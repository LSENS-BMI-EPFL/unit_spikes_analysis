"""
neuron_report.py

Generates a per-neuron PDF containing:
- title/table with metadata
- rasters and PSTHs under different conditions and alignments
- spike amplitude vs time
- performance plot

Dependencies: numpy, pandas, matplotlib, scipy
"""

import os
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

from typing import Dict, Any, Optional, Tuple, List

import NWB_reader_functions as nwb_reader
import allen_utils as allen
import neural_utils
import plotting_utils as plutils

# DREDGE_DATA_ROOT: the Linux root find_kilosort_paths already relies on
# (cicada_analysis/templates/baseline_analysis.py's own DATA_ROOT is a
# Windows-only path with no Linux override, so not reused here even now
# that _load_dredge is copied locally below).
DREDGE_DATA_ROOT = "/mnt/lsens-analysis/Axel_Bisi/data"


def _find_motion_folder(data_root, mouse_id, session_name, imec):
    """Return Path to the DREDge motion folder for one probe, or None.

    Copied from cicada_analysis/templates/baseline_analysis.py so this file
    doesn't depend on that separate project.
    """
    ephys_root = pathlib.Path(data_root) / mouse_id / session_name / "Ephys"
    if not ephys_root.exists():
        return None
    for catgt in sorted(ephys_root.glob("catgt_*")):
        for imec_dir in sorted(catgt.glob(f"*_{imec}")):
            for dredge_folder in ("dredge_fast", "dredge"):
                motion_dir = imec_dir / dredge_folder / "motion" / "motion"
                if motion_dir.exists():
                    return motion_dir
    return None


def _load_dredge(data_root, mouse_id, session_name, imec):
    """Load DREDge (disp, tbins, sbins) for one probe, or (None, None, None) if unavailable.

    disp : (n_time, n_sbins) um
    tbins: (n_time,) s
    sbins: (n_sbins,) um — None if spatial_bins file is absent

    Copied from cicada_analysis/templates/baseline_analysis.py so this file
    doesn't depend on that separate project.
    """
    folder = _find_motion_folder(data_root, mouse_id, session_name, imec)
    if folder is None:
        return None, None, None
    disp_path  = folder / "displacement_seg0.npy"
    tbins_path = folder / "temporal_bins_s_seg0.npy"
    if not (disp_path.exists() and tbins_path.exists()):
        return None, None, None
    disp  = np.load(disp_path)
    tbins = np.load(tbins_path)
    if disp.ndim == 1:
        disp = disp[:, np.newaxis]
    if disp.shape[0] != len(tbins):
        disp = disp.T
    sbins = None
    for sbins_name in ("spatial_bins_um.npy", "spatial_bins_um_seg0.npy", "spatial_bins_seg0.npy"):
        p = folder / sbins_name
        if p.exists():
            sbins = np.load(p)
            break
    return disp, tbins, sbins

# Thresholds for generate_unit_spike_report's nested output-quality tiers
# (see 'quality_tiers' in its per-unit loop).
PRESENCE_RATIO_THRESHOLD = 0.8
COVERAGE_RATIO_THRESHOLD = 0.8

DEFAULT_TMIN = -0.1
DEFAULT_TMAX = 0.2

# ---------------------------
# Utility / smoothing helpers
# ---------------------------
def compute_psth_from_spikes(spikes: np.ndarray,
                             align_times: np.ndarray,
                             tmin: float,
                             tmax: float,
                             bin_size: float,
                             sigma_smooth_ms: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    spikes: 1D numpy array of spike times (seconds)
    align_times: 1D numpy array of per-trial alignment times (seconds)
    returns (time_centers, mean_rate_Hz, sem_rate_Hz, raw_counts_per_trial)
    """
    nbins = int(np.round((tmax - tmin) / bin_size))
    edges = np.linspace(tmin, tmax, nbins + 1)
    centers = edges[:-1] + bin_size/2
    counts = np.zeros((len(align_times), nbins), dtype=float)
    for i, align in enumerate(align_times):
        rel_spikes = spikes - align
        # select spikes in window
        mask = (rel_spikes >= tmin) & (rel_spikes < tmax)
        selected = rel_spikes[mask]
        if selected.size:
            # histogram in this trial
            c, _ = np.histogram(selected, bins=edges)
            counts[i, :] = c
    # convert to rate in Hz
    rates = counts / bin_size  # spikes/bin -> spikes/sec
    mean_rate = rates.mean(axis=0)
    sem_rate = rates.std(axis=0, ddof=1) / np.sqrt(max(1, rates.shape[0]))

    return centers, mean_rate, sem_rate, counts


# ---------------------------
# Plotting primitives
# ---------------------------
def make_title_table(fig, metadata: Dict[str, Any]):
    """
    ax: matplotlib ax where the metadata text will be placed (no axes)
    metadata example keys: cluster_id, neuron_id, area_parent, area, layer, mouse_name
    """
    reward_txt = 'R+' if metadata.get('reward_group', 1) == 1 else 'R-'
    txt_one_line = f"Mouse {metadata['mouse_id']} ({reward_txt}), {metadata['area_acronym_custom']}, {metadata.get('imec', '?')}, neuron ID {metadata['neuron_id']}, cluster ID {metadata['cluster_id']}, bc_label {metadata.get('bc_label', '?')}"
    # big title text
    fig.suptitle(txt_one_line, x=0.5, y=0.95, ha='center', va='center', fontsize=16, family='monospace', fontweight='semibold')
    return txt_one_line


def add_extra_metrics_table(ax, metadata: Dict[str, Any]):
    """Draw metadata['extra_metrics'] ({name: value}, see
    generate_unit_spike_report's extra_metrics param) as a text table filling
    its own dedicated axes, rather than a small floating corner box.

    Each row is bolded when it flags a quality concern (presence_ratio or
    coverage_ratio below their PRESENCE_RATIO_THRESHOLD/COVERAGE_RATIO_THRESHOLD,
    or significant correlation to motion) and colored red when it indicates
    significant correlation to the learning curve — both read off the
    "correlated"/"not correlated" text the motion/learning_curve entries
    were formatted with above, and the row's own key name.
    """
    ax.axis('off')
    extra_metrics = metadata.get('extra_metrics') or {}
    if not extra_metrics:
        ax.text(0.5, 0.5, "No extra metrics", ha='center', va='center', fontsize=12)
        return

    def _row_style(name, value):
        bold, color = False, 'black'
        if name in ('presence_ratio', 'coverage_ratio'):
            threshold = PRESENCE_RATIO_THRESHOLD if name == 'presence_ratio' else COVERAGE_RATIO_THRESHOLD
            val = pd.to_numeric(value, errors='coerce')
            if pd.notna(val) and val < threshold:
                bold = True
        is_significant = str(value).startswith('correlated')
        if 'motion' in name and is_significant:
            bold = True
        if 'learning_curve' in name and is_significant:
            color = 'red'
        return bold, color

    ax.set_title('Additional metrics', fontsize=13)
    line_height = min(0.09, 0.85 / max(len(extra_metrics), 1))
    y = 0.97
    for name, value in extra_metrics.items():
        bold, color = _row_style(name, value)
        ax.text(0.03, y, f"{name}: {value}", ha='left', va='top', fontsize=14,
                family='monospace', color=color,
                fontweight='bold' if bold else 'normal',
                transform=ax.transAxes)
        y -= line_height

def plot_raster(
    ax,
    spikes: np.ndarray,
    trials_df: pd.DataFrame,
    align_col: str,
    tmin: float,
    tmax: float,
    sort_by: Optional[str] = None,
    condition_mask: Optional[np.ndarray] = None,
    cmap: Optional[Dict[str, str]] = None,
    context_cmap: Optional[Dict[str, str]] = None,
    trial_type_col: str = "trial_type",
    context_col: str = "context",
    dot_size: float = 8.0
):
    """
    Draw raster for trials in trials_df.
    Colors spikes by trial type (if cmap provided) and/or by context (if context_cmap provided).
    Adds shading for passive trials.
    """

    # Select subset
    if condition_mask is None:
        sel_df = trials_df.copy()
    else:
        sel_df = trials_df[condition_mask].copy()
    if sel_df.empty:
        ax.text(0.5, 0.5, "No trials", ha='center', va='center')
        ax.set_xlim(tmin, tmax)
        ax.set_ylim(0, 1)
        return

    # Sort trials
    if sort_by and (sort_by in sel_df.columns):
        sel_df = sel_df.sort_values(by=sort_by, ascending=True)
    else:
        sel_df = sel_df.reset_index(drop=True)

    n_trials = len(sel_df)

    # Highlight passive trials (NO flip now)
    if context_col in sel_df.columns:
        passive_mask = sel_df[context_col].str.contains("passive", case=False, na=False)
        for row_i in np.where(passive_mask)[0]:
            ax.axhspan(row_i - 0.5, row_i + 0.5,
                       facecolor='lightgray', edgecolor="none", alpha=0.3, zorder=0)

    # Plot spikes (NO flip → y = i instead of n_trials-1-i)
    for i, (_, row) in enumerate(sel_df.iterrows()):
        rel_spikes = spikes - row[align_col]
        xs = rel_spikes[(rel_spikes >= tmin) & (rel_spikes < tmax)]
        ys = np.full(xs.shape, i)  # first trial = row 0, last = bottom
        if xs.size:
            if context_cmap is not None and pd.notnull(row.get(context_col)):
                color = context_cmap.get(row[context_col], 'k')
            elif cmap is not None and pd.notnull(row.get(trial_type_col)):
                color = cmap.get(row[trial_type_col], 'k')
            else:
                color = 'k'
            ax.scatter(xs, ys, s=dot_size, marker='o', color=color, edgecolors='none')

    # Axis formatting
    ax.set_xlim(tmin, tmax)
    ax.set_ylabel("Trials (first → last)")
    ax.set_xlabel(f"Time from {align_col} (s)")
    ax.axvline(0, color='k', linestyle='--', linewidth=1)

    # Make trial 0 at top
    ax.invert_yaxis()


    return

def plot_psth(ax,
              spikes: np.ndarray,
              trials_df: pd.DataFrame,
              align_col: str,
              tmin: float,
              tmax: float,
              bin_size: float = 0.01,
              groupby: Optional[str] = None,
              group_values: Optional[List[Any]] = None,
              legend: bool = True,
              colors: Optional[Dict[Any, str]] = None,
              linestyle_map: Optional[Dict[Any, str]] = None,
              label: Optional[str] = None):
    """
    Plot PSTHs for groups defined in trials_df[groupby].
    - If groupby is None, plot all trials in a single color.
    - If groupby is 'lick_flag' or similar, color fixed by trial type, linestyle changes per group.
    """
    if groupby is None:
        align_times = trials_df[align_col].values
        centers, mean_rate, sem_rate, _ = compute_psth_from_spikes(spikes, align_times, tmin, tmax, bin_size)
        ax.plot(centers, mean_rate, label=label, color=colors if isinstance(colors, str) else None)
        ax.fill_between(centers, mean_rate - sem_rate, mean_rate + sem_rate, alpha=0.3, color=colors if isinstance(colors, str) else None)
    else:
        if group_values is None:
            group_values = trials_df[groupby].unique()

        # Determine if we are using linestyle for grouping
        use_linestyle = groupby.lower() in ['lick_flag', 'lick'] and linestyle_map is not None

        for gv in group_values:
            mask = trials_df[groupby] == gv
            if mask.sum() == 0:
                continue
            align_times = trials_df.loc[mask, align_col].values
            centers, mean_rate, sem_rate, _ = compute_psth_from_spikes(spikes, align_times, tmin, tmax, bin_size)

            if use_linestyle:
                # color fixed, linestyle varies
                color = 'k'
                ls = linestyle_map.get(gv, '-') if linestyle_map else '-'
            else:
                color = colors.get(gv) if colors else None
                ls = '-'

            ax.plot(centers, mean_rate, label=str(gv), color=color, linestyle=ls, lw=2.0)
            ax.fill_between(centers, mean_rate - sem_rate, mean_rate + sem_rate,
                            alpha=0.2, lw=0, color=color)

    ax.set_xlim(tmin, tmax)

    # Axis labels
    if 'jaw' in align_col:
        ax.set_xlabel('Time from jaw (s)')
    else:
        ax.set_xlabel('Time from start (s)')
    ax.set_ylabel('Firing rate (spks/s)')

    ax.axvline(x=0, color='k', linestyle='--', linewidth=1)

    if legend:
        ax.legend(fontsize='small', frameon=False, loc='upper right')

    # Integer y-axis
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x)}"))

    return

def plot_spike_amplitudes(ax,
                          spike_times: np.ndarray,
                          spike_amps: np.ndarray,
                          tmin: float,
                          tmax: float,
                          passive_windows: Optional[List[Tuple[float, float]]] = None):
    """
    Plot spike amplitude vs time for that neuron (global time).
    """
    mask = (spike_times >= tmin) & (spike_times <= tmax)
    ax.plot(spike_times[mask], spike_amps[mask], marker='.', linestyle='None', markersize=1, alpha=0.5)
    for window in passive_windows:
        ax.axvspan(window[0], window[1], color='lightgray', alpha=0.5, zorder=0)
    ax.set_xlabel('Session time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_xlim(tmin, tmax)
    # Only clip the top of the y-axis (99.5th percentile) so a handful of
    # extreme-high outliers don't flatten the rest of the distribution —
    # the lower bound is always the true min, never clipped, since the low
    # end (near baseline/noise floor) is exactly what needs to stay visible.
    amps_in_window = spike_amps[mask]
    if amps_in_window.size:
        lo = amps_in_window.min()
        hi = np.percentile(amps_in_window, 99.5)
        if hi > lo:
            margin = 0.05 * (hi - lo)
            ax.set_ylim(lo - margin, hi + margin)
    ax.set_title('Spike amplitudes')

def plot_dredge_motion(ax, disp, tbins, sbins, unit_depth=None):
    """
    Plot DREDge estimated motion (displacement) for this unit's probe, for
    session-wide drift context alongside the spike-amplitude-vs-time panel.
    One line per spatial (depth) bin — same style as baseline_analysis.py's
    probe-overview DREDge panel. If unit_depth is given, the line for the
    spatial bin closest to this unit's own depth is drawn bold so it stands
    out among the others.
    disp : (n_time, n_sbins) um; tbins : (n_time,) s; sbins : (n_sbins,) um or None.
    """
    if disp is None or tbins is None:
        ax.text(0.5, 0.5, "No DREDge motion data", ha='center', va='center')
        ax.set_xlabel('Session time (s)')
        ax.set_ylabel('Displacement (um)')
        ax.set_title('Estimated motion')
        return
    n_sbins    = disp.shape[1]
    depth_cmap = plt.cm.tab10
    depth_norm = mcolors.Normalize(vmin=0, vmax=max(n_sbins - 1, 1))
    # pd.to_numeric: sbins/unit_depth (the latter from the raw NWB units
    # table) are occasionally a non-numeric placeholder (e.g. the literal
    # string "None") rather than an actual missing value — same root cause
    # as presenceRatio elsewhere in this file. Coerce instead of crashing.
    sbins_arr = (pd.to_numeric(pd.Series(np.asarray(sbins).ravel()), errors='coerce').to_numpy()
                if sbins is not None else None)
    unit_bin = None
    if unit_depth is not None and sbins_arr is not None:
        unit_depth_val = pd.to_numeric(unit_depth, errors='coerce')
        if pd.notna(unit_depth_val) and np.isfinite(sbins_arr).any():
            unit_bin = int(np.nanargmin(np.abs(sbins_arr - unit_depth_val)))
    for k in range(n_sbins):
        label = f"{sbins_arr[k]:.0f} um" if sbins_arr is not None and np.isfinite(sbins_arr[k]) else f"bin {k}"
        is_unit_bin = (k == unit_bin)
        if is_unit_bin:
            label += " (this unit)"
        ax.plot(tbins, disp[:, k], color=depth_cmap(depth_norm(k)),
                lw=2.5 if is_unit_bin else 0.7,
                alpha=1.0 if is_unit_bin else 0.85,
                zorder=3 if is_unit_bin else 2,
                label=label)
    ax.set_xlabel('Session time (s)')
    ax.set_ylabel('Displacement (um)')
    ax.legend(fontsize=6, frameon=False, ncol=min(n_sbins, 5),
             loc='upper right', handlelength=1.0)
    ax.set_title('Estimated motion')

def plot_performance(ax, trials_df: pd.DataFrame, block_size: int = 20, time_col: str = 'start_time', type_colors: Optional[Dict[str, str]] = None, metadata: Optional[Dict[str, Any]] = None):
    trials_df = trials_df[(trials_df.context=='active')
                        & (trials_df.early_lick==0)]
    trials_df  = trials_df.reset_index(drop=True)
    trials_df['outcome_w'] = trials_df.loc[(trials_df.trial_type=='whisker_trial')]['lick_flag']
    trials_df['outcome_a'] = trials_df.loc[(trials_df.trial_type=='auditory_trial')]['lick_flag']
    trials_df['outcome_n'] = trials_df.loc[(trials_df.trial_type=='no_stim_trial')]['lick_flag']

    # Add the block info
    block_length = 20
    trials_df['trial'] = trials_df.index
    trials_df['block'] = trials_df.loc[trials_df.early_lick == 0, 'trial'].transform(
        lambda x: x // block_length)

    # Compute hit rates. Use transform to propagate hit rate to all entries.
    # 'mean' (not np.nanmean) - pandas' groupby transform already skips NaN,
    # and the callable form is deprecated in favor of the string name.
    trials_df['hr_w'] = trials_df.groupby(['block', 'opto_stim'], as_index=False, dropna=False)[
        'outcome_w'].transform('mean')
    trials_df['hr_a'] = trials_df.groupby(['block', 'opto_stim'], as_index=False, dropna=False)[
        'outcome_a'].transform('mean')
    trials_df['hr_n'] = trials_df.groupby(['block', 'opto_stim'], as_index=False, dropna=False)[
        'outcome_n'].transform('mean')

    if trials_df.empty:
        ax.text(0.5, 0.5, "No trials", ha='center', va='center')
        return

    # Plot performance
    if type_colors is None:
        type_colors = {'whisker_trial':'forestgreen' if metadata['reward_group']==1 else 'crimson',
                       'auditory_trial':'mediumblue',
                       'no_stim_trial':'k'}
    # Update whisker color based on reward group
    if metadata is not None:
        if metadata.get('reward_group', 1) == 1:
            type_colors['whisker_trial'] = 'forestgreen'
        else:
            type_colors['whisker_trial'] = 'crimson'
    sns.lineplot(data=trials_df, x='block', y='hr_a', ax=ax, label='Auditory', color=type_colors['auditory_trial'], markers='o', lw=2)
    sns.lineplot(data=trials_df, x='block', y='hr_w', ax=ax, label='Whisker', color=type_colors['whisker_trial'], markers='o', lw=2)
    sns.lineplot(data=trials_df, x='block', y='hr_n', ax=ax, label='False alarm', color=type_colors['no_stim_trial'], markers='o', lw=2)

    # Set x axis as trials
    x_ticks = ax.get_xticks()
    x_ticklabels = [str(int(tick * block_length)) for tick in x_ticks]
    ax.set_xticks(x_ticks)  # pin ticks first — set_xticklabels alone warns otherwise
    ax.set_xticklabels(x_ticklabels)
    ax.set_xlabel('Trials')
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('P(lick)')
    # loc='best' (not the previous outside-axes bbox_to_anchor) since this
    # panel no longer has guaranteed empty space to its right — extra_metrics
    # now sits immediately next to it.
    ax.legend(frameon=False, fontsize=6, loc='best')
    ax.set_title(f'Mouse performance')
    ax.grid(alpha=0.3)
    return

# ---------------------------
# Layout / PDF generator
# ---------------------------


def default_layout_map():
    """
    3 active rows x 6 columns (rasters+performance, PSTHs+motion, summaries).
    """

    lm = {
        # -------- Row 0: rasters + performance, 2 cols each --------
        'combined_raster_whisker':   (0, 0, 1, 2),
        'combined_raster_auditory':  (0, 2, 1, 2),
        'performance':               (0, 4, 1, 2),

        # -------- Row 3: PSTHs + motion (all same width: 2 cols each) --------
        'psth_whisker':        (3, 0, 1, 2),
        'psth_auditory':       (3, 2, 1, 2),
        'motion':              (3, 4, 1, 2),

        # -------- Row 4: summaries (all same width: 2 cols each) --------
        'amp_time':        (4, 0, 1, 2),
        'waveform_mean':   (4, 2, 1, 2),
        'extra_metrics':   (4, 4, 1, 2),
    }

    return lm

def generate_neuron_pdf(neuron_id: Any,
                        spikes: np.ndarray,
                        spike_times: np.ndarray,
                        spike_amps: np.ndarray,
                        trials_df: pd.DataFrame,
                        metadata: Dict[str, Any],
                        outpath: str,
                        layout_map: Optional[Dict[str, Tuple[int,int,int,int]]] = None,
                        tmin: float = DEFAULT_TMIN,
                        tmax: float = DEFAULT_TMAX,
                        bin_size: float = 0.005):
    """
    Main function to produce PDF for a single neuron.

    Parameters:
    - neuron_id: identifier (for filename or title)
    - spikes: 1D numpy array of spike times (seconds) for this neuron (global timestamps)
    - spike_times: same as 'spikes' (kept separate name if you prefer)
    - spike_amps: array of spike amplitudes (same length as spikes)
    - trials_df: pandas DataFrame with at least these columns:
         - 'trial_start' or 'trial_start_time' (seconds or datetime) -- for 'raster' alignment if desired
         - 'trial_start_time' in seconds (or supply align columns used below)
         - 'trial_type' (strings like 'whisker'/'auditory')
         - 'is_whisker', 'is_auditory' boolean optional
         - 'jaw_onset' (seconds) (alignment)
         - 'trial_id' or similar for sorting early->late
         - 'passive_pre'/'passive_post'/'active' booleans or a 'behavioral_state' column
         - 'lick_flag' boolean (or 'licked')
         - 'trial_outcome' or 'correct' column (0/1) for performance
    - metadata: dict with keys for title table
    - outpath: path to save the PDF (e.g. 'neuron_123_report.pdf')
    - layout_map: optional mapping to override default positions
    - tmin/tmax, bin_size: time window and PSTH bin size
    """
    if layout_map is None:
        layout_map = default_layout_map()

    # default_layout_map() only defines rows 0, 3, 4 (rasters+performance,
    # PSTHs+motion, summaries) — remap to a compact 3-row grid.
    _row_remap = {0: 0, 3: 1, 4: 2}
    layout_map = {
        k: (_row_remap[r], c, rs, cs)
        for k, (r, c, rs, cs) in layout_map.items()
    }

    # create pdf + figure with GridSpec
    nrows = 3
    ncols = 6
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(nrows, ncols, figure=fig, wspace=0.5, hspace=0.4,
                  height_ratios=[1, 1, 1])

    # helper to add subplot from key
    def add_ax_for_key(key):
        if key not in layout_map:
            return None
        r, c, rs, cs = layout_map[key]
        ax = fig.add_subplot(gs[r:r+rs, c:c+cs])
        plutils.remove_top_right_frame(ax)
        return ax

    # Trial colors
    type_colors = {'whisker_trial':'forestgreen',
                   'auditory_trial':'mediumblue',
                   'no_stim_trial':'k'}

    # Full trial set (active + passive), kept aside for the rasters below —
    # everything else in this function works off the passive-only trials_df
    # split out next.
    trials_df_all = trials_df

    trials_df_actif = trials_df[trials_df['context'] == 'active']
    trials_df = trials_df[trials_df['context'] != 'active']

    # Needed by the row-0 rasters and row-3 PSTH panels below (and by the
    # still-disabled rows 1-2 correlation plots if those get restored later).
    align_col_start = 'start_time'

    # ============= ROW 0: spike rasters, aligned to start_time (no DLC) =========
    # Every whisker/auditory trial (active + passive), not just passive ones.
    sort_col = 'trial_id' if 'trial_id' in trials_df_all.columns else None

    # Whisker raster
    ax_raster_whisk = add_ax_for_key('combined_raster_whisker')
    if ax_raster_whisk is not None:
        mask = (trials_df_all['trial_type'] == 'whisker_trial')
        context_cmap = {
            'passive_pre': plutils.adjust_lightness(type_colors['whisker_trial'], 1.5),
            'active': type_colors['whisker_trial'],
            'passive_post': plutils.adjust_lightness(type_colors['whisker_trial'], 0.4)
        }
        plot_raster(
            ax=ax_raster_whisk,
            spikes=spikes,
            trials_df=trials_df_all,
            align_col=align_col_start,
            tmin=tmin,
            tmax=tmax,
            condition_mask=mask,
            sort_by=sort_col,
            context_cmap=context_cmap,
        )
        ax_raster_whisk.set_title('Whisker trials')

    # Auditory raster
    ax_raster_aud = add_ax_for_key('combined_raster_auditory')
    if ax_raster_aud is not None:
        mask = (trials_df_all['trial_type'] == 'auditory_trial')
        context_cmap = {
            'passive_pre': plutils.adjust_lightness(type_colors['auditory_trial'], 2.0),
            'active': type_colors['auditory_trial'],
            'passive_post': plutils.adjust_lightness(type_colors['auditory_trial'], 0.6)
        }
        plot_raster(
            ax=ax_raster_aud,
            spikes=spikes,
            trials_df=trials_df_all,
            align_col=align_col_start,
            tmin=tmin,
            tmax=tmax,
            condition_mask=mask,
            sort_by=sort_col,
            context_cmap=context_cmap,
        )
        ax_raster_aud.set_title('Auditory trials')

    # Mouse performance — sits next to the rasters now, for a consistent
    # 2-col-per-panel layout across all active rows.
    ax_perf = add_ax_for_key('performance')
    if ax_perf is not None:
        plot_performance(ax_perf, trials_df_actif, time_col='start_time', type_colors=type_colors, metadata=metadata)

    # ============= ROW 3: spike PSTHs (by context, not baseline-corrected) ====
    # Color adjustment helper
    def ctx_colors(base_color):
        """Return a dict of context → color with lightness variations."""
        return {
            'passive_pre': plutils.adjust_lightness(base_color, 1.5),
            'passive_post': plutils.adjust_lightness(base_color, 0.6), # darker
            'active': plutils.adjust_lightness(base_color, 1.0)
        }

    # Contexts in preferred plotting order
    context_order = ['passive_pre', 'passive_post']


    # --- Whisker trials PSTH (passive_pre vs passive_post) ---
    ax_psth_whisk = add_ax_for_key('psth_whisker')
    if ax_psth_whisk is not None:
        whisk_df = trials_df[trials_df['trial_type'] == 'whisker_trial']
        if 'context' in whisk_df.columns and not whisk_df.empty:
            available_contexts = [ctx for ctx in context_order if ctx in whisk_df['context'].unique()]
            colors = ctx_colors(type_colors['whisker_trial'])
            plot_psth(
                ax=ax_psth_whisk,
                spikes=spikes,
                trials_df=whisk_df,
                align_col=align_col_start,
                tmin=tmin,
                tmax=tmax,
                bin_size=bin_size,
                groupby='context',
                group_values=available_contexts,
                colors={k: colors[k] for k in available_contexts}
            )
        else:
            ax_psth_whisk.text(0.5, 0.5, "No whisker or context data", ha='center', va='center')
        ax_psth_whisk.set_title("Whisker (by context)")

    # --- Auditory trials PSTH (passive_pre vs passive_post) ---
    ax_psth_aud = add_ax_for_key('psth_auditory')
    if ax_psth_aud is not None:
        aud_df = trials_df[trials_df['trial_type'] == 'auditory_trial']
        if 'context' in aud_df.columns and not aud_df.empty:
            available_contexts = [ctx for ctx in context_order if ctx in aud_df['context'].unique()]
            colors = ctx_colors(type_colors['auditory_trial'])
            plot_psth(
                ax=ax_psth_aud,
                spikes=spikes,
                trials_df=aud_df,
                align_col=align_col_start,
                tmin=tmin,
                tmax=tmax,
                bin_size=bin_size,
                groupby='context',
                group_values=available_contexts,
                colors={k: colors[k] for k in available_contexts}
            )
        else:
            ax_psth_aud.text(0.5, 0.5, "No auditory or context data", ha='center', va='center')
        ax_psth_aud.set_title("Auditory (by context)")

    # --- DREDge estimated motion for this unit's probe ---
    ax_motion = add_ax_for_key('motion')
    if ax_motion is not None:
        disp, dredge_tbins, sbins = metadata.get('motion', (None, None, None))
        plot_dredge_motion(ax_motion, disp, dredge_tbins, sbins, unit_depth=metadata.get('depth'))

    # ============= ROW 4: spike amplitudes, waveform, mouse performance ============
    ax_amp = add_ax_for_key('amp_time')
    if ax_amp is not None:
        # For ampt plot we typically want to show spikes across session time.
        # We'll define tmin/tmax in absolute session times; if user's spike_times array is in seconds from session start, that's fine.
        # Here we assume spike_times is same as spikes argument.
        # Provide full session window (or tmin/tmax)
        # We'll just plot spike amplitudes across session (not aligned).
        # If spike_times are absolute, you can supply window in metadata or param, here use spikes min->max
        tmin_session = spike_times.min() if spike_times.size else 0
        tmax_session = spike_times.max() if spike_times.size else 1
        # passive pre start and end time from trials_df
        passive_pre_start = trials_df[trials_df['context']=='passive_pre']['start_time'].min()
        passive_pre_end = trials_df[trials_df['context']=='passive_pre']['stop_time'].max()
        passive_post_start = trials_df[trials_df['context']=='passive_post']['start_time'].min()
        passive_post_end = trials_df[trials_df['context']=='passive_post']['stop_time'].max()
        passive_windows = [(passive_pre_start, passive_pre_end), (passive_post_start, passive_post_end)]
        plot_spike_amplitudes(ax_amp, spike_times, spike_amps, tmin_session, tmax_session, passive_windows)

    ax_wf = add_ax_for_key('waveform_mean')
    if ax_wf is not None:
        waveform_mean = metadata['waveform_mean']
        ax_wf.plot(waveform_mean, lw=2.5)
        ax_wf.set_xlabel('Time (ms)')
        ax_wf.set_ylabel(r'Amplitude ($mu$V)')
        ax_wf.set_xlim(tmin, tmax)
        n_points = len(waveform_mean)
        ax_wf.set_xticks(np.linspace(0, n_points, 5))
        ax_wf.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, n_points / 30000 * 1000, 5)]) #assumes 30kHz sampling
        ax_wf.set_title('Mean waveform')

    ax_extra = add_ax_for_key('extra_metrics')
    if ax_extra is not None:
        add_extra_metrics_table(ax_extra, metadata)

    # -----------------------------------
    # Adjust figure, make title, and save
    # -----------------------------------
    fig.align_ylabels()
    _ = make_title_table(fig, metadata)
    mouse_id_str = metadata.get('mouse_id', '')
    filename = f"{mouse_id_str}_neuron_{neuron_id}_report"
    # outpath: a single folder, or a list of folders to save the same figure
    # into (e.g. this unit's quality-tier folders — see generate_unit_spike_report).
    outpaths = [outpath] if isinstance(outpath, (str, pathlib.Path)) else list(outpath)
    for op in outpaths:
        fig.savefig(os.path.join(op, f"{filename}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return

def find_kilosort_paths(base_dir, experimenter, mouse_id, session_id, probe_id='imec0'):
    """
    Find Kilosort output paths flexibly under:
        M:/analysis/{experimenter}/data/{mouse_id}/{session_id}/Ephys/
    even if the run name (e.g. _g0) or CatGT folder differs.

    Parameters
    ----------
    base_dir : str or pathlib.Path
        Root path, e.g. 'M:/analysis'
    experimenter : str
    mouse_id : str
    session_id : str
    probe : str
        Probe name, e.g. 'imec0', 'imec1'

    Returns
    -------
    dict with keys:
        'spike_clusters', 'amplitudes', 'spike_times', 'spike_templates'
        (any that exist)
    """

    ephys_dir = pathlib.Path(base_dir) / 'Axel_Bisi' / "data" / mouse_id / session_id / "Ephys"

    # Look for a CatGT folder (e.g., catgt_AB147_g0)
    catgt_folders = sorted(ephys_dir.glob("catgt_*"))
    if not catgt_folders:
        raise FileNotFoundError(f"No 'catgt_*' folder found in {ephys_dir}")
    catgt_folder = catgt_folders[0]  # pick the first one if multiple

    # Inside it, look for a matching probe folder
    probe_folders = sorted(catgt_folder.glob(f"*_{probe_id}"))
    if not probe_folders:
        raise FileNotFoundError(f"No probe folder matching '*_{probe_id}' found in {catgt_folder}")
    probe_folder = probe_folders[0]

    # Look for the kilosort folder (usually 'kilosort2', 'kilosort3', or 'kilosort4').
    # Prefer kilosort2 specifically when multiple versions exist for the same
    # probe; fall back to whatever's found (sorted()[0]) if there's no
    # kilosort2 folder.
    ks_folders = sorted(probe_folder.glob("kilosort*"))
    if not ks_folders:
        raise FileNotFoundError(f"No kilosort folder found in {probe_folder}")

    ks2_folders = [p for p in ks_folders if p.name == 'kilosort4']
    ks_folder = ks2_folders[0] if ks2_folders else ks_folders[0]

    # Newer runs (kilosort4 via SpikeInterface) nest the actual output files
    # one level deeper, under 'sorter_output/', instead of directly inside
    # the kilosort folder (e.g. .../kilosort4/sorter_output/spike_clusters.npy).
    # Use that subfolder when present, else fall back to the kilosort folder
    # itself so older (kilosort2/3) runs without it still resolve correctly.
    sorter_output = ks_folder / "sorter_output"
    npy_dir = sorter_output if sorter_output.is_dir() else ks_folder

    # Gather relevant files if they exist
    result = {}
    for fname in ["spike_clusters.npy", "amplitudes.npy", "spike_times.npy", "spike_templates.npy"]:
        fpath = npy_dir / fname
        if fpath.exists():
            result[fname.replace(".npy", "")] = fpath

    return result



def compute_coverage_ratio(unit_df, spike_times_col='spike_times'):
    """Fraction of the recording duration spanned by each unit's own spike
    train, reimplemented from baseline_analysis.py's filter_units_by_quality
    (cicada_analysis/templates/baseline_analysis.py) — same formula, applied
    here directly rather than importing that function (which expects to run
    as part of its own quality-filtering pipeline).

    Recording duration is derived from the population: earliest first-spike
    and latest last-spike across ALL units in unit_df (should be the full,
    unfiltered per-session unit table — not already subset to e.g. bc_label
    =='good' — since the population span is what "coverage" is relative to).

    Returns a pd.Series aligned to unit_df's index; 1.0 = spikes span the
    full recording, 0.0 for units with <2 spikes or an unusable duration.
    """
    all_spikes = [np.asarray(s) for s in unit_df[spike_times_col]]
    firsts = [s[0]  for s in all_spikes if len(s) > 0]
    lasts  = [s[-1] for s in all_spikes if len(s) > 0]
    if firsts and lasts:
        rec_start = min(firsts)
        rec_dur   = max(lasts) - rec_start
    else:
        rec_start, rec_dur = 0.0, 0.0
    ratios = [
        (s[-1] - s[0]) / rec_dur if (len(s) > 1 and rec_dur > 0) else 0.0
        for s in all_spikes
    ]
    return pd.Series(ratios, index=unit_df.index)


_SHIFT_TEST_CACHE: Dict[Any, Optional[pd.DataFrame]] = {}


def _load_shift_test_results(combined_results_path, mouse_id, session_day):
    """Load single_neuron_shift_tests.py's per-session shift-test CSV, if it
    exists, so its per-neuron r-values can be surfaced here without
    re-running that (expensive, permutation-based) analysis. Returns None if
    that analysis hasn't been run for this mouse/session."""
    key = (str(combined_results_path), mouse_id, session_day)
    if key not in _SHIFT_TEST_CACHE:
        csv_path = (pathlib.Path(combined_results_path) / mouse_id / session_day
                   / 'single_neuron_shift_test' / f'{mouse_id}_{session_day}_shift_test_results.csv')
        _SHIFT_TEST_CACHE[key] = pd.read_csv(csv_path) if csv_path.exists() else None
    return _SHIFT_TEST_CACHE[key]


# Main function
def generate_unit_spike_report(nwb_file, mouse_res_path, combined_results_path, extra_metrics=None, unit_table=None):
    """
    extra_metrics : optional list of unit_table column names to look up per
        unit and print in the report title (see make_title_table), in
        addition to the fixed set already shown (imec, cluster_id,
        area_acronym_custom, bc_label, presence_ratio, fractionRPV).
    unit_table : optional combined (all-mice) unit_table — the same one
        run_shift_test_analysis (single_neuron_shift_tests.py) was called
        with — used only to recover its 'unit_id' (a positional index into
        that combined table, not otherwise reproducible here) for each unit
        in this NWB file, by matching on cluster_id within this mouse/session.
        Needed to join this report's units against the shift-test CSVs,
        which are keyed by that same unit_id. Without it, shift-test
        correlations are simply omitted from extra_metrics.
    """

    combined_results_path = pathlib.Path(combined_results_path)

    # Get session info
    mouse_id = nwb_reader.get_mouse_id(nwb_file)
    session_id = nwb_reader.get_session_id(nwb_file)
    initials = nwb_reader.get_experimenter(nwb_file)
    sess_metadata = nwb_reader.get_session_metadata(nwb_file)
    behavior_type, day = nwb_reader.get_bhv_type_and_training_day_index(nwb_file)
    if day != 0 :
        return 0
    # Matches single_neuron_shift_tests.py's session_day folder/filename convention
    # (e.g. "whisker_0"), used below to locate its per-session results CSV.
    session_day = f"{behavior_type}_{day}"

    # Load trial and unit tables from NWB
    trial_df = nwb_reader.get_trial_table(nwb_file)
    trial_df['mouse_id'] = mouse_id
    trial_df['session_id'] = session_id
    unit_df = nwb_reader.get_unit_table(nwb_file)
    unit_df = allen.create_area_custom_column(unit_df)
    # Computed over ALL units in this session (before the bc_label=='good'
    # filter below), since coverage is relative to the whole recording span.
    unit_df['coverage_ratio'] = compute_coverage_ratio(unit_df)


    # Define passive pre and passive post
    n_trials = len(trial_df)
    mid_session_trial_idx = n_trials // 2
     # Passive pre are early and passive post are late in session
    mask_pre = (trial_df['context'] == 'passive') & (trial_df.index < mid_session_trial_idx)
    mask_post = (trial_df['context'] == 'passive') & (trial_df.index >= mid_session_trial_idx)
    trial_df.loc[mask_pre, 'context'] = 'passive_pre'
    trial_df.loc[mask_post, 'context'] = 'passive_post'

    # Filter, format data. bc_label in {'good','mua'} (i.e. not noise) is the
    # base population for the 'all' output tier — see the per-unit
    # qualifying-folder logic below for the stricter, nested tiers.
    unit_df_subset = unit_df[unit_df['bc_label'].isin(['good', 'mua'])]
    unit_df_subset = neural_utils.convert_electrode_group_object_to_columns(unit_df_subset)

    # Recover each unit's shift-test 'unit_id' (see generate_unit_spike_report's
    # docstring) by matching (cluster_id, electrode_group) within this
    # mouse/session against the combined unit_table that produced it, then
    # load that session's shift-test results (if any) for the correlation
    # lookup per unit below. electrode_group is part of the key because
    # cluster_id is only unique per probe, not per session — a session with
    # multiple probes can reuse the same cluster_id on different imecs.
    if unit_table is not None:
        id_map = (unit_table[(unit_table['mouse_id'] == mouse_id) &
                             (unit_table['session_id'] == session_id)]
                 [['cluster_id', 'electrode_group', 'unit_id']]
                 .drop_duplicates(['cluster_id', 'electrode_group']))
        unit_df_subset = unit_df_subset.merge(id_map, on=['cluster_id', 'electrode_group'], how='left')
    shift_results = _load_shift_test_results(combined_results_path, mouse_id, session_day)
    print(unit_df_subset['area_acronym_custom'].unique())
    print('Unit cols', unit_df_subset.columns)
    if initials=='AB':
        experimenter = 'Axel_Bisi'
    elif initials=='MH':
        experimenter = 'Myriam_Hamon'

    for imec_id in unit_df_subset['electrode_group'].unique():
        unit_df_imec = unit_df_subset[unit_df_subset['electrode_group']==imec_id]

        # Get paths to spike clusters and amplitudes
        imec_id = imec_id.split('_')[0]
        ks_paths = find_kilosort_paths(base_dir="/mnt/lsens-analysis", experimenter=experimenter, mouse_id=mouse_id,
                                       session_id=session_id, probe_id=imec_id)
        print(f"  [{mouse_id}/{session_id}/{imec_id}] Kilosort paths: {ks_paths}")

        # DREDge estimated motion for this probe (one per imec, reused across
        # its units) — (None, None, None) if unavailable for this probe/session.
        motion = _load_dredge(DREDGE_DATA_ROOT, mouse_id, session_id, imec_id)

        for idx, row in unit_df_imec.iterrows():

            # single_neuron_shift_tests.py's r (baseline/evoked x motion/learning_curve)
            # for this unit, if that analysis has been run and unit_id resolved above.
            # pd.to_numeric: presenceRatio/coverage_ratio are occasionally
            # strings in the raw NWB data (seen on AB143_20241126_115737.nwb)
            # rather than floats — coerce to NaN instead of crashing on
            # formatting/comparison.
            presence_ratio_val = pd.to_numeric(row['presenceRatio'], errors='coerce')
            coverage_ratio_val = pd.to_numeric(row['coverage_ratio'], errors='coerce')

            unit_shift_rows = None
            corr_metrics = {}
            if shift_results is not None and pd.notna(row.get('unit_id')):
                unit_shift_rows = shift_results[shift_results['unit_id'] == row['unit_id']]
                for _, tr in unit_shift_rows.iterrows():
                    label = "correlated" if tr['significant'] else "not correlated"
                    corr_metrics[f"{tr['epoch']}_{tr['factor']}"] = f"{label} (r={tr['r']:.2f})"
            corr_metrics['coverage_ratio'] = (f"{coverage_ratio_val:.2f}"
                                              if pd.notna(coverage_ratio_val) else str(row['coverage_ratio']))
            corr_metrics['presence_ratio'] = (f"{presence_ratio_val:.2f}"
                                              if pd.notna(presence_ratio_val) else str(row['presenceRatio']))
            # Same string-placeholder issue as presenceRatio can affect any
            # raw NWB column, so format these numeric-safe too.
            for col in ('percentageSpikesMissing_gaussian', 'fractionRPVs_estimatedTauR', 'signalToNoiseRatio'):
                val = pd.to_numeric(row[col], errors='coerce')
                corr_metrics[col] = f"{val:.2f}" if pd.notna(val) else str(row[col])

            # Nested quality tiers: each tier requires everything the previous
            # one did, plus one more criterion. A unit's report is saved into
            # every tier it qualifies for (not just the strictest one) — see
            # PRESENCE_RATIO_THRESHOLD/COVERAGE_RATIO_THRESHOLD near the top
            # of this file to tune the thresholds.
            quality_tiers = ['all']   # bc_label in {'good','mua'}, already the unit_df_subset filter
            if row['bc_label'] == 'good':
                quality_tiers.append('good')
                if pd.notna(presence_ratio_val) and presence_ratio_val > PRESENCE_RATIO_THRESHOLD:
                    quality_tiers.append(f'good_presence_ratio_{PRESENCE_RATIO_THRESHOLD}')
                    if pd.notna(coverage_ratio_val) and coverage_ratio_val > COVERAGE_RATIO_THRESHOLD:
                        quality_tiers.append(
                            f'good_presence_ratio_{PRESENCE_RATIO_THRESHOLD}'
                            f'_coverage_ratio_{COVERAGE_RATIO_THRESHOLD}')
                        motion_rows = (unit_shift_rows[unit_shift_rows['factor'] == 'motion']
                                      if unit_shift_rows is not None else None)
                        # Require the shift-test to have actually run for this unit
                        # (motion_rows non-empty) — can't claim "uncorrelated" without it.
                        if motion_rows is not None and not motion_rows.empty and not motion_rows['significant'].any():
                            quality_tiers.append(
                                f'good_presence_ratio_{PRESENCE_RATIO_THRESHOLD}'
                                f'_coverage_ratio_{COVERAGE_RATIO_THRESHOLD}_uncorrelated_motion')

            metadata = {
            'mouse_id':mouse_id,
            'imec':imec_id,
            'reward_group':sess_metadata['wh_reward'],
            'neuron_id':row['neuron_id'],
            'cluster_id':row['cluster_id'],
            'area_acronym_custom':row['area_acronym_custom'],
            'bc_label':row['bc_label'],
            'depth': row.get('depth'),
            'waveform_mean': row['waveform_mean'],
            'presence_ratio': row['presenceRatio'],
            'fractionRPV': row['fractionRPVs_estimatedTauR'],
            'extra_metrics': {**{m: row[m] for m in (extra_metrics or [])}, **corr_metrics},
            'motion': motion,
            }

            unit_spikes = row['spike_times']
            cluster_id = int(row['cluster_id'])

            # Get cluster spikes
            spike_clusters = np.load(ks_paths['spike_clusters'])
            spike_clusters = np.array(spike_clusters)
            amplitudes = np.load(ks_paths['amplitudes']) #Note: these are spike TEMPLATE amplitudes
            amplitudes = np.array(amplitudes)

            spike_indices = np.where(spike_clusters==cluster_id)
            unit_spike_amplitudes = amplitudes[spike_indices]
            if len(unit_spike_amplitudes) != len(unit_spikes):
                print(f"  [MISMATCH] cluster_id={cluster_id}: NWB has "
                      f"{len(unit_spikes)} spike(s) but {ks_paths.get('spike_clusters')} "
                      f"has {len(unit_spike_amplitudes)} matching cluster_id={cluster_id} "
                      f"(spike_clusters.npy has {len(spike_clusters)} total spikes, "
                      f"{len(set(spike_clusters.tolist()))} unique cluster ids)")


            tier_paths = []
            for tier in quality_tiers:
                tier_path = combined_results_path / 'unit_spike_report' / tier
                tier_path.mkdir(parents=True, exist_ok=True)
                tier_paths.append(tier_path)

            generate_neuron_pdf(row['neuron_id'], unit_spikes, unit_spikes, unit_spike_amplitudes, trial_df, metadata,
                                outpath=tier_paths)
