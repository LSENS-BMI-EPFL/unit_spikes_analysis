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
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

from typing import Dict, Any, Optional, Tuple, List

import NWB_reader_functions as nwb_reader
import neural_utils
import plotting_utils as plutils


TRIAL_MAP = {
    0: 'whisker_miss',
    1: 'auditory_miss',
    2: 'whisker_hit',
    3: 'auditory_hit',
    4: 'correct_rejection',
    5: 'false_alarm',
    6: 'association',
}
# ---------------------------
# Utility / smoothing helpers
# ---------------------------
def compute_psth(spike_times_list: List[np.ndarray],
                 align_times: np.ndarray,
                 tmin: float,
                 tmax: float,
                 bin_size: float,
                 sigma_smooth_ms: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PSTH: returns (time_edges_centers, rateHz) averaged across trials.
    spike_times_list: list of spike-time arrays (seconds) for the neuron (global times).
    align_times: per-trial alignment times (seconds) (same order/length as trials)
    bin_size: in seconds (e.g. 0.01)
    sigma_smooth_ms: gaussian smoothing sigma in milliseconds (applied in bins)
    """
    nbins = int(np.round((tmax - tmin) / bin_size))
    edges = np.linspace(tmin, tmax, nbins + 1)
    centers = edges[:-1] + bin_size/2
    # prepare counts per trial
    counts = np.zeros((len(align_times), nbins), dtype=float)
    for i, align_t in enumerate(align_times):
        # get spikes in window around align
        rel_spikes = spike_times_list - align_t  # vectorized if spike_times_list is np array
        # however spike_times_list is global array; we want spikes of neuron (global)
        # We'll instead pass the neuron's full spike array; here left for clarity.
        # So this function expects spike_times_list to be the neuron's global spike times (1d np array)
        pass
    # Implementation note: the caller uses compute_psth_from_spikes for simplicity below.
    raise RuntimeError("Do not call this function directly; use compute_psth_from_spikes in this file.")

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
    lines = []
    order = ['mouse_id', 'neuron_id', 'ccf_atlas_acronym']
    for k in order:
        if k in metadata:
            lines.append(f"{metadata[k]}")
    txt = "\n".join(lines)
    txt_one_line = " , ".join(lines)
    txt_one_line = f"Mouse {metadata['mouse_id']}, neuron ID {metadata['neuron_id']}, {metadata['ccf_atlas_acronym']}"
    # big title text
    fig.suptitle(txt_one_line, x=0.5, y=0.95, ha='center', va='center', fontsize=16, family='monospace', fontweight='semibold')
    return txt

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

    # Highlight passive trials
    if context_col in sel_df.columns:
        passive_idx = sel_df.index[sel_df[context_col].str.contains("passive", case=False, na=False)].to_numpy()
        for idx in passive_idx:
            # y-coordinate (flipped so early trials on top)
            y0 = n_trials - 1 - (sel_df.index.get_loc(idx) if idx in sel_df.index else idx)
            ax.axhspan(
                y0 - 0.5, y0 + 0.5,
                facecolor='lightgray',
                edgecolor="none",
                alpha=0.3,
                zorder=0
            )

    # Plot spikes
    for i, (_, row) in enumerate(sel_df.iterrows()):
        align = row[align_col]
        rel_spikes = spikes - align
        mask = (rel_spikes >= tmin) & (rel_spikes < tmax)
        xs = rel_spikes[mask]
        ys = np.full(xs.shape, n_trials - 1 - i)  # flipped y
        if xs.size:
            # Determine color
            color = 'k'
            if context_cmap is not None and context_col in row and pd.notnull(row[context_col]):
                color = context_cmap.get(row[context_col], 'k')
            elif cmap is not None and trial_type_col in row and pd.notnull(row[trial_type_col]):
                color = cmap.get(row[trial_type_col], 'k')
            ax.scatter(xs, ys, s=dot_size, marker='o', color=color, edgecolors='none')

    # Formatting
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(-1, n_trials)
    ax.set_ylabel('Trials (early → late)')
    if 'jaw' in align_col:
        ax.set_xlabel('Time from jaw (s)')
    else:
        ax.set_xlabel('Time from start (s)')
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
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
    ax.set_title('Spike amplitudes')

def plot_performance(ax, trials_df: pd.DataFrame, block_size: int = 20, time_col: str = 'start_time', type_colors: Optional[Dict[str, str]] = None):
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
    trials_df['hr_w'] = trials_df.groupby(['block', 'opto_stim'], as_index=False, dropna=False)[
        'outcome_w'].transform(np.nanmean)
    trials_df['hr_a'] = trials_df.groupby(['block', 'opto_stim'], as_index=False, dropna=False)[
        'outcome_a'].transform(np.nanmean)
    trials_df['hr_n'] = trials_df.groupby(['block', 'opto_stim'], as_index=False, dropna=False)[
        'outcome_n'].transform(np.nanmean)

    if trials_df.empty:
        ax.text(0.5, 0.5, "No trials", ha='center', va='center')
        return

    # Plot performance
    if type_colors is None:
        type_colors = {'whisker_trial':'forestgreen',
                       'auditory_trial':'mediumblue',
                       'no_stim_trial':'k'}
    sns.lineplot(data=trials_df, x='block', y='hr_a', ax=ax, label='Auditory', color=type_colors['auditory_trial'], markers='o', lw=2)
    sns.lineplot(data=trials_df, x='block', y='hr_w', ax=ax, label='Whisker', color=type_colors['whisker_trial'], markers='o', lw=2)
    sns.lineplot(data=trials_df, x='block', y='hr_n', ax=ax, label='False alarm', color=type_colors['no_stim_trial'], markers='o', lw=2)

    # Set x axis as trials
    x_ticks = ax.get_xticks()
    x_ticklabels = [str(int(tick * block_length)) for tick in x_ticks]
    ax.set_xticklabels(x_ticklabels)
    ax.set_xlabel('Trials')
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('P(lick)')
    ax.legend(frameon=False, fontsize=8, loc='center right')
    ax.set_title(f'Mouse performance')
    ax.grid(alpha=0.3)
    return


# ---------------------------
# Layout / PDF generator
# ---------------------------
DEFAULT_TMIN = -0.2
DEFAULT_TMAX = 0.5


def default_layout_map():
    """
    Returns a mapping of "plot keys" to grid positions (row, col, rowspan, colspan).
    We create a 5x5 grid by default (5 columns x 5 rows). Most requested plots mapped here.
    Keys you can place in layout_map:
      - 'title'
      - 'raster_all', 'raster_whisker', 'raster_auditory', 'raster_jaw'
      - 'psth_all', 'psth_whisker', 'psth_auditory', 'psth_jaw'
      - 'psth_lick', 'psth_whisker_lick', 'psth_auditory_lick', 'psth_jaw_lick'
      - 'psth_quartiles_active' (this can be placed over columns)
      - 'amp_time'
      - 'performance'
    """
    # grid: rows x cols = 5 x 5
    lm = {
        # FIRST ROW: 5 columns
        #'title': (0, 0, 1, 1),
        'raster_all': (0, 0, 2, 1),
        'raster_whisker': (0, 1, 2, 1),
        'raster_auditory': (0, 2, 2, 1),
        'raster_jaw': (0, 3, 2, 1),
        # SECOND ROW: PSTHs (4 col) - we use columns 0..3, leave col4 for something else or blank
        'psth_all': (2, 0, 1, 1),
        'psth_whisker': (2, 1, 1, 1),
        'psth_auditory': (2, 2, 1, 1),
        'psth_jaw': (2, 3, 1, 1),
        # THIRD ROW: PSTHs differentiating lick/nolick
        'psth_no_stim_lick': (3, 0, 1, 1),
        'psth_whisker_lick': (3, 1, 1, 1),
        'psth_auditory_lick': (3, 2, 1, 1),
        'psth_jaw_lick': (3, 3, 1, 1),
        # FOURTH ROW: active-only quartiles (4 cols)
        #'psth_active_quart_no_stim': (4, 0, 1, 1),
        #'psth_active_quart_whisker': (4, 1, 1, 1),
        #'psth_active_quart_aud': (4, 2, 1, 1),
        #'psth_active_quart_jaw': (4, 3, 1, 1),
        # FIFTH ROW:
        'amp_time': (4, 0, 1, 1),
        'waveform_mean': (4, 1, 1, 1),
        'performance': (4, 2, 1, 1),
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
                        bin_size: float = 0.01):
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
         - 'trial_order' or similar for sorting early->late
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

    # create pdf + figure with GridSpec
    nrows = 5
    ncols = 4
    fig = plt.figure(figsize=(20, 25))
    gs = GridSpec(nrows, ncols, figure=fig, wspace=0.6, hspace=0.7)

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


    # ============= ROW 1: rasters =============
    # Raster all trials aligned at trial start (assume align column 'trial_start_time' or 'trial_start')
    align_col_start = 'start_time'
    ax_raster_all = add_ax_for_key('raster_all')
    if ax_raster_all is not None:
        plot_raster(ax_raster_all, spikes, trials_df, align_col=align_col_start, tmin=tmin, tmax=tmax,
                    sort_by='trial_order' if 'trial_order' in trials_df.columns else None,
                    cmap=type_colors, trial_type_col='trial_type')

    # Raster whisker trials
    ax_raster_whisk = add_ax_for_key('raster_whisker')
    if ax_raster_whisk is not None:
        if 'is_whisker' in trials_df.columns:
            mask = trials_df['is_whisker'].astype(bool)
        else:
            mask = (trials_df.get('trial_type') == 'whisker_trial') if 'trial_type' in trials_df.columns else np.zeros(len(trials_df), dtype=bool)
        mask = (trials_df.get('trial_type') == 'whisker_trial')
        context_cmap = {
            'passive_pre': plutils.adjust_lightness(type_colors['whisker_trial'], 1.5),
            'active': type_colors['whisker_trial'],
            'passive_post': plutils.adjust_lightness(type_colors['whisker_trial'], 0.4)
        }
        plot_raster(ax_raster_whisk, spikes, trials_df, align_col=align_col_start, tmin=tmin, tmax=tmax,
                    sort_by='trial_order' if 'trial_order' in trials_df.columns else None,
                    condition_mask=mask, cmap=type_colors, trial_type_col='trial_type', context_cmap=context_cmap)

    # Raster auditory trials
    ax_raster_aud = add_ax_for_key('raster_auditory')
    if ax_raster_aud is not None:
        if 'is_auditory' in trials_df.columns:
            mask = trials_df['is_auditory'].astype(bool)
        else:
            mask = (trials_df.get('trial_type') == 'auditory_trial') if 'trial_type' in trials_df.columns else np.zeros(len(trials_df), dtype=bool)
        mask = (trials_df.get('trial_type') == 'auditory_trial')
        context_cmap = {
            'passive_pre': plutils.adjust_lightness(type_colors['auditory_trial'], 2.0),
            'active': type_colors['auditory_trial'],
            'passive_post': plutils.adjust_lightness(type_colors['auditory_trial'], 0.6)
        }
        plot_raster(ax_raster_aud, spikes, trials_df, align_col=align_col_start, tmin=tmin, tmax=tmax,
                    sort_by='trial_order' if 'trial_order' in trials_df.columns else None,
                    condition_mask=mask, cmap=type_colors, trial_type_col='trial_type', context_cmap=context_cmap)

    # Raster aligned at jaw onsetY
    align_col_jaw = 'jaw_onset_time' if 'jaw_onset_time' in trials_df.columns else None
    ax_raster_jaw = add_ax_for_key('raster_jaw')
    if ax_raster_jaw is not None:
        if align_col_jaw is None:
            ax_raster_jaw.text(0.5,0.5, "No jaw_onset_time column", ha='center', va='center')
        else:
            plot_raster(ax_raster_jaw, spikes, trials_df, align_col=align_col_jaw, tmin=-0.5, tmax=0.2, # different window for jaw
                        sort_by='trial_order' if 'trial_order' in trials_df.columns else None,
                        cmap=type_colors, trial_type_col='trial_type')

    # ============= ROW 2: PSTHs (by context) =============
    # Color adjustment helper
    def ctx_colors(base_color):
        """Return a dict of context → color with lightness variations."""
        return {
            'passive_pre': plutils.adjust_lightness(base_color, 1.5),
            'passive_post': plutils.adjust_lightness(base_color, 0.6), # darker
            'active': plutils.adjust_lightness(base_color, 1.0)
        }

    # Contexts in preferred plotting order
    context_order = ['passive_pre', 'active', 'passive_post']

    # --- (1) All trials PSTH ---
    ax_psth_all = add_ax_for_key('psth_all')
    if ax_psth_all is not None:
        # Select only active trials
        if 'context' in trials_df.columns:
            active_df = trials_df[trials_df['context'] == 'active'].copy()
        else:
            active_df = trials_df.copy()

        if active_df.empty:
            ax_psth_all.text(0.5, 0.5, "No active trials", ha='center', va='center')
        else:
            # Define trial types and corresponding colors
            trial_types = ['no_stim_trial', 'whisker_trial', 'auditory_trial']
            colors = {
                'no_stim_trial': type_colors.get('no_stim_trial', 'k'),
                'whisker_trial': type_colors.get('whisker_trial', 'C0'),
                'auditory_trial': type_colors.get('auditory_trial', 'C1'),
            }

            # Plot one PSTH per trial type (lick trials only)
            for tt in trial_types:
                df_tt = active_df[(active_df['trial_type'] == tt) & (active_df['lick_flag'] == 1)]
                if df_tt.empty:
                    continue

                plot_psth(
                    ax=ax_psth_all,
                    spikes=spikes,
                    trials_df=df_tt,
                    align_col=align_col_start,
                    tmin=tmin,
                    tmax=tmax,
                    bin_size=bin_size,
                    groupby=None,
                    legend=False,
                    colors=colors[tt],
                    #label=tt.replace('_trial', '').replace('_', ' ').capitalize(),
                )

            # Formatting
            ax_psth_all.axvline(0, color='k', linestyle='--', linewidth=1)
            ax_psth_all.set_xlim(tmin, tmax)
            ax_psth_all.set_xlabel('Time from start (s)')
            ax_psth_all.set_ylabel('Firing rate (spks/s)')
            ax_psth_all.legend(
                frameon=False,
                fontsize=8,
                loc='upper right',
                handlelength=1.5,
            )
            ax_psth_all.set_title("Lick active trials (by trial type)")

    # --- (2) Whisker trials PSTHs ---
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

    # --- (3) Auditory trials PSTHs ---
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

    # --- (4) Jaw-onset aligned PSTHs ---
    ax_psth_jaw = add_ax_for_key('psth_jaw')
    trials_df_jaw = trials_df[(trials_df['lick_flag'] == 1) & (trials_df['context'] == 'active')]
    if ax_psth_jaw is not None:
        if align_col_jaw is None or align_col_jaw not in trials_df_jaw.columns:
            ax_psth_jaw.text(0.5, 0.5, "No jaw_onset_time column", ha='center', va='center')
        else:
            if 'context' in trials_df_jaw.columns:
                plot_psth(
                    ax=ax_psth_jaw,
                    spikes=spikes,
                    trials_df=trials_df_jaw,
                    align_col=align_col_jaw,
                    tmin=-0.5,
                    tmax=0.2,
                    bin_size=bin_size,
                    groupby='trial_type',
                    group_values=type_colors.keys(),
                    colors=type_colors,
                )
            else:
                plot_psth(
                    ax=ax_psth_jaw,
                    spikes=spikes,
                    trials_df=trials_df_jaw,
                    align_col=align_col_jaw,
                    tmin=-0.5,
                    tmax=0.2,
                    bin_size=bin_size,
                    groupby=None
                )
        ax_psth_jaw.set_title("Lick trials (by trial type)")

    # ============= ROW 3: PSTHs differentiating lick/nolick ============
    # Note: in passive there is no lick — we'll handled missing by plotting only if data exists
    def safe_plot_lick(ax_key, df_subset, align_col, title_suffix='', colors=None):
        ax = add_ax_for_key(ax_key)
        if ax is None:
            return
        linestyle_map = {
            True: '-',  # lick / hit
            False: '--',  # no-lick / miss
        }
        if 'lick_flag' in df_subset.columns:
            if 'jaw_onset_time' in align_col:
                tmin, tmax = -0.5, 0.2
            else:
                tmin, tmax = DEFAULT_TMIN, DEFAULT_TMAX

            for lick_flag, linestyle in linestyle_map.items():
                mask = df_subset['lick_flag'] == lick_flag
                if not mask.any():
                    continue

                sub_df = df_subset[mask]
                label = 'Lick' if lick_flag else 'No lick'
                plot_psth(
                    ax=ax,
                    spikes=spikes,
                    trials_df=sub_df,
                    align_col=align_col,
                    tmin=tmin,
                    tmax=tmax,
                    bin_size=bin_size,
                    groupby='lick_flag',
                    group_values=[True, False],
                    colors=colors,
                    linestyle_map=linestyle_map,
                    #groupby=None,  # no grouping within lick_flag subsets
                    legend=True,
                    #colors=colors,  # keep external color scheme (e.g. trial type)
                    #ls=linestyle,
                    label=label
                )
            # Update legend to be lick / no lick
            legend_handles, _ = ax.get_legend_handles_labels()
            ax.legend(legend_handles, ['Lick', 'No lick'], frameon=False, fontsize=8, loc='upper right')
            ax.set_title(title_suffix)
        else:
            ax.text(0.5, 0.5, "No 'lick' column", ha='center', va='center')
        return

    trials_df_active = trials_df[trials_df['context'] == 'active']
    safe_plot_lick('psth_all_lick', trials_df_active, align_col_start, title_suffix='All active trials',
                   colors=type_colors)
    # whisker
    if 'trial_type' in trials_df.columns:
        safe_plot_lick('psth_no_stim_lick', trials_df_active.loc[trials_df_active['trial_type']=='no_stim_trial'], align_col_start, title_suffix='Active no stim.',
                       colors=type_colors)
        safe_plot_lick('psth_whisker_lick', trials_df_active.loc[trials_df_active['trial_type']=='whisker_trial'], align_col_start, title_suffix='Active whisker',
                       colors=type_colors)
        safe_plot_lick('psth_auditory_lick', trials_df_active.loc[trials_df_active['trial_type']=='auditory_trial'], align_col_start, title_suffix='Active auditory',
                       colors=type_colors)
    # jaw aligned
    if align_col_jaw:
        safe_plot_lick('psth_jaw_lick', trials_df_jaw, align_col_jaw, title_suffix='Lick trials (by trial type)')


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
        ax_wf.set_ylabel('Amplitude')
        ax_wf.set_xlim(tmin, tmax)
        n_points = len(waveform_mean)
        ax_wf.set_xticks(np.linspace(0, n_points, 5))
        ax_wf.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, n_points / 30000 * 1000, 5)]) #assumes 30kHz sampling
        ax_wf.set_title('Mean waveform')

    ax_perf = add_ax_for_key('performance')
    if ax_perf is not None:
        plot_performance(ax_perf, trials_df, time_col='start_time', type_colors=type_colors)

    # -----------------------------------
    # Adjust figure, make title, and save
    # -----------------------------------
    fig.align_ylabels()
    _ = make_title_table(fig, metadata)
    filename = f"neuron_{neuron_id}_report.pdf"
    with PdfPages(os.path.join(outpath, filename)) as pdf:
        pdf.savefig()
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

    ephys_dir = pathlib.Path(base_dir) / experimenter / "data" / mouse_id / session_id / "Ephys"

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

    # Look for the kilosort folder (usually 'kilosort2' or 'kilosort3')
    ks_folders = sorted(probe_folder.glob("kilosort*"))
    if not ks_folders:
        raise FileNotFoundError(f"No kilosort folder found in {probe_folder}")
    ks_folder = ks_folders[0]

    # Gather relevant files if they exist
    result = {}
    for fname in ["spike_clusters.npy", "amplitudes.npy", "spike_times.npy", "spike_templates.npy"]:
        fpath = ks_folder / fname
        if fpath.exists():
            result[fname.replace(".npy", "")] = fpath

    return result

# Main function
def generate_unit_spike_report(nwb_file, mouse_res_path, res_path):

    res_path = pathlib.Path(res_path)
    res_path.mkdir(parents=True, exist_ok=True)

    # Get session info
    mouse_id = nwb_reader.get_mouse_id(nwb_file)
    session_id = nwb_reader.get_session_id(nwb_file)
    initials = nwb_reader.get_experimenter(nwb_file)

    # Load trial and unit tables from NWB
    trial_df = nwb_reader.get_trial_table(nwb_file)
    trial_df['mouse_id'] = mouse_id
    trial_df['session_id'] = session_id
    unit_df = nwb_reader.get_unit_table(nwb_file)
    print('Unit cols', unit_df.columns)

    # Define passive pre and passive post
    n_trials = len(trial_df)
    mid_session_trial_idx = n_trials // 2
     # Passive pre are early and passive post are late in session
    mask_pre = (trial_df['context'] == 'passive') & (trial_df.index < mid_session_trial_idx)
    mask_post = (trial_df['context'] == 'passive') & (trial_df.index >= mid_session_trial_idx)
    trial_df.loc[mask_pre, 'context'] = 'passive_pre'
    trial_df.loc[mask_post, 'context'] = 'passive_post'


    # Load jaw onset, merge onto trial_df
    jaw_data_path = os.path.join(mouse_res_path, 'dlc_jaw_onset_times.pkl')
    jaw_onset_df = pd.read_pickle(jaw_data_path)
    trial_df = trial_df.merge(
        jaw_onset_df[['mouse_id', 'session_id', 'trial_id', 'jaw_dlc_onset', 'piezo_lick_time']],
        on=['mouse_id', 'session_id', 'trial_id'], how='left')
    trial_df['jaw_onset_time'] = trial_df['start_time'] + trial_df['jaw_dlc_onset']

    # Filter, format data
    unit_df_subset = unit_df[(unit_df['bc_label']=='good') & (unit_df['firing_rate']>=1)]
    unit_df_subset = neural_utils.convert_electrode_group_object_to_columns(unit_df_subset)


    if initials=='AB':
        experimenter = 'Axel_Bisi'
    elif initials=='MH':
        experimenter = 'Myriam_Hamon'

    for imec_id in unit_df_subset['electrode_group'].unique():
        unit_df_imec = unit_df_subset[unit_df_subset['electrode_group']==imec_id]

        # Get paths to spike clusters and amplitudes
        imec_id = imec_id.split('_')[0]
        ks_paths = find_kilosort_paths(base_dir="M:/analysis", experimenter=experimenter, mouse_id=mouse_id,
                                       session_id=session_id, probe_id=imec_id)

        for idx, row in unit_df_imec.iterrows():

            metadata = {
            'mouse_id':mouse_id,
            'neuron_id':row['neuron_id'],
            'ccf_atlas_acronym':row['ccf_atlas_acronym'],
            'waveform_mean': row['waveform_mean'],
            }

            unit_spikes = row['spike_times']
            cluster_id = int(row['cluster_id'])


            # Get cluster spikes
            spike_clusters = np.load(ks_paths['spike_clusters'])
            spike_clusters = np.array(spike_clusters)
            amplitudes = np.load(ks_paths['amplitudes'])
            amplitudes = np.array(amplitudes)

            spike_indices = np.where(spike_clusters==cluster_id)
            unit_spike_amplitudes = amplitudes[spike_indices]

            generate_neuron_pdf(row['neuron_id'], unit_spikes, unit_spikes, unit_spike_amplitudes, trial_df, metadata,
                                outpath=res_path)


        return
