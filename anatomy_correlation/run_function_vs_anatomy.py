"""
fraction_vs_anatomy.py
──────────────────────
Correlate per-area functional metrics against anatomical variables
(S1 axonal innervation from Liu et al., cortical hierarchy from Harris et al.).

Metrics computed per area × reward_group × encoding_variable:
  ROC
    - fraction_significant      : fraction of units with significant ROC
    - mean_selectivity          : mean ROC selectivity index (signed)
    - mean_abs_selectivity      : mean absolute ROC selectivity index
    (all three split by direction: positive / negative)
  GLM
    - fraction_significant      : fraction of units with significant LRT
    - mean_delta_test_corr      : mean(test_corr_full - test_corr_reduced) / test_corr_full

  Latency (whisker stimulus + lick onset, area-averaged PSTH step-function fit)
    - response_latency_ms       : onset latency per area, correlated vs anatomy
    Vectorized + parallelized across (reward_group, area, event) jobs.
    A stimulus-locked artifact (e.g. magnetic/electrical) in [-5, +5] ms around
    t=0 is removed and replaced with Poisson-simulated spikes drawn from each
    unit's own pre-stimulus baseline rate, so the PSTH is smooth through t=0.

Stratifications applied to every metric:
  - Reward group  : R+ vs R- overlaid | separate | R+−R− difference | pooled
  - Cell type     : RSU (duration >= 0.35 ms) vs FSU (duration < 0.35 ms)
  - Cortical depth: superficial (L1/L2/L3) vs deep (L4/L5/L6)

Pipeline
--------
1.  Load mouse metadata
2.  Load trial_table + unit_table from NWB files
3.  Load ROC or GLM results; merge area labels
4.  keep_shared_areas (good units only, per-RG thresholds + total minimum)
5.  compute_roc_metrics / compute_glm_metrics  per area × stratification
6.  compute_latency_metrics  from area-averaged PSTHs (whisker + lick),
    vectorized PSTH construction, parallel across (rg, area, event)
7.  Merge anatomical variables (Liu log + raw, Harris hierarchy)
8.  plot_metric_vs_anatomy for every metric × anatomical variable
      Fig A : R+ and R- overlaid
      Fig B : R+ and R- separate panels
      Fig C : R+−R− difference vs anatomy
      Fig D : pooled (both RGs)
"""

import os, socket, pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, ndimage

import allen_utils as allen
import neural_utils as nutils
import plotting_utils
from roc_analysis.roc_analysis_utils import load_roc_results, compute_prop_significant

# ── anatomical column names ───────────────────────────────────────────────────
LIU_COL       = 'avg_ipsi_corr'
LIU_RAW_COL   = 'liu_raw'
HIERARCHY_COL = 'cc_tc_ct_iterated'

ANATOMICAL_VARS = {
    HIERARCHY_COL : 'Cortical hierarchy score\n(Harris et al.)',
    LIU_COL       : 'S1 axonal innervation\n(Liu et al., log)',
    LIU_RAW_COL   : 'S1 axonal innervation\n(Liu et al., raw)',
}

RG_COLORS  = {'R+': 'forestgreen', 'R-': 'crimson'}
AREA_COL   = 'ccf_acronym_no_layer'
CCF_PARENT = 'ccf_atlas_parent_acronym'

# ── waveform threshold (RS vs FS) ─────────────────────────────────────────────
# Gentet et al. 2010, Barthó et al. 2004: FS < 0.35 ms trough-to-peak duration
FS_DURATION_THRESHOLD_MS = 0.35

# ── layer groupings ───────────────────────────────────────────────────────────
SUPERFICIAL_LAYERS = {'1', '2', '3', '2/3'}
DEEP_LAYERS        = {'4', '5', '6', '6a', '6b'}

MAX_WORKERS = 29


# ── unit classification ───────────────────────────────────────────────────────

def add_cell_type(unit_table):
    """
    Add 'cell_type' column: 'FSU' if duration < 0.35 ms, else 'RSU'.
    Requires 'duration' column (ms) in unit_table.
    """
    if 'duration' not in unit_table.columns:
        print('[WARN] "duration" column not found; cell_type set to "unknown".')
        unit_table['cell_type'] = 'unknown'
        return unit_table
    unit_table['duration'] = unit_table['duration'].astype(float)
    unit_table['cell_type'] = np.where(
        unit_table['duration'] < FS_DURATION_THRESHOLD_MS, 'FSU', 'RSU')
    n_fsu = (unit_table['cell_type'] == 'FSU').sum()
    n_rsu = (unit_table['cell_type'] == 'RSU').sum()
    print(f'  Cell types: RSU={n_rsu}, FSU={n_fsu} '
          f'(threshold={FS_DURATION_THRESHOLD_MS} ms)')
    return unit_table


def add_layer_group(unit_table):
    """
    Add 'layer_group' column from 'layer_number' column produced by
    allen.process_allen_labels: 'superficial', 'deep', or 'unknown'.
    """
    if 'layer_number' not in unit_table.columns:
        print('[WARN] "layer_number" column not found; layer_group set to "unknown".')
        unit_table['layer_group'] = 'unknown'
        return unit_table

    def _group(layer_val):
        if not isinstance(layer_val, str):
            return 'unknown'
        l = layer_val.strip()
        if l in SUPERFICIAL_LAYERS:
            return 'superficial'
        if l in DEEP_LAYERS:
            return 'deep'
        return 'unknown'

    unit_table['layer_group'] = unit_table['layer_number'].apply(_group)
    print(f'  Layer groups: '
          f'{unit_table["layer_group"].value_counts().to_dict()}')
    return unit_table


# ── area filtering ────────────────────────────────────────────────────────────

def keep_shared_areas(data_df, nomenclature,
                      n_min_units=5, n_min_mice=3, n_min_good_total=50):
    """
    Keep areas present in R+ ∩ R- with sufficient good units and mice.
    All counts use bc_label == 'good' only.
    """
    print(f'Filtering [{nomenclature}] '
          f'>={n_min_units}u / >={n_min_mice}m per RG, '
          f'>={n_min_good_total} total good...')
    n0      = len(data_df)
    data_df = data_df.dropna(subset=[nomenclature])
    if (d := n0 - len(data_df)):
        print(f'  Dropped {d} NaN-area rows.')

    areas_rp = set(data_df[data_df['reward_group'] == 'R+'][nomenclature].unique())
    areas_rm = set(data_df[data_df['reward_group'] == 'R-'][nomenclature].unique())
    cands    = areas_rp & areas_rm
    good     = data_df[data_df['bc_label'] == 'good']

    def _cts(rg):
        s = good[good['reward_group'] == rg]
        return (s.groupby(nomenclature)['unit_id'].nunique(),
                s.groupby(nomenclature)['mouse_id'].nunique())

    u_rp, m_rp = _cts('R+')
    u_rm, m_rm = _cts('R-')
    u_tot = good.groupby(nomenclature)['unit_id'].nunique()

    shared = [a for a in cands
              if (u_rp.get(a,0) >= n_min_units
                  and u_rm.get(a,0) >= n_min_units
                  and m_rp.get(a,0) >= n_min_mice
                  and m_rm.get(a,0) >= n_min_mice
                  and u_tot.get(a,0) >= n_min_good_total)]

    removed = cands - set(shared)
    if removed:
        print(f'  Removed {len(removed)} areas:')
        for a in sorted(removed):
            print(f'    {a}: R+{u_rp.get(a,0)}u/{m_rp.get(a,0)}m '
                  f'R-{u_rm.get(a,0)}u/{m_rm.get(a,0)}m '
                  f'tot={u_tot.get(a,0)}')
    print(f'  Keeping {len(shared)}: {sorted(shared)}')
    return data_df[data_df[nomenclature].isin(shared)], shared


# ── ROC metrics ───────────────────────────────────────────────────────────────

def compute_roc_metrics(roc_df, area_col=AREA_COL):
    """
    Per area × reward_group × analysis_type × direction:
      fraction_significant, mean_selectivity, mean_abs_selectivity.
    """
    rows = []
    for (rg, atype, direction), sub in roc_df.groupby(
            ['reward_group', 'analysis_type', 'direction']):
        for area, asub in sub.groupby(area_col):
            if len(asub) == 0:
                continue
            rows.append({
                'reward_group'        : rg,
                'analysis_type'       : atype,
                'direction'           : direction,
                area_col              : area,
                'fraction_significant': asub['significant'].mean(),
                'mean_selectivity'    : asub['selectivity'].mean(),
                'mean_abs_selectivity': asub['selectivity'].abs().mean(),
                'n_units'             : len(asub),
                'n_mice'              : asub['mouse_id'].nunique(),
            })
    return pd.DataFrame(rows)


# ── GLM metrics ───────────────────────────────────────────────────────────────

def compute_glm_metrics(glm_df, area_col=AREA_COL):
    """
    Per area × reward_group × model_name:
      fraction_significant, mean_delta_test_corr.
    Excludes 'full' model rows.
    """
    sub = glm_df[glm_df['model_name'] != 'full'].copy()
    rows = []
    for (rg, model, area), asub in sub.groupby(
            ['reward_group', 'model_name', area_col]):
        if len(asub) == 0:
            continue
        rows.append({
            'reward_group'        : rg,
            'analysis_type'       : model,
            area_col              : area,
            'fraction_significant': asub['significant'].mean(),
            'mean_delta_test_corr': asub['delta_test_corr'].mean(),
            'n_units'             : len(asub),
            'n_mice'              : asub['mouse_id'].nunique(),
        })
    return pd.DataFrame(rows)


# ── PSTH + latency (vectorized, parallelized, artifact-corrected) ────────────

def _remove_stimulus_artifact(spike_times, artifact_win=(-0.005, 0.005),
                              baseline_win=(-0.2, -0.01), t_ref=0.0,
                              rng=None):
    """
    Remove spikes within `artifact_win` of `t_ref` (absolute time) and
    replace them with Poisson-simulated spikes drawn from the unit's own
    baseline firing rate (computed in `baseline_win`, also relative to
    `t_ref`). This is applied per-trial-alignment by the caller; here
    `spike_times` are already the trial-relative spike times for one trial
    (i.e. spikes - t0), and t_ref = 0.

    Returns a new array of trial-relative spike times.
    """
    if rng is None:
        rng = np.random.default_rng()

    lo, hi = artifact_win
    in_artifact = (spike_times >= lo) & (spike_times <= hi)
    clean = spike_times[~in_artifact]

    # Estimate baseline rate from this trial's own pre-stimulus window
    blo, bhi = baseline_win
    base_dur = bhi - blo
    if base_dur <= 0:
        return clean
    n_base_spikes = np.sum((spike_times >= blo) & (spike_times < bhi))
    base_rate = n_base_spikes / base_dur  # spikes/s

    # Simulate Poisson spikes within the artifact window at baseline rate
    art_dur = hi - lo
    n_sim = rng.poisson(base_rate * art_dur)
    if n_sim > 0:
        sim_spikes = rng.uniform(lo, hi, size=n_sim)
        clean = np.concatenate([clean, sim_spikes])

    return clean


def _build_psth_for_job(spike_times_list, trial_starts, pre_s, post_s,
                        bin_ms, artifact_win, baseline_win, seed):
    """
    Vectorized PSTH construction for one (area, reward_group, event) job,
    with stimulus-artifact correction applied per-trial per-unit.

    Returns (times, psth) — psth is mean firing rate across units, spikes/s.
    """
    rng = np.random.default_rng(seed)

    bin_s  = bin_ms / 1000.0
    edges  = np.arange(-pre_s, post_s + bin_s, bin_s)
    times  = (edges[:-1] + edges[1:]) / 2
    n_bins = len(times)

    n_units  = len(spike_times_list)
    n_trials = len(trial_starts)
    if n_units == 0 or n_trials == 0:
        return times, np.full(n_bins, np.nan)

    counts = np.zeros((n_units, n_bins))

    for ui, spikes in enumerate(spike_times_list):
        spikes = np.asarray(spikes)
        if spikes.size == 0:
            continue

        # Vectorized trial-relative spike extraction:
        # for each trial start, find spikes within [t0-pre_s, t0+post_s]
        # using searchsorted on sorted spike times (fast, no python loop
        # over individual spikes).
        sorted_spikes = np.sort(spikes)

        all_rel = []
        for t0 in trial_starts:
            lo_t = t0 - pre_s
            hi_t = t0 + post_s
            i0 = np.searchsorted(sorted_spikes, lo_t, side='left')
            i1 = np.searchsorted(sorted_spikes, hi_t, side='right')
            if i1 <= i0:
                continue
            rel = sorted_spikes[i0:i1] - t0
            # artifact correction (per trial, per unit)
            rel = _remove_stimulus_artifact(
                rel, artifact_win=artifact_win,
                baseline_win=baseline_win, rng=rng)
            all_rel.append(rel)

        if all_rel:
            rel_concat = np.concatenate(all_rel)
            h, _ = np.histogram(rel_concat, bins=edges)
            counts[ui] = h

    rate = counts / (n_trials * bin_s)
    psth = rate.mean(axis=0)
    return times, psth


def fit_step_latency(times, psth, baseline_win=(-0.2, 0.0),
                     smooth_sigma_ms=20, bin_ms=10,
                     threshold_sd=3.0):
    """
    Estimate response onset latency by fitting a step function.

    Strategy:
      1. Smooth PSTH with a Gaussian kernel.
      2. Compute baseline mean ± SD in the pre-stimulus window.
      3. Latency = first bin post-stimulus where smoothed PSTH exceeds
                   baseline_mean + threshold_sd * baseline_sd,
                   sustained for at least 2 consecutive bins.

    Returns latency in ms, or np.nan if no crossing found.
    """
    sigma_bins = (smooth_sigma_ms / bin_ms)
    smoothed   = ndimage.gaussian_filter1d(psth.astype(float), sigma=sigma_bins)

    base_mask  = (times >= baseline_win[0]) & (times < baseline_win[1])
    if base_mask.sum() < 3:
        return np.nan
    base_mean = smoothed[base_mask].mean()
    base_sd   = smoothed[base_mask].std()
    if base_sd == 0:
        return np.nan

    threshold  = base_mean + threshold_sd * base_sd
    post_mask  = times >= 0
    post_times = times[post_mask]
    post_vals  = smoothed[post_mask]

    above = post_vals > threshold
    for i in range(len(above) - 1):
        if above[i] and above[i+1]:
            return float(post_times[i]) * 1000.0
    return np.nan


def _latency_job(job):
    """
    Worker function for one (reward_group, area, event) job.
    job is a dict with keys:
      rg, area, event, spike_times_list, trial_starts,
      pre_s, post_s, bin_ms, smooth_sigma_ms, threshold_sd,
      artifact_win, baseline_win, seed
    Returns a result dict (one row) or None if too few trials/units.
    """
    if job['spike_times_list'] is None or len(job['spike_times_list']) == 0:
        return None
    if len(job['trial_starts']) < 5:
        return None

    times, psth = _build_psth_for_job(
        job['spike_times_list'], job['trial_starts'],
        pre_s=job['pre_s'], post_s=job['post_s'], bin_ms=job['bin_ms'],
        artifact_win=job['artifact_win'], baseline_win=job['baseline_win'],
        seed=job['seed'])

    latency = fit_step_latency(
        times, psth,
        baseline_win=(-job['pre_s'], 0.0),
        smooth_sigma_ms=job['smooth_sigma_ms'],
        bin_ms=job['bin_ms'],
        threshold_sd=job['threshold_sd'])

    return {
        AREA_COL              : job['area'],
        'reward_group'        : job['rg'],
        'event'               : job['event'],
        'response_latency_ms' : latency,
        'n_units'             : len(job['spike_times_list']),
    }


def compute_latency_metrics(unit_table, trial_table,
                            area_col=AREA_COL,
                            pre_s=0.2, post_s=0.6,
                            bin_ms=10, smooth_sigma_ms=20,
                            threshold_sd=3.0,
                            artifact_win=(-0.005, 0.005),
                            artifact_baseline_win=(-0.2, -0.01),
                            max_workers=MAX_WORKERS):
    """
    Compute response onset latency per area for two events:
      1. Whisker stimulus onset  (trial_type == 'whisker_trial')
      2. Lick onset              (lick_flag in [0, 1])

    Vectorized PSTH construction (searchsorted-based spike extraction per
    trial instead of full-array spike subtraction) and parallelized across
    (reward_group, area, event) jobs using ProcessPoolExecutor.

    A stimulus-locked artifact in `artifact_win` (default ±5 ms around the
    alignment time) is removed and replaced with Poisson-simulated spikes
    drawn from each unit's own per-trial baseline rate (estimated in
    `artifact_baseline_win`), so the PSTH is smooth through t=0.

    unit_table must have: unit_id, mouse_id, reward_group, area_col,
                          spike_times (array or list).
    trial_table must have: trial_type, lick_flag,
                           start_time (alignment time in seconds).

    Returns a DataFrame with columns:
      area_col, reward_group, event, response_latency_ms, n_units.
    """
    events = {
        'whisker_stimulus': trial_table['trial_type'] == 'whisker_trial',
        'lick_onset'      : trial_table['lick_flag'].isin([0, 1]),
    }

    # Build job list
    jobs = []
    seed_counter = 0
    for (rg, area), area_units in unit_table.groupby(['reward_group', area_col]):
        if len(area_units) == 0:
            continue
        spike_times_list = [
            np.asarray(row['spike_times'])
            for _, row in area_units.iterrows()
            if 'spike_times' in area_units.columns
               and row['spike_times'] is not None
        ]
        if len(spike_times_list) == 0:
            continue

        for event_label, trial_mask in events.items():
            trial_starts = trial_table.loc[trial_mask, 'start_time'].values
            jobs.append({
                'rg'               : rg,
                'area'             : area,
                'event'            : event_label,
                'spike_times_list' : spike_times_list,
                'trial_starts'     : trial_starts,
                'pre_s'            : pre_s,
                'post_s'           : post_s,
                'bin_ms'           : bin_ms,
                'smooth_sigma_ms'  : smooth_sigma_ms,
                'threshold_sd'     : threshold_sd,
                'artifact_win'     : artifact_win,
                'baseline_win'     : artifact_baseline_win,
                'seed'             : seed_counter,
            })
            seed_counter += 1

    print(f'  {len(jobs)} latency jobs (area × reward_group × event)...')

    rows = []
    if max_workers and max_workers > 1:
        with ProcessPoolExecutor(max_workers=25) as executor:
            futures = {executor.submit(_latency_job, job): job for job in jobs}
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    rows.append(res)
    else:
        for job in jobs:
            res = _latency_job(job)
            if res is not None:
                rows.append(res)

    return pd.DataFrame(rows)


# ── anatomical merge ──────────────────────────────────────────────────────────

def merge_anatomy(df, area_col=AREA_COL):
    """Merge Liu (raw + log-transformed) and Harris hierarchy onto df."""
    data_areas = df[area_col].dropna().unique()

    liu_areas = allen.load_liu_et_al_avg_ipsi()
    print(f'  Liu:    {len(liu_areas)} areas | data: {len(data_areas)} | '
          f'intersect: {len(set(liu_areas.keys()) & set(data_areas))}')
    df = allen.merge_liu_avg_ipsi_opt(df, cols_priority=area_col)
    df[LIU_RAW_COL] = df[LIU_COL].copy()
    df[LIU_COL]     = np.log(df[LIU_COL] + 1e-5)

    harris_df    = allen.load_process_hierarchy_from_harris()
    harris_areas = harris_df[area_col].dropna().unique()
    print(f'  Harris: {len(harris_areas)} areas | data: {len(data_areas)} | '
          f'intersect: {len(set(harris_areas) & set(data_areas))}')
    df = allen.merge_hierarchy_from_harris(df, merge_on=area_col)
    return df


# ── plotting ──────────────────────────────────────────────────────────────────

def _fmt_p(p):
    return 'p<0.001' if p < 0.001 else f'p={p:.3f}'


def _draw_scatter(ax, sub, anat_col, metric_col, color, label=None):
    """Scatter + OLS regression. Returns (rP,pP,rS,pS) or None."""
    sub = sub.dropna(subset=[anat_col, metric_col])
    sub = sub[sub[metric_col] != 0]
    if len(sub) < 3:
        return None
    x, y  = sub[anat_col].values, sub[metric_col].values
    sizes = 35 + 12 * np.log1p(sub['n_units'].values)
    ax.scatter(x, y, c=color, s=sizes, alpha=0.75,
               edgecolors='white', linewidths=0.4, zorder=3, label=label)
    for _, row in sub.iterrows():
        ax.annotate(row[AREA_COL],
                    xy=(row[anat_col], row[metric_col]),
                    fontsize=5, alpha=0.5,
                    xytext=(3, 2), textcoords='offset points')
    sl, ic, rP, pP, _ = stats.linregress(x, y)
    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, sl*xl + ic, color=color, lw=1.5, ls='--', alpha=0.85, zorder=2)
    rS, pS = stats.spearmanr(x, y)
    return rP, pP, rS, pS


def _style_ax(ax, sub_list, metric_col, title, xlabel, ylabel):
    ax.set_title(title, fontsize=7.5, pad=4)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    plotting_utils.remove_top_right_frame(ax)
    all_y = pd.concat([s.dropna(subset=[metric_col])[metric_col]
                       for s in sub_list if len(s)], ignore_index=True)
    all_y = all_y[all_y != 0]
    if len(all_y):
        ax.set_ylim(min(0, all_y.min()*1.15), all_y.max()*1.15)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')


def plot_metric_vs_anatomy(metrics_df, anat_col, x_label,
                           metric_col, y_label,
                           model_name_dict, saving_path, tag=''):
    """
    Four figures per metric × anatomical variable:
      A : R+ and R- overlaid
      B : R+ and R- separate panels
      C : R+−R− difference vs anatomy
      D : pooled (both RGs)
    """
    os.makedirs(saving_path, exist_ok=True)
    models = [m for m in model_name_dict
              if m in metrics_df['analysis_type'].unique()]
    if not models:
        return

    n_models   = len(models)
    n_cols     = min(4, n_models)
    n_rows     = int(np.ceil(n_models / n_cols))
    ps         = 3.5

    fig_A, axes_A = plt.subplots(n_rows, n_cols,
                                  figsize=(ps*n_cols, ps*n_rows), dpi=300)
    fig_B, axes_B = plt.subplots(n_rows, n_cols*2,
                                  figsize=(ps*n_cols*2, ps*n_rows), dpi=300)
    axes_A = np.array(axes_A).flatten()
    axes_B = np.array(axes_B).flatten()

    for idx, model in enumerate(models):
        disp = model_name_dict.get(model, model)

        ax_a, lines, subs = axes_A[idx], [], []
        for rg in ['R+', 'R-']:
            s = metrics_df[(metrics_df['analysis_type'] == model) &
                           (metrics_df['reward_group']  == rg)]
            subs.append(s)
            res = _draw_scatter(ax_a, s, anat_col, metric_col,
                                RG_COLORS[rg], label=rg)
            if res:
                rP,pP,rS,pS = res
                lines.append(f"{rg} r_P={rP:.2f}({_fmt_p(pP)}) "
                             f"r_S={rS:.2f}({_fmt_p(pS)})")
        ax_a.legend(fontsize=6, frameon=False)
        _style_ax(ax_a, subs, metric_col,
                  title=f"{disp}\n"+"\n".join(lines),
                  xlabel=x_label, ylabel=y_label)

        for ri, rg in enumerate(['R+', 'R-']):
            ax_b = axes_B[idx*2+ri]
            s    = metrics_df[(metrics_df['analysis_type'] == model) &
                              (metrics_df['reward_group']  == rg)]
            res  = _draw_scatter(ax_b, s, anat_col, metric_col, RG_COLORS[rg])
            corr = ''
            if res:
                rP,pP,rS,pS = res
                corr = f"r_P={rP:.2f} ({_fmt_p(pP)})\nr_S={rS:.2f} ({_fmt_p(pS)})"
            _style_ax(ax_b, [s], metric_col,
                      title=f"{disp} — {rg}\n{corr}",
                      xlabel=x_label, ylabel=y_label)

    for i in range(n_models, len(axes_A)): axes_A[i].set_visible(False)
    for i in range(n_models*2, len(axes_B)): axes_B[i].set_visible(False)

    for fig, ftag in [(fig_A,'overlaid'), (fig_B,'separate')]:
        fig.suptitle(f'{y_label} vs {x_label.split(chr(10))[0]} '
                     f'[{ftag}]{tag}', fontsize=9, y=1.01)
        fig.tight_layout()
        plotting_utils.save_figure_with_options(
            fig, ['png','pdf','svg'],
            f'{metric_col}_vs_{anat_col}_{ftag}{tag}',
            saving_path, dark_background=False)
        plt.close(fig)

    anat_map = (metrics_df[[AREA_COL, anat_col]]
                .dropna(subset=[anat_col]).drop_duplicates(AREA_COL)
                .set_index(AREA_COL)[anat_col])
    rp = (metrics_df[metrics_df['reward_group']=='R+']
          .set_index(['analysis_type', AREA_COL])[metric_col])
    rm = (metrics_df[metrics_df['reward_group']=='R-']
          .set_index(['analysis_type', AREA_COL])[metric_col])
    diff_df = (rp - rm).dropna().reset_index()
    diff_df.columns = ['analysis_type', AREA_COL, metric_col]
    n_map = (metrics_df.groupby(['analysis_type', AREA_COL])['n_units']
             .sum().reset_index())
    diff_df = diff_df.merge(n_map, on=['analysis_type', AREA_COL], how='left')
    diff_df[anat_col] = diff_df[AREA_COL].map(anat_map)

    fig_C, axes_C = plt.subplots(n_rows, n_cols,
                                  figsize=(ps*n_cols, ps*n_rows), dpi=300)
    axes_C = np.array(axes_C).flatten()
    for idx, model in enumerate(models):
        disp = model_name_dict.get(model, model)
        ax_c = axes_C[idx]
        s    = diff_df[diff_df['analysis_type']==model].dropna(subset=[anat_col])
        res  = _draw_scatter(ax_c, s, anat_col, metric_col, 'slateblue')
        corr = ''
        if res:
            rP,pP,rS,pS = res
            corr = f"r_P={rP:.2f} ({_fmt_p(pP)})\nr_S={rS:.2f} ({_fmt_p(pS)})"
        _style_ax(ax_c, [s], metric_col,
                  title=f"{disp} (R+−R−)\n{corr}",
                  xlabel=x_label, ylabel=f'Δ {y_label} (R+−R−)')
        ax_c.axhline(0, color='k', lw=0.7, ls=':', zorder=1)
    for i in range(n_models, len(axes_C)): axes_C[i].set_visible(False)
    fig_C.suptitle(f'Δ{y_label} (R+−R−) vs '
                   f'{x_label.split(chr(10))[0]}{tag}', fontsize=9, y=1.01)
    fig_C.tight_layout()
    plotting_utils.save_figure_with_options(
        fig_C, ['png','pdf','svg'],
        f'{metric_col}_vs_{anat_col}_RG_diff{tag}',
        saving_path, dark_background=False)
    plt.close(fig_C)

    pooled = (metrics_df.groupby(['analysis_type', AREA_COL])
              .agg(**{metric_col: (metric_col,'mean'),
                      'n_units'  : ('n_units','sum')})
              .reset_index())
    pooled[anat_col] = pooled[AREA_COL].map(anat_map)

    fig_D, axes_D = plt.subplots(n_rows, n_cols,
                                  figsize=(ps*n_cols, ps*n_rows), dpi=300)
    axes_D = np.array(axes_D).flatten()
    for idx, model in enumerate(models):
        disp = model_name_dict.get(model, model)
        ax_d = axes_D[idx]
        s    = pooled[pooled['analysis_type']==model].dropna(subset=[anat_col])
        res  = _draw_scatter(ax_d, s, anat_col, metric_col, 'steelblue')
        corr = ''
        if res:
            rP,pP,rS,pS = res
            corr = f"r_P={rP:.2f} ({_fmt_p(pP)})\nr_S={rS:.2f} ({_fmt_p(pS)})"
        _style_ax(ax_d, [s], metric_col,
                  title=f"{disp} (pooled)\n{corr}",
                  xlabel=x_label, ylabel=y_label)
    for i in range(n_models, len(axes_D)): axes_D[i].set_visible(False)
    fig_D.suptitle(f'{y_label} (pooled) vs '
                   f'{x_label.split(chr(10))[0]}{tag}', fontsize=9, y=1.01)
    fig_D.tight_layout()
    plotting_utils.save_figure_with_options(
        fig_D, ['png','pdf','svg'],
        f'{metric_col}_vs_{anat_col}_pooled{tag}',
        saving_path, dark_background=False)
    plt.close(fig_D)

    metrics_df.to_csv(
        os.path.join(saving_path, f'{metric_col}_vs_{anat_col}{tag}.csv'),
        index=False)


def run_all_plots(metrics_df, model_name_dict, result_label,
                  FIGURE_PATH, tag=''):
    """Loop over every metric × anatomical variable."""
    all_metrics = {
        'fraction_significant' : 'Fraction significant',
        'mean_selectivity'     : 'Mean selectivity (signed)',
        'mean_abs_selectivity' : 'Mean |selectivity|',
        'mean_delta_test_corr' : 'Mean Δtest corr (full−reduced)/full',
        'response_latency_ms'  : 'Response onset latency (ms)',
    }
    present = {k: v for k,v in all_metrics.items()
               if k in metrics_df.columns}

    for metric_col, y_label in present.items():
        for anat_col, x_label in ANATOMICAL_VARS.items():
            if anat_col not in metrics_df.columns:
                continue
            if metrics_df[anat_col].isna().all():
                continue
            out = str(FIGURE_PATH / result_label / anat_col / metric_col)

            if 'direction' in metrics_df.columns:
                for direction in metrics_df['direction'].dropna().unique():
                    s = metrics_df[metrics_df['direction']==direction].copy()
                    print(f'  {metric_col} | {anat_col} | {direction}{tag}')
                    plot_metric_vs_anatomy(s, anat_col, x_label,
                                           metric_col, y_label,
                                           model_name_dict, out,
                                           tag=f'{tag}_{direction}')
            else:
                print(f'  {metric_col} | {anat_col}{tag}')
                plot_metric_vs_anatomy(metrics_df, anat_col, x_label,
                                       metric_col, y_label,
                                       model_name_dict, out, tag=tag)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ====================================================================
    # USER SETTINGS
    # ====================================================================
    MODE             = 'roc'
    GLM_EXPERIMENTER = 'Myriam_Hamon'
    GIT_VERSION      = '1b14083'

    N_UNITS_MIN            = 5
    N_MICE_PER_AREA_MIN    = 2
    N_MIN_GOOD_UNITS_TOTAL = 10

    RUN_LATENCY = True   # set False to skip PSTH latency computation
    # ====================================================================

    hostname = socket.gethostname()
    if 'haas' in hostname:
        DATA_PATH       = pathlib.Path('/mnt/lsens-analysis/')
        AXEL_NWB_PATH   = pathlib.Path('/mnt/lsens-analysis/Axel_Bisi/NWB_combined')
        MYRIAM_NWB_PATH = pathlib.Path('/mnt/lsens-analysis/Myriam_Hamon/NWB')
        FIGURE_PATH     = pathlib.Path('/mnt/lsens-analysis/Axel_Bisi/combined_results/fraction_vs_anatomy')
        INFO_PATH       = pathlib.Path('/mnt/share_internal/Axel_Bisi_Share/dataset_info')
    else:
        DATA_PATH       = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis')
        AXEL_NWB_PATH   = pathlib.Path(r'M:\analysis\Axel_Bisi\NWB_combined')
        MYRIAM_NWB_PATH = pathlib.Path(r'M:\analysis\Myriam_Hamon\NWB')
        FIGURE_PATH     = pathlib.Path(r'M:\analysis\Axel_Bisi\combined_results\fraction_vs_anatomy')
        INFO_PATH       = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\share_internal\Axel_Bisi_Share\dataset_info')

    os.makedirs(FIGURE_PATH, exist_ok=True)

    # ── mouse metadata ────────────────────────────────────────────────────────
    print('Loading mouse metadata...')
    mouse_info_df = pd.read_excel(INFO_PATH / 'joint_mouse_reference_weight.xlsx')
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df = mouse_info_df[
        (mouse_info_df['exclude']       == 0) &
        (mouse_info_df['exclude_ephys'] == 0) &
        (mouse_info_df['reward_group'].isin(['R+', 'R-'])) &
        (mouse_info_df['recording']     == 1)]
    valid_mice = mouse_info_df['mouse_id'].unique()

    # ── NWB loading ───────────────────────────────────────────────────────────
    print('Loading NWB files...')
    nwb_list = []
    for m in valid_mice:
        path = (AXEL_NWB_PATH   if m.startswith('AB') else
                MYRIAM_NWB_PATH if m.startswith('MH') else None)
        if path:
            nwb_list.extend(str(path/f) for f in os.listdir(path) if m in f)

    # combine_ephys_nwb returns (trial_table, unit_table, ephys_nwb_list)
    trial_table, unit_table, ephys_nwb_list = nutils.combine_ephys_nwb(
        nwb_list, max_workers=MAX_WORKERS)
    unit_table['reward_group'] = unit_table['reward_group'].map({1:'R+',0:'R-'})

    # ── unit table: filter + enrich ───────────────────────────────────────────
    unit_table = unit_table[unit_table['bc_label'] == 'good']
    unit_table = allen.process_allen_labels(unit_table, subdivide_areas=True)
    unit_table = add_cell_type(unit_table)    # uses 'duration', threshold 0.35 ms
    unit_table = add_layer_group(unit_table)  # uses 'layer_number' from process_allen_labels
    print(f'  {len(unit_table)} good units | {unit_table["mouse_id"].nunique()} mice')

    UNIT_COLS = ['mouse_id', 'neuron_id', 'unit_id', 'bc_label',
                 'area_acronym_custom', AREA_COL, CCF_PARENT,
                 'cell_type', 'layer_group', 'reward_group']

    # ── load metric data ──────────────────────────────────────────────────────
    if MODE == 'roc':
        print('Loading ROC results...')
        roc_df = load_roc_results(DATA_PATH / 'Axel_Bisi' / 'combined_results',
                                  max_workers=MAX_WORKERS)
        roc_df = roc_df[roc_df['mouse_id'].isin(unit_table['mouse_id'].unique())]
        roc_df = roc_df.merge(mouse_info_df[['mouse_id','reward_group']],
                              on='mouse_id', how='left')
        roc_df = roc_df.drop(
            columns=list(set(UNIT_COLS)-{'mouse_id','neuron_id'}), errors='ignore')
        roc_df = roc_df.merge(unit_table[UNIT_COLS],
                              on=['mouse_id','neuron_id'], how='left')
        roc_df, _ = keep_shared_areas(
            roc_df, AREA_COL,
            n_min_units=N_UNITS_MIN,
            n_min_mice=N_MICE_PER_AREA_MIN,
            n_min_good_total=N_MIN_GOOD_UNITS_TOTAL)

        result_label    = 'roc'
        compute_metrics = compute_roc_metrics
        src_df          = roc_df
        MODEL_NAME_DICT = {
            'whisker_active'               : 'Whisker (active)',
            'auditory_active'              : 'Auditory (active)',
            'wh_vs_aud_active'             : 'Modality selective',
            'choice'                       : 'Choice',
            'whisker_choice'               : 'Whisker choice',
            'baseline_choice'              : 'Baseline choice',
            'whisker_passive_pre'          : 'Whisker passive pre',
            'whisker_passive_post'         : 'Whisker passive post',
            'auditory_passive_pre'         : 'Auditory passive pre',
            'auditory_passive_post'        : 'Auditory passive post',
            'whisker_pre_vs_post_learning' : 'Whisker pre->post',
            'auditory_pre_vs_post_learning': 'Auditory pre->post',
            'spontaneous_licks'            : 'Spontaneous licks',
        }

    elif MODE == 'glm':
        print(f'Loading GLM results (git={GIT_VERSION})...')
        glm_data_path = DATA_PATH / GLM_EXPERIMENTER / 'combined_results'
        glm_dfs = []
        for mouse_id in valid_mice:
            fpath = (glm_data_path / mouse_id / 'whisker_0' / 'unit_glm'
                     / GIT_VERSION
                     / f'summary_{mouse_id}_unit_glm_{GIT_VERSION}.parquet')
            if not fpath.exists():
                print(f'  [WARN] {fpath}')
                continue
            glm_dfs.append(pd.read_parquet(fpath))
        if not glm_dfs:
            raise FileNotFoundError('No GLM files found.')

        glm_df = pd.concat(glm_dfs, ignore_index=True)
        glm_df['significant']  = ((glm_df['test_corr'] > 0.2)
                                   & glm_df['lrt_significant'])
        glm_df['reward_group'] = glm_df['reward_group'].map({1:'R+', 0:'R-'})

        # delta_test_corr = (test_corr_full - test_corr_reduced) / test_corr_full
        full_corr = (glm_df[glm_df['model_name'] == 'full']
                     .drop_duplicates(subset=['mouse_id', 'neuron_id', 'reward_group'])
                     .set_index(['mouse_id', 'neuron_id', 'reward_group'])['test_corr']
                     .rename('test_corr_full'))
        glm_df = glm_df.join(full_corr, on=['mouse_id', 'neuron_id', 'reward_group'])
        glm_df['delta_test_corr'] = (
            (glm_df['test_corr_full'] - glm_df['test_corr']) / glm_df['test_corr_full'])

        glm_df = glm_df.drop(
            columns=list(set(UNIT_COLS)-{'mouse_id','neuron_id'}), errors='ignore')
        glm_df = glm_df.merge(unit_table[UNIT_COLS],
                              on=['mouse_id','neuron_id'], how='left')
        glm_df, _ = keep_shared_areas(
            glm_df, AREA_COL,
            n_min_units=N_UNITS_MIN,
            n_min_mice=N_MICE_PER_AREA_MIN,
            n_min_good_total=N_MIN_GOOD_UNITS_TOTAL)

        result_label    = f'glm_{GIT_VERSION}'
        compute_metrics = compute_glm_metrics
        src_df          = glm_df
        MODEL_NAME_DICT = ({
            'auditory_encoding'        : 'Auditory stimulus',
            'whisker_encoding'         : 'Whisker stimulus',
            'jaw_onset_encoding'       : 'Lick initiation',
            'reward_encoding'          : 'Reward time',
            'motor_encoding'           : 'Orofacial motion',
            'pupil_area'               : 'Pupil area',
            'time_since_whisker_reward': 'Whisker reward recency',
            'block_perf_type'          : 'High/low performance',
            'session_progress_encoding': 'Trial index',
        } if GIT_VERSION == '1b14083' else {
            'auditory_encoding'    : 'Auditory stimulus',
            'whisker_encoding'     : 'Whisker stimulus',
            'jaw_onset_encoding'   : 'Lick initiation',
            'motor_encoding'       : 'Orofacial motion',
            'last_whisker_reward'  : 'Prev. whisker rewarded',
            'prev_success'         : 'Previous trial success',
            'block_perf_type'      : 'High/low performance',
            'sum_rewards'          : 'Cumulative rewards',
            'whisker_reward_rate_5': 'Perf. last 5 whisker trials',
        })
    else:
        raise ValueError(f'Unknown MODE {MODE}')

    # ── latency metrics ───────────────────────────────────────────────────────
    if RUN_LATENCY:
        print('\nComputing PSTH latencies (vectorized, parallel, '
              'artifact-corrected)...')
        latency_df = compute_latency_metrics(
            unit_table  = unit_table,
            trial_table = trial_table,
            area_col    = AREA_COL,
            pre_s=0.2, post_s=0.6, bin_ms=10,
            smooth_sigma_ms=20, threshold_sd=3.0,
            artifact_win=(-0.005, 0.005),
            artifact_baseline_win=(-0.2, -0.01),
            max_workers=MAX_WORKERS)
        latency_df = merge_anatomy(latency_df, area_col=AREA_COL)
        lat_out = FIGURE_PATH / result_label / 'latency'
        os.makedirs(lat_out, exist_ok=True)
        latency_df.to_csv(lat_out / 'response_latency.csv', index=False)

        for event in latency_df['event'].unique():
            ev_df = latency_df[latency_df['event'] == event].copy()
            ev_df['analysis_type'] = event
            for anat_col, x_label in ANATOMICAL_VARS.items():
                if anat_col not in ev_df.columns or ev_df[anat_col].isna().all():
                    continue
                plot_metric_vs_anatomy(
                    ev_df, anat_col, x_label,
                    'response_latency_ms', 'Response onset latency (ms)',
                    {event: event}, str(lat_out), tag=f'_{event}')

    # ── stratified runs ───────────────────────────────────────────────────────
    RUNS = [
        ('',             None),
        ('_RSU',         'cell_type == "RSU"'),
        ('_FSU',         'cell_type == "FSU"'),
        ('_superficial', 'layer_group == "superficial"'),
        ('_deep',        'layer_group == "deep"'),
    ]

    for run_tag, query in RUNS:
        print(f'\n=== Stratification: {run_tag or "all units"} ===')
        sub_df = src_df.query(query) if query else src_df
        if len(sub_df) == 0:
            print('  [SKIP] no rows.')
            continue

        metrics_df = compute_metrics(sub_df, area_col=AREA_COL)
        print(f'  {len(metrics_df)} area×RG×variable rows')

        print('  Merging anatomy...')
        metrics_df = merge_anatomy(metrics_df, area_col=AREA_COL)

        run_all_plots(metrics_df, MODEL_NAME_DICT,
                      result_label, FIGURE_PATH, tag=run_tag)

    print('\nDone.')