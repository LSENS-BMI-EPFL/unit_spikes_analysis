# rastermap_utils.py
# Shared utilities for build_feature_matrix.py and run_clustering.py.
# Import with:  from rastermap_utils import *

from __future__ import annotations

from locale import normalize
from pathlib import Path
from typing import Any

import os
import re
import socket
import numpy as np
import pandas as pd
import matplotlib
import yaml
import cmasher as cmr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from rastermap import Rastermap
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import fisher_exact, kruskal, mannwhitneyu
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import plotting_utils

# ── config ─────────────────────────────────────────────────────────────────────
# TODO: Implement trad spectral emb/GMM approach and compare clustering / have it in different script for all the methods
# TODO: order of clustering kmeans/GMMs applied on ordered rastermap_psth cmatrix
# TODO: fix missing jaw onset files / missing pre / missing post data
# TODO: make axon cmap logscale too? or data as logscale? not mean over neurons? but areas?
# TODO: average of neurons, mice, areas,geometric mean for anatomical axes?


hostname = socket.gethostname()
if 'haas' in hostname:
    N_WORKERS = 100
    ANALYSIS_ROOT_PATH = '/mnt/lsens-analysis'
    INFO_ROOT_PATH = '/mnt/share_internal'
else:
    N_WORKERS = 30
    ANALYSIS_ROOT_PATH = r'\\sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis'
    INFO_ROOT_PATH = r'\\sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal'

MOUSE_INFO = os.path.join(INFO_ROOT_PATH, 'Axel_Bisi_Share', 'dataset_info', 'joint_mouse_reference_weight.xlsx')
WAVEFORM_ANALYSIS_ROOT = os.path.join(ANALYSIS_ROOT_PATH, 'Axel_Bisi', 'combined_results')


DEFAULT_CFG: dict[str, Any] = dict(
    recompute_feature_matrix = True,
    n_min_trial_per_condition = 5, # trial count for spit-trials
    period                = 'active_passive', # "active_passive", "passive"', "active"
    t_pre_passive        = 0.2,    # pre-stimulus window for passive conditions (s)
    t_post_passive       = 0.5,    # post-stimulus window for passive conditions (s)
    t_pre_active         = 0.2,    # pre-stimulus window for active conditions (s)
    t_post_active        = 0.5,    # post-stimulus window for active conditions (s)
    t_pre_jaw            = 0.35,   # pre-jaw-onset window (s)
    t_post_jaw           = 0.35,   # post-jaw-onset window (s)
    bin_ms               = 10,
    sigma_ms             = 5,
    artifact_win_s       = (-0.005, 0.005),
    whisker_trial_label  = "whisker_trial",
    global_fr_hz         = 0.01,
    fr_threshold_hz      = 0.2,
    square_fr            = False,
    baseline_removal     = False,   # if False: skip all per-trial baseline subtraction (own-window AND jaw-borrowed), for each neuron and trial
    include_lick_conditions = True,  # if True: add spontaneous_lick + reward_lick conditions (requires lick_df passed to run_rastermap_psth)
    lick_quiet_window_s     = 0.2,   # duration (s) of quiet baseline window ending at closest preceding trial start_time, for lick_time conditions
    reward_group_col     = "reward_group",
    area_col             = "area_acronym_custom",
    normalize            = "zscore", #"zscore" or "baseline"
    zscore_full          = True,
    context_col          = "context",
    trial_type_col       = "trial_type",
    mouse_id_col         = "mouse_id",
    session_id_col       = "session_id",
    n_rastermap_clusters = 20,
    locality             = 0.75,
    time_lag_window      = 20,
    grid_upsample        = 0,
    k_means_k            = 25,
    k_elbow_range        = range(2, 30),
    umap_n_neighbors     = 20,
    umap_min_dist        = 0.15,
    vmax_pct             = 75,
    n_sample_neurons     = 24,
    n_jobs               = N_WORKERS,
    stride_ms            = 5,
    modality             = "whisker_auditory",    # "both" | "whisker" | "auditory"
    reward_filter        = "combined",    # "both" | "R+" | "R-"
    n_example_neurons    = 30,
    example_alpha        = 0.5,
    use_wf_classification_csv = True, # if True: load RSU(WW)/FSU(NW) labels from per-mouse waveform_analysis CSV instead of duration-percentile split
    do_global       = True,     # if True: fit rastermap_psth on all trials
    cross_validate       = True,     # if True: fit rastermap_psth on odd trials, evaluate on even
    cv_zscore_independent = True,    # if True: z-score odd and even trials independently; if False: fit normaliser on odd, apply to even
    # Anatomical axis annotations (matrix side columns + per-cluster stats; {standard_name: actual_column_name_in_units_good}).
    anatomy_score_cols   = {
    "avg_ipsi":      "avg_ipsi",       # ← right side = actual column name in unit_table
    "cc_tc_ct_iterated": "cc_tc_ct_iterated",
    "cc_hierarchy_score_columns":    "cc_hierarchy_score_columns",
        },
)

layer_number_mapper = {
    '1':'supragranular',
    '2/3':'supragranular',
    '4':'granular',
    '4a':'granular',
    '4b':'granular',
    '5':'infragranular',
    '6a':'infragranular',
    '6b':'infragranular',
    '6':'infragranular',
}
def get_conditions(cfg):
    period = cfg.get("period", "passive")
    mod    = cfg.get("modality", "whisker_auditory")

    passive_all = [
        ("whisker_trial",  "passive_pre",    "Whisker pre",        "#32a852", "start_time"),
        ("whisker_trial",  "passive_post",   "Whisker post",       "#085c1f", "start_time"),
        ("auditory_trial", "passive_pre",    "Auditory pre",       "#4158d9", "start_time"),
        ("auditory_trial", "passive_post",   "Auditory post",      "#0f2187", "start_time"),
    ]
    # Active conditions: trial_type × lick context (active_lick / active_nolick)
    # whisker hit/miss, auditory hit, false alarm, correct rejection
    # Jaw-aligned: same trial subsets but aligned to jaw_onset_time instead of start_time
    active_all = [
        ("whisker_trial",  "active_lick",    "Whisker hit",        "#fcba03", "start_time"),
        ("whisker_trial",  "active_nolick",  "Whisker miss",       "#d6371e", "start_time"),
        ("auditory_trial", "active_lick",    "Auditory hit",       "#7c0082", "start_time"),
        ("no_stim_trial",  "active_lick",    "False alarm",        "#211f21", "start_time"), #TODO: use spontaneous licks for this
        #("no_stim_trial",  "active_nolick",  "Correct rej.",       "#a6a4a1", "start_time"),
        # Jaw-aligned conditions (lighter shades of the start_time counterparts)
        ("whisker_trial",  "active_lick",    "Whisker hit (jaw)",  "#fde89a", "jaw_onset_time"),
        ("auditory_trial", "active_lick",    "Auditory hit (jaw)", "#c9aadd", "jaw_onset_time"),
        #("no_stim_trial",  "active_lick",    "False alarm (jaw)",  "#999999", "jaw_onset_time"),
    ]

    # Lick-aligned conditions — only included when include_lick_conditions=True
    # and a lick_df has been provided (populated into event_map via add_lick_event_map).
    # Appended last so existing condition indices are never disturbed when flag is False.
    if cfg.get("include_lick_conditions", False):
        active_all += [
            ("spontaneous_lick", "spontaneous", "Spont. lick",  "#999999", "lick_time"),
            ("reward_lick",      "spontaneous", "Reward time",  "#bf1a1a", "lick_time"),
        ]

    if mod == "whisker":
        passive_all = [c for c in passive_all if c[0] == "whisker_trial"]
        active_all  = [c for c in active_all  if c[0] in ("whisker_trial", "no_stim_trial")]
    elif mod == "auditory":
        passive_all = [c for c in passive_all if c[0] == "auditory_trial"]
        active_all  = [c for c in active_all  if c[0] in ("auditory_trial", "no_stim_trial")]

    if period == "passive":
        all_conds = passive_all
    elif period == "active":
        all_conds = active_all
    else:  # both
        all_conds = passive_all + active_all

    conds              = [(c[0], c[1]) for c in all_conds]
    cond_labels        = [c[2] for c in all_conds]
    cond_colors        = [c[3] for c in all_conds]
    cond_align_cols    = [c[4] for c in all_conds]
    cond_labels_matrix = [c[2].replace(" ", "\n") for c in all_conds]
    return conds, cond_labels, cond_colors, cond_labels_matrix, cond_align_cols


def get_t_window(cfg, context: str, align_col: str = "start_time"):
    if align_col == "jaw_onset_time":# or align_col == "lick_time":
        return cfg["t_pre_jaw"], cfg["t_post_jaw"]
    if context.startswith("passive"):
        return cfg["t_pre_passive"], cfg["t_post_passive"]
    else:
        return cfg["t_pre_active"], cfg["t_post_active"]


def get_cond_infos(cfg, conds,cond_align_cols=None):
    """Return per-condition list of (t_pre, t_post, t_ctr, n_bins, base_mask).

    Each condition may have a different time window (passive vs active), so
    n_bins can differ across conditions.  All downstream code should use this
    list rather than a single scalar n_bins.
    """
    if cond_align_cols is None:
        cond_align_cols = ["start_time"] * len(conds)
    dt = cfg["stride_ms"] / 1000
    infos = []
    for (tt, ctx), acol in zip(conds, cond_align_cols):
        t_pre, t_post = get_t_window(cfg, ctx, acol)
        n_out = int(round((t_pre + t_post) / dt))
        t_ctr = np.linspace(-t_pre, t_post, n_out, endpoint=False)
        infos.append((t_pre, t_post, t_ctr, n_out, t_ctr < 0))
    return infos

CONDITIONS = [
    ("whisker_trial",  "passive_pre"),
    ("whisker_trial",  "passive_post"),
    ("auditory_trial", "passive_pre"),
    ("auditory_trial", "passive_post"),
]
COND_LABELS        = ["Whisker pre", "Whisker post", "Auditory pre", "Auditory post"]
COND_LABELS_MATRIX = ["Whisker\npre-learning", "Whisker\npost-learning", "Auditory\npre-learning", "Auditory\npost-learning"]
COND_COLORS  = ["#32a852", "#085c1f", "#4158d9", "#0f2187"]
COND_ALIGN_COLS = ["start_time"] * 4

CONDITIONS, COND_LABELS, COND_COLORS, COND_LABELS_MATRIX, COND_ALIGN_COLS = get_conditions(DEFAULT_CFG)


# ── spikes helpers ──────────────────────────────────────────────────────────────

def assign_passive_context(trials: pd.DataFrame, mouse_id_col: str,
                           session_id_col: str) -> pd.DataFrame:
    """Split context=='passive' into passive_pre/post per (mouse, session)."""
    trials = trials.copy()
    for _, grp in trials[trials["context"] == "passive"].groupby([mouse_id_col, session_id_col]):
        mid = len(grp) // 2
        trials.loc[grp.index[:mid], "context"] = "passive_pre"
        trials.loc[grp.index[mid:], "context"] = "passive_post"
    return trials


def assign_active_context(trials: pd.DataFrame) -> pd.DataFrame:
    """Label active/NaN-context rows as 'active_lick' or 'active_nolick' via lick_flag."""
    trials = trials.copy()
    active_mask = trials["context"].isna() | (trials["context"]=='nan') | (trials["context"] == "active")
    #active_mask = trials['context'] != "passive"
    trials.loc[active_mask & (trials["lick_flag"] == 1), "context"] = "active_lick"
    trials.loc[active_mask & (trials["lick_flag"] == 0), "context"] = "active_nolick"
    return trials


def get_spike_times(row: pd.Series) -> np.ndarray:
    return np.asarray(row["spike_times"])


def _replace_artifact(rel, t_pre, lo, hi, rng):
    pre  = rel[rel < lo]
    rate = len(pre) / (t_pre + lo) if (t_pre + lo) > 0 else 0.0
    keep = rel[(rel < lo) | (rel > hi)]
    n    = rng.poisson(rate * (hi - lo))
    return np.sort(np.concatenate([keep, rng.uniform(lo, hi, n)])) if n > 0 else np.sort(keep)


def spikes_around_events(spk, events, t_pre, t_post,
                         is_whisker=False, artifact_win_s=(-0.005, 0.005), rng=None,
                         align_col="start_time"):
    """
    align_col : alignment column name for this condition ('start_time' or
                'jaw_onset_time').  Artifact correction (_replace_artifact) is
                only ever applied for start_time-aligned whisker trials —
                jaw-aligned conditions skip it entirely, since the artifact
                window is defined relative to stimulus onset, not jaw onset.
    """
    if rng is None:
        rng = np.random.default_rng()
    lo_a, hi_a = artifact_win_s
    i_lo   = np.searchsorted(spk, events - t_pre)
    i_hi   = np.searchsorted(spk, events + t_post, side="left")
    raster = [spk[i_lo[i]:i_hi[i]] - events[i] for i in range(len(events))]
    if is_whisker and align_col == "start_time":
        raster = [_replace_artifact(r, t_pre, lo_a, hi_a, rng) for r in raster]
    return raster


def _bin_rates(raster, bins, dt):
    return np.vstack([np.histogram(r, bins=bins)[0] for r in raster]).astype(float) / dt

def _bin_rates_strided_withartifact(raster, t_pre, t_post, bin_ms, stride_ms, trim=True):
    dt_stride  = stride_ms / 1000
    dt_bin     = bin_ms    / 1000
    k          = max(1, int(round(bin_ms / stride_ms)))   # boxcar width
    pad        = k // 2
    # extend window by pad bins on each side
    t_pre_ext  = t_pre  + pad * dt_stride
    t_post_ext = t_post + pad * dt_stride
    n_out_ext  = int(round((t_pre_ext + t_post_ext) / dt_stride))
    bins_fine  = np.linspace(-t_pre_ext, t_post_ext, n_out_ext + 1)
    boxcar     = np.ones(k) / dt_bin
    n_out      = int(round((t_pre + t_post) / dt_stride))
    rows = []
    for r in raster:
        counts, _ = np.histogram(r, bins=bins_fine)
        sm        = np.convolve(counts.astype(float), boxcar, mode="valid")
        # trim to exact window, dropping the edge-contaminated bins
        rows.append(sm[pad: pad + n_out])
    return np.vstack(rows)

def _bin_rates_strided(raster, t_pre_ext, t_post_ext, bin_ms, stride_ms, n_out):
    """Valid convolution on extended window — no edge padding needed."""
    dt_stride = stride_ms / 1000
    dt_bin    = bin_ms    / 1000
    k         = max(1, int(round(bin_ms / stride_ms)))
    n_ext     = int(round((t_pre_ext + t_post_ext) / dt_stride))
    bins_fine = np.linspace(-t_pre_ext, t_post_ext, n_ext + 1)
    boxcar    = np.ones(k) / dt_bin
    rows = []
    for r in raster:
        counts, _ = np.histogram(r, bins=bins_fine)
        #sm = np.convolve(counts.astype(float), boxcar, mode="valid") #acausal, creates shift
        padded = np.pad(counts.astype(float), (k - 1, 0), mode="constant", constant_values=0)
        sm = np.convolve(padded, boxcar, mode="valid")[:n_out]
        rows.append(sm[:n_out])
    return np.vstack(rows)

def _bin_rates_strided_old(raster, t_pre, t_post, bin_ms, stride_ms):
    """Sliding-window firing rate at stride_ms resolution, bin_ms window."""
    dt_stride = stride_ms / 1000
    dt_bin    = bin_ms   / 1000
    n_out     = int(round((t_pre + t_post) / dt_stride))
    bins_fine = np.linspace(-t_pre, t_post, n_out + 1)
    k         = max(1, int(round(bin_ms / stride_ms)))   # boxcar width in fine bins
    boxcar    = np.ones(k) / dt_bin                      # → Hz after convolution
    rows = []
    for r in raster:
        counts, _ = np.histogram(r, bins=bins_fine)
        # valid convolution avoids edge contamination; pad to restore length
        pad    = k // 2
        #padded = np.pad(counts.astype(float), (pad, pad), mode="edge")
        padded = np.pad(counts.astype(float), (pad, pad), mode="constant", constant_values=0)

        sm     = np.convolve(padded, boxcar, mode="valid")[:n_out]
        rows.append(sm)
    return np.vstack(rows)

from scipy.ndimage import gaussian_filter1d

def _bin_and_smooth_original(raster, t_pre, t_post, stride_ms, sigma_ms):
    """Bin spikes at stride_ms resolution, smooth with Gaussian of sigma_ms."""
    dt    = stride_ms / 1000
    n_out = int(round((t_pre + t_post) / dt))
    bins  = np.linspace(-t_pre, t_post, n_out + 1)
    sigma = sigma_ms / stride_ms          # sigma in bins
    rows  = []
    for r in raster:
        counts, _ = np.histogram(r, bins=bins)
        rate      = counts.astype(float) / dt   # Hz
        rows.append(gaussian_filter1d(rate, sigma=sigma, mode="reflect"))
    return np.vstack(rows)                # (n_trials, n_out)


def _bin_and_smooth(raster, t_pre, t_post, stride_ms, sigma_ms):
    """
    Bin spikes at stride_ms resolution and smooth with a Gaussian kernel
    without edge artifacts by padding with real data and cropping.
    """
    dt = stride_ms / 1000
    sigma_bins = sigma_ms / stride_ms

    # kernel support (~4 sigma on each side)
    pad_bins = int(np.ceil(4 * sigma_bins))
    pad_t = pad_bins * dt

    # extended window
    t_pre_ext = t_pre + pad_t
    t_post_ext = t_post + pad_t

    n_ext = int(round((t_pre_ext + t_post_ext) / dt))
    bins_ext = np.linspace(-t_pre_ext, t_post_ext, n_ext + 1)

    rows = []

    for r in raster:
        # strict clipping to extended window
        r = r[(r >= -t_pre_ext) & (r < t_post_ext)]

        counts, _ = np.histogram(r, bins=bins_ext)
        rate = counts.astype(float) / dt  # Hz

        # no reflection artifacts
        smooth = gaussian_filter1d(
            rate,
            sigma=sigma_bins,
            mode="nearest",
            #cval=0.0
        )

        # crop back to requested interval
        smooth = smooth[pad_bins:-pad_bins]

        rows.append(smooth)

    return np.vstack(rows)

def _causal_gaussian(x, sigma_bins, truncate=4.0):
    r = int(truncate * sigma_bins)
    k = np.exp(-0.5 * (np.arange(r + 1) / sigma_bins) ** 2)
    k /= k.sum()
    return np.convolve(x, k, mode="full")[:len(x)]




# ── pre-grouping (key speedup) ─────────────────────────────────────────────────

def precompute_event_map(trials: pd.DataFrame, cfg: dict,
                         conds=None, cond_align_cols=None) -> dict:
    """
    Group event times by (mouse_id, session_id, context, trial_type, align_col) once.
    Workers receive a plain dict[tuple, np.ndarray] — no DataFrame per neuron.

    Each condition may have its own alignment column (e.g. 'start_time' or
    'jaw_onset_time').  The align_col name is included as the 5th key element
    to disambiguate conditions that share the same (context, trial_type) but
    differ in alignment.  Rows where the alignment column is NaN are dropped.
    """

    # For passive/both: drop mice that have no passive data at all
    mice_beh_epochs = trials.groupby('mouse_id')['context'].nunique()
    mice_with_no_passive = mice_beh_epochs[mice_beh_epochs == 2].index
    print('Excluding mice without passive data...')
    trials = trials[~trials['mouse_id'].isin(mice_with_no_passive)]


    if conds is None:
        conds = CONDITIONS
    if cond_align_cols is None:
        cond_align_cols = COND_ALIGN_COLS

    # Build a set of (context, trial_type, afalign_col) triples we actually need
    needed = set(
        (ctx, tt, acol)
        for (tt, ctx), acol in zip(conds, cond_align_cols)
    )

    event_map = {}
    for keys, grp in trials.groupby([cfg["mouse_id_col"], cfg["session_id_col"],
                                     cfg["context_col"],   cfg["trial_type_col"]]):
        mouse_id, session_id, context, trial_type = keys
        for align_col in {acol for (ctx, tt, acol) in needed
                          if ctx == context and tt == trial_type}:
            if align_col not in grp.columns:
                continue
            col_vals = grp[align_col].dropna()
            if len(col_vals) == 0:
                continue
            # Use only rows where this align_col is non-NaN
            valid = grp[grp[align_col].notna()]
            event_map[(mouse_id, session_id, context, trial_type, align_col)] = \
                valid[align_col].to_numpy()
    return event_map


def add_lick_event_map(lick_df: pd.DataFrame, event_map: dict, cfg: dict) -> None:
    """Append spontaneous-lick and reward-lick events to an existing event_map in-place.

    lick_df : one row per session, columns:
        mouse_id, session_id,
        spontaneous_licks  (np.ndarray of lick times),
        reward_times       (np.ndarray of lick times)
    As returned by load_spontaneous_reward_lick_times().
    """
    mid_col = cfg.get("mouse_id_col", "mouse_id")
    sid_col = cfg.get("session_id_col", "session_id")
    n_spont = n_reward = 0

    missing_spont = []
    missing_reward = []

    for _, row in lick_df.iterrows():
        mouse_id, session_id = row[mid_col], row[sid_col]

        spont_times  = row["spontaneous_licks"]
        reward_times = row["reward_times"]

        if spont_times is not None and len(spont_times):
            event_map[(mouse_id, session_id, "spontaneous", "spontaneous_lick", "lick_time")] = np.sort(spont_times)
            n_spont += 1

        if reward_times is not None and len(reward_times):
            event_map[(mouse_id, session_id, "spontaneous", "reward_lick", "lick_time")] = np.sort(reward_times)
            n_reward += 1

        if spont_times is None:
            missing_spont.append(mouse_id)
        if reward_times is None:
            missing_reward.append(mouse_id)




    print(f"  add_lick_event_map: {n_spont} spontaneous_lick, {n_reward} reward_lick entries added")
    print('      missing spontaneous_licks:', missing_spont)
    print('      missing reward_times:', missing_reward)

def _get_events(event_map, mouse_id, session_id, context, trial_type,
                align_col="start_time") -> np.ndarray:
    return event_map.get((mouse_id, session_id, context, trial_type, align_col), np.array([]))


# ── FR filter ──────────────────────────────────────────────────────────────────

def _unit_fr(st, mouse_id, session_id, event_map, cfg) -> float:
    """Mean FR across all conditions, using per-condition time windows."""
    total_spk  = 0
    total_time = 0.0
    for (tt, ctx), acol in zip(CONDITIONS, COND_ALIGN_COLS):
        events = _get_events(event_map, mouse_id, session_id, ctx, tt, acol)
        if len(events) == 0:
            continue
        t_pre, t_post = get_t_window(cfg, ctx, acol)
        i_lo = np.searchsorted(st, events - t_pre)
        i_hi = np.searchsorted(st, events + t_post, side="left")
        total_spk  += (i_hi - i_lo).sum()
        total_time += len(events) * (t_pre + t_post)
    return total_spk / total_time if total_time > 0 else 0.0


def apply_fr_filter(unit_ids, st_map, mouse_map, session_map, event_map, cfg):
    thr     = cfg["fr_threshold_hz"]
    fr_vals = Parallel(n_jobs=cfg["n_jobs"], prefer="threads")(
        delayed(_unit_fr)(st_map[uid], mouse_map[uid], session_map[uid], event_map, cfg)
        for uid in unit_ids
    )
    fr_map = dict(zip(unit_ids, fr_vals))
    kept   = [uid for uid in unit_ids if fr_map[uid] >= thr]
    print(f"  FR filter (>={thr} Hz): {len(unit_ids)} → {len(kept)}")
    return kept, fr_map


# ── feature matrix ─────────────────────────────────────────────────────────────

def _neuron_vector(st, mouse_id, session_id, event_map, bins, t_ctr, base_mask, cfg):
    """
    (4*n_bins,) z-scored PSTH.

    Z-score: z(t) = (mean_psth(t) - mean_bl) / std_bl
    where mean_bl and std_bl are estimated from trial-to-trial baseline firing
    rates pooled across all 4 conditions — giving a single shared normalizer so
    pre and post are directly comparable.
    """
    dt  = cfg["bin_ms"] / 1000
    rng = np.random.default_rng()

    # Compute smoothed mean PSTH per condition
    def _mean_psth(trial_type, context):
        events = _get_events(event_map, mouse_id, session_id, context, trial_type)
        if len(events) == 0:
            return np.zeros(len(t_ctr))
        raster = spikes_around_events(st, events, cfg["t_pre"], cfg["t_post"],
                                      is_whisker=(trial_type == cfg["whisker_trial_label"]),
                                      artifact_win_s=cfg["artifact_win_s"], rng=rng)
        rates = _bin_rates(raster, bins, dt) #wo stride

        if cfg.get("square_fr", False):
            rates = rates ** 2

        return rates.mean(axis=0)


    wh_pre  = _mean_psth("whisker_trial",  "passive_pre")
    wh_post = _mean_psth("whisker_trial",  "passive_post")
    au_pre  = _mean_psth("auditory_trial", "passive_pre")
    au_post = _mean_psth("auditory_trial", "passive_post")

    def _zscore_pair(a, b):
        v    = np.concatenate([a, b])
        mean = v.mean()
        std  = v.std() + 1e-9
        return (a - mean) / std, (b - mean) / std

    def _zscore_pair_with_bas(a, b, base_mask):
        bl = np.concatenate([a[base_mask], b[base_mask]])
        mean = bl.mean()
        std = bl.std() + 1e-9
        return (a - mean) / std, (b - mean) / std

    #wh_pre,  wh_post = _zscore_pair(wh_pre,  wh_post)
    #au_pre,  au_post = _zscore_pair(au_pre,  au_post)
    if cfg.get("zscore_full", True):
        wh_pre,  wh_post = _zscore_pair(wh_pre,  wh_post)
        au_pre,  au_post = _zscore_pair(au_pre,  au_post)
    else: # then z-score using baseline stats only
        wh_pre, wh_post = _zscore_pair_with_bas(wh_pre, wh_post, base_mask)
        au_pre, au_post = _zscore_pair_with_bas(au_pre, au_post, base_mask)

    return np.concatenate([wh_pre, wh_post, au_pre, au_post])

def _neuron_vector_strided(st, mouse_id, session_id, event_map, cond_infos, cfg, conds,
                           cond_align_cols=None,
                           norm_params=None, return_norm=False,
                           start_times_map=None):
    """
    Pipeline order (per condition), per neuron:

    1. Raw per-trial smoothed rates (Hz).

    2. [Optional] Per-trial baseline subtraction (cfg["baseline_removal"], default True):
         • start_time-aligned: subtract each trial's own pre-window mean (t_ctr < 0).
         • jaw_onset_time-aligned: borrow pooled mean from the matched start_time
           sibling (same trial_type/context).
         • lick_time-aligned: per-event quiet-window baseline — for each lick event,
           find the most recent preceding trial start_time in this session, compute
           mean spikes/s in [start_time - 0.5 s, start_time] from the raw spike
           train, and subtract that scalar from every bin of that lick event.
           Falls back to the earliest trial if no preceding trial exists.
           Multiple lick events sharing the same nearest preceding trial get the
           same baseline scalar.
       When False: raw rates, no subtraction.

    3. Average across trials → mean PSTH per condition.
    4. [Optional] Signed-square (cfg["square_fr"]).
    5. Z-score on full concatenated trace (single mean/std across all conditions).

    start_times_map : {(mouse_id, session_id): sorted start_time array} — required
                      for lick_time baseline estimation.
    norm_params     : pre-fitted (mean, std) from training split (CV even path).
    """
    if cond_align_cols is None:
        cond_align_cols = ["start_time"] * len(conds)
    rng              = np.random.default_rng()
    baseline_removal = cfg.get("baseline_removal", True)

    # Build lookup: (trial_type, context, align_col) -> condition index,
    # used to find the start_time sibling for each jaw condition.
    cond_index = {(tt, ctx, acol): i
                  for i, ((tt, ctx), acol) in enumerate(zip(conds, cond_align_cols))}

    # Cache raw per-trial rates so each condition is computed only once
    # (the start_time sibling is needed by its jaw counterpart).
    rates_cache = {}

    def _get_rates(idx):
        if idx not in rates_cache:
            (tt, ctx), (t_pre, t_post, _, n_out, _), acol = \
                conds[idx], cond_infos[idx], cond_align_cols[idx]
            events = _get_events(event_map, mouse_id, session_id, ctx, tt, acol)
            if len(events) == 0:
                rates_cache[idx] = None
            else:
                raster = spikes_around_events(
                    st, events, t_pre, t_post,
                    is_whisker=(tt == cfg["whisker_trial_label"]),
                    artifact_win_s=cfg["artifact_win_s"], rng=rng)
                rates_cache[idx] = _bin_and_smooth(
                    raster, t_pre, t_post, cfg["stride_ms"], cfg["sigma_ms"])
        return rates_cache[idx]

    def _mean_psth(idx):
        (tt, ctx), (t_pre, t_post, _, n_out, base_mask), acol = \
            conds[idx], cond_infos[idx], cond_align_cols[idx]
        rates = _get_rates(idx)
        if rates is None:
            return np.zeros(n_out)

        if baseline_removal:
            if acol == "start_time":
                # Own-window per-trial baseline: subtract each trial's mean
                # pre-stimulus activity before averaging.
                rates = rates - rates[:, base_mask].mean(axis=1, keepdims=True)
            elif acol == "lick_time":
                # Per-event quiet-window baseline: for each lick event, find
                # the most recent preceding trial start_time in this session,
                # compute mean spikes/s in [start_time - 0.5 s, start_time]
                # from the raw spike train, and subtract as a per-event scalar.
                # Falls back to the earliest trial if no preceding trial exists.
                BL_WIN = cfg.get("lick_quiet_window_s", 0.5)
                session_starts = (start_times_map or {}).get(
                    (mouse_id, session_id), np.array([]))

                if len(session_starts) == 0:
                    # No trial info available — fall back to zero
                    rates = rates - 0.0
                else:
                    events = _get_events(
                        event_map, mouse_id, session_id, ctx, tt, acol)
                    per_event_bl = np.empty(len(events))
                    for ei, t_lick in enumerate(events):
                        # Most recent start_time before this lick
                        idx_prec = np.searchsorted(session_starts, t_lick, side="left") - 1
                        if idx_prec < 0:
                            # No preceding trial — use earliest trial
                            idx_prec = 0
                        t_ref = session_starts[idx_prec]
                        # Count spikes in [t_ref - BL_WIN, t_ref]
                        n_spk = np.searchsorted(st, t_ref, side="left") - \
                                np.searchsorted(st, t_ref - BL_WIN, side="left")
                        per_event_bl[ei] = n_spk / BL_WIN  # Hz
                    # per_event_bl shape: (n_events,) → (n_events, 1) for broadcast
                    rates = rates - per_event_bl[:, np.newaxis]
            else:
                # Jaw-aligned: borrow baseline mean from the matched
                # start_time sibling (same trial_type, same context).
                sib_idx   = cond_index.get((tt, ctx, "start_time"))
                sib_rates = _get_rates(sib_idx) if sib_idx is not None else None
                if sib_rates is not None:
                    sib_base_mask = cond_infos[sib_idx][4]
                    borrowed_mean = sib_rates[:, sib_base_mask].mean()
                else:
                    borrowed_mean = 0.0
                rates = rates - borrowed_mean

        if cfg.get("square_fr", False):
            rates = np.sign(rates) * rates ** 2

        return rates.mean(axis=0)

    psths = [_mean_psth(idx) for idx in range(len(conds))]

    # ── Z-score on full concatenated trace (single normaliser) ───────────────
    # Always uses the entire concatenated mean PSTH vector (all conditions),
    # not baseline bins only. One mean/std per neuron, applied uniformly.
    if norm_params is None:
        full_vec = np.concatenate(psths)
        mean_    = full_vec.mean()
        std_     = full_vec.std() + 1e-9 if cfg.get("normalize", "zscore") == "zscore" else 1.0
        norms    = (mean_, std_)
    else:
        norms    = norm_params
        mean_, std_ = norms

    psths = [(p - mean_) / std_ for p in psths]

    vec = np.concatenate(psths)
    if return_norm:
        return vec, norms
    return vec

def _neuron_vector_strided_keep(st, mouse_id, session_id, event_map, cond_infos, cfg, conds,
                           cond_align_cols=None,
                           norm_params=None, return_norm=False):
    """
    Pipeline order (per condition), textbook baseline z-score style:
        1. raw per-trial smoothed rates (Hz)
        2. subtract each trial's own baseline-window mean (own-window for
           start_time-aligned conditions; borrowed from the matched
           start_time sibling for jaw_onset_time-aligned conditions)
        3. divide every bin (still per-trial) by ONE condition-level baseline
           std, computed by pooling all (trial x baseline-bin) RAW values of
           that condition (pre-subtraction, pre-square) into a single array
           and taking its std. Jaw conditions borrow this std from their
           start_time sibling the same way they borrow the mean.
        4. average across trials -> z-scored PSTH
        5. square (signed) if cfg["square_fr"] is True -- squaring happens on
           the z-scored values, AFTER normalisation, so the std fitted in
           step 3 reflects genuine baseline firing-rate variability rather
           than a residual of squared, trial-averaged deviations.

    Baseline removal  (cfg["baseline_removal"], default True)
    ────────────────────────────────────────────────────────
    When True: steps 2-3 above are applied as described.
    When False: no baseline mean subtraction AND no baseline-std division —
        the PSTH is just the trial-averaged raw rate (optionally squared).

    cfg["normalize"]: "zscore" divides by the baseline std (step 3); "baseline"
        skips the division (std=1), keeping only the mean subtraction.
    The std computed here is always baseline-window-only, per condition, by
    design -- there is no longer a "whole window vs baseline-only" choice for
    this strided pipeline (that distinction only ever applied to the legacy,
    unused _neuron_vector function).
    """
    if cond_align_cols is None:
        cond_align_cols = ["start_time"] * len(conds)
    rng = np.random.default_rng()
    baseline_removal = cfg.get("baseline_removal", True)
    do_zscore = cfg.get("normalize", "zscore") == "zscore"

    # Map (trial_type, context) -> index, to find each jaw condition's
    # start_time-aligned sibling (same trial_type/context, align_col="start_time").
    cond_index = {(tt, ctx, acol): i
                  for i, ((tt, ctx), acol) in enumerate(zip(conds, cond_align_cols))}

    def _raw_rates(trial_type, context, t_pre, t_post, n_out, align_col):
        """Smoothed per-trial rates (Hz), no baseline subtraction, no squaring."""
        events = _get_events(event_map, mouse_id, session_id, context, trial_type, align_col)
        if len(events) == 0:
            return None
        raster = spikes_around_events(st, events, t_pre, t_post,
                                      is_whisker=(trial_type == cfg["whisker_trial_label"]),
                                      artifact_win_s=cfg["artifact_win_s"], rng=rng,
                                      align_col=align_col)
        return _bin_and_smooth(raster, t_pre, t_post, cfg["stride_ms"], cfg["sigma_ms"])

    # Cache of raw per-trial rates per condition index, so the start_time
    # sibling (needed for jaw borrowed-baseline/std) is computed only once.
    rates_cache = {}

    def _get_rates(idx):
        if idx not in rates_cache:
            (tt, ctx), (t_pre, t_post, t_ctr_c, n_out, base_mask), acol = \
                conds[idx], cond_infos[idx], cond_align_cols[idx]
            rates_cache[idx] = _raw_rates(tt, ctx, t_pre, t_post, n_out, acol)
        return rates_cache[idx]

    def _baseline_mean_and_std(idx):
        """Per-neuron baseline mean & std for condition idx, computed from RAW
        (pre-subtraction, pre-square) rates -- own baseline window for
        start_time conditions, borrowed from the start_time sibling for jaw
        conditions. Returns (mean, std) as scalars; std=None if unavailable."""
        (tt, ctx), (t_pre, t_post, t_ctr_c, n_out, base_mask), align_col = \
            conds[idx], cond_infos[idx], cond_align_cols[idx]
        if align_col == "start_time":
            rates = _get_rates(idx)
            if rates is None:
                return 0.0, None
            bl_vals = rates[:, base_mask]
        else:
            sib_idx = cond_index.get((tt, ctx, "start_time"))
            sib_rates = _get_rates(sib_idx) if sib_idx is not None else None
            if sib_rates is None:
                return 0.0, None
            sib_base_mask = cond_infos[sib_idx][4]
            bl_vals = sib_rates[:, sib_base_mask]
        return bl_vals.mean(), bl_vals.std()

    def _mean_psth(idx, mean_, std_):
        rates = _get_rates(idx)
        n_out = cond_infos[idx][3]
        if rates is None:
            return np.zeros(n_out)

        if baseline_removal:
            rates = rates - mean_
            if do_zscore and std_ is not None:
                rates = rates / (std_ + 1e-9)

        if cfg.get("square_fr", False):
            rates = np.sign(rates) * rates ** 2  # negative stays negative, applied post z-score

        return rates.mean(axis=0)

    # ── per-condition baseline (mean, std), fit on this split or reused ────
    if norm_params is None:
        norms = [_baseline_mean_and_std(idx) for idx in range(len(conds))]
    else:
        norms = norm_params

    psths = [_mean_psth(idx, m, s) for idx, (m, s) in enumerate(norms)]
    vec = np.concatenate(psths)
    if return_norm:
        return vec, norms
    return vec

def build_feature_matrix(unit_ids, st_map, mouse_map, session_map, event_map, cfg):
    """(n_neurons, 4*n_bins) z-scored PSTH matrix, fully parallelised."""
    dt    = cfg["bin_ms"] / 1000
    n_bins = int(round((cfg["t_pre"] + cfg["t_post"]) / dt))
    #bins  = np.arange(-cfg["t_pre"], cfg["t_post"] + dt, dt)
    bins = np.linspace(-cfg["t_pre"], cfg["t_post"], n_bins + 1)
    t_ctr = bins[:-1] + dt / 2
    rows  = Parallel(n_jobs=cfg["n_jobs"], prefer="threads")(
        delayed(_neuron_vector)(
            st_map[uid], mouse_map[uid], session_map[uid],
            event_map, bins, t_ctr, t_ctr < 0, cfg,
        )
        for uid in unit_ids
    )
    return np.vstack(rows), t_ctr, len(t_ctr)

def build_feature_matrix_strided(unit_ids, st_map, mouse_map, session_map, event_map, cfg,
                         conds, cond_labels, cond_colors, cond_labels_matrix,
                         cond_align_cols=None,
                         norm_params_list=None, return_norms=False,
                         start_times_map=None):
    """Build (n_neurons, sum(n_bins_per_cond)) z-scored PSTH matrix.

    Each condition may have a different window (passive vs active), so the
    feature vector is the concatenation of variable-length PSTHs.

    cond_align_cols  : per-condition alignment column names (e.g. 'start_time' or
                       'jaw_onset_time').  Defaults to 'start_time' for all conditions.
    norm_params_list : per-neuron norm tuples fitted on another split (odd trials).
                       When None, norms are computed from the data.
    return_norms     : if True, also return the list of per-neuron norms.
    start_times_map  : dict {(mouse_id, session_id): sorted start_time array},
                       used for lick-aligned quiet-window baseline estimation.

    Returns
    -------
    X            : (n_neurons, total_bins) array
    t_ctrs       : list of per-condition t_ctr arrays
    n_bins_list  : list of per-condition bin counts
    [norms_list] : only when return_norms=True
    """
    assert norm_params_list is None

    if cond_align_cols is None:
        cond_align_cols = ["start_time"] * len(conds)
    cond_infos = get_cond_infos(cfg, conds, COND_ALIGN_COLS)
    t_ctrs     = [info[2] for info in cond_infos]
    n_bins_list = [info[3] for info in cond_infos]

    results = Parallel(n_jobs=cfg["n_jobs"], prefer="threads")(
        delayed(_neuron_vector_strided)(
            st_map[uid], mouse_map[uid], session_map[uid],
            event_map, cond_infos, cfg, conds,
            cond_align_cols = cond_align_cols,
            norm_params = norm_params_list[i] if norm_params_list is not None else None,
            return_norm = return_norms,
            start_times_map = start_times_map,
        )
        for i, uid in enumerate(unit_ids)
    )

    if return_norms:
        rows       = [r[0] for r in results]
        norms_list = [r[1] for r in results]
        return np.vstack(rows), t_ctrs, n_bins_list, norms_list
    return np.vstack(results), t_ctrs, n_bins_list


# ── rastermap_psth ──────────────────────────────────────────────────────────────────

def fit_rastermap(X, n_clusters):
    n_pcs  = min(200, X.shape[0] - 1, X.shape[1] - 1)
    model  = Rastermap(n_clusters=n_clusters, #clusters for initial k_means
                       n_PCs=n_pcs,
                       locality=DEFAULT_CFG['locality'],
                       time_lag_window=DEFAULT_CFG['time_lag_window'],
                       grid_upsample=DEFAULT_CFG['grid_upsample'],
                    verbose =False).fit(X)
    isort  = model.isort
    bounds = np.round(np.linspace(0, len(isort), n_clusters + 1)[1:-1]).astype(int)
    return isort, bounds


def _kmeans_inertia(X, k):
    return KMeans(n_clusters=k, random_state=42, n_init=5).fit(X).inertia_



# ── plotting helpers ───────────────────────────────────────────────────────────
def _draw_matrix(ax, mat, n_bins_list, boundaries, vmax, cfg, title):
    """Draw the population PSTH matrix.

    n_bins_list : list of per-condition bin counts (may differ for passive vs active).
    Condition separator lines, onset lines and x-tick positions are all derived
    from cumulative bin offsets so they stay correct with variable-length windows.
    """
    n, n_total = mat.shape
    dt      = cfg["stride_ms"] / 1000
    offsets = np.concatenate([[0], np.cumsum(n_bins_list)])

    if not cfg.get("baseline_removal"):
        # Compute vcenter as the mean of all baseline-window bins (t < 0) across
        # all conditions and all neurons in this matrix.
        bl_cols = []
        for (tt, ctx), acol, start in zip(CONDITIONS, COND_ALIGN_COLS, offsets[:-1]):
            t_pre, _ = get_t_window(cfg, ctx, acol)
            n_bl = int(round(t_pre / dt))
            if n_bl > 0:
                bl_cols.append(mat[:, start : start + n_bl])
        vcenter = float(np.nanmean(np.concatenate(bl_cols, axis=1))) if bl_cols else 0.0
        cmap = 'coolwarm'
    else:
        # Baseline was subtracted per trial: baseline mean is ~0 by construction.
        vcenter = 0.0
        cmap = 'coolwarm'

    # Always diverging, centred on vcenter.
    norm = matplotlib.colors.TwoSlopeNorm(
        vcenter=vcenter, #TODO: hard-coded but to be fixed, fix also so that colored
        vmin=min(vcenter - vmax, vcenter - 1e-6),
        vmax=max(vcenter + vmax, vcenter + 1e-6),
    )
    #norm = matplotlib.colors.CenteredNorm(vcenter=vcenter, halfrange=vmax)

    im = ax.imshow(mat, aspect="auto", interpolation="none", cmap=cmap,
                   norm=norm, extent=[0, n_total, n, 0])

    for i, ((tt, ctx), start, end, acol) in enumerate(
            zip(CONDITIONS, offsets[:-1], offsets[1:], COND_ALIGN_COLS)):
        t_pre, _ = get_t_window(cfg, ctx, acol)
        onset_bin = int(round(t_pre / dt))
        if i > 0:
            ax.axvline(start, color="k", lw=1.5)
        ax.axvline(start + onset_bin, color="white", lw=0.8, ls="--")
    ticks = [(offsets[i] + offsets[i + 1]) / 2 for i in range(len(n_bins_list))]
    ax.set_xticks(ticks)
    ax.set_xticklabels(COND_LABELS_MATRIX, fontsize=8)

    ax.set_ylabel("Neuron")
    ax.set_title(title, fontsize=10)
    return im

def _draw_matrix_old(ax, mat, n_bins_list, boundaries, vmax, cfg, title):
    """Draw the population PSTH matrix.

    n_bins_list : list of per-condition bin counts (may differ for passive vs active).
    Condition separator lines, onset lines and x-tick positions are all derived
    from cumulative bin offsets so they stay correct with variable-length windows.
    """
    n, n_total = mat.shape
    if DEFAULT_CFG['baseline_removal']==False:
        vmin=0
        cmap = "coolwarm"

    else:
        vmin=-vmax
        cmap="coolwarm"

    im = ax.imshow(mat, aspect="auto", interpolation="none", cmap=cmap,
                   vmin=vmin, vmax=vmax, extent=[0, n_total, n, 0])
    dt        = cfg["stride_ms"] / 1000
    offsets   = np.concatenate([[0], np.cumsum(n_bins_list)])
    for i, ((tt, ctx), start, end, acol) in enumerate(
            zip(CONDITIONS, offsets[:-1], offsets[1:], COND_ALIGN_COLS)):
        t_pre, _ = get_t_window(cfg, ctx, acol)
        onset_bin = int(round(t_pre / dt))
        if i > 0:
            ax.axvline(start, color="k", lw=1.5)
        ax.axvline(start + onset_bin, color="white", lw=0.8, ls="--")
    ticks = [(offsets[i] + offsets[i + 1]) / 2 for i in range(len(n_bins_list))]
    ax.set_xticks(ticks)
    ax.set_xticklabels(COND_LABELS_MATRIX, fontsize=8)
    #for b in boundaries:
    #    ax.axhline(b, color="k", lw=0.1)

    ax.set_ylabel("Neuron")
    ax.set_title(title, fontsize=10)
    return im


def _save(fig, path,dpi):
    for fmt in ['.png', '.pdf', '.svg']:
        path_suffix = path.with_suffix(fmt)
        fig.savefig(path_suffix, dpi=dpi, bbox_inches="tight")

    plt.close(fig)
    print(f"  saved {Path(path).name}")


# ── figures ────────────────────────────────────────────────────────────────────

def order_area_groups(area_group_arr):
    """Order brain region groups by the canonical key order of get_custom_area_groups()
    (primary), then by descending population frequency within any unrecognised groups
    (secondary).
    """
    try:
        from allen_utils import get_custom_area_groups
        canonical_order = list(get_custom_area_groups().keys())
    except Exception:
        canonical_order = []

    # count population frequency per group
    groups, counts = np.unique(area_group_arr, return_counts=True)
    freq = dict(zip(groups, counts))

    def _sort_key(g):
        try:
            return (canonical_order.index(g), -freq.get(g, 0))
        except ValueError:
            return (len(canonical_order), -freq.get(g, 0))

    return sorted(freq.keys(), key=_sort_key)


def _draw_prop_column(ax, sorted_arr, categories, cat_colors, edges, n_neurons, title,
                      exclude=None):
    """Narrow stacked horizontal bar chart: one bar per cluster, showing category proportions."""
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        n_cl = hi - lo
        #if n_cl == 0:
        #    continue
        chunk = sorted_arr[lo:hi]
        if exclude is not None:
            chunk = chunk[~np.isin(chunk, exclude)]
        #valid = chunk != np.nan  # or "nan", "unknown" — whatever missing values look like
        n_valid = len(chunk)
        if n_valid == 0:
            continue
        #n_cl = len(chunk)
        #if n_cl == 0:
        #    continue
        left = 0.0
        for cat in categories:
            prop = (chunk == cat).sum() / n_valid  # ← denominator is n_valid, not n_cl
            if prop == 0:
                continue
            ax.barh((lo + hi) / 2, prop, left=left, height=n_cl * 0.92,
                    color=cat_colors.get(cat, "#aaaaaa"), edgecolor="none", align="center")
            left += prop
    ax.set_xlim(0, 1)
    ax.set_ylim(n_neurons, 0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"], fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel("Prop.", fontsize=7)
    ax.set_title(title, fontsize=8)


#ANATOMY_SCORE_COLS = [
#    "avg_ipsi",
#    "cc_tc_ct_iterated",
#    "cc_hierarchy_score_columns",
#]

def load_anatomy_scores(units_good, score_col_map):
    """Pull per-neuron anatomical axis scores directly from units_good/unit_table.

    score_col_map : dict mapping standard score name -> actual column name in
                    units_good, e.g.
                    {"axonal_innervation_s1":      "axonal_innervation_s1",
                     "hierarchy_score_harris2019": "hierarchy_score_harris2019",
                     "hierarchy_score_gao2026":    "hierarchy_score_gao2026"}
                    Override the values if unit_table uses different column names.
    Missing values (NaN in units_good) are preserved as NaN, not dropped.

    Returns
    -------
    dict[standard_score_name] -> dict[unit_id] -> value (float, NaN if missing)
    """
    missing = [col for col in score_col_map.values() if col not in units_good.columns]
    if missing:
        raise ValueError(
            f"anatomy score columns not found in units_good: {missing}. "
            f"Available columns: {list(units_good.columns)}")

    out = {}
    for std_name, col in score_col_map.items():
        col_map = units_good[col].to_dict()
        out[std_name] = {
            uid: (float(v) if pd.notna(v) else np.nan) for uid, v in col_map.items()
        }
    n_matched = sum(not np.isnan(v) for v in next(iter(out.values())).values())
    print(f"  Anatomy scores loaded from units_good: "
          f"{n_matched}/{len(units_good)} neurons have a non-NaN value "
          f"(checked on '{next(iter(score_col_map.values()))}')")
    return out
    return out


def _truncate_colormap(cmap, minval=0.15, maxval=0.85, n=256):
    """Return a colormap using only the [minval, maxval] slice of *cmap*'s range."""
    colors = cmap(np.linspace(minval, maxval, n))
    name   = f"trunc_{getattr(cmap, 'name', 'cmap')}"
    return mcolors.LinearSegmentedColormap.from_list(name, colors)


def _sequential_cmap_from_color(hex_color, from_color="#ffffff"):
    """White → *hex_color* sequential colormap (for e.g. axonal innervation strength)."""
    return mcolors.LinearSegmentedColormap.from_list(
        f"seq_{hex_color.lstrip('#')}", [from_color, hex_color])


def _robust_sequential_norm(arr):
    """Normalize(vmin, vmax) from the 1st/99th percentile of the non-NaN values."""
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return mcolors.Normalize(vmin=0, vmax=1)
    vmin, vmax = np.nanpercentile(valid, [1, 99])
    #vmax = vmax * 1.05          # pad top by 5%
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return mcolors.Normalize(vmin=vmin, vmax=vmax)
    #return mcolors.LogNorm(vmin=vmin, vmax=vmax)



def _robust_diverging_norm(arr):
    """TwoSlopeNorm centered at 0 (if the data spans both signs) or the median."""
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    vmin, vmax = np.nanpercentile(valid, [15, 85])
    vcenter = 0.0 if vmin < 0 < vmax else float(np.median(valid))
    if vmin >= vcenter:
        vmin = vcenter - 1e-6
    if vmax <= vcenter:
        vmax = vcenter + 1e-6
    return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


def build_anatomy_cmaps(axon_arr, harris_arr, gao_arr):
    """Build (cmap, norm) pairs for the three anatomical axis columns.

    axon    : sequential, white → #f2b01f
    harris/gao : diverging, cmasher 'holly' truncated to its middle 70%
                 (first/last 15% excluded), centered at 0 where possible
    """
    axon_cmap = _sequential_cmap_from_color("#f2b01f")
    axon_cmap = cmr.ember

    try:
        hier_base = cmr.holly
    except ImportError:
        print("  cmasher not installed (pip install cmasher) — "
              "falling back to matplotlib 'RdBu_r' for hierarchy-score colormaps")
        hier_base = matplotlib.colormaps["RdBu_r"]
    hier_cmap = _truncate_colormap(hier_base, 0.15, 0.85)

    return dict(
        avg_ipsi      = (axon_cmap, _robust_sequential_norm(axon_arr)),
        cc_tc_ct_iterated = (hier_cmap, _robust_diverging_norm(harris_arr)),
        cc_hierarchy_score_columns    = (hier_cmap, _robust_diverging_norm(gao_arr)),
    )

def _draw_continuous_cluster_column(ax, sorted_arr, area_sorted, edges, n_neurons, title, cmap, norm):
    """One solid-colored bar per cluster: mean-over-unique-areas (not mean-over-neurons)
    of a continuous anatomical axis — so an over-sampled area doesn't dominate the
    color just because more neurons were recorded from it.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        n_cl = hi - lo
        if n_cl == 0:
            continue
        chunk    = sorted_arr[lo:hi]
        areas_cl = area_sorted[lo:hi]
        valid    = ~np.isnan(chunk)
        if valid.sum() == 0:
            color = "#dddddd"
        else:
            area_means = pd.Series(chunk[valid]).groupby(areas_cl[valid]).mean()
            color = cmap(norm(area_means.mean()))
        ax.barh((lo + hi) / 2, 1.0, left=0.0, height=n_cl * 0.92,
                color=color, edgecolor="none", align="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(n_neurons, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)

def _draw_continuous_cluster_column_neurons(ax, sorted_arr, edges, n_neurons, title, cmap, norm):
    """Narrow column: one solid-colored bar per cluster, colored by that cluster's
    mean value on a continuous anatomical axis (e.g. hierarchy score).

    NaN-only clusters (no neurons with a matched area score) are drawn light gray.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        n_cl = hi - lo
        if n_cl == 0:
            continue
        chunk = sorted_arr[lo:hi]
        valid = chunk[~np.isnan(chunk)]
        color = "#dddddd" if valid.size == 0 else cmap(norm(np.nanmean(valid)))
        ax.barh((lo + hi) / 2, 1.0, left=0.0, height=n_cl * 0.92,
                color=color, edgecolor="none", align="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(n_neurons, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=5)


def load_waveform_classification(units_good, mouse_id_col, session_id_col,
                                  electrode_group_col="electrode_group",
                                  cluster_id_col="cluster_id"):
    """Load RSU(WW)/FSU(NW) waveform-type labels from per-mouse classification CSVs.

    For each mouse, reads:
        {WAVEFORM_ANALYSIS_ROOT}/{mouse_id}/whisker_0/waveform_analysis/{mouse_id}_cortical_wf_type.csv
    and merges onto units_good on (mouse_id, session_id, electrode_group, cluster_id).
    Expected CSV columns: mouse_id, session_id, electrode_group, cluster_id, and a
    label column identifying RSU ("ww", "rsu", "regular") vs FSU ("nw", "fsu", "narrow",
    "fast").  RSU -> "WW", FSU -> "NW".

    Returns
    -------
    dict[unit_id] -> "WW" | "NW" | "unknown"
    """
    units_good['cluster_id'] = units_good['cluster_id'].astype(int)

    label_col_candidates = ["waveform_type"]
    join_cols = [mouse_id_col, session_id_col, electrode_group_col, cluster_id_col]

    def _to_ww_nw(val):
        s = str(val).strip().lower()
        if s in ("ww", "rsu", "regular", "regular-spiking", "wide", "wide-waveform"):
            return "WW"
        if s in ("nw", "fsu", "narrow", "fast", "fast-spiking", "narrow-waveform"):
            return "NW"
        return "unknown"

    wf_map  = {}
    mice    = units_good[mouse_id_col].unique()

    missing_units_good_cols = [c for c in join_cols if c not in units_good.columns]
    if missing_units_good_cols:
        raise KeyError(
            f"units_good is missing join column(s) {missing_units_good_cols} needed to "
            f"merge waveform classification. Pass the correct column name(s) via "
            f"electrode_group_col / cluster_id_col if they're named differently in units_good.")

    for mouse_id in mice:
        mouse_units = units_good[units_good[mouse_id_col] == mouse_id]
        csv_path = Path(WAVEFORM_ANALYSIS_ROOT, str(mouse_id), "whisker_0",
                        "waveform_analysis", f"{mouse_id}_cortical_wf_type.csv")

        if not csv_path.exists():
            print(f"  [waveform classification] No CSV found for mouse {mouse_id} "
                 f"({csv_path}) — marking {len(mouse_units)} units as 'unknown'")
            wf_map.update({uid: "unknown" for uid in mouse_units.index})
            continue

        try:
            wf_df = pd.read_csv(csv_path)
            wf_df['cluster_id'] = wf_df['cluster_id'].astype(int)
        except Exception as e:
            print(f"  [waveform classification] Failed to read {csv_path}: {e} "
                 f"— marking {len(mouse_units)} units as 'unknown'")
            wf_map.update({uid: "unknown" for uid in mouse_units.index})
            continue

        label_col = next((c for c in label_col_candidates if c in wf_df.columns), None)
        if label_col is None:
            print(f"  [waveform classification] No recognised label column in {csv_path} "
                 f"(columns: {list(wf_df.columns)}) — marking {len(mouse_units)} units as 'unknown'")
            wf_map.update({uid: "unknown" for uid in mouse_units.index})
            continue

        missing_join_cols = [c for c in join_cols if c not in wf_df.columns]
        if missing_join_cols:
            print(f"  [waveform classification] CSV {csv_path} missing join column(s) "
                 f"{missing_join_cols} — marking {len(mouse_units)} units as 'unknown'")
            wf_map.update({uid: "unknown" for uid in mouse_units.index})
            continue

        wf_df = wf_df[join_cols + [label_col]].copy()
        wf_df["_wf_label"] = wf_df[label_col].map(_to_ww_nw)

        # Merge on the four key columns; keep unit_id (units_good's index) intact.
        merged = mouse_units[join_cols].reset_index(names="_uid").merge(
            wf_df[join_cols + ["_wf_label"]], on=join_cols, how="left")
        merged["_wf_label"] = merged["_wf_label"].fillna("unknown")

        # Guard against a unit matching >1 CSV row (duplicate keys) — keep first match,
        # but warn since this indicates the CSV/join keys aren't actually unique per unit.
        n_dupes = merged["_uid"].duplicated().sum()
        if n_dupes:
            print(f"  [waveform classification] {n_dupes} duplicate matches for mouse "
                 f"{mouse_id} in {csv_path} — keeping first match per unit")
            merged = merged.drop_duplicates(subset="_uid", keep="first")

        wf_map.update(dict(zip(merged["_uid"], merged["_wf_label"])))

    n_unknown = sum(1 for v in wf_map.values() if v == "unknown")
    if n_unknown:
        print(f"  [waveform classification] {n_unknown}/{len(wf_map)} units could not be "
             f"matched to a waveform-type CSV row and were marked 'unknown'")
    return wf_map

def split_event_map(event_map):
    """Split each (mouse, session, context, trial_type) group into odd/even by
    positional order within the group.  Odd = indices 0,2,4,… ; even = 1,3,5,…"""
    event_map_odd, event_map_even = {}, {}
    for key, events in event_map.items():
        event_map_odd[key]  = events[0::2]
        event_map_even[key] = events[1::2]
    return event_map_odd, event_map_even


def run_reward_group_stats(out_folder: Path | str) -> dict: # this is for rastermap only
    """Mouse-level R+/R− enrichment analysis across rastermap_psth clusters.

    Loads embedding_results.npz from *out_folder* (must contain mouse_arr and
    reward_arr saved by run_rastermap_psth).  All outputs go to
    out_folder/stats/.

    Pipeline
    --------
    1. Build f_{m,k} matrix  (n_mice × n_clusters): fractional neuron yield
       of each mouse in each cluster.  Mouse-level aggregation avoids
       pseudoreplication from neuron non-independence.
    2. PERMANOVA on full f matrix → global test of whether R+/R− centroids
       differ (permutation-based, no distributional assumptions).
    3. Per-cluster Mann-Whitney U (R+ vs R− on f_{m,k}) + BH-FDR correction.
    4. Strip/box plots per enriched cluster (and top-10 by p-value) showing
       per-mouse fractional yield colored by group.
    5. Sanity check: total neuron yield per mouse (raw counts) to confirm
       recording-size balance between groups.

    Returns
    -------
    dict with keys: f_matrix, mouse_ids, reward_groups, pvals_raw,
                    pvals_fdr, reject, permanova_p, permanova_F
    """
    from scipy.stats import mannwhitneyu
    out_folder = Path(out_folder)
    stats_dir  = out_folder / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load embedding results ──────────────────────────────────────────
    cv_result_file = [f for f in os.listdir(out_folder) if f.endswith('results_cv.npz')][0]
    emb_path = out_folder / cv_result_file
    if not emb_path.exists():
        raise FileNotFoundError(
            f"embedding_results.npz not found at {emb_path}.\n"
            "Run run_rastermap_psth first to generate it.")

    data           = np.load(emb_path, allow_pickle=True)
    cluster_labels = data["cluster_labels"]          # (n_neurons,)
    mouse_arr      = data["mouse_arr"].astype(str)   # (n_neurons,)
    reward_arr     = data["reward_arr"].astype(str)  # (n_neurons,) "R+" | "R-"
    n_clusters     = int(cluster_labels.max()) + 1

    # Needed for the combined per-cluster stats-matrix figure (column 3:
    # mean PSTH difference) — X is per-neuron concatenated PSTH, isort/boundaries
    # define rastermap_psth cluster order, n_bins_list/t_ctrs define condition layout.
    X              = data["X"]
    # isort and boundaries are rastermap-specific; derive from cluster_labels for other methods.
    if "isort" in data and "boundaries" in data:
        isort = data["isort"]
        boundaries = list(data["boundaries"])
    else:
        n_clusters_for_sort = int(cluster_labels.max()) + 1
        isort = np.argsort(cluster_labels, kind="stable")
        boundaries = np.cumsum(
            [(cluster_labels == k).sum() for k in range(n_clusters_for_sort)]
        )[:-1].tolist()
    #isort          = data["isort"]
    #boundaries     = data["boundaries"]
    n_bins_list    = list(data["n_bins_list"])
    n_conds        = int(data["n_conds"])
    t_ctrs         = [data[f"t_ctr_{ci}"] for ci in range(n_conds)]

    # ── 2. Build f_{m,k} matrix ───────────────────────────────────────────
    # Per-mouse fractional representation in each cluster.
    # f[m, k] = neurons_from_mouse_m_in_cluster_k / total_neurons_from_mouse_m
    # This corrects for different recording sizes across mice.
    mouse_ids_all = np.unique(mouse_arr)

    # Map each mouse to its reward group (R+ / R-)
    mouse_reward = {}
    for mid in mouse_ids_all:
        mask   = mouse_arr == mid
        groups = reward_arr[mask]
        vals, counts = np.unique(groups, return_counts=True)
        mouse_reward[mid] = vals[counts.argmax()]   # majority label (should be unique)

    # Keep only R+ and R- mice (drop "unknown")
    valid_mice    = [m for m in mouse_ids_all if mouse_reward[m] in ("R+", "R-")]
    reward_groups = np.array([mouse_reward[m] for m in valid_mice])
    n_mice        = len(valid_mice)

    f_matrix = np.zeros((n_mice, n_clusters), dtype=float)
    neuron_counts = np.zeros(n_mice, dtype=int)
    for i, mid in enumerate(valid_mice):
        mask              = mouse_arr == mid
        total             = mask.sum()
        neuron_counts[i]  = total
        if total == 0:
            continue
        for k in range(n_clusters):
            f_matrix[i, k] = ((cluster_labels[mask] == k).sum()) / total

    np.savez_compressed(
        stats_dir / "f_matrix.npz",
        f_matrix      = f_matrix,
        mouse_ids     = np.array(valid_mice),
        reward_groups = reward_groups,
        neuron_counts = neuron_counts,
    )
    print(f"  f_matrix shape: {f_matrix.shape}  "
          f"(R+: {(reward_groups=='R+').sum()}, R-: {(reward_groups=='R-').sum()} mice)")

    # ── 3. PERMANOVA (global test) ─────────────────────────────────────────
    # Permutation MANOVA on Euclidean distances between per-mouse f-vectors.
    # Tests whether R+/R- group centroids differ in cluster-usage space.
    # H0: group label is exchangeable with the distance matrix.
    permanova_F, permanova_p = _permanova(f_matrix, reward_groups, n_perm=9999)
    print(f"  PERMANOVA — F={permanova_F:.3f}  p={permanova_p:.4f}  (9999 permutations)")

    # ── 4. Per-cluster Mann-Whitney + BH-FDR ─────────────────────────────
    # For each cluster k, compare f_{m,k} between R+ and R- mice.
    # Mann-Whitney U is non-parametric and appropriate for small n_mice.
    mask_rp = reward_groups == "R+"
    mask_rm = reward_groups == "R-"
    pvals_raw = np.zeros(n_clusters)
    u_stats   = np.zeros(n_clusters)
    for k in range(n_clusters):
        fp = f_matrix[mask_rp, k]
        fm = f_matrix[mask_rm, k]
        if len(fp) < 2 or len(fm) < 2:
            pvals_raw[k] = 1.0
            continue
        stat, p        = mannwhitneyu(fp, fm, alternative="two-sided")
        u_stats[k]     = stat
        pvals_raw[k]   = p

    # Benjamini-Hochberg FDR correction
    pvals_fdr, reject = _bh_correction(pvals_raw, alpha=0.05)

    n_sig = reject.sum()
    print(f"  Per-cluster MW: {n_sig}/{n_clusters} clusters significant after BH-FDR (α=0.05)")

    # Save stats table
    stats_df = pd.DataFrame(dict(
        cluster    = np.arange(1, n_clusters + 1),
        U_stat     = u_stats,
        p_raw      = pvals_raw,
        p_fdr      = pvals_fdr,
        significant = reject,
        mean_frac_rplus = [f_matrix[mask_rp, k].mean() for k in range(n_clusters)],
        mean_frac_rminus= [f_matrix[mask_rm, k].mean() for k in range(n_clusters)],
    ))
    stats_df.to_csv(stats_dir / "per_cluster_stats.csv", index=False)

    # Save summary
    with open(stats_dir / "permanova_summary.txt", "w") as fh:
        fh.write(f"PERMANOVA (Euclidean, 9999 permutations)\n")
        fh.write(f"  F-statistic : {permanova_F:.4f}\n")
        fh.write(f"  p-value     : {permanova_p:.4f}\n\n")
        fh.write(f"Per-cluster Mann-Whitney U (BH-FDR alpha=0.05)\n")
        fh.write(f"  Significant clusters: {n_sig}/{n_clusters}\n")
        sig_clusters = np.where(reject)[0] + 1
        fh.write(f"  Cluster IDs: {list(sig_clusters)}\n")

    # ── 5. Figures ────────────────────────────────────────────────────────
    _fig_stats_overview(f_matrix, reward_groups, pvals_raw, pvals_fdr, reject,
                        permanova_F, permanova_p, n_clusters, stats_dir)

    _fig_strip_plots(f_matrix, reward_groups, pvals_fdr, reject, valid_mice, stats_dir)

    _fig_neuron_yield(neuron_counts, reward_groups, valid_mice, stats_dir)

    _fig_cluster_stats_matrix(f_matrix, reward_groups, pvals_fdr, reject,
                              permanova_F, permanova_p,
                              X, cluster_labels, reward_arr, isort, boundaries,
                              n_bins_list, t_ctrs,
                              n_clusters, stats_dir)

    return dict(
        f_matrix     = f_matrix,
        mouse_ids    = valid_mice,
        reward_groups= reward_groups,
        pvals_raw    = pvals_raw,
        pvals_fdr    = pvals_fdr,
        reject       = reject,
        permanova_F  = permanova_F,
        permanova_p  = permanova_p,
    )


# ── stats helper functions ─────────────────────────────────────────────────────

def _permanova(X: np.ndarray, groups: np.ndarray, n_perm: int = 9999) -> tuple[float, float]:
    """Pseudo-F PERMANOVA on Euclidean distances.

    Partitions total sum of squares into between-group and within-group
    components using Euclidean distance matrix, then permutes group labels
    to build a null distribution of the pseudo-F statistic.

    Parameters
    ----------
    X      : (n_samples, n_features) array of per-mouse cluster fractions
    groups : (n_samples,) array of group labels ("R+" / "R-")
    n_perm : number of permutations for the null distribution

    Returns
    -------
    F_obs : observed pseudo-F statistic
    p     : permutation p-value
    """
    labels       = np.unique(groups)
    n            = len(X)
    # Squared Euclidean distance matrix
    D2           = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)

    def _pseudo_f(D2, grp):
        """Compute pseudo-F from squared distance matrix and group vector."""
        SS_total = D2.sum() / (2 * n)
        SS_w = 0.0
        for lab in labels:
            idx    = np.where(grp == lab)[0]
            ni     = len(idx)
            if ni < 2:
                continue
            SS_w  += D2[np.ix_(idx, idx)].sum() / (2 * ni)
        SS_b    = SS_total - SS_w
        df_b    = len(labels) - 1
        df_w    = n - len(labels)
        if df_w <= 0 or SS_w == 0:
            return np.nan
        return (SS_b / df_b) / (SS_w / df_w)

    F_obs  = _pseudo_f(D2, groups)
    rng    = np.random.default_rng(42)
    null_F = np.array([_pseudo_f(D2, rng.permutation(groups)) for _ in range(n_perm)])
    p      = ((null_F >= F_obs).sum() + 1) / (n_perm + 1)
    return float(F_obs), float(p)


def _bh_correction(pvals: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Returns
    -------
    pvals_adj : BH-adjusted p-values (same length as pvals)
    reject    : boolean array, True where adjusted p < alpha
    """
    n     = len(pvals)
    order = np.argsort(pvals)
    rank  = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)

    # Adjusted p-values: p_adj[i] = min over j>=rank[i] of (p[j] * n / j)
    pvals_adj = np.minimum(1.0, pvals * n / rank)
    # Enforce monotonicity from largest to smallest rank
    pvals_adj_sorted = pvals_adj[order]
    for i in range(n - 2, -1, -1):
        pvals_adj_sorted[i] = min(pvals_adj_sorted[i], pvals_adj_sorted[i + 1])
    pvals_adj[order] = pvals_adj_sorted

    reject = pvals_adj < alpha
    return pvals_adj, reject


def _fig_stats_overview(f_matrix, reward_groups, pvals_raw, pvals_fdr, reject,
                        permanova_F, permanova_p, n_clusters, stats_dir):
    """Three-panel overview figure.

    Panel A: mean fractional yield per cluster for R+ vs R-, with FDR-sig markers.
    Panel B: -log10(p_fdr) per cluster with significance threshold line.
    Panel C: PERMANOVA result as text summary.
    """
    mask_rp   = reward_groups == "R+"
    mask_rm   = reward_groups == "R-"
    mean_rp   = f_matrix[mask_rp].mean(axis=0)
    mean_rm   = f_matrix[mask_rm].mean(axis=0)
    sem_rp    = f_matrix[mask_rp].std(axis=0) / np.sqrt(mask_rp.sum())
    sem_rm    = f_matrix[mask_rm].std(axis=0) / np.sqrt(mask_rm.sum())
    x         = np.arange(n_clusters)

    fig, axes = plt.subplots(3, 1, figsize=(max(10, n_clusters * 0.35), 12),
                             gridspec_kw={"height_ratios": [3, 2, 1]})

    # ── A: mean ± SEM per cluster ─────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(x, mean_rp - sem_rp, mean_rp + sem_rp, alpha=0.25, color="forestgreen")
    ax.fill_between(x, mean_rm - sem_rm, mean_rm + sem_rm, alpha=0.25, color="crimson")
    ax.plot(x, mean_rp, color="forestgreen", lw=1.5, label=f"R+ (n={mask_rp.sum()})")
    ax.plot(x, mean_rm, color="crimson",     lw=1.5, label=f"R− (n={mask_rm.sum()})")
    for k in np.where(reject)[0]:
        ax.axvspan(k - 0.4, k + 0.4, color="gold", alpha=0.35, zorder=0)
    ax.set_xlim(-0.5, n_clusters - 0.5)
    ax.set_ylabel("Mean fractional yield  f_{m,k}")
    ax.set_title("Mean ± SEM cluster fraction per reward group\n"
                 "(gold shading = FDR-significant clusters)", fontsize=9)
    ax.legend(fontsize=8)

    # ── B: -log10(p_fdr) ─────────────────────────────────────────────────
    ax = axes[1]
    neg_log_p = -np.log10(np.clip(pvals_fdr, 1e-10, 1))
    colors    = ["gold" if r else "steelblue" for r in reject]
    ax.bar(x, neg_log_p, color=colors, edgecolor="none", width=0.8)
    ax.axhline(-np.log10(0.05), color="k", ls="--", lw=1.2,
               label="FDR α = 0.05")
    ax.set_xlim(-0.5, n_clusters - 0.5)
    ax.set_xlabel("Rastermap cluster")
    ax.set_ylabel("−log₁₀(p_FDR)")
    ax.set_title("Per-cluster significance (BH-corrected Mann-Whitney U)", fontsize=9)
    ax.legend(fontsize=8)

    # ── C: PERMANOVA summary ──────────────────────────────────────────────
    ax = axes[2]
    ax.axis("off")
    p_str = f"p = {permanova_p:.4f}" if permanova_p >= 0.001 else f"p < 0.001"
    ax.text(0.5, 0.6,
            f"PERMANOVA (global test, 9999 permutations): "
            f"F = {permanova_F:.3f},  {p_str}",
            ha="center", va="center", fontsize=10,
            transform=ax.transAxes,
            color="darkred" if permanova_p < 0.05 else "dimgrey")

    fig.suptitle("R+/R− enrichment across rastermap_psth clusters", fontsize=11)
    fig.tight_layout()
    _save(fig, stats_dir / "figS1_stats_overview", dpi=400)


def _fig_strip_plots(f_matrix, reward_groups, pvals_fdr, reject, mouse_ids, stats_dir):
    """Strip + box plots for clusters of interest.

    Shows all FDR-significant clusters; if none, shows the top-10 by p_fdr.
    Each panel: per-mouse fractional yield as dots (R+ green, R- red) with
    a box overlaid.  Title shows cluster index and FDR p-value.
    """
    sig_idx = np.where(reject)[0]
    if len(sig_idx) == 0:
        # Fall back to top-10 by p_fdr so the figure is always informative
        sig_idx = np.argsort(pvals_fdr)[:10]
        fallback = True
    else:
        fallback = False

    n_panels = len(sig_idx)
    ncols    = min(5, n_panels)
    nrows    = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5 * ncols, 3.5 * nrows),
                             squeeze=False)

    rng_jitter = np.random.default_rng(0)
    for ax_i, k in enumerate(sig_idx):
        ax     = axes[ax_i // ncols][ax_i % ncols]
        fp     = f_matrix[reward_groups == "R+", k]
        fm     = f_matrix[reward_groups == "R-", k]

        # box plot underneath (no fliers — dots will show them)
        bp = ax.boxplot(
            [fp, fm], positions=[0, 1], widths=0.35,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="k", lw=2),
        )
        for patch, col in zip(bp["boxes"], ["forestgreen", "crimson"]):
            patch.set_facecolor(col); patch.set_alpha(0.25)

        # strip plot with jitter
        for xi, (vals, col) in enumerate([(fp, "forestgreen"), (fm, "crimson")]):
            jitter = rng_jitter.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(xi + jitter, vals, color=col, s=30, alpha=0.85,
                       edgecolors="none", zorder=3)

        p_str = f"p_FDR={pvals_fdr[k]:.3f}" if pvals_fdr[k] >= 0.001 else "p_FDR<0.001"
        ax.set_title(f"Cluster {k+1}\n{p_str}", fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["R+", "R−"], fontsize=9)
        ax.set_ylabel("f_{m,k}" if ax_i % ncols == 0 else "")
        ax.set_xlim(-0.5, 1.5)

    # Hide unused panels
    for ax_i in range(n_panels, nrows * ncols):
        axes[ax_i // ncols][ax_i % ncols].set_visible(False)

    title = ("Top-10 clusters by FDR p-value (none significant at α=0.05)"
             if fallback else
             f"FDR-significant clusters (n={n_panels}, α=0.05)")
    fig.suptitle(f"Per-mouse fractional yield by reward group — {title}", fontsize=10)
    fig.tight_layout()
    _save(fig, stats_dir / "figS2_strip_plots", dpi=400)


def _fig_neuron_yield(neuron_counts, reward_groups, mouse_ids, stats_dir):
    """Sanity check: total recorded neurons per mouse, split by R+/R-.

    A large imbalance in neuron counts between groups could confound
    the f_{m,k} analysis (lower-yield mice have noisier fractions).
    """
    from scipy.stats import mannwhitneyu
    mask_rp = reward_groups == "R+"
    mask_rm = reward_groups == "R-"
    fp      = neuron_counts[mask_rp].astype(float)
    fm      = neuron_counts[mask_rm].astype(float)

    _, p = mannwhitneyu(fp, fm, alternative="two-sided") if (len(fp) > 1 and len(fm) > 1) else (np.nan, np.nan)

    fig, ax = plt.subplots(figsize=(5, 4))
    rng_j   = np.random.default_rng(1)
    for xi, (vals, col, lab) in enumerate(
            [(fp, "forestgreen", "R+"), (fm, "crimson", "R-")]):
        jitter = rng_j.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(xi + jitter, vals, color=col, s=40, alpha=0.8,
                   edgecolors="none", label=lab, zorder=3)
        ax.plot([xi - 0.2, xi + 0.2], [np.median(vals)] * 2,
                color=col, lw=2.5, zorder=4)

    p_str = f"MW p = {p:.3f}" if (p is not np.nan and p >= 0.001) else ("MW p < 0.001" if p is not np.nan else "")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["R+", "R−"], fontsize=11)
    ax.set_ylabel("Neurons recorded per mouse")
    ax.set_title(f"Sanity check: neuron yield per mouse\n{p_str}", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, stats_dir / "figS3_neuron_yield_sanity", dpi=400)

def _fig_cluster_stats_matrix(f_matrix, reward_groups, pvals_fdr, reject,
                              permanova_F, permanova_p,
                              X, cluster_labels, reward_arr, isort, boundaries,
                              n_bins_list, t_ctrs,
                              n_clusters, stats_dir):
    """Vertical, cluster-indexed summary figure (rows = clusters 1..n, in
    rastermap_psth output order — matches fig5_population_matrix / fig6 layout).

    Five columns, sharing the y-axis (cluster index):
      1. f_{m,k}        — group mean ± SEM as a dot-and-errorbar per group
                          (R+ green, R− red), no per-mouse dots.
      2. p-value        — dot-and-line (lollipop) per cluster: grey if not
                          significant, black if FDR-significant, annotated
                          with the cluster id when significant.
      3. Mean PSTH      — pooled (both groups) mean response vector per
                          cluster, drawn as one thin line trace per row
                          (vertically rescaled per cluster for visibility),
                          black for FDR-significant clusters, grey otherwise.
      4. Mean PSTH      — same data as column 3, but as a heatmap.
      5. ΔPSTH (R+−R−)  — mean PSTH difference between groups for that
                          cluster's neurons, one heatmap row per cluster,
                          condition boundaries marked as in _draw_matrix.

    Annotated with the global PERMANOVA result (F, p) as a figure-level title.
    """
    edges      = [0] + list(boundaries) + [len(isort)]
    mask_rp    = reward_groups == "R+"
    mask_rm    = reward_groups == "R-"

    # ── precompute columns 3/4 & 5: per-cluster mean PSTH and ΔPSTH ────────
    offsets   = np.concatenate([[0], np.cumsum(n_bins_list)]).astype(int)
    n_total   = offsets[-1]
    mean_mat  = np.full((n_clusters, n_total), np.nan)
    diff_mat  = np.full((n_clusters, n_total), np.nan)
    for k in range(n_clusters):
        idx       = isort[edges[k]:edges[k + 1]]
        if len(idx):
            mean_mat[k] = X[idx].mean(0)
        rg_mask   = reward_arr[idx] == "R+"
        if rg_mask.any() and (~rg_mask).any():
            diff_mat[k] = X[idx[rg_mask]].mean(0) - X[idx[~rg_mask]].mean(0)
    vmax_mean = np.nanpercentile(np.abs(mean_mat), 95)
    vmax_mean = vmax_mean if vmax_mean > 0 else 1.0
    vmax_diff = np.nanpercentile(np.abs(diff_mat), 95)
    vmax_diff = vmax_diff if vmax_diff > 0 else 1.0

    # ── figure & axes (rows = clusters, top row = cluster 1) ───────────────
    fig_h = max(10, n_clusters * 0.16)
    fig, axes = plt.subplots(
        1, 5, figsize=(19, fig_h),
        gridspec_kw={"width_ratios": [1.6, 1.2, 3.0, 3.0, 3.0], "wspace": 0.08})
    ax_f, ax_p, ax_line, ax_mean, ax_diff = axes

    y_inv = n_clusters - np.arange(n_clusters)   # cluster k(0-idx) -> row position, cluster 1 at top

    # ── Column 1: f_{m,k} group mean ± SEM (no per-mouse dots) ─────────────
    mean_rp  = f_matrix[mask_rp].mean(axis=0)
    mean_rm  = f_matrix[mask_rm].mean(axis=0)
    sem_rp   = f_matrix[mask_rp].std(axis=0) / np.sqrt(max(mask_rp.sum(), 1))
    sem_rm   = f_matrix[mask_rm].std(axis=0) / np.sqrt(max(mask_rm.sum(), 1))
    dy       = 0.18   # vertical offset between the two group dots within a row
    ax_f.errorbar(mean_rp, y_inv + dy, xerr=sem_rp, fmt="o", color="forestgreen",
                 ms=3, lw=0.8, elinewidth=0.8, capsize=0, zorder=3)
    ax_f.errorbar(mean_rm, y_inv - dy, xerr=sem_rm, fmt="o", color="crimson",
                 ms=3, lw=0.8, elinewidth=0.8, capsize=0, zorder=3)
    ax_f.set_ylim(0.5, n_clusters + 0.5)
    ax_f.set_yticks([])
    ax_f.set_ylabel("")
    ax_f.spines[["top", "right", "left"]].set_visible(False)
    ax_f.tick_params(left=False)
    ax_f.set_xlabel(r"Fraction $f_{m,k}$", fontsize=8)
    ax_f.set_title("Mean ± SEM", fontsize=9)
    from matplotlib.lines import Line2D
    ax_f.legend(handles=[
        Line2D([0], [0], marker="o", color="forestgreen", lw=0, label="R+"),
        Line2D([0], [0], marker="o", color="crimson",     lw=0, label="R−"),
    ], fontsize=6, loc="upper right", frameon=False)

    # ── Column 2: p-value lollipop plot ─────────────────────────────────────
    neg_log_p = -np.log10(np.clip(pvals_fdr, 1e-10, 1))
    dot_colors = ["black" if r else "darkgrey" for r in reject]
    ax_p.hlines(y_inv, 0, neg_log_p, color=dot_colors, lw=1.0, zorder=2)
    ax_p.scatter(neg_log_p, y_inv, s=10, color=dot_colors, zorder=3, edgecolors="none")
    ax_p.axvline(-np.log10(0.05), color="k", ls="--", lw=0.8, alpha=0.6, zorder=1)
    for k in np.where(reject)[0]:
        ax_p.annotate(str(k + 1), (neg_log_p[k], y_inv[k]),
                      xytext=(4, 0), textcoords="offset points",
                      fontsize=5.5, va="center", ha="left", color="black")
    ax_p.set_ylim(0.5, n_clusters + 0.5)
    ax_p.set_yticks([])
    ax_p.spines[["top", "right", "left"]].set_visible(False)
    ax_p.tick_params(left=False)
    ax_p.set_xlabel("−log₁₀(p-val)", fontsize=8)
    ax_p.set_title("Significance", fontsize=9)

    # x-tick positions (condition centers), shared by all PSTH-based columns
    tick_pos = [(offsets[ci] + offsets[ci + 1]) / 2 for ci in range(len(n_bins_list))]

    # ── Column 3: mean PSTH as one line trace per cluster row ──────────────
    # Each cluster's mean PSTH vector is z-scored to its own peak-to-peak range
    # and rescaled to a fixed vertical amplitude so all 100 traces stack
    # within their row without overlapping neighbours. The trace is drawn as
    # one separate line segment per condition (a gap at each boundary instead
    # of a continuous line), and a thin grey vertical marks each
    # segment's alignment bin (t=0, i.e. start_time or jaw_onset_time).
    trace_amp = 0.42   # max vertical excursion from row center, in row units
    for k in range(n_clusters):
        v = mean_mat[k]
        if not np.isfinite(v).any():
            continue
        rng_v = np.nanmax(v) - np.nanmin(v)
        rng_v = rng_v if rng_v > 0 else 1.0
        v_norm = (v - np.nanmean(v)) / rng_v * (2 * trace_amp)
        color  = "black" if reject[k] else "darkgrey"
        lw     = 0.8 if reject[k] else 0.5
        zord   = 3 if reject[k] else 2
        for ci, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            ax_line.plot(np.arange(start, end), y_inv[k] + v_norm[start:end],
                        color=color, lw=lw, zorder=zord)
    for ci, t_ctr_c in enumerate(t_ctrs):
        onset_idx = np.argmin(np.abs(t_ctr_c))      # alignment bin (start_time or jaw_onset_time)
        onset_x   = offsets[ci] + onset_idx
        ax_line.axvline(onset_x, color="grey", lw=0.5, ls=":", alpha=0.8, zorder=1)
    ax_line.set_xlim(0, n_total)
    ax_line.set_ylim(0.5, n_clusters + 0.5)
    ax_line.set_yticks([])
    ax_line.set_xticks(tick_pos)
    ax_line.set_xticklabels(COND_LABELS_MATRIX, fontsize=6)
    for s in ax_line.spines.values():
        s.set_visible(False)
    ax_line.set_title("Mean PSTH (pooled)\nline per cluster", fontsize=9)

    # ── Column 4: mean PSTH heatmap (pooled across groups) ─────────────────
    im_mean = ax_mean.imshow(mean_mat, aspect="auto", interpolation="none",
                             cmap="viridis", vmin=-vmax_mean, vmax=vmax_mean,
                             extent=[0, n_total, 0.5, n_clusters + 0.5])
    for ci, start in enumerate(offsets[1:-1]):
        ax_mean.axvline(start, color="w", lw=0.8)
    ax_mean.set_xticks(tick_pos)
    ax_mean.set_xticklabels(COND_LABELS_MATRIX, fontsize=4)
    ax_mean.set_yticks([])
    for s in ax_mean.spines.values():
        s.set_visible(False)
    ax_mean.set_title("Mean PSTH (pooled)\nheatmap", fontsize=9)
    cbar_mean = fig.colorbar(im_mean, ax=ax_mean, shrink=0.25, pad=0.01)
    label_txt ='Firing rate (z-score)' if DEFAULT_CFG['normalize']=='zscore' else 'Firing rate (spks/s)'
    cbar_mean.set_label(label_txt, fontsize=7)
    cbar_mean.ax.tick_params(labelsize=6)
    cbar_mean.outline.set_visible(False)

    # ── Column 4: ΔPSTH heatmap (R+ − R−), one row per cluster ─────────────
    im = ax_diff.imshow(diff_mat, aspect="auto", interpolation="none",
                        cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff,
                        extent=[0, n_total, 0.5, n_clusters + 0.5])
    for ci, start in enumerate(offsets[1:-1]):
        ax_diff.axvline(start, color="k", lw=0.8)
    ax_diff.set_xticks(tick_pos)
    ax_diff.set_xticklabels(COND_LABELS_MATRIX, fontsize=6)
    ax_diff.set_yticks([])
    for s in ax_diff.spines.values():
        s.set_visible(False)
    ax_diff.set_title("Mean PSTH difference\n(R+ − R−)", fontsize=9)
    cbar = fig.colorbar(im, ax=ax_diff, shrink=0.25, pad=0.01)
    cbar.set_label(f"Δ {label_txt}", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_visible(False)

    p_str = f"p = {permanova_p:.4f}" if permanova_p >= 0.001 else "p < 0.001"
    fig.suptitle(
        f"Per-cluster R+/R− statistics  —  PERMANOVA: F = {permanova_F:.3f},  {p_str}  "
        f"({reject.sum()}/{n_clusters} clusters FDR-significant)",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, stats_dir / "figS4_cluster_stats_matrix",dpi=400)




# ── stats-only entry point ───────────────────H─────────────────────────────────

def run_stats_only(out_folder: str | Path) -> dict:
    """Re-run the reward-group enrichment analysis on an existing embedding.

    Use this when you want to run or re-run statistics without recomputing
    the full rastermap_psth pipeline.  All required data (cluster_labels, mouse_arr,
    reward_arr) must already be present in embedding_results.npz — which is
    guaranteed if the embedding was produced by run_rastermap_psth with the
    current version of this script.

    Parameters
    ----------
    out_folder : path to the output folder that contains embedding_results.npz
                 (i.e. the same folder passed to run_rastermap_psth as out_root/
                 sub-path).

    Returns
    -------
    Same dict as run_reward_group_stats.
    """
    out_folder = Path(out_folder)
    emb_path   = out_folder / "cv_embedding_results.npz"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"No embedding_results.npz found at:\n  {emb_path}\n"
            "Run run_rastermap_psth first.")

    # Check that required per-neuron arrays were saved
    with np.load(emb_path, allow_pickle=True) as data:
        for key in ("mouse_arr", "reward_arr", "cluster_labels"):
            if key not in data:
                raise KeyError(
                    f"Key '{key}' missing from embedding_results.npz.\n"
                    "Re-run run_rastermap_psth with the current script to "
                    "regenerate the embedding with metadata included.")

    print(f"Running stats-only pipeline on {out_folder} ...")
    return run_reward_group_stats(out_folder)

# ── config loading ─────────────────────────────────────────────────────────────

def load_cfg(yaml_path: str | Path, **overrides) -> dict:
    """Load config.yaml and apply any keyword overrides.

    Reconstructs Python types that YAML cannot express natively:
      k_elbow_min / k_elbow_max  ->  k_elbow_range = range(min, max)
      artifact_win_s             ->  tuple
    """
    import yaml
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    cfg["k_elbow_range"] = range(
        cfg.pop("k_elbow_min", 2),
        cfg.pop("k_elbow_max", 30),
    )

    if "artifact_win_s" in cfg:
        cfg["artifact_win_s"] = tuple(cfg["artifact_win_s"])

    cfg.setdefault("n_jobs", N_WORKERS)
    cfg.update(overrides)
    return cfg