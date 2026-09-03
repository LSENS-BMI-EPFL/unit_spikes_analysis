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

from ephys_utilities.plotting_utils import plotting_utils
import ephys_utilities.allen_utils.allen_utils as allen_utils
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from rastermap import Rastermap
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import fisher_exact, kruskal, mannwhitneyu, chisquare
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from ephys_utilities.plotting_utils import *
rplus_color = plotting_utils.GROUP_COLORS['rplus']
rminus_color = plotting_utils.GROUP_COLORS['rminus']

# ── config ─────────────────────────────────────────────────────────────────────
# TODO: collapse pre/post passive trials into one for only sensory responses?
# TODO: save more anat and metadata of neurons for analysis to avoid loading tables several times,.,.,
# TODO: order of clustering kmeans/GMMs applied on ordered rastermap_psth cmatrix
# TODO: make axon cmap logscale too? or data as logscale? not mean over neurons? but areas?


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
    n_min_trial_per_condition = 3, # trial count for spit-trials
    period                = 'active_passive', # "active_passive", "passive"', "active"
    t_pre_passive        = 0.2,    # pre-stimulus window for passive conditions (s)
    t_post_passive       = 0.5,    # post-stimulus window for passive conditions (s)
    t_pre_active         = 0.2,    # pre-stimulus window for active conditions (s)
    t_post_active        = 0.5,    # post-stimulus window for active conditions (s)
    t_pre_jaw            = 0.35,   # pre-jaw-onset window (s)
    t_post_jaw           = 0.35,   # post-jaw-onset window (s)
    bin_ms               = 10,
    sigma_ms             = 0,
    artifact_win_s       = (-0.010, 0.005),
    whisker_trial_label  = "whisker_trial",
    global_fr_hz         = 0.01, # threshold on the whole-session
    fr_threshold_hz      = 0.1, # treshold on psth data
    square_fr            = False,
    baseline_removal     = True,   # if False: skip all per-trial baseline subtraction (own-window AND jaw-borrowed), for each neuron and trial
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
    n_rastermap_clusters = 100,
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
        ("whisker_trial",  "active_nolick",  "Whisker miss",       "#d6371e", "start_time"),
        ("whisker_trial",  "active_lick",    "Whisker hit",        "#fcba03", "start_time"),
        ("auditory_trial", "active_lick",    "Auditory hit",       "#7c0082", "start_time"),
        #("no_stim_trial",  "active_lick",    "False alarm",        "#211f21", "start_time"),
        #("no_stim_trial",  "active_nolick",  "Correct rej.",       "#a6a4a1", "start_time"),
        # Jaw-aligned conditions (lighter shades of the start_time counterparts)
        #("whisker_trial",  "active_lick",    "Whisker hit (jaw)",  "#fde89a", "jaw_onset_time"),
        #("auditory_trial", "active_lick",    "Auditory hit (jaw)", "#c9aadd", "jaw_onset_time"),
        #("no_stim_trial",  "active_lick",    "False alarm (jaw)",  "#999999", "jaw_onset_time"),
    ]

    # Lick-aligned conditions — only included when include_lick_conditions=True
    # and a lick_df has been provided (populated into event_map via add_lick_event_map).
    # Appended last so existing condition indices are never disturbed when flag is False.
    if cfg.get("include_lick_conditions", False):
        active_all += [
            ("spontaneous_lick", "spontaneous", "Spont. lick", "#999999", "lick_time"),
            ("whisker_trial", "active_lick", "Whisker hit (lick)",  "#fcba03", "lick_time"),
            ("auditory_trial", "active_lick", "Auditory hit (lick)",  "#7c0082", "lick_time"),
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
    if align_col == "lick_time":
        return cfg["t_pre_lick"], cfg["t_post_lick"]
    if align_col == "jaw_onset_time":
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
                         is_whisker=False, artifact_win_s=(-0.01, 0.005), rng=None,
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
    print('Excluding mice without passive data...:', mice_with_no_passive)
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

    estimated_second_lick_latency = 0.15

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
            event_map[(mouse_id, session_id, "spontaneous", "reward_lick", "lick_time")] = np.sort(reward_times) + estimated_second_lick_latency
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
                    artifact_win_s=cfg["artifact_win_s"], rng=rng,
                    align_col=acol)
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

    #psths = [(p - mean_) / std_ for p in psths]
    psths = [(p) / std_ for p in psths]

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
    canonical_order = allen_utils.get_area_group_custom_order()

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



def run_reward_group_stats(out_folder: Path | str, cfg: dict, unit_table: pd.DataFrame,
                            plot_all_clusters: bool = False) -> dict:
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
    permanova_F, permanova_p = _permanova(f_matrix, reward_groups, n_perm=500)
    print(f"  PERMANOVA — F={permanova_F:.3f}  p={permanova_p:.4f}  (500 permutations)")

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

    # ── condition metadata for post-hoc figures ────────────────────────────
    conds, cond_labels, cond_colors, _, cond_align_cols = get_conditions(cfg)
    row_groups = _condition_rows(conds, cond_align_cols)
    edge_trim = _edge_trim_bins(cfg)
    area_arr, area_src = _load_area_arr(out_folder, data["unit_ids"])
    print(f"  Loaded area_acronym from {area_src.name}")

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

    # Area compo all clusters
   # _fig_cluster_area_composition(cluster_labels, area_arr, mouse_arr, pvals_fdr, reject, stats_dir.parent, mode="all")


    # Activity profiles by trial types
    _fig_sigclusters_by_trialtype(X, cluster_labels, mouse_arr, pvals_fdr, reject,
                                   n_bins_list, t_ctrs, cond_labels, row_groups,
                                   edge_trim, stats_dir)

    # TODO: plot the activity of these neurons, not the z-scored.
    _fig_sigclusters_by_trialtype_by_rewardgroup(X, cluster_labels, mouse_arr, reward_arr, pvals_fdr, reject,
                                   n_bins_list, t_ctrs, cond_labels, row_groups,
                                   edge_trim, stats_dir)

    # ── area composition ────────────────────────────────────────────────────
    mode = "significant"

    area_arr, area_src = _load_area_arr(out_folder, data["unit_ids"])
    print(f"  Loaded area_acronym from {area_src.name}")
    _fig_cluster_area_composition(cluster_labels, area_arr, mouse_arr,
                                  pvals_fdr, reject, stats_dir, mode=mode)

    # ── spatial location ─────────────────────────────────────────────────────
    mode = "significant"
    sig_clusters = _select_clusters_to_plot(pvals_fdr, reject, mode=mode)

    coords_bregma = _load_bregma_coords(unit_table, data["unit_ids"])
    coords_ccf = _load_ccf_atlas_coords(unit_table, data["unit_ids"])

    loc_dir = stats_dir / "sig_clusters_location"
    if not os.path.exists(loc_dir) and mode=="significant":
        os.makedirs(loc_dir)

    loc_df, baseline_centroid = _permutation_location_vs_dataset(
        cluster_labels, coords_bregma, mouse_arr, sig_clusters, n_perm=500)
    loc_df.to_csv(loc_dir / "location_vs_dataset_stats.csv", index=False)

    rg_loc_df = _permutation_location_by_rewardgroup(
        cluster_labels, coords_bregma, mouse_arr, reward_arr, sig_clusters, n_perm=500)
    rg_loc_df.to_csv(loc_dir / "location_by_rewardgroup_stats.csv", index=False)

    _fig_location_vs_dataset_detail(cluster_labels, coords_bregma, mouse_arr, sig_clusters,
                                    loc_df, baseline_centroid, loc_dir)
    _fig_location_by_rewardgroup(cluster_labels, coords_bregma, mouse_arr, reward_arr,
                                 sig_clusters, rg_loc_df, loc_dir)

    dev_df, baseline_centroid = compute_cluster_centroid_deviation(
        cluster_labels, coords_bregma, mouse_arr, clusters_to_test=sig_clusters, n_perm=2000)
    dev_df.to_csv(loc_dir / "cluster_centroid_deviation.csv", index=False)
    _fig_cluster_centroid_deviation(dev_df, loc_dir)

    #render_cluster_location_and_density_grid(cluster_labels, coords_ccf, sig_clusters, loc_dir,
    #                                          cluster_selection=mode)

    ATLAS_PATH = Path("/mnt/lsens-analysis/Axel_Bisi/Anatomy/allen_mouse_bluebrain_barrels_10um_v1.0")
    _fig_cluster_atlas_projections(cluster_labels, coords_ccf, sig_clusters, loc_dir,
                                    atlas_path=ATLAS_PATH)

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
    ax.fill_between(x, mean_rp - sem_rp, mean_rp + sem_rp, alpha=0.25, color=rplus_color)
    ax.fill_between(x, mean_rm - sem_rm, mean_rm + sem_rm, alpha=0.25, color=rminus_color)
    ax.plot(x, mean_rp, color=rplus_color, lw=1.5, label=f"R+ (n={mask_rp.sum()})")
    ax.plot(x, mean_rm, color=rminus_color,     lw=1.5, label=f"R− (n={mask_rm.sum()})")
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
        for patch, col in zip(bp["boxes"], [rplus_color, rminus_color]):
            patch.set_facecolor(col); patch.set_alpha(0.25)

        # strip plot with jitter
        for xi, (vals, col) in enumerate([(fp, rplus_color), (fm, rminus_color)]):
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
            [(fp, rplus_color, "R+"), (fm, rminus_color, "R-")]):
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
    ax_f.errorbar(mean_rp, y_inv + dy, xerr=sem_rp, fmt="o", color=rplus_color,
                 ms=3, lw=0.8, elinewidth=0.8, capsize=0, zorder=3)
    ax_f.errorbar(mean_rm, y_inv - dy, xerr=sem_rm, fmt="o", color=rminus_color,
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
        Line2D([0], [0], marker="o", color=rplus_color, lw=0, label="R+"),
        Line2D([0], [0], marker="o", color=rminus_color,     lw=0, label="R−"),
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
    return


def _condition_rows(conds, cond_align_cols):
    """Group condition indices into display rows: passive, active (start_time),
    active (jaw), active (lick) — in that canonical order, skipping empty rows.
    """
    rows = {"passive": [], "active_start": [], "active_jaw": [], "active_lick": []}
    for i, ((tt, ctx), acol) in enumerate(zip(conds, cond_align_cols)):
        if ctx.startswith("passive"):
            rows["passive"].append(i)
        elif acol == "lick_time":
            rows["active_lick"].append(i)
        elif acol == "jaw_onset_time":
            rows["active_jaw"].append(i)
        else:
            rows["active_start"].append(i)
    order = ["passive", "active_start", "active_jaw", "active_lick"]
    return [idxs for name in order if (idxs := rows[name])]


def _edge_trim_bins(cfg):
    """Number of bins at each edge of every condition segment biased by the
    zero-padding artifact in _bin_and_smooth (see notes)."""
    sigma_bins = cfg["sigma_ms"] / cfg["stride_ms"]
    return int(np.ceil(4 * sigma_bins))


def _load_area_arr(out_folder, unit_ids):
    """Load area_acronym per neuron from the CV metadata CSV, aligned to
    unit_ids by an explicit merge (not position)."""
    out_folder = Path(out_folder)

    for name in ("neuron_cluster_labels_cv.csv",""):
        for candidate in (out_folder / name, out_folder.parent / name):
            if candidate.exists():
                meta_df = pd.read_csv(candidate).set_index("unit_ids")
                area_arr = meta_df.reindex(unit_ids)["area_acronym"].to_numpy()
                missing = pd.isna(area_arr).sum()
                if missing:
                    print(f"  Warning: {missing}/{len(unit_ids)} unit_ids not found in {candidate.name}")
                return area_arr, candidate
    raise FileNotFoundError(
        f"Could not find neuron_cluster_labels[_cv].csv near {out_folder}")


def _select_clusters_to_plot(pvals_fdr, reject, top_n_fallback=10, mode="significant"):
    """mode: "significant" (BH-FDR sig, fallback top-N by p_fdr) or "all"
    (every cluster, ignoring the reward-group stats entirely)."""
    if mode == "all":
        return np.arange(len(pvals_fdr))
    sig_idx = np.where(reject)[0]
    if len(sig_idx) == 0:
        print(f"  No clusters significant after FDR — plotting top-{top_n_fallback} by p_fdr instead")
        sig_idx = np.argsort(pvals_fdr)[:top_n_fallback]
    return sig_idx


def _fig_sigclusters_by_trialtype(X, cluster_labels, mouse_arr, pvals_fdr, reject,
                                   n_bins_list, t_ctrs, cond_labels, row_groups,
                                   edge_trim, stats_dir):
    """One figure per sig cluster: grid of subplots (rows = passive / active-start
    / active-lick), each cell one trial-type condition, mean ± SEM over neurons.
    Edge bins biased by the smoothing/window artifact are trimmed from display.
    Y-axis shared across all subplots in the figure; x shared within each row
    (rows have different alignment windows). Every axis is individually labeled.
    """
    out_dir = stats_dir / "sig_clusters_by_trialtype"
    out_dir.mkdir(parents=True, exist_ok=True)
    starts = np.cumsum([0] + list(n_bins_list[:-1]))
    ends   = np.cumsum(n_bins_list)
    slices = list(zip(starts, ends))

    n_rows = len(row_groups)
    n_cols = max(len(r) for r in row_groups)

    for k in _select_clusters_to_plot(pvals_fdr, reject):
        neurons_k = cluster_labels == k
        n_k = int(neurons_k.sum())
        if n_k == 0:
            continue
        n_mice_k = len(np.unique(mouse_arr[neurons_k]))
        X_k = X[neurons_k]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows),
                                  squeeze=False)

        # first pass: compute all traces + a shared y-range
        all_means, all_sems = {}, {}
        y_min, y_max = np.inf, -np.inf
        for row_idx, idxs in enumerate(row_groups):
            for col_idx, ci in enumerate(idxs):
                s, e = slices[ci]
                t = t_ctrs[ci]
                trim = min(edge_trim, (e - s) // 3)  # never trim more than 1/3 of a window
                seg  = X_k[:, s + trim : e - trim]
                t_tr = t[trim: len(t) - trim] if trim else t
                mean = seg.mean(axis=0)
                sem  = seg.std(axis=0, ddof=1) / np.sqrt(n_k) if n_k > 1 else np.zeros_like(mean)
                all_means[(row_idx, col_idx)] = (t_tr, mean, sem, ci)
                y_min = min(y_min, (mean - sem).min())
                y_max = max(y_max, (mean + sem).max())

        pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
        y_min, y_max = y_min - pad, y_max + pad

        for row_idx, idxs in enumerate(row_groups):
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                if col_idx >= len(idxs):
                    ax.axis("off")
                    continue
                t_tr, mean, sem, ci = all_means[(row_idx, col_idx)]
                ax.plot(t_tr, mean, color="k", lw=1.5)
                ax.fill_between(t_tr, mean - sem, mean + sem, color="k", alpha=0.25)
                ax.axvline(0, ls="--", color="gray", lw=0.8)
                ax.set_ylim(y_min, y_max)
                ax.set_title(cond_labels[ci], fontsize=9)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("z-score (mean ± SEM)")

        fig.suptitle(f"Cluster {k + 1}  (n={n_k} neurons, {n_mice_k} mice, "
                     f"p_fdr={pvals_fdr[k]:.3g})")
        fig.tight_layout()
        fig.savefig(out_dir / f"cluster_{k + 1:03d}_by_trialtype.png", dpi=150)
        plt.close(fig)

    print(f"  Saved per-cluster trial-type figures → {out_dir}")

def _condition_slices(n_bins_list):
    """(start, end) index pairs into the concatenated PSTH axis of X, one per condition."""
    starts = np.cumsum([0] + list(n_bins_list[:-1]))
    ends   = np.cumsum(n_bins_list)
    return list(zip(starts, ends))

def _fig_sigclusters_by_trialtype_by_rewardgroup(X, cluster_labels, mouse_arr, reward_arr,
                                                   pvals_fdr, reject,
                                                   n_bins_list, t_ctrs, cond_labels, row_groups,
                                                   edge_trim, stats_dir):
    """One figure per sig cluster: grid of subplots (rows = passive / active-start
    / active-lick), each cell one trial-type condition, overlaying R+ vs R-
    mean ± SEM. Edge bins biased by the smoothing/window artifact are trimmed
    from display. Y-axis shared across all subplots in the figure; x shared
    within each row (rows have different alignment windows). Every axis is
    individually labeled.
    """
    out_dir = stats_dir / "sig_clusters_by_trialtype_by_rewardgroup"
    out_dir.mkdir(parents=True, exist_ok=True)
    starts = np.cumsum([0] + list(n_bins_list[:-1]))
    ends   = np.cumsum(n_bins_list)
    slices = list(zip(starts, ends))

    group_colors = {"R+": rplus_color, "R-": rminus_color}
    n_rows = len(row_groups)
    n_cols = max(len(r) for r in row_groups)

    for k in _select_clusters_to_plot(pvals_fdr, reject):
        neurons_k = cluster_labels == k
        if neurons_k.sum() == 0:
            continue

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows),
                                  squeeze=False)

        # ── first pass: compute traces for both groups + a shared y-range ──
        traces = {}          # (row_idx, col_idx) -> ci -> {group: (t_tr, mean, sem)}
        n_per_group, mice_per_group = {}, {}
        y_min, y_max = np.inf, -np.inf

        for group, color in group_colors.items():
            grp_mask = neurons_k & (reward_arr == group)
            n_grp = int(grp_mask.sum())
            n_per_group[group]    = n_grp
            mice_per_group[group] = len(np.unique(mouse_arr[grp_mask])) if n_grp > 0 else 0
            if n_grp == 0:
                continue
            X_grp = X[grp_mask]

            for row_idx, idxs in enumerate(row_groups):
                for col_idx, ci in enumerate(idxs):
                    s, e = slices[ci]
                    t = t_ctrs[ci]
                    trim = min(edge_trim, (e - s) // 3)
                    seg  = X_grp[:, s + trim : e - trim]
                    t_tr = t[trim: len(t) - trim] if trim else t
                    mean = seg.mean(axis=0)
                    sem  = seg.std(axis=0, ddof=1) / np.sqrt(n_grp) if n_grp > 1 else np.zeros_like(mean)

                    cell = traces.setdefault((row_idx, col_idx), {})
                    cell[group] = (t_tr, mean, sem, ci)
                    y_min = min(y_min, (mean - sem).min())
                    y_max = max(y_max, (mean + sem).max())

        if not np.isfinite(y_min) or not np.isfinite(y_max):
            plt.close(fig)
            continue
        pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
        y_min, y_max = y_min - pad, y_max + pad

        # ── second pass: draw ───────────────────────────────────────────────
        for row_idx, idxs in enumerate(row_groups):
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                if col_idx >= len(idxs):
                    ax.axis("off")
                    continue

                cell = traces.get((row_idx, col_idx), {})
                ci = idxs[col_idx]
                for group, color in group_colors.items():
                    if group not in cell:
                        continue
                    t_tr, mean, sem, _ = cell[group]
                    ax.plot(t_tr, mean, color=color, lw=1.5,
                             label=f"{group} (n={n_per_group[group]})")
                    ax.fill_between(t_tr, mean - sem, mean + sem, color=color, alpha=0.2)

                ax.axvline(0, ls="--", color="gray", lw=0.8)
                ax.set_ylim(y_min, y_max)
                ax.set_title(cond_labels[ci], fontsize=9)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("z-score (mean ± SEM)")
                ax.legend(fontsize=7, frameon=False)

        fig.suptitle(
            f"Cluster {k + 1}  "
            f"(R+={n_per_group.get('R+', 0)} neu / {mice_per_group.get('R+', 0)} mice, "
            f"R-={n_per_group.get('R-', 0)} neu / {mice_per_group.get('R-', 0)} mice, "
            f"p_fdr={pvals_fdr[k]:.3g})"
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"cluster_{k + 1:03d}_by_trialtype_by_rewardgroup.png", dpi=150)
        plt.close(fig)

    print(f"  Saved per-cluster trial-type-by-reward-group figures → {out_dir}")

# ---------------------
# Area-level analysis
# ---------------------

def _mouse_level_area_baseline(group_arr, mouse_arr, all_groups):
    """Dataset-wide 'sampling bias' baseline: each mouse's own area-group
    composition (all of that mouse's neurons, any cluster), averaged with
    equal weight across mice — not pooled at the neuron level, so a
    heavily-recorded mouse doesn't dominate the baseline."""
    mice = np.unique(mouse_arr)
    per_mouse = []
    for m in mice:
        mask = mouse_arr == m
        n_m = mask.sum()
        if n_m == 0:
            continue
        counts = pd.Series(group_arr[mask]).value_counts().reindex(all_groups, fill_value=0)
        per_mouse.append(counts.values / n_m)
    per_mouse = np.array(per_mouse)
    baseline = per_mouse.mean(axis=0)
    return baseline / baseline.sum(), mice


def _cluster_mouse_level_composition(cluster_labels, group_arr, mouse_arr, k, all_groups):
    """Per-mouse area-group proportions among the neurons of mouse m that
    fall in cluster k, averaged with equal weight across contributing mice."""
    neurons_k = cluster_labels == k
    mice_in_k = np.unique(mouse_arr[neurons_k])
    per_mouse = []
    for m in mice_in_k:
        mask = neurons_k & (mouse_arr == m)
        n_km = mask.sum()
        if n_km == 0:
            continue
        counts = pd.Series(group_arr[mask]).value_counts().reindex(all_groups, fill_value=0)
        per_mouse.append(counts.values / n_km)
    if not per_mouse:
        return None, 0
    per_mouse = np.array(per_mouse)
    dist = per_mouse.mean(axis=0)
    return dist / dist.sum(), len(per_mouse)


def _permutation_area_divergence_test(cluster_labels, group_arr, mouse_arr, sig_clusters,
                                       all_groups, n_perm=500, rng=None):
    """Mouse-level permutation test: does a cluster's area-group composition
    (mouse-averaged) diverge from the dataset-wide baseline (also
    mouse-averaged) more than expected by chance, given each mouse's own area
    sampling? Effect size = Jensen-Shannon distance (bounded [0, 1], base 2).

    Null model: for each mouse contributing n_km neurons to cluster k, redraw
    a random subsample of n_km neurons (without replacement) from that same
    mouse's full neuron pool. This preserves each mouse's own area
    composition exactly, and only asks whether *which* of its neurons happen
    to land in cluster k depends on area group — the same logic as the
    mouse-level PERMANOVA used for reward-group enrichment elsewhere in this
    pipeline, applied here to area composition instead of R+/R−.
    """
    from scipy.spatial.distance import jensenshannon
    if rng is None:
        rng = np.random.default_rng()

    baseline_dist, _ = _mouse_level_area_baseline(group_arr, mouse_arr, all_groups)
    mice_all = np.unique(mouse_arr)
    mouse_pool_idx = {m: np.where(mouse_arr == m)[0] for m in mice_all}

    js_obs, pvals_raw, n_mice_used = [], [], []

    for k in sig_clusters:
        obs_dist, n_mice_k = _cluster_mouse_level_composition(
            cluster_labels, group_arr, mouse_arr, k, all_groups)
        n_mice_used.append(n_mice_k)
        if obs_dist is None or n_mice_k == 0:
            js_obs.append(np.nan); pvals_raw.append(1.0)
            continue

        obs_js = jensenshannon(obs_dist, baseline_dist, base=2)
        js_obs.append(obs_js)

        neurons_k = cluster_labels == k
        contrib = {m: int((neurons_k & (mouse_arr == m)).sum())
                   for m in mice_all if (neurons_k & (mouse_arr == m)).sum() > 0}

        null_js = np.empty(n_perm)
        for p in range(n_perm):
            per_mouse = []
            for m, n_km in contrib.items():
                sub_idx = rng.choice(mouse_pool_idx[m], size=n_km, replace=False)
                counts = pd.Series(group_arr[sub_idx]).value_counts().reindex(all_groups, fill_value=0)
                per_mouse.append(counts.values / n_km)
            perm_dist = np.array(per_mouse).mean(axis=0)
            perm_dist = perm_dist / perm_dist.sum()
            null_js[p] = jensenshannon(perm_dist, baseline_dist, base=2)

        pvals_raw.append((1 + np.sum(null_js >= obs_js)) / (n_perm + 1))

    pvals_raw = np.array(pvals_raw)
    pvals_fdr, reject = _bh_correction(pvals_raw, alpha=0.05)

    return pd.DataFrame(dict(
        cluster       = [k + 1 for k in sig_clusters],
        js_divergence = js_obs,
        n_mice        = n_mice_used,
        p_raw         = pvals_raw,
        p_fdr         = pvals_fdr,
        significant   = reject,
    ))

def _resolve_area_group_map(acronyms):
    """Map each raw area_acronym_custom value to its custom group name."""
    groups = allen_utils.get_custom_area_groups()
    acronym_to_group = {}
    # Handle dict[group_name] -> list[acronym]
    if isinstance(next(iter(groups.values())), (list, tuple, set)):
        for group_name, acs in groups.items():
            for ac in acs:
                acronym_to_group[ac] = group_name
    else:
        # Handle dict[acronym] -> group_name directly
        acronym_to_group = dict(groups)
    ac_dict = {ac: acronym_to_group.get(ac, "unassigned") for ac in np.unique(acronyms)}
    unassigned = [ac for ac, grp in ac_dict.items() if grp == "unassigned"]
    if unassigned:
        print(f"  Warning: {len(unassigned)} area_acronym_custom values not found in custom groups, "
          f"mapped to 'unassigned': {unassigned}.")
    return ac_dict

def _wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return max((center - margin) / denom, 0.0), min((center + margin) / denom, 1.0)


def _compute_area_enrichment_stats(cluster_group_counts, global_prop, all_groups, sig_clusters):
    """Per-(cluster, area-group) log2 fold-change vs. dataset baseline, Wilson CI
    (also in log2 space), and BH-FDR-corrected binomial significance.
    Neuron-level (not mouse-level) — see caveat in accompanying notes.
    """
    from scipy.stats import binomtest
    eps = 1e-6
    n_groups, n_clusters = len(all_groups), len(sig_clusters)
    log2fc   = np.zeros((n_clusters, n_groups))
    ci_lo    = np.zeros((n_clusters, n_groups))
    ci_hi    = np.zeros((n_clusters, n_groups))
    pvals    = np.ones((n_clusters, n_groups))
    counts_m = np.zeros((n_clusters, n_groups), dtype=int)

    for i, k in enumerate(sig_clusters):
        counts = cluster_group_counts[k]
        n_k = int(counts.sum())
        for j, g in enumerate(all_groups):
            c = int(counts.get(g, 0))
            counts_m[i, j] = c
            p0 = global_prop[g]

            phat = c / n_k if n_k > 0 else 0.0
            wilson_lo, wilson_hi = _wilson_ci(c, n_k)

            # identical eps-shift + denominator applied to phat and both bounds,
            # so ci_lo <= log2fc <= ci_hi holds exactly (wilson_lo <= phat <= wilson_hi
            # is preserved through the same monotonic transform on all three).
            log2fc[i, j] = np.log2((phat + eps) / (p0 + eps))
            ci_lo[i, j]  = np.log2((wilson_lo + eps) / (p0 + eps))
            ci_hi[i, j]  = np.log2((wilson_hi + eps) / (p0 + eps))

            if n_k > 0:
                pvals[i, j] = binomtest(c, n_k, p0, alternative="two-sided").pvalue

    flat_fdr, flat_reject = _bh_correction(pvals.flatten(), alpha=0.05)
    return dict(log2fc=log2fc, ci_lo=ci_lo, ci_hi=ci_hi, pvals=pvals, counts=counts_m,
                pvals_fdr=flat_fdr.reshape(n_clusters, n_groups),
                reject=flat_reject.reshape(n_clusters, n_groups))


# ── Recommendation 1: enrichment dot-heatmap ────────────────────────────────
def _fig_area_enrichment_dotplot(stats, all_groups, sig_clusters, out_dir, js_df=None):
    n_groups, n_clusters = len(all_groups), len(sig_clusters)
    n_cols = n_groups + (1 if js_df is not None else 0)
    log2fc = stats["log2fc"]
    reject = stats["reject"]

    vmax = 4

    side = 0.55 * max(n_cols, n_clusters) + 2.6
    fig, ax = plt.subplots(figsize=(side, side))

    xs, ys = np.meshgrid(np.arange(n_groups) + 0.5, np.arange(n_clusters) + 0.5)
    xs, ys = xs.flatten(), ys.flatten()
    c_flat   = log2fc.flatten()
    sig_flat = reject.flatten()

    # sizing stays on the true (unrounded) magnitude
    abs_fc = np.abs(c_flat)
    max_abs = max(abs_fc.max(), 1e-6)
    sig_size_min, sig_size_max = 90, 500
    sig_sizes = sig_size_min + (sig_size_max - sig_size_min) * (abs_fc / max_abs)
    nonsig_color = (0.80, 0.80, 0.80)

    cmap = plt.get_cmap("PiYG")
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    base_colors = cmap(norm(c_flat))[:, :3]

    m = ~sig_flat
    ax.scatter(xs[m], ys[m], facecolors="none", edgecolors=nonsig_color,
               linewidths=0.8, s=sig_sizes[m], marker="o", zorder=2)
    m = sig_flat
    ax.scatter(xs[m], ys[m], facecolors=base_colors[m], edgecolors="black",
               linewidths=0.0, s=sig_sizes[m], marker="o", zorder=3)

    # ── extra column: overall JS-divergence effect size per cluster ────────
    js_cmap, js_norm = None, None
    if js_df is not None:
        js_vals = js_df["js_divergence"].to_numpy()
        js_sig  = js_df["significant"].to_numpy()
        js_cmap = plt.get_cmap("Purples")
        js_vmax = max(np.nanmax(js_vals), 1e-3)
        js_norm = plt.Normalize(vmin=0, vmax=js_vmax)
        js_x = np.full(n_clusters, n_groups + 0.5)
        js_y = np.arange(n_clusters) + 0.5
        js_colors = js_cmap(js_norm(js_vals))[:, :3]
        js_sizes = sig_size_min + (sig_size_max - sig_size_min) * (js_vals / js_vmax)

        m = ~js_sig
        ax.scatter(js_x[m], js_y[m], facecolors="none", edgecolors=nonsig_color,
                   linewidths=0.8, s=js_sizes[m], marker="o", zorder=2)
        m = js_sig
        ax.scatter(js_x[m], js_y[m], facecolors=js_colors[m], edgecolors="black",
                   linewidths=0.3, s=js_sizes[m], marker="o", zorder=3)

        ax.axvline(n_groups, color="black", lw=0.8, ls="--", alpha=0.5)

    ax.set_xlim(0, n_cols); ax.set_ylim(0, n_clusters)
    ax.set_aspect("equal")
    xtick_labels = list(all_groups) + (["Overall\ndivergence"] if js_df is not None else [])
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(np.arange(n_clusters) + 0.5)
    ax.set_yticklabels([f"Cluster {k+1}" for k in sig_clusters], fontsize=11)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # ── colorbars: fixed inset axes, stacked, hugging the right edge ───────
    cax1 = ax.inset_axes([1.04, 0.56, 0.025, 0.36], transform=ax.transAxes)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax1)
    cbar.set_label("log2 Fold-Change vs. dataset\n(significant cells only)", fontsize=10)
    cbar.ax.tick_params(labelsize=9, length=2)
    cbar.outline.set_visible(False)

    if js_df is not None:
        cax2 = ax.inset_axes([1.04, 0.06, 0.025, 0.36], transform=ax.transAxes)
        sm2 = plt.cm.ScalarMappable(cmap=js_cmap, norm=js_norm)
        cbar2 = fig.colorbar(sm2, cax=cax2)
        cbar2.set_label("JS divergence\n(mouse-level, sig. only)", fontsize=10)
        cbar2.ax.tick_params(labelsize=9, length=2)
        cbar2.outline.set_visible(False)

    legend_vals = sorted(set(v for v in
                             [round(max_abs), max(round(max_abs) / 2, 1), 1] if v >= 1))
    size_handles = [
        ax.scatter([], [], s=sig_size_min + (sig_size_max - sig_size_min) * (v / max_abs),
                   facecolors="grey", edgecolors="black", linewidths=0.3, label=f"{v:.0f}")
        for v in legend_vals
    ]
    sig_handle    = ax.scatter([], [], facecolors="grey", edgecolors="black",
                                linewidths=0.3, s=150, label="Significant")
    nonsig_handle = ax.scatter([], [], facecolors="none", edgecolors=nonsig_color,
                                linewidths=0.8, s=150, label="Not significant")
    all_handles = size_handles + [sig_handle, nonsig_handle]
    ax.legend(handles=all_handles,
              title="|log2 Fold-Change| (rounded, sig. only)          Significance",
              loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=len(all_handles), fontsize=10, title_fontsize=11,
              frameon=False, handletextpad=0.6, columnspacing=1.6,
              borderaxespad=1.2)

    ax.set_title("Area-group enrichment vs. dataset baseline\n"
                  "(rightmost column: overall composition divergence, mouse-level permutation test)",
                  fontsize=14, pad=14)
    fig.tight_layout(rect=[0, 0, 0.98, 1])
    fig.savefig(out_dir / "area_enrichment_dotplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Recommendation 2: forest plot, one subplot per cluster ─────────────────
def _fig_area_enrichment_forest(stats, all_groups, sig_clusters, group_color_map, out_dir):
    n_sig = len(sig_clusters)
    ncols = min(4, n_sig)
    nrows = int(np.ceil(n_sig / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(3.2 * ncols, 0.5 * len(all_groups) + 1.8 * nrows),
                              squeeze=False)
    y = np.arange(len(all_groups))
    colors = [group_color_map.get(g, "grey") for g in all_groups]

    for idx, k in enumerate(sig_clusters):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        fc, lo, hi, sig = stats["log2fc"][idx], stats["ci_lo"][idx], stats["ci_hi"][idx], stats["reject"][idx]
        xerr_lo = np.maximum(0, fc - lo)
        xerr_hi = np.maximum(0, hi - fc)
        ax.errorbar(fc, y, xerr=[xerr_lo, xerr_hi], fmt="none", ecolor="grey", capsize=2, zorder=2)
        ax.scatter(fc, y, c=colors, s=60, zorder=3, edgecolors="black", linewidths=0.5)
        for yi in np.where(sig)[0]:
            ax.text(fc[yi], yi, " *", va="center", fontsize=11, fontweight="bold")
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_yticks(y); ax.set_yticklabels(all_groups, fontsize=7)
        ax.set_title(f"Cluster {k+1}", fontsize=9)
        ax.set_xlabel("log2 FC vs. dataset")

    for idx in range(n_sig, nrows * ncols):
        axes[divmod(idx, ncols)].axis("off") if False else axes[idx // ncols, idx % ncols].axis("off")

    fig.suptitle("Per-cluster area-group enrichment (Wilson 95% CI vs. dataset baseline)")
    fig.tight_layout()
    fig.savefig(out_dir / "area_enrichment_forest.png", dpi=150)
    plt.close(fig)


# ── Recommendation 3: stacked composition (panel A) + enrichment heatmap strip (panel B) ──
def _fig_area_stacked_with_heatmap(cluster_group_counts, global_prop, all_groups, sig_clusters,
                                    group_color_map, stats, out_dir):
    n_sig = len(sig_clusters)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(max(7, 1.1 * (n_sig + 1)), 8),
        gridspec_kw={"height_ratios": [3, 3]})

    x = np.arange(n_sig + 1)
    bottom = np.zeros(n_sig + 1)
    for group in all_groups:
        cluster_fracs = np.array([
            cluster_group_counts[k].get(group, 0) / max(1, cluster_group_counts[k].sum())
            for k in sig_clusters])
        fracs = np.concatenate([[global_prop[group]], cluster_fracs])
        color = group_color_map.get(group, "lightgrey")
        ax_top.bar(x, fracs, bottom=bottom, color=color, edgecolor="white", linewidth=0.5, label=group)
        for xi, (frac, base) in enumerate(zip(fracs, bottom)):
            if frac >= 0.04:
                ax_top.text(xi, base + frac / 2, f"{frac*100:.0f}%", ha="center", va="center",
                            fontsize=7, color="white")
        bottom += fracs

    ax_top.axvline(0.5, color="black", lw=1, ls="--", alpha=0.6)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(["Dataset"] + [f"Cluster {k+1}" for k in sig_clusters], rotation=45, ha="right")
    ax_top.set_ylabel("Fraction"); ax_top.set_ylim(0, 1.05)
    ax_top.set_title("Area-group composition")
    ax_top.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, frameon=False)
    ax_top.set_xlim(-0.5, n_sig + 0.5)

    hm = stats["log2fc"].T  # groups × clusters
    vmax = max(np.nanmax(np.abs(hm)), 1e-3)
    im = ax_bot.imshow(hm, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for gi in range(hm.shape[0]):
        for ci in range(hm.shape[1]):
            if stats["reject"][ci, gi]:
                ax_bot.text(ci, gi, "*", ha="center", va="center", fontsize=9, fontweight="bold")
    ax_bot.set_yticks(np.arange(len(all_groups))); ax_bot.set_yticklabels(all_groups)
    ax_bot.set_xticks(np.arange(n_sig))
    ax_bot.set_xticklabels([f"Cluster {k+1}" for k in sig_clusters], rotation=45, ha="right")
    ax_bot.set_xlim(-0.5, n_sig - 0.5)
    ax_bot.set_title("log2 fold-enrichment vs. dataset (* = significant, BH-FDR)")
    fig.colorbar(im, ax=ax_bot, orientation="horizontal", fraction=0.05, pad=0.35, label="log2FC")

    fig.tight_layout()
    fig.savefig(out_dir / "area_stacked_with_enrichment_heatmap.png", dpi=150)
    plt.close(fig)


# ── Recommendation 4: radar plot, one subplot per cluster ──────────────────
def _fig_area_radar(cluster_group_counts, global_prop, all_groups, sig_clusters, group_color_map, out_dir):
    n_groups = len(all_groups)
    angles = np.linspace(0, 2 * np.pi, n_groups, endpoint=False).tolist()
    angles += angles[:1]
    baseline_vals = [global_prop[g] for g in all_groups]; baseline_vals += baseline_vals[:1]

    ncols = min(4, len(sig_clusters))
    nrows = int(np.ceil(len(sig_clusters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows),
                              subplot_kw=dict(polar=True), squeeze=False)

    for idx, k in enumerate(sig_clusters):
        ax = axes[idx // ncols, idx % ncols]
        n_k = max(1, cluster_group_counts[k].sum())
        cluster_vals = [cluster_group_counts[k].get(g, 0) / n_k for g in all_groups]
        cluster_vals += cluster_vals[:1]

        ax.plot(angles, baseline_vals, color="grey", lw=1.5, ls="--", label="Dataset")
        ax.fill(angles, baseline_vals, color="grey", alpha=0.1)
        ax.plot(angles, cluster_vals, color="black", lw=1.5, label=f"Cluster {k+1}")
        for a, v, g in zip(angles[:-1], cluster_vals[:-1], all_groups):
            ax.scatter([a], [v], color=group_color_map.get(g, "grey"), s=50,
                       zorder=5, edgecolors="black", linewidths=0.5)

        ax.set_xticks(angles[:-1]); ax.set_xticklabels(all_groups, fontsize=7)
        ax.set_title(f"Cluster {k+1}", fontsize=9, pad=15)
        ax.set_ylim(0, max(max(baseline_vals), max(cluster_vals)) * 1.15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=6, frameon=False)

    for idx in range(len(sig_clusters), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    fig.suptitle("Area-group composition: cluster (solid) vs. dataset baseline (dashed)")
    fig.tight_layout()
    fig.savefig(out_dir / "area_radar_per_cluster.png", dpi=150)
    plt.close(fig)


# ── Recommendation 5: spatial CCF maps, one subplot per cluster ────────────
def _fig_area_spatial_ccf(cluster_labels, group_arr, sig_clusters, group_color_map,
                           ccf_ap, ccf_ml, out_dir):
    if ccf_ap is None or ccf_ml is None:
        print("  Skipping spatial CCF enrichment maps — ccf_ap/ccf_ml not provided.")
        return
    n_sig = len(sig_clusters)
    ncols = min(4, n_sig); nrows = int(np.ceil(n_sig / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows), squeeze=False)

    for idx, k in enumerate(sig_clusters):
        ax = axes[idx // ncols, idx % ncols]
        ax.scatter(ccf_ml, ccf_ap, s=3, color="lightgrey", alpha=0.3)
        mask = cluster_labels == k
        colors = [group_color_map.get(g, "grey") for g in group_arr[mask]]
        ax.scatter(ccf_ml[mask], ccf_ap[mask], s=10, c=colors, edgecolors="black", linewidths=0.2)
        ax.set_title(f"Cluster {k+1} (n={int(mask.sum())})", fontsize=9)
        ax.set_xlabel("ML (CCF)"); ax.set_ylabel("AP (CCF)")
        ax.invert_yaxis()

    for idx in range(n_sig, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    fig.suptitle("Spatial distribution of cluster neurons (colored by area group) vs. all neurons (grey)")
    fig.tight_layout()
    fig.savefig(out_dir / "area_spatial_ccf_maps.png", dpi=150)
    plt.close(fig)

def _fig_cluster_area_composition(cluster_labels, area_arr, mouse_arr, pvals_fdr, reject, stats_dir,
                                   ccf_ap=None, ccf_ml=None, mode='significant'):
    """All sig clusters on shared figures (not one file per cluster):
    (1) a grid of paired bar charts (area group / area acronym) per cluster,
        colored by allen_utils custom area group colors;
    (2) a stacked bar chart of area-group proportion per cluster, annotated
        per-segment, with significance from the mouse-level permutation test
        (observed area-group composition vs. dataset-wide "sampling bias"
        proportions), BH-FDR corrected across clusters.
        [Chi-square goodness-of-fit is still computed for CSV/figure-1 output,
        but is no longer used for the stacked-figure annotation — see below.]
    """
    out_dir = stats_dir / "sig_clusters_area_composition"
    out_dir.mkdir(parents=True, exist_ok=True)
    group_map        = _resolve_area_group_map(area_arr)
    group_arr        = np.array([group_map[a] for a in area_arr])
    group_color_map  = allen_utils.get_custom_area_groups_colors()          # {group: color}
    canonical_groups = list(allen_utils.get_custom_area_groups().keys())   # stacking order
    default_color    = "lightgrey"
    sig_clusters = _select_clusters_to_plot(pvals_fdr, reject, mode=mode)
    n_sig = len(sig_clusters)
    if n_sig == 0:
        print("  No clusters to plot for area composition.")
        return
    # ── dataset-wide baseline proportions (the "sampling bias" null) ───────
    global_group_counts = pd.Series(group_arr).value_counts()
    all_groups = sorted(set(canonical_groups) | set(global_group_counts.index),
                         key=lambda g: canonical_groups.index(g) if g in canonical_groups else 999)
    global_prop = (global_group_counts.reindex(all_groups, fill_value=0) /
                   global_group_counts.sum())
    #
    js_df = _permutation_area_divergence_test(
        cluster_labels, group_arr, mouse_arr, sig_clusters, all_groups, n_perm=500)
    js_df.to_csv(out_dir / "area_composition_permutation_stats.csv", index=False)
    n_sig_js = int(js_df["significant"].sum())
    print(f"  Mouse-level permutation test: {n_sig_js}/{len(sig_clusters)} clusters "
          f"deviate from sampling baseline after BH-FDR (α=0.05)")
    # lookup keyed by cluster id, for the stacked-figure annotation below
    js_sig_by_cluster = dict(zip(js_df["cluster"], js_df["significant"]))
    # ── pass 1: per-cluster counts, proportions, and chi-square test ───────
    rows = []
    cluster_group_counts = {}   # k -> Series indexed by all_groups
    chi2_stats, pvals_raw = [], []
    for k in sig_clusters:
        neurons_k = cluster_labels == k
        n_k = int(neurons_k.sum())
        if n_k == 0:
            cluster_group_counts[k] = pd.Series(0, index=all_groups)
            chi2_stats.append(np.nan); pvals_raw.append(1.0)
            continue
        fine_counts  = pd.Series(area_arr[neurons_k]).value_counts()
        group_counts = pd.Series(group_arr[neurons_k]).value_counts().reindex(all_groups, fill_value=0)
        cluster_group_counts[k] = group_counts
        expected = n_k * global_prop.values
        low_exp = (expected < 5).sum()
        if low_exp:
            print(f"  Cluster {k+1}: {low_exp}/{len(expected)} area groups have "
                  f"expected count < 5 — chi-square p-value may be unreliable")
        # --- Chi-square goodness-of-fit test (kept for CSV / figure-1 output; ---
        # --- no longer drives significance annotation on the stacked figure) ---
        chi2, p = chisquare(f_obs=group_counts.values, f_exp=expected)
        chi2_stats.append(chi2); pvals_raw.append(p)
        for area_val, cnt in fine_counts.items():
            n_mice_area = len(np.unique(mouse_arr[neurons_k & (area_arr == area_val)]))
            rows.append(dict(cluster=k+1, area_acronym_custom=area_val,
                              area_group=group_map[area_val],
                              n_neurons=int(cnt), frac_of_cluster=cnt / n_k,
                              n_mice=n_mice_area))
    pvals_raw = np.array(pvals_raw)
    area_pvals_fdr, area_reject = _bh_correction(pvals_raw, alpha=0.05)
    stats_df = pd.DataFrame(dict(
        cluster      = [k + 1 for k in sig_clusters],
        n_neurons    = [int((cluster_labels == k).sum()) for k in sig_clusters],
        chi2         = chi2_stats,
        p_raw        = pvals_raw,
        p_fdr        = area_pvals_fdr,
        significant  = area_reject,
    ))
    stats_df.to_csv(out_dir / "area_composition_stats.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "area_composition_table.csv", index=False)
    n_sig_area = int(area_reject.sum())
    print(f"  Area composition chi-square: {n_sig_area}/{n_sig} clusters deviate "
          f"from sampling bias after BH-FDR (α=0.05)")
    # ── figure 1: grid of paired bar charts, all sig clusters, one figure ──
    fig1, axes1 = plt.subplots(n_sig, 2, figsize=(8, 4 * n_sig), squeeze=False)
    for row, k in enumerate(sig_clusters):
        neurons_k = cluster_labels == k
        n_k = int(neurons_k.sum())
        group_counts = cluster_group_counts[k].sort_values(ascending=False)
        group_counts = group_counts[group_counts > 0]
        colors_g = [group_color_map.get(g, default_color) for g in group_counts.index]
        ax = axes1[row, 0]
        bars = ax.bar(group_counts.index.astype(str), group_counts.values, color=colors_g)
        ax.bar_label(bars, fontsize=7)
        ax.set_title(f"Cluster {k + 1}: area groups (n={n_k})", fontsize=9)
        ax.set_ylabel("N neurons")
        ax.tick_params(axis="x", labelbottom=False, bottom=False)
        ax.set_box_aspect(1)
        fine_counts = (pd.Series(area_arr[neurons_k]).value_counts()
                       .sort_values(ascending=False)) if n_k else pd.Series(dtype=int)
        colors_f = [group_color_map.get(group_map.get(a, "Other"), default_color)
                    for a in fine_counts.index]
        ax = axes1[row, 1]
        bars = ax.bar(fine_counts.index.astype(str), fine_counts.values, color=colors_f)
        ax.bar_label(bars, fontsize=7)
        ax.set_title(f"Cluster {k + 1}: area_acronym_custom (n={n_k})", fontsize=9)
        ax.set_ylabel("N neurons")
        ax.tick_params(axis="x", rotation=45)
        ax.set_box_aspect(1)
    fig1.tight_layout()
    fig1.savefig(out_dir / "all_sig_clusters_area_composition.png", dpi=150)
    plt.close(fig1)
    # ── figure 2: stacked proportion bar chart, all sig clusters ───────────
    fig2, ax2 = plt.subplots(figsize=(max(6, 1.1 * n_sig), 6))
    x = np.arange(n_sig)
    bottom = np.zeros(n_sig)
    for group in all_groups:
        fracs = np.array([
            (cluster_group_counts[k].get(group, 0) / max(1, cluster_group_counts[k].sum()))
            for k in sig_clusters
        ])
        color = group_color_map.get(group, default_color)
        bar_container = ax2.bar(x, fracs, bottom=bottom, color=color, label=group,
                                 edgecolor="white", linewidth=0.5)
        for xi, (frac, base) in enumerate(zip(fracs, bottom)):
            if frac >= 0.04:   # only annotate segments large enough to hold text
                ax2.text(xi, base + frac / 2, f"{frac*100:.0f}%",
                          ha="center", va="center", fontsize=7, color="white")
        bottom += fracs
    for xi, k in enumerate(sig_clusters):
        # was: star = "*" if area_reject[xi] else "ns"   (chi-square)
        star = "*" if js_sig_by_cluster.get(k + 1, False) else "ns"
        ax2.text(xi, 1.02, star, ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0, 1.12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Cluster {k+1}" for k in sig_clusters], rotation=45, ha="right")
    ax2.set_ylabel("Fraction of cluster")
    ax2.set_title("Area-group composition per cluster\n"
                   "(* = significantly deviates from dataset-wide sampling proportions,\n"
                   "mouse-level permutation test, BH-FDR α=0.05)")
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)
    fig2.tight_layout()
    fig2.savefig(out_dir / "all_sig_clusters_area_proportion_stacked.png", dpi=150)
    plt.close(fig2)

    # ── recommendations 1–5 ─────────────────────────────────────────────────
    enrich_stats = _compute_area_enrichment_stats(cluster_group_counts, global_prop, all_groups, sig_clusters)
    _fig_area_enrichment_dotplot(enrich_stats, all_groups, sig_clusters, out_dir, js_df=js_df)
    _fig_area_enrichment_forest(enrich_stats, all_groups, sig_clusters, group_color_map, out_dir)
    _fig_area_stacked_with_heatmap(cluster_group_counts, global_prop, all_groups, sig_clusters,
                                   group_color_map, enrich_stats, out_dir)
    _fig_area_radar(cluster_group_counts, global_prop, all_groups, sig_clusters, group_color_map, out_dir)
    _fig_area_spatial_ccf(cluster_labels, group_arr, sig_clusters, group_color_map, ccf_ap, ccf_ml, out_dir)

    print(f"  Saved enrichment dotplot, forest, stacked+heatmap, radar "
          f"{'and spatial CCF ' if ccf_ap is not None else ''}figures → {out_dir}")


def _load_bregma_coords(unit_table, unit_ids):
    """Per-neuron bregma-relative AP/ML/DV, for statistics and the matplotlib
    projection figures."""
    ut = unit_table.set_index("unit_id") if unit_table.index.name != "unit_id" else unit_table
    sub = ut.reindex(unit_ids)
    coords = np.column_stack([
        pd.to_numeric(sub["ap"], errors="coerce").to_numpy(),
        pd.to_numeric(sub["ml"], errors="coerce").to_numpy(),
        pd.to_numeric(sub["dv"], errors="coerce").to_numpy(),
    ])
    n_bad = np.isnan(coords).any(axis=1).sum()
    if n_bad:
        print(f"  Warning: {n_bad}/{len(unit_ids)} neurons missing bregma AP/ML/DV")
    return coords  # (n_neurons, 3), columns = AP, ML, DV, bregma-relative


def _load_ccf_atlas_coords(unit_table, unit_ids):
    """Per-neuron true CCF atlas coordinates, for brainrender only (brainrender
    needs actual atlas-space microns, not bregma-relative offsets)."""
    ut = unit_table.set_index("unit_id") if unit_table.index.name != "unit_id" else unit_table
    sub = ut.reindex(unit_ids)
    coords = np.column_stack([
        pd.to_numeric(sub["ccf_atlas_ap"], errors="coerce").to_numpy(),
        pd.to_numeric(sub["ccf_atlas_ml"], errors="coerce").to_numpy(),
        pd.to_numeric(sub["ccf_atlas_dv"], errors="coerce").to_numpy(),
    ])
    n_bad = np.isnan(coords).any(axis=1).sum()
    if n_bad:
        print(f"  Warning: {n_bad}/{len(unit_ids)} neurons missing ccf_atlas_ap/ml/dv")
    return coords  # (n_neurons, 3), columns = AP, ML, DV, true CCF atlas space

def _mouse_level_centroid(coords, mouse_arr, mask):
    """Equal-weight-per-mouse centroid: average each mouse's own mean
    coordinate, then average across mice — avoids one heavily-recorded mouse
    dominating, same principle as _mouse_level_area_baseline."""
    mice = np.unique(mouse_arr[mask])
    per_mouse = []
    for m in mice:
        sub = coords[mask & (mouse_arr == m)]
        sub = sub[~np.isnan(sub).any(axis=1)]
        if len(sub) == 0:
            continue
        per_mouse.append(sub.mean(axis=0))
    if not per_mouse:
        return None, 0
    per_mouse = np.array(per_mouse)
    return per_mouse.mean(axis=0), len(per_mouse)

from scipy.stats import gaussian_kde


def _kde_contour_on_ax(ax, x, y, color, levels=4, alpha_lines=0.9, fill_alpha=0.08,
                        grid_n=120, min_points=8):
    """Fit a 2D Gaussian KDE to (x, y) and draw it as light filled contours +
    solid contour lines in `color`. Falls back to a plain scatter if there
    aren't enough points for a stable KDE fit (avoids gaussian_kde's singular-
    covariance errors on tiny/degenerate point sets)."""
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < min_points or np.std(x) == 0 or np.std(y) == 0:
        ax.scatter(x, y, s=8, color=color, alpha=0.5, zorder=2)
        return

    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except np.linalg.LinAlgError:
        ax.scatter(x, y, s=8, color=color, alpha=0.5, zorder=2)
        return

    pad_x = 0.1 * (x.max() - x.min() + 1e-9)
    pad_y = 0.1 * (y.max() - y.min() + 1e-9)
    xx, yy = np.meshgrid(
        np.linspace(x.min() - pad_x, x.max() + pad_x, grid_n),
        np.linspace(y.min() - pad_y, y.max() + pad_y, grid_n))
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("kde_cmap", ["white", color])
    ax.contourf(xx, yy, zz, levels=levels, cmap=cmap, alpha=fill_alpha, zorder=1)
    ax.contour(xx, yy, zz, levels=levels, colors=[color], linewidths=1.0,
               alpha=alpha_lines, zorder=2)


def _fig_location_by_rewardgroup(cluster_labels, coords, mouse_arr, reward_arr,
                                  sig_clusters, rg_loc_df, out_dir):
    """One row per cluster, one column per projection (AP-ML, AP-DV, ML-DV).
    R+ vs R- shown as overlaid KDE contours (not raw scatter) so density
    differences between cohorts are visible rather than obscured by
    overplotting."""
    n_sig = len(sig_clusters)
    fig, axes = plt.subplots(n_sig, 3, figsize=(9, 2.8 * n_sig), squeeze=False)

    for row, k in enumerate(sig_clusters):
        neurons_k = cluster_labels == k
        rp_mask = neurons_k & (reward_arr == "R+")
        rm_mask = neurons_k & (reward_arr == "R-")
        rg_row = rg_loc_df.iloc[row]
        star = "*" if rg_row["significant"] else "ns"

        for col, (xlabel, ylabel, xi, yi) in enumerate(_PROJECTIONS):
            ax = axes[row, col]
            _kde_contour_on_ax(ax, coords[rp_mask, xi], coords[rp_mask, yi], rplus_color)
            _kde_contour_on_ax(ax, coords[rm_mask, xi], coords[rm_mask, yi], rminus_color)

            ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            if yi == 0:
                ax.invert_yaxis()
            if col == 0:
                ax.set_title(f"Cluster {k+1}: {rg_row['within_cluster_distance']:.0f} vs "
                              f"{rg_row['global_dataset_distance']:.0f} μm baseline ({star})",
                              fontsize=8, loc="left")
            if row == 0 and col == 0:
                from matplotlib.lines import Line2D
                handles = [Line2D([0], [0], color=rplus_color, lw=1.5, label="R+"),
                           Line2D([0], [0], color=rminus_color, lw=1.5, label="R-")]
                ax.legend(handles=handles, fontsize=7, frameon=False)

    fig.suptitle("Within-cluster R+ vs. R− centroid location, all three projections\n"
                  "(KDE contours; subplot title: within-cluster centroid distance vs. "
                  "dataset-wide R+/R- baseline gap)")
    fig.tight_layout()
    fig.savefig(out_dir / "location_by_rewardgroup_3proj.png", dpi=150)
    plt.close(fig)

def _permutation_location_vs_dataset(cluster_labels, coords, mouse_arr, sig_clusters, n_perm=500, rng=None):
    """Mouse-level permutation test: does a cluster's centroid (mouse-averaged
    AP/ML/DV) deviate from the dataset-wide baseline centroid more than
    expected, given each mouse's own spatial sampling? Effect size = Euclidean
    centroid distance in CCF units. Null: for each mouse contributing n_km
    neurons, resample n_km neurons from that mouse's full pool (any cluster).
    """
    if rng is None:
        rng = np.random.default_rng()
    all_mask = np.ones(len(mouse_arr), dtype=bool)
    baseline_centroid, _ = _mouse_level_centroid(coords, mouse_arr, all_mask)
    mice_all = np.unique(mouse_arr)
    mouse_pool_idx = {m: np.where((mouse_arr == m) & ~np.isnan(coords).any(axis=1))[0] for m in mice_all}

    dist_obs, pvals_raw, n_mice_used = [], [], []
    for k in sig_clusters:
        neurons_k = cluster_labels == k
        obs_centroid, n_mice_k = _mouse_level_centroid(coords, mouse_arr, neurons_k)
        n_mice_used.append(n_mice_k)
        if obs_centroid is None:
            dist_obs.append(np.nan); pvals_raw.append(1.0)
            continue
        obs_dist = np.linalg.norm(obs_centroid - baseline_centroid)
        dist_obs.append(obs_dist)

        contrib = {m: int((neurons_k & (mouse_arr == m)).sum())
                   for m in mice_all if (neurons_k & (mouse_arr == m)).sum() > 0}

        null_dist = np.empty(n_perm)
        for p in range(n_perm):
            per_mouse = []
            for m, n_km in contrib.items():
                pool = mouse_pool_idx[m]
                if len(pool) < n_km:
                    continue
                sub_idx = rng.choice(pool, size=n_km, replace=False)
                per_mouse.append(coords[sub_idx].mean(axis=0))
            if not per_mouse:
                null_dist[p] = np.nan
                continue
            perm_centroid = np.array(per_mouse).mean(axis=0)
            null_dist[p] = np.linalg.norm(perm_centroid - baseline_centroid)

        valid = ~np.isnan(null_dist)
        pvals_raw.append((1 + np.sum(null_dist[valid] >= obs_dist)) / (valid.sum() + 1))

    pvals_raw = np.array(pvals_raw)
    pvals_fdr, reject = _bh_correction(pvals_raw, alpha=0.05)
    return pd.DataFrame(dict(
        cluster=[k + 1 for k in sig_clusters], centroid_distance=dist_obs,
        n_mice=n_mice_used, p_raw=pvals_raw, p_fdr=pvals_fdr, significant=reject,
    )), baseline_centroid



def _permutation_location_by_rewardgroup(cluster_labels, coords, mouse_arr, reward_arr,
                                          sig_clusters, n_perm=500, rng=None):
    """For each cluster already found significant in the R+/R- enrichment
    test: do R+ and R- mice's neurons within THIS cluster differ in centroid
    location more than expected by chance, given which mice contribute?
    Also reports the dataset-wide global R+/R- centroid distance for context
    — if the within-cluster distance isn't bigger than the global baseline
    gap, location isn't a special confound for this cluster specifically.

    Null: shuffle reward-group labels among the mice CONTRIBUTING TO THIS
    CLUSTER (their neuron counts/locations held fixed) — tests whether actual
    R+/R- assignment separates centroids more than random relabeling of the
    same contributing mice.
    """
    if rng is None:
        rng = np.random.default_rng()

    # global (dataset-wide) R+/R- centroid distance, for context
    mouse_reward = {m: reward_arr[mouse_arr == m][0] for m in np.unique(mouse_arr)}
    global_rp_mask = np.array([mouse_reward[m] == "R+" for m in mouse_arr])
    global_rm_mask = np.array([mouse_reward[m] == "R-" for m in mouse_arr])
    c_rp, _ = _mouse_level_centroid(coords, mouse_arr, global_rp_mask)
    c_rm, _ = _mouse_level_centroid(coords, mouse_arr, global_rm_mask)
    global_dist = np.linalg.norm(c_rp - c_rm) if (c_rp is not None and c_rm is not None) else np.nan

    dist_obs, pvals_raw, n_mice_rp, n_mice_rm = [], [], [], []
    for k in sig_clusters:
        neurons_k = cluster_labels == k
        mice_k = np.unique(mouse_arr[neurons_k])
        group_of = {m: mouse_reward[m] for m in mice_k}

        rp_mice = [m for m in mice_k if group_of[m] == "R+"]
        rm_mice = [m for m in mice_k if group_of[m] == "R-"]
        n_mice_rp.append(len(rp_mice)); n_mice_rm.append(len(rm_mice))

        if len(rp_mice) < 2 or len(rm_mice) < 2:
            dist_obs.append(np.nan); pvals_raw.append(1.0)
            continue

        c1, _ = _mouse_level_centroid(coords, mouse_arr, neurons_k & np.isin(mouse_arr, rp_mice))
        c2, _ = _mouse_level_centroid(coords, mouse_arr, neurons_k & np.isin(mouse_arr, rm_mice))
        obs_dist = np.linalg.norm(c1 - c2)
        dist_obs.append(obs_dist)

        # per-mouse coordinate pool WITHIN this cluster (locations fixed, only relabel group)
        mouse_coords_k = {m: coords[neurons_k & (mouse_arr == m)] for m in mice_k}
        null_dist = np.empty(n_perm)
        for p in range(n_perm):
            shuffled = rng.permutation(mice_k)
            perm_rp = shuffled[:len(rp_mice)]
            perm_rm = shuffled[len(rp_mice):len(rp_mice) + len(rm_mice)]
            pc1 = np.array([mouse_coords_k[m].mean(axis=0) for m in perm_rp
                             if len(mouse_coords_k[m])]).mean(axis=0)
            pc2 = np.array([mouse_coords_k[m].mean(axis=0) for m in perm_rm
                             if len(mouse_coords_k[m])]).mean(axis=0)
            null_dist[p] = np.linalg.norm(pc1 - pc2)

        pvals_raw.append((1 + np.sum(null_dist >= obs_dist)) / (n_perm + 1))

    pvals_raw = np.array(pvals_raw)
    pvals_fdr, reject = _bh_correction(pvals_raw, alpha=0.05)
    return pd.DataFrame(dict(
        cluster=[k + 1 for k in sig_clusters], within_cluster_distance=dist_obs,
        global_dataset_distance=global_dist, n_mice_rplus=n_mice_rp, n_mice_rminus=n_mice_rm,
        p_raw=pvals_raw, p_fdr=pvals_fdr, significant=reject,
    ))

_PROJECTIONS = [
    ("ML", "AP", 1, 0),   # (xlabel, ylabel, x_col_idx, y_col_idx) -- coords columns = [AP, ML, DV]
    ("DV", "AP", 2, 0),
    ("ML", "DV", 1, 2),
]
def _fig_location_vs_dataset_overview(cluster_labels, coords, mouse_arr, sig_clusters, loc_df, baseline_centroid, out_dir):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(coords[:, 1], coords[:, 0], s=1, color="lightgrey", alpha=0.15, zorder=1)
    ax.scatter(baseline_centroid[1], baseline_centroid[0], marker="+", s=200,
               color="black", linewidths=2, zorder=4, label="Dataset centroid")

    dmax = max(loc_df["centroid_distance"].max(), 1e-3)
    for i, k in enumerate(sig_clusters):
        row = loc_df.iloc[i]
        c, _ = _mouse_level_centroid(coords, mouse_arr, cluster_labels == k)
        if c is None:
            continue
        size = 60 + 300 * (row["centroid_distance"] / dmax)
        if row["significant"]:
            ax.scatter(c[1], c[0], s=size, facecolors="crimson", edgecolors="black",
                       linewidths=0.5, zorder=3)
            ax.annotate(f"{k+1}", (c[1], c[0]), fontsize=7, ha="center", va="center", zorder=5)
        else:
            ax.scatter(c[1], c[0], s=30, facecolors="none", edgecolors="grey",
                       linewidths=0.8, zorder=2)

    ax.set_xlabel("ML (CCF)"); ax.set_ylabel("AP (CCF)")
    ax.set_title("Cluster centroid location vs. dataset baseline\n"
                  "(filled = significant deviation, size ∝ centroid distance, mouse-level permutation)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "location_vs_dataset_overview.png", dpi=150)
    plt.close(fig)

def _fig_location_by_rewardgroup_old(cluster_labels, coords, mouse_arr, reward_arr,
                                  sig_clusters, rg_loc_df, out_dir):
    """One row per cluster, one column per projection (AP-ML, AP-DV, ML-DV)."""
    n_sig = len(sig_clusters)
    fig, axes = plt.subplots(n_sig, 3, figsize=(9, 2.8 * n_sig), squeeze=False)

    for row, k in enumerate(sig_clusters):
        neurons_k = cluster_labels == k
        rp_mask = neurons_k & (reward_arr == "R+")
        rm_mask = neurons_k & (reward_arr == "R-")
        rg_row = rg_loc_df.iloc[row]
        star = "*" if rg_row["significant"] else "ns"

        for col, (xlabel, ylabel, xi, yi) in enumerate(_PROJECTIONS):
            ax = axes[row, col]
            ax.scatter(coords[rp_mask, xi], coords[rp_mask, yi], s=8, color=rplus_color, alpha=0.6, label="R+")
            ax.scatter(coords[rm_mask, xi], coords[rm_mask, yi], s=8, color=rminus_color, alpha=0.6, label="R-")
            ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            if yi == 0:  # AP on y-axis in this projection -> anatomical "up" convention
                ax.invert_yaxis()
            if col == 0:
                ax.set_title(f"Cluster {k+1}: {rg_row['within_cluster_distance']:.0f} vs "
                              f"{rg_row['global_dataset_distance']:.0f} μm baseline ({star})",
                              fontsize=8, loc="left")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, frameon=False)

    fig.suptitle("Within-cluster R+ vs. R− centroid location, all three projections\n"
                  "(subplot title: within-cluster centroid distance vs. dataset-wide R+/R- baseline gap)")
    fig.tight_layout()
    fig.savefig(out_dir / "location_by_rewardgroup_3proj.png", dpi=150)
    plt.close(fig)


def _fig_location_vs_dataset_detail(cluster_labels, coords, mouse_arr, sig_clusters,
                                     loc_df, baseline_centroid, out_dir):
    """Per-cluster (Q2) detail: all neurons in grey, cluster neurons colored,
    dataset centroid (cross) vs. cluster centroid (dot), all three projections."""
    n_sig = len(sig_clusters)
    fig, axes = plt.subplots(n_sig, 3, figsize=(9, 2.8 * n_sig), squeeze=False)

    for row, k in enumerate(sig_clusters):
        neurons_k = cluster_labels == k
        centroid, _ = _mouse_level_centroid(coords, mouse_arr, neurons_k)
        loc_row = loc_df.iloc[row]
        star = "*" if loc_row["significant"] else "ns"

        for col, (xlabel, ylabel, xi, yi) in enumerate(_PROJECTIONS):
            ax = axes[row, col]
            ax.scatter(coords[:, xi], coords[:, yi], s=2, color="lightgrey", alpha=0.2, zorder=1)
            ax.scatter(coords[neurons_k, xi], coords[neurons_k, yi], s=8, color="steelblue", alpha=0.6, zorder=2)
            ax.scatter(baseline_centroid[xi], baseline_centroid[yi], marker="+", s=120,
                       color="black", linewidths=2, zorder=4)
            if centroid is not None:
                ax.scatter(centroid[xi], centroid[yi], marker="o", s=60,
                           color="crimson", edgecolors="black", linewidths=0.5, zorder=4)
            ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            if yi == 0:
                ax.invert_yaxis()
            if col == 0:
                ax.set_title(f"Cluster {k+1}: {loc_row['centroid_distance']:.0f} μm from "
                              f"dataset centroid ({star})", fontsize=8, loc="left")

    fig.suptitle("Cluster centroid location vs. dataset baseline, all three projections\n"
                  "(cross = dataset centroid, red dot = cluster centroid)")
    fig.tight_layout()
    fig.savefig(out_dir / "location_vs_dataset_3proj.png", dpi=150)
    plt.close(fig)

def _reorder_coords_for_brainrender(coords):
    """coords columns are [AP, ML, DV] (this pipeline's convention). brainrender
    / bg-atlasapi expects [AP, DV, ML] in microns for allen_mouse atlases.
    Sanity-check axis order once against a known landmark before trusting this
    (e.g. bregma AP should be near the atlas's AP midpoint, DV should increase
    ventrally, ML should be symmetric around the midline) — print a quick
    range summary to catch an accidental unit or axis-order mismatch early.
    """
    reordered = coords[:, [0, 2, 1]]  # AP, DV, ML
    print(f"  brainrender coords range — AP: [{reordered[:,0].min():.0f}, {reordered[:,0].max():.0f}]  "
          f"DV: [{reordered[:,1].min():.0f}, {reordered[:,1].max():.0f}]  "
          f"ML: [{reordered[:,2].min():.0f}, {reordered[:,2].max():.0f}]  (μm, verify against known landmarks)")
    return reordered

import vedo
vedo.settings.use_depth_peeling = True  # fixes VTK's default incorrect ordering
                                       # of overlapping semi-transparent actors
                                       # (root mesh + points) — without this,
                                       # points can render invisible behind a
                                       # low-alpha root mesh even though both
                                       # are logically in the scene


def _brainrender_multiview_screenshots(build_actor_fn, tmp_dir, tag):
    from brainrender import Scene
    paths = {}
    for label, camera in _BRAINRENDER_VIEWS:
        scene = Scene(atlas_name="allen_mouse_25um", title=None, inset=False)
        scene.add_brain_region("root", alpha=0.1, color="grey")  # bumped 0.06->0.1
        build_actor_fn(scene)
        scene.render(camera=camera, interactive=False, zoom=1.3)
        out_path = tmp_dir / f"{tag}_{camera}.png"
        scene.screenshot(name=str(out_path.with_suffix("")))
        scene.close()
        paths[label] = out_path if out_path.exists() else out_path.with_suffix(".png")
    return paths


def render_cluster_location_and_density_grid_old(cluster_labels, coords_ccf, sig_clusters, out_dir,
                                              cluster_selection="significant"):
    try:
        from brainrender.actors import Points, PointsDensity
    except ImportError:
        print("  brainrender not installed — skipping location/density grid renders.")
        return

    br_dir = out_dir / "brainrender"
    tmp_dir = br_dir / "_tmp_views"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    coords_br = _reorder_coords_for_brainrender(coords_ccf)

    clusters_to_render = (np.unique(cluster_labels) if cluster_selection == "all" else sig_clusters)
    print(f"  Rendering location+density grids for {len(clusters_to_render)} cluster(s)")

    for k in clusters_to_render:
        mask = (cluster_labels == k) & ~np.isnan(coords_br).any(axis=1)
        if mask.sum() < 10:
            print(f"  Cluster {k+1}: too few valid-coordinate neurons ({mask.sum()}), skipping")
            continue
        pts = coords_br[mask]

        loc_paths = _brainrender_multiview_screenshots(
            # radius bumped 25->40, alpha set to fully opaque (was 0.85) so
            # points aren't stacking transparency with the root mesh
            lambda scene, p=pts: scene.add(Points(p, radius=40, colors="crimson", alpha=1.0)),
            tmp_dir, tag=f"cl{k+1:03d}_loc")
        dens_paths = _brainrender_multiview_screenshots(
            lambda scene, p=pts: scene.add(PointsDensity(p, colors=_DENSITY_CMAP)),
            tmp_dir, tag=f"cl{k+1:03d}_dens")

        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        for col, (label, _) in enumerate(_BRAINRENDER_VIEWS):
            for row, paths, row_label in [(0, loc_paths, "Cell locations"), (1, dens_paths, "Density")]:
                ax = axes[row, col]
                img_path = paths[label]
                if img_path.exists():
                    ax.imshow(plt.imread(img_path))
                else:
                    ax.text(0.5, 0.5, "render failed", ha="center", va="center", fontsize=8)
                ax.axis("off")
                if row == 0:
                    ax.set_title(label, fontsize=10)
                if col == 0:
                    ax.text(-0.05, 0.5, row_label, transform=ax.transAxes, fontsize=10,
                             rotation=90, va="center", ha="right")

        fig.suptitle(f"Cluster {k+1} (n={int(mask.sum())} neurons)", fontsize=12)
        fig.tight_layout()
        fig.savefig(br_dir / f"cluster_{k+1:03d}_location_density_grid.png", dpi=180)
        plt.close(fig)

    for f in tmp_dir.glob("*"):
        f.unlink()
    tmp_dir.rmdir()
    print(f"  Saved location+density grids → {br_dir}")

def render_cluster_locations_brainrender_old(cluster_labels, coords, sig_clusters, out_dir,
                                          atlas_name="allen_mouse_25um", one_scene_per_cluster=True):
    """Static 3D brainrender screenshots of significant clusters' neuron
    locations against the Allen CCF brain outline. Runs off-screen (no
    interactive window needed) and saves PNG screenshots.

    Requires: pip install brainrender  (pulls in vedo + bg-atlasapi; the
    atlas itself downloads on first use, ~100s of MB).
    """
    try:
        from brainrender import Scene, settings
        from brainrender.actors import Points
    except ImportError:
        print("  brainrender not installed — skipping 3D renders. "
              "Install with: pip install brainrender")
        return

    settings.OFFSCREEN = True  # no display needed on a headless workstation
    br_dir = out_dir / "brainrender"
    br_dir.mkdir(parents=True, exist_ok=True)

    coords_br = _reorder_coords_for_brainrender(coords)
    cmap = plt.get_cmap("tab20")

    if one_scene_per_cluster:
        for i, k in enumerate(sig_clusters):
            scene = Scene(atlas_name=atlas_name, title=f"Cluster {k+1}")
            scene.add_brain_region("root", alpha=0.06, color="grey")
            mask = cluster_labels == k
            pts = Points(coords_br[mask], radius=25, colors="crimson", alpha=0.85)
            scene.add(pts)
            scene.render(interactive=False, zoom=1.3)
            scene.screenshot(name=str(br_dir / f"cluster_{k+1:03d}_3d"))
            scene.close()
    else:
        # all significant clusters in one scene, colored distinctly
        scene = Scene(atlas_name=atlas_name, title="All significant clusters")
        scene.add_brain_region("root", alpha=0.05, color="grey")
        for i, k in enumerate(sig_clusters):
            mask = cluster_labels == k
            color = cmap(i % 20)[:3]
            scene.add(Points(coords_br[mask], radius=20, colors=color, alpha=0.8, name=f"Cluster {k+1}"))
        scene.render(interactive=False, zoom=1.3)
        scene.screenshot(name=str(br_dir / "all_sig_clusters_3d"))
        scene.close()

    print(f"  Saved brainrender 3D screenshots → {br_dir}")

import brainrender
brainrender.settings.SHOW_AXES = False  # remove the default axes/ruler overlay, module-level, set once

_BRAINRENDER_VIEWS = [
    ("Angled", "three_quarters"),
    ("Top",    "top"),
    ("Frontal","frontal"),
    ("Sagittal","sagittal"),
]
_DENSITY_CMAP = "magma"   # distinct from PiYG/Purples/tab20 used elsewhere in this pipeline


def _brainrender_multiview_screenshots_old(build_actor_fn, tmp_dir, tag):
    """Render one Scene per camera view (a fresh Scene per view avoids stale-
    camera issues some brainrender versions have when reusing a Scene across
    multiple scene.render(camera=...) calls) and return {view_label: path}."""
    from brainrender import Scene
    paths = {}
    for label, camera in _BRAINRENDER_VIEWS:
        scene = Scene(atlas_name="allen_mouse_25um", title=None)
        scene.add_brain_region("root", alpha=0.06, color="grey")
        build_actor_fn(scene)
        scene.render(camera=camera, interactive=False, zoom=1.3)
        out_path = tmp_dir / f"{tag}_{camera}.png"
        scene.screenshot(name=str(out_path.with_suffix("")))
        scene.close()
        paths[label] = out_path if out_path.exists() else out_path.with_suffix(".png")
    return paths


def render_cluster_location_and_density_grid(cluster_labels, coords_ccf, sig_clusters, out_dir,
                                              cluster_selection="significant"):
    """For each cluster: a single combined figure, 2 rows x 4 columns —
    row 1 = actual cell locations (Points), row 2 = density (PointsDensity,
    unique colormap) — each column a different camera view (angled/top/
    frontal/sagittal). Off-screen, static screenshots stitched via matplotlib.
    """
    try:
        from brainrender.actors import Points, PointsDensity
    except ImportError:
        print("  brainrender not installed — skipping location/density grid renders.")
        return

    br_dir = out_dir / "brainrender"
    tmp_dir = br_dir / "_tmp_views"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    coords_br = _reorder_coords_for_brainrender(coords_ccf)  # AP,ML,DV -> AP,DV,ML

    if cluster_selection == "all":
        clusters_to_render = np.unique(cluster_labels)
    elif cluster_selection == "significant":
        clusters_to_render = sig_clusters
    else:
        raise ValueError('cluster_selection must be "significant" or "all"')
    print(f"  Rendering location+density grids for {len(clusters_to_render)} cluster(s)")

    for k in clusters_to_render:
        mask = (cluster_labels == k) & ~np.isnan(coords_br).any(axis=1)
        if mask.sum() < 10:
            print(f"  Cluster {k+1}: too few valid-coordinate neurons ({mask.sum()}), skipping")
            continue
        pts = coords_br[mask]

        loc_paths = _brainrender_multiview_screenshots(
            lambda scene, p=pts: scene.add(Points(p, radius=50, colors="crimson", alpha=0.85)),
            tmp_dir, tag=f"cl{k+1:03d}_loc")
        dens_paths = _brainrender_multiview_screenshots(
            lambda scene, p=pts: scene.add(PointsDensity(p, colors=_DENSITY_CMAP, radius=100)),
            tmp_dir, tag=f"cl{k+1:03d}_dens")

        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        for col, (label, _) in enumerate(_BRAINRENDER_VIEWS):
            for row, paths, row_label in [(0, loc_paths, "Cell locations"), (1, dens_paths, "Density")]:
                ax = axes[row, col]
                img_path = paths[label]
                if img_path.exists():
                    ax.imshow(plt.imread(img_path))
                else:
                    ax.text(0.5, 0.5, "render failed", ha="center", va="center", fontsize=8)
                ax.axis("off")
                if row == 0:
                    ax.set_title(label, fontsize=10)
                if col == 0:
                    ax.text(-0.05, 0.5, row_label, transform=ax.transAxes, fontsize=10,
                             rotation=90, va="center", ha="right")

        fig.suptitle(f"Cluster {k+1} (n={int(mask.sum())} neurons)", fontsize=12)
        fig.tight_layout()
        fig.savefig(br_dir / f"cluster_{k+1:03d}_location_density_grid.png", dpi=180)
        plt.close(fig)

    for f in tmp_dir.glob("*"):
        f.unlink()
    tmp_dir.rmdir()
    print(f"  Saved location+density grids → {br_dir}")

def render_cluster_density_brainrender(cluster_labels, coords_ccf, sig_clusters, out_dir,
                                        atlas_name="allen_mouse_25um",
                                        cluster_selection="significant"):
    """Volumetric density maps in true CCF atlas space, one per cluster plus
    a dataset-wide reference density. Off-screen, static screenshots.

    cluster_selection: "significant" -> only sig_clusters (as passed in);
                        "all"        -> every cluster present in cluster_labels.

    Requires: pip install brainrender  (pulls in vedo + bg-atlasapi; the
    atlas downloads on first use).
    """
    try:
        from brainrender import Scene, settings
        from brainrender.actors import PointsDensity
    except ImportError:
        print("  brainrender not installed — skipping density renders. "
              "Install with: pip install brainrender")
        return

    settings.OFFSCREEN = True
    br_dir = out_dir / "brainrender" / "density"
    br_dir.mkdir(parents=True, exist_ok=True)

    coords_br = _reorder_coords_for_brainrender(coords_ccf)  # AP,ML,DV -> AP,DV,ML

    if cluster_selection == "all":
        clusters_to_render = np.unique(cluster_labels)
        print(f"  Rendering density for ALL {len(clusters_to_render)} clusters")
    elif cluster_selection == "significant":
        clusters_to_render = sig_clusters
        print(f"  Rendering density for {len(clusters_to_render)} significant cluster(s)")
    else:
        raise ValueError('cluster_selection must be "significant" or "all"')

    # dataset-wide density (reference, always rendered)
    valid_all = ~np.isnan(coords_br).any(axis=1)
    scene = Scene(atlas_name=atlas_name, title="Dataset density")
    scene.add_brain_region("root", alpha=0.05, color="grey")
    scene.add(PointsDensity(coords_br[valid_all]))
    scene.render(interactive=False, zoom=1.3)
    scene.screenshot(name=str(br_dir / "dataset_density"))
    scene.close()

    for k in clusters_to_render:
        mask = (cluster_labels == k) & ~np.isnan(coords_br).any(axis=1)
        if mask.sum() < 10:
            print(f"  Cluster {k+1}: too few valid-coordinate neurons ({mask.sum()}) for a density map, skipping")
            continue
        scene = Scene(atlas_name=atlas_name, title=f"Cluster {k+1} density")
        scene.add_brain_region("root", alpha=0.05, color="grey")
        scene.add(PointsDensity(coords_br[mask]))
        scene.render(interactive=False, zoom=1.3)
        scene.screenshot(name=str(br_dir / f"cluster_{k+1:03d}_density"))
        scene.close()

    print(f"  Saved brainrender density maps → {br_dir}")

import tifffile

VOXEL_SIZE_UM = 10.0  # must match the resolution of REF_PATH's atlas

def load_reference(atlas_path) -> np.ndarray:
    """Load and normalize the atlas reference volume (axis order: AP, DV, ML),
    for the grayscale background of the three projection panels."""
    ref = tifffile.imread(atlas_path / "reference.tiff").astype(np.float32)
    ref = (ref - ref.min()) / (ref.max() - ref.min())
    return ref

def load_annotation_mask(atlas_path) -> np.ndarray:
    """Boolean mask, True = inside the brain, from the atlas annotation
    volume (nonzero structure ID = brain tissue, 0 = background)."""
    ann = tifffile.imread(atlas_path / "annotation.tiff")
    return ann != 0

def _filter_points_inside_brain(pts_px, brain_mask):
    """pts_px: (N, 3) neuron coords in voxel-index units, columns AP,DV,ML,
    same order/scale as brain_mask's axes. Returns points whose voxel index
    falls inside the mask (out-of-bounds also treated as outside)."""
    idx = np.round(pts_px).astype(int)
    inside = np.zeros(len(pts_px), dtype=bool)
    valid_bounds = (
        (idx[:, 0] >= 0) & (idx[:, 0] < brain_mask.shape[0]) &
        (idx[:, 1] >= 0) & (idx[:, 1] < brain_mask.shape[1]) &
        (idx[:, 2] >= 0) & (idx[:, 2] < brain_mask.shape[2])
    )
    inside[valid_bounds] = brain_mask[
        idx[valid_bounds, 0], idx[valid_bounds, 1], idx[valid_bounds, 2]]
    return inside

def _voxelize_cluster_occupancy(coords_atlas, mask, ref_shape, voxel_size_um):
    """Boolean occupancy volume (shape = ref_shape) marking every voxel that
    contains >=1 neuron of this cluster. coords_atlas columns must already be
    in the SAME axis order as ref (AP, DV, ML), true CCF atlas-space microns —
    NOT bregma-relative, and NOT the (AP, ML, DV) order used elsewhere in this
    pipeline for the matplotlib bregma-relative figures."""
    pts = coords_atlas[mask]
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) == 0:
        return np.zeros(ref_shape, dtype=bool)
    idx = np.round(pts / voxel_size_um).astype(int)
    for d in range(3):
        idx[:, d] = np.clip(idx[:, d], 0, ref_shape[d] - 1)
    occ = np.zeros(ref_shape, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ

import colorsys
from scipy.stats import gaussian_kde


def _generate_bright_colors(n, seed=None):
    """Randomly pick n bright, mutually separable colors — golden-ratio hue
    stepping from a random start hue guarantees even hue spacing (so colors
    stay visually distinguishable even for large n), with saturation/value
    randomized within a bright, saturated range so it still reads as
    'randomly picked' rather than a fixed palette."""
    rng = np.random.default_rng(seed)
    golden_ratio_conj = 0.618033988749895
    hue = rng.random()
    colors = []
    for _ in range(n):
        hue = (hue + golden_ratio_conj) % 1.0
        s = rng.uniform(0.65, 0.95)
        v = rng.uniform(0.85, 1.0)
        colors.append(colorsys.hsv_to_rgb(hue, s, v))
    return colors


def _translucent_gray_rgba(ref_slice, brain_alpha_max=0.85):
    """RGBA image: alpha scales with tissue intensity (outside-brain stays
    transparent), tissue itself rendered as a darker, more visible gray."""
    rgba = np.zeros((*ref_slice.shape, 4), dtype=np.float32)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = 1.0 - 0.75 * ref_slice  # darker than before (was 0.55)
    rgba[..., 3] = np.clip(ref_slice, 0, 1) * brain_alpha_max            # 0.85 max opacity (was 0.7)
    return rgba


def _kde_density_bands_on_ax(ax, x, y, color, grid_n=150, min_points=10):
    """Two shaded density bands (50-80% and 80-100% of peak KDE density) plus
    a single solid contour line at the 80% isoline — no cluster label."""
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < min_points or np.std(x) == 0 or np.std(y) == 0:
        return
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except np.linalg.LinAlgError:
        return
    pad_x = 0.15 * (x.max() - x.min() + 1e-9)
    pad_y = 0.15 * (y.max() - y.min() + 1e-9)
    xx, yy = np.meshgrid(np.linspace(x.min() - pad_x, x.max() + pad_x, grid_n),
                          np.linspace(y.min() - pad_y, y.max() + pad_y, grid_n))
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    zz = zz / zz.max()  # normalize to peak density = 1.0

    from matplotlib.colors import LinearSegmentedColormap
    band_cmap = LinearSegmentedColormap.from_list("band_cmap", ["white", color])
    ax.contourf(xx, yy, zz, levels=[0.5, 0.8, 1.0], cmap=band_cmap, alpha=0.35, zorder=2)
    ax.contour(xx, yy, zz, levels=[0.8], colors=[color], linewidths=1.4, alpha=0.95, zorder=3)


def _fig_cluster_atlas_projections(cluster_labels, coords_ccf, sig_clusters, out_dir, atlas_path,
                                    voxel_size_um=VOXEL_SIZE_UM, seed=None):
    ref = load_reference(atlas_path)
    brain_mask  = load_annotation_mask(atlas_path)
    coords_atlas = _reorder_coords_for_brainrender(coords_ccf)
    px = coords_atlas / voxel_size_um

    ref_coronal    = ref.max(axis=0)
    ref_sagittal   = ref.max(axis=2).T
    ref_transverse = ref.max(axis=1).T

    out_dir = Path(out_dir) / "atlas_proj"
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = _generate_bright_colors(len(sig_clusters), seed=seed)

    for i, k in enumerate(sig_clusters):
        mask = cluster_labels == k
        pts = px[mask]
        pts = pts[~np.isnan(pts).any(axis=1)]
        n_before = len(pts)
        inside = _filter_points_inside_brain(pts, brain_mask)   # brain_mask loaded once outside the loop
        pts = pts[inside]
        n_outside = n_before - len(pts)
        if n_outside:
            print(f"  Cluster {k+1}: excluded {n_outside}/{n_before} neurons outside brain mask")
        n_k = len(pts)
        if n_k == 0:
            print(f"  Cluster {k+1}: no valid coordinates, skipping")
            continue
        ap_p, dv_p, ml_p = pts[:, 0], pts[:, 1], pts[:, 2]
        clr = colors[i]

        fig, (ax_cor, ax_sag, ax_tra) = plt.subplots(1, 3, figsize=(13, 4.5))
        fig.patch.set_facecolor("#f7f7f7")
        for ax, img in [(ax_cor, ref_coronal), (ax_sag, ref_sagittal), (ax_tra, ref_transverse)]:
            ax.set_facecolor("#f7f7f7")
            ax.imshow(_translucent_gray_rgba(img), origin="upper", interpolation="nearest", zorder=1)
            ax.axis("off")
            # titles removed

        centroid_kw = dict(marker="+", s=320, linewidths=3.0, color=clr,
                            edgecolors="black", zorder=5)
        dot_kw = dict(s=14, color=clr, alpha=0.7, edgecolors="none", zorder=4)  # slightly smaller

        # Coronal: x=ML, y=DV
        ax_cor.scatter(ml_p, dv_p, **dot_kw)
        _kde_density_bands_on_ax(ax_cor, ml_p, dv_p, clr)
        ax_cor.scatter([ml_p.mean()], [dv_p.mean()], **centroid_kw)

        # Sagittal: x=AP, y=DV
        ax_sag.scatter(ap_p, dv_p, **dot_kw)
        _kde_density_bands_on_ax(ax_sag, ap_p, dv_p, clr)
        ax_sag.scatter([ap_p.mean()], [dv_p.mean()], **centroid_kw)

        # Transverse: x=AP, y=ML
        ax_tra.scatter(ap_p, ml_p, **dot_kw)
        _kde_density_bands_on_ax(ax_tra, ap_p, ml_p, clr)
        ax_tra.scatter([ap_p.mean()], [ml_p.mean()], **centroid_kw)

        fig.suptitle(f"Cluster {k+1} (n={n_k} neurons) — Allen reference atlas projections", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"cluster_{k+1:03d}_atlas_proj.png", dpi=200,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    print(f"  Saved {len(sig_clusters)} per-cluster atlas projection figures → {out_dir}")

def compute_cluster_centroid_deviation(cluster_labels, coords, mouse_arr, clusters_to_test,
                                        n_perm=2000, rng=None):
    """Mouse-level permutation test: does each cluster's centroid (mean
    AP/ML/DV, equal-weight-averaged across contributing mice) deviate from
    the dataset-wide baseline centroid more than expected by chance, given
    each mouse's own spatial sampling?

    coords: (n_neurons, 3) array, columns = AP, ML, DV (any consistent units
    — bregma-relative or CCF atlas space, just be consistent).
    clusters_to_test: list/array of cluster IDs (0-indexed, matching
    cluster_labels) to test — pass whichever selection you want (e.g.
    BH-FDR-significant reward-group clusters, or any custom subset).

    Returns a DataFrame with one row per cluster: centroid_distance (effect
    size, same units as coords), n_mice, p_raw, p_fdr, significant.
    """
    if rng is None:
        rng = np.random.default_rng()

    valid = ~np.isnan(coords).any(axis=1)
    baseline_centroid, n_mice_baseline = _mouse_level_centroid(coords, mouse_arr, valid)
    print(f"  Dataset baseline centroid (AP, ML, DV) = "
          f"({baseline_centroid[0]:.0f}, {baseline_centroid[1]:.0f}, {baseline_centroid[2]:.0f}), "
          f"averaged over {n_mice_baseline} mice")

    mice_all = np.unique(mouse_arr[valid])
    mouse_pool_idx = {m: np.where((mouse_arr == m) & valid)[0] for m in mice_all}

    rows = []
    for k in clusters_to_test:
        neurons_k = cluster_labels == k
        obs_centroid, n_mice_k = _mouse_level_centroid(coords, mouse_arr, neurons_k & valid)
        if obs_centroid is None:
            rows.append(dict(cluster=k + 1, centroid_distance=np.nan, n_mice=0,
                              p_raw=1.0))
            continue
        obs_dist = np.linalg.norm(obs_centroid - baseline_centroid)

        contrib = {m: int((neurons_k & (mouse_arr == m) & valid).sum())
                   for m in mice_all if (neurons_k & (mouse_arr == m) & valid).sum() > 0}

        null_dist = np.empty(n_perm)
        for p in range(n_perm):
            per_mouse = []
            for m, n_km in contrib.items():
                pool = mouse_pool_idx[m]
                if len(pool) < n_km:
                    continue
                sub_idx = rng.choice(pool, size=n_km, replace=False)
                per_mouse.append(coords[sub_idx].mean(axis=0))
            if not per_mouse:
                null_dist[p] = np.nan
                continue
            perm_centroid = np.array(per_mouse).mean(axis=0)
            null_dist[p] = np.linalg.norm(perm_centroid - baseline_centroid)

        valid_null = ~np.isnan(null_dist)
        p_raw = (1 + np.sum(null_dist[valid_null] >= obs_dist)) / (valid_null.sum() + 1)
        rows.append(dict(cluster=k + 1, centroid_distance=obs_dist, n_mice=n_mice_k, p_raw=p_raw))

    df = pd.DataFrame(rows)
    df["p_fdr"], df["significant"] = _bh_correction(df["p_raw"].to_numpy(), alpha=0.05)
    return df, baseline_centroid


def _fig_cluster_centroid_deviation(dev_df, out_dir, unit_label="μm"):
    """Bar/point plot: first xtick = dataset baseline (reference, distance
    0), subsequent xticks = each cluster's centroid distance from baseline.
    Significant clusters (BH-FDR) are annotated with a star."""
    n = len(dev_df)
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * (n + 1)), 4.5))

    x = np.arange(n + 1)
    values = np.concatenate([[0.0], dev_df["centroid_distance"].to_numpy()])
    colors = ["black"] + ["crimson" if sig else "lightgrey" for sig in dev_df["significant"]]

    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5, width=0.6)

    for xi, (dist, sig, p) in enumerate(zip(dev_df["centroid_distance"], dev_df["significant"],
                                             dev_df["p_fdr"]), start=1):
        if sig:
            ax.text(xi, dist, "*", ha="center", va="bottom", fontsize=14, fontweight="bold")

    labels = ["Dataset\n(reference)"] + [f"Cluster {c}" for c in dev_df["cluster"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(f"Centroid distance from dataset baseline ({unit_label})")
    ax.set_title("Cluster centroid deviation from dataset baseline\n"
                  "(mouse-level permutation test, * = significant BH-FDR α=0.05)")
    ax.axhline(0, color="black", lw=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_centroid_deviation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved centroid deviation figure → {out_dir / 'cluster_centroid_deviation.png'}")


# ── stats-only entry point ───────────────────H─────────────────────────────────

def run_stats_only(out_folder: str | Path,  cfg: dict, unit_table: pd.DataFrame) -> dict:
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
    cv_result_file = [f for f in os.listdir(out_folder) if f.endswith('results_cv.npz')][0]
    emb_path = out_folder / cv_result_file
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
    return run_reward_group_stats(out_folder, cfg, unit_table)

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