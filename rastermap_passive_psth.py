"""
rastermap_psth.py — Population PSTH matrix with rastermap ordering, UMAP, k-means.

Condition order: whisker_pre | whisker_post | auditory_pre | auditory_post
Z-score: per neuron, single std from baseline bins pooled across all 4 conditions.

Entry point
───────────
    run_rastermap_psth(units, trials, out_root, **cfg_overrides)

units  : DataFrame, unique index = unit_id. Required: spike_times, mouse_id, session_id, bc_label.
trials : All mice/sessions concatenated. Required: start_time, context, trial_type, mouse_id, session_id.
         context=='passive' split into passive_pre / passive_post per (mouse, session) internally.

Figures
───────
    fig0_data_summary.png      — counts at each filter stage
    fig1_trial_counts.png      — trial counts per condition
    fig2_fr_distribution.png   — FR histogram with threshold
    fig3_neuron_counts.png     — neurons per mouse pre/post filters
    fig4_sample_neurons.png    — random sample, 4-cond PSTHs overlaid
    fig5_population_matrix.png — input order | rastermap order (subplots)
    fig6_cluster_profiles.png  — mean PSTH per rastermap cluster
    fig7_pca_variance.png      — scree + cumulative variance
    fig8_umap.png              — unicolor | rastermap-cluster coloured
    fig9_kmeans.png            — k-means UMAP | elbow
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

import os
import socket
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from rastermap import Rastermap
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import fisher_exact
from matplotlib.colors import ListedColormap

import plotting_utils

# ── config ─────────────────────────────────────────────────────────────────────
# TODO: remove stride? or keep stride but fix artifact on edges (find again vrsion that did not have them)
# TODO: active PETHs too (change some parameters accordingly)
# TODO: odd trials train, apply embedding on test trials PETHs
# TODO: remove unused PCA?

DEFAULT_CFG: dict[str, Any] = dict(
    period                = 'passive', #"active_passive", "passive"', "active"
    t_pre                = 0.1,
    t_post               = 0.2,
    bin_ms               = 10,
    sigma_ms             = 2,
    artifact_win_s       = (-0.005, 0.005),
    whisker_trial_label  = "whisker_trial",
    fr_threshold_hz      = 0.1,
    square_fr            = True,
    reward_group_col     = "reward_group",
    area_col             = "area_acronym_custom",
    zscore_full           = True,
    align_col            = "start_time",
    context_col          = "context",
    trial_type_col       = "trial_type",
    mouse_id_col         = "mouse_id",
    session_id_col       = "session_id",
    n_rastermap_clusters = 20,
    k_means_k            = 8,
    k_elbow_range        = range(2, 12),
    umap_n_neighbors     = 50,
    umap_min_dist        = 0.2,
    vmax_pct             = 95,
    n_sample_neurons     = 24,
    n_jobs               = 60,
    stride_ms            = 2,
    modality             = "whisker_auditory",    # "both" | "whisker" | "auditory"
    reward_filter        = "combined",    # "both" | "R+" | "R-"
    n_example_neurons    = 30,
    example_alpha        = 0.5,       # weight: 0=centroid only, 1=LZ only
)

def get_conditions(cfg):
    mod = cfg.get("modality", "whisker_auditory")
    all_conds = [
        ("whisker_trial",  "passive_pre",  "Whisker pre",  "#32a852"),
        ("whisker_trial",  "passive_post", "Whisker post", "#085c1f"),
        ("auditory_trial", "passive_pre",  "Auditory pre", "#4158d9"),
        ("auditory_trial", "passive_post", "Auditory post","#0f2187"),
    ]
    if mod == "whisker":
        all_conds = [c for c in all_conds if c[0] == "whisker_trial"]
    elif mod == "auditory":
        all_conds = [c for c in all_conds if c[0] == "auditory_trial"]
    conds        = [(c[0], c[1]) for c in all_conds]
    cond_labels  = [c[2] for c in all_conds]
    cond_colors  = [c[3] for c in all_conds]
    cond_labels_matrix = [c[2].replace(" ", "\n") for c in all_conds]

    return conds, cond_labels, cond_colors, cond_labels_matrix

CONDITIONS = [
    ("whisker_trial",  "passive_pre"),
    ("whisker_trial",  "passive_post"),
    ("auditory_trial", "passive_pre"),
    ("auditory_trial", "passive_post"),
]
COND_LABELS        = ["Whisker pre", "Whisker post", "Auditory pre", "Auditory post"]
COND_LABELS_MATRIX = ["Whisker\npre-learning", "Whisker\npost-learning", "Auditory\npre-learning", "Auditory\npost-learning"]
COND_COLORS  = ["#32a852", "#085c1f", "#4158d9", "#0f2187"]

CONDITIONS, COND_LABELS, COND_COLORS, COND_LABELS_MATRIX = get_conditions(DEFAULT_CFG)


hostname = socket.gethostname()
if 'haas' in hostname:
    ROOT_PATH = '/mnt/share_internal'
else:
    ROOT_PATH = '/Petersen-Lab/share_internal'

MOUSE_INFO = os.path.join(ROOT_PATH,'Axel_Bisi_Share','dataset_info','joint_mouse_reference_weight.xlsx')

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


def get_spike_times(row: pd.Series) -> np.ndarray:
    return np.asarray(row["spike_times"])


def _replace_artifact(rel, t_pre, lo, hi, rng):
    pre  = rel[rel < lo]
    rate = len(pre) / (t_pre + lo) if (t_pre + lo) > 0 else 0.0
    keep = rel[(rel < lo) | (rel > hi)]
    n    = rng.poisson(rate * (hi - lo))
    return np.sort(np.concatenate([keep, rng.uniform(lo, hi, n)])) if n > 0 else np.sort(keep)


def spikes_around_events(spk, events, t_pre, t_post,
                         is_whisker=False, artifact_win_s=(-0.005, 0.005), rng=None):
    if rng is None:
        rng = np.random.default_rng()
    lo_a, hi_a = artifact_win_s
    i_lo   = np.searchsorted(spk, events - t_pre)
    i_hi   = np.searchsorted(spk, events + t_post, side="left")
    raster = [spk[i_lo[i]:i_hi[i]] - events[i] for i in range(len(events))]
    if is_whisker:
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
            mode="constant",
            cval=0.0
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

def precompute_event_map(trials: pd.DataFrame, cfg: dict) -> dict:
    """
    Group event times by (mouse_id, session_id, context, trial_type) once.
    Workers receive a plain dict[tuple, np.ndarray] — no DataFrame per neuron.
    """
    event_map = {}
    for keys, grp in trials.groupby([cfg["mouse_id_col"], cfg["session_id_col"],
                                     cfg["context_col"],   cfg["trial_type_col"]]):
        event_map[keys] = grp[cfg["align_col"]].to_numpy()
    return event_map


def _get_events(event_map, mouse_id, session_id, context, trial_type) -> np.ndarray:
    return event_map.get((mouse_id, session_id, context, trial_type), np.array([]))


# ── FR filter ──────────────────────────────────────────────────────────────────

def _unit_fr(st, mouse_id, session_id, event_map, cfg) -> float:
    events = np.concatenate([
        _get_events(event_map, mouse_id, session_id, ctx, tt)
        for tt, ctx in CONDITIONS
    ])
    if len(events) == 0:
        return 0.0
    i_lo = np.searchsorted(st, events - cfg["t_pre"])
    i_hi = np.searchsorted(st, events + cfg["t_post"], side="left")
    return (i_hi - i_lo).sum() / (len(events) * (cfg["t_pre"] + cfg["t_post"]))


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

    if cfg["zscore_full"]:
        wh_pre,  wh_post = _zscore_pair(wh_pre,  wh_post)
        au_pre,  au_post = _zscore_pair(au_pre,  au_post)
    else: # then z-score using baseline stats only
        wh_pre, wh_post = _zscore_pair_with_bas(wh_pre, wh_post, base_mask)
        au_pre, au_post = _zscore_pair_with_bas(au_pre, au_post, base_mask)

    return np.concatenate([wh_pre, wh_post, au_pre, au_post])

def _neuron_vector_strided(st, mouse_id, session_id, event_map, t_ctr, base_mask, cfg, conds):
    dt_stride = cfg["stride_ms"] / 1000
    dt_bin    = cfg["bin_ms"]    / 1000
    rng = np.random.default_rng()

    def _mean_psth_noclip(trial_type, context):
        events = _get_events(event_map, mouse_id, session_id, context, trial_type)
        if len(events) == 0:
            return np.zeros(len(t_ctr))
        raster = spikes_around_events(st, events, cfg["t_pre"], cfg["t_post"],
                                      is_whisker=(trial_type == cfg["whisker_trial_label"]),
                                      artifact_win_s=cfg["artifact_win_s"], rng=rng)
        rates = _bin_rates_strided(raster, cfg["t_pre"], cfg["t_post"],
                                   cfg["bin_ms"], cfg["stride_ms"])
        if cfg.get("square_fr", False):
            rates = rates ** 2
        return rates.mean(axis=0)

    def _mean_psth_withartifact(trial_type, context):
        events = _get_events(event_map, mouse_id, session_id, context, trial_type)
        if len(events) == 0:
            return np.zeros(len(t_ctr))
        raster = spikes_around_events(st, events, cfg["t_pre"], cfg["t_post"],
                                      is_whisker=(trial_type == cfg["whisker_trial_label"]),
                                      artifact_win_s=cfg["artifact_win_s"], rng=rng)
        # strictly exclude spikes at or beyond t_post regardless of ITI
        raster = [r[(r > -cfg["t_pre"]) & (r < cfg["t_post"])] for r in raster]
        rates = _bin_rates_strided(raster, cfg["t_pre"], cfg["t_post"],
                                   cfg["bin_ms"], cfg["stride_ms"])
        if cfg.get("square_fr", False):
            rates = rates ** 2
        return rates.mean(axis=0)

    def _mean_psth_presimple(trial_type, context):
        events = _get_events(event_map, mouse_id, session_id, context, trial_type)
        if len(events) == 0:
            return np.zeros(len(t_ctr))

        dt_stride = cfg["stride_ms"] / 1000
        k = max(1, int(round(cfg["bin_ms"] / cfg["stride_ms"])))
        pad = k // 2
        t_pre_ext = cfg["t_pre"] + pad * dt_stride
        t_post_ext = cfg["t_post"] + pad * dt_stride
        n_out = len(t_ctr)

        # extract wider window so kernel has real data at both edges
        raster = spikes_around_events(st, events, t_pre_ext, t_post_ext,
                                      is_whisker=(trial_type == cfg["whisker_trial_label"]),
                                      artifact_win_s=cfg["artifact_win_s"], rng=rng)
        # strict clip — no spikes outside extended window
        raster = [r[(r >= -t_pre_ext) & (r < t_post_ext)] for r in raster]

        rates = _bin_rates_strided(raster, t_pre_ext, t_post_ext,
                                   cfg["bin_ms"], cfg["stride_ms"], n_out)
        if cfg.get("square_fr", False):
            rates = rates ** 2
        return rates.mean(axis=0)

    def _mean_psth(trial_type, context):
        events = _get_events(event_map, mouse_id, session_id, context, trial_type)
        if len(events) == 0:
            return np.zeros(len(t_ctr))
        raster = spikes_around_events(st, events, cfg["t_pre"], cfg["t_post"],
                                      is_whisker=(trial_type == cfg["whisker_trial_label"]),
                                      artifact_win_s=cfg["artifact_win_s"], rng=rng)
        rates = _bin_and_smooth(raster, cfg["t_pre"], cfg["t_post"],
                                cfg["stride_ms"], cfg["sigma_ms"])
        if cfg.get("square_fr", False):
            rates = rates ** 2
        return rates.mean(axis=0)

    psths = [_mean_psth(tt, ctx) for tt, ctx in conds]

    def _zscore_pair(a, b):
        v = np.concatenate([a, b]); mean = v.mean(); std = v.std() + 1e-9
        return (a - mean) / std, (b - mean) / std

    def _zscore_pair_with_bas(a, b):
        bl = np.concatenate([a[base_mask], b[base_mask]]); mean = bl.mean(); std = bl.std() + 1e-9
        return (a - mean) / std, (b - mean) / std

    zscore_fn = _zscore_pair if cfg["zscore_full"] else _zscore_pair_with_bas

    if len(psths) == 4:   # both modalities
        psths[0], psths[1] = zscore_fn(psths[0], psths[1])
        psths[2], psths[3] = zscore_fn(psths[2], psths[3])
    elif len(psths) == 2: # single modality
        psths[0], psths[1] = zscore_fn(psths[0], psths[1])

    return np.concatenate(psths)

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
                         conds, cond_labels, cond_colors, cond_labels_matrix):
    dt_stride = cfg["stride_ms"] / 1000
    dt_bin    = cfg["bin_ms"]    / 1000
    dt    = cfg["stride_ms"] / 1000
    n_out     = int(round((cfg["t_pre"] + cfg["t_post"]) / dt_stride))
    #t_ctr     = np.linspace(-cfg["t_pre"], cfg["t_post"], n_out, endpoint=False) + dt_stride / 2
    t_ctr = np.linspace(-cfg["t_pre"], cfg["t_post"], n_out, endpoint=False)

    rows = Parallel(n_jobs=cfg["n_jobs"], prefer="threads")(
        delayed(_neuron_vector_strided)(
            st_map[uid], mouse_map[uid], session_map[uid],
            event_map, t_ctr, t_ctr < 0, cfg, conds,
        )
        for uid in unit_ids
    )
    return np.vstack(rows), t_ctr, n_out


# ── rastermap ──────────────────────────────────────────────────────────────────

def fit_rastermap(X, n_clusters):
    n_pcs  = min(200, X.shape[0] - 1, X.shape[1] - 1)
    model  = Rastermap(n_clusters=n_clusters,
                       n_PCs=n_pcs,
                       locality=0.75,
                       time_lag_window=0,
                       verbose =True).fit(X)
    isort  = model.isort
    bounds = np.round(np.linspace(0, len(isort), n_clusters + 1)[1:-1]).astype(int)
    return isort, bounds


def _kmeans_inertia(X, k):
    return KMeans(n_clusters=k, random_state=42, n_init=5).fit(X).inertia_



# ── plotting helpers ───────────────────────────────────────────────────────────

def _draw_matrix(ax, mat, n_bins, boundaries, vmax, cfg, title):
    n, n_total = mat.shape
    im = ax.imshow(mat, aspect="auto", interpolation="none", cmap="Greys",
                   vmin=-vmax, vmax=vmax, extent=[0, n_total, n, 0])
    #onset_bin = int(cfg["t_pre"] / (cfg["bin_ms"] / 1000)) #wo stride
    onset_bin = int(cfg["t_pre"] / (cfg["stride_ms"] / 1000))  # ← was bin_ms
    for i in range(1, n_total // n_bins):
        ax.axvline(i * n_bins, color="k", lw=1.5)
    for i in range(n_total//n_bins):
        ax.axvline(i * n_bins + onset_bin, color="white", lw=0.8, ls="--")
    for b in boundaries:
        ax.axhline(b, color="k", lw=0.8)
    ticks = [(i + 0.5) * n_bins for i in range(n_total // n_bins)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(COND_LABELS_MATRIX, fontsize=8)
    ax.set_ylabel("Neuron")
    ax.set_title(title, fontsize=10)
    return im


def _save(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {Path(path).name}")


# ── figures ────────────────────────────────────────────────────────────────────

def fig0_data_summary(units_raw, units_good, unit_ids, trials, cfg, out_dir):
    mid = cfg["mouse_id_col"]
    sid = cfg["session_id_col"]
    rows = [
        ("Mice",              trials[mid].nunique()),
        ("Sessions",          trials.groupby([mid, sid]).ngroups),
        ("Neurons (raw)",     len(units_raw)),
        ("Neurons (bc=good)", len(units_good)),
        ("Neurons (FR pass)", len(unit_ids)),
        ("Trials total",      len(trials)),
    ]
    for (tt, ctx), label in zip(CONDITIONS, COND_LABELS):
        n = ((trials[cfg["context_col"]] == ctx) & (trials[cfg["trial_type_col"]] == tt)).sum()
        rows.append((f"  {label}", n))

    fig, ax = plt.subplots(figsize=(5, 0.45 * len(rows) + 1))
    ax.axis("off")
    tbl = ax.table(cellText=[[k, str(v)] for k, v in rows],
                   colLabels=["", "Count"], loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    ax.set_title("Data summary", fontsize=11, pad=12)
    fig.tight_layout()
    _save(fig, out_dir / "fig0_data_summary.png")


def fig1_trial_counts(trials, cfg, out_dir):
    counts = [((trials[cfg["context_col"]] == ctx) & (trials[cfg["trial_type_col"]] == tt)).sum()
               for tt, ctx in CONDITIONS]
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(COND_LABELS, counts, color=COND_COLORS, edgecolor="none")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + max(counts) * 0.01,
                str(c), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Trial count")
    ax.set_title("Trial counts per condition")
    fig.tight_layout()
    _save(fig, out_dir / "fig1_trial_counts.png")


def fig2_fr_distribution(fr_map, unit_ids, thr, out_dir):
    all_vals  = np.array(list(fr_map.values()))
    kept_vals = np.array([fr_map[uid] for uid in unit_ids])
    fig, ax   = plt.subplots(figsize=(5, 3))
    ax.hist(all_vals,  bins=80, color="steelblue", alpha=0.5, label="all",  edgecolor="none")
    ax.hist(kept_vals, bins=80, color="navy",      alpha=0.8, label="kept", edgecolor="none")
    ax.axvline(thr, color="r", lw=1.5, ls="--", label=f"threshold = {thr} Hz")
    ax.set_xlabel("Mean FR (Hz)")
    ax.set_ylabel("Neuron count")
    ax.set_yscale("log")
    ax.set_title(f"FR distribution  ({len(all_vals)} total → {len(kept_vals)} kept)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir / "fig2_fr_distribution.png")


def fig3_neuron_counts(units_raw, units_good, unit_ids_final, cfg, out_dir):
    mid     = cfg["mouse_id_col"]
    mice    = sorted(units_raw[mid].unique())
    uid_set = set(unit_ids_final)
    raw     = [len(units_raw[units_raw[mid] == m])  for m in mice]
    good    = [len(units_good[units_good[mid] == m]) for m in mice]
    final   = [sum(1 for uid in unit_ids_final
                   if uid in units_good.index and units_good.loc[uid, mid] == m)
               for m in mice]
    x, w = np.arange(len(mice)), 0.25
    fig, ax = plt.subplots(figsize=(max(5, len(mice) * 1.0), 3.5))
    ax.bar(x - w, raw,   w, label="raw",     color="lightsteelblue", edgecolor="none")
    ax.bar(x,     good,  w, label="bc=good", color="steelblue",      edgecolor="none")
    ax.bar(x + w, final, w, label="FR pass", color="navy",           edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in mice], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Neuron count")
    ax.set_title("Neurons per mouse at each filter stage")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir / "fig3_neuron_counts.png")


def fig4_sample_neurons(unit_ids, st_map, mouse_map, session_map, event_map, t_ctr, cfg, out_dir):
    """Random sample of neurons, 4 conditions overlaid per subplot."""
    n         = min(cfg["n_sample_neurons"], len(unit_ids))
    rng       = np.random.default_rng(0)
    sample    = rng.choice(len(unit_ids), size=n, replace=False)
    ncols     = 6
    nrows     = int(np.ceil(n / ncols))
    base_mask = t_ctr < 0                          # derived from t_ctr, always consistent
    n_out     = len(t_ctr)

    dt_stride  = cfg["stride_ms"] / 1000
    k          = max(1, int(round(cfg["bin_ms"] / cfg["stride_ms"])))
    pad        = k // 2
    t_pre_ext  = cfg["t_pre"]  + pad * dt_stride
    t_post_ext = cfg["t_post"] + pad * dt_stride

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows),
                              sharey=False, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for i, idx in enumerate(sample):
        uid, ax = unit_ids[idx], axes[i]
        st      = st_map[uid]
        for c, (trial_type, context) in enumerate(CONDITIONS):
            events = _get_events(event_map, mouse_map[uid], session_map[uid], context, trial_type)
            if len(events) == 0:
                continue
            #aster   = spikes_around_events(st, events, cfg["t_pre"], cfg["t_post"],
            #                               is_whisker=(trial_type == cfg["whisker_trial_label"]),
            #                               artifact_win_s=cfg["artifact_win_s"],
            #                               rng=np.random.default_rng(c))
            raster = spikes_around_events(st, events, t_pre_ext, t_post_ext, is_whisker=(trial_type == cfg["whisker_trial_label"]),
                                            artifact_win_s=cfg["artifact_win_s"],
                                            rng=np.random.default_rng(c))
            raster = [r[(r >= -t_pre_ext) & (r < t_post_ext)] for r in raster]
            #rates    = _bin_rates(raster, bins, dt) #wo stride
            rates = _bin_rates_strided(raster, t_pre_ext, t_post_ext,
                                       cfg["bin_ms"], cfg["stride_ms"], n_out)
            rates_bc = rates - rates[:, base_mask].mean(axis=1, keepdims=True)
            mean_sm = rates_bc.mean(0)
            #mean_sm  = _causal_gaussian(rates_bc.mean(0), cfg["sigma_ms"] / cfg["bin_ms"])
            ax.plot(t_ctr, mean_sm, color=COND_COLORS[c], lw=1.0, label=COND_LABELS[c])
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_title(f"uid={uid}", fontsize=7)
        if i % ncols == 0:
            ax.set_ylabel("ΔFR (Hz)", fontsize=7)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Time (s)", fontsize=7)
        if i == 0:
            ax.legend(fontsize=5, loc="upper left")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Sample neurons (n={n}, random)", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / "fig4_sample_neurons.png", dpi=120)


def fig5_population_matrix_old(X, n_bins, isort, boundaries, vmax, cfg, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=400)
    im1 = _draw_matrix(axes[0], X,        n_bins, [],         vmax, cfg,  f"Input order  (n={len(X)})")
    im2 = _draw_matrix(axes[1], X[isort], n_bins, boundaries, vmax, cfg, f"Rastermap order (n={len(X)}")
    for ax, im in zip(axes, [im1, im2]):
        fig.colorbar(im, ax=ax, label="z-score", shrink=0.3, pad=0.01)
    fig.tight_layout()
    _save(fig, out_dir / "fig5_population_matrix.png", dpi=200)


def fig6_cluster_profiles(X, t_ctr, n_bins, isort, boundaries, out_dir):
    n_clusters = len(boundaries) + 1
    edges      = [0] + list(boundaries) + [len(isort)]
    ncols      = (n_clusters + 1) // 2
    nrows      = 4 if n_clusters > 1 else 1
    fig, axes  = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                               sharey=True, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for k in range(n_clusters):
        ax       = axes[k]
        idx      = isort[edges[k]:edges[k + 1]]
        mean_vec = X[idx].mean(0)
        sem_vec  = X[idx].std(0) / np.sqrt(len(idx))
        for c, (label, color) in enumerate(zip(COND_LABELS, COND_COLORS)):
            mean_vec_plot = mean_vec[1+c * n_bins:(c + 1) * n_bins -1]
            sem_vec_plot = sem_vec[1+c * n_bins:(c + 1) * n_bins -1]
            ax.plot(t_ctr[1:-1], mean_vec_plot,
                    color=color, lw=1.5, label=label)

            ax.fill_between(t_ctr[1:-1], mean_vec_plot - sem_vec_plot,
                             mean_vec_plot + sem_vec_plot,
                             color=color, alpha=0.3)
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_title(f"C{k+1}  (n={len(idx)})", fontsize=8)
        if k % ncols == 0:
            ax.set_ylabel("z-score")
        if k >= (nrows - 1) * ncols:
            ax.set_xlabel("Time (s)")
        ax.legend(fontsize=6, loc="upper right", frameon=False)
    for ax in axes[n_clusters:]:
        ax.set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir / "fig6_cluster_profiles.png")

def fig6b_cluster_profiles_reward_groups(X, t_ctr, n_bins, isort, boundaries,
                                   cond_labels, cond_colors, reward_arr, out_dir):
    wh_idx = [i for i, l in enumerate(cond_labels) if "Whisker" in l]
    if len(wh_idx) == 0:
        print("  No whisker conditions found, skipping fig6b")
        return

    n_clusters = len(boundaries) + 1
    edges      = [0] + list(boundaries) + [len(isort)]
    ncols      = (n_clusters + 1) // 2
    nrows      = 4 if n_clusters > 1 else 1
    fig, axes  = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                               sharey=True, sharex=True)
    axes = np.atleast_1d(axes).ravel()

    reward_groups = [("R+", "forestgreen"), ("R-", "crimson")]

    for k in range(n_clusters):
        ax  = axes[k]
        idx = isort[edges[k]:edges[k + 1]]

        for rg, rg_color in reward_groups:
            rg_mask   = reward_arr[idx] == rg
            rg_idx    = idx[rg_mask]
            if len(rg_idx) == 0:
                continue
            mean_vec  = X[rg_idx].mean(0)
            sem_vec   = X[rg_idx].std(0) / np.sqrt(len(rg_idx))

            for c in wh_idx:
                label = f"{cond_labels[c]} {rg}"
                ls    = "-" if "pre" in cond_labels[c].lower() else "--"
                ls     = "-"
                m     = mean_vec[c * n_bins:(c + 1) * n_bins]
                s     = sem_vec[c * n_bins:(c + 1) * n_bins]
                color = plotting_utils.adjust_lightness(rg_color, 1.5) if "pre" in cond_labels[c].lower() else rg_color
                ax.plot(t_ctr, m, color=color, lw=1.5, ls=ls, label=label)
                ax.fill_between(t_ctr, m - s, m + s, color=rg_color, alpha=0.2)

        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        n_rp = (reward_arr[idx] == "R+").sum()
        n_rm = (reward_arr[idx] == "R-").sum()
        ax.set_title(f"C{k+1}  (R+={n_rp}, R−={n_rm})", fontsize=8)
        if k % ncols == 0:
            ax.set_ylabel("z-score")
        if k >= (nrows - 1) * ncols:
            ax.set_xlabel("Time (s)")
        ax.legend(fontsize=6, loc="upper right", frameon=False)

    for ax in axes[n_clusters:]:
        ax.set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir / "fig6b_cluster_profiles_reward_groups.png")

def fig7_pca_variance(X, n_pca, out_dir):
    pca    = PCA(n_components=n_pca).fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.plot(np.arange(1, n_pca + 1), pca.explained_variance_ratio_ * 100,
             "o-", ms=3, lw=1, color="steelblue")
    ax1.set_xlabel("PC"); ax1.set_ylabel("Variance explained (%)")
    ax1.set_title("Scree plot")
    ax2.plot(np.arange(1, n_pca + 1), cumvar, "o-", ms=3, lw=1, color="steelblue")
    ax2.axhline(80, color="r", ls="--", lw=1, label="80%")
    ax2.axhline(95, color="r", ls=":",  lw=1, label="95%")
    ax2.set_xlabel("PC"); ax2.set_ylabel("Cumulative variance (%)")
    ax2.set_title("Cumulative variance"); ax2.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir / "fig7_pca_variance.png")


def fig8_umap(emb, cluster_labels, n_clusters, out_dir):
    fig, (ax1, ax2, cax) = plt.subplots(1, 3, figsize=(11, 5),gridspec_kw={"width_ratios": [1, 1, 0.05]})
    ax1.scatter(emb[:, 0], emb[:, 1], s=2, c="steelblue", alpha=0.8, linewidths=0)
    ax1.set_title("UMAP"); ax1.set_xlabel("UMAP 1"); ax1.set_ylabel("UMAP 2")
    sc = ax2.scatter(emb[:, 0], emb[:, 1], s=2, c=cluster_labels,
                     cmap="tab20", alpha=0.8, linewidths=0, vmin=0, vmax=n_clusters - 1)
    ax2.set_title("UMAP — rastermap clusters"); ax2.set_xlabel("UMAP 1")
    cb = plt.colorbar(sc, cax=cax, label="Rastermap cluster")
    fig.tight_layout()
    _save(fig, out_dir / "fig8_umap.png")


def fig9_kmeans(emb, km_labels, k, k_range, inertias, out_dir):
    fig, (ax1, ax2, cax) = plt.subplots(1, 3, figsize=(11, 5),gridspec_kw={"width_ratios": [1, 1, 0.05]})
    sc = ax1.scatter(emb[:, 0], emb[:, 1], s=2, c=km_labels,
                     cmap="tab10", alpha=0.6, linewidths=0, vmin=0, vmax=k - 1)
    ax1.set_title(f"UMAP — k-means  (k={k})")
    ax1.set_xlabel("UMAP 1"); ax1.set_ylabel("UMAP 2")
    cb = plt.colorbar(sc, cax=cax, label="Rastermap cluster")
    ax2.plot(list(k_range), inertias, "o-", color="k", lw=1.5)
    ax2.set_xlabel("k"); ax2.set_ylabel("Inertia"); ax2.set_title("K-means elbow")
    fig.tight_layout()
    _save(fig, out_dir / "fig9_kmeans.png")

def fig10_kmeans_profiles(X, t_ctr, n_bins, km_labels, k, out_dir):
    ncols = (k + 1) // 2
    nrows = 2 if k > 1 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                              sharey=True, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ki in range(k):
        ax       = axes[ki]
        idx      = np.where(km_labels == ki)[0]
        mean_vec = X[idx].mean(0)
        sem_vec = X[idx].std(0) / np.sqrt(len(idx))
        for c, (label, color) in enumerate(zip(COND_LABELS, COND_COLORS)):
            mean_vec_plot = mean_vec[1+ c* n_bins:(c + 1) * n_bins-1]
            sem_vec_plot = sem_vec[1+ c* n_bins:(c + 1) * n_bins-1]
            ax.plot(t_ctr[1:-1], mean_vec_plot,
                    color=color, lw=1.2, label=label)
            ax.fill_between(t_ctr[1:-1], mean_vec_plot - sem_vec_plot,
                            mean_vec_plot + sem_vec_plot,
                            color=color, alpha=0.3)
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_title(f"K{ki + 1}  (n={len(idx)})", fontsize=8)
        if ki % ncols == 0:
            ax.set_ylabel("z-score")
        if ki >= (nrows - 1) * ncols:
            ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8, loc="upper right", frameon=False)
    for ax in axes[k:]:
        ax.set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir / "fig10_kmeans_profiles.png")

def fig5_population_matrix(X, n_bins, isort, boundaries, vmax, cfg, reward_arr, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(22, 9), dpi=400,
                              gridspec_kw={"width_ratios": [10, 10, 1]})
    im1 = _draw_matrix(axes[0], X,        n_bins, [],         vmax, cfg, f"Input order  (n={len(X)})")
    im2 = _draw_matrix(axes[1], X[isort], n_bins, boundaries, vmax, cfg, f"Rastermap order (n={len(X)})")
    for ax, im in zip(axes[:2], [im1, im2]):
        fig.colorbar(im, ax=ax, label="z-score", shrink=0.3, pad=0.01)

    # Third panel: R+/R- proportion per cluster, vertical stacked bars
    ax3          = axes[2]
    ax3.spines["top"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    n_neurons    = len(isort)
    edges        = [0] + list(boundaries) + [n_neurons]
    n_clusters   = len(boundaries) + 1
    reward_sorted = reward_arr[isort]
    is_rplus      = reward_sorted == "R+"

    for k in range(n_clusters):
        lo, hi  = edges[k], edges[k + 1]
        n_cl    = hi - lo
        if n_cl == 0:
            continue
        prop_rp = is_rplus[lo:hi].sum() / n_cl
        center  = (lo + hi) / 2
        ax3.barh(center, prop_rp,        height=n_cl * 0.92,
                 color="forestgreen", edgecolor="none", align="center")
        ax3.barh(center, 1 - prop_rp,   height=n_cl * 0.92, left=prop_rp,
                 color="crimson",     edgecolor="none", align="center")

    ax3.set_xlim(0, 1)
    ax3.set_ylim(n_neurons, 0)
    ax3.set_xticks([0, 0.5, 1])
    ax3.set_xticklabels(["0", ".5", "1"], fontsize=8)
    ax3.set_yticks([])
    ax3.set_xlabel("Proportion", fontsize=8)
    ax3.set_title("Reward group", fontsize=10)
    # legend patches
    #from matplotlib.patches import Patch
    #ax3.legend(handles=[Patch(color="forestgreen", label="R+"),
    #                     Patch(color="crimson",     label="R−")],
    #           fontsize=6, frameon=False,
    #           loc="lower right", bbox_to_anchor=(1, 0))

    fig.tight_layout()
    _save(fig, out_dir / "fig5_population_matrix.png", dpi=400)


def fig11_area_per_cluster(unit_ids, cluster_labels, area_arr, n_clusters, out_dir):
    """Stacked bar: area composition per rastermap cluster."""
    all_areas  = sorted(set(area_arr))
    cmap       = plt.cm.get_cmap("tab20", len(all_areas))
    area_color = {a: cmap(i) for i, a in enumerate(all_areas)}

    ncols = (n_clusters + 1) // 2
    nrows = 2 if n_clusters > 1 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for k in range(n_clusters):
        ax     = axes[k]
        mask   = cluster_labels == k
        counts = pd.Series(area_arr[mask]).value_counts()
        total  = mask.sum()
        bottom = 0.0
        for area, cnt in counts.items():
            pct = cnt / total
            ax.bar(0, pct, bottom=bottom, color=area_color[area],
                   edgecolor="none", width=0.7)
            bottom += pct
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylim(0, 1)
        ax.set_title(f"C{k+1}\n(n={total})", fontsize=7)
        if k % ncols == 0:
            ax.set_ylabel("Proportion", fontsize=8)

    for ax in axes[n_clusters:]:
        ax.set_visible(False)

    from matplotlib.patches import Patch
    handles = [Patch(color=area_color[a], label=a) for a in all_areas]
    fig.legend(handles, all_areas, loc="lower center",
               ncol=min(8, len(all_areas)), fontsize=6,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Area composition per rastermap cluster", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / "fig11_area_per_cluster.png")


def fig12_reward_per_cluster(unit_ids, cluster_labels, reward_arr, n_clusters, out_dir):
    """R+/R- proportion per rastermap cluster with per-cluster Fisher's exact + BH correction."""
    is_rplus       = reward_arr == "R+"
    n_tot_rp       = is_rplus.sum()
    n_tot_rm       = (~is_rplus).sum()
    overall_prop   = n_tot_rp / len(reward_arr) if len(reward_arr) > 0 else 0

    props_rp, props_rm, ns, pvals = [], [], [], []
    for k in range(n_clusters):
        mask    = cluster_labels == k
        n_cl_rp = (mask & is_rplus).sum()
        n_cl_rm = (mask & ~is_rplus).sum()
        total   = n_cl_rp + n_cl_rm
        table   = [[n_cl_rp, n_cl_rm],
                   [n_tot_rp - n_cl_rp, n_tot_rm - n_cl_rm]]
        _, p    = fisher_exact(table)
        pvals.append(p)
        props_rp.append(n_cl_rp / total if total > 0 else 0)
        props_rm.append(n_cl_rm / total if total > 0 else 0)
        ns.append(total)

    # Benjamini-Hochberg correction
    pvals  = np.array(pvals)
    order  = np.argsort(pvals)
    thresh = (np.arange(1, n_clusters + 1) / n_clusters) * 0.05
    sig    = pvals[order] <= thresh
    reject = np.zeros(n_clusters, dtype=bool)
    if sig.any():
        reject[order[:np.where(sig)[0].max() + 1]] = True

    x   = np.arange(n_clusters)
    fig, ax = plt.subplots(figsize=(max(6, n_clusters * 0.7), 4))
    ax.bar(x, props_rp, color="forestgreen", label="R+", edgecolor="none")
    ax.bar(x, props_rm, bottom=props_rp, color="crimson", label="R−", edgecolor="none")
    ax.axhline(overall_prop, color="forestgreen", ls="--", lw=1.2, alpha=0.7,
               label=f"R+ overall ({overall_prop:.2f})")

    for k in range(n_clusters):
        if reject[k]:
            ax.text(k, 1.03, "*", ha="center", va="bottom", fontsize=11, color="k")

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{k+1}\n(n={ns[k]})" for k in range(n_clusters)], fontsize=7)
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.12)
    ax.set_title("Reward group composition per rastermap cluster\n"
                 "(* BH-corrected Fisher's exact, α=0.05)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir / "fig12_reward_per_cluster.png")

def _lz76(seq):
    """Normalised LZ76 complexity of a binary sequence."""
    s, n = list(seq), len(seq)
    if n == 0:
        return 0.0
    c, l, k, i, kmax = 1, 1, 1, 1, 1
    while i + k <= n:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
        else:
            kmax = max(kmax, k)
            l += 1
            if l == i:
                c += 1; i += kmax; l = 1; k = 1; kmax = 1
            else:
                k = 1
    return c / (n / np.log2(n + 1e-9))


def plot_example_neurons_clusters(X, t_ctr, n_bins, cluster_labels, unit_ids,
                                  units_good, cfg, cond_labels, cond_colors, out_dir):
    """
    One figure per cluster. Neurons selected by combined score:
        score = alpha * norm(dist_to_centroid) + (1-alpha) * norm(|lz - median_lz|)
    Lower score = closer to centroid shape AND typical temporal complexity.
    Traces are the concatenated z-scored feature vector (all conditions as one trace).
    Annotated with area_acronym_custom and reward_group.
    """
    try:
        from allen_utils import get_custom_area_color_per_group  # local module; fallback below
        def _area_color(area):
            try:
                area_dict, _ = get_custom_area_color_per_group()
                return area_dict[area]
            except:
                return "steelblue"
    except ImportError as err:
        print(err)
        cmap_areas = plt.cm.get_cmap("tab20")
        all_areas  = sorted(set(units_good["area_acronym_custom"].dropna().unique()))
        area_idx   = {a: i for i, a in enumerate(all_areas)}
        def _area_color(area):
            return cmap_areas(area_idx.get(area, 0) % 20)

    ex_dir = out_dir / "example_neurons"
    ex_dir.mkdir(exist_ok=True)

    n_clusters  = len(np.unique(cluster_labels))
    alpha       = cfg.get("example_alpha", 0.5)
    n_plot      = cfg.get("n_example_neurons", 15)
    area_map    = units_good["area_acronym_custom"].to_dict()
    reward_map  = units_good[cfg["reward_group_col"]].to_dict()

    # full time axis for concatenated vector
    n_conds   = len(cond_labels)
    t_full    = np.concatenate([t_ctr + i * (t_ctr[-1] - t_ctr[0] + (t_ctr[1]-t_ctr[0]))
                                 for i in range(n_conds)])

    for k in range(n_clusters):
        print('Plotting for cluster', k)

        idx = np.where(cluster_labels == k)[0]
        if len(idx) == 0:
            continue

        Xk       = X[idx]
        centroid = Xk.mean(0)

        # centroid distance
        dists = np.linalg.norm(Xk - centroid, axis=1)
        #dists_n = (dists - dists.min()) / (dists.ptp() + 1e-9)
        dists_n = (dists - dists.min()) / (dists.max() - dists.min() + 1e-9)

        # LZ complexity on binarized feature vector
        lz_vals = np.array([_lz76((row > 0).astype(int)) for row in Xk])
        med_lz  = np.median(lz_vals)
        lz_dev  = np.abs(lz_vals - med_lz)
        #lz_n    = (lz_dev - lz_dev.min()) / (lz_dev.ptp() + 1e-9)
        lz_n = (lz_dev - lz_dev.min()) / (lz_dev.max() - lz_dev.min() + 1e-9)
        score   = alpha * dists_n + (1 - alpha) * lz_n
        chosen  = idx[np.argsort(score)[:n_plot]]

        # stacking offset
        offset  = np.abs(X[chosen]).max() * 1.5

        fig, ax = plt.subplots(figsize=(8, 0.9 * len(chosen) + 1))

        # condition boundary lines on x-axis
        onset_bin = int(cfg["t_pre"] / (cfg["stride_ms"] / 1000))  # ← was bin_ms
        onset_bins = [onset_bin + ci * n_bins for ci in range(n_conds)]
        for ci in range(1, n_conds):
            ax.axvline(ci * n_bins, color="lightgrey", lw=2, ls="-", zorder=0)
            ax.axvline(onset_bins[ci-1], color="k", lw=1, ls="--", zorder=0)
        ax.axhline(0, color="k", lw=0.3, alpha=0.3)

        for j, neuron_idx in enumerate(chosen):
            uid    = unit_ids[neuron_idx]
            area   = area_map.get(uid, "unknown")
            rgroup = reward_map.get(uid, "?")
            color  = _area_color(area)
            trace  = X[neuron_idx][1:] #exclude first point to add discontinuity
            y      = trace + j * offset
            ax.plot(np.arange(len(trace)), y, color="k", lw=2, alpha=1.0)
            ax.text(-len(t_ctr) * 0.03, j * offset,
                    f"{area}  {rgroup}", fontsize=7, va="center",
                    ha="right", color=color)

        # x-tick labels: one per condition at center
        tick_pos    = [(ci + 0.5) * n_bins for ci in range(n_conds)]
        tick_labels = cond_labels
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticks([])
        ax.set_xlim(-n_bins * 0.15, n_conds * n_bins)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_xlabel("Condition / time")
        ax.set_ylabel("Neuron (stacked)")
        ax.set_title(f"Cluster {k+1}  (n shown={len(chosen)} of {len(idx)})", fontsize=9)
        fig.tight_layout()
        _save(fig, ex_dir / f"cluster_{k+1:02d}_examples.png", dpi=300)

    return

# ── entry point ────────────────────────────────────────────────────────────────

def run_rastermap_psth(units: pd.DataFrame,
                       trials: pd.DataFrame,
                       out_root: str | Path = "rastermap_psth_out",
                       **cfg_overrides) -> dict:
    cfg      = {**DEFAULT_CFG, **cfg_overrides}

    # ── output path ────────────────────────────────────────────────────────
    zscore_tag = "zscore_full" if cfg["zscore_full"] else "zscore_bl"
    period_tag = cfg["period"]  # both|whisker|auditory
    mod_tag = cfg["modality"]  # both|whisker|auditory
    reward_tag = cfg["reward_filter"].replace("+", "plus").replace("-", "minus")
    out_folder = Path(out_root, "passive_rastermap_psth", period_tag, zscore_tag, mod_tag, reward_tag)
    out_folder.mkdir(parents=True, exist_ok=True)

    #out_root = Path(out_root, "passive_rastermap_psth")
    #out_root.mkdir(parents=True, exist_ok=True)
    #if cfg['zscore_full']==True:
    #    out_folder = out_root / "zscore_full"
    #elif cfg["zscore_full"]==False:
    #    out_folder = out_root / "zscore"
    #out_folder.mkdir(parents=True, exist_ok=True)
    assert units.index.is_unique, "units index must be unique"


    print('Excluding non-learners...')
    mouse_info = pd.read_excel(MOUSE_INFO)
    valid_mice = mouse_info[mouse_info['learning_category'].isin(['good','moderate'])]['mouse_id'].unique()
    units = units[units.mouse_id.isin(valid_mice)]
    trials = trials[trials.mouse_id.isin(valid_mice)]


    print("Assigning passive contexts...")
    trials     = assign_passive_context(trials, cfg["mouse_id_col"], cfg["session_id_col"])
    units_raw  = units.copy()

    print("Unit selection (bc_label == 'good')...")
    units_good = units[units.bc_label.isin(["good"])]
    all_ids    = units_good.index.tolist()
    print(f"  {len(units_raw)} → {len(units_good)} units")

    # Pre-extract into plain dicts — avoids repeated DataFrame .loc in workers
    print("Pre-extracting spike times and metadata...")
    st_map      = {uid: get_spike_times(units_good.loc[uid]) for uid in all_ids}
    mouse_map   = units_good[cfg["mouse_id_col"]].to_dict()
    session_map = units_good[cfg["session_id_col"]].to_dict()

    # One groupby scan of trials instead of N_neurons × 4 boolean masks
    print("Pre-grouping trial event times...")
    event_map = precompute_event_map(trials, cfg)
    print(f"  {len(event_map)} (mouse, session, context, trial_type) groups found")

    # Check which mice have passive data
    mice_with_passive = {
        mouse_id
        for (mouse_id, session_id, context, trial_type) in event_map
        if context in ("passive_pre", "passive_post")
    }
    mice_all = set(mouse_map[uid] for uid in all_ids)
    mice_missing = mice_all - mice_with_passive
    if mice_missing:
        print(f"  Dropping {len(mice_missing)} mice with no passive data: {mice_missing}")
        all_ids = [uid for uid in all_ids if mouse_map[uid] in mice_with_passive]
        print(f"  {len(all_ids)} units remaining")

    print("Applying FR filter...")
    unit_ids, fr_map = apply_fr_filter(all_ids, st_map, mouse_map, session_map, event_map, cfg)

    # extract per-neuron metadata arrays aligned to unit_ids
    reward_map = units_good[cfg["reward_group_col"]].to_dict()
    reward_map = {uid: ("R+" if r == 1 else "R-" if r == 0 else "unknown") for uid, r in reward_map.items()}


    area_map = units_good[cfg["area_col"]].to_dict()
    reward_arr = np.array([reward_map.get(uid, "unknown") for uid in unit_ids])
    #reward_arr = np.where(reward_arr == 1, "R+", np.where(reward_arr == 0, "R-", "unknown"))
    area_arr = np.array([area_map.get(uid, "unknown") for uid in unit_ids])

    # Diagnostic figures
    fig0_data_summary(units_raw, units_good, unit_ids, trials, cfg, out_folder)
    fig1_trial_counts(trials, cfg, out_folder)
    fig2_fr_distribution(fr_map, unit_ids, cfg["fr_threshold_hz"], out_folder)
    fig3_neuron_counts(units_raw, units_good, unit_ids, cfg, out_folder)

    dt = cfg["bin_ms"] / 1000
    bins = np.arange(-cfg["t_pre"], cfg["t_post"] + dt, dt)
    t_ctr = bins[:-1] + dt / 2
    n_bins = len(t_ctr)

    fname = "feature_matrix.npy" if cfg["zscore_full"] else "feature_matrix_no_baseline.npy"
    fpath = out_folder / fname

    #if fpath.exists():
    #    print(f"Loading feature matrix from {fname}...")
    #    X = np.load(fpath)
    #else:

    print(f"Building feature matrix ({len(unit_ids)} neurons × 4 conditions)...")
    #X, t_ctr, n_bins = build_feature_matrix(
    #    unit_ids, st_map, mouse_map, session_map, event_map, cfg)
    X, t_ctr, n_bins = build_feature_matrix_strided(
        unit_ids, st_map, mouse_map, session_map, event_map, cfg,
    CONDITIONS, COND_LABELS, COND_COLORS, COND_LABELS_MATRIX)

    onset_idx = np.argmin(np.abs(t_ctr)) # diag to see alignment
    print(f"  t_ctr[0]={t_ctr[0] * 1000:.2f}ms, t_ctr[onset]={t_ctr[onset_idx] * 1000:.2f}ms, "
          f"t_ctr[-1]={t_ctr[-1] * 1000:.2f}ms")

    print(f"  X shape: {X.shape}")
    np.save(fpath, X)

    fig4_sample_neurons(unit_ids, st_map, mouse_map, session_map, event_map, t_ctr, cfg, out_folder)

    n_k = cfg["n_rastermap_clusters"]
    print(f"Running rastermap (n_clusters={n_k})...")
    isort, boundaries = fit_rastermap(X, n_k)

    cluster_labels = np.empty(len(unit_ids), dtype=int)
    edges = [0] + list(boundaries) + [len(isort)]
    for k in range(n_k):
        cluster_labels[isort[edges[k]:edges[k + 1]]] = k

    vmax = np.nanpercentile(np.abs(X), cfg["vmax_pct"])
    #fig5_population_matrix_old(X, n_bins, isort, boundaries, vmax, cfg, out_folder)
    fig5_population_matrix(X, n_bins, isort, boundaries, vmax, cfg, reward_arr, out_folder)
    fig6_cluster_profiles(X, t_ctr, n_bins, isort, boundaries, out_folder)
    fig6b_cluster_profiles_reward_groups(X, t_ctr, n_bins, isort, boundaries,
                                   COND_LABELS, COND_COLORS, reward_arr, out_folder)

    # ── feature matrix diagnostics ─────────────────────────────────────────
    nan_rows = np.isnan(X).any(axis=1)
    inf_rows = np.isinf(X).any(axis=1)
    zero_rows = (X == 0).all(axis=1)
    constant_rows = X.std(axis=1) < 1e-6
    print(f"  NaN rows:      {nan_rows.sum()}")
    print(f"  Inf rows:      {inf_rows.sum()}")
    print(f"  All-zero rows: {zero_rows.sum()}")
    print(f"  Constant rows (std < 1e-6): {constant_rows.sum()}")

    odd = nan_rows | inf_rows | zero_rows | constant_rows
    if odd.sum() > 0:
        print(f"  Total odd rows: {odd.sum()} — dropping before rastermap/UMAP")
        odd_uids = [unit_ids[i] for i in np.where(odd)[0]]
        print(f"  Example odd uids: {odd_uids[:10]}")
        unit_ids = [uid for uid, o in zip(unit_ids, odd) if not o]
        X = X[~odd]
        reward_arr = reward_arr[~odd]
        area_arr = area_arr[~odd]
        cluster_labels = cluster_labels[~odd] if len(cluster_labels) == len(odd) else cluster_labels
        print(f"  X shape after dropping: {X.shape}")


    n_pca = min(150, len(unit_ids) - 1, X.shape[1] - 1)
    print(f"Running PCA ({n_pca} components) + UMAP...")
    X_pca = PCA(n_components=n_pca).fit_transform(X)
    fig7_pca_variance(X, n_pca, out_folder)
    emb = umap.UMAP(n_neighbors=cfg["umap_n_neighbors"],
                    min_dist=cfg["umap_min_dist"],
                    n_components=2,
                    random_state=42).fit_transform(X)
    fig8_umap(emb, cluster_labels, n_k, out_folder)

    k = cfg["k_means_k"]
    print(f"Running k-means (k={k}) + elbow...")
    km_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    inertias  = Parallel(n_jobs=cfg["n_jobs"])(
        delayed(_kmeans_inertia)(X, ki) for ki in cfg["k_elbow_range"]
    )
    fig9_kmeans(emb, km_labels, k, cfg["k_elbow_range"], inertias, out_folder)

    # Now plot k-means clusters
    fig10_kmeans_profiles(X, t_ctr, n_bins, km_labels, k, out_folder)

    # Compare contributions of areas and reward groups
    fig11_area_per_cluster(unit_ids, cluster_labels, area_arr, n_k, out_folder)
    fig12_reward_per_cluster(unit_ids, cluster_labels, reward_arr, n_k, out_folder)

    # Plot example neurons
    plot_example_neurons_clusters(X, t_ctr, n_bins, cluster_labels, unit_ids,
                                  units_good, cfg, COND_LABELS, COND_COLORS, out_folder)

    print(f"\nDone. Outputs → {out_folder}")
    return dict(
        X=X, t_ctr=t_ctr, n_bins=n_bins, unit_ids=unit_ids,
        isort=isort, boundaries=boundaries, cluster_labels=cluster_labels,
        umap_embedding=emb, km_labels=km_labels,
    )
