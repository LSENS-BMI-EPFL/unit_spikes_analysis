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

# ── config ─────────────────────────────────────────────────────────────────────

DEFAULT_CFG: dict[str, Any] = dict(
    t_pre                = 0.1,
    t_post               = 0.2,
    bin_ms               = 5,
    sigma_ms             = 1,
    artifact_win_s       = (-0.005, 0.005),
    whisker_trial_label  = "whisker_trial",
    fr_threshold_hz      = 1.0,
    zscore_full           = False,
    align_col            = "start_time",
    context_col          = "context",
    trial_type_col       = "trial_type",
    mouse_id_col         = "mouse_id",
    session_id_col       = "session_id",
    n_rastermap_clusters = 16,
    k_means_k            = 8,
    k_elbow_range        = range(2, 12),
    umap_n_neighbors     = 15,
    umap_min_dist        = 0.5,
    vmax_pct             = 95,
    n_sample_neurons     = 24,
    n_jobs               = 60,
)

CONDITIONS = [
    ("whisker_trial",  "passive_pre"),
    ("whisker_trial",  "passive_post"),
    ("auditory_trial", "passive_pre"),
    ("auditory_trial", "passive_post"),
]
COND_LABELS        = ["Whisker pre", "Whisker post", "Auditory pre", "Auditory post"]
COND_LABELS_MATRIX = ["Whisker\npre", "Whisker\npost", "Auditory\npre", "Auditory\npost"]
COND_COLORS  = ["#32a852", "#085c1f", "#4158d9", "#0f2187"]


hostname = socket.gethostname()
if 'haas' in hostname:
    ROOT_PATH = '/mnt/share_internal'
else:
    ROOT_PATH = '/Petersen-Lab/share_internal'

MOUSE_INFO = os.path.join(ROOT_PATH,'Axel_Bisi_Share','dataset_info','joint_mouse_reference_weight.xlsx')

# ── spike helpers ──────────────────────────────────────────────────────────────

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
    i_hi   = np.searchsorted(spk, events + t_post, side="right")
    raster = [spk[i_lo[i]:i_hi[i]] - events[i] for i in range(len(events))]
    if is_whisker:
        raster = [_replace_artifact(r, t_pre, lo_a, hi_a, rng) for r in raster]
    return raster


def _bin_rates(raster, bins, dt):
    return np.vstack([np.histogram(r, bins=bins)[0] for r in raster]).astype(float) / dt


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
    i_hi = np.searchsorted(st, events + cfg["t_post"], side="right")
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
        rates = _bin_rates(raster, bins, dt)
        rates = rates.mean(axis=0)
        #rates = _causal_gaussian(rates.mean(axis=0), cfg["sigma_ms"] / cfg["bin_ms"])
        return rates

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


def build_feature_matrix(unit_ids, st_map, mouse_map, session_map, event_map, cfg):
    """(n_neurons, 4*n_bins) z-scored PSTH matrix, fully parallelised."""
    dt    = cfg["bin_ms"] / 1000
    bins  = np.arange(-cfg["t_pre"], cfg["t_post"] + dt, dt)
    t_ctr = bins[:-1] + dt / 2
    rows  = Parallel(n_jobs=cfg["n_jobs"], prefer="threads")(
        delayed(_neuron_vector)(
            st_map[uid], mouse_map[uid], session_map[uid],
            event_map, bins, t_ctr, t_ctr < 0, cfg,
        )
        for uid in unit_ids
    )
    return np.vstack(rows), t_ctr, len(t_ctr)


# ── rastermap ──────────────────────────────────────────────────────────────────

def fit_rastermap(X, n_clusters):
    n_pcs  = min(200, X.shape[0] - 1, X.shape[1] - 1)
    model  = Rastermap(n_clusters=n_clusters,
                       n_PCs=n_pcs,
                       locality=0.75,
                       time_lag_window=1,
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
    onset_bin = int(cfg["t_pre"] / (cfg["bin_ms"] / 1000))
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
    n      = min(cfg["n_sample_neurons"], len(unit_ids))
    rng    = np.random.default_rng(0)
    sample = rng.choice(len(unit_ids), size=n, replace=False)
    ncols  = 6
    nrows  = int(np.ceil(n / ncols))
    dt     = cfg["bin_ms"] / 1000
    bins   = np.arange(-cfg["t_pre"], cfg["t_post"] + dt, dt)
    base_mask = t_ctr < 0

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
            raster   = spikes_around_events(st, events, cfg["t_pre"], cfg["t_post"],
                                            is_whisker=(trial_type == cfg["whisker_trial_label"]),
                                            artifact_win_s=cfg["artifact_win_s"],
                                            rng=np.random.default_rng(c))
            rates    = _bin_rates(raster, bins, dt)
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


def fig5_population_matrix(X, n_bins, isort, boundaries, vmax, cfg, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=200)
    im1 = _draw_matrix(axes[0], X,        n_bins, [],         vmax, cfg,  f"Input order  (n={len(X)})")
    im2 = _draw_matrix(axes[1], X[isort], n_bins, boundaries, vmax, cfg, "Rastermap order")
    for ax, im in zip(axes, [im1, im2]):
        fig.colorbar(im, ax=ax, label="z-score", shrink=0.5, pad=0.02)
    fig.tight_layout()
    _save(fig, out_dir / "fig5_population_matrix.png", dpi=200)


def fig6_cluster_profiles(X, t_ctr, n_bins, isort, boundaries, out_dir):
    n_clusters = len(boundaries) + 1
    edges      = [0] + list(boundaries) + [len(isort)]
    ncols      = (n_clusters + 1) // 2
    nrows      = 2 if n_clusters > 1 else 1
    fig, axes  = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                               sharey=True, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for k in range(n_clusters):
        ax       = axes[k]
        idx      = isort[edges[k]:edges[k + 1]]
        mean_vec = X[idx].mean(0)
        for c, (label, color) in enumerate(zip(COND_LABELS, COND_COLORS)):
            ax.plot(t_ctr, mean_vec[c * n_bins:(c + 1) * n_bins],
                    color=color, lw=1.2, label=label)
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_title(f"C{k+1}  (n={len(idx)})", fontsize=8)
        if k % ncols == 0:
            ax.set_ylabel("z-score")
        if k >= (nrows - 1) * ncols:
            ax.set_xlabel("Time (s)")
        if k == 0:
            ax.legend(fontsize=6, loc="upper left")
    for ax in axes[n_clusters:]:
        ax.set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir / "fig6_cluster_profiles.png")


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(emb[:, 0], emb[:, 1], s=2, c="steelblue", alpha=0.4, linewidths=0)
    ax1.set_title("UMAP"); ax1.set_xlabel("UMAP 1"); ax1.set_ylabel("UMAP 2")
    sc = ax2.scatter(emb[:, 0], emb[:, 1], s=2, c=cluster_labels,
                     cmap="tab20", alpha=0.4, linewidths=0, vmin=0, vmax=n_clusters - 1)
    ax2.set_title("UMAP — rastermap clusters"); ax2.set_xlabel("UMAP 1")
    plt.colorbar(sc, ax=ax2, label="Rastermap cluster", ticks=range(n_clusters))
    fig.tight_layout()
    _save(fig, out_dir / "fig8_umap.png")


def fig9_kmeans(emb, km_labels, k, k_range, inertias, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    sc = ax1.scatter(emb[:, 0], emb[:, 1], s=2, c=km_labels,
                     cmap="tab10", alpha=0.4, linewidths=0, vmin=0, vmax=k - 1)
    ax1.set_title(f"UMAP — k-means  (k={k})")
    ax1.set_xlabel("UMAP 1"); ax1.set_ylabel("UMAP 2")
    plt.colorbar(sc, ax=ax1, label="k-means cluster")
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
        for c, (label, color) in enumerate(zip(COND_LABELS, COND_COLORS)):
            ax.plot(t_ctr, mean_vec[c * n_bins:(c + 1) * n_bins],
                    color=color, lw=1.2, label=label)
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_title(f"K{ki + 1}  (n={len(idx)})", fontsize=8)
        if ki % ncols == 0:
            ax.set_ylabel("z-score")
        if ki >= (nrows - 1) * ncols:
            ax.set_xlabel("Time (s)")
        if ki == 0:
            ax.legend(fontsize=6, loc="upper left")
    for ax in axes[k:]:
        ax.set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir / "fig10_kmeans_profiles.png")


# ── entry point ────────────────────────────────────────────────────────────────

def run_rastermap_psth(units: pd.DataFrame,
                       trials: pd.DataFrame,
                       out_root: str | Path = "rastermap_psth_out",
                       **cfg_overrides) -> dict:
    cfg      = {**DEFAULT_CFG, **cfg_overrides}
    out_root = Path(out_root, "passive_rastermap_psth")
    out_root.mkdir(parents=True, exist_ok=True)
    if cfg['zscore_full']==True:
        out_root = out_root / "zscore_full"
    else:
        out_root = out_root / "zscore"
    out_root.mkdir(parents=True, exist_ok=True)
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


    # Diagnostic figures
    fig0_data_summary(units_raw, units_good, unit_ids, trials, cfg, out_root)
    fig1_trial_counts(trials, cfg, out_root)
    fig2_fr_distribution(fr_map, unit_ids, cfg["fr_threshold_hz"], out_root)
    fig3_neuron_counts(units_raw, units_good, unit_ids, cfg, out_root)

    dt = cfg["bin_ms"] / 1000
    bins = np.arange(-cfg["t_pre"], cfg["t_post"] + dt, dt)
    t_ctr = bins[:-1] + dt / 2
    n_bins = len(t_ctr)

    fname = "feature_matrix.npy" if cfg["zscore_full"] else "feature_matrix_no_baseline.npy"
    fpath = out_root / fname

    #if fpath.exists():
    #    print(f"Loading feature matrix from {fname}...")
    #    X = np.load(fpath)
    #else:

    print(f"Building feature matrix ({len(unit_ids)} neurons × 4 conditions)...")
    X, t_ctr, n_bins = build_feature_matrix(
        unit_ids, st_map, mouse_map, session_map, event_map, cfg)
    print(f"  X shape: {X.shape}")
    np.save(fpath, X)

    fig4_sample_neurons(unit_ids, st_map, mouse_map, session_map, event_map, t_ctr, cfg, out_root)

    n_k = cfg["n_rastermap_clusters"]
    print(f"Running rastermap (n_clusters={n_k})...")
    isort, boundaries = fit_rastermap(X, n_k)

    cluster_labels = np.empty(len(unit_ids), dtype=int)
    edges = [0] + list(boundaries) + [len(isort)]
    for k in range(n_k):
        cluster_labels[isort[edges[k]:edges[k + 1]]] = k

    vmax = np.nanpercentile(np.abs(X), cfg["vmax_pct"])
    fig5_population_matrix(X, n_bins, isort, boundaries, vmax, cfg, out_root)
    fig6_cluster_profiles(X, t_ctr, n_bins, isort, boundaries, out_root)

    n_pca = min(150, len(unit_ids) - 1, X.shape[1] - 1)
    print(f"Running PCA ({n_pca} components) + UMAP...")
    X_pca = PCA(n_components=n_pca).fit_transform(X)
    fig7_pca_variance(X, n_pca, out_root)
    emb = umap.UMAP(n_neighbors=cfg["umap_n_neighbors"],
                    min_dist=cfg["umap_min_dist"],
                    n_components=2,
                    random_state=42).fit_transform(X)
    fig8_umap(emb, cluster_labels, n_k, out_root)

    k = cfg["k_means_k"]
    print(f"Running k-means (k={k}) + elbow...")
    km_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    inertias  = Parallel(n_jobs=cfg["n_jobs"])(
        delayed(_kmeans_inertia)(X, ki) for ki in cfg["k_elbow_range"]
    )
    fig9_kmeans(emb, km_labels, k, cfg["k_elbow_range"], inertias, out_root)

    # Now plot k-means clusters
    fig10_kmeans_profiles(X, t_ctr, n_bins, km_labels, k, out_root)

    print(f"\nDone. Outputs → {out_root}")
    return dict(
        X=X, t_ctr=t_ctr, n_bins=n_bins, unit_ids=unit_ids,
        isort=isort, boundaries=boundaries, cluster_labels=cluster_labels,
        umap_embedding=emb, km_labels=km_labels,
    )
