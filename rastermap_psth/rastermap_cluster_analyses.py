"""
cluster_postproc_analyses.py — Post-hoc analyses on a rastermap_psth cluster output.

Loads a completed rastermap_psth run (neuron_cluster_labels_cv.csv +
neuron_psth_by_condition.npz) and cross-references it against a unit_table
(ROC selectivity results) and a trial_table (behavioral d-prime), independent
of the original Rastermap fitting step.

Four toggleable analyses, each producing intermediate/diagnostic outputs plus
a summary panel, and a final combined summary figure across all four.

    1. Per-neuron / per-cluster response latency (threshold-crossing)
    2. Meta-clustering of cluster-mean PSTHs (hierarchical, 1-correlation dist)
    3. ROC cross-reference (per analysis_type, % significant + direction)
    4. Behavioral d-prime vs. per-cluster fractional representation (F_mk)

Usage
-----
    Called programmatically, the same way run_rastermap_psth(units, trials, ...)
    is called elsewhere in the pipeline — unit_table and trial_table are
    passed in as already-loaded DataFrames, not file paths:

        from cluster_postproc_analyses import run_all

        run_all(
            rastermap_dir = "/path/to/rastermap_output_folder",
            unit_table    = unit_table,     # in-memory DataFrame
            trial_table   = trial_table,    # in-memory DataFrame
            out_root      = None,           # default: rastermap_dir
            run_analyses  = None,           # default: all four analyses
        )

Inputs expected inside rastermap_dir
---------------------------------------
    neuron_cluster_labels_cv.csv   (unit_id, cluster_label_cv, km_label, cluster_id,
                                     mouse_id, session_id, electrode_group,
                                     reward_group, area_acronym, waveform_type,
                                     layer_number, [avg_ipsi, cc_tc_ct_iterated,
                                     cc_hierarchy_score])
    neuron_psth_by_condition.npz   (unit_ids, cond_labels, psth__<tag>, t_ctr__<tag>)

unit_table columns expected (long format, one row per unit x analysis_type)
-----------------------------------------------------------------------------
    mouse_id, session_id, electrode_group, cluster_id,
    analysis_type   (whisker_active | auditory_active | choice | spontaneous_licks |
                     whisker_vs_aud)
    significant     (bool)
    direction       (positive | negative,  or  whisker | auditory  for whisker_vs_aud)

trial_table columns expected
------------------------------
    mouse_id, session_id, perf, context, trial_type, lick_flag
    trial_type values used: "whisker_trial" (hits), "no_stim_trial" (false alarms)
    excluded: perf == 6, context == "passive"

Output
------
    <out_root>/analyses/analyses_<n>/
        01_latency/            (if run)
        02_meta_clustering/    (if run)
        03_roc/                (if run)
        04_dprime/              (if run)
        summary_figure.png/.pdf/.svg
        run_config.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, norm as scipy_norm
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist


# ══════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════

DEFAULT_CFG: dict[str, Any] = dict(
    # ── which analyses to run ──────────────────────────────────────────
    run_latency         = True,
    run_meta_clustering = True,
    run_roc             = False,
    run_dprime          = True,

    # ── 1. latency ──────────────────────────────────────────────────────
    latency_window_s    = (0.0, 0.5),     # post start_time search window
    latency_threshold_frac = 0.5,          # fraction of peak for crossing
    # Run latency separately per condition below (tag -> exact cond_labels string).
    # One full set of latency outputs is produced per entry.
    latency_conditions = {"whisker_hit": "Whisker hit", "auditory_hit": "Auditory hit"},
    latency_n_example_neurons = 12,

    # ── 2. meta-clustering ──────────────────────────────────────────────
    meta_linkage_method  = "average",      # linkage on 1-corr distance (Pearson correlation)
    meta_dendro_cut_dist = 0.35,           # distance threshold -> families (1-corr; 0.35 => r>=0.65)

    # ── 3. ROC cross-reference ───────────────────────────────────────────
    roc_analysis_types   = ["whisker_active", "auditory_active", "choice",
                             "spontaneous_licks", "whisker_vs_aud"],

    # ── 4. d-prime ────────────────────────────────────────────────────────
    dprime_exclude_perf   = 6,
    dprime_exclude_context = "passive",
    dprime_hit_trial_type  = "whisker_trial",
    dprime_fa_trial_type   = "no_stim_trial",
    dprime_min_trials       = 5,           # min trials per rate to trust a mouse

    # ── join keys ─────────────────────────────────────────────────────────
    join_keys = ["mouse_id", "session_id", "electrode_group", "cluster_id"],

    fdr_alpha = 0.05,
)


# ══════════════════════════════════════════════════════════════════════════
# Small shared helpers
# ══════════════════════════════════════════════════════════════════════════

def _save(fig, path: Path, dpi=300):
    for fmt in [".png", ".pdf", ".svg"]:
        fig.savefig(path.with_suffix(fmt), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def _bh_correction(pvals: np.ndarray, alpha: float = 0.05):
    """Benjamini-Hochberg FDR correction. Returns (pvals_adj, reject)."""
    pvals = np.asarray(pvals, dtype=float)
    n     = len(pvals)
    if n == 0:
        return pvals, np.array([], dtype=bool)
    order = np.argsort(pvals)
    rank  = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    pvals_adj = np.minimum(1.0, pvals * n / rank)
    sorted_adj = pvals_adj[order]
    for i in range(n - 2, -1, -1):
        sorted_adj[i] = min(sorted_adj[i], sorted_adj[i + 1])
    pvals_adj[order] = sorted_adj
    reject = pvals_adj < alpha
    return pvals_adj, reject


def _sanitize_cond_label(label: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_").lower()


def make_run_dir(out_root: Path) -> Path:
    analyses_root = out_root / "analyses"
    analyses_root.mkdir(parents=True, exist_ok=True)
    existing = [p for p in analyses_root.iterdir()
                if p.is_dir() and re.match(r"analyses_\d+$", p.name)]
    nums = [int(p.name.split("_")[1]) for p in existing] or [0]
    run_dir = analyses_root / f"analyses_{max(nums) + 1}"
    run_dir.mkdir(parents=True)
    return run_dir


# ══════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════

def load_cluster_table(rastermap_dir: Path) -> pd.DataFrame:
    path = rastermap_dir / "neuron_cluster_labels_cv.csv"
    if not path.exists():
        # fall back to the non-CV table if cross_validate=False was used upstream
        alt = rastermap_dir / "neuron_cluster_labels.csv"
        if alt.exists():
            print(f"  [warn] {path.name} not found, falling back to {alt.name} "
                  f"(cluster_label used as cluster_label_cv)")
            df = pd.read_csv(alt)
            df = df.rename(columns={"cluster_label": "cluster_label_cv"})
            return df
        raise FileNotFoundError(f"Neither neuron_cluster_labels_cv.csv nor "
                                 f"neuron_cluster_labels.csv found in {rastermap_dir}")
    df = pd.read_csv(path)
    print(f"  Loaded cluster table: {len(df)} neurons, "
          f"{df['cluster_label_cv'].nunique()} clusters")
    return df


def load_psth_npz(rastermap_dir: Path):
    path = rastermap_dir / "neuron_psth_by_condition.npz"
    if not path.exists():
        raise FileNotFoundError(f"neuron_psth_by_condition.npz not found in {rastermap_dir}")
    data = np.load(path, allow_pickle=True)
    unit_ids    = data["unit_ids"]
    cond_labels = [str(c) for c in data["cond_labels"]]
    psth = {}
    t_ctr = {}
    for label in cond_labels:
        tag = _sanitize_cond_label(label)
        psth[label]  = data[f"psth__{tag}"]
        t_ctr[label] = data[f"t_ctr__{tag}"]
    print(f"  Loaded PSTHs: {len(unit_ids)} neurons x {len(cond_labels)} conditions "
          f"({', '.join(cond_labels)})")
    return unit_ids, cond_labels, psth, t_ctr


# ══════════════════════════════════════════════════════════════════════════
# 1. Latency
# ══════════════════════════════════════════════════════════════════════════

def _threshold_crossing_latency(trace, t_ctr, window_s, thresh_frac):
    """Return (latency_s, sign, peak_val) for one neuron, one condition.

    sign: +1 excited, -1 suppressed. Latency is NaN if no crossing found.
    """
    lo, hi = window_s
    win_mask = (t_ctr >= lo) & (t_ctr < hi)
    if win_mask.sum() == 0:
        return np.nan, 0, np.nan

    seg   = trace[win_mask]
    t_seg = t_ctr[win_mask]
    pos_peak = np.nanmax(seg)
    neg_peak = np.nanmin(seg)

    if abs(neg_peak) > abs(pos_peak):
        sign = -1
        peak = neg_peak
        signed_seg = -seg
        signed_peak = -peak
    else:
        sign = 1
        peak = pos_peak
        signed_seg = seg
        signed_peak = peak

    if signed_peak <= 0 or not np.isfinite(signed_peak):
        return np.nan, sign, peak

    thresh = thresh_frac * signed_peak
    above  = np.where(signed_seg >= thresh)[0]
    if len(above) == 0:
        return np.nan, sign, peak
    latency = t_seg[above[0]]
    return float(latency), sign, float(peak)


def _plot_latency_pointplot(cluster_lat, out_dir, tag, cond_label):
    """Two versions of the per-cluster mean-latency pointplot (SEM error bars,
    larger dots):
      (a) latency_vs_cluster_position — y = cluster_label_cv, matrix order
          (inverted y-axis so cluster 0 sits at top, matching the population
          matrix / rastermap figure convention).
      (b) latency_vs_latency_order — same plot but clusters sorted by their
          own mean latency, so the range/ordering of response timing is
          immediately visible.
    """
    # (a) aligned to matrix (cluster_label_cv) order
    df1 = cluster_lat.sort_values("cluster_label_cv")
    fig, ax = plt.subplots(figsize=(5, max(4, len(df1) * 0.18)))
    ax.errorbar(df1["mean"] * 1000, df1["cluster_label_cv"], xerr=df1["sem"] * 1000,
                fmt="o", ms=9, mec="white", mew=0.6, color="darkorange",
                ecolor="darkorange", elinewidth=1.5, capsize=3, zorder=3)
    ax.invert_yaxis()   # cluster 0 at top, matches population-matrix orientation
    ax.set_xlabel("Mean latency (ms) ± SEM")
    ax.set_ylabel("cluster_label_cv")
    ax.set_title(f"{cond_label}: latency vs. cluster position\n(aligned to matrix order)")
    fig.tight_layout()
    _save(fig, out_dir / f"latency_vs_cluster_position_{tag}", dpi=300)

    # (b) sorted by latency value
    df2 = cluster_lat.sort_values("mean").reset_index(drop=True)
    y_pos = np.arange(len(df2))
    fig, ax = plt.subplots(figsize=(5, max(4, len(df2) * 0.18)))
    ax.errorbar(df2["mean"] * 1000, y_pos, xerr=df2["sem"] * 1000,
                fmt="o", ms=9, mec="white", mew=0.6, color="darkorange",
                ecolor="darkorange", elinewidth=1.5, capsize=3, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df2["cluster_label_cv"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean latency (ms) ± SEM")
    ax.set_ylabel("cluster_label_cv (sorted by latency)")
    ax.set_title(f"{cond_label}: clusters sorted by latency")
    fig.tight_layout()
    _save(fig, out_dir / f"latency_vs_latency_order_{tag}", dpi=300)


def run_latency_analysis(cluster_df, psth, t_ctr, unit_ids, cfg, out_dir, tag, cond_label):
    """Latency analysis for a single condition (e.g. 'Whisker hit'). Called
    once per entry in cfg['latency_conditions'] — each call is fully
    self-contained and writes to its own out_dir."""
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"\n[1] Latency analysis — {cond_label} ({tag})")

    if cond_label not in psth:
        print(f"  [warn] condition '{cond_label}' not found in PSTH file — skipping")
        return None

    uid_to_row = {uid: i for i, uid in enumerate(unit_ids)}
    window     = cfg["latency_window_s"]
    thresh     = cfg["latency_threshold_frac"]
    mat        = psth[cond_label]
    t_c        = t_ctr[cond_label]

    per_neuron_rows = []
    for uid in cluster_df["unit_id"]:
        row_idx = uid_to_row.get(uid)
        if row_idx is None:
            continue
        trace = mat[row_idx]
        lat, sign, peak = _threshold_crossing_latency(trace, t_c, window, thresh)
        per_neuron_rows.append(dict(unit_id=uid, condition=cond_label,
                                     latency_s=lat, sign=sign, peak=peak))
    lat_df = pd.DataFrame(per_neuron_rows)
    lat_df = lat_df.rename(columns={"latency_s": "latency_s_mean"})
    lat_df.to_csv(out_dir / f"latency_per_neuron_{tag}.csv", index=False)

    merged = cluster_df.merge(lat_df[["unit_id", "latency_s_mean", "sign", "peak"]],
                              on="unit_id", how="left")

    # per-cluster aggregation — mean ± SEM (primary) plus median/IQR (kept for reference)
    def _sem(x):
        x = x.dropna()
        return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0

    cluster_lat = (merged.groupby("cluster_label_cv")["latency_s_mean"]
                          .agg(mean="mean",
                               sem=_sem,
                               median="median",
                               q25=lambda x: x.quantile(0.25),
                               q75=lambda x: x.quantile(0.75),
                               n_valid=lambda x: x.notna().sum(),
                               n_total="size")
                          .reset_index()
                          .sort_values("cluster_label_cv"))
    cluster_lat.to_csv(out_dir / f"latency_per_cluster_{tag}.csv", index=False)

    # ── quality/intermediate diagnostics ──────────────────────────────────
    frac_missing = merged["latency_s_mean"].isna().mean()
    with open(out_dir / f"latency_diagnostics_{tag}.txt", "w") as fh:
        fh.write(f"Condition: {cond_label}\n")
        fh.write(f"Window: {window[0]*1000:.0f}-{window[1]*1000:.0f} ms\n")
        fh.write(f"Threshold: {thresh*100:.0f}% of peak\n")
        fh.write(f"Fraction with no threshold crossing: {frac_missing:.3f}\n")
        fh.write(f"Suppressed (sign=-1) fraction among valid: "
                 f"{(merged['sign']==-1).mean():.3f}\n")

    # (a) histogram of per-neuron latencies
    fig, ax = plt.subplots(figsize=(5, 3.5))
    vals = merged["latency_s_mean"].dropna() * 1000
    ax.hist(vals, bins=40, color="steelblue", edgecolor="none")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Neuron count")
    ax.set_title(f"{cond_label}: per-neuron latency distribution (n={len(vals)})")
    fig.tight_layout()
    _save(fig, out_dir / f"qc_latency_histogram_{tag}", dpi=300)

    # (b) example neuron traces with detected crossing marked
    n_ex = min(cfg["latency_n_example_neurons"], len(unit_ids))
    rng  = np.random.default_rng(0)
    ex_uids  = rng.choice(cluster_df["unit_id"].values, size=n_ex, replace=False)
    ncols = 4
    nrows = int(np.ceil(n_ex / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for i, uid in enumerate(ex_uids):
        ax = axes[i]
        row_idx = uid_to_row.get(uid)
        if row_idx is None:
            ax.set_visible(False)
            continue
        trace = mat[row_idx]
        lat, sign, peak = _threshold_crossing_latency(trace, t_c, window, thresh)
        ax.plot(t_c, trace, color="k", lw=1)
        ax.axvspan(window[0], window[1], color="grey", alpha=0.1)
        if np.isfinite(lat):
            ax.axvline(lat, color="crimson", lw=1.2, ls="--")
            ax.scatter([lat], [thresh * peak], color="crimson", zorder=5, s=20)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"uid={uid}  lat={lat*1000:.0f}ms" if np.isfinite(lat) else f"uid={uid}  no crossing",
                     fontsize=7)
    for ax in axes[n_ex:]:
        ax.set_visible(False)
    fig.suptitle(f"Example latency detections — {cond_label}", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / f"qc_example_traces_{tag}", dpi=300)

    # ── summary figures ─────────────────────────────────────────────────
    _plot_latency_pointplot(cluster_lat, out_dir, tag, cond_label)

    print(f"  Done. {merged['latency_s_mean'].notna().sum()}/{len(merged)} neurons "
          f"with a valid latency.")
    return cluster_lat


# ══════════════════════════════════════════════════════════════════════════
# 2. Meta-clustering
# ══════════════════════════════════════════════════════════════════════════

def run_meta_clustering(cluster_df, cond_labels, psth, unit_ids, cfg, out_dir):
    out_dir.mkdir(exist_ok=True)
    print("\n[2] Meta-clustering (cluster-of-clusters)")

    uid_to_row = {uid: i for i, uid in enumerate(unit_ids)}
    # build full concatenated feature vector per neuron, in cond_labels order
    row_idx_for_uid = np.array([uid_to_row.get(uid, -1) for uid in cluster_df["unit_id"]])
    valid = row_idx_for_uid >= 0
    if not valid.all():
        print(f"  [warn] {(~valid).sum()} cluster-table neurons not found in PSTH file, dropping")
    sub_cluster_df = cluster_df[valid].reset_index(drop=True)
    row_idx_for_uid = row_idx_for_uid[valid]

    X_full = np.concatenate([psth[label][row_idx_for_uid] for label in cond_labels], axis=1)

    clusters = np.sort(sub_cluster_df["cluster_label_cv"].unique())
    cluster_means = np.vstack([
        X_full[sub_cluster_df["cluster_label_cv"].values == c].mean(axis=0)
        for c in clusters
    ])

    # pairwise correlation -> distance
    corr_mat = np.corrcoef(cluster_means)
    dist_mat = 1 - corr_mat
    np.fill_diagonal(dist_mat, 0.0)
    dist_mat = np.clip(dist_mat, 0, None)  # guard tiny negative floating noise
    condensed = squareform(dist_mat, checks=False)

    Z = linkage(condensed, method=cfg["meta_linkage_method"])
    coph_corr, coph_dists = cophenet(Z, condensed)
    print(f"  Cophenetic correlation coefficient: {coph_corr:.3f} "
          f"(closer to 1 = dendrogram faithfully represents pairwise distances)")

    family_labels = fcluster(Z, t=cfg["meta_dendro_cut_dist"], criterion="distance")
    families_df = pd.DataFrame({"cluster_label_cv": clusters, "family": family_labels})
    families_df.to_csv(out_dir / "cluster_families.csv", index=False)

    family_sizes = families_df["family"].value_counts().sort_index()
    n_singletons = (family_sizes == 1).sum()
    print(f"  {family_labels.max()} families found at cut distance "
          f"{cfg['meta_dendro_cut_dist']} ({n_singletons} singleton families)")

    # ── quality/intermediate outputs ────────────────────────────────────────
    pd.DataFrame(corr_mat, index=clusters, columns=clusters).to_csv(
        out_dir / "cluster_correlation_matrix.csv")

    with open(out_dir / "meta_clustering_diagnostics.txt", "w") as fh:
        fh.write(f"Similarity measure: Pearson correlation (distance = 1 - r)\n")
        fh.write(f"Linkage method: {cfg['meta_linkage_method']}\n")
        fh.write(f"Cut distance: {cfg['meta_dendro_cut_dist']}\n")
        fh.write(f"Cophenetic correlation coefficient: {coph_corr:.4f}\n")
        fh.write(f"Number of families: {family_labels.max()}\n")
        fh.write(f"Singleton families: {n_singletons}\n")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(family_sizes.index.astype(str), family_sizes.values, color="steelblue")
    ax.set_xlabel("Family")
    ax.set_ylabel("Number of clusters")
    ax.set_title(f"Family sizes (n={family_labels.max()} families,\n"
                 f"{n_singletons} singletons)")
    fig.tight_layout()
    _save(fig, out_dir / "qc_family_sizes", dpi=300)

    # ── STEP 1: raw input — cluster-mean PSTH matrix, no clustering applied yet ──
    cond_lengths = [psth[label].shape[1] for label in cond_labels]
    offsets      = np.concatenate([[0], np.cumsum(cond_lengths)])
    tick_pos     = [(offsets[i] + offsets[i + 1]) / 2 for i in range(len(cond_labels))]

    def _draw_cluster_matrix(ax, matrix, row_labels, title):
        vmax = np.nanpercentile(np.abs(matrix), 95)
        vmax = vmax if vmax > 0 else 1.0
        im = ax.imshow(matrix, aspect="auto", interpolation="none", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, extent=[0, offsets[-1], len(row_labels), 0])
        for start in offsets[1:-1]:
            ax.axvline(start, color="k", lw=0.8)
        ax.set_yticks(np.arange(len(row_labels)) + 0.5)
        ax.set_yticklabels(row_labels, fontsize=6)
        ax.set_ylabel("cluster_label_cv")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(cond_labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(title)
        return im

    fig, ax = plt.subplots(figsize=(9, max(6, len(clusters) * 0.15)))
    im = _draw_cluster_matrix(ax, cluster_means, clusters,
                              "STEP 1 — Input: cluster-mean PSTH matrix\n"
                              "(cluster_label_cv order, before any clustering)")
    fig.colorbar(im, ax=ax, label="Firing rate (z-score)", shrink=0.6, pad=0.02)
    fig.tight_layout()
    _save(fig, out_dir / "qc_input_matrix_original_order", dpi=300)

    # ── STEP 2: pairwise similarity (correlation matrix) ─────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("STEP 2 — Pairwise similarity\n(Pearson correlation, original order)")
    ax.set_xlabel("cluster_label_cv"); ax.set_ylabel("cluster_label_cv")
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    fig.tight_layout()
    _save(fig, out_dir / "qc_correlation_matrix_original_order", dpi=300)

    # ── STEP 3: dendrogram (from linkage on 1-r distance) next to the ────
    # INPUT matrix, in its ORIGINAL cluster_label_cv order. The tree's leaves
    # are NOT in this order, so rows here are NOT vertically aligned with the
    # dendrogram — this view is for comparing "before" layout against the
    # tree structure, not for reading off which family a given row belongs to.
    def _draw_dendrogram_panel(ax, row_labels_for_ticks):
        dn = dendrogram(Z, orientation="left", labels=[str(c) for c in row_labels_for_ticks],
                        color_threshold=cfg["meta_dendro_cut_dist"], ax=ax)
        ax.axvline(cfg["meta_dendro_cut_dist"], color="k", ls="--", lw=0.8)
        ax.set_xlabel("1 − correlation distance")
        ax.set_title("Dendrogram")
        return dn

    fig = plt.figure(figsize=(12, max(6, len(clusters) * 0.18)))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 3], wspace=0.05)
    ax_dendro = fig.add_subplot(gs[0])
    ax_heat   = fig.add_subplot(gs[1])
    _draw_dendrogram_panel(ax_dendro, clusters)
    im = _draw_cluster_matrix(ax_heat, cluster_means, clusters,
                              "STEP 3 — Dendrogram next to INPUT matrix\n"
                              "(original order — rows NOT aligned with tree leaves)")
    fig.colorbar(im, ax=ax_heat, label="Firing rate (z-score)", shrink=0.6, pad=0.02)
    fig.suptitle(f"Meta-clustering: dendrogram vs. original-order input "
                 f"(cophenetic r={coph_corr:.2f})", fontsize=10)
    _save(fig, out_dir / "meta_clustering_dendrogram_matrix_original_order", dpi=300)

    # ── STEP 4: dendrogram next to the matrix REORDERED by leaf order — ──
    # now rows ARE aligned with the tree, so a family in the dendrogram
    # visibly corresponds to a contiguous block of similarly-shaped rows.
    leaf_order_dn = dendrogram(Z, no_plot=True)
    leaf_order = leaf_order_dn["leaves"]
    reordered_clusters = clusters[leaf_order]
    reordered_means = cluster_means[leaf_order]

    fig = plt.figure(figsize=(12, max(6, len(clusters) * 0.18)))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 3], wspace=0.05)
    ax_dendro = fig.add_subplot(gs[0])
    ax_heat   = fig.add_subplot(gs[1])
    _draw_dendrogram_panel(ax_dendro, clusters)
    im = _draw_cluster_matrix(ax_heat, reordered_means, reordered_clusters,
                              "STEP 4 — Dendrogram next to REORDERED matrix\n"
                              "(rows aligned to dendrogram leaf order)")
    fig.colorbar(im, ax=ax_heat, label="Firing rate (z-score)", shrink=0.6, pad=0.02)
    fig.suptitle(f"Meta-clustering of cluster-mean PSTHs "
                 f"(cophenetic r={coph_corr:.2f})", fontsize=10)
    _save(fig, out_dir / "meta_clustering_dendrogram_matrix_reordered", dpi=300)

    print(f"  Done. Families saved to cluster_families.csv")
    return families_df


# ══════════════════════════════════════════════════════════════════════════
# 3. ROC cross-reference
# ══════════════════════════════════════════════════════════════════════════

def run_roc_analysis(cluster_df, unit_table, cfg, out_dir):
    """unit_table : already-loaded DataFrame (long format, one row per unit x analysis_type)."""
    out_dir.mkdir(exist_ok=True)
    print("\n[3] ROC cross-reference")

    unit_table = unit_table.copy()
    join_keys  = cfg["join_keys"]
    missing_cols = [c for c in join_keys + ["analysis_type", "significant", "direction"]
                    if c not in unit_table.columns]
    if missing_cols:
        raise KeyError(f"unit_table is missing expected column(s): {missing_cols}. "
                        f"Available columns: {list(unit_table.columns)}")

    for c in join_keys:
        if c in cluster_df.columns:
            cluster_df[c] = cluster_df[c].astype(str)
        unit_table[c] = unit_table[c].astype(str)

    merged = cluster_df.merge(unit_table[join_keys + ["analysis_type", "significant", "direction"]],
                               on=join_keys, how="left")
    merged.to_csv(out_dir / "cluster_roc_merged.csv", index=False)

    # ── merge diagnostics ──────────────────────────────────────────────────
    n_neurons = cluster_df["unit_id"].nunique()
    n_matched = merged.dropna(subset=["analysis_type"])["unit_id"].nunique()
    print(f"  Merge: {n_matched}/{n_neurons} neurons matched at least one ROC row in unit_table")

    with open(out_dir / "roc_merge_diagnostics.txt", "w") as fh:
        fh.write(f"Total neurons in cluster table: {n_neurons}\n")
        fh.write(f"Neurons with >=1 matched ROC row: {n_matched}\n")
        fh.write(f"Unmatched: {n_neurons - n_matched}\n\n")
        for at in cfg["roc_analysis_types"]:
            sub = merged[merged["analysis_type"] == at]
            fh.write(f"{at}: {sub['unit_id'].nunique()} matched neurons, "
                     f"baseline significance rate = "
                     f"{sub['significant'].mean() if len(sub) else float('nan'):.3f}\n")

    # ── per-cluster % significant + direction, per analysis_type ──────────
    summary_rows = []
    for at in cfg["roc_analysis_types"]:
        sub = merged[merged["analysis_type"] == at]
        if sub.empty:
            print(f"  [warn] analysis_type '{at}' not found in unit_table — skipping")
            continue
        for c, grp in sub.groupby("cluster_label_cv"):
            n_total = len(grp)
            n_sig   = grp["significant"].sum()
            pct_sig = n_sig / n_total if n_total > 0 else np.nan
            sig_dirs = grp.loc[grp["significant"], "direction"]
            dir_counts = sig_dirs.value_counts(normalize=True).to_dict()
            summary_rows.append(dict(cluster_label_cv=c, analysis_type=at,
                                      n_total=n_total, n_significant=n_sig,
                                      pct_significant=pct_sig,
                                      **{f"dir_frac_{k}": v for k, v in dir_counts.items()}))
    roc_summary = pd.DataFrame(summary_rows)
    roc_summary.to_csv(out_dir / "roc_summary_per_cluster.csv", index=False)

    # ── quality output: population baseline vs per-cluster spread ─────────
    fig, axes = plt.subplots(1, len(cfg["roc_analysis_types"]), figsize=(3.2 * len(cfg["roc_analysis_types"]), 3.5),
                              sharey=False)
    axes = np.atleast_1d(axes)
    for ax, at in zip(axes, cfg["roc_analysis_types"]):
        sub = roc_summary[roc_summary["analysis_type"] == at]
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.hist(sub["pct_significant"].dropna(), bins=15, color="steelblue", edgecolor="none")
        baseline = merged[merged["analysis_type"] == at]["significant"].mean()
        ax.axvline(baseline, color="crimson", ls="--", lw=1, label=f"pop. baseline={baseline:.2f}")
        ax.set_title(at, fontsize=8)
        ax.set_xlabel("% significant")
        ax.legend(fontsize=6)
    fig.suptitle("Per-cluster % significant vs. population baseline, by analysis_type", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / "qc_pct_significant_distributions", dpi=300)

    # ── summary figure: % significant per cluster, one panel per analysis_type ──
    n_at = roc_summary["analysis_type"].nunique()
    fig, axes = plt.subplots(1, n_at, figsize=(2.4 * n_at, max(4, cluster_df['cluster_label_cv'].nunique() * 0.15)),
                              sharey=True)
    axes = np.atleast_1d(axes)
    for ax, at in zip(axes, roc_summary["analysis_type"].unique()):
        sub = roc_summary[roc_summary["analysis_type"] == at].sort_values("cluster_label_cv")
        ax.scatter(sub["pct_significant"], sub["cluster_label_cv"], s=10, color="darkorange")
        ax.set_title(at, fontsize=8)
        ax.set_xlabel("% sig.")
    axes[0].set_ylabel("cluster_label_cv")
    fig.suptitle("Cluster ROC selectivity summary", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / "roc_summary_figure", dpi=300)

    print(f"  Done. Summary saved to roc_summary_per_cluster.csv")
    return roc_summary


# ══════════════════════════════════════════════════════════════════════════
# 4. d-prime
# ══════════════════════════════════════════════════════════════════════════

def _corrected_rate(n_events, n_trials):
    """Log-linear (Hautus) correction: add 0.5 event and 1 trial."""
    if n_trials == 0:
        return np.nan
    return (n_events + 0.5) / (n_trials + 1)


def run_dprime_analysis(cluster_df, trial_table, cfg, out_dir):
    """trial_table : already-loaded DataFrame."""
    out_dir.mkdir(exist_ok=True)
    print("\n[4] d-prime vs. cluster occupancy")

    trials = trial_table.copy()
    required = ["mouse_id", "session_id", "perf", "context", "trial_type", "lick_flag"]
    missing = [c for c in required if c not in trials.columns]
    if missing:
        raise KeyError(f"trial_table is missing expected column(s): {missing}. "
                        f"Available columns: {list(trials.columns)}")

    filt = trials[(trials["perf"] != cfg["dprime_exclude_perf"]) &
                   (trials["context"] != cfg["dprime_exclude_context"])]

    dprime_rows = []
    for mouse_id, grp in filt.groupby("mouse_id"):
        hit_trials = grp[grp["trial_type"] == cfg["dprime_hit_trial_type"]]
        fa_trials  = grp[grp["trial_type"] == cfg["dprime_fa_trial_type"]]
        n_hit_trials, n_fa_trials = len(hit_trials), len(fa_trials)
        n_hits = hit_trials["lick_flag"].sum()
        n_fas  = fa_trials["lick_flag"].sum()

        hit_rate_raw = n_hits / n_hit_trials if n_hit_trials > 0 else np.nan
        fa_rate_raw  = n_fas / n_fa_trials if n_fa_trials > 0 else np.nan

        hit_rate_c = _corrected_rate(n_hits, n_hit_trials)
        fa_rate_c  = _corrected_rate(n_fas, n_fa_trials)

        if np.isnan(hit_rate_c) or np.isnan(fa_rate_c):
            dprime = np.nan
        else:
            dprime = scipy_norm.ppf(hit_rate_c) - scipy_norm.ppf(fa_rate_c)

        dprime_rows.append(dict(mouse_id=mouse_id, n_hit_trials=n_hit_trials,
                                 n_fa_trials=n_fa_trials, n_hits=n_hits, n_fas=n_fas,
                                 hit_rate_raw=hit_rate_raw, fa_rate_raw=fa_rate_raw,
                                 hit_rate_corrected=hit_rate_c, fa_rate_corrected=fa_rate_c,
                                 dprime=dprime))
    dprime_df = pd.DataFrame(dprime_rows)
    dprime_df.to_csv(out_dir / "dprime_per_mouse.csv", index=False)

    low_trial_mice = dprime_df[(dprime_df["n_hit_trials"] < cfg["dprime_min_trials"]) |
                                (dprime_df["n_fa_trials"] < cfg["dprime_min_trials"])]
    if len(low_trial_mice):
        print(f"  [warn] {len(low_trial_mice)} mice have <{cfg['dprime_min_trials']} trials "
              f"for hit or FA rate — d' may be unreliable for these mice: "
              f"{list(low_trial_mice['mouse_id'])}")

    # ── quality/intermediate outputs ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].hist(dprime_df["dprime"].dropna(), bins=20, color="steelblue", edgecolor="none")
    axes[0].set_xlabel("d'"); axes[0].set_ylabel("Mouse count")
    axes[0].set_title("d' distribution across mice")
    axes[1].scatter(dprime_df["hit_rate_raw"], dprime_df["fa_rate_raw"],
                     c=dprime_df["dprime"], cmap="viridis", s=30)
    axes[1].set_xlabel("Hit rate"); axes[1].set_ylabel("FA rate")
    axes[1].set_title("Hit vs FA rate (colored by d')")
    fig.tight_layout()
    _save(fig, out_dir / "qc_dprime_distributions", dpi=300)

    # ── F_mk: per-mouse fractional representation per cluster ──────────────
    cluster_df = cluster_df.copy()
    cluster_df["mouse_id"] = cluster_df["mouse_id"].astype(str)
    dprime_df["mouse_id"]  = dprime_df["mouse_id"].astype(str)

    ct = pd.crosstab(cluster_df["mouse_id"], cluster_df["cluster_label_cv"])
    f_mk = ct.div(ct.sum(axis=1), axis=0)   # fraction per mouse

    common_mice = [m for m in f_mk.index if m in set(dprime_df["mouse_id"])]
    f_mk_common = f_mk.loc[common_mice]
    dprime_common = dprime_df.set_index("mouse_id").loc[common_mice, "dprime"]

    print(f"  {len(common_mice)} mice with both cluster occupancy and valid d' data")

    corr_rows = []
    for cluster in f_mk_common.columns:
        x = f_mk_common[cluster].values
        y = dprime_common.values
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 3:
            rho, p = np.nan, np.nan
        else:
            rho, p = spearmanr(x[valid], y[valid])
        corr_rows.append(dict(cluster_label_cv=cluster, rho=rho, p_raw=p, n_mice=valid.sum()))
    corr_df = pd.DataFrame(corr_rows).sort_values("cluster_label_cv")
    pvals = corr_df["p_raw"].fillna(1.0).values
    p_fdr, reject = _bh_correction(pvals, alpha=cfg["fdr_alpha"])
    corr_df["p_fdr"] = p_fdr
    corr_df["significant"] = reject
    corr_df.to_csv(out_dir / "dprime_fmk_correlation_per_cluster.csv", index=False)

    n_sig = reject.sum()
    print(f"  {n_sig}/{len(corr_df)} clusters show significant d'-occupancy correlation (BH-FDR)")

    # ── summary figure ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, max(4, len(corr_df) * 0.15)))
    colors = ["crimson" if s else "grey" for s in corr_df["significant"]]
    ax.scatter(corr_df["rho"], corr_df["cluster_label_cv"], color=colors, s=20)
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Spearman ρ (F_mk vs d')")
    ax.set_ylabel("cluster_label_cv")
    ax.set_title(f"d'-occupancy correlation per cluster\n"
                 f"(red = BH-FDR significant, α={cfg['fdr_alpha']})")
    fig.tight_layout()
    _save(fig, out_dir / "dprime_correlation_summary", dpi=300)

    print(f"  Done.")
    return corr_df


# ══════════════════════════════════════════════════════════════════════════
# Combined summary figure
# ══════════════════════════════════════════════════════════════════════════

def make_combined_summary(latency_results, families_df, roc_summary, dprime_corr,
                          all_clusters, out_dir):
    """latency_results : dict {tag: cluster_lat_df} — one entry per entry in
    cfg['latency_conditions'], or {} if latency wasn't run."""
    print("\n[Summary] Building combined summary figure")
    n_panels = (len(latency_results) +
                (1 if roc_summary is not None else 0) +
                (1 if dprime_corr is not None else 0))
    if n_panels == 0:
        print("  No analyses produced results — skipping combined summary figure")
        return

    panels = []
    for tag, df in latency_results.items():
        if df is not None:
            panels.append((f"latency__{tag}", df))
    roc_types = []
    if roc_summary is not None:
        roc_types = list(roc_summary["analysis_type"].unique())
        for at in roc_types:
            panels.append((f"roc__{at}", roc_summary[roc_summary["analysis_type"] == at]))
    if dprime_corr is not None:
        panels.append(("dprime", dprime_corr))

    n_cols = len(panels)
    fig_h = max(5, len(all_clusters) * 0.15)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.4 * n_cols, fig_h), sharey=True)
    axes = np.atleast_1d(axes)

    # family color strip on first axis
    family_color_map = {}
    if families_df is not None:
        fam_ids = sorted(families_df["family"].unique())
        cmap = matplotlib.colormaps.get_cmap("tab20")
        family_color_map = {f: cmap(i % 20) for i, f in enumerate(fam_ids)}

    for ax, (name, df) in zip(axes, panels):
        if name.startswith("latency__"):
            ax.errorbar(df["mean"] * 1000, df["cluster_label_cv"], xerr=df["sem"] * 1000,
                       fmt="o", ms=6, color="darkorange", ecolor="darkorange",
                       elinewidth=1.0, capsize=2)
            ax.set_xlabel("Latency (ms)")
        elif name.startswith("roc__"):
            ax.scatter(df["pct_significant"], df["cluster_label_cv"], s=12, color="steelblue")
            ax.set_xlabel("% sig.")
        elif name == "dprime":
            colors = ["crimson" if s else "grey" for s in df["significant"]]
            ax.scatter(df["rho"], df["cluster_label_cv"], s=12, color=colors)
            ax.axvline(0, color="k", lw=0.4, ls="--")
            ax.set_xlabel("ρ (d')")
        title = (name.replace("roc__", "ROC:\n").replace("latency__", "Latency:\n")
                 if ("roc__" in name or "latency__" in name) else name)
        ax.set_title(title, fontsize=8)

        if family_color_map:
            for _, row in (df.merge(families_df, on="cluster_label_cv", how="left")
                             if "family" not in df.columns else df).iterrows():
                fam = row.get("family", None)
                if fam is not None and pd.notna(fam):
                    ax.axhspan(row["cluster_label_cv"] - 0.4, row["cluster_label_cv"] + 0.4,
                               color=family_color_map.get(fam, "grey"), alpha=0.08, zorder=0)

    axes[0].set_ylabel("cluster_label_cv")
    fig.suptitle("Combined cluster-level analysis summary\n"
                 "(shaded bands = meta-clustering family, if computed)", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / "summary_figure", dpi=300)


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_rastermap_analyses(rastermap_dir: Path, unit_table: pd.DataFrame, trial_table: pd.DataFrame,
           out_root: Path | None = None, run_analyses: list[str] | None = None,
           **cfg_overrides):
    """
    rastermap_dir : path to a completed rastermap_psth output folder (read from disk).
    unit_table    : already-loaded DataFrame, long format (one row per unit x analysis_type).
    trial_table   : already-loaded DataFrame.
    """
    cfg = {**DEFAULT_CFG, **cfg_overrides}
    if run_analyses is not None:
        for key in ["run_latency", "run_meta_clustering", "run_roc", "run_dprime"]:
            cfg[key] = False
        name_map = dict(latency="run_latency", meta_clustering="run_meta_clustering",
                        roc="run_roc", dprime="run_dprime")
        for name in run_analyses:
            cfg[name_map[name]] = True

    rastermap_dir = Path(rastermap_dir)
    out_root      = Path(out_root) if out_root is not None else rastermap_dir
    run_dir       = out_root
    print(f"Output directory: {run_dir}")

    with open(run_dir / "run_config.txt", "w") as fh:
        fh.write(f"rastermap_dir: {rastermap_dir}\n")
        fh.write(f"unit_table:    DataFrame, {len(unit_table)} rows\n")
        fh.write(f"trial_table:   DataFrame, {len(trial_table)} rows\n\n")
        for k, v in cfg.items():
            fh.write(f"{k}: {v}\n")

    print("Loading rastermap output...")
    cluster_df = load_cluster_table(rastermap_dir)
    unit_ids, cond_labels, psth, t_ctr = load_psth_npz(rastermap_dir)
    all_clusters = np.sort(cluster_df["cluster_label_cv"].unique())

    latency_results = {}
    families_df = roc_summary = dprime_corr = None

    if cfg["run_latency"]:
        for tag, cond_label in cfg["latency_conditions"].items():
            latency_results[tag] = run_latency_analysis(
                cluster_df, psth, t_ctr, unit_ids, cfg,
                run_dir / f"01_latency_{tag}", tag, cond_label)

    if cfg["run_meta_clustering"]:
        families_df = run_meta_clustering(cluster_df, cond_labels, psth, unit_ids, cfg,
                                          run_dir / "02_meta_clustering")

    if cfg["run_roc"]:
        roc_summary = run_roc_analysis(cluster_df, unit_table, cfg,
                                       run_dir / "03_roc")

    if cfg["run_dprime"]:
        dprime_corr = run_dprime_analysis(cluster_df, trial_table, cfg,
                                          run_dir / "04_dprime")

    make_combined_summary(latency_results, families_df, roc_summary, dprime_corr,
                          all_clusters, run_dir)

    print(f"\nDone. All outputs in {run_dir}")
    return dict(run_dir=run_dir, latency_results=latency_results, families_df=families_df,
                roc_summary=roc_summary, dprime_corr=dprime_corr)


def _load_table(path: Path) -> pd.DataFrame:
    """Convenience loader for the optional CLI path below — supports pickle,
    csv, and parquet by extension. Not used when run_all() is called
    programmatically with DataFrames already in memory (the normal path)."""
    suffix = path.suffix.lower()
    if suffix in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".csv", ".tsv"):
        return pd.read_csv(path)
    raise ValueError(f"Don't know how to load table with extension '{suffix}': {path}")


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__ + (
            "\n\nNOTE: the primary interface is the run_all() function, called "
            "with unit_table/trial_table already loaded as DataFrames (see Usage "
            "above). This CLI is a convenience wrapper for when the tables happen "
            "to also exist as files on disk (pickle/csv/parquet)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rastermap-dir", required=True, type=Path,
                   help="Path to a completed rastermap_psth output folder "
                        "(must contain neuron_cluster_labels_cv.csv and "
                        "neuron_psth_by_condition.npz)")
    p.add_argument("--unit-table", required=True, type=Path,
                   help="Path to a unit_table file (.pkl/.csv/.parquet) — "
                        "loaded here only for CLI convenience")
    p.add_argument("--trial-table", required=True, type=Path,
                   help="Path to a trial_table file (.pkl/.csv/.parquet) — "
                        "loaded here only for CLI convenience")
    p.add_argument("--out-root", type=Path, default=None,
                   help="Where to create analyses/analyses_N/ (default: rastermap-dir)")
    p.add_argument("--run", nargs="+", default=None,
                   choices=["latency", "meta_clustering", "dprime"], #["latency", "meta_clustering", "roc", "dprime"]
                   help="Subset of analyses to run (default: all four)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    unit_table_df  = _load_table(args.unit_table)
    trial_table_df = _load_table(args.trial_table)
    run_rastermap_analyses(args.rastermap_dir, unit_table_df, trial_table_df, out_root=args.out_root,
                           run_analyses=args.run)