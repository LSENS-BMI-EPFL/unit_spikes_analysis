"""
cluster_postproc_analyses.py — Post-hoc analyses on a rastermap_psth cluster output.

Loads a completed rastermap_psth run (neuron_cluster_labels_cv.csv +
neuron_psth_by_condition.npz) and cross-references it against a unit_table
(ROC selectivity results) and a trial_table (behavioral d-prime), independent
of the original Rastermap fitting step.

Five toggleable analyses, each producing intermediate/diagnostic outputs plus
a summary panel, and a final combined summary figure across the first four.

    1. Per-neuron / per-cluster response latency (threshold-crossing)
    2. Meta-clustering of cluster-mean PSTHs (hierarchical, 1-correlation dist)
    3. ROC cross-reference (per analysis_type, % significant + direction)
    4. Behavioral d-prime vs. per-cluster fractional representation (F_mk)
    5. Sensorimotor arc: anatomical position + functional-fraction profiles
       as a function of rastermap cluster index (see run_sensorimotor_arc)

Usage
-----
    Called programmatically, the same way run_rastermap_psth(units, trials, ...)
    is called elsewhere in the pipeline — unit_table and trial_table are
    passed in as already-loaded DataFrames, not file paths:

        from cluster_postproc_analyses import run_rastermap_analyses

        run_rastermap_analyses(
            rastermap_dir = "/path/to/rastermap_output_folder",
            unit_table    = unit_table,     # in-memory DataFrame
            trial_table   = trial_table,    # in-memory DataFrame
            out_root      = None,           # default: rastermap_dir
            run_analyses  = None,           # default: all enabled analyses
        )

Inputs expected inside rastermap_dir
---------------------------------------
    neuron_cluster_labels_cv.csv   (unit_ids, cluster_label, km_label, cluster_id,
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
    ap, ml, dv   (CCF coordinates — static per neuron,
                     used by analysis 5; repeated across each unit's analysis_type rows)

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
        05_sensorimotor_arc/   (if run)
        summary_figure.png/.pdf/.svg
        run_config.txt
"""
from __future__ import annotations

import argparse
import itertools
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import spearmanr, norm as scipy_norm
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist


# ══════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════

DEFAULT_CFG: dict[str, Any] = dict(
    # ── which analyses to run ──────────────────────────────────────────
    run_latency          = True,
    run_meta_clustering  = True,
    run_roc              = False,
    run_dprime           = True,
    run_sensorimotor_arc = True,    # needs only unit_table (ap/ml/dv columns)

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
                             "spontaneous_licks", "wh_vs_aud_active"],

    # ── 4. d-prime ────────────────────────────────────────────────────────
    dprime_exclude_perf   = 6,
    dprime_exclude_context = "passive",
    dprime_hit_trial_type  = "whisker_trial",
    dprime_fa_trial_type   = "no_stim_trial",
    dprime_min_trials       = 5,           # min trials per rate to trust a mouse

    # ── 5. sensorimotor arc ──────────────────────────────────────────────
    anatomical_axes_cols = ["ap", "ml", "dv"],  # cols in unit_table
    sensorimotor_analysis_types = dict(
        whisker="whisker_active", auditory="auditory_active",
        choice="choice", lick="spontaneous_licks",
    ),
    # which 2D CCF localization plot variants to generate, per axis pair —
    # see run_sensorimotor_arc for what each one shows
    ccf_localization_plots = ["cluster_means", "point_cloud", "density", "ellipses"],
    ccf_point_cloud_alpha  = 0.35,   # per-neuron marker transparency
    ccf_hexbin_gridsize    = 25,     # density-plot hexbin resolution
    ccf_ellipse_n_std      = 1.0,    # ellipse radius, in SDs, for the "ellipses" variant
    # additional CCF plots colored by ROC-based metrics instead of cluster_label —
    # directly tests whether anatomical position tracks a functional property
    ccf_metric_plots = ["cluster_mean_by_metric", "point_cloud_by_significance"],

    # ── join keys ─────────────────────────────────────────────────────────
    join_keys = ["mouse_id", "session_id", "electrode_group"],

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


def _sem(x):
    """Standard error of the mean, NaN-safe, 0 for n<=1."""
    x = pd.Series(x).dropna()
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


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


def _normalize_path_string(path):
    """Guards against a Windows-style path (e.g. copied from a UNC/mapped-drive
    location like M:\\...) being passed while running on a POSIX system (e.g. via
    a mounted network share). Backslashes are literal filename characters on POSIX,
    not separators, so an unnormalized path silently produces a bogus nested
    directory name instead of an error until mkdir rejects it. Only touches str
    input with backslashes on non-Windows; Path objects and pure-Windows runs are
    left alone."""
    if isinstance(path, str) and os.name != "nt" and "\\" in path:
        normalized = path.replace("\\", "/")
        print(f"  [warn] '{path}' looks like a Windows path but this is a POSIX "
              f"system — normalizing to '{normalized}'")
        return normalized
    return path


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
    """Loads the cluster table and normalizes column names to the current
    convention (unit_ids, cluster_label), regardless of whether the
    underlying CSV still uses the older unit_id / cluster_label_cv names."""
    def _normalize_columns(df):
        return df.rename(columns={"unit_id": "unit_ids", "cluster_label_cv": "cluster_label"})

    path = rastermap_dir / "neuron_cluster_labels_cv.csv"
    if not path.exists():
        # fall back to the non-CV table if cross_validate=False was used upstream
        alt = rastermap_dir / "neuron_cluster_labels.csv"
        if alt.exists():
            print(f"  [warn] {path.name} not found, falling back to {alt.name}")
            df = _normalize_columns(pd.read_csv(alt))
            return df
        raise FileNotFoundError(f"Neither neuron_cluster_labels_cv.csv nor "
                                 f"neuron_cluster_labels.csv found in {rastermap_dir}")
    df = _normalize_columns(pd.read_csv(path))
    print(f"  Loaded cluster table: {len(df)} neurons, "
          f"{df['cluster_label'].nunique()} clusters")
    return df


def load_psth_npz(rastermap_dir: Path):
    path = rastermap_dir / "neuron_psth_by_condition.npz"
    if not path.exists():
        # falls back one level up — neuron_psth_by_condition.npz is written by the
        # upstream clustering step alongside the rastermap/ subfolder, not inside it
        alt = rastermap_dir.parent / "neuron_psth_by_condition.npz"
        if alt.exists():
            print(f"  [info] neuron_psth_by_condition.npz not in {rastermap_dir}, "
                  f"using {alt} instead")
            path = alt
        else:
            raise FileNotFoundError(f"neuron_psth_by_condition.npz not found in "
                                     f"{rastermap_dir} or its parent {rastermap_dir.parent}")
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
      (a) latency_vs_cluster_position — y = cluster_label, matrix order
          (inverted y-axis so cluster 0 sits at top, matching the population
          matrix / rastermap figure convention).
      (b) latency_vs_latency_order — same plot but clusters sorted by their
          own mean latency, so the range/ordering of response timing is
          immediately visible.
    """
    # (a) aligned to matrix (cluster_label) order
    df1 = cluster_lat.sort_values("cluster_label")
    fig, ax = plt.subplots(figsize=(5, max(4, len(df1) * 0.18)))
    ax.errorbar(df1["mean"] * 1000, df1["cluster_label"], xerr=df1["sem"] * 1000,
                fmt="o", ms=9, mec="white", mew=0.6, color="darkorange",
                ecolor="darkorange", elinewidth=1.5, capsize=3, zorder=3)
    ax.invert_yaxis()   # cluster 0 at top, matches population-matrix orientation
    ax.set_xlabel("Mean latency (ms) ± SEM")
    ax.set_ylabel("cluster_label")
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
    ax.set_yticklabels(df2["cluster_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean latency (ms) ± SEM")
    ax.set_ylabel("cluster_label (sorted by latency)")
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
    for uid in cluster_df["unit_ids"]:
        row_idx = uid_to_row.get(uid)
        if row_idx is None:
            continue
        trace = mat[row_idx]
        lat, sign, peak = _threshold_crossing_latency(trace, t_c, window, thresh)
        per_neuron_rows.append(dict(unit_ids=uid, condition=cond_label,
                                     latency_s=lat, sign=sign, peak=peak))
    lat_df = pd.DataFrame(per_neuron_rows)
    lat_df = lat_df.rename(columns={"latency_s": "latency_s_mean"})
    lat_df.to_csv(out_dir / f"latency_per_neuron_{tag}.csv", index=False)

    merged = cluster_df.merge(lat_df[["unit_ids", "latency_s_mean", "sign", "peak"]],
                              on="unit_ids", how="left")

    # per-cluster aggregation — mean ± SEM (primary) plus median/IQR (kept for reference)
    cluster_lat = (merged.groupby("cluster_label")["latency_s_mean"]
                          .agg(mean="mean",
                               sem=_sem,
                               median="median",
                               q25=lambda x: x.quantile(0.25),
                               q75=lambda x: x.quantile(0.75),
                               n_valid=lambda x: x.notna().sum(),
                               n_total="size")
                          .reset_index()
                          .sort_values("cluster_label"))
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
    ex_uids  = rng.choice(cluster_df["unit_ids"].values, size=n_ex, replace=False)
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
    row_idx_for_uid = np.array([uid_to_row.get(uid, -1) for uid in cluster_df["unit_ids"]])
    valid = row_idx_for_uid >= 0
    if not valid.all():
        print(f"  [warn] {(~valid).sum()} cluster-table neurons not found in PSTH file, dropping")
    sub_cluster_df = cluster_df[valid].reset_index(drop=True)
    row_idx_for_uid = row_idx_for_uid[valid]

    X_full = np.concatenate([psth[label][row_idx_for_uid] for label in cond_labels], axis=1)

    clusters = np.sort(sub_cluster_df["cluster_label"].unique())
    cluster_means = np.vstack([
        X_full[sub_cluster_df["cluster_label"].values == c].mean(axis=0)
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
    families_df = pd.DataFrame({"cluster_label": clusters, "family": family_labels})
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
        ax.set_ylabel("cluster_label")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(cond_labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(title)
        return im

    fig, ax = plt.subplots(figsize=(9, max(6, len(clusters) * 0.15)))
    im = _draw_cluster_matrix(ax, cluster_means, clusters,
                              "STEP 1 — Input: cluster-mean PSTH matrix\n"
                              "(cluster_label order, before any clustering)")
    fig.colorbar(im, ax=ax, label="Firing rate (z-score)", shrink=0.6, pad=0.02)
    fig.tight_layout()
    _save(fig, out_dir / "qc_input_matrix_original_order", dpi=300)

    # ── STEP 2: pairwise similarity (correlation matrix) ─────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("STEP 2 — Pairwise similarity\n(Pearson correlation, original order)")
    ax.set_xlabel("cluster_label"); ax.set_ylabel("cluster_label")
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    fig.tight_layout()
    _save(fig, out_dir / "qc_correlation_matrix_original_order", dpi=300)

    # ── STEP 3: dendrogram (from linkage on 1-r distance) next to the ────
    # INPUT matrix, in its ORIGINAL cluster_label order. The tree's leaves
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
    n_neurons = cluster_df["unit_ids"].nunique()
    n_matched = merged.dropna(subset=["analysis_type"])["unit_ids"].nunique()
    print(f"  Merge: {n_matched}/{n_neurons} neurons matched at least one ROC row in unit_table")

    with open(out_dir / "roc_merge_diagnostics.txt", "w") as fh:
        fh.write(f"Total neurons in cluster table: {n_neurons}\n")
        fh.write(f"Neurons with >=1 matched ROC row: {n_matched}\n")
        fh.write(f"Unmatched: {n_neurons - n_matched}\n\n")
        for at in cfg["roc_analysis_types"]:
            sub = merged[merged["analysis_type"] == at]
            fh.write(f"{at}: {sub['unit_ids'].nunique()} matched neurons, "
                     f"baseline significance rate = "
                     f"{sub['significant'].mean() if len(sub) else float('nan'):.3f}\n")

    # ── per-cluster % significant + direction, per analysis_type ──────────
    summary_rows = []
    for at in cfg["roc_analysis_types"]:
        sub = merged[merged["analysis_type"] == at]
        if sub.empty:
            print(f"  [warn] analysis_type '{at}' not found in unit_table — skipping")
            continue
        for c, grp in sub.groupby("cluster_label"):
            n_total = len(grp)
            n_sig   = grp["significant"].sum()
            pct_sig = n_sig / n_total if n_total > 0 else np.nan
            sig_dirs = grp.loc[grp["significant"], "direction"]
            dir_counts = sig_dirs.value_counts(normalize=True).to_dict()
            summary_rows.append(dict(cluster_label=c, analysis_type=at,
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
    fig, axes = plt.subplots(1, n_at, figsize=(2.4 * n_at, max(4, cluster_df['cluster_label'].nunique() * 0.15)),
                              sharey=True)
    axes = np.atleast_1d(axes)
    for ax, at in zip(axes, roc_summary["analysis_type"].unique()):
        sub = roc_summary[roc_summary["analysis_type"] == at].sort_values("cluster_label")
        ax.scatter(sub["pct_significant"], sub["cluster_label"], s=10, color="darkorange")
        ax.set_title(at, fontsize=8)
        ax.set_xlabel("% sig.")
    axes[0].set_ylabel("cluster_label")
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

    ct = pd.crosstab(cluster_df["mouse_id"], cluster_df["cluster_label"])
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
        corr_rows.append(dict(cluster_label=cluster, rho=rho, p_raw=p, n_mice=valid.sum()))
    corr_df = pd.DataFrame(corr_rows).sort_values("cluster_label")
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
    ax.scatter(corr_df["rho"], corr_df["cluster_label"], color=colors, s=20)
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Spearman ρ (F_mk vs d')")
    ax.set_ylabel("cluster_label")
    ax.set_title(f"d'-occupancy correlation per cluster\n"
                 f"(red = BH-FDR significant, α={cfg['fdr_alpha']})")
    fig.tight_layout()
    _save(fig, out_dir / "dprime_correlation_summary", dpi=300)

    print(f"  Done.")
    return corr_df


# ══════════════════════════════════════════════════════════════════════════
# 5. Sensorimotor arc
# ══════════════════════════════════════════════════════════════════════════

def _extract_per_neuron_anatomical(unit_table, join_keys, anatomical_cols):
    """unit_table is long (one row per unit x analysis_type), but CCF
    coordinates are static per neuron and repeated across its analysis_type
    rows — collapse to one row per neuron via drop_duplicates on join_keys."""
    cols_present = [c for c in anatomical_cols if c in unit_table.columns]
    missing = [c for c in anatomical_cols if c not in unit_table.columns]
    if missing:
        print(f"  [warn] anatomical column(s) not found in unit_table, skipping: {missing}")
    if not cols_present:
        return pd.DataFrame(columns=join_keys), []

    sub = unit_table[join_keys + cols_present].drop_duplicates(subset=join_keys).copy()
    for col in cols_present:
        if not pd.api.types.is_numeric_dtype(sub[col]):
            coerced = pd.to_numeric(sub[col], errors="coerce")
            n_bad = coerced.isna().sum() - sub[col].isna().sum()
            if n_bad > 0:
                print(f"  [warn] '{col}': {n_bad} non-numeric value(s) coerced to NaN "
                      f"(dtype was {sub[col].dtype})")
            sub[col] = coerced
    return sub, cols_present


def _per_cluster_fraction_significant(cluster_df, unit_table, analysis_type, join_keys):
    """Fraction of neurons per cluster with significant==True for a given
    analysis_type in unit_table, plus n_valid = the number of neurons that
    actually had a matching ROC row (the denominator of that fraction — NaN
    rows from an unmatched left-merge are excluded from both count and mean).
    Returns a DataFrame with columns ['frac_significant', 'n_valid'], indexed
    by cluster_label."""
    sub = unit_table[unit_table["analysis_type"] == analysis_type]
    if sub.empty:
        print(f"  [warn] analysis_type '{analysis_type}' not found in unit_table")
        return pd.DataFrame(columns=["frac_significant", "n_valid"], dtype=float)

    merged = cluster_df[["unit_ids", "cluster_label"] + join_keys].merge(
        sub[join_keys + ["significant"]], on=join_keys, how="left")
    merged["significant"] = merged["significant"].astype(float)
    return merged.groupby("cluster_label")["significant"].agg(
        frac_significant="mean", n_valid="count")


def _draw_sensorimotor_arc_figure(rows_data, clusters, out_path, title_suffix, reverse):
    """rows_data : list of (row_label, kind, data) tuples, in the order they
    should stack top-to-bottom. kind == 'value' -> data is a DataFrame with
    'mean'/'sem' columns indexed by cluster_label. kind == 'fraction' ->
    data is a Series indexed by cluster_label."""
    order = clusters[::-1] if reverse else clusters
    x = np.arange(len(order))
    n_rows = len(rows_data)

    # squarish per-panel sizing: taller rows, more modest width growth per
    # cluster than a flat horizontal-strip layout would use
    fig, axes = plt.subplots(n_rows, 1, figsize=(max(7, len(order) * 0.13), 3.0 * n_rows),
                              sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (label, kind, data) in zip(axes, rows_data):
        vals_ordered = data.reindex(order)
        if kind == "value":
            ax.errorbar(x, vals_ordered["mean"], yerr=vals_ordered["sem"],
                        fmt="o-", ms=4, lw=1, color="darkslateblue",
                        ecolor="darkslateblue", elinewidth=1, capsize=2)
        else:  # "fraction"
            ax.plot(x, vals_ordered.values, "o-", ms=4, lw=1, color="teal")
            finite = vals_ordered.values[np.isfinite(vals_ordered.values)]
            top = np.nanmax(finite) * 1.1 if len(finite) else 1.0
            ax.set_ylim(0, max(top, 0.1))
        ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
        ax.grid(axis="x", alpha=0.15)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(order, rotation=90, fontsize=6)
    axes[-1].set_xlabel(f"cluster_label "
                        f"({'decreasing' if reverse else 'increasing'} index)")
    fig.suptitle(f"Sensorimotor arc across rastermap clusters{title_suffix}", fontsize=10)
    fig.tight_layout()
    _save(fig, out_path, dpi=300)


def _short_axis_name(col: str) -> str:
    return col.replace("ccf_atlas_", "")


def _draw_ccf_cluster_means_figure(x_stats, y_stats, clusters, x_col, y_col, out_path):
    """Variant 'cluster_means': one point per cluster at its mean CCF position
    (± SEM), connected in cluster order. The clean, high-level view of any
    trajectory through anatomical space — but it hides overlap and within-
    cluster spread, which the other variants below show instead."""
    x = x_stats["mean"].reindex(clusters)
    y = y_stats["mean"].reindex(clusters)
    xerr = x_stats["sem"].reindex(clusters)
    yerr = y_stats["sem"].reindex(clusters)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor="grey",
                elinewidth=0.6, capsize=1.5, alpha=0.4, zorder=1)
    ax.plot(x, y, "-", color="grey", lw=0.8, alpha=0.5, zorder=2)
    sc = ax.scatter(x, y, c=clusters, cmap="viridis", s=50, zorder=3,
                     edgecolors="white", linewidths=0.5)
    fig.colorbar(sc, ax=ax, label="cluster_label", shrink=0.8)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Cluster means ± SEM, connected in order\n{x_col} vs. {y_col}", fontsize=10)
    fig.tight_layout()
    _save(fig, out_path, dpi=300)


def _draw_ccf_point_cloud_figure(anat_merged, clusters, x_col, y_col, alpha, out_path):
    """Variant 'point_cloud': every single neuron plotted at its own CCF
    position, colored by its cluster_label (viridis) with low alpha so
    overlapping regions read as denser. Cluster means are overlaid as small
    white-edged markers for reference. Shows how much clusters' anatomical
    footprints actually overlap, which the means-only view can't."""
    sub = anat_merged.dropna(subset=[x_col, y_col])
    means = sub.groupby("cluster_label")[[x_col, y_col]].mean().reindex(clusters)

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(sub[x_col], sub[y_col], c=sub["cluster_label"], cmap="viridis",
                     s=8, alpha=alpha, linewidths=0, zorder=2)
    ax.scatter(means[x_col], means[y_col], c=clusters, cmap="viridis", s=35,
               edgecolors="white", linewidths=0.7, zorder=3)
    fig.colorbar(sc, ax=ax, label="cluster_label", shrink=0.8)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Single-neuron positions, colored by cluster\n"
                 f"(small markers = per-cluster mean)\n{x_col} vs. {y_col}", fontsize=10)
    fig.tight_layout()
    _save(fig, out_path, dpi=300)


def _draw_ccf_density_figure(anat_merged, x_col, y_col, gridsize, out_path):
    """Variant 'density': hexbin of all neurons in this anatomical plane,
    independent of cluster assignment — shows where tissue was actually
    sampled, as context for interpreting the other variants (e.g. a gap in
    a cluster's spread might just be a gap in sampling, not a real boundary)."""
    sub = anat_merged.dropna(subset=[x_col, y_col])
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(sub[x_col], sub[y_col], gridsize=gridsize, cmap="inferno", mincnt=1)
    fig.colorbar(hb, ax=ax, label="# neurons", shrink=0.8)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Neuron sampling density (all clusters pooled)\n{x_col} vs. {y_col}", fontsize=10)
    fig.tight_layout()
    _save(fig, out_path, dpi=300)


def _draw_ccf_ellipses_figure(anat_merged, clusters, x_col, y_col, n_std, out_path):
    """Variant 'ellipses': one covariance ellipse per cluster (n_std SDs,
    from that cluster's own neurons) plus its mean, colored by cluster_label.
    A middle ground between 'cluster_means' (too coarse) and 'point_cloud'
    (can be too dense to read with many clusters) — summarizes each
    cluster's spread and lets overlap between neighboring clusters be judged
    directly from how much their ellipses intersect."""
    cmap = matplotlib.colormaps.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=np.min(clusters), vmax=np.max(clusters))

    fig, ax = plt.subplots(figsize=(6, 6))
    for c in clusters:
        sub = anat_merged.loc[anat_merged["cluster_label"] == c, [x_col, y_col]].dropna()
        if len(sub) < 3:
            continue
        mx, my = sub[x_col].mean(), sub[y_col].mean()
        cov = np.cov(sub[x_col], sub[y_col])
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * n_std * np.sqrt(np.clip(vals, 0, None))
        color = cmap(norm(c))
        ax.add_patch(Ellipse((mx, my), width, height, angle=angle, facecolor=color,
                             edgecolor=color, alpha=0.15, lw=1.2, zorder=2))
        ax.scatter([mx], [my], color=color, s=20, edgecolors="white", linewidths=0.5, zorder=3)

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="cluster_label", shrink=0.8)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_aspect("equal", adjustable="box")
    ax.relim(); ax.autoscale_view()
    ax.set_title(f"Per-cluster spread ({n_std:g}-SD ellipse)\n{x_col} vs. {y_col}", fontsize=10)
    fig.tight_layout()
    _save(fig, out_path, dpi=300)


def _merge_per_neuron_significance(cluster_df, unit_table, analysis_type, join_keys):
    """Per-neuron (not per-cluster) significant flag for one analysis_type —
    the row-level table _per_cluster_fraction_significant aggregates away.
    Returns a DataFrame with ['unit_ids', 'significant'] (float, NaN where
    unmatched), or None if analysis_type isn't in unit_table at all."""
    sub = unit_table[unit_table["analysis_type"] == analysis_type]
    if sub.empty:
        return None
    merged = cluster_df[["unit_ids"] + join_keys].merge(
        sub[join_keys + ["significant"]], on=join_keys, how="left")
    merged["significant"] = merged["significant"].astype(float)
    return merged[["unit_ids", "significant"]]


def _draw_ccf_cluster_metric_figure(x_stats, y_stats, metric_series, clusters,
                                    x_col, y_col, metric_label, out_path):
    """Cluster means in CCF space (as in 'cluster_means'), but colored by a
    per-cluster ROC-based metric (e.g. fraction whisker-responsive) instead
    of cluster_label — the direct test of whether anatomical position tracks
    a specific functional property, rather than just cluster identity."""
    x = x_stats["mean"].reindex(clusters)
    y = y_stats["mean"].reindex(clusters)
    xerr = x_stats["sem"].reindex(clusters)
    yerr = y_stats["sem"].reindex(clusters)
    c = metric_series.reindex(clusters)
    finite_c = c.values[np.isfinite(c.values)]
    vmax = np.nanmax(finite_c) if len(finite_c) else 1.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor="grey",
                elinewidth=0.6, capsize=1.5, alpha=0.3, zorder=1)
    ax.plot(x, y, "-", color="grey", lw=0.6, alpha=0.3, zorder=2)
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=60, zorder=3,
                    edgecolors="white", linewidths=0.6, vmin=0, vmax=max(vmax, 0.1))
    fig.colorbar(sc, ax=ax, label=metric_label, shrink=0.8)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Cluster means, colored by {metric_label}\n"
                 f"(thin line = cluster order, for reference)\n{x_col} vs. {y_col}", fontsize=10)
    fig.tight_layout()
    _save(fig, out_path, dpi=300)


def _draw_ccf_point_cloud_by_significance_figure(anat_with_sig, x_col, y_col, metric_label, out_path):
    """Single-neuron point cloud colored by whether each neuron was
    ROC-significant for one analysis_type — shows whether responsive neurons
    cluster anatomically, independent of the rastermap cluster assignment
    (a complementary check to the cluster-level 'cluster_mean_by_metric' view)."""
    sub = anat_with_sig.dropna(subset=[x_col, y_col])
    unmatched = sub[sub["significant"].isna()]
    nonsig    = sub[sub["significant"] == 0.0]
    sig       = sub[sub["significant"] == 1.0]

    fig, ax = plt.subplots(figsize=(6, 6))
    if len(unmatched):
        ax.scatter(unmatched[x_col], unmatched[y_col], color="lightgrey", s=8,
                   alpha=0.25, linewidths=0, zorder=1, label="no ROC match")
    ax.scatter(nonsig[x_col], nonsig[y_col], color="silver", s=8, alpha=0.4,
              linewidths=0, zorder=2, label="non-significant")
    ax.scatter(sig[x_col], sig[y_col], color="crimson", s=10, alpha=0.6,
              linewidths=0, zorder=3, label="significant")
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Single-neuron positions, colored by {metric_label} significance\n"
                 f"{x_col} vs. {y_col}", fontsize=10)
    fig.tight_layout()
    _save(fig, out_path, dpi=300)


def run_sensorimotor_arc(cluster_df, unit_table, cfg, out_dir):
    """Tests whether the rastermap continuous embedding orders clusters along
    a sensorimotor arc, by plotting anatomical position and functional-response
    fractions (both from unit_table) as a function of cluster_label, in
    both increasing and mirrored (decreasing) order."""
    out_dir.mkdir(exist_ok=True)
    print("\n[5] Sensorimotor arc across cluster index")

    join_keys = cfg["join_keys"]
    cluster_df = cluster_df.copy()
    unit_table = unit_table.copy()
    for c in join_keys:
        if c in cluster_df.columns:
            cluster_df[c] = cluster_df[c].astype(str)
        if c in unit_table.columns:
            unit_table[c] = unit_table[c].astype(str)

    clusters = np.sort(cluster_df["cluster_label"].unique())
    cluster_sizes = cluster_df.groupby("cluster_label").size().reindex(clusters)

    # ── anatomical axes (static per-neuron columns in unit_table) ─────────
    anat_per_neuron, anat_cols_present = _extract_per_neuron_anatomical(
        unit_table, join_keys, cfg["anatomical_axes_cols"])
    anat_merged = cluster_df[["unit_ids", "cluster_label"] + join_keys].merge(
        anat_per_neuron, on=join_keys, how="left")
    anat_merged.to_csv(out_dir / "cluster_anatomical_merged.csv", index=False)

    anat_rows = []
    anat_stats_by_col = {}
    anat_summary_cols = {}
    for col in anat_cols_present:
        stats = (anat_merged.groupby("cluster_label")[col]
                            .agg(mean="mean", sem=_sem, n="count")
                            .reindex(clusters))
        anat_rows.append((col, "value", stats))
        anat_stats_by_col[col] = stats
        anat_summary_cols[f"{col}_mean"] = stats["mean"]
        anat_summary_cols[f"{col}_sem"] = stats["sem"]

    # ── functional fractions (from unit_table, all analysis_types) ────────
    at_map = cfg["sensorimotor_analysis_types"]
    frac_stats = {key: _per_cluster_fraction_significant(cluster_df, unit_table, at, join_keys).reindex(clusters)
                  for key, at in at_map.items()}
    frac_whisker  = frac_stats["whisker"]["frac_significant"]
    frac_auditory = frac_stats["auditory"]["frac_significant"]
    frac_choice   = frac_stats["choice"]["frac_significant"]
    frac_lick     = frac_stats["lick"]["frac_significant"]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_wa = (frac_whisker / frac_auditory).replace([np.inf, -np.inf], np.nan)

    # ── match-rate diagnostics: how many neurons per cluster actually had a
    # ROC row for each analysis_type (n_valid), vs. the cluster's total size ──
    with open(out_dir / "sensorimotor_arc_diagnostics.txt", "w") as fh:
        for key, at in at_map.items():
            n_valid = frac_stats[key]["n_valid"]
            total_valid, total_n = n_valid.sum(), cluster_sizes.sum()
            fh.write(f"{key} ({at}): {int(total_valid)}/{int(total_n)} neurons matched overall "
                     f"({total_valid/total_n:.1%})\n")
            low_match = cluster_sizes.index[(n_valid / cluster_sizes) < 0.5]
            if len(low_match):
                fh.write(f"  [warn] clusters with <50% match rate: {list(low_match)}\n")
        low_match_any = [key for key in at_map
                          if ((frac_stats[key]["n_valid"] / cluster_sizes) < 0.5).any()]
    if low_match_any:
        print(f"  [warn] some clusters have <50% ROC match rate for: {low_match_any} "
              f"— see sensorimotor_arc_diagnostics.txt")

    frac_df = pd.DataFrame({
        "cluster_label": clusters,
        "n_neurons": cluster_sizes.values,
        **anat_summary_cols,
        "frac_whisker": frac_whisker.values,
        "n_valid_whisker": frac_stats["whisker"]["n_valid"].values,
        "frac_auditory": frac_auditory.values,
        "n_valid_auditory": frac_stats["auditory"]["n_valid"].values,
        "ratio_whisker_auditory": ratio_wa.values,
        "frac_choice": frac_choice.values,
        "n_valid_choice": frac_stats["choice"]["n_valid"].values,
        "frac_lick": frac_lick.values,
        "n_valid_lick": frac_stats["lick"]["n_valid"].values,
    })
    frac_df.to_csv(out_dir / "cluster_sensorimotor_profile.csv", index=False)

    rows_data = anat_rows + [
        ("frac.\nwhisker", "fraction", frac_whisker),
        ("frac.\nauditory", "fraction", frac_auditory),
        ("frac.\nchoice", "fraction", frac_choice),
        ("frac.\nlick", "fraction", frac_lick),
    ]

    _draw_sensorimotor_arc_figure(
        rows_data, clusters, out_dir / "sensorimotor_arc_increasing",
        " (increasing cluster index)", reverse=False)
    _draw_sensorimotor_arc_figure(
        rows_data, clusters, out_dir / "sensorimotor_arc_decreasing",
        " (mirrored: decreasing cluster index)", reverse=True)

    # ── 2D CCF localization: several alternative views, per axis pair ────
    if len(anat_cols_present) < 2:
        print(f"  [warn] fewer than 2 anatomical axes available — skipping 2D "
              f"CCF localization figures")
    else:
        ccf_dir = out_dir / "ccf_localization"
        ccf_dir.mkdir(exist_ok=True)
        variants = cfg["ccf_localization_plots"]
        for col_x, col_y in itertools.combinations(anat_cols_present, 2):
            pair_tag = f"{_short_axis_name(col_x)}_vs_{_short_axis_name(col_y)}"
            if "cluster_means" in variants:
                _draw_ccf_cluster_means_figure(
                    anat_stats_by_col[col_x], anat_stats_by_col[col_y], clusters,
                    col_x, col_y, ccf_dir / f"{pair_tag}__cluster_means")
            if "point_cloud" in variants:
                _draw_ccf_point_cloud_figure(
                    anat_merged, clusters, col_x, col_y, cfg["ccf_point_cloud_alpha"],
                    ccf_dir / f"{pair_tag}__point_cloud")
            if "density" in variants:
                _draw_ccf_density_figure(
                    anat_merged, col_x, col_y, cfg["ccf_hexbin_gridsize"],
                    ccf_dir / f"{pair_tag}__density")
            if "ellipses" in variants:
                _draw_ccf_ellipses_figure(
                    anat_merged, clusters, col_x, col_y, cfg["ccf_ellipse_n_std"],
                    ccf_dir / f"{pair_tag}__ellipses")
        print(f"  Saved {len(variants)} CCF localization variant(s) x "
              f"{len(list(itertools.combinations(anat_cols_present, 2)))} axis pair(s) "
              f"to {ccf_dir.name}/")

        # ── same axis pairs, but colored by a ROC-based metric instead of
        # cluster_label — tests whether anatomical position tracks a specific
        # functional property (whisker/auditory/choice/lick), not just cluster identity ──
        metric_variants = cfg["ccf_metric_plots"]
        metric_frac_series = dict(whisker=frac_whisker, auditory=frac_auditory,
                                  choice=frac_choice, lick=frac_lick)
        if metric_variants:
            roc_dir = ccf_dir / "by_roc_metric"
            roc_dir.mkdir(exist_ok=True)
            n_saved = 0
            for key, at in at_map.items():
                metric_label = f"frac_{key}"
                per_neuron_sig = None
                if "point_cloud_by_significance" in metric_variants:
                    per_neuron_sig = _merge_per_neuron_significance(cluster_df, unit_table, at, join_keys)
                    if per_neuron_sig is None:
                        print(f"  [warn] '{at}' not found in unit_table — skipping "
                              f"point_cloud_by_significance for {metric_label}")

                for col_x, col_y in itertools.combinations(anat_cols_present, 2):
                    pair_tag = f"{_short_axis_name(col_x)}_vs_{_short_axis_name(col_y)}"
                    if "cluster_mean_by_metric" in metric_variants:
                        _draw_ccf_cluster_metric_figure(
                            anat_stats_by_col[col_x], anat_stats_by_col[col_y],
                            metric_frac_series[key], clusters, col_x, col_y, metric_label,
                            roc_dir / f"{pair_tag}__mean_by_{key}")
                        n_saved += 1
                    if per_neuron_sig is not None:
                        anat_with_sig = anat_merged.merge(per_neuron_sig, on="unit_ids", how="left")
                        _draw_ccf_point_cloud_by_significance_figure(
                            anat_with_sig, col_x, col_y, metric_label,
                            roc_dir / f"{pair_tag}__points_by_{key}_sig")
                        n_saved += 1
            print(f"  Saved {n_saved} ROC-metric-colored CCF figure(s) to "
                  f"{ccf_dir.name}/{roc_dir.name}/")

    print(f"  Done. Profile saved to cluster_sensorimotor_profile.csv")
    return frac_df


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
            ax.errorbar(df["mean"] * 1000, df["cluster_label"], xerr=df["sem"] * 1000,
                       fmt="o", ms=6, color="darkorange", ecolor="darkorange",
                       elinewidth=1.0, capsize=2)
            ax.set_xlabel("Latency (ms)")
        elif name.startswith("roc__"):
            ax.scatter(df["pct_significant"], df["cluster_label"], s=12, color="steelblue")
            ax.set_xlabel("% sig.")
        elif name == "dprime":
            colors = ["crimson" if s else "grey" for s in df["significant"]]
            ax.scatter(df["rho"], df["cluster_label"], s=12, color=colors)
            ax.axvline(0, color="k", lw=0.4, ls="--")
            ax.set_xlabel("ρ (d')")
        title = (name.replace("roc__", "ROC:\n").replace("latency__", "Latency:\n")
                 if ("roc__" in name or "latency__" in name) else name)
        ax.set_title(title, fontsize=8)

        if family_color_map:
            for _, row in (df.merge(families_df, on="cluster_label", how="left")
                             if "family" not in df.columns else df).iterrows():
                fam = row.get("family", None)
                if fam is not None and pd.notna(fam):
                    ax.axhspan(row["cluster_label"] - 0.4, row["cluster_label"] + 0.4,
                               color=family_color_map.get(fam, "grey"), alpha=0.08, zorder=0)

    axes[0].set_ylabel("cluster_label")
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
                    Also carries ap/ml/dv (used by analysis 5).
    trial_table   : already-loaded DataFrame.
    """
    cfg = {**DEFAULT_CFG, **cfg_overrides}

    if run_analyses is not None:
        for key in ["run_latency", "run_meta_clustering", "run_roc", "run_dprime",
                    "run_sensorimotor_arc"]:
            cfg[key] = False
        name_map = dict(latency="run_latency", meta_clustering="run_meta_clustering",
                        roc="run_roc", dprime="run_dprime",
                        sensorimotor_arc="run_sensorimotor_arc")
        for name in run_analyses:
            cfg[name_map[name]] = True

    rastermap_dir = Path(_normalize_path_string(rastermap_dir))
    out_root      = Path(_normalize_path_string(out_root)) if out_root is not None else rastermap_dir
    run_dir       = make_run_dir(out_root)
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
    all_clusters = np.sort(cluster_df["cluster_label"].unique())

    latency_results = {}
    families_df = roc_summary = dprime_corr = sensorimotor_profile = None

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

    if cfg["run_sensorimotor_arc"]:
        sensorimotor_profile = run_sensorimotor_arc(cluster_df, unit_table, cfg,
                                                     run_dir / "05_sensorimotor_arc")

    make_combined_summary(latency_results, families_df, roc_summary, dprime_corr,
                          all_clusters, run_dir)

    print(f"\nDone. All outputs in {run_dir}")
    return dict(run_dir=run_dir, latency_results=latency_results, families_df=families_df,
                roc_summary=roc_summary, dprime_corr=dprime_corr,
                sensorimotor_profile=sensorimotor_profile)


def _load_table(path: Path) -> pd.DataFrame:
    """Convenience loader for the optional CLI path below — supports pickle,
    csv, and parquet by extension. Not used when run_rastermap_analyses() is
    called programmatically with DataFrames already in memory (the normal path)."""
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
            "\n\nNOTE: the primary interface is the run_rastermap_analyses() function, "
            "called with unit_table/trial_table already loaded as DataFrames (see "
            "Usage above). This CLI is a convenience wrapper for when the tables "
            "happen to also exist as files on disk (pickle/csv/parquet)."
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
                   choices=["latency", "meta_clustering", "roc", "dprime", "sensorimotor_arc"],
                   help="Subset of analyses to run (default: all enabled analyses)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    unit_table_df  = _load_table(args.unit_table)
    trial_table_df = _load_table(args.trial_table)
    run_rastermap_analyses(args.rastermap_dir, unit_table_df, trial_table_df,
                           out_root=args.out_root, run_analyses=args.run)