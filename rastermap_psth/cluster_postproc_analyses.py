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
from sklearn.metrics import silhouette_score

from ephys_utilities.plotting_utils.plotting_utils import *
# ══════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════

DEFAULT_CFG: dict[str, Any] = dict(
    # ── which analyses to run ──────────────────────────────────────────
    run_latency          = True,
    run_meta_clustering  = True,
    run_roc              = False,
    run_dprime           = False,
    run_sensorimotor_arc = True,    # needs only unit_table (ap/ml/dv columns)

    # ── 1. latency ──────────────────────────────────────────────────────
    latency_window_s    = (0.0, 0.5),     # post start_time search window
    latency_threshold_frac = 0.5,          # fraction of peak for crossing
    # Run latency separately per condition below (tag -> exact cond_labels string).
    # One full set of latency outputs is produced per entry.
    latency_conditions = {"whisker_hit": "Whisker hit", "auditory_hit": "Auditory hit"},
    latency_n_example_neurons = 12,
    # bins trimmed from BOTH ends of every trace before any latency
    # quantification (sections 1 and 1b) — gaussian_filter1d's zero-padded
    # convolution biases the first/last few bins toward 0, which shows up as
    # a sharp deflection at the trace edges and gets mis-detected as a real
    # peak/threshold crossing. Trimming, not further smoothing, is the fix —
    # see _trim_edge_bins.
    latency_edge_trim_bins = 2,

    # ── 1b. cluster-mean multi-method latency vs. rastermap sorting index ────
    # (see run_cluster_mean_latency_vs_sorting) — averages across trial types
    # per cluster, then compares several latency definitions against
    # cluster_label (the rastermap sorting index)
    #
    # IMPORTANT: latency_mm_baseline_window_s / latency_mm_search_window_s are
    # both defined relative to t=0 for the *condition's own alignment event*.
    # That's only meaningful if every condition fed into this analysis shares
    # the same alignment (start_time), since a single global window is applied
    # to all of them. Lick-aligned (or any other non-start_time-aligned)
    # conditions are therefore excluded automatically — see
    # latency_mm_conditions / latency_alignment_exclude_keywords below.
    latency_mm_conditions = None,   # None => auto: every cond_label NOT matching
                                     # latency_alignment_exclude_keywords (case-insensitive
                                     # substring match). Or pass an explicit list of exact
                                     # cond_labels to use, bypassing the keyword filter
                                     # entirely (e.g. ["Whisker hit", "Auditory hit"]).
    latency_alignment_exclude_keywords = ["lick", "jaw"],  # substrings (case-insensitive)
                                     # identifying non-start_time-aligned conditions to
                                     # exclude from the auto-detected condition list
    latency_mm_baseline_window_s = (-0.2, 0.0),  # pre-start_time window, "baseline_deviation" method
    latency_mm_search_window_s   = (0.0, 0.5),   # post-start_time window searched by all 3 methods
    latency_mm_half_thresh_frac  = 0.5,          # half-rise/-fall threshold, as frac of peak
    latency_mm_baseline_n_std    = 2.0,          # deviation threshold, in baseline SDs
    latency_mm_n_example_clusters      = 6,      # clusters shown in the method-illustration figures
    latency_mm_example_axis_group_index = 0,     # which time-axis group (see _group_conditions_
                                                  # by_time_axis) to use for illustration + the
                                                  # cluster-matrix-with-markers figure

    # ── 2. meta-clustering ──────────────────────────────────────────────
    meta_linkage_method  = "average",      # linkage on 1-corr distance (Pearson correlation)
    meta_dendro_cut_dist = 0.35,           # distance threshold -> families (1-corr; 0.35 => r>=0.65)

    # ── 2b. per-cluster trial-type grid (pooled + reward-group split) ────────
    # produced as part of meta-clustering — see _fig_cluster_trialtype_grid
    trialtype_row_groups     = None,   # None => single row, all conditions in npz order.
                                        # Or dict {row_label: [cond_label, ...]} for custom
                                        # rows (e.g. {"passive": [...], "active_start": [...],
                                        # "active_lick": [...]}), mirroring row_groups in the
                                        # original _fig_sigclusters_by_trialtype_by_rewardgroup
    trialtype_max_cols_per_row = 4,    # wraps any row with more conditions than this into
                                        # several figure rows (see _wrap_trialtype_row_groups),
                                        # so a long condition list doesn't produce one very
                                        # wide row. Set to None/0 to disable wrapping.
    trialtype_edge_trim_bins = 2,       # bins trimmed off each condition segment's edges
                                        # (gaussian_filter1d smoothing artifact)
    trialtype_pooled_color   = "black",
    rplus_color               = GROUP_COLORS['rplus'],
    rminus_color              = GROUP_COLORS['rminus'],

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


def _trim_edge_bins(trace, t_ctr, edge_trim_bins):
    """Trims `edge_trim_bins` samples off BOTH ends of the last axis of
    `trace` (1D: a single trace; 2D: neurons x time — works either way) and
    the matching `t_ctr`. Removes the region most affected by
    gaussian_filter1d's zero-padded convolution, where the first/last few
    bins are biased toward 0 and can look like a sharp real deflection to a
    peak-finding or threshold-crossing method. No-op if edge_trim_bins is
    falsy; clamped so it never eats more than a third of the trace.
    """
    if not edge_trim_bins:
        return trace, t_ctr
    trim = min(edge_trim_bins, len(t_ctr) // 3)
    if trim <= 0:
        return trace, t_ctr
    return trace[..., trim: trace.shape[-1] - trim], t_ctr[trim: len(t_ctr) - trim]


def run_latency_analysis(cluster_df, psth, t_ctr, unit_ids, cfg, out_dir, tag, cond_label):
    """Latency analysis for a single condition (e.g. 'Whisker hit'). Called
    once per entry in cfg['latency_conditions'] — each call is fully
    self-contained and writes to its own out_dir."""
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"\n[1] Latency analysis — {cond_label} ({tag})")

    if cond_label not in psth:
        print(f"  [warn] condition '{cond_label}' not found in PSTH file — skipping")
        return None

    if any(kw.lower() in cond_label.lower() for kw in cfg.get("latency_alignment_exclude_keywords", [])):
        print(f"  [warn] '{cond_label}' matches a latency_alignment_exclude_keywords entry "
              f"(looks lick/jaw-aligned, not start_time-aligned). latency_window_s is "
              f"defined relative to start_time, so results for this condition may not be "
              f"meaningful — double-check cfg['latency_conditions'] if this is unintended.")

    uid_to_row = {uid: i for i, uid in enumerate(unit_ids)}
    window     = cfg["latency_window_s"]
    thresh     = cfg["latency_threshold_frac"]
    mat        = psth[cond_label]
    t_c        = t_ctr[cond_label]

    edge_trim_bins = cfg.get("latency_edge_trim_bins", 0)
    mat, t_c = _trim_edge_bins(mat, t_c, edge_trim_bins)
    if edge_trim_bins:
        print(f"  Trimmed {min(edge_trim_bins, len(t_ctr[cond_label]) // 3)} bin(s) off each "
              f"trace edge (smoothing-artifact guard) — usable range now "
              f"[{t_c[0]:.3f}, {t_c[-1]:.3f}] s")

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
        fh.write(f"Edge-trim: {min(edge_trim_bins, len(t_ctr[cond_label]) // 3) if edge_trim_bins else 0} "
                 f"bin(s) removed from each end (usable range [{t_c[0]:.3f}, {t_c[-1]:.3f}] s)\n")
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


# ── 1b. Cluster-mean multi-method latency vs. rastermap sorting index ──────

def _method_peak_time(trace, t_ctr, search_window_s, **_):
    """Latency = time of max |deviation| within the search window. Simplest
    and most robust method; sign (+1 excited / -1 suppressed) comes from the
    sign of the trace at that time, so increasing and decreasing responses
    are both handled."""
    lo, hi = search_window_s
    mask = (t_ctr >= lo) & (t_ctr < hi)
    if mask.sum() == 0:
        return np.nan, 0
    seg, t_seg = trace[mask], t_ctr[mask]
    idx = int(np.nanargmax(np.abs(seg)))
    return float(t_seg[idx]), int(np.sign(seg[idx]))


def _method_half_rise(trace, t_ctr, search_window_s, half_thresh_frac, **_):
    """Latency = first crossing of `half_thresh_frac` of the peak (default:
    half-rise for excitation, half-fall for suppression — same convention as
    `_threshold_crossing_latency`, reused here directly)."""
    lat, sign, _peak = _threshold_crossing_latency(trace, t_ctr, search_window_s, half_thresh_frac)
    return lat, sign


def _method_baseline_deviation(trace, t_ctr, baseline_window_s, search_window_s, baseline_n_std, **_):
    """Latency = first bin in the search window where the trace deviates from
    the pre-stim baseline by more than `baseline_n_std` baseline SDs, in
    either direction. Most conservative of the three (data-driven threshold
    rather than a fixed fraction of peak)."""
    lo_b, hi_b = baseline_window_s
    bmask = (t_ctr >= lo_b) & (t_ctr < hi_b)
    if bmask.sum() < 2:
        return np.nan, 0
    baseline_mean = np.nanmean(trace[bmask])
    baseline_std  = np.nanstd(trace[bmask], ddof=1)
    if not np.isfinite(baseline_std) or baseline_std == 0:
        return np.nan, 0

    lo_s, hi_s = search_window_s
    smask = (t_ctr >= lo_s) & (t_ctr < hi_s)
    if smask.sum() == 0:
        return np.nan, 0
    seg, t_seg = trace[smask], t_ctr[smask]
    dev = (seg - baseline_mean) / baseline_std
    above = np.where(np.abs(dev) >= baseline_n_std)[0]
    if len(above) == 0:
        return np.nan, 0
    idx = above[0]
    return float(t_seg[idx]), int(np.sign(dev[idx]))


LATENCY_MM_METHODS = {
    "peak_time":          _method_peak_time,
    "half_rise":          _method_half_rise,
    "baseline_deviation": _method_baseline_deviation,
}


def _filter_start_time_conditions(cond_labels, cfg):
    """Restricts the condition list used by run_cluster_mean_latency_vs_sorting
    to start_time-aligned conditions only, since a single global search/
    baseline window (both defined relative to t=0) is applied to every
    condition that survives this filter — that's only valid if they all
    share the same alignment event.

    If cfg['latency_mm_conditions'] is given, it's used verbatim (exact
    cond_label match) and the keyword filter is skipped entirely — use this
    when your naming doesn't cleanly separate alignments by keyword. Missing
    entries are warned about and dropped.

    Otherwise, every cond_label containing one of
    cfg['latency_alignment_exclude_keywords'] (case-insensitive substring,
    e.g. "lick", "jaw") is excluded, and the rest are kept.

    Returns the filtered list, and prints exactly what was kept/excluded so
    this never silently changes which conditions feed the analysis.
    """
    explicit = cfg.get("latency_mm_conditions")
    if explicit is not None:
        missing = [c for c in explicit if c not in cond_labels]
        if missing:
            print(f"  [warn] latency_mm_conditions entries not found in cond_labels, "
                  f"dropping: {missing}")
        kept = [c for c in explicit if c in cond_labels]
        print(f"  Using explicit latency_mm_conditions: {kept}")
        return kept

    keywords = [kw.lower() for kw in cfg.get("latency_alignment_exclude_keywords", [])]
    kept, excluded = [], []
    for label in cond_labels:
        (excluded if any(kw in label.lower() for kw in keywords) else kept).append(label)
    print(f"  Auto-filtered to start_time-aligned conditions: {kept}")
    if excluded:
        print(f"  Excluded (matched latency_alignment_exclude_keywords): {excluded}")
    return kept


def _group_conditions_by_time_axis(cond_labels, t_ctr):
    """Groups cond_labels that share an identical time axis (same length and
    values), since only those can be averaged elementwise into one trace.
    Conditions with a different alignment/window (e.g. jaw/lick-aligned vs.
    stimulus-aligned) end up in separate groups; a cluster's final per-method
    latency is the average of that method's latency across all groups (see
    run_cluster_mean_latency_vs_sorting), not an elementwise average of
    incompatible time axes.
    Returns a list of (group_key, [cond_label, ...]) in first-seen order.
    """
    groups: dict[tuple, list[str]] = {}
    for label in cond_labels:
        t = t_ctr[label]
        key = (len(t), tuple(np.round(t, 6)))
        groups.setdefault(key, []).append(label)
    return list(groups.items())


def _select_example_clusters(clusters, n_examples):
    """Picks up to n_examples clusters spread evenly across the sorted
    cluster list (always including the first and last), for illustration
    figures where showing every cluster would be too dense to read."""
    clusters = np.asarray(clusters)
    if len(clusters) <= n_examples:
        return list(clusters)
    idx = np.unique(np.linspace(0, len(clusters) - 1, n_examples).round().astype(int))
    return [clusters[i] for i in idx]


def _draw_latency_mm_method_examples(trace_store, group_idx, clusters, method_kwargs, out_dir):
    """One figure per method: a small grid of example clusters showing the
    actual cluster-mean trace with that method's quantification overlaid
    (search window, threshold/baseline band, detected crossing point) — so
    the method's behavior can be checked visually rather than trusted
    blindly. Uses the trace already computed for `group_idx` in the main
    loop (trace_store), no recomputation.
    """
    ex_clusters = _select_example_clusters(clusters, len(clusters))  # filtered to n below
    ex_clusters = [c for c in ex_clusters if (group_idx, c) in trace_store]
    if not ex_clusters:
        print("  [warn] no example clusters available for method-illustration figures")
        return
    lo_s, hi_s = method_kwargs["search_window_s"]
    lo_b, hi_b = method_kwargs["baseline_window_s"]

    for method_name, method_fn in LATENCY_MM_METHODS.items():
        ncols = min(3, len(ex_clusters))
        nrows = int(np.ceil(len(ex_clusters) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharex=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, c in zip(axes, ex_clusters):
            t_c, trace = trace_store[(group_idx, c)]
            lat, sign = method_fn(trace, t_c, **method_kwargs)
            ax.plot(t_c, trace, color="k", lw=1.3, zorder=3)
            ax.axhline(0, color="grey", lw=0.5)
            ax.axvspan(lo_s, hi_s, color="grey", alpha=0.08, zorder=1, label="search window")

            if method_name == "baseline_deviation":
                bmask = (t_c >= lo_b) & (t_c < hi_b)
                if bmask.sum() >= 2:
                    bmean = np.nanmean(trace[bmask])
                    bstd  = np.nanstd(trace[bmask], ddof=1)
                    n_std = method_kwargs["baseline_n_std"]
                    ax.axvspan(lo_b, hi_b, color="steelblue", alpha=0.08, zorder=1)
                    ax.axhspan(bmean - n_std * bstd, bmean + n_std * bstd,
                               color="steelblue", alpha=0.15, zorder=2,
                               label=f"±{n_std:g} SD baseline")
                    ax.axhline(bmean, color="steelblue", lw=0.8, ls=":", zorder=2)
            elif method_name == "half_rise":
                smask = (t_c >= lo_s) & (t_c < hi_s)
                if smask.sum():
                    seg = trace[smask]
                    pos_peak, neg_peak = np.nanmax(seg), np.nanmin(seg)
                    peak = neg_peak if abs(neg_peak) > abs(pos_peak) else pos_peak
                    thresh_val = method_kwargs["half_thresh_frac"] * peak
                    ax.axhline(thresh_val, color="darkorange", lw=0.9, ls=":", zorder=2,
                               label=f"{method_kwargs['half_thresh_frac']*100:.0f}% of peak")
                    ax.scatter([t_c[smask][np.nanargmax(np.abs(seg))]], [peak],
                               marker="*", color="darkorange", s=60, zorder=4, label="peak")
            elif method_name == "peak_time":
                smask = (t_c >= lo_s) & (t_c < hi_s)
                if smask.sum():
                    seg, t_seg = trace[smask], t_c[smask]
                    idx = np.nanargmax(np.abs(seg))
                    ax.scatter([t_seg[idx]], [seg[idx]], marker="*", color="darkorange",
                               s=60, zorder=4, label="max |deviation|")

            if np.isfinite(lat):
                ax.axvline(lat, color="crimson", lw=1.4, ls="--", zorder=4)
                ax.scatter([lat], [np.interp(lat, t_c, trace)], color="crimson",
                           edgecolors="white", s=45, zorder=5)
                ax.set_title(f"cluster {c}: lat={lat*1000:.0f} ms "
                             f"({'excited' if sign >= 0 else 'suppressed'})", fontsize=8)
            else:
                ax.set_title(f"cluster {c}: no latency detected", fontsize=8)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("z-score")
            ax.legend(fontsize=6, frameon=False, loc="best")
        for ax in axes[len(ex_clusters):]:
            ax.set_visible(False)
        fig.suptitle(f"{method_name}: quantification on example cluster-mean traces", fontsize=10)
        fig.tight_layout()
        _save(fig, out_dir / f"qc_method_example_{method_name}", dpi=300)


def _draw_latency_mm_cluster_matrix_with_markers(trace_store, group_idx, clusters, latency_mm_df, out_dir):
    """Single figure: a cluster-mean-trace matrix (clusters x time, one
    time-axis group), with every method's detected latency overlaid as a
    distinctly-shaped/colored marker at (latency, cluster row). Ties the
    per-method quantification directly back to the traces it was computed
    from, across the whole cluster range at once (not just the examples in
    _draw_latency_mm_method_examples)."""
    valid_clusters, rows, t_c_ref = [], [], None
    for c in clusters:
        key = (group_idx, c)
        if key not in trace_store:
            continue
        t_c, trace = trace_store[key]
        t_c_ref = t_c
        rows.append(trace)
        valid_clusters.append(c)
    if not rows:
        print("  [warn] no traces available for the cluster-matrix-with-markers figure")
        return

    matrix = np.vstack(rows)
    vmax = np.nanpercentile(np.abs(matrix), 95)
    vmax = vmax if vmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(7.5, max(4, len(valid_clusters) * 0.3)))
    im = ax.imshow(matrix, aspect="auto", interpolation="none", cmap="bwr",
                    vmin=-vmax, vmax=vmax,
                    extent=[t_c_ref[0], t_c_ref[-1], len(valid_clusters), 0])
    ax.set_yticks(np.arange(len(valid_clusters)) + 0.5)
    ax.set_yticklabels(valid_clusters, fontsize=7)
    ax.set_ylabel("cluster_label")
    ax.set_xlabel("Time (s)")
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)

    method_markers = {"peak_time": ("o", "black"), "half_rise": ("s", "white"),
                       "baseline_deviation": ("^", "gold")}
    row_of = {c: i for i, c in enumerate(valid_clusters)}
    for method_name, (marker, color) in method_markers.items():
        sub = latency_mm_df[latency_mm_df["method"] == method_name].set_index("cluster_label")
        xs, ys = [], []
        for c in valid_clusters:
            if c in sub.index and np.isfinite(sub.loc[c, "latency_s"]):
                xs.append(sub.loc[c, "latency_s"])
                ys.append(row_of[c] + 0.5)
        ax.scatter(xs, ys, marker=marker, color=color, edgecolors="black",
                   linewidths=0.6, s=45, label=method_name, zorder=5)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.colorbar(im, ax=ax, label="Firing rate (z-score)", shrink=0.7)
    ax.set_title("Cluster-mean traces with all methods' detected latency overlaid", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / "qc_cluster_matrix_with_latency_markers", dpi=300)


def _draw_latency_mm_histograms(latency_mm_df, out_dir):
    """Distribution of per-cluster latency, one panel per method — the
    same QC role as the per-neuron latency histogram in section 1, but at
    the cluster level."""
    methods = list(latency_mm_df["method"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, method_name in zip(axes, methods):
        vals = latency_mm_df.loc[latency_mm_df["method"] == method_name, "latency_s"].dropna() * 1000
        n_bins = min(10, max(3, len(vals)))
        ax.hist(vals, bins=n_bins, color="steelblue", edgecolor="none")
        ax.set_title(method_name, fontsize=9)
        ax.set_xlabel("Latency (ms)")
    axes[0].set_ylabel("Cluster count")
    fig.suptitle("Per-method distribution of cluster-mean latency", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / "qc_latency_distributions_by_method", dpi=300)


def _draw_latency_mm_cross_method_comparison(latency_mm_df, out_dir):
    """Two figures assessing whether the three methods agree with each
    other (independent of the sorting-index correlation): a pairwise
    Spearman correlation heatmap between methods' per-cluster latencies,
    and a single dot plot showing all three methods' latency per cluster
    side by side."""
    pivot = latency_mm_df.pivot(index="cluster_label", columns="method", values="latency_s")
    pivot.to_csv(out_dir / "latency_multimethod_wide.csv")
    methods = list(pivot.columns)
    if len(methods) < 2:
        return

    corr = pivot.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(corr.values, cmap="bwr", vmin=-1, vmax=1)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=8)
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman r", shrink=0.8)
    ax.set_title("Cross-method agreement\n(per-cluster latency, Spearman r)", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir / "qc_cross_method_agreement", dpi=300)

    fig, ax = plt.subplots(figsize=(6, max(4, pivot.shape[0] * 0.25)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    for color, method_name in zip(colors, methods):
        ax.scatter(pivot[method_name] * 1000, pivot.index, color=color, s=28, label=method_name)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("cluster_label")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("All methods' per-cluster latency, side by side", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir / "qc_all_methods_per_cluster", dpi=300)


def run_cluster_mean_latency_vs_sorting(cluster_df, cond_labels, psth, t_ctr, unit_ids, cfg, out_dir):
    """Cluster-mean, multi-method latency vs. rastermap sorting index.

    For each cluster: build one cluster-mean activity trace per time-axis
    group by averaging over (a) all neurons in the cluster and (b) all trial
    types that share that time axis (see _group_conditions_by_time_axis).
    Compute latency of that trace with three independent methods
    (peak_time, half_rise, baseline_deviation — all sign-aware, so both
    increasing and decreasing responses are handled), then average each
    method's latency across time-axis groups to get one scalar per cluster
    per method. Finally, correlate (Spearman) each method's per-cluster
    latency against cluster_label (the rastermap sorting index), with a
    scatter + OLS regression line and reported rho/p.

    Only start_time-aligned conditions are used (see
    _filter_start_time_conditions) — a single global search/baseline window
    is applied to every condition here, which is only valid if they all
    share the same t=0 alignment event. Lick/jaw-aligned conditions are
    excluded by default.

    Produces, in order: (1) per-method example figures showing the
    quantification directly on real cluster-mean traces
    (qc_method_example_<method>), (2) a cluster-mean matrix with every
    method's detected latency overlaid (qc_cluster_matrix_with_latency_
    markers), (3) per-method latency distributions (qc_latency_
    distributions_by_method), (4) cross-method agreement — a Spearman
    correlation heatmap plus a side-by-side dot plot (qc_cross_method_
    agreement, qc_all_methods_per_cluster) — and finally (5) the
    per-method scatter + regression line against the sorting index
    (latency_vs_sorting_<method>).
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    print("\n[1b] Cluster-mean multi-method latency vs. sorting index")

    uid_to_row = {uid: i for i, uid in enumerate(unit_ids)}
    row_idx_for_uid = np.array([uid_to_row.get(uid, -1) for uid in cluster_df["unit_ids"]])
    valid = row_idx_for_uid >= 0
    if not valid.all():
        print(f"  [warn] {(~valid).sum()} cluster-table neurons not found in PSTH file, dropping")
    sub_cluster_df = cluster_df[valid].reset_index(drop=True)
    row_idx_for_uid = row_idx_for_uid[valid]
    clusters = np.sort(sub_cluster_df["cluster_label"].unique())

    latency_cond_labels = _filter_start_time_conditions(cond_labels, cfg)
    if not latency_cond_labels:
        print("  [warn] no start_time-aligned conditions available after filtering — "
              "skipping cluster-mean multi-method latency analysis")
        return None

    axis_groups = _group_conditions_by_time_axis(latency_cond_labels, t_ctr)
    print(f"  {len(latency_cond_labels)} condition(s) grouped into {len(axis_groups)} "
          f"distinct time-axis group(s): "
          + "; ".join(f"[{', '.join(labels)}]" for _, labels in axis_groups))

    method_kwargs = dict(
        baseline_window_s=cfg["latency_mm_baseline_window_s"],
        search_window_s=cfg["latency_mm_search_window_s"],
        half_thresh_frac=cfg["latency_mm_half_thresh_frac"],
        baseline_n_std=cfg["latency_mm_baseline_n_std"],
    )
    edge_trim_bins = cfg.get("latency_edge_trim_bins", 0)
    if edge_trim_bins:
        print(f"  Trimming {edge_trim_bins} bin(s) off each trace edge before quantification "
              f"(smoothing-artifact guard, see _trim_edge_bins)")

    # per_cluster_group_latency[method][cluster] -> list of per-group latencies (s)
    per_cluster_group_latency = {m: {c: [] for c in clusters} for m in LATENCY_MM_METHODS}
    per_cluster_group_sign    = {m: {c: [] for c in clusters} for m in LATENCY_MM_METHODS}
    trace_store = {}   # (group_idx, cluster) -> (t_c, cluster_mean_trace), for QC figures below

    for group_idx, (_, labels_in_group) in enumerate(axis_groups):
        t_c_full = t_ctr[labels_in_group[0]]   # identical across labels in this group, by construction
        for c in clusters:
            neurons_k = sub_cluster_df["cluster_label"].values == c
            rows_k = row_idx_for_uid[neurons_k]
            if len(rows_k) == 0:
                continue
            # cluster-mean trace: mean over neurons, then mean over conditions in this group
            cond_means = [psth[label][rows_k].mean(axis=0) for label in labels_in_group]
            group_trace_full = np.mean(cond_means, axis=0)
            # trim edge bins (smoothing-artifact guard, see _trim_edge_bins) before any
            # method sees the trace, so a false edge deflection can't be picked up as
            # the peak/threshold crossing
            group_trace, t_c = _trim_edge_bins(group_trace_full, t_c_full, edge_trim_bins)
            trace_store[(group_idx, c)] = (t_c, group_trace)
            for method_name, method_fn in LATENCY_MM_METHODS.items():
                lat, sign = method_fn(group_trace, t_c, **method_kwargs)
                per_cluster_group_latency[method_name][c].append(lat)
                per_cluster_group_sign[method_name][c].append(sign)

    # ── collapse across time-axis groups -> one row per cluster per method ──
    summary_rows = []
    for method_name in LATENCY_MM_METHODS:
        for c in clusters:
            lats  = np.array(per_cluster_group_latency[method_name][c], dtype=float)
            signs = np.array(per_cluster_group_sign[method_name][c], dtype=float)
            valid_lats = lats[np.isfinite(lats)]
            summary_rows.append(dict(
                method=method_name, cluster_label=c,
                latency_s=np.nanmean(lats) if len(valid_lats) else np.nan,
                n_groups_valid=len(valid_lats), n_groups_total=len(lats),
                mean_sign=np.nanmean(signs) if np.isfinite(signs).any() else np.nan,
            ))
    latency_mm_df = pd.DataFrame(summary_rows)
    latency_mm_df.to_csv(out_dir / "cluster_mean_latency_multimethod.csv", index=False)

    # ── per-method correlation vs. sorting index (cluster_label) ───────────
    corr_rows = []
    for method_name in LATENCY_MM_METHODS:
        sub = latency_mm_df[latency_mm_df["method"] == method_name].sort_values("cluster_label")
        x = sub["cluster_label"].values.astype(float)
        y = sub["latency_s"].values.astype(float)
        valid_mask = np.isfinite(x) & np.isfinite(y)
        n_valid = int(valid_mask.sum())
        if n_valid < 3:
            rho, p, slope, intercept = np.nan, np.nan, np.nan, np.nan
        else:
            rho, p = spearmanr(x[valid_mask], y[valid_mask])
            slope, intercept = np.polyfit(x[valid_mask], y[valid_mask], 1)
        corr_rows.append(dict(method=method_name, rho=rho, p_raw=p, n_valid=n_valid,
                               slope=slope, intercept=intercept))
    corr_df = pd.DataFrame(corr_rows)
    pvals = corr_df["p_raw"].fillna(1.0).values
    p_fdr, reject = _bh_correction(pvals, alpha=cfg["fdr_alpha"])
    corr_df["p_fdr"] = p_fdr
    corr_df["significant"] = reject
    corr_df.to_csv(out_dir / "latency_multimethod_vs_sorting_correlation.csv", index=False)

    # ── process figures: quantification examples -> cluster overview ──────
    # -> per-method distributions -> cross-method agreement -> final
    # correlation (below) ──────────────────────────────────────────────────
    example_group_idx = min(cfg["latency_mm_example_axis_group_index"], len(axis_groups) - 1)
    ex_clusters = _select_example_clusters(clusters, cfg["latency_mm_n_example_clusters"])
    _draw_latency_mm_method_examples(
        {k: v for k, v in trace_store.items() if k[0] == example_group_idx and k[1] in ex_clusters},
        example_group_idx, ex_clusters, method_kwargs, out_dir)
    _draw_latency_mm_cluster_matrix_with_markers(trace_store, example_group_idx, clusters,
                                                 latency_mm_df, out_dir)
    _draw_latency_mm_histograms(latency_mm_df, out_dir)
    _draw_latency_mm_cross_method_comparison(latency_mm_df, out_dir)

    # ── one scatter + regression figure per method ──────────────────────────
    for _, row in corr_df.iterrows():
        method_name = row["method"]
        sub = latency_mm_df[latency_mm_df["method"] == method_name].sort_values("cluster_label")
        x = sub["cluster_label"].values.astype(float)
        y = sub["latency_s"].values.astype(float) * 1000  # ms for display
        valid_mask = np.isfinite(x) & np.isfinite(y)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.scatter(x[valid_mask], y[valid_mask], color="steelblue", s=25, zorder=3)
        if (~valid_mask).any():
            ax.scatter(x[~valid_mask], np.zeros(np.sum(~valid_mask)), marker="x",
                       color="lightgrey", s=25, zorder=2, label="no valid latency")
        if np.isfinite(row["slope"]):
            x_line = np.linspace(x[valid_mask].min(), x[valid_mask].max(), 100)
            y_line = (row["slope"] * x_line + row["intercept"]) * 1000
            ax.plot(x_line, y_line, color="crimson", lw=1.5, zorder=4,
                    label=f"OLS fit (slope={row['slope']*1000:.2f} ms/cluster)")
        ax.set_xlabel("cluster_label (rastermap sorting index)")
        ax.set_ylabel("Cluster-mean latency (ms)")
        sig_tag = " *" if row["significant"] else ""
        ax.set_title(f"{method_name}: latency vs. sorting index{sig_tag}\n"
                     f"Spearman ρ={row['rho']:.3f}, p={row['p_raw']:.3g} "
                     f"(p_fdr={row['p_fdr']:.3g}, n={row['n_valid']})", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        _save(fig, out_dir / f"latency_vs_sorting_{method_name}", dpi=300)

    n_sig = int(reject.sum())
    print(f"  {n_sig}/{len(corr_df)} methods show a significant latency-vs-sorting-index "
          f"correlation (BH-FDR, α={cfg['fdr_alpha']})")
    print(f"  Done. Summary saved to latency_multimethod_vs_sorting_correlation.csv")
    return corr_df


# ══════════════════════════════════════════════════════════════════════════
# 2. Meta-clustering
# ══════════════════════════════════════════════════════════════════════════

def _build_neuron_matrix(cluster_df, cond_labels, psth, unit_ids):
    """Concatenate psth[label] for every condition into one neuron x time
    matrix (X_full), row-aligned with a filtered copy of cluster_df
    (sub_cluster_df) restricted to neurons actually present in the PSTH
    file. Also returns n_bins_list (per-condition segment length, in
    cond_labels order) so callers can slice X_full back into per-condition
    segments. Shared by run_meta_clustering and _fig_cluster_trialtype_grid
    so both operate on the exact same neuron ordering."""
    uid_to_row = {uid: i for i, uid in enumerate(unit_ids)}
    row_idx_for_uid = np.array([uid_to_row.get(uid, -1) for uid in cluster_df["unit_ids"]])
    valid = row_idx_for_uid >= 0
    if not valid.all():
        print(f"  [warn] {(~valid).sum()} cluster-table neurons not found in PSTH file, dropping")
    sub_cluster_df = cluster_df[valid].reset_index(drop=True)
    row_idx_for_uid = row_idx_for_uid[valid]
    X_full = np.concatenate([psth[label][row_idx_for_uid] for label in cond_labels], axis=1)
    n_bins_list = [psth[label].shape[1] for label in cond_labels]
    return X_full, sub_cluster_df, n_bins_list


def _resolve_trialtype_row_groups(cond_labels, cfg):
    """Returns (row_groups, row_titles): row_groups is a list of lists of
    indices into cond_labels (one sub-list per figure row); row_titles is
    the parallel list of row labels.

    Default (cfg['trialtype_row_groups'] is None): a single row containing
    every condition, in npz order. If cfg['trialtype_row_groups'] is a dict
    {row_label: [cond_label, ...]}, each key becomes a row (dict order
    preserved) mapped to indices via an exact cond_labels string match
    (mirrors the row_groups convention in
    _fig_sigclusters_by_trialtype_by_rewardgroup, e.g. passive / active-start
    / active-lick). Any cond_label not claimed by a named group is appended
    as a final 'other' row so nothing silently disappears from the grid.
    """
    custom = cfg.get("trialtype_row_groups")
    if not custom:
        return [list(range(len(cond_labels)))], ["all conditions"]

    label_to_idx = {label: i for i, label in enumerate(cond_labels)}
    row_groups, row_titles, used = [], [], set()
    for row_label, labels in custom.items():
        idxs = []
        for lab in labels:
            if lab not in label_to_idx:
                print(f"  [warn] trialtype_row_groups['{row_label}']: condition "
                      f"'{lab}' not found in cond_labels — skipping")
                continue
            idxs.append(label_to_idx[lab])
            used.add(lab)
        if idxs:
            row_groups.append(idxs)
            row_titles.append(row_label)
    leftover = [i for i, lab in enumerate(cond_labels) if lab not in used]
    if leftover:
        row_groups.append(leftover)
        row_titles.append("other")
    return row_groups, row_titles


def _wrap_trialtype_row_groups(row_groups, row_titles, max_cols):
    """Splits any row with more than max_cols conditions into several
    consecutive figure rows of at most max_cols columns each, so a long
    condition list (or a coarse 'all conditions' default row) doesn't
    produce one very wide row. Continuation rows repeat the same
    conditions-group but get an empty title (row-title annotation is only
    drawn once, on the first chunk) so the grid reads as one logical block
    spanning several rows rather than several separately-labeled groups.
    Pass max_cols=None or 0 to disable wrapping and return the input as-is.
    """
    if not max_cols:
        return row_groups, row_titles
    wrapped_groups, wrapped_titles = [], []
    for idxs, title in zip(row_groups, row_titles):
        if len(idxs) <= max_cols:
            wrapped_groups.append(idxs)
            wrapped_titles.append(title)
            continue
        for start in range(0, len(idxs), max_cols):
            chunk = idxs[start: start + max_cols]
            wrapped_groups.append(chunk)
            wrapped_titles.append(title if start == 0 else "")
    return wrapped_groups, wrapped_titles


def _draw_one_trialtype_grid(X, neurons_k, mouse_arr, reward_arr, group_id,
                              slices, t_ctrs, cond_labels, row_groups, row_titles,
                              edge_trim, group_colors, split_tag, out_dir,
                              n_rows, n_cols, group_label="Cluster", file_prefix="cluster"):
    """Draws and saves one grid figure (rows = row_groups, cols = conditions
    within that row) for a single group of neurons (a rastermap cluster, or
    a coarser meta-clustering family — see group_label/file_prefix).
    group_colors == {"all": color} produces the pooled (not color-coded)
    variant; group_colors == {"R+": ..., "R-": ...} produces the
    reward-group-split variant, both following the same two-pass
    compute-then-draw structure (and the same edge-trim / shared-y-axis /
    per-axis-label conventions) as the original
    _fig_sigclusters_by_trialtype_by_rewardgroup.
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.set_box_aspect(1)

    # ── first pass: compute traces for each group + a shared y-range ───────
    traces = {}   # (row_idx, col_idx) -> group -> (t_tr, mean, sem)
    n_per_group, mice_per_group = {}, {}
    y_min, y_max = np.inf, -np.inf
    for group, color in group_colors.items():
        grp_mask = neurons_k if group == "all" else (neurons_k & (reward_arr == group))
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
                seg  = X_grp[:, s + trim: e - trim]
                t_tr = t[trim: len(t) - trim] if trim else t
                mean = seg.mean(axis=0)
                sem  = seg.std(axis=0, ddof=1) / np.sqrt(n_grp) if n_grp > 1 else np.zeros_like(mean)
                traces.setdefault((row_idx, col_idx), {})[group] = (t_tr, mean, sem)
                y_min = min(y_min, (mean - sem).min())
                y_max = max(y_max, (mean + sem).max())

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        plt.close(fig)
        return
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    y_min, y_max = y_min - pad, y_max + pad

    # ── second pass: draw ────────────────────────────────────────────────
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
                t_tr, mean, sem = cell[group]
                label = "all neurons" if group == "all" else f"{group} (n={n_per_group[group]})"
                ax.plot(t_tr, mean, color=color, lw=1.5, label=label)
                ax.fill_between(t_tr, mean - sem, mean + sem, color=color, alpha=0.2)
            ax.axvline(0, ls="--", color="gray", lw=0.8)
            ax.set_ylim(y_min, y_max)
            ax.set_title(cond_labels[ci], fontsize=9)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("z-score (mean ± SEM)")
            ax.legend(fontsize=7, frameon=False)
        if row_titles[row_idx] not in ("all conditions", ""):
            axes[row_idx, 0].annotate(row_titles[row_idx], xy=(-0.35, 0.5),
                                      xycoords="axes fraction", rotation=90,
                                      ha="center", va="center", fontsize=9, fontweight="bold")

    if "R+" in group_colors:
        title_counts = (f"R+={n_per_group.get('R+', 0)} neu / {mice_per_group.get('R+', 0)} mice, "
                        f"R-={n_per_group.get('R-', 0)} neu / {mice_per_group.get('R-', 0)} mice")
    else:
        title_counts = f"n={n_per_group.get('all', 0)} neu / {mice_per_group.get('all', 0)} mice"
    fig.suptitle(f"{group_label} {group_id}  ({title_counts})")
    fig.tight_layout()
    fig.savefig(out_dir / f"{file_prefix}_{group_id:03d}_{split_tag}.png", dpi=150)
    plt.close(fig)


def _fig_cluster_trialtype_grid(X, sub_cluster_df, n_bins_list, cond_labels, t_ctr, cfg, out_dir,
                                 group_values=None, group_label="Cluster", file_prefix="cluster"):
    """For every group, saves two grid figures — one pooling all neurons in
    the group (not color-coded), one splitting R+ vs R- (color-coded,
    overlaid mean ± SEM per condition) — laid out as rows = row_groups
    (default: a single row with all conditions; configurable via
    cfg['trialtype_row_groups']) x cols = conditions within that row.
    Requires a 'reward_group' column in sub_cluster_df for the split
    variant; the pooled variant is produced regardless.

    group_values : array aligned with sub_cluster_df rows assigning each
        neuron to a group. Defaults to sub_cluster_df['cluster_label'] (the
        raw rastermap clusters) if not given — pass the meta-clustering
        family assignment instead to plot the merged (dendrogram-cut)
        groups, which is what run_meta_clustering does by default.
    group_label / file_prefix : used in figure titles / filenames
        ("Cluster"/"cluster" vs. "Family"/"family").
    """
    if group_values is None:
        group_values = sub_cluster_df["cluster_label"].values
    group_values = np.asarray(group_values)
    mouse_arr = sub_cluster_df["mouse_id"].values
    groups = np.sort(np.unique(group_values))

    row_groups, row_titles = _resolve_trialtype_row_groups(cond_labels, cfg)
    row_groups, row_titles = _wrap_trialtype_row_groups(
        row_groups, row_titles, cfg.get("trialtype_max_cols_per_row"))
    t_ctrs = [t_ctr[label] for label in cond_labels]
    starts = np.cumsum([0] + list(n_bins_list[:-1]))
    ends   = np.cumsum(n_bins_list)
    slices = list(zip(starts, ends))
    n_rows = len(row_groups)
    n_cols = max(len(r) for r in row_groups)
    edge_trim = cfg["trialtype_edge_trim_bins"]

    pooled_dir = out_dir / f"{file_prefix}_trialtype_pooled"
    pooled_dir.mkdir(parents=True, exist_ok=True)
    pooled_colors = {"all": cfg["trialtype_pooled_color"]}
    for k in groups:
        _draw_one_trialtype_grid(X, group_values == k, mouse_arr, None, k, slices, t_ctrs,
                                  cond_labels, row_groups, row_titles, edge_trim,
                                  pooled_colors, "pooled", pooled_dir, n_rows, n_cols,
                                  group_label=group_label, file_prefix=file_prefix)
    print(f"  Saved per-{group_label.lower()} trial-type grids (pooled, {len(groups)} "
          f"{group_label.lower()}(s)) → {pooled_dir.name}/")

    if "reward_group" not in sub_cluster_df.columns:
        print("  [warn] 'reward_group' not found in cluster table — skipping "
              "reward-group-split trial-type grids")
        return

    reward_arr = sub_cluster_df["reward_group"].values
    split_dir = out_dir / f"{file_prefix}_trialtype_by_rewardgroup"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_colors = {"R+": cfg["rplus_color"], "R-": cfg["rminus_color"]}
    for k in groups:
        _draw_one_trialtype_grid(X, group_values == k, mouse_arr, reward_arr, k, slices, t_ctrs,
                                  cond_labels, row_groups, row_titles, edge_trim,
                                  split_colors, "by_rewardgroup", split_dir, n_rows, n_cols,
                                  group_label=group_label, file_prefix=file_prefix)
    print(f"  Saved per-{group_label.lower()} trial-type grids (by reward group, {len(groups)} "
          f"{group_label.lower()}(s)) → {split_dir.name}/")


import colorsys

def deterministic_palette(n, hue_start=0.13, s=0.75, v=0.85):
    """Deterministic, vivid, maximally-separated palette of n colors.
    Golden-ratio hue stepping guarantees near-uniform hue spacing (the
    standard trick for perceptually distinct categorical colors), and a
    fixed starting hue makes it reproducible — same n, same colors, always.
    s/v fixed at moderately high, non-pastel values so nothing looks washed
    out or neon; tune s/v together if you want more/less saturation overall."""
    golden_ratio_conj = 0.618033988749895
    return [colorsys.hsv_to_rgb((hue_start + i * golden_ratio_conj) % 1.0, s, v)
            for i in range(n)]


def metacluster_color_map(metacluster_ids, hue_start=0.13, s=0.75, v=0.85):
    """{metacluster_id: hex_color}. Sorts ids first, so the mapping depends
    only on each id's RANK among the full id set — passing the same ids in
    a different order still reproduces identical colors."""
    ids_sorted = sorted(set(metacluster_ids))
    palette = deterministic_palette(len(ids_sorted), hue_start, s, v)
    to_hex = lambda rgb: '#%02x%02x%02x' % tuple(int(round(c * 255)) for c in rgb)
    return {mid: to_hex(c) for mid, c in zip(ids_sorted, palette)}

def build_link_color_func(Z, leaf_metacluster, palette, default_color="grey"):
    """leaf_metacluster must be in Z's original observation order (same
    order as the rows Z was built from), NOT the dendrogram's display order.
    A merge node is colored by its metacluster only if every leaf beneath
    it shares one; any node spanning multiple metaclusters gets default_color."""
    node_cluster = list(leaf_metacluster)
    for left, right, dist, count in Z:
        left, right = int(left), int(right)
        cl, cr = node_cluster[left], node_cluster[right]
        node_cluster.append(cl if cl == cr else None)

    def link_color_func(node_id):
        cl = node_cluster[node_id]
        return palette.get(cl, default_color) if cl is not None else default_color

    return link_color_func

def run_meta_clustering(cluster_df, cond_labels, psth, t_ctr, unit_ids, cfg, out_dir):
    out_dir.mkdir(exist_ok=True)
    print("\n[2] Meta-clustering (cluster-of-clusters)")

    X_full, sub_cluster_df, n_bins_list = _build_neuron_matrix(cluster_df, cond_labels, psth, unit_ids)
    clusters = np.sort(sub_cluster_df["cluster_label"].unique())
    cluster_means = np.vstack([
        X_full[sub_cluster_df["cluster_label"].values == c].mean(axis=0)
        for c in clusters
    ])

    # pairwise correlation -> distance
    #corr_mat = np.corrcoef(cluster_means)
    #dist_mat = 1 - corr_mat
    #np.fill_diagonal(dist_mat, 0.0)
    #dist_mat = np.clip(dist_mat, 0, None)  # guard tiny negative floating noise
    #condensed = squareform(dist_mat, checks=False)
    condensed = pdist(cluster_means, metric="correlation")  # same as 1 - Pearson r
    condensed = np.clip(condensed, 0, None)  # still guard tiny negative floating noise

    # ── linkage method: validate/select among distance-agnostic methods ────
    # Ward/centroid/median assume true Euclidean coordinates and silently
    # distort the tree on a non-Euclidean (1-r) distance — only these four
    # are valid here.
    valid_methods = ["single", "complete", "average", "weighted"]
    method_coph = {}
    for m in valid_methods:
        Z_m = linkage(condensed, method=m)
        c, coph_d = cophenet(Z_m, condensed)
        method_coph[m] = c
    print("  Cophenetic correlation by linkage method:",
          {m: f"{c:.3f}" for m, c in method_coph.items()})

    method = cfg["meta_linkage_method"]
    if method == "auto":
        method = max(method_coph, key=method_coph.get)
        print(f"  meta_linkage_method='auto' → selected '{method}'")
    elif method not in valid_methods:
        print(f"  Warning: '{method}' isn't valid for a non-Euclidean (1-r) distance — "
              f"Ward/centroid/median assume true Euclidean coordinates")

    Z = linkage(condensed, method=method)
    coph_corr, coph_dists = cophenet(Z, condensed)
    print(f"  Cophenetic correlation coefficient: {coph_corr:.3f} "
          f"(closer to 1 = dendrogram faithfully represents pairwise distances)")

    # ── cut distance: validate/select via silhouette sweep ──────────────────
    cut_dist = cfg["meta_dendro_cut_dist"]
    if cut_dist == "auto":
        thresholds = np.linspace(Z[:, 2].min(), Z[:, 2].max(), 60)[1:-1]
        sil_scores = []
        for t in thresholds:
            labels_t = fcluster(Z, t=t, criterion="distance")
            k = len(np.unique(labels_t))
            sil_scores.append(silhouette_score(squareform(condensed), labels_t, metric="precomputed")
                              if 1 < k < len(labels_t) else np.nan)
        cut_dist = thresholds[np.nanargmax(sil_scores)]
        print(f"  meta_dendro_cut_dist='auto' → selected {cut_dist:.3f} "
              f"(silhouette={np.nanmax(sil_scores):.3f})")

    family_labels = fcluster(Z, t=cut_dist, criterion="distance")
    families_df = pd.DataFrame({"cluster_label": clusters, "family": family_labels})
    families_df.to_csv(out_dir / "cluster_families.csv", index=False)

    family_sizes = families_df["family"].value_counts().sort_index()
    n_singletons = (family_sizes == 1).sum()
    print(f"  {family_labels.max()} families found at cut distance "
          f"{cut_dist:.3f} ({n_singletons} singleton families)")

    # ── quality/intermediate outputs ────────────────────────────────────────
    corr_mat = 1 - squareform(condensed)
    np.fill_diagonal(corr_mat, 1.0)
    pd.DataFrame(corr_mat, index=clusters, columns=clusters).to_csv(
        out_dir / "cluster_correlation_matrix.csv")

    with open(out_dir / "meta_clustering_diagnostics.txt", "w") as fh:
        fh.write(f"Similarity measure: Pearson correlation (distance = 1 - r)\n")
        fh.write(f"Linkage method: {cfg['meta_linkage_method']}\n")
        fh.write(f"Cut distance: {cfg['meta_dendro_cut_dist']}\n")
        fh.write(f"Cophenetic correlation coefficient: {coph_corr:.4f}\n")
        fh.write(f"Number of families: {family_labels.max()}\n")
        fh.write(f"Singleton families: {n_singletons}\n")

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_box_aspect(aspect=1)
    ax.bar(family_sizes.index.astype(str), family_sizes.values, color="k")
    xticks = family_sizes.index.astype(str)
    # label every 5 ticks
    xticklabels = [xticks[i] if i % 5 == 0 else "" for i in range(len(xticks))]
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Meta-cluster")
    ax.set_ylabel("Number of original clusters")
    ax.set_title(f"Meta-cluster sizes (n={family_labels.max()} families, {n_singletons} singletons)")
    fig.tight_layout()
    _save(fig, out_dir / "qc_family_sizes", dpi=300)

    # ── STEP 1: raw input — cluster-mean PSTH matrix, no clustering applied yet ──
    cond_lengths = [psth[label].shape[1] for label in cond_labels]
    offsets      = np.concatenate([[0], np.cumsum(cond_lengths)])
    tick_pos     = [(offsets[i] + offsets[i + 1]) / 2 for i in range(len(cond_labels))]

    def _draw_cluster_matrix(ax, matrix, row_labels, title):
        vmax = np.nanpercentile(np.abs(matrix), 95)
        vmax = vmax if vmax > 0 else 1.0
        im = ax.imshow(matrix, aspect="auto", interpolation="none", cmap="bwr",
                       vmin=-vmax, vmax=vmax, extent=[0, offsets[-1], len(row_labels), 0])
        for start in offsets[1:-1]:
            ax.axvline(start, color="k", lw=0.8)
        ax.set_yticks(np.arange(len(row_labels)) + 0.5)
        # label every 20 ticks only
        row_labels = [str(row_labels[i]) if i % 20 == 0 else "" for i in range(len(row_labels))]
        ax.set_yticklabels(row_labels, fontsize=6)
        #ax.set_ylabel("cluster_label")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(cond_labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(title)
        return im

    fig, ax = plt.subplots(figsize=(9, max(6, len(clusters) * 0.15)))
    im = _draw_cluster_matrix(ax, cluster_means, clusters,
                              "Input: cluster-mean PSTH matrix\n"
                              "(cluster_label order, before any clustering)")
    fig.colorbar(im, ax=ax, label="Firing rate (z-score)", shrink=0.2, pad=0.02, aspect=20)
    fig.tight_layout()
    _save(fig, out_dir / "qc_input_matrix_original_order", dpi=300)

    # ── STEP 2: pairwise similarity (correlation matrix) ─────────────────
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(corr_mat, cmap="bwr", vmin=-1, vmax=1)
    ax.set_title("Pairwise similarity (original order)")
    ax.set_xlabel("Cluster index"); ax.set_ylabel("Cluster index")
    fig.colorbar(im, ax=ax, label=r"Pearson's $r$", shrink=0.3, pad= 0.01, ticks=[-1,0,1])
    fig.tight_layout()
    _save(fig, out_dir / "qc_correlation_matrix_original_order", dpi=300)

    # ── STEP 3: dendrogram (from linkage on 1-r distance) next to the ────
    # INPUT matrix, in its ORIGINAL cluster_label order. The tree's leaves
    # are NOT in this order, so rows here are NOT vertically aligned with the
    # dendrogram — this view is for comparing "before" layout against the
    # tree structure, not for reading off which family a given row belongs to.
    def _draw_dendrogram_panel(ax, row_labels_for_ticks):
        leaf_metacluster = fcluster(Z, t=cfg["meta_dendro_cut_dist"], criterion="distance")
        palette = metacluster_color_map(np.unique(leaf_metacluster))
        lcf = build_link_color_func(Z, leaf_metacluster, palette)
        dn = dendrogram(Z, orientation="left",
                        #labels=[str(c) for c in row_labels_for_ticks],
                        color_threshold=cfg["meta_dendro_cut_dist"],
                        ax=ax,
                        link_color_func=lcf,
                        above_threshold_color='grey',
                        no_labels=True,
        )
        ax.axvline(cfg["meta_dendro_cut_dist"], color="k", ls="--", lw=1.2)
        ax.set_xlabel("1 − correlation distance")
        ax.set_title("Dendrogram")
        return dn, leaf_metacluster, palette

    fig = plt.figure(figsize=(12, max(6, len(clusters) * 0.18)))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 3], wspace=0.05)
    ax_dendro = fig.add_subplot(gs[0])
    ax_heat   = fig.add_subplot(gs[1])
    _draw_dendrogram_panel(ax_dendro, clusters)
    im = _draw_cluster_matrix(ax_heat, cluster_means, clusters,
                              "Dendrogram next to INPUT matrix\n"
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
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.04, 3], wspace=0.02)
    ax_dendro = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1])  # ← sharey removed
    ax_heat = fig.add_subplot(gs[2])

    dn, leaf_metacluster, palette = _draw_dendrogram_panel(ax_dendro, clusters)
    leaf_order = dn["leaves"]
    leaf_order = dn["leaves"][::-1]  # scipy draws leaves[0] at the BOTTOM for orientation="left";

    reordered_clusters = np.asarray(clusters)[leaf_order]
    reordered_means = np.asarray(cluster_means)[leaf_order]

    reordered_clusters = np.asarray(clusters)[leaf_order]
    reordered_means = np.asarray(cluster_means)[leaf_order]

    strip_rgb = np.array([matplotlib.colors.to_rgb(palette[m])
                          for m in np.asarray(leaf_metacluster)[leaf_order]]).reshape(-1, 1, 3)
    ax_strip.imshow(strip_rgb, aspect="auto")
    ax_strip.set_xticks([]);
    ax_strip.set_yticks([])

    im = _draw_cluster_matrix(ax_heat, reordered_means, reordered_clusters,
                              f"Reordered matrix (cophenetic r={coph_corr:.2f})\n"
                              "(rows aligned to dendrogram leaf order)")
    fig.colorbar(im, ax=ax_heat, label="Firing rate (z-score)", shrink=0.2, pad=0.02, aspect=20)
    _save(fig, out_dir / "meta_clustering_dendrogram_matrix_reordered", dpi=300)


    # ── per-family trial-type grids (pooled + reward-group split) ──────────
    # Grouped by the merged meta-clustering family (dendrogram cut), NOT the
    # raw rastermap cluster_label — there should be far fewer of these, and
    # each one summarizes a whole family of similarly-shaped clusters.
    family_map = dict(zip(families_df["cluster_label"], families_df["family"]))
    family_per_neuron = sub_cluster_df["cluster_label"].map(family_map).values
    _fig_cluster_trialtype_grid(X_full, sub_cluster_df, n_bins_list, cond_labels, t_ctr,
                                cfg, out_dir, group_values=family_per_neuron,
                                group_label="Family", file_prefix="family")

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
        run_cluster_mean_latency_vs_sorting(
            cluster_df, cond_labels, psth, t_ctr, unit_ids, cfg,
            run_dir / "01b_latency_multimethod_vs_sorting")

    if cfg["run_meta_clustering"]:
        families_df = run_meta_clustering(cluster_df, cond_labels, psth, t_ctr, unit_ids, cfg,
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