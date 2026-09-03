"""
cluster_comparison.py
─────────────────────
Compare any two clustering results (rastermap, kmeans, or gmm) that were
produced by run_clustering.py on the same neuron population.

Usage
-----
    # From Python
    from cluster_comparison import run_cluster_comparison
    run_cluster_comparison(out_folder, method_a="rastermap", method_b="kmeans")
    run_cluster_comparison(out_folder, method_a="rastermap", method_b="gmm", cv=False)

    # CLI
    python cluster_comparison.py <out_folder> [--method_a rastermap] [--method_b kmeans]
                                              [--label_a Rastermap] [--label_b KMeans]
                                              [--global_mode]

Required .npz fields (same for all methods — standard format)
──────────────────────────────────────────────────────────────
    X              : (n_neurons, total_bins)   feature matrix
    cluster_labels : (n_neurons,)              integer assignments
    n_bins_list    : (n_conds,)
    t_ctr_0 …      : per-condition time axes
    n_conds        : scalar

Outputs  (written to <out_folder>/comparison_<method_a>_vs_<method_b>[_cv]/)
─────────────────────────────────────────────────────────────────────────────
    fig1_cluster_sizes.png/pdf/svg
    fig2_mean_psth_heatmaps.png/pdf/svg
    fig3_raw_corr_matrix.png/pdf/svg
    fig4_matched_r_distribution.png/pdf/svg
    fig5_confusion_matrix.png/pdf/svg
    fig6_summary.png/pdf/svg
    fig7_row_coloring.png/pdf/svg
    fig8_aligned_population_matrix.png/pdf/svg
    cluster_comparison.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})
DPI = 200


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_results(folder: Path, npz_name: str):
    d       = dict(np.load(folder / npz_name, allow_pickle=True))
    n_conds = int(d["n_conds"])
    t_ctrs  = [d[f"t_ctr_{c}"] for c in range(n_conds)]
    return (d["X"],
            d["cluster_labels"].astype(int),
            d["n_bins_list"].astype(int),
            t_ctrs)


def _save(fig, path):
    """Save figure as PNG, PDF and SVG."""
    path = Path(path)
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(path.with_suffix(ext), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.stem}  [png | pdf | svg]")


# ── metrics ───────────────────────────────────────────────────────────────────

def cluster_means(X, labels):
    """(k, n_bins) mean PSTH matrix; noise label -1 skipped."""
    ids = np.unique(labels[labels >= 0])
    return np.vstack([X[labels == k].mean(0) for k in ids]), ids


def psth_corr_matrix(means_a, means_b):
    """Pearson r matrix (k_a x k_b)."""
    def _norm(m):
        m = m - m.mean(1, keepdims=True)
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    return _norm(means_a) @ _norm(means_b).T


def confusion_matrix_norm(labels_a, labels_b, ids_a, ids_b):
    """Row-normalised confusion matrix: C[i,j] = fraction of A-cluster-i in B-cluster-j."""
    k_a, k_b = len(ids_a), len(ids_b)
    C = np.zeros((k_a, k_b))
    for i, ia in enumerate(ids_a):
        mask = labels_a == ia
        for j, ib in enumerate(ids_b):
            C[i, j] = (labels_b[mask] == ib).sum()
        if C[i].sum() > 0:
            C[i] /= C[i].sum()
    return C


def compute_metrics(labels_a, labels_b, means_a, means_b, ids_a, ids_b):
    mask     = (labels_a >= 0) & (labels_b >= 0)
    nmi      = normalized_mutual_info_score(labels_a[mask], labels_b[mask])
    ari      = adjusted_rand_score(labels_a[mask], labels_b[mask])
    corr_mat = psth_corr_matrix(means_a, means_b)
    row, col = linear_sum_assignment(-corr_mat)
    conf_mat = confusion_matrix_norm(labels_a, labels_b, ids_a, ids_b)
    return dict(nmi=nmi, ari=ari,
                corr_mat=corr_mat, row=row, col=col,
                matched_r=corr_mat[row, col],
                conf_mat=conf_mat)


# ── figures ───────────────────────────────────────────────────────────────────

def fig1_cluster_sizes(labels_a, labels_b, ids_a, ids_b, label_a, label_b, out):
    """Step 1 — Are cluster sizes balanced and comparable between methods?"""
    sizes_a = np.array([(labels_a == k).sum() for k in ids_a])
    sizes_b = np.array([(labels_b == k).sum() for k in ids_b])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, sizes, ids, label, color in zip(
            axes, [sizes_a, sizes_b], [ids_a, ids_b],
            [label_a, label_b], ["#4C72B0", "#DD8452"]):
        ax.bar(np.arange(len(ids)), sizes, color=color, edgecolor="none")
        ax.set_xlabel("Cluster index")
        ax.set_ylabel("Neuron count")
        ax.set_title(f"{label}  (k={len(ids)},  "
                     f"median={int(np.median(sizes))},  "
                     f"noise={( labels_a if label == label_a else labels_b == -1).sum()})")
        ax.set_xticks(np.arange(len(ids)))

    fig.suptitle("Step 1 — Cluster size distributions", fontsize=10)
    fig.tight_layout()
    _save(fig, out)


def fig2_mean_psth_heatmaps(means_a, means_b, n_bins_list, label_a, label_b, out):
    """Step 2 — What does each method's cluster mean PSTH look like?"""
    offsets = np.concatenate([[0], np.cumsum(n_bins_list)])
    vmax    = np.nanpercentile(np.abs(np.vstack([means_a, means_b])), 95)
    norm    = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.18 * (len(means_a) + len(means_b)))))
    for ax, means, label in zip(axes, [means_a, means_b], [label_a, label_b]):
        im = ax.imshow(means, aspect="auto", interpolation="none",
                       cmap="coolwarm", norm=norm,
                       extent=[0, means.shape[1], means.shape[0], 0])
        for s in offsets[1:-1]:
            ax.axvline(s, color="k", lw=0.8)
        ax.set_xlabel("Bin (concatenated conditions)")
        ax.set_ylabel("Cluster")
        ax.set_title(f"{label}  (k={len(means)})")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="z-score")

    fig.suptitle("Step 2 — Per-cluster mean PSTH heatmaps", fontsize=10)
    fig.tight_layout()
    _save(fig, out)


def fig3_raw_corr_matrix(corr_mat, label_a, label_b, out):
    """Step 3 — Raw PSTH correlation matrix before matching.
    A clean block-diagonal indicates good agreement."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xlabel(f"{label_b} cluster")
    ax.set_ylabel(f"{label_a} cluster")
    ax.set_title(f"Step 3 — Raw PSTH correlation matrix\n"
                 f"{label_a} (k={corr_mat.shape[0]}) × "
                 f"{label_b} (k={corr_mat.shape[1]})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    _save(fig, out)


def fig4_matched_r_distribution(matched_r, row, col, label_a, label_b, out):
    """Step 4 — Distribution of matched Pearson r values (after Hungarian matching).
    Shows whether agreement is uniform or driven by a few well-matched clusters."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # histogram
    ax = axes[0]
    ax.hist(matched_r, bins=min(20, len(matched_r)), color="#4C72B0",
            edgecolor="white", linewidth=0.5)
    ax.axvline(matched_r.mean(),   color="crimson", lw=1.5,
               ls="--", label=f"mean={matched_r.mean():.2f}")
    ax.axvline(np.median(matched_r), color="orange", lw=1.5,
               ls="--", label=f"median={np.median(matched_r):.2f}")
    ax.set_xlabel("Matched PSTH Pearson r")
    ax.set_ylabel("Cluster pair count")
    ax.set_title("Matched r distribution")
    ax.legend(fontsize=7, frameon=False)

    # ranked bar
    ax = axes[1]
    order = np.argsort(matched_r)[::-1]
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(matched_r)))
    bars = ax.bar(np.arange(len(matched_r)), matched_r[order],
                  color=colors, edgecolor="none")
    ax.set_xlabel("Matched pair rank")
    ax.set_ylabel("Pearson r")
    ax.set_title("Ranked matched PSTH correlations")
    ax.axhline(0, color="k", lw=0.5)
    for i, (b, ri) in enumerate(zip(bars, order)):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + 0.01,
                f"A{row[ri]}↔B{col[ri]}", ha="center", va="bottom",
                fontsize=5, rotation=90)

    fig.suptitle("Step 4 — Matched pair Pearson r (Hungarian assignment)", fontsize=10)
    fig.tight_layout()
    _save(fig, out)


def fig5_confusion_matrix(conf_mat, row, col, ids_a, ids_b, label_a, label_b, out):
    """Step 5 — Row-normalised confusion matrix: fraction of A-cluster neurons in each B-cluster.
    Reordered by Hungarian matching. Diagonal dominance = neuron-level agreement."""
    reordered = conf_mat[:, col][row, :]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, mat, title in zip(
            axes,
            [conf_mat, reordered],
            ["Raw order", f"Reordered by Hungarian matching\n(rows: {label_a}, cols: {label_b})"]):
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xlabel(f"{label_b} cluster")
        ax.set_ylabel(f"{label_a} cluster")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Fraction of A-cluster neurons")

    fig.suptitle("Step 5 — Row-normalised confusion matrix (neuron assignment overlap)",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, out)


def fig7_row_coloring(X, labels_a, labels_b, ids_a, ids_b,
                      n_bins_list, label_a, label_b, out):
    """Step 7 — Population matrix sorted by A's cluster order, rows colored by B's labels.

    Left strip  : A's cluster identity (color per A-cluster).
    Right strip : B's cluster identity for the same neuron (color per B-cluster).
    Main panel  : mean-PSTH heatmap of X sorted by A's isort.

    Solid blocks in the right strip = A and B agree on that cluster.
    Mixed colors in the right strip within an A-cluster block = neurons that
    B assigns differently, revealing exactly where the two methods disagree.
    """
    from matplotlib.colors import BoundaryNorm
    import matplotlib as mpl

    # Sort neurons by A's cluster order
    isort_a = np.argsort(labels_a, kind="stable")
    sorted_labels_a = labels_a[isort_a]
    sorted_labels_b = labels_b[isort_a]

    k_a = len(ids_a)
    k_b = len(ids_b)

    # Discrete colormaps — one color per cluster index
    cmap_a = mpl.colormaps["tab20"].resampled(k_a)
    cmap_b = mpl.colormaps["tab20b"].resampled(k_b)

    # Color strips: shape (n_neurons, 1) with integer cluster index
    strip_a = sorted_labels_a[:, np.newaxis]   # A labels in A-sorted order
    strip_b = sorted_labels_b[:, np.newaxis]   # B labels in A-sorted order

    # A cluster boundary positions (for horizontal lines)
    boundaries_a = np.where(np.diff(sorted_labels_a) != 0)[0] + 1

    vmax = np.nanpercentile(np.abs(X), 95)
    norm_psth = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    # Layout: [A strip | B strip | PSTH matrix | colorbars]
    fig = plt.figure(figsize=(14, max(5, 0.015 * len(isort_a))))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.03,
                            width_ratios=[0.015, 0.015, 0.92, 0.05])

    ax_a    = fig.add_subplot(gs[0])   # A cluster strip
    ax_b    = fig.add_subplot(gs[1])   # B cluster strip
    ax_mat  = fig.add_subplot(gs[2])   # PSTH matrix
    ax_cbar = fig.add_subplot(gs[3])   # colorbar for PSTH

    n_neurons = len(isort_a)
    offsets   = np.concatenate([[0], np.cumsum(n_bins_list)])

    # PSTH matrix
    im = ax_mat.imshow(X[isort_a], aspect="auto", interpolation="none",
                       cmap="coolwarm", norm=norm_psth,
                       extent=[0, X.shape[1], n_neurons, 0])
    for s in offsets[1:-1]:
        ax_mat.axvline(s, color="k", lw=0.8)
    for b in boundaries_a:
        ax_mat.axhline(b, color="white", lw=0.3, alpha=0.5)
    ax_mat.set_xlabel("Bin (concatenated conditions)")
    ax_mat.set_yticks([])
    ax_mat.set_title(f"Population matrix sorted by {label_a} cluster order", fontsize=9)
    plt.colorbar(im, cax=ax_cbar, label="z-score")

    # A strip
    ax_a.imshow(strip_a, aspect="auto", interpolation="none",
                cmap=cmap_a, vmin=-0.5, vmax=k_a - 0.5,
                extent=[0, 1, n_neurons, 0])
    for b in boundaries_a:
        ax_a.axhline(b, color="white", lw=0.3)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.set_ylabel("Neuron (sorted by A)", fontsize=7)
    ax_a.set_title(label_a, fontsize=7, rotation=90, va="bottom")

    # B strip — same neuron order, B's label per neuron
    # noise neurons (label -1) shown in grey
    strip_b_plot = np.where(strip_b >= 0, strip_b, k_b)   # remap -1 to last slot

    cmap_b_ext = plt.get_cmap("tab20b", k_b + 1)
    ax_b.imshow(strip_b_plot, aspect="auto", interpolation="none",
                cmap=cmap_b_ext, vmin=-0.5, vmax=k_b + 0.5,
                extent=[0, 1, n_neurons, 0])
    for b in boundaries_a:
        ax_b.axhline(b, color="white", lw=0.3)
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    ax_b.set_title(label_b, fontsize=7, rotation=90, va="bottom")

    # Cluster label annotations on A strip (centred in each block)
    edges_a = np.concatenate([[0], boundaries_a, [n_neurons]])
    for ki, (lo, hi) in enumerate(zip(edges_a[:-1], edges_a[1:])):
        ax_a.text(0.5, (lo + hi) / 2, str(ki),
                  ha="center", va="center", fontsize=5,
                  color="white", fontweight="bold",
                  transform=ax_a.get_yaxis_transform())

    fig.suptitle(
        f"Step 7 — {label_a} matrix rows colored by {label_b} cluster membership\n"
        f"Solid blocks in '{label_b}' strip = agreement; "
        f"mixed colors = neurons assigned differently by {label_b}",
        fontsize=9)
    _save(fig, out)


def fig6_summary(metrics, means_a, means_b, t_ctrs, n_bins_list,
                 label_a, label_b, out, top_n=4):
    """Step 6 — Summary metrics + best and worst matched cluster PSTH pairs."""
    offsets   = np.concatenate([[0], np.cumsum(n_bins_list)])
    matched_r = metrics["matched_r"]
    row, col  = metrics["row"], metrics["col"]
    order     = np.argsort(matched_r)[::-1]
    top       = order[:top_n]
    bottom    = order[-top_n:][::-1]

    fig = plt.figure(figsize=(2.8 * (top_n + 1), 10))
    gs  = gridspec.GridSpec(3, top_n + 1, figure=fig,
                            hspace=0.55, wspace=0.4,
                            height_ratios=[1.2, 1, 1])

    # ── summary text ─────────────────────────────────────────────────────
    ax_txt = fig.add_subplot(gs[0, :2])
    ax_txt.axis("off")
    lines = [
        f"{'Method A:':<22} {label_a}  (k={len(means_a)})",
        f"{'Method B:':<22} {label_b}  (k={len(means_b)})",
        "",
        f"{'NMI:':<22} {metrics['nmi']:.3f}",
        f"{'ARI:':<22} {metrics['ari']:.3f}",
        f"{'Mean matched r:':<22} {matched_r.mean():.3f}",
        f"{'Median matched r:':<22} {np.median(matched_r):.3f}",
        f"{'Pairs r > 0.8:':<22} {(matched_r > 0.8).sum()} / {len(matched_r)}",
        f"{'Pairs r < 0.3:':<22} {(matched_r < 0.3).sum()} / {len(matched_r)}",
    ]
    ax_txt.text(0.05, 0.95, "\n".join(lines), va="top", ha="left",
                fontsize=9, transform=ax_txt.transAxes, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#cccccc"))

    # ── reordered corr heatmap (square) ──────────────────────────────────
    ax_heat = fig.add_subplot(gs[0, 2:])
    reordered = metrics["corr_mat"][:, col][row, :]
    im = ax_heat.imshow(reordered, vmin=-1, vmax=1, cmap="coolwarm", aspect="equal")
    ax_heat.set_xlabel(f"{label_b} (matched)", fontsize=8)
    ax_heat.set_ylabel(label_a, fontsize=8)
    ax_heat.set_title("PSTH correlation (matched)", fontsize=8)
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="r")

    # x-axis for concatenated PSTH: one continuous index across all bins
    total_bins = sum(n_bins_list)
    x_concat   = np.arange(total_bins)
    # vertical lines at condition boundaries
    boundaries_x = np.cumsum(n_bins_list[:-1])

    def _plot_pair(ax, ri, ci, r_val, title_prefix):
        ax.plot(x_concat, means_a[ri], lw=1.2, color="#4C72B0", label=label_a)
        ax.plot(x_concat, means_b[ci], lw=1.2, ls="--", color="#DD8452", label=label_b)
        for bx in boundaries_x:
            ax.axvline(bx, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.axhline(0, color="k", lw=0.3, ls=":")
        ax.set_title(f"{title_prefix}\nA{ri}↔B{ci}  r={r_val:.2f}", fontsize=7)
        ax.set_xlabel("Bin (concat. conditions)", fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_xlim(0, total_bins - 1)

    # ── top matched pairs ─────────────────────────────────────────────────
    for i in range(top_n + 1):
        ax = fig.add_subplot(gs[1, i])
        if i == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, f"Best matched\npairs\n(highest r)",
                    ha="center", va="center", fontsize=9,
                    transform=ax.transAxes, color="#4C72B0")
        elif i - 1 < len(top):
            ri, ci = row[top[i - 1]], col[top[i - 1]]
            _plot_pair(ax, ri, ci, matched_r[top[i - 1]], f"Top {i}")
            if i == 1:
                ax.set_ylabel("z-score", fontsize=6)
                ax.legend(fontsize=5, frameon=False)

    # ── worst matched pairs ───────────────────────────────────────────────
    for i in range(top_n + 1):
        ax = fig.add_subplot(gs[2, i])
        if i == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, f"Worst matched\npairs\n(lowest r)",
                    ha="center", va="center", fontsize=9,
                    transform=ax.transAxes, color="crimson")
        elif i - 1 < len(bottom):
            ri, ci = row[bottom[i - 1]], col[bottom[i - 1]]
            _plot_pair(ax, ri, ci, matched_r[bottom[i - 1]], f"Bottom {i}")
            if i == 1:
                ax.set_ylabel("z-score", fontsize=6)

    fig.suptitle(f"Step 6 — Summary: {label_a} vs {label_b}", fontsize=11)
    _save(fig, out)


def fig8_aligned_population_matrix(
        X_a, X_b,
        labels_a, labels_b,
        ids_a, ids_b,
        metrics,
        n_bins_list,
        label_a, label_b,
        out,
        vmax=None,
):
    """Population matrices of both methods with clusters aligned by Hungarian matching.

    Layout
    ------
    Two panels side by side sharing the same colour scale and y-axis scale.
    Method A (left) is sorted by its natural cluster order (0, 1, 2, …).
    Method B (right) is sorted so that cluster col[i] appears at the same
    vertical position as method A's cluster row[i] — i.e. matched pairs
    occupy the same row band in both panels.

    When k_a ≠ k_b, unmatched clusters are appended below the matched block,
    separated by a thicker boundary line and annotated.

    Cluster index labels are shown on the y-axis at each cluster centre.
    Vertical lines mark condition boundaries.  The matched-pair Pearson r is
    annotated on the right panel beside each matched cluster band.

    Parameters
    ----------
    X_a, X_b      : (N_neurons, T) feature matrices (should be identical since
                    both methods cluster the same neurons)
    labels_a/b    : (N,) integer cluster assignments for each method
    ids_a/b       : sorted unique cluster ids returned by cluster_means()
    metrics       : output of compute_metrics() — provides row, col, matched_r
    n_bins_list   : (n_conds,) number of time bins per condition
    vmax          : colour saturation; default = 95th percentile of |X_a|
    """
    row_idx = metrics["row"]   # indices into ids_a  (length = min(k_a, k_b))
    col_idx = metrics["col"]   # indices into ids_b
    matched_r = metrics["matched_r"]   # Pearson r per matched pair

    # ── 1. Build matched cluster ordering ─────────────────────────────────────
    # Sort pairs by method A cluster position so the A side reads in order.
    pair_sort     = np.argsort(row_idx)
    a_cluster_ord = ids_a[row_idx[pair_sort]]   # actual A cluster labels
    b_cluster_ord = ids_b[col_idx[pair_sort]]   # corresponding B cluster labels
    r_ord         = matched_r[pair_sort]         # r values in same order

    # Unmatched clusters (only non-empty when k_a != k_b)
    matched_a_set  = set(a_cluster_ord.tolist())
    matched_b_set  = set(b_cluster_ord.tolist())
    unmatched_a    = [k for k in ids_a if k not in matched_a_set]
    unmatched_b    = [k for k in ids_b if k not in matched_b_set]
    n_unmatched    = max(len(unmatched_a), len(unmatched_b))
    n_matched      = len(a_cluster_ord)

    full_a_ord = list(a_cluster_ord) + unmatched_a
    full_b_ord = list(b_cluster_ord) + unmatched_b

    # ── 2. Build isort and boundaries for each method ─────────────────────────
    def _isort(labels, cluster_order):
        parts = [np.where(labels == k)[0]
                 for k in cluster_order if (labels == k).any()]
        return np.concatenate(parts) if parts else np.arange(len(labels))

    def _boundaries(labels, cluster_order):
        sizes = [(labels == k).sum() for k in cluster_order
                 if (labels == k).any()]
        return np.cumsum(sizes)[:-1].tolist()

    isort_a  = _isort(labels_a, full_a_ord)
    isort_b  = _isort(labels_b, full_b_ord)
    bounds_a = _boundaries(labels_a, full_a_ord)
    bounds_b = _boundaries(labels_b, full_b_ord)

    # Position of matched/unmatched separator in each panel (neuron index)
    sep_a = int(sum((labels_a == k).sum() for k in a_cluster_ord
                    if (labels_a == k).any()))
    sep_b = int(sum((labels_b == k).sum() for k in b_cluster_ord
                    if (labels_b == k).any()))

    # ── 3. Cluster centre positions (for y-axis ticks and r annotations) ──────
    def _centres(labels, cluster_order):
        edges = [0] + _boundaries(labels, cluster_order) + [len(labels)]
        return [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]

    centres_a = _centres(labels_a, full_a_ord)
    centres_b = _centres(labels_b, full_b_ord)

    # ── 4. Colour scale ───────────────────────────────────────────────────────
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(X_a), 95))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = "RdBu_r"

    offsets   = np.concatenate([[0], np.cumsum(n_bins_list)])
    n_bins    = int(offsets[-1])
    n_neurons = len(isort_a)

    # ── 5. Figure layout ──────────────────────────────────────────────────────
    fig_h = max(8, n_neurons // 80)
    fig, axes = plt.subplots(
        1, 2, figsize=(22, fig_h),
        gridspec_kw={"wspace": 0.06})

    panels = [
        (axes[0], X_a, isort_a, bounds_a, sep_a, centres_a,
         full_a_ord, label_a, None),
        (axes[1], X_b, isort_b, bounds_b, sep_b, centres_b,
         full_b_ord, label_b, r_ord),
    ]

    for (ax, X_plot, isort, bounds, sep, centres, cluster_ord,
         lbl, r_vals) in panels:

        mat = X_plot[isort]
        im  = ax.imshow(mat, aspect="auto", interpolation="none",
                        norm=norm, cmap=cmap, origin="upper")

        # Condition separators (vertical, white)
        for off in offsets[1:-1]:
            ax.axvline(off - 0.5, color="white", lw=0.6, alpha=0.6)

        # Cluster boundaries (horizontal)
        for b in bounds:
            is_sep = (n_unmatched > 0 and b == sep)
            ax.axhline(b - 0.5, color="white",
                       lw=2.2 if is_sep else 0.4,
                       ls="--" if is_sep else "-",
                       alpha=0.9)

        # Y-axis: cluster index at each cluster centre
        ax.set_yticks(centres)
        ax.set_yticklabels([str(k) for k in cluster_ord], fontsize=5)
        ax.set_ylim(n_neurons - 0.5, -0.5)

        # X-axis: condition index at block centres
        xtick_pos = [(offsets[i] + offsets[i+1]) / 2
                     for i in range(len(n_bins_list))]
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels([f"Cond {i}" for i in range(len(n_bins_list))],
                           fontsize=7, rotation=30, ha="right")
        ax.set_xlim(-0.5, n_bins - 0.5)

        k_total = len(cluster_ord)
        ax.set_title(f"{lbl}  (k={k_total})", fontsize=10, fontweight="bold")

        cb = fig.colorbar(im, ax=ax, shrink=0.25, pad=0.01)
        cb.set_label("z-score", fontsize=7)

        # Annotate matched-pair r values on method-B panel right margin
        if r_vals is not None:
            for i, (ctr, r) in enumerate(zip(centres[:n_matched], r_vals)):
                color = plt.cm.RdYlGn(0.1 + 0.8 * max(0.0, r))
                ax.text(n_bins + n_bins * 0.01, ctr,
                        f"r={r:.2f}", va="center", ha="left",
                        fontsize=4.5, color=color,
                        transform=ax.transData)

        # Label the unmatched region if present
        if n_unmatched > 0 and sep < n_neurons:
            ax.text(n_bins * 0.01, sep + (n_neurons - sep) / 2,
                    "unmatched", va="center", ha="left",
                    fontsize=6, color="yellow", alpha=0.85,
                    rotation=90)

    fig.suptitle(
        f"Population matrices — Hungarian-matched cluster alignment\n"
        f"{label_a}  vs  {label_b}   |   "
        f"matched pairs: {n_matched}   "
        f"mean r = {r_ord.mean():.3f}   "
        f"ARI = {metrics['ari']:.3f}   "
        f"NMI = {metrics['nmi']:.3f}",
        fontsize=9, y=1.005,
    )
    _save(fig, out)


# ── main ──────────────────────────────────────────────────────────────────────

def run_cluster_comparison(out_folder, method_a="rastermap", method_b="kmeans",
                           label_a=None, label_b=None, cv=True):
    """
    Compare two clustering results from any pair of run_rastermap / run_kmeans /
    run_gmm outputs that share the same out_folder.

    Parameters
    ----------
    out_folder : root clustering folder produced by run_rastermap/run_kmeans/run_gmm
                 (contains <method_a>/ and <method_b>/ subfolders)
    method_a   : subfolder name for method A ("rastermap", "kmeans", or "gmm")
    method_b   : subfolder name for method B
    label_a    : display label for method A (defaults to method_a.capitalize())
    label_b    : display label for method B (defaults to method_b.capitalize())
    cv         : if True, load *_results_cv.npz; if False, load *_results.npz
                 Note: GMM does not produce CV results — use cv=False when
                 comparing against GMM.

    Outputs  (written to <out_folder>/comparison_<method_a>_vs_<method_b>[_cv]/)
    ──────────────────────────────────────────────────────────────────────────────
    fig1_cluster_sizes.png/pdf/svg
    fig2_mean_psth_heatmaps.png/pdf/svg
    fig3_raw_corr_matrix.png/pdf/svg
    fig4_matched_r_distribution.png/pdf/svg
    fig5_confusion_matrix.png/pdf/svg
    fig6_summary.png/pdf/svg
    fig7_row_coloring.png/pdf/svg
    cluster_comparison.csv
    """
    out_folder = Path(out_folder)
    label_a    = label_a or method_a.capitalize()
    label_b    = label_b or method_b.capitalize()
    suffix     = "_cv" if cv else ""
    npz_a      = f"{method_a}_results{suffix}.npz"
    npz_b      = f"{method_b}_results{suffix}.npz"
    folder_a   = out_folder / method_a
    folder_b   = out_folder / method_b

    for p in [folder_a / npz_a, folder_b / npz_b]:
        if not p.exists():
            raise FileNotFoundError(
                f"Expected results file not found: {p}\n"
                f"If comparing against GMM, use cv=False (GMM has no CV results).")

    out_dir = out_folder / f"comparison_{method_a}_vs_{method_b}{suffix}"
    out_dir.mkdir(exist_ok=True)
    print(f"Output → {out_dir}")

    print(f"Loading {label_a} from {folder_a / npz_a} ...")
    X_a, labels_a, n_bins_list, t_ctrs = load_results(folder_a, npz_a)
    print(f"Loading {label_b} from {folder_b / npz_b} ...")
    X_b, labels_b, _, _ = load_results(folder_b, npz_b)

    assert X_a.shape == X_b.shape, \
        f"Feature matrix shape mismatch: {X_a.shape} vs {X_b.shape}"
    assert len(labels_a) == len(labels_b), \
        f"Label length mismatch: {len(labels_a)} vs {len(labels_b)}"

    X = X_a

    print("Computing cluster means ...")
    means_a, ids_a = cluster_means(X, labels_a)
    means_b, ids_b = cluster_means(X, labels_b)

    print("Computing metrics ...")
    metrics = compute_metrics(labels_a, labels_b, means_a, means_b, ids_a, ids_b)
    print(f"  NMI={metrics['nmi']:.3f}  ARI={metrics['ari']:.3f}  "
          f"k_a={len(ids_a)}  k_b={len(ids_b)}  "
          f"mean matched r={metrics['matched_r'].mean():.3f}")

    print("Saving figures ...")
    fig1_cluster_sizes(labels_a, labels_b, ids_a, ids_b,
                       label_a, label_b, out_dir / "fig1_cluster_sizes.png")
    fig2_mean_psth_heatmaps(means_a, means_b, n_bins_list,
                            label_a, label_b, out_dir / "fig2_mean_psth_heatmaps.png")
    fig3_raw_corr_matrix(metrics["corr_mat"],
                         label_a, label_b, out_dir / "fig3_raw_corr_matrix.png")
    fig4_matched_r_distribution(metrics["matched_r"], metrics["row"], metrics["col"],
                                label_a, label_b, out_dir / "fig4_matched_r_distribution.png")
    fig5_confusion_matrix(metrics["conf_mat"], metrics["row"], metrics["col"],
                          ids_a, ids_b, label_a, label_b, out_dir / "fig5_confusion_matrix.png")
    fig6_summary(metrics, means_a, means_b, t_ctrs, n_bins_list,
                 label_a, label_b, out_dir / "fig6_summary.png")
    fig7_row_coloring(X, labels_a, labels_b, ids_a, ids_b,
                      n_bins_list, label_a, label_b, out_dir / "fig7_row_coloring.png")
    fig8_aligned_population_matrix(
        X_a, X_b, labels_a, labels_b, ids_a, ids_b,
        metrics, n_bins_list,
        label_a, label_b,
        out_dir / "fig8_aligned_population_matrix.png",
    )

    pd.DataFrame([{
        "method_a": method_a, "method_b": method_b,
        "label_a": label_a,   "label_b": label_b,
        "npz_a": npz_a, "npz_b": npz_b,
        "k_a": len(ids_a), "k_b": len(ids_b),
        "nmi": metrics["nmi"], "ari": metrics["ari"],
        "mean_matched_r":   metrics["matched_r"].mean(),
        "median_matched_r": float(np.median(metrics["matched_r"])),
        "n_pairs_r_gt_0.8": int((metrics["matched_r"] > 0.8).sum()),
        "n_pairs_r_lt_0.3": int((metrics["matched_r"] < 0.3).sum()),
    }]).to_csv(out_dir / "cluster_comparison.csv", index=False)
    print(f"  cluster_comparison.csv")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_folder", type=Path,
                        help="Root clustering folder (contains method subfolders)")
    parser.add_argument("--method_a", default="rastermap",
                        help="Method A subfolder name: rastermap | kmeans | gmm  (default: rastermap)")
    parser.add_argument("--method_b", default="kmeans",
                        help="Method B subfolder name: rastermap | kmeans | gmm  (default: kmeans)")
    parser.add_argument("--label_a", default=None,
                        help="Display label for method A (default: method_a.capitalize())")
    parser.add_argument("--label_b", default=None,
                        help="Display label for method B (default: method_b.capitalize())")
    parser.add_argument("--global_mode", action="store_true",
                        help="Compare global (non-CV) results; required when method_b=gmm")
    args = parser.parse_args()
    run_cluster_comparison(
        args.out_folder,
        method_a=args.method_a, method_b=args.method_b,
        label_a=args.label_a,   label_b=args.label_b,
        cv=not args.global_mode,
    )