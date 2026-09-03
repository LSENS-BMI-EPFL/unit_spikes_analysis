"""
run_clustering.py  —  Step 2 of the rastermap pipeline.
Load the feature matrix produced by build_feature_matrix.py and run one or
more clustering methods.  Each method is an independent entry point that
writes all its outputs to its own subfolder.

Entry points
------------
    run_rastermap(data_folder, config_path, **overrides)
        → <out_folder>/rastermap/

    run_kmeans(data_folder, config_path, **overrides)
        → <out_folder>/kmeans/

    run_gmm(data_folder, config_path, **overrides)
        → <out_folder>/gmm/

Output folder
-------------
    <data_folder>/../clustering/n{k}_{period}_{norm}_{mod}_{rw}_{bl}/
        rastermap/           rastermap_results[_cv].npz, figures, CSV
        kmeans/              kmeans_results[_cv].npz, figures, CSV
        gmm/                 gmm_results.npz, figures, CSV  (no CV)
        neuron_psth_by_condition.npz   (method-independent, written by any entry point)

Comparison
----------
    from cluster_comparison import run_cluster_comparison
    run_cluster_comparison(out_folder, method_a="rastermap", method_b="kmeans")
    run_cluster_comparison(out_folder, method_a="rastermap", method_b="gmm", cv=False)

NPZ format (standard — all methods)
-------------------------------------
    cluster_labels : (n_neurons,) int
    X              : (n_neurons, n_bins_total) feature matrix
    unit_ids, n_bins_list, t_ctr_*, n_conds, mouse_arr, reward_arr
    Method-specific extras stored in the same file (isort/boundaries for
    rastermap; gmm_probs/spectral_data for GMM).
"""

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from scipy.stats import fisher_exact, kruskal, mannwhitneyu
import matplotlib.patches as mpatches

import ephys_utilities.plotting_utils.plotting_utils as plotting_utils
from rastermap_psth.rastermap_utils import (
    load_cfg, N_WORKERS,
    get_conditions, get_cond_infos,
    fit_rastermap, _kmeans_inertia,
    _draw_matrix, _save,
    order_area_groups, _draw_prop_column,
    _draw_continuous_cluster_column_neurons,
    build_anatomy_cmaps,
    run_reward_group_stats, _bh_correction
)

try:
    from allen_utils import get_custom_area_groups, get_custom_area_groups_colors
    _HAS_ALLEN = True
except Exception:
    _HAS_ALLEN = False

from rastermap_psth.gmm_utils_new import * #(
        #GMM_CFG_DEFAULTS,
        #fit_spectral_gmm,
        #figGMM_pca_variance, figGMM_bic_curve, figGMM_eigenvalues,
        #figGMM_spectral_scatter, figGMM_assignment_entropy,
        #)

#try:
#    from rastermap_psth.gmm_utils import (
#        GMM_CFG_DEFAULTS,
#        fit_spectral_gmm,
#        figGMM_pca_variance, figGMM_bic_curve, figGMM_eigenvalues,
#        figGMM_spectral_scatter, figGMM_assignment_entropy,
#    )
_HAS_GMM = True
#except Exception as _gmm_err:
#    _HAS_GMM = False
#    GMM_CFG_DEFAULTS: dict = {}
#    print(f"[warn] gmm_utils not available: {_gmm_err}")

from rastermap_psth.rastermap_utils import (
    DEFAULT_CFG,
    CONDITIONS as CONDITIONS,
    COND_LABELS as COND_LABELS,
    COND_COLORS as COND_COLORS,
    COND_LABELS_MATRIX as COND_LABELS_MATRIX,
    COND_ALIGN_COLS as COND_ALIGN_COLS,
)


# ── figure functions (fig5-fig13, figCV) extracted from original script ───────
def fig5b_kmeans_matrix(X, n_bins_list, km_labels, k, vmax, cfg,
                        reward_arr, waveform_arr, layer_arr, area_group_arr,
                        group_colors_map, out_dir,
                        axon_arr=None, harris_arr=None, gao_arr=None,
                        anatomy_cmaps=None, area_arr=None,
                        method_label="K-means", fig_name="fig5b_kmeans_matrix"):
    """Population PSTH matrix sorted by k-means cluster, with metadata side panels.

    Same layout as fig5_population_matrix but:
      - neurons sorted by k-means cluster label
      - cluster boundaries are hard cuts between consecutive clusters
      - horizontal annotations (condition separators, onset lines) identical to fig5
    """
    # Build isort and boundaries from km_labels
    isort_km   = np.argsort(km_labels, kind="stable")   # group neurons by cluster
    boundaries_km = []
    for ki in range(k - 1):
        boundaries_km.append(int((km_labels[isort_km] == ki).sum() +
                                 sum((km_labels[isort_km] == kj).sum() for kj in range(ki))))
    # simpler: cumulative cluster sizes
    boundaries_km = np.cumsum([(km_labels == ki).sum() for ki in range(k)])[:-1].tolist()

    extra_cols = []
    if (axon_arr is not None or harris_arr is not None or gao_arr is not None):
        assert area_arr is not None, (
            "area_arr must be provided when any anatomy score array is set — "
            "_draw_continuous_cluster_column needs it to average per unique area.")
    if axon_arr is not None:
        cmap, norm = anatomy_cmaps["avg_ipsi"]
        extra_cols.append(("S1 axonal\ninnervation", axon_arr, cmap, norm))
    if harris_arr is not None:
        cmap, norm = anatomy_cmaps["cc_tc_ct_iterated"]
        extra_cols.append(("Hierarchy\n(Harris '19)", harris_arr, cmap, norm))
    if gao_arr is not None:
        cmap, norm = anatomy_cmaps["cc_hierarchy_score_columns"]
        extra_cols.append(("Hierarchy\n(Gao '26)", gao_arr, cmap, norm))

    n_side = 4 + len(extra_cols)
    width_ratios = [10, 10, 0.5, 0.5, 1.0] + [0.3] * len(extra_cols) + [0.5]
    fig, axes = plt.subplots(
        1, 2 + n_side, figsize=(24 + 2 * len(extra_cols), 12), dpi=400,
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.05})

    im1 = _draw_matrix(axes[0], X,            n_bins_list, [],            vmax, cfg,
                       f"Input order  (n={len(X)})")
    im2 = _draw_matrix(axes[1], X[isort_km],  n_bins_list, boundaries_km, vmax, cfg,
                       f"{method_label} order  (k={k}, n={len(X)})")
    for ax, im in zip(axes[:2], [im1, im2]):
        label_txt = 'Firing rate (z-score)' if DEFAULT_CFG['normalize']=='zscore' else 'Firing rate (spks/s)'
        fig.colorbar(im, ax=ax, label=label_txt, shrink=0.3, pad=0.01)

    n_neurons = len(isort_km)
    edges     = [0] + list(boundaries_km) + [n_neurons]

    reward_s   = reward_arr[isort_km]
    waveform_s = waveform_arr[isort_km]
    layer_s    = layer_arr[isort_km]
    agroup_s   = area_group_arr[isort_km]

    _draw_prop_column(axes[2], waveform_s,
                      ["NW", "WW"],
                      {"NW": "#83b1ff", "WW": "#ff8783"},
                      edges, n_neurons, "Waveform",
                      exclude=["unknown", "None", "nan"])

    layer_base   = ["supragranular", "granular", "infragranular"]
    layer_colors = {"supragranular": "#9B59B6",
                    "granular":      "#E74C3C",
                    "infragranular": "#194882"}
    extra_layers = [l for l in sorted(set(layer_s)) if l not in layer_base]
    layer_cats   = layer_base + extra_layers
    for l in extra_layers:
        layer_colors[l] = "#aaaaaa"
    _draw_prop_column(axes[3], layer_s, layer_cats, layer_colors,
                      edges, n_neurons, "Layer",
                      exclude=["None", "nan", "unknown"])

    all_groups = order_area_groups(agroup_s)
    _draw_prop_column(axes[4], agroup_s, all_groups, group_colors_map,
                      edges, n_neurons, "Area")


    # ── Anatomical axes (continuous, per-cluster mean) ──────────────────────────
    anatomy_axes = []
    for i, (title, arr, cmap, norm) in enumerate(extra_cols):
        ax = axes[5 + i]
        _draw_continuous_cluster_column_neurons(ax, arr[isort_km], edges, n_neurons, title, cmap, norm)
        #_draw_continuous_cluster_column(ax, arr[isort_km], area_arr[isort_km], edges, n_neurons, title, cmap, norm)
        anatomy_axes.append((ax, cmap, norm))

    fig.tight_layout()

    # Colorbars are added as separate axes positioned below each column using its
    # finalized (post-tight_layout) figure coordinates. Attaching via
    # fig.colorbar(..., ax=ax) instead would carve space out of the same ax,
    # squeezing the full-height per-cluster strip into a sliver at the top.
    for ax, cmap, norm in anatomy_axes:
        pos = ax.get_position()
        ax.spines["bottom"].set_visible(False)
        cax = fig.add_axes([pos.x0, pos.y0 - 0.035, pos.width, 0.012])
        cb  = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                           orientation="horizontal")
        cb.ax.tick_params(labelsize=5, length=2, pad=1)

    _draw_prop_column(axes[-1], reward_s,
                      ["R+", "R-"],
                      {"R+": "forestgreen", "R-": "crimson"},
                      edges, n_neurons, "Group")

    _save(fig, out_dir / fig_name, dpi=400)

def fig6_cluster_profiles(X, t_ctrs, n_bins_list, isort, boundaries, out_dir, prefix=None):
    n_clusters = len(boundaries) + 1
    edges      = [0] + list(boundaries) + [len(isort)]
    offsets    = np.concatenate([[0], np.cumsum(n_bins_list)])
    ncols      = (n_clusters + 1) // 2
    nrows      = 4 if n_clusters > 1 else 1
    fig, axes  = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                               sharey=True, sharex=False)
    axes = np.atleast_1d(axes).ravel()
    for k in range(n_clusters):
        ax       = axes[k]
        idx      = isort[edges[k]:edges[k + 1]]
        if len(idx) == 0:
            # Can happen when n_rastermap_clusters is close to/exceeds the
            # number of neurons: the even-width boundary split degenerates
            # to a zero-width bin for this cluster. X[idx].mean(0) on an
            # empty selection would be all-NaN and crash matplotlib's axis
            # autoscaling downstream, so skip plotting this (empty) cluster
            # entirely and just hide its axis.
            ax.set_title(f"C{k+1}  (n=0)", fontsize=8)
            ax.set_visible(False)
            continue

        #print(f"NaN in X before fig6: {np.isnan(X).any(axis=1).sum()} rows")
        mean_vec = X[idx].mean(0)
        sem_vec  = X[idx].std(0) / np.sqrt(len(idx))
        mean_vec = np.where(np.isfinite(mean_vec), mean_vec, 0.0)
        sem_vec = np.where(np.isfinite(sem_vec), sem_vec, 0.0)

        if not np.isfinite(mean_vec).any():
            ax.set_title(f"C{k + 1}  (n={len(idx)}, all-NaN)", fontsize=8)
            ax.set_visible(False)
            continue
        for c, (label, color, t_ctr_c) in enumerate(zip(COND_LABELS, COND_COLORS, t_ctrs)):
            sl = slice(offsets[c] + 1, offsets[c + 1] - 1)
            ax.plot(t_ctr_c[1:-1], mean_vec[sl], color=color, lw=1.5, label=label)
            ax.fill_between(t_ctr_c[1:-1], mean_vec[sl] - sem_vec[sl],
                             mean_vec[sl] + sem_vec[sl], color=color, alpha=0.3)
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_title(f"C{k+1}  (n={len(idx)})", fontsize=8)
        #if k % ncols == 0:
        #    ax.set_ylabel(label_txt)
        label_txt = 'Firing rate (z-score)' if DEFAULT_CFG['normalize']=='zscore' else 'Firing rate (spks/s)'
        #if k >= (nrows - 1) * ncols:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(label_txt)
        ax.legend(fontsize=5, loc="upper right", frameon=False)
    for ax in axes[n_clusters:]:
        ax.set_visible(False)   # BUG FIX: was True
    fig.tight_layout()
    if prefix is not None:
        figname = f"fig6_cluster_profiles_{prefix}"
    else:
        figname = "fig6_cluster_profiles"
    _save(fig, out_dir / figname, dpi=400)


def fig6b_cluster_profiles_reward_groups(X, t_ctrs, n_bins_list, isort, boundaries,
                                   cond_labels, cond_colors, reward_arr, out_dir, prefix=None):
    wh_idx = [i for i, l in enumerate(cond_labels) if "Whisker" in l]
    if len(wh_idx) == 0:
        print("  No whisker conditions found, skipping fig6b")
        return

    offsets    = np.concatenate([[0], np.cumsum(n_bins_list)])
    n_clusters = len(boundaries) + 1
    edges      = [0] + list(boundaries) + [len(isort)]
    ncols      = (n_clusters + 1) // 2
    nrows      = 4 if n_clusters > 1 else 1
    fig, axes  = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                               sharey=True, sharex=False)
    axes = np.atleast_1d(axes).ravel()

    reward_groups = [("R+", "forestgreen"), ("R-", "crimson")]

    for k in range(n_clusters):
        ax  = axes[k]
        idx = isort[edges[k]:edges[k + 1]]
        if len(idx) == 0:
            ax.set_visible(False)
            continue

        for rg, rg_color in reward_groups:
            rg_mask   = reward_arr[idx] == rg
            rg_idx    = idx[rg_mask]
            if len(rg_idx) == 0:
                continue
            mean_vec  = X[rg_idx].mean(0)
            sem_vec   = X[rg_idx].std(0) / np.sqrt(len(rg_idx))

            for c in wh_idx:
                t_ctr_c = t_ctrs[c]
                sl = slice(offsets[c] + 1, offsets[c + 1] - 1)
                label = f"{cond_labels[c]} {rg}"
                color = plotting_utils.adjust_lightness(rg_color, 1.5) if "pre" in cond_labels[c].lower() else rg_color
                ax.plot(t_ctr_c[1:-1], mean_vec[sl], color=color, lw=1.5, label=label)
                ax.fill_between(t_ctr_c[1:-1], mean_vec[sl] - sem_vec[sl],
                                mean_vec[sl] + sem_vec[sl], color=rg_color, alpha=0.2)

        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        n_rp = (reward_arr[idx] == "R+").sum()
        n_rm = (reward_arr[idx] == "R-").sum()
        ax.set_title(f"C{k+1}  (R+={n_rp}, R−={n_rm})", fontsize=8)
        #if k % ncols == 0:
        #    ax.set_ylabel("z-score")
        label_txt = 'Firing rate (z-score)' if DEFAULT_CFG['normalize']=='zscore' else 'Firing rate (spks/s)'
        #if k >= (nrows - 1) * ncols:
        ax.set_ylabel(label_txt)
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=5, loc="upper right", frameon=False)

    for ax in axes[n_clusters:]:
        ax.set_visible(False)
    fig.tight_layout()
    if prefix is not None:
        figname = f"fig6b_cluster_profiles_reward_groups_{prefix}"
    else:
        figname = "fig6b_cluster_profiles_reward_groups"

    _save(fig, out_dir / figname, dpi=400)
def fig7_pca_variance(X, n_pca, out_dir): # TODO: keep
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
    _save(fig, out_dir / "fig7_pca_variance", dpi=400)

def fig8_umap(emb, cluster_labels, n_clusters, out_dir, prefix=None):

    vmax = int(cluster_labels.max())
    fig, (ax1, ax2, cax) = plt.subplots(1, 3,figsize=(11, 5),gridspec_kw={"width_ratios": [1, 1, 0.025]})  # thinner colorbar

    ax1.scatter(emb[:, 0], emb[:, 1],s=3, c="steelblue", alpha=0.6, linewidths=0)
    ax1.set_title("UMAP")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")

    sc = ax2.scatter(emb[:, 0], emb[:, 1],s=3, c=cluster_labels,cmap="turbo",alpha=0.6,linewidths=0,vmin=0,vmax=vmax,)
    ax2.set_title("UMAP — rastermap_psth clusters")
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")

    cb = plt.colorbar(sc, cax=cax, label="Rastermap cluster")

    # 6 evenly spaced ticks from 0 to vmax
    ticks = np.linspace(0, vmax, 6)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{int(round(t))}" for t in ticks])
    cb.ax.tick_params(labelsize=8)
    cb.set_label("Rastermap cluster", fontsize=9)

    fig.tight_layout()
    if prefix is not None:
        figname = f"fig8_umap_{prefix}"
    else:
        figname = "fig8_umap"
    _save(fig, out_dir / figname, dpi=400)
    return


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
    _save(fig, out_dir / "fig9_kmeans", dpi=400)

def fig10_kmeans_profiles(X, t_ctrs, n_bins_list, km_labels, k, out_dir):
    offsets = np.concatenate([[0], np.cumsum(n_bins_list)])
    ncols   = (k + 1) // 2
    nrows   = 2 if k > 1 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                              sharey=True, sharex=False)
    axes = np.atleast_1d(axes).ravel()
    for ki in range(k):
        ax       = axes[ki]
        idx      = np.where(km_labels == ki)[0]
        if len(idx) == 0:
            ax.set_title(f"K{ki + 1}  (n=0)", fontsize=8)
            ax.set_visible(False)
            continue
        mean_vec = X[idx].mean(0)
        sem_vec  = X[idx].std(0) / np.sqrt(len(idx))
        for c, (label, color, t_ctr_c) in enumerate(zip(COND_LABELS, COND_COLORS, t_ctrs)):
            sl = slice(offsets[c] + 1, offsets[c + 1] - 1)
            ax.plot(t_ctr_c[1:-1], mean_vec[sl], color=color, lw=1.2, label=label)
            ax.fill_between(t_ctr_c[1:-1], mean_vec[sl] - sem_vec[sl],
                            mean_vec[sl] + sem_vec[sl], color=color, alpha=0.3)
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_title(f"K{ki + 1}  (n={len(idx)})", fontsize=8)
        #if ki % ncols == 0:
        #    ax.set_ylabel("z-score")
        label_txt = 'Firing rate (z-score)' if DEFAULT_CFG['normalize']=='zscore' else 'Firing rate (spks/s)'
        if ki >= (nrows - 1) * ncols:
            ax.set_ylabel(label_txt)
            ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8, loc="upper right", frameon=False)
    for ax in axes[k:]:
        ax.set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir / "fig10_kmeans_profiles", dpi=400)

def fig5_population_matrix(X, n_bins_list, isort, boundaries, vmax, cfg,
                           reward_arr, waveform_arr, layer_arr, area_group_arr,
                           group_colors_map, out_dir,
                           axon_arr=None, harris_arr=None, gao_arr=None,
                           anatomy_cmaps=None, prefix=None, area_arr=None):
    """Population PSTH matrix with metadata side panels.

    Categorical side panels (one per cluster, stacked horizontal bars):
      • Reward group  (R+ / R-)
      • Waveform type (NW narrow / WW wide, split at population median)
      • Layer         (supragranular / granular / infragranular)
      • Brain region  (area_acronym_custom mapped to allen_utils groups)

    Continuous side panels (one per cluster, solid color = cluster mean; optional,
    only drawn when the corresponding *_arr is not None):
      • S1 axonal innervation
      • Hierarchy score (Harris et al. 2019)
      • Hierarchy score (Gao et al. 2026)
    """
    extra_cols = []
    if (axon_arr is not None or harris_arr is not None or gao_arr is not None):
        assert area_arr is not None, (
            "area_arr must be provided when any anatomy score array is set — "
            "_draw_continuous_cluster_column needs it to average per unique area.")
    if axon_arr is not None:
        cmap, norm = anatomy_cmaps["avg_ipsi"]
        extra_cols.append(("wS1/2 proj. strength\n(Liu '24)", axon_arr, cmap, norm))
    if harris_arr is not None:
        cmap, norm = anatomy_cmaps["cc_tc_ct_iterated"]
        extra_cols.append(("Hierarchy\n(Harris '19)", harris_arr, cmap, norm))
    if gao_arr is not None:
        cmap, norm = anatomy_cmaps["cc_hierarchy_score_columns"]
        extra_cols.append(("Hierarchy\n(Gao '26)", gao_arr, cmap, norm))

    n_side = 4 + len(extra_cols)
    width_ratios = [10, 10, 0.5, 0.5, 1.0] + [0.3] * len(extra_cols) + [0.5]
    fig, axes = plt.subplots(
        1, 2 + n_side, figsize=(24 + 2 * len(extra_cols), 12), dpi=400,
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.05})

    im1 = _draw_matrix(axes[0], X,        n_bins_list, [],         vmax, cfg, f"Input order  (n={len(X)})")
    im2 = _draw_matrix(axes[1], X[isort], n_bins_list, boundaries, vmax, cfg, f"Rastermap order (n={len(X)})")
    for ax, im in zip(axes[:2], [im1, im2]):
        label_txt = 'Firing rate (z-score)' if DEFAULT_CFG['normalize']=='zscore' else 'Firing rate (spks/s)'
        fig.colorbar(im, ax=ax, label=label_txt, shrink=0.3, pad=0.005)

    n_neurons = len(isort)
    edges     = [0] + list(boundaries) + [n_neurons]

    # sort all metadata arrays by rastermap_psth order
    reward_s   = reward_arr[isort]
    waveform_s = waveform_arr[isort]
    layer_s    = layer_arr[isort]
    agroup_s   = area_group_arr[isort]

    # ── Waveform type (NW = fast-spiking, WW = regular-spiking) ──────────────
    _draw_prop_column(axes[2], waveform_s,
                      ["NW", "WW"],
                      {"NW": "#83b1ff", "WW": "#ff8783"},
                      edges, n_neurons, "Waveform",
                      exclude=["unknown", "None", "nan"])

    # ── Layer ─────────────────────────────────────────────────────────────────
    layer_base   = ["supragranular", "granular", "infragranular"]
    layer_colors = {"supragranular":  "#9B59B6",
                    "granular":       "#E74C3C",
                    "infragranular":  "#194882"}
    extra_layers = [l for l in sorted(set(layer_s)) if l not in layer_base]
    layer_cats   = layer_base + extra_layers
    for l in extra_layers:
        layer_colors[l] = "#aaaaaa"
    _draw_prop_column(axes[3], layer_s, layer_cats, layer_colors,
                      edges, n_neurons, "Layer",
                      exclude=["None", "nan", "unknown"])


    # ── Brain region (allen_utils groups) ─────────────────────────────────────
    all_groups = order_area_groups(agroup_s)   # anatomy-first, then frequency within group
    _draw_prop_column(axes[4], agroup_s, all_groups, group_colors_map,
                      edges, n_neurons, "Area")


    # ── Anatomical axes (continuous, per-cluster mean) ──────────────────────────
    anatomy_axes = []
    for i, (title, arr, cmap, norm) in enumerate(extra_cols):
        ax = axes[5 + i]
        _draw_continuous_cluster_column_neurons(ax, arr[isort], edges, n_neurons, title, cmap, norm)
        #_draw_continuous_cluster_column_neurons(ax, arr[isort], area_arr[isort], edges, n_neurons, title, cmap, norm)
        anatomy_axes.append((ax, cmap, norm, title))

    fig.tight_layout()

    # Colorbars are added as separate axes positioned below each column using its
    # finalized (post-tight_layout) figure coordinates. Attaching via
    # fig.colorbar(..., ax=ax) instead would carve space out of the same ax,
    # squeezing the full-height per-cluster strip into a sliver at the top.
    for ax, cmap, norm, title in anatomy_axes:
        ax.spines["bottom"].set_visible(False)
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0, pos.y0 - 0.035, pos.width, 0.012])
        cb  = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                           orientation="horizontal", aspect=10)
        cb.ax.tick_params(labelsize=5, length=2, pad=1, rotation=45)
        if 'Harris' in title and 'Gao' in title:
            cb.set_ticks([norm.vmin, 0, norm.vmax])
            cb.set_ticklabels([f"{norm.vmin:.1f}",'0',f"{norm.vmax:.1f}"])
        elif 'Liu' in title:
            cb.set_ticks([0, norm.vmax])
            cb.set_ticklabels([0, f"{norm.vmax:.0e}"])

    # ── Reward group ──────────────────────────────────────────────────────────
    _draw_prop_column(axes[-1], reward_s,
                      ["R+", "R-"],
                      {"R+": "forestgreen", "R-": "crimson"},
                      edges, n_neurons, "Group")

    if prefix is not None:
        fig_name = f"fig5_population_matrix_{prefix}"
    else:
        fig_name = "fig5_population_matrix"

    _save(fig, out_dir / fig_name, dpi=400)


def fig11_area_per_cluster(unit_ids, cluster_labels, area_arr, n_clusters, out_dir):
    """Stacked bar: area composition per rastermap_psth cluster."""
    all_areas  = sorted(set(area_arr))
    cmap       = matplotlib.colormaps.get_cmap("tab20")#plt.cm.get_cmap("tab20", len(all_areas))
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

    handles = [mpatches.Patch(color=area_color[a], label=a) for a in all_areas]
    fig.legend(handles, all_areas, loc="lower center",
               ncol=min(8, len(all_areas)), fontsize=6,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Area composition per rastermap_psth cluster", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / "fig11_area_per_cluster", dpi=400)


def fig12_reward_per_cluster(unit_ids, cluster_labels, reward_arr, n_clusters, out_dir):
    """R+/R- proportion per rastermap_psth cluster with per-cluster Fisher's exact + BH correction."""
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
    ax.set_title("Reward group composition per rastermap_psth cluster\n"
                 "(* BH-corrected Fisher's exact, α=0.05)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir / "fig12_reward_per_cluster", dpi=400)


def fig13_anatomy_axis_per_cluster(cluster_labels, axis_arr, axis_title, axis_tag,
                                   n_clusters, out_dir):
    """Per-cluster summary of a continuous anatomical axis (e.g. hierarchy score).

    For each cluster: mean ± SEM of axis_arr, tested against the rest of the
    population with a two-sided Mann-Whitney U (per-cluster vs. rest) + BH-FDR
    across clusters. An omnibus Kruskal-Wallis test (does the axis vary at all
    across clusters) is also reported. NaN entries (neurons whose area had no
    matched score) are dropped before testing.

    Saves
    -----
    stats/anatomy_axis_stats_{axis_tag}.csv  — per-cluster mean/SEM/n/U/p_raw/p_fdr/significant
    stats/anatomy_axis_kruskal_{axis_tag}.txt
    fig13_anatomy_axis_per_cluster_{axis_tag}.png/.pdf/.svg
    """
    valid        = ~np.isnan(axis_arr)
    axis_valid   = axis_arr[valid]
    labels_valid = cluster_labels[valid]
    overall_mean = axis_valid.mean() if axis_valid.size else np.nan

    # ── Omnibus test: does the axis differ at all across clusters? ─────────
    groups = [axis_valid[labels_valid == k] for k in range(n_clusters)]
    groups_nonempty = [g for g in groups if len(g) >= 1]
    if len(groups_nonempty) >= 2 and any(len(g) >= 2 for g in groups_nonempty):
        kw_stat, kw_p = kruskal(*groups_nonempty)
    else:
        kw_stat, kw_p = np.nan, np.nan

    stats_dir = Path(out_dir) / "stats"
    stats_dir.mkdir(exist_ok=True)
    with open(stats_dir / f"anatomy_axis_kruskal_{axis_tag}.txt", "w") as fh:
        fh.write(f"{axis_title}\n")
        if not np.isnan(kw_p):
            fh.write(f"Kruskal-Wallis across {n_clusters} clusters: "
                     f"H={kw_stat:.4f}  p={kw_p:.4g}\n")
        else:
            fh.write("Kruskal-Wallis: not enough valid clusters to test\n")

    # ── Per-cluster mean/SEM + cluster-vs-rest Mann-Whitney U ──────────────
    means, sems, ns, pvals = [], [], [], []
    for k in range(n_clusters):
        cl_vals   = axis_valid[labels_valid == k]
        rest_vals = axis_valid[labels_valid != k]
        n_cl      = len(cl_vals)
        ns.append(n_cl)
        means.append(cl_vals.mean() if n_cl > 0 else np.nan)
        sems.append(cl_vals.std(ddof=1) / np.sqrt(n_cl) if n_cl > 1 else 0.0)
        if n_cl >= 2 and len(rest_vals) >= 2:
            _, p = mannwhitneyu(cl_vals, rest_vals, alternative="two-sided")
        else:
            p = 1.0
        pvals.append(p)

    pvals_fdr, reject = _bh_correction(np.array(pvals), alpha=0.05)
    n_sig = int(reject.sum())
    if not np.isnan(kw_p):
        print(f"  {axis_title}: {n_sig}/{n_clusters} clusters significant vs. rest "
              f"(BH-FDR, α=0.05); Kruskal-Wallis p={kw_p:.4g}")
    else:
        print(f"  {axis_title}: not enough valid data for cluster stats")

    stats_df = pd.DataFrame(dict(
        cluster      = np.arange(1, n_clusters + 1),
        n             = ns,
        mean          = means,
        sem           = sems,
        p_raw         = pvals,
        p_fdr         = pvals_fdr,
        significant   = reject,
    ))
    stats_df.to_csv(stats_dir / f"anatomy_axis_stats_{axis_tag}.csv", index=False)

    # ── Summary bar figure ───────────────────────────────────────────────
    x = np.arange(n_clusters)
    fig, ax = plt.subplots(figsize=(max(6, n_clusters * 0.7), 4))
    ax.bar(x, means, yerr=sems, color="#4158d9", edgecolor="none", capsize=2)
    if not np.isnan(overall_mean):
        ax.axhline(overall_mean, color="k", ls="--", lw=1.0, alpha=0.7,
                   label=f"Overall mean ({overall_mean:.2f})")

    means_arr = np.array(means, dtype=float)
    sems_arr  = np.array(sems, dtype=float)
    if np.all(np.isnan(means_arr)):
        y_top, y_bot = 1.0, 0.0
    else:
        y_top = np.nanmax(means_arr + sems_arr)
        y_bot = np.nanmin(means_arr - sems_arr)
    pad   = 0.08 * (y_top - y_bot if y_top > y_bot else 1.0)
    for k in range(n_clusters):
        if reject[k] and not np.isnan(means[k]):
            ax.text(k, means[k] + sems[k] + pad * 0.3, "*", ha="center", va="bottom",
                    fontsize=11, color="k")

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{k+1}\n(n={ns[k]})" for k in range(n_clusters)], fontsize=7)
    ax.set_ylabel(axis_title)
    ax.set_ylim(y_bot - pad, y_top + pad)
    kw_txt = f"H={kw_stat:.2f}, p={kw_p:.3g}" if not np.isnan(kw_p) else "n/a"
    ax.set_title(f"{axis_title} per rastermap_psth cluster\n"
                f"(* BH-corrected Mann-Whitney vs. rest, α=0.05; Kruskal-Wallis {kw_txt})",
                fontsize=9)
    if not np.isnan(overall_mean):
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir / f"fig13_anatomy_axis_per_cluster_{axis_tag}", dpi=400)


def figCV_rastermap_comparison(X_odd, X_even, n_bins_list, isort, boundaries,
                               vmax, cfg, out_dir):
    """Two-panel figure: X_odd[isort] | X_even[isort]."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 9), dpi=400,
                             gridspec_kw={"width_ratios": [10, 10, 1]})
    im1 = _draw_matrix(axes[0], X_odd[isort],  n_bins_list, boundaries, vmax, cfg,
                       f"Odd trials — rastermap_psth order (n={len(X_odd)})")
    im2 = _draw_matrix(axes[1], X_even[isort], n_bins_list, boundaries, vmax, cfg,
                       f"Even trials — same order (n={len(X_even)})")
    for ax, im in zip(axes[:2], [im1, im2]):
        label_txt = 'Firing rate (z-score)' if DEFAULT_CFG['normalize']=='zscore' else 'Firing rate (spks/s)'
        fig.colorbar(im, ax=ax, label=label_txt, shrink=0.3, pad=0.01)
    axes[2].axis("off")
    fig.suptitle("Cross-validation: embedding fitted on odd trials", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir / "figCV_rastermap_comparison", dpi=400)


def figCV_similarity_metrics(X_odd, X_even, isort, boundaries, cluster_labels,
                             n_bins_list, t_ctrs, cond_labels, cond_colors, out_dir):
    """
    Four-panel figure:
      A) Per-cluster R² between cluster mean vectors (odd vs even)
      B) Cosine similarity per neuron (histogram)
      C) Principal angles between column spaces of X_odd and X_even
      D) Global metrics: normalised Frobenius norm, mean cosine, mean principal angle
    """
    from scipy.linalg import subspace_angles

    n_clusters = len(boundaries) + 1
    edges      = [0] + list(boundaries) + [len(isort)]

    # ── per-cluster R² ────────────────────────────────────────────────────────
    r2_per_cluster = []
    for k in range(n_clusters):
        idx = isort[edges[k]:edges[k + 1]]
        if len(idx) == 0:
            r2_per_cluster.append(np.nan)
            continue
        mu_odd  = X_odd[idx].mean(0)
        mu_even = X_even[idx].mean(0)
        # Pearson R² between the two mean PSTH vectors
        mask = np.isfinite(mu_odd) & np.isfinite(mu_even)
        if mask.sum() < 2:
            r2_per_cluster.append(np.nan)
            continue
        r = np.corrcoef(mu_odd[mask], mu_even[mask])[0, 1]
        r2_per_cluster.append(float(np.nan_to_num(r ** 2, nan=0.0, posinf=0.0)))
    r2_arr = np.array(r2_per_cluster)

    # ── per-neuron cosine similarity ──────────────────────────────────────────
    norms_odd  = np.linalg.norm(X_odd,  axis=1, keepdims=True) + 1e-12
    norms_even = np.linalg.norm(X_even, axis=1, keepdims=True) + 1e-12
    cos_sim    = ((X_odd / norms_odd) * (X_even / norms_even)).sum(axis=1)
    cos_sim    = np.clip(cos_sim, -1.0, 1.0)   # guard against numerical noise

    # ── principal angles ──────────────────────────────────────────────────────
    n_angle_vecs = min(50, X_odd.shape[0], X_odd.shape[1])
    angles_rad   = subspace_angles(X_odd.T[:, :n_angle_vecs],
                                   X_even.T[:, :n_angle_vecs])
    angles_deg   = np.degrees(angles_rad)

    # ── global metrics ────────────────────────────────────────────────────────
    frob_diff  = np.linalg.norm(X_odd[isort] - X_even[isort], "fro")
    frob_odd   = np.linalg.norm(X_odd[isort], "fro") + 1e-12
    frob_norm  = float(frob_diff / frob_odd)
    mean_cos   = float(np.nanmean(cos_sim))
    mean_angle = float(np.nanmean(angles_deg))

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
    axA = fig.add_subplot(gs[0, :])   # full top row: per-cluster R²
    axB = fig.add_subplot(gs[1, 0])   # cosine histogram
    axC = fig.add_subplot(gs[1, 1])   # principal angles
    axD = fig.add_subplot(gs[1, 2])   # global metric summary (text)

    # A — per-cluster R²
    x      = np.arange(n_clusters)
    colors = ["steelblue" if not np.isnan(v) else "lightgrey" for v in r2_arr]
    axA.bar(x, np.nan_to_num(r2_arr), color=colors, edgecolor="none")
    axA.axhline(np.nanmean(r2_arr), color="k", ls="--", lw=1.2,
                label=f"Mean R² = {np.nanmean(r2_arr):.3f}")
    axA.set_xlim(-0.5, n_clusters - 0.5)
    axA.set_ylim(0, 1.05)
    axA.set_xlabel("Rastermap cluster")
    axA.set_ylabel("R² (odd vs even cluster mean)")
    axA.set_title("Per-cluster R²: consistency of mean PSTH (odd → even)")
    axA.legend(fontsize=9)

    # B — cosine similarity histogram
    axB.hist(cos_sim, bins=40, color="steelblue", edgecolor="none", alpha=0.85)
    axB.axvline(mean_cos, color="k", ls="--", lw=1.2,
                label=f"Mean = {mean_cos:.3f}")
    axB.set_xlabel("Cosine similarity")
    axB.set_ylabel("Neuron count")
    axB.set_title("Per-neuron cosine similarity\n(odd vs even PSTH vectors)")
    axB.legend(fontsize=8)

    # C — principal angles
    axC.plot(angles_deg, "o-", ms=3, lw=1, color="darkorange")
    axC.axhline(mean_angle, color="k", ls="--", lw=1.2,
                label=f"Mean = {mean_angle:.1f}°")
    axC.set_xlabel("Principal angle index")
    axC.set_ylabel("Angle (degrees)")
    axC.set_title("Principal angles between\nodd and even column spaces")
    axC.legend(fontsize=8)

    # D — global summary as formatted text (avoids bbox blowup from bar+text combos)
    axD.axis("off")
    axD.set_title("Global similarity metrics", fontsize=10, pad=6)
    summary_lines = [
        ("Frobenius norm (norm.)",  f"{frob_norm:.3f}",  "(↓ better)", "#4c72b0"),
        ("Mean cosine similarity",  f"{mean_cos:.3f}",   "(↑ better)", "#55a868"),
        (f"Mean principal angle",   f"{mean_angle:.1f}°","(↓ better)", "#c44e52"),
    ]
    for j, (name, val, hint, col) in enumerate(summary_lines):
        y = 0.78 - j * 0.28
        axD.text(0.05, y,       name,  transform=axD.transAxes,
                 fontsize=10, color="k",  va="top", ha="left")
        axD.text(0.95, y,       val,   transform=axD.transAxes,
                 fontsize=13, color=col, va="top", ha="right", fontweight="bold")
        axD.text(0.95, y-0.09,  hint,  transform=axD.transAxes,
                 fontsize=8,  color="grey", va="top", ha="right")

    fig.suptitle("Cross-validation similarity: odd-trial embedding vs even-trial PSTHs",
                 fontsize=11)
    _save(fig, out_dir / "figCV_similarity_metrics", dpi=400)

    print(f"  CV metrics — mean R²={np.nanmean(r2_arr):.3f}  "
          f"mean cosine={mean_cos:.3f}  "
          f"Frobenius(norm)={frob_norm:.3f}  "
          f"mean principal angle={mean_angle:.1f}°")
    return dict(r2_per_cluster=r2_arr, cos_sim=cos_sim,
                angles_deg=angles_deg, frob_norm=frob_norm)




# ── helpers ───────────────────────────────────────────────────────────────────

def _load_npz(path):
    d       = dict(np.load(path, allow_pickle=True))
    n_conds = int(d["n_conds"])
    t_ctrs  = [d[f"t_ctr_{ci}"] for ci in range(n_conds)]
    return d, t_ctrs


def _load_feature_matrix(data_folder: Path) -> dict:
    """Load feature_matrix.npz and neuron_metadata.csv from data_folder.

    Returns a flat dict with all arrays and the meta DataFrame,
    ready to be unpacked by any entry point.
    """
    d, t_ctrs   = _load_npz(data_folder / "feature_matrix.npz")
    meta_df     = pd.read_csv(data_folder / "neuron_metadata.csv")
    unit_ids    = d["unit_ids"].tolist()
    n_bins_list = d["n_bins_list"].astype(int)
    n           = len(unit_ids)
    axon_arr    = d.get("axon_arr",   None)
    harris_arr  = d.get("harris_arr", None)
    gao_arr     = d.get("gao_arr",    None)
    if axon_arr is not None:
        axon_arr   = axon_arr.astype(float)
        harris_arr = harris_arr.astype(float)
        gao_arr    = gao_arr.astype(float)
    print(f"  X={d['X'].shape}  X_odd={d['X_odd'].shape}  X_even={d['X_even'].shape}")
    return dict(
        X=d["X"], X_odd=d["X_odd"], X_even=d["X_even"],
        unit_ids=unit_ids, n_bins_list=n_bins_list, t_ctrs=t_ctrs,
        mouse_arr=d["mouse_arr"], reward_arr=d["reward_arr"],
        area_arr       = d.get("area_arr",       np.array(["unknown"] * n)),
        waveform_arr   = d.get("waveform_arr",   np.array(["unknown"] * n)),
        layer_arr      = d.get("layer_arr",      np.array(["unknown"] * n)),
        area_group_arr = d.get("area_group_arr", np.array(["Other"]   * n)),
        axon_arr=axon_arr, harris_arr=harris_arr, gao_arr=gao_arr,
        meta_df=meta_df,
    )


def _resolve_dirs(data_folder: Path, cfg: dict, method_name: str):
    """Compute and create the clustering root folder and the method subfolder.

    Returns (out_folder, method_dir).
    """
    n_k        = cfg["n_rastermap_clusters"]
    norm_tag   = cfg.get("normalize", "zscore")
    period_tag = cfg["period"]
    mod_tag    = cfg["modality"]
    rw_tag     = cfg.get("reward_filter", "combined").replace("+","plus").replace("-","minus")
    bl_tag     = "nobl" if cfg.get("baseline_removal", False) else "bl"
    out_folder = (data_folder.parent /
                  f"clustering/n{n_k}_{period_tag}_{norm_tag}_{mod_tag}_{rw_tag}_{bl_tag}")
    method_dir = out_folder / method_name
    out_folder.mkdir(parents=True, exist_ok=True)
    method_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out_folder}")
    print(f"  {method_name}/ → {method_dir}")
    return out_folder, method_dir


def _cluster_labels_from_isort(isort, boundaries, n_neurons):
    labels = np.empty(n_neurons, dtype=int)
    edges  = [0] + list(boundaries) + [n_neurons]
    for k in range(len(boundaries) + 1):
        labels[isort[edges[k]:edges[k+1]]] = k
    return labels


def _labels_to_isort_boundaries(labels: np.ndarray, k: int):
    """Convert integer cluster labels → (isort, boundaries).

    Produces the same (isort, boundaries) format expected by fig5/fig6
    so that k-means and GMM can reuse the rastermap figure functions.
    Neurons are sorted by label (cluster 0 first, then 1, …).
    """
    isort      = np.argsort(labels, kind="stable")
    boundaries = np.cumsum(
        [(labels == ki).sum() for ki in range(k)]
    )[:-1].tolist()
    return isort, boundaries


def _base_save_dict(X, unit_ids, n_bins_list, t_ctrs, mouse_arr, reward_arr):
    """Common fields shared by all result .npz files."""
    sd = dict(X=X,
              unit_ids=np.array(unit_ids),
              n_bins_list=np.array(n_bins_list),
              n_conds=np.array(len(t_ctrs)),
              mouse_arr=mouse_arr,
              reward_arr=reward_arr)
    for ci, tc in enumerate(t_ctrs):
        sd[f"t_ctr_{ci}"] = tc
    return sd


def _save_rastermap_npz(path, X, isort, boundaries, cluster_labels,
                         umap_emb, unit_ids, n_bins_list, t_ctrs,
                         mouse_arr, reward_arr):
    """Save rastermap results to .npz (standard format + rastermap extras).

    Standard fields (required by cluster_comparison.py)
    ─────────────────────────────────────────────────────
    X, cluster_labels, unit_ids, n_bins_list, t_ctr_*, n_conds,
    mouse_arr, reward_arr

    Rastermap-specific extras
    ─────────────────────────
    isort, boundaries, umap_embedding
    """
    sd = _base_save_dict(X, unit_ids, n_bins_list, t_ctrs, mouse_arr, reward_arr)
    sd.update(cluster_labels=cluster_labels, isort=isort,
              boundaries=boundaries, umap_embedding=umap_emb)
    np.savez_compressed(path, **sd)
    print(f"  Saved → {Path(path).name}")


def _save_labels_npz(path, X, cluster_labels, unit_ids, n_bins_list,
                     t_ctrs, mouse_arr, reward_arr, **extras):
    """Save any label-only result to .npz (k-means, GMM, …).

    Standard fields (required by cluster_comparison.py)
    ─────────────────────────────────────────────────────
    X, cluster_labels, unit_ids, n_bins_list, t_ctr_*, n_conds,
    mouse_arr, reward_arr

    Pass method-specific arrays via **extras (e.g. gmm_probs, spectral_data).
    """
    sd = _base_save_dict(X, unit_ids, n_bins_list, t_ctrs, mouse_arr, reward_arr)
    sd["cluster_labels"] = cluster_labels
    sd.update(extras)
    np.savez_compressed(path, **sd)
    print(f"  Saved → {Path(path).name}")


def _save_meta_csv(path, unit_ids, cluster_labels, base_meta):
    """Save one row per neuron with cluster_label + neuron metadata."""
    df = pd.DataFrame({"unit_ids": unit_ids,
                       "cluster_label": cluster_labels,
                       **base_meta})
    df.to_csv(path, index=False)
    print(f"  Saved → {Path(path).name}  ({len(df)} neurons)")


def _save_psth_npz(out_folder, X, unit_ids, n_bins_list, t_ctrs, cond_labels):
    """Save per-condition PSTH slices to out_folder/neuron_psth_by_condition.npz.

    Method-independent; safe to call from any entry point since the content
    depends only on the feature matrix, not on the clustering.
    """
    def _sanitize(s):
        return re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    psth_d = dict(unit_ids=np.array(unit_ids), cond_labels=np.array(cond_labels))
    col = 0
    for lbl, nb, tc in zip(cond_labels, n_bins_list, t_ctrs):
        tag = _sanitize(lbl)
        psth_d[f"psth__{tag}"]  = X[:, col:col+nb]
        psth_d[f"t_ctr__{tag}"] = tc
        col += nb
    np.savez_compressed(out_folder / "neuron_psth_by_condition.npz", **psth_d)
    print(f"  Saved → neuron_psth_by_condition.npz")


def _base_meta(meta_df):
    """Extract scalar metadata columns available in neuron_metadata.csv."""
    cols = ["mouse_id", "session_id", "cluster_id", "electrode_group",
            "reward_group", "area_acronym", "waveform_type", "layer_number"]
    return {c: meta_df[c].values for c in cols if c in meta_df.columns}


def _anatomy_figures(cluster_labels, axon_arr, harris_arr, gao_arr, n_k, out_dir):
    """Call fig13 for each anatomy axis when arrays are present."""
    if axon_arr is None:
        return
    fig13_anatomy_axis_per_cluster(cluster_labels, axon_arr,
        "wS1/2 proj.", "avg_ipsi", n_k, out_dir)
    fig13_anatomy_axis_per_cluster(cluster_labels, harris_arr,
        "Hierarchy (Harris)", "cc_tc_ct_iterated", n_k, out_dir)
    fig13_anatomy_axis_per_cluster(cluster_labels, gao_arr,
        "Hierarchy (Gao)", "cc_hierarchy_score_columns", n_k, out_dir)


# ── entry point: rastermap ────────────────────────────────────────────────────

def run_rastermap(
        data_folder:  str | Path,
        config_path:  str | Path = "config.yaml",
        **cfg_overrides,
) -> dict:
    """Run rastermap clustering.  All outputs go to <out_folder>/rastermap/.

    Global mode  (cfg['do_global'] = True)
        Fit rastermap on all trials → population matrix, cluster profiles,
        reward/area/anatomy per-cluster figures, UMAP, PCA scree.

    CV mode  (cfg['cross_validate'] = True)
        Fit rastermap on odd trials, evaluate on even trials →
        CV comparison panels + all cluster figures repeated for even-trial data.

    Saves
    -----
    rastermap/rastermap_results.npz
    rastermap/rastermap_results_cv.npz
    rastermap/neuron_cluster_labels.csv
    rastermap/neuron_cluster_labels_cv.csv
    neuron_psth_by_condition.npz   (clustering root, method-independent)
    """

    print('Running rastermap clustering...')
    cfg         = load_cfg(config_path, **cfg_overrides)
    data_folder = Path(data_folder)

    import rastermap_psth.rastermap_utils as _u
    conds = get_conditions(cfg)
    (_u.CONDITIONS, _u.COND_LABELS, _u.COND_COLORS,
     _u.COND_LABELS_MATRIX, _u.COND_ALIGN_COLS) = conds
    CONDITIONS, COND_LABELS, COND_COLORS, COND_LABELS_MATRIX, COND_ALIGN_COLS = conds

    n_k = cfg["n_rastermap_clusters"]
    out_folder, method_dir = _resolve_dirs(data_folder, cfg, "rastermap")

    print("Loading feature_matrix.npz ...")
    fm = _load_feature_matrix(data_folder)
    X, X_odd, X_even         = fm["X"], fm["X_odd"], fm["X_even"]
    unit_ids, n_bins_list     = fm["unit_ids"], fm["n_bins_list"]
    t_ctrs                    = fm["t_ctrs"]
    mouse_arr, reward_arr     = fm["mouse_arr"], fm["reward_arr"]
    area_arr, waveform_arr    = fm["area_arr"],  fm["waveform_arr"]
    layer_arr, area_group_arr = fm["layer_arr"], fm["area_group_arr"]
    axon_arr, harris_arr, gao_arr = fm["axon_arr"], fm["harris_arr"], fm["gao_arr"]
    meta_df                   = fm["meta_df"]

    group_colors_map = get_custom_area_groups_colors() if _HAS_ALLEN else {}
    anatomy_cmaps    = build_anatomy_cmaps(axon_arr, harris_arr, gao_arr) if axon_arr is not None else None
    vmax    = np.nanpercentile(np.abs(X),     cfg["vmax_pct"])
    vmax_cv = np.nanpercentile(np.abs(X_odd), cfg["vmax_pct"])

    cluster_labels    = np.zeros(len(unit_ids), int)
    cluster_labels_cv = np.zeros(len(unit_ids), int)
    cv_metrics        = None

    # ── global ────────────────────────────────────────────────────────────────
    if cfg.get("do_global", True):
        print(f"\n[rastermap] Global mode ...")
        isort, boundaries = fit_rastermap(X, n_k)
        cluster_labels    = _cluster_labels_from_isort(isort, boundaries, len(unit_ids))

        n_pca = min(150, len(unit_ids)-1, X.shape[1]-1)
        emb   = umap.UMAP(n_neighbors=cfg["umap_n_neighbors"],
                          min_dist=cfg["umap_min_dist"],
                          n_components=2, random_state=42).fit_transform(X)

        fig7_pca_variance(X, n_pca, method_dir)
        fig8_umap(emb, cluster_labels, n_k, method_dir)
        fig5_population_matrix(X, n_bins_list, isort, boundaries, vmax, cfg,
                               reward_arr, waveform_arr, layer_arr, area_group_arr,
                               group_colors_map, method_dir,
                               axon_arr=axon_arr, harris_arr=harris_arr, gao_arr=gao_arr,
                               anatomy_cmaps=anatomy_cmaps, area_arr=area_arr)
        fig6_cluster_profiles(X, t_ctrs, n_bins_list, isort, boundaries, method_dir)
        fig6b_cluster_profiles_reward_groups(X, t_ctrs, n_bins_list, isort, boundaries,
                                             COND_LABELS, COND_COLORS, reward_arr, method_dir)
        fig11_area_per_cluster(unit_ids, cluster_labels, area_arr, n_k, method_dir)
        fig12_reward_per_cluster(unit_ids, cluster_labels, reward_arr, n_k, method_dir)
        _anatomy_figures(cluster_labels, axon_arr, harris_arr, gao_arr, n_k, method_dir)
        _save_rastermap_npz(method_dir / "rastermap_results.npz",
                            X, isort, boundaries, cluster_labels, emb,
                            unit_ids, n_bins_list, t_ctrs, mouse_arr, reward_arr)

    # ── CV ────────────────────────────────────────────────────────────────────
    if cfg.get("cross_validate", True):
        print(f"\n[rastermap] CV mode (odd → even) ...")
        isort_cv, boundaries_cv = fit_rastermap(X_odd, n_k)
        cluster_labels_cv = _cluster_labels_from_isort(isort_cv, boundaries_cv, len(unit_ids))

        emb_cv = umap.UMAP(n_neighbors=cfg["umap_n_neighbors"],
                           min_dist=cfg["umap_min_dist"],
                           n_components=2, random_state=42).fit_transform(X_even)

        figCV_rastermap_comparison(X_odd, X_even, n_bins_list, isort_cv,
                                   boundaries_cv, vmax_cv, cfg, method_dir)
        cv_metrics = figCV_similarity_metrics(
            X_odd, X_even, isort_cv, boundaries_cv, cluster_labels_cv,
            n_bins_list, t_ctrs, COND_LABELS, COND_COLORS, method_dir)
        fig5_population_matrix(X_even, n_bins_list, isort_cv, boundaries_cv,
                               vmax_cv, cfg, reward_arr, waveform_arr, layer_arr,
                               area_group_arr, group_colors_map, method_dir,
                               prefix="cv", axon_arr=axon_arr,
                               harris_arr=harris_arr, gao_arr=gao_arr,
                               anatomy_cmaps=anatomy_cmaps, area_arr=area_arr)
        fig6_cluster_profiles(X_even, t_ctrs, n_bins_list, isort_cv,
                              boundaries_cv, method_dir, prefix="cv")
        fig6b_cluster_profiles_reward_groups(
            X_even, t_ctrs, n_bins_list, isort_cv, boundaries_cv,
            COND_LABELS, COND_COLORS, reward_arr, method_dir, prefix="cv")
        fig8_umap(emb_cv, cluster_labels_cv, n_k, method_dir, prefix="cv")
        fig11_area_per_cluster(unit_ids, cluster_labels_cv, area_arr, n_k, method_dir)
        fig12_reward_per_cluster(unit_ids, cluster_labels_cv, reward_arr, n_k, method_dir)
        _anatomy_figures(cluster_labels_cv, axon_arr, harris_arr, gao_arr, n_k, method_dir)
        _save_rastermap_npz(method_dir / "rastermap_results_cv.npz",
                            X_even, isort_cv, boundaries_cv, cluster_labels_cv, emb_cv,
                            unit_ids, n_bins_list, t_ctrs, mouse_arr, reward_arr)

    # ── shared outputs ────────────────────────────────────────────────────────
    bm = _base_meta(meta_df)
    _save_meta_csv(method_dir / "neuron_cluster_labels.csv",
                   unit_ids, cluster_labels, bm)
    if cfg.get("cross_validate", True):
        _save_meta_csv(method_dir / "neuron_cluster_labels_cv.csv",
                       unit_ids, cluster_labels_cv, bm)
    _save_psth_npz(out_folder, X, unit_ids, n_bins_list, t_ctrs, COND_LABELS)
    run_reward_group_stats(method_dir)

    print(f"\n[rastermap] Done.  Output → {method_dir}")
    return dict(out_folder=out_folder, method_dir=method_dir,
                cluster_labels=cluster_labels, cluster_labels_cv=cluster_labels_cv,
                cv_metrics=cv_metrics)


# ── entry point: k-means ──────────────────────────────────────────────────────

def run_kmeans(
        data_folder:  str | Path,
        config_path:  str | Path = "config.yaml",
        **cfg_overrides,
) -> dict:
    """Run k-means clustering.  All outputs go to <out_folder>/kmeans/.

    Global mode  (cfg['do_global'] = True)
        Fit k-means on all trials → population matrix, cluster profiles,
        elbow curve, UMAP, reward/area/anatomy per-cluster figures.

    CV mode  (cfg['cross_validate'] = True)
        Fit k-means on odd trials; assign even-trial neurons via nearest
        centroid → cluster profiles on even-trial data + npz.

    Saves
    -----
    kmeans/kmeans_results.npz
    kmeans/kmeans_results_cv.npz
    kmeans/neuron_cluster_labels.csv
    kmeans/neuron_cluster_labels_cv.csv
    neuron_psth_by_condition.npz   (clustering root, method-independent)
    """
    print('Running kmeans clustering...')

    cfg         = load_cfg(config_path, **cfg_overrides)
    data_folder = Path(data_folder)

    import rastermap_psth.rastermap_utils as _u
    conds = get_conditions(cfg)
    (_u.CONDITIONS, _u.COND_LABELS, _u.COND_COLORS,
     _u.COND_LABELS_MATRIX, _u.COND_ALIGN_COLS) = conds
    CONDITIONS, COND_LABELS, COND_COLORS, COND_LABELS_MATRIX, COND_ALIGN_COLS = conds

    out_folder, method_dir = _resolve_dirs(data_folder, cfg, "kmeans")

    print("Loading feature_matrix.npz ...")
    fm = _load_feature_matrix(data_folder)
    X, X_odd, X_even         = fm["X"], fm["X_odd"], fm["X_even"]
    unit_ids, n_bins_list     = fm["unit_ids"], fm["n_bins_list"]
    t_ctrs                    = fm["t_ctrs"]
    mouse_arr, reward_arr     = fm["mouse_arr"], fm["reward_arr"]
    area_arr, waveform_arr    = fm["area_arr"],  fm["waveform_arr"]
    layer_arr, area_group_arr = fm["layer_arr"], fm["area_group_arr"]
    axon_arr, harris_arr, gao_arr = fm["axon_arr"], fm["harris_arr"], fm["gao_arr"]
    meta_df                   = fm["meta_df"]

    k = cfg["k_means_k"]
    group_colors_map = get_custom_area_groups_colors() if _HAS_ALLEN else {}
    anatomy_cmaps    = build_anatomy_cmaps(axon_arr, harris_arr, gao_arr) if axon_arr is not None else None
    vmax    = np.nanpercentile(np.abs(X),     cfg["vmax_pct"])
    vmax_cv = np.nanpercentile(np.abs(X_odd), cfg["vmax_pct"])

    km_labels    = np.zeros(len(unit_ids), int)
    km_labels_cv = np.zeros(len(unit_ids), int)

    # ── global ────────────────────────────────────────────────────────────────
    if cfg.get("do_global", True):
        print(f"\n[kmeans] Global mode (k={k}) ...")
        km_fit    = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        km_labels = km_fit.labels_
        isort, boundaries = _labels_to_isort_boundaries(km_labels, k)

        emb = umap.UMAP(n_neighbors=cfg["umap_n_neighbors"],
                        min_dist=cfg["umap_min_dist"],
                        n_components=2, random_state=42).fit_transform(X)
        inertias = Parallel(n_jobs=cfg["n_jobs"])(
            delayed(_kmeans_inertia)(X, ki) for ki in cfg["k_elbow_range"])

        fig9_kmeans(emb, km_labels, k, cfg["k_elbow_range"], inertias, method_dir)
        fig10_kmeans_profiles(X, t_ctrs, n_bins_list, km_labels, k, method_dir)
        fig5b_kmeans_matrix(X, n_bins_list, km_labels, k, vmax, cfg,
                            reward_arr, waveform_arr, layer_arr, area_group_arr,
                            group_colors_map, method_dir,
                            axon_arr=axon_arr, harris_arr=harris_arr, gao_arr=gao_arr,
                            anatomy_cmaps=anatomy_cmaps, area_arr=area_arr)
        fig6_cluster_profiles(X, t_ctrs, n_bins_list, isort, boundaries, method_dir)
        fig6b_cluster_profiles_reward_groups(X, t_ctrs, n_bins_list, isort, boundaries,
                                             COND_LABELS, COND_COLORS, reward_arr, method_dir)
        fig11_area_per_cluster(unit_ids, km_labels, area_arr, k, method_dir)
        fig12_reward_per_cluster(unit_ids, km_labels, reward_arr, k, method_dir)
        _anatomy_figures(km_labels, axon_arr, harris_arr, gao_arr, k, method_dir)
        _save_labels_npz(method_dir / "kmeans_results.npz",
                         X, km_labels, unit_ids, n_bins_list, t_ctrs, mouse_arr, reward_arr)

    # ── CV ────────────────────────────────────────────────────────────────────
    if cfg.get("cross_validate", True):
        print(f"\n[kmeans] CV mode: fit on X_odd (k={k}), assign X_even by nearest centroid ...")
        km_odd       = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_odd)
        km_labels_cv = np.argmin(cdist(X_even, km_odd.cluster_centers_), axis=1)
        isort_cv, boundaries_cv = _labels_to_isort_boundaries(km_labels_cv, k)

        emb_cv = umap.UMAP(n_neighbors=cfg["umap_n_neighbors"],
                           min_dist=cfg["umap_min_dist"],
                           n_components=2, random_state=42).fit_transform(X_even)

        fig5b_kmeans_matrix(X_even, n_bins_list, km_labels_cv, k, vmax_cv, cfg,
                            reward_arr, waveform_arr, layer_arr, area_group_arr,
                            group_colors_map, method_dir,
                            axon_arr=axon_arr, harris_arr=harris_arr, gao_arr=gao_arr,
                            anatomy_cmaps=anatomy_cmaps, area_arr=area_arr,
                            fig_name="fig5b_kmeans_matrix_cv")
        fig6_cluster_profiles(X_even, t_ctrs, n_bins_list, isort_cv,
                              boundaries_cv, method_dir, prefix="cv")
        fig6b_cluster_profiles_reward_groups(
            X_even, t_ctrs, n_bins_list, isort_cv, boundaries_cv,
            COND_LABELS, COND_COLORS, reward_arr, method_dir, prefix="cv")
        fig8_umap(emb_cv, km_labels_cv, k, method_dir, prefix="cv")
        fig11_area_per_cluster(unit_ids, km_labels_cv, area_arr, k, method_dir)
        fig12_reward_per_cluster(unit_ids, km_labels_cv, reward_arr, k, method_dir)
        _anatomy_figures(km_labels_cv, axon_arr, harris_arr, gao_arr, k, method_dir)
        _save_labels_npz(method_dir / "kmeans_results_cv.npz",
                         X_even, km_labels_cv, unit_ids, n_bins_list, t_ctrs,
                         mouse_arr, reward_arr)

    # ── shared outputs ────────────────────────────────────────────────────────
    bm = _base_meta(meta_df)
    _save_meta_csv(method_dir / "neuron_cluster_labels.csv",
                   unit_ids, km_labels, bm)
    if cfg.get("cross_validate", True):
        _save_meta_csv(method_dir / "neuron_cluster_labels_cv.csv",
                       unit_ids, km_labels_cv, bm)
    _save_psth_npz(out_folder, X, unit_ids, n_bins_list, t_ctrs, COND_LABELS)
    run_reward_group_stats(method_dir)

    print(f"\n[kmeans] Done.  Output → {method_dir}")
    return dict(out_folder=out_folder, method_dir=method_dir,
                cluster_labels=km_labels, cluster_labels_cv=km_labels_cv)


# ── entry point: GMM (Spectral Embedding + Gaussian Mixture Model) ────────────
def run_gmm(
        data_folder: str | Path,
        config_path: str | Path = "config.yaml",
        **cfg_overrides,
) -> dict:
    """Run spectral-embedding clustering with Standard GMM and/or Bayesian GMM.

    Both variants share the same PCA + spectral embedding pipeline, computed
    once per mode (global / CV).  Each writes to its own subfolder.

    Standard GMM  (gmm_do_standard=True)
        BIC sweep -> pick K -> GaussianMixture. Output: <out_folder>/gmm/

    Bayesian GMM  (gmm_do_bayesian=True)
        DP-GMM, K_max=100, large concentration -> ~100 active clusters.
        Output: <out_folder>/gmm_bayesian/

    CV (cross_validate=True)
        Nystrom out-of-sample extension maps X_even into X_odd's spectral
        manifold without re-running eigendecomposition. GMM.predict() is then
        called on the Nystrom coordinates.

    Saves
    -----
    gmm/gmm_results.npz  +  gmm/gmm_results_cv.npz
    gmm_bayesian/gmm_bayesian_results.npz  +  gmm_bayesian/gmm_bayesian_results_cv.npz
    neuron_psth_by_condition.npz  (clustering root, method-independent)
    """
    if not _HAS_GMM:
        raise ImportError("gmm_utils unavailable. "
                          "Ensure rastermap_psth/gmm_utils.py is on the Python path.")

    from rastermap_psth.gmm_utils_new import (
        GMM_CFG_DEFAULTS as _GMM_DEFS,
        _row_center, _fit_pca, determine_n_pca,
        tune_sigma, spectral_embedding, nystrom_extension,
        select_k_bic, fit_gmm, fit_bayesian_gmm, bayesian_active_labels,
        figGMM_pca_variance, figGMM_bic_curve, figGMM_eigenvalues,
        figGMM_spectral_scatter, figGMM_assignment_entropy,
        figGMM_active_components, figGMM_nystrom_sanity,
    )

    cfg = load_cfg(config_path, **cfg_overrides)
    data_folder = Path(data_folder)
    gmm_cfg = {**_GMM_DEFS, **cfg}

    import rastermap_psth.rastermap_utils as _u
    conds = get_conditions(cfg)
    (_u.CONDITIONS, _u.COND_LABELS, _u.COND_COLORS,
     _u.COND_LABELS_MATRIX, _u.COND_ALIGN_COLS) = conds
    CONDITIONS, COND_LABELS, COND_COLORS, COND_LABELS_MATRIX, COND_ALIGN_COLS = conds

    out_folder, gmm_dir = _resolve_dirs(data_folder, cfg, "gmm")
    _, gmm_bayes_dir = _resolve_dirs(data_folder, cfg, "gmm_bayesian")

    print("Loading feature_matrix.npz ...")
    fm = _load_feature_matrix(data_folder)
    X, X_odd, X_even = fm["X"], fm["X_odd"], fm["X_even"]
    unit_ids, n_bins_list = fm["unit_ids"], fm["n_bins_list"]
    t_ctrs = fm["t_ctrs"]
    mouse_arr, reward_arr = fm["mouse_arr"], fm["reward_arr"]
    area_arr, waveform_arr = fm["area_arr"], fm["waveform_arr"]
    layer_arr, area_group_arr = fm["layer_arr"], fm["area_group_arr"]
    axon_arr, harris_arr, gao_arr = fm["axon_arr"], fm["harris_arr"], fm["gao_arr"]
    meta_df = fm["meta_df"]

    group_colors_map = get_custom_area_groups_colors() if _HAS_ALLEN else {}
    anatomy_cmaps = build_anatomy_cmaps(axon_arr, harris_arr, gao_arr) \
        if axon_arr is not None else None
    vmax = np.nanpercentile(np.abs(X), cfg["vmax_pct"])
    vmax_cv = np.nanpercentile(np.abs(X_odd), cfg["vmax_pct"])

    n_spectral = int(gmm_cfg.get("gmm_n_spectral", 13))
    do_std = gmm_cfg.get("gmm_do_standard", True)
    do_bay = gmm_cfg.get("gmm_do_bayesian", True)
    do_cv = cfg.get("cross_validate", True)
    K_max = gmm_cfg.get("gmm_bayesian_n_components", 100)

    # ═══════════════════════════════════════════════════════════════════
    # Shared spectral pipeline (computed once, reused by both variants)
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n[gmm] Building global spectral pipeline ...")
    n_pca = determine_n_pca(X, gmm_cfg)
    X_c = _row_center(X)
    PCA_data_global, _ = _fit_pca(X_c, n_pca)
    sigma_g = gmm_cfg.get("gmm_sigma") or tune_sigma(PCA_data_global)
    spectral_g, eigvals_g, eigvecs_g, d_g = \
        spectral_embedding(PCA_data_global, sigma_g, n_spectral)

    spectral_odd = spectral_even = eigvecs_odd = eigvals_odd = d_odd = None
    PCA_data_odd = sigma_cv = None

    if do_cv:
        print(f"\n[gmm] Building CV spectral pipeline (Nystrom) ...")
        X_odd_c = _row_center(X_odd)
        X_even_c = _row_center(X_even)
        PCA_data_odd, pca_cv = _fit_pca(X_odd_c, n_pca)
        sigma_cv = gmm_cfg.get("gmm_sigma") or tune_sigma(PCA_data_odd)
        spectral_odd, eigvals_odd, eigvecs_odd, d_odd = \
            spectral_embedding(PCA_data_odd, sigma_cv, n_spectral)
        PCA_data_even = pca_cv.transform(X_even_c)  # same PCA space as X_odd
        spectral_even = nystrom_extension(
            PCA_data_even, PCA_data_odd, sigma_cv, d_odd, eigvecs_odd, eigvals_odd)
        print(f"  [gmm] Nystrom spectral_even: {spectral_even.shape}")

    # ═══════════════════════════════════════════════════════════════════
    # Standard GMM  →  gmm/
    # ═══════════════════════════════════════════════════════════════════

    std_labels_g = std_labels_cv = None

    if do_std:
        print(f"\n[gmm] Standard GMM ...")

        # global
        if gmm_cfg.get("gmm_k") is None:
            best_k, k_arr, bic_vals = select_k_bic(spectral_g, gmm_cfg)
        else:
            best_k, k_arr, bic_vals = int(gmm_cfg["gmm_k"]), None, None

        best_k = cfg['n_rastermap_clusters']
        gmm_g = fit_gmm(spectral_g, best_k, gmm_cfg)
        std_labels_g = gmm_g.predict(spectral_g)
        std_probs_g = gmm_g.predict_proba(spectral_g)
        isort_g, bounds_g = _labels_to_isort_boundaries(std_labels_g, best_k)

        figGMM_pca_variance(X, n_pca, gmm_dir)
        figGMM_eigenvalues(eigvals_g, n_spectral, gmm_dir)
        figGMM_spectral_scatter(spectral_g, std_labels_g, best_k, gmm_dir)
        figGMM_assignment_entropy(std_probs_g, gmm_dir)
        if k_arr is not None:
            figGMM_bic_curve(k_arr, bic_vals, best_k, gmm_dir)

        fig5b_kmeans_matrix(X, n_bins_list, std_labels_g, best_k, vmax, cfg,
                            reward_arr, waveform_arr, layer_arr, area_group_arr,
                            group_colors_map, gmm_dir,
                            axon_arr=axon_arr, harris_arr=harris_arr, gao_arr=gao_arr,
                            anatomy_cmaps=anatomy_cmaps, area_arr=area_arr,
                            method_label="GMM", fig_name="figGMM_population_matrix")
        fig6_cluster_profiles(X, t_ctrs, n_bins_list, isort_g, bounds_g, gmm_dir)
        fig6b_cluster_profiles_reward_groups(X, t_ctrs, n_bins_list, isort_g, bounds_g,
                                             COND_LABELS, COND_COLORS, reward_arr, gmm_dir)
        fig11_area_per_cluster(unit_ids, std_labels_g, area_arr, best_k, gmm_dir)
        fig12_reward_per_cluster(unit_ids, std_labels_g, reward_arr, best_k, gmm_dir)
        _anatomy_figures(std_labels_g, axon_arr, harris_arr, gao_arr, best_k, gmm_dir)
        _save_labels_npz(gmm_dir / "gmm_results.npz",
                         X, std_labels_g, unit_ids, n_bins_list, t_ctrs,
                         mouse_arr, reward_arr,
                         gmm_probs=std_probs_g, spectral_data=spectral_g,
                         eigenvalues=eigvals_g,
                         n_pca=np.array(n_pca), sigma=np.array(sigma_g),
                         k=np.array(best_k))

        # CV
        if do_cv:
            print(f"  [gmm] Standard GMM CV: fit on X_odd (K={best_k}), predict X_even ...")
            best_k = cfg['n_rastermap_clusters']
            gmm_cv_model = fit_gmm(spectral_odd, best_k, gmm_cfg)
            std_labels_cv = gmm_cv_model.predict(spectral_even)
            std_probs_cv = gmm_cv_model.predict_proba(spectral_even)
            isort_cv, bounds_cv = _labels_to_isort_boundaries(std_labels_cv, best_k)

            figGMM_pca_variance(X_odd, n_pca, gmm_dir, prefix="cv")
            figGMM_eigenvalues(eigvals_odd, n_spectral, gmm_dir, prefix="cv")
            figGMM_spectral_scatter(spectral_even, std_labels_cv, best_k, gmm_dir, prefix="cv")
            figGMM_nystrom_sanity(spectral_odd, spectral_even, gmm_dir)
            figGMM_assignment_entropy(std_probs_cv, gmm_dir, prefix="cv")

            fig5b_kmeans_matrix(X_even, n_bins_list, std_labels_cv, best_k, vmax_cv, cfg,
                                reward_arr, waveform_arr, layer_arr, area_group_arr,
                                group_colors_map, gmm_dir,
                                axon_arr=axon_arr, harris_arr=harris_arr, gao_arr=gao_arr,
                                anatomy_cmaps=anatomy_cmaps, area_arr=area_arr,
                                method_label="GMM", fig_name="cv_figGMM_population_matrix")
            fig6_cluster_profiles(X_even, t_ctrs, n_bins_list, isort_cv,
                                  bounds_cv, gmm_dir, prefix="cv")
            fig6b_cluster_profiles_reward_groups(
                X_even, t_ctrs, n_bins_list, isort_cv, bounds_cv,
                COND_LABELS, COND_COLORS, reward_arr, gmm_dir, prefix="cv")
            fig11_area_per_cluster(unit_ids, std_labels_cv, area_arr, best_k, gmm_dir)
            fig12_reward_per_cluster(unit_ids, std_labels_cv, reward_arr, best_k, gmm_dir)
            _anatomy_figures(std_labels_cv, axon_arr, harris_arr, gao_arr, best_k, gmm_dir)
            _save_labels_npz(gmm_dir / "gmm_results_cv.npz",
                             X_even, std_labels_cv, unit_ids, n_bins_list, t_ctrs,
                             mouse_arr, reward_arr,
                             gmm_probs=std_probs_cv, spectral_data=spectral_even,
                             n_pca=np.array(n_pca), sigma=np.array(sigma_cv),
                             k=np.array(best_k))

        bm = _base_meta(meta_df)
        _save_meta_csv(gmm_dir / "neuron_cluster_labels.csv",
                       unit_ids, std_labels_g, bm)
        if do_cv and std_labels_cv is not None:
            _save_meta_csv(gmm_dir / "neuron_cluster_labels_cv.csv",
                           unit_ids, std_labels_cv, bm)
        run_reward_group_stats(gmm_dir)
        print(f"  [gmm] Standard GMM done. K={best_k}  Output -> {gmm_dir}")

    # ═══════════════════════════════════════════════════════════════════
    # Bayesian GMM  →  gmm_bayesian/
    # ═══════════════════════════════════════════════════════════════════

    bay_labels_g = bay_labels_cv = None

    if do_bay:
        print(f"\n[gmm] Bayesian GMM (DP-GMM, K_max={K_max}) ...")

        # global
        bgmm_g = fit_bayesian_gmm(spectral_g, gmm_cfg)
        bay_labels_g, k_act = bayesian_active_labels(bgmm_g, spectral_g, gmm_cfg)
        bay_probs_g = bgmm_g.predict_proba(spectral_g)
        isort_bg, bounds_bg = _labels_to_isort_boundaries(bay_labels_g, k_act)

        figGMM_pca_variance(X, n_pca, gmm_bayes_dir)
        figGMM_eigenvalues(eigvals_g, n_spectral, gmm_bayes_dir)
        figGMM_spectral_scatter(spectral_g, bay_labels_g, k_act, gmm_bayes_dir)
        figGMM_active_components(bgmm_g, K_max, gmm_cfg, gmm_bayes_dir)
        figGMM_assignment_entropy(bay_probs_g, gmm_bayes_dir)

        fig5b_kmeans_matrix(X, n_bins_list, bay_labels_g, k_act, vmax, cfg,
                            reward_arr, waveform_arr, layer_arr, area_group_arr,
                            group_colors_map, gmm_bayes_dir,
                            axon_arr=axon_arr, harris_arr=harris_arr, gao_arr=gao_arr,
                            anatomy_cmaps=anatomy_cmaps, area_arr=area_arr,
                            method_label="Bayesian GMM",
                            fig_name="figGMM_bayesian_population_matrix")
        fig6_cluster_profiles(X, t_ctrs, n_bins_list, isort_bg, bounds_bg, gmm_bayes_dir)
        fig6b_cluster_profiles_reward_groups(X, t_ctrs, n_bins_list, isort_bg, bounds_bg,
                                             COND_LABELS, COND_COLORS, reward_arr, gmm_bayes_dir)
        fig11_area_per_cluster(unit_ids, bay_labels_g, area_arr, k_act, gmm_bayes_dir)
        fig12_reward_per_cluster(unit_ids, bay_labels_g, reward_arr, k_act, gmm_bayes_dir)
        _anatomy_figures(bay_labels_g, axon_arr, harris_arr, gao_arr, k_act, gmm_bayes_dir)
        _save_labels_npz(gmm_bayes_dir / "gmm_bayesian_results.npz",
                         X, bay_labels_g, unit_ids, n_bins_list, t_ctrs,
                         mouse_arr, reward_arr,
                         gmm_probs=bay_probs_g, spectral_data=spectral_g,
                         eigenvalues=eigvals_g,
                         n_pca=np.array(n_pca), sigma=np.array(sigma_g),
                         k=np.array(k_act), k_max=np.array(K_max))

        # CV
        if do_cv:
            print(f"  [gmm] Bayesian GMM CV: fit on X_odd, predict X_even via Nystrom ...")
            bgmm_cv = fit_bayesian_gmm(spectral_odd, gmm_cfg)
            bay_labels_cv, k_act_cv = bayesian_active_labels(bgmm_cv, spectral_even, gmm_cfg)
            bay_probs_cv = bgmm_cv.predict_proba(spectral_even)
            isort_bcv, bounds_bcv = _labels_to_isort_boundaries(bay_labels_cv, k_act_cv)

            figGMM_pca_variance(X_odd, n_pca, gmm_bayes_dir, prefix="cv")
            figGMM_eigenvalues(eigvals_odd, n_spectral, gmm_bayes_dir, prefix="cv")
            figGMM_spectral_scatter(spectral_even, bay_labels_cv, k_act_cv,
                                    gmm_bayes_dir, prefix="cv")
            figGMM_active_components(bgmm_cv, K_max, gmm_cfg, gmm_bayes_dir, prefix="cv")
            figGMM_nystrom_sanity(spectral_odd, spectral_even, gmm_bayes_dir)
            figGMM_assignment_entropy(bay_probs_cv, gmm_bayes_dir, prefix="cv")

            fig5b_kmeans_matrix(X_even, n_bins_list, bay_labels_cv, k_act_cv, vmax_cv, cfg,
                                reward_arr, waveform_arr, layer_arr, area_group_arr,
                                group_colors_map, gmm_bayes_dir,
                                axon_arr=axon_arr, harris_arr=harris_arr, gao_arr=gao_arr,
                                anatomy_cmaps=anatomy_cmaps, area_arr=area_arr,
                                method_label="Bayesian GMM",
                                fig_name="cv_figGMM_bayesian_population_matrix")
            fig6_cluster_profiles(X_even, t_ctrs, n_bins_list, isort_bcv,
                                  bounds_bcv, gmm_bayes_dir, prefix="cv")
            fig6b_cluster_profiles_reward_groups(
                X_even, t_ctrs, n_bins_list, isort_bcv, bounds_bcv,
                COND_LABELS, COND_COLORS, reward_arr, gmm_bayes_dir, prefix="cv")
            fig11_area_per_cluster(unit_ids, bay_labels_cv, area_arr, k_act_cv, gmm_bayes_dir)
            fig12_reward_per_cluster(unit_ids, bay_labels_cv, reward_arr, k_act_cv, gmm_bayes_dir)
            _anatomy_figures(bay_labels_cv, axon_arr, harris_arr, gao_arr,
                             k_act_cv, gmm_bayes_dir)
            _save_labels_npz(gmm_bayes_dir / "gmm_bayesian_results_cv.npz",
                             X_even, bay_labels_cv, unit_ids, n_bins_list, t_ctrs,
                             mouse_arr, reward_arr,
                             gmm_probs=bay_probs_cv, spectral_data=spectral_even,
                             n_pca=np.array(n_pca), sigma=np.array(sigma_cv),
                             k=np.array(k_act_cv), k_max=np.array(K_max))

        bm = _base_meta(meta_df)
        _save_meta_csv(gmm_bayes_dir / "neuron_cluster_labels.csv",
                       unit_ids, bay_labels_g, bm)
        if do_cv and bay_labels_cv is not None:
            _save_meta_csv(gmm_bayes_dir / "neuron_cluster_labels_cv.csv",
                           unit_ids, bay_labels_cv, bm)
        run_reward_group_stats(gmm_bayes_dir)
        print(f"  [gmm] Bayesian GMM done. K_active={k_act}  Output -> {gmm_bayes_dir}")

    # shared
    _save_psth_npz(out_folder, X, unit_ids, n_bins_list, t_ctrs, COND_LABELS)
    print(f"\n[gmm] All done.")
    return dict(
        out_folder=out_folder,
        gmm_dir=gmm_dir,
        gmm_bayes_dir=gmm_bayes_dir,
        cluster_labels=std_labels_g if do_std else bay_labels_g,
        cluster_labels_cv=std_labels_cv if do_std else bay_labels_cv,
        std_labels=std_labels_g,
        std_labels_cv=std_labels_cv,
        bay_labels=bay_labels_g,
        bay_labels_cv=bay_labels_cv,
    )

def run_gmm_old(
        data_folder:  str | Path,
        config_path:  str | Path = "config.yaml",
        **cfg_overrides,
) -> dict:
    """Run spectral-embedding + GMM clustering.  All outputs go to <out_folder>/gmm/.

    Pipeline
    --------
    PCA neuron coordinates → Gaussian similarity matrix (tuned σ) →
    Normalized Laplacian eigenvectors → GMM with BIC-selected K.

    CV is not supported: the spectral embedding is non-parametric and
    cannot be applied to held-out data.  Use run_cluster_comparison()
    to compare GMM global results against rastermap or k-means.

    Saves
    -----
    gmm/gmm_results.npz
    gmm/neuron_cluster_labels.csv
    neuron_psth_by_condition.npz   (clustering root, method-independent)
    """
    print('Running spectral embedding/gaussian mixture clustering...')

    #if not _HAS_GMM:
    #    raise ImportError(
    #        "gmm_utils is not available.  "
    #        "Ensure rastermap_psth/gmm_utils.py is on the Python path.")

    cfg         = load_cfg(config_path, **cfg_overrides)
    data_folder = Path(data_folder)

    import rastermap_psth.rastermap_utils as _u
    conds = get_conditions(cfg)
    (_u.CONDITIONS, _u.COND_LABELS, _u.COND_COLORS,
     _u.COND_LABELS_MATRIX, _u.COND_ALIGN_COLS) = conds
    CONDITIONS, COND_LABELS, COND_COLORS, COND_LABELS_MATRIX, COND_ALIGN_COLS = conds

    out_folder, method_dir = _resolve_dirs(data_folder, cfg, "gmm")

    print("Loading feature_matrix.npz ...")
    fm = _load_feature_matrix(data_folder)
    X                         = fm["X"]
    unit_ids, n_bins_list     = fm["unit_ids"], fm["n_bins_list"]
    t_ctrs                    = fm["t_ctrs"]
    mouse_arr, reward_arr     = fm["mouse_arr"], fm["reward_arr"]
    area_arr, waveform_arr    = fm["area_arr"],  fm["waveform_arr"]
    layer_arr, area_group_arr = fm["layer_arr"], fm["area_group_arr"]
    axon_arr, harris_arr, gao_arr = fm["axon_arr"], fm["harris_arr"], fm["gao_arr"]
    meta_df                   = fm["meta_df"]

    group_colors_map = get_custom_area_groups_colors() if _HAS_ALLEN else {}
    anatomy_cmaps    = build_anatomy_cmaps(axon_arr, harris_arr, gao_arr) if axon_arr is not None else None
    vmax             = np.nanpercentile(np.abs(X), cfg["vmax_pct"])

    cluster_labels    = np.zeros(len(unit_ids), int)
    cluster_labels_cv = np.zeros(len(unit_ids), int)
    cv_metrics        = None

    # ── global ────────────────────────────────────────────────────────────────
    print(f"\n[gmm] Global mode ...")
    gmm_cfg    = {**GMM_CFG_DEFAULTS, **cfg}
    gmm_result = fit_spectral_gmm(X, gmm_cfg, out_dir=None)
    gmm_labels = gmm_result["gmm_labels"]
    k_gmm      = gmm_result["k"]
    isort, boundaries = _labels_to_isort_boundaries(gmm_labels, k_gmm)

    # GMM-specific diagnostic figures
    figGMM_pca_variance(X, gmm_result["n_pca"], method_dir)
    figGMM_eigenvalues(gmm_result["eigenvalues"],
                       int(gmm_cfg.get("gmm_n_spectral", 13)), method_dir)
    figGMM_spectral_scatter(gmm_result["spectral_data"], gmm_labels, k_gmm, method_dir)
    figGMM_assignment_entropy(gmm_result["gmm_probs"], method_dir)
    if gmm_result["bic_curve"] is not None:
        figGMM_bic_curve(*gmm_result["bic_curve"], k_gmm, method_dir)

    # Shared cluster figures (all methods produce these)
    fig5b_kmeans_matrix(X, n_bins_list, gmm_labels, k_gmm, vmax, cfg,
                        reward_arr, waveform_arr, layer_arr, area_group_arr,
                        group_colors_map, method_dir,
                        axon_arr=axon_arr, harris_arr=harris_arr, gao_arr=gao_arr,
                        anatomy_cmaps=anatomy_cmaps, area_arr=area_arr,
                        method_label="GMM", fig_name="figGMM_population_matrix")
    fig6_cluster_profiles(X, t_ctrs, n_bins_list, isort, boundaries, method_dir)
    fig6b_cluster_profiles_reward_groups(X, t_ctrs, n_bins_list, isort, boundaries,
                                         COND_LABELS, COND_COLORS, reward_arr, method_dir)
    fig11_area_per_cluster(unit_ids, gmm_labels, area_arr, k_gmm, method_dir)
    fig12_reward_per_cluster(unit_ids, gmm_labels, reward_arr, k_gmm, method_dir)
    _anatomy_figures(gmm_labels, axon_arr, harris_arr, gao_arr, k_gmm, method_dir)

    _save_labels_npz(method_dir / "gmm_results.npz",
                     X, gmm_labels, unit_ids, n_bins_list, t_ctrs, mouse_arr, reward_arr,
                     gmm_probs=gmm_result["gmm_probs"],
                     spectral_data=gmm_result["spectral_data"],
                     eigenvalues=gmm_result["eigenvalues"],
                     n_pca=np.array(gmm_result["n_pca"]),
                     sigma=np.array(gmm_result["sigma"]),
                     k=np.array(k_gmm))

    # ── shared outputs ────────────────────────────────────────────────────────
    bm = _base_meta(meta_df)
    _save_meta_csv(method_dir / "neuron_cluster_labels.csv",
                   unit_ids, gmm_labels, bm)
    _save_psth_npz(out_folder, X, unit_ids, n_bins_list, t_ctrs, COND_LABELS)
    run_reward_group_stats(method_dir)

    print(f"\n[gmm] Done.  K={k_gmm}  σ={gmm_result['sigma']:.6f}  "
          f"N_PCA={gmm_result['n_pca']}.  Output → {method_dir}")
    return dict(out_folder=out_folder, method_dir=method_dir,
                cluster_labels=gmm_labels, gmm_result=gmm_result)