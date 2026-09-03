"""
cluster_area_correlations.py
==============================
Correlate anatomical, functional (ROC tuning/selectivity), and
specialization measures at the cluster level and at the area level, from a
rastermap clustering output + a unit_table stacked with ROC results (one row
per unit x analysis_type).

Key design points
------------------
- ROC measures are computed PER analysis_type, not collapsed into one
  "tuning" number. For each analysis_type present in the data:
    * if it's a SELECTIVITY-style comparison (e.g. "wh_vs_aud_active" —
      selective for whisker vs. auditory rather than tuned/untuned), the
      continuous `selectivity_index` column is used, mean-aggregated ->
      "selectivity__<analysis_type>"
    * otherwise the boolean `significant` column is mean-aggregated across
      any duplicate rows for that (unit, analysis_type) -> a continuous
      "degree of tuning" -> "tuning__<analysis_type>"
- All three anatomical hierarchy columns (avg_ipsi, cc_hierarchy_score_columns,
  cc_tc_ct_iterated) are kept and correlated separately, not collapsed to one.
- sorting_index = the rastermap cluster index itself (constant within a
  cluster), a proxy for position along rastermap's continuous 1D ordering.
- Cluster similarity matrices (one per metric, side by side) are plotted
  BEFORE the RSA comparison, so you can visually inspect each metric's
  structure prior to the second-order RSA correlation.
- Area specialization gets its own figures: a sorted bar chart across all
  areas, and a small-multiples "most specialized vs. most generalist"
  cluster-occupancy comparison against the uniform baseline.

Results are saved to:  <rastermap_out_folder>/correlations/
    correlations/by_cluster/       - cluster-level measures, correlations, figures
    correlations/by_area/          - area-level measures (incl. specialization),
                                      correlations, figures, specialization figures
    correlations/rsa/              - similarity matrices + RSA correlation matrix

Usage
-----
    python cluster_area_correlations.py

or import and call run_all(...) — see the __main__ block at the bottom.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon


# ─────────────────────────────────────────────────────────────────────────
# 0. shared stats helper
# ─────────────────────────────────────────────────────────────────────────
def _bh_correction(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns (pvals_fdr, reject)."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    fdr = ranked * n / (np.arange(n) + 1)
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    fdr = np.clip(fdr, 0, 1)
    pvals_fdr = np.empty(n)
    pvals_fdr[order] = fdr
    reject = pvals_fdr < alpha
    return pvals_fdr, reject


# ─────────────────────────────────────────────────────────────────────────
# 1. loading rastermap output
# ─────────────────────────────────────────────────────────────────────────
def _load_rastermap_output(out_folder):
    """Load the pieces of a rastermap embedding_results*.npz needed for
    downstream correlation/RSA analyses."""
    out_folder = Path(out_folder)
    cv_result_file = [f for f in os.listdir(out_folder) if f.endswith("results_cv.npz")]
    if not cv_result_file:
        raise FileNotFoundError(f"No *results_cv.npz found in {out_folder}")
    data = np.load(out_folder / cv_result_file[0], allow_pickle=True)
    return dict(
        cluster_labels=data["cluster_labels"],
        mouse_arr=data["mouse_arr"].astype(str),
        reward_arr=data["reward_arr"].astype(str),
        unit_ids=data["unit_ids"],
    )


def _load_area_arr(out_folder, unit_ids):
    """Load area_acronym per neuron from the CV metadata CSV, aligned to
    unit_ids by an explicit merge (not position)."""
    out_folder = Path(out_folder)
    for name in ("neuron_cluster_labels_cv.csv", "neuron_cluster_labels.csv"):
        for candidate in (out_folder / name, out_folder.parent / name):
            if candidate.exists():
                meta_df = pd.read_csv(candidate)
                id_col = "unit_id" if "unit_id" in meta_df.columns else "unit_ids"
                meta_df = meta_df.set_index(id_col)
                meta_df.index = meta_df.index.astype(str)
                area_arr = meta_df.reindex(np.asarray(unit_ids).astype(str))["area_acronym"].to_numpy()
                missing = pd.isna(area_arr).sum()
                if missing:
                    print(f"  Warning: {missing}/{len(unit_ids)} unit_ids not found in {candidate.name}")
                return area_arr, candidate
    raise FileNotFoundError(f"Could not find neuron_cluster_labels[_cv].csv near {out_folder}")


# ─────────────────────────────────────────────────────────────────────────
# 2. ROC measures, per analysis_type (tuning OR selectivity, generalized)
# ─────────────────────────────────────────────────────────────────────────
# analysis_types whose meaningful summary is direction/degree of SELECTIVITY
# (continuous, can be signed) rather than significant/not-significant.
DEFAULT_SELECTIVITY_ANALYSIS_TYPES = {"wh_vs_aud_active"}


def build_roc_measures_wide(unit_table, id_col_candidates=("unit_id", "unit_ids"),
                             analysis_type_col="analysis_type",
                             sig_col="significant", selectivity_col="selectivity_index",
                             selectivity_analysis_types=DEFAULT_SELECTIVITY_ANALYSIS_TYPES):
    """Pivot a long unit_table (one row per unit x analysis_type) into one
    row per unit_id, with one measure column per analysis_type:
        - "tuning__<analysis_type>"      = mean(sig_col)          [default]
        - "selectivity__<analysis_type>" = mean(selectivity_col)  [if that
          analysis_type is in selectivity_analysis_types]

    Mean-aggregation across any residual duplicate rows for the same
    (unit, analysis_type) turns a boolean into a continuous "degree of
    tuning" (fraction of repeats/sessions significant), rather than
    collapsing to a single yes/no.

    Returns: (wide_df indexed by unit_id (str), list of measure column names created)
    """
    ut = unit_table.copy()
    id_col = next((c for c in id_col_candidates if c in ut.columns), None)
    if id_col is None:
        raise KeyError(f"Could not find a unit id column among {id_col_candidates} "
                        f"in unit_table (columns: {list(ut.columns)[:10]}...)")
    ut[id_col] = ut[id_col].astype(str)

    if analysis_type_col not in ut.columns:
        raise KeyError(f"analysis_type_col '{analysis_type_col}' not found in unit_table "
                        f"(columns: {list(ut.columns)[:10]}...)")

    analysis_types = sorted(ut[analysis_type_col].dropna().unique())
    print(f"  Found {len(analysis_types)} ROC analysis_type(s): {analysis_types}")

    measure_series = {}
    measure_cols = []
    for atype in analysis_types:
        sub = ut[ut[analysis_type_col] == atype]
        if atype in selectivity_analysis_types:
            if selectivity_col not in sub.columns:
                print(f"  Warning: analysis_type '{atype}' requested selectivity_index but "
                      f"column '{selectivity_col}' not found — skipping")
                continue
            col_name = f"selectivity__{atype}"
            vals = pd.to_numeric(sub[selectivity_col], errors="coerce")
        else:
            if sig_col not in sub.columns:
                print(f"  Warning: analysis_type '{atype}' expected significance column "
                      f"'{sig_col}' but it was not found — skipping")
                continue
            col_name = f"tuning__{atype}"
            vals = pd.to_numeric(sub[sig_col], errors="coerce")

        agg = vals.groupby(sub[id_col]).mean()
        measure_series[col_name] = agg
        measure_cols.append(col_name)

    wide_df = pd.DataFrame(measure_series)
    wide_df.index.name = id_col
    return wide_df, measure_cols


# ─────────────────────────────────────────────────────────────────────────
# 3. neuron-level merge: rastermap clusters + ROC measures + anatomy
# ─────────────────────────────────────────────────────────────────────────
def merge_units_with_clusters(out_folder, unit_table, anatomical_cols,
                               analysis_type_col="analysis_type",
                               sig_col="significant", selectivity_col="selectivity_index",
                               selectivity_analysis_types=DEFAULT_SELECTIVITY_ANALYSIS_TYPES,
                               id_col_candidates=("unit_id", "unit_ids")):
    """Build one neuron-level DataFrame combining rastermap cluster
    assignment, per-analysis_type ROC tuning/selectivity measures, and
    anatomical hierarchy scores — matched by unit_id.

    unit_table: long format, one row per unit x analysis_type (e.g. from a
    merge with roc_df); may have duplicate unit_id rows.
    anatomical_cols: list of unit_table column names that are constant per
    unit (not analysis_type-dependent), e.g.
        ["avg_ipsi", "cc_hierarchy_score_columns", "cc_tc_ct_iterated"]
    kept and correlated as SEPARATE measures, not collapsed to one score.

    Returns: (neuron_df, cluster_labels, roc_measure_cols, anatomical_cols_found)
    """
    rm = _load_rastermap_output(out_folder)
    cluster_labels, unit_ids = rm["cluster_labels"], rm["unit_ids"]

    area_arr, area_src = _load_area_arr(out_folder, unit_ids)
    print(f"  Loaded area_acronym from {area_src.name}")

    ut = unit_table.reset_index() if unit_table.index.name in id_col_candidates else unit_table.copy()
    id_col = next((c for c in id_col_candidates if c in ut.columns), None)
    if id_col is None:
        raise KeyError(f"Could not find a unit id column among {id_col_candidates} "
                        f"in unit_table (columns: {list(ut.columns)[:10]}...)")
    ut[id_col] = ut[id_col].astype(str)

    # ── per-analysis_type ROC measures (wide) ───────────────────────────────
    roc_wide, roc_measure_cols = build_roc_measures_wide(
        ut, id_col_candidates=(id_col,), analysis_type_col=analysis_type_col,
        sig_col=sig_col, selectivity_col=selectivity_col,
        selectivity_analysis_types=selectivity_analysis_types)

    # ── anatomical columns: constant per unit across analysis_type rows —
    #    sanity-check that, then take one value per unit ──────────────────
    anatomical_cols_found = [c for c in anatomical_cols if c in ut.columns]
    missing_anat = [c for c in anatomical_cols if c not in ut.columns]
    if missing_anat:
        print(f"  Warning: anatomical columns not found in unit_table, skipping: {missing_anat}")

    for c in anatomical_cols_found:
        n_unique_per_unit = ut.groupby(id_col)[c].nunique(dropna=True)
        inconsistent = int((n_unique_per_unit > 1).sum())
        if inconsistent:
            print(f"  Warning: '{c}' varies across analysis_type rows for {inconsistent} units "
                  f"— using each unit's first value. Verify this column is truly unit-constant.")

    anat_df = (ut.drop_duplicates(subset=id_col)
                 .set_index(id_col)[anatomical_cols_found]) if anatomical_cols_found else pd.DataFrame(index=[])

    ut_dedup = roc_wide.join(anat_df, how="outer")
    sub = ut_dedup.reindex(np.asarray(unit_ids).astype(str))

    df = pd.DataFrame({
        "unit_id": np.asarray(unit_ids).astype(str),
        "cluster": cluster_labels,
        "area_acronym": area_arr,
    })
    for col in roc_measure_cols + anatomical_cols_found:
        if col in sub.columns:
            df[col] = pd.to_numeric(sub[col], errors="coerce").to_numpy()

    # sorting_index = the rastermap cluster index itself (constant within a
    # cluster by construction) — a proxy for position along rastermap's
    # continuous 1D functional-similarity ordering.
    df["sorting_index"] = cluster_labels.astype(float)

    measure_cols_all = roc_measure_cols + anatomical_cols_found
    if measure_cols_all:
        n_bad = df[measure_cols_all].isna().all(axis=1).sum()
        if n_bad:
            print(f"  Warning: {n_bad}/{len(df)} neurons have no matched unit_table data at all "
                  f"(unit_id mismatch between npz and unit_table?)")

    return df, cluster_labels, roc_measure_cols, anatomical_cols_found


# ─────────────────────────────────────────────────────────────────────────
# 4. area specialization + figures
# ─────────────────────────────────────────────────────────────────────────
def compute_area_specialization(cluster_labels, area_arr, n_clusters=None, min_neurons=10,
                                 baseline="uniform"):
    """For each area: how non-uniformly its neurons are distributed across
    clusters. High = area's neurons concentrate in few clusters
    ('specialized'). Low = spread across many clusters ('generalist').

    baseline: "uniform" (1/n_clusters everywhere) or "dataset" (dataset-wide
    cluster-size distribution — controls for clusters not being equal-sized).
    Effect size = Jensen-Shannon distance, base 2, bounded [0, 1].
    """
    if n_clusters is None:
        n_clusters = int(cluster_labels.max()) + 1

    if baseline == "uniform":
        ref_dist = np.full(n_clusters, 1.0 / n_clusters)
    elif baseline == "dataset":
        counts_all = np.bincount(cluster_labels, minlength=n_clusters).astype(float)
        ref_dist = counts_all / counts_all.sum()
    else:
        raise ValueError('baseline must be "uniform" or "dataset"')

    rows = []
    for area in np.unique(area_arr):
        mask = area_arr == area
        n_a = int(mask.sum())
        if n_a < min_neurons:
            continue
        counts = np.bincount(cluster_labels[mask], minlength=n_clusters).astype(float)
        dist = counts / counts.sum()
        js = jensenshannon(dist, ref_dist, base=2)
        rows.append(dict(area_acronym=area, n_neurons=n_a, specialization=js))

    df = pd.DataFrame(rows).set_index("area_acronym")
    print(f"  Computed specialization for {len(df)} areas (min {min_neurons} neurons each, "
          f"baseline='{baseline}')")
    return df


def _fig_area_specialization_bar(spec_df, out_dir):
    """Sorted bar chart of specialization across all qualifying areas."""
    sorted_df = spec_df.sort_values("specialization", ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(sorted_df)), 4.5))
    ax.bar(range(len(sorted_df)), sorted_df["specialization"],
           color="steelblue", edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(sorted_df)))
    ax.set_xticklabels(sorted_df.index, rotation=75, ha="right", fontsize=7)
    for i, n in enumerate(sorted_df["n_neurons"]):
        ax.text(i, sorted_df["specialization"].iloc[i], f"n={int(n)}",
                 ha="center", va="bottom", fontsize=5, rotation=90)
    ax.set_ylabel("Specialization (JS distance from baseline)")
    ax.set_title("Area specialization across clusters\n(higher = neurons concentrate in fewer clusters)")
    fig.tight_layout()
    fig.savefig(out_dir / "area_specialization_bar.png", dpi=150)
    plt.close(fig)


def _fig_area_cluster_occupancy_examples(cluster_labels, area_arr, spec_df, out_dir,
                                          n_examples=6, n_clusters=None):
    """Intermediate figure: for the N most specialized and N most generalist
    areas, show their actual cluster-occupancy distribution against the
    uniform baseline — makes the specialization number concrete."""
    if n_clusters is None:
        n_clusters = int(cluster_labels.max()) + 1
    n_examples = min(n_examples, len(spec_df) // 2) if len(spec_df) >= 2 else 0
    if n_examples == 0:
        print("  Not enough areas to plot specialization examples, skipping")
        return

    top = spec_df.sort_values("specialization", ascending=False).head(n_examples)
    bottom = spec_df.sort_values("specialization", ascending=True).head(n_examples)

    fig, axes = plt.subplots(2, n_examples, figsize=(2.4 * n_examples, 5), squeeze=False)
    for row, (label, subset) in enumerate([("Most specialized", top), ("Most generalist", bottom)]):
        for col, (area, r) in enumerate(subset.iterrows()):
            ax = axes[row, col]
            mask = area_arr == area
            counts = np.bincount(cluster_labels[mask], minlength=n_clusters)
            frac = counts / counts.sum()
            ax.bar(range(n_clusters), frac, color="steelblue", width=1.0)
            ax.axhline(1.0 / n_clusters, color="crimson", lw=1, ls="--")
            ax.set_title(f"{area}\nJS={r['specialization']:.2f}, n={int(r['n_neurons'])}", fontsize=7)
            ax.set_xticks([]); ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)

    fig.suptitle("Cluster-occupancy distribution: most specialized vs. most generalist areas\n"
                 "(dashed red = uniform baseline)")
    fig.tight_layout()
    fig.savefig(out_dir / "area_specialization_occupancy_examples.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# 5. generic neuron -> entity aggregation
# ─────────────────────────────────────────────────────────────────────────
def compute_entity_level_measures(neuron_df, group_col, measure_cols, min_neurons=10):
    """Generic neuron -> entity aggregation (entity = 'cluster' or
    'area_acronym', whichever column name is passed as group_col)."""
    grouped = neuron_df.groupby(group_col)
    n_neurons = grouped.size()
    valid = n_neurons[n_neurons >= min_neurons].index

    out = pd.DataFrame(index=valid)
    out.index.name = group_col
    out["n_neurons"] = n_neurons.reindex(valid)
    for col in measure_cols:
        if col in neuron_df.columns:
            out[col] = grouped[col].mean().reindex(valid)
        else:
            print(f"  Warning: '{col}' not in neuron_df, skipping for {group_col}-level aggregation")
    return out


# ─────────────────────────────────────────────────────────────────────────
# 6. generic pairwise correlation with permutation significance
# ─────────────────────────────────────────────────────────────────────────
def compute_measure_correlations(entity_df, measures, n_perm=5000, rng=None):
    """Pairwise Spearman correlations across entity-level measures, with
    permutation-based p-values and BH-FDR correction across all pairs.
    Rows with NaN in either of a pair are dropped pairwise, not listwise.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_m = len(measures)
    corr = np.full((n_m, n_m), np.nan)
    pval_raw = np.full((n_m, n_m), np.nan)
    n_used = np.full((n_m, n_m), np.nan)

    pairs = [(i, j) for i in range(n_m) for j in range(i + 1, n_m)]
    for i, j in pairs:
        a = entity_df[measures[i]].to_numpy()
        b = entity_df[measures[j]].to_numpy()
        valid = np.isfinite(a) & np.isfinite(b)
        n = valid.sum()
        n_used[i, j] = n_used[j, i] = n
        if n < 4:
            continue
        a_v, b_v = a[valid], b[valid]
        obs_r, _ = spearmanr(a_v, b_v)
        corr[i, j] = corr[j, i] = obs_r

        null_r = np.empty(n_perm)
        for p in range(n_perm):
            null_r[p], _ = spearmanr(a_v, rng.permutation(b_v))
        p_raw = (1 + np.sum(np.abs(null_r) >= np.abs(obs_r))) / (n_perm + 1)
        pval_raw[i, j] = pval_raw[j, i] = p_raw

    np.fill_diagonal(corr, 1.0)
    np.fill_diagonal(pval_raw, 0.0)

    flat_p = np.array([pval_raw[i, j] for i, j in pairs])
    valid_p = ~np.isnan(flat_p)
    flat_fdr = np.full(len(flat_p), np.nan)
    if valid_p.any():
        flat_fdr[valid_p], _ = _bh_correction(flat_p[valid_p], alpha=0.05)
    pval_fdr = np.full((n_m, n_m), np.nan)
    for (i, j), p in zip(pairs, flat_fdr):
        pval_fdr[i, j] = pval_fdr[j, i] = p
    np.fill_diagonal(pval_fdr, 0.0)

    corr_df   = pd.DataFrame(corr, index=measures, columns=measures)
    pval_df   = pd.DataFrame(pval_raw, index=measures, columns=measures)
    pfdr_df   = pd.DataFrame(pval_fdr, index=measures, columns=measures)
    n_used_df = pd.DataFrame(n_used, index=measures, columns=measures)
    return corr_df, pval_df, pfdr_df, n_used_df


# ─────────────────────────────────────────────────────────────────────────
# 7. figures: correlation heatmap + scatter grid (reused for cluster/area/RSA)
# ─────────────────────────────────────────────────────────────────────────
def _fig_measure_correlation_heatmap(corr_df, pfdr_df, out_dir, alpha=0.05,
                                      fname="measure_correlation_heatmap.png",
                                      title="Measure correlations\n(* = significant after BH-FDR, permutation p-values)"):
    n = len(corr_df)
    fig, ax = plt.subplots(figsize=(max(4, 1.0 * n + 2), max(4, 1.0 * n + 1)))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = corr_df.values[i, j]
            if np.isnan(r):
                continue
            star = "*" if (not np.isnan(pfdr_df.values[i, j]) and pfdr_df.values[i, j] < alpha) else ""
            ax.text(j, i, f"{r:.2f}{star}", ha="center", va="center", fontsize=7,
                     color="white" if abs(r) > 0.5 else "black")

    ax.set_xticks(range(n)); ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(corr_df.index, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Spearman r", fontsize=9)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def _fig_measure_scatter_grid(entity_df, measures, corr_df, pfdr_df, out_dir, alpha=0.05,
                               fname="measure_scatter_grid.png"):
    pairs = [(a, b) for i, a in enumerate(measures) for b in measures[i + 1:]]
    if not pairs:
        return
    ncols = min(5, len(pairs)); nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows), squeeze=False)

    for idx, (ma, mb) in enumerate(pairs):
        ax = axes[idx // ncols, idx % ncols]
        x, y = entity_df[ma], entity_df[mb]
        valid = x.notna() & y.notna()
        sig = (not np.isnan(pfdr_df.loc[ma, mb])) and pfdr_df.loc[ma, mb] < alpha
        ax.scatter(x[valid], y[valid], s=26, color="crimson" if sig else "grey",
                   edgecolors="black", linewidths=0.4)
        r = corr_df.loc[ma, mb]
        ax.set_title(f"{ma}\nvs {mb}\nr={r:.2f}{' *' if sig else ''}", fontsize=7)
        ax.set_xlabel(ma, fontsize=6); ax.set_ylabel(mb, fontsize=6)
        ax.tick_params(labelsize=6)

    for idx in range(len(pairs), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    fig.suptitle("Pairwise scatter, all tested measure pairs (red = significant, BH-FDR)")
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# 8. cluster-level and area-level correlation runners
# ─────────────────────────────────────────────────────────────────────────
def run_cluster_level_correlations(neuron_df, measure_cols, correlations_dir,
                                    min_neurons=10, n_perm=5000):
    out_dir = correlations_dir / "by_cluster"
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_measures_df = compute_entity_level_measures(
        neuron_df, group_col="cluster", measure_cols=measure_cols, min_neurons=min_neurons)

    measures = [c for c in measure_cols if c in cluster_measures_df.columns]
    corr_df, pval_df, pfdr_df, n_used_df = compute_measure_correlations(
        cluster_measures_df, measures, n_perm=n_perm)

    cluster_measures_df.to_csv(out_dir / "cluster_level_measures.csv")
    corr_df.to_csv(out_dir / "cluster_measure_correlations.csv")
    pfdr_df.to_csv(out_dir / "cluster_measure_correlations_pfdr.csv")
    n_used_df.to_csv(out_dir / "cluster_measure_correlations_n.csv")

    _fig_measure_correlation_heatmap(corr_df, pfdr_df, out_dir,
                                      title="Cluster-level measure correlations")
    _fig_measure_scatter_grid(cluster_measures_df, measures, corr_df, pfdr_df, out_dir)

    print(f"  Cluster-level correlations ({len(cluster_measures_df)} clusters, "
          f"{len(measures)} measures) → {out_dir}")
    return cluster_measures_df, corr_df, pfdr_df


def run_area_level_correlations(neuron_df, cluster_labels, measure_cols, correlations_dir,
                                 min_neurons=10, n_perm=5000, specialization_baseline="uniform"):
    out_dir = correlations_dir / "by_area"
    out_dir.mkdir(parents=True, exist_ok=True)

    area_arr = neuron_df["area_acronym"].to_numpy()
    spec_df = compute_area_specialization(cluster_labels, area_arr, min_neurons=min_neurons,
                                           baseline=specialization_baseline)
    _fig_area_specialization_bar(spec_df, out_dir)
    _fig_area_cluster_occupancy_examples(cluster_labels, area_arr, spec_df, out_dir)

    area_measures_df = compute_entity_level_measures(
        neuron_df, group_col="area_acronym", measure_cols=measure_cols, min_neurons=min_neurons)

    area_df = spec_df.join(area_measures_df.drop(columns=["n_neurons"], errors="ignore"), how="inner")

    measures = ["specialization"] + [c for c in measure_cols if c in area_df.columns]
    corr_df, pval_df, pfdr_df, n_used_df = compute_measure_correlations(
        area_df, measures, n_perm=n_perm)

    area_df.to_csv(out_dir / "area_level_measures.csv")
    corr_df.to_csv(out_dir / "area_measure_correlations.csv")
    pfdr_df.to_csv(out_dir / "area_measure_correlations_pfdr.csv")
    n_used_df.to_csv(out_dir / "area_measure_correlations_n.csv")

    _fig_measure_correlation_heatmap(corr_df, pfdr_df, out_dir,
                                      title="Area-level measure correlations (incl. specialization)")
    _fig_measure_scatter_grid(area_df, measures, corr_df, pfdr_df, out_dir)

    print(f"  Area-level correlations ({len(area_df)} areas, "
          f"{len(measures)} measures, incl. specialization) → {out_dir}")
    return area_df, corr_df, pfdr_df


# ─────────────────────────────────────────────────────────────────────────
# 9. cluster similarity matrices (before RSA) + RSA
# ─────────────────────────────────────────────────────────────────────────
def build_rdm_from_scalar(entity_df, measure, metric="abs_diff"):
    """K x K representational dissimilarity matrix from a single per-entity
    scalar column. metric: 'abs_diff' or 'sq_diff'."""
    v = entity_df[measure].to_numpy()
    if metric == "abs_diff":
        return np.abs(v[:, None] - v[None, :])
    elif metric == "sq_diff":
        return (v[:, None] - v[None, :]) ** 2
    raise ValueError('metric must be "abs_diff" or "sq_diff"')


def _fig_cluster_similarity_matrices(rdm_dict, out_dir, fname="cluster_similarity_matrices.png"):
    """Side-by-side similarity matrices (1 - normalized RDM), one panel per
    metric, same cluster ordering across all panels — an intermediate,
    purely descriptive figure shown BEFORE the second-order RSA comparison,
    so each metric's structure can be inspected on its own first."""
    names = list(rdm_dict.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4), squeeze=False)
    axes = axes[0]
    im = None
    for ax, name in zip(axes, names):
        rdm = rdm_dict[name]
        denom = np.nanmax(rdm)
        sim = 1 - rdm / denom if denom > 0 else np.ones_like(rdm)
        im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    cbar = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02, shrink=0.85)
    cbar.set_label("Similarity (1 - normalized |Δ|)", fontsize=9)
    fig.suptitle("Cluster similarity matrices per metric (same cluster order, before RSA)")
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compare_rdms(rdm_dict, n_perm=5000, rng=None):
    """Second-order RSA comparison: for every pair of RDMs (same entity
    ordering across all of them), Spearman-correlate their vectorized upper
    triangles, with permutation significance and BH-FDR across all pairs.
    """
    if rng is None:
        rng = np.random.default_rng()
    names = list(rdm_dict.keys())
    n = rdm_dict[names[0]].shape[0]
    iu = np.triu_indices(n, k=1)

    corr = np.full((len(names), len(names)), np.nan)
    pval_raw = np.full((len(names), len(names)), np.nan)

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if j <= i:
                continue
            a, b = rdm_dict[ni][iu], rdm_dict[nj][iu]
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() < 6:
                continue
            obs_r, _ = spearmanr(a[valid], b[valid])
            corr[i, j] = corr[j, i] = obs_r

            null_r = np.empty(n_perm)
            b_full = rdm_dict[nj]
            for p in range(n_perm):
                perm = rng.permutation(n)
                shuffled = b_full[np.ix_(perm, perm)][iu]
                null_r[p], _ = spearmanr(a[valid], shuffled[valid])
            p_raw = (1 + np.sum(np.abs(null_r) >= np.abs(obs_r))) / (n_perm + 1)
            pval_raw[i, j] = pval_raw[j, i] = p_raw

    np.fill_diagonal(corr, 1.0); np.fill_diagonal(pval_raw, 0.0)
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    flat_p = np.array([pval_raw[i, j] for i, j in pairs])
    valid_p = ~np.isnan(flat_p)
    flat_fdr = np.full(len(flat_p), np.nan)
    if valid_p.any():
        flat_fdr[valid_p], _ = _bh_correction(flat_p[valid_p], alpha=0.05)
    pfdr = np.full_like(corr, np.nan)
    for (i, j), p in zip(pairs, flat_fdr):
        pfdr[i, j] = pfdr[j, i] = p
    np.fill_diagonal(pfdr, 0.0)

    return (pd.DataFrame(corr, index=names, columns=names),
            pd.DataFrame(pval_raw, index=names, columns=names),
            pd.DataFrame(pfdr, index=names, columns=names))


def run_cluster_rsa(cluster_measures_df, correlations_dir, extra_rdms=None, n_perm=5000):
    """Build one RDM per cluster-level measure plus any extra precomputed
    RDMs (e.g. spatial centroid distance — caller must ensure these use the
    SAME cluster ordering as cluster_measures_df). Plots the per-metric
    similarity matrices side by side FIRST, then runs the RSA comparison.
    """
    out_dir = correlations_dir / "rsa"
    out_dir.mkdir(parents=True, exist_ok=True)

    rdms = {m: build_rdm_from_scalar(cluster_measures_df, m)
            for m in cluster_measures_df.columns if m != "n_neurons"}
    if extra_rdms:
        rdms.update(extra_rdms)

    _fig_cluster_similarity_matrices(rdms, out_dir)

    rsa_corr, rsa_pval, rsa_pfdr = compare_rdms(rdms, n_perm=n_perm)

    rsa_corr.to_csv(out_dir / "rsa_correlation_matrix.csv")
    rsa_pfdr.to_csv(out_dir / "rsa_correlation_matrix_pfdr.csv")
    _fig_measure_correlation_heatmap(rsa_corr, rsa_pfdr, out_dir, fname="rsa_correlation_heatmap.png",
                                      title="RSA: second-order correlation across metric RDMs")

    print(f"  RSA across {len(rdms)} RDMs, {len(cluster_measures_df)} clusters → {out_dir}")
    return rsa_corr, rsa_pfdr, rdms


# ─────────────────────────────────────────────────────────────────────────
# 10. top-level orchestrator
# ─────────────────────────────────────────────────────────────────────────
def run_all(rastermap_out_folder, unit_table, anatomical_cols,
            analysis_type_col="analysis_type", sig_col="significant",
            selectivity_col="selectivity_index",
            selectivity_analysis_types=DEFAULT_SELECTIVITY_ANALYSIS_TYPES,
            min_neurons=10, n_perm=5000, specialization_baseline="uniform",
            extra_rdms=None):
    """Run the full correlation + RSA pipeline. Results saved under
    <rastermap_out_folder>/correlations/.
    """
    rastermap_out_folder = Path(rastermap_out_folder)
    correlations_dir = rastermap_out_folder / "correlations"
    correlations_dir.mkdir(parents=True, exist_ok=True)

    neuron_df, cluster_labels, roc_measure_cols, anatomical_cols_found = merge_units_with_clusters(
        rastermap_out_folder, unit_table, anatomical_cols,
        analysis_type_col=analysis_type_col, sig_col=sig_col, selectivity_col=selectivity_col,
        selectivity_analysis_types=selectivity_analysis_types)

    measure_cols = roc_measure_cols + anatomical_cols_found + ["sorting_index"]
    print(f"  Full measure set ({len(measure_cols)}): {measure_cols}")

    cluster_measures_df, cluster_corr_df, cluster_pfdr_df = run_cluster_level_correlations(
        neuron_df, measure_cols, correlations_dir, min_neurons=min_neurons, n_perm=n_perm)

    area_df, area_corr_df, area_pfdr_df = run_area_level_correlations(
        neuron_df, cluster_labels, measure_cols, correlations_dir,
        min_neurons=min_neurons, n_perm=n_perm, specialization_baseline=specialization_baseline)

    rsa_corr, rsa_pfdr, rdms = run_cluster_rsa(
        cluster_measures_df, correlations_dir, extra_rdms=extra_rdms, n_perm=n_perm)

    print(f"\nAll correlation/RSA results saved under: {correlations_dir}")
    return dict(
        neuron_df=neuron_df,
        cluster_measures_df=cluster_measures_df, cluster_corr_df=cluster_corr_df, cluster_pfdr_df=cluster_pfdr_df,
        area_df=area_df, area_corr_df=area_corr_df, area_pfdr_df=area_pfdr_df,
        rsa_corr=rsa_corr, rsa_pfdr=rsa_pfdr,
    )


# ─────────────────────────────────────────────────────────────────────────
# example call
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RASTERMAP_OUT_FOLDER = (
        "/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_clustering/"
        "passive_active/whisker_auditory/clustering/"
        "n100_passive_active_baseline_whisker_auditory_combined_nobl/rastermap"
    )

    # unit_table: long format, one row per unit x analysis_type, from a
    # merge with roc_df. Must have columns for unit id, "analysis_type",
    # "significant" (bool), "selectivity_index" (continuous), and the
    # anatomical hierarchy score columns.
    # unit_table = pd.read_csv("path/to/unit_table_merged_with_roc.csv")

    anatomical_cols = ["avg_ipsi", "cc_hierarchy_score_columns", "cc_tc_ct_iterated"]

    results = run_all(
        RASTERMAP_OUT_FOLDER,
        unit_table,               # noqa: F821 -- supply this before running
        anatomical_cols,
        analysis_type_col="analysis_type",
        sig_col="significant",
        selectivity_col="selectivity_index",
        selectivity_analysis_types={"wh_vs_aud_active"},  # add more selectivity-style types here if needed
        min_neurons=10,
        n_perm=5000,
        specialization_baseline="uniform",
    )