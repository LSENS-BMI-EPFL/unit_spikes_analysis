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
- Area specialization: 'specialization' is the raw Jensen-Shannon distance
  of an area's cluster-occupancy distribution from the baseline (uniform or
  dataset-wide) — its difference from uniformity. This raw value is
  CONFOUNDED by sample size (areas with fewer neurons show inflated apparent
  specialization purely from multinomial sampling noise, the same bias that
  affects naive entropy/diversity estimators at small N). Alongside it,
  'specialization_null_*' columns report a sample-size-corrected version:
  a null distribution of JS-divergence is simulated for each area's actual N
  (assuming pure random sampling from the baseline), and the observed value
  is expressed as a z-score / p-value / bias-subtracted value relative to
  that null. Downstream correlations use specialization_null_z, not the raw
  (confounded) 'specialization'.
- Correlations use Pearson r. Two p-values are reported per pair:
    * a permutation-based p-value (BH-FDR corrected across all pairs tested,
      though the star currently uses the UNCORRECTED permutation p by
      request — BH-FDR is computed and saved to CSV but unused for now);
    * the analytic Pearson p-value from scipy's pearsonr — reported
      alongside r in the heatmap and scatter grid as a familiar reference
      value, NOT used for the significance star.
- Area-level correlation scatter points are colored by each area's custom
  anatomical group (allen_utils.get_custom_area_groups), using Allen's
  default group colors (allen_utils.get_custom_area_groups_allen_colors),
  with each point individually labeled by its area acronym and a single
  shared group->color legend per figure.
- In addition to the compact scatter GRID (all pairs, small panels), a
  single full-size figure with an OLS regression line is saved per pair,
  under a 'scatter_pairs/' subfolder — useful for inspecting or presenting
  one relationship at a time.

Results are saved to:  <rastermap_out_folder>/correlations/
    correlations/by_cluster/                 - cluster-level measures, correlations, figures
        by_cluster/scatter_pairs/            - one regression-line figure per measure pair
    correlations/by_area/                    - area-level measures (incl. specialization),
                                                correlations, figures, specialization figures
        by_area/scatter_pairs/               - one regression-line figure per measure pair
    correlations/rsa/                        - similarity matrices + RSA correlation matrix

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
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from ephys_utilities.allen_utils.allen_utils import (
    get_custom_area_groups, get_custom_area_groups_allen_colors, get_isocortex_mapping)


# ─────────────────────────────────────────────────────────────────────────
# 0. shared stats helpers
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


def _format_p(p):
    """Compact p-value string for figure annotations."""
    if p is None or np.isnan(p):
        return "p=n/a"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


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
    """Load area_acronym_custom per neuron from the CV metadata CSV, aligned
    to unit_ids by an explicit merge (not position)."""
    out_folder = Path(out_folder)
    for name in ("neuron_cluster_labels_cv.csv", "neuron_cluster_labels.csv"):
        for candidate in (out_folder / name, out_folder.parent / name):
            if candidate.exists():
                meta_df = pd.read_csv(candidate)
                id_col = "unit_id" if "unit_id" in meta_df.columns else "unit_ids"
                meta_df = meta_df.set_index(id_col)
                meta_df.index = meta_df.index.astype(str)
                area_arr = meta_df.reindex(np.asarray(unit_ids).astype(str))["area_acronym_custom"].to_numpy()
                missing = pd.isna(area_arr).sum()
                if missing:
                    print(f"  Warning: {missing}/{len(unit_ids)} unit_ids not found in {candidate.name}")
                return area_arr, candidate
    raise FileNotFoundError(f"Could not find neuron_cluster_labels[_cv].csv near {out_folder}")


def _build_area_group_colors(areas):
    """Map each area_acronym_custom to its custom area group (via
    allen_utils.get_custom_area_groups) and the corresponding Allen default
    color (via allen_utils.get_custom_area_groups_allen_colors).

    Returns: (area_colors, area_groups) — both dicts keyed by area acronym.
    Areas not found in any custom group are colored grey under 'Other'.
    """
    groups = get_custom_area_groups()                     # {group_name: [acronyms]}
    group_colors = get_custom_area_groups_allen_colors()   # {group_name: color}
    acronym_to_group = {a: g for g, acs in groups.items() for a in acs}

    area_groups, area_colors = {}, {}
    unmapped = []
    for area in areas:
        group = acronym_to_group.get(area)
        if group is None:
            group = "Other"
            unmapped.append(area)
        area_groups[area] = group
        area_colors[area] = group_colors.get(group, "grey")
    if unmapped:
        print(f"  Warning: {len(unmapped)} area(s) not found in get_custom_area_groups(), "
              f"colored grey under 'Other': {unmapped}")
    return area_colors, area_groups


# ─────────────────────────────────────────────────────────────────────────
# 2. ROC measures, per analysis_type (tuning OR selectivity, generalized)
# ─────────────────────────────────────────────────────────────────────────
# analysis_types whose meaningful summary is direction/degree of SELECTIVITY
# (continuous, can be signed) rather than significant/not-significant.
DEFAULT_SELECTIVITY_ANALYSIS_TYPES = {"wh_vs_aud_active"}

# Some anatomical hierarchy scores are only meaningful for specific brain
# regions (e.g. a corticothalamic hierarchy score computed only from
# isocortex + thalamic circuitry doesn't mean anything for a striatal
# neuron). For each such column, list the (region_column, allowed_value)
# conditions that qualify a neuron to be INCLUDED (OR'd together) — neurons
# not matching any condition get that one column set to NaN (so they're
# excluded only from analyses of that specific measure, not from everything).
DEFAULT_ANATOMICAL_REGION_FILTERS = {
    "cc_tc_ct_iterated":          [("isocortex_group", "Isocortex"), ("area_group", "Thalamus")],
    "cc_hierarchy_score_columns": [("isocortex_group", "Isocortex")],
}


def _apply_anatomical_region_filters(neuron_df, region_filters):
    """For each measure_col in region_filters, keep only neurons matching at
    least one of its (region_column, allowed_value) conditions; everyone
    else gets that column set to NaN. Filtering is per-column, so a neuron
    excluded from one anatomical measure can still contribute to others.
    """
    for measure_col, conditions in region_filters.items():
        if measure_col not in neuron_df.columns:
            continue
        mask = np.zeros(len(neuron_df), dtype=bool)
        for region_col, allowed_val in conditions:
            if region_col not in neuron_df.columns:
                print(f"  Warning: region-filter column '{region_col}' not found in unit_table — "
                      f"skipping that condition for '{measure_col}'")
                continue
            mask |= (neuron_df[region_col] == allowed_val)

        n_before = neuron_df[measure_col].notna().sum()
        neuron_df.loc[~mask, measure_col] = np.nan
        n_after = neuron_df[measure_col].notna().sum()
        cond_str = " OR ".join(f"{c}=='{v}'" for c, v in conditions)
        print(f"  Restricted '{measure_col}' to neurons where {cond_str}: "
              f"{n_after}/{n_before} non-null values kept")
    return neuron_df


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
def _load_neuron_join_keys(out_folder, unit_ids, key_cols=("session_id", "electrode_group", "cluster_id")):
    """Load (session_id, electrode_group, cluster_id) per neuron from the CV
    metadata CSV, aligned to unit_ids by an explicit merge (not position).
    Used as the join key against unit_table instead of unit_id, since
    unit_id construction can differ between the rastermap pipeline's saved
    metadata and unit_table."""
    out_folder = Path(out_folder)
    for name in ("neuron_cluster_labels_cv.csv", "neuron_cluster_labels.csv"):
        for candidate in (out_folder / name, out_folder.parent / name):
            if candidate.exists():
                meta_df = pd.read_csv(candidate)
                id_col = "unit_id" if "unit_id" in meta_df.columns else "unit_ids"
                missing_cols = [c for c in key_cols if c not in meta_df.columns]
                if missing_cols:
                    raise KeyError(f"{candidate.name} missing join-key column(s) {missing_cols}")
                meta_df = meta_df.set_index(id_col)
                meta_df.index = meta_df.index.astype(str)
                sub = meta_df.reindex(np.asarray(unit_ids).astype(str))[list(key_cols)]
                missing = sub.isna().any(axis=1).sum()
                if missing:
                    print(f"  Warning: {missing}/{len(unit_ids)} unit_ids not found in {candidate.name} "
                          f"(or missing one of {key_cols})")
                return sub, candidate
    raise FileNotFoundError(f"Could not find neuron_cluster_labels[_cv].csv near {out_folder}")


def _make_join_key(df, key_cols):
    """Concatenate key_cols into one string key per row; NaN in any key_col
    propagates to NaN for the whole key (never silently coerced to the
    literal string 'nan', which could spuriously collide across rows)."""
    parts = [df[c].astype(str) for c in key_cols]
    key = parts[0]
    for p in parts[1:]:
        key = key + "||" + p
    any_na = df[list(key_cols)].isna().any(axis=1)
    return key.mask(any_na)


# ─────────────────────────────────────────────────────────────────────────
def merge_units_with_clusters(out_folder, unit_table, anatomical_cols,
                               analysis_type_col="analysis_type",
                               sig_col="significant",
                              selectivity_col="selectivity",
                               selectivity_analysis_types=DEFAULT_SELECTIVITY_ANALYSIS_TYPES,
                               region_filters=DEFAULT_ANATOMICAL_REGION_FILTERS,
                               join_key_cols=("session_id", "electrode_group", "cluster_id")):
    """Build one neuron-level DataFrame combining rastermap cluster
    assignment, per-analysis_type ROC tuning/selectivity measures, and
    anatomical hierarchy scores — matched by (session_id, electrode_group,
    cluster_id), NOT unit_id (unit_id construction can differ between the
    rastermap pipeline's saved metadata and unit_table, so the compound key
    is the reliable match).

    unit_table: long format, one row per unit x analysis_type (e.g. from a
    merge with roc_df); may have duplicate rows per join key.
    anatomical_cols: list of unit_table column names that are constant per
    unit (not analysis_type-dependent), e.g.
        ["avg_ipsi", "cc_hierarchy_score_columns", "cc_tc_ct_iterated"]
    kept and correlated as SEPARATE measures, not collapsed to one score.
    region_filters: dict restricting specific anatomical measures to
    specific regions (see DEFAULT_ANATOMICAL_REGION_FILTERS) — any
    region-group column referenced there (e.g. "isocortex_group",
    "area_group") is recomputed from area_acronym_custom, not fetched.
    join_key_cols: unit_table columns forming the unique per-neuron key,
    matched against the same-named columns in the rastermap CV metadata CSV.

    Returns: (neuron_df, cluster_labels, roc_measure_cols, anatomical_cols_found)
    """
    rm = _load_rastermap_output(out_folder)
    cluster_labels, unit_ids = rm["cluster_labels"], rm["unit_ids"]

    area_arr, area_src = _load_area_arr(out_folder, unit_ids)
    print(f"  Loaded area_custom_acronym from {area_src.name}")

    neuron_keys_df, keys_src = _load_neuron_join_keys(out_folder, unit_ids, key_cols=join_key_cols)
    print(f"  Loaded join keys {join_key_cols} from {keys_src.name}")
    neuron_join_key = _make_join_key(neuron_keys_df, join_key_cols).to_numpy()

    ut = unit_table.reset_index() if unit_table.index.name in ("unit_id", "unit_ids") else unit_table.copy()
    missing_key_cols = [c for c in join_key_cols if c not in ut.columns]
    if missing_key_cols:
        raise KeyError(f"unit_table missing join-key column(s) {missing_key_cols} "
                        f"(columns: {list(ut.columns)[:10]}...)")
    ut["_join_key"] = _make_join_key(ut, join_key_cols)
    n_bad_ut_key = ut["_join_key"].isna().sum()
    if n_bad_ut_key:
        print(f"  Warning: {n_bad_ut_key}/{len(ut)} unit_table rows have a NaN in "
              f"{join_key_cols} and are dropped before joining")
        ut = ut[ut["_join_key"].notna()]

    # ── per-analysis_type ROC measures (wide), grouped by the composite key ──
    roc_wide, roc_measure_cols = build_roc_measures_wide(
        ut, id_col_candidates=("_join_key",), analysis_type_col=analysis_type_col,
        sig_col=sig_col, selectivity_col=selectivity_col,
        selectivity_analysis_types=selectivity_analysis_types)

    # ── anatomical columns, constant per unit across analysis_type rows ────
    # region_cols_needed (isocortex_group / area_group) is recomputed below
    # directly from area_arr, never fetched from unit_table.
    region_cols_needed = sorted({col for conds in region_filters.values() for col, _ in conds})
    anatomical_cols_found = [c for c in anatomical_cols if c in ut.columns]
    missing_anat = [c for c in anatomical_cols if c not in ut.columns]
    if missing_anat:
        print(f"  Warning: anatomical columns not found in unit_table, skipping: {missing_anat}")

    for c in anatomical_cols_found:
        n_unique_per_key = ut.groupby("_join_key")[c].nunique(dropna=True)
        inconsistent = int((n_unique_per_key > 1).sum())
        if inconsistent:
            print(f"  Warning: '{c}' varies across analysis_type rows for {inconsistent} units "
                  f"— using each unit's first value. Verify this column is truly unit-constant.")

    anat_df = (ut.drop_duplicates(subset="_join_key")
                 .set_index("_join_key")[anatomical_cols_found]) if anatomical_cols_found else pd.DataFrame(index=[])

    ut_dedup = roc_wide.join(anat_df, how="outer")
    sub = ut_dedup.reindex(neuron_join_key)

    df = pd.DataFrame({
        "unit_id": np.asarray(unit_ids).astype(str),
        "cluster": cluster_labels,
        "area_acronym_custom": area_arr,
    })
    for col in roc_measure_cols + anatomical_cols_found:
        if col in sub.columns:
            df[col] = pd.to_numeric(sub[col], errors="coerce").to_numpy()

    # Region-group columns (isocortex_group, area_group) are recomputed here
    # directly from area_arr — the SAME area_acronym_custom source used for
    # df["area_acronym_custom"] above — rather than pulled from unit_table.
    # unit_table's own isocortex_group/area_group were baked in from
    # whatever area_acronym_custom it held at a possibly different pipeline
    # stage; if that ever disagrees with area_arr (the rastermap CV CSV),
    # a neuron could be grouped under one area here while its region-filter
    # columns reflect a different area's classification — silently letting
    # e.g. hippocampal neurons pass an Isocortex-only filter. Recomputing
    # from area_arr guarantees the grouping and the filter can never disagree.
    if "isocortex_group" in region_cols_needed:
        isocortex2group = {a: g for g, areas in get_isocortex_mapping().items() for a in areas}
        df["isocortex_group"] = pd.Series(area_arr).map(isocortex2group).fillna("Other").to_numpy()
    if "area_group" in region_cols_needed:
        area2group = {a: g for g, areas in get_custom_area_groups().items() for a in areas}
        df["area_group"] = pd.Series(area_arr).map(area2group).fillna("Other").to_numpy()

    # sorting_index = the rastermap cluster index itself (constant within a
    # cluster by construction) — a proxy for position along rastermap's
    # continuous 1D functional-similarity ordering.
    df["sorting_index"] = cluster_labels.astype(float)

    df = _apply_anatomical_region_filters(df, region_filters)

    # drop the region-group columns now that filtering is done — they're not
    # measures to correlate, just the criteria used to restrict others
    region_only_cols = [c for c in region_cols_needed if c not in anatomical_cols_found]
    df = df.drop(columns=region_only_cols, errors="ignore")

    measure_cols_all = roc_measure_cols + anatomical_cols_found
    if measure_cols_all:
        n_bad = df[measure_cols_all].isna().all(axis=1).sum()
        if n_bad:
            print(f"  WARNING TO SOLVE: {n_bad}/{len(df)} neurons have no matched unit_table data at all "
                  f"(join-key mismatch between rastermap CSV and unit_table?)")

    return df, cluster_labels, roc_measure_cols, anatomical_cols_found


# ─────────────────────────────────────────────────────────────────────────
# 4. area specialization + figures
# ─────────────────────────────────────────────────────────────────────────
def _simulate_js_null(n_a, ref_dist, n_null=2000, rng=None):
    """Null distribution of JS-divergence-from-ref_dist for a sample of n_a
    categorical draws generated EXACTLY according to ref_dist — i.e. the
    divergence expected from pure finite-sample multinomial noise alone,
    with zero true specialization. Depends only on n_a and the number of
    clusters, not on any real cluster assignment.
    """
    if rng is None:
        rng = np.random.default_rng()
    null_js = np.empty(n_null)
    for i in range(n_null):
        counts = rng.multinomial(n_a, ref_dist)
        dist = counts / n_a
        null_js[i] = jensenshannon(dist, ref_dist, base=2)
    return null_js


def compute_area_specialization(cluster_labels, area_arr, n_clusters=None, min_neurons=100,
                                 baseline="uniform", n_null=2000, rng=None):
    """For each area: how non-uniformly its neurons are distributed across
    clusters. High = area's neurons concentrate in few clusters
    ('specialized'). Low = spread across many clusters ('generalist').

    baseline: "uniform" (1/n_clusters everywhere) or "dataset" (dataset-wide
    cluster-size distribution — controls for clusters not being equal-sized).

    'specialization' = raw Jensen-Shannon distance from the baseline (its
    difference from uniformity), base 2, bounded [0, 1]. This raw value is
    CONFOUNDED by sample size — areas with fewer neurons show inflated
    apparent specialization purely from multinomial sampling noise (same
    bias affecting naive entropy/diversity estimators at small N).

    'specialization_null_*' columns correct for this: for each area, a null
    distribution of JS-divergence values expected from pure sampling noise
    AT THAT AREA'S OWN N is simulated (_simulate_js_null), and the observed
    value is reported relative to it:
        - specialization_null_mean / _null_std : the simulated null's moments
        - specialization_null_z    : (raw - null_mean) / null_std — comparable
                                      across areas regardless of N
        - specialization_null_p    : empirical one-sided p-value (fraction of
                                      null draws >= observed)
        - specialization_null_p_fdr / _significant : BH-FDR across areas
        - specialization_null_corrected : raw - null_mean, bias-subtracted,
                                      same units as the raw JS distance, ~0
                                      for a truly non-specialized area
                                      regardless of N

    Use specialization_null_z (or specialization_null_corrected) instead of
    the raw 'specialization' column for any downstream correlation/RSA
    analysis, since only the null-corrected version is comparable across
    areas of differing sample size.
    """
    if rng is None:
        rng = np.random.default_rng()
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

        null_js = _simulate_js_null(n_a, ref_dist, n_null=n_null, rng=rng)
        null_mean, null_std = null_js.mean(), null_js.std()
        z = (js - null_mean) / null_std if null_std > 0 else np.nan
        p = (1 + np.sum(null_js >= js)) / (n_null + 1)

        rows.append(dict(area_acronym_custom=area, n_neurons=n_a, specialization=js,
                          specialization_null_mean=null_mean, specialization_null_std=null_std,
                          specialization_null_z=z, specialization_null_p=p,
                          specialization_null_corrected=js - null_mean))

    df = pd.DataFrame(rows).set_index("area_acronym_custom")
    if len(df):
        df["specialization_null_p_fdr"], df["specialization_null_significant"] = _bh_correction(
            df["specialization_null_p"].to_numpy(), alpha=0.05)
    print(f"  Computed specialization (raw + sample-size-corrected 'specialization_null_*') for "
          f"{len(df)} areas (min {min_neurons} neurons each, baseline='{baseline}', "
          f"{n_null} null draws/area)")
    return df


def _fig_area_specialization_bar(spec_df, out_dir):
    """Sorted bar chart of sample-size-corrected specialization (z-score)
    across all qualifying areas, colored by significance (BH-FDR). The raw
    (confounded) 'specialization' value is annotated for reference."""
    sorted_df = spec_df.sort_values("specialization_null_z", ascending=False)
    colors = ["crimson" if s else "steelblue" for s in sorted_df["specialization_null_significant"]]
    fig, ax = plt.subplots(figsize=(max(6, 0.15 * len(sorted_df)), 4.5))
    ax.bar(range(len(sorted_df)), sorted_df["specialization_null_z"],
           color=colors, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(sorted_df)))
    ax.set_xticklabels(sorted_df.index, rotation=75, ha="right", fontsize=7)
    for i, n in enumerate(sorted_df["n_neurons"]):
        ax.text(i, sorted_df["specialization_null_z"].iloc[i], f"n={int(n)}",
                 ha="center", va="bottom", fontsize=5, rotation=90)
    ax.set_ylabel("Specialization z-score (sample-size corrected, vs. null)")
    ax.set_title("Area specialization across clusters, bias-corrected\n"
                 "(higher = concentrates in fewer clusters beyond chance; red = significant, BH-FDR)")
    fig.tight_layout()
    fig.savefig(out_dir / "area_specialization_bar.pdf", dpi=150)
    plt.close(fig)


def _fig_specialization_bias_check(spec_df, out_dir):
    """Diagnostic figure: raw specialization (difference to uniformity) vs.
    N — should show the sample-size bias — next to the null-corrected
    z-score vs. N, which should show that bias removed."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(spec_df["n_neurons"], spec_df["specialization"],
                     color="steelblue", edgecolors="black", linewidths=0.4)
    axes[0].set_xlabel("N neurons in area"); axes[0].set_ylabel("Raw specialization\n(difference to uniformity)")
    axes[0].set_title("Raw: confounded by sample size")

    colors = ["crimson" if s else "grey" for s in spec_df["specialization_null_significant"]]
    axes[1].scatter(spec_df["n_neurons"], spec_df["specialization_null_z"],
                     color=colors, edgecolors="black", linewidths=0.4)
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    axes[1].set_xlabel("N neurons in area"); axes[1].set_ylabel("specialization_null_z")
    axes[1].set_title("Corrected: red = significant, BH-FDR")

    fig.suptitle("Sample-size bias check for area specialization")
    fig.tight_layout()
    fig.savefig(out_dir / "area_specialization_bias_check.pdf", dpi=150)
    plt.close(fig)


    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(spec_df["n_neurons"], spec_df["specialization"],
                     color="steelblue", edgecolors="black", linewidths=0.4)
    axes[0].set_xlabel("N neurons in area"); axes[0].set_ylabel("Raw specialization\n(difference to uniformity)")
    axes[0].set_title("Raw: confounded by sample size")

    colors = ["crimson" if s else "grey" for s in spec_df["specialization_null_significant"]]
    axes[1].scatter(spec_df["n_neurons"], spec_df["specialization_null_corrected"],
                     color=colors, edgecolors="black", linewidths=0.4)
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    axes[1].set_xlabel("N neurons in area"); axes[1].set_ylabel("specialization_null_corrected")
    axes[1].set_title("Corrected: red = significant, BH-FDR")

    fig.suptitle("Sample-size bias check for area specialization corrected")
    fig.tight_layout()
    fig.savefig(out_dir / "area_specialization_corrected_bias_check.pdf", dpi=150)
    plt.close(fig)
    return


def _fig_area_cluster_occupancy_examples(cluster_labels, area_arr, spec_df, out_dir,
                                          n_examples=6, n_clusters=None):
    """Intermediate figure: for the N most specialized (by null-corrected
    z-score) and N most generalist areas, show their actual cluster-
    occupancy distribution against the uniform baseline — makes the
    specialization number concrete."""
    if n_clusters is None:
        n_clusters = int(cluster_labels.max()) + 1
    n_examples = min(n_examples, len(spec_df) // 2) if len(spec_df) >= 2 else 0
    if n_examples == 0:
        print("  Not enough areas to plot specialization examples, skipping")
        return

    top = spec_df.sort_values("specialization_null_z", ascending=False).head(n_examples)
    bottom = spec_df.sort_values("specialization_null_z", ascending=True).head(n_examples)

    fig, axes = plt.subplots(2, n_examples, figsize=(2.4 * n_examples, 5), squeeze=False,
                             sharex=False, sharey=False)
    for row, (label, subset) in enumerate([("Most specialized", top), ("Most generalist", bottom)]):
        for col, (area, r) in enumerate(subset.iterrows()):
            ax = axes[row, col]
            mask = area_arr == area
            counts = np.bincount(cluster_labels[mask], minlength=n_clusters)
            frac = counts / counts.sum()
            ax.bar(range(n_clusters), frac, color="steelblue", width=1.0)
            ax.axhline(1.0 / n_clusters, color="crimson", lw=1, ls="--")
            ax.set_title(f"{area}\nz={r['specialization_null_z']:.2f}, n={int(r['n_neurons'])}", fontsize=7)
            ax.set_xticks([]); ax.tick_params(labelsize=6)
            ax.set_ylabel('Cluster occupancy', fontsize=8)
            ax.set_xlabel('Clusters', fontsize=8)

    fig.suptitle("Cluster-occupancy distribution: most specialized to most generalist areas\n"
                 "(dashed red = uniform baseline; ranked by sample-size-corrected z-score)")
    fig.tight_layout()
    fig.savefig(out_dir / "area_specialization_occupancy_examples.pdf", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# 5. generic neuron -> entity aggregation
# ─────────────────────────────────────────────────────────────────────────
def compute_entity_level_measures(neuron_df, group_col, measure_cols, min_neurons=10):
    """Generic neuron -> entity aggregation (entity = 'cluster' or
    'area_acronym_custom', whichever column name is passed as group_col)."""
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
# 6. generic pairwise correlation with permutation + analytic p-values
# ─────────────────────────────────────────────────────────────────────────
def compute_measure_correlations(entity_df, measures, n_perm=5000, rng=None):
    """Pairwise Pearson correlations across entity-level measures.

    Two kinds of p-value are computed per pair:
      - pval_perm: permutation-based (BH-FDR corrected across all pairs) —
        computed and saved, but NOT currently used for the significance star
        (uncorrected pval_perm is used instead, by request).
      - pval_analytic: the analytic p-value returned directly by
        scipy.stats.pearsonr — reported alongside r as a familiar reference
        value, NOT used for significance decisions.

    Rows with NaN in either of a pair are dropped pairwise, not listwise.

    Returns: corr_df, pval_perm_df (raw), pval_perm_fdr_df, n_used_df,
             pval_analytic_df
    """
    if rng is None:
        rng = np.random.default_rng()

    n_m = len(measures)
    corr          = np.full((n_m, n_m), np.nan)
    pval_perm_raw = np.full((n_m, n_m), np.nan)
    pval_analytic = np.full((n_m, n_m), np.nan)
    n_used        = np.full((n_m, n_m), np.nan)

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

        obs_r, p_analytic = pearsonr(a_v, b_v)
        corr[i, j] = corr[j, i] = obs_r
        pval_analytic[i, j] = pval_analytic[j, i] = p_analytic

        null_r = np.empty(n_perm)
        for p in range(n_perm):
            null_r[p], _ = pearsonr(a_v, rng.permutation(b_v))
        p_perm = (1 + np.sum(np.abs(null_r) >= np.abs(obs_r))) / (n_perm + 1)
        pval_perm_raw[i, j] = pval_perm_raw[j, i] = p_perm

    np.fill_diagonal(corr, 1.0)
    np.fill_diagonal(pval_perm_raw, 0.0)
    np.fill_diagonal(pval_analytic, 0.0)

    flat_p = np.array([pval_perm_raw[i, j] for i, j in pairs])
    valid_p = ~np.isnan(flat_p)
    flat_fdr = np.full(len(flat_p), np.nan)
    if valid_p.any():
        flat_fdr[valid_p], _ = _bh_correction(flat_p[valid_p], alpha=0.05)
    pval_perm_fdr = np.full((n_m, n_m), np.nan)
    for (i, j), p in zip(pairs, flat_fdr):
        pval_perm_fdr[i, j] = pval_perm_fdr[j, i] = p
    np.fill_diagonal(pval_perm_fdr, 0.0)

    corr_df           = pd.DataFrame(corr, index=measures, columns=measures)
    pval_perm_df      = pd.DataFrame(pval_perm_raw, index=measures, columns=measures)
    pval_perm_fdr_df  = pd.DataFrame(pval_perm_fdr, index=measures, columns=measures)
    n_used_df         = pd.DataFrame(n_used, index=measures, columns=measures)
    pval_analytic_df  = pd.DataFrame(pval_analytic, index=measures, columns=measures)
    return corr_df, pval_perm_df, pval_perm_fdr_df, n_used_df, pval_analytic_df


# ─────────────────────────────────────────────────────────────────────────
# 7. figures: correlation heatmap + scatter grid (reused for cluster/area/RSA)
# ─────────────────────────────────────────────────────────────────────────
def _fig_measure_correlation_heatmap(corr_df, pfdr_df, out_dir, alpha=0.05,
                                      fname="measure_correlation_heatmap.pdf",
                                      title="Measure correlations\n(* = significant, permutation p-values)",
                                      p_analytic_df=None):
    """pfdr_df drives the significance star. NOTE: call sites currently pass
    the UNCORRECTED permutation p-values here (not BH-FDR) by request — BH-FDR
    is still computed and saved to CSV alongside, just not used for the star
    yet. p_analytic_df, if given, is displayed as an extra annotation line
    (analytic Pearson p) alongside r — purely descriptive, not used for the
    star either way.
    """
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
            text = f"r={r:.2f}{star}"
            if p_analytic_df is not None:
                p_val = p_analytic_df.values[i, j]
                text += f"\n{_format_p(p_val)}"
            ax.text(j, i, text, ha="center", va="center", fontsize=6,
                     color="white" if abs(r) > 0.5 else "black")

    ax.set_xticks(range(n)); ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(corr_df.index, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Pearson r", fontsize=9)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def _fig_measure_scatter_grid(entity_df, measures, corr_df, pfdr_df, out_dir, alpha=0.05,
                               fname="measure_scatter_grid.pdf", p_analytic_df=None,
                               point_colors=None, point_groups=None):
    """
    point_colors: optional dict {entity index value: color}. If given,
    points are colored by this (e.g. Allen area-group colors) instead of the
    default significance-based red/grey, and each point is annotated with
    its own entity label (small font, offset) since color alone can't
    distinguish areas sharing the same group.
    point_groups: optional dict {entity index value: group_name}, used only
    to build one shared legend (group -> color) for the whole figure.
    """
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

        if point_colors is not None:
            entities = list(entity_df.index[valid])
            colors = [point_colors.get(e, "grey") for e in entities]
            ax.scatter(x[valid], y[valid], s=34, c=colors, alpha=0.8,
                       linewidths=0.0, zorder=3)
            for xv, yv, e in zip(x[valid], y[valid], entities):
                ax.annotate(str(e), (xv, yv), fontsize=4.5, xytext=(2, 2),
                            textcoords="offset points", alpha=0.85, zorder=4)
        else:
            ax.scatter(x[valid], y[valid], s=26, color="crimson" if sig else "grey",
                       edgecolors="black", linewidths=0.4)

        r = corr_df.loc[ma, mb]
        title = f"{ma}\nvs {mb}\nr={r:.2f}{' *' if sig else ''}"
        if p_analytic_df is not None:
            title += f"\n{_format_p(p_analytic_df.loc[ma, mb])}"
        ax.set_title(title, fontsize=7)
        ax.set_xlabel(ma, fontsize=6); ax.set_ylabel(mb, fontsize=6)
        ax.tick_params(labelsize=6)

    for idx in range(len(pairs), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    suptitle = "Pairwise scatter, all tested measure pairs"
    if point_groups is not None and point_colors is not None:
        group_to_color = {}
        for e, g in point_groups.items():
            if e in point_colors and g not in group_to_color:
                group_to_color[g] = point_colors[e]
        handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                               markeredgecolor="black", markersize=7, label=g)
                   for g, c in sorted(group_to_color.items())]
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6),
                   fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.03))
        suptitle += " (colored by Allen area group; points labeled by area)"
    else:
        suptitle += " (red = significant, BH-FDR)"

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_measure_scatter_single(entity_df, ma, mb, corr_df, pfdr_df, out_dir, alpha=0.05,
                                 p_analytic_df=None, point_colors=None, point_groups=None):
    """Single full-size scatter for one measure pair, with an OLS regression
    line overlaid. Companion to the compact grid — one file per pair."""
    x, y = entity_df[ma], entity_df[mb]
    valid = x.notna() & y.notna()
    sig = (not np.isnan(pfdr_df.loc[ma, mb])) and pfdr_df.loc[ma, mb] < alpha

    fig, ax = plt.subplots(figsize=(5.2, 4.6))

    if point_colors is not None:
        entities = list(entity_df.index[valid])
        colors = [point_colors.get(e, "grey") for e in entities]
        ax.scatter(x[valid], y[valid], s=55, c=colors, alpha=0.85, linewidths=0.0, zorder=3)
        for xv, yv, e in zip(x[valid], y[valid], entities):
            ax.annotate(str(e), (xv, yv), fontsize=6.5, xytext=(3, 3),
                        textcoords="offset points", alpha=0.85, zorder=4)
    else:
        ax.scatter(x[valid], y[valid], s=40, color="crimson" if sig else "grey",
                   edgecolors="black", linewidths=0.5, zorder=3)

    xv_arr, yv_arr = x[valid].to_numpy(), y[valid].to_numpy()
    if xv_arr.size >= 2 and np.std(xv_arr) > 0:
        slope, intercept = np.polyfit(xv_arr, yv_arr, 1)
        x_line = np.linspace(xv_arr.min(), xv_arr.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="black", lw=1.5, ls="--", zorder=2,
                label="OLS fit")

    r = corr_df.loc[ma, mb]
    title = f"{ma}  vs  {mb}\nr={r:.2f}{' *' if sig else ''}"
    if p_analytic_df is not None:
        title += f"   ({_format_p(p_analytic_df.loc[ma, mb])})"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(ma, fontsize=9); ax.set_ylabel(mb, fontsize=9)
    ax.tick_params(labelsize=8)

    if point_groups is not None and point_colors is not None:
        group_to_color = {}
        for e, g in point_groups.items():
            if e in point_colors and g not in group_to_color:
                group_to_color[g] = point_colors[e]
        handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                               markeredgecolor="black", markersize=6, label=g)
                   for g, c in sorted(group_to_color.items())]
        ax.legend(handles=handles, fontsize=6, frameon=False, loc="best")

    fig.tight_layout()
    safe_ma = ma.replace("/", "_"); safe_mb = mb.replace("/", "_")
    fig.savefig(out_dir / f"{safe_ma}__vs__{safe_mb}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_measure_scatter_all_pairs(entity_df, measures, corr_df, pfdr_df, out_dir, alpha=0.05,
                                    p_analytic_df=None, point_colors=None, point_groups=None,
                                    subfolder="scatter_pairs"):
    """Save one single-pair regression figure per measure pair into
    out_dir/subfolder/, in addition to the compact grid figure."""
    pairs = [(a, b) for i, a in enumerate(measures) for b in measures[i + 1:]]
    if not pairs:
        return
    pair_dir = out_dir / subfolder
    pair_dir.mkdir(parents=True, exist_ok=True)
    for ma, mb in pairs:
        _fig_measure_scatter_single(entity_df, ma, mb, corr_df, pfdr_df, pair_dir, alpha=alpha,
                                     p_analytic_df=p_analytic_df, point_colors=point_colors,
                                     point_groups=point_groups)
    print(f"  Saved {len(pairs)} single-pair regression figures → {pair_dir}")


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
    corr_df, pval_perm_df, pfdr_df, n_used_df, p_analytic_df = compute_measure_correlations(
        cluster_measures_df, measures, n_perm=n_perm)

    cluster_measures_df.to_csv(out_dir / "cluster_level_measures.csv")
    corr_df.to_csv(out_dir / "cluster_measure_correlations.csv")
    pval_perm_df.to_csv(out_dir / "cluster_measure_correlations_pval.csv")
    pfdr_df.to_csv(out_dir / "cluster_measure_correlations_pfdr.csv")  # computed, unused for now
    n_used_df.to_csv(out_dir / "cluster_measure_correlations_n.csv")
    p_analytic_df.to_csv(out_dir / "cluster_measure_correlations_p_analytic.csv")

    _fig_measure_correlation_heatmap(corr_df, pval_perm_df, out_dir,
                                      title="Cluster-level measure correlations",
                                      p_analytic_df=p_analytic_df)
    _fig_measure_scatter_grid(cluster_measures_df, measures, corr_df, pval_perm_df, out_dir,
                               p_analytic_df=p_analytic_df)
    _fig_measure_scatter_all_pairs(cluster_measures_df, measures, corr_df, pval_perm_df, out_dir,
                                    p_analytic_df=p_analytic_df)

    print(f"  Cluster-level correlations ({len(cluster_measures_df)} clusters, "
          f"{len(measures)} measures) → {out_dir}")
    return cluster_measures_df, corr_df, pfdr_df


def run_area_level_correlations(neuron_df, cluster_labels, measure_cols, correlations_dir,
                                 min_neurons=10, n_perm=5000, specialization_baseline="uniform"):
    out_dir = correlations_dir / "by_area"
    out_dir.mkdir(parents=True, exist_ok=True)

    area_arr = neuron_df["area_acronym_custom"].to_numpy()
    spec_df = compute_area_specialization(cluster_labels, area_arr,
                                           min_neurons=min_neurons,
                                           baseline=specialization_baseline)
    _fig_area_specialization_bar(spec_df, out_dir)
    _fig_area_cluster_occupancy_examples(cluster_labels, area_arr, spec_df, out_dir)
    _fig_specialization_bias_check(spec_df, out_dir)

    area_measures_df = compute_entity_level_measures(
        neuron_df, group_col="area_acronym_custom", measure_cols=measure_cols, min_neurons=min_neurons)

    # NOTE: raw 'specialization' is confounded by N (see
    # area_specialization_bias_check.pdf) — 'specialization_null_z' is the
    # sample-size-corrected version and is what downstream correlations use.
    area_df = spec_df.join(area_measures_df.drop(columns=["n_neurons"], errors="ignore"), how="inner")

    # Allen custom-group colors per area, for the correlation scatter grid.
    area_colors, area_groups = _build_area_group_colors(area_df.index.tolist())
    pd.DataFrame({"area_acronym_custom": list(area_groups.keys()),
                  "area_group": list(area_groups.values())}).to_csv(
        out_dir / "area_group_mapping.csv", index=False)

    measures = [c for c in measure_cols if c in area_df.columns] + ["specialization_null_z" ,"specialization_null_corrected"]
    corr_df, pval_perm_df, pfdr_df, n_used_df, p_analytic_df = compute_measure_correlations(
        area_df, measures, n_perm=n_perm)

    area_df.to_csv(out_dir / "area_level_measures.csv")
    corr_df.to_csv(out_dir / "area_measure_correlations.csv")
    pval_perm_df.to_csv(out_dir / "area_measure_correlations_pval.csv")
    pfdr_df.to_csv(out_dir / "area_measure_correlations_pfdr.csv")  # computed, unused for now
    n_used_df.to_csv(out_dir / "area_measure_correlations_n.csv")
    p_analytic_df.to_csv(out_dir / "area_measure_correlations_p_analytic.csv")

    _fig_measure_correlation_heatmap(corr_df, pval_perm_df, out_dir,
                                      title="Area-level measure correlations (incl. specialization_null_z)",
                                      p_analytic_df=p_analytic_df)
    _fig_measure_scatter_grid(area_df, measures, corr_df, pval_perm_df, out_dir,
                               p_analytic_df=p_analytic_df,
                               point_colors=area_colors, point_groups=area_groups)
    _fig_measure_scatter_all_pairs(area_df, measures, corr_df, pval_perm_df, out_dir,
                                    p_analytic_df=p_analytic_df,
                                    point_colors=area_colors, point_groups=area_groups)

    print(f"  Area-level correlations ({len(area_df)} areas, "
          f"{len(measures)} measures, incl. specialization_null_z) → {out_dir}")
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


def _fig_cluster_similarity_matrices(rdm_dict, out_dir, fname="cluster_similarity_matrices.pdf"):
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
    ordering across all of them), Pearson-correlate their vectorized upper
    triangles, with permutation significance (BH-FDR across all pairs) and
    the analytic Pearson p-value reported alongside (descriptive only).
    """
    if rng is None:
        rng = np.random.default_rng()
    names = list(rdm_dict.keys())
    n = rdm_dict[names[0]].shape[0]
    iu = np.triu_indices(n, k=1)

    corr          = np.full((len(names), len(names)), np.nan)
    pval_perm_raw = np.full((len(names), len(names)), np.nan)
    pval_analytic = np.full((len(names), len(names)), np.nan)

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if j <= i:
                continue
            a, b = rdm_dict[ni][iu], rdm_dict[nj][iu]
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() < 6:
                continue
            obs_r, p_analytic = pearsonr(a[valid], b[valid])
            corr[i, j] = corr[j, i] = obs_r
            pval_analytic[i, j] = pval_analytic[j, i] = p_analytic

            null_r = np.empty(n_perm)
            b_full = rdm_dict[nj]
            for p in range(n_perm):
                perm = rng.permutation(n)
                shuffled = b_full[np.ix_(perm, perm)][iu]
                null_r[p], _ = pearsonr(a[valid], shuffled[valid])
            p_perm = (1 + np.sum(np.abs(null_r) >= np.abs(obs_r))) / (n_perm + 1)
            pval_perm_raw[i, j] = pval_perm_raw[j, i] = p_perm

    np.fill_diagonal(corr, 1.0); np.fill_diagonal(pval_perm_raw, 0.0); np.fill_diagonal(pval_analytic, 0.0)
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    flat_p = np.array([pval_perm_raw[i, j] for i, j in pairs])
    valid_p = ~np.isnan(flat_p)
    flat_fdr = np.full(len(flat_p), np.nan)
    if valid_p.any():
        flat_fdr[valid_p], _ = _bh_correction(flat_p[valid_p], alpha=0.05)
    pfdr = np.full_like(corr, np.nan)
    for (i, j), p in zip(pairs, flat_fdr):
        pfdr[i, j] = pfdr[j, i] = p
    np.fill_diagonal(pfdr, 0.0)

    return (pd.DataFrame(corr, index=names, columns=names),
            pd.DataFrame(pval_perm_raw, index=names, columns=names),
            pd.DataFrame(pfdr, index=names, columns=names),
            pd.DataFrame(pval_analytic, index=names, columns=names))


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

    rsa_corr, rsa_pval_perm, rsa_pfdr, rsa_p_analytic = compare_rdms(rdms, n_perm=n_perm)

    rsa_corr.to_csv(out_dir / "rsa_correlation_matrix.csv")
    rsa_pval_perm.to_csv(out_dir / "rsa_correlation_matrix_pval.csv")
    rsa_pfdr.to_csv(out_dir / "rsa_correlation_matrix_pfdr.csv")  # computed, unused for now
    rsa_p_analytic.to_csv(out_dir / "rsa_correlation_matrix_p_analytic.csv")
    _fig_measure_correlation_heatmap(rsa_corr, rsa_pval_perm, out_dir, fname="rsa_correlation_heatmap.pdf",
                                      title="RSA: second-order correlation across metric RDMs",
                                      p_analytic_df=rsa_p_analytic)

    print(f"  RSA across {len(rdms)} RDMs, {len(cluster_measures_df)} clusters → {out_dir}")
    return rsa_corr, rsa_pfdr, rdms


# ─────────────────────────────────────────────────────────────────────────
# 10. top-level orchestrator
# ─────────────────────────────────────────────────────────────────────────
def run_all(rastermap_out_folder, unit_table, anatomical_cols,
            analysis_type_col="analysis_type", sig_col="significant",
            selectivity_col="selectivity_index",
            selectivity_analysis_types=DEFAULT_SELECTIVITY_ANALYSIS_TYPES,
            region_filters=DEFAULT_ANATOMICAL_REGION_FILTERS,
            min_neurons=100,
            n_perm=5000, specialization_baseline="uniform",
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
        selectivity_analysis_types=selectivity_analysis_types, region_filters=region_filters)

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