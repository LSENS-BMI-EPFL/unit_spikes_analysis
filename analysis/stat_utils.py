import numpy as np
import pandas as pd
from roc_analysis_utils import compute_prop_significant


def compute_mean_selectivity(
    roc_df: pd.DataFrame,
    area_col: str,
) -> pd.DataFrame:
    """
    Compute mean absolute selectivity per mouse × area × analysis_type.

    Parameters
    ----------
    roc_df   : must contain columns mouse_id, reward_group, analysis_type,
               <area_col>, selectivity
    area_col : e.g. 'area_group' or 'area'

    Returns
    -------
    DataFrame: mouse_id, reward_group, analysis_type, <area_col>, mean_abs_selectivity
    """
    return (
        roc_df
        .groupby(["mouse_id", "reward_group", "analysis_type", area_col])
        ["selectivity"]
        .apply(lambda x: np.mean(np.abs(x)))
        .reset_index()
        .rename(columns={"selectivity": "mean_abs_selectivity"})
    )


def compute_delta_proportion_sel(
    roc_df: pd.DataFrame,
    area_col: str,
    pre_label: str,
    post_label: str,
    value_col: str = "mean_abs_selectivity",   # updated default
) -> pd.DataFrame:
    """
    Compute post - pre delta in mean absolute selectivity per mouse × area.
    """
    metric_df = compute_mean_selectivity(roc_df, area_col)

    idx = ["mouse_id", "reward_group", area_col]

    pre  = (metric_df[metric_df["analysis_type"] == pre_label]
            .set_index(idx)[value_col].rename("pre"))
    post = (metric_df[metric_df["analysis_type"] == post_label]
            .set_index(idx)[value_col].rename("post"))

    delta = (
        pd.concat([pre, post], axis=1)
        .dropna(subset=["pre", "post"])
        .assign(delta=lambda x: x["post"] - x["pre"])
        .drop(columns=["pre", "post"])
        .reset_index()
    )

    return delta
# ── test statistic ────────────────────────────────────────────────────────────

def _group_diff_statistic(
    delta_df: pd.DataFrame,
    area_col: str,
    group_col: str = "reward_group",
) -> float:
    """
    For each area, compute the difference in mean Δ between groups
    using only mice recorded in that area. Average the absolute
    differences across areas.

    This naturally handles missing areas: each area contributes its
    own group contrast from whichever mice have it.
    """
    diffs = []
    for area, grp in delta_df.groupby(area_col):
        group_means = grp.groupby(group_col)["delta"].mean()
        if len(group_means) < 2:
            continue  # area only present in one group — skip
        diffs.append(group_means.iloc[0] - group_means.iloc[1])

    if len(diffs) == 0:
        return np.nan

    # mean squared difference across areas (sign-insensitive, more powerful)
    return np.mean(np.array(diffs) ** 2)


# ── permutation test ──────────────────────────────────────────────────────────

def _hierarchical_permutation_test(
    delta_df: pd.DataFrame,
    area_col: str,
    group_col: str = "reward_group",
    n_perm: int    = 9999,
    seed: int      = 42,
) -> dict:
    """
    Permutation test on the per-area group difference statistic.

    Permutation is at the mouse level: reward_group labels are shuffled
    across mice, preserving each mouse's full area vector. This respects
    the hierarchical structure (areas nested within mice) and naturally
    handles each mouse having a unique set of recorded areas.

    Parameters
    ----------
    delta_df : long-format DataFrame with columns
               mouse_id, reward_group, <area_col>, delta
    n_perm   : number of permutations

    Returns
    -------
    dict with observed statistic, p-value, and null distribution
    """
    rng = np.random.default_rng(seed)

    # mouse-level group label lookup (one row per mouse)
    mouse_groups = (
        delta_df[["mouse_id", group_col]]
        .drop_duplicates()
        .set_index("mouse_id")[group_col]
    )
    mouse_ids = mouse_groups.index.to_numpy()
    labels    = mouse_groups.to_numpy()

    n_mice  = len(mouse_ids)
    groups  = np.unique(labels)
    n_per_group = {g: (labels == g).sum() for g in groups}

    print(f"  mice: {n_mice}  |  "
          + "  ".join(f"{g}: n={n}" for g, n in n_per_group.items()))
    print(f"  areas: {delta_df[area_col].nunique()}")

    # observed statistic
    stat_obs = _group_diff_statistic(delta_df, area_col, group_col)

    # permutation null: shuffle group labels across mice
    stat_null = np.empty(n_perm)
    for i in range(n_perm):
        perm_labels = rng.permutation(labels)
        perm_map    = dict(zip(mouse_ids, perm_labels))

        delta_perm = delta_df.copy()
        delta_perm[group_col] = delta_perm["mouse_id"].map(perm_map)

        stat_null[i] = _group_diff_statistic(delta_perm, area_col, group_col)

    p_value = (np.sum(stat_null >= stat_obs) + 1) / (n_perm + 1)

    return {
        "statistic": stat_obs,
        "p_value":   p_value,
        "n_mice":    n_mice,
        "n_perm":    n_perm,
        "null_dist": stat_null,   # keep for plotting
    }


# ── delta computation (unchanged) ─────────────────────────────────────────────

def compute_delta_proportion(
    roc_df: pd.DataFrame,
    area_col: str,
    pre_label: str,
    post_label: str,
    value_col: str = "proportion_all",
) -> pd.DataFrame:
    prop_df = compute_prop_significant(roc_df, area_col, per_subject=True)

    prop_df = (
        prop_df
        .drop_duplicates(subset=["mouse_id", "reward_group", "analysis_type", area_col])
        [["mouse_id", "reward_group", "analysis_type", area_col, value_col]]
    )

    idx = ["mouse_id", "reward_group", area_col]

    pre  = (prop_df[prop_df["analysis_type"] == pre_label]
            .set_index(idx)[value_col].rename("pre"))
    post = (prop_df[prop_df["analysis_type"] == post_label]
            .set_index(idx)[value_col].rename("post"))

    return (
        pd.concat([pre, post], axis=1)
        .dropna(subset=["pre", "post"])
        .assign(delta=lambda x: x["post"] - x["pre"])
        .drop(columns=["pre", "post"])
        .reset_index()
    )


# ── full pipeline ─────────────────────────────────────────────────────────────

def run_proportion_permanova(
    roc_df: pd.DataFrame,
    area_col: str,
    value_col: str  = "proportion_all",
    n_permutations: int = 9999,
) -> tuple[pd.DataFrame, dict]:
    """
    Hierarchical permutation test on learning-induced selectivity change.

    Condition A (whisker):  Δ = whisker_passive_post  − whisker_passive_pre
    Condition B (auditory): Δ = auditory_passive_post − auditory_passive_pre

    Tests
    -----
    1. Permutation test on whisker Δ   — expect group effect
    2. Permutation test on auditory Δ  — expect null
    3. Permutation test on whisker Δ − auditory Δ per mouse  — interaction
    """
    conditions = {
        "whisker":  ("whisker_passive_pre",  "whisker_passive_post"),
        "auditory": ("auditory_passive_pre", "auditory_passive_post"),
    }

    deltas = {}
    for name, (pre, post) in conditions.items():
        print(f"\nComputing Δ for '{name}'  ({pre} → {post})")
        value_col = 'mean_abs_selectivity'
        deltas[name] = compute_delta_proportion_sel(
            roc_df, area_col, pre, post, value_col
        )

    # ── per-condition ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Hierarchical permutation test per condition  (factor = reward_group)")
    print(f"  metric : {value_col}")
    print("=" * 65)

    rows = []
    for name, delta_df in deltas.items():
        print(f"\n  condition = {name}")
        res = _hierarchical_permutation_test(
            delta_df, area_col, n_perm=n_permutations
        )
        rows.append({"condition": name, **{k: v for k, v in res.items()
                                           if k != "null_dist"}})
        print(f"    statistic : {res['statistic']:.4f}")
        print(f"    p-value   : {res['p_value']:.4f}")

    per_cond = pd.DataFrame(rows).set_index("condition")

    # ── interaction ───────────────────────────────────────────────────────
    # Build contrast per mouse × area: whisker Δ − auditory Δ
    # Only mice and areas present in both conditions contribute
    print("\n" + "=" * 65)
    print("Interaction test  (reward_group × condition)")
    print("  contrast : whisker Δ − auditory Δ  per mouse × area")
    print("=" * 65)

    merged = deltas["whisker"].merge(
        deltas["auditory"],
        on=["mouse_id", "reward_group", area_col],
        suffixes=("_w", "_a"),
    )
    merged["delta"] = merged["delta_w"] - merged["delta_a"]
    contrast_df = merged[["mouse_id", "reward_group", area_col, "delta"]]

    print(f"\n  condition = whisker − auditory")
    inter_res = _hierarchical_permutation_test(
        contrast_df, area_col, n_perm=n_permutations
    )
    inter = {
        "contrast": "whisker − auditory",
        "n_areas":  contrast_df[area_col].nunique(),
        **{k: v for k, v in inter_res.items() if k != "null_dist"},
    }
    print(f"    statistic : {inter['statistic']:.4f}")
    print(f"    p-value   : {inter['p_value']:.4f}")
    print(f"    n_areas   : {inter['n_areas']}")

    return per_cond, inter