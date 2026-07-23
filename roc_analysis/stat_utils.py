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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats


def plot_permanova_results(
    roc_df: pd.DataFrame,
    area_col: str,
    per_cond: pd.DataFrame,
    interaction: dict,
    value_col: str = "mean_abs_selectivity",
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    Four-panel summary figure for the hierarchical permutation results.

    Panel A : per-area Δ per group, whisker condition (strip + mean ± SEM)
    Panel B : per-area Δ per group, auditory condition
    Panel C : per-area contrast (whisker Δ − auditory Δ) per group
    Panel D : null distributions with observed statistic for all three tests
    """
    conditions = {
        "whisker":  ("whisker_passive_pre",  "whisker_passive_post"),
        "auditory": ("auditory_passive_pre", "auditory_passive_post"),
    }

    # ── compute deltas ────────────────────────────────────────────────────
    deltas = {}
    for name, (pre, post) in conditions.items():
        value_col='mean_abs_selectivity'
        deltas[name] = compute_delta_proportion_sel(roc_df, area_col, pre, post, value_col)

    # per-area group means for A and B
    def area_group_means(delta_df):
        return (
            delta_df.groupby([area_col, "reward_group"])["delta"]
            .agg(mean="mean", sem=lambda x: x.sem())
            .reset_index()
        )

    means = {name: area_group_means(d) for name, d in deltas.items()}

    # contrast per mouse × area
    merged = deltas["whisker"].merge(
        deltas["auditory"],
        on=["mouse_id", "reward_group", area_col],
        suffixes=("_w", "_a"),
    )
    merged["delta"] = merged["delta_w"] - merged["delta_a"]
    contrast_means  = area_group_means(merged.rename(columns={"delta": "delta"}))

    # ── layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_w    = fig.add_subplot(gs[0, 0])   # whisker per-area
    ax_a    = fig.add_subplot(gs[0, 1])   # auditory per-area
    ax_int  = fig.add_subplot(gs[0, 2])   # contrast per-area
    ax_null = fig.add_subplot(gs[1, :])   # null distributions

    colors  = {"R+": "forestgreen", "R-": "crimson"}
    jitter  = {"R+": -0.15, "R+": 0.15}
    x_pos   = {"R+": 0, "R-": 1}

    # ── helper: per-area strip plot ───────────────────────────────────────
    def plot_per_area(ax, means_df, title, ylabel, p_value):
        areas = sorted(means_df[area_col].unique())
        n_areas = len(areas)
        area_idx = {a: i for i, a in enumerate(areas)}

        for _, row in means_df.iterrows():
            xi = area_idx[row[area_col]]
            g  = row["reward_group"]
            jit = -0.15 if g == "R+" else 0.15
            ax.errorbar(
                xi + jit, row["mean"], yerr=row["sem"],
                fmt="o", color=colors[g], markersize=4,
                capsize=2, linewidth=1, elinewidth=0.8,
                label=g if xi == 0 else "_nolegend_",
            )

        # connect group means per area with a thin line
        for area in areas:
            xi   = area_idx[area]
            sub  = means_df[means_df[area_col] == area].set_index("reward_group")
            if "R+" in sub.index and "R-" in sub.index:
                ax.plot(
                    [xi - 0.15, xi + 0.15],
                    [sub.loc["R+", "mean"], sub.loc["R-", "mean"]],
                    color="gray", linewidth=0.6, alpha=0.5, zorder=0,
                )

        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
        ax.set_xticks(range(n_areas))
        ax.set_xticklabels(areas, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(
            f"{title}\np = {p_value:.3f}",
            fontsize=9,
        )
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    # ── panels A, B, C ───────────────────────────────────────────────────
    plot_per_area(
        ax_w, means["whisker"],
        title  = "Whisker condition",
        ylabel = "Δ mean |selectivity|\n(post − pre)",
        p_value= per_cond.loc["whisker", "p_value"],
    )
    plot_per_area(
        ax_a, means["auditory"],
        title  = "Auditory condition",
        ylabel = "Δ mean |selectivity|\n(post − pre)",
        p_value= per_cond.loc["auditory", "p_value"],
    )
    plot_per_area(
        ax_int, contrast_means,
        title  = "Interaction\n(whisker − auditory)",
        ylabel = "Δ whisker − Δ auditory",
        p_value= interaction["p_value"],
    )

    # ── panel D: null distributions ───────────────────────────────────────
    # re-run tests to get null distributions (or pass them in if cached)
    null_colors = {
        "whisker":   "#E8593C",
        "auditory":  "#3B8BD4",
        "interaction": "#888780",
    }
    test_results = {
        "whisker":    _hierarchical_permutation_test(deltas["whisker"],  area_col),
        "auditory":   _hierarchical_permutation_test(deltas["auditory"], area_col),
        "interaction": _hierarchical_permutation_test(
            merged[[area_col, "mouse_id", "reward_group", "delta"]], area_col
        ),
    }
    p_vals = {
        "whisker":     per_cond.loc["whisker",  "p_value"],
        "auditory":    per_cond.loc["auditory", "p_value"],
        "interaction": interaction["p_value"],
    }

    offset = {"whisker": -0.25, "auditory": 0, "interaction": 0.25}
    for name, res in test_results.items():
        null  = res["null_dist"]
        obs   = res["statistic"]
        color = null_colors[name]
        kde_x = np.linspace(null.min(), max(null.max(), obs * 1.1), 300)
        kde   = stats.gaussian_kde(null)
        ax_null.plot(kde_x, kde(kde_x), color=color, linewidth=1.5,
                     label=f"{name}  (p={p_vals[name]:.3f})")
        ax_null.axvline(obs, color=color, linewidth=1.5,
                        linestyle="--", alpha=0.9)
        ax_null.fill_between(
            kde_x, kde(kde_x),
            where=kde_x >= obs,
            color=color, alpha=0.15,
        )

    ax_null.set_xlabel("Test statistic  (mean squared group difference)", fontsize=9)
    ax_null.set_ylabel("Density", fontsize=9)
    ax_null.set_title("Permutation null distributions\n(dashed = observed statistic)", fontsize=9)
    ax_null.legend(fontsize=8, frameon=False)
    ax_null.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Learning-induced selectivity change by reward group",
                 fontsize=11, y=1.01)

    return fig