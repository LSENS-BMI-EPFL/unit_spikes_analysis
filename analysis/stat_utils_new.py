import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from roc_analysis_utils import compute_prop_significant


# ── metric computation ────────────────────────────────────────────────────────

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
) -> pd.DataFrame:
    """
    Compute post - pre delta in mean absolute selectivity per mouse × area.

    Returns
    -------
    DataFrame: mouse_id, reward_group, <area_col>, delta
    """
    metric_df = compute_mean_selectivity(roc_df, area_col)
    value_col = "mean_abs_selectivity"

    idx = ["mouse_id", "reward_group", area_col]

    pre  = (metric_df[metric_df["analysis_type"] == pre_label]
            .set_index(idx)[value_col].rename("pre"))
    post = (metric_df[metric_df["analysis_type"] == post_label]
            .set_index(idx)[value_col].rename("post"))

    return (
        pd.concat([pre, post], axis=1)
        .dropna(subset=["pre", "post"])
        .assign(delta=lambda x: x["post"] - x["pre"])
        .drop(columns=["pre", "post"])
        .reset_index()
    )


# ── weights ───────────────────────────────────────────────────────────────────

def _compute_area_weights(
    roc_df: pd.DataFrame,
    area_col: str,
    pre_label: str,
    post_label: str,
    group_col: str = "reward_group",
) -> pd.Series:
    """
    Compute per-area weights as the harmonic mean of per-group mean neuron
    counts per mouse, averaged over pre and post analysis_types.

    Weight reflects how reliably the per-mouse mean selectivity is estimated
    in each area: areas with many neurons per mouse in both groups get high
    weight; areas where one group has very few neurons per mouse get low weight
    (harmonic mean is sensitive to imbalance).

    Returns
    -------
    pd.Series indexed by area, normalised so weights sum to 1.
    """
    subset = roc_df[roc_df["analysis_type"].isin([pre_label, post_label])]

    # neuron count per mouse × area × analysis_type
    counts = (
        subset
        .groupby(["mouse_id", group_col, area_col, "analysis_type"])
        .size()
        .reset_index(name="n_neurons")
    )

    # average neuron count per mouse over pre and post, then average over mice
    # within each group × area to get mean neurons per mouse per group per area
    mean_per_mouse = (
        counts
        .groupby(["mouse_id", group_col, area_col])["n_neurons"]
        .mean()                              # average over pre/post per mouse
        .reset_index()
        .groupby([group_col, area_col])["n_neurons"]
        .mean()                              # average over mice per group
        .reset_index()
    )

    def harmonic_mean(x):
        if (x == 0).any() or len(x) < 2:
            return 0.0
        return len(x) / np.sum(1.0 / x)

    weights = (
        mean_per_mouse
        .groupby(area_col)["n_neurons"]
        .apply(harmonic_mean)
        .rename("weight")
    )

    total = weights.sum()
    if total > 0:
        weights = weights / total

    return weights


# ── test statistic ────────────────────────────────────────────────────────────

def _group_diff_statistic(
    delta_df: pd.DataFrame,
    area_col: str,
    group_col: str = "reward_group",
    weights: pd.Series = None,
) -> float:
    """
    Mean (weighted) squared group difference in Δ across areas.

    For each area, compute (mean_delta_R+ - mean_delta_R-)^2 using only
    mice recorded in that area. Average across areas, optionally weighted
    by per-area neuron count reliability.

    Parameters
    ----------
    weights : pd.Series indexed by area (from _compute_area_weights).
              If None, all areas are weighted equally (unweighted test).
    """
    diffs  = []
    w_used = []

    for area, grp in delta_df.groupby(area_col):
        group_means = grp.groupby(group_col)["delta"].mean()
        if len(group_means) < 2:
            continue   # area present in only one group — skip

        if weights is not None:
            if area not in weights.index or weights[area] == 0:
                continue
            w_used.append(weights[area])

        diffs.append(group_means.iloc[0] - group_means.iloc[1])

    if len(diffs) == 0:
        return np.nan

    diffs = np.array(diffs)

    if weights is not None:
        w = np.array(w_used)
        w = w / w.sum()          # re-normalise over contributing areas
        return np.sum(w * diffs ** 2)
    else:
        return np.mean(diffs ** 2)


# ── permutation test ──────────────────────────────────────────────────────────

def _hierarchical_permutation_test(
    delta_df: pd.DataFrame,
    area_col: str,
    group_col: str = "reward_group",
    n_perm: int    = 9999,
    seed: int      = 42,
    weights: pd.Series = None,
) -> dict:
    """
    Hierarchical permutation test on the per-area group difference statistic.

    Group labels (reward_group) are permuted at the mouse level, preserving
    each mouse's full area vector. This respects the nested structure and
    handles mice with unique sets of recorded areas.

    Parameters
    ----------
    delta_df : long-format DataFrame: mouse_id, reward_group, <area_col>, delta
    weights  : optional per-area weights from _compute_area_weights.
               If None, all areas contribute equally (unweighted).

    Returns
    -------
    dict: statistic, p_value, n_mice, n_perm, null_dist, weighted
    """
    rng = np.random.default_rng(seed)

    mouse_groups = (
        delta_df[["mouse_id", group_col]]
        .drop_duplicates()
        .set_index("mouse_id")[group_col]
    )
    mouse_ids = mouse_groups.index.to_numpy()
    labels    = mouse_groups.to_numpy()
    n_mice    = len(mouse_ids)
    groups    = np.unique(labels)
    n_per_group = {g: (labels == g).sum() for g in groups}

    weighted_str = "weighted" if weights is not None else "unweighted"
    print(f"  [{weighted_str}] mice: {n_mice}  |  "
          + "  ".join(f"{g}: n={n}" for g, n in n_per_group.items()))
    print(f"  areas: {delta_df[area_col].nunique()}")

    stat_obs = _group_diff_statistic(delta_df, area_col, group_col, weights)

    stat_null = np.empty(n_perm)
    for i in range(n_perm):
        perm_labels           = rng.permutation(labels)
        perm_map              = dict(zip(mouse_ids, perm_labels))
        delta_perm            = delta_df.copy()
        delta_perm[group_col] = delta_perm["mouse_id"].map(perm_map)
        stat_null[i]          = _group_diff_statistic(
            delta_perm, area_col, group_col, weights
        )

    p_value = (np.sum(stat_null >= stat_obs) + 1) / (n_perm + 1)

    return {
        "statistic": stat_obs,
        "p_value":   p_value,
        "n_mice":    n_mice,
        "n_perm":    n_perm,
        "null_dist": stat_null,
        "weighted":  weights is not None,
    }


# ── full pipeline ─────────────────────────────────────────────────────────────

def run_proportion_permanova(
    roc_df: pd.DataFrame,
    area_col: str,
    n_permutations: int = 1000,
    weighted: bool      = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Hierarchical permutation test on learning-induced selectivity change.

    Condition A (whisker):  Δ = whisker_passive_post  − whisker_passive_pre
    Condition B (auditory): Δ = auditory_passive_post − auditory_passive_pre

    Tests
    -----
    1. Permutation test on whisker Δ   — expect group effect
    2. Permutation test on auditory Δ  — expect null
    3. Permutation test on whisker Δ − auditory Δ  — interaction

    Parameters
    ----------
    weighted : if True, weight each area by the harmonic mean of per-group
               mean neuron counts per mouse. Areas with more reliable
               selectivity estimates (more neurons) contribute more to the
               test statistic.
    """
    conditions = {
        "whisker":  ("whisker_passive_pre",  "whisker_passive_post"),
        "auditory": ("auditory_passive_pre", "auditory_passive_post"),
    }

    # ── deltas and (optionally) weights ──────────────────────────────────
    deltas  = {}
    weights = {}
    for name, (pre, post) in conditions.items():
        print(f"\nComputing Δ for '{name}'  ({pre} → {post})")
        deltas[name] = compute_delta_proportion_sel(roc_df, area_col, pre, post)
        if weighted:
            weights[name] = _compute_area_weights(roc_df, area_col, pre, post)
            print(f"  weights computed for {len(weights[name])} areas")
        else:
            weights[name] = None

    # ── per-condition ─────────────────────────────────────────────────────
    weighted_str = "weighted " if weighted else ""
    print("\n" + "=" * 65)
    print(f"Hierarchical {weighted_str}permutation test  (factor = reward_group)")
    print("=" * 65)

    rows = []
    for name, delta_df in deltas.items():
        print(f"\n  condition = {name}")
        res = _hierarchical_permutation_test(
            delta_df, area_col,
            n_perm  = n_permutations,
            weights = weights[name],
        )
        rows.append({"condition": name,
                     **{k: v for k, v in res.items() if k != "null_dist"}})
        print(f"    statistic : {res['statistic']:.4f}")
        print(f"    p-value   : {res['p_value']:.4f}")

    per_cond = pd.DataFrame(rows).set_index("condition")

    # ── interaction ───────────────────────────────────────────────────────
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

    # interaction weights: harmonic mean of per-condition weights per area
    if weighted:
        w_w = weights["whisker"]
        w_a = weights["auditory"]
        common = w_w.index.intersection(w_a.index)
        w_interaction = (
            2 * w_w[common] * w_a[common] / (w_w[common] + w_a[common])
        )
        w_interaction = w_interaction / w_interaction.sum()
    else:
        w_interaction = None

    print(f"\n  condition = whisker − auditory")
    inter_res = _hierarchical_permutation_test(
        contrast_df, area_col,
        n_perm  = n_permutations,
        weights = w_interaction,
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


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_permanova_results(
    roc_df: pd.DataFrame,
    area_col: str,
    per_cond: pd.DataFrame,
    interaction: dict,
    figsize: tuple = (14, 10),
    weighted: bool = False,
) -> plt.Figure:
    """
    Four-panel summary figure for the hierarchical permutation results.

    Panel A : per-area Δ per group, whisker condition (mean ± SEM)
    Panel B : per-area Δ per group, auditory condition
    Panel C : per-area contrast (whisker Δ − auditory Δ) per group
    Panel D : null distributions with observed statistic for all three tests
    """
    conditions = {
        "whisker":  ("whisker_passive_pre",  "whisker_passive_post"),
        "auditory": ("auditory_passive_pre", "auditory_passive_post"),
    }

    # ── compute deltas and weights ────────────────────────────────────────
    deltas  = {}
    weights = {}
    for name, (pre, post) in conditions.items():
        deltas[name]  = compute_delta_proportion_sel(roc_df, area_col, pre, post)
        weights[name] = (_compute_area_weights(roc_df, area_col, pre, post)
                         if weighted else None)

    def area_group_means(delta_df):
        return (
            delta_df.groupby([area_col, "reward_group"])["delta"]
            .agg(mean="mean", sem=lambda x: x.sem())
            .reset_index()
        )

    means = {name: area_group_means(d) for name, d in deltas.items()}

    merged = deltas["whisker"].merge(
        deltas["auditory"],
        on=["mouse_id", "reward_group", area_col],
        suffixes=("_w", "_a"),
    )
    merged["delta"] = merged["delta_w"] - merged["delta_a"]
    contrast_df     = merged[["mouse_id", "reward_group", area_col, "delta"]]
    contrast_means  = area_group_means(contrast_df)

    if weighted:
        w_w = weights["whisker"]
        w_a = weights["auditory"]
        common = w_w.index.intersection(w_a.index)
        w_interaction = (
            2 * w_w[common] * w_a[common] / (w_w[common] + w_a[common])
        )
        w_interaction = w_interaction / w_interaction.sum()
    else:
        w_interaction = None

    # ── layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_w    = fig.add_subplot(gs[0, 0])
    ax_a    = fig.add_subplot(gs[0, 1])
    ax_int  = fig.add_subplot(gs[0, 2])
    ax_null = fig.add_subplot(gs[1, :])

    colors = {"R+": "#E8593C", "R-": "#3B8BD4"}

    # ── helper: per-area strip plot ───────────────────────────────────────
    def plot_per_area(ax, means_df, title, ylabel, p_value):
        areas    = sorted(means_df[area_col].unique())
        area_idx = {a: i for i, a in enumerate(areas)}

        for _, row in means_df.iterrows():
            xi  = area_idx[row[area_col]]
            g   = row["reward_group"]
            jit = -0.15 if g == "R+" else 0.15
            ax.errorbar(
                xi + jit, row["mean"], yerr=row["sem"],
                fmt="o", color=colors[g], markersize=5,
                capsize=2, linewidth=1, elinewidth=0.8,
                label=g if xi == 0 else "_nolegend_",
            )

        for area in areas:
            xi  = area_idx[area]
            sub = means_df[means_df[area_col] == area].set_index("reward_group")
            if "R+" in sub.index and "R-" in sub.index:
                ax.plot(
                    [xi - 0.15, xi + 0.15],
                    [sub.loc["R+", "mean"], sub.loc["R-", "mean"]],
                    color="gray", linewidth=0.6, alpha=0.5, zorder=0,
                )

        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
        ax.set_xticks(range(len(areas)))
        ax.set_xticklabels(areas, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{title}\np = {p_value:.3f}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    plot_per_area(ax_w,   means["whisker"],  "Whisker condition",
                  "Δ mean |selectivity|\n(post − pre)",
                  per_cond.loc["whisker",  "p_value"])
    plot_per_area(ax_a,   means["auditory"], "Auditory condition",
                  "Δ mean |selectivity|\n(post − pre)",
                  per_cond.loc["auditory", "p_value"])
    plot_per_area(ax_int, contrast_means,    "Interaction\n(whisker − auditory)",
                  "Δ whisker − Δ auditory",
                  interaction["p_value"])

    # ── panel D: null distributions ───────────────────────────────────────
    null_colors = {
        "whisker":     "#E8593C",
        "auditory":    "#3B8BD4",
        "interaction": "#888780",
    }
    test_results = {
        "whisker":  _hierarchical_permutation_test(
            deltas["whisker"],  area_col, weights=weights["whisker"]),
        "auditory": _hierarchical_permutation_test(
            deltas["auditory"], area_col, weights=weights["auditory"]),
        "interaction": _hierarchical_permutation_test(
            contrast_df, area_col, weights=w_interaction),
    }
    p_vals = {
        "whisker":     per_cond.loc["whisker",  "p_value"],
        "auditory":    per_cond.loc["auditory", "p_value"],
        "interaction": interaction["p_value"],
    }

    for name, res in test_results.items():
        null  = res["null_dist"]
        obs   = res["statistic"]
        color = null_colors[name]
        kde_x = np.linspace(null.min(), max(null.max(), obs * 1.1), 300)
        kde   = stats.gaussian_kde(null)
        ax_null.plot(kde_x, kde(kde_x), color=color, linewidth=1.5,
                     label=f"{name}  (p={p_vals[name]:.3f})")
        ax_null.axvline(obs, color=color, linewidth=1.5, linestyle="--", alpha=0.9)
        ax_null.fill_between(kde_x, kde(kde_x),
                             where=kde_x >= obs, color=color, alpha=0.15)

    weighted_str = " (weighted)" if weighted else ""
    ax_null.set_xlabel("Test statistic  (mean squared group difference)", fontsize=9)
    ax_null.set_ylabel("Density", fontsize=9)
    ax_null.set_title(
        f"Permutation null distributions{weighted_str}\n(dashed = observed statistic)",
        fontsize=9, fontweight="bold"
    )
    ax_null.legend(fontsize=8, frameon=False)
    ax_null.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Learning-induced selectivity change by reward group{weighted_str}",
        fontsize=11, fontweight="bold", y=1.01
    )

    return fig
