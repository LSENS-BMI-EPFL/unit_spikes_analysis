"""
Dyad CCG connectivity: cross-session summary figures + statistics.

Loads the already-computed per-session score-change results (the
*_score_changes.csv files written by dyad_pipeline.py, or the combined CSV
if present) -- never touches raw CCG/.npy data, so this is fast and safe to
re-run as often as you like.

SIMPLIFIED SCORE-CHANGE DESIGN (see CONDITIONS below):
Every score-change figure is one of three types, each a 2x2 grid over the
four (sign, comparison) conditions -- E/inflection, E/passive, I/inflection,
I/passive (CONDITIONS, fixed order/layout: row=sign, col=comparison) -- and
each produced 3x: pooled (all areas), within-area only, across-area only.
  1. plot_delta_histogram   -- raw pair-level score_diff histogram, R+ vs R-
  2. plot_delta_boxplot     -- session-level mean score_diff boxplot, R+ vs R-
  3. plot_delta_by_area     -- per-area mean+/-SEM bars, R+ vs R-, with
                                omnibus PERMANOVA + conditional per-area
                                Mann-Whitney post-hoc (only run if the
                                omnibus is significant)
E and I are NEVER combined or tested against each other -- different
underlying models (different SCORE_TAG versions), not comparable on the
same scale.

STATISTICAL APPROACH -- read before interpreting p-values:
- Raw pair-level rows are heavily pseudo-replicated: thousands of pairs from
  the same session are not independent observations. Every statistical test
  is run on SESSION-LEVEL MEANS (one value per session, per area, per
  group), never on raw per-pair rows -- even in plot_delta_histogram, whose
  histogram itself DOES show raw pair-level values (that's the plot), but
  whose p-value annotation is computed on session-level aggregates.
- "PERMANOVA" here means a permutation-based ANOVA (pseudo-F computed via
  permutation of group labels). For a single continuous response this is
  the exact univariate case of PERMANOVA (Euclidean-distance pseudo-F on
  one variable == classic ANOVA F-statistic) -- verified against scipy's
  f_oneway during development.
- Post-hoc (plot_delta_by_area only): per-area Mann-Whitney U, BH-FDR
  corrected across areas, run ONLY IF the omnibus PERMANOVA for that panel
  is significant (alpha=ALPHA) -- avoids fishing for area-level differences
  without an overall effect to explain. Areas with fewer than
  MIN_N_PER_GROUP sessions in either group are excluded and flagged.

SCORE-DIFF VALIDITY FILTER:
- score_pre/score_post can individually be the -7788 sentinel (unscoreable
  in that specific epoch) even when the pair passed the WHOLE-SESSION
  threshold used to mark it "connected". Every score-change figure filters
  to rows where BOTH score_pre > CONNECTED_THRESH AND
  score_post > CONNECTED_THRESH first (see _valid_score_diff). Connectivity
  /composition figures (which count pairs, not score_diff magnitude) are
  NOT filtered this way -- a pair's connected status only ever depends on
  the whole-session score.

WITHIN vs ACROSS AREA:
- Every score-change figure is produced 3x via _filter_area_relation:
  pooled (all pairs), within-area only (area_source == area_target), and
  across-area only. Filenames get a _within / _across suffix (none for
  pooled).

TRUE CONNECTIVITY RATES vs COMPOSITION -- two different denominators:
- plot_connectivity_by_reward_group uses combined_connectivity_summary.csv
  (written by dyad_pipeline.py's session_connectivity_summary()), which has
  REAL total-candidate-pair denominators -- these are true connectivity
  RATES, session-level, not broken down by specific area.
- plot_connectivity_by_area_and_reward_group breaks the SAME three
  categories (global/within/across) down by specific area, but the
  pipeline only saves within/across totals aggregated over ALL areas, not
  per individual area -- so no true per-area denominator exists. This
  figure instead shows COMPOSITION: of the connected pairs, what % falls
  in each area. Proportions of the connected set, not rates -- don't
  conflate the two.
"""
import os
import pickle
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

sns.set_theme()  # seaborn's default aesthetic (darkgrid style, deep palette), applied globally

# ============================== CONFIG ======================================
OUTPUT_ROOT = Path(r"M:\analysis\Axel_Bisi\combined_results")
COMBINED_CSV = OUTPUT_ROOT / "combined_score_changes.csv"
COMBINED_CONNECTIVITY_CSV = OUTPUT_ROOT / "combined_connectivity_summary.csv"
SUMMARY_DIR = OUTPUT_ROOT / "dyad"
# Raw per-session data root -- ONLY pairs_light.pkl is read from here (small
# per-pair metadata, not the CCG arrays). Everything else in this script
# stays CSV-only by design; this is a deliberate, narrow exception for the
# firing-rate confound check.
DATA_ROOT = Path(r"M:\share_external\Dyad_collaboration\processed_data")
# Another deliberate, narrow exception: small Excel metadata, read directly
# to filter mice by learning_category. Same file dyad_pipeline.py already
# reads for reward_group.
WEIGHT_XLSX = Path(r"M:\share_internal\Axel_Bisi_Share\dataset_info\joint_mouse_reference_weight.xlsx")

# EDIT these to match your actual Excel column headers / category values
EXCLUDE_BAD_LEARNING_MICE = True  # False to include every mouse regardless of learning_category
MOUSE_ID_COL = "mouse_id"         # mirrors dyad_pipeline.py's WEIGHT_COLS["mouse"]
LEARNING_CATEGORY_COL = "learning_category"
BAD_LEARNING_VALUES = {"bad"}     # case-insensitive match; add more labels if needed

CONNECTED_THRESH = -7.5  # must match dyad_pipeline.py
N_PERM = 999
SEED = 0
MIN_N_PER_GROUP = 3   # minimum sessions per group for a per-area post-hoc test to run
MAX_AREAS_SHOWN = 20  # cap per-area bar charts to the N areas with the largest |effect|
MAX_AREAS_STACKED = 10  # cap for stacked per-area composition bars; rest grouped as "Other"
N_WORKERS = min(8, os.cpu_count() or 4)  # EDIT: tune for your machine. Used for both
# thread-parallel loading (CSV/pickle I/O) and process-parallel figure generation.

SIGNS = ("E", "I")
SIGN_COLORS = {"E": "tab:orange", "I": "tab:purple"}
# EDIT if your reward_group values differ from these
REWARD_COLORS = {"R+": "forestgreen", "R-": "crimson"}

# Fixed 2x2 layout for every score-change figure: row=sign, col=comparison,
# in this exact order (top-left to bottom-right): E/inflection, E/passive,
# I/inflection, I/passive.
CONDITIONS = [("E", "inflection"), ("E", "passive"), ("I", "inflection"), ("I", "passive")]

SAVE_DPI = 300
# Only font-size/DPI overrides needed for our many small, dense subplots --
# everything else (background, grid, spines, palette) stays seaborn's
# default theme, set via sns.set_theme() above.
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
})


# ============================== LOADING ======================================
def _read_csv(path):
    """Module-level (picklable/thread-safe) single-file CSV reader, used by
    the thread-parallel loaders below."""
    return pd.read_csv(path)


def load_all_session_results():
    """Prefer the combined CSV; fall back to concatenating every per-session
    *_score_changes.csv found under OUTPUT_ROOT/{mouse}/whisker_*/dyad/.
    Per-session files are read in parallel (threads -- this is I/O-bound,
    especially over a network drive, so threads avoid both the GIL for I/O
    waits and the pickling overhead a process pool would add here)."""
    files = sorted(OUTPUT_ROOT.glob("*/whisker_*/dyad/*_score_changes.csv"))
    if files:
        # Per-session CSVs are the freshest source of truth -- one gets
        # written every time a session finishes processing. The combined
        # CSV is only written once per full pipeline run and can go stale
        # (e.g. left over from an earlier debug=True run with fewer mice),
        # so it is never preferred over per-session files when both exist.
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            dfs = list(ex.map(_read_csv, files))
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(files)} per-session CSVs ({N_WORKERS} threads): {len(df):,} rows, "
              f"{df['mouse'].nunique()} mice")
        try:
            df.to_csv(COMBINED_CSV, index=False)
            print(f"Refreshed combined CSV at {COMBINED_CSV}")
        except OSError as e:
            warnings.warn(f"Could not refresh combined CSV ({e}); continuing anyway")
    elif COMBINED_CSV.exists():
        df = pd.read_csv(COMBINED_CSV)
        print(f"No per-session CSVs found; loaded combined CSV: {len(df):,} rows "
              f"from {COMBINED_CSV}")
    else:
        raise FileNotFoundError(
            f"No combined CSV at {COMBINED_CSV} and no per-session "
            f"*_score_changes.csv files found under {OUTPUT_ROOT}"
        )

    df["area_pair"] = df["area_source"].astype(str) + "\u2192" + df["area_target"].astype(str)
    return df


def load_connectivity_summary():
    """Load the TRUE connectivity-rate summary (global/within/across, with
    real total-candidate-pair denominators) written by dyad_pipeline.py's
    session_connectivity_summary(). Same freshness preference as
    load_all_session_results(): per-session CSVs over the combined one."""
    files = sorted(OUTPUT_ROOT.glob("*/whisker_*/dyad/*_connectivity_summary.csv"))
    if files:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            dfs = list(ex.map(_read_csv, files))
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(files)} per-session connectivity CSVs ({N_WORKERS} threads): {len(df):,} rows")
        try:
            df.to_csv(COMBINED_CONNECTIVITY_CSV, index=False)
            print(f"Refreshed combined connectivity CSV at {COMBINED_CONNECTIVITY_CSV}")
        except OSError as e:
            warnings.warn(f"Could not refresh combined connectivity CSV ({e}); continuing anyway")
    elif COMBINED_CONNECTIVITY_CSV.exists():
        df = pd.read_csv(COMBINED_CONNECTIVITY_CSV)
        print(f"No per-session connectivity CSVs found; loaded combined: {len(df):,} rows")
    else:
        raise FileNotFoundError(
            f"No combined connectivity CSV at {COMBINED_CONNECTIVITY_CSV} and no "
            f"per-session *_connectivity_summary.csv files found under {OUTPUT_ROOT}. "
            f"Re-run the updated dyad_pipeline.py to generate these."
        )
    return df


def load_excluded_mice():
    """Mice to exclude because their learning_category in
    joint_mouse_reference_weight.xlsx is flagged 'bad' (or whatever
    BAD_LEARNING_VALUES contains). Returns an empty set (excludes nobody)
    if EXCLUDE_BAD_LEARNING_MICE is False, the file can't be read, or the
    expected columns aren't found -- fails safe rather than silently
    dropping data on an unexpected schema."""
    if not EXCLUDE_BAD_LEARNING_MICE:
        return set()

    try:
        weight = pd.read_excel(WEIGHT_XLSX)
    except (FileNotFoundError, OSError) as e:
        warnings.warn(f"Could not read {WEIGHT_XLSX} ({e}); not excluding any mice by learning_category")
        return set()

    if MOUSE_ID_COL not in weight.columns or LEARNING_CATEGORY_COL not in weight.columns:
        warnings.warn(f"'{MOUSE_ID_COL}' or '{LEARNING_CATEGORY_COL}' not found in {WEIGHT_XLSX} "
                       f"(columns present: {list(weight.columns)}); not excluding any mice")
        return set()

    bad_lower = {v.strip().lower() for v in BAD_LEARNING_VALUES}
    is_bad = weight[LEARNING_CATEGORY_COL].astype(str).str.strip().str.lower().isin(bad_lower)
    excluded = set(weight.loc[is_bad, MOUSE_ID_COL].astype(str))
    print(f"learning_category filter: excluding {len(excluded)} mice "
          f"({LEARNING_CATEGORY_COL} in {BAD_LEARNING_VALUES}): {sorted(excluded)}")
    return excluded


def _filter_excluded_mice(df, excluded_mice, mouse_col="mouse"):
    """Drop rows for mice in excluded_mice. No-op if the set is empty."""
    if not excluded_mice:
        return df
    before = df[mouse_col].nunique()
    filtered = df[~df[mouse_col].astype(str).isin(excluded_mice)].copy()
    after = filtered[mouse_col].nunique()
    print(f"learning_category filter: kept {after}/{before} mice in this dataframe")
    return filtered


def _dedupe_connected_pairs(df):
    """Each connected pair appears once per comparison (passive/inflection)
    in the tidy dataframe, even though 'connected' status comes from the
    whole-session score and is independent of comparison. Any analysis of
    WHERE connected pairs are (not their score changes) must collapse back
    to one row per (mouse, session, sign, pair_id) first, or pairs present
    in both comparisons get counted twice."""
    return df.drop_duplicates(subset=["mouse", "session_id", "sign", "pair_id"])[
        ["mouse", "session_id", "sign", "reward_group", "day_type",
         "pair_id", "area_source", "area_target", "area_pair"]
    ].copy()


def _valid_score_diff(df):
    """Filter to rows where BOTH score_pre and score_post individually
    exceed CONNECTED_THRESH -- see module docstring. Only used for
    score_diff-based figures; connectivity/composition figures use the
    unfiltered df since 'connected' status doesn't depend on this."""
    mask = (df["score_pre"] > CONNECTED_THRESH) & (df["score_post"] > CONNECTED_THRESH)
    n_before, n_after = len(df), int(mask.sum())
    print(f"score_diff validity filter: {n_after:,}/{n_before:,} rows kept "
          f"({n_before - n_after:,} dropped for score_pre/post <= {CONNECTED_THRESH})")
    return df.loc[mask].copy()


def _filter_area_relation(df, area_relation):
    """Filter to within-area (area_source == area_target) or across-area
    pairs. Uses the area_relation column if present (written by the
    updated dyad_pipeline.py's process_session), else derives it from
    area_source/area_target directly. area_relation=None returns df
    unchanged (pooled)."""
    if area_relation is None:
        return df
    if "area_relation" in df.columns:
        return df[df["area_relation"] == area_relation]
    is_within = df["area_source"] == df["area_target"]
    return df[is_within] if area_relation == "within" else df[~is_within]


def _tag_suffix(area_relation):
    """Filename suffix for a given area_relation split ('' for pooled)."""
    return f"_{area_relation}" if area_relation else ""


def _tag_title(area_relation):
    """Title annotation for a given area_relation split."""
    return f" [{area_relation}-area]" if area_relation else " [pooled: within+across]"


# ============================== STATISTICS ===================================
def permanova_oneway(values, groups, n_perm=N_PERM, seed=SEED):
    """Permutation-based one-way ANOVA (pseudo-F via label permutation).
    Returns (F, p). See module docstring for why this is the correct
    univariate case of PERMANOVA. Verified against scipy.stats.f_oneway."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    mask = ~np.isnan(values)
    values, groups = values[mask], groups[mask]
    uniq, codes = np.unique(groups, return_inverse=True)
    k, n = len(uniq), len(values)
    if k < 2 or n <= k:
        return np.nan, np.nan

    ss_total = np.sum((values - values.mean()) ** 2)
    df_between, df_within = k - 1, n - k

    def pseudo_F(codes_):
        sums = np.bincount(codes_, weights=values, minlength=k)
        sumsq = np.bincount(codes_, weights=values**2, minlength=k)
        counts = np.bincount(codes_, minlength=k).astype(float)
        counts_safe = np.where(counts > 0, counts, 1)
        ss_within = np.sum(sumsq - (sums**2) / counts_safe)
        ss_between = ss_total - ss_within
        return np.inf if ss_within <= 0 else (ss_between / df_between) / (ss_within / df_within)

    F_obs = pseudo_F(codes)
    rng = np.random.default_rng(seed)
    perm_F = np.empty(n_perm)
    for i in range(n_perm):
        perm_F[i] = pseudo_F(rng.permutation(codes))
    p = (np.sum(perm_F >= F_obs) + 1) / (n_perm + 1)
    return F_obs, p


def _bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction. Verified against
    statsmodels.stats.multitest.multipletests(method='fdr_bh')."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    return adj


def mannwhitney_twogroup(values, groups):
    """Two-sided Mann-Whitney U test: unpaired, non-parametric, exactly two
    independent groups. Returns (U, p). NaN if there aren't exactly two
    non-empty groups."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    mask = ~np.isnan(values)
    values, groups = values[mask], groups[mask]
    uniq = np.unique(groups)
    if len(uniq) != 2:
        return np.nan, np.nan
    a = values[groups == uniq[0]]
    b = values[groups == uniq[1]]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    return U, p


def posthoc_per_area(df_area_level, group_col, area_col="area_pair", value_col="score_diff",
                      n_perm=N_PERM, min_n=MIN_N_PER_GROUP, test_func=None, stat_name="F"):
    """Re-run a statistical test independently within each area, then
    BH-FDR-correct across areas. df_area_level must already be aggregated
    to one row per (session, area, group) -- not raw pairs.

    test_func defaults to permanova_oneway (stat_name 'F'). Pass
    mannwhitney_twogroup (stat_name 'U') for a strictly-two-group
    non-parametric post-hoc -- used for all reward_group (R+/R-) post-hocs
    per the omnibus-then-conditional-posthoc pattern (see
    omnibus_and_conditional_posthoc)."""
    if test_func is None:
        test_func = lambda v, g: permanova_oneway(v, g, n_perm=n_perm)

    rows = []
    for area, sub in df_area_level.groupby(area_col):
        counts = sub.groupby(group_col).size()
        if (counts >= min_n).sum() < 2:
            rows.append(dict(area=area, stat=np.nan, p_raw=np.nan,
                              n_sessions=len(sub), note="insufficient data"))
            continue
        stat, p = test_func(sub[value_col], sub[group_col])
        rows.append(dict(area=area, stat=stat, p_raw=p, n_sessions=len(sub), note=""))

    out = pd.DataFrame(rows)
    out.insert(1, "stat_name", stat_name)
    valid = out["p_raw"].notna()
    out["p_adj"] = np.nan
    if valid.sum():
        out.loc[valid, "p_adj"] = _bh_fdr(out.loc[valid, "p_raw"].to_numpy())
    return out.sort_values("p_adj", na_position="last").reset_index(drop=True)


ALPHA = 0.05  # significance threshold gating reward-group post-hoc tests


def omnibus_and_conditional_posthoc(session_df, area_df, group_col="reward_group",
                                     value_col="score_diff", area_col="area_pair",
                                     alpha=ALPHA, n_perm=N_PERM, min_n=MIN_N_PER_GROUP):
    """Omnibus PERMANOVA (permutation-ANOVA, shuffling group_col labels --
    session_df is the session-level, de-pseudoreplicated unit) followed by
    a per-area post-hoc ONLY IF the omnibus is significant at alpha. The
    post-hoc uses Mann-Whitney U (non-parametric, appropriate since
    group_col is always exactly two groups here -- R+ vs R-), BH-FDR
    corrected across areas.

    Returns (F, p, posthoc_df_or_None). posthoc_df is None when the omnibus
    isn't significant (or can't be computed) -- callers should handle that.
    """
    F, p = permanova_oneway(session_df[value_col], session_df[group_col], n_perm=n_perm)
    if p is None or np.isnan(p) or p >= alpha:
        return F, p, None
    posthoc_df = posthoc_per_area(area_df, group_col=group_col, area_col=area_col,
                                   value_col=value_col, min_n=min_n,
                                   test_func=mannwhitney_twogroup, stat_name="U")
    return F, p, posthoc_df


# ============================== PLOT HELPERS =================================
def _style_ax(ax, xlabel="", ylabel="Mean CCG score change", zero_line=True, vertical_zero=False):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params()
    if zero_line and not vertical_zero:
        ax.axhline(0, color="grey", lw=0.8, linestyle="--", zorder=0)
    if vertical_zero:
        ax.axvline(0, color="grey", lw=0.8, linestyle="--", zorder=0)


def _boxplot_with_points(ax, groups_data, labels, colors, jitter=0.06, seed=SEED):
    """Boxplot + jittered individual points. showmeans=True already draws
    BOTH mean (triangle marker) and median (the box's center line) --
    satisfies 'show mean and median' without extra work for any boxplot
    figure."""
    rng = np.random.default_rng(seed)
    bp = ax.boxplot(groups_data, tick_labels=labels, showmeans=True,
                     patch_artist=True, widths=0.6, zorder=2,
                     medianprops=dict(color="black", linewidth=1.2),
                     meanprops=dict(marker="^", markerfacecolor="white",
                                    markeredgecolor="black", markersize=5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for i, y in enumerate(groups_data, start=1):
        if len(y) == 0:
            continue
        xj = rng.normal(i, jitter, size=len(y))
        ax.scatter(xj, y, s=8, color="black", alpha=0.3, zorder=3, linewidths=0)
    return bp


def _p_str(p):
    if p is None or np.isnan(p):
        return "p = NA"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _sig_stars(p):
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _add_mean_median_lines(ax, values, color):
    """Vertical lines marking mean (dashed) and median (dotted) of values,
    in the given color. Used on histogram panels to satisfy 'show mean and
    median' there too."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return
    ax.axvline(np.mean(values), color=color, linestyle="--", linewidth=1.4, alpha=0.95, zorder=4)
    ax.axvline(np.median(values), color=color, linestyle=":", linewidth=1.6, alpha=0.95, zorder=4)


def _condition_grid(figsize_per_panel=3.2, sharex=True, sharey=True):
    """2x2 figure, one panel per (sign, comparison) in CONDITIONS, fixed
    layout (row=sign, col=comparison). Axes shared for direct visual
    comparison (per request), but EVERY panel still shows its own tick
    labels (overriding matplotlib's default hiding of inner labels under
    sharex/sharey), and every panel is forced square via set_box_aspect.
    Returns (fig, axes, ax_map) where ax_map[(sign, comparison)] -> Axes."""
    fig, axes = plt.subplots(2, 2, figsize=(figsize_per_panel * 2, figsize_per_panel * 2),
                              sharex=sharex, sharey=sharey, squeeze=False)
    ax_map = {}
    for idx, (sign, comparison) in enumerate(CONDITIONS):
        i, j = idx // 2, idx % 2
        ax = axes[i][j]
        ax.set_box_aspect(1)
        ax.tick_params(labelbottom=True, labelleft=True)
        ax_map[(sign, comparison)] = ax
    return fig, axes, ax_map


def _area_barh(ax, cell, area_col="area_pair", mean_col="mean", err_col="sem",
               color="tab:blue", max_areas=MAX_AREAS_SHOWN, sig_col=None):
    """Horizontal bar chart of mean +/- error per area, capped to the
    max_areas largest |effect| areas (kept sorted by value for readability).
    If sig_col is given (a column of stars strings, e.g. from _sig_stars),
    each bar is annotated with its stars just past the error bar, on
    whichever side the bar extends (handles negative values correctly).
    Returns (n_total_areas, n_shown)."""
    n_total = len(cell)
    if n_total == 0:
        ax.axis("off")
        return 0, 0
    if n_total > max_areas:
        cell = cell.reindex(cell[mean_col].abs().sort_values(ascending=False).index[:max_areas])
    cell = cell.sort_values(mean_col)
    y = np.arange(len(cell))
    ax.barh(y, cell[mean_col], xerr=cell[err_col], color=color, capsize=2,
            edgecolor="black", linewidth=0.4, height=0.72, error_kw=dict(linewidth=0.8))
    ax.set_yticks(y)
    ax.set_yticklabels(cell[area_col])
    ax.margins(y=0.02)

    if sig_col is not None and sig_col in cell.columns:
        span = cell[mean_col].abs().max()
        pad = 0.03 * span if span else 0.1
        for yi, (val, err, stars) in enumerate(zip(cell[mean_col], cell[err_col], cell[sig_col])):
            if not stars or stars == "n.s.":
                continue
            err = err if pd.notna(err) else 0
            xpos = (val + err + pad) if val >= 0 else (val - err - pad)
            ax.text(xpos, yi, stars, va="center", ha="left" if val >= 0 else "right",
                    fontsize=7, fontweight="bold")

    return n_total, len(cell)


def _area_barh_grouped(ax, table, area_col="area_pair", value_col="pct", group_col="reward_group",
                        colors=REWARD_COLORS, max_areas=MAX_AREAS_SHOWN, sig_map=None,
                        err_col=None, median_col=None):
    """Grouped horizontal bars: one bar per group_col value, per area.
    Capped to the max_areas areas with the largest combined value.

    err_col: optional column for xerr (error bars).
    median_col: optional column of median values -- drawn as a small black
    diamond marker at the median position for each bar (the bar length IS
    the mean, so this satisfies 'show mean and median' for bar figures).
    sig_map: optional dict area -> stars string, one annotation per area
    row past its tallest bar (+error).
    Returns (n_total_areas, n_shown)."""
    pivot = table.pivot_table(index=area_col, columns=group_col, values=value_col, fill_value=0)
    n_total = len(pivot)
    if n_total == 0:
        ax.axis("off")
        return 0, 0
    combined = pivot.sum(axis=1)
    if n_total > max_areas:
        top_idx = combined.sort_values(ascending=False).index[:max_areas]
        pivot = pivot.loc[top_idx]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]

    err_pivot = None
    if err_col:
        err_pivot = table.pivot_table(index=area_col, columns=group_col, values=err_col, fill_value=0)
        err_pivot = err_pivot.reindex(index=pivot.index, columns=pivot.columns, fill_value=0)
    med_pivot = None
    if median_col:
        med_pivot = table.pivot_table(index=area_col, columns=group_col, values=median_col, fill_value=0)
        med_pivot = med_pivot.reindex(index=pivot.index, columns=pivot.columns, fill_value=0)

    groups = list(pivot.columns)
    n_groups = len(groups)
    y = np.arange(len(pivot))
    bar_h = 0.8 / max(n_groups, 1)
    for i, g in enumerate(groups):
        offset = (i - (n_groups - 1) / 2) * bar_h
        xerr = err_pivot[g] if err_pivot is not None else None
        ax.barh(y + offset, pivot[g], xerr=xerr, height=bar_h * 0.9, color=colors.get(g, "tab:gray"),
                 label=g, edgecolor="black", linewidth=0.3, capsize=1.5,
                 error_kw=dict(linewidth=0.6))
        if med_pivot is not None:
            ax.scatter(med_pivot[g], y + offset, marker="D", color="black", s=10,
                       zorder=5, label="median" if i == 0 else None)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.margins(y=0.02)

    if sig_map:
        err_max = err_pivot.values.max() if err_pivot is not None and err_pivot.size else 0
        max_val = pivot.values.max() + err_max if pivot.size else 1
        pad = 0.03 * max_val if max_val else 0.1
        for yi, area in enumerate(pivot.index):
            stars = sig_map.get(area, "")
            if not stars or stars == "n.s.":
                continue
            row_err = err_pivot.loc[area].max() if err_pivot is not None else 0
            row_max = pivot.loc[area].max() + row_err
            ax.text(row_max + pad, yi, stars, va="center", ha="left", fontsize=7, fontweight="bold")

    return n_total, len(pivot)


def _barh_height(n_bars, per_bar=0.22, base=0.9):
    return max(base, per_bar * n_bars + 0.6)


# ============================== FIGURE 1: delta-score histogram =============
def plot_delta_histogram(df, out_dir, stats_dir, area_relation=None):
    """Raw pair-level score_diff histogram, all pairs (within+across, or
    within-only / across-only per area_relation). 2x2 grid over CONDITIONS,
    R+ vs R- overlaid, mean (dashed) + median (dotted) lines. p-value
    annotation uses session-level means (avoids pseudoreplication) even
    though the histogram itself shows raw pair-level values.
    df must already be filtered via _valid_score_diff()."""
    df = _filter_area_relation(df, area_relation)
    tag, ttag = _tag_suffix(area_relation), _tag_title(area_relation)

    fig, axes, ax_map = _condition_grid()
    stats_rows = []

    for sign, comparison in CONDITIONS:
        ax = ax_map[(sign, comparison)]
        sub = df[(df["sign"] == sign) & (df["comparison"] == comparison)]
        groups = sorted(sub["reward_group"].dropna().unique())
        if not groups:
            ax.axis("off")
            continue

        all_vals = sub["score_diff"].to_numpy()
        bins = np.linspace(all_vals.min(), all_vals.max(), 30) if len(all_vals) else 30
        for g in groups:
            d = sub.loc[sub["reward_group"] == g, "score_diff"].to_numpy()
            color = REWARD_COLORS.get(g, "tab:gray")
            ax.hist(d, bins=bins, alpha=0.45, color=color, label=f"{g} (n={len(d)})", density=True)
            _add_mean_median_lines(ax, d, color)

        session_agg = sub.groupby(["mouse", "session_id", "reward_group"])["score_diff"].mean().reset_index()
        F, p = permanova_oneway(session_agg["score_diff"], session_agg["reward_group"])
        stats_rows.append(dict(sign=sign, comparison=comparison, area_relation=area_relation or "pooled",
                                F=F, p=p, n_sessions=len(session_agg)))

        ax.legend(fontsize=6, frameon=False, loc="upper right")
        ax.set_title(f"{sign} \u2014 {comparison}\n{_p_str(p)} {_sig_stars(p)}", fontsize=8, fontweight="bold")
        _style_ax(ax, xlabel="Score change (raw pairs)", ylabel="Density", zero_line=True)

    fig.suptitle(f"Histogram of score change{ttag}\n(dashed=mean, dotted=median; per reward group)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_dir / f"delta_histogram{tag}.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(stats_rows).to_csv(stats_dir / f"delta_histogram_stats{tag}.csv", index=False)


# ============================== FIGURE 2: delta-score boxplot ===============
def plot_delta_boxplot(df, out_dir, stats_dir, area_relation=None):
    """Session-level mean score_diff, boxplot, all pairs (within+across, or
    within-only / across-only per area_relation). 2x2 grid over CONDITIONS,
    R+ vs R-. Boxplot already shows both mean (triangle) and median (box
    line) -- see _boxplot_with_points. df must already be filtered via
    _valid_score_diff()."""
    df = _filter_area_relation(df, area_relation)
    tag, ttag = _tag_suffix(area_relation), _tag_title(area_relation)

    fig, axes, ax_map = _condition_grid()
    stats_rows = []

    for sign, comparison in CONDITIONS:
        ax = ax_map[(sign, comparison)]
        sub = df[(df["sign"] == sign) & (df["comparison"] == comparison)]
        session_agg = sub.groupby(["mouse", "session_id", "reward_group"])["score_diff"].mean().reset_index()
        groups = sorted(session_agg["reward_group"].dropna().unique())
        if not groups:
            ax.axis("off")
            continue
        data = [session_agg.loc[session_agg["reward_group"] == g, "score_diff"].to_numpy() for g in groups]
        colors = [REWARD_COLORS.get(g, "tab:gray") for g in groups]
        _boxplot_with_points(ax, data, groups, colors)

        F, p = permanova_oneway(session_agg["score_diff"], session_agg["reward_group"])
        stats_rows.append(dict(sign=sign, comparison=comparison, area_relation=area_relation or "pooled",
                                F=F, p=p, n_sessions=len(session_agg)))
        ax.set_title(f"{sign} \u2014 {comparison}\n{_p_str(p)} {_sig_stars(p)}", fontsize=8, fontweight="bold")
        _style_ax(ax, xlabel="Reward group", ylabel="Mean score change / session")

    fig.suptitle(f"Boxplot of session-level mean score change{ttag}\n(\u25b3=mean, line=median)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_dir / f"delta_boxplot{tag}.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(stats_rows).to_csv(stats_dir / f"delta_boxplot_stats{tag}.csv", index=False)


# ============================== FIGURE 3: delta-score by area ===============
def plot_delta_by_area(df, out_dir, stats_dir, area_relation=None):
    """Per-area version of the histogram/boxplot above: mean +/- SEM score
    change per area (median also marked, black diamond), R+ vs R- grouped
    bars, 2x2 grid over CONDITIONS. Omnibus PERMANOVA (session-level,
    pooled across areas) + conditional per-area Mann-Whitney post-hoc
    (stars), post-hoc only run if that panel's omnibus is significant --
    see omnibus_and_conditional_posthoc. df must already be filtered via
    _valid_score_diff().

    Only the x-axis (score-change magnitude) is shared across panels --
    the y-axis (which areas appear) is NOT shared, since the top-N areas
    shown can legitimately differ per condition; sharing categorical labels
    that differ per panel would be misleading, not just cosmetic."""
    df = _filter_area_relation(df, area_relation)
    tag, ttag = _tag_suffix(area_relation), _tag_title(area_relation)
    fig_dir = out_dir / "by_area"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes, ax_map = _condition_grid(figsize_per_panel=4.2, sharex=True, sharey=False)
    stats_rows = []

    for sign, comparison in CONDITIONS:
        ax = ax_map[(sign, comparison)]
        sub = df[(df["sign"] == sign) & (df["comparison"] == comparison)]
        if sub.empty:
            ax.axis("off")
            continue

        session_agg = sub.groupby(["mouse", "session_id", "reward_group"])["score_diff"].mean().reset_index()
        area_agg = (sub.groupby(["mouse", "session_id", "reward_group", "area_pair"])["score_diff"]
                      .mean().reset_index())

        F, p, posthoc = omnibus_and_conditional_posthoc(session_agg, area_agg)
        sig_map = {}
        if posthoc is not None:
            posthoc.to_csv(stats_dir / f"posthoc_area_delta_{sign}_{comparison}{tag}.csv", index=False)
            sig_map = dict(zip(posthoc["area"], posthoc["p_adj"].apply(_sig_stars)))
        stats_rows.append(dict(sign=sign, comparison=comparison, area_relation=area_relation or "pooled",
                                F=F, p=p, posthoc_ran=posthoc is not None, n_sessions=len(session_agg)))

        table = (area_agg.groupby(["reward_group", "area_pair"])["score_diff"]
                   .agg(mean="mean",
                        sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0,
                        median="median")
                   .reset_index())
        n_total, n_shown = _area_barh_grouped(ax, table, value_col="mean", err_col="sem",
                                               median_col="median", sig_map=sig_map)
        cap_note = f" (top {n_shown}/{n_total})" if n_shown < n_total else f" (n={n_shown})"
        ax.set_title(f"{sign} \u2014 {comparison}{cap_note}\nomnibus {_p_str(p)} {_sig_stars(p)}",
                     fontsize=7, fontweight="bold")
        _style_ax(ax, xlabel="Mean score change \u00b1 SEM", ylabel="", vertical_zero=True, zero_line=False)
        if (sign, comparison) == CONDITIONS[0]:
            ax.legend(fontsize=6, frameon=False, loc="lower right")

    fig.suptitle(f"Score change by area{ttag}\n(\u2666=median; stars: per-area Mann-Whitney "
                 "post-hoc, only shown if omnibus PERMANOVA significant)", fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(fig_dir / f"delta_by_area{tag}.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(stats_rows).to_csv(stats_dir / f"delta_by_area_stats{tag}.csv", index=False)


# ============ FIGURES: sign-of-change (% pairs increased / decreased) ======
def _session_sign_stats(df, area_relation=None):
    """Per session (and per session x area): % of pairs with score_diff > 0
    (pct_increase) and < 0 (pct_decrease). Threshold-agnostic -- depends
    only on the SIGN of an already-computed score_diff, not on
    CONNECTED_THRESH (unlike which pairs were connected in the first
    place, which is a pipeline-level decision made before this script ever
    sees the data). Returns (session_level_df, session_x_area_df)."""
    df = _filter_area_relation(df, area_relation)

    def _pct_inc(s):
        return (s > 0).mean() * 100

    def _pct_dec(s):
        return (s < 0).mean() * 100

    session = (df.groupby(["mouse", "session_id", "reward_group", "sign", "comparison"])["score_diff"]
                 .agg(pct_increase=_pct_inc, pct_decrease=_pct_dec, n="size").reset_index())
    area = (df.groupby(["mouse", "session_id", "reward_group", "sign", "comparison", "area_pair"])["score_diff"]
              .agg(pct_increase=_pct_inc, pct_decrease=_pct_dec, n="size").reset_index())
    return session, area


def plot_sign_histogram(df, out_dir, stats_dir, direction="increase", area_relation=None):
    """Histogram (across sessions) of per-session % pairs increased/
    decreased, R+ vs R- overlaid, 2x2 grid over CONDITIONS. Unlike
    plot_delta_histogram, there's no raw-pair-level unit for this metric (a
    single pair doesn't have a '% increased') -- this histogram is already
    over the same session-level values used by plot_sign_boxplot, just
    visualized as a distribution instead of a box. df must already be
    filtered via _valid_score_diff()."""
    value_col = f"pct_{direction}"
    session, _ = _session_sign_stats(df, area_relation)
    tag, ttag = _tag_suffix(area_relation), _tag_title(area_relation)

    fig, axes, ax_map = _condition_grid()
    stats_rows = []

    for sign, comparison in CONDITIONS:
        ax = ax_map[(sign, comparison)]
        sub = session[(session["sign"] == sign) & (session["comparison"] == comparison)]
        groups = sorted(sub["reward_group"].dropna().unique())
        if not groups:
            ax.axis("off")
            continue

        bins = np.linspace(0, 100, 21)
        for g in groups:
            d = sub.loc[sub["reward_group"] == g, value_col].to_numpy()
            color = REWARD_COLORS.get(g, "tab:gray")
            ax.hist(d, bins=bins, alpha=0.45, color=color, label=f"{g} (n={len(d)})", density=True)
            _add_mean_median_lines(ax, d, color)

        F, p = permanova_oneway(sub[value_col], sub["reward_group"])
        stats_rows.append(dict(sign=sign, comparison=comparison, direction=direction,
                                area_relation=area_relation or "pooled", F=F, p=p, n_sessions=len(sub)))
        ax.legend(fontsize=6, frameon=False, loc="upper right")
        ax.set_title(f"{sign} \u2014 {comparison}\n{_p_str(p)} {_sig_stars(p)}", fontsize=8, fontweight="bold")
        _style_ax(ax, xlabel=f"% pairs {direction}d / session", ylabel="Density", zero_line=False)

    fig.suptitle(f"Histogram of % pairs {direction}d (across sessions){ttag}\n"
                 "(dashed=mean, dotted=median; threshold-agnostic)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_dir / f"sign_{direction}_histogram{tag}.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(stats_rows).to_csv(stats_dir / f"sign_{direction}_histogram_stats{tag}.csv", index=False)


def plot_sign_boxplot(df, out_dir, stats_dir, direction="increase", area_relation=None):
    """Boxplot of per-session % pairs increased/decreased, R+ vs R-, 2x2
    grid over CONDITIONS. Boxplot already shows both mean (triangle) and
    median (box line). df must already be filtered via _valid_score_diff()."""
    value_col = f"pct_{direction}"
    session, _ = _session_sign_stats(df, area_relation)
    tag, ttag = _tag_suffix(area_relation), _tag_title(area_relation)

    fig, axes, ax_map = _condition_grid()
    stats_rows = []

    for sign, comparison in CONDITIONS:
        ax = ax_map[(sign, comparison)]
        sub = session[(session["sign"] == sign) & (session["comparison"] == comparison)]
        groups = sorted(sub["reward_group"].dropna().unique())
        if not groups:
            ax.axis("off")
            continue
        data = [sub.loc[sub["reward_group"] == g, value_col].to_numpy() for g in groups]
        colors = [REWARD_COLORS.get(g, "tab:gray") for g in groups]
        _boxplot_with_points(ax, data, groups, colors)

        F, p = permanova_oneway(sub[value_col], sub["reward_group"])
        stats_rows.append(dict(sign=sign, comparison=comparison, direction=direction,
                                area_relation=area_relation or "pooled", F=F, p=p, n_sessions=len(sub)))
        ax.set_title(f"{sign} \u2014 {comparison}\n{_p_str(p)} {_sig_stars(p)}", fontsize=8, fontweight="bold")
        _style_ax(ax, xlabel="Reward group", ylabel=f"% pairs {direction}d / session", zero_line=False)

    fig.suptitle(f"Boxplot of % pairs {direction}d / session{ttag}\n"
                 "(\u25b3=mean, line=median; threshold-agnostic)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_dir / f"sign_{direction}_boxplot{tag}.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(stats_rows).to_csv(stats_dir / f"sign_{direction}_boxplot_stats{tag}.csv", index=False)


def plot_sign_by_area(df, out_dir, stats_dir, direction="increase", area_relation=None):
    """Per-area version: mean +/- SEM % pairs increased/decreased per area
    (median also marked), R+ vs R- grouped bars, 2x2 grid over CONDITIONS.
    Omnibus PERMANOVA + conditional per-area Mann-Whitney post-hoc, same
    gating pattern as plot_delta_by_area. df must already be filtered via
    _valid_score_diff()."""
    value_col = f"pct_{direction}"
    session, area = _session_sign_stats(df, area_relation)
    tag, ttag = _tag_suffix(area_relation), _tag_title(area_relation)
    fig_dir = out_dir / "by_area"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes, ax_map = _condition_grid(figsize_per_panel=4.2, sharex=True, sharey=False)
    stats_rows = []

    for sign, comparison in CONDITIONS:
        ax = ax_map[(sign, comparison)]
        sess_sub = session[(session["sign"] == sign) & (session["comparison"] == comparison)]
        area_sub = area[(area["sign"] == sign) & (area["comparison"] == comparison)]
        if sess_sub.empty:
            ax.axis("off")
            continue

        F, p, posthoc = omnibus_and_conditional_posthoc(sess_sub, area_sub, value_col=value_col)
        sig_map = {}
        if posthoc is not None:
            posthoc.to_csv(stats_dir / f"posthoc_area_sign_{direction}_{sign}_{comparison}{tag}.csv", index=False)
            sig_map = dict(zip(posthoc["area"], posthoc["p_adj"].apply(_sig_stars)))
        stats_rows.append(dict(sign=sign, comparison=comparison, direction=direction,
                                area_relation=area_relation or "pooled",
                                F=F, p=p, posthoc_ran=posthoc is not None, n_sessions=len(sess_sub)))

        table = (area_sub.groupby(["reward_group", "area_pair"])[value_col]
                   .agg(mean="mean",
                        sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0,
                        median="median")
                   .reset_index())
        n_total, n_shown = _area_barh_grouped(ax, table, value_col="mean", err_col="sem",
                                               median_col="median", sig_map=sig_map)
        cap_note = f" (top {n_shown}/{n_total})" if n_shown < n_total else f" (n={n_shown})"
        ax.set_title(f"{sign} \u2014 {comparison}{cap_note}\nomnibus {_p_str(p)} {_sig_stars(p)}",
                     fontsize=7, fontweight="bold")
        _style_ax(ax, xlabel=f"Mean % {direction}d \u00b1 SEM", ylabel="", zero_line=False)
        if (sign, comparison) == CONDITIONS[0]:
            ax.legend(fontsize=6, frameon=False, loc="lower right")

    fig.suptitle(f"% pairs {direction}d by area{ttag}\n(\u2666=median; stars: per-area Mann-Whitney "
                 "post-hoc, only shown if omnibus PERMANOVA significant)", fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(fig_dir / f"sign_{direction}_by_area{tag}.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(stats_rows).to_csv(stats_dir / f"sign_{direction}_by_area_stats{tag}.csv", index=False)


# ============================== FIGURE: where connected pairs are found =====
def plot_connected_pairs_by_area(df, out_dir):
    """Descriptive (no stats): of all connected pairs in the ENTIRE dataset
    (pooled across mice, sessions, reward groups, and comparisons), what
    proportion falls in each area pair? One figure per sign, never combined.
    Full (untruncated) table saved as CSV alongside the capped figure.
    Uses UNFILTERED df (connectivity presence, not score_diff)."""
    fig_dir = out_dir / "connectivity"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dedup = _dedupe_connected_pairs(df)

    for sign in [s for s in SIGNS if s in dedup["sign"].unique()]:
        sub = dedup[dedup["sign"] == sign]
        counts = sub["area_pair"].value_counts()
        table = pd.DataFrame({
            "area_pair": counts.index,
            "mean": counts.values / counts.sum() * 100,  # 'mean' col name for _area_barh reuse
            "sem": 0.0,
            "n_pairs": counts.values,
        })
        table.sort_values("mean", ascending=False).to_csv(
            fig_dir / f"connected_pairs_by_area_{sign}.csv", index=False)

        height = _barh_height(min(len(table), MAX_AREAS_SHOWN))
        fig, ax = plt.subplots(figsize=(4.4, height))
        n_total, n_shown = _area_barh(ax, table, color=SIGN_COLORS[sign])
        cap_note = f" (top {n_shown}/{n_total})" if n_shown < n_total else f" (all {n_shown})"
        ax.set_xlim(left=0)
        _style_ax(ax, xlabel="% of all connected pairs", ylabel="", zero_line=False)
        ax.set_title(f"{sign} connections{cap_note}", fontweight="bold")
        fig.suptitle("Where connected pairs are found\n"
                     "(pooled: all mice, sessions, reward groups, comparisons)",
                     fontsize=10, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        fig.savefig(fig_dir / f"connected_pairs_by_area_{sign}.png", dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)


# ============================== FIGURE: within- vs across-area =============
def plot_within_vs_across_area(df, out_dir):
    """Descriptive (no stats): of all connected pairs in the ENTIRE dataset,
    what proportion are within-area (area_source == area_target) vs
    across-area (area_source != area_target)? Pooled over all areas, one
    panel per sign, never combined. Uses UNFILTERED df."""
    fig_dir = out_dir / "connectivity"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dedup = _dedupe_connected_pairs(df)
    dedup = dedup.copy()
    dedup["pair_type"] = np.where(dedup["area_source"] == dedup["area_target"],
                                   "Within-area", "Across-area")

    signs_present = [s for s in SIGNS if s in dedup["sign"].unique()]
    fig, axes = plt.subplots(1, len(signs_present), figsize=(2.4 * len(signs_present), 3.2),
                              squeeze=False, sharey=True)
    rows_out = []

    for j, sign in enumerate(signs_present):
        ax = axes[0][j]
        sub = dedup[dedup["sign"] == sign]
        counts = sub["pair_type"].value_counts().reindex(["Within-area", "Across-area"]).fillna(0)
        pct = counts / counts.sum() * 100

        ax.bar(pct.index, pct.values, color=[SIGN_COLORS[sign], "lightgray"],
               edgecolor="black", linewidth=0.6, width=0.6)
        for x, (v, n) in enumerate(zip(pct.values, counts.values)):
            ax.text(x, v + 1.5, f"{v:.1f}%\n(n={int(n):,})", ha="center", va="bottom", fontsize=7)
        ax.set_ylim(0, max(pct.values.max() * 1.3, 10))
        _style_ax(ax, xlabel="", ylabel="% of connected pairs" if j == 0 else "", zero_line=False)
        ax.tick_params(axis="x", rotation=15)
        ax.set_title(sign, fontweight="bold")

        rows_out.append(dict(sign=sign,
                              within_pct=pct.get("Within-area", 0.0), across_pct=pct.get("Across-area", 0.0),
                              within_n=int(counts.get("Within-area", 0)), across_n=int(counts.get("Across-area", 0))))

    fig.suptitle("Connected pairs: within-area vs across-area\n"
                 "(pooled: all mice, sessions, reward groups, comparisons)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(fig_dir / "connected_pairs_within_vs_across_area.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows_out).to_csv(fig_dir / "within_vs_across_area_summary.csv", index=False)


# ============================== FIGURE: connected pairs by mouse ============
def plot_connected_pairs_by_mouse(df, out_dir):
    """Single figure (not one file per mouse): for each mouse, proportion
    of its connected pairs found in each area (top areas + 'Other' bucket
    so bars still sum to 100%). One panel per sign, never combined.
    Uses UNFILTERED df."""
    fig_dir = out_dir / "connectivity"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dedup = _dedupe_connected_pairs(df)

    signs_present = [s for s in SIGNS if s in dedup["sign"].unique()]
    mice = sorted(dedup["mouse"].unique())
    n_mice = len(mice)
    if n_mice == 0:
        warnings.warn("plot_connected_pairs_by_mouse: no data to plot")
        return
    height = max(3.0, 0.22 * n_mice + 1.0)

    fig, axes = plt.subplots(1, len(signs_present), figsize=(5.2 * len(signs_present), height),
                              squeeze=False, sharey=True)

    for j, sign in enumerate(signs_present):
        ax = axes[0][j]
        sub = dedup[dedup["sign"] == sign]

        # Top areas by overall count (this sign, across all mice) fix a
        # consistent stacking order/color mapping across every mouse's bar.
        overall_counts = sub["area_pair"].value_counts()
        top_areas = overall_counts.index[:MAX_AREAS_STACKED].tolist()

        counts = sub.groupby(["mouse", "area_pair"]).size().unstack(fill_value=0)
        counts = counts.reindex(index=mice, fill_value=0)
        top_present = [a for a in top_areas if a in counts.columns]
        other = counts.drop(columns=top_present).sum(axis=1) if len(top_present) < counts.shape[1] else pd.Series(0, index=mice)

        plot_df = pd.DataFrame(index=mice)
        for area in top_areas:
            plot_df[area] = counts[area] if area in counts.columns else 0
        plot_df["Other"] = other

        row_totals = plot_df.sum(axis=1).replace(0, np.nan)
        proportions = (plot_df.div(row_totals, axis=0) * 100).fillna(0)

        colors = list(plt.cm.tab10.colors)[:len(top_areas)] + ["lightgray"]
        y = np.arange(n_mice)
        left = np.zeros(n_mice)
        for area, color in zip(proportions.columns, colors):
            ax.barh(y, proportions[area], left=left, color=color,
                     edgecolor="white", linewidth=0.3, height=0.75, label=area)
            left += proportions[area].to_numpy()

        ax.set_yticks(y)
        ax.set_yticklabels(mice)
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        _style_ax(ax, xlabel="% of mouse's connected pairs",
                  ylabel="Mouse" if j == 0 else "", zero_line=False)
        ax.set_title(f"{sign}  (top {len(top_areas)} areas + Other)", fontweight="bold")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    fig.suptitle("Proportion of connected pairs by area, per mouse", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.97, 0.95])
    fig.savefig(fig_dir / "connected_pairs_by_mouse.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# ============ FIGURE: TRUE connectivity rate by reward group (global/within/across)
def plot_connectivity_by_reward_group(connectivity_df, out_dir, stats_dir):
    """R+ vs R- comparison of TRUE connectivity rates (real total-candidate
    -pair denominators, from combined_connectivity_summary.csv), three
    separate barplots -- global / within-area / across-area. E vs I as
    separate sub-panels (never pooled). 'global' uses Mann-Whitney U
    (unpaired, non-parametric, two groups, per explicit prior instruction);
    'within'/'across' use the omnibus permutation-ANOVA as elsewhere."""
    fig_dir = out_dir / "connectivity"
    fig_dir.mkdir(parents=True, exist_ok=True)

    signs_present = [s for s in SIGNS if s in connectivity_df["sign"].unique()]
    if not signs_present:
        warnings.warn("plot_connectivity_by_reward_group: no data to plot")
        return

    metrics = [
        ("global", "pct_connected_overall", "% connected / session (overall)", "mannwhitney"),
        ("within", "pct_connected_within_area", "% connected / session (within-area)", "permanova"),
        ("across", "pct_connected_across_area", "% connected / session (across-area)", "permanova"),
    ]
    all_stats_rows = []

    for tag, value_col, ylabel, test_type in metrics:
        fig, axes = plt.subplots(1, len(signs_present), figsize=(2.8 * len(signs_present), 3.6),
                                  squeeze=False, sharey=True)
        for j, sign in enumerate(signs_present):
            ax = axes[0][j]
            sub = connectivity_df[connectivity_df["sign"] == sign]
            groups = sorted(sub["reward_group"].dropna().unique())
            data = [sub.loc[sub["reward_group"] == g, value_col].to_numpy() for g in groups]
            colors = [REWARD_COLORS.get(g, "tab:gray") for g in groups]
            _boxplot_with_points(ax, data, groups, colors)
            _style_ax(ax, xlabel="Reward group", ylabel=ylabel if j == 0 else "", zero_line=False)

            if test_type == "mannwhitney":
                stat, p = mannwhitney_twogroup(sub[value_col], sub["reward_group"])
                stat_name = "U"
            else:
                stat, p = permanova_oneway(sub[value_col], sub["reward_group"])
                stat_name = "F"

            all_stats_rows.append(dict(metric=tag, sign=sign, test_type=test_type,
                                        stat_name=stat_name, stat=stat, p=p, n_sessions=len(sub)))
            ax.set_title(f"{sign}\n{_p_str(p)} {_sig_stars(p)}", fontweight="bold")

        titles = {"global": "TRUE connectivity rate (overall) by reward group\n(Mann-Whitney U, unpaired)",
                  "within": "TRUE within-area connectivity rate by reward group",
                  "across": "TRUE across-area connectivity rate by reward group"}
        fig.suptitle(titles[tag], fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.85])
        fig.savefig(fig_dir / f"connectivity_rate_{tag}_by_reward_group.png", dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(all_stats_rows).to_csv(stats_dir / "connectivity_rate_by_reward_group_stats.csv", index=False)


# ======== FIGURE: connectivity COMPOSITION by area (pooled + split), one figure
def _session_area_composition(df, category):
    """Session-level % composition by area for one of 'global'/'within'/
    'across' -- one row per (mouse, session, reward_group, sign, area_pair).
    Percentages are within-category (e.g. 'within' rows sum to 100% across
    areas within the within-only subset), matching the dataset-pooled
    version used for the figure. This is the proper de-pseudoreplicated
    unit for the per-area post-hoc test."""
    dedup = _dedupe_connected_pairs(df)
    dedup = dedup.copy()
    dedup["pair_type"] = np.where(dedup["area_source"] == dedup["area_target"], "within", "across")
    if category != "global":
        dedup = dedup[dedup["pair_type"] == category]

    counts = (dedup.groupby(["mouse", "session_id", "reward_group", "sign", "area_pair"])
                    .size().reset_index(name="n"))
    totals = (dedup.groupby(["mouse", "session_id", "reward_group", "sign"])
                    .size().reset_index(name="n_total"))
    merged = counts.merge(totals, on=["mouse", "session_id", "reward_group", "sign"])
    merged["pct"] = merged["n"] / merged["n_total"] * 100
    return merged


def plot_connectivity_by_area_and_reward_group(df, out_dir, stats_dir):
    """Per-area breakdown of connected-pair COMPOSITION (see module
    docstring -- this is NOT the true rate from plot_connectivity_by_
    reward_group, since no per-area candidate-pair denominator exists).
    One figure per sign: row 0 = R+/R- pooled together, row 1 = split by
    reward group (grouped bars); columns = global / within-only / across-
    only. Per-area post-hoc (R+ vs R-, Mann-Whitney U) saved as CSV for
    each category, using session-level composition (proper unit, avoids
    pseudoreplication).

    NOTE ON OMNIBUS GATING: unlike the other reward-group figures, this
    post-hoc is NOT gated behind an omnibus test. A composition breakdown
    spans many areas simultaneously (not one scalar), so a proper omnibus
    here would need a genuine multivariate test (e.g. PERMANOVA on a full
    composition dissimilarity matrix), which isn't implemented -- gating on
    e.g. the TRUE within/across rate omnibus from plot_connectivity_by_
    reward_group would be testing a different quantity (rate vs
    composition) and could mislead. BH-FDR correction across areas is the
    multiple-comparisons safeguard here instead."""
    fig_dir = out_dir / "connectivity"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dedup = _dedupe_connected_pairs(df)
    dedup = dedup.copy()
    dedup["pair_type"] = np.where(dedup["area_source"] == dedup["area_target"], "within", "across")

    categories = ["global", "within", "across"]
    signs_present = [s for s in SIGNS if s in dedup["sign"].unique()]

    for sign in signs_present:
        sub_sign = dedup[dedup["sign"] == sign]
        panel_data = {
            "global": sub_sign,
            "within": sub_sign[sub_sign["pair_type"] == "within"],
            "across": sub_sign[sub_sign["pair_type"] == "across"],
        }

        # Session-level post-hoc per category (proper unit for testing),
        # computed BEFORE drawing so significance stars can be annotated on
        # the split-row bars below. Not omnibus-gated -- see docstring.
        sig_map_by_cat = {}
        for cat in categories:
            session_comp = _session_area_composition(df, cat)
            session_comp_sign = session_comp[session_comp["sign"] == sign]
            if session_comp_sign.empty:
                sig_map_by_cat[cat] = {}
                continue
            posthoc = posthoc_per_area(session_comp_sign, group_col="reward_group", value_col="pct",
                                        test_func=mannwhitney_twogroup, stat_name="U")
            posthoc.to_csv(stats_dir / f"posthoc_area_composition_{cat}_{sign}.csv", index=False)
            sig_map_by_cat[cat] = dict(zip(posthoc["area"], posthoc["p_adj"].apply(_sig_stars)))

        fig, axes = plt.subplots(2, 3, figsize=(4.2 * 3, 4.0 * 2), squeeze=False)

        # Row 0: pooled across reward groups
        for col, cat in enumerate(categories):
            ax = axes[0][col]
            panel_df = panel_data[cat]
            counts = panel_df["area_pair"].value_counts()
            if counts.empty:
                ax.axis("off")
                continue
            table = pd.DataFrame({"area_pair": counts.index,
                                   "mean": counts.values / counts.sum() * 100,
                                   "sem": 0.0})
            n_total, n_shown = _area_barh(ax, table, color=SIGN_COLORS[sign])
            cap_note = f" (top {n_shown}/{n_total})" if n_shown < n_total else f" (n={n_shown})"
            ax.set_xlim(left=0)
            ax.set_title(f"{cat} \u2014 pooled{cap_note}", fontweight="bold")
            _style_ax(ax, xlabel="% of connected pairs" if col == 0 else "", ylabel="", zero_line=False)

        # Row 1: split by reward group (grouped bars, each group normalized to its own total)
        for col, cat in enumerate(categories):
            ax = axes[1][col]
            panel_df = panel_data[cat]
            if panel_df.empty:
                ax.axis("off")
                continue
            counts = panel_df.groupby(["reward_group", "area_pair"]).size().reset_index(name="n")
            totals = panel_df.groupby("reward_group").size()
            if totals.empty:
                ax.axis("off")
                continue
            counts["pct"] = counts.apply(lambda r: r["n"] / totals[r["reward_group"]] * 100, axis=1)
            n_total, n_shown = _area_barh_grouped(ax, counts, sig_map=sig_map_by_cat[cat])
            cap_note = f" (top {n_shown}/{n_total})" if n_shown < n_total else f" (n={n_shown})"
            ax.set_xlim(left=0)
            ax.set_title(f"{cat} \u2014 R+ vs R-{cap_note}", fontweight="bold")
            _style_ax(ax, xlabel="% of connected pairs (within own group)", ylabel="", zero_line=False)
            if col == len(categories) - 1:
                ax.legend(fontsize=7, frameon=False, loc="lower right")

        fig.suptitle(f"{sign} connections: connected-pair composition by area\n"
                     "(composition of the connected set -- not a true rate; "
                     "see connectivity_rate_*_by_reward_group.png for true rates)\n"
                     "stars (bottom row): per-area Mann-Whitney U post-hoc, BH-FDR (not omnibus-gated -- see docstring)",
                     fontsize=9, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.86])
        fig.savefig(fig_dir / f"connectivity_by_area_{sign}.png", dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)


# ============ CONFOUND CHECK: firing rate / spike count by reward group ====
def _load_one_firing_stat(session_row):
    """Module-level (thread-safe) worker for one session -- see
    load_session_firing_stats. session_row: (mouse, session_id, reward_group)
    namedtuple. Returns a result dict, or None if this session is skipped."""
    mouse, session_id, reward_group = session_row.mouse, session_row.session_id, session_row.reward_group
    pkl_path = DATA_ROOT / session_id / "pairs_light.pkl"
    if not pkl_path.exists():
        warnings.warn(f"pairs_light.pkl not found for {session_id}, skipping")
        return None
    with open(pkl_path, "rb") as f:
        pairs = pickle.load(f)
    pairs = pairs if isinstance(pairs, pd.DataFrame) else pd.DataFrame(pairs)

    needed = {"preIdx", "postIdx", "pre_firing_rate", "post_firing_rate",
              "pre_Nspikes", "post_Nspikes"}
    missing = needed - set(pairs.columns)
    if missing:
        warnings.warn(f"{session_id}: pairs_light.pkl missing {missing}, skipping")
        return None

    units_pre = pairs[["preIdx", "pre_firing_rate", "pre_Nspikes"]].rename(
        columns={"preIdx": "unit_idx", "pre_firing_rate": "firing_rate", "pre_Nspikes": "n_spikes"})
    units_post = pairs[["postIdx", "post_firing_rate", "post_Nspikes"]].rename(
        columns={"postIdx": "unit_idx", "post_firing_rate": "firing_rate", "post_Nspikes": "n_spikes"})
    units = pd.concat([units_pre, units_post], ignore_index=True).drop_duplicates(subset="unit_idx")

    return dict(
        mouse=mouse, session_id=session_id, reward_group=reward_group,
        n_units=len(units),
        mean_firing_rate=units["firing_rate"].mean(),
        median_firing_rate=units["firing_rate"].median(),
        mean_n_spikes=units["n_spikes"].mean(),
    )


def load_session_firing_stats(df):
    """For every (mouse, session_id, reward_group) present in df, load
    pairs_light.pkl directly (small per-pair metadata, not a CCG array) and
    compute session-level mean firing rate / spike count across UNIQUE
    recorded units. One file read per session, in parallel (threads --
    I/O-bound, same rationale as the CSV loaders).

    NOTE: pre_firing_rate/post_firing_rate/pre_Nspikes/post_Nspikes are
    PER-PAIR columns -- the same unit appears once per pair it participates
    in, so a highly-connected unit would be over-weighted by a naive
    pair-level mean. Units are deduplicated by index (preIdx/postIdx) first."""
    sessions = df[["mouse", "session_id", "reward_group"]].drop_duplicates()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        results = list(ex.map(_load_one_firing_stat, sessions.itertuples(index=False)))
    rows = [r for r in results if r is not None]
    return pd.DataFrame(rows)


def plot_firing_rate_confound(df, out_dir, stats_dir):
    """Confound check: does mean firing rate / spike count per session
    differ between R+ and R-? If so, connectivity-detection power (and
    therefore every R+/R- connectivity comparison above) could be
    confounded by this rather than reflecting a real circuit difference.
    Two-sided Mann-Whitney U (unpaired, non-parametric), matching the
    'global' connectivity test above. Uses UNFILTERED df."""
    fig_dir = out_dir / "confounds"
    fig_dir.mkdir(parents=True, exist_ok=True)

    firing_stats = load_session_firing_stats(df)
    if firing_stats.empty:
        warnings.warn("plot_firing_rate_confound: no firing-rate data loaded, skipping")
        return
    firing_stats.to_csv(fig_dir / "session_firing_stats.csv", index=False)

    groups = sorted(firing_stats["reward_group"].dropna().unique())
    if len(groups) < 2:
        warnings.warn("plot_firing_rate_confound: fewer than 2 reward groups present, skipping stats")

    metrics = [("mean_firing_rate", "Mean firing rate / session (Hz)"),
               ("mean_n_spikes", "Mean spike count / session")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.8 * len(metrics), 3.6), squeeze=False)
    stats_rows = []

    for j, (value_col, ylabel) in enumerate(metrics):
        ax = axes[0][j]
        data = [firing_stats.loc[firing_stats["reward_group"] == g, value_col].to_numpy() for g in groups]
        colors = [REWARD_COLORS.get(g, "tab:gray") for g in groups]
        _boxplot_with_points(ax, data, groups, colors)
        _style_ax(ax, xlabel="Reward group", ylabel=ylabel, zero_line=False)

        U, p = mannwhitney_twogroup(firing_stats[value_col], firing_stats["reward_group"])
        stats_rows.append(dict(metric=value_col, test_type="mannwhitney", stat_name="U",
                                stat=U, p=p, n_sessions=len(firing_stats)))
        ax.set_title(f"{_p_str(p)} {_sig_stars(p)}", fontweight="bold")

    fig.suptitle("Confound check: firing rate / spike count by reward group\n(Mann-Whitney U, unpaired)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig(fig_dir / "firing_rate_by_reward_group.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(stats_rows).to_csv(stats_dir / "firing_rate_confound_stats.csv", index=False)


# ============================== MAIN =========================================
# ============================== PARALLEL FIGURE DISPATCH ====================
# Figure-generating functions are independent of each other (each reads
# from an already-loaded dataframe and writes its own files), so they run
# as separate jobs across a process pool. The dataframes are passed ONCE
# via the pool initializer (not re-pickled per job) -- with ~33 jobs this
# avoids serializing potentially large DataFrames repeatedly. out_dir/
# stats_dir are also passed explicitly (not read from module-level
# SUMMARY_DIR inside the worker) since spawned worker processes re-import
# this module fresh and would otherwise see stale/default paths rather
# than whatever was set at call time.
_WORKER = {}


def _init_worker(df, df_valid, connectivity_df, out_dir, stats_dir):
    _WORKER["df"] = df
    _WORKER["df_valid"] = df_valid
    _WORKER["connectivity_df"] = connectivity_df
    _WORKER["out_dir"] = out_dir
    _WORKER["stats_dir"] = stats_dir


def _run_job(func, data_key, needs_stats, kwargs):
    """Runs in a worker process. Returns (func_name, kwargs, traceback_or_None)."""
    try:
        data = _WORKER[data_key]
        if data is None:
            return func.__name__, kwargs, f"SKIPPED: '{data_key}' unavailable"
        if needs_stats:
            func(data, _WORKER["out_dir"], _WORKER["stats_dir"], **kwargs)
        else:
            func(data, _WORKER["out_dir"], **kwargs)
        return func.__name__, kwargs, None
    except Exception:
        return func.__name__, kwargs, traceback.format_exc()


def _build_jobs():
    """(function, data_key, needs_stats_dir, kwargs) for every figure in
    the script. data_key indexes into _WORKER: 'df' (unfiltered),
    'df_valid' (score-diff-valid filtered), or 'connectivity_df'."""
    jobs = []
    for area_relation in (None, "within", "across"):
        jobs.append((plot_delta_histogram, "df_valid", True, dict(area_relation=area_relation)))
        jobs.append((plot_delta_boxplot, "df_valid", True, dict(area_relation=area_relation)))
        jobs.append((plot_delta_by_area, "df_valid", True, dict(area_relation=area_relation)))
        for direction in ("increase", "decrease"):
            jobs.append((plot_sign_histogram, "df_valid", True,
                         dict(direction=direction, area_relation=area_relation)))
            jobs.append((plot_sign_boxplot, "df_valid", True,
                         dict(direction=direction, area_relation=area_relation)))
            jobs.append((plot_sign_by_area, "df_valid", True,
                         dict(direction=direction, area_relation=area_relation)))

    # Connectivity/composition figures -- NOT related to score change, kept
    # as-is. Use the UNFILTERED df: 'connected' status depends only on the
    # whole-session score, not score_pre/post.
    jobs.append((plot_connected_pairs_by_area, "df", False, {}))
    jobs.append((plot_within_vs_across_area, "df", False, {}))
    jobs.append((plot_connected_pairs_by_mouse, "df", False, {}))
    jobs.append((plot_connectivity_by_area_and_reward_group, "df", True, {}))
    jobs.append((plot_connectivity_by_reward_group, "connectivity_df", True, {}))
    jobs.append((plot_firing_rate_confound, "df", True, {}))
    return jobs


# ============================== MAIN =========================================
def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    stats_dir = SUMMARY_DIR / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_session_results()
    excluded_mice = load_excluded_mice()
    df = _filter_excluded_mice(df, excluded_mice)
    df_valid = _valid_score_diff(df)

    try:
        connectivity_df = load_connectivity_summary()
        connectivity_df = _filter_excluded_mice(connectivity_df, excluded_mice)
    except FileNotFoundError as e:
        warnings.warn(f"TRUE connectivity-rate figures will be skipped: {e}")
        connectivity_df = None

    jobs = _build_jobs()
    print(f"Running {len(jobs)} figure jobs across up to {N_WORKERS} worker processes...")

    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker,
                              initargs=(df, df_valid, connectivity_df, SUMMARY_DIR, stats_dir)) as ex:
        futures = {ex.submit(_run_job, func, data_key, needs_stats, kwargs): (func.__name__, kwargs)
                   for func, data_key, needs_stats, kwargs in jobs}
        for fut in as_completed(futures):
            name, kwargs = futures[fut]
            _, _, err = fut.result()
            if err:
                print(f"FAILED {name}({kwargs}):\n{err}")
            else:
                print(f"done: {name}({kwargs})")

    print(f"Done. Figures + stats saved under {SUMMARY_DIR}")


if __name__ == "__main__":
    main()