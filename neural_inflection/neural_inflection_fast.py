"""
sigmoid_modulation.py
---------------------
Per-neuron sigmoid vs linear vs constant model comparison on trial-by-trial
firing rates (baseline and sensory-evoked, baseline-corrected), for
whisker-active and auditory-active trials separately.

Drift-correlated neurons (p_conservative < 0.05 on motion shift test,
baseline epoch) are excluded before any analysis.

Sigmoid validity constraints
  - t0 within [t[0]+BORDER_MARGIN, t[-1]-BORDER_MARGIN]
  - S-shaped plateau: |k| * (t[-1]-t[0]) > S_SHAPE_MIN_SPAN

Outputs
-------
Per-session CSV, per-mouse figure folders, summary figures (R+ vs R-).

Modes
-----
run_analysis()     — full pipeline
run_figures_only() — reload existing CSVs, regenerate all figures
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')   # non-interactive backend — prevents GUI buffer refs on Windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.special import expit
from scipy.stats import mannwhitneyu, gaussian_kde
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

from allen_utils import get_custom_area_order
from functools import lru_cache

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
    "figure.dpi":       150,
    "axes.linewidth":   1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "lines.linewidth":  1.8,
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_OUT        = Path(r"M:\analysis\Axel_Bisi\combined_results\sigmoid_analysis_new")
SHIFT_TEST_ROOT = Path(r"M:\analysis\Axel_Bisi\combined_results")
BASE_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT = BASE_OUT / "figures"
FIG_OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASELINE_START   = -2.0
BASELINE_END     = -0.005
EVOKED_START     =  0.005
EVOKED_END       =  0.050

AIC_DELTA_THRESH = 2.0
BORDER_MARGIN    = 5
S_SHAPE_MIN_SPAN = 4.0
N_EXAMPLE        = 30
N_JOBS           = 25
N_PERM           = 1000
DRIFT_P_THRESH   = 0.05
LAG_RANGE        = (-60, 60)
HEATMAP_MIN_NEURONS = 30   # per group
HEATMAP_MIN_MICE    = 3    # per group

CATS       = ["constant", "linear", "sigmoid_down", "sigmoid_up"]
CAT_COLORS = ["#aaaaaa", "#f0a500", "#4477aa", "#cc3333"]
RG_COLORS  = {1: "#228B22", 0: "#DC143C"}
RG_LABELS  = {1: "R+", 0: "R-"}

# ---------------------------------------------------------------------------
# Save helper: png + pdf + svg
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Path):
    """Save figure as PNG, PDF and SVG sequentially.
    Explicit canvas clear + gc between formats prevents CPython 3.14 Windows
    BytesIO/numpy buffer finalisation race (BufferError on BytesIO resize)."""
    import gc
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(path.with_suffix(ext), bbox_inches="tight")
        fig.canvas.flush_events()
        gc.collect()


# ---------------------------------------------------------------------------
# Selection criteria text file
# ---------------------------------------------------------------------------

def write_selection_criteria(out_dir: Path = BASE_OUT):
    txt = f"""Sigmoid Modulation Analysis — Selection & Parameter Criteria
=============================================================

DRIFT EXCLUSION
  Neurons with p_conservative < {DRIFT_P_THRESH} on the motion shift test
  (baseline epoch, factor=motion) are excluded before fitting.
  Source: {{mouse_id}}_whisker_0_shift_test_results.csv, merged on
  mouse_id, session_id, imec_id, unit_id.

EPOCH DEFINITIONS
  Baseline  : [{BASELINE_START}, {BASELINE_END}] s relative to stimulus onset
  Evoked    : [{EVOKED_START}, {EVOKED_END}] s relative to stimulus onset
  Evoked is baseline-subtracted (per trial).

TRIAL SELECTION
  Whisker-active : trial_type == 'whisker_trial' AND context != 'passive'
  Auditory-active: trial_type == 'auditory_trial' AND context != 'passive'
  Minimum 10 active trials required per neuron.

THREE COMPETING MODELS (fit by least squares)
  Constant : y = mu                          (1 parameter)
  Linear   : y = a*t + b                     (2 parameters)
  Sigmoid  : y = L / (1+exp(-k*(t-t0))) + b  (4 parameters)
    L  = amplitude (Hz); positive -> increasing, negative -> decreasing
    k  = steepness; bounded to [-5, 5] in optimizer
    t0 = inflection trial (continuous); bounded to [t0_lo, t0_hi] (see below)
    b  = baseline offset (Hz)

MODEL SELECTION — AIC
  AIC = N * log(RSS/N) + 2*K   (Gaussian log-likelihood, N trials, K params)
  Sigmoid wins if ALL of:
    (1) AIC_sigmoid < AIC_constant
    (2) AIC_sigmoid < AIC_linear
    (3) min(AIC_constant, AIC_linear) - AIC_sigmoid > {AIC_DELTA_THRESH}

SIGMOID VALIDITY CONSTRAINTS (beyond AIC)
  Border margin : t0 within [t[0]+{BORDER_MARGIN}, t[-1]-{BORDER_MARGIN}]
  S-shape check : |k| * (t[-1]-t[0]) > {S_SHAPE_MIN_SPAN}
  Convergence   : curve_fit must succeed without exception.

DIRECTION
  sigmoid_up   : L > 0  (firing rate increases with trials)
  sigmoid_down : L < 0  (firing rate decreases with trials)

LAG COMPUTATION (whisker evoked only)
  lag = t0_idx - learning_trial
  Negative lag -> neural change precedes behavioral learning.
  Positive lag -> neural change follows behavioral learning.

SUMMARY STATISTICS
  Mouse level : median lag per mouse per area per reward group.
  Area inclusion (heatmap): >= {HEATMAP_MIN_NEURONS} neurons AND >= {HEATMAP_MIN_MICE} mice per group.
  Area inclusion (pointplot): >= 3 mice per group.
  PERMANOVA: permutation test (N={N_PERM}) on reward_group at mouse level.
  Post-hoc: Mann-Whitney U per area, FDR-BH corrected.
"""
    (out_dir / "selection_criteria.txt").write_text(txt)


# ---------------------------------------------------------------------------
# Drift filter
# ---------------------------------------------------------------------------

def load_shift_test_results(mouse_ids: list[str]) -> pd.DataFrame:
    dfs = []
    for m in mouse_ids:
        p = (SHIFT_TEST_ROOT / m / "whisker_0" / "single_neuron_shift_test" /
             f"{m}_whisker_0_shift_test_results.csv")
        if not p.exists():
            print(f"WARNING: shift test CSV not found for {m}: {p}")
            continue
        try:
            df = pd.read_csv(p)
            df["mouse_id"] = m
            mask = (df["factor"] == "motion") & (df["epoch"] == "baseline")
            dfs.append(df.loc[mask, ["mouse_id", "session_id", "imec_id",
                                     "unit_id", "p_conservative"]])
        except Exception as e:
            print(f"WARNING: could not read shift test for {m}: {e}")
    if not dfs:
        print("WARNING: no shift test results loaded — drift filter not applied.")
        return pd.DataFrame(columns=["mouse_id", "session_id", "imec_id",
                                     "unit_id", "p_conservative"])
    return pd.concat(dfs, ignore_index=True)


def apply_drift_filter(unit_table: pd.DataFrame,
                       shift_df: pd.DataFrame) -> pd.DataFrame:
    if shift_df.empty:
        return unit_table
    drift = shift_df[shift_df["p_conservative"] < DRIFT_P_THRESH][
        ["mouse_id", "session_id", "imec_id", "unit_id"]
    ].drop_duplicates()
    drift["_drift"] = True
    unit_table['imec_id'] = unit_table['electrode_group'].str.slice(4, 5).astype(int)
    merged = unit_table.merge(drift, on=["mouse_id", "session_id", "imec_id", "unit_id"],
                              how="left")
    n_before = len(unit_table)
    filtered = merged[merged["_drift"].isna()].drop(columns=["_drift"])
    print(f"Drift filter: {n_before - len(filtered)} / {n_before} neurons excluded.")
    return filtered


# ---------------------------------------------------------------------------
# Firing rate helpers (vectorized)
# ---------------------------------------------------------------------------

def compute_fr_vectorized(spike_times: np.ndarray, trial_starts: np.ndarray,
                           t_start: float, t_end: float) -> np.ndarray:
    dur    = t_end - t_start
    lo_idx = np.searchsorted(spike_times, trial_starts + t_start, side="left")
    hi_idx = np.searchsorted(spike_times, trial_starts + t_end,   side="left")
    return (hi_idx - lo_idx).astype(float) / dur


def compute_epoch_fr(spike_times: np.ndarray, trial_starts: np.ndarray):
    baseline = compute_fr_vectorized(spike_times, trial_starts, BASELINE_START, BASELINE_END)
    evoked   = compute_fr_vectorized(spike_times, trial_starts, EVOKED_START, EVOKED_END)
    return baseline, evoked - baseline


# ---------------------------------------------------------------------------
# Trial selection
# ---------------------------------------------------------------------------

def get_active_trial_starts(trial_table: pd.DataFrame, trial_type: str):
    mask = (
        (trial_table["trial_type"] == f"{trial_type}_trial") &
        (trial_table["context"]    != "passive")
    )
    return trial_table.loc[mask, "start_time"].values, int(mask.sum())


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def _aic(n: int, rss: float, k: int) -> float:
    if rss <= 0 or n <= k:
        return np.inf
    return n * np.log(rss / n) + 2 * k


def _fit_constant(y: np.ndarray):
    mu  = y.mean()
    rss = float(np.sum((y - mu) ** 2))
    return mu, rss, _aic(len(y), rss, 1)


def _fit_linear(t: np.ndarray, y: np.ndarray):
    tm, ym = t.mean(), y.mean()
    denom  = np.sum((t - tm) ** 2)
    if denom == 0:
        return 0.0, ym, float(np.sum((y - ym) ** 2)), np.inf
    slope = np.sum((t - tm) * (y - ym)) / denom
    ic    = ym - slope * tm
    rss   = float(np.sum((y - slope * t - ic) ** 2))
    return slope, ic, rss, _aic(len(t), rss, 2)


def _sigmoid_fn(t, L, k, t0, b):
    return L * expit(k * (t - t0)) + b


def _fit_sigmoid(t: np.ndarray, y: np.ndarray):
    t_range = float(t[-1] - t[0])
    t_lo    = float(t[0]  + BORDER_MARGIN)
    t_hi    = float(t[-1] - BORDER_MARGIN)
    if t_lo >= t_hi:
        return np.nan, np.nan, np.nan, np.nan, np.inf, np.inf, False
    n  = len(t)
    p0 = [float(np.ptp(y)), 0.1, float(t[n // 2]), float(y.min())]
    bounds = ([-np.inf, -5.0, t_lo, -np.inf],
              [ np.inf,  5.0, t_hi,  np.inf])
    try:
        popt, _ = curve_fit(_sigmoid_fn, t, y, p0=p0, bounds=bounds, maxfev=10_000)
        L, k, t0, b = popt
        yhat     = _sigmoid_fn(t, *popt)
        rss      = float(np.sum((y - yhat) ** 2))
        aic      = _aic(n, rss, 4)
        s_shaped = abs(k) * t_range > S_SHAPE_MIN_SPAN
        return L, k, t0, b, rss, aic, s_shaped
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.inf, np.inf, False


# Column names matching model_comparison tuple return order
MC_FIELDS = (
    "aic_constant","aic_linear","aic_sigmoid",
    "rss_constant","rss_linear","rss_sigmoid",
    "best_model","sigmoid_wins","linear_wins",
    "L","k","t0","b",
    "linear_slope","linear_intercept",
    "t0_idx","direction","linear_direction",
)


def model_comparison(t: np.ndarray, y: np.ndarray) -> tuple:
    """Returns a flat tuple in MC_FIELDS order — avoids per-row dict allocation."""
    mu,    rss_c, aic_c  = _fit_constant(y)
    sl, ic, rss_l, aic_l = _fit_linear(t, y)
    if rss_c == 0:
        L, k, t0, b, rss_s, aic_s, valid = np.nan, np.nan, np.nan, np.nan, np.inf, np.inf, False
    else:
        L, k, t0, b, rss_s, aic_s, valid = _fit_sigmoid(t, y)

    sigmoid_wins = (
        valid and aic_s < aic_c and aic_s < aic_l and
        (aic_s - min(aic_c, aic_l)) < -AIC_DELTA_THRESH
    )
    linear_wins = (not sigmoid_wins) and (aic_l < aic_c) and \
                  ((aic_l - aic_c) < -AIC_DELTA_THRESH)

    if sigmoid_wins:
        t0_idx    = int(np.round(t0))
        direction = 1 if L > 0 else -1
    else:
        t0 = t0_idx = direction = np.nan

    linear_direction = np.nan if np.isnan(sl) else (1 if sl > 0 else -1)

    return (
        aic_c, aic_l, aic_s,
        rss_c, rss_l, rss_s,
        "sigmoid" if sigmoid_wins else ("linear" if linear_wins else "constant"),
        sigmoid_wins, linear_wins,
        L, k, t0, b,
        sl, ic,
        t0_idx, direction, linear_direction,
    )


# ---------------------------------------------------------------------------
# Per-neuron worker
# ---------------------------------------------------------------------------

def _neuron_worker(unit_id, area, mouse_id, session_id, imec_id, probe,
                   reward_group, spike_times,
                   trial_starts_w, n_w, trial_starts_a, n_a):
    """
    Returns (rows, fr_map) where:
      rows   : list of flat tuples, one per (stim, epoch) combination
      fr_map : dict (trial_type, epoch) -> fr array  (for example figures)
    Each tuple layout: MC_FIELDS + meta + (trial_type, epoch, n_trials, mean_fr)
    """
    spk = np.asarray(spike_times)
    if spk.size > 1 and not np.all(spk[:-1] <= spk[1:]):
        spk = np.sort(spk)

    rows   = []
    fr_map = {}
    for trial_type, trial_starts, n_trials in [
        ("whisker",  trial_starts_w, n_w),
        ("auditory", trial_starts_a, n_a),
    ]:
        if n_trials < 10:
            continue
        baseline, evoked_bc = compute_epoch_fr(spk, trial_starts)
        t = np.arange(n_trials, dtype=float)
        for epoch, y in (("baseline", baseline), ("evoked", evoked_bc)):
            mc  = model_comparison(t, y)
            row = mc + (
                unit_id, area, mouse_id, session_id, imec_id, probe, reward_group,
                trial_type, epoch, n_trials, float(y.mean()),
            )
            rows.append(row)
            fr_map[(trial_type, epoch)] = y
    return rows, fr_map


# Column names for the full result DataFrame
_RESULT_COLS = MC_FIELDS + (
    "unit_id","area_acronym_custom","mouse_id","session_id",
    "imec_id","probe","reward_group",
    "trial_type","epoch","n_trials","mean_fr",
)


def analyse_session(unit_table: pd.DataFrame, trial_table: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame directly — built from columnar tuple data.
    fr_series stored only for top-N_EXAMPLE sigmoid neurons per stim/epoch.
    """
    unit_table = unit_table.copy()
    unit_table["presenceRatio"] = unit_table["presenceRatio"].astype(float)
    good = unit_table[
        (unit_table["bc_label"] == "good") &
        (unit_table["presenceRatio"] > 0.8)
    ]
    if good.empty:
        return pd.DataFrame(columns=_RESULT_COLS)

    trial_starts_w, n_w = get_active_trial_starts(trial_table, "whisker")
    trial_starts_a, n_a = get_active_trial_starts(trial_table, "auditory")

    ids    = good["unit_id"].values
    areas  = good["area_acronym_custom"].values
    mids   = good["mouse_id"].values
    sids   = good["session_id"].values
    imecs  = good["imec_id"].values   if "imec_id"      in good.columns else np.full(len(good), np.nan)
    probes = good["probe"].values     if "probe"        in good.columns else np.full(len(good), np.nan)
    rgs    = good["reward_group"].values if "reward_group" in good.columns else np.full(len(good), np.nan)
    spikes = good["spike_times"].values

    worker_args = [
        (ids[i], areas[i], mids[i], sids[i], imecs[i], probes[i], rgs[i],
         spikes[i], trial_starts_w, n_w, trial_starts_a, n_a)
        for i in range(len(good))
    ]

    try:
        outputs = Parallel(n_jobs=N_JOBS, backend="loky", timeout=300)(
            delayed(_neuron_worker)(*args) for args in worker_args
        )
    except Exception as e:
        print(f"WARNING: parallel processing failed ({e}), falling back to sequential.")
        outputs = [_neuron_worker(*args) for args in worker_args]

    # outputs is list of (rows, fr_map) per neuron
    all_rows = []
    fr_store = {}   # (unit_id, trial_type, epoch) -> fr array
    for (rows, fr_map), uid in zip(outputs, ids):
        all_rows.extend(rows)
        for (tt, ep), arr in fr_map.items():
            fr_store[(uid, tt, ep)] = arr

    if not all_rows:
        return pd.DataFrame(columns=_RESULT_COLS)

    # Build DataFrame from columnar data — much faster than list-of-dicts
    df = pd.DataFrame.from_records(all_rows, columns=_RESULT_COLS)

    # Attach fr_series only for sigmoid-winning neurons (for example figures)
    sig_mask = df["sigmoid_wins"].astype(bool)
    df["fr_series"] = None
    if sig_mask.any():
        for idx in df.index[sig_mask]:
            key = (df.at[idx, "unit_id"], df.at[idx, "trial_type"], df.at[idx, "epoch"])
            arr = fr_store.get(key)
            if arr is not None:
                df.at[idx, "fr_series"] = arr.tolist()

    return df


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_results(df: pd.DataFrame, session_id: str) -> pd.DataFrame:
    """df is already a DataFrame from analyse_session."""
    df.drop(columns=["fr_series"], errors="ignore").to_csv(
        BASE_OUT / f"{session_id}_sigmoid_results.csv", index=False)
    return df


def load_all_results() -> pd.DataFrame:
    csvs = sorted(BASE_OUT.glob("*_sigmoid_results.csv"))
    if not csvs:
        raise FileNotFoundError(f"No sigmoid result CSVs found in {BASE_OUT}")
    dfs = []
    for p in csvs:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"WARNING: could not read {p.name}: {e}")
    if not dfs:
        raise RuntimeError("No result files could be loaded.")
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Area ordering helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_allen_order() -> tuple:
    """Cached wrapper — get_custom_area_order() called only once."""
    return tuple(get_custom_area_order())


def _ordered_areas(areas: list[str]) -> list[str]:
    allen   = _get_allen_order()
    area_set = set(areas)
    known   = [a for a in allen if a in area_set]
    unknown = sorted(a for a in areas if a not in set(allen))
    out = known + unknown
    print(out)
    return out


# ---------------------------------------------------------------------------
# Lag computation & shared areas
# ---------------------------------------------------------------------------

def compute_lags(results_df: pd.DataFrame, learning_df: pd.DataFrame,
                  trial_type: str = "whisker", epoch: str = "evoked") -> pd.DataFrame:
    """lag = learning_trial - t0_idx  (positive = neural precedes behaviour)."""
    sub = results_df[
        (results_df["trial_type"]   == trial_type) &
        (results_df["epoch"]        == epoch)       &
        (results_df["sigmoid_wins"] == True)
    ].copy()
    merged = sub.merge(learning_df, on="mouse_id", how="left")
    merged["lag"] = merged["learning_trial"] - merged["t0_idx"]
    merged["lag_trial_type"] = trial_type
    merged["lag_epoch"]      = epoch
    return merged


def get_shared_areas(lag_df: pd.DataFrame, min_mice: int = 3) -> list[str]:
    """Areas with >= min_mice per reward group (for pointplot)."""
    counts = (
        lag_df.groupby(["area_acronym_custom", "reward_group"])["mouse_id"]
        .nunique().unstack(fill_value=0)
    )
    mask = (counts.get(1, pd.Series(0, index=counts.index)) >= min_mice) & \
           (counts.get(0, pd.Series(0, index=counts.index)) >= min_mice)
    return _ordered_areas(counts.index[mask].tolist())


def get_heatmap_areas(lag_df: pd.DataFrame,
                      min_neurons: int = HEATMAP_MIN_NEURONS,
                      min_mice:    int = HEATMAP_MIN_MICE) -> list[str]:
    """Areas with >= min_neurons AND >= min_mice per reward group."""
    neuron_counts = (
        lag_df.groupby(["area_acronym_custom", "reward_group"]).size()
        .unstack(fill_value=0)
    )
    mice_counts = (
        lag_df.groupby(["area_acronym_custom", "reward_group"])["mouse_id"]
        .nunique().unstack(fill_value=0)
    )
    mask = (
        (neuron_counts.get(1, pd.Series(0, index=neuron_counts.index)) >= min_neurons) &
        (neuron_counts.get(0, pd.Series(0, index=neuron_counts.index)) >= min_neurons) &
        (mice_counts.get(1,   pd.Series(0, index=mice_counts.index))   >= min_mice)    &
        (mice_counts.get(0,   pd.Series(0, index=mice_counts.index))   >= min_mice)
    )
    return _ordered_areas(neuron_counts.index[mask].tolist())


# ---------------------------------------------------------------------------
# PERMANOVA
# ---------------------------------------------------------------------------

def _f_stat_vec(x: np.ndarray, mask1: np.ndarray) -> float:
    """F-statistic for two groups defined by boolean mask1."""
    n    = len(x)
    x1   = x[mask1];  x0 = x[~mask1]
    n1, n0 = len(x1), len(x0)
    if n1 < 1 or n0 < 1:
        return np.nan
    grand = x.mean()
    ss_b  = n1 * (x1.mean() - grand) ** 2 + n0 * (x0.mean() - grand) ** 2
    ss_w  = np.sum((x1 - x1.mean()) ** 2) + np.sum((x0 - x0.mean()) ** 2)
    df_w  = n - 2
    if ss_w == 0 or df_w == 0:
        return np.nan
    return (ss_b / 1.0) / (ss_w / df_w)


def _permanova(vals_rp: np.ndarray, vals_rm: np.ndarray,
               n_perm: int = N_PERM) -> float:
    """Vectorised permutation F-test. All permutations computed in one numpy batch."""
    n1, n2 = len(vals_rp), len(vals_rm)
    if n1 < 2 or n2 < 2:
        return np.nan
    combined = np.concatenate([vals_rp, vals_rm])  # (n,)
    n        = len(combined)
    mask_obs = np.array([True] * n1 + [False] * n2)
    obs      = _f_stat_vec(combined, mask_obs)
    if np.isnan(obs):
        return np.nan
    # generate all permuted index matrices at once: (n_perm, n)
    rng   = np.random.default_rng(42)
    perms = np.argsort(rng.random((n_perm, n)), axis=1)  # each row is a permutation
    grand = combined.mean()
    # vectorised F over all permutations
    # group1 size is always n1; group0 size n2
    # for each permuted row, first n1 elements are "group1"
    combined_perms = combined[perms]          # (n_perm, n)
    g1 = combined_perms[:, :n1]              # (n_perm, n1)
    g0 = combined_perms[:, n1:]              # (n_perm, n2)
    grand_p = combined_perms.mean(axis=1, keepdims=True)  # (n_perm, 1)
    ss_b_p  = (n1 * (g1.mean(axis=1) - grand_p[:, 0]) ** 2 +
               n2 * (g0.mean(axis=1) - grand_p[:, 0]) ** 2)
    ss_w_p  = (np.sum((g1 - g1.mean(axis=1, keepdims=True)) ** 2, axis=1) +
               np.sum((g0 - g0.mean(axis=1, keepdims=True)) ** 2, axis=1))
    df_w    = n - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        f_perm = np.where(ss_w_p > 0, ss_b_p / (ss_w_p / df_w), np.nan)
    return float((np.nansum(f_perm >= obs)) + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Classify categories
# ---------------------------------------------------------------------------

def _classify_category(df: pd.DataFrame) -> pd.Series:
    cat = df["best_model"].copy()
    sig = df["sigmoid_wins"]
    cat[sig & (df["direction"] ==  1)] = "sigmoid_up"
    cat[sig & (df["direction"] == -1)] = "sigmoid_down"
    return cat


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def _mouse_fig_dir(mouse_id: str) -> Path:
    d = FIG_OUT / mouse_id
    d.mkdir(exist_ok=True)
    return d


def _example_neuron_dir(mouse_id: str) -> Path:
    d = FIG_OUT / mouse_id / "example_neurons"
    d.mkdir(exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Figures — example neurons
# ---------------------------------------------------------------------------

def _plot_example_neuron(row: pd.Series, out_path: Path):
    y      = np.asarray(row["fr_series"])
    t      = np.arange(len(y), dtype=float)
    t_fine = np.linspace(t[0], t[-1], 300)

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.40)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(t, y, "o", ms=4, color="gray", alpha=0.55, label="FR data", zorder=1)
    ax1.axhline(y.mean(), ls="--", color="steelblue", lw=2, label="Constant")
    sl, ic, *_ = _fit_linear(t, y)
    ax1.plot(t_fine, sl * t_fine + ic, "--", color="darkorange", lw=2, label="Linear")
    ax1.plot(t_fine, _sigmoid_fn(t_fine, row["L"], row["k"], row["t0"], row["b"]),
             "-", color="crimson", lw=2.5, label="Sigmoid", zorder=3)
    ax1.axvline(row["t0"], color="crimson", ls=":", lw=1.5, label=f"t0 = {row['t0']:.1f}")
    ax1.set_xlabel("Active trial index")
    ax1.set_ylabel("Firing rate (Hz)")
    ax1.set_title(f"{row['unit_id']}  |  {row['area_acronym_custom']}  |  "
                  f"{row['trial_type']} {row['epoch']}")
    ax1.legend(fontsize=10, framealpha=0.8)

    ax2    = fig.add_subplot(gs[0, 2])
    models = ["constant", "linear", "sigmoid"]
    aics   = [row[f"aic_{m}"] for m in models]
    bars   = ax2.bar(models, aics, color=["steelblue", "darkorange", "crimson"], alpha=0.8)
    min_a  = min(aics)
    for bar, aic in zip(bars, aics):
        ax2.text(bar.get_x() + bar.get_width() / 2, aic + abs(min_a) * 0.01,
                 f"{aic - min_a:+.1f}", ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel("AIC"); ax2.set_title("Model comparison (AIC)")

    ax3 = fig.add_subplot(gs[1, 0])
    resid = y - _sigmoid_fn(t, row["L"], row["k"], row["t0"], row["b"])
    ax3.plot(t, resid, "o", ms=4, color="gray", alpha=0.55)
    ax3.axhline(0, color="k", lw=1.2)
    ax3.set_xlabel("Active trial index"); ax3.set_ylabel("Residual (Hz)")
    ax3.set_title("Sigmoid residuals")

    ax4 = fig.add_subplot(gs[1, 1]); ax4.axis("off")
    ax4.text(0.05, 0.95, (
        f"L         = {row['L']:+.3f} Hz\n"
        f"k         = {row['k']:+.4f}\n"
        f"t0        = {row['t0']:.2f} trials\n"
        f"b         = {row['b']:+.3f} Hz\n"
        f"t0_idx    = {row['t0_idx']}\n"
        f"direction = {'↑ up' if row['direction']==1 else '↓ down'}\n"
        f"|k|×range = {abs(row['k'])*(len(y)-1):.2f}  (>{S_SHAPE_MIN_SPAN})\n"
        f"ΔAIC(s−l) = {row['aic_sigmoid']-row['aic_linear']:.2f}\n"
        f"ΔAIC(s−c) = {row['aic_sigmoid']-row['aic_constant']:.2f}"
    ), transform=ax4.transAxes, va="top", fontsize=10, family="monospace")
    ax4.set_title("Parameters")

    ax5 = fig.add_subplot(gs[1, 2])
    deltas = [row["aic_sigmoid"] - row["aic_linear"],
              row["aic_sigmoid"] - row["aic_constant"]]
    ax5.bar(["vs Linear", "vs Constant"], deltas,
            color=["darkorange", "steelblue"], alpha=0.8)
    ax5.axhline(-AIC_DELTA_THRESH, color="k", ls="--", lw=1.5,
                label=f"threshold = −{AIC_DELTA_THRESH}")
    ax5.set_ylabel("ΔAIC (sigmoid − other)"); ax5.set_title("Sigmoid advantage")
    ax5.legend(fontsize=10)

    import gc
    _savefig(fig, out_path)
    plt.close(fig)
    del fig
    gc.collect()


def plot_example_neurons(results_df: pd.DataFrame, n: int = N_EXAMPLE,
                          tag: str = "", out_dir: Path = FIG_OUT):
    if "fr_series" not in results_df.columns:
        return
    sig = results_df[results_df["sigmoid_wins"]].copy()
    if sig.empty:
        return
    examples = sig.loc[sig["L"].abs().nlargest(n).index]
    for _, row in examples.iterrows():
        uid = str(row["unit_id"]).replace("/", "_")
        _plot_example_neuron(row, out_dir / f"example_{tag}_{uid}.pdf")


# ---------------------------------------------------------------------------
# Figures — fraction modulated (3×3 grid)
# ---------------------------------------------------------------------------

def plot_fraction_modulated(results_df: pd.DataFrame, trial_type: str, epoch: str,
                             out_dir: Path = FIG_OUT, title_prefix: str = ""):
    sub = results_df[
        (results_df["trial_type"] == trial_type) &
        (results_df["epoch"]      == epoch)
    ].copy()
    if sub.empty:
        return

    sub["category"] = _classify_category(sub)
    areas = _ordered_areas(sorted(sub["area_acronym_custom"].dropna().unique()))
    if not areas:
        return

    x  = np.arange(len(areas))
    bw = 0.6
    groups = [("All", sub),
              (RG_LABELS[1], sub[sub["reward_group"] == 1]),
              (RG_LABELS[0], sub[sub["reward_group"] == 0])]

    fig, axes = plt.subplots(3, 3, figsize=(max(10, len(areas) * 1.1), 14), sharex=True)
    fig.suptitle(f"{title_prefix}Modulation fractions — {trial_type} {epoch}",
                 fontsize=15, y=1.01)

    for col, (grp_label, gdf) in enumerate(groups):
        if gdf.empty:
            for row in range(3):
                axes[row, col].set_visible(False)
            continue

        # row 0: constant / linear / sigmoid fractions
        ax = axes[0, col]
        ct = pd.crosstab([gdf["mouse_id"], gdf["area_acronym_custom"]], gdf["category"])
        ct = ct.reindex(columns=CATS, fill_value=0)
        ct = ct.div(ct.sum(axis=1), axis=0)
        mdf = ct.groupby(level="area_acronym_custom").mean().reindex(
            index=areas, columns=CATS, fill_value=0)
        bottom = np.zeros(len(areas))
        for cat, color in zip(CATS, CAT_COLORS):
            vals = mdf[cat].values
            ax.bar(x, vals, bottom=bottom, color=color, label=cat, alpha=0.85, width=bw)
            bottom += vals
        ax.set_ylim(0, 1); ax.set_ylabel("Fraction of neurons")
        ax.set_title(f"{grp_label} — model type")
        if col == 0:
            ax.legend(loc="upper right", fontsize=9)

        # row 1: sigmoid up vs down within sigmoid neurons
        ax = axes[1, col]
        sig_only = gdf[gdf["sigmoid_wins"]].copy()
        if not sig_only.empty:
            ct2 = pd.crosstab([sig_only["mouse_id"], sig_only["area_acronym_custom"]],
                              sig_only["direction"].map({1: "up", -1: "down"}))
            ct2 = ct2.reindex(columns=["up", "down"], fill_value=0)
            ct2 = ct2.div(ct2.sum(axis=1), axis=0)
            mdf2 = ct2.groupby(level="area_acronym_custom").mean().reindex(index=areas, fill_value=0)
            ups  = mdf2.get("up",   pd.Series(0, index=areas)).values
            downs = mdf2.get("down", pd.Series(0, index=areas)).values
            ax.bar(x, ups,   color="#cc3333", alpha=0.85, width=bw, label="↑ up")
            ax.bar(x, downs, bottom=ups, color="#4477aa", alpha=0.85, width=bw, label="↓ down")
        ax.set_ylim(0, 1); ax.set_ylabel("Fraction of sigmoid neurons")
        ax.set_title(f"{grp_label} — sigmoid direction")
        if col == 0:
            ax.legend(loc="upper right", fontsize=9)

        # row 2: linear up vs down within linear neurons
        ax = axes[2, col]
        lin_only = gdf[gdf["linear_wins"]].copy()
        if not lin_only.empty:
            ct3 = pd.crosstab([lin_only["mouse_id"], lin_only["area_acronym_custom"]],
                              lin_only["linear_direction"].map({1: "up", -1: "down",
                                                                np.nan: "unknown"}))
            ct3 = ct3.reindex(columns=["up", "down"], fill_value=0)
            ct3 = ct3.div(ct3.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
            mdf3 = ct3.groupby(level="area_acronym_custom").mean().reindex(index=areas, fill_value=0)
            ups  = mdf3.get("up",   pd.Series(0, index=areas)).values
            downs = mdf3.get("down", pd.Series(0, index=areas)).values
            ax.bar(x, ups,   color="#cc3333", alpha=0.85, width=bw, label="↑ positive slope")
            ax.bar(x, downs, bottom=ups, color="#4477aa", alpha=0.85, width=bw, label="↓ negative slope")
        ax.set_ylim(0, 1); ax.set_ylabel("Fraction of linear neurons")
        ax.set_title(f"{grp_label} — linear direction")
        if col == 0:
            ax.legend(loc="upper right", fontsize=9)

    for col in range(3):
        axes[2, col].set_xticks(x)
        axes[2, col].set_xticklabels(areas, rotation=45, ha="right")

    fig.tight_layout()
    _savefig(fig, out_dir / f"fraction_modulated_{trial_type}_{epoch}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures — pie chart (5 categories, 2×2 grid)
# ---------------------------------------------------------------------------

def plot_pie_chart(results_df: pd.DataFrame, out_dir: Path = FIG_OUT):
    """
    2×2 grid: whisker baseline / whisker evoked / auditory baseline / auditory evoked.
    5 categories: constant / linear_up / linear_down / sigmoid_up / sigmoid_down.
    """
    pie_cats   = ["constant", "linear_up", "linear_down", "sigmoid_up", "sigmoid_down"]
    pie_colors = ["#aaaaaa", "#f0a500", "#f5c842", "#cc3333", "#4477aa"]
    pie_labels = ["Constant", "Linear ↑", "Linear ↓", "Sigmoid ↑", "Sigmoid ↓"]

    def _get_cat5(df):
        cat = pd.Series("constant", index=df.index)
        cat[df["sigmoid_wins"] & (df["direction"] ==  1)] = "sigmoid_up"
        cat[df["sigmoid_wins"] & (df["direction"] == -1)] = "sigmoid_down"
        lw = df["linear_wins"]
        cat[lw & (df["linear_direction"] ==  1)] = "linear_up"
        cat[lw & (df["linear_direction"] == -1)] = "linear_down"
        return cat

    combos = [("whisker",  "baseline"), ("whisker",  "evoked"),
              ("auditory", "baseline"), ("auditory", "evoked")]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for ax, (tt, ep) in zip(axes, combos):
        sub = results_df[
            (results_df["trial_type"] == tt) &
            (results_df["epoch"]      == ep)
        ].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sub["cat5"] = _get_cat5(sub)
        counts = sub["cat5"].value_counts().reindex(pie_cats, fill_value=0)
        total  = counts.sum()
        pct    = counts / total * 100
        wedge_labels = [f"{l}\n{p:.1f}%" for l, p in zip(pie_labels, pct)]
        ax.pie(counts.values, labels=wedge_labels, colors=pie_colors,
               autopct=None, startangle=90,
               wedgeprops=dict(linewidth=1.2, edgecolor="white"))
        ax.set_title(f"{tt} — {ep}\n(n = {total:,})", fontsize=13)

    fig.suptitle("Model category distribution (whole dataset)", fontsize=15, y=1.01)
    fig.tight_layout()
    _savefig(fig, out_dir / "pie_chart_model_categories.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures — slope consistency
# ---------------------------------------------------------------------------

def plot_slope_consistency(results_df: pd.DataFrame, trial_type: str, epoch: str,
                            out_dir: Path = FIG_OUT, title_prefix: str = ""):
    sub = results_df[
        (results_df["trial_type"]   == trial_type) &
        (results_df["epoch"]        == epoch)      &
        (results_df["sigmoid_wins"] == True)
    ].dropna(subset=["k", "linear_slope"]).copy()
    if sub.empty:
        return

    sub["sign_match"] = np.sign(sub["k"]) == np.sign(sub["linear_slope"])

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    groups = [("All", sub),
              (RG_LABELS[1], sub[sub["reward_group"] == 1]),
              (RG_LABELS[0], sub[sub["reward_group"] == 0])]

    for ax, (label, gdf) in zip(axes, groups):
        if gdf.empty:
            ax.set_visible(False); continue
        pct_match    = 100 * gdf["sign_match"].mean()
        pct_mismatch = 100 - pct_match
        ax.bar(["Match", "Mismatch"], [pct_match, pct_mismatch],
               color=["#4dac26", "#d01c8b"], alpha=0.85, width=0.5)
        ax.set_ylim(0, 100); ax.set_ylabel("% of sigmoid neurons")
        ax.set_title(f"{label}  (n = {len(gdf)})")
        ax.text(0, pct_match    + 1, f"{pct_match:.1f}%",    ha="center", va="bottom", fontsize=11)
        ax.text(1, pct_mismatch + 1, f"{pct_mismatch:.1f}%", ha="center", va="bottom", fontsize=11)

    fig.suptitle(f"{title_prefix}Slope consistency — {trial_type} {epoch}", fontsize=14)
    fig.tight_layout()
    _savefig(fig, out_dir / f"slope_consistency_{trial_type}_{epoch}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures — lag violin per area
# ---------------------------------------------------------------------------

def _violin_lag_panel(ax, rp_vals, rm_vals, area):
    for pos, vals, color in [(0, rp_vals, RG_COLORS[1]),
                              (1, rm_vals, RG_COLORS[0])]:
        if len(vals) == 0:
            continue
        parts = ax.violinplot([vals], positions=[pos], showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color); pc.set_alpha(0.5)
        ax.scatter(np.random.normal(pos, 0.05, size=len(vals)),
                   vals, color=color, s=20, alpha=0.65, zorder=3)
        ax.axhline(np.median(vals), color=color, ls="--", lw=1.5)
    ax.axhline(0, color="k", lw=1.2, ls=":")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["R+", "R-"])
    ax.set_xlabel("Reward group"); ax.set_title(area)


def plot_lag_distribution(lag_df: pd.DataFrame, direction: int,
                           areas: list[str], tag: str = "",
                           out_dir: Path = FIG_OUT):
    areas     = _ordered_areas(areas)   # enforce Allen order defensively
    dir_label = "increased" if direction == 1 else "decreased"
    sub = lag_df[
        (lag_df["direction"] == direction) &
        (lag_df["area_acronym_custom"].isin(areas))
    ].dropna(subset=["lag"])
    if sub.empty or not areas:
        return

    mouse_df = (sub.groupby(["mouse_id", "area_acronym_custom", "reward_group"])
                ["lag"].median().reset_index())

    for level, df in [("mouse", mouse_df), ("neuron", sub)]:
        n_areas = len(areas)
        # square subplots
        sq = max(4.0, sub["lag"].std() * 0.3)
        fig, axes = plt.subplots(1, n_areas,
                                 figsize=(3.8 * n_areas, 5), sharey=True)
        axes = [axes] if n_areas == 1 else list(axes)
        y_all = np.concatenate([
            df[df["area_acronym_custom"] == a]["lag"].dropna().values for a in areas
        ])
        ylim = (np.percentile(y_all, 2), np.percentile(y_all, 98)) if len(y_all) else (-30, 30)

        pvals = {}
        for ax, area in zip(axes, areas):
            rp = df[(df["area_acronym_custom"] == area) &
                    (df["reward_group"] == 1)]["lag"].dropna().values
            rm = df[(df["area_acronym_custom"] == area) &
                    (df["reward_group"] == 0)]["lag"].dropna().values
            _violin_lag_panel(ax, rp, rm, area)
            ax.set_ylim(ylim)
            ax.set_ylabel("Lag (behavioral − neural, trials)" if ax is axes[0] else "")
            if len(rp) >= 3 and len(rm) >= 3:
                _, p = mannwhitneyu(rp, rm, alternative="two-sided")
                pvals[area] = p

        areas_tested = list(pvals.keys())
        if areas_tested:
            _, pvals_corr, _, _ = multipletests(
                [pvals[a] for a in areas_tested], method="fdr_bh")
            for area, pc in zip(areas_tested, pvals_corr):
                axes[areas.index(area)].set_title(f"{area}\np_fdr = {pc:.3f}")

        fig.suptitle(f"Lag — {dir_label} ({level} level) {tag}", y=1.02)
        fig.tight_layout()
        _savefig(fig, out_dir / f"lag_{dir_label}_{level}_{tag}.png")
        plt.close(fig)


def plot_lag_difference_distribution(lag_df: pd.DataFrame, direction: int,
                                      tag: str = "", out_dir: Path = FIG_OUT):
    dir_label = "increased" if direction == 1 else "decreased"
    sub = lag_df[lag_df["direction"] == direction].dropna(subset=["lag"])
    if sub.empty:
        return
    mouse_lags = sub.groupby(["mouse_id", "reward_group"])["lag"].median().reset_index()
    xlim = (sub["lag"].quantile(0.01), sub["lag"].quantile(0.99))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for col, (data_all, mouse_data, label) in enumerate([
        (sub["lag"].values,
         sub.groupby("mouse_id")["lag"].median().values,
         "Neuron level"),
        (None, None, "Mouse level"),  # placeholder, handled per-group below
    ]):
        # ── row 0: single grey histogram (pooled) ──
        ax = axes[0, col]
        if col == 0:
            data = sub["lag"].values
        else:
            data = sub.groupby("mouse_id")["lag"].median().values
        ax.hist(data, bins=30, color="#888888", alpha=0.60, density=True)
        ax.axvline(data.mean(),     color="crimson",   lw=2, ls="--",
                   label=f"Mean = {data.mean():.1f}")
        ax.axvline(np.median(data), color="steelblue", lw=2, ls="-",
                   label=f"Median = {np.median(data):.1f}")
        ax.axvline(0, color="k", lw=1.2, ls=":")
        ax.set_xlim(xlim)
        ax.set_xlabel("Lag (behavioral − neural, trials)")
        ax.set_ylabel("Density")
        ax.set_title(f"{dir_label} — {'Neuron' if col==0 else 'Mouse'} level  (n = {len(data)})")
        ax.legend(fontsize=11)

        # ── row 1: overlapping histograms per reward group ──
        ax2 = axes[1, col]
        for rg in [1, 0]:
            if col == 0:
                vals = sub[sub["reward_group"] == rg]["lag"].dropna().values
            else:
                vals = mouse_lags[mouse_lags["reward_group"] == rg]["lag"].dropna().values
            if len(vals) == 0:
                continue
            color = RG_COLORS[rg]
            ax2.hist(vals, bins=25, color=color, alpha=0.45, density=True,
                     label=f"{RG_LABELS[rg]} (n={len(vals)})")
            try:
                xs  = np.linspace(xlim[0], xlim[1], 200)
                kde = gaussian_kde(vals, bw_method="scott")
                ax2.plot(xs, kde(xs), color=color, lw=2)
                ax2.axvline(np.median(vals), color=color, lw=1.5, ls="--")
            except Exception:
                pass
        ax2.axvline(0, color="k", lw=1.2, ls=":")
        ax2.set_xlim(xlim)
        ax2.set_xlabel("Lag (behavioral − neural, trials)")
        ax2.set_ylabel("Density")
        ax2.set_title(f"{dir_label} — {'Neuron' if col==0 else 'Mouse'} level by group")
        ax2.legend(fontsize=10)

    fig.suptitle(f"Lag distribution — {dir_label} neurons  {tag}")
    fig.tight_layout()
    _savefig(fig, out_dir / f"lag_distribution_{dir_label}_{tag}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Lag summary — split by reward group, square subplots
# ---------------------------------------------------------------------------

def plot_lag_distribution_by_group(lag_df: pd.DataFrame, shared_areas: list[str],
                                    out_dir: Path = FIG_OUT):
    """
    One figure per direction (increased / decreased).
    Columns = R+ / R-, rows = areas (Allen-ordered).
    Each subplot is square; shows histogram + KDE + median/mean lines.
    """
    ordered = _ordered_areas(shared_areas)
    if not ordered:
        return

    xlim_all = (lag_df["lag"].quantile(0.01), lag_df["lag"].quantile(0.99))

    for direction in [1, -1]:
        dir_label = "increased" if direction == 1 else "decreased"
        sub = lag_df[
            (lag_df["direction"] == direction) &
            (lag_df["area_acronym_custom"].isin(ordered))
        ].dropna(subset=["lag"])
        if sub.empty:
            continue

        n_areas = len(ordered)
        cell_sz = 3.2
        fig, axes = plt.subplots(n_areas, 2,
                                  figsize=(cell_sz * 2, cell_sz * n_areas),
                                  sharex=True, sharey=False)
        if n_areas == 1:
            axes = axes[np.newaxis, :]

        for row_i, area in enumerate(ordered):
            for col_i, rg in enumerate([1, 0]):
                ax   = axes[row_i, col_i]
                vals = sub[(sub["area_acronym_custom"] == area) &
                           (sub["reward_group"] == rg)]["lag"].dropna().values
                color = RG_COLORS[rg]

                if len(vals) >= 2:
                    ax.hist(vals, bins=20, density=True, color=color,
                            alpha=0.45, edgecolor="none")
                    try:
                        kde  = gaussian_kde(vals, bw_method="scott")
                        xs   = np.linspace(xlim_all[0], xlim_all[1], 200)
                        ax.plot(xs, kde(xs), color=color, lw=2)
                    except Exception:
                        pass
                    ax.axvline(np.median(vals), color=color, lw=1.8, ls="-",
                               label=f"Median={np.median(vals):.1f}")
                    ax.axvline(vals.mean(),     color=color, lw=1.8, ls="--",
                               label=f"Mean={vals.mean():.1f}")
                    ax.legend(fontsize=8, handlelength=1.2)

                ax.axvline(0, color="k", lw=1.0, ls=":")
                ax.set_xlim(xlim_all)
                # square aspect via set_box_aspect
                ax.set_box_aspect(1)
                ax.set_ylabel("Density" if col_i == 0 else "")
                if row_i == 0:
                    ax.set_title(RG_LABELS[rg], fontsize=12)
                if row_i == n_areas - 1:
                    ax.set_xlabel("Lag (behavioral − neural, trials)")
                # area label on y-axis left panel
                if col_i == 0:
                    ax.set_ylabel(area, fontsize=11)

        fig.suptitle(f"Lag by area & reward group — {dir_label} neurons",
                     fontsize=14, y=1.01)
        fig.tight_layout()
        _savefig(fig, out_dir / f"lag_by_group_{dir_label}.png")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Lag summary pointplot (Allen order + lag-difference order)
# ---------------------------------------------------------------------------

def _build_pointplot(lag_df, ordered, direction, xlim, ax, annotate_sig=True):
    """Draw pointplot onto ax, return dict of pvals per area."""
    dir_label = "Increased" if direction == 1 else "Decreased"
    sub = lag_df[
        (lag_df["direction"] == direction) &
        (lag_df["area_acronym_custom"].isin(ordered))
    ].dropna(subset=["lag"])

    mouse_med = (sub.groupby(["mouse_id", "area_acronym_custom", "reward_group"])
                 ["lag"].median().reset_index())


    rp_all = mouse_med[mouse_med["reward_group"] == 1]["lag"].dropna().values
    rm_all = mouse_med[mouse_med["reward_group"] == 0]["lag"].dropna().values
    p_omni = _permanova(rp_all, rm_all)

    pvals = {}
    for i, area in enumerate(ordered):
        for jitter, rg in [(-0.18, 1), (0.18, 0)]:
            vals  = mouse_med[(mouse_med["area_acronym_custom"] == area) &
                              (mouse_med["reward_group"] == rg)]["lag"].dropna().values
            if len(vals) == 0:
                continue
            color  = RG_COLORS[rg]
            mean_v = vals.mean()
            sem_v  = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            ax.scatter(vals, np.full(len(vals), i + jitter),
                       color=color, s=28, alpha=0.3, zorder=3)
            ax.errorbar(mean_v, i + jitter, xerr=sem_v, fmt="o",
                        color=color, ms=12, lw=2, capsize=4, zorder=4)

        rp_v = mouse_med[(mouse_med["area_acronym_custom"] == area) &
                          (mouse_med["reward_group"] == 1)]["lag"].dropna().values
        rm_v = mouse_med[(mouse_med["area_acronym_custom"] == area) &
                          (mouse_med["reward_group"] == 0)]["lag"].dropna().values
        if len(rp_v) >= 3 and len(rm_v) >= 3:
            _, p = mannwhitneyu(rp_v, rm_v, alternative="two-sided")
            pvals[area] = p

    areas_tested = list(pvals.keys())
    sig_areas = set()
    if areas_tested and annotate_sig and (not np.isnan(p_omni)) and p_omni < 0.05:
        _, pvals_corr, _, _ = multipletests(
            [pvals[a] for a in areas_tested], method="fdr_bh")
        for area, pc in zip(areas_tested, pvals_corr):
            if pc < 0.05:
                print('sig-posthoc')
                sig_areas.add(area)
                i = ordered.index(area)
                ax.text(xlim[1] + (xlim[1] - xlim[0]) * 0.02, i,
                        f"* p={pc:.3f}", va="center", fontsize=10,
                        color="k", clip_on=False)


    ax.axvline(0, color="k", lw=1.2, ls=":")
    ax.set_xlim(xlim)
    ax.set_yticks(range(len(ordered)))
    # bold y-tick labels for significant areas
    print('sig areas', sig_areas)
    ax.set_yticklabels(
        [f"* {a}" if a in sig_areas else a for a in ordered],
        fontsize=11
    )
    ax.set_xlabel("Lag (behavioral − neural, trials)")
    p_txt = f"PERMANOVA p = {p_omni:.3f}" if not np.isnan(p_omni) else ""
    sig_note = "  (post-hoc gated on PERMANOVA)" if (not np.isnan(p_omni)) and p_omni < 0.05 else "  (PERMANOVA n.s. — post-hoc not shown)"
    ax.set_title(f"{dir_label}\n{p_txt}{sig_note}")
    ax.invert_yaxis()
    return pvals, mouse_med


def plot_lag_summary_pointplot(lag_df: pd.DataFrame, shared_areas: list[str],
                                out_dir: Path = FIG_OUT, tag: str = ""):
    """
    Two figures:
      1. Allen order  — increased | decreased
      2. Lag-diff order (|median_R+ - median_R-|, computed on increased,
         applied to decreased)
    """
    if not shared_areas:
        return
    lag_vals = lag_df[lag_df["area_acronym_custom"].isin(shared_areas)]["lag"].dropna()
    if lag_vals.empty:
        return
    xlim = (lag_vals.quantile(0.02), lag_vals.quantile(0.98))

    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=RG_COLORS[rg], ms=9, label=RG_LABELS[rg])
               for rg in [1, 0]]

    # --- figure 1: Allen order ---
    allen_ordered = _ordered_areas(shared_areas)
    allen_ordered = allen_ordered[::-1]
    fig, axes = plt.subplots(1, 2,
                              figsize=(14, max(5, len(allen_ordered) * 0.60 + 1.5)),
                              sharey=True)
    pvals_inc, mmed_inc = _build_pointplot(lag_df, allen_ordered, 1, xlim, axes[0])
    pvals_inc, mmed_inc = _build_pointplot(lag_df, allen_ordered, -1, xlim, axes[1])

    axes[1].legend(handles=handles, loc="lower right", fontsize=11)
    fig.suptitle("Lag by area — mouse-median ± SEM  (Allen order)", fontsize=15, y=1.01)
    fig.tight_layout()
    _savefig(fig, out_dir / f"lag_summary_pointplot_allen_{tag}.png")
    plt.close(fig)

    # --- figure 2: lag-difference order (compute on increased, apply to both) ---
    # median per area per group, then |diff|
    sub_inc = lag_df[
        (lag_df["direction"] == 1) &
        (lag_df["area_acronym_custom"].isin(shared_areas))
        ].dropna(subset=["lag"])

    # Mouse medians (same quantity as in the plot)
    mouse_med = (
        sub_inc
        .groupby(["mouse_id", "area_acronym_custom", "reward_group"])["lag"]
        .median()
        .reset_index()
    )

    # Median across mice
    med_inc = (
        mouse_med
        .groupby(["area_acronym_custom", "reward_group"])["lag"]
        .median()
        .unstack()
    )

    med_inc["abs_diff"] = (med_inc[1] - med_inc[0]).abs()
    diff_ordered = med_inc.sort_values("abs_diff", ascending=False).index.tolist()
    diff_ordered = [a for a in diff_ordered if a in shared_areas]  # safety
    print(diff_ordered)

    fig, axes = plt.subplots(1, 2,
                              figsize=(14, max(5, len(diff_ordered) * 0.60 + 1.5)),
                              sharey=True)
    _build_pointplot(lag_df, diff_ordered, 1, xlim, axes[0])
    _build_pointplot(lag_df, diff_ordered, -1, xlim, axes[1])
    axes[1].legend(handles=handles, loc="lower right", fontsize=11)
    fig.suptitle("Lag by area — mouse-median ± SEM  (ordered by |lag diff| R+−R-)",
                 fontsize=15, y=1.01)
    fig.tight_layout()
    _savefig(fig, out_dir / f"lag_summary_pointplot_lagdiff_order_{tag}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Lag summary heatmap + KDE stacked lines
# ---------------------------------------------------------------------------

def _compute_heatmap_matrix(sub, ordered, rg, bin_ctr, lag_range):
    """Returns (mat, peak_lags) where mat shape = (n_areas, n_bins), row-normalized."""
    n_bins = len(bin_ctr)
    mat       = np.zeros((len(ordered), n_bins))
    peak_lags = [np.nan] * len(ordered)
    for i, area in enumerate(ordered):
        vals = sub[(sub["area_acronym_custom"] == area) &
                   (sub["reward_group"] == rg)]["lag"].dropna().values
        vals_clip = vals[(vals >= lag_range[0]) & (vals <= lag_range[1])]
        if len(vals_clip) < 2:
            continue
        try:
            kde = gaussian_kde(vals_clip, bw_method="scott")
            row = kde(bin_ctr)
            if row.max() > 0:
                mat[i]       = row / row.max()
                peak_lags[i] = float(bin_ctr[np.argmax(row)])
        except Exception:
            pass
    return mat, peak_lags


def plot_lag_summary_heatmap(lag_df: pd.DataFrame, shared_areas: list[str],
                              out_dir: Path = FIG_OUT,
                              lag_range: tuple = LAG_RANGE, n_bins: int = 120):
    """
    For each direction (increased / decreased):
      Figure A — heatmap: 4 panels (R+ heatmap | R+ marginal | R- heatmap | R- marginal)
        Areas ordered by R+ KDE peak lag; same order applied to R-.
      Figure B — KDE stacked lines: 2 panels (R+ | R-), one KDE line per area, y-offset.
    """
    hmap_areas = get_heatmap_areas(lag_df)
    if not hmap_areas:
        print("WARNING: no areas meet heatmap inclusion criteria.")
        return

    bins    = np.linspace(lag_range[0], lag_range[1], n_bins + 1)
    bin_ctr = 0.5 * (bins[:-1] + bins[1:])

    for direction in [1, -1]:
        dir_label = "increased" if direction == 1 else "decreased"
        sub = lag_df[
            (lag_df["direction"] == direction) &
            (lag_df["area_acronym_custom"].isin(hmap_areas))
        ].dropna(subset=["lag"])
        if sub.empty:
            continue

        # Compute R+ matrix first to get peak-lag order
        mat_rp, peaks_rp = _compute_heatmap_matrix(sub, hmap_areas, 1, bin_ctr, lag_range)
        # Sort areas by R+ peak lag
        sort_idx  = np.argsort([p if not np.isnan(p) else np.inf for p in peaks_rp])
        peak_ordered = [hmap_areas[i] for i in sort_idx]

        # Recompute matrices in peak order
        mat_rp, peaks_rp = _compute_heatmap_matrix(sub, peak_ordered, 1, bin_ctr, lag_range)
        mat_rm, peaks_rm = _compute_heatmap_matrix(sub, peak_ordered, 0, bin_ctr, lag_range)

        y_extent = [len(peak_ordered) - 0.5, -0.5]

        # ---- Figure A: heatmap + marginal ----
        fig_h = plt.figure(figsize=(20, max(5, len(peak_ordered) * 0.60 + 2)))
        gs_h  = gridspec.GridSpec(1, 4, figure=fig_h, wspace=0.30,
                                  width_ratios=[3, 1, 3, 1])
        hmap_axes = [fig_h.add_subplot(gs_h[0, 0]), fig_h.add_subplot(gs_h[0, 2])]
        marg_axes = [fig_h.add_subplot(gs_h[0, 1]), fig_h.add_subplot(gs_h[0, 3])]

        for hax, max_, rg, mat, peaks in [
            (hmap_axes[0], marg_axes[0], 1, mat_rp, peaks_rp),
            (hmap_axes[1], marg_axes[1], 0, mat_rm, peaks_rm),
        ]:
            im = hax.imshow(mat, aspect="auto", origin="upper",
                            extent=[lag_range[0], lag_range[1]] + y_extent,
                            cmap="plasma", vmin=0, vmax=1)
            hax.axvline(0, color="w", lw=1.5, ls="--")
            # mark peak per area
            for i, pk in enumerate(peaks):
                if not np.isnan(pk):
                    hax.plot(pk, i, "k0", ms=5, zorder=5)
            hax.set_yticks(range(len(peak_ordered)))
            hax.set_yticklabels(peak_ordered, fontsize=10)
            hax.set_xlabel("Lag (behavioral − neural, trials)"); hax.set_xlim(lag_range)
            hax.set_title(f"{RG_LABELS[rg]} — density per area\n(ordered by R+ peak lag)")
            cb = plt.colorbar(im, ax=hax, label="Norm. density", shrink=0.5, pad=0.02)
            cb.ax.tick_params(labelsize=9)

            # marginal
            all_vals = sub[sub["reward_group"] == rg]["lag"].dropna().values
            all_clip = all_vals[(all_vals >= lag_range[0]) & (all_vals <= lag_range[1])]
            if len(all_clip) >= 2:
                max_.hist(all_clip, bins=30, density=True, color=RG_COLORS[rg],
                          alpha=0.45, orientation="horizontal")
                try:
                    kde_all  = gaussian_kde(all_clip, bw_method="scott")
                    kde_vals = kde_all(bin_ctr)
                    max_.plot(kde_vals, bin_ctr, color=RG_COLORS[rg], lw=2)
                    peak_lag = bin_ctr[np.argmax(kde_vals)]
                    max_.axhline(peak_lag, color="k", lw=1.5, ls="--",
                                 label=f"Peak = {peak_lag:.1f}")
                    max_.axhline(0, color="k", lw=1.0, ls=":")
                    max_.legend(fontsize=10)
                except Exception:
                    pass
            max_.set_ylim(lag_range); max_.set_xlabel("Density")
            max_.set_title(f"{RG_LABELS[rg]} — pooled"); max_.set_yticks([])

        fig_h.suptitle(f"Lag density heatmap — {dir_label} neurons", fontsize=15, y=1.02)
        _savefig(fig_h, out_dir / f"lag_summary_heatmap_{dir_label}.png")
        plt.close(fig_h)

        # ---- Figure B: KDE stacked lines ----
        fig_k, axes_k = plt.subplots(1, 2,
                                      figsize=(14, max(5, len(peak_ordered) * 0.55 + 2)),
                                      sharey=True)
        #cmap_areas = plt.cm.get_cmap("tab20", len(peak_ordered))

        for ax, rg, mat in [(axes_k[0], 1, mat_rp), (axes_k[1], 0, mat_rm)]:
            for i, area in enumerate(peak_ordered):
                row   = mat[i]
                y_off = i
                ax.fill_between(bin_ctr, y_off, y_off + row * 0.85,
                                color=cmap_areas(i), alpha=0.55)
                ax.plot(bin_ctr, y_off + row * 0.85,
                        color=cmap_areas(i), lw=1.2)
                # mark peak
                if not np.isnan(peaks_rp[i] if rg == 1 else peaks_rm[i]):
                    pk = peaks_rp[i] if rg == 1 else peaks_rm[i]
                    ax.axvline(pk, ymin=i / len(peak_ordered),
                               ymax=(i + 1) / len(peak_ordered),
                               color=cmap_areas(i), lw=1.5, ls=":")

            ax.axvline(0, color="k", lw=1.2, ls="--")
            ax.set_xlim(lag_range)
            ax.set_yticks(range(len(peak_ordered)))
            ax.set_yticklabels(peak_ordered, fontsize=10)
            ax.set_xlabel("Lag (behavioral − neural, trials)")
            ax.set_title(f"{RG_LABELS[rg]}")
            ax.invert_yaxis()

        fig_k.suptitle(f"Lag KDE stacked lines — {dir_label} neurons", fontsize=15, y=1.02)
        fig_k.tight_layout()
        _savefig(fig_k, out_dir / f"lag_summary_kde_lines_{dir_label}.png")
        plt.close(fig_k)


# ---------------------------------------------------------------------------
# Per-mouse figures
# ---------------------------------------------------------------------------

def make_mouse_figures(mouse_id: str, results_df: pd.DataFrame,
                        lag_df: pd.DataFrame):
    mdf   = results_df[results_df["mouse_id"] == mouse_id]
    m_lag = lag_df[lag_df["mouse_id"] == mouse_id]
    out   = _mouse_fig_dir(mouse_id)
    ex_out = _example_neuron_dir(mouse_id)

    # example neurons in subfolder
    for tt, tag in [("whisker", "whisker_evoked"), ("auditory", "auditory_evoked")]:
        plot_example_neurons(
            mdf[(mdf["trial_type"] == tt) & (mdf["epoch"] == "evoked")],
            tag=tag, out_dir=ex_out)

    # fraction modulated + slope consistency
    for tt in ["whisker", "auditory"]:
        for epoch in ["baseline", "evoked"]:
            plot_fraction_modulated(mdf, tt, epoch,
                                    out_dir=out, title_prefix=f"{mouse_id} — ")
            plot_slope_consistency(mdf, tt, epoch,
                                   out_dir=out, title_prefix=f"{mouse_id} — ")

    if m_lag.empty:
        return
    areas_mouse = _ordered_areas(
        sorted(m_lag["area_acronym_custom"].dropna().unique().tolist()))
    for direction in [1, -1]:
        plot_lag_distribution(m_lag, direction, areas_mouse,
                               tag=mouse_id, out_dir=out)
        plot_lag_difference_distribution(m_lag, direction,
                                          tag=mouse_id, out_dir=out)


# ---------------------------------------------------------------------------
# Summary figures
# ---------------------------------------------------------------------------

def _make_summary_figures(results_df: pd.DataFrame, learning_df: pd.DataFrame):
    print("Generating summary figures ...")

    for tt in ["whisker", "auditory"]:
        for epoch in ["baseline", "evoked"]:
            plot_fraction_modulated(results_df, tt, epoch,
                                    out_dir=FIG_OUT, title_prefix="Summary — ")
            plot_slope_consistency(results_df, tt, epoch,
                                   out_dir=FIG_OUT, title_prefix="Summary — ")

    plot_pie_chart(results_df, out_dir=FIG_OUT)

    # Three lag datasets: whisker_evoked, whisker_baseline, auditory_evoked
    lag_specs = [
        ("whisker",  "evoked",   "whisker_evoked"),
        ("whisker",  "baseline", "whisker_baseline"),
        ("auditory", "evoked",   "auditory_evoked"),
    ]

    for tt, ep, tag in lag_specs:
        lag_df = compute_lags(results_df, learning_df, trial_type=tt, epoch=ep)
        if "reward_group" in results_df.columns and "reward_group" not in lag_df.columns:
            rg = results_df[["mouse_id", "reward_group"]].drop_duplicates()
            lag_df = lag_df.merge(rg, on="mouse_id", how="left")
        lag_df.to_csv(BASE_OUT / f"lag_analysis_{tag}.csv", index=False)

        sub_dir = FIG_OUT / f"lag_{tag}"
        sub_dir.mkdir(exist_ok=True)

        shared_areas = get_shared_areas(lag_df)
        print(f"[{tag}] Shared areas (≥3 mice/group): {shared_areas}")

        for direction in [1, -1]:
            plot_lag_distribution(lag_df, direction, shared_areas,
                                   tag=tag, out_dir=sub_dir)
            plot_lag_difference_distribution(lag_df, direction,
                                              tag=tag, out_dir=sub_dir)

        #plot_lag_distribution_by_group(lag_df, shared_areas, out_dir=sub_dir)
        plot_lag_summary_pointplot(lag_df, shared_areas, out_dir=sub_dir, tag=tag)
        plot_lag_summary_heatmap(lag_df, shared_areas, out_dir=sub_dir)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_analysis(unit_table: pd.DataFrame,
                 trial_table: pd.DataFrame,
                 learning_df: pd.DataFrame,
                 shift_df: pd.DataFrame | None = None):
    write_selection_criteria()

    if shift_df is None:
        print("WARNING: shift_df not provided — drift-correlated neurons will NOT be excluded.")
    else:
        unit_table = apply_drift_filter(unit_table, shift_df)

    sessions = unit_table["session_id"].unique()

    def _process_session(sid):
        print(f"Processing session {sid} ...")
        u  = unit_table[unit_table["session_id"] == sid]
        t  = trial_table[trial_table["session_id"] == sid]
        df = analyse_session(u, t)
        save_results(df, sid)
        return df

    # Outer parallelism over sessions using threads (inner uses loky processes).
    # prefer="threads" avoids nested process pool conflicts on Windows.
    try:
        all_results = Parallel(n_jobs=min(N_JOBS, len(sessions)), prefer="threads")(
            delayed(_process_session)(sid) for sid in sessions
        )
    except Exception as e:
        print(f"WARNING: session-level parallelism failed ({e}), falling back to sequential.")
        all_results = [_process_session(sid) for sid in sessions]

    results_df = pd.concat([df for df in all_results if not df.empty], ignore_index=True)

    if "reward_group" not in results_df.columns:
        rg = unit_table[["unit_id", "reward_group"]].drop_duplicates()
        results_df = results_df.merge(rg, on="unit_id", how="left")

    lag_df = compute_lags(results_df, learning_df)
    if "reward_group" not in lag_df.columns:
        rg = unit_table[["mouse_id", "reward_group"]].drop_duplicates()
        lag_df = lag_df.merge(rg, on="mouse_id", how="left")

    results_df.drop(columns=["fr_series"], errors="ignore").to_csv(
        BASE_OUT / "all_sessions_sigmoid_results.csv", index=False)
    lag_df.to_csv(BASE_OUT / "lag_analysis.csv", index=False)

    for mouse_id in sorted(results_df["mouse_id"].unique()):
        print(f"  Mouse figures: {mouse_id}")
        make_mouse_figures(mouse_id, results_df, lag_df)

    # summary example neurons (fr_series in-memory only)
    for tt, tag in [("whisker", "whisker_evoked"), ("auditory", "auditory_evoked")]:
        ex_dir = FIG_OUT / "example_neurons"
        ex_dir.mkdir(exist_ok=True)
        #plot_example_neurons(
        #    results_df[(results_df["trial_type"] == tt) &
        #               (results_df["epoch"]      == "evoked")],
        #    tag=tag, out_dir=ex_dir)

    _make_summary_figures(results_df, learning_df)

    print(f"Done. Results saved to {BASE_OUT}")
    return results_df, lag_df


# ---------------------------------------------------------------------------
# Figures-only mode
# ---------------------------------------------------------------------------

def run_figures_only(learning_df: pd.DataFrame):
    print("Loading existing results ...")
    results_df = load_all_results()

    lag_df = compute_lags(results_df, learning_df)
    if "reward_group" in results_df.columns and "reward_group" not in lag_df.columns:
        rg = results_df[["mouse_id", "reward_group"]].drop_duplicates()
        lag_df = lag_df.merge(rg, on="mouse_id", how="left")

    missing = set(learning_df["mouse_id"]) - set(results_df["mouse_id"])
    for m in sorted(missing):
        print(f"WARNING: no results found for mouse {m}")

    #for mouse_id in sorted(results_df["mouse_id"].unique()):
    #    print(f"  Mouse figures: {mouse_id}")
    #    make_mouse_figures(mouse_id, results_df, lag_df)

    # recompute lag for run_figures_only default (whisker evoked) for per-mouse figures
    lag_df = compute_lags(results_df, learning_df)
    if "reward_group" in results_df.columns and "reward_group" not in lag_df.columns:
        rg = results_df[["mouse_id", "reward_group"]].drop_duplicates()
        lag_df = lag_df.merge(rg, on="mouse_id", how="left")
    _make_summary_figures(results_df, learning_df)

    lag_df.to_csv(BASE_OUT / "lag_analysis.csv", index=False)
    print(f"Done. Figures saved to {FIG_OUT}")
    return results_df, lag_df


# ---------------------------------------------------------------------------
# Learning curve loader
# ---------------------------------------------------------------------------

def load_learning_curves_data(path_to_data, subject_ids) -> pd.DataFrame:
    data = []
    for m in subject_ids:
        try:
            fname = f"{m}_whisker_0_whisker_trial_learning_curve_interp.h5"
            df_w  = pd.read_hdf(os.path.join(
                path_to_data, m, "whisker_0", "learning_curve", fname))
            df_w["mouse_id"] = m
            data.append(df_w)
        except FileNotFoundError:
            print(f"No whisker curve for: {m}")
    return pd.concat(data).reset_index(drop=True)


def get_learning_df(path_to_data, subject_ids) -> pd.DataFrame:
    data_df = load_learning_curves_data(path_to_data, subject_ids)
    corrections = {
        'MH036': 20, 'MH035': 20, 'MH031': 18, 'MH030': 34, 'MH028': 12,
        'MH011': 31, 'AB163': 15, 'AB159': 44, 'AB153': 20, 'AB141': np.nan,
        'AB140': 20, 'AB104': 25, 'AB102': 20, 'AB094': 20, 'AB093': 15,
        'AB116': 79,
    }
    for m_id, val in corrections.items():
        data_df.loc[data_df["mouse_id"] == m_id, "learning_trial"] = val
    return data_df.groupby("mouse_id")["learning_trial"].first().reset_index()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Full pipeline:
    #   shift_df    = load_shift_test_results(subject_ids)
    #   learning_df = get_learning_df(path_to_data, subject_ids)
    #   run_analysis(unit_table, trial_table, learning_df, shift_df)
    #
    # Figures only:
    #   learning_df = get_learning_df(path_to_data, subject_ids)
    #   run_figures_only(learning_df)
    print("Import and call run_analysis() or run_figures_only().")