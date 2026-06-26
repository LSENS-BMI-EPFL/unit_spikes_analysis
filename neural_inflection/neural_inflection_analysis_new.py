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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.special import expit
from scipy.stats import mannwhitneyu, gaussian_kde
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

from allen_utils import get_custom_area_order

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
    "xtick.major.width":1.2,
    "ytick.major.width":1.2,
    "lines.linewidth":  1.8,
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_OUT      = Path(r"M:\analysis\Axel_Bisi\combined_results\sigmoid_analysis")
SHIFT_TEST_ROOT = Path(r"M:\analysis\Axel_Bisi\combined_results")
BASE_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT = BASE_OUT / "figures"
FIG_OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASELINE_START    = -2.0
BASELINE_END      = -0.005
EVOKED_START      =  0.005
EVOKED_END        =  0.050

AIC_DELTA_THRESH  = 2.0
BORDER_MARGIN     = 5
S_SHAPE_MIN_SPAN  = 4.0
N_EXAMPLE         = 30
N_JOBS            = -1
N_PERM            = 1000   # PERMANOVA permutations
DRIFT_P_THRESH    = 0.05

CATS       = ["constant", "linear", "sigmoid_down", "sigmoid_up"]
CAT_COLORS = ["#aaaaaa", "#f0a500", "#4477aa", "#cc3333"]
RG_COLORS  = {1: "#228B22", 0: "#DC143C"}

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
  Constant : y = mu                        (1 parameter)
  Linear   : y = a*t + b                   (2 parameters)
  Sigmoid  : y = L / (1+exp(-k*(t-t0))) + b (4 parameters)
    L  = amplitude (Hz); positive -> increasing, negative -> decreasing
    k  = steepness; bounded to [-5, 5] in optimizer
    t0 = inflection trial (continuous); bounded to [t0_lo, t0_hi] (see below)
    b  = baseline offset (Hz)

MODEL SELECTION — AIC
  AIC = N * log(RSS/N) + 2*K   (Gaussian log-likelihood, N trials, K params)
  Sigmoid wins if ALL of:
    (1) AIC_sigmoid < AIC_constant
    (2) AIC_sigmoid < AIC_linear
    (3) min(AIC_constant, AIC_linear) - AIC_sigmoid > {AIC_DELTA_THRESH}  (meaningful advantage)

SIGMOID VALIDITY CONSTRAINTS (beyond AIC)
  Border margin : t0 must fall within [t[0]+{BORDER_MARGIN}, t[-1]-{BORDER_MARGIN}]
                  Hard bound in curve_fit optimizer. Ensures both pre- and
                  post-transition plateaus are observed within the session.
  S-shape check : |k| * (t[-1]-t[0]) > {S_SHAPE_MIN_SPAN}
                  At |k|*(range)=4, sigmoid spans >96% of its amplitude
                  within the data, guaranteeing visible plateaus on both ends.
                  Prevents exponential-looking fits with one extrapolated plateau.
  Convergence   : curve_fit must succeed without exception.

DIRECTION
  sigmoid_up   : L > 0  (firing rate increases with trials)
  sigmoid_down : L < 0  (firing rate decreases with trials)

LAG COMPUTATION (whisker evoked only)
  lag = t0_idx - learning_trial
  t0_idx = round(t0) in whisker-active trial index space.
  learning_trial from behavioral learning curve (hand-corrected per mouse).
  Negative lag -> neural change precedes behavioral learning.
  Positive lag -> neural change follows behavioral learning.

SUMMARY STATISTICS
  Mouse level: median lag per mouse per area per reward group.
  Area inclusion: both R+ and R- represented by >= 3 mice.
  PERMANOVA: permutation test (N={N_PERM}) on reward_group at mouse level.
  Post-hoc: Mann-Whitney U per area, FDR-BH corrected.
"""
    (out_dir / "selection_criteria.txt").write_text(txt)


# ---------------------------------------------------------------------------
# Drift filter
# ---------------------------------------------------------------------------

def load_shift_test_results(mouse_ids: list[str]) -> pd.DataFrame:
    """
    Load per-mouse shift test CSVs and return rows where:
      factor == 'motion' AND epoch == 'baseline'
    with columns [mouse_id, session_id, imec_id, unit_id, p_conservative].
    """
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
            # keep motion / baseline rows only
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
    """
    Exclude neurons with p_conservative < DRIFT_P_THRESH.
    Returns filtered unit_table.
    """
    if shift_df.empty:
        return unit_table
    drift = shift_df[shift_df["p_conservative"] < DRIFT_P_THRESH][
        ["mouse_id", "session_id", "imec_id", "unit_id"]
    ].drop_duplicates()
    drift["_drift"] = True
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
    evoked   = compute_fr_vectorized(spike_times, trial_starts, EVOKED_START,   EVOKED_END)
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
    slope  = np.sum((t - tm) * (y - ym)) / denom
    ic     = ym - slope * tm
    rss    = float(np.sum((y - slope * t - ic) ** 2))
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


def model_comparison(t: np.ndarray, y: np.ndarray) -> dict:
    mu,    rss_c, aic_c              = _fit_constant(y)
    sl, ic, rss_l, aic_l             = _fit_linear(t, y)
    L, k, t0, b, rss_s, aic_s, valid = _fit_sigmoid(t, y)

    sigmoid_wins = (
        valid and
        aic_s < aic_c and aic_s < aic_l and
        (aic_s - min(aic_c, aic_l)) < -AIC_DELTA_THRESH
    )
    linear_wins = (not sigmoid_wins) and (aic_l < aic_c) and \
                  ((aic_l - aic_c) < -AIC_DELTA_THRESH)

    if sigmoid_wins:
        t0_idx    = int(np.round(t0))
        direction = 1 if L > 0 else -1
    else:
        t0 = t0_idx = direction = np.nan

    linear_direction = np.nan
    if not np.isnan(sl):
        linear_direction = 1 if sl > 0 else -1

    return {
        "aic_constant": aic_c, "aic_linear": aic_l, "aic_sigmoid": aic_s,
        "rss_constant": rss_c, "rss_linear": rss_l, "rss_sigmoid": rss_s,
        "best_model":   ("sigmoid" if sigmoid_wins else
                         ("linear" if linear_wins else "constant")),
        "sigmoid_wins": sigmoid_wins,
        "linear_wins":  linear_wins,
        "L": L, "k": k, "t0": t0, "b": b,
        "linear_slope": sl, "linear_intercept": ic,
        "t0_idx": t0_idx, "direction": direction,
        "linear_direction": linear_direction,
    }


# ---------------------------------------------------------------------------
# Per-neuron worker
# ---------------------------------------------------------------------------

def _neuron_worker(unit_id, area, mouse_id, session_id, imec_id, probe,
                   reward_group, spike_times,
                   trial_starts_w, n_w, trial_starts_a, n_a):
    spk  = np.sort(np.asarray(spike_times))
    meta = dict(unit_id=unit_id, area_acronym_custom=area,
                mouse_id=mouse_id, session_id=session_id,
                imec_id=imec_id, probe=probe, reward_group=reward_group)
    results = []
    for trial_type, trial_starts, n_trials in [
        ("whisker",  trial_starts_w, n_w),
        ("auditory", trial_starts_a, n_a),
    ]:
        if n_trials < 10:
            continue
        baseline, evoked_bc = compute_epoch_fr(spk, trial_starts)
        t = np.arange(n_trials, dtype=float)
        for epoch, y in [("baseline", baseline), ("evoked", evoked_bc)]:
            res = model_comparison(t, y)
            res.update(meta)
            res.update({"trial_type": trial_type, "epoch": epoch,
                        "n_trials": n_trials, "mean_fr": float(y.mean()),
                        "fr_series": y.tolist()})
            results.append(res)
    return results


def analyse_session(unit_table: pd.DataFrame, trial_table: pd.DataFrame) -> list[dict]:
    good = unit_table[unit_table["bc_label"] == "good"]
    if good.empty:
        return []
    trial_starts_w, n_w = get_active_trial_starts(trial_table, "whisker")
    trial_starts_a, n_a = get_active_trial_starts(trial_table, "auditory")
    rows    = good.to_dict("records")
    batches = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_neuron_worker)(
            r["unit_id"], r["area_acronym_custom"], r["mouse_id"],
            r["session_id"], r.get("imec_id", np.nan),
            r.get("probe", np.nan), r.get("reward_group", np.nan),
            r["spike_times"], trial_starts_w, n_w, trial_starts_a, n_a,
        )
        for r in rows
    )
    return [item for batch in batches for item in batch]


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_results(results: list[dict], session_id: str) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df.drop(columns=["fr_series"], errors="ignore").to_csv(
        BASE_OUT / f"{session_id}_sigmoid_results.csv", index=False
    )
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
# Area ordering
# ---------------------------------------------------------------------------

def _ordered_areas(areas: list[str]) -> list[str]:
    allen = get_custom_area_order()
    known   = [a for a in allen if a in set(areas)]
    unknown = sorted(a for a in areas if a not in set(allen))
    return known + unknown


# ---------------------------------------------------------------------------
# Lag computation & shared areas
# ---------------------------------------------------------------------------

def compute_lags(results_df: pd.DataFrame, learning_df: pd.DataFrame) -> pd.DataFrame:
    sub = results_df[
        (results_df["trial_type"]    == "whisker") &
        (results_df["epoch"]        == "evoked")  &
        (results_df["sigmoid_wins"] == True)
    ].copy()
    merged = sub.merge(learning_df, on="mouse_id", how="left")
    merged["lag"] = merged["t0_idx"] - merged["learning_trial"]
    return merged


def get_shared_areas(lag_df: pd.DataFrame, min_mice: int = 3) -> list[str]:
    counts = (
        lag_df.groupby(["area_acronym_custom", "reward_group"])["mouse_id"]
        .nunique().unstack(fill_value=0)
    )
    mask = (counts.get(1, pd.Series(0, index=counts.index)) >= min_mice) & \
           (counts.get(0, pd.Series(0, index=counts.index)) >= min_mice)
    return _ordered_areas(counts.index[mask].tolist())


# ---------------------------------------------------------------------------
# PERMANOVA (permutation test on group label, mouse-level medians)
# ---------------------------------------------------------------------------

def _permanova(vals_rp: np.ndarray, vals_rm: np.ndarray,
               n_perm: int = N_PERM) -> float:
    """
    Univariate PERMANOVA: F = (SS_between / df_between) / (SS_within / df_within)
    Permutes group labels, returns p-value.
    """
    n1, n2 = len(vals_rp), len(vals_rm)
    if n1 < 2 or n2 < 2:
        return np.nan
    combined = np.concatenate([vals_rp, vals_rm])
    n        = len(combined)
    labels   = np.array([0] * n1 + [1] * n2)

    def _f_stat(x, lab):
        grand = x.mean()
        grp   = np.unique(lab)
        ss_b  = sum(((x[lab == g].mean() - grand) ** 2) * (lab == g).sum() for g in grp)
        ss_w  = sum(np.sum((x[lab == g] - x[lab == g].mean()) ** 2) for g in grp)
        df_b  = len(grp) - 1
        df_w  = n - len(grp)
        if ss_w == 0 or df_w == 0:
            return np.nan
        return (ss_b / df_b) / (ss_w / df_w)

    obs = _f_stat(combined, labels)
    if np.isnan(obs):
        return np.nan
    rng  = np.random.default_rng(42)
    perm = np.array([
        _f_stat(combined, rng.permutation(labels)) for _ in range(n_perm)
    ])
    return float((perm >= obs).sum() + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Classify categories (vectorized)
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


# ---------------------------------------------------------------------------
# Figures — example neurons (30 per call)
# ---------------------------------------------------------------------------

def _plot_example_neuron(row: pd.Series, out_path: Path):
    y      = np.asarray(row["fr_series"])
    t      = np.arange(len(y), dtype=float)
    t_fine = np.linspace(t[0], t[-1], 300)

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.40)

    # panel 1 — FR + all fits
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

    # panel 2 — AIC bars
    ax2    = fig.add_subplot(gs[0, 2])
    models = ["constant", "linear", "sigmoid"]
    aics   = [row[f"aic_{m}"] for m in models]
    bars   = ax2.bar(models, aics, color=["steelblue", "darkorange", "crimson"], alpha=0.8)
    min_a  = min(aics)
    for bar, aic in zip(bars, aics):
        ax2.text(bar.get_x() + bar.get_width() / 2, aic + abs(min_a) * 0.01,
                 f"{aic - min_a:+.1f}", ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel("AIC")
    ax2.set_title("Model comparison (AIC)")

    # panel 3 — residuals
    ax3 = fig.add_subplot(gs[1, 0])
    resid = y - _sigmoid_fn(t, row["L"], row["k"], row["t0"], row["b"])
    ax3.plot(t, resid, "o", ms=4, color="gray", alpha=0.55)
    ax3.axhline(0, color="k", lw=1.2)
    ax3.set_xlabel("Active trial index")
    ax3.set_ylabel("Residual (Hz)")
    ax3.set_title("Sigmoid residuals")

    # panel 4 — parameters text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
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

    # panel 5 — ΔAIC advantage
    ax5 = fig.add_subplot(gs[1, 2])
    deltas = [row["aic_sigmoid"] - row["aic_linear"],
              row["aic_sigmoid"] - row["aic_constant"]]
    ax5.bar(["vs Linear", "vs Constant"], deltas,
            color=["darkorange", "steelblue"], alpha=0.8)
    ax5.axhline(-AIC_DELTA_THRESH, color="k", ls="--", lw=1.5,
                label=f"threshold = −{AIC_DELTA_THRESH}")
    ax5.set_ylabel("ΔAIC (sigmoid − other)")
    ax5.set_title("Sigmoid advantage")
    ax5.legend(fontsize=10)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
        _plot_example_neuron(row, out_dir / f"example_{tag}_{uid}.png")


# ---------------------------------------------------------------------------
# Figures — combined fraction modulated (single figure, many subplots)
# ---------------------------------------------------------------------------

def plot_fraction_modulated(results_df: pd.DataFrame, trial_type: str, epoch: str,
                             out_dir: Path = FIG_OUT, title_prefix: str = ""):
    """
    One figure with 3 rows × 3 columns:
      Row 0: all neurons stacked bar (all / R+ / R-) → constant / linear / sigmoid
      Row 1: sigmoid-up vs sigmoid-down as fraction of sigmoid neurons (all / R+ / R-)
      Row 2: linear-up vs linear-down as fraction of linear neurons (all / R+ / R-)
    All area bars aligned (same x-axis), all rows share area order.
    """
    sub = results_df[
        (results_df["trial_type"] == trial_type) &
        (results_df["epoch"]     == epoch)
    ].copy()
    if sub.empty:
        return

    sub["category"] = _classify_category(sub)
    areas = _ordered_areas(sorted(sub["area_acronym_custom"].dropna().unique()))
    if not areas:
        return

    x    = np.arange(len(areas))
    bw   = 0.6   # bar width
    groups = [("All",  sub),
              ("R+",   sub[sub["reward_group"] == 1]),
              ("R-",   sub[sub["reward_group"] == 0])]

    fig, axes = plt.subplots(3, 3, figsize=(max(10, len(areas) * 1.1), 14),
                              sharex=True)
    fig.suptitle(f"{title_prefix}Modulation fractions — {trial_type} {epoch}",
                 fontsize=15, y=1.01)

    for col, (grp_label, gdf) in enumerate(groups):
        if gdf.empty:
            for row in range(3):
                axes[row, col].set_visible(False)
            continue

        # ---- row 0: constant / linear / sigmoid fraction of all neurons ----
        ax = axes[0, col]
        ct = pd.crosstab([gdf["mouse_id"], gdf["area_acronym_custom"]], gdf["category"])
        ct = ct.reindex(columns=CATS, fill_value=0)
        ct = ct.div(ct.sum(axis=1), axis=0)
        mdf = ct.groupby(level="area_acronym_custom").mean().reindex(
            index=areas, columns=CATS, fill_value=0)

        bottom = np.zeros(len(areas))
        for cat, color in zip(CATS, CAT_COLORS):
            vals = mdf[cat].values
            ax.bar(x, vals, bottom=bottom, color=color, label=cat,
                   alpha=0.85, width=bw)
            bottom += vals
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of neurons")
        ax.set_title(f"{grp_label} — model type")
        if col == 0:
            ax.legend(loc="upper right", fontsize=9)

        # ---- row 1: sigmoid up vs down within sigmoid neurons ----
        ax = axes[1, col]
        sig_only = gdf[gdf["sigmoid_wins"]].copy()
        if not sig_only.empty:
            ct2 = pd.crosstab([sig_only["mouse_id"], sig_only["area_acronym_custom"]],
                              sig_only["direction"].map({1: "up", -1: "down"}))
            ct2 = ct2.reindex(columns=["up", "down"], fill_value=0)
            ct2 = ct2.div(ct2.sum(axis=1), axis=0)
            mdf2 = ct2.groupby(level="area_acronym_custom").mean().reindex(
                index=areas, fill_value=0)
            ax.bar(x, mdf2.get("up",   pd.Series(0, index=areas)).values,
                   color="#cc3333", alpha=0.85, width=bw, label="↑ up")
            ax.bar(x, mdf2.get("down", pd.Series(0, index=areas)).values,
                   bottom=mdf2.get("up", pd.Series(0, index=areas)).values,
                   color="#4477aa", alpha=0.85, width=bw, label="↓ down")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of sigmoid neurons")
        ax.set_title(f"{grp_label} — sigmoid direction")
        if col == 0:
            ax.legend(loc="upper right", fontsize=9)

        # ---- row 2: linear up vs down within linear neurons ----
        ax = axes[2, col]
        #lin_only = gdf[gdf["linear_wins"]].copy()
        lin_only = pd.DataFrame() #empty for now
        if not lin_only.empty:
            ct3 = pd.crosstab([lin_only["mouse_id"], lin_only["area_acronym_custom"]],
                              lin_only["linear_direction"].map({1: "up", -1: "down",
                                                                np.nan: "unknown"}))
            ct3 = ct3.reindex(columns=["up", "down"], fill_value=0)
            ct3 = ct3.div(ct3.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
            mdf3 = ct3.groupby(level="area_acronym_custom").mean().reindex(
                index=areas, fill_value=0)
            ax.bar(x, mdf3.get("up",   pd.Series(0, index=areas)).values,
                   color="#cc3333", alpha=0.85, width=bw, label="↑ positive slope")
            ax.bar(x, mdf3.get("down", pd.Series(0, index=areas)).values,
                   bottom=mdf3.get("up", pd.Series(0, index=areas)).values,
                   color="#4477aa", alpha=0.85, width=bw, label="↓ negative slope")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of linear neurons")
        ax.set_title(f"{grp_label} — linear direction")
        if col == 0:
            ax.legend(loc="upper right", fontsize=9)

    # shared x-axis labels on bottom row
    for col in range(3):
        axes[2, col].set_xticks(x)
        axes[2, col].set_xticklabels(areas, rotation=45, ha="right")

    fig.tight_layout()
    fname = f"fraction_modulated_{trial_type}_{epoch}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures — slope consistency
# ---------------------------------------------------------------------------

def plot_slope_consistency(results_df: pd.DataFrame, trial_type: str, epoch: str,
                            out_dir: Path = FIG_OUT, title_prefix: str = ""):
    """
    For sigmoid-winning neurons: % where sign(k) == sign(linear_slope).
    Single bar per (trial_type, epoch), broken down by reward group.
    """
    sub = results_df[
        (results_df["trial_type"]    == trial_type) &
        (results_df["epoch"]        == epoch)      &
        (results_df["sigmoid_wins"] == True)
    ].dropna(subset=["k", "linear_slope"]).copy()

    if sub.empty:
        return

    sub["sign_match"] = np.sign(sub["k"]) == np.sign(sub["linear_slope"])

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    groups = [("All", sub),
              ("R+",  sub[sub["reward_group"] == 1]),
              ("R-",  sub[sub["reward_group"] == 0])]

    for ax, (label, gdf) in zip(axes, groups):
        if gdf.empty:
            ax.set_visible(False)
            continue
        pct_match    = 100 * gdf["sign_match"].mean()
        pct_mismatch = 100 - pct_match
        ax.bar(["Match", "Mismatch"], [pct_match, pct_mismatch],
               color=["#4dac26", "#d01c8b"], alpha=0.85, width=0.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of sigmoid neurons")
        ax.set_title(f"{label}  (n={len(gdf)})")
        ax.text(0, pct_match + 1, f"{pct_match:.1f}%", ha="center", va="bottom", fontsize=11)
        ax.text(1, pct_mismatch + 1, f"{pct_mismatch:.1f}%", ha="center", va="bottom", fontsize=11)

    fig.suptitle(f"{title_prefix}Slope consistency (sigmoid k vs linear slope) — "
                 f"{trial_type} {epoch}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f"slope_consistency_{trial_type}_{epoch}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures — lag distribution (violin per area, existing style)
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
    dir_label = "increased" if direction == 1 else "decreased"
    sub = lag_df[
        (lag_df["direction"] == direction) &
        (lag_df["area_acronym_custom"].isin(areas))
    ].dropna(subset=["lag"])
    if sub.empty or not areas:
        return

    mouse_df = sub.groupby(["mouse_id", "area_acronym_custom",
                             "reward_group"])["lag"].median().reset_index()

    for level, df in [("mouse", mouse_df), ("neuron", sub)]:
        fig, axes = plt.subplots(1, len(areas),
                                 figsize=(3.8 * len(areas), 5), sharey=True)
        axes = [axes] if len(areas) == 1 else list(axes)
        pvals = {}
        for ax, area in zip(axes, areas):
            rp = df[(df["area_acronym_custom"] == area) &
                    (df["reward_group"] == 1)]["lag"].dropna().values
            rm = df[(df["area_acronym_custom"] == area) &
                    (df["reward_group"] == 0)]["lag"].dropna().values
            _violin_lag_panel(ax, rp, rm, area)
            ax.set_ylabel("Lag (trials)" if ax is axes[0] else "")
            if len(rp) >= 3 and len(rm) >= 3:
                _, p = mannwhitneyu(rp, rm, alternative="two-sided")
                pvals[area] = p

        areas_tested = list(pvals.keys())
        if areas_tested:
            _, pvals_corr, _, _ = multipletests(
                [pvals[a] for a in areas_tested], method="fdr_bh")
            for area, pc in zip(areas_tested, pvals_corr):
                axes[areas.index(area)].set_title(f"{area}\np_fdr = {pc:.3f}")

        fig.suptitle(f"Lag — {dir_label} neurons ({level} level) {tag}", y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / f"lag_{dir_label}_{level}_{tag}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_lag_difference_distribution(lag_df: pd.DataFrame, direction: int,
                                      tag: str = "", out_dir: Path = FIG_OUT):
    dir_label = "increased" if direction == 1 else "decreased"
    sub = lag_df[lag_df["direction"] == direction].dropna(subset=["lag"])
    if sub.empty:
        return
    mouse_lags = sub.groupby("mouse_id")["lag"].median()
    xlim = (sub["lag"].quantile(0.01), sub["lag"].quantile(0.99))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, label in [
        (axes[0], sub["lag"].values,  "Neuron level"),
        (axes[1], mouse_lags.values,  "Mouse level"),
    ]:
        ax.hist(data, bins=30, color="#888888", alpha=0.60, density=True)
        ax.axvline(data.mean(),     color="crimson",   lw=2, ls="--",
                   label=f"Mean = {data.mean():.1f}")
        ax.axvline(np.median(data), color="steelblue", lw=2, ls="-",
                   label=f"Median = {np.median(data):.1f}")
        ax.axvline(0, color="k", lw=1.2, ls=":")
        ax.set_xlim(xlim)
        ax.set_xlabel("Lag (neural − behavioral trial, trials)")
        ax.set_ylabel("Density")
        ax.set_title(f"{dir_label} — {label}  (n = {len(data)})")
        ax.legend(fontsize=11)

    fig.suptitle(f"Lag distribution — {dir_label} neurons  {tag}")
    fig.tight_layout()
    fig.savefig(out_dir / f"lag_distribution_{dir_label}_{tag}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures — summary pointplot (Allen-ordered, PERMANOVA)
# ---------------------------------------------------------------------------

def plot_lag_summary_pointplot(lag_df: pd.DataFrame, shared_areas: list[str],
                                out_dir: Path = FIG_OUT):
    ordered = _ordered_areas(shared_areas)
    if not ordered:
        return

    # global x-range for both panels
    lag_vals = lag_df[lag_df["area_acronym_custom"].isin(ordered)]["lag"].dropna()
    if lag_vals.empty:
        return
    xlim = (lag_vals.quantile(0.02), lag_vals.quantile(0.98))

    fig, axes = plt.subplots(1, 2,
                              figsize=(14, max(5, len(ordered) * 0.60 + 1.5)),
                              sharey=True)

    for ax, direction in zip(axes, [1, -1]):
        dir_label = "Increased" if direction == 1 else "Decreased"
        sub = lag_df[
            (lag_df["direction"] == direction) &
            (lag_df["area_acronym_custom"].isin(ordered))
        ].dropna(subset=["lag"])

        mouse_med = (sub.groupby(["mouse_id", "area_acronym_custom", "reward_group"])
                     ["lag"].median().reset_index())

        # PERMANOVA omnibus
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
                y_pos = i + jitter
                color = RG_COLORS[rg]
                mean_v = vals.mean()
                sem_v  = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                ax.scatter(vals, np.full(len(vals), y_pos),
                           color=color, s=22, alpha=0.50, zorder=3)
                ax.errorbar(mean_v, y_pos, xerr=sem_v, fmt="o",
                            color=color, ms=8, lw=2, capsize=4, zorder=4,
                            label=rg if i == 0 else "")

            rp_v = mouse_med[(mouse_med["area_acronym_custom"] == area) &
                              (mouse_med["reward_group"] == 1)]["lag"].dropna().values
            rm_v = mouse_med[(mouse_med["area_acronym_custom"] == area) &
                              (mouse_med["reward_group"] == 0)]["lag"].dropna().values
            if len(rp_v) >= 3 and len(rm_v) >= 3:
                _, p = mannwhitneyu(rp_v, rm_v, alternative="two-sided")
                pvals[area] = p

        areas_tested = list(pvals.keys())
        sig_areas    = set()
        if areas_tested:
            _, pvals_corr, _, _ = multipletests(
                [pvals[a] for a in areas_tested], method="fdr_bh")
            for area, pc in zip(areas_tested, pvals_corr):
                if pc < 0.05:
                    sig_areas.add(area)
                    i = ordered.index(area)
                    ax.text(xlim[1] + (xlim[1] - xlim[0]) * 0.02, i,
                            f"* p={pc:.3f}", va="center", fontsize=10,
                            color="k", clip_on=False)

        ax.axvline(0, color="k", lw=1.2, ls=":")
        ax.set_xlim(xlim)
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels(ordered, fontsize=11)
        ax.set_xlabel("Lag (neural − behavioral trial, trials)")
        p_txt = f"PERMANOVA p = {p_omni:.3f}" if not np.isnan(p_omni) else ""
        ax.set_title(f"{dir_label}\n{p_txt}")
        ax.invert_yaxis()

    # legend
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=RG_COLORS[rg], ms=9, label=rg)
               for rg in [1, 0]]
    axes[1].legend(handles=handles, loc="lower right", fontsize=11)

    fig.suptitle("Lag by area — mouse-median ± SEM", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "lag_summary_pointplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures — summary heatmap + marginal lag distribution
# ---------------------------------------------------------------------------

def plot_lag_summary_heatmap(lag_df: pd.DataFrame, shared_areas: list[str],
                              out_dir: Path = FIG_OUT,
                              lag_range: tuple = (-40, 40), n_bins: int = 80):
    """
    For each direction (increased/decreased):
      4-panel figure: R+ heatmap | R+ marginal | R- heatmap | R- marginal
      Heatmap: KDE density per area (Allen-ordered y), lag on x.
      Marginal: pooled lag distribution across all areas, with KDE peak marked.
      All panels share the same x-axis range.
    """
    ordered = _ordered_areas(shared_areas)
    if not ordered:
        return

    bins    = np.linspace(lag_range[0], lag_range[1], n_bins + 1)
    bin_ctr = 0.5 * (bins[:-1] + bins[1:])

    for direction in [1, -1]:
        dir_label = "increased" if direction == 1 else "decreased"
        sub = lag_df[
            (lag_df["direction"] == direction) &
            (lag_df["area_acronym_custom"].isin(ordered))
        ].dropna(subset=["lag"])
        if sub.empty:
            continue

        fig = plt.figure(figsize=(18, max(5, len(ordered) * 0.60 + 2)))
        # 4 columns: heatmap R+, marginal R+, heatmap R-, marginal R-
        gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35,
                               width_ratios=[3, 1, 3, 1])

        hmap_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]
        marg_axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 3])]

        y_extent = [len(ordered) - 0.5, -0.5]

        for hax, max_, rg in zip(hmap_axes, marg_axes, [1, 0]):
            mat  = np.zeros((len(ordered), n_bins))
            peak_lags = []

            for i, area in enumerate(ordered):
                vals = sub[(sub["area_acronym_custom"] == area) &
                           (sub["reward_group"] == rg)]["lag"].dropna().values
                vals_clip = vals[(vals >= lag_range[0]) & (vals <= lag_range[1])]
                if len(vals_clip) < 2:
                    continue
                try:
                    kde  = gaussian_kde(vals_clip, bw_method="scott")
                    row  = kde(bin_ctr)
                    if row.max() > 0:
                        mat[i]    = row / row.max()
                        peak_lags.append(bin_ctr[np.argmax(row)])
                except Exception:
                    pass

            # heatmap
            im = hax.imshow(mat, aspect="auto", origin="upper",
                            extent=[lag_range[0], lag_range[1]] + y_extent,
                            cmap="YlOrRd", vmin=0, vmax=1)
            hax.axvline(0, color="w", lw=1.5, ls="--")
            for i, area in enumerate(ordered):
                if peak_lags and i < len(peak_lags) and not np.isnan(peak_lags[i]):
                    pass  # per-area peaks already visible in heatmap
            hax.set_yticks(range(len(ordered)))
            hax.set_yticklabels(ordered, fontsize=10)
            hax.set_xlabel("Lag (trials)")
            hax.set_title(f"{rg}  —  density per area")
            hax.set_xlim(lag_range)
            plt.colorbar(im, ax=hax, label="Norm. density", shrink=0.8)

            # marginal distribution (all areas pooled)
            all_vals = sub[sub["reward_group"] == rg]["lag"].dropna().values
            all_clip = all_vals[(all_vals >= lag_range[0]) &
                                (all_vals <= lag_range[1])]
            if len(all_clip) >= 2:
                max_.hist(all_clip, bins=30, density=True,
                          color=RG_COLORS[rg], alpha=0.55, orientation="horizontal")
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
            max_.set_ylim(lag_range)
            max_.set_xlabel("Density")
            max_.set_title(f"{rg}  —  pooled")
            max_.set_yticks([])   # y shared visually with heatmap

        fig.suptitle(f"Lag density — {dir_label} neurons", fontsize=15, y=1.02)
        fig.savefig(out_dir / f"lag_summary_heatmap_{dir_label}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Per-mouse figures
# ---------------------------------------------------------------------------

def make_mouse_figures(mouse_id: str, results_df: pd.DataFrame,
                        lag_df: pd.DataFrame):
    mdf   = results_df[results_df["mouse_id"] == mouse_id]
    m_lag = lag_df[lag_df["mouse_id"] == mouse_id]
    out   = _mouse_fig_dir(mouse_id)

    # example neurons
    for stim, tag in [("whisker", "whisker_evoked"), ("auditory", "auditory_evoked")]:
        plot_example_neurons(
            mdf[(mdf["trial_type"] == stim) & (mdf["epoch"] == "evoked")],
            tag=tag, out_dir=out
        )

    # fraction modulated + slope consistency
    for stim in ["whisker", "auditory"]:
        for epoch in ["baseline", "evoked"]:
            plot_fraction_modulated(mdf, stim, epoch,
                                    out_dir=out, title_prefix=f"{mouse_id} — ")
            plot_slope_consistency(mdf, stim, epoch,
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

def _make_summary_figures(results_df: pd.DataFrame, lag_df: pd.DataFrame):
    print("Generating summary figures ...")
    for stim in ["whisker", "auditory"]:
        for epoch in ["baseline", "evoked"]:
            plot_fraction_modulated(results_df, stim, epoch,
                                    out_dir=FIG_OUT, title_prefix="Summary — ")
            plot_slope_consistency(results_df, stim, epoch,
                                   out_dir=FIG_OUT, title_prefix="Summary — ")

    shared_areas = get_shared_areas(lag_df)
    print(f"Shared areas (R+/R- ≥3 mice): {shared_areas}")

    for direction in [1, -1]:
        plot_lag_distribution(lag_df, direction, shared_areas,
                               tag="summary", out_dir=FIG_OUT)
        plot_lag_difference_distribution(lag_df, direction,
                                          tag="summary", out_dir=FIG_OUT)

    plot_lag_summary_pointplot(lag_df, shared_areas, out_dir=FIG_OUT)
    plot_lag_summary_heatmap(lag_df, shared_areas, out_dir=FIG_OUT)


# ---------------------------------------------------------------------------
# Main entry point — full pipeline
# ---------------------------------------------------------------------------

def run_analysis(unit_table: pd.DataFrame,
                 trial_table: pd.DataFrame,
                 learning_df: pd.DataFrame,
                 shift_df: pd.DataFrame | None = None):
    """
    unit_table  : unit_id, mouse_id, session_id, imec_id, probe,
                  area_acronym_custom, bc_label, spike_times, reward_group
    trial_table : start_time, trial_type, context, session_id
    learning_df : mouse_id, learning_trial
    shift_df    : output of load_shift_test_results() — optional; if None,
                  no drift filtering is applied.
    """
    write_selection_criteria()

    if shift_df is not None:
        unit_table = apply_drift_filter(unit_table, shift_df)

    all_results = []
    for sid in unit_table["session_id"].unique():
        print(f"Processing session {sid} ...")
        u   = unit_table[unit_table["session_id"] == sid]
        t   = trial_table[trial_table["session_id"] == sid]
        res = analyse_session(u, t)
        df  = save_results(res, sid)
        all_results.append(df)

    results_df = pd.concat(all_results, ignore_index=True)

    if "reward_group" not in results_df.columns:
        rg = unit_table[["unit_id", "reward_group"]].drop_duplicates()
        results_df = results_df.merge(rg, on="unit_id", how="left")

    lag_df = compute_lags(results_df, learning_df)
    if "reward_group" not in lag_df.columns:
        rg = unit_table[["mouse_id", "reward_group"]].drop_duplicates()
        lag_df = lag_df.merge(rg, on="mouse_id", how="left")

    # save CSVs before figures
    results_df.drop(columns=["fr_series"], errors="ignore").to_csv(
        BASE_OUT / "all_sessions_sigmoid_results.csv", index=False)
    lag_df.to_csv(BASE_OUT / "lag_analysis.csv", index=False)

    # per-mouse figures
    for mouse_id in sorted(results_df["mouse_id"].unique()):
        print(f"  Mouse figures: {mouse_id}")
        make_mouse_figures(mouse_id, results_df, lag_df)

    # summary example neurons (fr_series in-memory only)
    for stim, tag in [("whisker", "whisker_evoked"), ("auditory", "auditory_evoked")]:
        plot_example_neurons(
            results_df[(results_df["trial_type"] == stim) &
                       (results_df["epoch"]     == "evoked")],
            tag=tag, out_dir=FIG_OUT)

    _make_summary_figures(results_df, lag_df)

    print(f"Done. Results saved to {BASE_OUT}")
    return results_df, lag_df


# ---------------------------------------------------------------------------
# Figures-only mode
# ---------------------------------------------------------------------------

def run_figures_only(learning_df: pd.DataFrame,
                     subject_ids: list[str] | None = None):
    """
    Load existing CSVs and regenerate all figures.
    shift_df is reloaded from disk if subject_ids is provided.
    Example neurons skipped (fr_series not on disk).
    """
    print("Loading existing results ...")
    results_df = load_all_results()

    lag_df = compute_lags(results_df, learning_df)
    if "reward_group" in results_df.columns and "reward_group" not in lag_df.columns:
        rg = results_df[["mouse_id", "reward_group"]].drop_duplicates()
        lag_df = lag_df.merge(rg, on="mouse_id", how="left")

    missing = set(learning_df["mouse_id"]) - set(results_df["mouse_id"])
    for m in sorted(missing):
        print(f"WARNING: no results found for mouse {m}")

    for mouse_id in sorted(results_df["mouse_id"].unique()):
        print(f"  Mouse figures: {mouse_id}")
        make_mouse_figures(mouse_id, results_df, lag_df)

    _make_summary_figures(results_df, lag_df)

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