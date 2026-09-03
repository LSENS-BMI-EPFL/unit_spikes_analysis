"""
single_neuron_shift_test_figs.py  (v3)
---------------------------------------
Per-neuron shift-test analysis: firing rate vs probe drift (motion) and
behavioural learning curves.

Four shift tests per neuron
---------------------------
  baseline_vs_motion      all trials
  baseline_vs_learning    whisker active trials
  evoked_vs_motion        whisker active trials
  evoked_vs_learning      whisker active trials

Neuron classification (four-quadrant, preferred over hard exclusion)
--------------------------------------------------------------------
  sig_motion & sig_lc   → ambiguous
  sig_motion & ~sig_lc  → drift-correlated only
  ~sig_motion & sig_lc  → clean learner  ← population of interest
  ~sig_motion & ~sig_lc → unresponsive

Partial r (continuous effect-size complement to the binary quadrant)
--------------------------------------------------------------------
  partial_r = (r_lc - r_motion * r_lm) / sqrt((1-r_motion²)(1-r_lm²))
  where r_lm = Pearson(motion, learning_curve)
  This estimates the learning-curve correlation orthogonal to drift.

New in v3
---------
* Four-quadrant classification + partial_r column in all CSVs
* Peak-lag column (argmax of shift scores) saved per neuron per test
* PERMANOVA omnibus now operates on mouse-level means (not neurons)
* Partial-r forest plot alongside the signed-r forest plot
* Cross-correlation figure: per-area mean |Pearson r| at every lag for
  motion and learning curve side-by-side (compares which areas track each)
* Single-neuron example figure: learning curve / motion / firing rate
  on one page, multiple neurons as subplot rows
* Bootstrapped 95 % CI on forest plot means (replaces SEM for small n)
* Session-level drift-vs-population-rate sanity check figure
* Centralised exclude / classify helpers; DRY throughout
* Parallelised per-session computation via ProcessPoolExecutor
"""
from __future__ import annotations
import socket
import json
import pickle
import re
import warnings
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

from noise_correlations import allen_utils

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

import ephys_utilities.allen_utils as allen_utils
from single_neuron_shift_test.shift_test import shift_test, shift_test_many

# ============================================================================
# GLOBAL STYLE
# ============================================================================

RC = {
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 9, "axes.titleweight": "normal",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False,
    "figure.titlesize": 10, "figure.titleweight": "normal",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8,
    "ytick.major.width": 0.8, "xtick.major.size": 3.5,
    "ytick.major.size": 3.5, "lines.linewidth": 1.2,
    "patch.linewidth": 0.8, "pdf.fonttype": 42,
    "svg.fonttype": "none", "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}
plt.rcParams.update(RC)

REWARD_COLORS  = {0: "#C0392B", 1: "#27AE60"}
REWARD_LABELS  = {0: "R−", 1: "R+"}
EPOCH_COLORS   = {"baseline": "#2980B9", "evoked": "#E74C3C"}
FACTOR_COLORS  = {"motion": "#27AE60", "learning_curve": "#8E44AD"}
BC_COLORS      = {"good": "#2471A3", "mua": "#AAB7B8", "both": "#566573"}
QUAD_COLORS    = {
    "clean_learner":    "#27AE60",
    "ambiguous":        "#E67E22",
    "drift_only":       "#C0392B",
    "unresponsive":     "#BDC3C7",
}
QUAD_LABELS = {
    "clean_learner":  "Clean learner\n(~motion, lc)",
    "ambiguous":      "Ambiguous\n(motion, lc)",
    "drift_only":     "Drift only\n(motion, ~lc)",
    "unresponsive":   "Unresponsive\n(~motion, ~lc)",
}
SIG_ALPHA   = 0.05

FIG_W_SINGLE  = 3.5
FIG_W_DOUBLE  = 6.8
FIG_W_TRIPLE  = 9.5
FIG_W_FULL    = 12.0
FIG_H_UNIT    = 2.8
FIG_H_SQUARE  = 3.5
JITTER        = 0.14
DOT_SIZE      = 4
DOT_ALPHA     = 0.40
MEAN_LW       = 2.2
CAP_SIZE      = 3

# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_CONFIG = {
    "baseline_window": (-2.0, -0.005),
    "evoked_window":   (0.005, 0.050),
    "shift_N":         39,
    "alpha":           SIG_ALPHA,
    "use_p":           "p_conservative",

    "data_root": r"M:\analysis\Axel_Bisi\data",
    "motion_subpath_parts": (
        "Ephys", "catgt_{mouse_id}_{gate}",
        "{mouse_id}_{gate}_imec{imec_id}", "dredge", "motion", "motion",
    ),
    "combined_results_root": r"M:\analysis\Axel_Bisi\combined_results",
    "learning_curve_subpath_parts": (
        "{mouse_id}", "{session_day}", "learning_curve",
        "{mouse_id}_{session_day}_whisker_trial_learning_curve_interp.h5",
    ),
    "per_session_results_subfolder": "single_neuron_shift_test",
    "summary_results_subfolder":     "single_neuron_shift_test",

    "n_example_neurons": 10,
    "random_seed":       0,
    "figure_dpi":        300,
    "min_trials_for_test": 2 * 10 + 1 + 5,
    "n_bootstrap":       2000,       # for CI on forest-plot means
    "n_workers":         1,          # parallel sessions

    # --- time-binned drift (motion) test, replaces trial-indexed shift test
    #     for baseline_vs_motion / evoked_vs_motion only ---
    "drift_test_corr_method":        "pearson",   # "pearson" or "spearman"
    "drift_test_shift_frac":         (0.10, 0.90),  # |lag| as frac of series length
    "drift_test_min_bins":           10,
    "drift_test_bin_width_baseline": 1.0,    # seconds; fine bins, all-trials continuous FR
    "drift_test_n_shifts_baseline":  1000,
    "drift_test_bin_width_evoked":   7.5,    # seconds; coarser, whisker-evoked FR
    "drift_test_n_shifts_evoked":    200,

    "n_process_examples_per_group":  6,   # example units shown per sig/n.s. group
}

# ============================================================================
# SMALL HELPERS
# ============================================================================

def _get_imec_id(electrode_group) -> Optional[int]:
    if electrode_group is None:
        return None
    if isinstance(electrode_group, dict):
        val = electrode_group.get("imecID", electrode_group.get("imec_id"))
        return int(val) if val is not None else None
    if isinstance(electrode_group, str):
        try:
            d = json.loads(electrode_group.replace("'", '"'))
            if isinstance(d, dict):
                return int(d.get("imecID", d.get("imec_id")))
        except (json.JSONDecodeError, ValueError):
            pass
        m = re.search(r"imec(\d+)", electrode_group, re.IGNORECASE)
        if m:
            return int(m.group(1))
    try:
        return int(electrode_group)
    except (TypeError, ValueError):
        return None


def _find_session_data_root(data_root: str, mouse_id: str, session_id) -> Optional[Path]:
    mouse_dir = Path(data_root) / mouse_id
    if not mouse_dir.exists():
        return None
    candidates = sorted(p for p in mouse_dir.iterdir()
                        if p.is_dir() and p.name.startswith(mouse_id))
    if not candidates:
        return None
    sid_str  = str(session_id)
    matches  = [p for p in candidates if sid_str in p.name]
    if len(matches) == 1:
        return matches[0]
    if len(candidates) == 1:
        return candidates[0]
    warnings.warn(f"Ambiguous session folder for {mouse_id}/{session_id}; using most recent.")
    return candidates[-1]


def build_motion_path(config, mouse_id, session_id, imec_id) -> Path:
    session_root = _find_session_data_root(config["data_root"], mouse_id, session_id)
    if session_root is None:
        raise FileNotFoundError(f"No session folder: {mouse_id}/{session_id}")
    glob_pattern = "/".join(
        p.format(mouse_id=mouse_id, imec_id=imec_id, gate="g*")
        for p in config["motion_subpath_parts"]
    )
    matches = sorted(session_root.glob(glob_pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected 1 motion match, found {len(matches)}: {matches}")
    return matches[0]


def build_learning_curve_path(config, mouse_id, session_day) -> Path:
    root  = Path(config["combined_results_root"])
    parts = [p.format(mouse_id=mouse_id, session_day=session_day)
             for p in config["learning_curve_subpath_parts"]]
    return root.joinpath(*parts)


def get_per_session_output_dir(config, mouse_id, session_day) -> Path:
    return (Path(config["combined_results_root"])
            / mouse_id / session_day / config["per_session_results_subfolder"])


def get_summary_output_dir(config) -> Path:
    return Path(config["combined_results_root"]) / config["summary_results_subfolder"]


# ============================================================================
# LOADERS  (cached)
# ============================================================================

_MOTION_CACHE: dict = {}


def load_motion(motion_path: Path):
    key = str(motion_path)
    if key not in _MOTION_CACHE:
        import spikeinterface.core as sc
        _MOTION_CACHE[key] = sc.load(str(motion_path))
    return _MOTION_CACHE[key]


def get_motion_at_times_and_depth(motion, times_s, depth_um) -> np.ndarray:
    times_s = np.asarray(times_s, dtype=float)
    depths  = np.full_like(times_s, float(depth_um))
    if hasattr(motion, "get_displacement_at_time_and_depth"):
        return np.asarray(
            motion.get_displacement_at_time_and_depth(times_s, depths, segment_index=0),
            dtype=float).ravel()
    warnings.warn("Motion fallback to manual nearest-bin lookup.")
    temporal_bins = np.asarray(motion.temporal_bins_s[0])
    spatial_bins  = np.asarray(motion.spatial_bins_um)
    displacement  = np.asarray(motion.displacement[0])
    t_idx = np.searchsorted(temporal_bins, times_s)
    t_idx = np.clip(t_idx, 0, len(temporal_bins) - 1)
    left  = np.clip(t_idx - 1, 0, len(temporal_bins) - 1)
    t_idx = np.where(np.abs(temporal_bins[left] - times_s) <
                     np.abs(temporal_bins[t_idx] - times_s), left, t_idx)
    s_idx = int(np.argmin(np.abs(spatial_bins - depth_um)))
    return displacement[t_idx, s_idx]


def load_learning_curve(h5_path: Path) -> pd.DataFrame:
    return pd.read_hdf(h5_path)


# ============================================================================
# ACTIVITY
# ============================================================================

def compute_firing_rate_per_trial(spike_times, event_times, window) -> np.ndarray:
    spike_times = np.asarray(spike_times, dtype=float)
    if spike_times.size == 0:
        return np.zeros(len(event_times), dtype=float)
    spike_times = np.sort(spike_times)
    win_start, win_end = window
    duration = win_end - win_start
    starts   = np.asarray(event_times, dtype=float) + win_start
    ends     = np.asarray(event_times, dtype=float) + win_end
    counts   = (np.searchsorted(spike_times, ends, "left") -
                np.searchsorted(spike_times, starts, "left"))
    return counts / duration


# ============================================================================
# PARTIAL CORRELATION HELPER
# ============================================================================

def partial_r(r_lc: float, r_motion: float, r_lm: float) -> float:
    """
    Partial correlation of learning curve with firing rate, controlling for motion.
    r_lc    : Pearson r(firing_rate, learning_curve)   — per neuron
    r_motion: Pearson r(firing_rate, motion)            — per neuron
    r_lm    : Pearson r(learning_curve, motion)         — session scalar,
              computed from raw trial-level time series in analyze_session.

    Values are clipped to [-1, 1]: slight exceedance is a known numerical
    artifact of the formula when r_lm is large (small residual denominator).
    """
    denom = np.sqrt((1.0 - r_motion**2) * (1.0 - r_lm**2))
    if denom < 1e-12:
        return np.nan
    return float(np.clip((r_lc - r_motion * r_lm) / denom, -1.0, 1.0))


def _compute_partial_r_column(df: pd.DataFrame,
                               r_lm_by_epoch: dict[str, float]) -> pd.Series:
    """
    Compute the partial correlation for every LC row AND every motion row:
      - LC rows:     partial_r = r_{FR,LC | motion}
                    = (r_LC - r_motion*r_LM) / sqrt((1-r_motion^2)(1-r_LM^2))
      - motion rows: partial_r = r_{FR,motion | LC}
                    = (r_motion - r_LC*r_LM) / sqrt((1-r_LC^2)(1-r_LM^2))
    Same generic partial_r() formula both times, arguments swapped for the
    motion case. r_lm_by_epoch : {epoch: r_LC_M} — Pearson r between the
    learning curve and probe motion computed from the raw trial-level time
    series (session-level scalar, one per epoch); this is the only quantity
    that correctly orthogonalises FR-LC / FR-motion with respect to each
    other -- computing it across neurons would reintroduce the confound.
    """
    result   = pd.Series(np.nan, index=df.index)
    lc_mask  = df["factor"] == "learning_curve"
    mot_mask = df["factor"] == "motion"

    for epoch, r_lm in r_lm_by_epoch.items():
        if not np.isfinite(r_lm):
            continue
        lc_sub  = df[lc_mask  & (df["epoch"] == epoch)].set_index("unit_id")
        mot_sub = df[mot_mask & (df["epoch"] == epoch)].set_index("unit_id")
        shared  = lc_sub.index.intersection(mot_sub.index)
        if shared.empty:
            continue
        r_lc_vals  = lc_sub.loc[shared, "r"].to_numpy(float)
        r_mot_vals = mot_sub.loc[shared, "r"].to_numpy(float)

        # LC rows: r_{FR,LC | motion}
        pr_lc = np.array([partial_r(rl, rm, r_lm)
                          for rl, rm in zip(r_lc_vals, r_mot_vals)])
        lc_rows   = df.index[(df["unit_id"].isin(shared)) &
                             (df["epoch"] == epoch) &
                             (df["factor"] == "learning_curve")]
        uid_to_pr_lc = dict(zip(shared, pr_lc))
        result.loc[lc_rows] = [uid_to_pr_lc.get(u, np.nan)
                               for u in df.loc[lc_rows, "unit_id"]]

        # motion rows: r_{FR,motion | LC} -- same formula, args swapped
        pr_mot = np.array([partial_r(rm, rl, r_lm)
                           for rl, rm in zip(r_lc_vals, r_mot_vals)])
        mot_rows = df.index[(df["unit_id"].isin(shared)) &
                            (df["epoch"] == epoch) &
                            (df["factor"] == "motion")]
        uid_to_pr_mot = dict(zip(shared, pr_mot))
        result.loc[mot_rows] = [uid_to_pr_mot.get(u, np.nan)
                                for u in df.loc[mot_rows, "unit_id"]]
    return result


# ============================================================================
# QUADRANT CLASSIFICATION
# ============================================================================

QUAD_ORDER = ["clean_learner", "ambiguous", "drift_only", "unresponsive"]


def classify_quadrant(sig_motion: bool, sig_lc: bool) -> str:
    if not sig_motion and sig_lc:
        return "clean_learner"
    if sig_motion and sig_lc:
        return "ambiguous"
    if sig_motion and not sig_lc:
        return "drift_only"
    return "unresponsive"


def add_quadrant_column(df: pd.DataFrame, use_p: str) -> pd.DataFrame:
    """Add 'quadrant' column (object dtype) to learning-curve rows, vectorised."""
    df = df.copy()
    df["quadrant"] = pd.array([pd.NA] * len(df), dtype=object)

    for epoch in ["baseline", "evoked"]:
        mot_mask = (df["factor"] == "motion")          & (df["epoch"] == epoch)
        lc_mask  = (df["factor"] == "learning_curve") & (df["epoch"] == epoch)
        if not mot_mask.any() or not lc_mask.any():
            continue
        # map significant flag from motion rows onto each lc row by unit_id
        sig_motion_map = df.loc[mot_mask].set_index("unit_id")["significant"]
        lc_idx = df.index[lc_mask]
        sig_m  = df.loc[lc_mask, "unit_id"].map(sig_motion_map).fillna(False).astype(bool)
        sig_l  = df.loc[lc_mask, "significant"].astype(bool)
        quads  = np.where(~sig_m & sig_l,  "clean_learner",
                 np.where( sig_m & sig_l,  "ambiguous",
                 np.where( sig_m & ~sig_l, "drift_only",
                                            "unresponsive")))
        df.loc[lc_idx, "quadrant"] = quads
    return df


# ============================================================================
# CORE PER-NEURON TEST
# ============================================================================

@dataclass
class NeuronTestResult:
    session_id:     object
    mouse_id:       str
    unit_id:        object
    bc_label:       object
    depth:          float
    imec_id:        Optional[int]
    area:           Optional[str]
    reward_group:   Optional[int]
    epoch:          str
    factor:         str
    n_trials:       int
    r:              float
    m:              int
    p_conservative: float
    p_approx:       float
    significant:    bool
    peak_lag:       float = field(default=np.nan)   # lag of max |score| (in trials)
    error:          Optional[str] = None
    p_fdr:          float = field(default=np.nan)   # BH-FDR corrected p (motion test only)


def run_single_test(activity, factor_vals, N, alpha, use_p,
                    epoch, factor, meta) -> NeuronTestResult:
    n_trials = len(activity)
    bad = lambda **kw: NeuronTestResult(
        **meta, epoch=epoch, factor=factor, n_trials=n_trials,
        r=np.nan, m=np.nan, p_conservative=np.nan, p_approx=np.nan,
        significant=False, peak_lag=np.nan, **kw)

    if n_trials <= 2 * N:
        return bad(error=f"too few trials ({n_trials})")
    if np.all(activity == activity[0]) or np.all(factor_vals == factor_vals[0]):
        # zero variance -> correlation undefined but not missing: a constant
        # signal cannot covary with anything, so r=0, p=1 (fully non-sig)
        # is the well-defined answer, not NaN.
        return NeuronTestResult(
            **meta, epoch=epoch, factor=factor, n_trials=n_trials,
            r=0.0, m=0, p_conservative=1.0, p_approx=1.0,
            significant=False, peak_lag=np.nan,
            error="constant vector (r set to 0, p to 1)")
    try:
        res = shift_test(activity, factor_vals, N)
    except Exception as exc:
        return bad(error=f"shift_test failed: {exc}")

    scores   = np.asarray(res["scores"])
    shifts   = np.arange(-N, N + 1)
    peak_lag = float(shifts[np.argmax(np.abs(scores))])
    p_val    = res[use_p]

    return NeuronTestResult(
        **meta, epoch=epoch, factor=factor, n_trials=n_trials,
        r=res["sign_at_shift0"], m=res["m"],
        p_conservative=res["p_conservative"], p_approx=res["p_approx"],
        significant=bool(p_val <= alpha),
        peak_lag=peak_lag, error=None)


# ============================================================================
# TIME-BINNED DRIFT (MOTION) TEST
# ----------------------------------------------------------------------------
# Replaces the trial-indexed shift test for baseline_vs_motion / evoked_vs_motion.
# Session is binned into fixed-width time bins; firing rate and DREDge
# displacement are each aggregated per bin, giving two time-aligned series.
# The null preserves each series' own autocorrelation/trend by shifting one
# series by a random lag (drawn away from 0 and from the series length) and
# recomputing the correlation on the trimmed overlap -- true alignment is
# destroyed, internal structure is not.
# ============================================================================

def _shifted_pair(x: np.ndarray, y: np.ndarray, k: int):
    """Shift y by k samples relative to x, trim to the overlapping region."""
    n = len(x)
    if k > 0:
        return x[k:], y[:n - k]
    if k < 0:
        return x[:n + k], y[-k:]
    return x, y


def _corr_fn(method: str):
    if method == "spearman":
        return lambda a, b: stats.spearmanr(a, b)[0]
    return lambda a, b: stats.pearsonr(a, b)[0]


def shift_null_test(x: np.ndarray, y: np.ndarray, n_shifts: int, rng,
                    method: str = "pearson",
                    shift_frac=(0.10, 0.90), return_null: bool = False):
    """
    Two-sided shift test: observed corr(x, y) vs. a null built from random
    lags of y relative to x, |lag| drawn uniformly within shift_frac * len(x)
    (excludes near-zero lags, where the shift barely changes anything, and
    near-full-length lags, which leave almost no overlap).

    Returns (r_obs, p, m) where m is the number of usable null draws.
    If return_null=True, also returns the null_vals array (for diagnostics).
    """
    n = len(x)
    lo = max(1, int(round(shift_frac[0] * n)))
    hi = int(round(shift_frac[1] * n))
    corr = _corr_fn(method)
    if hi <= lo:
        out = (np.nan, np.nan, 0)
        return (*out, np.array([])) if return_null else out
    try:
        r_obs = corr(x, y)
    except Exception:
        out = (np.nan, np.nan, 0)
        return (*out, np.array([])) if return_null else out
    if not np.isfinite(r_obs):
        out = (np.nan, np.nan, 0)
        return (*out, np.array([])) if return_null else out

    signs = rng.choice([-1, 1], size=n_shifts)
    mags  = rng.integers(lo, hi + 1, size=n_shifts)
    lags  = signs * mags

    null_vals = np.full(n_shifts, np.nan)
    for i, k in enumerate(lags):
        xa, ya = _shifted_pair(x, y, int(k))
        if len(xa) < 3 or np.all(xa == xa[0]) or np.all(ya == ya[0]):
            continue
        try:
            null_vals[i] = corr(xa, ya)
        except Exception:
            continue

    finite = np.isfinite(null_vals)
    m = int(finite.sum())
    if m == 0:
        out = (r_obs, np.nan, 0)
        return (*out, null_vals) if return_null else out
    p = float(np.mean(np.abs(null_vals[finite]) >= abs(r_obs)))
    out = (r_obs, p, m)
    return (*out, null_vals) if return_null else out


def compute_binned_firing_rate(spike_times, t_start, t_end, bin_width,
                               event_times=None, event_window=None):
    """
    Bin firing rate over fixed-width time bins spanning [t_start, t_end].

    Continuous mode (event_times=None): rate per bin from all spikes in that
    bin (baseline).  Event mode: per-trial firing rate in `event_window`
    relative to each event onset, averaged over events whose onset falls in
    that bin; bins with no events are NaN (evoked -- stimulus-locked, so only
    bins containing whisker trials carry information).

    Returns (bin_centers, rate_per_bin).
    """
    n_bins = int(np.floor((t_end - t_start) / bin_width))
    if n_bins < 1:
        return np.array([]), np.array([])
    edges   = t_start + np.arange(n_bins + 1) * bin_width
    centers = (edges[:-1] + edges[1:]) / 2.0

    if event_times is None:
        spk    = np.sort(np.asarray(spike_times, dtype=float))
        counts = np.histogram(spk, bins=edges)[0]
        rate   = counts / bin_width
        return centers, rate.astype(float)

    ev       = np.asarray(event_times, dtype=float)
    ev_in    = (ev >= t_start) & (ev < t_end)
    ev       = ev[ev_in]
    ev_rate  = compute_firing_rate_per_trial(spike_times, ev, event_window)
    bin_idx  = np.clip(np.digitize(ev, edges) - 1, 0, n_bins - 1)
    rate     = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = ev_rate[bin_idx == b]
        if len(sel):
            rate[b] = float(np.mean(sel))
    return centers, rate


def time_binned_drift_test(spike_times, motion, depth, t_start, t_end,
                           bin_width, n_shifts, seed, config,
                           event_times=None, event_window=None):
    """
    Full recipe: bin FR and DREDge displacement over time, correlate, test
    against a shift-randomized null.  Returns a dict with r, p, m, n_bins.
    """
    method     = config.get("drift_test_corr_method", "pearson")
    shift_frac = config.get("drift_test_shift_frac", (0.10, 0.90))
    min_bins   = config.get("drift_test_min_bins", 10)

    centers, rate = compute_binned_firing_rate(
        spike_times, t_start, t_end, bin_width,
        event_times=event_times, event_window=event_window)
    if len(centers) < min_bins:
        return dict(r=np.nan, p=np.nan, m=0, n_bins=len(centers),
                    error=f"too few bins ({len(centers)})")

    try:
        disp = get_motion_at_times_and_depth(motion, centers, depth)
    except Exception as exc:
        return dict(r=np.nan, p=np.nan, m=0, n_bins=len(centers),
                    error=f"motion lookup failed: {exc}")

    valid = np.isfinite(rate) & np.isfinite(disp)
    r_v, d_v = rate[valid], np.asarray(disp, dtype=float)[valid]
    if len(r_v) < min_bins:
        return dict(r=np.nan, p=np.nan, m=0, n_bins=len(r_v),
                    error=f"too few valid bins ({len(r_v)})")
    if np.all(r_v == r_v[0]) or np.all(d_v == d_v[0]):
        # zero variance in one series -> Pearson/Spearman r is undefined
        # (0/0), but this is NOT missing data: a constant signal cannot
        # covary with anything, so the well-defined answer is r=0, p=1
        # (no evidence of correlation), not NaN.
        return dict(r=0.0, p=1.0, m=0, n_bins=len(r_v),
                    error="constant vector (r set to 0, p to 1)")

    rng = np.random.default_rng(seed)
    r_obs, p, m = shift_null_test(r_v, d_v, n_shifts, rng,
                                  method=method, shift_frac=shift_frac)
    return dict(r=r_obs, p=p, m=m, n_bins=len(r_v), error=None)


def _drift_test_seed(base_seed, session_id, unit_id, epoch) -> np.random.SeedSequence:
    """Deterministic, unit/epoch-specific seed derived from the repo's base
    `random_seed`, so shift draws are reproducible but not identical
    across neurons."""
    tag = abs(hash((str(session_id), str(unit_id), epoch))) % (2**32)
    return np.random.SeedSequence([int(base_seed), tag])


# Error prefixes that mean "could not compute" (missing/insufficient data),
# as opposed to "computed and the answer is a well-defined null" (constant
# vector -> r=0, p=1). Only the former gets flagged/warned.
DRIFT_TEST_DATA_WARNING_PREFIXES = (
    "too few bins", "motion lookup failed", "too few valid bins",
)


def write_drift_test_warnings(results: pd.DataFrame, out_dir: Path,
                              mouse_id: str, session_day) -> Optional[Path]:
    """
    Flag motion-test rows that could not be computed due to missing/
    insufficient data (too few bins, motion lookup failure, too few valid
    bins after alignment) -- NOT the constant-vector case, which is a valid
    r=0/p=1 result, not a data problem. Prints each flagged unit/epoch and
    writes a warning.txt summary to out_dir.
    """
    mot = results[results["factor"] == "motion"]
    if mot.empty or "error" not in mot.columns:
        return None
    flagged = mot[mot["error"].notna() &
                 mot["error"].str.startswith(DRIFT_TEST_DATA_WARNING_PREFIXES)]
    if flagged.empty:
        return None

    lines = [f"Drift-test data-quality warnings -- {mouse_id} / {session_day}",
             f"{len(flagged)} of {len(mot)} motion-test rows flagged "
             f"(missing/insufficient data, not a computed null result):", ""]
    for _, row in flagged.iterrows():
        msg = (f"unit {row['unit_id']}  epoch={row['epoch']}  "
              f"area={row['area']}  -- {row['error']}")
        lines.append("  " + msg)
        warnings.warn(f"[{mouse_id}/{session_day}] {msg}")

    warn_path = out_dir / "warning.txt"
    warn_path.write_text("\n".join(lines) + "\n")
    print(f"  -> {len(flagged)} drift-test data-quality warning(s) -> {warn_path}")
    return warn_path


def analyze_session(unit_table, trial_table, session_id, mouse_id,
                    session_day, config) -> pd.DataFrame:
    N               = config["shift_N"]
    alpha           = config["alpha"]
    use_p           = config["use_p"]
    baseline_window = config["baseline_window"]
    evoked_window   = config["evoked_window"]

    trials = (trial_table[(trial_table["session_id"] == session_id)
              &(trial_table["context"] != "passive")]
              .sort_values("start_time").reset_index(drop=True))

    wt_mask             = ((trials["trial_type"] == "whisker_trial") &
                           (trials["context"] != "passive"))

    whisker_trials      = trials.loc[wt_mask].reset_index(drop=True)

    all_start_times     = trials["start_time"].to_numpy(float)
    whisker_start_times = whisker_trials["start_time"].to_numpy(float)

    reward_group = None
    if "reward_group" in trials.columns:
        rg_vals = trials["reward_group"].dropna()
        if len(rg_vals):
            reward_group = int(rg_vals.mode().iloc[0])

    learning_curve = None
    lc_path = build_learning_curve_path(config, mouse_id, session_day)
    if lc_path.exists():
        try:
            df_lc          = load_learning_curve(lc_path)
            learning_curve = df_lc["p_mean"][0]
            n_lc           = len(learning_curve)
            n_wt           = len(whisker_start_times)
            if n_lc < n_wt:
                warnings.warn(
                    f"LC length ({n_lc}) < whisker trials ({n_wt}) for "
                    f"{mouse_id}/{session_day}; truncating trials.")
            n = min(n_lc, n_wt)
            learning_curve      = learning_curve[:n]
            whisker_start_times = whisker_start_times[:n]
            whisker_trials      = whisker_trials.iloc[:n].reset_index(drop=True)
        except Exception as exc:
            warnings.warn(f"Learning curve load failed: {exc}")
    else:
        warnings.warn(f"Learning curve not found: {lc_path}")

    units = unit_table[unit_table["session_id"] == session_id].copy()

    # session time bounds for the time-binned drift test (point 3: start/end
    # of trial_table start_time for this session)
    t_start = float(trials["start_time"].min()) if len(trials) else np.nan
    t_end   = float(trials["start_time"].max()) if len(trials) else np.nan
    base_seed = config.get("random_seed", 0)

    rows          = []
    motion_raw    = {"baseline": [], "evoked": []}   # collected for within-session FDR
    lc_raw        = {"baseline": [], "evoked": []}   # collected for within-session FDR

    for _, unit in units.iterrows():
        unit_id   = unit["unit_id"]
        bc_label  = unit.get("bc_label", None)
        depth     = float(unit["depth"])
        spk       = np.asarray(unit["spike_times"], dtype=float)
        imec_id   = _get_imec_id(unit.get("electrode_group"))
        area      = unit.get("area_custom_acronym",
                    unit.get("area_acronym_custom", None))
        rg        = int(unit["reward_group"]) if "reward_group" in unit.index else reward_group
        meta      = dict(session_id=session_id, mouse_id=mouse_id, unit_id=unit_id,
                         bc_label=bc_label, depth=depth, imec_id=imec_id,
                         area=area, reward_group=rg)

        motion_obj   = None
        motion_error = None
        if imec_id is None:
            motion_error = "could not determine imecID"
        else:
            try:
                mp         = build_motion_path(config, mouse_id, session_id, imec_id)
                motion_obj = load_motion(mp)
            except Exception as exc:
                motion_error = str(exc)

        # --- time-binned drift (motion) test: baseline (continuous, fine
        #     bins) and evoked (whisker-locked, coarser bins) -- raw p-values
        #     collected here, BH-FDR applied per epoch after the unit loop.
        for ep, event_times, event_window, bin_width, n_shifts in [
            ("baseline", None,                 None,           config["drift_test_bin_width_baseline"], config["drift_test_n_shifts_baseline"]),
            ("evoked",   whisker_start_times,  evoked_window,  config["drift_test_bin_width_evoked"],   config["drift_test_n_shifts_evoked"]),
        ]:
            if motion_error or np.isnan(t_start) or np.isnan(t_end):
                res = dict(r=np.nan, p=np.nan, m=0, n_bins=0,
                          error=motion_error or "no trials for session bounds")
            else:
                seed = _drift_test_seed(base_seed, session_id, unit_id, ep)
                res = time_binned_drift_test(
                    spk, motion_obj, depth, t_start, t_end, bin_width, n_shifts,
                    seed, config, event_times=event_times, event_window=event_window)
            motion_raw[ep].append(dict(meta=meta, **res))

        def _lc(activity, ep):
            if learning_curve is None:
                return NeuronTestResult(
                    **meta, epoch=ep, factor="learning_curve",
                    n_trials=len(activity), r=np.nan, m=np.nan,
                    p_conservative=np.nan, p_approx=np.nan,
                    significant=False, error="learning curve unavailable")
            lc = learning_curve[:len(activity)]
            return run_single_test(activity, lc, N, alpha, use_p, ep, "learning_curve", meta)

        baseline_whisker = compute_firing_rate_per_trial(spk, whisker_start_times, baseline_window)
        evoked_whisker   = compute_firing_rate_per_trial(spk, whisker_start_times, evoked_window)
        lc_raw["baseline"].append(_lc(baseline_whisker, "baseline"))
        lc_raw["evoked"].append(_lc(evoked_whisker,   "evoked"))

    # --- within-session BH-FDR on the motion-test p-values, per epoch ---
    # `significant` is explicitly derived from p_fdr (q <= alpha), not taken
    # from statsmodels' `rejected` output directly -- same result by
    # definition of BH-FDR, but makes the dependency on p_fdr explicit.
    def _fdr_build_motion(epoch, raw_list):
        pvals = np.array([e["p"] for e in raw_list], dtype=float)
        finite = np.isfinite(pvals)
        qvals  = np.full(len(pvals), np.nan)
        if finite.any():
            _, qcorr, _, _ = multipletests(
                pvals[finite], alpha=alpha, method="fdr_bh")
            qvals[finite] = qcorr
        sig = np.isfinite(qvals) & (qvals <= alpha)
        out = []
        for e, s, q in zip(raw_list, sig, qvals):
            out.append(NeuronTestResult(
                **e["meta"], epoch=epoch, factor="motion",
                n_trials=e["n_bins"], r=e["r"], m=e["m"],
                p_conservative=e["p"], p_approx=e["p"], p_fdr=q,
                significant=bool(s), peak_lag=np.nan, error=e["error"]))
        return out

    for epoch in ["baseline", "evoked"]:
        rows.extend(_fdr_build_motion(epoch, motion_raw[epoch]))

    # --- within-session BH-FDR on the learning-curve-test p-values, per
    #     epoch, same treatment as motion. Correction input is res[use_p]
    #     (the same p already stored on each NeuronTestResult); everything
    #     else about the result (r, peak_lag, etc.) is kept as computed by
    #     run_single_test -- only `significant` and `p_fdr` are overwritten.
    def _fdr_build_lc(raw_list):
        pvals = np.array([getattr(r, use_p) for r in raw_list], dtype=float)
        finite = np.isfinite(pvals)
        sig    = np.zeros(len(pvals), dtype=bool)
        qvals  = np.full(len(pvals), np.nan)
        if finite.any():
            rejected, qcorr, _, _ = multipletests(
                pvals[finite], alpha=alpha, method="fdr_bh")
            sig[finite]   = rejected
            qvals[finite] = qcorr
        return [replace(r, significant=bool(s), p_fdr=q)
                for r, s, q in zip(raw_list, sig, qvals)]

    for epoch in ["baseline", "evoked"]:
        rows.extend(_fdr_build_lc(lc_raw[epoch]))

    df = pd.DataFrame([r.__dict__ for r in rows])

    # --- r_LC_M: session-level Pearson r(learning_curve, motion) per epoch ---
    # Computed from the raw trial-level time series, not across neurons.
    # Uses the median-depth unit motion trace as a representative probe signal.
    # baseline epoch uses all trials; evoked epoch uses whisker trials only.
    r_lm_by_epoch: dict[str, float] = {}
    if learning_curve is not None:
        # pick one representative motion trace (median-depth unit, first imec)
        _mot_traces: dict[str, np.ndarray] = {}   # epoch -> motion array
        for _ep, _st, _wlen in [
            ("baseline", all_start_times,     len(all_start_times)),
            ("evoked",   whisker_start_times,  len(learning_curve)),
        ]:
            for _, _u in units.iterrows():
                _imec = _get_imec_id(_u.get("electrode_group"))
                if _imec is None:
                    continue
                try:
                    _mp  = build_motion_path(config, mouse_id, session_id, _imec)
                    _mo  = load_motion(_mp)
                    _mot = get_motion_at_times_and_depth(_mo, _st[:_wlen], float(_u["depth"]))
                    _mot_traces[_ep] = _mot
                    break
                except Exception:
                    continue
        lc_arr = np.asarray(learning_curve, dtype=float)
        for _ep, _mot in _mot_traces.items():
            n = min(len(lc_arr), len(_mot))
            if n < 3:
                continue
            _lc_s  = lc_arr[:n];  _m_s = _mot[:n]
            finite = np.isfinite(_lc_s) & np.isfinite(_m_s)
            if finite.sum() < 3:
                continue
            r_lm_by_epoch[_ep] = float(np.corrcoef(_lc_s[finite], _m_s[finite])[0, 1])

    df["partial_r"] = _compute_partial_r_column(df, r_lm_by_epoch)
    df = add_quadrant_column(df, use_p)
    return df


# ============================================================================
# PERFORMANCE TERTILE
# ============================================================================

def add_performance_tertile(df: pd.DataFrame, trial_table: pd.DataFrame) -> pd.DataFrame:
    if "lick_flag" not in trial_table.columns:
        df["perf_tertile"] = np.nan
        return df
    mask = ((trial_table["trial_type"] == "whisker_trial") &
            (trial_table["context"] != "passive"))
    wt   = trial_table[mask]
    hr   = (wt.groupby("session_id")["lick_flag"].mean()
              .rename("hit_rate").reset_index())
    q    = hr["hit_rate"].quantile([1/3, 2/3]).values
    try:
        bins   = [-np.inf] + sorted(set(q.tolist())) + [np.inf]
        labels = list(range(len(bins) - 1))
        hr["perf_tertile"] = pd.cut(hr["hit_rate"], bins=bins,
                                     labels=labels, duplicates="drop").astype(float)
    except Exception:
        hr["perf_tertile"] = (
            pd.qcut(hr["hit_rate"].rank(method="first"),
                    q=min(3, len(hr)), labels=False, duplicates="drop").astype(float))
    return df.merge(hr[["session_id", "perf_tertile"]], on="session_id", how="left")


# ============================================================================
# CENTRALISED DATA PREPARATION  (single call site, no duplication)
# ============================================================================

def prepare_clean(df: pd.DataFrame, use_p: str,
                  bc: str = "both",
                  epoch: Optional[str] = None,
                  factor: Optional[str] = None,
                  quadrant: Optional[str] = None) -> pd.DataFrame:
    """
    Filter + cast in one place.  All figure helpers call this.
    bc='both' keeps all bc_label values.
    """
    out = df.copy()
    for col in ["r", "p_conservative", "p_approx", "p_fdr", "partial_r"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if bc != "both":
        out = out[out["bc_label"] == bc]
    if epoch is not None:
        out = out[out["epoch"] == epoch]
    if factor is not None:
        out = out[out["factor"] == factor]
    if quadrant is not None:
        out = out[out["quadrant"] == quadrant]
    return out.dropna(subset=["r"])


def mouse_means(df: pd.DataFrame, value_col: str = "r") -> pd.DataFrame:
    """Collapse to one mean per (mouse_id, area, reward_group)."""
    return (df.groupby(["mouse_id", "area", "reward_group"])[value_col]
              .mean().reset_index())


# ============================================================================
# FIGURE I/O HELPERS
# ============================================================================

def _save_fig(fig, subfolder: Path, stem: str, dpi: int = 300):
    subfolder.mkdir(parents=True, exist_ok=True)
    base = subfolder / stem
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".png", dpi=dpi)
    plt.close(fig)
    print(f"    → {base}.[pdf|png]")


def _stat_label(p: float) -> str:
    if not np.isfinite(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _ax_defaults(ax, xlabel="", ylabel="", title=""):
    ax.tick_params(which="both", direction="out",
                   labelbottom=True, labelleft=True)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title:  ax.set_title(title)
    return ax


def _area_color_map(areas):
    cmap = plt.get_cmap("tab20")
    return {a: cmap(i / max(len(areas) - 1, 1)) for i, a in enumerate(sorted(areas))}


# ============================================================================
# PRIMITIVE PLOT FUNCTIONS
# ============================================================================

def _strip_mean_sem(ax, pos: int, vals: np.ndarray, color,
                    jitter=JITTER, dot_size=DOT_SIZE, dot_alpha=DOT_ALPHA):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
    rng = np.random.default_rng(int(round(abs(pos) * 100)) + 42)
    x   = pos + rng.uniform(-jitter, jitter, len(vals))
    ax.scatter(x, vals, s=dot_size, color=color, alpha=dot_alpha,
               linewidths=0, zorder=2, rasterized=True)
    m   = vals.mean()
    sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
    ax.plot([pos - 0.28, pos + 0.28], [m, m], color=color, lw=MEAN_LW, zorder=4)
    ax.errorbar(pos, m, yerr=sem, fmt="none", color=color,
                capsize=CAP_SIZE, capthick=1.0, lw=1.2, zorder=4)


def _bootstrap_ci(vals: np.ndarray, n_boot: int = 2000, ci: float = 0.95,
                  seed: int = 0) -> tuple[float, float]:
    """Return (lo, hi) bootstrap CI of the mean."""
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        m = float(vals.mean()) if len(vals) == 1 else np.nan
        return m, m
    rng     = np.random.default_rng(seed)
    samples = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo, hi  = np.percentile(samples, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def _sig_bracket(ax, x1, x2, y_top, label, dy_frac=0.06):
    span = abs(ax.get_ylim()[1] - ax.get_ylim()[0]) or 1
    dy   = span * dy_frac
    ax.plot([x1, x2], [y_top, y_top], color="k", lw=0.9, clip_on=False)
    ax.text((x1 + x2) / 2, y_top + dy * 0.3, label,
            ha="center", va="bottom", fontsize=8, clip_on=False)


def _one_sample_wilcoxon(vals):
    v = vals[np.isfinite(vals)]
    if len(v) < 5:
        return np.nan
    try:
        _, p = stats.wilcoxon(v)
        return p
    except Exception:
        return np.nan


def _shared_areas_min_mice(df, rg_vals, min_mice=3):
    if len(rg_vals) < 2:
        return sorted(df["area"].dropna().unique())
    shared = None
    for rg in rg_vals:
        sub = df[df["reward_group"] == rg]
        qualifying = {a for a in sub["area"].dropna().unique()
                      if sub[sub["area"] == a]["mouse_id"].nunique() >= min_mice}
        shared = qualifying if shared is None else shared & qualifying
    return sorted(shared) if shared else []


# ============================================================================
# PERMANOVA
# ============================================================================

def _permanova(df, group_col="reward_group", value_col="r", n_perm=999, seed=0):
    rng   = np.random.default_rng(seed)
    vals  = df[value_col].dropna().values
    grps  = df.loc[df[value_col].notna(), group_col].values
    n     = len(vals)
    if n < 4:
        return np.nan, np.nan
    labels = np.unique(grps)
    if len(labels) < 2:
        return np.nan, np.nan

    def pseudo_f(v, g):
        grand  = v.mean()
        ss_tot = np.sum((v - grand) ** 2)
        ss_res = sum(np.sum((v[g == lbl] - v[g == lbl].mean()) ** 2) for lbl in labels)
        ss_bet = ss_tot - ss_res
        df_bet = len(labels) - 1
        df_res = n - len(labels)
        if df_res == 0 or ss_res == 0:
            return np.nan
        return (ss_bet / df_bet) / (ss_res / df_res)

    f_obs = pseudo_f(vals, grps)
    if np.isnan(f_obs):
        return np.nan, np.nan
    count = sum(1 for _ in range(n_perm)
                if (f_p := pseudo_f(vals, rng.permutation(grps))) >= f_obs
                and not np.isnan(f_p))
    return float(f_obs), float((count + 1) / (n_perm + 1))


# ============================================================================
# SESSION DRIFT SANITY CHECK
# ============================================================================

def plot_drift_sanity(unit_table, trial_table, session_id, mouse_id,
                      session_day, config, out_dir):
    """
    One figure per session: population mean firing rate vs probe motion over trials.
    Quick visual check that DREDge estimates are sensible.
    """
    trials = (trial_table[trial_table["session_id"] == session_id]
              .sort_values("start_time").reset_index(drop=True))
    all_st  = trials["start_time"].to_numpy(float)
    units   = unit_table[unit_table["session_id"] == session_id]
    bw      = config["baseline_window"]

    # population mean baseline FR
    all_fr = []
    for _, u in units.iterrows():
        spk = np.asarray(u["spike_times"], dtype=float)
        all_fr.append(compute_firing_rate_per_trial(spk, all_st, bw))
    if not all_fr:
        return
    pop_fr = np.nanmean(all_fr, axis=0)

    # one motion trace (first available imec)
    motion_trace = None
    for _, u in units.iterrows():
        imec = _get_imec_id(u.get("electrode_group"))
        if imec is None:
            continue
        try:
            mp = build_motion_path(config, mouse_id, session_id, imec)
            mo = load_motion(mp)
            motion_trace = get_motion_at_times_and_depth(mo, all_st, float(u["depth"]))
            break
        except Exception:
            continue

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W_FULL, FIG_H_UNIT * 1.6),
                             sharex=True, gridspec_kw=dict(hspace=0.15))
    tx = np.arange(len(all_st))
    axes[0].plot(tx, pop_fr, color="#2980B9", lw=0.8)
    _ax_defaults(axes[0], "", "Pop. mean FR (Hz)", "Population firing rate")
    if motion_trace is not None:
        axes[1].plot(tx, motion_trace, color="#27AE60", lw=0.8)
        _ax_defaults(axes[1], "Trial index", "Motion (µm)", "Probe motion (DREDge)")
    else:
        axes[1].text(0.5, 0.5, "Motion unavailable", ha="center",
                     va="center", transform=axes[1].transAxes)
    fig.suptitle(f"{mouse_id} / {session_day} — drift sanity check")
    _save_fig(fig, out_dir / "drift_sanity", f"{mouse_id}_{session_day}_drift_sanity",
              config["figure_dpi"])


# ============================================================================
# MOTION ALIGNMENT DIAGNOSTIC
# ============================================================================

def plot_motion_alignment(unit_table, trial_table, session_id, mouse_id,
                          session_day, config, out_dir):
    """
    Verify that the single start_time motion sample correctly represents
    the neural activity epoch for every trial.

    Three panels per probe (imec):

    Panel A — Continuous DREDge motion trace + trial-epoch markers
        Grey shading: DREDge temporal bin width around each sampled point.
        Blue ticks  : start_time + baseline_window midpoint  (where FR is measured)
        Red ticks   : start_time + evoked_window midpoint
        Orange dots : the actual DREDge temporal-bin centre nearest to start_time
        Demonstrates that bin width >> epoch offsets, so one sample per trial is valid.

    Panel B — Histogram: |t_start − nearest_bin_centre| across all trials
        Should be << DREDge bin width (typically ~2 s).
        Red dashed line marks the bin width.

    Panel C — Histogram: DREDge bin width distribution
        Confirms bin width is uniform and much larger than the ~55 ms evoked window.

    Saved as motion_alignment/<mouse>_<day>_motion_alignment_imec<N>.[pdf|png]
    """
    trials  = (trial_table[trial_table["session_id"] == session_id]
               .sort_values("start_time").reset_index(drop=True))
    all_st  = trials["start_time"].to_numpy(float)
    units   = unit_table[unit_table["session_id"] == session_id]

    bw = config["baseline_window"]
    ew = config["evoked_window"]
    bw_mid = (bw[0] + bw[1]) / 2.0   # midpoint of baseline window rel. to start_time
    ew_mid = (ew[0] + ew[1]) / 2.0   # midpoint of evoked window

    # collect one (imec_id, depth, motion_object) per probe
    probes_seen = {}
    for _, u in units.iterrows():
        imec = _get_imec_id(u.get("electrode_group"))
        if imec is None or imec in probes_seen:
            continue
        try:
            mp = build_motion_path(config, mouse_id, session_id, imec)
            mo = load_motion(mp)
            probes_seen[imec] = (float(u["depth"]), mo)
        except Exception:
            continue

    if not probes_seen:
        warnings.warn(f"No motion objects found for {mouse_id}/{session_day}; "
                      "skipping motion alignment figure.")
        return

    subfolder = out_dir / "motion_alignment"

    for imec, (depth, mo) in probes_seen.items():
        # ── extract DREDge internal bin structure ────────────────────────────
        if hasattr(mo, "temporal_bins_s"):
            t_bins = np.asarray(mo.temporal_bins_s[0], dtype=float)
        else:
            # SpikeInterface >= 0.101 stores it differently; try attribute
            try:
                t_bins = np.asarray(mo.temporal_bins_s, dtype=float).ravel()
            except Exception:
                warnings.warn("Cannot access DREDge temporal_bins_s; skipping.")
                continue

        bin_edges    = (t_bins[:-1] + t_bins[1:]) / 2.0   # midpoints between centres
        bin_widths   = np.diff(t_bins)                      # (N_bins-1,) seconds

        # For each trial: nearest bin centre index and residual
        nearest_idx  = np.argmin(np.abs(t_bins[:, None] - all_st[None, :]), axis=0)
        nearest_time = t_bins[nearest_idx]
        residuals    = np.abs(all_st - nearest_time)       # |t_start - bin_centre|

        # Local bin width at each trial (use the bin the trial falls in)
        # clip so we don't go out of bounds for the last bin
        local_bw = bin_widths[np.minimum(nearest_idx, len(bin_widths) - 1)]

        # Sampled motion displacement at start_time (what the analysis actually uses)
        sampled_motion = get_motion_at_times_and_depth(mo, all_st, depth)

        # ── figure ───────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(FIG_W_FULL, FIG_H_UNIT * 3.5))
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.40,
                                height_ratios=[2.5, 1, 1])

        # Panel A: continuous motion + epoch markers
        ax_A  = fig.add_subplot(gs[0, :])
        n_show = min(len(all_st), 300)        # cap at 300 trials for clarity
        tx     = all_st[:n_show]
        mot_s  = sampled_motion[:n_show]

        # grey band = ± half local bin width around each sampled point
        half_bw = local_bw[:n_show] / 2.0
        ax_A.fill_between(tx, mot_s - half_bw * 0,   # band on time axis, not y
                          mot_s, alpha=0,              # invisible; we draw error bars below
                          )
        # horizontal error bars showing ± half bin width
        ax_A.errorbar(tx, mot_s, xerr=half_bw,
                      fmt="none", ecolor="#BBBBBB", elinewidth=0.6,
                      alpha=0.5, zorder=1, label="± ½ bin width")
        ax_A.plot(tx, mot_s, color="#27AE60", lw=0.8, zorder=3, label="Motion (sampled)")

        # epoch midpoint markers (vertical ticks below the trace)
        y_min = ax_A.get_ylim()[0] if ax_A.get_ylim()[0] != ax_A.get_ylim()[1] else -1
        for t_offset, color, lbl in [
            (bw_mid, "#2980B9", f"Baseline mid ({bw_mid*1000:.0f} ms)"),
            (ew_mid, "#E74C3C", f"Evoked mid (+{ew_mid*1000:.0f} ms)"),
        ]:
            ax_A.vlines(tx + t_offset, ymin=np.min(mot_s) - 0.3,
                        ymax=np.min(mot_s) - 0.05,
                        color=color, lw=0.6, alpha=0.7, label=lbl)

        ax_A.scatter(nearest_time[:n_show], mot_s, s=6, color="darkorange",
                     zorder=4, alpha=0.6, label="Nearest bin centre")
        ax_A.legend(fontsize=7, loc="upper right", ncol=2)
        _ax_defaults(ax_A, "Time (s)", "Motion (µm)",
                     f"Motion alignment — imec{imec}  (first {n_show} trials shown)")
        ax_A.set_xlim(tx[0] - 1, tx[-1] + 1)

        # Panel B: histogram of |t_start - nearest_bin_centre|
        ax_B = fig.add_subplot(gs[1, 0])
        typical_bw = float(np.median(bin_widths))
        ax_B.hist(residuals * 1e3, bins=40, color="#27AE60", edgecolor="none", alpha=0.8)
        ax_B.axvline(typical_bw * 1e3 / 2, color="crimson", lw=1.2, ls="--",
                     label=f"½ bin width ({typical_bw/2*1e3:.0f} ms)")
        ax_B.axvline(abs(ew_mid) * 1e3, color="#E74C3C", lw=1.0, ls=":",
                     label=f"Evoked mid ({abs(ew_mid)*1e3:.0f} ms)")
        ax_B.legend(fontsize=7)
        _ax_defaults(ax_B, "|t_start − bin centre|  (ms)",
                     "# trials", "Temporal alignment residual")

        # Panel C: bin-width distribution
        ax_C = fig.add_subplot(gs[1, 1])
        ax_C.hist(bin_widths, bins=30, color="#8E44AD", edgecolor="none", alpha=0.8)
        ax_C.axvline(abs(ew[1] - ew[0]) * 1e3 / 1e3, color="#E74C3C", lw=1.0, ls=":",
                     label=f"Evoked window ({(ew[1]-ew[0])*1e3:.0f} ms)")
        ax_C.legend(fontsize=7)
        _ax_defaults(ax_C, "DREDge bin width (s)", "# bins", "Bin width distribution")

        # Panel D: scatter — residual vs motion magnitude (check if errors cluster)
        ax_D = fig.add_subplot(gs[2, 0])
        ax_D.scatter(residuals * 1e3, np.abs(sampled_motion),
                     s=4, alpha=0.35, color="#27AE60", linewidths=0, rasterized=True)
        _ax_defaults(ax_D, "|t_start − bin centre|  (ms)",
                     "|Motion| (µm)", "Residual vs motion magnitude")

        # Panel E: fraction of trials within ½ bin width (text summary)
        ax_E = fig.add_subplot(gs[2, 1])
        ax_E.axis("off")
        frac_ok  = float((residuals < typical_bw / 2).mean())
        frac_ew  = float((residuals < abs(ew_mid)).mean())
        summary  = (
            f"DREDge bin width:  {typical_bw:.2f} s  (median)\n"
            f"Evoked window:     {(ew[1]-ew[0])*1e3:.0f} ms\n"
            f"Baseline midpoint: {bw_mid*1e3:.0f} ms re. start\n"
            f"Evoked midpoint:   {ew_mid*1e3:.1f} ms re. start\n\n"
            f"Trials within ½ bin width: {frac_ok*100:.1f}%\n"
            f"Max residual: {residuals.max()*1e3:.1f} ms\n"
            f"Median residual: {np.median(residuals)*1e3:.1f} ms\n\n"
            f"Conclusion: bin width ({typical_bw:.1f} s) >> evoked\n"
            f"window ({(ew[1]-ew[0])*1e3:.0f} ms).  Single start_time\n"
            f"sample is adequate for both epochs."
        )
        ax_E.text(0.05, 0.95, summary, transform=ax_E.transAxes,
                  va="top", ha="left", fontsize=8.5,
                  fontfamily="monospace",
                  bbox=dict(facecolor="#F8F8F8", edgecolor="#CCCCCC",
                            boxstyle="round,pad=0.4"))

        fig.suptitle(
            f"{mouse_id} / {session_day} — Motion temporal alignment  (imec{imec})\n"
            f"Confirms DREDge sample at start_time covers both baseline and evoked epochs",
            y=1.01)
        _save_fig(fig, subfolder,
                  f"{mouse_id}_{session_day}_motion_alignment_imec{imec}",
                  config["figure_dpi"])


# ============================================================================
# SINGLE-NEURON EXAMPLE FIGURE
# ============================================================================

def plot_single_neuron_examples(results, unit_table, trial_table,
                                session_id, mouse_id, session_day,
                                config, out_dir, rng: np.random.Generator):
    """
    Per session: show n_example_neurons neurons (mix of quadrants) as rows.
    Columns: baseline FR | evoked FR | learning curve | motion
    Saved as single_neuron_examples.pdf/png.
    """
    n_ex  = config["n_example_neurons"]
    use_p = config["use_p"]
    N     = config["shift_N"]

    trials  = (trial_table[trial_table["session_id"] == session_id]
               .sort_values("start_time").reset_index(drop=True))
    wt_mask = ((trials["trial_type"] == "whisker_trial") &
               (trials["context"] != "passive"))
    all_st  = trials["start_time"].to_numpy(float)
    wh_st   = trials.loc[wt_mask, "start_time"].to_numpy(float)
    units   = unit_table[unit_table["session_id"] == session_id].set_index("unit_id")

    lc_path = build_learning_curve_path(config, mouse_id, session_day)
    learning_curve = None
    if lc_path.exists():
        try:
            learning_curve = np.asarray(load_learning_curve(lc_path)["p_mean"][0], dtype=float)
        except Exception:
            pass

    # pick a balanced sample across quadrants
    ev_lc = results[(results["epoch"] == "evoked") &
                    (results["factor"] == "learning_curve") &
                    results["quadrant"].notna()].copy()
    chosen_ids = []
    for q in QUAD_ORDER:
        pool = ev_lc[ev_lc["quadrant"] == q]["unit_id"].unique()
        n_q  = max(1, n_ex // len(QUAD_ORDER))
        chosen_ids.extend(rng.choice(pool, size=min(n_q, len(pool)), replace=False).tolist())
    chosen_ids = chosen_ids[:n_ex]
    if not chosen_ids:
        return

    n_rows = len(chosen_ids)
    fig, axes = plt.subplots(n_rows, 4,
                             figsize=(FIG_W_FULL, FIG_H_UNIT * n_rows),
                             gridspec_kw=dict(wspace=0.38, hspace=0.65),
                             squeeze=False)
    col_titles = ["Baseline FR (Hz)\n(all trials)",
                 "Evoked FR (Hz)\n(whisker trials)",
                 "Learning curve\n(r_LC below: trial-indexed shift test)",
                 "Motion @ whisker trials (µm)\n(display only — r_motion uses a\n"
                 "different test, see label)"]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=8)

    for ri, uid in enumerate(chosen_ids):
        try:
            unit = units.loc[uid]
        except KeyError:
            continue
        spk  = np.asarray(unit["spike_times"], dtype=float)
        dep  = float(unit["depth"])
        imec = _get_imec_id(unit.get("electrode_group"))
        area = unit.get("area_custom_acronym", unit.get("area_acronym_custom", "?"))
        bc   = unit.get("bc_label", "?")

        fr_base = compute_firing_rate_per_trial(spk, all_st, config["baseline_window"])
        fr_evok = compute_firing_rate_per_trial(spk, wh_st,  config["evoked_window"])

        n_wh = len(wh_st)
        lc   = learning_curve[:n_wh] if learning_curve is not None else None

        mot = None
        if imec is not None:
            try:
                mp  = build_motion_path(config, mouse_id, session_id, imec)
                mo  = load_motion(mp)
                mot = get_motion_at_times_and_depth(mo, wh_st, dep)
            except Exception:
                pass

        # retrieve test results for annotation -- two DIFFERENT tests feed
        # this figure: r_LC is the trial-indexed shift test (evoked epoch,
        # whisker trials, vs learning curve); r_motion is the time-binned
        # drift test (baseline epoch, continuous 1s bins over the whole
        # session, vs DREDge displacement). They are not directly comparable
        # and are NOT computed on the traces shown in the same column position.
        row_ev_lc = results[(results["unit_id"] == uid) &
                            (results["epoch"] == "evoked") &
                            (results["factor"] == "learning_curve")]
        row_bs_mo = results[(results["unit_id"] == uid) &
                            (results["epoch"] == "baseline") &
                            (results["factor"] == "motion")]

        q_label = row_ev_lc.iloc[0]["quadrant"] if not row_ev_lc.empty else "?"
        r_lc    = row_ev_lc.iloc[0]["r"]        if not row_ev_lc.empty else np.nan
        p_lc    = row_ev_lc.iloc[0][use_p]      if not row_ev_lc.empty else np.nan
        q_lc    = row_ev_lc.iloc[0]["p_fdr"]    if not row_ev_lc.empty else np.nan
        sig_lc  = bool(row_ev_lc.iloc[0]["significant"]) if not row_ev_lc.empty else False
        pr_val  = row_ev_lc.iloc[0]["partial_r"] if not row_ev_lc.empty else np.nan
        peak_l  = row_ev_lc.iloc[0]["peak_lag"] if not row_ev_lc.empty else np.nan

        r_mot   = row_bs_mo.iloc[0]["r"]        if not row_bs_mo.empty else np.nan
        p_mot   = row_bs_mo.iloc[0]["p_conservative"] if not row_bs_mo.empty else np.nan
        q_mot   = row_bs_mo.iloc[0]["p_fdr"]    if not row_bs_mo.empty else np.nan
        sig_mot = bool(row_bs_mo.iloc[0]["significant"]) if not row_bs_mo.empty else False
        pr_mot  = row_bs_mo.iloc[0]["partial_r"] if not row_bs_mo.empty else np.nan

        # 'significant' (and thus 'quadrant') is decided by the within-session
        # BH-FDR corrected q, not the raw p shown here -- raw p kept for
        # reference only. partial_r(LC|motion) uses the evoked-epoch r_LM;
        # partial_r(motion|LC) uses the baseline-epoch r_LM -- each factor's
        # partial_r matches its own row's epoch, not a shared one.
        row_label = (
            f"uid={uid} | {bc} | {area}\n"
            f"quadrant={q_label}\n"
            f"r_LC(evoked, shift-test)={r_lc:.2f}  raw p={p_lc:.3f}  "
            f"FDR-q={q_lc:.3f}  {'Sig' if sig_lc else 'n.s.'}\n"
            f"partial_r(LC | motion)={pr_val:.2f}  peak_lag={peak_l:.0f} trials\n"
            f"r_motion(baseline, time-binned)={r_mot:.2f}  "
            f"raw p={p_mot:.3f}  FDR-q={q_mot:.3f}  "
            f"{'Sig' if sig_mot else 'n.s.'}\n"
            f"partial_r(motion | LC)={pr_mot:.2f}"
        )
        qcol       = QUAD_COLORS.get(q_label, "grey")

        tx_all = np.arange(len(all_st))
        tx_wh  = np.arange(n_wh)

        for ci, (tx, data, ec) in enumerate([
            (tx_all, fr_base, "#2980B9"),
            (tx_wh,  fr_evok, "#E74C3C"),
            (tx_wh,  lc,      "#8E44AD"),
            (tx_wh,  mot,     "#27AE60"),
        ]):
            ax = axes[ri, ci]
            if data is not None:
                n = min(len(tx), len(data))
                ax.plot(tx[:n], np.asarray(data[:n], dtype=float),
                        color=ec, lw=0.85, alpha=0.85)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="grey")
            ax.tick_params(labelbottom=(ri == n_rows - 1), labelleft=True)
            if ci == 0:
                ax.set_ylabel(f"unit {uid}", fontsize=7, color=qcol)
                # side label
                ax.text(-0.34, 0.5, row_label, transform=ax.transAxes,
                        va="center", ha="right", fontsize=6, color=qcol,
                        rotation=0, wrap=True)
        if ri == n_rows - 1:
            for ci, xl in enumerate(["Trial (all)", "Trial (whisker)",
                                     "Trial (whisker)", "Trial (whisker)"]):
                axes[ri, ci].set_xlabel(xl, fontsize=8)

    fig.suptitle(f"{mouse_id} / {session_day} — single-neuron examples", y=1.01)
    fig.text(0.5, -0.01,
             "r_LC: trial-indexed shift test, evoked FR (whisker trials) vs. "
             "learning curve. r_motion: time-binned drift test, baseline-epoch "
             "FR (continuous 1s bins, whole session) vs. DREDge displacement. "
             "The 'Motion' column plots displacement at whisker-trial times for "
             "visual reference only -- it is not the series r_motion was computed on.",
             ha="center", fontsize=6.5, color="grey", wrap=True)
    _save_fig(fig, out_dir / "single_neuron_examples",
              f"{mouse_id}_{session_day}_single_neuron_examples",
              config["figure_dpi"])


# ============================================================================
# PROCESS FIGURE  (shift-test mechanics)
# ============================================================================

def plot_process_figure(results, unit_table, trial_table,
                        session_id, mouse_id, session_day,
                        config, out_dir, learning_curve, motion_cache):
    rng   = np.random.default_rng(config["random_seed"])
    N     = config["shift_N"]
    use_p = config["use_p"]

    trials  = (trial_table[trial_table["session_id"] == session_id]
               .sort_values("start_time").reset_index(drop=True))
    wt_mask = ((trials["trial_type"] == "whisker_trial") &
               (trials["context"] != "passive"))
    all_st  = trials["start_time"].to_numpy(float)
    wh_st   = trials.loc[wt_mask, "start_time"].to_numpy(float)
    units   = unit_table[unit_table["session_id"] == session_id].set_index("unit_id")

    t_start = float(all_st.min()) if len(all_st) else np.nan
    t_end   = float(all_st.max()) if len(all_st) else np.nan

    for factor in ["motion", "learning_curve"]:
        epoch_use = "evoked" if factor == "learning_curve" else "baseline"
        window    = (config["evoked_window"] if epoch_use == "evoked"
                     else config["baseline_window"])

        sub = results[(results["epoch"] == epoch_use) &
                      (results["factor"] == factor)].dropna(subset=[use_p])
        if factor == "learning_curve":
            # exclude ambiguous/drift from example
            sub = sub[sub["quadrant"].isin(["clean_learner", "unresponsive"])]

        chosen = []
        n_examples = config.get("n_process_examples_per_group", 6)
        for pool in [sub[sub["significant"]], sub[~sub["significant"]]]:
            if len(pool):
                k    = min(n_examples, len(pool))
                samp = pool.sample(n=k, random_state=int(rng.integers(1e6)))
                chosen.extend(samp.iloc[i] for i in range(k))
        if not chosen:
            continue

        n_rows = len(chosen)
        fig, axes = plt.subplots(n_rows, 5,
                                 figsize=(FIG_W_FULL * 1.25, FIG_H_UNIT * n_rows),
                                 gridspec_kw=dict(wspace=0.55, hspace=0.55),
                                 squeeze=False)

        for ri, nrow in enumerate(chosen):
            uid = nrow["unit_id"]
            try:
                unit = units.loc[uid]
            except KeyError:
                continue
            spk = np.asarray(unit["spike_times"], dtype=float)

            if factor == "motion":
                # mirror analyze_session's time-binned drift test exactly:
                # same binning, same event/window rule, same seed -- so this
                # panel shows the actual test that produced nrow['significant'].
                imec = _get_imec_id(unit.get("electrode_group"))
                try:
                    mp = build_motion_path(config, mouse_id, session_id, imec)
                    mo = load_motion(mp)
                except Exception:
                    continue
                bin_width = config["drift_test_bin_width_baseline"]
                n_shifts  = config["drift_test_n_shifts_baseline"]
                tx, act = compute_binned_firing_rate(spk, t_start, t_end, bin_width)
                if len(tx) == 0 or np.isnan(t_start) or np.isnan(t_end):
                    continue
                try:
                    fv = get_motion_at_times_and_depth(mo, tx, float(unit["depth"]))
                except Exception:
                    continue
                fv = np.asarray(fv, dtype=float)
                valid = np.isfinite(act) & np.isfinite(fv)
                tx, act, fv = tx[valid], act[valid], fv[valid]
                f_lbl, x_lbl = "Motion (µm)", "Time (s)"
            else:
                if learning_curve is None:
                    continue
                act = compute_firing_rate_per_trial(spk, wh_st, window)
                fv  = learning_curve[:len(act)]
                n   = min(len(act), len(fv))
                act = act[:n]; fv = np.asarray(fv[:n], dtype=float)
                tx  = np.arange(n)
                f_lbl, x_lbl = "Learning curve", "Trial"

            n = len(act)
            if n < 3:
                continue
            col = EPOCH_COLORS[epoch_use]

            axes[ri, 0].plot(tx, act, color=col, lw=0.9, alpha=0.85)
            _ax_defaults(axes[ri, 0], x_lbl, "FR (Hz)",
                         f"{'Sig' if nrow['significant'] else 'n.s.'} — unit {uid}")

            axes[ri, 1].plot(tx, fv, color=FACTOR_COLORS[factor], lw=0.9)
            _ax_defaults(axes[ri, 1], x_lbl, f_lbl, f_lbl)

            # aligned overlay: FR and factor share the x-axis (trial or time),
            # separate y-axes, so co-drift over the session is visible directly
            ax_ov  = axes[ri, 2]
            ax_ov2 = ax_ov.twinx()
            l1, = ax_ov.plot(tx, act, color=col, lw=0.9, alpha=0.85, label="FR")
            l2, = ax_ov2.plot(tx, fv, color=FACTOR_COLORS[factor], lw=0.9,
                              alpha=0.85, label=f_lbl)
            ax_ov.set_ylabel("FR (Hz)", color=col)
            ax_ov.tick_params(axis="y", labelcolor=col)
            ax_ov2.set_ylabel(f_lbl, color=FACTOR_COLORS[factor])
            ax_ov2.tick_params(axis="y", labelcolor=FACTOR_COLORS[factor])
            ax_ov2.spines["top"].set_visible(False)
            ax_ov.set_xlabel(x_lbl)
            ax_ov.set_title("FR vs. " + f_lbl.split(" (")[0] + " (aligned)")
            ax_ov.legend(handles=[l1, l2], fontsize=6, loc="upper right")

            axes[ri, 3].scatter(fv, act, s=5, alpha=0.4, color=col,
                                linewidths=0, rasterized=True)
            if n > 2:
                sl, ic, *_ = stats.linregress(fv, act)
                xr = np.array([fv.min(), fv.max()])
                axes[ri, 3].plot(xr, sl * xr + ic, "k--", lw=1.1)
            q_row = nrow["p_fdr"] if "p_fdr" in nrow and np.isfinite(nrow["p_fdr"]) else np.nan
            q_str = f"  FDR-q={q_row:.3f}" if np.isfinite(q_row) else ""
            _ax_defaults(axes[ri, 3], f_lbl, "FR (Hz)",
                         f"r={nrow['r']:.3f}  raw p={nrow[use_p]:.3f}{q_str}")

            ax4 = axes[ri, 4]
            if factor == "motion":
                # shift-randomized null: histogram of null |r|, observed r
                # marked -- this is the actual test behind 'significant'.
                seed = _drift_test_seed(config.get("random_seed", 0),
                                        session_id, uid, epoch_use)
                rng4 = np.random.default_rng(seed)
                method = config.get("drift_test_corr_method", "pearson")
                sfrac  = config.get("drift_test_shift_frac", (0.10, 0.90))
                r_obs, p_val, m, null_vals = shift_null_test(
                    act, fv, n_shifts, rng4, method=method,
                    shift_frac=sfrac, return_null=True)
                nv = null_vals[np.isfinite(null_vals)]
                if nv.size:
                    ax4.hist(nv, bins=30, color="steelblue", alpha=0.7)
                    ax4.axvline(r_obs, color=col, lw=1.8,
                               label=f"obs r={r_obs:.3f}")
                    ax4.axvline(-r_obs, color=col, lw=1.0, ls=":", alpha=0.6)
                    q = nrow["p_fdr"] if "p_fdr" in nrow and np.isfinite(nrow["p_fdr"]) else np.nan
                    q_str = f", q={q:.3f}" if np.isfinite(q) else ""
                    ax4.legend(fontsize=7)
                    _ax_defaults(ax4, "Null shift r", "Count",
                                f"Shift null (p={p_val:.3f}{q_str}, m={m})")
                else:
                    ax4.text(0.5, 0.5, "unavailable", ha="center",
                             va="center", transform=ax4.transAxes)
            else:
                # trial-indexed shift-score curve (unchanged: learning-curve
                # rows still use the original trial-indexed shift test)
                try:
                    res_full = shift_test(act, fv, N)
                    scores   = np.asarray(res_full["scores"])
                    shifts   = np.arange(-N, N + 1)
                    ax4.plot(shifts, scores, color="steelblue", lw=1.0)
                    ax4.axvline(0, color="k", lw=0.8, ls="--")
                    peak_idx = int(np.argmax(np.abs(scores)))
                    ax4.scatter([shifts[peak_idx]], [scores[peak_idx]],
                                color="darkorange", s=35, zorder=5,
                                label=f"peak lag={shifts[peak_idx]:+d}")
                    ax4.scatter([0], [scores[N]], color=col, s=30, zorder=5)
                    ax4.legend(fontsize=7)
                    _ax_defaults(ax4, "Shift (trials)", "|Pearson r|", "Shift-test scores")
                except Exception:
                    ax4.text(0.5, 0.5, "unavailable", ha="center",
                             va="center", transform=ax4.transAxes)

        fig.suptitle(f"{mouse_id} / {session_day} — shift-test process: "
                     f"{epoch_use} vs {factor}", y=1.02)
        stem = f"{mouse_id}_{session_day}_process_{'motion' if factor=='motion' else 'lc'}"
        _save_fig(fig, out_dir / "process_figures", stem, config["figure_dpi"])


# ============================================================================
# CROSS-CORRELATION FIGURE
# ============================================================================

def plot_cross_correlation_by_area(results, unit_table, trial_table,
                                    session_id, mouse_id, session_day,
                                    config, out_dir, learning_curve):
    """
    For each brain area: mean |Pearson r| at every shift lag for
    motion and learning curve side-by-side.
    Uses shift_test_many for vectorised computation over all neurons in the area.
    """
    N     = config["shift_N"]
    shifts = np.arange(-N, N + 1)

    trials  = (trial_table[trial_table["session_id"] == session_id]
               .sort_values("start_time").reset_index(drop=True))
    wt_mask = ((trials["trial_type"] == "whisker_trial") &
               (trials["context"] != "passive"))
    all_st  = trials["start_time"].to_numpy(float)
    wh_st   = trials.loc[wt_mask, "start_time"].to_numpy(float)
    units   = unit_table[unit_table["session_id"] == session_id].copy()
    areas   = sorted(units["area"].dropna().unique()
                     if "area" in units.columns else [])
    if not areas:
        return

    # build activity matrices (whisker trials) per area
    area_data: dict = {}  # area -> {"fr": (T, C), "motion": (T,), "lc": (T,)}
    for area in areas:
        au = units[units.get("area_custom_acronym",
                   units.get("area_acronym_custom", pd.Series())).fillna("") == area
                   if ("area_custom_acronym" in units.columns or
                       "area_acronym_custom" in units.columns) else pd.Series(False, index=units.index)]
        if "area_custom_acronym" in units.columns:
            au = units[units["area_custom_acronym"].fillna("") == area]
        elif "area_acronym_custom" in units.columns:
            au = units[units["area_acronym_custom"].fillna("") == area]
        else:
            continue
        if au.empty:
            continue

        fr_cols = []
        for _, u in au.iterrows():
            spk = np.asarray(u["spike_times"], dtype=float)
            fr_cols.append(compute_firing_rate_per_trial(spk, wh_st,
                                                          config["evoked_window"]))
        if not fr_cols:
            continue
        fr_mat = np.stack(fr_cols, axis=1)  # (T, C)

        # motion: use depth of first unit with a valid probe
        mot_trace = None
        for _, u in au.iterrows():
            imec = _get_imec_id(u.get("electrode_group"))
            if imec is None:
                continue
            try:
                mp  = build_motion_path(config, mouse_id, session_id, imec)
                mo  = load_motion(mp)
                mot_trace = get_motion_at_times_and_depth(mo, wh_st, float(u["depth"]))
                break
            except Exception:
                continue

        n_lc = len(wh_st)
        lc   = learning_curve[:n_lc] if learning_curve is not None else None
        area_data[area] = {"fr": fr_mat, "motion": mot_trace, "lc": lc}

    if not area_data:
        return

    n_areas = len(area_data)
    fig, axes = plt.subplots(n_areas, 2,
                             figsize=(FIG_W_DOUBLE, FIG_H_UNIT * n_areas),
                             gridspec_kw=dict(wspace=0.35, hspace=0.55),
                             squeeze=False)
    axes[0, 0].set_title("vs Motion", fontsize=9)
    axes[0, 1].set_title("vs Learning curve", fontsize=9)

    for ri, (area, dat) in enumerate(area_data.items()):
        fr_mat = dat["fr"]
        T, C   = fr_mat.shape
        if T <= 2 * N or C == 0:
            for ci in range(2):
                axes[ri, ci].text(0.5, 0.5, "insufficient data",
                                  ha="center", va="center",
                                  transform=axes[ri, ci].transAxes, fontsize=7)
            axes[ri, 0].set_ylabel(area, fontsize=8)
            continue

        for ci, (ref, col, factor_lbl) in enumerate([
            (dat["motion"], FACTOR_COLORS["motion"], "motion"),
            (dat["lc"],     FACTOR_COLORS["learning_curve"], "learning_curve"),
        ]):
            ax = axes[ri, ci]
            ax.set_ylabel(area if ci == 0 else "", fontsize=8)
            if ref is None or len(ref) < T:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
                continue
            ref_arr = np.asarray(ref[:T], dtype=float)
            if np.all(ref_arr == ref_arr[0]):
                ax.text(0.5, 0.5, "constant", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
                continue
            # vectorised cross-correlation across all neurons in area
            try:
                res_many = shift_test_many(ref_arr, fr_mat, N,
                                           shift_ref=True)
                # scores: (2N+1, C), take mean |r| across neurons
                mean_xcorr = np.nanmean(np.abs(res_many["scores"]), axis=1)
            except Exception:
                ax.text(0.5, 0.5, "error", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
                continue

            ax.fill_between(shifts, mean_xcorr, alpha=0.25, color=col)
            ax.plot(shifts, mean_xcorr, color=col, lw=1.1)
            ax.axvline(0, color="k", lw=0.8, ls="--")
            # mark peak lag
            peak_i = int(np.argmax(mean_xcorr))
            ax.scatter([shifts[peak_i]], [mean_xcorr[peak_i]],
                       color="darkorange", s=30, zorder=5,
                       label=f"peak={shifts[peak_i]:+d}")
            ax.legend(fontsize=7, loc="upper right")
            ax.set_xlim(-N, N)
            _ax_defaults(ax,
                         "Lag (trials)" if ri == n_areas - 1 else "",
                         "Mean |r|", "")

    fig.suptitle(f"{mouse_id} / {session_day} — cross-correlation by area", y=1.01)
    _save_fig(fig, out_dir / "cross_correlation",
              f"{mouse_id}_{session_day}_xcorr_by_area", config["figure_dpi"])


# ============================================================================
# PER-SESSION SUMMARY
# ============================================================================

def plot_per_session_summary(results, config, mouse_id, session_day, out_dir):
    use_p = config["use_p"]
    sub   = out_dir / "per_session_summary"

    TESTS  = [("baseline","motion"), ("evoked","motion"),
              ("baseline","learning_curve"), ("evoked","learning_curve")]
    TLBLS  = ["Baseline\nvs Motion", "Evoked\nvs Motion",
               "Baseline\nvs LC", "Evoked\nvs LC"]
    TCOLORS= [FACTOR_COLORS["motion"], FACTOR_COLORS["motion"],
               FACTOR_COLORS["learning_curve"], FACTOR_COLORS["learning_curve"]]

    # ── distributions ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_FULL, FIG_H_SQUARE),
                             gridspec_kw=dict(wspace=0.40))
    vals_list = [prepare_clean(results, use_p, epoch=ep, factor=fa)["r"].dropna().values
                 for ep, fa in TESTS]

    ax = axes[0]
    for i, (v, c) in enumerate(zip(vals_list, TCOLORS)):
        _strip_mean_sem(ax, i, v, c)
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xticks(range(4)); ax.set_xticklabels(TLBLS, rotation=20, ha="right")
    _ax_defaults(ax, "", "Pearson r", "Signed r")

    ax = axes[1]
    for i, (v, c) in enumerate(zip(vals_list, TCOLORS)):
        _strip_mean_sem(ax, i, np.abs(v), c)
    ax.set_xticks(range(4)); ax.set_xticklabels(TLBLS, rotation=20, ha="right")
    _ax_defaults(ax, "", "|Pearson r|", "|r|")

    ax = axes[2]
    fracs = [prepare_clean(results, use_p, epoch=ep, factor=fa)["significant"].mean()
             if len(prepare_clean(results, use_p, epoch=ep, factor=fa)) else 0
             for ep, fa in TESTS]
    ax.bar(range(4), fracs, color=TCOLORS, edgecolor="k", lw=0.7, width=0.6)
    ax.axhline(SIG_ALPHA, color="crimson", ls="--", lw=0.9, label=f"α={SIG_ALPHA}")
    ax.legend(); ax.set_ylim(0, max(max(fracs) * 1.25, SIG_ALPHA * 2))
    ax.set_xticks(range(4)); ax.set_xticklabels(TLBLS, rotation=20, ha="right")
    _ax_defaults(ax, "", "Fraction sig", "Fraction significant")

    fig.suptitle(f"{mouse_id} / {session_day}")
    _save_fig(fig, sub, f"{mouse_id}_{session_day}_distributions", config["figure_dpi"])

    # ── quadrant pie / bar ───────────────────────────────────────────────────
    ev_lc = prepare_clean(results, use_p, epoch="evoked", factor="learning_curve")
    ev_lc = ev_lc.dropna(subset=["quadrant"])
    if not ev_lc.empty:
        counts = [ev_lc["quadrant"].value_counts().get(q, 0) for q in QUAD_ORDER]
        colors = [QUAD_COLORS[q] for q in QUAD_ORDER]
        labels = [QUAD_LABELS[q] for q in QUAD_ORDER]
        fig, ax = plt.subplots(figsize=(FIG_W_SINGLE + 1, FIG_H_SQUARE))
        ax.bar(range(4), counts, color=colors, edgecolor="k", lw=0.7, width=0.7)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, rotation=15, ha="right",
                                                      fontsize=7.5)
        _ax_defaults(ax, "", "# neurons", "Quadrant classification (evoked vs LC)")
        fig.suptitle(f"{mouse_id} / {session_day}")
        _save_fig(fig, sub, f"{mouse_id}_{session_day}_quadrant_bar", config["figure_dpi"])

    # ── peak lag distribution ────────────────────────────────────────────────
    for (ep, fa), lbl, col in [
        (("evoked", "learning_curve"), "Evoked vs LC", FACTOR_COLORS["learning_curve"]),
        (("evoked", "motion"),         "Evoked vs Motion", FACTOR_COLORS["motion"]),
    ]:
        sub_df = prepare_clean(results, use_p, epoch=ep, factor=fa)
        if sub_df.empty or "peak_lag" not in sub_df.columns:
            continue
        pl = sub_df["peak_lag"].dropna().values
        if len(pl) < 3:
            continue
        fig, ax = plt.subplots(figsize=(FIG_W_SINGLE + 0.5, FIG_H_SQUARE))
        N = config["shift_N"]
        bins = np.arange(-N - 0.5, N + 1.5, 1)
        ax.hist(pl, bins=bins, color=col, edgecolor="k", lw=0.5)
        ax.axvline(0, color="k", lw=1.2, ls="--", label="lag=0")
        ax.legend()
        _ax_defaults(ax, "Peak lag (trials)", "# neurons",
                     f"Peak-lag distribution — {lbl}")
        fig.suptitle(f"{mouse_id} / {session_day}")
        _save_fig(fig, sub,
                  f"{mouse_id}_{session_day}_peak_lag_{ep}_{fa}",
                  config["figure_dpi"])


# ============================================================================
# GRAND SUMMARY
# ============================================================================

def plot_grand_summary(combined, config, out_dir, trial_table):
    use_p = config["use_p"]
    dpi   = config["figure_dpi"]
    clean = add_performance_tertile(combined, trial_table)
    sub_dir = out_dir / "grand_summary"
    sub_dir.mkdir(parents=True, exist_ok=True)

    rg_vals   = sorted(clean["reward_group"].dropna().unique())
    rg_labels = [REWARD_LABELS.get(int(r), str(r)) for r in rg_vals]
    rg_colors = [REWARD_COLORS.get(int(r), "grey") for r in rg_vals]
    areas     = sorted(clean["area"].dropna().unique())
    acolors   = _area_color_map(areas)

    TESTS  = [("baseline","motion"), ("evoked","motion"),
              ("baseline","learning_curve"), ("evoked","learning_curve")]
    TLBLS  = ["Baseline\nvs Motion", "Evoked\nvs Motion",
               "Baseline\nvs LC", "Evoked\nvs LC"]
    TCOLORS= [FACTOR_COLORS["motion"], FACTOR_COLORS["motion"],
               FACTOR_COLORS["learning_curve"], FACTOR_COLORS["learning_curve"]]

    for bc in ["both", "good", "mua"]:

        # ── 01: overall distributions ──────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(FIG_W_FULL, FIG_H_SQUARE),
                                 gridspec_kw=dict(wspace=0.42))
        vals_list = [prepare_clean(clean, use_p, bc=bc, epoch=ep, factor=fa)
                     ["r"].dropna().values for ep, fa in TESTS]
        ax = axes[0]
        for i, (v, c) in enumerate(zip(vals_list, TCOLORS)):
            _strip_mean_sem(ax, i, v, c)
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ymax = ax.get_ylim()[1] or 0.5
        for i, v in enumerate(vals_list):
            p = _one_sample_wilcoxon(v)
            lbl = _stat_label(p)
            if lbl and lbl != "ns":
                ax.text(i, ymax * 1.05, lbl, ha="center", fontsize=9)
        ax.set_xticks(range(4)); ax.set_xticklabels(TLBLS, rotation=20, ha="right")
        _ax_defaults(ax, "", "Pearson r", "Signed r")
        for i, (v, c) in enumerate(zip(vals_list, TCOLORS)):
            _strip_mean_sem(axes[1], i, np.abs(v), c)
        axes[1].set_xticks(range(4)); axes[1].set_xticklabels(TLBLS, rotation=20, ha="right")
        _ax_defaults(axes[1], "", "|r|", "|r|")
        fracs = [prepare_clean(clean, use_p, bc=bc, epoch=ep, factor=fa)
                 ["significant"].mean() for ep, fa in TESTS]
        axes[2].bar(range(4), fracs, color=TCOLORS, edgecolor="k", lw=0.7, width=0.6)
        axes[2].axhline(SIG_ALPHA, color="crimson", ls="--", lw=0.9)
        axes[2].set_xticks(range(4)); axes[2].set_xticklabels(TLBLS, rotation=20, ha="right")
        _ax_defaults(axes[2], "", "Fraction sig", "Fraction sig")
        fig.suptitle(f"Overall distributions (bc={bc})")
        _save_fig(fig, sub_dir, f"01_overall_distributions_bc{bc}", dpi)

        # ── 02: by reward group ────────────────────────────────────────────
        ev_lc = prepare_clean(clean, use_p, bc=bc, epoch="evoked", factor="learning_curve")
        fig, axes = plt.subplots(1, 3, figsize=(FIG_W_TRIPLE, FIG_H_SQUARE),
                                 gridspec_kw=dict(wspace=0.42))
        vals = [ev_lc[ev_lc["reward_group"] == rg]["r"].dropna().values for rg in rg_vals]
        for i, (v, c) in enumerate(zip(vals, rg_colors)):
            _strip_mean_sem(axes[0], i, v, c)
        axes[0].axhline(0, color="k", lw=0.7, ls="--")
        axes[0].set_xticks(range(len(rg_vals))); axes[0].set_xticklabels(rg_labels)
        _ax_defaults(axes[0], "Reward group", "Pearson r", "Signed r — evoked vs LC")
        if len(rg_vals) == 2 and all(len(v) > 2 for v in vals):
            _, p = stats.mannwhitneyu(*vals, alternative="two-sided")
            _, p = stats.ttest_ind(*vals, equal_var=False)
            _sig_bracket(axes[0], 0, 1, axes[0].get_ylim()[1], _stat_label(p))
        for i, (v, c) in enumerate(zip(vals, rg_colors)):
            _strip_mean_sem(axes[1], i, np.abs(v), c)
        axes[1].set_xticks(range(len(rg_vals))); axes[1].set_xticklabels(rg_labels)
        _ax_defaults(axes[1], "Reward group", "|r|", "|r| — evoked vs LC")
        fracs = [ev_lc[ev_lc["reward_group"] == rg]["significant"].mean()
                 if len(ev_lc[ev_lc["reward_group"] == rg]) else 0 for rg in rg_vals]
        axes[2].bar(range(len(rg_vals)), fracs, color=rg_colors,
                    edgecolor="k", lw=0.7, width=0.5)
        axes[2].axhline(SIG_ALPHA, color="crimson", ls="--", lw=0.9)
        axes[2].set_xticks(range(len(rg_vals))); axes[2].set_xticklabels(rg_labels)
        _ax_defaults(axes[2], "Reward group", "Fraction sig", "Fraction sig")
        fig.suptitle(f"By reward group — evoked vs LC (bc={bc})")
        _save_fig(fig, sub_dir, f"02_by_reward_group_bc{bc}", dpi)

        # ── 03: quadrant fractions by area ────────────────────────────────
        ev_lc_q = ev_lc.dropna(subset=["quadrant"])
        if not ev_lc_q.empty and areas:
            fig, ax = plt.subplots(figsize=(max(FIG_W_FULL, len(areas) * 1.2), FIG_H_SQUARE))
            bar_w = 0.7 / len(QUAD_ORDER)
            for qi, q in enumerate(QUAD_ORDER):
                fracs_q = [(ev_lc_q[(ev_lc_q["area"] == a) &
                                    (ev_lc_q["quadrant"] == q)].shape[0] /
                            max(ev_lc_q[ev_lc_q["area"] == a].shape[0], 1))
                           for a in areas]
                offsets = np.arange(len(areas)) + (qi - len(QUAD_ORDER)/2 + 0.5) * bar_w
                ax.bar(offsets, fracs_q, width=bar_w * 0.9,
                       color=QUAD_COLORS[q], edgecolor="k", lw=0.5,
                       label=QUAD_LABELS[q])
            ax.set_xticks(range(len(areas))); ax.set_xticklabels(areas, rotation=40, ha="right")
            ax.set_ylim(0, 1.05)
            ax.legend(loc="upper right", fontsize=7)
            _ax_defaults(ax, "Brain area", "Fraction", f"Quadrant fractions by area (bc={bc})")
            fig.tight_layout()
            _save_fig(fig, sub_dir, f"03_quadrant_by_area_bc{bc}", dpi)

        # ── 04: by performance tertile ────────────────────────────────────
        tvals = [0.0, 1.0, 2.0]; tlbls = ["Low perf", "Mid perf", "High perf"]
        tcols = ["#E08080", "#8080C0", "#60C060"]
        fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H_SQUARE),
                                 gridspec_kw=dict(wspace=0.42))
        for ax_i, (use_abs, lbl) in enumerate([(False, "Signed r"), (True, "|r|")]):
            ax = axes[ax_i]
            ev_t = prepare_clean(clean, use_p, bc=bc, epoch="evoked", factor="learning_curve")
            for i, (t, c) in enumerate(zip(tvals, tcols)):
                v = ev_t[ev_t.get("perf_tertile", pd.Series(np.nan, index=ev_t.index)) == t
                          ]["r"].dropna().values if "perf_tertile" in ev_t.columns else np.array([])
                _strip_mean_sem(ax, i, np.abs(v) if use_abs else v, c)
            ax.axhline(0, color="k", lw=0.7, ls="--")
            ax.set_xticks(range(3)); ax.set_xticklabels(tlbls, rotation=20, ha="right")
            _ax_defaults(ax, "", lbl, f"{lbl} — perf tertile")
        fig.suptitle(f"By performance tertile — evoked vs LC (bc={bc})")
        _save_fig(fig, sub_dir, f"04_by_perf_tertile_bc{bc}", dpi)

        # ── 05: scatter evoked_lc vs evoked_motion with quadrant colours ──
        ev_lc2 = prepare_clean(clean, use_p, bc=bc, epoch="evoked", factor="learning_curve")
        ev_mo  = prepare_clean(clean, use_p, bc=bc, epoch="evoked", factor="motion")
        lc_idx = ev_lc2.set_index("unit_id")
        mo_idx = ev_mo.set_index("unit_id")
        idx    = lc_idx.index.intersection(mo_idx.index)
        if len(idx) >= 5:
            fig, axes2 = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H_SQUARE),
                                      gridspec_kw=dict(wspace=0.38))
            x  = mo_idx.loc[idx, "r"].values
            y  = lc_idx.loc[idx, "r"].values
            qc = [QUAD_COLORS.get(str(lc_idx.loc[i, "quadrant"]), "grey") for i in idx]
            for ax, (xd, yd, xl, yl) in zip(
                axes2,
                [(x, y, "Motion r", "LC r"),
                 (mo_idx.loc[idx, "r"].values,
                  lc_idx.loc[idx, "partial_r"].values
                  if "partial_r" in lc_idx.columns else np.full(len(idx), np.nan),
                  "Motion r", "Partial r (LC|motion)")],
            ):
                fin = np.isfinite(xd) & np.isfinite(yd)
                ax.scatter(xd[fin], yd[fin], s=3, c=[qc[i] for i in np.where(fin)[0]],
                           alpha=0.4, linewidths=0, rasterized=True)
                lim = np.nanmax(np.abs([xd[fin], yd[fin]])) * 1.15 if fin.any() else 0.3
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
                ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
                ax.set_aspect("equal")
                _ax_defaults(ax, xl, yl, "")
            leg_h = [Line2D([0], [0], marker="o", color="w",
                            markerfacecolor=QUAD_COLORS[q], markersize=6,
                            label=QUAD_LABELS[q]) for q in QUAD_ORDER]
            axes2[0].legend(handles=leg_h, fontsize=6)
            fig.suptitle(f"Motion vs LC — evoked (bc={bc})")
            _save_fig(fig, sub_dir, f"05_scatter_motion_vs_lc_bc{bc}", dpi)


# ============================================================================
# FOREST PLOT + STATISTICS (key figure)
# ============================================================================

def _forest_plot(ax, areas, mouse_means_df, rg_vals, rg_labels, rg_colors,
                 value_col, mwu_fdr_by_area, wilcoxon_pfdr_by_area, n_boot,
                 xlabel="Mean  ±  95% CI", area_order=None):
    """
    Shared forest-plot renderer.
    mouse_means_df must have columns: mouse_id, area, reward_group, <value_col>

    mwu_fdr_by_area      : dict {area: fdr_p}  — MWU R+ vs R− per area
    wilcoxon_pfdr_by_area: dict {area: fdr_p}  — Wilcoxon-vs-0 per area
                           Both are keyed by area name so ordering is safe.

    area_order : optional list from allen_utils.get_custom_area_order().
                 Present areas are sorted to match its position; areas absent
                 from area_order are appended alphabetically at the bottom.
    """
    if area_order is not None:
        order_map = {a: i for i, a in enumerate(area_order)}
        areas = sorted(areas, key=lambda a: (order_map.get(a, len(area_order)), a))
    y_pos = np.arange(len(areas), dtype=float)
    n_rg = len(rg_vals)
    step = 0.22
    half = (n_rg - 1) / 2.0 * step
    all_means = []

    for rgi, (rg, rg_lbl, rg_col) in enumerate(zip(rg_vals, rg_labels, rg_colors)):
        sub_rg = mouse_means_df[mouse_means_df["reward_group"] == rg]
        means, lo_ci, hi_ci, ns = [], [], [], []
        for a in areas:
            v = sub_rg[sub_rg["area"] == a][value_col].dropna().values
            m = v.mean() if len(v) else np.nan
            lo, hi = _bootstrap_ci(v, n_boot=n_boot) if len(v) >= 2 else (m, m)
            means.append(m)
            lo_ci.append(lo)
            hi_ci.append(hi)
            ns.append(len(v))
        offset = rgi * step - half
        y = y_pos + offset
        means = np.array(means, dtype=float)
        lo_ci = np.array(lo_ci, dtype=float)
        hi_ci = np.array(hi_ci, dtype=float)
        all_means.extend(means[np.isfinite(means)])

        ax.errorbar(means, y,
                    xerr=[means - lo_ci, hi_ci - means],
                    fmt="o", color=rg_col, markersize=7,
                    markeredgewidth=0.6, markeredgecolor="white",
                    elinewidth=2.0, capsize=4, capthick=2.0,
                    lw=0, label=rg_lbl, zorder=4)
        for yi, (m, n_n) in enumerate(zip(means, ns)):
            if np.isfinite(m):
                ax.text(0.02, y[yi], f"n={n_n}",
                        transform=ax.get_yaxis_transform(),
                        fontsize=7, va="center", color=rg_col)

    ax.axvline(0, color="k", lw=1.0, ls="--", zorder=2)
    x_max = max(np.nanmax(np.abs(all_means)) * 1.6, 0.15) if all_means else 0.3

    # Stars are looked up by area name — safe regardless of sort order
    for yi, a in enumerate(areas):
        mwu_lbl = _stat_label(mwu_fdr_by_area.get(a, np.nan))
        if mwu_lbl and mwu_lbl != "ns":
            ax.text(x_max * 0.97, y_pos[yi], mwu_lbl, ha="right", va="center",
                    fontsize=11, color="k", fontweight="bold")
        wil_lbl = _stat_label(wilcoxon_pfdr_by_area.get(a, np.nan))
        if wil_lbl and wil_lbl != "ns":
            ax.text(-x_max * 0.97, y_pos[yi], wil_lbl, ha="left", va="center",
                    fontsize=9, color="#555555")

    ax.set_xlim(-x_max, x_max)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(areas, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.invert_yaxis()


def _compute_metric_stats(mm_df, val_col, areas, rg_vals, rg_labels, n_perm=999):
    """
    For one mouse-means DataFrame and one value column, compute:
      - MWU R+ vs R− per area, FDR-BH corrected  → {area: fdr_p}
      - Wilcoxon-vs-0 per area collapsed over RG, FDR-BH corrected → {area: fdr_p}
      - PERMANOVA omnibus (reward_group factor)   → (F, p)

    All tests operate on mouse-level means (no pseudoreplication).
    Wilcoxon deduplicates across reward groups before testing.
    """
    # ── PERMANOVA ────────────────────────────────────────────────────────────
    pf, pp = _permanova(mm_df, group_col="reward_group",
                        value_col=val_col, n_perm=n_perm)

    # ── MWU R+ vs R− per area ────────────────────────────────────────────────
    mwu_fdr_by_area: dict[str, float] = {}
    if len(rg_vals) == 2:
        mwu_pv, mwu_areas = [], []
        for a in areas:
            v1 = mm_df[(mm_df["area"] == a) & (mm_df["reward_group"] == rg_vals[0])
                       ][val_col].dropna().values
            v2 = mm_df[(mm_df["area"] == a) & (mm_df["reward_group"] == rg_vals[1])
                       ][val_col].dropna().values
            if len(v1) >= 3 and len(v2) >= 3:
                _, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
                _, p = stats.ttest_ind(v1, v2, equal_var=False)
                mwu_pv.append(p)
                mwu_areas.append(a)
        if mwu_pv:
            _, pfdr_mwu, _, _ = multipletests(mwu_pv, method="fdr_bh")
            mwu_fdr_by_area = dict(zip(mwu_areas, pfdr_mwu))

    # ── Wilcoxon-vs-0, one mean per mouse per area (collapsed over RG) ───────
    mm_collapsed = mm_df.groupby(["mouse_id", "area"])[val_col].mean().reset_index()
    pv_all = [_one_sample_wilcoxon(
                  mm_collapsed[mm_collapsed["area"] == a][val_col].dropna().values)
              for a in areas]
    pv_arr = np.array(pv_all, dtype=float)
    pfdr_arr = np.ones(len(areas))
    valid = np.isfinite(pv_arr)
    if valid.any():
        _, pfdr_arr[valid], _, _ = multipletests(pv_arr[valid], method="fdr_bh")
    wilcoxon_pfdr_by_area = dict(zip(areas, pfdr_arr))

    return mwu_fdr_by_area, wilcoxon_pfdr_by_area, (pf, pp)


def area_reward_group_statistics(combined, config, out_dir, trial_table, area_order=None):
    use_p = config["use_p"]
    dpi = config["figure_dpi"]
    n_boot = config["n_bootstrap"]
    alpha = config["alpha"]
    clean = add_performance_tertile(combined, trial_table)
    sub_dir = out_dir / "statistics"

    rg_vals = sorted(clean["reward_group"].dropna().unique())
    rg_labels = [REWARD_LABELS.get(int(r), str(r)) for r in rg_vals]
    rg_colors = [REWARD_COLORS.get(int(r), "grey") for r in rg_vals]

    stat_rows = []

    # factor -> (own-signal label, conditioning-factor label, suptitle tag)
    # e.g. learning_curve rows show "Partial r (LC | M)" -- partialling out
    # motion; motion rows show the mirror, "Partial r (Motion | LC)".
    FACTOR_SPECS = {
        "learning_curve": dict(self_lbl="LC",     cond_lbl="M",  suptitle="LC",
                               factor_tag="lc"),
        "motion":         dict(self_lbl="Motion", cond_lbl="LC", suptitle="Motion",
                               factor_tag="motion"),
    }

    for epoch in ["baseline", "evoked"]:
        for bc in ["both", "good", "mua"]:
            for factor, spec in FACTOR_SPECS.items():
                f_all = prepare_clean(clean, use_p, bc=bc, epoch=epoch,
                                      factor=factor)
                areas = _shared_areas_min_mice(f_all, rg_vals, min_mice=3)
                if not areas:
                    continue

                mm = mouse_means(f_all)  # (mouse_id, area, reward_group, r)

                # ── Mouse means for all three metrics ─────────────────────
                mm_partial = mouse_means(
                    f_all.dropna(subset=["partial_r"]), value_col="partial_r")
                f_abs = f_all.dropna(subset=["partial_r"]).copy()
                f_abs["abs_partial_r"] = f_abs["partial_r"].abs()
                mm_abs = mouse_means(f_abs, value_col="abs_partial_r")

                self_lbl, cond_lbl = spec["self_lbl"], spec["cond_lbl"]
                metrics = [
                    ("r",             f"Signed r  ({self_lbl})",
                     mm,         "signed_r"),
                    ("partial_r",     f"Partial r  ({self_lbl} | {cond_lbl})",
                     mm_partial, "partial_r"),
                    ("abs_partial_r", f"|Partial r|  ({self_lbl} | {cond_lbl})",
                     mm_abs,     "abs_partial_r"),
                ]

                # ── Per-metric stats (PERMANOVA + MWU post-hoc + Wilcoxon-vs-0)
                metric_stats = {}   # val_col → (mwu_by_area, wil_by_area, (F, p))
                perm_strs = {}
                for val_col, ttl, mm_df, metric_tag in metrics:
                    mwu_d, wil_d, (pf_m, pp_m) = _compute_metric_stats(
                        mm_df, val_col, areas, rg_vals, rg_labels)
                    metric_stats[val_col] = (mwu_d, wil_d)
                    perm_strs[val_col] = f"p={pp_m:.3f}" if np.isfinite(pp_m) else "p=n/a"

                    # save to stat_rows
                    stat_rows.append(dict(test=f"permanova_rg_{metric_tag}",
                                          factor=factor,
                                          epoch=epoch, bc=bc, area="all",
                                          reward_group="all", stat=pf_m, p=pp_m,
                                          p_fdr=np.nan))
                    for a, p_fdr in mwu_d.items():
                        stat_rows.append(dict(
                            test=f"mwu_rplus_vs_rminus_{metric_tag}",
                            factor=factor,
                            epoch=epoch, bc=bc, area=a,
                            reward_group=f"{rg_labels[0]}_vs_{rg_labels[1]}",
                            stat=np.nan, p=p_fdr, p_fdr=p_fdr))
                    for a, p_fdr in wil_d.items():
                        stat_rows.append(dict(
                            test=f"wilcoxon_vs0_{metric_tag}",
                            factor=factor,
                            epoch=epoch, bc=bc, area=a,
                            reward_group="all", stat=np.nan,
                            p=p_fdr, p_fdr=p_fdr))

                # ── Per-neuron fraction significant per area, using the
                #    within-session BH-FDR corrected 'significant' column
                #    (p_fdr <= alpha) -- not a raw-p threshold. Recorded for
                #    both factors; this is what was previously missing for
                #    motion specifically.
                for a in areas:
                    a_sub = f_all[f_all["area"] == a]
                    n_a = len(a_sub)
                    frac_sig = float(a_sub["significant"].mean()) if n_a else np.nan
                    stat_rows.append(dict(
                        test="fraction_significant_p_fdr",
                        factor=factor, epoch=epoch, bc=bc, area=a,
                        reward_group="all", stat=n_a, p=frac_sig, p_fdr=np.nan))

                suptitle_base = (
                    f"{epoch} vs {spec['suptitle']}  (bc={bc})\n"
                    f"PERMANOVA — "
                    f"signed r: {perm_strs['r']}  ·  "
                    f"partial r: {perm_strs['partial_r']}  ·  "
                    f"|partial r|: {perm_strs['abs_partial_r']}\n"
                    "Stars right: MWU R+/R− post-hoc FDR  ·  Stars left: Wilcoxon-vs-0 FDR  ·  "
                    f"significance (fraction sig, per-neuron) uses BH-FDR q<={alpha}"
                )

                # ── Sort helper: decreasing |mean(R+) − mean(R-)| per metric
                def _diff_order(mm_df, val_col):
                    if len(rg_vals) < 2:
                        return list(areas)
                    diffs = {}
                    for a in areas:
                        sub = mm_df[mm_df["area"] == a]
                        grp_means = [sub[sub["reward_group"] == rg][val_col].dropna().mean()
                                     for rg in rg_vals]
                        diffs[a] = (abs(grp_means[1] - grp_means[0])
                                    if all(np.isfinite(m) for m in grp_means) else np.nan)
                    return sorted(areas,
                                  key=lambda a: (-diffs[a]
                                                 if np.isfinite(diffs[a]) else np.inf))

                orderings = [
                    ("allen",    area_order, "Allen area order"),
                    ("diffmean", None,       "Decreasing |R+−R−| mean diff"),
                ]

                ftag = spec["factor_tag"]

                # ── 3-panel overview figure for each ordering ──────────────
                for ord_tag, ord_list, ord_label in orderings:
                    fig, axes = plt.subplots(
                        1, 3,
                        figsize=(21, max(3.2, len(areas) * 0.52 + 1.2)),
                        gridspec_kw=dict(wspace=0.55))
                    for ax, (val_col, ttl, mm_df, _) in zip(axes, metrics):
                        mwu_d, wil_d = metric_stats[val_col]
                        order = ord_list if ord_tag == "allen" else _diff_order(mm_df, val_col)
                        _forest_plot(ax, areas, mm_df, rg_vals, rg_labels, rg_colors,
                                     val_col, mwu_d, wil_d, n_boot, area_order=order)
                        ax.set_title(ttl, fontsize=10, pad=5)
                        ax.legend(loc="lower right", fontsize=10, handlelength=1.2)
                        if val_col == 'abs_partial_r':
                            ax.xlim(-0.05, 0.5)
                    fig.suptitle(f"{suptitle_base}\n({ord_label})", fontsize=9, y=1.01)
                    fig.tight_layout()
                    _save_fig(fig, sub_dir, f"forest_{ftag}_{epoch}_bc{bc}_{ord_tag}", dpi)

                # ── Standalone single-metric figures (2 orderings × 3 metrics)
                for val_col, ttl, mm_df, metric_tag in metrics:
                    mwu_d, wil_d = metric_stats[val_col]
                    for ord_tag, ord_list, ord_label in orderings:
                        order = ord_list if ord_tag == "allen" else _diff_order(mm_df, val_col)
                        figS, axS = plt.subplots(
                            figsize=(7, max(3.2, len(areas) * 0.52 + 1.2)))
                        _forest_plot(axS, areas, mm_df, rg_vals, rg_labels, rg_colors,
                                     val_col, mwu_d, wil_d, n_boot,
                                     xlabel=ttl, area_order=order)
                        axS.set_title(f"{ttl}\n{epoch}  bc={bc}  ({ord_label})",
                                      fontsize=10, pad=5)
                        axS.legend(loc="lower right", fontsize=10, handlelength=1.2)
                        figS.suptitle(suptitle_base, fontsize=9, y=1.01)
                        figS.tight_layout()
                        _save_fig(figS, sub_dir,
                                  f"forest_{ftag}_{metric_tag}_{epoch}_bc{bc}_{ord_tag}", dpi)

    if stat_rows:
        pd.DataFrame(stat_rows).to_csv(
            str(sub_dir / "area_reward_stats.csv"), index=False)
        print(f"  Stats CSV → {sub_dir / 'area_reward_stats.csv'}")
    return pd.DataFrame(stat_rows)


# ============================================================================
# CROSS-CORRELATION GRAND SUMMARY (across all sessions)
# ============================================================================

def plot_grand_xcorr(combined, config, out_dir):
    """
    Grand cross-correlation by area: per-neuron r at each shift lag,
    averaged across all sessions, separated by motion vs learning curve.
    Uses pre-stored scores — if not available, skips gracefully.
    NOTE: full scores are not stored in the CSV; this figure is generated
    per-session in plot_cross_correlation_by_area above.  This function
    plots the aggregate peak-lag distributions as a proxy.
    """
    use_p   = config["use_p"]
    dpi     = config["figure_dpi"]
    sub_dir = out_dir / "grand_summary"
    N       = config["shift_N"]
    shifts  = np.arange(-N, N + 1)

    areas = sorted(combined["area"].dropna().unique())
    if not areas:
        return

    fig, axes = plt.subplots(len(areas), 2,
                             figsize=(FIG_W_DOUBLE, FIG_H_UNIT * len(areas)),
                             gridspec_kw=dict(wspace=0.35, hspace=0.6),
                             squeeze=False)
    axes[0, 0].set_title("Peak-lag dist — vs Motion", fontsize=9)
    axes[0, 1].set_title("Peak-lag dist — vs LC", fontsize=9)

    for ri, area in enumerate(areas):
        for ci, (factor, col) in enumerate([
            ("motion", FACTOR_COLORS["motion"]),
            ("learning_curve", FACTOR_COLORS["learning_curve"]),
        ]):
            ax  = axes[ri, ci]
            sub = combined[(combined["area"] == area) &
                           (combined["factor"] == factor) &
                           (combined["epoch"] == "evoked")]
            ax.set_ylabel(area if ci == 0 else "", fontsize=8)
            if "peak_lag" not in sub.columns or sub["peak_lag"].dropna().empty:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
                continue
            pl = sub["peak_lag"].dropna().values
            bins = np.arange(-N - 0.5, N + 1.5, 1)
            ax.hist(pl, bins=bins, color=col, edgecolor="none", alpha=0.75)
            ax.axvline(0, color="k", lw=1.0, ls="--")
            ax.axvline(np.median(pl), color="darkorange", lw=1.0, ls=":",
                       label=f"med={np.median(pl):.0f}")
            ax.legend(fontsize=6, loc="upper right")
            if ri == len(areas) - 1:
                ax.set_xlabel("Peak lag (trials)", fontsize=8)

    fig.suptitle("Grand peak-lag distribution by area (evoked epoch)", y=1.01)
    _save_fig(fig, sub_dir, "grand_xcorr_peak_lag_by_area", dpi)


# ============================================================================
# FRACTION-SIGNIFICANT DETAIL
# ============================================================================

def plot_fraction_significant_detail(combined, config, out_dir):
    use_p   = config["use_p"]
    dpi     = config["figure_dpi"]
    sub_dir = out_dir / "fraction_sig_figures"

    TESTS   = [("baseline","motion"), ("evoked","motion"),
               ("baseline","learning_curve"), ("evoked","learning_curve")]
    TLBLS   = ["Baseline\nvs Motion", "Evoked\nvs Motion",
                "Baseline\nvs LC", "Evoked\nvs LC"]
    bc_groups  = ["both", "good", "mua"]
    bc_offsets = np.array([-0.26, 0, 0.26])
    bc_cols    = [BC_COLORS[b] for b in bc_groups]

    areas    = sorted(combined["area"].dropna().unique())
    rg_vals  = sorted(combined["reward_group"].dropna().unique())
    rg_labels= [REWARD_LABELS.get(int(r), str(r)) for r in rg_vals]
    rg_colors= [REWARD_COLORS.get(int(r), "grey") for r in rg_vals]

    fig, ax = plt.subplots(figsize=(FIG_W_TRIPLE, FIG_H_SQUARE))
    for bc, off, col in zip(bc_groups, bc_offsets, bc_cols):
        fracs = [prepare_clean(combined, use_p, bc=bc, epoch=ep, factor=fa)
                 ["significant"].mean() for ep, fa in TESTS]
        ax.bar(np.arange(len(TESTS)) + off, fracs, width=0.23,
               color=col, edgecolor="k", lw=0.6, label=bc)
    ax.axhline(SIG_ALPHA, color="crimson", ls="--", lw=0.9)
    ax.set_xticks(range(len(TESTS))); ax.set_xticklabels(TLBLS, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    _ax_defaults(ax, "", "Fraction significant", "Fraction sig — by test and unit quality")
    ax.legend()
    _save_fig(fig, sub_dir, "frac_sig_by_test", dpi)

    # clean-learner fraction per area
    if areas:
        fig, ax = plt.subplots(figsize=(max(FIG_W_FULL, len(areas) * 1.3), FIG_H_SQUARE))
        ev_lc = prepare_clean(combined, use_p, epoch="evoked", factor="learning_curve")
        ev_lc_q = ev_lc.dropna(subset=["quadrant"])
        bar_w = 0.7 / max(len(rg_vals), 1)
        for rgi, (rg, rg_lbl, rg_col) in enumerate(zip(rg_vals, rg_labels, rg_colors)):
            sub_rg = ev_lc_q[ev_lc_q["reward_group"] == rg]
            fracs_cl = [(sub_rg[(sub_rg["area"] == a) &
                                (sub_rg["quadrant"] == "clean_learner")].shape[0] /
                         max(sub_rg[sub_rg["area"] == a].shape[0], 1))
                        for a in areas]
            offsets = np.arange(len(areas)) + (rgi - len(rg_vals)/2 + 0.5) * bar_w
            ax.bar(offsets, fracs_cl, width=bar_w * 0.85,
                   color=rg_col, edgecolor="k", lw=0.6, label=rg_lbl)
        ax.axhline(SIG_ALPHA, color="crimson", ls="--", lw=0.9)
        ax.set_xticks(range(len(areas))); ax.set_xticklabels(areas, rotation=40, ha="right")
        ax.set_ylim(0, 1.0)
        _ax_defaults(ax, "Brain area", "Fraction clean learners",
                     "Fraction clean-learner neurons by area and reward group")
        ax.legend()
        _save_fig(fig, sub_dir, "frac_clean_learner_by_area_rg", dpi)

    print(f"  Fraction-sig figures → {sub_dir}/")


# ============================================================================
# LOAD EXISTING RESULTS
# ============================================================================

def load_existing_results(config, mouse_ids=None, session_days=None) -> pd.DataFrame:
    root      = Path(config["combined_results_root"])
    subfolder = config["per_session_results_subfolder"]
    pattern   = f"*/*/{subfolder}/*_shift_test_results.csv"
    csv_files = sorted(root.glob(pattern))
    if not csv_files:
        warnings.warn(f"No shift-test CSVs found under {root} (pattern: {pattern}).")
        return pd.DataFrame()
    frames = []
    for csv_path in csv_files:
        try:
            parts      = csv_path.relative_to(root).parts
            mouse_id   = parts[0]
            session_day = parts[1]
        except (ValueError, IndexError):
            mouse_id = session_day = None
        if mouse_ids    is not None and mouse_id    not in mouse_ids:
            continue
        if session_days is not None and session_day not in session_days:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            warnings.warn(f"Could not read {csv_path}: {exc}")
            continue
        if "mouse_id"    not in df.columns and mouse_id    is not None:
            df["mouse_id"]    = mouse_id
        if "session_day" not in df.columns and session_day is not None:
            df["session_day"] = session_day
        frames.append(df)
        print(f"  Loaded {len(df):>5} rows  ←  {csv_path.relative_to(root)}")
    if not frames:
        warnings.warn("No CSV files matched the filters.")
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    print(f"[load_existing_results] {len(combined)} rows from {len(frames)} session(s).")
    return combined


# ============================================================================
# PER-SESSION WORKER  (runs in subprocess for parallelism)
# ============================================================================

def _process_one_session(args):
    """Top-level function (picklable) for ProcessPoolExecutor."""
    (session_id, mouse_id, session_day,
     unit_table, trial_table, tmp_dir, cfg) = args

    import pickle as _pickle
    unit_table  = _pickle.loads((tmp_dir / "unit_table.pkl").read_bytes())
    trial_table = _pickle.loads((tmp_dir / "trial_table.pkl").read_bytes())


    print(f"[shift_test] {mouse_id} / {session_day}  (session {session_id})")

    try:
        results = analyze_session(unit_table, trial_table,
                                  session_id, mouse_id, session_day, cfg)
    except Exception as exc:
        warnings.warn(f"Session {mouse_id}/{session_day} failed: {exc}")
        return None

    out_dir = get_per_session_output_dir(cfg, mouse_id, session_day)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{mouse_id}_{session_day}_shift_test_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"  -> {len(results)} rows → {csv_path}")

    write_drift_test_warnings(results, out_dir, mouse_id, session_day)

    rng = np.random.default_rng(cfg["random_seed"])

    # load learning curve once for this session (reused by several figures)
    learning_curve = None
    lc_path = build_learning_curve_path(cfg, mouse_id, session_day)
    if lc_path.exists():
        try:
            learning_curve = np.asarray(
                load_learning_curve(lc_path)["p_mean"][0], dtype=float)
        except Exception:
            pass

    for fn, label in [
        (lambda: plot_drift_sanity(unit_table, trial_table,
                                   session_id, mouse_id, session_day,
                                   cfg, out_dir),
         "drift sanity"),
        (lambda: plot_motion_alignment(unit_table, trial_table,
                                       session_id, mouse_id, session_day,
                                       cfg, out_dir),
         "motion alignment"),
        (lambda: plot_single_neuron_examples(results, unit_table, trial_table,
                                             session_id, mouse_id, session_day,
                                             cfg, out_dir, rng),
         "single-neuron examples"),
        (lambda: plot_process_figure(results, unit_table, trial_table,
                                     session_id, mouse_id, session_day,
                                     cfg, out_dir, learning_curve, {}),
         "process figure"),
        (lambda: plot_cross_correlation_by_area(results, unit_table, trial_table,
                                                session_id, mouse_id, session_day,
                                                cfg, out_dir, learning_curve),
         "cross-correlation"),
        (lambda: plot_per_session_summary(results, cfg, mouse_id, session_day, out_dir),
         "per-session summary"),
    ]:
        try:
            fn()
        except Exception as exc:
            warnings.warn(f"{label} failed for {mouse_id}/{session_day}: {exc}")

    #return results
    return csv_path


# ============================================================================
# TOP-LEVEL ORCHESTRATION
# ============================================================================

def run_shift_test_analysis(unit_table, trial_table, output_path,
                            config=None, figures_only=False,
                            mouse_ids=None, session_days=None) -> pd.DataFrame:
    """
    Full shift-test analysis pipeline.

    Parameters
    ----------
    unit_table      : DataFrame (ignored when figures_only=True)
    trial_table     : DataFrame (always required for performance tertile)
    output_path     : overrides config['combined_results_root']
    config          : dict, optional; merged with DEFAULT_CONFIG
    figures_only    : if True, load existing CSVs and regenerate grand figures only
    mouse_ids       : optional list to restrict sessions
    session_days    : optional list to restrict sessions

    Returns
    -------
    pd.DataFrame  combined long-format results
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if output_path:
        cfg["combined_results_root"] = output_path
    print(cfg['data_root'])

    # ── FIGURES-ONLY ────────────────────────────────────────────────────────
    if figures_only:
        print("[shift_test] figures_only=True — loading existing CSVs...")
        combined = load_existing_results(cfg, mouse_ids=mouse_ids,
                                         session_days=session_days)
        if combined.empty:
            warnings.warn("No data loaded.")
            return combined
        summary_dir = get_summary_output_dir(cfg)
        summary_dir.mkdir(parents=True, exist_ok=True)
        for fn, label in [
            (lambda: plot_grand_summary(combined, cfg, summary_dir, trial_table),
             "grand summary"),
            (lambda: area_reward_group_statistics(combined, cfg, summary_dir, trial_table,
                                                  area_order=allen_utils.get_custom_area_order()),
             "area/reward stats"),
            (lambda: plot_fraction_significant_detail(combined, cfg, summary_dir),
             "fraction-sig detail"),
            (lambda: plot_grand_xcorr(combined, cfg, summary_dir),
             "grand xcorr"),
        ]:
            try:
                fn()
            except Exception as exc:
                warnings.warn(f"{label} failed: {exc}")
        return combined

    # ── FULL COMPUTATION ─────────────────────────────────────────────────────
    if unit_table is None:
        raise ValueError("unit_table required when figures_only=False.")

    required_unit  = {"session_id", "mouse_id", "unit_id", "electrode_group",
                      "depth", "bc_label", "spike_times"}
    required_trial = {"session_id", "start_time", "trial_type"}
    for missing, name in [
        (required_unit  - set(unit_table.columns),  "unit_table"),
        (required_trial - set(trial_table.columns), "trial_table"),
    ]:
        if missing:
            raise KeyError(f"{name} missing columns: {missing}")

    if "session_day" not in unit_table.columns:
        unit_table = unit_table.copy()
        unit_table["session_day"] = "whisker_0"

    session_keys = unit_table[["session_id", "mouse_id", "session_day"]].drop_duplicates()
    if mouse_ids    is not None:
        session_keys = session_keys[session_keys["mouse_id"].isin(mouse_ids)]
    if session_days is not None:
        session_keys = session_keys[session_keys["session_day"].isin(session_days)]

    # Write tables to disk once — avoids large IPC serialization (CPython 3.14 WinError 87)
    import tempfile, pickle as _pickle
    _tmp_dir = Path(tempfile.mkdtemp())
    (_tmp_dir / "unit_table.pkl").write_bytes(_pickle.dumps(unit_table))
    (_tmp_dir / "trial_table.pkl").write_bytes(_pickle.dumps(trial_table))


    args_list = [
        (row["session_id"], row["mouse_id"], row["session_day"],
         unit_table, trial_table, _tmp_dir, cfg)
        for _, row in session_keys.iterrows()
    ]

    n_workers = cfg.get("n_workers", 1)
    all_results = []

    #if n_workers > 1:
    #    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
    #        #for res in ex.map(_process_one_session, args_list):
    #        #    if res is not None:
    #        #        all_results.append(res)
    #        for csv_path  in ex.map(_process_one_session, args_list):
    #            if csv_path is not None:
    #                all_results.append(pd.read_csv(csv_path))
    #else:
    #    for args in args_list:
    #        res = _process_one_session(args)
    #        if res is not None:
    #            all_results.append(res)

    if n_workers > 1:
        pickle.dumps(args_list[0])
        csv_paths = Parallel(n_jobs=n_workers, backend='loky')(
            delayed(_process_one_session)(args) for args in args_list
        )
        for csv_path in csv_paths:
            if csv_path is not None:
                all_results.append(pd.read_csv(csv_path))
    else:
        for args in args_list:
            res = _process_one_session(args)
            if res is not None:
                all_results.append(pd.read_csv(res))

    if not all_results:
        warnings.warn("No sessions processed.")
        return pd.DataFrame()

    combined    = pd.concat(all_results, ignore_index=True)
    summary_dir = get_summary_output_dir(cfg)
    summary_dir.mkdir(parents=True, exist_ok=True)

    for fn, label in [
        (lambda: plot_grand_summary(combined, cfg, summary_dir, trial_table),
         "grand summary"),
        (lambda: area_reward_group_statistics(combined, cfg, summary_dir, trial_table,
                                              area_order=allen_utils.get_custom_area_order()),
         "area/reward stats"),
        (lambda: plot_fraction_significant_detail(combined, cfg, summary_dir),
         "fraction-sig detail"),
        (lambda: plot_grand_xcorr(combined, cfg, summary_dir),
         "grand xcorr"),
    ]:
        try:
            fn()
        except Exception as exc:
            warnings.warn(f"{label} failed: {exc}")

    print(f"[shift_test] Done. Summary in {summary_dir}")
    return combined


if __name__ == "__main__":
    print("Import run_shift_test_analysis and call with your unit_table/trial_table.")