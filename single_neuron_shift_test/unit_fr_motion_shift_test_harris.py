"""
single_neuron_motion_shift_test.py
-----------------------------------
Per-neuron shift test: does firing rate correlate with probe drift (DREDge
motion)? Both firing rate and motion are aggregated into fixed-width time
bins (~1s) spanning the whole session; the null is built by randomly
shifting the motion series relative to firing rate (preserving each
series' own autocorrelation) and asking how often the shifted correlation
is at least as extreme as the true one.

This is the reduced version of the pipeline: only the drift test survives.
No learning-curve test, no cross-session/summary statistics, no forest
plots -- single-mouse, single-probe diagnostic plots only.
"""
from __future__ import annotations
import json
import re
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

# ============================================================================
# STYLE
# ============================================================================

RC = {
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
}
plt.rcParams.update(RC)

MOTION_COLOR = "#27AE60"
FR_COLOR     = "#2980B9"
SIG_COLOR    = "#C0392B"
NONSIG_COLOR = "#95A5A6"
FIG_W_FULL   = 12.0
FIG_H_UNIT   = 2.8

# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_CONFIG = {
    "data_root": r"M:\analysis\Axel_Bisi\data",
    "motion_subpath_parts": (
        "Ephys", "catgt_{mouse_id}_{gate}",
        "{mouse_id}_{gate}_imec{imec_id}", "dredge", "motion", "motion",
    ),
    "combined_results_root": r"M:\analysis\Axel_Bisi\combined_results",
    "per_session_results_subfolder": "single_neuron_motion_shift_test",

    "random_seed": 0,     # only used for example-neuron selection, not the test itself
    "figure_dpi":  300,
    "alpha":       0.05,
    "n_workers":   25,          # parallel sessions (joblib)

    "drift_test_corr_method": "pearson",   # "pearson" or "spearman"
    "drift_test_bin_width":   1.0,     # seconds
    # N: half-width of the exhaustive shift sweep (Harris 2021). Comparison
    # uses ALL 2N+1 consecutive integer shifts -N..N, each correlating a
    # fixed central segment of firing rate (length D=T-2N) against the same-
    # length sliding segment of motion. N=500 -> 1001 shifts total, needs a
    # session with > 1000 one-second bins (~17 min) to be usable.
    "drift_test_N":        500,
    "drift_test_min_bins": 10,   # minimum D=T-2N required to run the test
    "use_p": "p_conservative",  # "p_conservative" (m/(N+1), proven for any
                                 # finite N) or "p_approx" (m/(2N+1), valid
                                 # as N->inf, ~2x power) -- which feeds FDR

    # BH-FDR correction scope: "imec_id" (default, per probe), "area"
    # (per brain area -- smaller families, eases the threshold, changes
    # what the correction means), or "session" (pooled across everything
    # in the session -- the original, most conservative scope).
    "fdr_group_by": "imec_id",
    # If False, skip BH-FDR entirely: p_fdr becomes a pass-through of the
    # raw p (config["use_p"]), and significance is a plain p <= alpha
    # threshold -- uncorrected, no multiple-comparisons control at all.
    "apply_fdr": False,

    "n_example_neurons": 30,    # example units per sig/n.s. group, per probe
    "plot_all_units": False,   # if True, also plot EVERY valid unit (not
                                # just n_example_neurons) into out_dir/diag/,
                                # split into significant/nonsignificant. Can
                                # produce hundreds of files -- off by default.
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
    sid_str = str(session_id)
    matches = [p for p in candidates if sid_str in p.name]
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
        print('Trying dredge_fast.')
        # Replace dredge by dredge_fast
        config["motion_subpath_parts"] = [p.replace("dredge", "dredge_fast") for p in config["motion_subpath_parts"]]
        matches = sorted(session_root.glob(glob_pattern))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected 1 motion match, found {len(matches)}: {matches}")
    return matches[0]


def get_per_session_output_dir(config, mouse_id, session_day) -> Path:
    return (Path(config["combined_results_root"])
            / mouse_id / session_day / config["per_session_results_subfolder"])


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


# ============================================================================
# THE TEST: time-binned FR vs. motion, shift-randomized null
# ----------------------------------------------------------------------------
# Firing rate and DREDge displacement are each binned into fixed-width time
# bins spanning the whole session, correlated, then tested against a null
# built by shifting the motion series by a random lag (drawn away from 0
# and away from the series length) and recomputing the correlation on the
# trimmed overlap. This preserves each series' own autocorrelation/trend
# while destroying true alignment -- the whole point being that a naive
# Pearson p-value badly inflates false positives on two autocorrelated
# series like these.
#
# This implements Harris (2021), "A Shift Test for Independence in Generic
# Time Series" (https://arxiv.org/abs/2012.06862): a FIXED central segment
# of firing rate (length D = T-2N) is compared against motion at EVERY
# consecutive integer shift s = -N..N (2N+1 total, exhaustive -- not a
# random subsample), using |correlation| as the (two-sided) association
# measure V. m = how many of those 2N+1 shifts are >= the true (s=0) value.
#   p_conservative = m/(N+1)   -- proven valid for any finite N
#   p_approx       = m/(2N+1)  -- valid as N->inf, ~2x power
# The test is fully deterministic given the data and N: no randomness, no
# seed needed.
# ============================================================================

def compute_binned_firing_rate(spike_times, t_start, t_end, bin_width):
    """Bin spike counts into fixed-width time bins spanning [t_start, t_end].
    Returns (bin_centers, rate_per_bin)."""
    n_bins = int(np.floor((t_end - t_start) / bin_width))
    if n_bins < 1:
        return np.array([]), np.array([])
    edges   = t_start + np.arange(n_bins + 1) * bin_width
    centers = (edges[:-1] + edges[1:]) / 2.0
    spk     = np.sort(np.asarray(spike_times, dtype=float))
    counts  = np.histogram(spk, bins=edges)[0]
    rate    = counts / bin_width
    return centers, rate.astype(float)


def _corr_fn(method: str):
    if method == "spearman":
        return lambda a, b: stats.spearmanr(a, b)[0]
    return lambda a, b: stats.pearsonr(a, b)[0]


def shift_test_exhaustive(x: np.ndarray, y: np.ndarray, N: int,
                          method: str = "pearson", return_scores: bool = False):
    """
    Harris (2021) shift test. x's central segment [N : T-N] (length D=T-2N)
    is fixed; y is compared against it at every consecutive integer shift
    s = -N..N (2N+1 total). Association measured as |correlation| (two-
    sided). Returns (r0_signed, p_conservative, p_approx, m, D); if
    return_scores=True also returns the (2N+1,) array of |corr| at every
    shift, index i corresponding to shift s = i - N.

    r0_signed is the actual signed correlation at shift 0 (for reporting
    direction/effect size) -- the two-sided test itself is built on |r|.
    """
    T = len(x)
    D = T - 2 * N
    corr = _corr_fn(method)

    if D < 1:
        out = (np.nan, np.nan, np.nan, 0, D)
        return (*out, np.array([])) if return_scores else out

    x_seg = x[N:T - N]
    scores = np.full(2 * N + 1, np.nan)
    for i, s in enumerate(range(-N, N + 1)):
        y_seg = y[s + N: s + T - N]
        if len(y_seg) != D or np.all(x_seg == x_seg[0]) or np.all(y_seg == y_seg[0]):
            continue
        try:
            scores[i] = abs(corr(x_seg, y_seg))
        except Exception:
            continue

    v0 = scores[N]   # shift s=0 is at index N
    if not np.isfinite(v0):
        out = (np.nan, np.nan, np.nan, 0, D)
        return (*out, scores) if return_scores else out

    try:
        r0_signed = float(corr(x_seg, y[N:T - N]))
    except Exception:
        r0_signed = np.nan

    finite = np.isfinite(scores)
    m = int(np.sum(scores[finite] >= v0))
    p_conservative = min(m / (N + 1), 1.0)
    p_approx       = min(m / (2 * N + 1), 1.0)
    out = (r0_signed, p_conservative, p_approx, m, D)
    return (*out, scores) if return_scores else out


def time_binned_drift_test(spike_times, motion, depth, t_start, t_end,
                           bin_width, N, config):
    """Full recipe: bin FR and displacement over time, run the exhaustive
    shift test. Returns a dict with r, p_conservative, p_approx, m, N,
    n_bins, error."""
    method   = config.get("drift_test_corr_method", "pearson")
    min_bins = config.get("drift_test_min_bins", 10)

    centers, rate = compute_binned_firing_rate(spike_times, t_start, t_end, bin_width)
    if len(centers) <= 2 * N:
        return dict(r=np.nan, p_conservative=np.nan, p_approx=np.nan, m=0, N=N,
                    n_bins=len(centers),
                    error=f"too few bins ({len(centers)}) for N={N} (need > {2*N})")

    try:
        disp = get_motion_at_times_and_depth(motion, centers, depth)
    except Exception as exc:
        return dict(r=np.nan, p_conservative=np.nan, p_approx=np.nan, m=0, N=N,
                    n_bins=len(centers), error=f"motion lookup failed: {exc}")

    valid = np.isfinite(rate) & np.isfinite(disp)
    r_v, d_v = rate[valid], np.asarray(disp, dtype=float)[valid]
    if len(r_v) <= 2 * N:
        return dict(r=np.nan, p_conservative=np.nan, p_approx=np.nan, m=0, N=N,
                    n_bins=len(r_v),
                    error=f"too few valid bins ({len(r_v)}) for N={N} (need > {2*N})")

    D = len(r_v) - 2 * N
    if D < min_bins:
        return dict(r=np.nan, p_conservative=np.nan, p_approx=np.nan, m=0, N=N,
                    n_bins=len(r_v),
                    error=f"central segment too short (D={D} < min_bins={min_bins})")

    if np.all(r_v == r_v[0]) or np.all(d_v == d_v[0]):
        # zero variance -> correlation undefined but not missing: a constant
        # signal cannot covary with anything, so r=0, p=1 is the well-defined
        # answer, not NaN.
        return dict(r=0.0, p_conservative=1.0, p_approx=1.0, m=0, N=N,
                    n_bins=len(r_v), error="constant vector (r set to 0, p to 1)")

    r_obs, p_cons, p_appx, m, D = shift_test_exhaustive(r_v, d_v, N, method=method)
    return dict(r=r_obs, p_conservative=p_cons, p_approx=p_appx, m=m, N=N,
               n_bins=len(r_v), error=None)


# Error prefixes meaning "could not compute" (missing/insufficient data),
# as opposed to "computed and the answer is a well-defined null" (constant
# vector -> r=0, p=1). Only the former gets flagged/warned.
DRIFT_TEST_DATA_WARNING_PREFIXES = (
    "too few bins", "motion lookup failed", "too few valid bins",
    "central segment too short", "could not determine imecID",
)


def write_drift_test_warnings(results: pd.DataFrame, out_dir: Path,
                              mouse_id: str, session_day) -> Optional[Path]:
    """Flag rows that could not be computed due to missing/insufficient data
    -- NOT the constant-vector case, which is a valid r=0/p=1 result, not a
    data problem. Prints each flagged unit and writes warning.txt."""
    if results.empty or "error" not in results.columns:
        return None
    flagged = results[results["error"].notna() &
                      results["error"].str.startswith(DRIFT_TEST_DATA_WARNING_PREFIXES)]
    if flagged.empty:
        return None

    lines = [f"Drift-test data-quality warnings -- {mouse_id} / {session_day}",
             f"{len(flagged)} of {len(results)} rows flagged "
             f"(missing/insufficient data, not a computed null result):", ""]
    for _, row in flagged.iterrows():
        msg = (f"unit {row['neuron_id']}  area={row['area']}  -- {row['error']}")
        lines.append("  " + msg)
        warnings.warn(f"[{mouse_id}/{session_day}] {msg}")

    warn_path = out_dir / "warning.txt"
    warn_path.write_text("\n".join(lines) + "\n")
    print(f"  -> {len(flagged)} drift-test data-quality warning(s) -> {warn_path}")
    return warn_path


# ============================================================================
# RESULT SCHEMA
# ============================================================================

@dataclass
class NeuronTestResult:
    session_id:      object
    mouse_id:        str
    neuron_id:       object
    cluster_id:      object = None
    electrode_group: object = None
    bc_label:        object = None
    depth:           float = np.nan
    imec_id:         Optional[int] = None
    area:            Optional[str] = None
    t_start:         float = np.nan
    t_end:           float = np.nan
    n_bins:          int = 0
    N:               int = 0     # shift half-width used (Harris 2021)
    r:               float = np.nan
    m:               int = 0
    p_conservative:  float = np.nan   # m/(N+1), proven for any finite N
    p_approx:        float = np.nan   # m/(2N+1), valid as N->inf, ~2x power
    p_fdr:           float = np.nan   # BH-FDR corrected p (config["use_p"]),
                                       # or a pass-through of the raw p if
                                       # config["apply_fdr"]=False
    significant:     bool = False
    error:           Optional[str] = None


# ============================================================================
# PER-SESSION ANALYSIS
# ============================================================================

def analyze_session(unit_table, trial_table, session_id, mouse_id,
                    session_day, config) -> pd.DataFrame:
    alpha     = config["alpha"]
    bin_width = config["drift_test_bin_width"]
    N         = config["drift_test_N"]
    use_p     = config.get("use_p", "p_conservative")

    # session time bounds: ALL trials, including passive -- the drift test
    # doesn't care about trial context, and restricting to active-only
    # trials would silently narrow the test window below both the raw
    # DREDge object's coverage and what the diagnostic plots show.
    trials_all = (trial_table[trial_table["session_id"] == session_id]
                  .sort_values("start_time").reset_index(drop=True))
    t_start = float(trials_all["start_time"].min()) if len(trials_all) else np.nan
    t_end   = float(trials_all["start_time"].max()) if len(trials_all) else np.nan

    units = unit_table[unit_table["session_id"] == session_id].copy()

    # neuron_id must be unique within this session: every neuron_id-keyed
    # join downstream silently misaligns instead of erroring on a collision,
    # so a duplicate would corrupt results without any visible sign.
    # cluster_id is NOT usable as this key on its own -- it can repeat
    # across different imec_id/electrode_group in the same file.
    dup_mask = units["neuron_id"].duplicated(keep=False)
    if dup_mask.any():
        dup_report = (units.loc[dup_mask, ["neuron_id", "cluster_id", "imec_id"]]
                      if "imec_id" in units.columns
                      else units.loc[dup_mask, ["neuron_id", "cluster_id"]])
        raise ValueError(
            f"Duplicate neuron_id within session {mouse_id}/{session_day} "
            f"(session_id={session_id}):\n{dup_report.to_string(index=False)}\n"
            f"neuron_id must be unique per session. If these rows differ in "
            f"electrode_group/imec_id but share the same neuron_id, neuron_id "
            f"is likely being derived from cluster_id alone upstream without "
            f"folding in the probe -- fix the neuron_id construction rather "
            f"than working around it here.")

    raw = []
    for _, unit in units.iterrows():
        neuron_id            = unit["neuron_id"]
        cluster_id           = unit.get("cluster_id", None)
        electrode_group_raw  = unit.get("electrode_group", None)
        bc_label             = unit.get("bc_label", None)
        depth                = float(unit["depth"])
        spk                  = np.asarray(unit["spike_times"], dtype=float)
        imec_id              = _get_imec_id(unit.get("electrode_group"))
        area                 = unit.get("area_custom_acronym",
                               unit.get("area_acronym_custom", None))
        meta = dict(session_id=session_id, mouse_id=mouse_id, neuron_id=neuron_id,
                   cluster_id=cluster_id, electrode_group=electrode_group_raw,
                   bc_label=bc_label, depth=depth, imec_id=imec_id, area=area,
                   t_start=t_start, t_end=t_end)

        motion_error = None
        motion_obj   = None
        if imec_id is None:
            motion_error = "could not determine imecID"
        else:
            try:
                mp         = build_motion_path(config, mouse_id, session_id, imec_id)
                motion_obj = load_motion(mp)
            except Exception as exc:
                motion_error = str(exc)

        if motion_error or np.isnan(t_start) or np.isnan(t_end):
            res = dict(r=np.nan, p_conservative=np.nan, p_approx=np.nan, m=0, N=N,
                      n_bins=0, error=motion_error or "no trials for session bounds")
        else:
            res = time_binned_drift_test(spk, motion_obj, depth, t_start, t_end,
                                         bin_width, N, config)
        raw.append(dict(meta=meta, **res))

    # BH-FDR on config["use_p"] (p_conservative by default), applied within
    # groups defined by config["fdr_group_by"] -- each group is its own
    # family of tests, so a unit's significance doesn't depend on the
    # p-value distribution of units outside its group. `significant` is
    # explicitly derived from p_fdr (q <= alpha), not statsmodels' `rejected`
    # output directly -- same result by definition of BH-FDR, but makes the
    # dependency explicit.
    #
    # config["apply_fdr"]=False skips correction entirely: p_fdr becomes a
    # pass-through of the raw p, so `p_fdr <= alpha` downstream (here and in
    # plot_session_diagnostics) reduces to a plain, uncorrected threshold
    # with no multiple-comparisons control.
    pvals = np.array([e[use_p] for e in raw], dtype=float)
    if config.get("apply_fdr", True):
        fdr_group_by = config.get("fdr_group_by", "imec_id")
        if fdr_group_by in (None, "session"):
            group_arr = np.zeros(len(raw), dtype=object)   # one pooled group
        else:
            group_arr = np.array([e["meta"].get(fdr_group_by) for e in raw], dtype=object)
        qvals = np.full(len(raw), np.nan)
        for grp in pd.unique(group_arr):
            mask   = group_arr == grp
            sub_p  = pvals[mask]
            finite = np.isfinite(sub_p)
            if finite.any():
                _, qcorr, _, _ = multipletests(sub_p[finite], alpha=alpha, method="fdr_bh")
                sub_q = np.full(finite.shape, np.nan)
                sub_q[finite] = qcorr
                qvals[mask] = sub_q
    else:
        qvals = pvals.copy()   # no correction: p_fdr is just the raw p
    sig = np.isfinite(qvals) & (qvals <= alpha)

    rows = [NeuronTestResult(**e["meta"], n_bins=e["n_bins"], N=e["N"], r=e["r"],
                             m=e["m"], p_conservative=e["p_conservative"],
                             p_approx=e["p_approx"], p_fdr=q, significant=bool(s),
                             error=e["error"])
            for e, s, q in zip(raw, sig, qvals)]
    return pd.DataFrame([r.__dict__ for r in rows])


# ============================================================================
# SINGLE-MOUSE, SINGLE-PROBE DIAGNOSTIC PLOTS
# ============================================================================

def plot_probe_sanity(unit_table, trial_table, session_id, mouse_id,
                      session_day, imec_id, config, out_dir):
    """One figure per probe: population mean firing rate vs. probe motion,
    both time-binned the same way the actual test uses. Quick visual check
    that DREDge estimates are sensible and roughly track population activity."""
    trials_all = (trial_table[trial_table["session_id"] == session_id]
                  .sort_values("start_time").reset_index(drop=True))
    t_start = float(trials_all["start_time"].min()) if len(trials_all) else np.nan
    t_end   = float(trials_all["start_time"].max()) if len(trials_all) else np.nan
    if np.isnan(t_start) or np.isnan(t_end):
        warnings.warn(f"[plot_probe_sanity] {mouse_id}/{session_day} imec{imec_id}: "
                      f"no trials found for session_id={session_id}; skipping.")
        return

    units = unit_table[(unit_table["session_id"] == session_id) &
                       (unit_table["electrode_group"].apply(_get_imec_id) == imec_id)]
    if units.empty:
        warnings.warn(f"[plot_probe_sanity] {mouse_id}/{session_day} imec{imec_id}: "
                      f"no units resolved to this imec_id via electrode_group; skipping.")
        return

    bin_width = config["drift_test_bin_width"]
    pop_rates = []
    centers = None
    for _, u in units.iterrows():
        spk = np.asarray(u["spike_times"], dtype=float)
        c, r = compute_binned_firing_rate(spk, t_start, t_end, bin_width)
        if len(c) == 0:
            continue
        centers = c
        pop_rates.append(r)
    if not pop_rates or centers is None:
        warnings.warn(f"[plot_probe_sanity] {mouse_id}/{session_day} imec{imec_id}: "
                      f"no unit produced any time bins (t_start={t_start}, t_end={t_end}, "
                      f"bin_width={bin_width}); skipping.")
        return
    pop_fr = np.nanmean(np.stack(pop_rates, axis=0), axis=0)

    depth_med = float(pd.to_numeric(units["depth"], errors="coerce").median())
    try:
        mp = build_motion_path(config, mouse_id, session_id, imec_id)
        mo = load_motion(mp)
        motion_trace = get_motion_at_times_and_depth(mo, centers, depth_med)
    except Exception as exc:
        warnings.warn(f"Motion unavailable for {mouse_id}/{session_day} imec{imec_id}: {exc}")
        motion_trace = None

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W_FULL, FIG_H_UNIT * 1.6),
                             sharex=True, gridspec_kw=dict(hspace=0.15))
    axes[0].plot(centers, pop_fr, color=FR_COLOR, lw=0.8)
    axes[0].set_ylabel("Pop. mean FR (Hz)")
    axes[0].set_title(f"{mouse_id} / {session_day} — imec{imec_id} — drift sanity check "
                      f"({len(units)} units, {bin_width}s bins)")
    if motion_trace is not None:
        axes[1].plot(centers, motion_trace, color=MOTION_COLOR, lw=0.8)
        axes[1].set_ylabel("Motion (µm)")
        axes[1].set_xlabel("Time (s)")
    else:
        axes[1].text(0.5, 0.5, "Motion unavailable", ha="center",
                     va="center", transform=axes[1].transAxes)
    _save_fig(fig, out_dir / "drift_sanity",
             f"{mouse_id}_{session_day}_imec{imec_id}_drift_sanity",
             config["figure_dpi"])


def _plot_unit_row(axes_row, uid, sig, r, use_p_label, use_p_val, p_fdr,
                   tx, act, fv, N, method, fdr_label="FDR-q"):
    """Fill one row of 5 axes (FR | motion | overlay | scatter | shift-sweep
    histogram) for a single unit. Shared by plot_process_figure (sampled
    examples) and plot_all_units_diagnostics (every unit). fdr_label should
    be "raw p (no FDR)" when config["apply_fdr"]=False, since p_fdr is then
    just a pass-through of the raw p, not an actual correction."""
    ax0, ax1, ax_ov, ax3, ax4 = axes_row

    ax0.plot(tx, act, color=FR_COLOR, lw=0.9, alpha=0.85)
    ax0.set_xlabel("Time (s)"); ax0.set_ylabel("FR (Hz)")
    ax0.set_title(f"{'Sig' if sig else 'n.s.'} — unit {uid}")

    ax1.plot(tx, fv, color=MOTION_COLOR, lw=0.9)
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Motion (µm)")

    ax_ov2 = ax_ov.twinx()
    l1, = ax_ov.plot(tx, act, color=FR_COLOR, lw=0.9, alpha=0.85, label="FR")
    l2, = ax_ov2.plot(tx, fv, color=MOTION_COLOR, lw=0.9, alpha=0.85, label="Motion")
    ax_ov.set_ylabel("FR (Hz)", color=FR_COLOR)
    ax_ov.tick_params(axis="y", labelcolor=FR_COLOR)
    ax_ov2.set_ylabel("Motion (µm)", color=MOTION_COLOR)
    ax_ov2.tick_params(axis="y", labelcolor=MOTION_COLOR)
    ax_ov2.spines["top"].set_visible(False)
    ax_ov.set_xlabel("Time (s)")
    ax_ov.set_title("FR vs. Motion (aligned)")
    ax_ov.legend(handles=[l1, l2], fontsize=6, loc="upper right")

    T = len(act)
    ax3.scatter(fv, act, s=5, alpha=0.4, color=FR_COLOR, linewidths=0, rasterized=True)
    if T > 2:
        sl, ic, *_ = stats.linregress(fv, act)
        xr = np.array([fv.min(), fv.max()])
        ax3.plot(xr, sl * xr + ic, "k--", lw=1.1)
    q_str = f"  {fdr_label}={p_fdr:.3f}" if np.isfinite(p_fdr) else ""
    ax3.set_xlabel("Motion (µm)"); ax3.set_ylabel("FR (Hz)")
    ax3.set_title(f"r={r:.3f}  {use_p_label}={use_p_val:.3f}{q_str}")

    # Harris (2021) exhaustive shift sweep: |corr| at every consecutive
    # shift -N..N (2N+1 total, deterministic -- no randomness).
    r0, p_cons, p_appx, m, D, scores = shift_test_exhaustive(
        act, fv, N, method=method, return_scores=True)
    sc = scores[np.isfinite(scores)]
    if sc.size:
        v0 = scores[N]
        ax4.hist(sc, bins=30, color="steelblue", alpha=0.7)
        ax4.axvline(v0, color=FR_COLOR, lw=1.8, label=f"|r| at shift 0 = {v0:.3f}")
        ax4.legend(fontsize=7)
        ax4.set_xlabel("|corr| across shifts -N..N"); ax4.set_ylabel("Count")
        ax4.set_title(f"Shift sweep (m={m}, N={N}, D={D})\n"
                      f"p_cons={p_cons:.3f}  p_approx={p_appx:.3f}")
    else:
        ax4.text(0.5, 0.5, "unavailable", ha="center", va="center",
                 transform=ax4.transAxes)


def _prep_unit_timeseries(unit, nrow, mo, bin_width, N):
    """Compute (tx, act, fv) for one unit, or None if data is insufficient
    -- same requirement as the real test (T > 2N)."""
    spk = np.asarray(unit["spike_times"], dtype=float)
    t_start, t_end = nrow.get("t_start"), nrow.get("t_end")
    if t_start is None or pd.isna(t_start):
        return None
    tx, act = compute_binned_firing_rate(spk, t_start, t_end, bin_width)
    if len(tx) == 0:
        return None
    try:
        fv = get_motion_at_times_and_depth(mo, tx, float(unit["depth"]))
    except Exception:
        return None
    fv = np.asarray(fv, dtype=float)
    valid = np.isfinite(act) & np.isfinite(fv)
    tx, act, fv = tx[valid], act[valid], fv[valid]
    if len(act) <= 2 * N:
        return None
    return tx, act, fv


def plot_process_figure(results, unit_table, session_id, mouse_id, session_day,
                        imec_id, config, out_dir, rng: np.random.Generator):
    """Process figure for one probe: n_example_neurons from the
    significant and non-significant pools, each shown as FR trace | motion
    trace | aligned overlay | scatter | shift-test score distribution
    (Harris 2021: |corr| at every consecutive shift -N..N)."""
    trials = results[(results["imec_id"] == imec_id) & results["r"].notna()]
    if trials.empty:
        n_total = len(results[results["imec_id"] == imec_id])
        warnings.warn(f"[plot_process_figure] {mouse_id}/{session_day} imec{imec_id}: "
                      f"0 of {n_total} units on this probe have a valid r (all NaN/errored). "
                      f"Check results['error'] / warning.txt for this session -- a common "
                      f"cause is drift_test_N too large for the session length (need "
                      f"n_bins > 2*N). Skipping process figure.")
        return

    n_ex = config.get("n_example_neurons", 6)
    chosen = []
    for pool in [trials[trials["significant"]], trials[~trials["significant"]]]:
        if len(pool):
            k = min(n_ex, len(pool))
            samp = pool.sample(n=k, random_state=int(rng.integers(1e6)))
            chosen.extend(samp.iloc[i] for i in range(k))
    if not chosen:
        warnings.warn(f"[plot_process_figure] {mouse_id}/{session_day} imec{imec_id}: "
                      f"{len(trials)} units had valid r, but none were sampled into "
                      f"significant/non-significant example pools; skipping.")
        return

    units = unit_table[unit_table["session_id"] == session_id].set_index("neuron_id")

    bin_width = config["drift_test_bin_width"]
    N         = config["drift_test_N"]
    method    = config.get("drift_test_corr_method", "pearson")
    use_p     = config.get("use_p", "p_conservative")
    fdr_label = "FDR-q" if config.get("apply_fdr", True) else "raw p (no FDR)"

    try:
        mp = build_motion_path(config, mouse_id, session_id, imec_id)
        mo = load_motion(mp)
    except Exception as exc:
        warnings.warn(f"[plot_process_figure] {mouse_id}/{session_day} imec{imec_id}: "
                      f"motion unavailable ({exc}); skipping.")
        return

    n_rows = len(chosen)
    fig, axes = plt.subplots(n_rows, 5,
                             figsize=(FIG_W_FULL * 1.25, FIG_H_UNIT * n_rows),
                             gridspec_kw=dict(wspace=0.55, hspace=0.55),
                             squeeze=False)

    for ri, nrow in enumerate(chosen):
        uid = nrow["neuron_id"]
        try:
            unit = units.loc[uid]
        except KeyError:
            continue
        prep = _prep_unit_timeseries(unit, nrow, mo, bin_width, N)
        if prep is None:
            continue
        tx, act, fv = prep
        _plot_unit_row(axes[ri, :], uid, nrow["significant"], nrow["r"],
                       use_p, nrow[use_p], nrow["p_fdr"], tx, act, fv, N, method,
                       fdr_label=fdr_label)

    fig.suptitle(f"{mouse_id} / {session_day} — imec{imec_id} — shift-test process figure", y=1.02)
    _save_fig(fig, out_dir / "process_figures",
             f"{mouse_id}_{session_day}_imec{imec_id}_process", config["figure_dpi"])


def plot_all_units_diagnostics(results, unit_table, session_id, mouse_id, session_day,
                               imec_id, config, out_dir):
    """
    Optional (config['plot_all_units']=True): one individual figure per
    UNIT on this probe (not a sample), saved into out_dir/diag/significant/
    or out_dir/diag/nonsignificant/. Can produce hundreds of files -- PNG
    only (no PDF) to keep this from being too heavy.
    """
    trials = results[(results["imec_id"] == imec_id) & results["r"].notna()]
    if trials.empty:
        return

    units = unit_table[unit_table["session_id"] == session_id].set_index("neuron_id")
    bin_width = config["drift_test_bin_width"]
    N         = config["drift_test_N"]
    method    = config.get("drift_test_corr_method", "pearson")
    use_p     = config.get("use_p", "p_conservative")
    fdr_label = "FDR-q" if config.get("apply_fdr", True) else "raw p (no FDR)"

    try:
        mp = build_motion_path(config, mouse_id, session_id, imec_id)
        mo = load_motion(mp)
    except Exception as exc:
        warnings.warn(f"[plot_all_units_diagnostics] {mouse_id}/{session_day} imec{imec_id}: "
                      f"motion unavailable ({exc}); skipping.")
        return

    n_done = 0
    for _, nrow in trials.iterrows():
        uid = nrow["neuron_id"]
        try:
            unit = units.loc[uid]
        except KeyError:
            continue
        prep = _prep_unit_timeseries(unit, nrow, mo, bin_width, N)
        if prep is None:
            continue
        tx, act, fv = prep

        fig, axes = plt.subplots(1, 5, figsize=(FIG_W_FULL * 1.25, FIG_H_UNIT),
                                 gridspec_kw=dict(wspace=0.55))
        _plot_unit_row(axes, uid, nrow["significant"], nrow["r"],
                       use_p, nrow[use_p], nrow["p_fdr"], tx, act, fv, N, method,
                       fdr_label=fdr_label)

        subfolder = "significant" if nrow["significant"] else "nonsignificant"
        stem = f"{mouse_id}_{session_day}_imec{imec_id}_unit{uid}"
        target_dir = out_dir / "diag" / subfolder
        target_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(target_dir / f"{stem}.png"), dpi=config["figure_dpi"])
        plt.close(fig)
        n_done += 1

    print(f"    -> {n_done} per-unit diagnostic figures -> {out_dir / 'diag'}/"
         f"{{significant,nonsignificant}}/ (imec{imec_id})")


def _strip_two_groups(ax, group_a_vals, group_b_vals, ylabel, jitter=0.14,
                      labels=("p>0.05", "p<=0.05")):
    """Jittered scatter + mean +/- SEM for two groups, side by side at
    x=0 (group_a, grey) and x=1 (group_b, red)."""
    rng = np.random.default_rng(0)
    group_a_vals = np.asarray(group_a_vals, dtype=float)
    group_a_vals = group_a_vals[np.isfinite(group_a_vals)]
    group_b_vals = np.asarray(group_b_vals, dtype=float)
    group_b_vals = group_b_vals[np.isfinite(group_b_vals)]
    for pos, vals, color in [(0, group_a_vals, NONSIG_COLOR), (1, group_b_vals, SIG_COLOR)]:
        if len(vals) == 0:
            continue
        x = pos + rng.uniform(-jitter, jitter, len(vals))
        ax.scatter(x, vals, s=4, color=color, alpha=0.35, linewidths=0, rasterized=True)
        mean = vals.mean()
        sem  = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        ax.plot([pos - 0.25, pos + 0.25], [mean, mean], color=color, lw=2.2, zorder=4)
        ax.errorbar(pos, mean, yerr=sem, color=color, fmt="none",
                    capsize=3, lw=1.2, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{labels[0]}\n(n={len(group_a_vals)})",
                        f"{labels[1]}\n(n={len(group_b_vals)})"])
    ax.set_ylabel(ylabel)


def _cdf_two_groups(ax, group_a_vals, group_b_vals, xlabel,
                    labels=("p>0.05", "p<=0.05")):
    """Empirical CDF of the same two groups, same colors as _strip_two_groups."""
    for vals, color, label in [(group_a_vals, NONSIG_COLOR, labels[0]),
                               (group_b_vals, SIG_COLOR, labels[1])]:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        xs = np.sort(vals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", color=color, lw=1.6, label=f"{label} (n={len(xs)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6, loc="lower right")


def plot_session_diagnostics(results, unit_table, session_id, mouse_id, session_day,
                             out_dir, config):
    """
    One figure per session summarizing the motion-test output: signed r,
    |r|, mean firing rate, spike amplitude -- top row as strip plots, bottom
    row as CDFs of the same quantities. Grouping is by RAW p <= alpha
    (config["use_p"]), explicitly NOT the FDR-corrected 'significant' column
    -- this matches the exclusion-criterion use case (flag units, don't
    treat this as a corrected discovery claim). Pools across probes -- this
    is a session-level overview, not a per-probe diagnostic.
    """
    valid = results[results["r"].notna()].copy()
    if valid.empty:
        warnings.warn(f"[plot_session_diagnostics] {mouse_id}/{session_day}: "
                      f"no units with a valid r; skipping.")
        return

    valid["abs_r"] = valid["r"].abs()
    alpha = config["alpha"]
    use_p = config.get("use_p", "p_conservative")
    valid["raw_sig"] = valid[use_p] <= alpha   # NaN -> False, correctly

    units_idx = unit_table[unit_table["session_id"] == session_id].set_index("neuron_id")

    # per-unit mean firing rate over [t_start, t_end] (not stored in results
    # -- needs the raw spike train, so pulled from unit_table here)
    fr_vals = []
    for _, row in valid.iterrows():
        try:
            spk = np.asarray(units_idx.loc[row["neuron_id"], "spike_times"], dtype=float)
            dur = row["t_end"] - row["t_start"]
            fr = float(np.sum((spk >= row["t_start"]) & (spk <= row["t_end"])) / dur) \
                if dur > 0 else np.nan
        except Exception:
            fr = np.nan
        fr_vals.append(fr)
    valid["mean_fr"] = fr_vals

    # spike amplitude -- unit_table's actual column is 'rawAmplitude'
    amp_col = "rawAmplitude" if "rawAmplitude" in unit_table.columns else None
    if amp_col is not None:
        amp_vals = []
        for _, row in valid.iterrows():
            try:
                amp_vals.append(float(units_idx.loc[row["neuron_id"], amp_col]))
            except Exception:
                amp_vals.append(np.nan)
        valid["amp"] = amp_vals
    else:
        warnings.warn(f"[plot_session_diagnostics] {mouse_id}/{session_day}: "
                      f"no 'rawAmplitude' column in unit_table; "
                      f"amplitude panel will be empty.")
        valid["amp"] = np.nan

    sig  = valid[valid["raw_sig"]]
    nsig = valid[~valid["raw_sig"]]
    has_amp = valid["amp"].notna().any()

    fig, axes = plt.subplots(2, 4, figsize=(FIG_W_FULL * 4 / 3, FIG_H_UNIT * 2.4),
                             gridspec_kw=dict(wspace=0.5, hspace=0.55))

    panels = [
        ("r",       "Signed r",     "Signed correlation"),
        ("abs_r",   "|r|",          "Absolute correlation"),
        ("mean_fr", "Mean FR (Hz)", "Firing rate"),
        ("amp",     "Amplitude",    "Spike amplitude"),
    ]
    for ci, (col, ylabel, title) in enumerate(panels):
        if col == "amp" and not has_amp:
            axes[0, ci].axis("off")
            axes[1, ci].axis("off")
            continue
        _strip_two_groups(axes[0, ci], nsig[col], sig[col], ylabel)
        axes[0, ci].set_title(title)
        if col == "r":
            axes[0, ci].axhline(0, color="k", lw=0.6, ls="--")
        _cdf_two_groups(axes[1, ci], nsig[col], sig[col], ylabel)

    frac_sig = len(sig) / max(len(valid), 1)
    fig.suptitle(f"{mouse_id} / {session_day} — motion-test diagnostics — "
                f"{len(sig)}/{len(valid)} raw p<={alpha} ({frac_sig*100:.1f}%)  "
                f"[top: strip, bottom: CDF]", y=1.03)
    _save_fig(fig, out_dir / "session_diagnostics",
             f"{mouse_id}_{session_day}_diagnostics", config["figure_dpi"])


def _save_fig(fig, subfolder: Path, stem: str, dpi: int = 300):
    subfolder.mkdir(parents=True, exist_ok=True)
    base = subfolder / stem
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".png", dpi=dpi)
    plt.close(fig)
    print(f"    -> {base}.[pdf|png]")


# ============================================================================
# PER-SESSION WORKER
# ============================================================================

def _process_one_session(args):
    (session_id, mouse_id, session_day, unit_table, trial_table, cfg) = args

    print(f"[motion_shift_test] {mouse_id} / {session_day}  (session {session_id})")

    try:
        results = analyze_session(unit_table, trial_table,
                                  session_id, mouse_id, session_day, cfg)
    except Exception as exc:
        warnings.warn(f"Session {mouse_id}/{session_day} failed: {exc}")
        return None

    out_dir = get_per_session_output_dir(cfg, mouse_id, session_day)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{mouse_id}_{session_day}_motion_shift_test_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"  -> {len(results)} rows -> {csv_path}")

    # diagnostic: where is the signal (or lack of it) actually lost?
    # per probe: how many valid tests, the raw p-value distribution before
    # correction, and how many survive vs. don't after BH-FDR. This is
    # printed unconditionally (not just on a suspicious result) so "zero
    # significant units" is diagnosable from the console instead of a guess.
    use_p = cfg.get("use_p", "p_conservative")
    fdr_on = cfg.get("apply_fdr", True)
    tail = "-> {n} survive FDR" if fdr_on else "-> {n} pass uncorrected threshold (apply_fdr=False)"
    for probe, sub in results.groupby("imec_id"):
        valid = sub[sub["r"].notna()]
        n_valid = len(valid)
        if n_valid == 0:
            print(f"  [diag] imec{probe}: 0/{len(sub)} units have a valid r "
                 f"(all failed -- check results['error'] / warning.txt)")
            continue
        raw_p = valid[use_p].dropna()
        n_raw_sig = int((raw_p <= cfg["alpha"]).sum())
        n_fdr_sig = int(valid["significant"].sum())
        print(f"  [diag] imec{probe}: {n_valid} valid units | "
             f"raw {use_p} min={raw_p.min():.4f} median={raw_p.median():.4f} | "
             f"{n_raw_sig} raw p<={cfg['alpha']} before FDR "
             + tail.format(n=n_fdr_sig))

    write_drift_test_warnings(results, out_dir, mouse_id, session_day)

    rng = np.random.default_rng(cfg["random_seed"])
    # derive probes directly from unit_table's electrode_group, not from
    # results['imec_id'] -- if every unit on a probe failed to resolve its
    # imec_id or the motion test errored out, results would silently have
    # no rows for that probe, and it would get skipped entirely, including
    # the sanity-check plot that would help diagnose why it failed.
    session_units = unit_table[unit_table["session_id"] == session_id]
    imec_ids = sorted({iid for iid in session_units["electrode_group"].apply(_get_imec_id)
                       if iid is not None})
    if not imec_ids:
        sample_vals = session_units["electrode_group"].dropna().unique()[:3].tolist()
        warnings.warn(
            f"[_process_one_session] {mouse_id}/{session_day}: could not resolve any "
            f"imec_id from {len(session_units)} units' electrode_group column -- "
            f"NO PLOTS will be generated for this session (CSV was still saved). "
            f"Sample electrode_group values seen: {sample_vals}. "
            f"_get_imec_id expects a dict with 'imecID'/'imec_id', a JSON-ish string, "
            f"or a string/number containing 'imecN'.")
    print(f"  -> {len(imec_ids)} probe(s) found: {imec_ids}")

    for imec_id in imec_ids:
        for fn, label in [
            (lambda: plot_probe_sanity(unit_table, trial_table, session_id, mouse_id,
                                       session_day, imec_id, cfg, out_dir),
             f"probe sanity (imec{imec_id})"),
            (lambda: plot_process_figure(results, unit_table, session_id, mouse_id,
                                         session_day, imec_id, cfg, out_dir, rng),
             f"process figure (imec{imec_id})"),
        ]:
            try:
                fn()
            except Exception as exc:
                warnings.warn(f"{label} failed for {mouse_id}/{session_day}: {exc}")

        if cfg.get("plot_all_units", False):
            try:
                plot_all_units_diagnostics(results, unit_table, session_id, mouse_id,
                                           session_day, imec_id, cfg, out_dir)
            except Exception as exc:
                warnings.warn(f"plot_all_units_diagnostics (imec{imec_id}) failed for "
                              f"{mouse_id}/{session_day}: {exc}")

    try:
        plot_session_diagnostics(results, unit_table, session_id, mouse_id,
                                 session_day, out_dir, cfg)
    except Exception as exc:
        warnings.warn(f"session diagnostics failed for {mouse_id}/{session_day}: {exc}")

    return csv_path


# ============================================================================
# TOP-LEVEL ORCHESTRATION
# ============================================================================

def run_motion_shift_test_analysis(unit_table, trial_table, output_path,
                                   config=None, mouse_ids=None,
                                   session_days=None) -> pd.DataFrame:
    """
    Run the motion shift test for every (session_id, mouse_id, session_day)
    in unit_table, save one CSV per session, and produce single-mouse,
    single-probe diagnostic plots. Returns the combined long-format results.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if output_path:
        cfg["combined_results_root"] = output_path

    required_unit  = {"session_id", "mouse_id", "cluster_id", "electrode_group",
                      "neuron_id", "depth", "bc_label", "spike_times"}
    required_trial = {"session_id", "start_time"}
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
    if mouse_ids is not None:
        session_keys = session_keys[session_keys["mouse_id"].isin(mouse_ids)]
    if session_days is not None:
        session_keys = session_keys[session_keys["session_day"].isin(session_days)]

    args_list = [
        (row["session_id"], row["mouse_id"], row["session_day"],
         unit_table[unit_table.session_id == row["session_id"]],
         trial_table[trial_table.session_id == row["session_id"]], cfg)
        for _, row in session_keys.iterrows()
    ]

    n_workers = cfg.get("n_workers", 1)
    all_results = []

    if n_workers > 1:
        csv_paths = Parallel(n_jobs=n_workers, backend="loky")(
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

    combined = pd.concat(all_results, ignore_index=True)
    print(f"[motion_shift_test] Done. {len(combined)} rows across "
         f"{combined[['mouse_id','session_id']].drop_duplicates().shape[0]} session(s).")
    return combined


if __name__ == "__main__":
    print("Import run_motion_shift_test_analysis and call with your "
         "unit_table/trial_table.")