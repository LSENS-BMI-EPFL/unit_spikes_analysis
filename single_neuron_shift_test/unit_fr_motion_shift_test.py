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
SIG_COLOR    = "#2980B9"
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

    "random_seed": 0,
    "figure_dpi":  300,
    "alpha":       0.05,
    "n_workers":   25,          # parallel sessions (joblib)

    "drift_test_corr_method": "pearson",   # "pearson" or "spearman"
    "drift_test_shift_frac":  (0.10, 0.90),
    "drift_test_min_bins":    10,
    "drift_test_bin_width":   1.0,     # seconds
    "drift_test_n_shifts":    100,

    "n_example_neurons": 6,    # example units per sig/n.s. group, per probe
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
    lags of y relative to x, |lag| drawn uniformly within shift_frac * len(x).
    Returns (r_obs, p, m); if return_null=True also returns null_vals array.
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


def time_binned_drift_test(spike_times, motion, depth, t_start, t_end,
                           bin_width, n_shifts, seed, config):
    """Full recipe: bin FR and displacement over time, correlate, shift-test.
    Returns a dict with r, p, m, n_bins, error."""
    method     = config.get("drift_test_corr_method", "pearson")
    shift_frac = config.get("drift_test_shift_frac", (0.10, 0.90))
    min_bins   = config.get("drift_test_min_bins", 10)

    centers, rate = compute_binned_firing_rate(spike_times, t_start, t_end, bin_width)
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
        # zero variance -> correlation undefined but not missing: a constant
        # signal cannot covary with anything, so r=0, p=1 is the well-defined
        # answer, not NaN.
        return dict(r=0.0, p=1.0, m=0, n_bins=len(r_v),
                    error="constant vector (r set to 0, p to 1)")

    rng = np.random.default_rng(seed)
    r_obs, p, m = shift_null_test(r_v, d_v, n_shifts, rng,
                                  method=method, shift_frac=shift_frac)
    return dict(r=r_obs, p=p, m=m, n_bins=len(r_v), error=None)


def _drift_test_seed(base_seed, session_id, neuron_id) -> np.random.SeedSequence:
    """Deterministic, per-neuron seed derived from the repo's base
    random_seed, so shift draws are reproducible but not identical
    across neurons."""
    tag = abs(hash((str(session_id), str(neuron_id)))) % (2**32)
    return np.random.SeedSequence([int(base_seed), tag])


# Error prefixes meaning "could not compute" (missing/insufficient data),
# as opposed to "computed and the answer is a well-defined null" (constant
# vector -> r=0, p=1). Only the former gets flagged/warned.
DRIFT_TEST_DATA_WARNING_PREFIXES = (
    "too few bins", "motion lookup failed", "too few valid bins",
    "could not determine imecID",
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
    r:               float = np.nan
    m:               int = 0
    p:               float = np.nan
    p_fdr:           float = np.nan
    significant:     bool = False
    error:           Optional[str] = None


# ============================================================================
# PER-SESSION ANALYSIS
# ============================================================================

def analyze_session(unit_table, trial_table, session_id, mouse_id,
                    session_day, config) -> pd.DataFrame:
    alpha     = config["alpha"]
    bin_width = config["drift_test_bin_width"]
    n_shifts  = config["drift_test_n_shifts"]
    base_seed = config.get("random_seed", 0)

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
            res = dict(r=np.nan, p=np.nan, m=0, n_bins=0,
                      error=motion_error or "no trials for session bounds")
        else:
            seed = _drift_test_seed(base_seed, session_id, neuron_id)
            res = time_binned_drift_test(spk, motion_obj, depth, t_start, t_end,
                                         bin_width, n_shifts, seed, config)
        raw.append(dict(meta=meta, **res))

    # within-session BH-FDR; `significant` is explicitly derived from p_fdr
    # (q <= alpha), not statsmodels' `rejected` output directly -- same
    # result by definition of BH-FDR, but makes the dependency explicit.
    pvals  = np.array([e["p"] for e in raw], dtype=float)
    finite = np.isfinite(pvals)
    qvals  = np.full(len(pvals), np.nan)
    if finite.any():
        _, qcorr, _, _ = multipletests(pvals[finite], alpha=alpha, method="fdr_bh")
        qvals[finite] = qcorr
    sig = np.isfinite(qvals) & (qvals <= alpha)

    rows = [NeuronTestResult(**e["meta"], n_bins=e["n_bins"], r=e["r"], m=e["m"],
                             p=e["p"], p_fdr=q, significant=bool(s), error=e["error"])
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
        return

    units = unit_table[(unit_table["session_id"] == session_id) &
                       (unit_table["electrode_group"].apply(_get_imec_id) == imec_id)]
    if units.empty:
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
        return
    pop_fr = np.nanmean(np.stack(pop_rates, axis=0), axis=0)

    depth_med = float(units["depth"].median())
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


def plot_probe_examples(results, unit_table, session_id, mouse_id, session_day,
                        imec_id, config, out_dir, rng: np.random.Generator):
    """Example-neuron figure for one probe: n_example_neurons from the
    significant and non-significant pools, each shown as FR trace | motion
    trace | aligned overlay | scatter | shift-null histogram."""
    trials = results[(results["imec_id"] == imec_id) & results["r"].notna()]
    if trials.empty:
        return

    n_ex = config.get("n_example_neurons", 6)
    chosen = []
    for pool in [trials[trials["significant"]], trials[~trials["significant"]]]:
        if len(pool):
            k = min(n_ex, len(pool))
            samp = pool.sample(n=k, random_state=int(rng.integers(1e6)))
            chosen.extend(samp.iloc[i] for i in range(k))
    if not chosen:
        return

    units = unit_table[unit_table["session_id"] == session_id].set_index("neuron_id")

    bin_width = config["drift_test_bin_width"]
    n_shifts  = config["drift_test_n_shifts"]
    method    = config.get("drift_test_corr_method", "pearson")
    sfrac     = config.get("drift_test_shift_frac", (0.10, 0.90))

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
        spk = np.asarray(unit["spike_times"], dtype=float)

        try:
            mp = build_motion_path(config, mouse_id, session_id, imec_id)
            mo = load_motion(mp)
        except Exception:
            continue

        t_start, t_end = nrow.get("t_start"), nrow.get("t_end")
        if t_start is None or pd.isna(t_start):
            continue

        tx, act = compute_binned_firing_rate(spk, t_start, t_end, bin_width)
        if len(tx) == 0:
            continue
        try:
            fv = get_motion_at_times_and_depth(mo, tx, float(unit["depth"]))
        except Exception:
            continue
        fv = np.asarray(fv, dtype=float)
        valid = np.isfinite(act) & np.isfinite(fv)
        tx, act, fv = tx[valid], act[valid], fv[valid]
        n = len(act)
        if n < 3:
            continue

        axes[ri, 0].plot(tx, act, color=FR_COLOR, lw=0.9, alpha=0.85)
        axes[ri, 0].set_xlabel("Time (s)"); axes[ri, 0].set_ylabel("FR (Hz)")
        axes[ri, 0].set_title(f"{'Sig' if nrow['significant'] else 'n.s.'} — unit {uid}")

        axes[ri, 1].plot(tx, fv, color=MOTION_COLOR, lw=0.9)
        axes[ri, 1].set_xlabel("Time (s)"); axes[ri, 1].set_ylabel("Motion (µm)")

        ax_ov  = axes[ri, 2]
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

        axes[ri, 3].scatter(fv, act, s=5, alpha=0.4, color=FR_COLOR,
                            linewidths=0, rasterized=True)
        if n > 2:
            sl, ic, *_ = stats.linregress(fv, act)
            xr = np.array([fv.min(), fv.max()])
            axes[ri, 3].plot(xr, sl * xr + ic, "k--", lw=1.1)
        q = nrow["p_fdr"] if np.isfinite(nrow["p_fdr"]) else np.nan
        q_str = f"  FDR-q={q:.3f}" if np.isfinite(q) else ""
        axes[ri, 3].set_xlabel("Motion (µm)"); axes[ri, 3].set_ylabel("FR (Hz)")
        axes[ri, 3].set_title(f"r={nrow['r']:.3f}  raw p={nrow['p']:.3f}{q_str}")

        seed = _drift_test_seed(config.get("random_seed", 0), session_id, uid)
        rng4 = np.random.default_rng(seed)
        r_obs, p_val, m, null_vals = shift_null_test(
            act, fv, n_shifts, rng4, method=method, shift_frac=sfrac, return_null=True)
        ax4 = axes[ri, 4]
        nv = null_vals[np.isfinite(null_vals)]
        if nv.size:
            ax4.hist(nv, bins=30, color="steelblue", alpha=0.7)
            ax4.axvline(r_obs, color=FR_COLOR, lw=1.8, label=f"obs r={r_obs:.3f}")
            ax4.axvline(-r_obs, color=FR_COLOR, lw=1.0, ls=":", alpha=0.6)
            q_str2 = f", q={q:.3f}" if np.isfinite(q) else ""
            ax4.legend(fontsize=7)
            ax4.set_xlabel("Null shift r"); ax4.set_ylabel("Count")
            ax4.set_title(f"Shift null (p={p_val:.3f}{q_str2}, m={m})")
        else:
            ax4.text(0.5, 0.5, "unavailable", ha="center", va="center",
                     transform=ax4.transAxes)

    fig.suptitle(f"{mouse_id} / {session_day} — imec{imec_id} — example neurons", y=1.02)
    _save_fig(fig, out_dir / "example_neurons",
             f"{mouse_id}_{session_day}_imec{imec_id}_examples", config["figure_dpi"])


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
    for imec_id in imec_ids:
        for fn, label in [
            (lambda: plot_probe_sanity(unit_table, trial_table, session_id, mouse_id,
                                       session_day, imec_id, cfg, out_dir),
             f"probe sanity (imec{imec_id})"),
            (lambda: plot_probe_examples(results, unit_table, session_id, mouse_id,
                                         session_day, imec_id, cfg, out_dir, rng),
             f"probe examples (imec{imec_id})"),
        ]:
            try:
                fn()
            except Exception as exc:
                warnings.warn(f"{label} failed for {mouse_id}/{session_day}: {exc}")

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