"""
noise_unit_detection.py
-----------------------
Modular pipeline for flagging residual noise units in Neuropixels recordings.

Each metric is an independent function with a standard signature:
    metric_fn(unit_row, spike_times, **kwargs) -> (score: float, flagged: bool)

Add / remove metrics from METRIC_REGISTRY to customise the pipeline.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d


# ─────────────────────────────────────────────────────────────────────────────
# Data contract
# ─────────────────────────────────────────────────────────────────────────────
# unit_table  : pd.DataFrame, one row per unit.
#   Required columns : "unit_id", "spike_times"  (array of spike times in seconds)
#   Optional columns : "waveform_mean"            (1-D array, µV)
#
# trial_table : pd.DataFrame, one row per trial.
#   Required columns : "start_time"               (float, seconds)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _isi_array(spike_times: np.ndarray) -> np.ndarray:
    st = np.sort(spike_times)
    return np.diff(st) if len(st) > 1 else np.array([])


# ─────────────────────────────────────────────────────────────────────────────
# Individual metric functions
#   Signature: (unit_row, spike_times, **kwargs) -> (score, flagged)
# ─────────────────────────────────────────────────────────────────────────────

def metric_firing_rate(
    unit_row: pd.Series,
    spike_times: np.ndarray,
    *,
    min_rate: float = 0.05,               # Hz – below this globally → likely dead channel
    max_rate: float = 200.0,              # Hz – above this globally → almost certainly noise
    n_windows: int = 10,                  # equal-duration windows to split recording into
    max_local_rate: float | None = None,  # Hz cap per window; defaults to 2× max_rate
    max_local_ratio: float = 5.0,         # flag if any window > ratio × global mean rate
    **_,
) -> tuple[float, bool]:
    """
    Mean firing rate over the whole recording, plus a per-window (decile) check.

    The recording is divided into ``n_windows`` equal-duration bins.  A unit
    is flagged if:
      - its global mean rate is outside [min_rate, max_rate], OR
      - any single window exceeds ``max_local_rate`` (defaults to 2× max_rate), OR
      - any single window rate is more than ``max_local_ratio`` × the global
        mean rate (catches transient bursts that look fine on average).

    Returns the global mean firing rate as the score.
    """
    if len(spike_times) < 2:
        return 0.0, True

    t_min, t_max = spike_times.min(), spike_times.max()
    duration = t_max - t_min
    if duration <= 0:
        return 0.0, True

    global_rate = len(spike_times) / duration

    # ── global check ─────────────────────────────────────────────────────────
    if global_rate < min_rate or global_rate > max_rate:
        return float(global_rate), True

    # ── per-window check ──────────────────────────────────────────────────────
    _max_local   = max_local_rate if max_local_rate is not None else 2.0 * max_rate
    window_edges = np.linspace(t_min, t_max, n_windows + 1)
    window_dur   = duration / n_windows

    counts, _    = np.histogram(spike_times, bins=window_edges)
    local_rates  = counts / window_dur   # Hz per window

    local_rate_exceeded = bool(local_rates.max() > _max_local)
    ratio_exceeded      = bool(local_rates.max() / global_rate > max_local_ratio)

    return float(global_rate), local_rate_exceeded or ratio_exceeded


def metric_isi_violations(
    unit_row: pd.Series,
    spike_times: np.ndarray,
    *,
    refractory_ms: float = 1.5,    # absolute refractory period in ms
    max_violation_rate: float = 0.10,  # flag if >10 % of ISIs are violations
    **_,
) -> tuple[float, bool]:
    """
    Fraction of ISIs that fall inside the absolute refractory period.
    High values indicate spike-sorting errors or noise contamination.
    """
    isis = _isi_array(spike_times)
    if len(isis) == 0:
        return 1.0, True
    violation_frac = np.mean(isis < refractory_ms * 1e-3)
    return float(violation_frac), bool(violation_frac > max_violation_rate)


def metric_isi_cv(
    unit_row: pd.Series,
    spike_times: np.ndarray,
    *,
    max_cv: float = 4.0,  # CV > 4 is pathologically irregular
    **_,
) -> tuple[float, bool]:
    """
    Coefficient of variation of the ISI distribution.
    Noise bursts produce very high CVs; silent units produce near-zero CVs.
    """
    isis = _isi_array(spike_times)
    if len(isis) < 5:
        return np.nan, True
    cv = isis.std() / (isis.mean() + 1e-12)
    return float(cv), bool(cv > max_cv)


def metric_fano_factor(
    unit_row: pd.Series,
    spike_times: np.ndarray,
    *,
    bin_size_s: float = 0.1,
    recording_duration_s: float | None = None,
    max_fano: float = 10.0,
    **_,
) -> tuple[float, bool]:
    """
    Fano factor (variance / mean of binned spike counts).
    Extremely high values indicate burst-noise behaviour.
    """
    if len(spike_times) < 5:
        return np.nan, True
    t_min = spike_times.min()
    t_max = spike_times.max() if recording_duration_s is None else t_min + recording_duration_s
    bins = np.arange(t_min, t_max + bin_size_s, bin_size_s)
    counts, _ = np.histogram(spike_times, bins=bins)
    mean_c = counts.mean()
    if mean_c < 1e-9:
        return np.nan, True
    fano = counts.var() / mean_c
    return float(fano), bool(fano > max_fano)


def metric_waveform_amplitude(
    unit_row: pd.Series,
    spike_times: np.ndarray,
    *,
    min_amplitude_uv: float = 30.0,   # µV – below this → noise floor
    max_amplitude_uv: float = 2000.0, # µV – above this → artefact
    waveform_col: str = "waveform_mean",
    **_,
) -> tuple[float, bool]:
    """
    Peak-to-trough amplitude of the mean waveform.
    Returns (amplitude_µV, flagged).
    """
    waveform = unit_row.get(waveform_col, None)
    if waveform is None or not hasattr(waveform, "__len__") or len(waveform) == 0:
        return np.nan, False   # no waveform data → skip silently
    wf = np.asarray(waveform, dtype=float)
    amplitude = wf.max() - wf.min()
    flagged = amplitude < min_amplitude_uv or amplitude > max_amplitude_uv
    return float(amplitude), flagged


def metric_waveform_snr(
    unit_row: pd.Series,
    spike_times: np.ndarray,
    *,
    min_snr: float = 2.0,
    waveform_col: str = "waveform_mean",
    noise_baseline_samples: int = 10,  # samples at start used to estimate noise
    **_,
) -> tuple[float, bool]:
    """
    Signal-to-noise ratio of the mean waveform:  peak / std(baseline).
    """
    waveform = unit_row.get(waveform_col, None)
    if waveform is None or not hasattr(waveform, "__len__") or len(waveform) == 0:
        return np.nan, False
    wf = np.asarray(waveform, dtype=float)
    baseline_std = wf[:noise_baseline_samples].std() + 1e-12
    snr = np.abs(wf).max() / baseline_std
    return float(snr), bool(snr < min_snr)


def metric_waveform_trough_width(
    unit_row: pd.Series,
    spike_times: np.ndarray,
    *,
    min_trough_width_samples: int = 3,
    waveform_col: str = "waveform_mean",
    **_,
) -> tuple[float, bool]:
    """
    Half-width of the waveform trough (in samples).
    Very narrow troughs (<3 samples) are characteristic of noise.
    """
    waveform = unit_row.get(waveform_col, None)
    if waveform is None or not hasattr(waveform, "__len__") or len(waveform) == 0:
        return np.nan, False
    wf = np.asarray(waveform, dtype=float)
    trough_idx = int(np.argmin(wf))
    half_min = wf[trough_idx] / 2.0
    left = trough_idx - np.searchsorted(wf[:trough_idx][::-1] > half_min, True)
    right = trough_idx + np.searchsorted(wf[trough_idx:] > half_min, True)
    width = right - left
    return float(width), bool(width < min_trough_width_samples)


def metric_burst_index(
    unit_row: pd.Series,
    spike_times: np.ndarray,
    *,
    burst_isi_ms: float = 10.0,   # ISIs below this are 'burst' spikes
    max_burst_fraction: float = 0.60,
    **_,
) -> tuple[float, bool]:
    """
    Fraction of spikes belonging to a burst (ISI < burst_isi_ms).
    Dominantly burst-firing units are often noise.
    """
    isis = _isi_array(spike_times)
    if len(isis) == 0:
        return np.nan, True
    burst_frac = np.mean(isis < burst_isi_ms * 1e-3)
    return float(burst_frac), bool(burst_frac > max_burst_fraction)


# ─────────────────────────────────────────────────────────────────────────────
# Metric registry
#   Edit this dict to add / remove / re-parameterise metrics at runtime.
#   Keys   → display name
#   Values → dict with  "fn"     : the metric callable
#                        "kwargs" : threshold overrides (optional)
#                        "active" : whether to run (optional, default True)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_METRIC_REGISTRY: dict[str, dict] = {
    "firing_rate": {
        "fn": metric_firing_rate,
        "kwargs": {
            "min_rate": 0.05,
            "max_rate": 100.0,
            "n_windows": 10,           # deciles of the recording
            "max_local_rate": None,    # defaults to 2× max_rate inside the fn
            "max_local_ratio": 5.0,    # flag if any decile > 5× the global mean
        },
        "active": True,
    },
    "isi_violations": {
        "fn": metric_isi_violations,
        "kwargs": {"refractory_ms": 1.5, "max_violation_rate": 0.10},
        "active": True,
    },
    "isi_cv": {
        "fn": metric_isi_cv,
        "kwargs": {"max_cv": 4.0},
        "active": True,
    },
    "fano_factor": {
        "fn": metric_fano_factor,
        "kwargs": {"bin_size_s": 0.1, "max_fano": 10.0},
        "active": True,
    },
    "waveform_amplitude": {
        "fn": metric_waveform_amplitude,
        "kwargs": {"min_amplitude_uv": 30.0, "max_amplitude_uv": 2000.0},
        "active": True,
    },
    "waveform_snr": {
        "fn": metric_waveform_snr,
        "kwargs": {"min_snr": 2.0, "noise_baseline_samples": 10},
        "active": True,
    },
    "waveform_trough_width": {
        "fn": metric_waveform_trough_width,
        "kwargs": {"min_trough_width_samples": 3}, # at 30Hz
        "active": True,
    },
    "burst_index": {
        "fn": metric_burst_index,
        "kwargs": {"burst_isi_ms": 10.0, "max_burst_fraction": 0.60},
        "active": True,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Core scoring function
# ─────────────────────────────────────────────────────────────────────────────

def score_units(
    unit_table: pd.DataFrame,
    metric_registry: dict | None = None,
    min_flags_to_reject: int = 2,
) -> pd.DataFrame:
    """
    Run all active metrics on every unit and return an enriched DataFrame.

    Parameters
    ----------
    unit_table : pd.DataFrame
        Must have columns  "unit_id"  and  "spike_times".
    metric_registry : dict, optional
        Custom registry.  Defaults to DEFAULT_METRIC_REGISTRY.
    min_flags_to_reject : int
        A unit is labelled "noise" if at least this many metrics flag it.

    Returns
    -------
    pd.DataFrame
        Original columns + one score column and one flag column per metric,
        plus summary columns  "n_flags"  and  "is_noise".
    """
    if metric_registry is None:
        metric_registry = DEFAULT_METRIC_REGISTRY

    result = unit_table.copy()
    active_metrics = {k: v for k, v in metric_registry.items() if v.get("active", True)}

    for name, cfg in active_metrics.items():
        fn: Callable = cfg["fn"]
        kwargs: dict = cfg.get("kwargs", {})
        scores, flags = [], []
        for _, row in unit_table.iterrows():
            spikes = np.asarray(row["spike_times"])
            try:
                score, flag = fn(row, spikes, **kwargs)
            except Exception as exc:
                warnings.warn(f"Metric '{name}' failed for unit {row.get('unit_id', '?')}: {exc}")
                score, flag = np.nan, False
            scores.append(score)
            flags.append(flag)
        result[f"score_{name}"] = scores
        result[f"flag_{name}"] = flags

    flag_cols = [f"flag_{k}" for k in active_metrics]
    result["n_flags"] = result[flag_cols].sum(axis=1)
    result["is_noise"] = result["n_flags"] >= min_flags_to_reject
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Raster + waveform plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_unit(
    ax_raster: plt.Axes,
    ax_wave: plt.Axes | None,
    ax_isi: plt.Axes | None,
    unit_row: pd.Series,
    trial_starts: np.ndarray,
    pre_s: float,
    post_s: float,
    flag_cols: list[str],
    waveform_col: str = "waveform_mean",
) -> None:
    spike_times = np.sort(np.asarray(unit_row["spike_times"]))
    unit_id = unit_row.get("unit_id", "?")

    # ── raster ──────────────────────────────────────────────────────────────
    for trial_idx, t0 in enumerate(trial_starts):
        mask = (spike_times >= t0 - pre_s) & (spike_times <= t0 + post_s)
        rel = spike_times[mask] - t0
        ax_raster.scatter(rel, np.full(rel.size, trial_idx),
                          s=1.5, c="k", linewidths=0, rasterized=True)

    ax_raster.axvline(0, color="crimson", lw=0.8, ls="--", label="trial onset")
    ax_raster.set_xlim(-pre_s, post_s)
    ax_raster.set_ylabel("Trial #", fontsize=7)
    ax_raster.set_xlabel("Time from trial onset (s)", fontsize=7)
    ax_raster.tick_params(labelsize=6)

    ax_raster.set_title("Raster (trial-aligned)", fontsize=7, loc="left", color="dimgrey")

    # ── waveform ─────────────────────────────────────────────────────────────
    if ax_wave is not None:
        wf = unit_row.get(waveform_col, None)
        if wf is not None and hasattr(wf, "__len__") and len(wf) > 0:
            wf = np.asarray(wf, dtype=float)
            ax_wave.plot(wf, color="#2166ac", lw=1.2)
            ax_wave.axhline(0, color="grey", lw=0.5, ls=":")
            ax_wave.set_xlabel("Sample", fontsize=7)
            ax_wave.set_ylabel("µV", fontsize=7)
            ax_wave.tick_params(labelsize=6)
            ax_wave.set_title("Mean waveform", fontsize=7)
        else:
            ax_wave.text(0.5, 0.5, "no waveform", ha="center", va="center",
                         transform=ax_wave.transAxes, fontsize=8, color="grey")
            ax_wave.set_axis_off()

    # ── ISI histogram ────────────────────────────────────────────────────────
    if ax_isi is not None:
        isis = _isi_array(spike_times) * 1e3  # convert to ms
        if len(isis) > 1:
            bins = np.linspace(0, min(isis.max(), 200), 80)
            ax_isi.hist(isis, bins=bins, color="#4dac26", edgecolor="none")
            ax_isi.axvline(1.5, color="crimson", lw=0.8, ls="--", label="1.5 ms")
            ax_isi.set_xlabel("ISI (ms)", fontsize=7)
            ax_isi.set_ylabel("Count", fontsize=7)
            ax_isi.set_title("ISI histogram", fontsize=7)
            ax_isi.tick_params(labelsize=6)
        else:
            ax_isi.text(0.5, 0.5, "< 2 spikes", ha="center", va="center",
                        transform=ax_isi.transAxes, fontsize=8, color="grey")
            ax_isi.set_axis_off()


def plot_noise_units(
    scored_table: pd.DataFrame,
    trial_table: pd.DataFrame,
    pre_s: float = 0.5,
    post_s: float = 1.5,
    waveform_col: str = "waveform_mean",
    show_waveform: bool = True,
    show_isi: bool = True,
    noise_only: bool = True,
    save_path: str | None = None,
) -> list[plt.Figure]:
    """
    Plot one figure per flagged (or all) unit, each containing:
      - a raster aligned to trial onset  (left, wide)
      - mean waveform panel              (middle, optional)
      - ISI histogram                    (right, optional)

    Parameters
    ----------
    scored_table : pd.DataFrame
        Output of score_units().
    trial_table : pd.DataFrame
        Must contain column "start_time".
    pre_s / post_s : float
        Seconds before / after trial onset to show in the raster.
    show_waveform / show_isi : bool
        Toggle the waveform / ISI side panels.
    noise_only : bool
        If True, only plot units labelled is_noise=True.
    save_path : str, optional
        Directory in which to save one PDF per unit, named
        ``unit_<unit_id>.pdf``.

    Returns
    -------
    list of plt.Figure  (one entry per unit)
    """
    trial_starts = np.sort(trial_table["start_time"].values)
    flag_cols = [c for c in scored_table.columns if c.startswith("flag_")]

    subset = scored_table[scored_table["is_noise"]] if noise_only else scored_table
    subset = subset.reset_index(drop=True)

    if len(subset) == 0:
        print("No noise units to plot.")
        return []

    n_side = int(show_waveform) + int(show_isi)
    width_ratios = [4] + [1.5] * n_side
    fig_width = 4 * (1 + n_side * 0.5)   # ~6–8 inches depending on panels

    figures = []
    for _, unit_row in subset.iterrows():
        fig = plt.figure(figsize=(fig_width, 3.8))
        gs = gridspec.GridSpec(
            1, 1 + n_side,
            figure=fig,
            width_ratios=width_ratios,
            wspace=0.38,
            left=0.09, right=0.97, top=0.82, bottom=0.18,
        )
        ax_raster = fig.add_subplot(gs[0])
        col = 1
        ax_wave = fig.add_subplot(gs[col]) if show_waveform else None
        if show_waveform:
            col += 1
        ax_isi = fig.add_subplot(gs[col]) if show_isi else None

        _plot_unit(ax_raster, ax_wave, ax_isi,
                   unit_row, trial_starts, pre_s, post_s,
                   flag_cols, waveform_col)

        unit_id = unit_row.get("unit_id", "?")
        n_flags = int(unit_row.get("n_flags", 0))
        active_flags = [c.replace("flag_", "") for c in flag_cols if unit_row.get(c, False)]
        fig.suptitle(
            f"Unit {unit_id}  ·  {n_flags} flag(s): {', '.join(active_flags) or 'none'}",
            fontsize=9, x=0.5, y=0.97,
            color="crimson" if n_flags else "black",
        )

        if save_path:
            fname = os.path.join(save_path, f"unit_{unit_id}.png")
            fig.savefig(fname, bbox_inches="tight", dpi=150)

        figures.append(fig)

    return figures


# ─────────────────────────────────────────────────────────────────────────────
# Top-level convenience function
# ─────────────────────────────────────────────────────────────────────────────

def identify_noise_units(
    unit_table: pd.DataFrame,
    trial_table: pd.DataFrame,
    *,
    metric_registry: dict | None = None,
    min_flags_to_reject: int = 4,
    pre_s: float = 0.5,
    post_s: float = 1.5,
    plot: bool = True,
    show_waveform: bool = True,
    show_isi: bool = True,
    output_path: str | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[plt.Figure]]:
    """
    Identify noise / non-neuron units using modular quality metrics and
    optionally display one review figure per flagged unit.

    Parameters
    ----------
    unit_table : pd.DataFrame
        Required columns: "unit_id", "spike_times" (array of float, seconds).
        Optional column : "waveform_mean" (1-D array, µV).
    trial_table : pd.DataFrame
        Required column : "start_time" (float, seconds).
    metric_registry : dict, optional
        Pass a custom registry to add / remove / tweak metrics.
        See DEFAULT_METRIC_REGISTRY for the expected structure.
    min_flags_to_reject : int
        Number of metric flags required to label a unit as noise.
    pre_s / post_s : float
        Window around trial onset for the raster plots.
    plot : bool
        Whether to generate review figures.
    show_waveform / show_isi : bool
        Toggle the waveform / ISI panel in each figure.
    output_path : str, optional
        Directory for saving figures.  One PDF per unit is written as
        ``<output_path>/noise_units/unit_<unit_id>.pdf``.
    verbose : bool
        Print a short summary.

    Returns
    -------
    scored_table : pd.DataFrame
        Original unit_table enriched with per-metric scores, flags,
        "n_flags", and "is_noise".
    figures : list of plt.Figure
        One figure per flagged unit (empty list if plot=False).
    """
    save_dir = None
    if output_path is not None:
        save_dir = os.path.join(output_path, "noise_units")
        os.makedirs(save_dir, exist_ok=True)

    # Raise the ISI violation threshold, disable burst_index
    my_registry = {**DEFAULT_METRIC_REGISTRY}
    for metric in my_registry.keys():
        my_registry[metric]['active']=False
    my_registry["firing_rate"]["active"] = True
    my_registry["isi_cv"]["active"] = True
    my_registry["fano_factor"]["active"] = True
    my_registry["waveform_trough_width"]["active"] = True


    scored = score_units(unit_table, metric_registry, min_flags_to_reject)

    if verbose:
        n_noise = scored["is_noise"].sum()
        n_total = len(scored)
        flag_cols = [c for c in scored.columns if c.startswith("flag_")]
        print(f"\n{'─'*60}")
        print(f"  Noise unit detection  |  {n_noise}/{n_total} units flagged")
        print(f"{'─'*60}")
        for fc in flag_cols:
            name = fc.replace("flag_", "")
            n = scored[fc].sum()
            print(f"  {name:<28}  {n:>4} units flagged")
        print(f"{'─'*60}\n")

    figures = []
    if plot:
        print(f' Plotting flagged putative noise units ({n_noise})... at {save_dir}')
        figures = plot_noise_units(
            scored, trial_table,
            pre_s=pre_s,
            post_s=post_s,
            show_waveform=show_waveform,
            show_isi=show_isi,
            save_path=save_dir,
        )

    return scored, figures


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo / smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 3600.0  # seconds

    def _make_unit(uid, kind):
        if kind == "regular":
            return {"unit_id": uid,
                    "spike_times": np.sort(rng.uniform(0, T, int(rng.uniform(5, 20) * T))),
                    "waveform_mean": np.concatenate([
                        rng.normal(0, 2, 10),
                        np.array([-80, -100, -60, 0, 40, 30, 15, 5, 2, 0]),
                        rng.normal(0, 2, 10)])}
        elif kind == "noise_burst":
            # many spikes in tight bursts → high Fano + ISI violations
            burst_centres = rng.uniform(0, T, 1000)
            spikes = np.concatenate([c + rng.uniform(0, 5e-4, 30) for c in burst_centres])
            return {"unit_id": uid, "spike_times": spikes,
                    "waveform_mean": rng.normal(0, 3, 30)}
        elif kind == "silent":
            return {"unit_id": uid,
                    "spike_times": rng.uniform(0, T, 2),
                    "waveform_mean": rng.normal(0, 2, 30)}

    units = [_make_unit(i, k) for i, k in enumerate(
        ["regular"] * 5 + ["noise_burst"] * 3 + ["silent"] * 2)]
    unit_table = pd.DataFrame(units)
    trial_table = pd.DataFrame({"start_time": np.arange(0, T, 30.0)})

    scored, figs = identify_noise_units(unit_table, trial_table, min_flags_to_reject=2)
    print(scored[["unit_id", "n_flags", "is_noise"]])
    plt.show()
