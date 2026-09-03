#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_spikes_analysis
@file: task_modulation_utils.py
@time: 17/11/2025 1:43 PM
"""
# Imports
import os
import numpy as np
import pandas as pd
from functools import partial
from scipy.stats import ranksums
from statsmodels.stats.multitest import fdrcorrection
import multiprocessing
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # headless-safe (HPC/no display)
import matplotlib.pyplot as plt
import NWB_reader_functions as nwb_reader

# Dead-zone around whisker_trial start_time to exclude from spike counting (e.g. stim artifact)
DEAD_ZONE_PRE = 0.010   # 10 ms before start_time
DEAD_ZONE_POST = 0.005  # 5 ms after start_time


def compute_firing_rate(spike_times, window_start, window_end):
    """Compute firing rate in Hz."""
    return np.sum((spike_times >= window_start) & (spike_times < window_end)) / (window_end - window_start)
# --- helper: vectorized spike counting using searchsorted ---
def count_spikes(spikes, starts, ends):
    """
    Count spikes in each interval [starts[i], ends[i]) for a sorted 1D spike array.
    spikes : 1D numpy array (must be sorted)
    starts : scalar or 1D array
    ends   : scalar or 1D array
    returns: numpy array of counts with same shape as starts/ends broadcast
    """
    # np.searchsorted handles arrays for starts/ends and returns indices
    left_idx = np.searchsorted(spikes, starts, side='left')
    right_idx = np.searchsorted(spikes, ends, side='left')
    return right_idx - left_idx


def count_spikes_excluding_deadzone(spikes, starts, ends, dead_starts, dead_ends):
    """
    Same as count_spikes, but subtracts spikes falling in the intersection of
    each [starts[i], ends[i]) window with a per-trial dead zone [dead_starts[i], dead_ends[i]).
    Also returns the effective (dead-zone-corrected) duration of each window.
    Dead zones with dead_ends[i] <= dead_starts[i] (e.g. non-whisker trials) contribute no exclusion.
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    dead_starts = np.asarray(dead_starts, dtype=np.float64)
    dead_ends = np.asarray(dead_ends, dtype=np.float64)

    counts = count_spikes(spikes, starts, ends)

    overlap_starts = np.maximum(starts, dead_starts)
    overlap_ends = np.minimum(ends, dead_ends)
    overlap_len = np.clip(overlap_ends - overlap_starts, a_min=0.0, a_max=None)

    valid = overlap_len > 0
    if np.any(valid):
        overlap_counts = np.zeros_like(counts)
        overlap_counts[valid] = count_spikes(spikes, overlap_starts[valid], overlap_ends[valid])
        counts = counts - overlap_counts

    durations = (ends - starts) - overlap_len
    return counts, durations


def unit_task_mod(neuron_id, units_dict, trial_starts, trial_is_whisker, epoch_windows, baseline_length=0.05):
    """
    units_dict: {neuron_id: 1D numpy array of spike times (must be sorted or will be sorted here)}
    trial_starts: 1D numpy array of trial start times (float seconds)
    trial_is_whisker: 1D boolean array (same length as trial_starts), True for whisker_trial trials,
        used to exclude the [-10ms, +5ms) dead zone around start_time from spike counting.
    epoch_windows: list of (offset_start, offset_end) tuples, in seconds relative to start_time,
        each defining one epoch to test against the baseline. E.g. task modulation uses ten
        50ms epochs [(0.0,0.05), (0.05,0.10), ...]; sensory modulation uses a single window
        [(0.005, 0.050)] (5-50ms post start_time, deliberately starting after the dead zone).
    baseline_length: baseline window is [start_time - baseline_length, start_time).
    Each epoch is tested against baseline with a two-sided Wilcoxon rank-sum test, i.e. testing
    for ANY difference in firing rate (increase or decrease), not just an increase.
    """
    # Get spikes and ensure sorted (cheap if already sorted) necessary for searchsorted in count_spikes()
    spikes = units_dict[neuron_id]
    if spikes.size and not np.all(spikes[:-1] <= spikes[1:]):
        spikes = np.sort(spikes)

    # Dead zone [-10ms, +5ms) around start_time, only active for whisker_trial trials
    # (zero-width, i.e. no-op, for all other trials)
    dead_starts = np.where(trial_is_whisker, trial_starts - DEAD_ZONE_PRE, trial_starts)
    dead_ends = np.where(trial_is_whisker, trial_starts + DEAD_ZONE_POST, trial_starts)

    # Baseline windows (vector) - always the same
    baseline_starts = trial_starts - baseline_length
    baseline_ends = trial_starts
    # Count baseline spikes vectorized, excluding the dead zone
    baseline_counts, baseline_durations = count_spikes_excluding_deadzone(
        spikes, baseline_starts, baseline_ends, dead_starts, dead_ends
    )
    baseline_rates = baseline_counts / baseline_durations

    raw_pvals = []
    for offset_start, offset_end in epoch_windows:
        e_starts = trial_starts + offset_start
        e_ends = trial_starts + offset_end
        epoch_counts, epoch_durations = count_spikes_excluding_deadzone(
            spikes, e_starts, e_ends, dead_starts, dead_ends
        )
        epoch_rates = epoch_counts / epoch_durations
        # Fast check: identical distributions -> p=1.0
        if (np.all(baseline_rates == baseline_rates[0]) and
                np.all(epoch_rates == epoch_rates[0])):
            raw_pvals.append(1.0)
        else:
            # two-sided: tests for a difference in firing rate (increase OR decrease), not just an increase
            _, p = ranksums(baseline_rates, epoch_rates, alternative='two-sided')
            raw_pvals.append(p)
    return {"neuron_id": neuron_id, "raw_pvals": raw_pvals}


def apply_fdr_correction(results, sig_level=0.05):
    """
    Per-epoch FDR correction across units: for each epoch (column of the unit x epoch p-value
    matrix), the raw p-values across ALL units are corrected together, independently per epoch
    (not pooled across epochs). A unit is "modulated" if any of its epochs is significant after
    this per-epoch, across-unit correction.
    With a single epoch (e.g. the sensory-modulation test), this reduces to exactly one FDR
    correction across all units.
    Mutates and returns `results` (adds "corrected_pvals" and "modulated" per entry).
    """
    raw_pvals_matrix = np.array([r["raw_pvals"] for r in results])  # (n_units, n_epochs)
    corrected_matrix = np.empty_like(raw_pvals_matrix)
    n_epochs = raw_pvals_matrix.shape[1]
    for e in range(n_epochs):
        _, corrected = fdrcorrection(raw_pvals_matrix[:, e], alpha=sig_level)
        corrected_matrix[:, e] = corrected
    for i, r in enumerate(results):
        r["corrected_pvals"] = corrected_matrix[i, :]
        r["modulated"] = bool(np.any(corrected_matrix[i, :] < sig_level))
    return results


def get_num_workers(reserve=4):
    """
    Number of worker processes to use. Prefers os.sched_getaffinity(0), which reflects the CPUs
    actually allocated to THIS process (e.g. by SLURM's --cpus-per-task / cgroup limits) - unlike
    os.cpu_count(), which reports the whole node's core count regardless of what was actually
    allocated to the job. On a shared HPC node this matters a lot: cpu_count() can both
    over-subscribe (if the node has more cores than were allocated to you) and under-utilize
    (if you were allocated more than a hardcoded assumption expects).
    sched_getaffinity is POSIX-only; falls back to cpu_count() on platforms without it (e.g. Windows).
    """
    try:
        n_available = len(os.sched_getaffinity(0))
    except AttributeError:
        n_available = os.cpu_count() or 1
    return max(1, n_available - reserve)


def _get_mp_context():
    """
    Explicitly use the 'fork' start method rather than relying on the platform default.
    With fork, worker processes share the parent's memory via copy-on-write, so a large
    units_dict (thousands of neurons' spike arrays) is never actually serialized - workers just
    see it directly. With 'spawn' (the default on Windows/macOS, and occasionally forced by some
    containerized/HPC environments), every dispatched task would instead re-pickle and transmit
    whatever is bound in the partial(...) call - including the ENTIRE units_dict - which at
    thousands of neurons could mean repeatedly serializing a multi-GB object per task. Falls back
    to the default context if 'fork' isn't available (e.g. native Windows).
    """
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return multiprocessing.get_context()


def run_modulation_test(neuron_ids, units_dict, trial_starts, trial_is_whisker, epoch_windows,
                         num_workers, chunksize, sig_level=0.05, baseline_length=0.05, desc="Analyzing neurons"):
    """
    Runs unit_task_mod across all neurons (multiprocessing) for the given epoch_windows, then
    applies per-epoch, across-units FDR correction. Returns (results, n_raw_modulated) where
    n_raw_modulated is the count of neurons significant in >=1 epoch BEFORE correction.
    Standalone single-test runner - task_modulation_analysis uses run_modulation_tests_combined
    instead, to share one pool across all 3 tests (task, sensory-whisker, sensory-auditory).
    """
    with _get_mp_context().Pool(num_workers) as pool:
        func = partial(unit_task_mod,
                       units_dict=units_dict,
                       trial_starts=trial_starts,
                       trial_is_whisker=trial_is_whisker,
                       epoch_windows=epoch_windows,
                       baseline_length=baseline_length)
        results = []
        for r in tqdm(pool.imap(func, neuron_ids, chunksize=chunksize),
                      total=len(neuron_ids),
                      desc=desc):
            results.append(r)
    n_raw_modulated = sum(np.any(np.array(r["raw_pvals"]) < sig_level) for r in results)
    results = apply_fdr_correction(results, sig_level=sig_level)
    return results, n_raw_modulated


def _unit_modulation_dispatch(work_item, units_dict, test_configs):
    """Worker-side entry point for run_modulation_tests_combined: runs one (test, neuron) unit."""
    test_id, neuron_id = work_item
    cfg = test_configs[test_id]
    result = unit_task_mod(
        neuron_id, units_dict, cfg["trial_starts"], cfg["trial_is_whisker"],
        cfg["epoch_windows"], baseline_length=cfg["baseline_length"]
    )
    return test_id, result


def run_modulation_tests_combined(neuron_ids, units_dict, test_configs, num_workers, sig_level=0.05,
                                   desc="Analyzing neurons (all tests)"):
    """
    Runs MULTIPLE modulation tests (e.g. task, sensory-whisker, sensory-auditory) through a
    SINGLE shared multiprocessing.Pool, instead of one pool per test. This both avoids paying
    pool spawn/teardown overhead once per test, and - more importantly with many cores - gives
    the scheduler len(test_configs) x n_neurons work items to spread across workers instead of
    just n_neurons per stage, which matters a lot when n_neurons is comparable to or smaller
    than the worker count (each stage alone would otherwise leave many workers idle).

    test_configs: dict of {test_id: {"trial_starts", "trial_is_whisker", "epoch_windows",
        "baseline_length"}}. Returns dict of {test_id: (results, n_raw_modulated)}, each already
        through apply_fdr_correction, exactly as run_modulation_test would return per test.
    """
    work_items = [(test_id, nid) for test_id in test_configs for nid in neuron_ids]
    chunksize = max(1, len(work_items) // (num_workers * 4))

    results_by_test = {test_id: [] for test_id in test_configs}
    with _get_mp_context().Pool(num_workers) as pool:
        func = partial(_unit_modulation_dispatch, units_dict=units_dict, test_configs=test_configs)
        for test_id, r in tqdm(pool.imap_unordered(func, work_items, chunksize=chunksize),
                                total=len(work_items), desc=desc):
            results_by_test[test_id].append(r)

    out = {}
    for test_id, results in results_by_test.items():
        n_raw = sum(np.any(np.array(r["raw_pvals"]) < sig_level) for r in results)
        results = apply_fdr_correction(results, sig_level=sig_level)
        out[test_id] = (results, n_raw)
    return out
# ----------------------------------------------------------
# DEBUG PSTH PLOTTING
# ----------------------------------------------------------
def _generate_deadzone_replacement_spikes(spikes, trial_starts, trial_is_whisker,
                                           epoch_length=0.05, rng=None):
    """
    COSMETIC / PLOTTING ONLY - does not affect statistics.
    Returns a copy of `spikes` where, for each whisker_trial, any real spikes falling in the
    [-10ms, +5ms) dead zone around that trial's start_time are removed and replaced by a
    homogeneous Poisson process realization (actual continuous spike times, not binned).
    The rate lambda for each trial's replacement is that trial's own baseline firing rate,
    estimated from [start-epoch_length, start) with the dead zone excluded (same baseline
    used for the analysis), so the cosmetic dead-zone fill matches that trial's local background rate.
    Non-whisker trials are untouched (no dead zone defined for them).
    """
    if rng is None:
        rng = np.random.default_rng()

    spikes = np.asarray(spikes, dtype=np.float64)
    if spikes.size and not np.all(spikes[:-1] <= spikes[1:]):
        spikes = np.sort(spikes)

    whisker_idx = np.where(trial_is_whisker)[0]
    if whisker_idx.size == 0:
        return spikes.copy()

    dead_starts = trial_starts[whisker_idx] - DEAD_ZONE_PRE
    dead_ends = trial_starts[whisker_idx] + DEAD_ZONE_POST
    baseline_starts = trial_starts[whisker_idx] - epoch_length
    baseline_ends = trial_starts[whisker_idx]

    # Per-trial baseline rate, dead zone excluded (matches the analysis baseline exactly)
    baseline_counts, baseline_durations = count_spikes_excluding_deadzone(
        spikes, baseline_starts, baseline_ends, dead_starts, dead_ends
    )
    lambdas = baseline_counts / baseline_durations

    keep_mask = np.ones(spikes.shape, dtype=bool)
    synthetic_chunks = []
    for dstart, dend, lam in zip(dead_starts, dead_ends, lambdas):
        keep_mask &= ~((spikes >= dstart) & (spikes < dend))
        duration = dend - dstart
        if lam > 0 and duration > 0:
            n_synth = rng.poisson(lam * duration)
            if n_synth > 0:
                synthetic_chunks.append(rng.uniform(dstart, dend, size=n_synth))

    new_spikes = spikes[keep_mask]
    if synthetic_chunks:
        new_spikes = np.concatenate([new_spikes] + synthetic_chunks)
    return np.sort(new_spikes)


def _compute_psth(units_dict, neuron_ids, trial_starts, bin_edges):
    """
    Mean firing rate per neuron, per 5ms bin, averaged across trials (not across neurons).
    Returns array of shape (n_neurons, n_bins), units Hz.
    """
    n_bins = len(bin_edges) - 1
    bin_widths = np.diff(bin_edges)
    psth = np.full((len(neuron_ids), n_bins), np.nan)
    if len(trial_starts) == 0:
        return psth
    for i, nid in enumerate(neuron_ids):
        spikes = units_dict[nid]
        if spikes.size and not np.all(spikes[:-1] <= spikes[1:]):
            spikes = np.sort(spikes)
        counts = np.empty((len(trial_starts), n_bins))
        for b in range(n_bins):
            starts = trial_starts + bin_edges[b]
            ends = trial_starts + bin_edges[b + 1]
            counts[:, b] = count_spikes(spikes, starts, ends)
        rates = counts / bin_widths[None, :]
        psth[i, :] = rates.mean(axis=0)  # average across trials
    return psth


def _plot_mean_sem(ax, bin_centers_ms, psth, group_mask, label, color):
    """Plot mean +/- SEM across neurons in group_mask, on ax. No-op if group is empty."""
    n = int(group_mask.sum())
    if n == 0:
        return
    mean = np.nanmean(psth[group_mask], axis=0)
    sem = np.nanstd(psth[group_mask], axis=0) / np.sqrt(n)
    ax.plot(bin_centers_ms, mean, color=color, label=f"{label} (n={n})")
    ax.fill_between(bin_centers_ms, mean - sem, mean + sem, color=color, alpha=0.2, linewidth=0)


def plot_debug_psth(units_dict, whisker_out_df, auditory_out_df, whisker_starts, auditory_starts,
                     trial_starts, trial_is_whisker,
                     mouse_id, session_id, results_path,
                     epoch_windows, sig_col, modulation_label,
                     baseline_length=0.05, sig_level=0.05,
                     plot_window=(-0.05, 0.5), bin_size=0.005, rng=None):
    """
    Save debug figure(s) per session, comparing significant vs non-significant neurons'
    average activity around start_time (5ms bins), for whisker and auditory trials.
    Whisker and auditory are each their OWN test (separate out_df, separate significance calls) -
    the auditory panel/lines use auditory_out_df's own significance, the whisker panel/lines use
    whisker_out_df's own significance; they are not sharing one pooled classification.
      - "overall" figure: split by <sig_col> (significant in >=1 tested epoch), one panel per modality.
      - "per_epoch" figure (one per trial type): grid with one subplot per entry in
        epoch_windows, split by significance in that specific epoch. Only generated when
        len(epoch_windows) > 1 (with a single epoch it would just duplicate the overall figure).
    whisker_out_df / auditory_out_df must each contain columns: neuron_id, <sig_col>,
    corrected_pvals (array of length len(epoch_windows)).

    epoch_windows: list of (offset_start, offset_end) tuples in seconds relative to start_time -
        the same windows that were tested (see unit_task_mod). Used here only for shading /
        subplot boundaries, not for the plotted time range (see plot_window).
    sig_col: name of the boolean "modulated" column in each out_df (e.g. "task_modulated" or
        "sensory_modulated").
    modulation_label: short string used in titles/filenames, e.g. "task" or "sensory".
    plot_window: (start, end) in seconds relative to start_time, shown in every panel - fixed
        regardless of epoch_windows so task and sensory figures are visually comparable.

    trial_starts / trial_is_whisker: the FULL combined set of active trials (both modalities),
    used only to locate each whisker_trial's dead zone for the cosmetic Poisson replacement below
    - this is independent of which modality is being tested/plotted, and does NOT affect statistics.

    COSMETIC NOTE: for plotting only, real spikes in the [-10ms, +5ms) dead zone around each
    whisker_trial's start_time are replaced with a per-neuron, per-trial Poisson-process
    realization (rate = that trial's dead-zone-excluded baseline rate), so the dead zone doesn't
    show up as a stim-artifact spike burst or an artificial gap in the PSTH.
    """
    if rng is None:
        rng = np.random.default_rng()

    whisker_neuron_ids = whisker_out_df["neuron_id"].values
    whisker_sig_any = whisker_out_df[sig_col].values.astype(bool)
    whisker_corrected = np.vstack(whisker_out_df["corrected_pvals"].values)  # (n_neurons, n_epoch_windows)
    whisker_sig_per_epoch = whisker_corrected < sig_level

    auditory_neuron_ids = auditory_out_df["neuron_id"].values
    auditory_sig_any = auditory_out_df[sig_col].values.astype(bool)
    auditory_corrected = np.vstack(auditory_out_df["corrected_pvals"].values)
    auditory_sig_per_epoch = auditory_corrected < sig_level

    bin_edges = np.arange(plot_window[0], plot_window[1] + bin_size / 2, bin_size)
    bin_centers_ms = (bin_edges[:-1] + bin_edges[1:]) / 2 * 1000

    # Cosmetic-only spike trains with dead-zone spikes replaced (plotting exclusively; stats untouched)
    all_neuron_ids = np.union1d(whisker_neuron_ids, auditory_neuron_ids)
    plot_units_dict = {
        nid: _generate_deadzone_replacement_spikes(
            units_dict[nid], trial_starts, trial_is_whisker, epoch_length=baseline_length, rng=rng
        )
        for nid in all_neuron_ids
    }

    psth_whisker = _compute_psth(plot_units_dict, whisker_neuron_ids, whisker_starts, bin_edges)
    psth_auditory = _compute_psth(plot_units_dict, auditory_neuron_ids, auditory_starts, bin_edges)

    os.makedirs(results_path, exist_ok=True)

    # --- Figure 1: significant in any tested epoch, each panel using its own modality's test ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    panels = [
        (axes[0], psth_auditory, auditory_sig_any, "Auditory trials"),
        (axes[1], psth_whisker, whisker_sig_any, "Whisker trials"),
    ]
    for ax, psth, sig_any, trial_label in panels:
        _plot_mean_sem(ax, bin_centers_ms, psth, sig_any, "Significant", "crimson")
        _plot_mean_sem(ax, bin_centers_ms, psth, ~sig_any, "Non-significant", "gray")
        for offset_start, offset_end in epoch_windows:
            ax.axvspan(offset_start * 1000, offset_end * 1000, color="gold", alpha=0.12)
        ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_title(trial_label)
        ax.set_xlabel("Time from start_time (ms)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Firing rate (Hz)")
    fig.suptitle(f"{mouse_id} {session_id} - {modulation_label}-modulated neurons")
    fig.tight_layout()
    fig.savefig(f"{results_path}/{mouse_id}_{session_id}_debug_psth_{modulation_label}_overall.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: significant per epoch (only meaningful with >1 epoch), each using its own modality's test ---
    if len(epoch_windows) > 1:
        n_ep = len(epoch_windows)
        ncols = min(5, n_ep)
        nrows = int(np.ceil(n_ep / ncols))
        for psth, sig_per_epoch, trial_label, fname_suffix in [
            (psth_auditory, auditory_sig_per_epoch, "Auditory", "auditory"),
            (psth_whisker, whisker_sig_per_epoch, "Whisker", "whisker"),
        ]:
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.75 * nrows), sharex=True, sharey=True)
            axes = np.atleast_1d(axes).ravel()
            for e, (offset_start, offset_end) in enumerate(epoch_windows):
                ax = axes[e]
                grp_sig = sig_per_epoch[:, e]
                _plot_mean_sem(ax, bin_centers_ms, psth, grp_sig, "Sig. this epoch", "crimson")
                _plot_mean_sem(ax, bin_centers_ms, psth, ~grp_sig, "Non-sig.", "gray")
                epoch_start_ms = offset_start * 1000
                epoch_end_ms = offset_end * 1000
                ax.axvspan(epoch_start_ms, epoch_end_ms, color="gold", alpha=0.15)
                ax.axvline(0, color="k", linestyle="--", linewidth=0.6)
                ax.set_title(f"Epoch {e} [{epoch_start_ms:.0f}-{epoch_end_ms:.0f} ms]", fontsize=9)
                if e == 0:
                    ax.legend(fontsize=6)
            for ax in axes[n_ep:]:
                ax.axis("off")
            fig.supxlabel("Time from start_time (ms)")
            fig.supylabel("Firing rate (Hz)")
            fig.suptitle(f"{mouse_id} {session_id} - {modulation_label}, {trial_label} trials, per-epoch significance")
            fig.tight_layout()
            fig.savefig(
                f"{results_path}/{mouse_id}_{session_id}_debug_psth_{modulation_label}_per_epoch_{fname_suffix}.png",
                dpi=150
            )
            plt.close(fig)

    print(f"Saved {modulation_label}-modulation debug PSTH figures to: {results_path}")


# ----------------------------------------------------------
# MAIN ANALYSIS FUNCTION
# ----------------------------------------------------------
def _build_out_df(results, mouse_id, session_id, unit_meta, modulated_col_name):
    df = pd.DataFrame(results)
    df['mouse_id'] = mouse_id
    df['session_id'] = session_id
    df = df.join(unit_meta, on="neuron_id")
    df["raw_pvals"] = df["raw_pvals"].apply(lambda x: np.array(x))
    df["corrected_pvals"] = df["corrected_pvals"].apply(lambda x: np.array(x))
    df = df.rename(columns={"modulated": modulated_col_name})
    return df


def _run_and_build(neuron_ids, units_dict, starts, is_whisker, epoch_windows,
                    num_workers, chunksize, sig_level, baseline_length, desc,
                    mouse_id, session_id, unit_meta, modulated_col_name):
    """Run unit_task_mod + FDR correction for one (modality, epoch_windows) combo, build its out_df."""
    results, n_raw = run_modulation_test(
        neuron_ids, units_dict, starts, is_whisker, epoch_windows,
        num_workers, chunksize, sig_level=sig_level, baseline_length=baseline_length, desc=desc
    )
    out_df = _build_out_df(results, mouse_id, session_id, unit_meta, modulated_col_name)
    return out_df, n_raw


def _print_fraction(mouse_id, label, n_num, n_total, stage):
    frac = 100 * n_num / n_total if n_total else 0.0
    print(f"Mouse {mouse_id}: {n_num}/{n_total} neurons are {label} {stage} ({frac:.2f}%)")


def task_modulation_analysis(nwb_file, results_path, make_debug_plots=False):
    print("Starting statistical tests task-modulation for:", nwb_file)
    SIG_LEVEL = 0.05
    BASELINE_LENGTH = 0.05  # 50 ms baseline, [start_time - BASELINE_LENGTH, start_time)
    TASK_EPOCH_LENGTH = 0.05  # 50 ms
    N_TASK_EPOCHS = 10
    TASK_EPOCH_WINDOWS = [(e * TASK_EPOCH_LENGTH, (e + 1) * TASK_EPOCH_LENGTH) for e in range(N_TASK_EPOCHS)]
    # Sensory modulation: single window, 5-50ms post start_time (starts after the dead zone's post-start tail)
    SENSORY_EPOCH_WINDOWS = [(0.005, 0.050)]

    # Load neural and trial data
    mouse_id = nwb_reader.get_mouse_id(nwb_file)
    session_id = nwb_reader.get_session_id(nwb_file)
    unit_table = nwb_reader.get_unit_table(nwb_file)
    unit_table = unit_table[unit_table.columns.intersection(
        ["neuron_id", "spike_times", "electrode_group", "cluster_id"]
    )]
    neuron_ids = unit_table["neuron_id"].unique()
    # Pre-convert data for fast pickling
    units_dict = {
        nid: spikes
        for nid, spikes in zip(unit_table["neuron_id"], unit_table["spike_times"])
    }
    # Per-neuron metadata to carry through for later merging (one row per neuron_id)
    meta_cols = unit_table.columns.intersection(["neuron_id", "electrode_group", "cluster_id"])
    unit_meta = unit_table[meta_cols].drop_duplicates(subset="neuron_id").set_index("neuron_id")

    raw_trial_table = nwb_reader.get_trial_table(nwb_file)

    # --- TASK MODULATION trial set: whisker + auditory combined, active context only ---
    task_trial_mask = (
        raw_trial_table.trial_type.isin(['whisker_trial', 'auditory_trial']) &
        (raw_trial_table.context != 'passive')
    )
    task_trial_table = raw_trial_table[task_trial_mask]
    task_trial_starts = task_trial_table["start_time"].values.astype(np.float64)  # combined, both modalities
    task_trial_is_whisker = (task_trial_table["trial_type"] == 'whisker_trial').values
    task_whisker_starts = task_trial_table.loc[task_trial_table.trial_type == 'whisker_trial', 'start_time'].values.astype(np.float64)
    task_auditory_starts = task_trial_table.loc[task_trial_table.trial_type == 'auditory_trial', 'start_time'].values.astype(np.float64)

    # --- SENSORY MODULATION trial set: per-modality, all whisker/auditory trials (any context, any lick_flag) ---
    sensory_trial_mask = raw_trial_table.trial_type.isin(['whisker_trial', 'auditory_trial'])
    sensory_trial_table = raw_trial_table[sensory_trial_mask]
    sensory_trial_starts = sensory_trial_table["start_time"].values.astype(np.float64)  # combined, both modalities
    sensory_trial_is_whisker = (sensory_trial_table["trial_type"] == 'whisker_trial').values
    sensory_whisker_starts = sensory_trial_table.loc[sensory_trial_table.trial_type == 'whisker_trial', 'start_time'].values.astype(np.float64)
    sensory_auditory_starts = sensory_trial_table.loc[sensory_trial_table.trial_type == 'auditory_trial', 'start_time'].values.astype(np.float64)
    sensory_whisker_is_whisker = np.ones(sensory_whisker_starts.shape, dtype=bool)
    sensory_auditory_is_whisker = np.zeros(sensory_auditory_starts.shape, dtype=bool)

    num_workers = get_num_workers()
    n_total = len(neuron_ids)

    # ------------------------------------------------------
    # Run all three tests (task; sensory-whisker; sensory-auditory) through ONE shared pool
    # ------------------------------------------------------
    test_configs = {
        "task": dict(trial_starts=task_trial_starts, trial_is_whisker=task_trial_is_whisker,
                     epoch_windows=TASK_EPOCH_WINDOWS, baseline_length=BASELINE_LENGTH),
        "sensory_whisker": dict(trial_starts=sensory_whisker_starts, trial_is_whisker=sensory_whisker_is_whisker,
                                epoch_windows=SENSORY_EPOCH_WINDOWS, baseline_length=BASELINE_LENGTH),
        "sensory_auditory": dict(trial_starts=sensory_auditory_starts, trial_is_whisker=sensory_auditory_is_whisker,
                                 epoch_windows=SENSORY_EPOCH_WINDOWS, baseline_length=BASELINE_LENGTH),
    }
    test_results = run_modulation_tests_combined(
        neuron_ids, units_dict, test_configs, num_workers, sig_level=SIG_LEVEL,
        desc="Analyzing neurons (task + sensory, whisker + auditory)"
    )

    # ------------------------------------------------------
    # TASK MODULATION: 10 x 50ms epochs, 0-500ms post start_time - whisker+auditory COMBINED, one table
    # ------------------------------------------------------
    task_results, n_task_raw = test_results["task"]
    task_out_df = _build_out_df(task_results, mouse_id, session_id, unit_meta, "task_modulated")
    _print_fraction(mouse_id, "task-modulated", n_task_raw, n_total, "before correction")
    _print_fraction(mouse_id, "task-modulated", task_out_df['task_modulated'].sum(), n_total, "after correction")

    # ------------------------------------------------------
    # SENSORY MODULATION: single 5-50ms epoch post start_time - per modality, combined into one table
    # ------------------------------------------------------
    sensory_whisker_results, n_sensory_whisker_raw = test_results["sensory_whisker"]
    sensory_auditory_results, n_sensory_auditory_raw = test_results["sensory_auditory"]
    sensory_whisker_out_df = _build_out_df(sensory_whisker_results, mouse_id, session_id, unit_meta, "sensory_modulated")
    sensory_auditory_out_df = _build_out_df(sensory_auditory_results, mouse_id, session_id, unit_meta, "sensory_modulated")
    _print_fraction(mouse_id, "sensory-modulated (whisker)", n_sensory_whisker_raw, n_total, "before correction")
    _print_fraction(mouse_id, "sensory-modulated (whisker)", sensory_whisker_out_df['sensory_modulated'].sum(), n_total, "after correction")
    _print_fraction(mouse_id, "sensory-modulated (auditory)", n_sensory_auditory_raw, n_total, "before correction")
    _print_fraction(mouse_id, "sensory-modulated (auditory)", sensory_auditory_out_df['sensory_modulated'].sum(), n_total, "after correction")

    sensory_out_df = pd.concat([
        sensory_whisker_out_df.assign(modality="whisker"),
        sensory_auditory_out_df.assign(modality="auditory"),
    ], ignore_index=True)

    # ------------------------------------------------------
    # Debug PSTH plots
    # ------------------------------------------------------
    if make_debug_plots:
        # Task: combined test -> same out_df/significance shown on both the whisker and auditory panel
        plot_debug_psth(
            units_dict=units_dict,
            whisker_out_df=task_out_df,
            auditory_out_df=task_out_df,
            whisker_starts=task_whisker_starts,
            auditory_starts=task_auditory_starts,
            trial_starts=task_trial_starts,
            trial_is_whisker=task_trial_is_whisker,
            mouse_id=mouse_id,
            session_id=session_id,
            results_path=results_path,
            epoch_windows=TASK_EPOCH_WINDOWS,
            sig_col="task_modulated",
            modulation_label="task",
            baseline_length=BASELINE_LENGTH,
            sig_level=SIG_LEVEL,
        )
        # Sensory: per-modality test -> each panel uses its own modality's significance (unchanged)
        plot_debug_psth(
            units_dict=units_dict,
            whisker_out_df=sensory_whisker_out_df,
            auditory_out_df=sensory_auditory_out_df,
            whisker_starts=sensory_whisker_starts,
            auditory_starts=sensory_auditory_starts,
            trial_starts=sensory_trial_starts,
            trial_is_whisker=sensory_trial_is_whisker,
            mouse_id=mouse_id,
            session_id=session_id,
            results_path=results_path,
            epoch_windows=SENSORY_EPOCH_WINDOWS,
            sig_col="sensory_modulated",
            modulation_label="sensory",
            baseline_length=BASELINE_LENGTH,
            sig_level=SIG_LEVEL,
        )
    # ------------------------------------------------------
    # Save results
    # ------------------------------------------------------
    os.makedirs(results_path, exist_ok=True)
    task_file = f"{results_path}/{mouse_id}_task_modulation_results.csv"
    sensory_file = f"{results_path}/{mouse_id}_sensory_modulation_results.csv"
    task_out_df.to_csv(task_file, index=False)
    sensory_out_df.to_csv(sensory_file, index=False)
    print("Saved task-modulation results to:", task_file)
    print("Saved sensory-modulation results to:", sensory_file)
    return