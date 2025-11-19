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

import NWB_reader_functions as nwb_reader



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

def unit_task_mod(neuron_id, units_dict, trial_starts):
    """
    units_dict: {neuron_id: 1D numpy array of spike times (must be sorted or will be sorted here)}
    trial_starts: 1D numpy array of trial start times (float seconds)
    """
    # Get spikes and ensure sorted (cheap if already sorted) necessary for searchsorted in count_spikes()
    spikes = units_dict[neuron_id]
    if spikes.size and not np.all(spikes[:-1] <= spikes[1:]):
        spikes = np.sort(spikes)

    n_epochs = 10
    epoch_length = 0.05  # 50 ms

    # Baseline windows (vector) - always the same
    baseline_starts = trial_starts - epoch_length
    baseline_ends = trial_starts

    # Count baseline spikes vectorized
    baseline_counts = count_spikes(spikes, baseline_starts, baseline_ends)
    baseline_rates = baseline_counts / epoch_length

    raw_pvals = []

    # Loop epochs (only 10 iterations; counts computed vectorized per epoch)
    for e in range(n_epochs):
        e_starts = trial_starts + e * epoch_length
        e_ends = trial_starts + (e + 1) * epoch_length

        epoch_counts = count_spikes(spikes, e_starts, e_ends)
        epoch_rates = epoch_counts / epoch_length

        # Fast check: identical distributions -> p=1.0
        if (np.all(baseline_rates == baseline_rates[0]) and
                np.all(epoch_rates == epoch_rates[0])):
            raw_pvals.append(1.0)
        else:
            _, p = ranksums(baseline_rates, epoch_rates, alternative='two-sided')
            raw_pvals.append(p)

    return {"neuron_id": neuron_id, "raw_pvals": raw_pvals}


# ----------------------------------------------------------
# MAIN ANALYSIS FUNCTION
# ----------------------------------------------------------

def task_modulation_analysis(nwb_file, results_path):

    print("Starting statistical tests task-modulation for:", nwb_file)
    SIG_LEVEL = 0.05

    # Load neural and trial data
    mouse_id = nwb_reader.get_mouse_id(nwb_file)
    unit_table = nwb_reader.get_unit_table(nwb_file)
    unit_table = unit_table[unit_table.columns.intersection(["neuron_id", "spike_times"])]
    neuron_ids = unit_table["neuron_id"].unique()

    # Pre-convert data for fast pickling
    units_dict = {
        nid: spikes
        for nid, spikes in zip(unit_table["neuron_id"], unit_table["spike_times"])
    }

    # For task-modulation, keep active trials that are stim trials or lick trials
    trial_table = nwb_reader.get_trial_table(nwb_file)
    trial_mask = ((
            trial_table.trial_type.isin(['whisker_trial', 'auditory_trial']) | (trial_table.lick_flag == 1)) &
            (trial_table.context != 'passive') &
                  (trial_table.perf != 6)
    )
    trial_table = trial_table[trial_mask]
    trial_starts = trial_table["start_time"].values.astype(np.float32)

    # Multiprocessing
    num_workers = max(1, os.cpu_count() - 4)
    chunksize = max(1, len(neuron_ids) // (num_workers * 4)) # to reduce overhead in imap

    with multiprocessing.Pool(num_workers) as pool:
        func = partial(unit_task_mod,
                       units_dict=units_dict,
                       trial_starts=trial_starts)

        results = []
        for r in tqdm(pool.imap(func, neuron_ids, chunksize=chunksize),
                      total=len(neuron_ids),
                      desc="Analyzing neurons"):
            results.append(r)

    # Fraction of task-modulated neurons before correction
    n_task_modulated_raw = sum(np.any(np.array(r["raw_pvals"]) < SIG_LEVEL) for r in results)
    n_total = len(results)
    fraction_task_modulated_raw = n_task_modulated_raw / n_total if n_total > 0 else 0.0
    print(f"Mouse {mouse_id}: {n_task_modulated_raw}/{n_total} neurons are task-modulated before correction "
          f"({fraction_task_modulated_raw*100:.2f}%)")

    # ------------------------------------------------------
    # FDR CORRECTION (across all neurons * all 10 epochs)
    # ------------------------------------------------------
    all_pvals = np.concatenate([r["raw_pvals"] for r in results])
    reject, corrected = fdrcorrection(all_pvals, alpha=SIG_LEVEL) # false discovery rate of 1%

    # Assign corrected p-values back to neurons
    idx = 0
    for r in results:
        r["corrected_pvals"] = corrected[idx:idx+10]
        r["task_modulated"] = np.any(r["corrected_pvals"] < 0.05)
        idx += 10

    # Convert to dataframe
    out_df = pd.DataFrame(results)
    out_df['mouse_id'] = mouse_id
    out_df["raw_pvals"] = out_df["raw_pvals"].apply(lambda x: np.array(x))
    out_df["corrected_pvals"] = out_df["corrected_pvals"].apply(lambda x: np.array(x))

    # Fraction of task-modulated neurons
    n_task_modulated = out_df['task_modulated'].sum()
    n_total = len(out_df)
    fraction_task_modulated = n_task_modulated / n_total if n_total > 0 else 0.0
    print(f"Mouse {mouse_id}: {n_task_modulated}/{n_total} neurons are task-modulated after correction "
          f"({fraction_task_modulated*100:.2f}%)")

    # ------------------------------------------------------
    # Save results
    # ------------------------------------------------------
    os.makedirs(results_path, exist_ok=True)
    out_file = f"{results_path}/{mouse_id}_task_modulation_results.csv"
    out_df.to_csv(out_file, index=False)
    print("Saved task-modulation results to:", out_file)

    return
