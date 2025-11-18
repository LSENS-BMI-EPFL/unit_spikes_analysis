#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_spikes_analysis
@file: noise_correl_utils.py
@time: 10/11/2025 9:39 PM
"""

# Imports
import os
import numpy as np
import pandas as pd
from pathlib import Path
#import pyarrow as pa
#import pyarrow.parquet as pq
#import dask.dataframe as dd


import NWB_reader_functions as nwb_reader
import allen_utils

TRIAL_MAP = {
    0: 'whisker_miss',
    1: 'auditory_miss',
    2: 'whisker_hit',
    3: 'auditory_hit',
    4: 'correct_rejection',
    5: 'false_alarm',
    6: 'association',
}

# ---------------------
# 1. Compute residuals
# --------------------
def compute_residuals_matrix(spikes_df, trials_df, bin_size=0.005, window=(0.005, 0.1), n_neurons_test=None):

    neurons = spikes_df['neuron_id'].to_numpy() # session-wide neuronal index
    if n_neurons_test is not None:
        neurons = np.random.choice(neurons, size=min(n_neurons_test, len(neurons)), replace=False)
        spikes_df = spikes_df[spikes_df['neuron_id'].isin(neurons)]

    n_neurons = len(neurons)
    n_trials = len(trials_df)
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    n_bins = len(bins) - 1

    counts = np.zeros((n_neurons, n_trials, n_bins), dtype=np.float32)
    for ni, nid in enumerate(neurons):
        spikes = np.asarray(spikes_df.loc[spikes_df.neuron_id == nid, 'spike_times'].values[0])
        for ti, tstart in enumerate(trials_df['start_time']):
            rel_spikes = spikes - tstart
            counts[ni, ti] = np.histogram(rel_spikes, bins=bins)[0]

    # Create "outcome" column
    conds = trials_df['outcome'].to_numpy() #TODO: change to better include passive?
    unique_conds, cond_idx = np.unique(conds, return_inverse=True)
    psth = np.zeros((n_neurons, len(unique_conds), n_bins), dtype=np.float32)
    for ci, cond in enumerate(unique_conds):
        psth[:, ci] = counts[:, cond_idx == ci].mean(axis=1)

    residuals = np.zeros_like(counts)
    for ti, ci in enumerate(cond_idx):
        residuals[:, ti] = counts[:, ti] - psth[:, ci]

    return residuals, neurons, trials_df


# -------------------------------------
# 2. Noise correlation across trials
# -------------------------------------
def compute_noise_correlation_across_trials(residuals, neurons, trials_df, spikes_df, pair_type, output_dir):
    """
    Compute noise correlations (across trials) for all neuron pairs.

    Parameters
    ----------
    residuals : np.ndarray
        Residual activity [neurons × trials × bins].
    neurons : np.ndarray
        Neuron IDs.
    trials_df : pd.DataFrame
        Trial metadata.
    spikes_df : pd.DataFrame
        Unit metadata with 'neuron_id' and 'area_acronym_custom'.
    pair_type : {'within', 'across'}
        Whether to compute within-area or across-area correlations.
    output_dir : str or Path
        Directory to save per-session Parquet file.

    Returns
    -------
    str
        Path to written Parquet file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_neurons, n_trials, n_bins = residuals.shape
    print(f"Computing noise correlation across {n_trials} trials, {n_neurons} neurons")

    # Collapse time bins into mean residual per trial (spike count deviation)
    trial_responses = residuals.mean(axis=2)  # shape = [neurons × trials]

    # Transpose to trials × neurons
    X = trial_responses.T
    X -= X.mean(axis=0, keepdims=True)
    X /= X.std(axis=0, keepdims=True)

    # Compute correlation matrix across trials
    R = np.corrcoef(X, rowvar=False)

    # Map neurons to areas
    area_map = dict(zip(spikes_df['neuron_id'], spikes_df['area_acronym_custom']))

    iu = np.triu_indices(n_neurons, 1)
    results = []
    for i, j in zip(iu[0], iu[1]):
        area_i = area_map[neurons[i]]
        area_j = area_map[neurons[j]]

        # Filter within/across area pairs
        if (pair_type == 'within' and area_i != area_j) or \
           (pair_type == 'across' and area_i == area_j):
            continue

        results.append({
            'neuron_i': neurons[i],
            'neuron_j': neurons[j],
            'area_i': area_i,
            'area_j': area_j,
            'pair_type': pair_type,
            'noise_corr': float(R[i, j]),
            'n_trials': n_trials,
            'mouse_id': trials_df.iloc[0]['mouse_id'],
            'session_id': trials_df.iloc[0].get('session_id', None),
        })

    if not results:
        print(f"No valid pairs found for {pair_type}.")
        return None

    df = pd.DataFrame(results)
    table = pa.Table.from_pandas(df)
    outfile = Path(output_dir) / f"noise_corr_{pair_type}_across_trials.parquet"
    pq.write_table(table, outfile)
    print(f"✅ Saved {pair_type}-area noise correlations to {outfile}")

    return str(outfile)


# ---------------
# 3. Orchestrator
# ---------------
def compute_noise_correlations_across_trials(spikes_df, trials_df, output_dir,
                                             bin_size=0.005, window=(0.005, 0.1),
                                             pair_type='within',
                                             n_neurons_test=None):
    """
    Compute across-trial noise correlations for all neuron pairs.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    residuals, neurons, trials_df = compute_residuals_matrix(
        spikes_df, trials_df, bin_size, window, n_neurons_test
    )

    print(f"Computing {pair_type}-area noise correlations across trials")

    outfile = compute_noise_correlation_across_trials(
        residuals, neurons, trials_df, spikes_df, pair_type, output_dir
    )

    return outfile
# ---------------------------------

def concat_parquet_chunks_dask(output_dir, pattern="*.parquet", save_path=None):
    """
    Concatenate large Parquet trial chunks using Dask.

    Parameters
    ----------
    output_dir : str or Path
        Directory containing trial Parquet files.
    pattern : str
        Glob pattern to select files (default: '*.parquet').
    save_path : str or Path, optional
        If provided, saves concatenated Dask DataFrame to a single Parquet file.

    Returns
    -------
    dask.dataframe.DataFrame
        Lazy Dask DataFrame for further processing.
    """
    output_dir = Path(output_dir)
    files = sorted(output_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {output_dir} with pattern {pattern}")

    print(f"Found {len(files)} parquet chunks. Loading as Dask DataFrame...")
    ddf = dd.read_parquet([str(f) for f in files])
    print(f"Dask DataFrame created with {ddf.npartitions} partitions.")

    if save_path is not None:
        print(f"Writing concatenated Dask DataFrame to {save_path} ...")
        ddf.to_parquet(save_path, write_index=False, engine='pyarrow', overwrite=True)
        print(f"Saved to {save_path}")

    return ddf


def noise_correlation_analysis(nwb_file, results_path):
    """
    Perform ROC analysis on spike data from a NWB file.
    :param nwb_file: path to NWB file
    :param results_path: path to save results
    :return:
    """
    print('Starting noise correlation analysis for file:', nwb_file)

    # Parameters
    n_neurons_test = 50  # For dev / debugging (optional)

    # Get data
    trial_table = nwb_reader.get_trial_table(nwb_file)
    trial_table['outcome'] = trial_table['perf'].astype('int32').map(TRIAL_MAP)
    unit_table = nwb_reader.get_unit_table(nwb_file)

    # Process anatomical labels
    mouse_id = nwb_reader.get_mouse_id(nwb_file)
    unit_table['mouse_id'] = mouse_id
    trial_table['mouse_id'] = mouse_id
    unit_table = allen_utils.process_allen_labels(unit_table)

    # Within-area noise correlation
    # -----------------------------
    results_path_within = os.path.join(results_path, 'noise_correlation', 'within')
    compute_noise_correlations_across_trials(
        unit_table, trial_table,
        output_dir=results_path_within,
        pair_type='within',
    )

    # Across-area noise correlation
    # -----------------------------
    results_path_across = os.path.join(results_path, 'noise_correlation', 'across')
    #compute_noise_correlations_split(
    #    spikes_df, trials_df,
    #    output_dir=results_path_across",
    #    pair_type='across',
    #)

    print('Noise correlation analysis completed for file:', nwb_file)

    # Concatenate trial results
    # ------------------------------
    save_path_within = os.path.join(results_path, 'noise_correlation',
                                    'noise_correl_within_all_dask.parquet')
    within_ddf = concat_parquet_chunks_dask(
        output_dir=results_path_within,
        save_path=save_path_within
    )
    print(within_ddf.head())
    print(within_ddf.columns)
    within_ddf.to_parquet(os.path.join(results_path, 'noise_correlation',
                                       'noise_correl_within_all_dask.parquet'), write_index=False)

    return
