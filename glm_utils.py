#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_spikes_analysis
@file: glm_utils.py
@time: 10/23/2024 11:300 PM
"""

# Imports
import os
import time
import pathlib
import numpy as np
import pandas as pd
import scipy as sp
import pickle
from pynwb import NWBHDF5IO
import json
import sys
import ast
import subprocess
from sklearn.exceptions import ConvergenceWarning
import warnings
from itertools import combinations
from joblib import Parallel, delayed


sys.path.append('/home/mhamon/Github/NWB_reader')
sys.path.append('/home/mhamon/Github/allen_utils')

import multiprocessing as mp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyglmnet import GLM, GLMCV
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.use('TkAgg')

# Custom imports
import NWB_reader_functions as nwbreader
import allen_utils
import plotting_utils as putils

# Set global variables
BIN_SIZE = 0.1 # in seconds

ROOT_PATH = '/home/mhamon/data/'
# ROOT_PATH = os.path.join(r'C:\Users\mhamon/')

def bin_spike_times(spike_times, start_time, end_time, bin_size):
    """
    Bins spike times into a histogram with specified bin size.

    :param spike_times: array-like, spike times in seconds
    :param start_time: float, start time of the binning period in seconds
    :param end_time: float, end time of the binning period in seconds
    :param bin_size: float, size of each bin in seconds
    """
    bins = np.arange(start_time, end_time + bin_size, bin_size)
    binned, _ = np.histogram(spike_times, bins=bins)
    return binned

def bin_behavior(series_time, series_values, trial_start, trial_end, n_bins):
    """
    Bins a time series into specified number of bins within a trial period, returning the mean value for each bin.
    :param series_time: array-like, time points of the series in seconds
    :param series_values: array-like, values corresponding to the time points
    :param trial_start: float, start time of the trial in seconds
    :param trial_end: float, end time of the trial in seconds
    :param n_bins: int, number of bins to create within the trial period
    """
    bin_edges = np.linspace(trial_start, trial_end, n_bins + 1) # get bin edges
    binned = np.zeros(n_bins)    # init.
    digitized = np.digitize(series_time, bin_edges) - 1
    for i in range(n_bins):
        mask = digitized == i
        if np.any(mask):
            binned[i] = np.mean(series_values[mask])
    return binned


def build_design_matrix(data_dict, event_defs, analog_keys, bin_size, scale = None ):
    """
    Builds a design matrix X for GLM using input data dict and event definitions from
    `load_nwb_spikes_and_predictors`.
    :param data_dict: dict of predictor arrays, each shape (n_trials, n_bins)
    :param event_defs: dict of {name: (event_times, (start_offset, end_offset))} for event-based predictors
    :param analog_keys: dict of analog predictor names to their keys in the predictors dict
    :param bin_size: duration of one time bin (s)
    :return: tuple of (X, feature_names) where:
    - X: np.ndarray, shape (n_trials * n_bins, n_features) input predictors
    - feature_names: list of feature column names
    """
    # Init. design matrix and new feature names
    design_cols = []
    feature_names = []

    # Iterate over predictory data
    for name, data in data_dict.items():

        # Event-based predictors e.g. lick onset time
        if name in event_defs:
            _, (start_offset, end_offset) = event_defs[name]
            # Get kernel bin range
            offset_bins_pre = int(np.round(abs(start_offset) / bin_size))
            offset_bins_post = int(np.round(end_offset / bin_size))

            # Temporally expand the predictor within the kernel range
            for shift in range(-offset_bins_pre, offset_bins_post):
                shifted = np.zeros_like(data)
                if shift < 0:
                    shifted[:, :shift] = data[:, -shift:]
                elif shift > 0:
                    shifted[:, shift:] = data[:, :-shift]
                else:
                    shifted = data.copy()
                if name == 'whisker_stim' and scale is not None:
                    shifted = shifted * scale
                col = shifted.flatten()
                design_cols.append(col)
                feature_names.append(f"{name}_t{shift * bin_size:+.2f}s")

        # Analog predictors e.g. time series from DeepLabCut
        elif name in analog_keys:
            data_reshaped = data.flatten() # just flatten the data, data already processed (z-scored)
            design_cols.append(data_reshaped)
            feature_names.append(name)

        # Binary predictors e.g.
        else:
            design_cols.append(data.flatten())
            feature_names.append(name)

    # Stack and reshape into a tensor of shape (n_features, n_trials, n_bins)
    X = np.stack(design_cols, axis=1)
    n_trials_bins = X.shape[0]
    n_trials = len(data_dict[list(data_dict.keys())[0]])
    n_bins = n_trials_bins // n_trials
    X = X.reshape(n_trials, n_bins, -1).transpose((2, 0, 1)) # (n_features, n_trials, n_bins)

    return X, feature_names

def get_reduced_matrix(X_full, predictor_names, predictors_to_remove):
    """
    Given a design matrix X_full and a list of predictor names, returns a reduced matrix with those predictors removed.
    :param X_full: np.ndarray, shape (n_features, n_trials, n_bins) full design matrix
    :param predictor_names: list of str, ordered names of predictors in X_full
    :param predictors_to_remove: list of str, names of predictors to remove from X_full
    :return: tuple of (X_reduced, kept_features) where:
    - X_reduced: np.ndarray, shape (n_features_kept, n_trials, n_bins) reduced design matrix
    - kept_features: list of str, names of predictors that were kept in the reduced matrix
    """
    keep_mask_ids = [i for i, name in enumerate(predictor_names) if name not in predictors_to_remove]
    kept_features = [name for name in predictor_names if name not in predictors_to_remove]
    return X_full[keep_mask_ids, :, :], kept_features

def compute_mutual_info(y_true, y_pred):
    """
    Compute instant mutual information, in bits per spike, with the spike trains likelihood.
    The gain in predictability provided by the GLM parameters over a homogeneous Poisson process with constant firing intensity.
    :param y_true: np.ndarray, shape (n_bins,) true spike counts in each bin
    :param y_pred: np.ndarray, shape (n_bins,) predicted spike counts in each bin
    :return: float, bits per spike, single-spike mutual information
    """
    eps = 1e-12 # to avoid log(0) in case no spikes are predicted
    y_pred = np.clip(y_pred, eps, None)

    # Log-likelihood of fitted GLM
    ll_glm = np.sum(y_true * np.log(y_pred) - y_pred)

    # Homogeneous Poisson with constant firing rate
    total_spikes = np.sum(y_true)
    total_num_bins = len(y_true)
    lambda_homogen_rate = total_spikes / total_num_bins # mean spike counts / bin
    ll_hom = total_spikes*np.log(lambda_homogen_rate) - total_num_bins*lambda_homogen_rate

    # Bits per spike
    bits_per_spk = (ll_glm - ll_hom) / (total_spikes * np.log(2))
    return bits_per_spk


def plot_design_matrix_heatmap_single_trial(X, feature_names, trial_index, n_bins, bin_size):
    """
    Plot the design matrix predictors for a single trial as a heatmap.
    :param X: np.ndarray, shape (n_trials * n_bins, n_features)
    :param feature_names: list of str, names of features in the design matrix
    :param trial_index: int, index of the trial to visualize
    :param n_bins: int, number of bins per trial
    :param bin_size: float, size of each bin in seconds
    """
    X = X.reshape(X.shape[0], -1).transpose(1,0)  # Flatten the design matrix
    n_trials = X.shape[0] // n_bins
    if trial_index >= n_trials:
        raise IndexError(f"Trial index {trial_index} is out of range (0–{n_trials - 1})")

    # Extract the design matrix segment for the given trial
    start = trial_index * n_bins
    end = (trial_index + 1) * n_bins
    trial_matrix = X[start:end, :].T  # shape (n_features, n_bins)

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(feature_names)))
    im = ax.imshow(trial_matrix, aspect='equal', cmap='Grays', interpolation='nearest',
                   extent=[0, n_bins * bin_size, 0, len(feature_names)])

    ax.set_yticks(np.arange(len(feature_names)) + 0.5)
    ax.set_yticklabels(feature_names[::-1], fontsize=12)
    ax.set_ylabel('Features', fontsize=12, labelpad=15)
    ax.set_xticks(range(4))
    ax.set_xticklabels([-1,0,1,2], fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_title(f"Design matrix — trial {trial_index}", fontsize=12)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='1%', pad=0.5)
    plt.colorbar(im, cax=cax, label='Feature value', orientation='horizontal')

    plt.tight_layout()
    plt.show()
    return


def plot_design_matrix_vector_single_trial(X, feature_names, trial_index, n_bins, bin_size):
    """
    Plot the design matrix predictors for a single trial as a vectors.
    :param X: np.ndarray, shape (n_trials * n_bins, n_features)
    :param feature_names: list of str, names of features in the design matrix
    :param trial_index: int, index of the trial to visualize
    :param n_bins: int, number of bins per trial
    :param bin_size: float, size of each bin in seconds
    """
    X = X.reshape(X.shape[0], -1).transpose(1,0)  # Flatten the design matrix
    n_trials = X.shape[0] // n_bins
    if trial_index >= n_trials:
        raise IndexError(f"Trial index {trial_index} is out of range (0–{n_trials - 1})")

    # Get trial slice
    start = trial_index * n_bins
    end = (trial_index + 1) * n_bins
    trial_X = X[start:end]

    # Time axis in seconds
    time = np.arange(n_bins) * bin_size

    # Limit number of predictors to plot
    n_features_to_plot = len(feature_names)
    fig, axes = plt.subplots(n_features_to_plot, 1, figsize=(12, 1 * n_features_to_plot), sharex=True)

    if n_features_to_plot == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        values = trial_X[:, i]
        name = feature_names[i]
        putils.remove_top_right_frame(ax)
        if i != n_features_to_plot - 1:
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_visible(False)

        # Binary-like (0 or 1) predictor → pulse plot
        if np.array_equal(np.unique(values), [0]) or np.array_equal(np.unique(values), [0, 1]):
            ax.step(time, values, where='mid', color='black', lw=1.5)
            ax.set_ylim(-0.2, 1.2)
        else:  # Analog trace
            ax.plot(time, values, color='black', linewidth=1.5)

        ax.set_ylabel(name, fontsize=12, rotation=0, labelpad=80)
        ax.tick_params(axis='y', labelsize=12)
        ax.axvline(x=3, color='k', linestyle='--', lw=1, clip_on=False)

    # Add x-axis to last subplot
    axes[-1].set_xticks(range(4))
    axes[-1].set_xticklabels([ -1, 0, 1, 2], fontsize=12)
    axes[-1].set_xlabel('Time (s)', fontsize=12)

    # Adjust figure
    fig.align_ylabels()
    plt.suptitle(f"Design matrix: trial {trial_index}", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.02, top=0.96)
    plt.show()
    return

def remove_outliers_zscore(trace, threshold=10):
    '''Replace outliers with NaNs based on z-score.'''
    trace = np.array(trace)
    z = (trace - np.nanmean(trace)) / np.nanstd(trace)
    trace[np.abs(z) > threshold] = np.nan
    return trace

def interpolate_nans(trace):
    '''Linearly interpolate NaNs in the trace.''' #TODO: try smooth CubicSpline interpolation
    trace = np.array(trace)
    nans = np.isnan(trace)
    if np.all(nans):
        return np.zeros_like(trace)  # fallback
    trace[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), trace[~nans])
    return trace

def smooth_trace(trace, sigma=2):
    '''Smooth the trace using Gaussian filter.'''
    return sp.ndimage.gaussian_filter1d(trace, sigma=sigma)

def smooth_trace_savgol(trace, window_length=11, polyorder=3):
    '''Smooth the trace using Savitzky-Golay filter.'''
    return sp.signal.savgol_filter(trace, window_length, polyorder)
def smooth_trace_median(trace, kernel_size=5):
    '''Smooth the trace using median filter.'''
    return sp.signal.medfilt(trace, kernel_size=kernel_size)

def preprocess_dlc_trace(trace):
    '''Preprocess DLC trace to remove statistical outliers.'''
    trace = remove_outliers_zscore(trace, threshold=20)
    trace = interpolate_nans(trace)
    #trace = smooth_trace_median(trace, kernel_size=3)
    trace = smooth_trace(trace, sigma=5)
    #trace = smooth_trace_savgol(trace, window_length=11, polyorder=3)
    return trace

def assign_expertise_blocks(trial_table, n_consecutive=5):
    """
    Assign 'expert' or 'naive' labels to each trial for each mouse based on
    p_low vs p_chance, separately for reward_group, identifying blocks of at
    least `n_consecutive` trials.

    Parameters
    ----------
    trial_table : pd.DataFrame
        Must contain ['mouse_id', 'reward_group', 'trial_id', 'p_low', 'p_chance'].
        Should be sorted by trial_id within each mouse.
    n_consecutive : int
        Minimum number of consecutive trials to label as expert.

    Returns
    -------
    pd.DataFrame
        Original dataframe with added column 'block_perf_type' = 'naive' or 'expert'.
    """
    trial_table = trial_table.copy()
    trial_table['block_perf_type'] = 0

    def process_mouse(df_mouse):
        reward_group = df_mouse['reward_group'].iloc[0]

        # Criterion per trial
        if reward_group == 1:
            criterion = df_mouse['p_low'] > df_mouse['p_chance']
        elif reward_group == 0:
            criterion = df_mouse['p_low'] < df_mouse['p_chance']
        else:
            raise ValueError(f"Unexpected reward_group: {reward_group}")

        # Convert boolean series to runs of consecutive True/False
        vals = criterion.values
        expert_mask = np.zeros(len(vals), dtype=bool)

        start_idx = 0
        while start_idx < len(vals):
            if vals[start_idx]:
                # find run of consecutive True
                end_idx = start_idx
                while end_idx < len(vals) and vals[end_idx]:
                    end_idx += 1
                run_length = end_idx - start_idx
                if run_length >= n_consecutive:
                    expert_mask[start_idx:end_idx] = True
                start_idx = end_idx
            else:
                start_idx += 1

        df_mouse.loc[expert_mask, 'block_perf_type'] = 1
        return df_mouse

    trial_table = trial_table.groupby('mouse_id', group_keys=False).apply(process_mouse)
    return trial_table

def propagate_expertise_inplace(trial_table):
    """
    Propagate 'block_perf_type' (expert/naive) from whisker trials
    to other trial types in-place based on closest start_time per mouse.

    Parameters
    ----------
    trial_table : pd.DataFrame
        Must contain ['mouse_id', trial_type_col, time_col, 'block_perf_type'].
        Only whisker trials will have non-null 'block_perf_type' initially.
    time_col : str
        Column name for trial start time (used to find nearest whisker trial).
    trial_type_col : str
        Column name that identifies trial type (e.g., 'whisker', 'auditory', 'no_stim').

    Returns
    -------
    pd.DataFrame
        The same table with 'block_perf_type' updated for all trials.
    """
    trial_table = trial_table.sort_values(['start_time'])
    updated_trials = []

    for mouse_id, df_mouse in trial_table.groupby('mouse_id', group_keys=False):
        df_mouse = df_mouse.sort_values('start_time').copy()
        whisker_df = df_mouse[df_mouse['trial_type'] == 'whisker_trial']

        if whisker_df.empty:
            df_mouse['block_perf_type'] = np.nan
        else:
            # Use merge_asof for nearest whisker match
            merged = pd.merge_asof(
                df_mouse,
                whisker_df[['start_time', 'block_perf_type']].sort_values('start_time'),
                on='start_time',
                direction='nearest',
                suffixes=('', '_whisker')
            )
            # Update block_perf_type with nearest whisker label where missing
            df_mouse['block_perf_type'] = np.where(
                df_mouse['block_perf_type'].isna(),
                merged['block_perf_type_whisker'],
                df_mouse['block_perf_type']
            )

        updated_trials.append(df_mouse)

    # Combine back into one table
    updated_table = pd.concat(updated_trials, ignore_index=True)
    return updated_table

def keep_active_from_whisker_onset(trial_df):
    """
    Remove auditory blocks at onset of session, where mice were not yet engaged in the task, before whisker introduction
    :param trial_df: trial table dataframe with active trials only
    :return:
    """
    print('Keeping active trials and removing auditory onset blocks. Getting whisker trial indices...')

    # Keep active trials
    trial_df = trial_df[
        (~trial_df['context'].isin(['passive']))
        & (trial_df['perf'] != 6)
        & (trial_df['early_lick'] == 0)]

    df = trial_df.copy()

    # Find first whisker trial per mouse
    first_whisker_id = (
        df[df['trial_type'] == 'whisker_trial']
        .groupby('mouse_id')['trial_id']
        .min()
        .rename('first_whisker_id')
    )

    # Merge to get first whisker trial per mouse
    df = df.merge(first_whisker_id, on='mouse_id', how='left')

    # Keep only trials >= first whisker trial
    df = df[df['trial_id'] >= df['first_whisker_id']].copy()

    # Reindex trial_id to start at 0 from first whisker trial
    df['trial_id'] = df['trial_id'] - df['first_whisker_id']

    # Define also a whisker_trial_id, just for whisker trials
    df['whisker_trial_id'] = np.nan
    whisker_mask = df['trial_type'] == 'whisker_trial'
    df.loc[whisker_mask, 'whisker_trial_id'] = df.loc[whisker_mask].groupby('mouse_id').cumcount()
    df['whisker_trial_id'] = df['whisker_trial_id'].astype('Int64') # keep as nullable integer

    # Drop helper column
    df.drop(columns='first_whisker_id', inplace=True)

    return df




def load_perf_blocks(trial_table, mouse_id):

    path_to_data = r'M:\analysis\Axel_Bisi\combined_results'
    # curves_df = load_helpers.load_learning_curves_data(path_to_data=path_to_data, subject_ids=subject_ids)

    file_name = f'{mouse_id}_whisker_0_whisker_trial_learning_curve_interp.h5'
    path_to_file = os.path.join(ROOT_PATH, mouse_id, file_name)
    path_to_file = os.path.join(path_to_data, mouse_id,  'whisker_0', 'learning_curve',file_name)

    df_w = pd.read_hdf(path_to_file)

    trial_curves = []
    array_cols = ['p_mean', 'p_low', 'p_high', 'p_chance']
    for _, row in df_w.iterrows():
        n_trials = len(row[array_cols[0]])
        for t in range(n_trials):
            trial_dict = {}
            for col in df_w.columns:
                if col in array_cols:
                    trial_dict[col] = row[col][t]
                else:
                    trial_dict[col] = row[col]
            trial_dict['whisker_trial_id'] = t
            trial_curves.append(trial_dict)
    trial_curves_df = pd.DataFrame(trial_curves)
    trial_curves_df = assign_expertise_blocks(trial_curves_df, n_consecutive=5)

    # Merge learning curve data into trial table onto trial_id, for each mouse and onto whisker trials only
    trial_table = keep_active_from_whisker_onset(trial_table)  # get whisker trial index

    trial_table = trial_table.merge(
        trial_curves_df[
            ['mouse_id', 'block_perf_type', 'whisker_trial_id', 'p_mean', 'p_low', 'p_high', 'p_chance', 'mouse_cat',
             'learning_trial']],
        on=['mouse_id', 'whisker_trial_id'], how='left'
    )

    # Assign block_perf_typ to auditory_trial and no_stim_trial depending on the closest previous whisker trial
    trial_table = propagate_expertise_inplace(trial_table)

    return trial_table

def load_jaw_onset_data(mouse_id):
    """
    Load jaw onset data from NWB files.
    :param nwb_files: List of NWB file paths.
    :return:
    """
    print('Loading jaw onset data...')

    jaw_onset_list = []
    path_to_data = r'M:\analysis\Axel_Bisi\combined_results'

    file_path = os.path.join(path_to_data, mouse_id, 'dlc_jaw_onset_times.pkl')
    if os.path.exists(file_path):
        df = pd.read_pickle(file_path)
        jaw_onset_list.append(df)
    else:
        print(f"[WARN] Jaw onset data file not found for {mouse_id} at {file_path}. Skipping.")
        return
    jaw_onset_table = pd.concat(jaw_onset_list, ignore_index=True)
    return jaw_onset_table

def load_nwb_spikes_and_predictors(nwb_path, bin_size=0.1, nb_of_whisker_kernel= None, reward_kernel_per_type = False):
    """
    Loads spike trains from unit table and predictors from an NWB file.
    :param nwb_path: str, path to the NWB file
    :param bin_size: float, size of time bin in seconds
    :return: tuple of (spike_array, predictors, predictor_types, n_bins, bin_size) where
    - spike_array: np.ndarray, shape (n_neurons, n_trials, n_bins)
    - predictors: dict, contains predictor arrays
    - predictor_types: dict, contains types of predictors
    - n_bins: int, number of bins per trial
    - bin_size: float, size of time bin in seconds
    - nb_of_whisker_kernel : int, number of whisker kernels
    """
    with NWBHDF5IO(nwb_path, mode='r', load_namespaces=True) as io:
        nwbfile = io.read()
        #
        window_bounds_sec = (-1, 2)
        trials_df = nwbfile.trials.to_dataframe()
        trials_df = trials_df[(trials_df['context'] != 'passif') & (trials_df['perf'] != 6)].copy()
        trials_df['mouse_id'] = nwbreader.get_mouse_id(nwb_path)
        trials_df['session_id'] = nwbreader.get_session_id(nwb_path)
        #
        trials_df = load_perf_blocks(trials_df, trials_df['mouse_id'].unique()[0])
        trial_starts = trials_df['start_time'].values + window_bounds_sec[0]
        trial_ends = trials_df['start_time'].values + window_bounds_sec[1]
        #
        n_trials = len(trial_starts)
        #
        win_start, win_end = window_bounds_sec
        n_bins = int(np.round((win_end - win_start) / bin_size))

        unit_table = nwbfile.units.to_dataframe()
        # unit_table = unit_table.sample(frac=1) why is there a random shuffle ????
        unit_table = unit_table[unit_table['bc_label']=='good']
        # unit_table = unit_table[unit_table['ccf_parent_acronym'].isin(['SSp-bfd', 'SSs'])]

        unit_table = unit_table[unit_table['firing_rate'].astype(float).ge(2.0)]
        unit_table = unit_table[~unit_table['ccf_acronym'].isin(allen_utils.get_excluded_areas())]
        print('CAREFUL THIS IS ONLY 2 RANDOM UNITS')
        unit_table = unit_table.sample(n=2, random_state=None)

        # Use index as new column named "unit_id", then reset
        unit_table['neuron_id'] = unit_table.index
        unit_table.reset_index(drop=True, inplace=True)
        neurons_ccf = unit_table['ccf_parent_acronym'].values
        # unit_table = unit_table.iloc[[21, 34]]



        # # Here we can simulate Poisson spikes OR make it strictly constant.
        # rate_hz = 10
        # lam = rate_hz * bin_size
        #
        # # Option A: deterministic constant rate
        # spike_array = np.full((1, n_trials, n_bins), lam)
        #
        # # Option B: realistic Poisson spikes
        # spike_array = np.random.poisson(lam, size=(1, n_trials, n_bins))
        #
        # # Build dummy metadata
        # unit_table = pd.DataFrame([{
        #     "neuron_id": 999999,
        #     "firing_rate": rate_hz,
        #     "ccf_parent_acronym": "FAKE"
        # }])
        # neurons_ccf = 'FAKE'
        # ------------------
        # Spike Trains
        # ------------------
        spike_array = []
        for unit in unit_table.itertuples():
            binned_trials = []
            for start, end in zip(trial_starts, trial_ends):
                bins = np.linspace(start, end, n_bins + 1)
                spike_times = unit.spike_times
                binned, _ = np.histogram(spike_times, bins=bins)
                padded = np.zeros(n_bins)
                padded[:len(binned)] = binned
                binned_trials.append(padded)
            spike_array.append(np.stack(binned_trials))
        spike_array = np.stack(spike_array)  # (n_neurons, n_trials, n_bins)


        # ------------------
        # Predictors
        # ------------------
        predictors = {}

        # Binary Predictors


        # Trial index scaled to total number of trials
        trial_idx_scaled = np.arange(n_trials) / (n_trials-1)
        predictors['trial_index_scaled'] = np.tile(trial_idx_scaled[:, None], (1, n_bins))

        predictors['offset'] = np.ones((n_trials, n_bins))
        # Now this is if the last whisker presented was rewarded or not and same for auditory
        stim_type = trials_df['trial_type'].fillna('').values
        rewarded = (
                trials_df['trial_type'].isin(['whisker_trial', 'auditory_trial']).astype(int)
                * trials_df['reward_available'].astype(int)
                * trials_df['lick_flag'].astype(int)
        ).fillna(0).values

        prev_whisker_reward = np.zeros(n_trials)

        # Running memory for last reward status
        last_whisker_reward = 0

        for i in range(n_trials):
            # Store the most recent reward info
            prev_whisker_reward[i] = last_whisker_reward
            if stim_type[i] == 'whisker_trial':
                last_whisker_reward = rewarded[i]


        # Broadcast to bins
        # predictors['last_whisker_reward'] = np.tile(prev_whisker_reward[:, None], (1, n_bins))
        #  Independent code to get the proportion of past whisker trials that were rewarded
        past_whisker_trials = 0
        past_whisker_rewards = 0
        prop_past_whisker_rewarded = np.zeros(n_trials)

        for i in range(1, n_trials):
            # Whisker trial history
            if stim_type[i - 1] == 'whisker_trial':
                past_whisker_trials += 1
                if rewarded[i - 1] > 0:
                    past_whisker_rewards += 1

            # Avoid divide-by-zero
            if past_whisker_trials > 0:
                prop_past_whisker_rewarded[i] = past_whisker_rewards / past_whisker_trials

        # predictors['prop_past_whisker_rewarded'] = np.tile(prop_past_whisker_rewarded[:, None], (1, n_bins))
        block_perf_type = trials_df['block_perf_type'].to_numpy()  # shape (n_trials,)
        predictors['block_perf_type'] = np.tile(block_perf_type[:, None], (1, n_bins))

        # Rolling reward proportion
        whisker_reward_rate = np.zeros(n_trials)
        for i in range(n_trials):
            start_idx = max(0, i - 5)
            recent_trials = stim_type[start_idx:i]
            recent_rewards = rewarded[start_idx:i]
            whisker_mask = recent_trials == 'whisker_trial'
            if np.sum(whisker_mask) > 0:
                whisker_reward_rate[i] = np.sum(recent_rewards[whisker_mask]) / np.sum(whisker_mask)
            else:
                whisker_reward_rate[i] = 0
        scale = None
        scale = np.tile(whisker_reward_rate[:, None], (1, n_bins))

        # predictors['whisker_reward_rate_5'] = np.tile(whisker_reward_rate[:, None], (1, n_bins))

        total_rewards = np.sum(rewarded > 0)
        total_rewards = total_rewards if total_rewards > 0 else 1
        cum_reward = np.zeros(n_trials)
        reward_so_far = 0
        for i in range(n_trials):
            cum_reward[i] = reward_so_far / total_rewards
            if rewarded[i] > 0:
                reward_so_far += 1

        # Add to predictors (scaled cumulative rewards across all trial types)
        predictors['sum_reward_scaled'] = np.tile(cum_reward[:, None], (1, n_bins))

        # Whisker hit predictor: 1 if whisker trial and actually rewarded, else 0
        # whisker_hit = ((stim_type == 'whisker_trial') & (rewarded > 0)).astype(int)
        # Add to predictors
        # predictors['whisker_hit'] = np.tile(whisker_hit[:, None], (1, n_bins))

        binary_keys ={
            'trial_index_scale':'trial_index_scaled',
            # 'last_whisker_reward':'last_whisker_reward',
            # 'prop_past_whisker_rewarded':'prop_past_whisker_rewarded',
            # 'whisker_reward_rate_5': 'whisker_reward_rate_5',
            'sum_reward_scaled':'sum_reward_scaled',
            'block_perf_type':'block_perf_type'
            # 'whisker_hit': 'whisker_hit'
        }

        # Event-based predictors (rasterized kernels will be applied later)
        def rasterize_event(event_times, first_only=False):
            """
            Rasterizes session-wide event times into a binary trial-by-time-bin matrix.
            :param event_times: array-like, 1D array of timestamps (e.g., lick times)
            :param first_only: bool, if True, only the first event per trial is rasterized
            :return: np.ndarray, shape (n_trials, n_bins) binary matrix with 1s marking event bins
            """
            matrix = np.zeros((n_trials, n_bins))
            for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
                bins = np.linspace(start, end, n_bins + 1)
                # Get events occuring within the trial time window
                trial_events = event_times[(event_times >= start) & (event_times < end)]

                # If first_only, only keep the first event after trial start e.g. for lick onset
                if first_only:
                    trial_events = [t for t in trial_events if
                                    t >= (start + abs(window_bounds_sec[0]))]  # keep dlc licks after trial start
                    if len(trial_events) > 0:
                        first_trial_event = trial_events[0]
                        idx = np.digitize(first_trial_event, bins) - 1
                        if 0 <= idx < n_bins:
                            matrix[i, idx] = 1
                # Otherwise, rasterize all events within the trial
                else:
                    if len(trial_events) > 0:
                        idxs = np.digitize(trial_events, bins) - 1
                        for idx in idxs:
                            if 0 <= idx < n_bins:
                                matrix[i, idx] = 1
            return matrix

        # Get video start time to align to other events
        video_start_time = nwbfile.processing['behavior']['BehavioralTimeSeries'].time_series['whisker_angle'].timestamps[0]

        def get_events(key):
            return (
                nwbfile.processing['behavior']['BehavioralEvents'].time_series[key].data[:],
                nwbfile.processing['behavior']['BehavioralEvents'].time_series[key].timestamps[:] #TODO: add delay between camera and session
            )

        # Get lick onset time
        piezo_licks = np.array(list(get_events('piezo_lick_times')[1]))
        # tongue_dlc_licks = np.array(list(get_events('tongue_dlc_licks')[1])) / 200 + video_start_time

        jaw_onset_table = load_jaw_onset_data(trials_df['mouse_id'].unique()[0] )

        trials_df = trials_df.merge(jaw_onset_table[['mouse_id', 'session_id', 'trial_id', 'jaw_dlc_onset', 'piezo_lick_time']],
                                on=['mouse_id', 'session_id', 'trial_id'], how='left')

        all_jaw_onsets = trials_df['jaw_dlc_onset'].values + trials_df['start_time'].values

        # Get available stimulus times
        try:
            auditory_times = list(get_events('auditory_hit_trial')[1]) + list(get_events('auditory_miss_trial')[1])
        except:
            auditory_times = list(get_events('auditory_hit_trial')[1])
        auditory_times = np.array(sorted(auditory_times))

        try:
            whisker_times = list(get_events('whisker_hit_trial')[1]) + list(get_events('whisker_miss_trial')[1])
        except:
            whisker_times = list(get_events('whisker_hit_trial')[1])
        whisker_times = np.array(sorted(whisker_times))

        if nb_of_whisker_kernel is None:
            # Define event kernels for single trial events
            event_defs = {
                # 'dlc_lick_onset': (tongue_dlc_licks, (-0.3, 0.6)), # in seconds
                'jaw_onset' : (all_jaw_onsets, (-0.5, 0)),
                'auditory_stim': (auditory_times, (-0.1, 0.6)),
                'whisker_stim': (whisker_times, (-0.1, 0.6)),
                'piezo_reward': (piezo_licks, (-0, 0.6)),
            }
            print(all_jaw_onsets)
            print(whisker_times)
        else :
            event_defs = {
                # 'dlc_lick_onset': (tongue_dlc_licks, (-0.3, 0.6)), # in seconds
                'jaw_onset' : (all_jaw_onsets, (-0.5, 0)),
                'auditory_stim': (auditory_times, (-0.1, 0.6)),
                'piezo_reward': (piezo_licks, (-0, 0.6)),
            }
            whisker_splits = np.array_split(whisker_times, nb_of_whisker_kernel)
            for nb in range(nb_of_whisker_kernel):
                event_defs[f'whisker_stim_{nb}'] = (whisker_splits[nb], (-0.1, 0.6))


        for name, (times, _) in event_defs.items():
            if name=='dlc_lick_onset':
                predictors['dlc_lick_onset'] = rasterize_event(times, first_only=True)

            else:
                predictors[name] = rasterize_event(times, first_only=False)

        # Rasterize the rewards, only for hit trials and only auditory if non-rewarded mouse
        all_piezo_events = np.array(list(get_events('piezo_lick_times')[1]))

        if not reward_kernel_per_type:
            piezo_reward_matrix = np.zeros((n_trials, n_bins))
            for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
                # Only keep licks if this trial was a HIT
                if  rewarded[i] > 0:
                    bins = np.linspace(start, end, n_bins + 1)
                    trial_events = all_piezo_events[(all_piezo_events >= start) & (all_piezo_events < end)]
                    if len(trial_events) > 0:
                        if len(trial_events) > 1:
                            first_event = trial_events[1]  # take the first one
                            idx = np.digitize(first_event, bins) - 1
                            if 0 <= idx < n_bins:
                                piezo_reward_matrix[i, idx] = 1
                        else:
                            first_event = trial_events[0]  # take the first one
                            idx = np.digitize(first_event, bins) - 1
                            if 0 <= idx < n_bins:
                                piezo_reward_matrix[i, idx] = 1
            predictors['piezo_reward'] = piezo_reward_matrix

        else :
            piezo_reward_matrix_wh = np.zeros((n_trials, n_bins))
            piezo_reward_matrix_au = np.zeros((n_trials, n_bins))
            event_defs['piezo_reward_au'] = (piezo_licks, (-0.1, 0.6))
            for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
                # Only keep licks if this trial was a HIT
                if rewarded[i] > 0:
                    bins = np.linspace(start, end, n_bins + 1)
                    trial_events = all_piezo_events[(all_piezo_events >= start) & (all_piezo_events < end)]
                    if len(trial_events) > 0:
                        if len(trial_events) > 1:
                            first_event = trial_events[1]  # take the first one
                            idx = np.digitize(first_event, bins) - 1
                            if 0 <= idx < n_bins:
                                if stim_type[i] == 'whisker_trial':
                                    piezo_reward_matrix_wh[i, idx] = 1
                                else :
                                    piezo_reward_matrix_au[i, idx] = 1
                        else:
                            first_event = trial_events[0]  # take the first one
                            idx = np.digitize(first_event, bins) - 1
                            if 0 <= idx < n_bins:
                                if stim_type[i] == 'whisker_trial':
                                    piezo_reward_matrix_wh[i, idx] = 1
                                else :
                                    piezo_reward_matrix_au[i, idx] = 1
            predictors['piezo_reward'] = piezo_reward_matrix_wh
            predictors['piezo_reward_au'] = piezo_reward_matrix_au


        # Analog predictors, already filtered by likelihood at NWB creation
        analog_keys = {
            'side_nose_dist': 'nose_distance',
            #'side_nose_vel': 'nose_velocity',
            'top_nose_dist': 'top_nose_distance',
            #'top_nose_vel': 'top_nose_velocity',
            'jaw_dist': 'jaw_distance',
            #'jaw_vel': 'jaw_velocity',
            'pupil_area': 'pupil_area',
            #'pupil_area_vel': 'pupil_area_velocity',
            #'whisker_angle': 'whisker_angle',
            'whisker_vel': 'whisker_velocity'
        }

        def get_series(key):
            return (
                nwbfile.processing['behavior']['BehavioralTimeSeries'].time_series[key].data[:],
                nwbfile.processing['behavior']['BehavioralTimeSeries'].time_series[key].timestamps[:]
            )

        def bin_behavior(data, times):
            result = []
            for start, end in zip(trial_starts, trial_ends):
                edges = np.linspace(start, end, n_bins + 1)
                values = np.zeros(n_bins)
                idxs = np.digitize(times, edges) - 1
                for i in range(n_bins):
                    mask = idxs == i
                    if np.any(mask):
                        values[i] = np.mean(data[mask])
                result.append(values)
            return np.stack(result)

        def pad_to_equal_length(a, b):
            n = max(len(a), len(b))
            a_padded = np.pad(a, (0, n - len(a)), constant_values=np.nan)
            b_padded = np.pad(b, (0, n - len(b)), constant_values=np.nan)
            return a_padded, b_padded

        for short_key, long_key in analog_keys.items():
            data, times = get_series(long_key)
            # Preprocess data to remove outliers
            data = preprocess_dlc_trace(data)
            assert np.isfinite(data).all() # make sure there are no NaNs or infs
            data, times = pad_to_equal_length(data, times)

            predictors[short_key] = bin_behavior(data, times)

        # Compute norm of nose movement
        predictors['nose_dist'] = np.sqrt(predictors['side_nose_dist']**2 + predictors['top_nose_dist']**2)
        #predictors['nose_vel'] = np.gradient(predictors['nose_dist'], bin_size, axis=1)
        analog_keys['nose_dist'] = 'nose_distance'
        predictors['whisker_vel'] = np.abs( predictors['whisker_vel'])

        #analog_keys['nose_vel'] = 'nose_velocity'
        predictors.pop('side_nose_dist')
        #predictors.pop('side_nose_vel')
        predictors.pop('top_nose_dist')
        #predictors.pop('top_nose_vel')


        # Prepare outputs
        predictor_types ={'binary_keys': binary_keys,
                         'analog_keys': analog_keys,
                         'event_defs': event_defs}

        return spike_array, predictors, predictor_types, n_bins, bin_size, neurons_ccf, scale


def fit_neuron_glm(neuron_id, spikes_trainval, X_trainval, spikes_test, X_test, lambdas):
    """
    Fit a GLM to the data for a single neuron, in multiprocessing context.
    :param neuron_id: int, ID of the neuron to fit
    :param spikes_trainval: np.ndarray, shape (n_neurons, n_trials, n_bins) spike counts for training and validation
    :param X_trainval: np.ndarray, shape (n_features, n_trials, n_bins) design matrix for training and validation
    :param spikes_test: np.ndarray, shape (n_neurons, n_trials, n_bins) spike counts for testing
    :param X_test: np.ndarray, shape (n_features, n_trials, n_bins) design matrix for testing
    :param lambdas: float or np.ndarray, regularization parameter(s) for Ridge regression
    :return: dict, results of the GLM fit to be appended in Pool

    """
    # Fitting parameters, N numbers
    cv_folds = 5
    n_features = X_trainval.shape[0]
    n_bins = X_trainval.shape[2]

    # Select unit spikes, and reshape data for GLM formulation
    y_trainval = spikes_trainval[neuron_id].reshape(spikes_trainval.shape[1], -1).flatten()  # (n_trials * n_bins,)
    X_trainval = X_trainval.transpose(1,2,0).reshape(-1, n_features)
    y_test = spikes_test[neuron_id].reshape(spikes_test.shape[1], -1).flatten()  # (n_trials * n_bins,)
    X_test = X_test.transpose(1,2,0).reshape(-1, n_features) # (n_trials * n_bins, n_predictors)


    # If provided lambdas is an array, cross-validate lambda
    if isinstance(lambdas, np.ndarray):
        # Cross-validation with warm-restarts and regularization paths
        glmcv = GLMCV(distr='poisson',
                    score_metric='pseudo_R2',
                    fit_intercept=False,
                    alpha=0.0,
                    reg_lambda=lambdas,
                    solver='cdfast',
                    cv=cv_folds,
                    tol=1e-4,
                    verbose=False,
                    random_state=42)
        glmcv.fit(X_trainval, y_trainval)
        lambda_opt = glmcv.reg_lambda_opt_

    elif isinstance(lambdas, float):
        lambda_opt = lambdas    #optimized previously from full

    # Final fit on full trainval with best lambda
    glm_final = GLM(distr='poisson',
                    score_metric='pseudo_R2',
                    fit_intercept=False,
                    alpha=0.0,
                    reg_lambda=lambda_opt,
                    solver='cdfast',
                    tol=1e-4,
                    verbose=False,
                    random_state=42)

    fit_success = True  # Will remain True if no convergence issues

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", category=ConvergenceWarning)
            glm_final.fit(X_trainval, y_trainval)

            # Check if any convergence warnings occurred
            for warning in w:
                if issubclass(warning.category, ConvergenceWarning):
                    print("Warning: Final GLM did not converge:", warning.message)
                    fit_success = False

        # -------------------------------------------
        # Evaluate final model on train and test sets
        # -------------------------------------------

        # Train and test scores (pseudo-R2)
        y_trainval_pred = glm_final.predict(X_trainval)
        train_score = glm_final.score(X_trainval, y_trainval)
        y_test_pred = glm_final.predict(X_test)
        test_score = glm_final.score(X_test, y_test)

        # Compute log-likelihoods
        train_ll = np.sum(y_trainval * np.log(y_trainval_pred + 1e-12) - y_trainval_pred)
        test_ll = np.sum(y_test * np.log(y_test_pred + 1e-12) - y_test_pred)

        # Single-trial correlation (r)
        train_corr, _ = sp.stats.pearsonr(y_trainval, y_trainval_pred)
        test_corr, _ = sp.stats.pearsonr(y_test, y_test_pred)

        # Predictor-spike mutual information (MI)
        train_mi = compute_mutual_info(y_trainval, y_trainval_pred)
        test_mi = compute_mutual_info(y_test, y_test_pred)

        # Format results
        result = {
            'neuron_id': neuron_id,
            'lambda_opt': lambda_opt,
            'train_ll': train_ll,
            'train_score': train_score,
            'train_corr':train_corr,
            'train_mi': train_mi,
            'test_ll': test_ll,
            'test_score': test_score,
            'coef': glm_final.beta_.copy(),
            'y_train': y_trainval,
            'y_test': y_test,
            'y_pred': y_test_pred,
            'y_train_pred' : y_trainval_pred,
            'test_corr': test_corr,
            'test_mi': test_mi,
            'n_bins': n_bins,
            'fit_success': fit_success,
        }
        return result

    except Exception as e:
        print(f"GLM failed to converge: {e}")
        return None



def fit_neuron_glm_wrapper(args):
    return fit_neuron_glm(*args)  # small wrapper because map needs a single argument

def parallel_fit_glms(spikes_trainval, X_trainval, spikes_test, X_test, lambdas, n_jobs=10):

    # Create a list of args to be passed on for each neuron
    n_neurons = spikes_trainval.shape[0]

    # Check lambdas input type
    if isinstance(lambdas, dict): # optimal lambdas per neuron pre-computed
        args_list = [
            (neuron_id, spikes_trainval, X_trainval, spikes_test, X_test, lambdas[neuron_id])
            for neuron_id in range(n_neurons)
        ]
    else:
        args_list = [
            (neuron_id, spikes_trainval, X_trainval, spikes_test, X_test, lambdas)
            for neuron_id in range(n_neurons)
        ]

    with mp.Pool(processes=n_jobs) as pool:
        results = list(pool.map(fit_neuron_glm_wrapper, args_list))

    return results

def save_model_input_output(X, spikes, feature_names, output_dir, neuron_ccf):
    commit_hash = get_git_revision_short_hash()
    data_path = pathlib.Path(output_dir, 'data')
    data_path = pathlib.Path(os.path.join(data_path, commit_hash))
    data_path.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(data_path, 'data.pkl'), 'wb') as f:
        pickle.dump({'input': X, 'output': spikes, 'feature_names': feature_names, 'commit_hash' : commit_hash, 'neurons_id': neuron_ccf}, f)

    print('Saved input predictor data:', X.shape)
    print('Saved neural output spike data:', spikes.shape)
    return

def load_model_input_output(output_dir):
    data_path = pathlib.Path(output_dir, 'data', 'ad9dbf2')
    with open(os.path.join(data_path,'data.pkl'), 'rb') as f:
        data = pickle.load(f)
    X = data['input']
    spikes = data['output']
    feature_names = data['feature_names']
    return X, spikes, feature_names

def save_model_results2(result_df, filename, output_dir):
    result_path = pathlib.Path(output_dir, 'models')
    result_path.mkdir(parents=True, exist_ok=True)
    if 'coef' in result_df.columns and result_df['coef'].apply(lambda x: isinstance(x, list)).any():
        result_df['coef'] = result_df['coef'].apply(lambda x: json.dumps(x))

    result_df.to_parquet(os.path.join(result_path, '{}_results.parquet'.format(filename)))
    print('Saved model results in: ', result_path)
    return
    



def save_model_results(result_df, filename, commit_hash, output_dir):
    result_path = pathlib.Path(output_dir, 'models')
    result_path.mkdir(parents=True, exist_ok=True)

    # Iterate through columns to identify and convert problematic 'object' types
    for col in result_df.columns:
        if result_df[col].dtype == 'object':
            # Check if this object column contains lists or numpy arrays
            types_in_col = result_df[col].apply(type).value_counts()

            # Define a helper function to convert
            def to_json_if_complex(x):
                if isinstance(x, (list, np.ndarray)):
                    try:
                        return json.dumps(x.tolist() if isinstance(x, np.ndarray) else x)
                    except TypeError as e:
                        print(f"Warning: Could not JSON dump an element in column '{col}': {e} - Element type: {type(x)} - Value: {x}")
                        return str(x) # Fallback to string representation if JSON fails
                return x # Return as is if not a list or numpy array

            if (result_df[col].apply(lambda x: isinstance(x, (list, np.ndarray)))).any():
                result_df[col] = result_df[col].apply(to_json_if_complex)

    try:

        result_df.to_parquet(os.path.join(result_path, '{}_{}_results.parquet'.format(commit_hash, filename)))
        print('Saved model results in: ', result_path)
    except ValueError as e:
        print(f"Failed to save to parquet: {e}")
        raise # Re-raise the exception to propagate the error

    return

def load_model_results(filename, output_dir):

    result_path = pathlib.Path(output_dir, 'models')
    file_path = os.path.join(result_path, '{}_results.parquet'.format(filename))
    try:
        result_df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print('No model results found in:', file_path)
        return None
    return result_df

def fit_one_reduced_model(combo_name, features_to_remove, spikes_trainval, X_trainval, spikes_test, X_test,
                          feature_names, full_optimal_lambdas, trainval_ids, test_ids, fold_idx, n_jobs):
    # Reduce design matrix
    X_trainval_reduced, kept_features = get_reduced_matrix(X_trainval, feature_names, features_to_remove)
    X_test_reduced, _ = get_reduced_matrix(X_test, feature_names, features_to_remove)

    # Fit GLM
    results_reduced = parallel_fit_glms(
        spikes_trainval=spikes_trainval,
        X_trainval=X_trainval_reduced,
        spikes_test=spikes_test,
        X_test=X_test_reduced,
        lambdas=full_optimal_lambdas,
        n_jobs=n_jobs
    )

    results_reduced_df = pd.DataFrame(results_reduced)
    results_reduced_df['fold'] = fold_idx
    results_reduced_df['train_trials'] = [trainval_ids] * len(results_reduced_df)
    results_reduced_df['test_trials'] = [test_ids] * len(results_reduced_df)
    results_reduced_df['model_name'] = combo_name
    results_reduced_df['predictors'] = [list(kept_features)] * len(results_reduced_df)
    return results_reduced_df


def run_unit_glm_pipeline_with_pool(nwb_path, output_dir, n_jobs=10):
    print('GLM fitting for:', pathlib.Path(nwb_path).name)

    commit_hash = get_git_revision_short_hash()
    mouse_id = nwbreader.get_mouse_id(nwb_path)
    mouse_output_path = pathlib.Path(output_dir, mouse_id, 'whisker_0', 'unit_glm')
    mouse_output_path.mkdir(parents=True, exist_ok=True)
    # -----------------------------------
    # Load and prepare data from NWB file
    # -----------------------------------

    # Get trial table, input and output formatted for GLM, list of predictor types
    trials_df = nwbreader.get_trial_table(nwb_path)
    trials_df = trials_df[(trials_df['context'] != 'passif') &(trials_df['perf'] != 6)].copy()
    trials_df['mouse_id'] = mouse_id
    trials_df = load_perf_blocks(trials_df, mouse_id)

    trials_df = trials_df.reset_index(drop=True)
    #try: #uncomment when design matrix fixed, so that no need to recompute it
    #    X, spikes, feature_names = load_model_input_output(output_dir)
    #

    #  Get trial table, input and output formatted for GLM, list of predictor types
    spikes, predictors, predictor_types, n_bins, bin_size, neurons_ccf, scale = load_nwb_spikes_and_predictors(nwb_path, bin_size=BIN_SIZE)
    event_defs = predictor_types['event_defs']
    analog_keys = predictor_types['analog_keys']

    # Build design matrix for entire dataset
    X, feature_names = build_design_matrix(predictors, event_defs, analog_keys, bin_size=BIN_SIZE, scale = None)
    X = np.nan_to_num(X)


    n_features = X.shape[0]

    # Save input/output data
    save_model_input_output(X, spikes, feature_names, mouse_output_path, neurons_ccf)
    whisker_kernels = False
    if whisker_kernels:
        all_Xs = []
        feature_namess = []
        nb_whisker_kernels = []
        for number_of_whisker_kernel in range(2,5):
            spikes, predictors, predictor_types, n_bins, bin_size, neurons_ccf, _ = load_nwb_spikes_and_predictors(nwb_path, bin_size=BIN_SIZE, nb_of_whisker_kernel = number_of_whisker_kernel)
            event_defs = predictor_types['event_defs']
            analog_keys = predictor_types['analog_keys']

            # Build design matrix for entire dataset
            X_extra, feature_names_extra = build_design_matrix(predictors, event_defs, analog_keys, bin_size=BIN_SIZE)
            all_Xs.append(X_extra)
            feature_namess.append(feature_names_extra)
            nb_whisker_kernels.append(number_of_whisker_kernel)

    reward_kernels = False
    if reward_kernels:

        X_rewards = []
        feature_names_rewards = []
        nb_whisker_kernel_rewards = []
        for number_of_whisker_kernel in [1,2]:
            spikes, predictors, predictor_types, n_bins, bin_size, neurons_ccf, _ = load_nwb_spikes_and_predictors(nwb_path, bin_size=BIN_SIZE, nb_of_whisker_kernel = number_of_whisker_kernel, reward_kernel_per_type = True)
            event_defs = predictor_types['event_defs']
            analog_keys = predictor_types['analog_keys']

            # Build design matrix for entire dataset
            X_extra, feature_names_extra = build_design_matrix(predictors, event_defs, analog_keys, bin_size=BIN_SIZE)
            X_rewards.append(X_extra)
            feature_names_rewards.append(feature_names_extra)
            nb_whisker_kernel_rewards.append(number_of_whisker_kernel)


    # ---------------------------------------
    # Train/test data cross-validation splits
    # ---------------------------------------
    model_res_df_outer = []
    cv_outer_folds = 5

    outer_kf = KFold(n_splits=cv_outer_folds, shuffle=True, random_state=42)

    for fold_idx, (trainval_ids, test_ids) in enumerate(outer_kf.split(trials_df.index)):

        print('Fold', fold_idx)

        # Get data splits
        X_trainval, X_test = X[:,trainval_ids,:], X[:,test_ids,:]
        spikes_trainval, spikes_test = spikes[:,trainval_ids,:], spikes[:,test_ids,:]

        # Reshape
        X_trainval = X_trainval.reshape(X_trainval.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        # Standardize DLC features using train statistics, apply on test
        for dlc_idx in range(n_features-4, n_features):
            mean = X_trainval[dlc_idx,:].mean()
            std = X_trainval[dlc_idx,:].std()
            X_trainval[dlc_idx,:] = (X_trainval[dlc_idx,:] - mean) / std
            X_test[dlc_idx,:] = (X_test[dlc_idx,:] - mean) / std

            X[dlc_idx,:] = (X[dlc_idx,:] - mean) / std # transform original X for plotting single trial input

        X_trainval = X_trainval.reshape(X_trainval.shape[0], len(trainval_ids), -1) # reshape
        X_test = X_test.reshape(X_test.shape[0], len(test_ids), -1)

        debug = False
        if debug:
            for test_idx in test_ids[0:5]:
                plot_design_matrix_heatmap_single_trial(X, feature_names, trial_index=test_idx, n_bins=n_bins, bin_size=BIN_SIZE)
                plot_design_matrix_vector_single_trial(X, feature_names, trial_index=test_idx, n_bins=n_bins,
                                                       bin_size=BIN_SIZE)
            return

        # -----------------------------------
        # Fit full GLMs using multiprocessing
        # -----------------------------------

        lambdas = np.exp(np.linspace(np.log(0.5), np.log(1e-5), 5)) # regul. strength hyperparam.
        #lambdas = np.array([0.001, 0.5])

        start_time = time.time()
        results = parallel_fit_glms(spikes_trainval=spikes_trainval,
                                    X_trainval=X_trainval,
                                    spikes_test=spikes_test,
                                    X_test=X_test,
                                    lambdas=lambdas,
                                    n_jobs=n_jobs)
        print('GLM fitting complete in ', time.time() - start_time)

        results = sorted(results, key=lambda r: r['neuron_id']) # sort for neuron_id--lambda_opt matching

        # Save mouse results for model type and fold index
        model_res_df = pd.DataFrame(results)
        model_res_df['fold'] = fold_idx
        model_res_df['train_trials'] = [trainval_ids] * len(model_res_df)
        model_res_df['test_trials'] = [test_ids] * len(model_res_df)
        model_res_df['model_name'] = 'full'
        model_res_df['predictors'] = [feature_names] * len(model_res_df)
        describe_cols = ['neuron_id', 'fold', 'train_score', 'test_score', 'train_corr', 'test_corr',
                         'train_mi', 'test_mi']
        print(model_res_df[describe_cols].describe())
        save_model_results(model_res_df, filename='model_full_fold{}'.format(fold_idx), commit_hash = commit_hash, output_dir=mouse_output_path)

        # Add to list of global dat
        model_res_df_outer.append(model_res_df)

        # Plot single trial predictions models
        debug=False
        if debug:
            for neuron_id in model_res_df.neuron_id.unique():
                plot_trial_grid_predictions(model_res_df, trials_df, neuron_id, bin_size=BIN_SIZE)
            return


        # --------------------------------------
        # Fit reduced GLMs using multiprocessing
        # --------------------------------------

        # Define reduced encoding model formulations
        reduced_models = {
            'whisker_encoding': [f for f in feature_names if 'whisker_stim_t' in f],
            'auditory_encoding': [f for f in feature_names if 'auditory_stim_t' in f],
            'whisker_reward_encoding': ['prev_whisker_reward'],
            'jaw_onset_encoding': [f for f in feature_names if 'jaw_onset' in f],
            'motor_encoding': [f for f in feature_names if 'dist' in f or 'vel' in f],
            'block_perf_type' : ['block_perf_type'],
            # 'whisker_move': ['whisker_vel'],
            'session_progress_encoding': ['trial_index_scaled'],
            # 'last_rewards_whisker': ['last_whisker_reward'],
            # 'whisker_hit' : ['whisker_hit'],
            # 'prop_rewards_whiskers': ['prop_past_whisker_rewarded'],
            # 'prop_last_5_whisker': ['whisker_reward_rate_5'],
            # 'cum_rewards_whisker': ['sum_whisker_reward_scaled'],
            # 'all_whisker_progression': ['prev_whisker_reward', 'last_whisker_reward', 'prop_past_whisker_rewarded',
                                        # 'whisker_reward_rate_5', 'sum_whisker_reward_scaled'],
            'sum_rewards': ['sum_reward_scaled']
        }
        # Get full model params for fair comparison
        full_optimal_lambdas = {neuron_id: lambda_opt for neuron_id, lambda_opt in
                                zip(model_res_df['neuron_id'], model_res_df['lambda_opt'])}

        # Iterate over reduced model formulations
        results_reduced_all = []
        for model_name, features_to_remove in reduced_models.items():
            # Select subset of feature matrix
            X_trainval_reduced, kept_features = get_reduced_matrix(X_trainval, feature_names, features_to_remove)
            X_test_reduced, kept_features = get_reduced_matrix(X_test, feature_names, features_to_remove)


            # Fit GLM with reduced feature set
            results_reduced = parallel_fit_glms(spikes_trainval=spikes_trainval,
                                                X_trainval=X_trainval_reduced,
                                                spikes_test=spikes_test,
                                                X_test=X_test_reduced,
                                                lambdas=full_optimal_lambdas,
                                                n_jobs=n_jobs)
            results_reduced_df = pd.DataFrame(results_reduced)
            results_reduced_df['fold'] = fold_idx
            results_reduced_df['train_trials'] = [trainval_ids] * len(results_reduced_df)
            results_reduced_df['test_trials'] = [test_ids] * len(results_reduced_df)
            results_reduced_df['model_name'] = model_name
            results_reduced_df['predictors'] = [list(kept_features)] * len(results_reduced_df)
            results_reduced_all.append(results_reduced_df)

            # Append reduced model to all models
            model_res_df_outer.append(results_reduced_df)

        if whisker_kernels:
            for number_of_whisker_kernel in range(len(all_Xs)):
                # Get data splits
                X_trainval, X_test = all_Xs[number_of_whisker_kernel][:, trainval_ids, :],  all_Xs[number_of_whisker_kernel][:, test_ids, :]
                X_trainval = X_trainval.reshape(X_trainval.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)

                X_trainval = X_trainval.reshape(X_trainval.shape[0], len(trainval_ids), -1)  # reshape
                X_test = X_test.reshape(X_test.shape[0], len(test_ids), -1)

                # Fit GLM with reduced feature set
                results_reduced = parallel_fit_glms(spikes_trainval=spikes_trainval,
                                                    X_trainval=X_trainval,
                                                    spikes_test=spikes_test,
                                                    X_test=X_test,
                                                    lambdas=full_optimal_lambdas,
                                                    n_jobs=n_jobs)
                results_reduced_df = pd.DataFrame(results_reduced)
                results_reduced_df['fold'] = fold_idx
                results_reduced_df['train_trials'] = [trainval_ids] * len(results_reduced_df)
                results_reduced_df['test_trials'] = [test_ids] * len(results_reduced_df)
                results_reduced_df['model_name'] = str(nb_whisker_kernels[number_of_whisker_kernel]) + 'whisker_kernels'
                results_reduced_df['predictors'] = [list(feature_namess[number_of_whisker_kernel])] * len(results_reduced_df)
                results_reduced_all.append(results_reduced_df)

                # Append reduced model to all models
                model_res_df_outer.append(results_reduced_df)

        if reward_kernels:

            for number_of_whisker_kernel in range(len(X_rewards)):
                # Get data splits
                X_trainval, X_test = X_rewards[number_of_whisker_kernel][:, trainval_ids, :], X_rewards[number_of_whisker_kernel][
                                                                                           :, test_ids, :]
                X_trainval = X_trainval.reshape(X_trainval.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)

                X_trainval = X_trainval.reshape(X_trainval.shape[0], len(trainval_ids), -1)  # reshape
                X_test = X_test.reshape(X_test.shape[0], len(test_ids), -1)

                # Fit GLM with reduced feature set
                results_reduced = parallel_fit_glms(spikes_trainval=spikes_trainval,
                                                    X_trainval=X_trainval,
                                                    spikes_test=spikes_test,
                                                    X_test=X_test,
                                                    lambdas=full_optimal_lambdas,
                                                    n_jobs=n_jobs)

                results_reduced_df = pd.DataFrame(results_reduced)
                results_reduced_df['fold'] = fold_idx
                results_reduced_df['train_trials'] = [trainval_ids] * len(results_reduced_df)
                results_reduced_df['test_trials'] = [test_ids] * len(results_reduced_df)
                results_reduced_df['model_name'] =  str(nb_whisker_kernel_rewards[number_of_whisker_kernel]) + 'whisker_kernels_2_rewards'
                results_reduced_df['predictors'] = [list(feature_names_rewards[number_of_whisker_kernel])] * len(
                    results_reduced_df)
                results_reduced_all.append(results_reduced_df)

                # Append reduced model to all models
                model_res_df_outer.append(results_reduced_df)

        # Merge results from all reduced models, then save
        results_reduced_all_df = pd.concat(results_reduced_all, ignore_index=True)
        save_model_results(results_reduced_all_df, filename='model_reduced_fold{}'.format(fold_idx), commit_hash = commit_hash,
                           output_dir=mouse_output_path)

        model_res_df_outer.append(results_reduced_all_df)

        debug=False
        if debug:
            for neuron_id in model_res_df.neuron_id.unique():
                plot_trial_grid_predictions(results_reduced_all_df[results_reduced_all_df.model_name=='motor_encoding'], trials_df, neuron_id, bin_size=BIN_SIZE)
                plot_trial_grid_predictions(results_reduced_all_df[results_reduced_all_df.model_name=='motor_encoding'], trials_df, neuron_id, bin_size=BIN_SIZE)
                plot_trial_grid_predictions(results_reduced_all_df[results_reduced_all_df.model_name=='whisker_encoding'], trials_df, neuron_id, bin_size=BIN_SIZE)


    # Save all model results in a single file
    model_res_df_outer_df = pd.concat(model_res_df_outer, ignore_index=True)
    save_model_results(model_res_df_outer_df, filename='model_results_all',commit_hash = commit_hash, output_dir=output_dir)

    # Inspect predictions for full models across folds
    neuron_id = 0
    model_res_df_outer_df_sub = model_res_df_outer_df[model_res_df_outer_df['neuron_id'] == neuron_id]
    model_res_df_outer_df_sub = model_res_df_outer_df_sub[model_res_df_outer_df_sub['model_name'] == 'full']
    describe_cols = ['neuron_id', 'fold', 'train_score', 'test_score', 'train_corr', 'test_corr',
                     'train_mi', 'test_mi']
    print(model_res_df_outer_df_sub[describe_cols].describe())

    print('Fitting done.')
    return



def plot_trial_grid_predictions(results_df, trial_table, neuron_id, bin_size):
    """
    Plot predictions for a single neuron across trials in a grid format.
    :param results_df: DataFrame with model results
    :param trial_table: DataFrame with trial information
    :param neuron_id: int, ID of the neuron to plot
    :param bin_size: float, size of time bin in seconds
    """

    # Plotting params
    n_rows, n_cols = 5, 5
    trials_to_plot = min(n_rows*n_cols, len(trial_table))

    # Get neuron results
    results_df_sub = results_df[results_df['neuron_id'] == neuron_id]
    y_test = results_df_sub['y_test'].values[0]
    y_pred = results_df_sub['y_pred'].values[0]
    n_bins = results_df_sub['n_bins'].values[0]
    y_test = np.array(ast.literal_eval(y_test))
    y_pred = np.array(ast.literal_eval(y_pred))

    # Format data into (n_trials, n_bins)
    n_trials = y_pred.shape[0] // n_bins
    y_test = y_test.reshape(n_trials, n_bins)
    y_pred = y_pred.reshape(n_trials, n_bins)

    # Order test trial temporally
    test_trial_ids =  np.array(ast.literal_eval(results_df_sub['test_trials'].values[0]))
    test_trial_id_order =  np.argsort(test_trial_ids)
    y_test = y_test[test_trial_id_order,:]
    y_pred = y_pred[test_trial_id_order,:]

    trials_test_df = trial_table[trial_table['trial_id'].isin(test_trial_ids)]
    trials_test_df = trials_test_df.sort_values(by='trial_id', ascending=True)
    trials_test_df = trials_test_df.reset_index(drop=True)
    trials_test_df = trials_test_df.iloc[:trials_to_plot]

    # Create figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 12), sharex=False, sharey=False)
    axs = axs.flatten()

    time = np.arange(n_bins) * bin_size
    time_stim = 3
    xticklabels = [-3, -2, -1, 0, 1, 2, 3, 4, 5]
    xticks = np.linspace(0, max(time), len(xticklabels))

    for idx, row in trials_test_df.iterrows():
        ax = axs[idx]
        ax.set_title('Trial {}'.format(row['trial_id']), fontsize=10)
        putils.remove_top_right_frame(ax)
        ax.set_ylim(0, 10)
        ax.set_ylabel('Spikes', fontsize=10)
        ax.set_yticks([0, 10])
        ax.set_yticklabels([0, 10], fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=10)

        trial_type = row['trial_type']
        if trial_type == 'whisker_trial':
            ax.axvline(time_stim, color='forestgreen', linestyle='-', linewidth=1)
        elif trial_type == 'auditory_trial':
            ax.axvline(time_stim, color='mediumblue', linestyle='-', linewidth=1)
        elif trial_type == 'no_stim_trial':
            ax.axvline(time_stim, color='k', linestyle='-', linewidth=1)

        # Plot target and predictions
        ax.plot(time, y_pred[idx], color='red', linewidth=1.5)
        ax.step(time, y_test[idx], where='mid', color='black', alpha=0.9, linewidth=1.5)


    title = (f'GLM predictions on test trials - unit {neuron_id}, '
             f'$pR^2$= {results_df_sub["test_score"].values[0]:.2f}')
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.align_ylabels()
    plt.show()

    return


def get_git_revision_short_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nwb', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--n_jobs', type=int, default=20)
    args = parser.parse_args()
    run_unit_glm_pipeline_with_pool(args.nwb, args.out, n_jobs=args.n_jobs)

    #experimenter = 'Axel_Bisi'
    #output_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'results')
    #root_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'NWBFull')

    #mouse_ids = ['AB131']

    #for mouse_id in mouse_ids:
        #nwb_file_list = [f for f in os.listdir(root_path) if mouse_id in f ]
        #nwb_files_to_process = []
        # Find file with neural recordings
        #if nwb_file_list:
            #for nwb_file in nwb_file_list:
                #nwb_path = os.path.join(root_path, nwb_file)
                # Check if NWB content has units
                #units = nwbreader.get_unit_table(nwb_path)
                #if units is not None:
#                    nwb_files_to_process.append(nwb_path)

#        for nwb_file in nwb_files_to_process:
#            nwb_file = pathlib.Path(nwb_file)
#            session_id = nwb_file.stem
#            mouse_output_path = pathlib.Path(output_path, mouse_id, 'whisker_0', 'unit_glm')
#            mouse_output_path.mkdir(parents=True, exist_ok=True)

#            run_unit_glm_pipeline_with_pool(nwb_file, mouse_output_path, n_jobs=10)

