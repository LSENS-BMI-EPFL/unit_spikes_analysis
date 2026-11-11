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
import shutil
import math
from sklearn.exceptions import ConvergenceWarning
import warnings
from itertools import combinations
from joblib import Parallel, delayed


sys.path.append('/home/mhamon/Github/NWB_reader')

import multiprocessing as mp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyglmnet import GLM, GLMCV
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.use('Agg')

# Custom imports
# allen_utils: always the ephys_utilities copy, loaded from its file under its own module name.
# A plain `import allen_utils` returns whichever allen_utils was imported first in the process
# (the standalone allen_utils repo has drifted: other area groupings, other argument names).
import importlib.util as _importlib_util
def _load_ephys_allen_utils():
    if 'ephys_allen_utils' in sys.modules:
        return sys.modules['ephys_allen_utils']
    path = pathlib.Path(__file__).resolve().parent.parent / 'ephys_utilities' / 'ephys_utilities' / 'allen_utils' / 'allen_utils.py'
    spec = _importlib_util.spec_from_file_location('ephys_allen_utils', path)
    module = _importlib_util.module_from_spec(spec)
    sys.modules['ephys_allen_utils'] = module
    spec.loader.exec_module(module)
    # Also serve it for every later `import allen_utils` (e.g. NWB_reader_functions)
    previous = sys.modules.get('allen_utils')
    if previous is not None and previous is not module:
        print(f"[WARN] another allen_utils was already imported ({getattr(previous, '__file__', None)}); "
              f"replacing it with the ephys_utilities copy")
    sys.modules['allen_utils'] = module
    return module
allen_utils = _load_ephys_allen_utils()
import NWB_reader_functions as nwbreader
import plotting_utils as putils
import drift_loading_utils as driftutils

# Set global variables
BIN_SIZE = 0.05 # in seconds

ROOT_PATH = '/work/lsens/Myriam_Hamon/glm_inputs/'  # jaw onset, learning curves, motion (was /scratch/mhamon/results_ks4/)
# ROOT_PATH = os.path.join(r'C:\Users\mhamon/')

# ---------------------------------------------------------------------------
# GLM run configuration: every switch of run_unit_glm_pipeline_with_pool lives here.
# A copy (plus git hash, inputs and design-matrix checks) is saved next to the results
# as models/<hash>_run_config.json. Override per run with
# run_unit_glm_pipeline_with_pool(..., config={'key': value}).
# ---------------------------------------------------------------------------
GLM_CONFIG = {
    # Per-unit, per-trial probe drift as an extra predictor (+ a 'no_drift' ablation), from this
    # session's motion: ROOT_PATH/<mouse>/<session_id>/imec<N>/motion/motion (see get_unit_drift_matrix).
    'compute_drift': True,
    # Temporal split of event kernels into `number_of_whisker_kernel` kernels
    'whisker_kernels': True,
    'number_of_whisker_kernel': 1,
    # Design-matrix variants: which event types get split ('full' drives the main model)
    'kernel_split_configs': {
        'full': ['whisker_hits', 'whisker_misses', 'auditory_stim'],
        # 'whisker_hits_split': ['whisker_hits'],
        # 'whisker_misses_split': ['whisker_misses'],
        # 'auditory_split': ['auditory_stim'],
    },
    # Control draws for the split: events reassigned to kernels at random / by circular shift
    'random_split': False,
    'linear_shift_split': False,
    'n_control_draws': 5,
    # One reward kernel per stimulus type (added model)
    'reward_kernels': False,
    # One 'whisker_stim' kernel for all whisker stimuli instead of separate whisker hit / miss kernels
    'merge_whisker': True,
    # Reward event: 'lick_time' = the trial's first response lick (valve opens within a few ms),
    # 'piezo' = previous behaviour (second piezo_lick_times event in the trial window)
    'reward_align': 'lick_time',
    # Lick regressor from the continuous piezo trace (lick_trace): per 50 ms bin, mean of the signal
    # above the session's noise ceiling, with lagged copies over lick_lags_s (start, end) like an event
    # kernel; z-scored with the DLC features (train folds only)
    'lick_regressor': False,
    # (0, 0.05) = a single same-bin column, like the DLC traces (no lags)
    'lick_lags_s': (0, 0.05),
    # Per-lick kernel on piezo_lick_times (every lick an event), window lick_kernel_window_s:
    #  'fine'       : n raised-cosine bases defined at 1 ms, averaged into each 50 ms bin (keeps where the lick
    #                 falls within the bin)
    #  'binned'     : each lick marked in its 50 ms bin, one column per lag (like the other event kernels)
    #  'fine_split' : two fine kernels, licks during reward collection (lick_time to +2 s) vs all other licks
    #  'fine_bout'  : fine kernel + lick-bout state (fraction of the bin inside a bout: first to last lick)
    #  None         : no lick kernel
    'lick_kernel': 'fine',
    'lick_kernel_window_s': (-0.05, 0.1),
    'lick_kernel_n_bases': 4,
    'lick_bout_gap_s': 0.35,  # licks closer than this belong to the same bout (inter-lick interval ~130 ms)
    # Reward kernel lags (s from the first lick); trial windows end 2 s after the stimulus, i.e. ~1.5-1.7 s
    # after the first lick
    'reward_window_s': (0, 1.6),
    # Reward kernel parameterisation in all models: 'cosine' = reward_n_cosine raised-cosine bases over the
    # reward lags (piezo_reward_cos<k>), None = one free weight per 50 ms lag (piezo_reward_t<lag>s)
    'reward_basis': 'cosine',
    # Pseudo-trials from inter-trial lick bouts (unrewarded "dry" licking, no stimulus), aligned to the bout's
    # first lick with the usual -1/+2 s window: bouts of >= iti_bout_min_licks piezo licks (gaps < lick_bout_gap_s),
    # no lick in the iti_bout_quiet_s before, window clear of every trial window. Not added when the session
    # has fewer than iti_bout_min_n such bouts.
    'iti_bout_trials': False,
    'iti_bout_min_licks': 5,
    'iti_bout_quiet_s': 1.0,
    'iti_bout_min_n': 20,
    # Other lick kernels fitted next to 'full' (model_added, full-model lambdas), named lick_<kind>
    'lick_kernel_variants': ['none'],
    # Extra models with other reward / lick designs, fitted next to 'full' with its lambdas (model_added):
    # reward_piezo_no_lick, reward_licktime_no_lick, reward_constant, reward_cosine (with reward_basis None),
    # reward_bins (free lags, with reward_basis 'cosine')
    'reward_variants': True,
    'reward_variant_names': [],  # e.g. 'reward_piezo_no_lick', 'reward_licktime_no_lick', 'reward_constant', 'reward_cosine', 'reward_bins'
    'reward_n_cosine': 6,
    # Extra model 'jaw_onset_cosine' (model_added): full design with the jaw-onset kernel (10 lags, -0.5 to
    # -0.05 s) replaced by jaw_onset_n_cosine raised-cosine bases. None = not fitted
    'jaw_onset_n_cosine': None,
    # Whisker history: 'prev_whisker_hit' baseline column (whole trial, all trial types, 1 after a
    # whisker hit) + 'whisker_stim_prevhit' kernel (whisker stimulus on trials after a whisker hit, on top
    # of the usual whisker kernels). Adds the 'whisker_history' reduced model (both removed) and
    # 'prev_whisker_shuffle_draws' controls with the previous-outcome labels shuffled across trials.
    'prev_whisker_history': False,
    'prev_whisker_shuffle_draws': 5,
    # Trial-history predictors, each fit as its own added model (None to skip)
    'add_perf_pred': None,
    # Reduced (ablation) models to fit, see REDUCED_MODEL_DEFINITIONS in the pipeline
    'reduced_models': ['whisker_encoding', 'auditory_encoding', 'reward_encoding', 'jaw_onset_encoding',
                       'motor_encoding', 'session_progress_encoding', 'sum_rewards',
                       'whisker_first_bin', 'auditory_first_bin', 'lick_encoding', 'licking_encoding'],
    # Outer cross-validation
    'cv_outer_folds': 5,
    'cv_random_state': 42,
    # Ridge penalty grid for the full model: logspace from lambda_max to lambda_min
    'lambda_max': 10,
    'lambda_min': 1e-5,
    'n_lambdas': 10,
    # Cross-validated predictions of the best-fitted neurons, saved to models/<hash>_top_predictions.pkl
    # (each trial predicted by the fold in which it was held out). Ranked by mean test R2 of 'full'.
    'save_top_predictions': True,
    'top_predictions_fraction': 0.05,
    'top_predictions_min_rate_hz': 1.0,  # neurons below this firing rate are not ranked
    'top_predictions_models': ['full', 'whisker_encoding'],  # models not fitted in this run are skipped
    # After fitting, build the per-mouse summaries (glms_plotting_utils.mouse_glm_results 'create_summary')
    # into <mouse>/whisker_0/unit_glm/<hash>/<unit filter>/. A failure here does not fail the run.
    'create_summary': True,
    'summary_info_path': '/work/lsens/Myriam_Hamon/glm_inputs/mice_info',  # joint_mouse_reference_weight.xlsx
}

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


def build_design_matrix(data_dict, event_defs, analog_keys, bin_size, scale = None, reward_basis=None, reward_n_cosine=6):
    """
    Builds a design matrix X for GLM using input data dict and event definitions from
    `load_nwb_spikes_and_predictors`.
    :param data_dict: dict of predictor arrays, each shape (n_trials, n_bins)
    :param event_defs: dict of {name: (event_times, (start_offset, end_offset))} for event-based predictors
    :param analog_keys: dict of analog predictor names to their keys in the predictors dict
    :param bin_size: duration of one time bin (s)
    :param reward_basis: None, or 'cosine' / 'constant' to replace the piezo_reward lag columns (reward_basis_transform)
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
    if reward_basis is not None:
        X, feature_names = reward_basis_transform(X, feature_names, reward_basis, reward_n_cosine)

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

    #path_to_data = r'M:\analysis\Axel_Bisi\combined_results'
    # curves_df = load_helpers.load_learning_curves_data(path_to_data=path_to_data, subject_ids=subject_ids)

    file_name = f'{mouse_id}_whisker_0_whisker_trial_learning_curve_interp.h5'
    path_to_file = os.path.join(ROOT_PATH, mouse_id, file_name)
    #path_to_file = os.path.join(path_to_data, mouse_id,  'whisker_0', 'learning_curve',file_name)

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
    file_path = os.path.join(ROOT_PATH, mouse_id, 'dlc_jaw_onset_times.pkl')
    if os.path.exists(file_path):
        df = pd.read_pickle(file_path)
        jaw_onset_list.append(df)
    else:
        print(f"[WARN] Jaw onset data file not found for {mouse_id} at {file_path}. Skipping.")
        return
    jaw_onset_table = pd.concat(jaw_onset_list, ignore_index=True)
    return jaw_onset_table

# Details of the last get_unit_drift_matrix call (motion files used / skipped), saved in the run config
LAST_DRIFT_INFO = {}


def get_unit_drift_matrix(unit_table, trial_times, mouse_id, session_id, motion_data_root, max_end_mismatch_s=60):
    """
    Compute per-unit, per-trial probe motion/drift (um) at each unit's own depth and probe, using this
    session's own motion estimate: `{motion_data_root}/{mouse_id}/{session_id}/imec{imec_id}/motion/motion`.
    There is deliberately no fallback to a per-mouse folder: with several sessions per mouse that held
    another recording's motion.

    A probe's motion is skipped (its units get no drift) if the motion trace ends before the recording
    does (last spike of the units on that probe, beyond max_end_mismatch_s): a trace from another
    recording would otherwise be used silently. A trace ending after the last spike is used (the
    recording can run on after the session) but flagged in LAST_DRIFT_INFO.

    :param unit_table: pd.DataFrame with 'electrode_group', 'depth' and 'spike_times' columns, one row per
        neuron, in the same order as the spike_array neuron axis.
    :param trial_times: array-like, shape (n_trials,), trial times in seconds (same clock as the
        motion object's temporal_bins_s, e.g. trial start_time).
    :param mouse_id: str
    :param session_id: str, NWB session id (= session folder name under the mouse folder)
    :param motion_data_root: str or Path
    :param max_end_mismatch_s: float, tolerated difference between motion end and last spike (s)
    :return: np.ndarray, shape (n_neurons, n_trials), NaN for units without usable motion, or None if no
        unit has usable motion.
    """
    global LAST_DRIFT_INFO
    imec_ids = [driftutils._get_imec_id(eg.name) for eg in unit_table['electrode_group'].values]
    depths = unit_table['depth'].astype(float).values
    trial_times = np.asarray(trial_times, dtype=float)

    n_neurons = len(unit_table)
    drift_matrix = np.full((n_neurons, len(trial_times)), np.nan)
    probes_info = {}

    motion_cache = {}
    for imec_id in sorted(set(i for i in imec_ids if i is not None)):
        motion_path = pathlib.Path(motion_data_root, mouse_id, session_id, f'imec{imec_id}', 'motion', 'motion')
        if not motion_path.exists():
            print(f"[WARN] No motion data for {session_id} imec{imec_id} at {motion_path}. "
                  f"Units on this probe get no drift.")
            probes_info[f'imec{imec_id}'] = {'status': 'missing', 'path': str(motion_path)}
            continue
        motion = driftutils.load_motion(motion_path)
        motion_end = float(np.asarray(motion.temporal_bins_s[0])[-1])
        on_probe = [i for i, x in enumerate(imec_ids) if x == imec_id]
        last_spike = max((float(np.max(st)) for st in unit_table['spike_times'].values[on_probe] if len(st)),
                         default=np.nan)
        info = {'path': str(motion_path), 'motion_end_s': round(motion_end, 1), 'last_spike_s': round(last_spike, 1)}
        if not np.isfinite(last_spike) or motion_end < last_spike - max_end_mismatch_s:
            print(f"[WARN] Motion for {session_id} imec{imec_id} ends at {motion_end:.0f} s, before the recording "
                  f"(last spike {last_spike:.0f} s): probably another recording, units on this probe get no drift.")
            probes_info[f'imec{imec_id}'] = {**info, 'status': 'skipped: motion ends before recording'}
            continue
        if motion_end > last_spike + max_end_mismatch_s:
            print(f"[INFO] Motion for {session_id} imec{imec_id} runs {motion_end - last_spike:.0f} s past the "
                  f"last spike (recording stopped later); used.")
            probes_info[f'imec{imec_id}'] = {**info, 'status': 'used (motion longer than last spike)'}
        else:
            probes_info[f'imec{imec_id}'] = {**info, 'status': 'used'}
        motion_cache[imec_id] = motion

    for i in range(n_neurons):
        imec_id = imec_ids[i]
        if imec_id is None or imec_id not in motion_cache:
            continue
        drift_matrix[i, :] = driftutils.get_motion_at_times_and_depth(motion_cache[imec_id], trial_times, depths[i])

    n_with_drift = int((~np.isnan(drift_matrix).all(axis=1)).sum())
    LAST_DRIFT_INFO = {'session_id': session_id, 'probes': probes_info,
                       'n_units_with_drift': n_with_drift, 'n_units': n_neurons}
    if n_with_drift == 0:
        return None
    return drift_matrix


def expand_scalar_predictor_to_time_kernel(values, n_bins, prefix):
    """
    values: shape (n_trials,) scalar predictor
    returns: dict of {f"{prefix}_bin{i}": (n_trials, n_bins)}
    """
    out = {}
    for b in range(n_bins):
        mat = np.zeros((len(values), n_bins), dtype=np.float32)
        mat[:, b] = values  # inject scalar per trial into one bin
        out[f"{prefix}_bin{b}"] = mat
    return out


def load_nwb_spikes_and_predictors(nwb_path, bin_size=0.1, nb_of_whisker_kernel= None, kernels_to_split=('whisker_hits', 'whisker_misses', 'auditory_stim'), split_mode='sequential', split_seed=0, reward_kernel_per_type = False, include_predictors=[], compute_drift=False, motion_data_root=None,
                                   prev_whisker_history=False, prev_whisker_shuffle_seed=None, merge_whisker=False,
                                   reward_align='piezo', lick_regressor=False, lick_lags_s=(-0.1, 0.25),
                                   lick_kernel=None, lick_kernel_window_s=(-0.05, 0.1), lick_kernel_n_bases=4,
                                   lick_bout_gap_s=0.35, reward_window_s=(0, 1), iti_bouts=None):
    """
    Loads spike trains from unit table and predictors from an NWB file.
    :param nwb_path: str, path to the NWB file
    :param bin_size: float, size of time bin in seconds
    :param nb_of_whisker_kernel: int or None, if not None, split each event type named in
        `kernels_to_split` into this many temporal kernels; event types not in
        `kernels_to_split` keep a single un-split kernel
    :param kernels_to_split: iterable of {'whisker_hits', 'whisker_misses', 'auditory_stim'},
        which event types to split into `nb_of_whisker_kernel` kernels (only used when
        `nb_of_whisker_kernel` is not None); pass a subset to split only some of them
    :param split_mode: 'sequential' (temporal blocks, in session order), 'random' (shuffled
        before splitting into blocks), or 'linear_shift' (temporal blocks circularly rolled
        by an amount set by `split_seed`, a deterministic alternative to 'random' for control draws)
    :param split_seed: int, seed/shift index used by 'random' and 'linear_shift' split modes
    :param compute_drift: bool, if True also compute a per-unit, per-trial probe drift matrix
    :param motion_data_root: str or Path, root containing '{mouse_id}/imec{imec_id}/motion/motion',
        required when compute_drift=True
    :return: tuple of (spike_array, predictors, predictor_types, n_bins, bin_size) where
    - spike_array: np.ndarray, shape (n_neurons, n_trials, n_bins)
    - predictors: dict, contains predictor arrays
    - predictor_types: dict, contains types of predictors
    - n_bins: int, number of bins per trial
    - bin_size: float, size of time bin in seconds
    - nb_of_whisker_kernel : int, number of whisker kernels
    - drift_matrix: np.ndarray, shape (n_neurons, n_trials), or None if compute_drift=False or
      unavailable for this session
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

        jaw_onset_table = load_jaw_onset_data(trials_df['mouse_id'].unique()[0] )
        # The jaw onset pickle can list a trial more than once; keep one row per trial,
        # otherwise the merge below duplicates trials.
        jaw_onset_table = jaw_onset_table.drop_duplicates(['mouse_id', 'session_id', 'trial_id'])

        trials_df = trials_df.merge(jaw_onset_table[['mouse_id', 'session_id', 'trial_id', 'jaw_dlc_onset', 'piezo_lick_time']],
                                on=['mouse_id', 'session_id', 'trial_id'], how='left')

        trials_df = load_perf_blocks(trials_df, trials_df['mouse_id'].unique()[0])
        if iti_bouts is not None:
            trials_df = add_iti_bout_trials(trials_df, nwbfile.trials.to_dataframe(), load_piezo_lick_times(nwb_path), **iti_bouts)
    #    trials_df = trials_df[trials_df['trial_type'] == 'whisker_trial']
        trial_starts = trials_df['start_time'].values + window_bounds_sec[0]
        trial_ends = trials_df['start_time'].values + window_bounds_sec[1]
        
        #
        n_trials = len(trial_starts)
        #
        win_start, win_end = window_bounds_sec
        n_bins = int(np.round((win_end - win_start) / bin_size))

        unit_table = nwbfile.units.to_dataframe()
        # unit_table = unit_table.sample(frac=1) why is there a random shuffle ????
        # unit_table = unit_table[unit_table['bc_label']=='good']
        # unit_table = unit_table[unit_table['ccf_parent_acronym'].isin(['SSp-bfd', 'SSs'])]

        # unit_table = unit_table[unit_table['firing_rate'].astype(float).ge(2.0)]
        # unit_table = unit_table[~unit_table['ccf_acronym'].isin(allen_utils.get_excluded_areas())]
        # unit_table = unit_table.sample(n=2, random_state=None)

        # Use index as new column named "unit_id", then reset
        unit_table['neuron_id'] = unit_table.index
        unit_table.reset_index(drop=True, inplace=True)
        neurons_ccf = unit_table['ccf_parent_acronym'].values
    #    unit_table = unit_table.iloc[[21, 34]]

        drift_matrix = None
        if compute_drift:
            drift_matrix = get_unit_drift_matrix(
                unit_table, trials_df['start_time'].values, trials_df['mouse_id'].unique()[0],
                trials_df['session_id'].unique()[0], motion_data_root)
            if drift_matrix is None:
                print(f"[WARN] No motion/drift data available for {trials_df['mouse_id'].unique()[0]}. "
                      f"Drift model will be skipped for this session.")



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
        lick = trials_df['lick_flag'].astype(int).values
        prev_whisker_reward = np.zeros(n_trials)

        # Running memory for last reward status
        last_whisker_reward = 0

        for i in range(n_trials):
            # Store the most recent reward info
            prev_whisker_reward[i] = last_whisker_reward
            if stim_type[i] == 'whisker_trial':
                last_whisker_reward = lick[i]
        # broadcast to bins
        if 'last_whisker_reward' in include_predictors :
            predictors.update(
                expand_scalar_predictor_to_time_kernel(prev_whisker_reward, n_bins, 'last_whisker_reward')
            )

        # Running memory for last no stim status
        last_no_stim = 0
        prev_no_stim = np.zeros(n_trials)
        licks =  trials_df['lick_flag'].astype(int).fillna(0).values
        for i in range(n_trials):
            # Store the most recent reward info
            prev_no_stim[i] = last_no_stim
            if stim_type[i] != 'whisker_trial' and stim_type[i] != 'auditory_trial':
                last_no_stim = licks[i]
        # broadcast to bins
        if 'last_false_alarm' in include_predictors:
            predictors.update(
                expand_scalar_predictor_to_time_kernel(prev_no_stim, n_bins, 'last_false_alarm')
            )



        # Compute trial outcome (success vs error)
        stim_type = trials_df['trial_type'].fillna('').values
        licks = trials_df['lick_flag'].astype(int).fillna(0).values
        reward_available = trials_df['reward_available'].astype(int).fillna(0).values

        success = np.zeros(n_trials)

        for i in range(n_trials):
            if stim_type[i] in ['whisker_trial', 'auditory_trial']:
                # GO trial → success if rewarded lick
                success[i] = 1 if (reward_available[i] == 1 and licks[i] == 1) else 0
            else:
                # NO-GO trial → success if no lick
                success[i] = 1 if licks[i] == 0 else 0

        # Shift by one trial to get previous trial's success/error
        prev_success = np.zeros(n_trials)
        prev_success[1:] = success[:-1]

        # Broadcast to bins if selected
        if 'prev_success' in include_predictors:
            predictors.update(
                expand_scalar_predictor_to_time_kernel(prev_success, n_bins, 'prev_success')
            )


        # get if last trial was rewarded regardless of trial type
        last_reward = 0
        prev_reward = np.zeros(n_trials)

        for i in range(n_trials):
            # Store the most recent reward info
            prev_reward[i] = last_reward
            last_reward = lick[i]

        if 'last_reward' in include_predictors :
            predictors.update(
                expand_scalar_predictor_to_time_kernel(prev_reward, n_bins, 'last_reward')
            )

        # get the proportion of past whisker trials that were rewarded
        past_whisker_trials = 0
        past_whisker_rewards = 0
        prop_past_whisker_rewarded = np.zeros(n_trials)

        for i in range(1, n_trials):
            # Whisker trial history
            if stim_type[i - 1] == 'whisker_trial':
                past_whisker_trials += 1
                if lick[i - 1] > 0:
                    past_whisker_rewards += 1

            # Avoid divide-by-zero
            if past_whisker_trials > 0:
                prop_past_whisker_rewarded[i] = past_whisker_rewards / past_whisker_trials

        if 'prop_past_whisker_rewarded' in include_predictors :
            predictors.update(
                expand_scalar_predictor_to_time_kernel(prop_past_whisker_rewarded, n_bins, 'prop_past_whisker_rewarded')
            )
        block_perf_type = trials_df['block_perf_type'].to_numpy()  # shape (n_trials,)

        if 'block_perf_type' in include_predictors :
            predictors.update(
                expand_scalar_predictor_to_time_kernel(block_perf_type, n_bins, 'block_perf_type')
            )
        # Rolling reward proportion
        whisker_reward_rate = np.zeros(n_trials)
        for i in range(n_trials):
            start_idx = max(0, i - 5)
            recent_trials = stim_type[start_idx:i]
            recent_rewards = lick[start_idx:i]
            whisker_mask = recent_trials == 'whisker_trial'
            if np.sum(whisker_mask) > 0:
                whisker_reward_rate[i] = np.sum(recent_rewards[whisker_mask]) / np.sum(whisker_mask)
            else:
                whisker_reward_rate[i] = 0

        scale = np.tile(whisker_reward_rate[:, None], (1, n_bins))

        if 'whisker_reward_rate_5' in include_predictors :
            predictors.update(
                expand_scalar_predictor_to_time_kernel(whisker_reward_rate, n_bins, 'whisker_reward_rate_5')
            )

        total_rewards = np.sum(rewarded > 0)
        total_rewards = total_rewards if total_rewards > 0 else 1
        cum_reward = np.zeros(n_trials)
        reward_so_far = 0
        for i in range(n_trials):
            cum_reward[i] = reward_so_far / total_rewards
            if lick[i] > 0:
                reward_so_far += 1

        # Add to predictors (scaled cumulative rewards across all trial types)
        predictors['sum_reward_scaled'] = np.tile(cum_reward[:, None], (1, n_bins))

        # Whisker history: 1 if the previous whisker trial (among the trials kept here) was a hit.
        # The first whisker trial of the session has no previous one: 0.
        prev_whisker_hit = np.zeros(n_trials)
        if prev_whisker_history:
            last_whisker_lick = None
            for i in range(n_trials):
                prev_whisker_hit[i] = 1.0 if last_whisker_lick == 1 else 0.0
                if stim_type[i] == 'whisker_trial':
                    last_whisker_lick = lick[i]
            if prev_whisker_shuffle_seed is not None:
                # Control: same number of "after a hit" trials, assigned at random
                prev_whisker_hit = np.random.default_rng(prev_whisker_shuffle_seed).permutation(prev_whisker_hit)
            # Baseline column: shift of the whole trial after a previous whisker hit (all trial types)
            predictors['prev_whisker_hit'] = np.tile(prev_whisker_hit[:, None], (1, n_bins))

        # Whisker hit predictor: 1 if whisker trial and actually rewarded, else 0
        # whisker_hit = ((stim_type == 'whisker_trial') & (rewarded > 0)).astype(int)
        # Add to predictors
        # predictors['whisker_hit'] = np.tile(whisker_hit[:, None], (1, n_bins))

        binary_keys ={
            'trial_index_scale':'trial_index_scaled',
            'offset' : 'offset',
            'sum_reward_scaled':'sum_reward_scaled',
        }
        for predictor in include_predictors:
            binary_keys[predictor] = predictor
        if prev_whisker_history:
            binary_keys['prev_whisker_hit'] = 'prev_whisker_hit'

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

        lick_trials = trials_df[trials_df['lick_flag'] == 1]
        all_jaw_onsets = lick_trials['jaw_dlc_onset'].values + lick_trials['start_time'].values

        # Get available stimulus times
        try:
            auditory_times = list(get_events('auditory_hit_trial')[1]) + list(get_events('auditory_miss_trial')[1])
        except:
            auditory_times = list(get_events('auditory_hit_trial')[1])
        auditory_times = np.array(sorted(auditory_times))

        try:
            whisker_times = list(get_events('whisker_hit_trial')[1]) + list(get_events('whisker_miss_trial')[1])
            whiskers_hits = list(get_events('whisker_hit_trial')[1])
            whiskers_misses = list(get_events('whisker_miss_trial')[1])
        except:
            whisker_times = list(get_events('whisker_hit_trial')[1])
        whisker_times = np.array(sorted(whisker_times))
        whisker_misses_times = np.array(whiskers_misses)
        whisker_hits_times = np.array(whiskers_hits)

        if nb_of_whisker_kernel is None:
            # Define event kernels for single trial events
            event_defs = {
                # 'dlc_lick_onset': (tongue_dlc_licks, (-0.3, 0.6)), # in seconds
                'jaw_onset' : (all_jaw_onsets, (-0.5, 0)),
                'auditory_stim': (auditory_times, (-0.1, 0.5)),
                'whisker_hits': (whisker_hits_times, (-0.1, 0.5)),
                'whisker_misses': (whisker_misses_times, (-0.1,0.5)),
                'piezo_reward': (piezo_licks, tuple(reward_window_s)),
            }

        else :
            event_defs = {
                # 'dlc_lick_onset': (tongue_dlc_licks, (-0.3, 0.6)), # in seconds
                'jaw_onset' : (all_jaw_onsets, (-0.5, 0)),
                'piezo_reward': (piezo_licks, tuple(reward_window_s)),

            }

            def _sequential_split(arr, n_splits):
                # Split by session time into n_splits equal-duration windows spanning the
                # whole session (trial_starts[0] to trial_ends[-1]), not by equal event
                # count — otherwise a burst of events concentrated early or late in the
                # session would shift the boundary away from the actual session midpoint.
                arr = np.sort(np.asarray(arr))
                if len(arr) == 0:
                    return [arr.copy() for _ in range(n_splits)]
                time_edges = np.linspace(trial_starts[0], trial_ends[-1], n_splits + 1)
                bin_idx = np.clip(np.digitize(arr, time_edges) - 1, 0, n_splits - 1)
                return [arr[bin_idx == k] for k in range(n_splits)]

            def _random_split(arr, n_splits, seed):
                arr = np.asarray(arr)
                rng = np.random.default_rng(seed=seed)
                perm = rng.permutation(len(arr))
                return np.array_split(arr[perm], n_splits)

            def _linear_shift_split(arr, n_splits, shift_idx):
                # Deterministic control: same temporally-sorted events as the sequential
                # split, but circularly rolled before splitting into contiguous chunks.
                # The shift follows a golden-ratio (low-discrepancy) sequence rather than
                # a step of N // n_splits: a step of N // n_splits only ever swaps whole
                # kernels with each other (e.g. for n_splits=2 it alternates between the
                # real split and its exact complement, cycling back to shift=0 after just
                # `n_splits` draws), which reshuffles nothing and isn't a real control.
                # The golden-ratio step instead lands each draw at a well-spread offset
                # that (for n_splits > 1) essentially never repeats and essentially never
                # falls exactly on a kernel boundary, so each draw genuinely reassigns
                # individual events across kernels.
                arr = np.sort(np.asarray(arr))
                n = len(arr)
                if n < 2:
                    return np.array_split(arr, n_splits)
                golden_step = 0.6180339887498949  # golden ratio conjugate
                shift = int(round((shift_idx + 1) * n * golden_step)) % n
                # Guard against landing exactly on a kernel-boundary multiple (would just
                # relabel whole kernels, same degenerate case the golden step is meant to avoid).
                boundary = max(n // n_splits, 1)
                if shift % boundary == 0:
                    shift = (shift + 1) % n
                return np.array_split(np.roll(arr, shift), n_splits)

            def _maybe_split(arr, key):
                if key not in kernels_to_split:
                    return None
                if split_mode == 'random':
                    return _random_split(arr, nb_of_whisker_kernel, split_seed)
                elif split_mode == 'linear_shift':
                    return _linear_shift_split(arr, nb_of_whisker_kernel, split_seed)
                else:
                    return _sequential_split(arr, nb_of_whisker_kernel)

            whisker_hits_splits = _maybe_split(whisker_hits_times, 'whisker_hits')
            whisker_misses_splits = _maybe_split(whisker_misses_times, 'whisker_misses')
            auditory_stim_splits = _maybe_split(auditory_times, 'auditory_stim')

            if whisker_hits_splits is not None:
                for nb in range(nb_of_whisker_kernel):
                    event_defs[f'whisker_hits_stim_{nb}'] = (whisker_hits_splits[nb], (-0.1, 0.5))
            else:
                event_defs['whisker_hits'] = (whisker_hits_times, (-0.1, 0.5))

            if whisker_misses_splits is not None:
                for nb in range(nb_of_whisker_kernel):
                    event_defs[f'whisker_misses_stim{nb}'] = (whisker_misses_splits[nb], (-0.1, 0.5))
            else:
                event_defs['whisker_misses'] = (whisker_misses_times, (-0.1, 0.5))

            if auditory_stim_splits is not None:
                for nb in range(nb_of_whisker_kernel):
                    event_defs[f'auditory_stim{nb}'] = (auditory_stim_splits[nb], (-0.1, 0.5))
            else:
                event_defs['auditory_stim'] = (auditory_times, (-0.1, 0.5))

        if merge_whisker:
            # Single kernel for all whisker stimuli, regardless of outcome (replaces hit / miss kernels)
            for k in [k for k in event_defs if k.startswith(('whisker_hits', 'whisker_misses'))]:
                del event_defs[k]
            event_defs['whisker_stim'] = (whisker_times, (-0.1, 0.5))

        if prev_whisker_history:
            # Extra whisker-stimulus kernel, only on whisker trials whose previous whisker trial was a hit,
            # on top of the usual whisker kernels: it is the change in the stimulus response after a hit.
            trial_start_times = trials_df['start_time'].values
            trial_of_event = np.searchsorted(trial_start_times, whisker_times, side='right') - 1
            valid = trial_of_event >= 0
            trial_of_event = np.clip(trial_of_event, 0, n_trials - 1)
            keep = (valid
                    & (whisker_times < trial_start_times[trial_of_event] + window_bounds_sec[1])
                    & (stim_type[trial_of_event] == 'whisker_trial')
                    & (prev_whisker_hit[trial_of_event] == 1))
            event_defs['whisker_stim_prevhit'] = (whisker_times[keep], (-0.1, 0.5))

        for name, (times, _) in event_defs.items():
            if name=='dlc_lick_onset':
                predictors['dlc_lick_onset'] = rasterize_event(times, first_only=True)

            else:
                predictors[name] = rasterize_event(times, first_only=False)
        # Rasterize the rewards, only for hit trials and only auditory if non-rewarded mouse
        all_piezo_events = np.array(list(get_events('piezo_lick_times')[1]))

        if not reward_kernel_per_type and reward_align == 'lick_time':
            # Reward onset = the trial's first response lick (lick_time); the valve opens within a few ms
            piezo_reward_matrix = np.zeros((n_trials, n_bins))
            reward_times = trials_df['lick_time'].to_numpy(dtype=float)
            for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
                if rewarded[i] > 0 and np.isfinite(reward_times[i]):
                    idx = np.digitize(reward_times[i], np.linspace(start, end, n_bins + 1)) - 1
                    if 0 <= idx < n_bins:
                        piezo_reward_matrix[i, idx] = 1
            predictors['piezo_reward'] = piezo_reward_matrix

        elif not reward_kernel_per_type:
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


        if lick_kernel is not None:
            lick_preds = build_lick_kernel_predictors(
                np.sort(np.asarray(all_piezo_events, float)), trial_starts, n_bins, bin_size, lick_kernel,
                lick_kernel_window_s, lick_kernel_n_bases, lick_bout_gap_s,
                reward_times=trials_df['lick_time'].to_numpy(dtype=float)[rewarded > 0])
            predictors.update(lick_preds)

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
            assert np.isfinite(data.shape).all() # make sure there are no NaNs or infs
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

        if lick_regressor:
            # Continuous piezo trace: it only moves when the mouse licks, on top of a small mains ripple.
            # Noise ceiling = 99.5th percentile in the quiet second before each stimulus; per bin, mean of
            # the signal above it ("how much licking in this bin"), then lagged copies like an event kernel.
            lick_x, lick_t = get_series('lick_trace')
            lick_x, lick_t = np.asarray(lick_x, float), np.asarray(lick_t, float)
            q0 = np.searchsorted(lick_t, trial_starts)
            q1 = np.searchsorted(lick_t, trial_starts - window_bounds_sec[0])
            quiet = np.concatenate([lick_x[a:b] for a, b in zip(q0, q1)]) if len(q0) else lick_x
            ceiling = np.percentile(quiet, 99.5) if len(quiet) else np.median(lick_x)
            cs = np.concatenate([[0.0], np.cumsum(np.clip(lick_x - ceiling, 0, None))])
            edges = trial_starts[:, None] + bin_size * np.arange(n_bins + 1)[None, :]
            ii = np.searchsorted(lick_t, edges)
            counts = np.diff(ii, axis=1)
            lick_binned = (cs[ii[:, 1:]] - cs[ii[:, :-1]]) / np.maximum(counts, 1)
            pre = int(np.round(abs(lick_lags_s[0]) / bin_size))
            post = int(np.round(lick_lags_s[1] / bin_size))
            for shift in range(-pre, post):
                shifted = np.zeros_like(lick_binned)
                if shift < 0:
                    shifted[:, :shift] = lick_binned[:, -shift:]
                elif shift > 0:
                    shifted[:, shift:] = lick_binned[:, :-shift]
                else:
                    shifted = lick_binned.copy()
                name = f'lick_piezo_t{shift * bin_size:+.2f}s'
                predictors[name] = shifted
                analog_keys[name] = name


        # Prepare outputs
        predictor_types ={'binary_keys': binary_keys,
                         'analog_keys': analog_keys,
                         'event_defs': event_defs}

        return spike_array, predictors, predictor_types, n_bins, bin_size, neurons_ccf, scale, drift_matrix


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


def fit_neuron_glm_gaussian(neuron_id, spikes_trainval, X_trainval, spikes_test, X_test, lambdas,
                             drift_trainval_row=None, drift_test_row=None):
    """
    Fit a non-Poisson (Gaussian) GLM to the data for a single neuron using closed-form ridge regression.
    Much faster than iterative GLM fitting.

    :param neuron_id: int, ID of the neuron to fit
    :param spikes_trainval: np.ndarray, (n_neurons, n_trials, n_bins)
    :param X_trainval: np.ndarray, (n_features, n_trials, n_bins)
    :param spikes_test: np.ndarray, (n_neurons, n_trials, n_bins)
    :param X_test: np.ndarray, (n_features, n_trials, n_bins)
    :param lambdas: float or np.ndarray, regularization parameter(s) for Ridge regression
    :param drift_trainval_row: np.ndarray, (n_trials,) optional, this neuron's own per-trial drift
        value (same value repeated for every bin of a trial), appended as an extra predictor column
    :param drift_test_row: np.ndarray, (n_trials,) optional, same as drift_trainval_row for the test split
    :return: dict, GLM fit results
    """
    from sklearn.linear_model import RidgeCV, Ridge
    from sklearn.metrics import r2_score

    cv_folds = 5
    n_features = X_trainval.shape[0]
    n_bins = X_trainval.shape[2]

    # Reshape into GLM-compatible matrices
    y_trainval = spikes_trainval[neuron_id].reshape(spikes_trainval.shape[1], -1).flatten()
    X_trainval_reshaped = X_trainval.transpose(1, 2, 0).reshape(-1, n_features)
    y_test = spikes_test[neuron_id].reshape(spikes_test.shape[1], -1).flatten()
    X_test_reshaped = X_test.transpose(1, 2, 0).reshape(-1, n_features)

    if drift_trainval_row is not None:
        # np.repeat (not tile): X_*_reshaped is flattened trial-major/bin-minor by the transpose+reshape
        # above, so each trial's scalar value must repeat n_bins times consecutively to line up.
        X_trainval_reshaped = np.hstack([X_trainval_reshaped, np.repeat(drift_trainval_row, n_bins)[:, None]])
        X_test_reshaped = np.hstack([X_test_reshaped, np.repeat(drift_test_row, n_bins)[:, None]])

    # -------------------------
    # Cross-validation for lambda (if needed)
    # -------------------------
    if isinstance(lambdas, np.ndarray):
        ridge_cv = RidgeCV(
            alphas=lambdas,
            cv=cv_folds,
            fit_intercept=False,
            scoring='r2',
        )
        ridge_cv.fit(X_trainval_reshaped, y_trainval)
        lambda_opt = ridge_cv.alpha_
        model = ridge_cv
    else:
        lambda_opt = float(lambdas)
        model = Ridge(alpha=lambda_opt, fit_intercept=False)
        model.fit(X_trainval_reshaped, y_trainval)

    fit_success = True

    # -------------------------
    # Predictions
    # -------------------------
    y_trainval_pred = model.predict(X_trainval_reshaped)
    y_test_pred = model.predict(X_test_reshaped)

    # -------------------------
    # Scores (pseudo R²)
    # -------------------------
    train_score = r2_score(y_trainval, y_trainval_pred)
    test_score = r2_score(y_test, y_test_pred)

    # Log-likelihood (Gaussian) - proportional to negative MSE
    train_ll = -0.5 * np.sum((y_trainval - y_trainval_pred) ** 2)
    test_ll = -0.5 * np.sum((y_test - y_test_pred) ** 2)

    # Correlations
    train_corr, _ = sp.stats.pearsonr(y_trainval, y_trainval_pred)
    test_corr, _ = sp.stats.pearsonr(y_test, y_test_pred)

    # Mutual information (optional)
    train_mi = compute_mutual_info(y_trainval, y_trainval_pred)
    test_mi = compute_mutual_info(y_test, y_test_pred)

    out = {
        'neuron_id': neuron_id,
        'lambda_opt': lambda_opt,
        'train_ll': train_ll,
        'train_score': train_score,
        'train_corr': train_corr,
        'train_mi': train_mi,
        'test_ll': test_ll,
        'test_score': test_score,
        'test_corr': test_corr,
        'test_mi': test_mi,
        'coef': model.coef_.copy(),
        'y_test': y_test.astype(np.float32),
        'y_test_pred': y_test_pred.astype(np.float32),
        'n_bins': n_bins,
        'fit_success': fit_success,
    }
    # Train predictions are ~4x the size of test ones and only needed for --save_predictions;
    # returning them for every neuron/model was a large part of the memory use.
    if _worker_keep_train_predictions:
        out['y_train'] = y_trainval
        out['y_train_pred'] = y_trainval_pred
    return out


# Set by run_unit_glm_pipeline_with_pool (True only with save_predictions); read by the workers.
_KEEP_TRAIN_PREDICTIONS = False
_worker_keep_train_predictions = False
_worker_spikes_trainval = None
_worker_X_trainval = None
_worker_spikes_test = None
_worker_X_test = None
_worker_drift_trainval = None
_worker_drift_test = None


def _init_glm_worker(spikes_trainval, X_trainval, spikes_test, X_test, drift_trainval=None, drift_test=None,
                     keep_train_predictions=False):
    """Pool initializer: stash the shared arrays once per worker instead of pickling them into every task."""
    global _worker_spikes_trainval, _worker_X_trainval, _worker_spikes_test, _worker_X_test
    global _worker_drift_trainval, _worker_drift_test, _worker_keep_train_predictions
    _worker_keep_train_predictions = keep_train_predictions
    _worker_spikes_trainval = spikes_trainval
    _worker_X_trainval = X_trainval
    _worker_spikes_test = spikes_test
    _worker_X_test = X_test
    _worker_drift_trainval = drift_trainval
    _worker_drift_test = drift_test


def fit_neuron_glm_wrapper(args):
    neuron_id, lambdas = args
    drift_trainval_row = _worker_drift_trainval[neuron_id] if _worker_drift_trainval is not None else None
    drift_test_row = _worker_drift_test[neuron_id] if _worker_drift_test is not None else None
    return fit_neuron_glm_gaussian(neuron_id, _worker_spikes_trainval, _worker_X_trainval,
                                   _worker_spikes_test, _worker_X_test, lambdas,
                                   drift_trainval_row=drift_trainval_row, drift_test_row=drift_test_row)

def parallel_fit_glms(spikes_trainval, X_trainval, spikes_test, X_test, lambdas, n_jobs=10,
                       drift_trainval=None, drift_test=None):

    # Create a list of args to be passed on for each neuron
    n_neurons = spikes_trainval.shape[0]

    # Check lambdas input type
    if isinstance(lambdas, dict): # optimal lambdas per neuron pre-computed
        args_list = [(neuron_id, lambdas[neuron_id]) for neuron_id in range(n_neurons)]
    else:
        args_list = [(neuron_id, lambdas) for neuron_id in range(n_neurons)]

    with mp.Pool(processes=n_jobs, initializer=_init_glm_worker,
                 initargs=(spikes_trainval, X_trainval, spikes_test, X_test, drift_trainval, drift_test,
                           _KEEP_TRAIN_PREDICTIONS)) as pool:
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
    



PREDICTION_COLUMNS = ['y_train', 'y_train_pred', 'y_test', 'y_test_pred']


def save_model_results(result_df, filename, commit_hash, output_dir, save_predictions=False):
    result_path = pathlib.Path(output_dir, 'models')
    result_path.mkdir(parents=True, exist_ok=True)

    # Predictions (y_train/y_train_pred/y_test/y_test_pred) are large per-trial arrays that bloat the
    # main results file. Keep them out of it by default; when requested, save them to a separate
    # companion file instead (as native parquet list columns, not JSON text), joinable back on
    # neuron_id/fold/model_name.
    prediction_cols = [c for c in PREDICTION_COLUMNS if c in result_df.columns]
    if prediction_cols:
        if save_predictions:
            id_cols = [c for c in ['neuron_id', 'fold', 'model_name'] if c in result_df.columns]
            predictions_df = result_df[id_cols + prediction_cols].copy()
            predictions_df.to_parquet(os.path.join(result_path, '{}_{}_predictions.parquet'.format(commit_hash, filename)))
            print('Saved predictions in: ', result_path)
        result_df = result_df.drop(columns=prediction_cols)

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


def restrict_predictor_to_bin(X, pred_idx, bin_idx, mask):
    """
    Keep only the specified bin(s) for a given predictor, zero out others.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_features, n_trials, n_bins)
    pred_idx : int
        Index of the predictor to mask
    bin_idx : int
        Which bin to keep
    mask : array_like
        Boolean mask of shape (n_bins,) indicating which bin(s) to keep

    Returns
    -------
    X_masked : np.ndarray
        Copy of X with only the selected bin(s) for pred_idx preserved, others zeroed
    """
    X_masked = X.copy()
    X_masked[pred_idx, :, ~mask] = 0
    return X_masked


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


# Number of trailing design-matrix rows z-scored with train-fold stats: 4 DLC features, plus the lick
# regressor columns when present (set per run in run_unit_glm_pipeline_with_pool)
N_STANDARDIZED_TAIL = 4
STANDARDIZED_NAMES = ('jaw_dist', 'pupil_area', 'whisker_vel', 'nose_dist')


LICK_KERNEL_PREFIXES = ('lick_ev_t', 'lick_b', 'lick_rew_b', 'lick_oth_b', 'lick_bout')


def lick_raised_cosines(window_s, n_bases, dt=0.001):
    """Raised-cosine bases tiling the lick window at resolution dt: (n_bases, n_samples), lags (s)."""
    lags = np.arange(int(round(window_s[0] / dt)), int(round(window_s[1] / dt))) * dt
    centers = np.linspace(window_s[0], window_s[1], n_bases)
    width = centers[1] - centers[0]  # neighbours overlap by half: the bases sum to 1 over the window
    B = np.array([np.where(np.abs(lags - c) < width, 0.5 * (1 + np.cos(np.pi * (lags - c) / width)), 0.0) for c in centers])
    return B, lags


def build_lick_kernel_predictors(lick_times, trial_starts, n_bins, bin_size, kind, window_s, n_bases, bout_gap_s,
                                 reward_times=None, dt=0.001):
    """Per-lick regressors, (n_trials, n_bins) each, for the GLM trial windows.
    'binned': lick indicator per bin, shifted like an event kernel (one column per 50 ms lag).
    'fine*' : each basis placed at every lick on a 1 ms grid, summed over licks, averaged within each bin.
    'fine_split': separate bases for licks within 2 s after a reward (lick_time) and for all other licks.
    'fine_bout' : + fraction of each bin inside a lick bout (first to last lick, gaps < bout_gap_s)."""
    n_trials = len(trial_starts)
    out = {}
    if kind == 'binned':
        ind = np.zeros((n_trials, n_bins))
        for i, st in enumerate(trial_starts):
            idx = np.floor((lick_times[(lick_times >= st) & (lick_times < st + n_bins * bin_size)] - st) / bin_size).astype(int)
            ind[i, np.clip(idx, 0, n_bins - 1)] = 1
        pre, post = int(round(abs(window_s[0]) / bin_size)), int(round(window_s[1] / bin_size))
        for shift in range(-pre, post):
            sh = np.zeros_like(ind)
            if shift < 0:
                sh[:, :shift] = ind[:, -shift:]
            elif shift > 0:
                sh[:, shift:] = ind[:, :-shift]
            else:
                sh = ind.copy()
            out[f'lick_ev_t{shift * bin_size:+.2f}s'] = sh
        return out

    B, lags = lick_raised_cosines(window_s, n_bases, dt)
    lag0 = int(round(-window_s[0] / dt))            # index of lag 0 in the basis
    spb = int(round(bin_size / dt))                 # fine samples per bin
    n_fine = n_bins * spb
    margin = B.shape[1]                             # licks this far outside the window still reach it

    def fine_regressors(times, prefix):
        R = np.zeros((n_bases, n_trials, n_bins))
        for i, st in enumerate(trial_starts):
            t = times[(times >= st - margin * dt) & (times < st + (n_fine + margin) * dt)]
            if len(t) == 0:
                continue
            train = np.zeros(n_fine + 2 * margin)
            np.add.at(train, np.clip(np.round((t - st) / dt).astype(int) + margin, 0, len(train) - 1), 1)
            for k in range(n_bases):
                # y[n] = sum over licks m of B_k(n - m); the basis starts at lag window_s[0]
                y = np.convolve(train, B[k])[lag0: lag0 + len(train)]
                R[k, i] = y[margin: margin + n_fine].reshape(n_bins, spb).mean(1)
        return {f'{prefix}{k}': R[k] for k in range(n_bases)}

    if kind in ('fine', 'fine_bout'):
        out.update(fine_regressors(lick_times, 'lick_b'))
    elif kind == 'fine_split':
        rt = np.sort(np.asarray(reward_times, float)) if reward_times is not None else np.array([])
        rt = rt[np.isfinite(rt)]
        j = np.searchsorted(rt, lick_times + 0.01, side='right') - 1   # last reward at or just before each lick
        during = (j >= 0) & (lick_times - rt[np.clip(j, 0, None)] <= 2.0) if len(rt) else np.zeros(len(lick_times), bool)
        out.update(fine_regressors(lick_times[during], 'lick_rew_b'))
        out.update(fine_regressors(lick_times[~during], 'lick_oth_b'))
    else:
        raise ValueError(f'unknown lick_kernel {kind}')

    if kind == 'fine_bout':
        # bouts: consecutive licks closer than bout_gap_s; state = fraction of the bin between first and last lick
        if len(lick_times):
            new = np.concatenate([[True], np.diff(lick_times) >= bout_gap_s])
            starts, ends = lick_times[new], lick_times[np.r_[new[1:], True]]
        else:
            starts = ends = np.array([])
        state = np.zeros((n_trials, n_bins))
        for i, st in enumerate(trial_starts):
            e0 = st + bin_size * np.arange(n_bins)
            for a, b in zip(starts, ends):
                if b < st or a > st + n_bins * bin_size:
                    continue
                state[i] += np.clip(np.minimum(e0 + bin_size, b) - np.maximum(e0, a), 0, None) / bin_size
        out['lick_bout'] = np.clip(state, 0, 1)
    return out


def _reward_lick_opts(cfg):
    return {'reward_align': cfg['reward_align'], 'lick_regressor': cfg['lick_regressor'], 'lick_lags_s': cfg['lick_lags_s'],
            'lick_kernel': cfg['lick_kernel'], 'lick_kernel_window_s': cfg['lick_kernel_window_s'],
            'lick_kernel_n_bases': cfg['lick_kernel_n_bases'], 'lick_bout_gap_s': cfg['lick_bout_gap_s'],
            'reward_window_s': cfg['reward_window_s'], 'iti_bouts': _iti_bout_opts(cfg)}


def _iti_bout_opts(cfg):
    if not cfg['iti_bout_trials']:
        return None
    return {'min_licks': cfg['iti_bout_min_licks'], 'quiet_s': cfg['iti_bout_quiet_s'],
            'gap_s': cfg['lick_bout_gap_s'], 'min_n': cfg['iti_bout_min_n']}


ITI_BOUT_TYPE = 'iti_lick_bout'


def load_piezo_lick_times(nwb_path):
    with NWBHDF5IO(nwb_path, mode='r', load_namespaces=True) as io:
        ev = io.read().processing['behavior']['BehavioralEvents']
        return np.sort(np.asarray(ev.time_series['piezo_lick_times'].timestamps[:], float))


def iti_lick_bout_starts(lick_times, all_trials, min_licks, quiet_s, gap_s, window_s=(-1, 2)):
    """First licks of inter-trial lick bouts usable as pseudo-trials: >= min_licks licks (gaps < gap_s), no lick
    in the quiet_s before, the whole window clear of every trial window (passive and early-lick trials included)
    and of the previous pseudo-trial, between two non-passive trials."""
    if len(lick_times) == 0:
        return np.array([])
    new = np.r_[True, np.diff(lick_times) >= gap_s]
    starts = lick_times[new]
    size = np.bincount(np.cumsum(new) - 1)
    quiet = starts - np.r_[-np.inf, lick_times[:-1]][new]
    trials = all_trials.sort_values('start_time')
    st = trials['start_time'].to_numpy(float)
    passive = (trials['context'] == 'passif').to_numpy()
    keep, last_end = [], -np.inf
    for s in starts[(size >= min_licks) & (quiet >= quiet_s)]:
        a, b = s + window_s[0], s + window_s[1]
        j = np.searchsorted(st, s)  # next trial
        if j == 0 or j == len(st) or passive[j - 1] or passive[j]:
            continue
        if a < last_end or np.any((st + window_s[1] > a) & (st + window_s[0] < b)):
            continue
        keep.append(s)
        last_end = b
    return np.array(keep)


def add_iti_bout_trials(trials_df, all_trials, lick_times, min_licks, quiet_s, gap_s, min_n):
    """Append inter-trial lick bouts as pseudo-trials (trial_type ITI_BOUT_TYPE, start_time = first lick, no
    stimulus, no reward, lick_flag 0 so they never count as rewarded), in time order. Unchanged if < min_n bouts."""
    starts = iti_lick_bout_starts(lick_times, all_trials, min_licks, quiet_s, gap_s)
    if len(starts) < min_n:
        print(f'[INFO] {len(starts)} inter-trial lick bouts (< {min_n}): no pseudo-trials added')
        return trials_df
    pseudo = pd.DataFrame({'start_time': starts, 'trial_type': ITI_BOUT_TYPE, 'lick_flag': 0, 'reward_available': 0,
                           'context': 'active', 'trial_id': -1 - np.arange(len(starts))})
    for col in ('mouse_id', 'session_id'):
        if col in trials_df:
            pseudo[col] = trials_df[col].iloc[0]
    print(f'[INFO] {len(starts)} inter-trial lick bouts added as pseudo-trials')
    return pd.concat([trials_df, pseudo], ignore_index=True).sort_values('start_time', kind='mergesort').reset_index(drop=True)


def _standardize_by_name(X_trainval, X_test, feature_names):
    """Z-score (train-fold stats only, in place) the DLC and lick regressor rows, found by name."""
    for idx, name in enumerate(feature_names):
        if name in STANDARDIZED_NAMES or name.startswith('lick_piezo_t'):
            mean, std = X_trainval[idx].mean(), X_trainval[idx].std()
            std = std if std > 0 else 1e-6
            X_trainval[idx] = (X_trainval[idx] - mean) / std
            X_test[idx] = (X_test[idx] - mean) / std
    return X_trainval, X_test


def reward_basis_transform(X, feature_names, basis, n_cosine=4, prefix='piezo_reward'):
    """Replace the <prefix> lag columns (default the reward kernel) by fewer, smooth regressors (same position in
    the matrix): 'constant' = one weight for the whole window; 'cosine' = n raised-cosine bases over the lags."""
    idx = [i for i, f in enumerate(feature_names) if f.startswith(f'{prefix}_t')]
    n_lags = len(idx)
    lags = np.arange(n_lags)
    if basis == 'constant':
        B, names = np.ones((1, n_lags)), [f'{prefix}_const']
    elif basis == 'cosine':
        centers = np.linspace(0, n_lags - 1, n_cosine)
        width = 2 * (centers[1] - centers[0])
        B = np.array([np.where(np.abs(lags - c) < width, 0.5 * (1 + np.cos(np.pi * (lags - c) / width)), 0.0) for c in centers])
        names = [f'{prefix}_cos{k}' for k in range(n_cosine)]
    else:
        raise ValueError(basis)
    new_rows = np.tensordot(B, X[idx], axes=(1, 0))  # (n_basis, n_trials, n_bins)
    first = idx[0]
    keep = [i for i in range(X.shape[0]) if i not in idx]
    X_new = np.concatenate([X[[i for i in keep if i < first]], new_rows, X[[i for i in keep if i > first]]], axis=0)
    names_new = [feature_names[i] for i in keep if i < first] + names + [feature_names[i] for i in keep if i > first]
    return X_new, names_new


def _standardize_dlc_features(X_trainval, X_test, n_dlc=None):
    """Standardize the last `n_dlc` predictor rows (DLC features + lick regressor) using train-fold stats only, in place."""
    n_dlc = N_STANDARDIZED_TAIL if n_dlc is None else n_dlc
    n_features = X_trainval.shape[0]
    for idx in range(n_features - n_dlc, n_features):
        mean = X_trainval[idx].mean()
        std = X_trainval[idx].std()
        if std == 0:
            std = 1e-6
        X_trainval[idx] = (X_trainval[idx] - mean) / std
        X_test[idx] = (X_test[idx] - mean) / std
    return X_trainval, X_test


def _fit_labeled_model(spikes_trainval, X_trainval, spikes_test, X_test, lambdas, n_jobs,
                        fold_idx, trainval_ids, test_ids, model_name, predictor_names,
                        drift_trainval=None, drift_test=None):
    """Fit a GLM and package the result df with the bookkeeping columns used across model variants."""
    results = parallel_fit_glms(spikes_trainval=spikes_trainval, X_trainval=X_trainval,
                                 spikes_test=spikes_test, X_test=X_test, lambdas=lambdas, n_jobs=n_jobs,
                                 drift_trainval=drift_trainval, drift_test=drift_test)
    if drift_trainval is not None:
        predictor_names = list(predictor_names) + ['drift']
    df = pd.DataFrame(results)
    df['fold'] = fold_idx
    df['train_trials'] = [trainval_ids] * len(df)
    df['test_trials'] = [test_ids] * len(df)
    df['model_name'] = model_name
    df['predictors'] = [list(predictor_names)] * len(df)
    return df


def _average_draw_dfs(draw_dfs, fold_idx, trainval_ids, test_ids, model_name, predictor_names):
    """Average coefficients and performance metrics across control draws (random/linear-shift), per neuron."""
    metric_cols = ['lambda_opt', 'train_ll', 'train_score', 'train_corr', 'train_mi',
                   'test_ll', 'test_score', 'test_corr', 'test_mi']
    avg_df = draw_dfs[0][['neuron_id', 'n_bins']].copy()
    for col in metric_cols:
        avg_df[col] = np.mean([d[col].values for d in draw_dfs], axis=0)
    avg_df['coef'] = [np.mean([d['coef'].iloc[i] for d in draw_dfs], axis=0) for i in range(len(avg_df))]
    avg_df['fit_success'] = np.all([d['fit_success'].values for d in draw_dfs], axis=0)
    avg_df['fold'] = fold_idx
    avg_df['train_trials'] = [trainval_ids] * len(avg_df)
    avg_df['test_trials'] = [test_ids] * len(avg_df)
    avg_df['model_name'] = model_name
    # Same predictors as the draws (incl. 'drift' when they were fit with it), matching the averaged coefs
    if 'predictors' in draw_dfs[0].columns:
        predictor_names = draw_dfs[0]['predictors'].iloc[0]
    avg_df['predictors'] = [list(predictor_names)] * len(avg_df)
    return avg_df


def _dump_fold_test_predictions(tmp_dir, result_dfs, models, fold_idx, test_ids, n_bins):
    """
    Write this fold's held-out predictions (all neurons) of the requested models to tmp_dir,
    one npz per model, so they don't stay in memory across folds.
    :param result_dfs: list of result DataFrames (one model each, as built in the pipeline)
    """
    for df in result_dfs:
        if len(df) == 0 or 'y_test_pred' not in df.columns:
            continue  # e.g. averaged control draws carry no predictions
        model_name = df['model_name'].iloc[0]
        if model_name not in models:
            continue
        df = df.sort_values('neuron_id')
        pred = np.stack([np.asarray(p, dtype=np.float32) for p in df['y_test_pred']])
        pred = pred.reshape(len(df), len(test_ids), n_bins)
        np.savez(pathlib.Path(tmp_dir, f'{model_name}_fold{fold_idx}.npz'),
                 neuron_id=df['neuron_id'].values, test_ids=np.asarray(test_ids), y_test_pred=pred)


def _save_top_predictions(tmp_dir, out_file, models, fold_scores, spikes, trials_df, neurons_ccf, cfg):
    """
    Rank neurons by mean held-out R2 of 'full' across folds, keep the top fraction, and save their
    full-session cross-validated predictions (each trial predicted by the fold where it was held out).
    :param fold_scores: np.ndarray, (n_neurons, n_folds) test R2 of the full model
    :return: dict summary for the run config
    """
    n_neurons, n_trials, n_bins = spikes.shape
    mean_score = np.nanmean(fold_scores, axis=1)
    firing_rate = spikes.sum(axis=(1, 2)) / (n_trials * n_bins * BIN_SIZE)  # Hz, within trial windows
    eligible = np.where((firing_rate >= cfg['top_predictions_min_rate_hz']) & np.isfinite(mean_score))[0]
    n_top = max(1, int(math.ceil(cfg['top_predictions_fraction'] * len(eligible)))) if len(eligible) else 0
    top = eligible[np.argsort(mean_score[eligible])[::-1][:n_top]]

    predictions, fold_of_trial = {}, np.full(n_trials, -1)
    for model in models:
        fold_files = sorted(pathlib.Path(tmp_dir).glob(f'{model}_fold*.npz'))
        if not fold_files:
            continue
        pred = np.full((len(top), n_trials, n_bins), np.nan, dtype=np.float32)
        for fold_file in fold_files:
            d = np.load(fold_file)
            row_of_neuron = {nid: i for i, nid in enumerate(d['neuron_id'])}
            rows = [row_of_neuron[nid] for nid in top]
            pred[:, d['test_ids'], :] = d['y_test_pred'][rows]
            fold_of_trial[d['test_ids']] = int(fold_file.stem.rsplit('_fold', 1)[1])
        predictions[model] = pred

    trial_cols = [c for c in ['trial_id', 'trial_type', 'lick_flag', 'reward_available', 'context', 'perf',
                              'start_time', 'block_perf_type'] if c in trials_df.columns]
    result = {
        'neuron_id': top,  # row index in the spikes array (same as neuron_id in the result parquets)
        'ccf_parent_acronym': np.asarray(neurons_ccf)[top],
        'mean_test_score': mean_score[top],
        'fold_test_scores': fold_scores[top],
        'firing_rate_hz': firing_rate[top],
        'y': spikes[top].astype(np.float32),  # (n_top, n_trials, n_bins) spike counts
        'predictions': predictions,  # {model: (n_top, n_trials, n_bins)}
        'fold_of_trial': fold_of_trial,
        'trials': trials_df[trial_cols].reset_index(drop=True),
        'bin_size': BIN_SIZE,
        'trial_window_s': (-1, 2),  # relative to trial start_time, as in load_nwb_spikes_and_predictors
        'ranking': {'metric': 'mean test R2 of full across folds', 'fraction': cfg['top_predictions_fraction'],
                    'min_rate_hz': cfg['top_predictions_min_rate_hz'], 'n_eligible': int(len(eligible))},
    }
    with open(out_file, 'wb') as f:
        pickle.dump(result, f)
    print(f'Saved predictions of the top {len(top)}/{len(eligible)} neurons ({list(predictions)}) in {out_file}')
    return {'file': str(out_file), 'n_top': int(len(top)), 'n_eligible': int(len(eligible)),
            'models': list(predictions)}


def _create_summary_after_fit(nwb_path, mouse_output_path, commit_hash, cfg):
    """Run the 'create_summary' step of glms_plotting_utils.mouse_glm_results on this session's
    results, writing <mouse_output_path>/<hash>/<unit filter>/summary_<mouse>_unit_glm_<hash>.parquet.
    Errors are caught and returned, so a summary problem never fails the fitting job."""
    start = time.time()
    try:
        import glms_plotting_utils as gpu  # imported here: heavy, and only needed for this step
        # mouse_glm_results only keeps sessions of the requested training day (0 = learning day, >0 = later days)
        _, day = nwbreader.get_bhv_type_and_training_day_index(nwb_path)
        summary_dir = pathlib.Path(mouse_output_path, commit_hash)
        summary_dir.mkdir(parents=True, exist_ok=True)
        gpu.mouse_glm_results(nwb_list=[str(nwb_path)], model_path=str(pathlib.Path(mouse_output_path, 'models')),
                              plots=['create_summary'], output_path=str(summary_dir), git_version=commit_hash,
                              info_path=cfg['summary_info_path'], day_to_analyze=day, overwrite=True)
        files = sorted(str(p) for p in summary_dir.glob(f'*/summary_*_unit_glm_{commit_hash}.parquet'))
        status = 'completed' if files else 'no summary written'
        print(f'Summary step: {status} ({len(files)} files) in {time.time() - start:.0f} s')
        return {'status': status, 'files': files, 'duration_s': round(time.time() - start)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'[WARN] Summary step failed (fit results are unaffected): {e!r}')
        return {'status': 'failed', 'error': repr(e), 'duration_s': round(time.time() - start)}


def write_run_config(output_dir, commit_hash, run_info):
    """Write models/<hash>_run_config.json next to the <hash>_model_*.parquet results, so it is
    overwritten exactly when they are (same hash, same folder)."""
    result_path = pathlib.Path(output_dir, 'models')
    result_path.mkdir(parents=True, exist_ok=True)
    path = result_path / f'{commit_hash}_run_config.json'
    with open(path, 'w') as f:
        json.dump(run_info, f, indent=2, default=str)  # default=str: numpy scalars, paths
    return path


def check_design_matrix(X, feature_names, name, n_trials):
    """
    Sanity checks on a design matrix before fitting. Prints and returns a summary.
    :param X: np.ndarray, (n_features, n_trials, n_bins)
    :param feature_names: list of str, one per row of X
    :param name: str, label for printing
    :param n_trials: int, expected number of trials
    :return: dict with the checks (saved into the run config)
    """
    n_features = X.shape[0]
    assert len(feature_names) == n_features, f"[{name}] {len(feature_names)} feature names for {n_features} rows"
    assert X.shape[1] == n_trials, f"[{name}] design matrix has {X.shape[1]} trials, expected {n_trials}"

    X2d = X.reshape(n_features, -1).T  # (n_trials * n_bins, n_features)

    # Columns that are zero everywhere: their coefficients are exactly 0 (ridge, no intercept)
    zero_cols = [feature_names[i] for i in np.where(~X2d.any(axis=0))[0]]
    # Identical columns: their weights are not identifiable separately
    _, first_idx, inverse = np.unique(X2d, axis=1, return_index=True, return_inverse=True)
    duplicated_cols = [feature_names[i] for i in range(n_features)
                       if first_idx[inverse.ravel()[i]] != i and X2d[:, i].any()]
    # Rank, ignoring all-zero columns: rank < number of columns means some predictors are a
    # linear combination of others (e.g. a kernel equal to the sum of two other kernels)
    nonzero = X2d[:, X2d.any(axis=0)]
    rank = int(np.linalg.matrix_rank(nonzero))
    rank_deficient = rank < nonzero.shape[1]

    # Number of trials in which each event kernel is active
    kernel_trials = {}
    for kernel in dict.fromkeys(f.rsplit('_t', 1)[0] for f in feature_names if '_t' in f and f.endswith('s')):
        idx = [i for i, f in enumerate(feature_names) if f.rsplit('_t', 1)[0] == kernel]
        kernel_trials[kernel] = int(X[idx].any(axis=(0, 2)).sum())

    checks = {'n_features': n_features, 'n_trials': n_trials, 'rank_nonzero_columns': rank,
              'n_nonzero_columns': int(nonzero.shape[1]), 'rank_deficient': bool(rank_deficient),
              'all_zero_columns': zero_cols, 'duplicated_columns': duplicated_cols,
              'trials_with_kernel_active': kernel_trials}

    print(f"[design check: {name}] {n_features} features, {n_trials} trials, rank {rank}/{nonzero.shape[1]} non-zero columns")
    if zero_cols:
        print(f"  [WARN] {len(zero_cols)} all-zero columns (coefficients will be 0): "
              f"{sorted(dict.fromkeys(c.rsplit('_t', 1)[0] for c in zero_cols))}")
    if duplicated_cols:
        print(f"  [WARN] duplicated columns: {duplicated_cols}")
    if rank_deficient:
        print(f"  [WARN] rank deficient: some predictors are linear combinations of others, "
              f"their weights are not identifiable")
    print(f"  trials with kernel active: {kernel_trials}")
    return checks


def run_unit_glm_pipeline_with_pool(nwb_path, output_dir, n_jobs=10, save_predictions=False, config=None,
                                    check_only=False):
    """
    :param config: dict, overrides for GLM_CONFIG (see top of file)
    :param check_only: bool, build the design matrices and run the checks, then stop before fitting
        (nothing is written)
    """
    print('GLM fitting for:', pathlib.Path(nwb_path).name)
    cfg = {**GLM_CONFIG, **(config or {})}
    global N_STANDARDIZED_TAIL
    n_lick_cols = (int(np.round(abs(cfg['lick_lags_s'][0]) / BIN_SIZE)) + int(np.round(cfg['lick_lags_s'][1] / BIN_SIZE))
                   if cfg['lick_regressor'] else 0)
    N_STANDARDIZED_TAIL = 4 + n_lick_cols
    def build_dm(*args, **kwargs):  # design matrices with the configured reward kernel basis
        return build_design_matrix(*args, reward_basis=cfg['reward_basis'], reward_n_cosine=cfg['reward_n_cosine'], **kwargs)
    if not cfg['lick_regressor'] and cfg['lick_kernel'] is None:
        cfg['reduced_models'] = [m for m in cfg['reduced_models'] if m not in ('lick_encoding', 'licking_encoding')]
    global _KEEP_TRAIN_PREDICTIONS
    _KEEP_TRAIN_PREDICTIONS = save_predictions  # workers return train predictions only if they get saved
    unknown = set(config or {}) - set(GLM_CONFIG)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    commit_hash = get_git_revision_short_hash()
    mouse_id = nwbreader.get_mouse_id(nwb_path)
    mouse_output_path = pathlib.Path(output_dir, mouse_id, 'whisker_0', 'unit_glm')
    if not check_only:
        mouse_output_path.mkdir(parents=True, exist_ok=True)
    run_info = {
        'git_hash': commit_hash,
        'status': 'started',
        'start_time': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'nwb': str(nwb_path),
        'mouse_id': mouse_id,
        'output_path': str(mouse_output_path),
        'root_path': ROOT_PATH,
        'bin_size': BIN_SIZE,
        'n_jobs': n_jobs,
        'save_predictions': save_predictions,
        'config': cfg,
    }
    # -----------------------------------
    # Load and prepare data from NWB file
    # -----------------------------------

    # Get trial table, input and output formatted for GLM, list of predictor types
    all_trials = nwbreader.get_trial_table(nwb_path)
    trials_df = all_trials[(all_trials['context'] != 'passif') &(all_trials['perf'] != 6)].copy()
    trials_df['mouse_id'] = mouse_id
    trials_df = load_perf_blocks(trials_df, mouse_id)
    if cfg['iti_bout_trials']:
        # Same pseudo-trials as load_nwb_spikes_and_predictors adds (the trial-count check below catches a mismatch)
        trials_df = add_iti_bout_trials(trials_df, all_trials, load_piezo_lick_times(nwb_path), **_iti_bout_opts(cfg))
    run_info['n_iti_bout_trials'] = int((trials_df['trial_type'] == ITI_BOUT_TYPE).sum())
    #trials_df = trials_df[trials_df['trial_type'] == 'whisker_trial']
    trials_df = trials_df.reset_index(drop=True)
    #try: #uncomment when design matrix fixed, so that no need to recompute it
    #    X, spikes, feature_names = load_model_input_output(output_dir)
    #

    #  Get trial table, input and output formatted for GLM, list of predictor types
    spikes, predictors, predictor_types, n_bins, bin_size, neurons_ccf, scale, drift_matrix = load_nwb_spikes_and_predictors(
        nwb_path, bin_size=BIN_SIZE, compute_drift=cfg['compute_drift'], motion_data_root=ROOT_PATH,
        prev_whisker_history=cfg['prev_whisker_history'], merge_whisker=cfg['merge_whisker'], **_reward_lick_opts(cfg))
    if cfg['compute_drift']:
        run_info['drift'] = dict(LAST_DRIFT_INFO)
    event_defs = predictor_types['event_defs']
    analog_keys = predictor_types['analog_keys']

    # Build design matrix for entire dataset
    X, feature_names = build_dm(predictors, event_defs, analog_keys, bin_size=BIN_SIZE, scale = None)
    X = np.nan_to_num(X)
    n_features = X.shape[0]

    # Save input/output data
    if not check_only:
        save_model_input_output(X, spikes, feature_names, mouse_output_path, neurons_ccf)
    whisker_kernels = cfg['whisker_kernels']
    number_of_whisker_kernel = cfg['number_of_whisker_kernel']

    # Each entry defines a design-matrix variant: which of the split-eligible event
    # kernels (whisker_hits, whisker_misses, auditory_stim) get split into
    # `number_of_whisker_kernel` temporal kernels vs. kept as a single un-split kernel.
    # 'full' splits all three together (the main model below); the others split only
    # one event type at a time so each can be evaluated as its own model.
    kernel_split_configs = {name: tuple(kernels) for name, kernels in cfg['kernel_split_configs'].items()}
    split_model_data = {}

    if whisker_kernels:
        for config_name, kernels_to_split in kernel_split_configs.items():
            spikes, predictors, predictor_types, n_bins, bin_size, neurons_ccf, _, _ = load_nwb_spikes_and_predictors(
                nwb_path, bin_size=BIN_SIZE, nb_of_whisker_kernel=number_of_whisker_kernel,
                kernels_to_split=kernels_to_split, split_mode='sequential', prev_whisker_history=cfg['prev_whisker_history'], merge_whisker=cfg['merge_whisker'], **_reward_lick_opts(cfg))
            event_defs = predictor_types['event_defs']
            analog_keys = predictor_types['analog_keys']

            # Build design matrix for entire dataset
            X_extra, feature_names_extra = build_dm(predictors, event_defs, analog_keys, bin_size=BIN_SIZE)
            X_extra = np.nan_to_num(X_extra)
            split_model_data[config_name] = {'X': X_extra, 'feature_names': feature_names_extra}

        # The 'full' config (all three split together) drives the main model + reduced ablations below
        all_Xs = split_model_data['full']['X']
        feature_namess = split_model_data['full']['feature_names']
        nb_whisker_kernels = number_of_whisker_kernel

    reward_kernels = cfg['reward_kernels']
    if reward_kernels:

        X_rewards = []
        spikes, predictors, predictor_types, n_bins, bin_size, neurons_ccf, _, _ = load_nwb_spikes_and_predictors(nwb_path, bin_size=BIN_SIZE, reward_kernel_per_type = True,
                                                                                                         prev_whisker_history=cfg['prev_whisker_history'], merge_whisker=cfg['merge_whisker'], **_reward_lick_opts(cfg))
        event_defs = predictor_types['event_defs']
        analog_keys = predictor_types['analog_keys']

        # Build design matrix for entire dataset
        X_extra, feature_names_extra = build_dm(predictors, event_defs, analog_keys, bin_size=BIN_SIZE)
        X_rewards.append(X_extra)
    
    random_split = cfg['random_split']
    linear_shift_split = cfg['linear_shift_split']
    n_control_draws = cfg['n_control_draws']
    if random_split or linear_shift_split:
        # For every split config, build control draws that re-fit the same kernel split
        # with the event-to-kernel assignment scrambled, as a baseline for how much of the
        # split model's gain is just from having more free parameters vs. genuine temporal
        # structure. 'random' reshuffles events before splitting (seed = draw_idx);
        # 'linear_shift' is a deterministic alternative that circularly rolls the
        # time-sorted events by an amount set by draw_idx instead of shuffling randomly.
        for config_name, kernels_to_split in kernel_split_configs.items():
            model_data = split_model_data[config_name]
            if random_split:
                X_randoms, feature_names_randoms = [], []
                for draw_idx in range(n_control_draws):
                    spikes, predictors, predictor_types, n_bins, bin_size, neurons_ccf, _, _ = load_nwb_spikes_and_predictors(
                        nwb_path, bin_size=BIN_SIZE, nb_of_whisker_kernel=number_of_whisker_kernel,
                        kernels_to_split=kernels_to_split, split_mode='random', split_seed=draw_idx,
                        prev_whisker_history=cfg['prev_whisker_history'], merge_whisker=cfg['merge_whisker'], **_reward_lick_opts(cfg))
                    event_defs = predictor_types['event_defs']
                    analog_keys = predictor_types['analog_keys']
                    X_r, feature_names_r = build_dm(predictors, event_defs, analog_keys, bin_size=BIN_SIZE)
                    X_randoms.append(np.nan_to_num(X_r))
                    feature_names_randoms.append(feature_names_r)
                model_data['random_Xs'] = X_randoms
                model_data['random_feature_names'] = feature_names_randoms

            if linear_shift_split:
                X_shifts, feature_names_shifts = [], []
                for draw_idx in range(n_control_draws):
                    spikes, predictors, predictor_types, n_bins, bin_size, neurons_ccf, _, _ = load_nwb_spikes_and_predictors(
                        nwb_path, bin_size=BIN_SIZE, nb_of_whisker_kernel=number_of_whisker_kernel,
                        kernels_to_split=kernels_to_split, split_mode='linear_shift', split_seed=draw_idx,
                        prev_whisker_history=cfg['prev_whisker_history'], merge_whisker=cfg['merge_whisker'], **_reward_lick_opts(cfg))
                    event_defs = predictor_types['event_defs']
                    analog_keys = predictor_types['analog_keys']
                    X_s, feature_names_s = build_dm(predictors, event_defs, analog_keys, bin_size=BIN_SIZE)
                    X_shifts.append(np.nan_to_num(X_s))
                    feature_names_shifts.append(feature_names_s)
                model_data['shift_Xs'] = X_shifts
                model_data['shift_feature_names'] = feature_names_shifts

    # Whisker-history control: same 'full' design, previous-outcome labels shuffled across trials
    history_shuffle_Xs, history_shuffle_feature_names = [], []
    if cfg['prev_whisker_history'] and whisker_kernels:
        for draw_idx in range(cfg['prev_whisker_shuffle_draws']):
            _, predictors_sh, predictor_types_sh, _, _, _, _, _ = load_nwb_spikes_and_predictors(
                nwb_path, bin_size=BIN_SIZE, nb_of_whisker_kernel=number_of_whisker_kernel,
                kernels_to_split=kernel_split_configs['full'], split_mode='sequential',
                prev_whisker_history=True, prev_whisker_shuffle_seed=draw_idx,
                merge_whisker=cfg['merge_whisker'], **_reward_lick_opts(cfg))
            X_sh, feature_names_sh = build_dm(predictors_sh, predictor_types_sh['event_defs'],
                                                         predictor_types_sh['analog_keys'], bin_size=BIN_SIZE)
            history_shuffle_Xs.append(np.nan_to_num(X_sh))
            history_shuffle_feature_names.append(feature_names_sh)

    # Available: 'last_whisker_reward', 'last_false_alarm', 'prev_success', 'last_reward',
    # 'prop_past_whisker_rewarded', 'block_perf_type', 'whisker_reward_rate_5'
    add_perf_pred = cfg['add_perf_pred']
    if add_perf_pred is not None:
        X_perfs = []
        feature_names_perfs = []
        for perf_predictors in add_perf_pred:
            spikes, predictors, predictor_types, n_bins, bin_size, neurons_ccf, _, _ = load_nwb_spikes_and_predictors(nwb_path, bin_size=BIN_SIZE,  include_predictors = [perf_predictors],
                                                                                                    prev_whisker_history=cfg['prev_whisker_history'], merge_whisker=cfg['merge_whisker'], **_reward_lick_opts(cfg))
            event_defs = predictor_types['event_defs']
            analog_keys = predictor_types['analog_keys']

            # Build design matrix for entire dataset
            X_extra, feature_names_extra = build_dm(predictors, event_defs, analog_keys, bin_size=BIN_SIZE)
            X_extra = np.nan_to_num(X_extra)
            X_perfs.append(X_extra)
            feature_names_perfs.append(feature_names_extra)

    # ---------------------------------------
    # Train/test data cross-validation splits
    # ---------------------------------------
    # model_res_df_outer = []  # disabled: duplicates data already written per-fold via save_model_results, causes OOM
    cv_outer_folds = cfg['cv_outer_folds']

    outer_kf = KFold(n_splits=cv_outer_folds, shuffle=True, random_state=cfg['cv_random_state'])

    # Folds index trials_df, but are applied to spikes / design matrices built separately
    # in load_nwb_spikes_and_predictors: all must have the same trials.
    if not (len(trials_df) == spikes.shape[1] == all_Xs.shape[1] == X.shape[1]):
        raise ValueError(f"Trial count mismatch: trials_df={len(trials_df)}, spikes={spikes.shape[1]}, "
                         f"split design matrix={all_Xs.shape[1]}, design matrix={X.shape[1]}")

    # Design-matrix checks (all-zero / duplicated columns, rank, events per kernel)
    run_info['n_trials'] = int(spikes.shape[1])
    run_info['n_neurons'] = int(spikes.shape[0])
    run_info['full_model_predictors'] = list(feature_namess)
    run_info['design_checks'] = {'full': check_design_matrix(all_Xs, feature_namess, 'full', len(trials_df))}
    if whisker_kernels:
        run_info['design_checks']['unsplit'] = check_design_matrix(X, feature_names, 'unsplit', len(trials_df))
        for config_name, model_data in split_model_data.items():
            if config_name != 'full':
                run_info['design_checks'][config_name] = check_design_matrix(
                    model_data['X'], model_data['feature_names'], config_name, len(trials_df))
    tail = feature_namess[-N_STANDARDIZED_TAIL:]
    assert all(n in STANDARDIZED_NAMES or n.startswith('lick_piezo_t') for n in tail) and \
        sum(n.startswith('lick_piezo_t') for n in feature_namess) == n_lick_cols, f'unexpected standardized tail: {tail}'

    # Reward / lick design variants, fitted per fold next to 'full' (with its lambdas)
    reward_variants = {}
    if cfg['reward_variants'] or cfg['lick_kernel_variants']:
        for variant_name, opts in [('reward_piezo_no_lick', {'reward_align': 'piezo', 'lick_regressor': False}),
                                   ('reward_licktime_no_lick', {'reward_align': 'lick_time', 'lick_regressor': False})]:
            if variant_name not in cfg['reward_variant_names']:
                continue
            _, predictors_v, predictor_types_v, _, _, _, _, _ = load_nwb_spikes_and_predictors(
                nwb_path, bin_size=BIN_SIZE, nb_of_whisker_kernel=number_of_whisker_kernel,
                kernels_to_split=kernel_split_configs['full'], split_mode='sequential',
                prev_whisker_history=cfg['prev_whisker_history'], merge_whisker=cfg['merge_whisker'],
                lick_lags_s=cfg['lick_lags_s'], **opts)
            X_v, names_v = build_dm(predictors_v, predictor_types_v['event_defs'],
                                               predictor_types_v['analog_keys'], bin_size=BIN_SIZE)
            reward_variants[variant_name] = (np.nan_to_num(X_v), names_v)
        for kind in cfg['lick_kernel_variants']:
            opts = {**_reward_lick_opts(cfg), 'lick_kernel': None if kind == 'none' else kind}
            _, predictors_v, predictor_types_v, _, _, _, _, _ = load_nwb_spikes_and_predictors(
                nwb_path, bin_size=BIN_SIZE, nb_of_whisker_kernel=number_of_whisker_kernel,
                kernels_to_split=kernel_split_configs['full'], split_mode='sequential',
                prev_whisker_history=cfg['prev_whisker_history'], merge_whisker=cfg['merge_whisker'], **opts)
            X_v, names_v = build_dm(predictors_v, predictor_types_v['event_defs'],
                                               predictor_types_v['analog_keys'], bin_size=BIN_SIZE)
            reward_variants[f'lick_{kind}'] = (np.nan_to_num(X_v), names_v)
        for basis in [b for b in ['constant', 'cosine'] if f'reward_{b}' in cfg['reward_variant_names']]:
            assert cfg['reward_basis'] is None, f'reward_{basis} variant needs the free-lag reward kernel in full'
            reward_variants[f'reward_{basis}'] = reward_basis_transform(all_Xs, feature_namess, basis, cfg['reward_n_cosine'])
        if cfg['jaw_onset_n_cosine']:
            reward_variants['jaw_onset_cosine'] = reward_basis_transform(all_Xs, feature_namess, 'cosine',
                                                                         cfg['jaw_onset_n_cosine'], prefix='jaw_onset')
        if 'reward_bins' in cfg['reward_variant_names'] and cfg['reward_basis'] is not None:
            # same design as 'full', reward kernel with one free weight per lag
            _, predictors_v, predictor_types_v, _, _, _, _, _ = load_nwb_spikes_and_predictors(
                nwb_path, bin_size=BIN_SIZE, nb_of_whisker_kernel=number_of_whisker_kernel,
                kernels_to_split=kernel_split_configs['full'], split_mode='sequential',
                prev_whisker_history=cfg['prev_whisker_history'], merge_whisker=cfg['merge_whisker'], **_reward_lick_opts(cfg))
            X_v, names_v = build_design_matrix(predictors_v, predictor_types_v['event_defs'],
                                               predictor_types_v['analog_keys'], bin_size=BIN_SIZE)
            reward_variants['reward_bins'] = (np.nan_to_num(X_v), names_v)
        run_info['reward_variants'] = {k: len(v[1]) for k, v in reward_variants.items()}

    if check_only:
        print('check_only: stopping before fitting, nothing written.')
        print(json.dumps(run_info['config'], indent=2, default=str))
        return run_info

    write_run_config(mouse_output_path, commit_hash, run_info)

    # Held-out predictions of the top neurons: dumped per fold to a temporary folder, assembled at the end
    save_top = cfg['save_top_predictions']
    if save_top:
        top_models = list(cfg['top_predictions_models'])
        tmp_pred_dir = pathlib.Path(mouse_output_path, 'models', f'tmp_predictions_{commit_hash}')
        if tmp_pred_dir.exists():
            shutil.rmtree(tmp_pred_dir)  # leftovers of a crashed run with the same hash
        tmp_pred_dir.mkdir(parents=True)
        fold_scores = np.full((spikes.shape[0], cv_outer_folds), np.nan)

    for fold_idx, (trainval_ids, test_ids) in enumerate(outer_kf.split(trials_df.index)):

        print('Fold', fold_idx)

        # ------------------------
        # 1. Split (keep 3D shape)
        # ------------------------
        X_trainval_full = all_Xs[:, trainval_ids, :]
        X_test_full = all_Xs[:, test_ids, :]
        
        spikes_trainval = spikes[:, trainval_ids, :]
        spikes_test = spikes[:, test_ids, :]
        n_features = all_Xs.shape[0]

        # Drift is a per-neuron predictor kept out-of-band from the shared design matrix (see
        # fit_neuron_glm_gaussian) and is included by default in the full/reduced models below.
        # Standardized per-neuron using train-fold statistics only, same convention as the DLC
        # features below - otherwise its raw um scale gets a different effective ridge penalty
        # than the other (already unit-scale or z-scored) predictors.
        drift_trainval = drift_matrix[:, trainval_ids] if drift_matrix is not None else None
        drift_test = drift_matrix[:, test_ids] if drift_matrix is not None else None
        if drift_trainval is not None:
            drift_mean = np.nanmean(drift_trainval, axis=1, keepdims=True)
            drift_std = np.nanstd(drift_trainval, axis=1, keepdims=True)
            drift_std[~(drift_std > 0)] = 1e-6
            drift_trainval = (drift_trainval - drift_mean) / drift_std
            drift_test = (drift_test - drift_mean) / drift_std
            # Units without usable motion (probe missing / skipped): all-zero drift column, so its
            # coefficient is 0 instead of NaNs breaking the fit
            drift_trainval = np.nan_to_num(drift_trainval)
            drift_test = np.nan_to_num(drift_test)
        # ----------------------------------------------
        # 2. Standardize DLC features using train only
        # ----------------------------------------------
        for dlc_idx in range(n_features - N_STANDARDIZED_TAIL, n_features):
            mean = X_trainval_full[dlc_idx].mean()
            std = X_trainval_full[dlc_idx].std()
            if std == 0: std = 1e-6

            X_trainval_full[dlc_idx] = (X_trainval_full[dlc_idx] - mean) / std
            X_test_full[dlc_idx] = (X_test_full[dlc_idx] - mean) / std

        X_trainval_r = X_trainval_full.reshape(n_features, len(trainval_ids), -1)
        X_test_r = X_test_full.reshape(n_features, len(test_ids), -1)
    
        debug = False
        if debug:
            for test_idx in test_ids[10:15]:
                plot_design_matrix_heatmap_single_trial(X_rewards[1], feature_names_rewards[1], trial_index=test_idx, n_bins=n_bins, bin_size=BIN_SIZE)
                plot_design_matrix_vector_single_trial(X_rewards[1], feature_names_rewards[1], trial_index=test_idx, n_bins=n_bins,
                                                       bin_size=BIN_SIZE)
            return

        # -----------------------------------
        # Fit full GLMs using multiprocessing
        # -----------------------------------

        lambdas = np.exp(np.linspace(np.log(cfg['lambda_max']), np.log(cfg['lambda_min']), cfg['n_lambdas'])) # regul. strength hyperparam.
        #lambdas = np.array([0.001, 0.5])

        start_time = time.time()

        feature_namess_with_drift = feature_namess + ['drift'] if drift_matrix is not None else feature_namess

        results = parallel_fit_glms(spikes_trainval=spikes_trainval,
                                    X_trainval=X_trainval_r,
                                    spikes_test=spikes_test,
                                    X_test=X_test_r,
                                    lambdas=lambdas,
                                    n_jobs=n_jobs,
                                    drift_trainval=drift_trainval,
                                    drift_test=drift_test)
        print('GLM fitting complete in ', time.time() - start_time)

        results = sorted(results, key=lambda r: r['neuron_id']) # sort for neuron_id--lambda_opt matching

        # Save mouse results for model type and fold index
        model_res_df = pd.DataFrame(results)
        model_res_df['fold'] = fold_idx
        model_res_df['train_trials'] = [trainval_ids] * len(model_res_df)
        model_res_df['test_trials'] = [test_ids] * len(model_res_df)
        model_res_df['model_name'] = 'full'
        model_res_df['predictors'] = [feature_namess_with_drift] * len(model_res_df)
        describe_cols = ['neuron_id', 'fold', 'train_score', 'test_score', 'train_corr', 'test_corr',
                         'train_mi', 'test_mi']
        print(model_res_df[describe_cols].describe())
        if save_top:
            fold_scores[model_res_df['neuron_id'].values, fold_idx] = model_res_df['test_score'].values
            _dump_fold_test_predictions(tmp_pred_dir, [model_res_df], top_models, fold_idx, test_ids, spikes.shape[2])
        save_model_results(model_res_df, filename='model_full_fold{}'.format(fold_idx), commit_hash = commit_hash, output_dir=mouse_output_path, save_predictions=save_predictions)

        # Add to list of global dat
        # model_res_df_outer.append(model_res_df)  # disabled: already saved above via save_model_results

        # Plot single trial predictions models
        debug=False
        if debug:
            for neuron_id in model_res_df.neuron_id.unique():
                plot_trial_grid_predictions(model_res_df, trials_df, neuron_id, bin_size=BIN_SIZE)
            return


        # --------------------------------------
        # Fit reduced GLMs using multiprocessing
        # --------------------------------------

        # Define reduced encoding model formulations (only those listed in cfg['reduced_models'] are fit)
        REDUCED_MODEL_DEFINITIONS = {
            'whisker_encoding': [f for f in feature_namess if ('whisker_hits_stim' in f) or ('whisker_misses_stim' in f ) or ('whisker_stim' in f)],
            'auditory_encoding': [f for f in feature_namess if 'auditory_stim' in f],
            'reward_encoding': [f for f in feature_namess if 'piezo_reward' in f],
            'jaw_onset_encoding': [f for f in feature_namess if 'jaw_onset' in f],
            'motor_encoding': [f for f in feature_namess if 'dist' in f or 'vel' in f],
            'session_progress_encoding': ['trial_index_scaled'],
            # Only the first post-stimulus bin (lag 0: 0-50 ms after the stimulus) of the stimulus kernels
            'whisker_first_bin': [f for f in feature_namess if ('whisker_hits_stim' in f or 'whisker_misses_stim' in f
                                                                   or 'whisker_stim' in f) and f.endswith('_t+0.00s')],
            'auditory_first_bin': [f for f in feature_namess if 'auditory_stim' in f and f.endswith('_t+0.00s')],
            'sum_rewards': ['sum_reward_scaled'],
            'lick_encoding': [f for f in feature_namess if f.startswith(('lick_piezo_t',) + LICK_KERNEL_PREFIXES)],
            # Licking as a whole: lick kernel + jaw distance (the jaw-onset kernel, before the first lick, stays)
            'licking_encoding': [f for f in feature_namess if f.startswith(('lick_piezo_t',) + LICK_KERNEL_PREFIXES) or f == 'jaw_dist'],
            'whisker_history': [f for f in feature_namess if f == 'prev_whisker_hit' or 'whisker_stim_prevhit' in f],

            # 'block_perf_type' : ['block_perf_type'],
            # 'whisker_move': ['whisker_vel'],
            # 'last_rewards_whisker': ['last_whisker_reward'],
            # 'whisker_hit' : ['whisker_hit'],
            # 'prop_rewards_whiskers': ['prop_past_whisker_rewarded'],
            # 'prop_last_5_whisker': ['whisker_reward_rate_5'],
            # 'cum_rewards_whisker': ['sum_whisker_reward_scaled'],
            # 'all_whisker_progression': ['prev_whisker_reward', 'last_whisker_reward', 'prop_past_whisker_rewarded',
                                        # 'whisker_reward_rate_5', 'sum_whisker_reward_scaled'],
        }
        unknown_reduced = set(cfg['reduced_models']) - set(REDUCED_MODEL_DEFINITIONS)
        if unknown_reduced:
            raise ValueError(f"Unknown reduced models in config: {sorted(unknown_reduced)}")
        reduced_model_names = list(cfg['reduced_models'])
        if cfg['prev_whisker_history'] and 'whisker_history' not in reduced_model_names:
            reduced_model_names.append('whisker_history')
        reduced_models = {name: REDUCED_MODEL_DEFINITIONS[name] for name in reduced_model_names}
        # Get full model params for fair comparison
        full_optimal_lambdas = {neuron_id: lambda_opt for neuron_id, lambda_opt in
                                zip(model_res_df['neuron_id'], model_res_df['lambda_opt'])}
        #
        # Iterate over reduced model formulations
        results_reduced_all = []
        for model_name, features_to_remove in reduced_models.items():
            # Select subset of feature matrix
            X_trainval_reduced, kept_features = get_reduced_matrix(X_trainval_r, feature_namess, features_to_remove)
            X_test_reduced, kept_features = get_reduced_matrix(X_test_r, feature_namess, features_to_remove)


            # Fit GLM with reduced feature set (drift stays in - these ablations remove other predictor groups)
            results_reduced = parallel_fit_glms(spikes_trainval=spikes_trainval,
                                                X_trainval=X_trainval_reduced,
                                                spikes_test=spikes_test,
                                                X_test=X_test_reduced,
                                                lambdas=full_optimal_lambdas,
                                                n_jobs=n_jobs,
                                                drift_trainval=drift_trainval,
                                                drift_test=drift_test)
            results_reduced_df = pd.DataFrame(results_reduced)
            results_reduced_df['fold'] = fold_idx
            results_reduced_df['train_trials'] = [trainval_ids] * len(results_reduced_df)
            results_reduced_df['test_trials'] = [test_ids] * len(results_reduced_df)
            results_reduced_df['model_name'] = model_name
            kept_features_with_drift = list(kept_features) + ['drift'] if drift_matrix is not None else list(kept_features)
            results_reduced_df['predictors'] = [kept_features_with_drift] * len(results_reduced_df)
            results_reduced_all.append(results_reduced_df)

            # Append reduced model to all models
            # model_res_df_outer.append(results_reduced_df)  # disabled: already saved per-fold below

        if drift_matrix is not None:
            # "no_drift" ablation: same design matrix as full, just without the drift column
            results_no_drift = parallel_fit_glms(spikes_trainval=spikes_trainval,
                                                 X_trainval=X_trainval_r,
                                                 spikes_test=spikes_test,
                                                 X_test=X_test_r,
                                                 lambdas=full_optimal_lambdas,
                                                 n_jobs=n_jobs)
            results_no_drift_df = pd.DataFrame(results_no_drift)
            results_no_drift_df['fold'] = fold_idx
            results_no_drift_df['train_trials'] = [trainval_ids] * len(results_no_drift_df)
            results_no_drift_df['test_trials'] = [test_ids] * len(results_no_drift_df)
            results_no_drift_df['model_name'] = 'no_drift'
            results_no_drift_df['predictors'] = [feature_namess] * len(results_no_drift_df)
            results_reduced_all.append(results_no_drift_df)

        # ---------------------------------------------------------------
        # Control draws and single-kernel-split models, per split config:
        # - 'full' (whisker_hits, whisker_misses, auditory_stim all split
        #   together) is already fit above as the 'full' model, so only its
        #   control draws are (re-)fit here.
        # - The other configs split one event type at a time (other two left
        #   un-split); each is fit as its own model plus its own controls.
        # Controls reshuffle which kernel each event lands in, either
        # randomly ('random', a different permutation per draw) or via a
        # deterministic circular roll of the time-sorted events
        # ('linear_shift'), as a baseline for how much of the split model's
        # gain is genuine temporal structure vs. just extra free parameters.
        # ---------------------------------------------------------------
        if whisker_kernels:
            for config_name, model_data in split_model_data.items():
                if config_name != 'full':
                    X_trainval_c = model_data['X'][:, trainval_ids, :].copy()
                    X_test_c = model_data['X'][:, test_ids, :].copy()
                    _standardize_dlc_features(X_trainval_c, X_test_c)

                    results_split_df = _fit_labeled_model(
                        spikes_trainval, X_trainval_c, spikes_test, X_test_c, full_optimal_lambdas, n_jobs,
                        fold_idx, trainval_ids, test_ids, config_name, model_data['feature_names'],
                        drift_trainval=drift_trainval, drift_test=drift_test)
                    results_reduced_all.append(results_split_df)

                for control_kind, draw_key, name_key in (
                    ('random', 'random_Xs', 'random_feature_names'),
                    ('linear_shift', 'shift_Xs', 'shift_feature_names'),
                ):
                    draw_Xs = model_data.get(draw_key, [])
                    draw_feature_names = model_data.get(name_key, [])
                    if not draw_Xs:
                        continue

                    draw_dfs = []
                    for draw_idx, X_draw in enumerate(draw_Xs):
                        X_trainval_draw = X_draw[:, trainval_ids, :].copy()
                        X_test_draw = X_draw[:, test_ids, :].copy()
                        _standardize_dlc_features(X_trainval_draw, X_test_draw)

                        draw_name = (f'{control_kind}_split_{draw_idx}' if config_name == 'full'
                                     else f'{config_name}_{control_kind}_{draw_idx}')
                        draw_df = _fit_labeled_model(
                            spikes_trainval, X_trainval_draw, spikes_test, X_test_draw, full_optimal_lambdas, n_jobs,
                            fold_idx, trainval_ids, test_ids, draw_name, draw_feature_names[draw_idx],
                            drift_trainval=drift_trainval, drift_test=drift_test)
                        results_reduced_all.append(draw_df)
                        draw_dfs.append(draw_df.sort_values('neuron_id').reset_index(drop=True))

                    avg_name = (f'average_{control_kind}' if config_name == 'full'
                                else f'{config_name}_average_{control_kind}')
                    avg_df = _average_draw_dfs(
                        draw_dfs, fold_idx, trainval_ids, test_ids, avg_name, draw_feature_names[0])
                    results_reduced_all.append(avg_df)

        # Whisker-history control: full model with shuffled previous-outcome labels (with drift, like 'full')
        if history_shuffle_Xs:
            shuffle_dfs = []
            for draw_idx, X_sh in enumerate(history_shuffle_Xs):
                X_trainval_sh = X_sh[:, trainval_ids, :].copy()
                X_test_sh = X_sh[:, test_ids, :].copy()
                _standardize_dlc_features(X_trainval_sh, X_test_sh)
                shuffle_df = _fit_labeled_model(
                    spikes_trainval, X_trainval_sh, spikes_test, X_test_sh, full_optimal_lambdas, n_jobs,
                    fold_idx, trainval_ids, test_ids, f'whisker_history_shuffle_{draw_idx}',
                    history_shuffle_feature_names[draw_idx], drift_trainval=drift_trainval, drift_test=drift_test)
                results_reduced_all.append(shuffle_df)
                shuffle_dfs.append(shuffle_df.sort_values('neuron_id').reset_index(drop=True))
            results_reduced_all.append(_average_draw_dfs(
                shuffle_dfs, fold_idx, trainval_ids, test_ids, 'whisker_history_average_shuffle',
                history_shuffle_feature_names[0]))

        # Merge results from all reduced models, then save
        if save_top:
            _dump_fold_test_predictions(tmp_pred_dir, results_reduced_all, top_models, fold_idx, test_ids, spikes.shape[2])
        results_reduced_all_df = pd.concat(results_reduced_all, ignore_index=True)
        save_model_results(results_reduced_all_df, filename='model_reduced_fold{}'.format(fold_idx),
                           commit_hash=commit_hash,
                           output_dir=mouse_output_path,
                           save_predictions=save_predictions)

        results_added_all = []

        # '1whisker_kernel' = unsplit design: identical to 'full' when there is no temporal split
        if whisker_kernels and number_of_whisker_kernel > 1:
          #  for w_idx, X_w in enumerate(all_Xs):
                n_features_w = X.shape[0]
                X_trainval = X[:, trainval_ids, :].copy()
                X_test = X[:, test_ids, :].copy()


                # Standardize DLC features (train only)
                for dlc_idx in range(n_features_w - N_STANDARDIZED_TAIL, n_features_w):
                    mean = X_trainval[dlc_idx].mean()
                    std = X_trainval[dlc_idx].std()
                    if std == 0: std = 1e-6
                    X_trainval[dlc_idx] = (X_trainval[dlc_idx] - mean) / std
                    X_test[dlc_idx] = (X_test[dlc_idx] - mean) / std

                # Fit GLM with reduced feature set
                results_added = parallel_fit_glms(spikes_trainval=spikes_trainval,
                                                    X_trainval=X_trainval,
                                                    spikes_test=spikes_test,
                                                    X_test=X_test,
                                                    lambdas=full_optimal_lambdas,
                                                    n_jobs=n_jobs,
                                                    drift_trainval=drift_trainval, drift_test=drift_test)
                results_added_df = pd.DataFrame(results_added)
                results_added_df['fold'] = fold_idx
                results_added_df['train_trials'] = [trainval_ids] * len(results_added_df)
                results_added_df['test_trials']  = [test_ids] * len(results_added_df)
                results_added_df['model_name'] = str(1) + 'whisker_kernel'
                results_added_df['predictors'] = [list(feature_names) + (['drift'] if drift_trainval is not None else [])] * len(results_added_df)
                results_added_all.append(results_added_df)

                # Append reduced model to all models
                # model_res_df_outer.append(results_added_df)  # disabled: already saved per-fold below

        if reward_kernels:
            print('fitting mulitple reward kernels')
            for k_idx, X_rk in enumerate(X_rewards):
                n_features_rk = X_rk.shape[0]
                X_trainval_rk = X_rk[:, trainval_ids, :].copy()
                X_test_rk = X_rk[:, test_ids, :].copy()

                # Standardize DLC features (train only)
                for dlc_idx in range(n_features_rk - N_STANDARDIZED_TAIL, n_features_rk):
                    mean = X_trainval_rk[dlc_idx].mean()
                    std = X_trainval_rk[dlc_idx].std()
                    if std == 0: std = 1e-6
                    X_trainval_rk[dlc_idx] = (X_trainval_rk[dlc_idx] - mean) / std
                    X_test_rk[dlc_idx] = (X_test_rk[dlc_idx] - mean) / std

                # reshape for GLM
                # X_trainval_rk2d = X_trainval_rk.reshape(X_trainval_rk.shape[0], -1)
                # X_test_rk2d = X_test_rk.reshape(X_test_rk.shape[0], -1)

                results_added = parallel_fit_glms(
                    spikes_trainval=spikes_trainval.reshape(spikes_trainval.shape[0], -1),
                    X_trainval=X_trainval_rk,
                    spikes_test=spikes_test.reshape(spikes_test.shape[0], -1),
                    X_test=X_test_rk,
                    lambdas=full_optimal_lambdas,
                    n_jobs=n_jobs,
                    drift_trainval=drift_trainval, drift_test=drift_test
                )

                results_added_df = pd.DataFrame(results_added)
                results_added_df['fold'] = fold_idx
                results_added_df['train_trials'] = [trainval_ids] * len(results_added_df)
                results_added_df['test_trials'] = [test_ids] * len(results_added_df)
                results_added_df['model_name'] =  '2_reward_kernels'
                results_added_df['predictors'] =  [feature_names_extra] * len(results_added_df)
                results_added_all.append(results_added_df)

                # Append reduced model to all models
                # model_res_df_outer.append(results_added_df)  # disabled: already saved per-fold below

        if add_perf_pred is not None:
            bin_masks = [np.zeros(n_bins, dtype=bool) for _ in range(n_bins)]
            for i in range(n_bins):
                bin_masks[i][i] = True

            for i, Xp in enumerate(X_perfs):
                n_features_p = Xp.shape[0]

                predictor_name = add_perf_pred[i]
                print(predictor_name)
                print(feature_names_perfs[i])
                X_trainval_p = Xp[:, trainval_ids, :].copy()
                X_test_p = Xp[:, test_ids, :].copy()

                # Standardize DLC
                for dlc_idx in range(n_features_p - N_STANDARDIZED_TAIL, n_features_p):
                    mean = X_trainval_p[dlc_idx].mean()
                    std = X_trainval_p[dlc_idx].std()
                    if std == 0: std = 1e-6
                    X_trainval_p[dlc_idx] = (X_trainval_p[dlc_idx] - mean) / std
                    X_test_p[dlc_idx] = (X_test_p[dlc_idx] - mean) / std
                    # Find index of this predictor in the feature names for this perf predictor set
       #         try:
      #              pred_idx = feature_names_perfs[i].index(predictor_name)
     #           except ValueError:
    #                raise ValueError(f"Predictor '{predictor_name}' not found in feature_names_perfs[{i}]")
        #        print(pred_idx)

                # Loop across bins
    #           for bin_idx in range(n_bins):
    #            print(f"Testing predictor {feature_names[i][pred_idx]} restricted to bin {bin_idx}")

                X_trainval_variant = X_trainval_p
                X_test_variant = X_test_p


                results_added = parallel_fit_glms(
                        spikes_trainval=spikes_trainval.reshape(spikes_trainval.shape[0], -1),
                        X_trainval=X_trainval_variant,
                        spikes_test=spikes_test.reshape(spikes_test.shape[0], -1),
                        X_test=X_test_variant,
                        lambdas=full_optimal_lambdas,
                        n_jobs=n_jobs,
                        drift_trainval=drift_trainval, drift_test=drift_test
                    )

                results_added_df = pd.DataFrame(results_added)
                results_added_df['fold'] = fold_idx
                results_added_df['train_trials'] = [trainval_ids] * len(results_added_df)
                results_added_df['test_trials'] = [test_ids] * len(results_added_df)
                results_added_df['model_name'] = f"{add_perf_pred[i]}"
                results_added_df['predictors'] = [list(feature_names_perfs[i])] * len(results_added_df)
                results_added_all.append(results_added_df)
                print(results_added_df[describe_cols].describe())

                    # Append reduced model to all models
                # model_res_df_outer.append(results_added_df)  # disabled: already saved per-fold below

        for variant_name, (X_v, names_v) in reward_variants.items():
            X_trainval_v = X_v[:, trainval_ids, :].copy()
            X_test_v = X_v[:, test_ids, :].copy()
            _standardize_by_name(X_trainval_v, X_test_v, names_v)
            results_v = parallel_fit_glms(spikes_trainval=spikes_trainval, X_trainval=X_trainval_v,
                                          spikes_test=spikes_test, X_test=X_test_v, lambdas=full_optimal_lambdas,
                                          n_jobs=n_jobs, drift_trainval=drift_trainval, drift_test=drift_test)
            results_v_df = pd.DataFrame(results_v)
            results_v_df['fold'] = fold_idx
            results_v_df['train_trials'] = [trainval_ids] * len(results_v_df)
            results_v_df['test_trials'] = [test_ids] * len(results_v_df)
            results_v_df['model_name'] = variant_name
            results_v_df['predictors'] = [list(names_v) + (['drift'] if drift_trainval is not None else [])] * len(results_v_df)
            results_added_all.append(results_v_df)

        if save_top:
            _dump_fold_test_predictions(tmp_pred_dir, results_added_all, top_models, fold_idx, test_ids, spikes.shape[2])
        if results_added_all:  # empty when no added model is configured
            results_added_all_df = pd.concat(results_added_all, ignore_index=True)
            save_model_results(results_added_all_df, filename='model_added_fold{}'.format(fold_idx),
                               commit_hash=commit_hash,
                               output_dir=mouse_output_path,
                               save_predictions=save_predictions)


        debug=False
        if debug:
            for neuron_id in model_res_df.neuron_id.unique():
                plot_trial_grid_predictions(results_reduced_all_df[results_reduced_all_df.model_name=='motor_encoding'], trials_df, neuron_id, bin_size=BIN_SIZE)
                plot_trial_grid_predictions(results_reduced_all_df[results_reduced_all_df.model_name=='motor_encoding'], trials_df, neuron_id, bin_size=BIN_SIZE)
                plot_trial_grid_predictions(results_reduced_all_df[results_reduced_all_df.model_name=='whisker_encoding'], trials_df, neuron_id, bin_size=BIN_SIZE)


    # Save all model results in a single file
    # Disabled: this re-accumulated every fold/model result (incl. raw y_train/y_test arrays) in memory
    # for the whole run just to re-save it combined, which was the main OOM driver. Each fold's results
    # are already saved individually above via save_model_results(). To get a combined file, concatenate
    # the per-fold parquet files from disk afterward instead.
    # model_res_df_outer_df = pd.concat(model_res_df_outer, ignore_index=True)
    # save_model_results(model_res_df_outer_df, filename='model_results_all',commit_hash = commit_hash, output_dir=output_dir)
    #
    # # Inspect predictions for full models across folds
    # neuron_id = 0
    # model_res_df_outer_df_sub = model_res_df_outer_df[model_res_df_outer_df['neuron_id'] == neuron_id]
    # model_res_df_outer_df_sub = model_res_df_outer_df_sub[model_res_df_outer_df_sub['model_name'] == 'full']
    # describe_cols = ['neuron_id', 'fold', 'train_score', 'test_score', 'train_corr', 'test_corr',
    #                  'train_mi', 'test_mi']
    # print(model_res_df_outer_df_sub[describe_cols].describe())

    if save_top:
        run_info['top_predictions'] = _save_top_predictions(
            tmp_pred_dir, pathlib.Path(mouse_output_path, 'models', f'{commit_hash}_top_predictions.pkl'),
            top_models, fold_scores, spikes, trials_df, neurons_ccf, cfg)
        shutil.rmtree(tmp_pred_dir)

    run_info['status'] = 'completed'
    run_info['end_time'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    write_run_config(mouse_output_path, commit_hash, run_info)
    print('Fitting done.')

    if cfg['create_summary']:
        run_info['summary'] = _create_summary_after_fit(nwb_path, mouse_output_path, commit_hash, cfg)
        write_run_config(mouse_output_path, commit_hash, run_info)
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
        # cwd = this file's folder, so the hash is this repo's even if the job runs from elsewhere
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       cwd=os.path.dirname(os.path.abspath(__file__))).decode('ascii').strip()
    except Exception:
        return "unknown"

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nwb', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--n_jobs', type=int, default=20)
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--check_only', action='store_true',
                        help='Build design matrices and run the checks, then stop before fitting (writes nothing)')
    args = parser.parse_args()
    run_unit_glm_pipeline_with_pool(args.nwb, args.out, n_jobs=args.n_jobs, save_predictions=args.save_predictions,
                                    check_only=args.check_only)

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

