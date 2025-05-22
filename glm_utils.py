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

import multiprocessing as mp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyglmnet import GLM, GLMCV
from functools import partial
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Custom imports
import NWB_reader_functions as nwbreader
import allen_utils
import plotting_utils as putils

# Set global variables
BIN_SIZE = 0.05 # seconds


def bin_spike_times(spike_times, start_time, end_time, bin_size):
    bins = np.arange(start_time, end_time + bin_size, bin_size)
    binned, _ = np.histogram(spike_times, bins=bins)
    return binned

def bin_behavior(series_time, series_values, trial_start, trial_end, n_bins):
    bin_edges = np.linspace(trial_start, trial_end, n_bins + 1)
    binned = np.zeros(n_bins)
    digitized = np.digitize(series_time, bin_edges) - 1
    for i in range(n_bins):
        mask = digitized == i
        if np.any(mask):
            binned[i] = np.mean(series_values[mask])
    return binned

def expand_event_predictors(predictor_matrix, window, bin_size):
    start_offset, end_offset = window
    lag_bins = np.arange(np.floor(start_offset / bin_size),
                         np.ceil(end_offset / bin_size) + 1).astype(int)
    n_lags = len(lag_bins)

    n_trials, n_timepoints = predictor_matrix.shape
    unfolded = np.zeros((n_trials, n_timepoints, n_lags))

    for i, lag in enumerate(lag_bins):
        if lag < 0:
            unfolded[:, -lag:, i] = predictor_matrix[:, :n_timepoints + lag]
        elif lag > 0:
            unfolded[:, :n_timepoints - lag, i] = predictor_matrix[:, lag:]
        else:
            unfolded[:, :, i] = predictor_matrix

    return unfolded

def build_design_matrix_old(predictors, kernel_defs, analog_keys, binary_keys, bin_size):
    X_parts = []
    feature_names = []

    for key in kernel_defs:
        unfolded = expand_event_predictors(kernel_defs[key][0], kernel_defs[key][1], bin_size)
        X_parts.append(unfolded)
        lags = unfolded.shape[-1]
        feature_names += [f"{key}_lag{l}" for l in range(lags)]

    for key in analog_keys + binary_keys:
        X_parts.append(predictors[key][..., np.newaxis])  # Add singleton feature dim
        feature_names.append(key)

    X = np.concatenate(X_parts, axis=-1)
    return X, feature_names

def build_design_matrix(predictors, event_defs, analog_keys, bin_size):
    """
    Builds a design matrix X for GLM using predictor dict and event definitions from
    `load_nwb_spikes_and_predictors`.

    Parameters:
    - predictors: dict of predictor arrays, each shape (n_trials, n_bins)
    - event_defs: dict of {name: (event_times, (start_offset, end_offset))}
    - bin_size: duration of one time bin (s)

    Returns:
    - X: np.ndarray, shape (n_trials * n_bins, n_features)
    - feature_names: list of feature column names
    """
    design_cols = []
    feature_names = []

    for name, data in predictors.items():
        if name in event_defs:
            _, (start_offset, end_offset) = event_defs[name]
            offset_bins_pre = int(np.round(abs(start_offset) / bin_size))
            offset_bins_post = int(np.round(end_offset / bin_size))

            for shift in range(-offset_bins_pre, offset_bins_post):
                shifted = np.zeros_like(data)
                if shift < 0:
                    shifted[:, :shift] = data[:, -shift:]
                elif shift > 0:
                    shifted[:, shift:] = data[:, :-shift]
                else:
                    shifted = data.copy()

                col = shifted.flatten()
                design_cols.append(col)
                feature_names.append(f"{name}_t{shift * bin_size:+.2f}s")
        elif name in analog_keys:
            # Z-score analog predictors before adding
            data_reshaped = data.reshape(-1, 1).flatten()
            #scaler = StandardScaler()
            #data_reshaped = scaler.fit_transform(data_reshaped).flatten() #TODO: remove that and do trina-test
            design_cols.append(data_reshaped)
            feature_names.append(name)

        else:
            # Binary or static predictors
            design_cols.append(data.flatten())
            feature_names.append(name)

    # Stack and reshape into a tensor
    X = np.stack(design_cols, axis=1)
    n_trials_bins = X.shape[0]
    n_trials = len(predictors[list(predictors.keys())[0]])
    n_bins = n_trials_bins // n_trials
    X = X.reshape(n_trials, n_bins, -1).transpose((2, 0, 1)) # ( n_features, n_trials, n_bins)
    return X, feature_names

def get_reduced_matrix(X_full, predictor_names, predictors_to_remove):
    keep_mask_ids = [i for i, name in enumerate(predictor_names) if name not in predictors_to_remove]
    kept_features = [name for name in predictor_names if name not in predictors_to_remove]
    return X_full[keep_mask_ids, :, :], kept_features

def compute_mutual_info(y_true, y_pred):
    """
    Compute instant mutual information, in bits per spike, with the spike trains likelihood.
    The gain in predictability provided by the GLM parameters over a homogeneous Poisson process with constant firing intensity.

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


def plot_design_matrix_heatmap_single_trial(X, feature_names, trial_index, n_bins, bin_size=0.05, cmap='viridis'):
    """
    Plots the design matrix as a heatmap for a single trial.

    Parameters:
    - X: np.ndarray, shape (n_trials * n_bins, n_features)
    - feature_names: list of str, feature names
    - trial_index: int, index of the trial to plot
    - n_bins: int, number of time bins per trial
    - bin_size: float, bin size in seconds
    - cmap: str, matplotlib colormap
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
    ax.set_xticks(range(9))
    ax.set_xticklabels([-3,-2,-1,0,1,2,3,4,5], fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_title(f"Design matrix — trial {trial_index}", fontsize=12)

    #cbar = fig.colorbar(im, ax=ax, shrink=0.7, location='bottom', pad=0.15)
    #cbar.set_label('Feature value', fontsize=12)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='1%', pad=0.5)
    plt.colorbar(im, cax=cax, label='Feature value', orientation='horizontal')

    plt.tight_layout()
    plt.show()
    return


def plot_design_matrix_single_trial(X, feature_names, trial_index=0, n_bins=None, bin_size=0.05):
    """
    Plots the design matrix predictors for a single trial as stacked time series.

    Parameters:
    - X: np.ndarray, shape (n_trials * n_bins, n_features)
    - feature_names: list of str, feature names
    - trial_index: int, index of the trial to visualize
    - n_bins: int, number of bins per trial (required)
    - bin_size: float, bin size in seconds
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
    axes[-1].set_xticks(range(9))
    axes[-1].set_xticklabels([-3, -2, -1, 0, 1, 2, 3, 4, 5], fontsize=12)
    axes[-1].set_xlabel('Time (s)', fontsize=12)

    # Adjust figure
    fig.align_ylabels()
    plt.suptitle(f"Design matrix: trial {trial_index}", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.02, top=0.96)
    plt.show()
    return

def remove_outliers_zscore(trace, threshold=10):
    '''Reaplce outliers with NaNs based on z-score.'''
    trace = np.array(trace)
    z = (trace - np.nanmean(trace)) / np.nanstd(trace)
    trace[np.abs(z) > threshold] = np.nan
    return trace

def interpolate_nans(trace):
    '''Linearly interpolate NaNs in the trace.'''
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
    '''Preprocess DLC trace to remove outliers.'''
    trace = remove_outliers_zscore(trace, threshold=10)
    trace = interpolate_nans(trace)
    i = np.argwhere(np.isnan(trace))
    assert not np.isnan(trace).any()

    trace = smooth_trace_median(trace, kernel_size=3)
    assert not np.isnan(trace).any()

    #trace = smooth_trace(trace, sigma=5)
    #trace = smooth_trace_savgol(trace, window_length=11, polyorder=3)
    return trace

def load_nwb_spikes_and_predictors(nwb_path, bin_size=0.05):
    with NWBHDF5IO(nwb_path, mode='r', load_namespaces=True) as io:
        nwbfile = io.read()

        window_bounds_sec = (-3, 5)
        trials_df = nwbfile.trials.to_dataframe()
        trials_df = trials_df[(trials_df['context'] == 'active') & (trials_df['perf'] != 6)].copy()

        trial_starts = trials_df['start_time'].values + window_bounds_sec[0]
        trial_ends = trials_df['start_time'].values + window_bounds_sec[1]

        n_trials = len(trial_starts)
        max_duration = np.max(trial_ends - trial_starts)
        n_bins = int(np.ceil(max_duration / bin_size))

        unit_table = nwbfile.units.to_dataframe()
        unit_table = unit_table.sample(frac=1)
        unit_table = unit_table[unit_table['bc_label']=='good']
        unit_table = unit_table[unit_table['ccf_parent_acronym'].isin(['SSp-bfd', 'SSs'])]
        unit_table = unit_table[unit_table['firing_rate'].astype(float).ge(0.5)]
        unit_table = unit_table[:2]
        unit_table = unit_table[~unit_table['ccf_acronym'].isin(allen_utils.get_excluded_areas())]

        # Use index as new column named "unit_id", then reset
        unit_table['neuron_id'] = unit_table.index
        unit_table.reset_index(drop=True, inplace=True)

        # ------------------
        # Spike Trains
        # ------------------
        spike_array = []
        for unit in unit_table.itertuples():
            binned_trials = []
            for start, end in zip(trial_starts, trial_ends):
                bins = np.arange(start, end + bin_size, bin_size)
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
        trial_idx_scaled = np.arange(n_trials) / (n_trials-1)
        predictors['trial_index_scaled'] = np.tile(trial_idx_scaled[:, None], (1, n_bins))

        stim_type = trials_df['trial_type'].fillna('').values
        trials_df['stimulus'] = trials_df['trial_type'].isin(['whisker_trial', 'auditory_trial'])
        trials_df['rewarded'] = (
            trials_df['stimulus'].astype(int)
            * trials_df['reward_available'].astype(int)
            * trials_df['lick_flag'].astype(int)
        )
        rewarded = trials_df['rewarded'].fillna(0).values

        prev_whisker_reward = np.zeros(n_trials)
        prev_auditory_reward = np.zeros(n_trials)

        for i in range(1, n_trials):
            if rewarded[i - 1] > 0:
                if stim_type[i - 1] == 'whisker_trial':
                    prev_whisker_reward[i] = 1
                elif stim_type[i - 1] == 'auditory_trial':
                    prev_auditory_reward[i] = 1

        predictors['prev_whisker_reward'] = np.tile(prev_whisker_reward[:, None], (1, n_bins))
        predictors['prev_auditory_reward'] = np.tile(prev_auditory_reward[:, None], (1, n_bins))
        binary_keys ={
            'trial_index_scale':'trial_index_scaled',
            'prev_whisker_reward':'prev_whisker_reward',
            'prev_auditory_reward':'prev_auditory_reward'
        }

        # Event-based predictors (rasterized kernels will be applied later)
        def rasterize_event(event_times, first_only=False):
            """
            Rasterizes event times into a binary trial-by-time-bin matrix.

            Parameters:
            - event_times (array): 1D array of timestamps (e.g., lick times).
            - first_only (bool): If True, only the first event per trial is rasterized.

            Returns:
            - matrix (n_trials, n_bins): Binary matrix with 1s marking event bins.
            """
            matrix = np.zeros((n_trials, n_bins))
            for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
                bins = np.linspace(start, end, n_bins + 1)
                trial_events = event_times[(event_times >= start) & (event_times < end)]

                if first_only:
                    trial_events = [t for t in trial_events if
                                    t >= (start + abs(window_bounds_sec[0]))]  # keep dlc licks after trial start
                    if len(trial_events) > 0:
                        first_trial_event = trial_events[0]
                        idx = np.digitize(first_trial_event, bins) - 1
                        if 0 <= idx < n_bins:
                            matrix[i, idx] = 1
                else:
                    if len(trial_events) > 0:
                        idxs = np.digitize(trial_events, bins) - 1
                        for idx in idxs:
                            if 0 <= idx < n_bins:
                                matrix[i, idx] = 1
            return matrix

        video_start_time = nwbfile.processing['behavior']['BehavioralTimeSeries'].time_series['whisker_angle'].timestamps[0]

        def get_events(key):
            return (
                nwbfile.processing['behavior']['BehavioralEvents'].time_series[key].data[:],
                nwbfile.processing['behavior']['BehavioralEvents'].time_series[key].timestamps[:] #TODO: add delay between camera and session
            )

        # Get lick onset time
        piezo_licks = np.array(list(get_events('piezo_lick_times')[1]))
        jaw_dlc_licks = np.array(list(get_events('jaw_dlc_licks')[1])) / 200 + video_start_time
        tongue_dlc_licks = np.array(list(get_events('tongue_dlc_licks')[1])) / 200 + video_start_time

        try:
            auditory_times = list(get_events('auditory_hit_trial')[1]) + list(get_events('auditory_miss_trial')[1])
        except:
            auditory_times = list(get_events('auditory_hit_trial')[1])
        auditory_times = np.array(sorted(auditory_times))

        whisker_times = list(get_events('whisker_hit_trial')[1]) + list(get_events('whisker_miss_trial')[1])
        whisker_times = np.array(sorted(whisker_times))

        # Define kernels for single trial events
        event_defs = {
            'dlc_lick_onset': (tongue_dlc_licks, (-0.2, 0.5)),
            'auditory_stim': (auditory_times, (-0.2, 0.5)),
            'whisker_stim': (whisker_times, (-0.2, 0.5)),
        }

        for name, (times, _) in event_defs.items():
            if name=='dlc_lick_onset':
                predictors['dlc_lick_onset'] = rasterize_event(times, first_only=True)
            else:
                predictors[name] = rasterize_event(times, first_only=False)

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
            'whisker_angle': 'whisker_angle',
            #'whisker_vel': 'whisker_velocity'
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


        for short_key, long_key in analog_keys.items():
            data, times = get_series(long_key)
            # Preprocess data to remove outliers
            data = preprocess_dlc_trace(data)
            assert np.isfinite(data.shape).all()
            predictors[short_key] = bin_behavior(data, times)

        # Compute norm of nose movement
        predictors['nose_dist'] = np.sqrt(predictors['side_nose_dist']**2 + predictors['top_nose_dist']**2)
        #predictors['nose_vel'] = np.gradient(predictors['nose_dist'], bin_size, axis=1)
        analog_keys['nose_dist'] = 'nose_distance'
        #analog_keys['nose_vel'] = 'nose_velocity'
        predictors.pop('side_nose_dist')
        #predictors.pop('side_nose_vel')
        predictors.pop('top_nose_dist')
        #predictors.pop('top_nose_vel')
        #TODO: calculate movement rather than distance, for whisker...

        # Prepare outputs
        predictor_types ={'binary_keys': binary_keys,
                         'analog_keys': analog_keys,
                         'event_defs': event_defs}
        return spike_array, predictors, predictor_types, n_bins, bin_size


def fit_neuron_glm(neuron_id, spikes_trainval, X_trainval, spikes_test, X_test, lambdas):
    """
    Fit a GLM to the data for a single neuron, in multiprocessing context.
    :return: dict, results of the GLM fit to be appended in Pool

    """
    # Fitting parameters, N numbers
    cv_folds = 2
    n_features = X_trainval.shape[0]
    n_bins = X_trainval.shape[2]

    # Select unit spikes, and reshape data for GLM formulation
    y_trainval = spikes_trainval[neuron_id].reshape(spikes_trainval.shape[1], -1).flatten()  # (n_trials * n_bins,)
    X_trainval = X_trainval.transpose(1,2,0).reshape(-1, n_features)
    y_test = spikes_test[neuron_id].reshape(spikes_test.shape[1], -1).flatten()  # (n_trials * n_bins,)
    X_test = X_test.transpose(1,2,0).reshape(-1, n_features) # (n_trials * n_bins, n_predictors)

    #best_lambda = None #Note: CV without warm restart and regularization paths
    #best_score = -np.inf
    #kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    #for lam in lambdas:
    #    fold_scores = []
    #    for train_idx, val_idx in kf.split(X_trainval):
    #        X_train = X_trainval[train_idx]
    #        y_train = y_trainval[train_idx]
    #        X_val = X_trainval[val_idx]
    #        y_val = y_trainval[val_idx]
#
    #        glm = GLM(distr='poisson',
    #                  score_metric='pseudo_R2',
    #                  fit_intercept=False,
    #                  alpha=0.0,
    #                  reg_lambda=lam,
    #                  solver='cdfast',
    #                  verbose=False,
    #                  random_state=42)
    #        glm.fit(X_train, y_train)
    #        y_pred = glm.predict(X_val)
    #        ll = -np.mean(y_val * np.log(y_pred + 1e-6) - y_pred)  # Negative log-likelihood
    #        fold_scores.append(-ll)
#
    #    mean_score = np.mean(fold_scores)
    #    if mean_score > best_score:
    #        best_score = mean_score
    #        best_lambda = lam

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
                    verbose=True,
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
    glm_final.fit(X_trainval, y_trainval)
    print('Feature weights', glm_final.beta_, glm_final.beta0_)

    # -------------------------------------------
    # Evaluate final model on train and test sets
    # -------------------------------------------

    # Train and test scores (pseudo-R2)
    y_trainval_pred = glm_final.predict(X_trainval)
    train_score = glm_final.score(X_trainval, y_trainval)
    y_test_pred = glm_final.predict(X_test)
    test_score = glm_final.score(X_test, y_test)

    # Compute log-likelihoods
    train_ll = np.sum(y_trainval * np.log(y_trainval_pred) - y_trainval_pred)
    test_ll = np.sum(y_test * np.log(y_test_pred) - y_test_pred)
    #train_ll = -np.mean(y_trainval * np.log(glm_final.predict(X_trainval) + 1e-6) - glm_final.predict(X_trainval)) #TODO: not sure it's correct
    #test_ll = -np.mean(y_test * np.log(y_test_pred + 1e-6) - y_test_pred) #TODO: not sure it's correct, sum instead of mean

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
        'y_test': y_test,
        'y_pred': y_test_pred,
        'test_corr': test_corr,
        'test_mi': test_mi,
        'n_bins': n_bins,
    }

    return result


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

def save_model_input_output(X, spikes, feature_names, output_dir):
    data_path = pathlib.Path(output_dir, 'data')
    data_path.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(data_path, 'data.pkl'), 'wb') as f:
        pickle.dump({'input': X, 'output': spikes, 'feature_names': feature_names}, f)

    print('Saved input predictor data:', X.shape)
    print('Saved neural output spike data:', spikes.shape)
    return

def load_model_input_output(output_dir):
    data_path = pathlib.Path(output_dir, 'data')
    with open(os.path.join(data_path,'data.pkl'), 'rb') as f:
        data = pickle.load(f)
    X = data['input']
    spikes = data['output']
    feature_names = data['feature_names']
    return X, spikes, feature_names

def save_model_results(result_df, filename, output_dir):
    result_path = pathlib.Path(output_dir, 'models')
    result_path.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(os.path.join(result_path, '{}_results.parquet'.format(filename)))
    print('Saved model results in: ', result_path)
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

def run_unit_glm_pipeline_with_pool(nwb_path, output_dir, n_jobs=10):
    print('GLM fitting for:', pathlib.Path(nwb_path).name)

    # -----------------------------------
    # Load and prepare data from NWB file
    # -----------------------------------

    # Get trial table, input and output formatted for GLM, list of predictor types
    trials_df = nwbreader.get_trial_table(nwb_path)
    trials_df = trials_df[(trials_df['context'] == 'active') &(trials_df['perf'] != 6)].copy()
    trials_df = trials_df.reset_index(drop=True)

    #try:
    #    X, spikes, feature_names = load_model_input_output(output_dir)
    #

    #  Get trial table, input and output formatted for GLM, list of predictor types
    spikes, predictors, predictor_types, n_bins, bin_size = load_nwb_spikes_and_predictors(nwb_path, bin_size=BIN_SIZE)
    event_defs = predictor_types['event_defs']
    analog_keys = predictor_types['analog_keys']
    binary_keys = predictor_types['binary_keys']

    # Build design matrix for entire dataset
    X, feature_names = build_design_matrix(predictors, event_defs, analog_keys, bin_size)
    save_model_input_output(X, spikes, feature_names, output_dir)

    debug=False
    if debug:
        plot_design_matrix_heatmap_single_trial(X, feature_names, trial_index=6, n_bins=n_bins, bin_size=bin_size)
        plot_design_matrix_single_trial(X, feature_names, trial_index=6, n_bins=n_bins, bin_size=bin_size)
        plot_design_matrix_heatmap_single_trial(X, feature_names, trial_index=10, n_bins=n_bins, bin_size=bin_size)
        plot_design_matrix_single_trial(X, feature_names, trial_index=10, n_bins=n_bins, bin_size=bin_size)
        plot_design_matrix_heatmap_single_trial(X, feature_names, trial_index=67, n_bins=n_bins, bin_size=bin_size)
        plot_design_matrix_single_trial(X, feature_names, trial_index=67, n_bins=n_bins, bin_size=bin_size)
        plot_design_matrix_heatmap_single_trial(X, feature_names, trial_index=86, n_bins=n_bins, bin_size=bin_size)
        plot_design_matrix_single_trial(X, feature_names, trial_index=86, n_bins=n_bins, bin_size=bin_size)

    # ---------------------------------------
    # Train/test data cross-validation splits
    # ---------------------------------------
    model_res_df_outer = []
    cv_outer_folds = 5
    outer_kf = KFold(n_splits=cv_outer_folds, shuffle=True, random_state=42)
    for fold_idx, (trainval_ids, test_ids) in enumerate(outer_kf.split(trials_df.index)):

        # Get data splits
        X_trainval, X_test = X[:,trainval_ids,:], X[:,test_ids,:]
        spikes_trainval, spikes_test = spikes[:,trainval_ids,:], spikes[:,test_ids,:]

        # Reshape and keep continuous predictors only
        X_trainval = X_trainval.reshape(X_trainval.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        X_trainval_dlc = X_trainval[:-4,:]
        X_test_dlc = X_test[:-4,:]

        # Normalize test data using train statistics
        train_mean = X_trainval_dlc.mean(axis=1, keepdims=True)
        train_std = X_trainval_dlc.std(axis=1, keepdims=True) + 1e-8  # avoid division by zero
        X_trainval_dlc_scaled = (X_trainval_dlc - train_mean) / train_std
        X_test_dlc_scaled = (X_test_dlc - train_mean) / train_std

        # Update and reshape data
        X_trainval[:-4,:] = X_trainval_dlc_scaled
        X_test[:-4,:] = X_test_dlc_scaled
        X_trainval = X_trainval.reshape(X_trainval.shape[0], len(trainval_ids), -1) # reshape
        X_test = X_test.reshape(X_test.shape[0], len(test_ids), -1)

        # -----------------------------------
        # Fit full GLMs using multiprocessing
        # -----------------------------------

        lambdas = np.exp(np.linspace(np.log(0.5), np.log(1e-5), 5)) # regul. strength hyperparam.
        lambdas = np.array([0.01])
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
        print(model_res_df.describe())
        save_model_results(model_res_df, filename='model_full_fold{}'.format(fold_idx), output_dir=output_dir)

        # Add to list of model folds
        model_res_df_outer.append(model_res_df)

        # Plot single trial predictions models
        debug=True
        if debug:
            for neuron_id in model_res_df.neuron_id.unique():
                plot_trial_grid_predictions(model_res_df, trials_df, neuron_id, bin_size)


        # --------------------------------------
        # Fit reduced GLMs using multiprocessing
        # --------------------------------------

        # Define reduced encoding model formulations
        reduced_models = {
            'whisker_encoding': [f for f in feature_names if 'whisker_stim_time' in f],
            'auditory_encoding': [f for f in feature_names if 'auditory_stim_time' in f],
            'whisker_reward_encoding': [f for f in feature_names if 'prev_whisker_reward' in f],
            'auditory_reward_encoding': [f for f in feature_names if 'prev_auditory_reward' in f],
            'lick_onset_encoding': [f for f in feature_names if 'dlc_lick_onset' in f],
            'motor_encoding': [f for f in feature_names if 'dist' in f or 'vel' in f],
            'session_progress_encoding': ['trial_index_scaled'],
        }

        # Get full model params for fair comparison
        full_optimal_lambdas = {neuron_id:lambda_opt for neuron_id, lambda_opt in zip(model_res_df['neuron_id'], model_res_df['lambda_opt'])}

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

            # Append reduced model to all models
            model_res_df_outer.append(results_reduced_df)

        # Merge results from all reduced models, then save
        results_reduced_all_df = pd.concat(results_reduced_all, ignore_index=True)
        print(results_reduced_all_df.head())
        save_model_results(results_reduced_all_df, filename='model_reduced_fold'.format(fold_idx), output_dir=output_dir)

        model_res_df_outer.append(results_reduced_all_df)

        # Merge full and reduced results
        #model_res_df = pd.concat([model_res_df, results_reduced_all], ignore_index=True)

        debug=True
        if debug:
            for neuron_id in model_res_df.neuron_id.unique():
                plot_trial_grid_predictions(results_reduced_all_df[results_reduced_all_df.model_name=='motor_encoding'], trials_df, neuron_id)
                plot_trial_grid_predictions(results_reduced_all_df[results_reduced_all_df.model_name=='whisker_encoding'], trials_df, neuron_id)


    # Save all model results in a single file
    model_res_df_outer_df = pd.concat(model_res_df_outer, ignore_index=True)
    save_model_results(model_res_df_outer_df, filename='model_results_all', output_dir=output_dir)

    # Inspect predictions for full models across folds
    neuron_id = 0
    model_res_df_outer_df_sub = model_res_df_outer_df[model_res_df_outer_df['neuron_id'] == neuron_id]
    model_res_df_outer_df_sub = model_res_df_outer_df_sub[model_res_df_outer_df_sub['model_name'] == 'full']
    print(model_res_df_outer_df_sub.describe())

    print('Fitting done.')
    return


def plot_single_neuron_predictions(neuron_id, y_test, y_pred, n_bins, n_trials_to_plot=20, trial_spacing=20, bin_size=0.05):
    # Reshape data to (n_trials, n_bins)
    n_trials = y_pred.shape[0] // n_bins
    y_test = y_test.reshape(n_trials, n_bins)
    y_pred = y_pred.reshape(n_trials, n_bins)

    plt.figure(figsize=(12, 4))
    current_x = 0

    for i in range(n_trials_to_plot):
        true_spikes = y_test[i,:]
        pred_spikes = y_pred[i,:]

        x = np.arange(n_bins) * bin_size + current_x

        plt.plot(x, pred_spikes, color='red', label='Model' if i == 0 else None)
        plt.step(x, true_spikes, where='mid', color='black', alpha=0.6, label='Data' if i == 0 else None)

        # Move x start position forward, including spacing
        current_x += (n_bins + trial_spacing) * bin_size

    plt.xlabel('Time (s)')
    plt.ylabel('Spike count')
    title_txt = 'Unit ID: {}'.format(neuron_id)
    plt.title(title_txt)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return


def plot_trial_grid_predictions(results_df, trial_table, neuron_id, bin_size):

    # Plotting params
    n_rows, n_cols = 5, 5
    trials_to_plot = min(n_rows*n_cols, len(trial_table))

    # Get neuron results
    results_df_sub = results_df[results_df['neuron_id'] == neuron_id]
    y_test = results_df_sub['y_test'].values[0]
    y_pred = results_df_sub['y_pred'].values[0]
    n_bins = results_df_sub['n_bins'].values[0]

    # Format data into (n_trials, n_bins)
    n_trials = y_pred.shape[0] // n_bins
    y_test = y_test.reshape(n_trials, n_bins)
    y_pred = y_pred.reshape(n_trials, n_bins)

    # Order trial temporally
    test_trial_ids = results_df_sub['test_trials'].values[0]
    test_trial_id_order = np.argsort(test_trial_ids)
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
        ax.set_ylim(0, 5)
        ax.set_ylabel('Spikes', fontsize=10)
        ax.set_yticks([0, 5])
        ax.set_yticklabels([0, 5], fontsize=10)
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



if __name__ == '__main__':

    import argparse

    #parser = argparse.ArgumentParser()
    #parser.add_argument('--nwb', type=str, required=True)
    #parser.add_argument('--out', type=str, required=True)
    #parser.add_argument('--n_jobs', type=int, default=4)
    #args = parser.parse_args()
    #run_pipeline_with_pool(args.nwb, args.out, n_jobs=args.n_jobs)

    experimenter = 'Axel_Bisi'
    output_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'results')
    root_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'NWBFull')

    mouse_ids = ['AB131']

    for mouse_id in mouse_ids:
        nwb_file_list = [f for f in os.listdir(root_path) if mouse_id in f ]
        nwb_files_to_process = []
        # Find file with neural recordings
        if nwb_file_list:
            for nwb_file in nwb_file_list:
                nwb_path = os.path.join(root_path, nwb_file)
                # Check if NWB content has units
                units = nwbreader.get_unit_table(nwb_path)
                if units is not None:
                    nwb_files_to_process.append(nwb_path)

        for nwb_file in nwb_files_to_process:
            nwb_file = pathlib.Path(nwb_file)
            session_id = nwb_file.stem
            mouse_output_path = pathlib.Path(output_path, mouse_id, 'whisker_0', 'unit_glm')
            mouse_output_path.mkdir(parents=True, exist_ok=True)

            run_unit_glm_pipeline_with_pool(nwb_file, mouse_output_path, n_jobs=10)

