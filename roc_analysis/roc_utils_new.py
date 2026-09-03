#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_spikes_analysis
@file: roc_utils.py
@time: 1/10/2025 1:43 PM
"""

# Imports
import os
import pathlib
import json
import datetime
import numpy as np
import pandas as pd
import tqdm as tqdm
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import NWB_reader_functions as nwb_reader
import neural_utils_old


N_WORKERS = 100
N_PERMUTATIONS = 1000
ALPHA = 0.05

# ----------------------------------------------------------------------------
# Fixed analysis windows (seconds, relative to event onset), hoisted to
# module level so both extract_spike_data() and select_spike_counts() can
# reference the same durations for single-trial baseline subtraction.
# ----------------------------------------------------------------------------
PRE_WINDOW = (-0.05, -0.015)
POST_WINDOW = (0.005, 0.035)
SPONT_LICK_WINDOW = (-0.4, -0.2, 0.0, 0.2)
BASELINE_WINDOW = (-1.0, -0.015)

PRE_DURATION = PRE_WINDOW[1] - PRE_WINDOW[0]
POST_DURATION = POST_WINDOW[1] - POST_WINDOW[0]
# spontaneous_licks uses its own (wider) pre/post window widths
SPONT_PRE_DURATION = SPONT_LICK_WINDOW[1] - SPONT_LICK_WINDOW[0]
SPONT_POST_DURATION = SPONT_LICK_WINDOW[3] - SPONT_LICK_WINDOW[2]
BASELINE_DURATION = BASELINE_WINDOW[1] - BASELINE_WINDOW[0]

TIME_WINDOWS = {
    'pre': PRE_WINDOW,
    'post': POST_WINDOW,
    'spontaneous_licks': SPONT_LICK_WINDOW,
    'baseline_pre': BASELINE_WINDOW,
}

# Event types for which an extended pre-event baseline window is computed.
# spontaneous_licks is now included: baseline is the extended window before
# the lick itself, same as for other events.
BASELINE_EVENTS = {
    'whisker', 'auditory', 'lick_trial', 'no_lick_trial',
    'whisker_hit', 'whisker_miss', 'auditory_hit', 'auditory_miss',
    'correct_rejection', 'spontaneous_licks',
}

# ----------------------------------------------------------------------------
# PSTH sanity-check plotting parameters.
# ----------------------------------------------------------------------------
PSTH_BIN_WIDTH = 0.010   # 10 ms sliding bin
PSTH_STRIDE = 0.010      # 2 ms stride between bin centers
PSTH_WINDOW_STIM = (-0.200, 0.500)   # around stimulus/trial-onset ('start_time') events
PSTH_WINDOW_LICK = (-0.350, 0.350)   # around lick-aligned events (spontaneous_licks)
# Shared window for panels that overlay a stim-aligned and a lick-aligned
# condition: the intersection of PSTH_WINDOW_STIM and PSTH_WINDOW_LICK, so
# both conditions have real extracted data across the full displayed range.
PSTH_WINDOW_MIXED = (-0.200, 0.350)

# Window used to baseline-correct every displayed PSTH: for each trial, its
# own mean rate in this pre-event window is subtracted from that trial's
# full rate curve before averaging across trials (and then across neurons),
# so panels show change-from-baseline rather than raw firing rate. Valid
# for both stim- and lick-aligned windows (-0.200 s sits inside all of
# PSTH_WINDOW_STIM, PSTH_WINDOW_LICK, and PSTH_WINDOW_MIXED).
PSTH_BASELINE_WINDOW = (-0.200, -0.010)

# Modality colors used throughout the sanity-check plots.
WHISKER_COLOR = '#ffb236'
AUDITORY_COLOR = '#322fe0'
# Colors for conditions that aren't tied to a stimulus modality (Correct
# Rejection, lick/no-lick trials, spontaneous licks).
NEUTRAL_COLORS = ['#4daf4a', '#999999']

# Piezo/electrical artifact around whisker stimulus delivery: raw spikes in
# this window are replaced by a Poisson spike train, using the baseline rate
# (estimated from raw, unbinned spike times in WHISKER_ARTIFACT_BASELINE_WINDOW)
# as lambda. Applied ONCE, globally, to every unit's full spike train for
# every whisker stimulus onset in the session (see correct_neuron_spike_train
# and process_nwb_tables) — not per analysis type or per PSTH panel — so
# every downstream consumer (ROC counting and PSTH extraction alike)
# automatically works on corrected spike times, regardless of which
# trial-type grouping (hit, miss, lick_trial, no_lick_trial, choice, ...)
# ends up referencing a given whisker trial's onset.
WHISKER_ARTIFACT_WINDOW = (-0.010, 0.006)
WHISKER_ARTIFACT_BASELINE_WINDOW = (-0.200, -0.010)

# (event_type -> (contexts, window)) for every event PSTH sanity-check plots
# need. 'whisker'/'auditory' contexts collapse to ('active',) at runtime for
# mice without a passive context. No per-event artifact-correction flag here
# anymore: correction is already baked into spike_times by process_nwb_tables.
PSTH_EVENT_SPECS = {
    'whisker':           (('passive_pre', 'passive_post', 'active'), PSTH_WINDOW_STIM),
    'auditory':          (('passive_pre', 'passive_post', 'active'), PSTH_WINDOW_STIM),
    'whisker_hit':       (('active',), PSTH_WINDOW_STIM),
    'whisker_miss':      (('active',), PSTH_WINDOW_STIM),
    'auditory_hit':      (('active',), PSTH_WINDOW_STIM),
    'auditory_miss':     (('active',), PSTH_WINDOW_STIM),
    'correct_rejection': (('active',), PSTH_WINDOW_STIM),
    'lick_trial':        (('active',), PSTH_WINDOW_STIM),
    'no_lick_trial':     (('active',), PSTH_WINDOW_STIM),
    'spontaneous_licks': (('',), PSTH_WINDOW_LICK),
}


def correct_neuron_spike_train(spike_times, whisker_onset_times, rng=None):
    """
    Apply the whisker stimulus-onset artifact correction directly to a
    neuron's full, session-wide raw spike train, once, for EVERY whisker
    stimulus onset in the session — regardless of which trial-type
    categorization (hit, miss, lick_trial, no_lick_trial, choice, passive,
    ...) later references that time. For each onset, spikes in
    WHISKER_ARTIFACT_WINDOW are replaced by a Poisson spike train whose
    rate is estimated from that same onset's raw spike times in
    WHISKER_ARTIFACT_BASELINE_WINDOW. Returns a corrected, sorted
    spike_times array covering the whole session.
    """
    if rng is None:
        rng = np.random.default_rng()
    spike_times = np.asarray(spike_times)
    if len(whisker_onset_times) == 0 or len(spike_times) == 0:
        return spike_times

    art_start, art_end = WHISKER_ARTIFACT_WINDOW
    bl_start, bl_end = WHISKER_ARTIFACT_BASELINE_WINDOW
    art_duration = art_end - art_start
    bl_duration = bl_end - bl_start

    remove_mask = np.zeros(len(spike_times), dtype=bool)
    synthetic_spikes = []

    for t in whisker_onset_times:
        bl_lo = np.searchsorted(spike_times, t + bl_start, side='left')
        bl_hi = np.searchsorted(spike_times, t + bl_end, side='left')
        rate = (bl_hi - bl_lo) / bl_duration

        art_lo = np.searchsorted(spike_times, t + art_start, side='left')
        art_hi = np.searchsorted(spike_times, t + art_end, side='left')
        remove_mask[art_lo:art_hi] = True

        n_synthetic = rng.poisson(rate * art_duration)
        if n_synthetic > 0:
            synthetic_spikes.append(rng.uniform(t + art_start, t + art_end, size=n_synthetic))

    kept = spike_times[~remove_mask]
    corrected = np.concatenate([kept] + synthetic_spikes) if synthetic_spikes else kept
    return np.sort(corrected)


def process_nwb_tables(nwb_file, apply_artifact_correction=True):
    """
    Process unit and trial table from a NWB file.
    :param nwb_file: path to NWB file
    :param apply_artifact_correction: if True (default), correct every
        unit's spike train once for the whisker stimulus-onset artifact
        (see correct_neuron_spike_train), using EVERY whisker trial in the
        session (any context, hit or miss) — not just the ones a given
        downstream analysis happens to select. This runs here so every
        consumer (extract_spike_data's counting and
        extract_psth_spike_times's raw extraction) automatically works on
        corrected spike_times without needing its own correction logic.
    :return:
    """
    # Convert NWB units and trials tables into Pandas DataFrames
    units = nwb_file.units.to_dataframe()
    trials = nwb_file.trials.to_dataframe()
    units['neuron_id'] = units.index # note before any filtering

    # Keep well-isolated units with a valid brain region label
    #units = units[(units['bc_label'].isin(['good','mua']))]
    units = neural_utils.convert_electrode_group_object_to_columns(units)

    # Keep fewer columns only
    ccf_cols = [c for c in units.columns if "ccf" in c]
    columns_to_keep = ["electrode_group", "cluster_id", "neuron_id", "spike_times", "firing_rate", "target_region"] + ccf_cols
    units = units[columns_to_keep]

    # Use index as new column named "neuron_id", then reset
    #units.reset_index(drop=True, inplace=True)

    # If context is only NaNs, set to 'active' for all trials
    if trials['context'].isnull().all():
        trials['context'] = 'active'
    # Mouse with passive and active periods
    else:
        # Separate passive trials from pre vs post trials
        trials_mid_index = len(trials) // 2  # find middle of session
        trials['context'] = trials.apply(lambda row:
                                         'active' if row['context'] == 'active' else
                                         ('passive_pre' if row['context'] == 'passive' and row.name < trials_mid_index else
                                          'passive_post'), axis=1)

    if apply_artifact_correction:
        # ALL whisker trials, any context, hit or miss — not filtered by
        # what a particular downstream analysis will later select.
        whisker_onset_times = trials[trials['whisker_stim'] == 1]['start_time'].values
        rng = np.random.default_rng()
        units = units.copy()
        units['spike_times'] = units['spike_times'].apply(
            lambda st: correct_neuron_spike_train(st, whisker_onset_times, rng)
        )

    return units, trials


def count_spikes_in_window(spike_times, start_time, end_time):
    """ Count spikes in a given time window. """
    return len(spike_times[(spike_times >= start_time) & (spike_times <= end_time)])

def filter_times_by_proximity(event_times, reference_times, proximity=1):
    """
    Filter out event times based on proximity to reference times.
    :param event_times: list of event times
    :param reference_times: list of reference times to avoid
    :param proximity: in seconds, minimum proximity to reference times
    :return:
    """
    reference_times = np.array(reference_times)
    return [time for time in event_times if not np.any((time - reference_times > 0) & (time - reference_times <= proximity))]

def filter_lick_times(lick_times, interval=1, **stimuli):
    """
    Filter out lick times based on interval spacing and proximity to stimulus events.
    :param lick_times: list of lick times
    :param interval: in seconds, minimum interval between licks
    :param stimuli: list of stimulus times
    :return:
    """
    filtered = [lick_times[0]]
    for i in range(1, len(lick_times)):
        if lick_times[i] - filtered[-1] > interval:
            filtered.append(lick_times[i])

    for stim_type, stim_times in stimuli.items():
        filtered = filter_times_by_proximity(filtered, stim_times)

    return filtered

def get_filtered_lick_times(nwbfile, interval=1):
    """ Extract and filter piezo lick times from NWB data. """
    behavior = nwbfile.processing['behavior']
    events = behavior.data_interfaces['BehavioralEvents']

    # Get piezo lick times
    piezo_lick_times = events.time_series['piezo_lick_times'].data[:]

    # Check available event types
    # Note: some may be missing, e.g. auditory_miss_trial (or whisker_hit_trial)
    # Gather all that are present in events as time_series
    stim_time_series = [key for key in events.time_series.keys() if 'trial' in key]
    stim_time_series = [key for key in stim_time_series if 'auditory' in key or 'whisker' in key]
    stim_times = {key: events.time_series[key].data[:] for key in stim_time_series}

    # Remove lick times within interval of stimulus events
    filtered_lick_times = filter_lick_times(piezo_lick_times, interval, **stim_times)

    return filtered_lick_times


def extract_event_times(nwb_file, event_type='whisker', context='passive', has_context=True):
    """ Extract event times based on stimulus type and context. """
    _, trials = process_nwb_tables(nwb_file)

    if event_type == 'spontaneous_licks':
        event_times = get_filtered_lick_times(nwb_file)
        return event_times

    elif event_type in ['whisker', 'auditory']:
        condition = trials[event_type + '_stim'] == 1
        if has_context:
            condition &= trials['context'] == context #add context condition on top of stimulus type

        # Keep Hits only in active context
        #condition &= trials['lick_flag'] == 1 if context == 'active' else True
        event_times = trials[condition]['start_time'].values

    elif event_type == 'lick_trial':
        condition = trials['lick_flag'] == 1
        if has_context:
            condition &= trials['context'] == context
        event_times = trials[condition]['start_time'].values

    elif event_type == 'no_lick_trial':
        condition = trials['lick_flag'] == 0
        if has_context:
            condition &= trials['context'] == context
        event_times = trials[condition]['start_time'].values

    elif event_type == 'whisker_hit':
        condition = trials['whisker_stim'] == 1
        condition &= trials['lick_flag'] == 1
        if has_context:
            condition &= trials['context'] == context
        event_times = trials[condition]['start_time'].values

    elif event_type == 'whisker_miss':
        condition = trials['whisker_stim'] == 1
        condition &= trials['lick_flag'] == 0
        if has_context:
            condition &= trials['context'] == context
        event_times = trials[condition]['start_time'].values

    elif event_type == 'auditory_hit':
        condition = trials['auditory_stim'] == 1
        condition &= trials['lick_flag'] == 1
        if has_context:
            condition &= trials['context'] == context
        event_times = trials[condition]['start_time'].values

    elif event_type == 'auditory_miss':
        condition = trials['auditory_stim'] == 1
        condition &= trials['lick_flag'] == 0
        if has_context:
            condition &= trials['context'] == context
        event_times = trials[condition]['start_time'].values

    elif event_type == 'correct_rejection':
        # No stimulus (neither modality) and no lick.
        condition = (trials['whisker_stim'] == 0) & (trials['auditory_stim'] == 0)
        condition &= trials['lick_flag'] == 0
        if has_context:
            condition &= trials['context'] == context
        event_times = trials[condition]['start_time'].values

    return event_times

_worker_combo_data = None  # set once per worker process, never re-pickled per task


def _extract_worker_init(combo_data):
    """Load combo data into each worker once at pool startup."""
    global _worker_combo_data
    _worker_combo_data = combo_data


def _process_unit_extract(unit_row):
    """
    For a single unit (pd.Series), compute pre/post spike counts for every
    (event_type, context) combo using vectorised numpy ops.
    Returns a list of dicts, one per combo.
    """
    spike_times = np.asarray(unit_row['spike_times'])
    rows = []

    for (event_type, context, event_times_arr,
         pre_start, pre_end, post_start, post_end,
         bl_pre_start, bl_pre_end) in _worker_combo_data:

        # Vectorised: one searchsorted call per window edge → no Python loop over events
        pre_counts  = (np.searchsorted(spike_times, event_times_arr + pre_end,  side='right') -
                       np.searchsorted(spike_times, event_times_arr + pre_start, side='left')).tolist()
        post_counts = (np.searchsorted(spike_times, event_times_arr + post_end,  side='right') -
                       np.searchsorted(spike_times, event_times_arr + post_start, side='left')).tolist()

        # Extended baseline window (None for events not used in baseline analyses)
        if bl_pre_start is not None:
            bl_pre_counts = (np.searchsorted(spike_times, event_times_arr + bl_pre_end, side='right') -
                             np.searchsorted(spike_times, event_times_arr + bl_pre_start, side='left')).tolist()
        else:
            bl_pre_counts = None

        row = unit_row.to_dict()
        row['event']      = event_type
        row['context']    = context
        row['pre_spikes'] = pre_counts
        row['post_spikes'] = post_counts
        row['baseline_pre_spikes'] = bl_pre_counts

        rows.append(row)

    return rows

def extract_spike_data(nwb_file):
    """
    Process spike data from a NWB file.
    :param nwb_file: path to NWB file
    :return: processed unit table (pd.DataFrame)
    """
    nwb_file_path = pathlib.Path(nwb_file)
    mouse_name = nwb_file_path.name[:5]
    session_id = nwb_file_path.stem
    nwb = nwb_reader.read_nwb_file(nwb_file_path)

    unit_table, trial_table = process_nwb_tables(nwb)
    unit_table['mouse_id']   = mouse_name
    unit_table['session_id'] = session_id

    contexts_available = trial_table['context'].unique()
    has_context = 'active' in contexts_available and 'passive_pre' in contexts_available

    event_types = ['whisker', 'auditory', 'spontaneous_licks',
                   'lick_trial', 'no_lick_trial', 'whisker_hit', 'whisker_miss',
                   'auditory_hit', 'auditory_miss', 'correct_rejection']
    baseline_events = BASELINE_EVENTS

    time_windows = TIME_WINDOWS


    # ------------------------------------------------------------------
    # Build combo list serially — extract_event_times touches the NWB
    # object and must not be called from worker processes.
    # Each entry: (event_type, context, event_times_arr, pre_s, pre_e, post_s, post_e)
    # ------------------------------------------------------------------
    combo_data = []
    for event_type in event_types:
        if event_type == 'spontaneous_licks':
            contexts = ['']
        elif event_type in ('lick_trial', 'no_lick_trial', 'whisker_hit', 'whisker_miss',
                             'auditory_hit', 'auditory_miss', 'correct_rejection'):
            contexts = ['active']
        else:
            contexts = ['active', 'passive_pre', 'passive_post'] if has_context else ['active']

        for context in contexts:
            event_times = extract_event_times(nwb, event_type, context, has_context)
            event_times_arr = np.asarray(event_times)

            if event_type == 'spontaneous_licks':
                pre_start, pre_end, post_start, post_end = time_windows['spontaneous_licks']
            else:
                pre_start, pre_end  = time_windows['pre']
                post_start, post_end = time_windows['post']

            if event_type in baseline_events:
                bl_pre_start, bl_pre_end = time_windows['baseline_pre']
            else:
                bl_pre_start, bl_pre_end = None, None

            combo_data.append((event_type, context, event_times_arr,
                               pre_start, pre_end, post_start, post_end,
                               bl_pre_start, bl_pre_end))

    # ------------------------------------------------------------------
    # Parallelise over units; combo_data is loaded once per worker
    # ------------------------------------------------------------------
    unit_rows = [row for _, row in unit_table.iterrows()]
    print(f'Extracting spike data: {len(unit_rows)} units × {len(combo_data)} combos')

    with multiprocessing.Pool(
        processes=N_WORKERS,
        initializer=_extract_worker_init,
        initargs=(combo_data,),
    ) as pool:
        nested = pool.map(
            _process_unit_extract,
            unit_rows,
            chunksize=max(1, len(unit_rows) // (N_WORKERS * 4)),
        )

    # Flatten list-of-lists and build DataFrame
    proc_unit_table = pd.DataFrame([row for rows in nested for row in rows])
    return proc_unit_table

def extract_spike_data_old(nwb_file):
    """
    Process spike data from a NWB file.
    :param nwb_file: path to NWB file
    :return:
    """

    nwb_file_path = pathlib.Path(nwb_file)
    mouse_name = nwb_file_path.name[:5]
    session_id = nwb_file_path.stem
    nwb = nwb_reader.read_nwb_file(nwb_file_path)

    unit_table, trial_table = process_nwb_tables(nwb)
    unit_table['mouse_id'] = mouse_name
    unit_table['session_id'] = session_id

    # Store processed data
    proc_unit_table = []

    # Define event types and contexts
    contexts = trial_table['context'].unique()
    has_context = 'active' in contexts and 'passive_pre' in contexts

    event_types = ['whisker', 'auditory', 'spontaneous_licks', 'lick_trial', 'no_lick_trial', 'whisker_hit', 'whisker_miss']
    time_windows = {
        'pre': (-0.05, -0.005),
        'post': (0.005, 0.05),
        'spontaneous_licks': (-0.4, -0.2, 0, 0.2)
    }

    for event_type in event_types:

        contexts = ['active', 'passive_pre', 'passive_post'] if has_context else ['active']

        # Initialize columns
        if event_type == 'spontaneous_licks':
            contexts = [''] # context irrelevant for spontaneous licks
        elif event_type in ['lick_trial', 'no_lick_trial', 'whisker_hit', 'whisker_miss']:
            contexts = ['active'] # choice only in active context

        # Get count data for each unit
        for context in contexts:

            # Extract list of event times
            event_times = extract_event_times(nwb, event_type, context, has_context)

            for idx, unit in unit_table.iterrows():
                spike_times = unit['spike_times']
                pre_counts, post_counts = [], []

                for event in event_times:
                    if event_type == 'spontaneous_licks':
                        pre_start, pre_end, post_start, post_end = time_windows['spontaneous_licks']
                    else:
                        pre_start, pre_end = time_windows['pre']
                        post_start, post_end = time_windows['post']

                    pre_counts.append(count_spikes_in_window(spike_times, event + pre_start, event + pre_end))
                    post_counts.append(count_spikes_in_window(spike_times, event + post_start, event + post_end))

                # Add metadata to unit table
                unit['event'] = event_type
                unit['context'] = context
                unit['pre_spikes'] = pre_counts
                unit['post_spikes'] = post_counts
                proc_unit_table.append(unit)

    # Convert to DataFrame
    proc_unit_table = pd.DataFrame(proc_unit_table)

    return proc_unit_table

def calculate_roc(class_1_counts, class_2_counts, shuffle=False):
    """
    Calculate receiver operating characteristic its area under the curve, between spike counts for two classes.
    :param class_1_counts: list of spike counts for class 1 for each event, e.g. pre-stim spikes
    :param class_2_counts: list of spike counts for class 2 for each event, e.g. post-stim spikes
    :param shuffle: whether to shuffle class labels
    :return:
    """
    # Combine spike count data and class labels
    spike_counts = np.concatenate([class_1_counts, class_2_counts])
    labels = np.concatenate([np.zeros(len(class_1_counts)), np.ones(len(class_2_counts))])

    if shuffle:
        labels = np.random.permutation(labels)
    # Check if labels have at least two classes
    if len(np.unique(labels)) < 2:
        return None, None, None, np.nan  # return NaN for ROC AUC or another default
    # Balance classes using sample weights inversely proportional to class frequency
    sample_weights = compute_sample_weight('balanced', labels)

    # Compute the ROC curve and area under the curve
    fpr, tpr, thresholds = roc_curve(labels, spike_counts, sample_weight=sample_weights, drop_intermediate=True)
    roc_auc = roc_auc_score(labels, spike_counts)

    return fpr, tpr, thresholds, roc_auc


def _baseline_subtract_column(df_subset, count_col, count_duration):
    """
    Subtract single-trial baseline firing (the extended pre-event window,
    scaled to the duration of `count_col`'s window) from a spike-count
    column, row by row. Each row of df_subset is one (event, context) combo
    holding a list of per-trial counts.

    If a row has no baseline_pre_spikes (not expected for any event type
    currently in BASELINE_EVENTS), that row's counts are returned untouched.

    :return: pd.Series of numpy arrays (float), same index as df_subset.
    """
    scale = count_duration / BASELINE_DURATION
    corrected = []
    for _, row in df_subset.iterrows():
        counts = np.asarray(row[count_col], dtype=float)
        baseline = row.get('baseline_pre_spikes', None)
        if baseline is None:
            corrected.append(counts)
            continue
        baseline = np.asarray(baseline, dtype=float)
        corrected.append(counts - baseline * scale)
    return pd.Series(corrected, index=df_subset.index)


def _get_counts(unit_data, event, context, col='post_spikes', apply_baseline=True):
    """
    Extract a flat array of (optionally baseline-corrected) spike counts
    for a given (event, context, column) combination.

    :param apply_baseline: if True, subtract the single-trial baseline
        (scaled to the window duration implied by `col`) before flattening.
        Has no effect on rows/events for which baseline_pre_spikes is None
        (e.g. spontaneous_licks).
    """
    sub = unit_data[(unit_data['event'] == event) & (unit_data['context'] == context)]
    if len(sub) == 0:
        return np.array([])

    if col == 'post_spikes':
        duration = SPONT_POST_DURATION if event == 'spontaneous_licks' else POST_DURATION
    elif col == 'pre_spikes':
        duration = SPONT_PRE_DURATION if event == 'spontaneous_licks' else PRE_DURATION
    else:
        duration = None  # e.g. col == 'baseline_pre_spikes' itself

    if apply_baseline and duration is not None:
        arrays = _baseline_subtract_column(sub, col, duration).values
    else:
        arrays = sub[col].values

    try:
        return np.concatenate(arrays)
    except ValueError:
        return np.array([])


def select_spike_counts(unit_data, analysis_type):
    """
    Select spike counts based on roc_analysis type.

    Single-trial baseline subtraction (extended pre-event window, scaled to
    the count window's duration) is applied to every analysis type except
    those starting with 'baseline_' (these already operate directly on the
    baseline_pre_spikes column and are not subtracted from themselves).
    spontaneous_licks now has its own extended pre-lick baseline and is
    baseline-corrected the same way as every other event.
    """
    is_baseline_type = analysis_type.startswith('baseline_')

    if analysis_type == 'whisker_passive_pre':
        spikes_1 = _get_counts(unit_data, 'whisker', 'passive_pre', 'pre_spikes')
        spikes_2 = _get_counts(unit_data, 'whisker', 'passive_pre', 'post_spikes')
    elif analysis_type == 'whisker_passive_post':
        spikes_1 = _get_counts(unit_data, 'whisker', 'passive_post', 'pre_spikes')
        spikes_2 = _get_counts(unit_data, 'whisker', 'passive_post', 'post_spikes')
    elif analysis_type == 'whisker_active':  # hit and miss trials
        spikes_1 = _get_counts(unit_data, 'whisker', 'active', 'pre_spikes')
        spikes_2 = _get_counts(unit_data, 'whisker', 'active', 'post_spikes')
    elif analysis_type == 'whisker_pre_vs_post_learning':
        spikes_1 = _get_counts(unit_data, 'whisker', 'passive_pre', 'post_spikes')
        spikes_2 = _get_counts(unit_data, 'whisker', 'passive_post', 'post_spikes')
    elif analysis_type == 'auditory_passive_pre':
        spikes_1 = _get_counts(unit_data, 'auditory', 'passive_pre', 'pre_spikes')
        spikes_2 = _get_counts(unit_data, 'auditory', 'passive_pre', 'post_spikes')
    elif analysis_type == 'auditory_passive_post':
        spikes_1 = _get_counts(unit_data, 'auditory', 'passive_post', 'pre_spikes')
        spikes_2 = _get_counts(unit_data, 'auditory', 'passive_post', 'post_spikes')
    elif analysis_type == 'auditory_active':  # hit and miss trials
        spikes_1 = _get_counts(unit_data, 'auditory', 'active', 'pre_spikes')
        spikes_2 = _get_counts(unit_data, 'auditory', 'active', 'post_spikes')
    elif analysis_type == 'auditory_pre_vs_post_learning':
        spikes_1 = _get_counts(unit_data, 'auditory', 'passive_pre', 'post_spikes')
        spikes_2 = _get_counts(unit_data, 'auditory', 'passive_post', 'post_spikes')
    elif analysis_type == 'wh_vs_aud_passive_pre':
        spikes_1 = _get_counts(unit_data, 'whisker', 'passive_pre', 'post_spikes')
        spikes_2 = _get_counts(unit_data, 'auditory', 'passive_pre', 'post_spikes')
    elif analysis_type == 'wh_vs_aud_passive_post':
        spikes_1 = _get_counts(unit_data, 'whisker', 'passive_post', 'post_spikes')
        spikes_2 = _get_counts(unit_data, 'auditory', 'passive_post', 'post_spikes')
    elif analysis_type == 'wh_vs_aud_active':
        spikes_1 = _get_counts(unit_data, 'whisker', 'active', 'post_spikes')
        spikes_2 = _get_counts(unit_data, 'auditory', 'active', 'post_spikes')
    elif analysis_type == 'wh_vs_aud_pre_vs_post_learning':
        spikes_1 = _get_counts(unit_data, 'whisker', 'passive_pre', 'post_spikes')
        spikes_2 = _get_counts(unit_data, 'auditory', 'passive_post', 'post_spikes')
    elif analysis_type == 'spontaneous_licks':
        # Pre vs post around the lick itself, now baseline-corrected using
        # the extended pre-lick window, same as every other event.
        spikes_1 = _get_counts(unit_data, 'spontaneous_licks', '', 'pre_spikes')
        spikes_2 = _get_counts(unit_data, 'spontaneous_licks', '', 'post_spikes')
    elif analysis_type == 'spontaneous_licks_vs_cr':
        # Pure-motor analysis: spontaneous-lick response vs Correct Rejection
        # response. Both sides baseline-corrected.
        spikes_1 = _get_counts(unit_data, 'correct_rejection', 'active', 'post_spikes')
        spikes_2 = _get_counts(unit_data, 'spontaneous_licks', '', 'post_spikes')
    elif analysis_type == 'choice':
        spikes_1 = _get_counts(unit_data, 'lick_trial', 'active', 'post_spikes')
        spikes_2 = _get_counts(unit_data, 'no_lick_trial', 'active', 'post_spikes')
    elif analysis_type == 'whisker_choice':
        spikes_1 = _get_counts(unit_data, 'whisker_hit', 'active', 'post_spikes')
        spikes_2 = _get_counts(unit_data, 'whisker_miss', 'active', 'post_spikes')
    elif analysis_type == 'auditory_choice':
        # Few Miss trials expected for auditory (near-ceiling performance),
        # so this analysis will have lower power than whisker_choice.
        spikes_1 = _get_counts(unit_data, 'auditory_hit', 'active', 'post_spikes')
        spikes_2 = _get_counts(unit_data, 'auditory_miss', 'active', 'post_spikes')
    elif analysis_type == 'whisker_sensory':
        # Pure-sensory analysis: stimulus-evoked (Miss, i.e. stim w/o lick)
        # vs no-stimulus (Correct Rejection), aligned to stimulus onset.
        spikes_1 = _get_counts(unit_data, 'correct_rejection', 'active', 'post_spikes')
        spikes_2 = _get_counts(unit_data, 'whisker_miss', 'active', 'post_spikes')
    elif analysis_type == 'auditory_sensory':
        spikes_1 = _get_counts(unit_data, 'correct_rejection', 'active', 'post_spikes')
        spikes_2 = _get_counts(unit_data, 'auditory_miss', 'active', 'post_spikes')
    elif analysis_type == 'whisker_hit_vs_cr':
        # Hit-trial response vs Correct Rejection, aligned to stimulus/trial onset.
        spikes_1 = _get_counts(unit_data, 'correct_rejection', 'active', 'post_spikes')
        spikes_2 = _get_counts(unit_data, 'whisker_hit', 'active', 'post_spikes')
    elif analysis_type == 'auditory_hit_vs_cr':
        spikes_1 = _get_counts(unit_data, 'correct_rejection', 'active', 'post_spikes')
        spikes_2 = _get_counts(unit_data, 'auditory_hit', 'active', 'post_spikes')
    elif analysis_type == 'whisker_hit_vs_spontaneous':
        # Decision-neuron component (b): Hit vs Spontaneous lick. Ideally
        # aligned to lick/jaw onset; for now both sides use their existing
        # (stim-onset for Hit, lick-time for spontaneous) windows, per
        # current instructions to defer jaw-onset alignment.
        spikes_1 = _get_counts(unit_data, 'spontaneous_licks', '', 'post_spikes')
        spikes_2 = _get_counts(unit_data, 'whisker_hit', 'active', 'post_spikes')
    elif analysis_type == 'auditory_hit_vs_spontaneous':
        spikes_1 = _get_counts(unit_data, 'spontaneous_licks', '', 'post_spikes')
        spikes_2 = _get_counts(unit_data, 'auditory_hit', 'active', 'post_spikes')
    elif analysis_type == 'baseline_choice':
        spikes_1 = _get_counts(unit_data, 'lick_trial', 'active', 'baseline_pre_spikes', apply_baseline=False)
        spikes_2 = _get_counts(unit_data, 'no_lick_trial', 'active', 'baseline_pre_spikes', apply_baseline=False)
    elif analysis_type == 'baseline_whisker_choice':
        spikes_1 = _get_counts(unit_data, 'whisker_hit', 'active', 'baseline_pre_spikes', apply_baseline=False)
        spikes_2 = _get_counts(unit_data, 'whisker_miss', 'active', 'baseline_pre_spikes', apply_baseline=False)
    elif analysis_type == 'baseline_auditory_choice':
        spikes_1 = _get_counts(unit_data, 'auditory_hit', 'active', 'baseline_pre_spikes', apply_baseline=False)
        spikes_2 = _get_counts(unit_data, 'auditory_miss', 'active', 'baseline_pre_spikes', apply_baseline=False)
    elif analysis_type == 'baseline_pre_vs_post_learning':
        spikes_1 = np.concatenate([
            _get_counts(unit_data, 'whisker', 'passive_pre', 'baseline_pre_spikes', apply_baseline=False),
            _get_counts(unit_data, 'auditory', 'passive_pre', 'baseline_pre_spikes', apply_baseline=False),
        ])
        spikes_2 = np.concatenate([
            _get_counts(unit_data, 'whisker', 'passive_post', 'baseline_pre_spikes', apply_baseline=False),
            _get_counts(unit_data, 'auditory', 'passive_post', 'baseline_pre_spikes', apply_baseline=False),
        ])
    else:  # TODO: hit. vs false alarm (per modality?) but aligned at jaw onset
        raise ValueError(f"Analysis type {analysis_type} not recognized.")

    return spikes_1, spikes_2


def process_unit(neuron_id, proc_unit_table, analysis_type, results_path):
    """
    Process a single unit for ROC roc_analysis.
    :param neuron_id: unit ID
    :param proc_unit_table: processed unit table
    :param analysis_type: type of roc_analysis
    :param results_path: path to save results
    :return:
    """
    unit_table = proc_unit_table[proc_unit_table['neuron_id'] == neuron_id]
    mouse_id = unit_table['mouse_id'].values[0]
    area = unit_table['ccf_atlas_parent_acronym'].values[0]

    # Keep relevant columns for results
    ccf_cols = [c for c in unit_table.columns if 'ccf' in c]
    cols_to_keep = ['mouse_id', 'session_id', 'electrode_group', 'neuron_id', 'cluster_id', 'firing_rate', 'target_region'] + ccf_cols
    res_dict = {col: unit_table[col].values[0] for col in cols_to_keep}
    res_dict.update({'analysis_type': analysis_type, 'neuron_id': neuron_id, 'mouse_id': mouse_id, 'area': area})

    # Select adequate spike counts and compute ROC
    spikes_1, spikes_2 = select_spike_counts(unit_table, analysis_type)
    fpr, tpr, thresholds, roc_auc = calculate_roc(spikes_1, spikes_2)
    selectivity_index = 2 * roc_auc - 1

    res_dict.update({'auc': roc_auc})
    res_dict.update({'selectivity': selectivity_index})
    res_dict.update({'fpr': fpr})
    res_dict.update({'tpr': tpr})
    res_dict.update({'thresholds': thresholds})

    # Perform class-label permutations to obtain a null distribution for significance
    n_permutations = N_PERMUTATIONS
    permuted_aucs = []
    for _ in range(n_permutations):
        _, _, _, roc_auc_permut = calculate_roc(spikes_1, spikes_2, shuffle=True)
        permuted_aucs.append(roc_auc_permut)
    permuted_aucs = np.array(permuted_aucs)

    # Calculate p-values as proportion of permuted AUCs greater than or equal to the observed AUC
    p_value_pos = np.sum(permuted_aucs >= roc_auc) / n_permutations  # one-tailed: AUC greater than chance
    p_value_neg = np.sum(permuted_aucs <= roc_auc) / n_permutations

    # Determine direction of significance based on roc_analysis type
    if 'wh_vs_aud' in analysis_type:
        directions = ['auditory', 'whisker']  # auditory is spikes_2!
    else:
        directions = ['positive', 'negative']

    # Significance from whichever tail is smaller; direction from the sign
    # of the selectivity index itself (equivalent in practice to checking
    # p_value_pos/p_value_neg separately, but avoids relying on evaluation
    # order and is robust to ties near AUC = 0.5).
    # Note: this is the raw, uncorrected per-analysis p-value (no multiple-
    # comparisons correction across the ~24 analysis types per neuron).
    p_value_to_show = p_value_pos if selectivity_index >= 0 else p_value_neg
    is_significant = p_value_to_show < ALPHA

    if is_significant:
        direction = directions[0] if selectivity_index > 0 else directions[1]
        res_dict.update({'significant': is_significant, 'direction': direction, 'p_value': p_value_to_show, 'p_value_to_show': p_value_to_show})
    else:
        res_dict.update({'significant': is_significant, 'direction': np.nan, 'p_value': p_value_pos, 'p_value_to_show': p_value_pos})

    debug = False
    if debug and is_significant:

        # Subplots: 1. ROC curve 2. Histogram of permuted AUCs
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        for ax in axs:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=15)

        pval = res_dict['p_value_to_show']
        suptitle_text = f'ROC Analysis for mouse {mouse_id} unit {neuron_id} ({area}) ({analysis_type})\n'
        suptitle_text += f'AUC = {roc_auc:.2f}, selectivity = {selectivity_index:.2f}, p-value = {pval:.3f}'
        fig.suptitle(suptitle_text)

        axs[0].plot(fpr, tpr, color='indianred', lw=2)
        axs[0].plot([0, 1], [0, 1], linestyle='--', color='k', lw=2)
        axs[0].set_xlabel('False positive rate', fontsize=15)
        axs[0].set_ylabel('True positive rate', fontsize=15)
        axs[0].set_title(f'ROC curve', fontsize=15)

        axs[1].hist(permuted_aucs, bins=30, color='grey', edgecolor='white')
        axs[1].axvline(roc_auc, color='r', linestyle='--', lw=2, label=f'Observed AUC = {roc_auc:.2f}')
        axs[1].set_xlabel('AUC', fontsize=15)
        axs[1].set_ylabel('Frequency', fontsize=15)
        axs[1].set_title(f'Permuted AUCs', fontsize=15)
        axs[1].legend(frameon=False, loc='upper right')

        # Show
        fig.tight_layout()
        plt.close()
        fname = f'{mouse_id}_{neuron_id}_{analysis_type}.png'
        results_path_fig = os.path.join(results_path, 'figures')
        os.makedirs(results_path_fig, exist_ok=True)
        fig.savefig(os.path.join(results_path_fig, fname))

    return res_dict

# Module-level worker state — set once per process via initializer, never re-pickled
_worker_unit_table = None

def _worker_init(unit_table):
    """Store the unit table in each worker process once at pool creation."""
    global _worker_unit_table
    _worker_unit_table = unit_table


def _process_unit_task(args):
    """Unpack args and call process_unit using the pre-loaded worker-local table."""
    neuron_id, analysis_type, results_path = args
    return process_unit(
        neuron_id,
        proc_unit_table=_worker_unit_table,
        analysis_type=analysis_type,
        results_path=results_path,
    )


def _neuron_significance(results_table, analysis_type, neuron_ids):
    """
    Boolean Series (indexed by neuron_id, aligned to neuron_ids) of
    `significant` for one analysis_type. Neurons for which that analysis
    wasn't computed (e.g. passive analyses on a mouse without a passive
    context) read as False, not missing.
    """
    sub = results_table[results_table['analysis_type'] == analysis_type]
    sig = sub.set_index('neuron_id')['significant'].astype(bool)
    return sig.reindex(neuron_ids).fillna(False)


def _neuron_direction(results_table, analysis_type, neuron_ids):
    """Direction values (indexed by neuron_id, aligned to neuron_ids) for one analysis_type."""
    sub = results_table[results_table['analysis_type'] == analysis_type]
    direction = sub.set_index('neuron_id')['direction']
    return direction.reindex(neuron_ids)


def compute_neuron_labels(results_table):
    """
    Build a per-neuron table of derived labels (one row per neuron_id),
    meant to be merged back onto results_table as new columns rather than
    encoded as extra analysis_type rows.

    Columns added:

    - modality_preference: one of {'whisker', 'auditory', 'bimodal', 'non_responsive'}.
        Decision order:
          1. 'bimodal' if both whisker_active AND auditory_active are significant
             (neuron responds to both modalities).
          2. otherwise, if wh_vs_aud_active is significant, use ITS direction
             ('whisker' or 'auditory') — this analysis directly compares
             whisker vs auditory evoked activity, so it's the more direct
             test of preference than comparing two separate significance
             flags.
          3. otherwise, fall back to whichever of whisker_active/auditory_active
             is significant alone (covers the case where wh_vs_aud_active is
             underpowered/not significant but the neuron is clearly
             modality-responsive on one side).
          4. 'non_responsive' if none of the above hold.

    - sensory_label: 'sensory' if significant in at least one of the four
        passive analyses (whisker_passive_pre, whisker_passive_post,
        auditory_passive_pre, auditory_passive_post), else 'non_sensory'.
        Note: mice without a passive context never have these analyses
        computed, so such neurons will read as 'non_sensory' by default
        rather than 'not tested' — worth filtering on mouse type if that
        distinction matters downstream.

    - whisker_decision, whisker_gated_decision,
      auditory_decision, auditory_gated_decision: bool, per modality M:
        M_decision       = significant M_choice AND significant M_hit_vs_spontaneous
        M_gated_decision = M_decision AND NOT M_sensory AND NOT spontaneous_licks_vs_cr
      (the motor/gating term, spontaneous_licks_vs_cr, is shared across
      modalities since it isn't stimulus-specific).

    Note: auditory_choice typically has few Miss trials, so auditory_decision/
    auditory_gated_decision will generally be lower-powered than their
    whisker counterparts.
    """
    neuron_ids = results_table['neuron_id'].unique()
    labels = pd.DataFrame(index=pd.Index(neuron_ids, name='neuron_id'))

    # --- modality preference ------------------------------------------------
    wh_active = _neuron_significance(results_table, 'whisker_active', neuron_ids)
    aud_active = _neuron_significance(results_table, 'auditory_active', neuron_ids)
    wh_vs_aud_sig = _neuron_significance(results_table, 'wh_vs_aud_active', neuron_ids)
    wh_vs_aud_dir = _neuron_direction(results_table, 'wh_vs_aud_active', neuron_ids)

    modality_preference = []
    for nid in neuron_ids:
        if wh_active.loc[nid] and aud_active.loc[nid]:
            modality_preference.append('bimodal')
        elif wh_vs_aud_sig.loc[nid] and wh_vs_aud_dir.loc[nid] in ('whisker', 'auditory'):
            modality_preference.append(wh_vs_aud_dir.loc[nid])
        elif wh_active.loc[nid]:
            modality_preference.append('whisker')
        elif aud_active.loc[nid]:
            modality_preference.append('auditory')
        else:
            modality_preference.append('non_responsive')
    labels['modality_preference'] = modality_preference

    # --- sensory label --------------------------------------------------------
    passive_types = ['whisker_passive_pre', 'whisker_passive_post',
                      'auditory_passive_pre', 'auditory_passive_post']
    passive_sig = pd.concat(
        [_neuron_significance(results_table, pt, neuron_ids) for pt in passive_types], axis=1
    )
    labels['sensory_label'] = np.where(passive_sig.any(axis=1), 'sensory', 'non_sensory')

    # --- decision / gated decision, per modality ------------------------------
    motor_sig = _neuron_significance(results_table, 'spontaneous_licks_vs_cr', neuron_ids)
    for modality in ('whisker', 'auditory'):
        choice_sig = _neuron_significance(results_table, f'{modality}_choice', neuron_ids)
        hit_vs_spont_sig = _neuron_significance(results_table, f'{modality}_hit_vs_spontaneous', neuron_ids)
        sensory_sig = _neuron_significance(results_table, f'{modality}_sensory', neuron_ids)

        is_decision = choice_sig & hit_vs_spont_sig
        is_gated = is_decision & (~sensory_sig) & (~motor_sig)

        labels[f'{modality}_decision'] = is_decision.values
        labels[f'{modality}_gated_decision'] = is_gated.values

    return labels.reset_index()


# ----------------------------------------------------------------------------
# PSTH sanity-check plotting: raw spike extraction, artifact correction,
# sliding-window PSTH, and per-panel population averaging.
# ----------------------------------------------------------------------------

def _extract_relative_spike_times(spike_times, event_times, window):
    """
    For each event time, return the neuron's spike times within
    event_time + window, shifted to be relative to the event (t=0 at onset).
    Returns a list of 1D numpy arrays, one per event/trial.
    """
    pre, post = window
    spike_times = np.asarray(spike_times)
    trial_spikes = []
    for t in event_times:
        lo = np.searchsorted(spike_times, t + pre, side='left')
        hi = np.searchsorted(spike_times, t + post, side='right')
        trial_spikes.append(spike_times[lo:hi] - t)
    return trial_spikes


def _compute_psth(trial_spikes, window, bin_width=PSTH_BIN_WIDTH, stride=PSTH_STRIDE,
                   baseline_window=PSTH_BASELINE_WINDOW):
    """
    Sliding-window PSTH from raw per-trial spike times: mean +/- SEM firing
    rate (Hz) across trials, using `bin_width`-wide bins centered every
    `stride`. If `baseline_window` is given (default PSTH_BASELINE_WINDOW),
    each trial's own mean rate in that window is subtracted from that
    trial's entire rate curve before averaging across trials, so the result
    is a change-from-baseline PSTH rather than a raw one. Pass
    baseline_window=None to skip this. Returns (bin_centers, mean_rate, sem_rate).
    """
    win_start, win_end = window
    half = bin_width / 2.0
    centers = np.arange(win_start + half, win_end - half + 1e-9, stride)
    n_trials = len(trial_spikes)
    if n_trials == 0:
        nan_arr = np.full(len(centers), np.nan)
        return centers, nan_arr, nan_arr

    rates = np.zeros((n_trials, len(centers)))
    for i, spikes in enumerate(trial_spikes):
        spikes = np.asarray(spikes)
        if spikes.size == 0:
            continue
        lo_idx = np.searchsorted(spikes, centers - half, side='left')
        hi_idx = np.searchsorted(spikes, centers + half, side='right')
        rates[i] = (hi_idx - lo_idx) / bin_width

    if baseline_window is not None:
        bl_start, bl_end = baseline_window
        bl_duration = bl_end - bl_start
        baseline_rates = np.zeros(n_trials)
        for i, spikes in enumerate(trial_spikes):
            spikes = np.asarray(spikes)
            bl_count = np.sum((spikes >= bl_start) & (spikes < bl_end))
            baseline_rates[i] = bl_count / bl_duration
        rates = rates - baseline_rates[:, None]

    mean_rate = rates.mean(axis=0)
    sem_rate = rates.std(axis=0, ddof=1) / np.sqrt(n_trials) if n_trials > 1 else np.zeros_like(mean_rate)
    return centers, mean_rate, sem_rate


def _cached_neuron_psth(psth_data, neuron_id, event, context, window, cache):
    """
    Per-neuron binned PSTH (centers, rate), memoized on (neuron_id, event,
    context, window) in `cache` so it's computed once regardless of how many
    panels reuse the same combo (e.g. whisker_hit/active feeds ~7 panels).
    """
    key = (neuron_id, event, context, window)
    if key in cache:
        return cache[key]
    trial_spikes = psth_data.get(neuron_id, {}).get((event, context))
    if not trial_spikes:
        cache[key] = None
        return None
    centers, rate, _ = _compute_psth(trial_spikes, window)
    cache[key] = (centers, rate)
    return cache[key]


def _population_psth(psth_data, neuron_ids, event, context, window, cache=None):
    """
    Population PSTH: for each neuron, first average across that neuron's own
    trials (via _compute_psth, memoized in `cache`), then average those
    per-neuron PSTHs across neurons — so every neuron is weighted equally
    regardless of firing rate or trial count. Returns
    (centers, mean_across_neurons, sem_across_neurons), or None if no neuron
    in neuron_ids has data for (event, context).
    """
    if cache is None:
        cache = {}
    per_neuron_rates = []
    centers = None
    for nid in neuron_ids:
        result = _cached_neuron_psth(psth_data, nid, event, context, window, cache)
        if result is None:
            continue
        c, rate = result
        centers = c
        per_neuron_rates.append(rate)

    if not per_neuron_rates:
        return None

    per_neuron_rates = np.array(per_neuron_rates)
    mean_rate = per_neuron_rates.mean(axis=0)
    n = per_neuron_rates.shape[0]
    sem_rate = per_neuron_rates.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean_rate)
    return centers, mean_rate, sem_rate


_psth_worker_combo_data = None


def _psth_worker_init(combo_data):
    """Load the (event, context, event_times, window) combo list into each worker once."""
    global _psth_worker_combo_data
    _psth_worker_combo_data = combo_data


def _process_unit_psth(unit_row):
    """
    For one unit, extract raw per-trial relative spike times for every
    (event, context) combo needed for PSTH plotting. spike_times is already
    whisker-artifact-corrected at this point (see correct_neuron_spike_train
    / process_nwb_tables), so no per-event correction happens here.
    Returns (neuron_id, {(event, context): [per-trial relative spike arrays]}).
    """
    spike_times = np.asarray(unit_row['spike_times'])
    out = {}
    for event_type, context, event_times_arr, window in _psth_worker_combo_data:
        out[(event_type, context)] = _extract_relative_spike_times(spike_times, event_times_arr, window)
    return unit_row['neuron_id'], out


def diagnose_whisker_artifact(nwb_file, half_range=0.030, bin_ms=1.0):
    """
    Diagnostic (not part of the main pipeline): pool raw whisker-aligned
    spike times across ALL neurons and trials into a fine, non-sliding
    histogram (bin_ms wide, no smoothing) around t=0, comparing
    process_nwb_tables with correction off vs on. Use this to check whether
    WHISKER_ARTIFACT_WINDOW actually covers the true suppression zone: if
    the 'raw' (uncorrected) histogram shows detection dropping out over a
    wider span than WHISKER_ARTIFACT_WINDOW, widen the constant accordingly
    — a residual dip right at the edge of the correction window after that
    fix usually means the true artifact is wider than assumed, not a bug in
    the Poisson replacement itself (which is an unbiased estimator of the
    surrounding baseline rate and shouldn't push the average below it).

    Returns (bin_edges, raw_counts, corrected_counts); also prints a quick
    text summary of where 'raw' counts drop noticeably below the
    surrounding baseline level.
    """
    nwb_file_path = pathlib.Path(nwb_file)
    nwb = nwb_reader.read_nwb_file(nwb_file_path)
    raw_units, trial_table = process_nwb_tables(nwb, apply_artifact_correction=False)
    corrected_units, _ = process_nwb_tables(nwb, apply_artifact_correction=True)

    contexts_available = trial_table['context'].unique()
    has_context = 'active' in contexts_available and 'passive_pre' in contexts_available
    event_times = np.asarray(extract_event_times(nwb, 'whisker', 'active', has_context))

    window = (-half_range, half_range)
    bin_edges = np.arange(window[0], window[1] + bin_ms / 1000, bin_ms / 1000)
    raw_counts = np.zeros(len(bin_edges) - 1)
    corrected_counts = np.zeros(len(bin_edges) - 1)

    for _, unit_row in raw_units.iterrows():
        trial_spikes = _extract_relative_spike_times(unit_row['spike_times'], event_times, window)
        raw_counts += np.histogram(np.concatenate(trial_spikes) if trial_spikes else [], bins=bin_edges)[0]
    for _, unit_row in corrected_units.iterrows():
        trial_spikes = _extract_relative_spike_times(unit_row['spike_times'], event_times, window)
        corrected_counts += np.histogram(np.concatenate(trial_spikes) if trial_spikes else [], bins=bin_edges)[0]

    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    baseline_mask = (centers >= WHISKER_ARTIFACT_BASELINE_WINDOW[0]) & (centers < WHISKER_ARTIFACT_BASELINE_WINDOW[1])
    baseline_level = raw_counts[baseline_mask].mean()
    suppressed = centers[raw_counts < 0.5 * baseline_level]
    if suppressed.size:
        print(f"Raw counts drop below half the baseline level over approximately "
              f"[{suppressed.min():.3f}, {suppressed.max():.3f}] s "
              f"(current WHISKER_ARTIFACT_WINDOW = {WHISKER_ARTIFACT_WINDOW}).")
    else:
        print("No clear suppression zone detected in raw counts at this resolution.")

    return bin_edges, raw_counts, corrected_counts


def extract_psth_spike_times(nwb_file):
    """
    Extract, for every neuron and every (event, context) combo used by the
    PSTH sanity-check plots, the raw per-trial spike times relative to event
    onset (PSTH_WINDOW_STIM for stimulus/trial-onset-aligned events,
    PSTH_WINDOW_LICK for spontaneous-lick-aligned events). Whisker stimulus-
    artifact correction is already applied to spike_times by
    process_nwb_tables (see correct_neuron_spike_train), globally, for
    every whisker trial in the session — no extra correction step happens
    here.

    :return: (psth_data, unit_table) where
        psth_data  = {neuron_id: {(event, context): [per-trial relative spike arrays]}}
        unit_table = processed unit metadata DataFrame (as in extract_spike_data)
    """
    nwb_file_path = pathlib.Path(nwb_file)
    mouse_name = nwb_file_path.name[:5]
    session_id = nwb_file_path.stem
    nwb = nwb_reader.read_nwb_file(nwb_file_path)

    unit_table, trial_table = process_nwb_tables(nwb)
    unit_table['mouse_id'] = mouse_name
    unit_table['session_id'] = session_id

    contexts_available = trial_table['context'].unique()
    has_context = 'active' in contexts_available and 'passive_pre' in contexts_available

    combo_data = []
    for event_type, (contexts, window) in PSTH_EVENT_SPECS.items():
        actual_contexts = contexts
        if event_type in ('whisker', 'auditory') and not has_context:
            actual_contexts = ('active',)
        for context in actual_contexts:
            event_times = extract_event_times(nwb, event_type, context, has_context)
            event_times_arr = np.asarray(event_times)
            combo_data.append((event_type, context, event_times_arr, window))

    unit_rows = [row for _, row in unit_table.iterrows()]
    print(f'Extracting PSTH data: {len(unit_rows)} units x {len(combo_data)} combos')

    with multiprocessing.Pool(
        processes=N_WORKERS,
        initializer=_psth_worker_init,
        initargs=(combo_data,),
    ) as pool:
        results = pool.map(_process_unit_psth, unit_rows)

    psth_data = {neuron_id: data for neuron_id, data in results}
    return psth_data, unit_table


# ----------------------------------------------------------------------------
# Declarative panel specs. Section 1 mirrors the ROC analysis types (many
# analyses that share the same underlying event/context pair, e.g.
# whisker_active's pre vs post, reuse a single PSTH trace, split by that
# analysis type's own significance label). Section 2 covers the derived
# per-neuron label columns from compute_neuron_labels.
# ----------------------------------------------------------------------------
PSTH_PANELS_BY_ANALYSIS = [
    dict(title='whisker_passive_pre', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='whisker', context='passive_pre', label='whisker')],
         split_analysis_type='whisker_passive_pre', shade=[PRE_WINDOW, POST_WINDOW]),
    dict(title='whisker_passive_post', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='whisker', context='passive_post', label='whisker')],
         split_analysis_type='whisker_passive_post', shade=[PRE_WINDOW, POST_WINDOW]),
    dict(title='whisker_active', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='whisker', context='active', label='whisker')],
         split_analysis_type='whisker_active', shade=[PRE_WINDOW, POST_WINDOW]),
    dict(title='whisker_pre_vs_post_learning', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='whisker', context='passive_pre', label='pre-learning'),
                     dict(event='whisker', context='passive_post', label='post-learning')],
         split_analysis_type='whisker_pre_vs_post_learning', shade=[POST_WINDOW]),
    dict(title='auditory_passive_pre', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='auditory', context='passive_pre', label='auditory')],
         split_analysis_type='auditory_passive_pre', shade=[PRE_WINDOW, POST_WINDOW]),
    dict(title='auditory_passive_post', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='auditory', context='passive_post', label='auditory')],
         split_analysis_type='auditory_passive_post', shade=[PRE_WINDOW, POST_WINDOW]),
    dict(title='auditory_active', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='auditory', context='active', label='auditory')],
         split_analysis_type='auditory_active', shade=[PRE_WINDOW, POST_WINDOW]),
    dict(title='auditory_pre_vs_post_learning', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='auditory', context='passive_pre', label='pre-learning'),
                     dict(event='auditory', context='passive_post', label='post-learning')],
         split_analysis_type='auditory_pre_vs_post_learning', shade=[POST_WINDOW]),
    dict(title='wh_vs_aud_passive_pre', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='whisker', context='passive_pre', label='whisker'),
                     dict(event='auditory', context='passive_pre', label='auditory')],
         split_analysis_type='wh_vs_aud_passive_pre', shade=[POST_WINDOW]),
    dict(title='wh_vs_aud_passive_post', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='whisker', context='passive_post', label='whisker'),
                     dict(event='auditory', context='passive_post', label='auditory')],
         split_analysis_type='wh_vs_aud_passive_post', shade=[POST_WINDOW]),
    dict(title='wh_vs_aud_active', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='whisker', context='active', label='whisker'),
                     dict(event='auditory', context='active', label='auditory')],
         split_analysis_type='wh_vs_aud_active', shade=[POST_WINDOW]),
    dict(title='wh_vs_aud_pre_vs_post_learning', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='whisker', context='passive_pre', label='whisker pre'),
                     dict(event='auditory', context='passive_post', label='auditory post')],
         split_analysis_type='wh_vs_aud_pre_vs_post_learning', shade=[POST_WINDOW]),
    dict(title='spontaneous_licks', window=PSTH_WINDOW_LICK,
         conditions=[dict(event='spontaneous_licks', context='', label='spontaneous lick')],
         split_analysis_type='spontaneous_licks', shade=[]),
    dict(title='spontaneous_licks_vs_cr', window=PSTH_WINDOW_MIXED,
         conditions=[dict(event='correct_rejection', context='active', label='CR'),
                     dict(event='spontaneous_licks', context='', label='spontaneous lick')],
         split_analysis_type='spontaneous_licks_vs_cr', shade=[]),
    dict(title='choice', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='lick_trial', context='active', label='lick'),
                     dict(event='no_lick_trial', context='active', label='no-lick')],
         split_analysis_type='choice', shade=[POST_WINDOW]),
    dict(title='whisker_choice', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='whisker_hit', context='active', label='Hit'),
                     dict(event='whisker_miss', context='active', label='Miss')],
         split_analysis_type='whisker_choice', shade=[POST_WINDOW], artifact_shade=WHISKER_ARTIFACT_WINDOW),
    dict(title='auditory_choice', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='auditory_hit', context='active', label='Hit'),
                     dict(event='auditory_miss', context='active', label='Miss')],
         split_analysis_type='auditory_choice', shade=[POST_WINDOW]),
    dict(title='whisker_sensory', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='correct_rejection', context='active', label='CR'),
                     dict(event='whisker_miss', context='active', label='Miss')],
         split_analysis_type='whisker_sensory', shade=[POST_WINDOW], artifact_shade=WHISKER_ARTIFACT_WINDOW),
    dict(title='auditory_sensory', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='correct_rejection', context='active', label='CR'),
                     dict(event='auditory_miss', context='active', label='Miss')],
         split_analysis_type='auditory_sensory', shade=[POST_WINDOW]),
    dict(title='whisker_hit_vs_cr', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='correct_rejection', context='active', label='CR'),
                     dict(event='whisker_hit', context='active', label='Hit')],
         split_analysis_type='whisker_hit_vs_cr', shade=[POST_WINDOW], artifact_shade=WHISKER_ARTIFACT_WINDOW),
    dict(title='auditory_hit_vs_cr', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='correct_rejection', context='active', label='CR'),
                     dict(event='auditory_hit', context='active', label='Hit')],
         split_analysis_type='auditory_hit_vs_cr', shade=[POST_WINDOW]),
    dict(title='whisker_hit_vs_spontaneous', window=PSTH_WINDOW_MIXED,
         conditions=[dict(event='spontaneous_licks', context='', label='spontaneous'),
                     dict(event='whisker_hit', context='active', label='Hit')],
         split_analysis_type='whisker_hit_vs_spontaneous', shade=[], artifact_shade=WHISKER_ARTIFACT_WINDOW),
    dict(title='auditory_hit_vs_spontaneous', window=PSTH_WINDOW_MIXED,
         conditions=[dict(event='spontaneous_licks', context='', label='spontaneous'),
                     dict(event='auditory_hit', context='active', label='Hit')],
         split_analysis_type='auditory_hit_vs_spontaneous', shade=[]),
    # baseline_* analyses reuse the same underlying PSTH traces as their
    # non-baseline counterparts, just split by the baseline_* significance
    # label instead, to check whether the baseline-window effect looks
    # different from the full pre/post effect.
    dict(title='baseline_choice', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='lick_trial', context='active', label='lick'),
                     dict(event='no_lick_trial', context='active', label='no-lick')],
         split_analysis_type='baseline_choice', shade=[]),
    dict(title='baseline_whisker_choice', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='whisker_hit', context='active', label='Hit'),
                     dict(event='whisker_miss', context='active', label='Miss')],
         split_analysis_type='baseline_whisker_choice', shade=[], artifact_shade=WHISKER_ARTIFACT_WINDOW),
    dict(title='baseline_auditory_choice', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='auditory_hit', context='active', label='Hit'),
                     dict(event='auditory_miss', context='active', label='Miss')],
         split_analysis_type='baseline_auditory_choice', shade=[]),
    dict(title='baseline_pre_vs_post_learning', window=PSTH_WINDOW_STIM,
         conditions=[dict(event='whisker', context='passive_pre', label='pre-learning (wh)'),
                     dict(event='whisker', context='passive_post', label='post-learning (wh)')],
         split_analysis_type='baseline_pre_vs_post_learning', shade=[]),
]

# Section 2a: single-event population panels split by a categorical label column.
PSTH_PANELS_BY_LABEL = [
    dict(title='modality_preference (whisker-aligned)', window=PSTH_WINDOW_STIM,
         event='whisker', context='active', label_column='modality_preference'),
    dict(title='modality_preference (auditory-aligned)', window=PSTH_WINDOW_STIM,
         event='auditory', context='active', label_column='modality_preference'),
    dict(title='sensory_label (whisker-aligned)', window=PSTH_WINDOW_STIM,
         event='whisker', context='active', label_column='sensory_label'),
    dict(title='sensory_label (auditory-aligned)', window=PSTH_WINDOW_STIM,
         event='auditory', context='active', label_column='sensory_label'),
]

# Section 2b: two-condition panels split by a boolean decision/gated-decision column.
PSTH_PANELS_DECISION = []
for _modality in ('whisker', 'auditory'):
    _hit, _miss = f'{_modality}_hit', f'{_modality}_miss'
    _artifact = WHISKER_ARTIFACT_WINDOW if _modality == 'whisker' else None
    PSTH_PANELS_DECISION.append(dict(
        title=f'{_modality}_decision (stim: Hit vs Miss)', window=PSTH_WINDOW_STIM,
        conditions=[dict(event=_hit, context='active', label='Hit'),
                    dict(event=_miss, context='active', label='Miss')],
        split_label_column=f'{_modality}_decision', shade=[POST_WINDOW], artifact_shade=_artifact,
    ))
    PSTH_PANELS_DECISION.append(dict(
        title=f'{_modality}_decision (lick: Hit vs Spont.)', window=PSTH_WINDOW_MIXED,
        conditions=[dict(event='spontaneous_licks', context='', label='spontaneous'),
                    dict(event=_hit, context='active', label='Hit')],
        split_label_column=f'{_modality}_decision', shade=[], artifact_shade=_artifact,
    ))
    PSTH_PANELS_DECISION.append(dict(
        title=f'{_modality}_gated_decision (stim, decision only)', window=PSTH_WINDOW_STIM,
        conditions=[dict(event=_hit, context='active', label='Hit'),
                    dict(event=_miss, context='active', label='Miss')],
        split_label_column=f'{_modality}_gated_decision', restrict_to_decision=_modality,
        shade=[POST_WINDOW], artifact_shade=_artifact,
    ))


def _render_condition_traces(ax, psth_data, event, context, window, groups, cache):
    """Plot one population-PSTH line + SEM band per group for one (event, context). `groups`: list of (label, neuron_ids, color, linestyle)."""
    for label, ids, color, linestyle in groups:
        result = _population_psth(psth_data, ids, event, context, window, cache=cache)
        if result is None:
            continue
        centers, mean_rate, sem_rate = result
        ax.plot(centers, mean_rate, color=color, linestyle=linestyle, linewidth=1.2, label=label)
        ax.fill_between(centers, mean_rate - sem_rate, mean_rate + sem_rate, color=color, alpha=0.15, linewidth=0)


def _event_color(event, fallback):
    """Whisker events get WHISKER_COLOR, auditory events get AUDITORY_COLOR, everything else uses `fallback`."""
    if event.startswith('whisker'):
        return WHISKER_COLOR
    if event.startswith('auditory'):
        return AUDITORY_COLOR
    return fallback


# Dedicated colors for the modality_preference / sensory_label categorical panels.
MODALITY_PREFERENCE_COLORS = {
    'whisker': WHISKER_COLOR,
    'auditory': AUDITORY_COLOR,
    'bimodal': '#7570b3',
    'non_responsive': '#999999',
}
SENSORY_LABEL_COLORS = {
    'sensory': '#4daf4a',
    'non_sensory': '#999999',
}


def _plot_panel(ax, psth_data, panel, results_table, cache):
    """Render one panel spec (see PSTH_PANELS_* above) onto a single Axes. `cache` is the shared per-neuron PSTH cache for the whole PDF build (see _cached_neuron_psth)."""
    window = panel['window']
    neuron_ids_all = list(psth_data.keys())
    conditions = panel.get('conditions')

    if conditions is not None:
        if 'split_analysis_type' in panel:
            sig = _neuron_significance(results_table, panel['split_analysis_type'], neuron_ids_all)
            groups_by_id = [
                ('sig', [n for n in neuron_ids_all if sig.loc[n]], '-'),
                ('n.s.', [n for n in neuron_ids_all if not sig.loc[n]], '--'),
            ]
        elif 'split_label_column' in panel:
            col = panel['split_label_column']
            lookup = results_table.drop_duplicates('neuron_id').set_index('neuron_id')[col]
            eligible_ids = neuron_ids_all
            if panel.get('restrict_to_decision'):
                dec_col = f"{panel['restrict_to_decision']}_decision"
                dec_lookup = results_table.drop_duplicates('neuron_id').set_index('neuron_id')[dec_col]
                eligible_ids = [n for n in neuron_ids_all if bool(dec_lookup.get(n, False))]
            groups_by_id = [
                ('True', [n for n in eligible_ids if bool(lookup.get(n, False))], '-'),
                ('False', [n for n in eligible_ids if not bool(lookup.get(n, False))], '--'),
            ]
        else:
            groups_by_id = [('all', neuron_ids_all, '-')]

        for i, cond in enumerate(conditions):
            base_color = _event_color(cond['event'], NEUTRAL_COLORS[i % len(NEUTRAL_COLORS)])
            cond_groups = [(f"{cond['label']} ({g_label})", g_ids, base_color, ls)
                           for g_label, g_ids, ls in groups_by_id]
            _render_condition_traces(ax, psth_data, cond['event'], cond['context'], window, cond_groups, cache)
    else:
        # Categorical single-event panel (modality_preference / sensory_label)
        col = panel['label_column']
        lookup = results_table.drop_duplicates('neuron_id').set_index('neuron_id')[col]
        categories = sorted(lookup.dropna().unique())
        color_map = MODALITY_PREFERENCE_COLORS if col == 'modality_preference' else SENSORY_LABEL_COLORS
        palette = plt.get_cmap('tab10').colors
        groups_by_id = [(cat, [n for n in neuron_ids_all if lookup.get(n) == cat], '-')
                        for cat in categories]
        cond_groups = [(g_label, g_ids, color_map.get(g_label, palette[i % len(palette)]), ls)
                       for i, (g_label, g_ids, ls) in enumerate(groups_by_id)]
        _render_condition_traces(ax, psth_data, panel['event'], panel['context'], window, cond_groups, cache)

    for shade in panel.get('shade', []):
        ax.axvspan(shade[0], shade[1], color='black', alpha=0.06, linewidth=0)
    if panel.get('artifact_shade'):
        ax.axvspan(panel['artifact_shade'][0], panel['artifact_shade'][1], color='red', alpha=0.10, linewidth=0)

    ax.axvline(0, color='k', linewidth=0.6, linestyle=':')
    ax.axhline(0, color='k', linewidth=0.5, linestyle='-', alpha=0.3)
    ax.set_title(panel['title'], fontsize=7, wrap=True)
    ax.set_xlabel('Time (s)', fontsize=7)
    ax.set_ylabel('\u0394 Firing rate (Hz)', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=5, frameon=False, loc='upper right')
    ax.set_box_aspect(1)


def plot_psth_sanity_check(nwb_file, results_table, results_path, n_cols=6, panels_per_page=None, dpi=150, panel_size=2.4):
    """
    Build a single-page PDF of PSTH sanity-check plots for one NWB file
    (all neurons pooled into population averages per group):

    Section 1 - one panel per analysis type: population PSTH split by that
                analysis's own 'significant' label (solid = significant,
                dashed = not).
    Section 2 - one panel per derived label column (modality_preference,
                sensory_label), plus per-modality decision/gated-decision
                Hit-vs-Miss and Hit-vs-Spontaneous comparisons.

    Every PSTH is baseline-corrected: each trial's own mean rate in
    PSTH_BASELINE_WINDOW (-200 to -10 ms pre-event) is subtracted from that
    trial's curve before averaging (see _compute_psth), so panels show
    change-from-baseline (Hz) rather than raw firing rate — a horizontal
    line at y=0 marks baseline.

    Whisker-related traces are colored WHISKER_COLOR ('#ffb236'), auditory
    traces AUDITORY_COLOR ('#322fe0'); conditions with no stimulus modality
    (Correct Rejection, lick/no-lick trials, spontaneous licks) use
    NEUTRAL_COLORS. modality_preference/sensory_label categorical panels
    use dedicated color maps built on the same two modality colors.

    10 ms sliding bins, 2 ms stride (PSTH_BIN_WIDTH/PSTH_STRIDE). Windows:
    -200 to +500 ms around stimulus/trial onset, +/-350 ms around lick
    times; panels overlaying a stim- and a lick-aligned condition use their
    -200 to +350 ms overlap (PSTH_WINDOW_MIXED) so both traces have real
    data across the full displayed range. The whisker stimulus-onset
    artifact (-10 to +5 ms) is Poisson-replaced once, globally, for every
    whisker trial in the session before this function ever runs (see
    correct_neuron_spike_train / process_nwb_tables), so it's already
    reflected here regardless of which event/analysis a panel uses.

    Note: direction-specific splits (e.g. wh_vs_aud_active's whisker- vs
    auditory-preferring direction) are simplified here to a plain
    significant/non-significant split; extend _plot_panel if you want the
    direction-colored version too.

    Per-neuron binned PSTHs (see _cached_neuron_psth) are computed once per
    unique (neuron, event, context, window) and reused across every panel
    that shares that combo — most (event, context) pairs feed several
    panels (e.g. whisker_hit/active feeds whisker_choice,
    baseline_whisker_choice, whisker_hit_vs_cr, whisker_hit_vs_spontaneous,
    and three whisker_decision panels), so this avoids re-binning the same
    raw spikes over and over.

    By default all panels are laid out on a single page (panels_per_page=
    None -> all of them, n_cols=6, panel_size=2.4in per panel — pass
    panels_per_page explicitly to paginate instead).

    Saved to {results_path}/figures/{mouse_id}_psth_sanity_check.pdf
    """
    print('Extracting PSTH data for sanity-check plots:', nwb_file)
    psth_data, unit_table = extract_psth_spike_times(nwb_file)
    mouse_id = unit_table['mouse_id'].values[0]

    all_panels = list(PSTH_PANELS_BY_ANALYSIS) + list(PSTH_PANELS_BY_LABEL) + list(PSTH_PANELS_DECISION)
    panels_per_page = panels_per_page or len(all_panels)  # default: everything on one page

    fig_dir = os.path.join(results_path, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, f'{mouse_id}_psth_sanity_check.pdf')

    psth_cache = {}  # shared across every panel: {(neuron_id, event, context, window): (centers, rate)}
    n_rows = int(np.ceil(panels_per_page / n_cols))
    with PdfPages(out_path) as pdf:
        for page_start in range(0, len(all_panels), panels_per_page):
            page_panels = all_panels[page_start:page_start + panels_per_page]
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * panel_size, n_rows * panel_size), dpi=dpi)
            axes = np.atleast_1d(axes).flatten()
            for ax, panel in zip(axes, page_panels):
                try:
                    _plot_panel(ax, psth_data, panel, results_table, psth_cache)
                except Exception as e:
                    ax.text(0.5, 0.5, f"error:\n{panel['title']}\n{e}", ha='center', va='center', fontsize=6, wrap=True)
                    ax.set_box_aspect(1)
            for ax in axes[len(page_panels):]:
                ax.axis('off')
            fig.suptitle(f'{mouse_id} - PSTH sanity check', fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

    print('Saved PSTH sanity-check PDF to:', out_path)
    return out_path


def _build_run_config(nwb_file, mouse_id, analyses_to_do):
    """
    Collect the run-level parameters that determine what a results CSV
    actually means (windows, permutation count, alpha, which analyses ran,
    and the definitions used for the derived per-neuron labels), so a CSV
    can always be traced back to the exact settings that produced it.
    """
    return {
        'source_nwb_file': str(nwb_file),
        'mouse_id': mouse_id,
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'n_workers': N_WORKERS,
        'n_permutations': N_PERMUTATIONS,
        'alpha': ALPHA,
        'multiple_comparisons_correction': None,  # not applied at this per-neuron stage
        'windows_sec': {
            'pre': list(PRE_WINDOW),
            'post': list(POST_WINDOW),
            'spontaneous_licks': list(SPONT_LICK_WINDOW),
            'baseline_pre': list(BASELINE_WINDOW),
        },
        'baseline_events': sorted(BASELINE_EVENTS),
        'analyses_to_do': analyses_to_do,
        'label_definitions': {
            'modality_preference': (
                "bimodal if whisker_active AND auditory_active both significant; "
                "else direction of wh_vs_aud_active if significant; "
                "else whichever of whisker_active/auditory_active is significant alone; "
                "else non_responsive"
            ),
            'sensory_label': (
                "sensory if any of whisker_passive_pre, whisker_passive_post, "
                "auditory_passive_pre, auditory_passive_post is significant; else non_sensory"
            ),
            'M_decision': "significant M_choice AND significant M_hit_vs_spontaneous, for M in {whisker, auditory}",
            'M_gated_decision': "M_decision AND NOT M_sensory AND NOT spontaneous_licks_vs_cr",
        },
    }


def compute_unit_roc(nwb_file, results_path, make_psth_plots=False):
    """
    Perform ROC roc_analysis on spike data from a NWB file.
    :param nwb_file: path to NWB file
    :param results_path: path to save results
    :param make_psth_plots: if True, also build a PSTH sanity-check PDF
        (see plot_psth_sanity_check). Off by default: it re-extracts raw
        spike times across all neurons/events, which is slower than the
        count-based ROC pipeline.
    """
    print('Starting ROC roc_analysis for file:', nwb_file)

    proc_unit_table = extract_spike_data(nwb_file)
    mouse_id = proc_unit_table['mouse_id'].values[0]

    if int(mouse_id[2:5]) < 115 and mouse_id[:2] == 'AB': #TODO: add data_utils for passive mcie
        analyses_to_do = [
            'whisker_active', 'auditory_active',
            'wh_vs_aud_active', 'spontaneous_licks',
            'choice', 'whisker_choice', 'auditory_choice',
            'baseline_choice', 'baseline_whisker_choice', 'baseline_auditory_choice',
            'whisker_sensory', 'auditory_sensory',
            'whisker_hit_vs_cr', 'auditory_hit_vs_cr',
            'spontaneous_licks_vs_cr',
            'whisker_hit_vs_spontaneous', 'auditory_hit_vs_spontaneous',
        ]
    else:
        analyses_to_do = [
            'whisker_passive_pre', 'whisker_passive_post',
            'whisker_active', 'whisker_pre_vs_post_learning',
            'auditory_passive_pre', 'auditory_passive_post',
            'auditory_active', 'auditory_pre_vs_post_learning',
            'wh_vs_aud_passive_pre', 'wh_vs_aud_passive_post',
            'wh_vs_aud_active', 'wh_vs_aud_pre_vs_post_learning',
            'spontaneous_licks', 'choice', 'whisker_choice', 'auditory_choice',
            'baseline_choice', 'baseline_whisker_choice', 'baseline_auditory_choice',
            'baseline_pre_vs_post_learning',
            'whisker_sensory', 'auditory_sensory',
            'whisker_hit_vs_cr', 'auditory_hit_vs_cr',
            'spontaneous_licks_vs_cr',
            'whisker_hit_vs_spontaneous', 'auditory_hit_vs_spontaneous',
        ]

    neuron_ids = proc_unit_table['neuron_id'].unique()

    # Build the full task list across ALL roc_analysis types and neurons at once
    tasks = [
        (neuron_id, analysis_type, results_path)
        for analysis_type in analyses_to_do
        for neuron_id in neuron_ids
    ]
    print(f'Dispatching {len(tasks)} tasks ({len(neuron_ids)} neurons × {len(analyses_to_do)} analyses) across {N_WORKERS} workers')

    # Single pool: proc_unit_table is loaded once per worker via the initializer,
    # never re-pickled across individual tasks
    chunksize = max(1, len(tasks) // (N_WORKERS * 4))
    with multiprocessing.Pool(
            processes=N_WORKERS,
            initializer=_worker_init,
            initargs=(proc_unit_table,),
    ) as pool:
        results = list(tqdm.tqdm(
            pool.imap(_process_unit_task, tasks, chunksize=chunksize),
            total=len(tasks),
            desc="Processing units",
        ))

    os.makedirs(results_path, exist_ok=True)
    results_table = pd.DataFrame(results)

    # Derive per-neuron labels (modality preference, sensory, decision /
    # gated-decision per modality) and merge them on as new columns —
    # constant across all analysis_type rows for a given neuron_id.
    neuron_labels = compute_neuron_labels(results_table)
    results_table = results_table.merge(neuron_labels, on='neuron_id', how='left')

    mouse_name = results_table['mouse_id'].values[0]
    out_path = f'{results_path}/{mouse_name}_roc_results_new.csv'
    print('Saving results to:', out_path)
    results_table.to_csv(out_path, index=False)

    config = _build_run_config(nwb_file, mouse_name, analyses_to_do)
    config_path = f'{results_path}/{mouse_name}_roc_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print('Saving run config to:', config_path)

    if make_psth_plots:
        plot_psth_sanity_check(nwb_file, results_table, results_path)

def roc_analysis_old(nwb_file, results_path):
    """
    Perform ROC roc_analysis on spike data from a NWB file.
    :param nwb_file: path to NWB file
    :param results_path: path to save results
    :return:
    """
    print('Starting ROC roc_analysis for file:', nwb_file)

    # Process spike data
    proc_unit_table = extract_spike_data(nwb_file)
    mouse_id = proc_unit_table['mouse_id'].values[0]

    # Select ROC analyses based on available data
    if int(mouse_id[2:5]) < 115 and mouse_id[:2] =='AB':            # mice without passive trials
        analyses_to_do = ['whisker_active', 'auditory_active',
                          'wh_vs_aud_active', 'spontaneous_licks',
                          'choice', 'whisker_choice',
                          'baseline_choice', 'baseline_whisker_choice']
    else:
        analyses_to_do = ['whisker_passive_pre', # comparing pre vs post whisker stim activity in passive pre-learning trials
                          'whisker_passive_post', # comparing pre vs post whisker stim activity in passive post-learning trials
                          'whisker_active', # comparing pre vs post whisker stim activity in active hit trials
                          'whisker_pre_vs_post_learning', # comparing post whisker stim activity in passive pre vs post-learning trials
                          'auditory_passive_pre', # idem for auditory stim
                          'auditory_passive_post',
                          'auditory_active',
                          'auditory_pre_vs_post_learning',
                          'wh_vs_aud_passive_pre',  # comparing whisker vs auditory post stim activity in passive pre-learning trials
                          'wh_vs_aud_passive_post', # comparing whisker vs auditory post stim activity in passive post-learning trials
                          'wh_vs_aud_active', # comparing whisker vs auditory post stim activity in active hit trials
                          'wh_vs_aud_pre_vs_post_learning', # comparing whisker vs auditory post stim activity in passive pre vs post-learning trials
                          'spontaneous_licks', # comparing pre vs post spontaneous lick activity
                          'choice', # comparing post. trial start spikes in lick vs no-lick trials
                          'whisker_choice', # comparing post. stim spikes in whisker hit vs miss trials
                          'baseline_choice', # comparing pre. trial start spikes in lick vs no-lick trials
                          'baseline_whisker_choice' # comparing pre. stim spikes in whisker hit vs miss trials
                          ]

    # Init. global results
    results = []

    for analysis_type in analyses_to_do:
        print(f'ROC roc_analysis type: {analysis_type}')
        # Use multiprocessing to process each neuron_id in parallel
        neuron_ids = proc_unit_table['neuron_id'].unique()

        with multiprocessing.Pool(N_WORKERS) as pool:
            func = partial(process_unit, proc_unit_table=proc_unit_table, analysis_type=analysis_type, results_path=results_path)
            analysis_results = pool.map(func, neuron_ids)
            results.extend(analysis_results)

    # Create and save individual mouse data
    results_table = pd.DataFrame(results)
    mouse_name = results_table['mouse_id'].values[0]
    os.makedirs(results_path, exist_ok=True)
    print('Saving results to:', results_path)
    results_table.to_csv(f'{results_path}/{mouse_name}_roc_results_new.csv', index=False)

    return