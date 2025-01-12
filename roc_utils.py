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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import NWB_reader_functions as nwb_reader


def process_nwb_tables(nwb_file):
    """
    Process unit and trial table from a NWB file.
    :param nwb_file:
    :return:
    """
    # Convert NWB units and trials tables into Pandas DataFrames
    units = nwb_file.units.to_dataframe()
    trials = nwb_file.trials.to_dataframe()

    # Keep well-isolated units with a valid brain region label
    units = units[(units['bc_label'] == 'good')
                  &
                  (units['ccf_acronym'].str.contains('[A-Z]'))]


    # Keep fewer columns only
    columns_to_keep = ["cluster_id", "firing_rate", "ccf_acronym", "ccf_name", "ccf_parent_acronym", "ccf_parent_id",
                       "ccf_parent_name", "spike_times"]
    units = units[columns_to_keep]

    # Use index as new column named "unit_id", then reset
    units['unit_id'] = units.index
    units.reset_index(drop=True, inplace=True)


    # Separate passive trials from pre vs post trials
    trials_mid_index = len(trials) // 2  # find middle of session
    trials['context'] = trials.apply(lambda row:
                                     'active' if row['context'] == 'active' else
                                     ('passive_pre' if row['context'] == 'passive' and row.name < trials_mid_index else
                                      'passive_post'), axis=1)
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

def filtered_lick_times(nwbfile, interval=1):
    """ Extract and filter piezo lick times from NWB data. """
    behavior = nwbfile.processing['behavior']
    events = behavior.data_interfaces['BehavioralEvents']

    # Get piezo lick times
    piezo_lick_times = events.time_series['piezo_lick_times'].data[:]

    # Get stimulus times
    stim_times = {
        'auditory_hit': events.time_series['auditory_hit_trial'].data[:],
        'auditory_miss': events.time_series['auditory_miss_trial'].data[:],
        'whisker_hit': events.time_series['whisker_hit_trial'].data[:],
        'whisker_miss': events.time_series['whisker_miss_trial'].data[:]
    }

    # Remove lick times within interval of stimulus events
    filtered_lick_times = filter_lick_times(piezo_lick_times, interval, **stim_times)
    return filtered_lick_times


def extract_event_times(nwb_file, event_type='whisker', context='passive', has_context=True):
    """ Extract event times based on stimulus type and context. """
    _, trials = process_nwb_tables(nwb_file)

    if event_type == 'spontaneous_licks':
        event_times = filtered_lick_times(nwb_file)
    else:
        condition = trials[event_type + '_stim'] == 1
        if has_context:
            condition &= trials['context'] == context # boolean refinement: add context condition on top of stimulus type

        # Keep Hits only in active context
        condition &= trials['lick_flag'] == 1 if context == 'active' else True

        event_times = trials[condition]['start_time'].values

    return event_times


def process_spike_data(nwb_file):
    """
    Process spike data from a NWB file.
    :param nwb_file: path to NWB file
    :return:
    """

    nwb_file_path = pathlib.Path(nwb_file)
    mouse_name = nwb_file_path.name[:5]
    nwb = nwb_reader.read_nwb_file(nwb_file_path)

    unit_table, trial_table = process_nwb_tables(nwb)
    unit_table['mouse_id'] = mouse_name

    # Store processed data
    proc_unit_table = []

    # Define event types and contexts
    contexts = trial_table['context'].unique()
    has_context = 'active' in contexts and 'passive_pre' in contexts

    event_types = ['whisker', 'auditory', 'spontaneous_licks']
    time_windows = {
        'pre': (-0.2, 0),
        'post': (0, 0.2),
        'spontaneous': (-0.4, -0.2, 0, 0.2)
    }

    for event_type in event_types:

        contexts = ['active', 'passive_pre', 'passive_post'] if has_context else ['active']

        # Initialize columns #TODO: improve this and table format
        if event_type == 'spontaneous_licks':
            contexts = [''] # context irrelevant for spontaneous licks
#
        #    #if event_type + '_pre_spikes' not in unit_table.columns:
        #    #    unit_table[event_type + '_pre_spikes'] = [[] for _ in range(len(unit_table))]
        #    #if event_type + '_post_spikes' not in unit_table.columns:
        #    #    unit_table[event_type + '_post_spikes'] = [[] for _ in range(len(unit_table))]
#
        #else:
        #    for context in contexts:
        #        if event_type + "_" + context + '_pre_spikes' not in unit_table.columns:
        #            unit_table[event_type + "_" + context + '_pre_spikes'] = [[] for _ in range(len(unit_table))]
        #        if event_type + "_" + context + '_post_spikes' not in unit_table.columns:
        #            unit_table[event_type + "_" + context + '_post_spikes'] = [[] for _ in range(len(unit_table))]

        # Get count data for each unit
        for context in contexts:

            # Extract list of event times
            event_times = extract_event_times(nwb, event_type, context, has_context)

            for idx, unit in unit_table.iterrows():
                spike_times = unit['spike_times']
                pre_counts, post_counts = [], []

                for event in event_times:
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

                #proc_unit_table.at[idx, 'event'] = event_type
                #proc_unit_table.at[idx, 'context'] = context
                #proc_unit_table.at[idx, 'pre_spikes'] = pre_counts
                #proc_unit_table.at[idx, 'post_spikes'] = post_counts

                # Add count data to unit table #TODO: reformat to have only categorical columns
                # TODO: would this allow easier comparison of different counts?
                #col_prefix = f"{event_type}_{context}".strip('_')
                #unit_table.at[idx, f'{col_prefix}_pre_spikes'] = pre_counts
                #unit_table.at[idx, f'{col_prefix}_post_spikes'] = post_counts

    # Convert to DataFrame
    proc_unit_table = pd.DataFrame(proc_unit_table)

    return proc_unit_table

def calculate_roc(class_1_counts, class_2_counts):
    """
    Calculate receiver operating characteristic its area under the curve, between spike counts for two classes.
    :param class_1_counts: list of spike counts for class 1 for each event, e.g. pre-stim spikes
    :param class_2_counts: list of spike counts for class 2 for each event, e.g. post-stim spikes
    :return:
    """
    # Combine spike count data and class labels
    spike_counts = np.concatenate([class_1_counts, class_2_counts])
    labels = np.concatenate([np.zeros(len(class_1_counts)), np.ones(len(class_2_counts))])

    # Balance classes using sample weights inversely proportional to class frequency
    sample_weights = compute_sample_weight('balanced', labels)

    # Check if labels have at least two classes
    if len(np.unique(labels)) < 2:
        return None, None, None, np.nan  # return NaN for ROC AUC or another default

    # Compute the ROC curve and area under the curve
    fpr, tpr, thresholds = roc_curve(labels, spike_counts, sample_weight=sample_weights)
    roc_auc = roc_auc_score(labels, spike_counts)
    return fpr, tpr, thresholds, roc_auc

def select_spike_counts(unit_data, analysis_type):
    """ Select spike counts based on analysis type. """

    if analysis_type == 'whisker_passive_pre':
        spikes_1 = unit_data[(unit_data['event'] == 'whisker') & (unit_data['context'] == 'passive_pre')]['pre_spikes']
        spikes_2 = unit_data[(unit_data['event'] == 'whisker') & (unit_data['context'] == 'passive_pre')]['post_spikes']
    elif analysis_type == 'whisker_passive_post':
        spikes_1 = unit_data[(unit_data['event'] == 'whisker') & (unit_data['context'] == 'passive_post')]['pre_spikes']
        spikes_2 = unit_data[(unit_data['event'] == 'whisker') & (unit_data['context'] == 'passive_post')]['post_spikes']
    elif analysis_type == 'whisker_active':
        spikes_1 = unit_data[(unit_data['event'] == 'whisker') & (unit_data['context'] == 'active')]['pre_spikes']
        spikes_2 = unit_data[(unit_data['event'] == 'whisker') & (unit_data['context'] == 'active')]['post_spikes']
    elif analysis_type == 'whisker_pre_vs_post_learning':
        spikes_1 = unit_data[(unit_data['event'] == 'whisker') & (unit_data['context'] == 'passive_pre')]['post_spikes']
        spikes_2 = unit_data[(unit_data['event'] == 'whisker') & (unit_data['context'] == 'passive_post')]['post_spikes']
    elif analysis_type == 'auditory_passive_pre':
        spikes_1 = unit_data[(unit_data['event'] == 'auditory') & (unit_data['context'] == 'passive_pre')]['pre_spikes']
        spikes_2 = unit_data[(unit_data['event'] == 'auditory') & (unit_data['context'] == 'passive_pre')]['post_spikes']
    elif analysis_type == 'auditory_passive_post':
        spikes_1 = unit_data[(unit_data['event'] == 'auditory') & (unit_data['context'] == 'passive_post')]['pre_spikes']
        spikes_2 = unit_data[(unit_data['event'] == 'auditory') & (unit_data['context'] == 'passive_post')]['post_spikes']
    elif analysis_type == 'auditory_active':
        spikes_1 = unit_data[(unit_data['event'] == 'auditory') & (unit_data['context'] == 'active')]['pre_spikes']
        spikes_2 = unit_data[(unit_data['event'] == 'auditory') & (unit_data['context'] == 'active')]['post_spikes']
    elif analysis_type == 'auditory_pre_vs_post_learning':
        spikes_1 = unit_data[(unit_data['event'] == 'auditory') & (unit_data['context'] == 'passive_pre')]['post_spikes']
        spikes_2 = unit_data[(unit_data['event'] == 'auditory') & (unit_data['context'] == 'passive_post')]['post_spikes']
    elif analysis_type == 'wh_vs_aud_passive_pre':
        spikes_1 = unit_data[(unit_data['event'] == 'whisker') & (unit_data['context'] == 'passive_pre')]['post_spikes']
        spikes_2 = unit_data[(unit_data['event'] == 'auditory') & (unit_data['context'] == 'passive_pre')]['post_spikes']
    elif analysis_type == 'wh_vs_aud_passive_post':
        spikes_1 = unit_data[(unit_data['event'] == 'whisker') & (unit_data['context'] == 'passive_post')]['post_spikes']
        spikes_2 = unit_data[(unit_data['event'] == 'auditory') & (unit_data['context'] == 'passive_post')]['post_spikes']
    elif analysis_type == 'wh_vs_aud_active':
        spikes_1 = unit_data[(unit_data['event'] == 'whisker') & (unit_data['context'] == 'active')]['post_spikes']
        spikes_2 = unit_data[(unit_data['event'] == 'auditory') & (unit_data['context'] == 'active')]['post_spikes']
    elif analysis_type == 'wh_vs_aud_pre_vs_post_learning':
        spikes_1 = unit_data[(unit_data['event'] == 'whisker') & (unit_data['context'] == 'passive_pre')]['post_spikes']
        spikes_2 = unit_data[(unit_data['event'] == 'auditory') & (unit_data['context'] == 'passive_post')]['post_spikes']
    elif analysis_type == 'spontaneous_licks':
        spikes_1 = unit_data[(unit_data['event'] == 'spontaneous_licks')]['pre_spikes']
        spikes_2 = unit_data[(unit_data['event'] == 'spontaneous_licks')]['post_spikes']
    else:
        raise ValueError(f"Analysis type {analysis_type} not recognized.")

    # Make these into flattened arrays
    spikes_counts_1 = np.concatenate(spikes_1.values)
    spikes_counts_2 = np.concatenate(spikes_2.values)

    return spikes_counts_1, spikes_counts_2

def roc_analysis(nwb_file, results_path):
    """
    Perform ROC analysis on spike data from a NWB file.
    :param nwb_file: path to NWB file
    :param results_path: path to save results
    :return:
    """
    print('Starting ROC analysis for NWB file:', nwb_file)

    # Process spike data
    proc_unit_table = process_spike_data(nwb_file)

    # Save processed unit table #TODO: remove?
    if 'electrode_group' in proc_unit_table.columns: # drop problematic column
        proc_unit_table = proc_unit_table.drop(columns=['electrode_group'])

    # Create and save individual mouse data to a parquet file
    mouse_name = proc_unit_table['mouse_id'].values[0]
    proc_unit_table.to_parquet(f'{results_path}/{mouse_name}_processed_data.parquet', index=False)

    # Perform ROC analysis
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
                      'spontaneous_licks' # comparing pre vs post spontaneous lick activity
                      ]
    # Init. global results
    results = []

    for analysis_type in analyses_to_do:

        for unit_id in proc_unit_table['unit_id'].unique():
            unit_table = proc_unit_table[proc_unit_table['unit_id'] == unit_id]

            # Keep relevant columns for results
            cols_to_keep = ['mouse_id', 'unit_id', 'cluster_id', 'firing_rate', 'ccf_acronym', 'ccf_name', 'ccf_parent_acronym', 'ccf_parent_id', 'ccf_parent_name']
            res_dict = {col: unit_table[col].values[0] for col in cols_to_keep}
            res_dict.update({'analysis_type': analysis_type})

            # Select adequate spike counts and compute ROC
            spikes_1, spikes_2 = select_spike_counts(unit_table, analysis_type)
            fpr, tpr, thresholds, roc_auc = calculate_roc(spikes_1, spikes_2)

            res_dict.update({'auc': roc_auc})
            res_dict.update({'selectivity': 2*roc_auc - 1})
            res_dict.update({'fpr': fpr})
            res_dict.update({'tpr': tpr})
            res_dict.update({'thresholds': thresholds})

            results.append(res_dict)

    # Create and save individual mouse data to a parquet file
    results_table = pd.DataFrame(results)
    mouse_name = results_table['mouse_id'].values[0]
    results_table.to_parquet(f'{results_path}/{mouse_name}_roc_results.parquet', index=False)

    return
