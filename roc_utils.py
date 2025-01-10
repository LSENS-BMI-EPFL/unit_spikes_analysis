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

    # Define event types and contexts
    periods = trial_table['context'].unique()
    has_context = 'active' in periods and 'passive_pre' in periods
    # context = ["passive", "active"]

    event_types = ['whisker', 'auditory', 'spontaneous_licks']
    time_windows = {
        'pre': (-0.2, 0),
        'post': (0, 0.2),
        'spontaneous': (-0.4, -0.2, 0, 0.2)
    }

    for event_type in event_types:

        specific_contexts = ['active', 'passive_pre', 'passive_post'] if has_context else ['active']

        # Initialize columns #TODO: improve this and table format
        if event_type == 'spontaneous_licks':
            specific_contexts = [''] # context irrelevant for spontaneous licks

            if event_type + '_pre_spikes' not in unit_table.columns:
                unit_table[event_type + '_pre_spikes'] = [[] for _ in range(len(unit_table))]
            if event_type + '_post_spikes' not in unit_table.columns:
                unit_table[event_type + '_post_spikes'] = [[] for _ in range(len(unit_table))]

        else:
            for context in specific_contexts:
                if event_type + "_" + context + '_pre_spikes' not in unit_table.columns:
                    unit_table[event_type + "_" + context + '_pre_spikes'] = [[] for _ in range(len(unit_table))]
                if event_type + "_" + context + '_post_spikes' not in unit_table.columns:
                    unit_table[event_type + "_" + context + '_post_spikes'] = [[] for _ in range(len(unit_table))]

        # Get count data for each unit
        for context in specific_contexts:

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

                # Add count data to unit table #TODO: reformat to have only categorical columns
                # TODO: would this allow easier comparison of different counts?
                col_prefix = f"{event_type}_{context}".strip('_')
                unit_table.at[idx, f'{col_prefix}_pre_spikes'] = pre_counts
                unit_table.at[idx, f'{col_prefix}_post_spikes'] = post_counts

    return unit_table


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

    # Save processed unit table
    if 'electrode_group' in proc_unit_table.columns: # drop problematic column
        proc_unit_table = proc_unit_table.drop(columns=['electrode_group'])

    # Create and save individual mouse data to a parquet file
    mouse_name = proc_unit_table['mouse_id'].values[0]
    proc_unit_table.to_parquet(f'{results_path}/{mouse_name}_processed_data.parquet', index=False)

    # Perform ROC analysis


    pass
