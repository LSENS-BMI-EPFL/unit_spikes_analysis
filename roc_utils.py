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
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import NWB_reader_functions as nwb_reader


def process_nwb_tables(nwb_file):
    """
    Process unit and trial table from a NWB file.
    :param nwb_file: path to NWB file
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
        condition &= trials['lick_flag'] == 1 if context == 'active' else True
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

    return event_times


def extract_spike_data(nwb_file):
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

def  select_spike_counts(unit_data, analysis_type):
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
    elif analysis_type == 'choice':
        spikes_1 = unit_data[(unit_data['event'] == 'lick_trial') & (unit_data['context'] == 'active')]['post_spikes']
        spikes_2 = unit_data[(unit_data['event'] == 'no_lick_trial') & (unit_data['context'] == 'active')]['post_spikes']
    elif analysis_type == 'whisker_choice':
        spikes_1 = unit_data[(unit_data['event'] == 'whisker_hit') & (unit_data['context'] == 'active')]['post_spikes']
        spikes_2 = unit_data[(unit_data['event'] == 'whisker_miss') & (unit_data['context'] == 'active')]['post_spikes']
    else:
        raise ValueError(f"Analysis type {analysis_type} not recognized.")

    # Make these into flattened arrays
    try :
        spikes_counts_1 = np.concatenate(spikes_1.values)
        spikes_counts_2 = np.concatenate(spikes_2.values)
    except :
        spikes_counts_1 = []
        spikes_counts_2 = []

    return spikes_counts_1, spikes_counts_2


def process_unit(unit_id, proc_unit_table, analysis_type, results_path):
    """
    Process a single unit for ROC analysis.
    :param unit_id: unit ID
    :param proc_unit_table: processed unit table
    :param analysis_type: type of analysis
    :param results_path: path to save results
    :return:
    """
    unit_table = proc_unit_table[proc_unit_table['unit_id'] == unit_id]
    mouse_id = unit_table['mouse_id'].values[0]
    area = unit_table['ccf_parent_acronym'].values[0]

    # Keep relevant columns for results
    cols_to_keep = ['mouse_id', 'unit_id', 'cluster_id', 'firing_rate', 'ccf_acronym', 'ccf_name',
                    'ccf_parent_acronym', 'ccf_parent_id', 'ccf_parent_name']
    res_dict = {col: unit_table[col].values[0] for col in cols_to_keep}
    res_dict.update({'analysis_type': analysis_type, 'unit_id': unit_id, 'mouse_id': mouse_id, 'area': area})

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
    n_permutations = 100
    permuted_aucs = []
    for _ in range(n_permutations):
        _, _, _, roc_auc_permut = calculate_roc(spikes_1, spikes_2, shuffle=True)
        permuted_aucs.append(roc_auc_permut)
    permuted_aucs = np.array(permuted_aucs)

    # Calculate p-values as proportion of permuted AUCs greater than or equal to the observed AUC
    p_value_pos = np.sum(permuted_aucs >= roc_auc) / n_permutations  # one-tailed test: AUC greater than chance
    p_value_neg = np.sum(permuted_aucs <= roc_auc) / n_permutations

    # Determine significance and direction of signifiance based on p-values and analysis type
    if 'wh_vs_aud' in analysis_type:
        directions = ['auditory', 'whisker'] # auditory is spikes_2!
    else:
        directions = ['positive', 'negative']

    if p_value_pos < 0.05:
        is_significant = True
        res_dict.update({'significant': is_significant, 'direction': directions[0], 'p_value': p_value_pos, 'p_value_to_show': p_value_pos})
    elif p_value_neg < 0.05:
        is_significant = True
        res_dict.update({'significant': is_significant, 'direction': directions[1], 'p_value': p_value_neg, 'p_value_to_show': p_value_neg})
    else:
        is_significant = False
        res_dict.update({'significant': is_significant, 'direction': np.nan, 'p_value': p_value_pos, 'p_value_to_show': p_value_pos}) # here only p-value for positive direction is kept

    debug = False
    if debug and is_significant:

        # Subplots: 1. ROC curve 2. Histogram of permutted AUCs
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        for ax in axs:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=15)

        pval = res_dict['p_value_to_show']
        suptitle_text = f'ROC Analysis for mouse {mouse_id} unit {unit_id} ({area}) ({analysis_type})\n'
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
        fname = f'{mouse_id}_{unit_id}_{analysis_type}.png'
        results_path_fig = os.path.join(results_path, 'figures')
        os.makedirs(results_path_fig, exist_ok=True)
        fig.savefig(os.path.join(results_path_fig, fname))

    return res_dict

def roc_analysis(nwb_file, results_path):
    """
    Perform ROC analysis on spike data from a NWB file.
    :param nwb_file: path to NWB file
    :param results_path: path to save results
    :return:
    """
    print('Starting ROC analysis for file:', nwb_file)

    # Process spike data
    proc_unit_table = extract_spike_data(nwb_file)
    mouse_id = proc_unit_table['mouse_id'].values[0]

    # Select ROC analyses based on available data
    if int(mouse_id[2:5]) < 115 and mouse_id[:2] =='AB':
        analyses_to_do = ['whisker_active', 'auditory_active',
                          'wh_vs_aud_active', 'spontaneous_licks',
                          'choice', 'whisker_choice']
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
                          'spontaneous_licks' # comparing pre vs post spontaneous lick activity
                          'choice', # comparing post. trial start spikes in lick vs no-lick trials
                          'whisker_choice' # comparing post. stim spikes in whisker hit vs miss trials
                          ]

    # Init. global results
    results = []

    for analysis_type in analyses_to_do:

        # Use multiprocessing to process each unit_id in parallel
        unit_ids = proc_unit_table['unit_id'].unique()

        with multiprocessing.Pool(os.cpu_count()-2) as pool:
            func = partial(process_unit, proc_unit_table=proc_unit_table, analysis_type=analysis_type, results_path=results_path)
            analysis_results = pool.map(func, unit_ids)
            results.extend(analysis_results)

    # Create and save individual mouse data to a parquet file
    results_table = pd.DataFrame(results)
    mouse_name = results_table['mouse_id'].values[0]
    os.makedirs(results_path, exist_ok=True)
    print('Saving results to:', results_path)
    results_table.to_csv(f'{results_path}/{mouse_name}_roc_results_new.csv', index=False)

    return