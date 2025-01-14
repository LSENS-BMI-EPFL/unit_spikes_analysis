#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_spikes_analysis
@file: raster_utils.py
@time: 1/14/2025 12:15 PM
"""

# Imports
import os
import pathlib
import numpy as np
import pandas as pd
import multiprocessing
import seaborn
import matplotlib.pyplot as plt
from matplotlib import gridspec

import NWB_reader_functions as nwb_reader

from plotting_utils import remove_top_right_frame, save_figure_with_options
from roc_utils import filter_lick_times

def plot_unit_raster(unit_data, event_times_dict, align_event, results_path):
    """
    Plot spike rasters for single-units.
    :param unit_data: single-unit data from pd.DataFrame
    :param event_times_dict: dict of event times for alignment
    :param align_event: event to align spikes to
    :param results_path: path to save results
    :return:
    """

    unit_id = unit_data['unit_id'].values[0]
    mouse_id = unit_data['mouse_id'].values[0]
    area = unit_data['ccf_acronym'].values[0]
    spike_times = unit_data['spike_times'].values[0]

    # Plot settings
    pre_event_win = 0.5  # in sec
    post_event_win = 0.5  # in sec
    time_ticks = np.arange(-pre_event_win, post_event_win, 0.25)
    time_ticks = [-0.5, 0, 0.5]
    #time_ticks_labels = np.arange(-pre_event_win, post_event_win, 0.25) * 1e3  # in msec
    #time_ticks_labels = np.round(time_ticks_labels).astype(int)  # format as int
    time_ticks_labels = [-0.5, 0, 0.5]

    if align_event == 'trial_start':
        event_times = [event_times_dict[k] for k in event_times_dict.keys() if 'lick_time' not in k]
        event_times = np.concatenate(event_times)
        n_events = len(event_times)

        event_type_dict = {'false_alarm_trial': 'k',
                            'correct_rejection_trial': 'grey',
                            'auditory_hit_trial': 'mediumblue',
                            'auditory_miss_trial': 'deepskyblue',
                            'whisker_hit_trial': 'forestgreen',
                            'whisker_miss_trial': 'crimson',
                           }


    elif align_event == 'piezo_lick_times':
        event_times = event_times_dict['piezo_lick_times']
        stimuli = {k:v for k,v in event_times_dict.items() if 'whisker' in k or 'auditory' in k}
        event_times = filter_lick_times(lick_times=event_times, interval=1, **stimuli)
        n_events = len(event_times)
        event_type_dict = {'piezo_lick_times': 'k'}

    trial_ticks = np.arange(0, n_events, 100)
    ft_size = 13
    #line_prop = dict(joinstyle='miter')

    # Make figure
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
    remove_top_right_frame(ax)

    if align_event == 'trial_start':
        ax.set_ylabel('Trials', fontsize=ft_size - 4)
        title_str = f'{mouse_id}, unit {unit_id} ({area})'
        ax.set_title(title_str, fontsize=ft_size / 2)

    elif align_event == 'piezo_lick_time':
        ax.set_ylabel('Event', fontsize=ft_size - 4)
        title_str = f'{mouse_id}, unit {unit_id}, {area}'
        ax.set_title(title_str, fontsize=ft_size / 2)

    ax.set_xlabel('Time [s]', fontsize=ft_size - 4)
    ax.set_yticks(ticks=trial_ticks, labels=trial_ticks, fontsize=ft_size - 4)
    ax.set_xticks(ticks=time_ticks, labels=time_ticks_labels, fontsize=ft_size - 4)
    ax.tick_params(axis='both', which='major', labelsize=ft_size - 4)
    trial_type_delimiters = []

    # Iterate over trial types
    for idx, (event_type, event_color) in enumerate(event_type_dict.items()):

        # Get event times for this condition
        trial_type_starts = event_times_dict[event_type]
        trial_type_delimiters.append(len(trial_type_starts))
        spike_times_aligned = []

        # Iterate over ordered trials
        for t_time in trial_type_starts:
            start_sec = t_time - pre_event_win
            end_sec = t_time + post_event_win
            spike_times_win = spike_times[(np.where((spike_times >= start_sec) & (spike_times <= end_sec)))] - t_time
            spike_times_aligned.append(spike_times_win)

        # Add raster per trial type
        if idx == 0:
            offset_start = 0
        else:
            offset_start = np.cumsum(trial_type_delimiters)[idx - 1]
        offset_end = np.cumsum(trial_type_delimiters)[idx]

        ax.eventplot(positions=spike_times_aligned,
                     lineoffsets=np.arange(offset_start, offset_end),
                     linewidths=1,
                     linelengths=2,
                     colors=[event_color] * len(spike_times_aligned),
                     )
    ax.axvline(x=0, lw=1, ls='--', c='k', zorder=0)
    ax.set_ylim(0, n_events)
    ax.set_xlim(-pre_event_win, post_event_win)

    fig.tight_layout()
    plt.show()

    # Save
    if align_event == 'trial_start':
        file_name = f'{mouse_id}_unit{unit_id}_raster_trial_starts'
    elif align_event == 'piezo_lick_times':
        file_name = f'{mouse_id}_unit{unit_id}_raster_piezo_lick_times'
    save_figure_with_options(fig, ['png','svg'], file_name,
                             results_path, dark_background=False)


    return

def plot_rasters(nwb_file, results_path):
    """
    Plot spike rasters for single-units.
    :param nwb_file: path to NWB file
    :param results_path: path to save results
    :return:
    """
    print('Plotting spike rasters for single-units...')

    nwb_file_path = pathlib.Path(nwb_file)
    mouse_name = nwb_file_path.name[:5]
    unit_table = nwb_reader.get_unit_table(nwb_file)
    trial_table = nwb_reader.get_trial_table(nwb_file)

    # Keep well-isolated unit_table with a valid brain region label
    unit_table = unit_table[(unit_table['bc_label'] == 'good')
                            &
                            (unit_table['ccf_acronym'].str.contains('[A-Z]'))]
    unit_table['mouse_id'] = mouse_name

    # Use index as new column named "unit_id", then reset
    unit_table['unit_id'] = unit_table.index
    unit_table.reset_index(drop=True, inplace=True)

    # If context is only NaNs, set to 'active' for all trials
    if trial_table['context'].isnull().all():
        trial_table['context'] = 'active'
    # Mouse with passive and active periods
    else:
        # Separate passive trials from pre vs post trials
        trials_mid_index = len(trial_table) // 2  # find middle of session
        trial_table['context'] = trial_table.apply(lambda row:
                                         'active' if row['context'] == 'active' else
                                         ('passive_pre' if row[
                                                               'context'] == 'passive' and row.name < trials_mid_index else
                                          'passive_post'), axis=1)

    # Get session event_times for alignment
    event_time_dict = nwb_reader.get_behavioral_events(nwb_file)

    align_events = ['trial_start', 'piezo_lick_times']
    align_events = ['piezo_lick_times']

    # Iterate over units
    for unit_id in unit_table['unit_id'].unique()[0:10]:

        for align_event in align_events:

            unit_data = unit_table[unit_table['unit_id'] == unit_id]
            plot_unit_raster(unit_data, event_time_dict, align_event, results_path)







    return
