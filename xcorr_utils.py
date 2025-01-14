#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_spikes_analysis
@file: xcorr_utils.py
@time: 1/14/2025 4:44 PM
"""

# Imports
import os
import time
import pathlib
import tqdm
from itertools import combinations
from multiprocessing import Pool

import neo
import numpy as np
import quantities as pq
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import NWB_reader_functions as nwb_reader
import cch_utils
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import cross_correlation_histogram
from elephant.spike_train_surrogates import jitter_spikes

# Declare constants globally
LAG_WINDOW = 100
PEAK_WINDOW = 13
NUM_PROCESSES = 10
BIN_SIZE = 1 * pq.ms
JITTER_WINDOW = 10 * pq.ms
NUM_SURROGATES = 10
SAMPLING_RATE = 1000 * pq.Hz
THRESHOLD = 7.0
LAGS = np.arange(-100, 101, 1) * pq.ms

def convert_electrode_group_object_to_columns(data):
    """
    Convert electrode group object to dictionary.
    Creates a new column in the dataframe.
    :param data: pd.DataFrame containing the NWB electrode group field.
    :return:
    """
    elec_group_list = data['electrode_group'].values
    elec_group_name = [e.name for e in elec_group_list]
    data['electrode_group'] = elec_group_name

    elec_group_location = [e.location.replace('nan', 'None') for e in elec_group_list]
    elec_group_location_dict = [eval(e) for e in elec_group_location]
    data['location'] = elec_group_location_dict
    data['target_region'] = [e.get('area') for e in elec_group_location_dict]

    return data

def extract_unit_data(unit_table, unit_id):
    unit_row = unit_table[unit_table['unit_id'] == unit_id]
    spike_times = unit_row['spike_times'].iloc[0]
    last_spike_time = spike_times.iloc[-1]
    unit_data = {
        'id': unit_id,
        't_start': 0,
        't_end': last_spike_time, # 10000 or last spike overall?
        'spike_times': spike_times,
        'ccf_acronym': unit_row['ccf_acronym'].iloc[0],
        'ccf_parent_acronym': unit_row['ccf_parent_acronym'].iloc[0],
        'firing_rate': unit_row['firing_rate'].iloc[0],
    }
    return unit_data

def process_pair(unit_pair):

    unit1_data, unit2_data = unit_pair

    # Create Neo SpikeTrain objects
    spike_train1 = neo.SpikeTrain(
        unit1_data['spike_times'] * pq.s,
        t_start=unit1_data['t_start'] * pq.s,
        t_stop=unit1_data['t_end'] * pq.s
    )

    spike_train2 = neo.SpikeTrain(
        unit2_data['spike_times'] * pq.s,
        t_start=unit2_data['t_start'] * pq.s,
        t_stop=unit2_data['t_end'] * pq.s
    )

    # Convert to BinnedSpikeTrain
    binned_spike_train_1 = BinnedSpikeTrain(spike_train1, BIN_SIZE)
    binned_spike_train_2 = BinnedSpikeTrain(spike_train2, BIN_SIZE)

    # Compute CCH
    cch = cch_utils.calculate_cch(binned_spike_train_1, binned_spike_train_2, LAG_WINDOW)

    # Compute jitter-corrected CCH
    jittered_cch = cch_utils.calculate_jitter_cch(
        binned_spike_train_1,
        spike_train2,
        JITTER_WINDOW,
        LAG_WINDOW,
        NUM_SURROGATES
    )

    # Calculate corrected CCH
    corrected_cch = cch[0] - jittered_cch

    # Find significant interactions
    is_significant, flank_sd, max_value, peak_lag = cch_utils.find_interactions(corrected_cch, THRESHOLD,

                                                                                PEAK_WINDOW)  # list of dictionaries for peak informations
    results = []

    # Plot and save figures for significant interactions
    if is_significant:
        #TODO: determine location of save_path
        # TODO: file name

        #plot_filename = f"./{FOLDER_NAME}/plots/cch_unit{unit1_data['id']}_unit{unit2_data['id']}.png"
        cch_utils.plot_cch_correction(
            cch=cch[0],
            mean_jittered_cch=jittered_cch,
            corrected_cch=corrected_cch,
            peak_value=max_value,
            peak_lag=peak_lag,
            flank_sd=flank_sd,
            save_path=plot_filename,
            bin_size=BIN_SIZE,
            maxlag=LAG_WINDOW,
            lag_window=PEAK_WINDOW,
            title=f'CCH unit{unit1_data["id"]}({unit1_data["ccf_acronym"]}) - unit{unit2_data["id"]}({unit2_data["ccf_acronym"]})'
        )
    # TODO: this could have potentially several interactions
    interaction_results = {
        'source_id': unit1_data['id'],
        'target_id': unit2_data['id'],
        'source_ccf': unit1_data['ccf_acronym'],
        'target_ccf': unit2_data['ccf_acronym'],
        'source_ccf_parent': unit1_data['ccf_parent_acronym'],
        'target_ccf_parent': unit2_data['ccf_parent_acronym'],
        'source_fr': unit1_data['firing_rate'],
        'target_fr': unit2_data['firing_rate'],
        'source_waveform_type': unit1_data['waveform_type'],
        'target_waveform_type': unit2_data['waveform_type'],
        'is_significant': is_significant,
        'cch_max_peak_value': max_value,
        'cch_max_peak_lag': peak_lag
        }

    results.append(interaction_results)
    return results


def xcorr_analysis(nwb_file, results_path):
    """
    Compute cross-correlation histograms for all single-unit pairs in a NWB file.
    :param nwb_file: path to NWB file
    :param results_path: path to save results
    :return:
    """
    print('Starting CCHs for file:', nwb_file)

    nwb_file_path = pathlib.Path(nwb_file)
    mouse_name = nwb_file_path.name[:5]
    unit_table = nwb_reader.get_unit_table(nwb_file)
    trial_table = nwb_reader.get_trial_table(nwb_file)

    # Keep well-isolated unit_table with a valid brain region label
    unit_table = unit_table[(unit_table['bc_label'] == 'good') & (unit_table['ccf_acronym'].str.contains('[A-Z]'))]
    unit_table['mouse_id'] = mouse_name
    unit_table = convert_electrode_group_object_to_columns(unit_table)
    unit_table.drop('electrode_group', axis=1, inplace=True)

    # Get all single-unit pairs
    unit_ids = unit_table['unit_id'].unique()
    unit_data = {unit_id: extract_unit_data(unit_table, unit_id)
                 for unit_id in unit_ids}
    unit_pairs = [(unit_data[id1], unit_data[id2])
                  for id1, id2 in combinations(unit_ids, 2)]

    start_time_total = time.perf_counter()

    with Pool(processes=os.cpu_count()-1) as pool:
        results_with_durations = list(tqdm(pool.imap(process_pair, unit_pairs), total=len(unit_pairs)))

    end_time_total = time.perf_counter()
    total_duration = end_time_total - start_time_total
    print(f"Total processing time: {total_duration / 60:.2f} minutes.")

    # Concatenate results and save
    xcorr_list = [item for sublist in results_with_durations for item in sublist]
    xcorr_df = pd.DataFrame(xcorr_list)
    xcorr_df.to_parquet(os.path.join(results_path, f'{mouse_name}_xcorr_df.parquet'), index=False)

    return