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
from tqdm import tqdm
from itertools import combinations
from multiprocessing import Pool

import neo
import numpy as np
import quantities as pq
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plotting_utils import get_excluded_areas
import NWB_reader_functions as nwb_reader
import cch_utils
from elephant.conversion import BinnedSpikeTrain

from elephant.spike_train_correlation import cross_correlation_histogram
from elephant.spike_train_surrogates import jitter_spikes

# Declare constants globally
LAG_WINDOW = 100
PEAK_WINDOW_MAX = 13
NUM_PROCESSES = 10
BIN_SIZE = 1
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
    unit_data = {
        'id': unit_id,
        't_start': 0,
        't_end': 1e5, #arbitrary long time
        'spike_times': spike_times,
        'ccf_acronym': unit_row['ccf_acronym'].iloc[0],
        'ccf_parent_acronym': unit_row['ccf_parent_acronym'].iloc[0],
        'firing_rate': unit_row['firing_rate'].iloc[0],
    }
    return unit_data

def process_pair(unit_pair):
    """
    Process a pair of units to compute cross-correlation histograms (CCH) and identify significant interactions.
    :param unit_pair: tuple of two unit data dictionaries, from unit_table pd.DataFrame
    """
    def create_interaction_result(source, target, cch, jittered_cch, corrected_cch, interaction_result, flipped=False):
        """ Helper function to create interaction result dictionary."""
        return {
            'source_id': source['id'],
            'target_id': target['id'],
            'source_ccf': source['ccf_acronym'],
            'target_ccf': target['ccf_acronym'],
            'source_ccf_parent': source['ccf_parent_acronym'],
            'target_ccf_parent': target['ccf_parent_acronym'],
            'source_fr': source['firing_rate'],
            'target_fr': target['firing_rate'],
            'cch': np.flip(cch) if flipped else cch,
            'bin_size_ms': BIN_SIZE,
            'jittered_cch': np.flip(jittered_cch) if flipped else jittered_cch,
            'corrected_cch': np.flip(corrected_cch) if flipped else corrected_cch,
            # 'source_waveform_type': source.get('waveform_type'),  # Uncomment if needed
            # 'target_waveform_type': target.get('waveform_type'),
            'flank_sd': interaction_result['flank_sd'],
            'significant': interaction_result['significant'],
            'lag_index': interaction_result['lag_index'],
            'cch_value': interaction_result['cch_value'],
            'int_type': interaction_result['int_type']
        }

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

    # Binarize spike trains
    binned_spike_train1 = BinnedSpikeTrain(spike_train1, BIN_SIZE * pq.ms)
    binned_spike_train2 = BinnedSpikeTrain(spike_train2, BIN_SIZE * pq.ms)

    # Compute CCH and jitter-corrected CCH
    cch_result = cch_utils.calculate_cch(binned_spike_train1, binned_spike_train2, LAG_WINDOW)
    jittered_cch = cch_utils.calculate_jitter_cch(binned_spike_train1, spike_train2, JITTER_WINDOW, LAG_WINDOW, NUM_SURROGATES)

    # Calculate corrected CCH
    corrected_cch = cch_result[0] - jittered_cch

    # Process significant interactions for both directions
    results = []

    # Direction: unit1 -> unit2
    interaction_result = cch_utils.find_interactions(corrected_cch, THRESHOLD, PEAK_WINDOW_MAX)
    results.append(create_interaction_result(unit1_data, unit2_data, cch_result[0], jittered_cch, corrected_cch, interaction_result))

    # Direction: unit2 -> unit1
    corrected_cch_flipped = np.flip(corrected_cch)
    interaction_result = cch_utils.find_interactions(corrected_cch_flipped, THRESHOLD, PEAK_WINDOW_MAX)
    results.append(create_interaction_result(unit2_data, unit1_data, cch_result[0], jittered_cch, corrected_cch, interaction_result, flipped=True))

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

    # Keep well-isolated unit_table with a valid brain region label

    unit_table = unit_table[(unit_table['bc_label'] == 'good') & (~unit_table['ccf_acronym'].isin(get_excluded_areas()))]
    unit_table['mouse_id'] = mouse_name
    unit_table = convert_electrode_group_object_to_columns(unit_table)
    unit_table.drop('electrode_group', axis=1, inplace=True)

    # Use index as new column named "unit_id", then reset
    unit_table['unit_id'] = unit_table.index
    unit_table.reset_index(drop=True, inplace=True)

    # Get all unique pairs of single units
    unit_ids = unit_table['unit_id'].unique()
    unit_data = {unit_id: extract_unit_data(unit_table, unit_id)
                 for unit_id in unit_ids}
    unit_pairs = [(unit_data[id1], unit_data[id2])
                  for id1, id2 in combinations(unit_ids, 2)][0:50] #TODO: remove slicing
    print('Number of unit pairs:', len(unit_pairs))
    start_time_total = time.perf_counter()

    with Pool(processes=2) as pool:
        results_with_durations = list(tqdm(pool.imap(process_pair, unit_pairs), total=len(unit_pairs)))

    end_time_total = time.perf_counter()
    total_duration = end_time_total - start_time_total
    print(f"Total processing time: {total_duration / 60:.2f} minutes.")

    # Concatenate results and save
    xcorr_list = [item for sublist in results_with_durations for item in sublist]
    xcorr_df = pd.DataFrame(xcorr_list)
    xcorr_df.to_parquet(os.path.join(results_path, f'{mouse_name}_xcorr_df.parquet'), index=False)
    xcorr_df.to_csv(os.path.join(results_path, f'{mouse_name}_xcorr_df.csv'), index=False) #TODO: after testing, remove csv export

    # Then, plot significant interactions
    xcorr_sig_df = xcorr_df[xcorr_df['significant'] == True]
    # Iterate over significant interactions
    for idx, row in xcorr_sig_df.iterrows():
        source_id = row['source_id']
        target_id = row['target_id']
        source_ccf = row['source_ccf']
        target_ccf = row['target_ccf']
        source_ccf_parent = row['source_ccf_parent']
        target_ccf_parent = row['target_ccf_parent']
        cch = row['cch']
        jittered_cch = row['jittered_cch']
        corrected_cch = row['corrected_cch']
        flank_sd = row['flank_sd']
        lag_index = row['lag_index']
        cch_value = row['cch_value']
        int_type = row['int_type']

        # Plot CCH
        plt.figure(figsize=(5, 5))
        plt.plot(LAGS, cch, color='black', label='CCH')
        plt.plot(LAGS, jittered_cch, color='red', label='Jittered CCH')
        plt.plot(LAGS, corrected_cch, color='blue', label='Corrected CCH')
        plt.axhspan(corrected_cch-flank_sd, corrected_cch+flank_sd, color='gray', alpha=0.5, label='Flank SD')
        plt.axvline(lag_index, color='green', linestyle='--', label='Lag index')
        plt.title(f'CCH: {source_ccf} -> {target_ccf}')
        plt.xlabel('Time lag [ms]')
        plt.ylabel('Spike count')
        plt.legend(frameon=False)
        plt.savefig(os.path.join(results_path, f'{mouse_name}_{source_id}_{target_id}_cch.png'))
        plt.close()


    return