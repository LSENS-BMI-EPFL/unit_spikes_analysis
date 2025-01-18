#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_spikes_analysis
@file: xcorr_utils.py
@time: 1/14/2025 4:44 PM
"""

# Imports
from mpi4py import MPI
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

from plotting_utils import get_excluded_areas, remove_top_right_frame
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
    # Get last spike of entire recording
    t_stop = unit_table['spike_times'].apply(lambda x: x[-1]).max()
    # Get unit data
    unit_row = unit_table[unit_table['unit_id'] == unit_id]
    spike_times = unit_row['spike_times'].iloc[0]
    unit_data = {
        'id': unit_id,
        't_start': 0,
        't_stop': t_stop, #arbitrary long time
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
            'cch_peak_value': interaction_result['cch_peak_value'],
            'int_type': interaction_result['int_type']
        }

    unit1_data, unit2_data = unit_pair
    print(f"Pair: {unit1_data['id']} - {unit2_data['id']}")


    # Create Neo SpikeTrain objects
    spike_train1 = neo.SpikeTrain(
        unit1_data['spike_times'] * pq.s,
        t_start=unit1_data['t_start'] * pq.s,
        t_stop=unit1_data['t_stop'] * pq.s
    )
    spike_train2 = neo.SpikeTrain(
        unit2_data['spike_times'] * pq.s,
        t_start=unit2_data['t_start'] * pq.s,
        t_stop=unit2_data['t_stop'] * pq.s
    )

    # Binarize spike trains
    binned_spike_train1 = BinnedSpikeTrain(spike_train1, BIN_SIZE * pq.ms, tolerance=None)
    binned_spike_train2 = BinnedSpikeTrain(spike_train2, BIN_SIZE * pq.ms, tolerance=None)

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


def xcorr_analysis_mpi(nwb_file, results_path):
    """
    Perform CCH analysis using mpi4py for distributed processing.
    """
    # MPI Initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Get the rank of the current process
    size = comm.Get_size() # Get the total number of processes


    if rank == 0: # Rank 0 reads the NWB file and scatters the data
        print(f"Starting CCH analysis with {size} MPI processes.")
        nwb_file_path = pathlib.Path(nwb_file)
        mouse_name = nwb_file_path.name[:5]
        unit_table = nwb_reader.get_unit_table(nwb_file)

        # Filter units
        unit_table = unit_table[(unit_table['bc_label'] == 'good') & (~unit_table['ccf_acronym'].isin(get_excluded_areas()))]
        unit_table['unit_id'] = unit_table.index
        unit_data = {unit_id: extract_unit_data(unit_table, unit_id) for unit_id in unit_table['unit_id'].unique()}
        unit_pairs = [(unit_data[id1], unit_data[id2]) for id1, id2 in combinations(unit_data.keys(), 2)]
    else:
        unit_pairs = None
        mouse_name = None

    # Broadcast mouse_name to all processes
    mouse_name = comm.bcast(mouse_name, root=0)

    # Scatter unit pairs across ranks
    unit_pairs = comm.bcast(unit_pairs, root=0)
    local_pairs = np.array_split(unit_pairs, size)[rank]

    # Process local pairs
    local_results = [process_pair(pair) for pair in tqdm(local_pairs, desc=f"Rank {rank}")]

    # Gather results at root
    all_results = comm.gather(local_results, root=0)

    if rank == 0: # Rank 0 saves the results and plots
        # Flatten results and save
        all_results = [item for sublist in all_results for item in sublist]

        # Again, flatten into a single list of dictionaries (in case two-sided interactions exist)
        all_results = [item for sublist in all_results for item in sublist]

        # Ensure it is a list of dictionaries
        keys = [set(d.keys()) for d in all_results]
        if len(set(frozenset(k) for k in keys)) > 1:
            print("Dictionaries have inconsistent keys!")

        results_df = pd.DataFrame.from_dict(all_results, orient='columns')
        output_path = pathlib.Path(results_path, mouse_name, 'xcorr_analysis')
        output_path.mkdir(parents=True, exist_ok=True)
        print('Saving in:', output_path)
        results_df.to_parquet(pathlib.Path(output_path, f'{mouse_name}_xcorr_df.parquet'), index=False)

        ### ---------------------------
        # Plot significant interactions
        # ----------------------------

        plt.switch_backend('Agg') # Switch to non-interactive backend
        fig_output = pathlib.Path(output_path, 'figures')
        fig_output.mkdir(parents=True, exist_ok=True)

        results_sig_df = results_df[results_df['significant'] == True]
        print(f"Plotting {len(results_sig_df)} significant interactions...")

        # Iterate over significant interactions
        for idx, row in results_sig_df.iterrows():
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
            fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
            title_txt = f'{mouse_name}: unit{source_id} ({source_ccf}) $\longrightarrow$ unit{target_id} ({target_ccf})\n'
            fig.suptitle(title_txt)

            for ax in axs:
                remove_top_right_frame(ax)
                ax.tick_params(axis='both', which='major', labelsize=15)
                ax.axvline(0, color='k', linestyle='--')
                ax.set_xlabel('Time lag [ms]', fontsize=15)
                ax.set_ylabel('Spike count', fontsize=15)
                ax.legend(frameon=False, loc='upper right')

            axs[0].plot(LAGS, cch, color='k', lw=1.5, label='CCH')
            axs[0].plot(LAGS, jittered_cch, color='peru', lw=1.5, label='Jittered CCH')

            axs[1].plot(LAGS, corrected_cch, color='k', lw=1.5, label='Corrected CCH')
            # Plot shaded area between corrected CCH +/- flank_sd
            axs[1].fill_between(LAGS, corrected_cch - flank_sd, corrected_cch + flank_sd, color='indianred',
                                edgecolor=None, alpha=0.4)

            fig.tight_layout()

            fig_name = pathlib.Path(fig_output, f'{mouse_name}_{source_id}_{target_id}_cch.png')
            plt.savefig(fig_name, format='png', dpi=300)
            plt.close()

    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed CCH Analysis")
    parser.add_argument("nwb_file", type=str, help="Path to the NWB file.")
    parser.add_argument("results_path", type=str, help="Path to save results.")
    args = parser.parse_args()

    xcorr_analysis_mpi(args.nwb_file, args.results_path)