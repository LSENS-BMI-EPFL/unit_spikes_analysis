#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: brain_wide_analysis
@file: neural_utils.py
@time: 2/11/2024 9:41 PM
"""
# Imports
import numpy as np
import json
import pandas as pd
import os
import scipy.ndimage
from multiprocessing import Pool
import matplotlib
matplotlib.use('Agg') # 'TkAgg' 'Agg' 'Qt5Agg'
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# Custom imports
import NWB_reader_functions as nwb_reader
import allen_utils as allen


TRIAL_MAP = {
    0: 'whisker_miss',
    1: 'auditory_miss',
    2: 'whisker_hit',
    3: 'auditory_hit',
    4: 'correct_rejection',
    5: 'false_alarm',
    6: 'association',
}


def process_single_nwb(nwb):
    try:
        beh_type, day = nwb_reader.get_bhv_type_and_training_day_index(nwb)
        if day != 0:
            return None

        unit_table = nwb_reader.get_unit_table(nwb)
        if unit_table is None or 'bc_label' not in unit_table.columns:
            return None

        trial_table = nwb_reader.get_trial_table(nwb)
        trial_table['trial_id'] = trial_table.index

        mouse_id = nwb_reader.get_mouse_id(nwb)
        session_id = nwb_reader.get_session_id(nwb)

        trial_table['mouse_id'] = mouse_id
        trial_table['session_id'] = session_id
        trial_table['context'] = trial_table['context'].astype(str)

        if trial_table['context'].str.contains('nan').all():
            trial_table['context'] = 'active'
        else:
            trial_table['context'] = trial_table['context'].fillna('active')
            trial_table['context'] = trial_table['context'].replace('nan','active')


        unit_table['mouse_id'] = mouse_id
        print('Warning: number of root neurons :', mouse_id, len(unit_table[unit_table.ccf_acronym=='root']))
        root_units = unit_table[unit_table.ccf_acronym=='root']
        if not root_units.empty:
            elec_groups = root_units['electrode_group'].unique()
            elec_names = [e.name for e in elec_groups]
            #print(f"Root units found in {mouse_id}: {len(root_units)} with electrode groups: {elec_names}")

        unit_table = convert_electrode_group_object_to_columns(unit_table)

        return {
            'nwb': nwb,
            'trial_table': trial_table,
            'unit_table': unit_table
        }

    except Exception as e:
        print(f"Error processing {nwb}: {e}")
        return None

def combine_ephys_nwb(nwb_list, max_workers=24):
    """
    Combine neural and behavioural data from multiple NWB files using multiprocessing and tqdm.
    :param nwb_list: list of NWB file paths.
    :param max_workers: number of parallel processes.
    :return: (trial_table, unit_table, ephys_nwb_list)
    """
    ephys_nwb_list = []
    trial_table_list = []
    unit_table_list = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_nwb, nwb): nwb for nwb in nwb_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading NWB files"):
            result = future.result()
            if result is None:
                continue
            ephys_nwb_list.append(result['nwb'])
            trial_table_list.append(result['trial_table'])
            unit_table_list.append(result['unit_table'])

    print(f"Found {len(ephys_nwb_list)} NWB files with ephys data.")
    print(f"Available NWB files {len(ephys_nwb_list)}:", sorted([os.path.basename(nwb) for nwb in ephys_nwb_list]))

    trial_table = pd.concat(trial_table_list, ignore_index=True) if trial_table_list else pd.DataFrame()
    unit_table = pd.concat(unit_table_list, ignore_index=True) if unit_table_list else pd.DataFrame()

    if not unit_table.empty:
        unit_table = unit_table[~unit_table['ccf_acronym'].isin(allen.get_excluded_areas())]
        unit_table = unit_table.reset_index(drop=True)
        unit_table['unit_id'] = unit_table.index

    return trial_table, unit_table, ephys_nwb_list

def convert_electrode_group_object_to_columns(data):
    """
    Convert electrode group object to dictionary.
    Creates a new column in the dataframe.
    :param data: pd.DataFrame containing the NWB electrode group field.
    :return: 
    """
    elec_group_list = data['electrode_group'].values
    #elec_group_name = [e.name for e in elec_group_list]
    elec_group_name = [e.name for e in elec_group_list]
    data['electrode_group'] = elec_group_name

    elec_group_location = [e.location.replace('nan', 'None') for e in elec_group_list]
    elec_group_location_dict = [eval(e) for e in elec_group_location]
    data['location'] = elec_group_location_dict
    data['target_region'] = [e.get('area') for e in elec_group_location_dict]

    return data

def compute_fano_factor_from_spike_train(spike_times, event_times, bin_size, time_start, time_stop):
    """
    Computes Fano factor for a single unit over trials.
    :param spike_times:  Spike times in seconds.
    :param event_times:  Stimulus times in seconds.
    :param bin_size: Bin size in seconds.
    :param time_start: Start of peri-stimulus time window.
    :param time_stop: End of peri-stimulus time window.
    :return: Fano factor of spike counts.
    """

    # Initialize Fano factor
    n_bins = int((time_stop - time_start) / bin_size)
    spike_counts = np.zeros((len(event_times), n_bins))

    # Compute spike counts
    for i, stim_time in enumerate(event_times):
        spike_times_in_window = spike_times[(spike_times >= stim_time + time_start) & (spike_times < stim_time + time_stop)]
        spike_times_in_window -= stim_time # align
        spike_counts[i,:] = np.histogram(spike_times_in_window, bins=np.arange(time_start, time_stop + bin_size, bin_size), density=False)[0]

    # Compute Fano factor
    fano_factor = np.var(spike_counts, axis=0) / np.mean(spike_counts, axis=0)

    return fano_factor


def compute_unit_peri_event_histogram(spike_times, event_times, bin_size, time_start, time_stop, artifact_correction=False):
    """
    Computes peri-stimulus time histogram for a single unit.
    :param spike_times:  Spike times in seconds.
    :param event_times:  Stimulus times in seconds.
    :param bin_size: Bin size in seconds.
    :param time_start: Start of peri-stimulus time window.
    :param time_stop: End of peri-stimulus time window.
    :param artifact_correction: Boolean to apply artifact correction.
    :return: Peri-stimulus time histogram of spike counts.
    """

    # Initialize histogram
    if artifact_correction:
        bin_size_hist = 0.001
        n_bins = int((time_stop - time_start) / 0.001)
    else:
        bin_size_hist = bin_size
        n_bins = int((time_stop - time_start) / bin_size)
        
    peri_stim_hist = np.zeros((len(event_times), n_bins))
        
    # Compute histogram
    for i, stim_time in enumerate(event_times):
        spike_times_in_window = spike_times[(spike_times >= stim_time + time_start) & (spike_times < stim_time + time_stop)]
        spike_times_in_window -= stim_time # align
        spike_counts = np.histogram(spike_times_in_window, bins=np.arange(time_start, time_stop + bin_size_hist, bin_size_hist), density=False)[0]
        peri_stim_hist[i,:] = spike_counts # add counts

    if artifact_correction:
        # Bin to correct for artifact
        if bin_size_hist==0.001:
            stim_dur = 3 # stimulus duration in msec
            art_start = -1  # ms before stim
            art_stop = stim_dur + 1    # ms after stim
            art_start_bin = int(abs(time_start) / bin_size_hist) + art_start
            art_stop_bin = int(abs(time_start) / bin_size_hist) + art_stop

        # Get baseline firing rate from PETH
        bas_stop = int(abs(time_start) / bin_size_hist) - 5 # 5 time bins before stim
        trial_baselines = compute_trial_baseline_from_peth(peri_stim_hist,
                                                           bas_start=0,
                                                           bas_stop=bas_stop)

        # Make Poisson noise based on baseline firing rate
        rng = np.random.default_rng(seed=None)  # no seed for variability
        poisson_noise = [rng.poisson(lam=trial_baselines[i], size=n_bins) for i in range(len(event_times))]
        poisson_noise = np.array(poisson_noise)

        # Replace spike counts with Poisson noise in artifact window
        try:
            #print('shape', peri_stim_hist.shape, 'art_start_bin:', art_start_bin, 'art_stop_bin:', art_stop_bin)
            peri_stim_hist[:, art_start_bin:art_stop_bin] = poisson_noise[:, art_start_bin:art_stop_bin]
        except IndexError:
            print('Index error in artifact correction. Skipping correction because no trials.')
            print('shape', peri_stim_hist.shape, 'art_start_bin:', art_start_bin, 'art_stop_bin:', art_stop_bin, 'events', len(event_times))
            print('art_start_bin:', art_start_bin)
            return peri_stim_hist

        # Rebin to desired bin size if there was artifact correction
        if artifact_correction and bin_size != 0.001:
            # Aggregate spike counts in bin_size in ms time bins using the sum over the bins
            current_bin_size_ms = int(bin_size_hist * 1000)
            new_bin_size_ms = int(bin_size * 1000)
            n_trials = peri_stim_hist.shape[0]

            peri_stim_hist_original = peri_stim_hist.copy()
            peri_stim_hist = peri_stim_hist.reshape(n_trials, -1, new_bin_size_ms // current_bin_size_ms).sum(axis=2)

            debug=False
            if debug:
                fig, ax = plt.subplots(1,1)
                time = np.linspace(time_start, time_stop, peri_stim_hist.shape[1])
                ax.plot(time,np.nanmean(peri_stim_hist_original, axis=0), c='k')
                ax.plot(time,np.nanmean(peri_stim_hist, axis=0), c='r')
                ax.axvline(0, c='k', linestyle='--')
                plt.show()
            
    return peri_stim_hist

def compute_fano_factor_from_peth(peth, time_start, time_stop):
    """
    Computes Fano factor from peri-event time histogram of a single unit.
    :param peth: Peri-event time histogram.
    :return: Fano factor of spike counts.
    """
    # Get window of interest
    peth = peth[:, int(time_start):int(time_stop)]
    # Compute Fano factor
    fano_factor = np.var(peth, axis=0) / np.mean(peth, axis=0)

    return fano_factor
def compute_baseline_from_spike_train(spike_times, event_times, bas_start, bas_stop):
    """
    Computes baseline firing rate for a unitspike train.
    :param spike_times:  Spike times in seconds.
    :param event_times:  Stimulus times in seconds.
    :param bas_start: Start of baseline time window.
    :param bas_stop: End of baseline time window.
    :return: Baseline firing rate, in Hz.
    """

    # Get all spikes in baseline window
    baseline_spikes = []
    for event in event_times:
        bas_spikes = spike_times[(spike_times >= event + bas_start) & (spike_times < event + bas_stop)]
        baseline_spikes.append(bas_spikes)
    # Calculate baseline firing rate
    baseline_rate = len(baseline_spikes) / (bas_stop - bas_start)

    return baseline_rate


def compute_baseline_from_peth(peth, bas_start, bas_stop):
    """
    Computes baseline firing rate from peri-event time histogram.
    :param peth: Peri-event time histogram.
    :param bas_start: Start of baseline time window.
    :param bas_stop: End of baseline time window.
    :return: Baseline firing rate, in Hz.
    """
    baseline_rate = np.mean(peth[:,bas_start:bas_stop])
    return baseline_rate

def compute_trial_baseline_from_peth(peth, bas_start, bas_stop):
    """
    Computes baseline firing rate from peri-event time histogram, for each trials.
    :param peth: Peri-event time histogram.
    :param bas_start: Start of baseline time window.
    :param bas_stop: End of baseline time window.
    :return: Baseline firing rate, in Hz.
    """
    baseline_rate = np.mean(peth[:, bas_start:bas_stop], axis=1)
    return baseline_rate

def compute_zscored_all(peths_array):
    """
    Z-score peri-event time histograms of a population of units.
    :param peths_array:
    :return:
    """
    mean_firing_rates = np.mean(peths_array)
    std_firing_rates = np.std(peths_array)
    return (peths_array - mean_firing_rates) / std_firing_rates

def normalize_by_std(peths_array):
    """
    Divide peri-event time histograms of a population of units by the standard deviation.
    PETHs are assumed already centered around zero.
    :param peths_array:
    :return:
    """
    std_firing_rates = np.std(peths_array, axis=1)
    return peths_array / std_firing_rates[:, np.newaxis]

def compute_zscored_peths(peths_array):
    """
    Z-score peri-event time histograms of a population of units.
    :param peths_array: Array of peri-event time histograms.
    :return: Z-scored peri-event time histograms.
    """
    mean_firing_rates = np.mean(peths_array, axis=1)
    std_firing_rates = np.std(peths_array, axis=1)
    return (peths_array - mean_firing_rates[:, np.newaxis]) / std_firing_rates[:, np.newaxis]

def compute_zscored_peths_per_unit(peths_array):
    """
    Z-score peri-event time histograms of a population of units, for each unit/population.
    :param peths_array:
    :return:
    """
    mean_firing_rates = np.mean(peths_array, axis=1)
    std_firing_rates = np.std(peths_array, axis=1)
    return (peths_array - mean_firing_rates[np.newaxis,:,:]) / std_firing_rates[np.newaxis,:,:]

def apply_moving_average(data, window_size):
    """ Computes a moving average of a 1-D array, with a window size centered on each data point.
    The size adapts at the edges of the array.
    """
    smoothed_data = []
    for i in range(len(data)):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(data), i + window_size // 2 + 1)
        window = data[start_index:end_index]
        smoothed_data.append(sum(window) / len(window))
    return smoothed_data

def halfgaussian_kernel1d(sigma, radius):
    """
    Computes a 1-D Half-Gaussian convolution kernel.
    """
    sigma2 = sigma * sigma
    x = np.arange(0, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    return phi_x

def halfgaussian_filter1d(input, sigma, axis=-1, output=None,
                      mode="nearest", cval=0.0, truncate=4.0):
    """
    Convolves a 1-D Half-Gaussian convolution kernel.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = halfgaussian_kernel1d(sigma, lw)
    origin = -lw // 2
    return scipy.ndimage.convolve1d(input, weights, axis, output, mode, cval, origin)


def half_gaussian_kernel(size, sigma):
    t = np.arange(0, size)
    kernel = np.exp(-t**2 / (2 * sigma**2))
    kernel = np.exp(-t**2 / (2 * sigma**2))
    return kernel / np.sum(kernel)  # Normalize the kernel

def causal_gaussian_filter(spike_train, sigma):
    size = int(3 * sigma)  # Choose the size of the kernel
    kernel = half_gaussian_kernel(size, sigma)
    smoothed_train = np.convolve(spike_train, kernel, mode='full')[:len(spike_train)]
    return smoothed_train


def compute_peth_for_unit(args):
    """
    Compute PETH for a single unit (cluster) across all trial types and lick flags.
    This function is designed to run in parallel processes.
    :param args: A tuple containing (cluster, trial_table_mouse, params).
    :return: List of dictionaries containing PETH information for the unit.
    """
    cluster, trial_table_mouse, params = args

    bin_size = params['bin_size']
    time_start = params['time_start']
    time_stop = params['time_stop']
    baseline_correction = params['baseline_correction']
    n_max_trials = params['n_max_trials']
    artifact_correction = params['artifact_correction']

    # Ensure spike_times is a numpy array (if it was an h5py object)
    spike_times = np.array(cluster['spike_times']) 

    peth_list = []

    # Iterate over all combinations of trial type and lick flag
    for context in ['active', 'passive']:
        for trial_type in ['whisker_trial', 'auditory_trial', 'no_stim_trial']:
            #if trial_type == 'no_stim_trial' and context != 'active':
            #    continue

            for lick_flag in [1, 0]:
                event_times_dict = {}
                trial_table_sub = pd.DataFrame()
                if context == 'active':
                    # Filter trials based on trial type and lick flag
                    trial_table_sub = trial_table_mouse[
                        (trial_table_mouse['trial_type'] == trial_type)
                        & (trial_table_mouse['lick_flag'] == lick_flag)
                        & (trial_table_mouse['context'] == context)
                        ]
                elif context == 'passive':
                    # Filter trials based on trial type and lick flag
                    trial_table_sub = trial_table_mouse[
                        (trial_table_mouse['trial_type'] == trial_type)
                        & (trial_table_mouse['context'] == context)
                        ]

               # if trial_table_sub.empty:
               #     #print('Empty trial table for', context, trial_type, 'lick flag', lick_flag,
               #     #      trial_table_mouse.mouse_id.unique(),
               #     #      trial_table_mouse.session_id.unique())
               #     continue

                # Get event times and outcome
                event_times = trial_table_sub['start_time'].values
                if len(event_times) == 0:
                    #print('No event times for', context, trial_type, 'lick flag', lick_flag, trial_table_mouse.mouse_id.unique())
                    continue
                outcome = trial_table_sub['outcome'].unique()[0]

                # Optionally, limit number of trials
                if n_max_trials is not None and len(event_times) > n_max_trials:
                    event_times = event_times[:n_max_trials]

                # For passive, get pre and post sessions
                if context == 'passive':
                    # Get trial index midway in session
                    idx_mid = len(trial_table_mouse) // 2
                    idx_mid_time = trial_table_mouse['start_time'].values[idx_mid]
                    # Get pre and post passive trials
                    event_times_pre = event_times[event_times < idx_mid_time]
                    event_times_post = event_times[event_times >= idx_mid_time]
                    event_times_dict['passive_pre'] = event_times_pre
                    event_times_dict['passive_post'] = event_times_post

                elif context == 'active':
                    event_times_dict['active'] = event_times

                # Iterate over event times
                for key, event_times in event_times_dict.items():
                    # Compute PETH
                    if len(event_times) == 0:
                        continue
                    peth = compute_unit_peri_event_histogram(spike_times=spike_times,
                                                             event_times=event_times,
                                                             bin_size=bin_size,
                                                             time_start=time_start,
                                                             time_stop=time_stop,
                                                             artifact_correction=artifact_correction)

                    # Apply baseline correction if enabled
                    if baseline_correction:
                        baseline_duration = abs(time_start)
                        bas_stop = int(baseline_duration / bin_size)
                        baseline = compute_baseline_from_peth(peth, bas_start=0, bas_stop=bas_stop)
                        peth = peth - baseline

                    # Average across trials
                    peth = np.mean(peth, axis=0)

                    # Add PETH info to cluster dict
                    cluster_info = cluster.to_dict()
                    cluster_info['peth'] = np.array(peth)
                    cluster_info['reward_group'] = trial_table_mouse['reward_group'].unique()[0]
                    cluster_info['session_id'] = trial_table_mouse['session_id'].unique()[0]
                    cluster_info['trial_type'] = trial_type
                    cluster_info['lick_flag'] = lick_flag if context == 'active' else 0
                    cluster_info['outcome'] = outcome if context == 'active' else 'passive'
                    cluster_info['artifact_correction'] = artifact_correction
                    cluster_info['context'] = key

                    # Append to list
                    peth_list.append(cluster_info)

    return peth_list

def compute_peth_for_unit_block(args):
    """
    Compute PETH for a single unit (cluster) across all trial types and lick flags.
    This function is designed to run in parallel processes.
    :param args: A tuple containing (cluster, trial_table_mouse, params).
    :return: List of dictionaries containing PETH information for the unit.
    """
    cluster, trial_table_mouse, params = args

    bin_size = params['bin_size']
    time_start = params['time_start']
    time_stop = params['time_stop']
    baseline_correction = params['baseline_correction']
    n_max_trials = params['n_max_trials']
    artifact_correction = params['artifact_correction']

    # Ensure spike_times is a numpy array (if it was an h5py object)
    spike_times = np.array(cluster['spike_times'])

    peth_list = []

    # Iterate over all combinations of trial type, lick flag, block
    for trial_type in ['whisker_trial', 'auditory_trial', 'no_stim_trial']:

        trial_table_type = trial_table_mouse[trial_table_mouse['trial_type'] == trial_type]
        if params['all_blocks']:
            block_id_list = trial_table_type['block_id'].unique()
        else: # early vs. late (take only these two)
            block_id_list = trial_table_type['block_id'].unique()
            block_id_list = [block_id_list[0], block_id_list[-1]]

        for block_id in block_id_list:
            event_times_dict = {}

            # Filter trials
            trial_table_sub = trial_table_type[
                #(trial_table_type['context'] == 'active') #combine both passive anda ctive (if early learning trial)
                #&
                        (trial_table_type['block_id'] == block_id)
                ]

            if trial_table_sub.empty:
                print('Missing condition for', trial_type, block_id, cluster, trial_table_mouse.mouse_id)
                continue

            # Get event times and outcome
            event_times = trial_table_sub['start_time'].values
            outcome = trial_table_sub['outcome'].unique()[0]

            # Optionally, limit number of trials
            if n_max_trials is not None and len(event_times) > n_max_trials:
                event_times = event_times[:n_max_trials]

            event_times_dict['active'] = event_times

            # Iterate over event times
            for key, event_times in event_times_dict.items():
                # Compute PETH
                peth = compute_unit_peri_event_histogram(spike_times=spike_times,
                                                         event_times=event_times,
                                                         bin_size=bin_size,
                                                         time_start=time_start,
                                                         time_stop=time_stop,
                                                         artifact_correction=artifact_correction)

                # Apply baseline correction if enabled
                if baseline_correction:
                    baseline_duration = abs(time_start)
                    bas_stop = int(baseline_duration / bin_size)
                    baseline = compute_baseline_from_peth(peth, bas_start=0, bas_stop=bas_stop)
                    peth = peth - baseline

                # Average across trials
                peth = np.mean(peth, axis=0)

                # Add PETH info to cluster dict
                cluster_info = cluster.to_dict()
                cluster_info['peth'] = np.array(peth)
                cluster_info['reward_group'] = trial_table_mouse['reward_group'].unique()[0]
                cluster_info['session_id'] = trial_table_mouse['session_id'].unique()[0]
                cluster_info['trial_type'] = trial_type
                cluster_info['lick_flag'] = 1
                cluster_info['outcome'] = outcome
                cluster_info['artifact_correction'] = artifact_correction
                cluster_info['context'] = key
                
                if params['all_blocks']: # at inflection trial
                    cluster_info['block'] = block_id
                else: # early vs late.
                    if block_id == 0:
                         cluster_info['block'] = 'early'
                    else:
                         cluster_info['block'] = 'late'

            # Append to list
            peth_list.append(cluster_info)

    return peth_list

def build_peth_table_parallel(trial_table, unit_table, params, proc_data_path, file_name):
    """
    Build peri-event time histogram dataframe from trial and unit tables using multiprocessing for speed.
    :param trial_table: Trial table.
    :param unit_table: Unit table.
    :param params: Dictionary of peri-event time histogram parameters.
    :param proc_data_path: Path to save processed data table.
    :param file_name: Name of the output file (without extension).
    :return: Peri-event time histogram table.
    """

    print('Building session-wide PETH table with multiprocessing...')
    # Map performance to outcome in trial_table
    trial_table['outcome'] = trial_table['perf'].astype('int32').map(TRIAL_MAP)

    # Group trial table by mouse_id and pass it along with the cluster for each unit
    unit_args = [(cluster, trial_table[trial_table['mouse_id'] == cluster['mouse_id']], params)
                 for idx, cluster in unit_table.iterrows()]

    # Ensure that all spike_times are numpy arrays (required by multiprocessing)
    for cluster in unit_args:
        cluster[0]['spike_times'] = np.array(cluster[0]['spike_times'])

    # Set up multiprocessing pool
    with Pool(os.cpu_count()-2) as pool:
        results = pool.map(compute_peth_for_unit, unit_args)

    # Flatten list of lists into a single list
    peth_list = [item for sublist in results for item in sublist]

    # Convert list to DataFrame
    peth_df = pd.DataFrame(peth_list)

    # Add metadata columns
    peth_df['bin_size'] = params['bin_size']
    peth_df['time_start'] = params['time_start']
    peth_df['time_stop'] = params['time_stop']
    peth_df['baseline_correction'] = params['baseline_correction']

    # Save the PETH table to HDF5 format
    if proc_data_path is not None:
        if not os.path.exists(proc_data_path):
            os.makedirs(proc_data_path)

        try:
            output_file = os.path.join(proc_data_path, file_name + '.h5')
            peth_df.to_hdf(output_file, key='df', mode='w')
            print(f'Saved multi-mouse PETH dataframe to {output_file}')


        except Exception as e:
            print('Could not save as HDF5:', e)
            output_file = os.path.join(proc_data_path, file_name + '.pkl')
            peth_df.to_pickle(output_file)
            print(f'Saved multi-mouse PETH dataframe to {output_file}')

        with open(os.path.join(proc_data_path, file_name+'_params.json'), 'w') as f:
            json.dump(params, f)

    return peth_df


def build_peth_table_parallel_block(trial_table, unit_table, params, proc_data_path):
    """
    Build peri-event time histogram dataframe from trial and unit tables using multiprocessing for speed,
    for different trial blocks.
    :param trial_table: Trial table.
    :param unit_table: Unit table.
    :param params: Dictionary of peri-event time histogram parameters.
    :param proc_data_path: Path to save processed data table.
    :return: Peri-event time histogram table.
    """

    print('Building block-wise PETH table with multiprocessing...')

    # Remove passive data, early licks, association trials
    trial_table = trial_table[trial_table['context'] != 'passive']
    trial_table = trial_table[trial_table['perf'] != 6]

    # Map performance to outcome in trial_table
    trial_table['outcome'] = trial_table['perf'].astype(int).map(TRIAL_MAP)
    trial_table = trial_table[trial_table['lick_flag']==1]
    trial_table['trial_id_type'] = trial_table.groupby(['mouse_id', 'context', 'trial_type'])['trial_id'].cumcount()

    # Create trial blocks
    params['all_blocks'] = False
    trial_table['block_id'] = trial_table['trial_id_type'] // params['block_size']

    # Convert info from electrode group object into columns (group name, probe location, target area)
    unit_table = convert_electrode_group_object_to_columns(unit_table)

    # Group trial table by mouse_id and pass it along with the cluster for each unit
    unit_args = [(cluster, trial_table[trial_table['mouse_id'] == cluster['mouse_id']], params)
                 for idx, cluster in unit_table.iterrows()]

    # Ensure that all spike_times are numpy arrays (required by multiprocessing)
    for cluster in unit_args:
        cluster[0]['spike_times'] = np.array(cluster[0]['spike_times'])

    # Set up multiprocessing pool
    with Pool(os.cpu_count() - 5) as pool:
        # Distribute computation of PETHs across processes
        results = pool.map(compute_peth_for_unit_block, unit_args)

    # Flatten list of lists into a single list
    peth_list = [item for sublist in results for item in sublist]

    # Convert list to DataFrame
    peth_df = pd.DataFrame(peth_list)


    # Add metadata columns
    peth_df['bin_size'] = params['bin_size']
    peth_df['time_start'] = params['time_start']
    peth_df['time_stop'] = params['time_stop']
    peth_df['baseline_correction'] = params['baseline_correction']

    # Save the PETH table to feather format
    if proc_data_path is not None:
        file_name = 'peth_table_block'
        if not os.path.exists(proc_data_path):
            os.makedirs(proc_data_path)

        try:
            output_file = os.path.join(proc_data_path, file_name + '.h5')
            peth_df.to_hdf(output_file, key='df', mode='w')
            print(f'Saved multi-mouse PETH dataframe to {output_file}')

        except Exception as e:
            print('Could not save as HDF5:', e)
            output_file = os.path.join(proc_data_path, file_name + '.pkl')
            peth_df.to_pickle(output_file)
            print(f'Saved multi-mouse PETH dataframe to {output_file}')

        with open(os.path.join(proc_data_path, file_name+'_params.json'), 'w') as f:
            json.dump(params, f)

    return peth_df


def build_peth_table_parallel_inflection(trial_table, unit_table, params, proc_data_path):
    """
    Build peri-event time histogram dataframe from trial and unit tables using multiprocessing for speed,
    before and after inflection points.
    :param trial_table: Trial table.
    :param unit_table: Unit table.
    :param params: Dictionary of peri-event time histogram parameters.
    :param proc_data_path: Path to save processed data table.
    :return: Peri-event time histogram table.
    """

    print('Building inflection PETH table with multiprocessing...')

    # Select for passive, optionally
    if not params['include_passive']:
        trial_table = trial_table[trial_table['context'].isin(['active'])]
        trial_table = trial_table[trial_table['perf'] != 6]

    # Map performance to outcome in trial_table
    trial_table['outcome'] = trial_table['perf'].astype(int).map(TRIAL_MAP)
    trial_table['trial_id_type'] = trial_table.groupby(['mouse_id', 'trial_type'])['trial_id'].cumcount()
    trial_table['trial_id_type_context'] = trial_table.groupby(['mouse_id', 'trial_type', 'context'])['trial_id'].cumcount()

    # Step 1: Get first_active_id per mouse_id
    first_active = trial_table[trial_table["context"] == "active"].groupby("mouse_id")["trial_id"].min().rename("first_active_id")
    trial_table = trial_table.merge(first_active, on="mouse_id", how="left")

    # Step 2: Get last_passive_id (trial_id just before first_active_id)
    def find_last_passive_id(sub_df):
        threshold = sub_df["first_active_id"].iloc[0]
        return sub_df[sub_df["trial_id"] < threshold]["trial_id"].max()

    last_passive = trial_table.groupby("mouse_id").apply(find_last_passive_id).rename("last_passive_id")
    trial_table = trial_table.merge(last_passive, on="mouse_id", how="left")

    # Step 3: convert learning trial index (whisker-based) to session_trial_id
    whisker_df = trial_table[(trial_table["trial_type"] == "whisker_trial") & (trial_table["context"] == "active")] # get trials where learning trial was computed with
    matched_trials = whisker_df[whisker_df["learning_trial"] == whisker_df["trial_id_type_context"]]# find match trial index matching learning trial (whisker only)
    learning_trial_id = matched_trials.groupby("mouse_id")["trial_id"].first().rename("learning_trial_id") #get session-wide trial id
    trial_table = trial_table.merge(learning_trial_id, on="mouse_id", how="left")

    # Check if trial is before inflection or after inflection row-wise
    trial_table['pre_inflection'] = trial_table.apply(lambda row: row['trial_id'] < row['learning_trial_id'], axis=1)
    trial_table['post_inflection'] = trial_table.apply(lambda row: row['trial_id'] >= row['learning_trial_id'], axis=1)

    # Step 4: Compute trial_id column relative to learning_trial_id, for each trial_type
    def compute_relative_trial_index(group, learning_trial_id):
        """Compute trial index relative to the learning trial (always a whisker trial)."""
        group = group.sort_values('trial_id')  # Ensure trials are sorted in order

        # Assign negative indices before, 0 at learning trial, and positive indices after
        group['trial_id_from_inflection'] = (
            group['trial_id'].apply(lambda x:
                                    -sum(group['trial_id'] < learning_trial_id) +
                                    sum(group['trial_id'] <= x) - 1
                                    )
        )

        return group

    # Apply function separately for each (mouse_id, trial_type)
    def process_mouse_group(mouse_group):
        try:
            learning_trial_id = mouse_group.loc[mouse_group['trial_id'] == mouse_group['learning_trial_id'].iloc[0], 'trial_id'].values[0]
        except IndexError as err:
            print('Error finding learning trial id for mouse:', mouse_group['mouse_id'].iloc[0], err)
            learning_trial_id = None
        return mouse_group.groupby("trial_type", group_keys=False).apply(compute_relative_trial_index,
                                                                         learning_trial_id)

    trial_table = trial_table.groupby("mouse_id", group_keys=False).apply(process_mouse_group)

    # Step 5: define custom trial ranges to compute PETHs over
    block_mapping = {
        (-10, -6): 'pre-learning',
        (-5, -1): 'during',  
        (0, 4): 'post-learning',
    }

    # Function to assign block name based on trial_id_from_inflection
    def assign_block(trial_id_from_inflection, min_trial, max_trial):
        # Adjust the ranges to fit within the available trials, if necessary
        for (start, end), block_name in block_mapping.items():
            # Ensure that ranges are within the possible trial range
            adjusted_start = max(start, min_trial)
            adjusted_end = min(end, max_trial)

            if adjusted_start <= trial_id_from_inflection <= adjusted_end:
                return block_name

        # If no block is found, assign 'Uncategorized' for trial_id_from_inflection outside defined ranges
        return 'Uncategorized'

    # Apply function to create block_id column
    trial_table['block_id'] = trial_table.groupby(['mouse_id', 'trial_type'])['trial_id_from_inflection'] \
        .transform(lambda x: x.apply(lambda y: assign_block(y, x.min(), x.max())))
    
    # Keep only specific blocks for PETH calculation
    params['all_blocks'] = True
    trial_table = trial_table[trial_table['block_id'].isin(list(block_mapping.values()))]

    # -------------
    # Conpute PETHs
    # -------------

    # Group trial table by mouse_id and pass it along with the cluster for each unit
    unit_args = [(cluster, trial_table[trial_table['mouse_id'] == cluster['mouse_id']], params)
                 for idx, cluster in unit_table.iterrows()]

    # Ensure that all spike_times are numpy arrays (required by multiprocessing)
    for cluster in unit_args:
        cluster[0]['spike_times'] = np.array(cluster[0]['spike_times'])

    # Set up multiprocessing pool
    with Pool(os.cpu_count() - 5) as pool:
        # Distribute computation of PETHs across processes
        results = pool.map(compute_peth_for_unit_block, unit_args)

    # Flatten list of lists into a single list
    peth_list = [item for sublist in results for item in sublist]

    # Convert list to DataFrame
    peth_df = pd.DataFrame(peth_list)

    # Add metadata columns
    peth_df['bin_size'] = params['bin_size']
    peth_df['time_start'] = params['time_start']
    peth_df['time_stop'] = params['time_stop']
    peth_df['baseline_correction'] = params['baseline_correction']

    # Save the PETH table to feather format
    if proc_data_path is not None:
        file_name = 'peth_table_inflection'
        if not os.path.exists(proc_data_path):
            os.makedirs(proc_data_path)

        try:
            output_file = os.path.join(proc_data_path, file_name + '.h5')
            peth_df.to_hdf(output_file, key='df', mode='w')
            print(f'Saved multi-mouse PETH dataframe to {output_file}')


        except Exception as e:
            print('Could not save as HDF5:', e)
            output_file = os.path.join(proc_data_path, file_name + '.pkl')
            peth_df.to_pickle(output_file)
            print(f'Saved multi-mouse PETH dataframe to {output_file}')

        with open(os.path.join(proc_data_path, file_name+'_params.json'), 'w') as f:
            json.dump(params, f)

    return peth_df


def build_session_dynamics_table(trial_table, unit_table, params, proc_data_path): #TODO: make it parallel
    """
    Build session trial-by-trial dynamics dataframe from trial and unit tables.
    :param trial_table:
    :param unit_table:
    :param params:
    :param proc_data_path:
    :return:
    """
    print('Building session dynamics PETH table...')

    trial_table = pd.DataFrame(trial_table)
    unit_table = pd.DataFrame(unit_table)

    # Remove passive data, early licks, association trials
    trial_table = trial_table[trial_table['context'] != 'passive']
    trial_table = trial_table[trial_table['perf'] != 6]

    # Get spike train data parameters
    bin_size = params['bin_size']
    time_start = params['time_start']
    time_stop = params['time_stop']
    baseline_correction = params['baseline_correction']

    # Initialize table
    sess_dyn_df = []

    # Map perf to outcome
    trial_table['outcome'] = trial_table['perf'].astype(int).map(TRIAL_MAP)

    for trial_type in ['whisker_trial', 'auditory_trial', 'no_stim_trial']:

        # Get trial subsets, skip if no such trial type
        trial_table_sub = trial_table[(trial_table['trial_type'] == trial_type)]
        if len(trial_table_sub.index) == 0:
            continue

        event_times = trial_table_sub['start_time'].values
        n_trials = len(event_times)

        if trial_type == 'whisker_trial':
            artifact_correction = params['artifact_correction']
        else:
            artifact_correction = False

        # Iterate over clusters to get spike times aligned
        for idx, cluster in unit_table.iterrows():
            spike_times = cluster['spike_times']

            peth = compute_unit_peri_event_histogram(spike_times=spike_times,
                                                     event_times=event_times,
                                                     bin_size=bin_size,
                                                     time_start=time_start,
                                                     time_stop=time_stop,
                                                     artifact_correction=artifact_correction)

            if baseline_correction:
                # Get baseline indices before alignment index
                bas_start = 0
                baseline_duration = abs(time_start)  # seconds
                bas_stop = int(baseline_duration / bin_size)  # this works if symmetrical window
                peth = peth - compute_baseline_from_peth(peth, bas_start=bas_start, bas_stop=bas_stop)

            cluster_dict = cluster.to_dict()
            cluster_info = {k: [v] * n_trials for k, v in cluster_dict.items()}
            cluster_info['spike_counts'] = list(peth) # splits along trials
            cluster_info['bin_size'] = [bin_size] * n_trials
            cluster_info['time_start'] = [time_start] * n_trials
            cluster_info['time_stop'] = [time_stop] * n_trials
            cluster_info['artifact_correction'] = [artifact_correction] * n_trials
            cluster_info['baseline_correction'] = [baseline_correction] * n_trials
            cluster_info['trial_type'] = [trial_type] * n_trials
            cluster_info['lick_flag'] = trial_table_sub['lick_flag'].values
            cluster_info['reward_group'] = trial_table['reward_group'].unique()[0]
            cluster_info['session_id'] = trial_table['session_id'].unique()[0]
            cluster_info['outcome'] = trial_table_sub['outcome'].values
            cluster_info['trial_index'] = trial_table_sub.index
            cluster_info['trial_type_index'] = np.arange(n_trials)
            cluster_info_df = pd.DataFrame.from_dict(cluster_info)
            sess_dyn_df.append(cluster_info_df)
            
        
            #for trial_type_index, event_time in enumerate(event_times):
#
#
            #    spike_times_in_window = spike_times[(spike_times >= event_time + time_start) & (spike_times < event_time + time_stop)]
            #    spike_times_in_window -= event_time
            #    spike_counts = np.histogram(spike_times_in_window, bins=np.arange(time_start, time_stop + bin_size, bin_size), density=False)[0]
#
            #    # Fill cluster info
            #    cluster_info = cluster.to_dict()
            #    cluster_info['spike_counts'] = spike_counts
            #    cluster_info['bin_size'] = bin_size
            #    cluster_info['time_start'] = time_start
            #    cluster_info['time_stop'] = time_stop
            #    cluster_info['artifact_correction'] = artifact_correction
            #    cluster_info['baseline_correction'] = baseline_correction
            #    cluster_info['trial_type'] = trial_type
            #    cluster_info['lick_flag'] = trial_table_sub['lick_flag'].values[trial_type_index]
            #    cluster_info['reward_group'] = trial_table['reward_group'].unique()[0]
            #    cluster_info['session_id'] = trial_table['session_id'].unique()[0]
            #    cluster_info['outcome'] = trial_table_sub['outcome'].values[trial_type_index]
            #    cluster_info['trial_type_index'] = trial_type_index
#
            #    # Append to list
            #    sess_dyn_df.append(cluster_info)

    # Concatenate all trial-type spikes
    sess_dyn_df = pd.concat(sess_dyn_df)

    # Save PETH table as processed data file
    if proc_data_path is not None:
        file_name = 'peth_table_trial_dynamics'
        if not os.path.exists(proc_data_path):
            os.makedirs(proc_data_path)

        try:
            output_file = os.path.join(proc_data_path, file_name + '.h5')
            sess_dyn_df.to_hdf(output_file, key='df', mode='w')
            print(f'Saved multi-mouse PETH dataframe to {output_file}')


        except Exception as e:
            print('Could not save as HDF5:', e)
            output_file = os.path.join(proc_data_path, file_name + '.pkl')
            sess_dyn_df.to_pickle(output_file)
            print(f'Saved multi-mouse PETH dataframe to {output_file}')

        with open(os.path.join(proc_data_path, file_name+'_params.json'), 'w') as f:
            json.dump(params, f)

    return sess_dyn_df
