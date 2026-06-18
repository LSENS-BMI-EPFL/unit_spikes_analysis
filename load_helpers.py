#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: brain_wide_analysis
@file: load_helpers.py
@time: 7/10/2025 12:58 PM
"""

# Imports
import os
import sys
import socket
import numpy as np
import pandas as pd
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


hostname = socket.gethostname()
print('Current host:', hostname)
if 'haas' in hostname:
    N_WORKERS = 120
    ROOT_PATH_AXEL = pathlib.Path('/mnt/lsens-analysis/Axel_Bisi/combined_results')
    ROOT_PATH_MYRIAM = pathlib.Path('/mnt/lsens-analysis/Myriam_Hamon/combined_results')

    sys.path.append("/home/bisi/code/NWB_reader")
    import NWB_reader_functions as nwb_reader

else:
    ROOT_PATH_AXEL = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Axel_Bisi\combined_results')
    ROOT_PATH_MYRIAM = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Axel_Bisi\combined_results')

import NWB_reader_functions as nwb_reader

def load_learning_curves_data(path_to_data, subject_ids):
    print('Loading learning curve data...')
    data = []
    for m in subject_ids:
        try:
            file_name = f'{m}_whisker_0_whisker_trial_learning_curve_interp.h5'
            path_to_file = os.path.join(path_to_data, m, 'whisker_0', 'learning_curve', file_name)
            df_w = pd.read_hdf(path_to_file)
            data.append(df_w)
        except FileNotFoundError as err:
            print('No whisker curve for:', m)

    data_df = pd.concat(data).reset_index(drop=True)
    return data_df


def load_jaw_onset_data(nwb_files, experimenter='AB', max_workers=12):
    """
    Load jaw onset data from NWB files in parallel.
    :param nwb_files: List of NWB file paths.
    :param experimenter: Experimenter identifier ('AB' or 'MH').
    :param max_workers: Number of threads for parallel loading.
    :return: Combined DataFrame of jaw onset times.
    """
    print('Loading jaw onset data...')

    def load_single(nwb_file):
        """Load jaw onset file for a single NWB file."""
        try:
            mouse_id = nwb_reader.get_mouse_id(nwb_file)
            if mouse_id.startswith('AB'):
                data_path = ROOT_PATH_AXEL
            elif mouse_id.startswith('MH'):
                data_path = ROOT_PATH_MYRIAM
            else:
                return None, f"[WARN] Unknown experimenter: {experimenter} for {mouse_id}"

            file_path = os.path.join(data_path, mouse_id, 'dlc_jaw_onset_times.pkl')
            if not os.path.exists(file_path):
                return None, f"[WARN] Jaw onset data not found for {mouse_id} ({file_path})"

            df = pd.read_pickle(file_path)
            df['mouse_id'] = mouse_id
            return df, None

        except Exception as e:
            return None, f"[ERROR] Failed to load {nwb_file}: {e}"

    # Parallel load with progress bar
    data_list, messages = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single, nwb_file): nwb_file for nwb_file in nwb_files}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Loading jaw onset files", ncols=100):
            df, msg = future.result()
            if msg:
                messages.append(msg)
            if df is not None and not df.empty:
                data_list.append(df)

    # Print warnings/errors after completion
    if messages:
        print("\n".join(messages))

    if not data_list:
        raise FileNotFoundError("No jaw onset .pkl files were loaded successfully.")

    # Combine into single DataFrame
    jaw_onset_table = pd.concat(data_list, ignore_index=True)
    return jaw_onset_table


def load_task_modulation_data(nwb_files, experimenter, max_workers=20):
    """
    Load ROC analysis data from NWB files in parallel.
    :param nwb_files: List of NWB file paths.
    :param experimenter: Experimenter identifier ('AB' or 'MH') for path selection.
    :param max_workers:
    :return:
    """
    print('Loading task modulation data...')

    def load_single(nwb_file):
        """Load one task mod CSV for a given NWB file."""
        try:
            mouse_id = nwb_reader.get_mouse_id(nwb_file)
            session_id = nwb_reader.get_session_id(nwb_file)

            if experimenter == 'AB':
                data_path = ROOT_PATH_AXEL
            elif experimenter == 'MH':
                data_path = ROOT_PATH_MYRIAM
            else:
                return None, f"[WARN] Unknown experimenter: {experimenter} for {mouse_id}"

            file_path = os.path.join(data_path, mouse_id, 'whisker_0', 'task_modulation',
                                     f'{mouse_id}_task_modulation_results.csv')

            if not os.path.exists(file_path):
                return None, f"[WARN] Missing task modulation: {mouse_id} ({file_path})"
            df = pd.read_csv(file_path)
            df['mouse_id'] = mouse_id
            df['session_id'] = session_id

            return df, None
        except Exception as e:
            return None, f"[ERROR] Failed for {nwb_file}: {e}"

    # Parallel load with progress bar
    data_list, messages = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single, nwb_file): nwb_file for nwb_file in nwb_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading task-mod. files", ncols=100):
            df, msg = future.result()
            if msg:
                messages.append(msg)
            if df is not None and not df.empty:
                data_list.append(df)

    # Print warnings/errors after the progress bar completes
    if messages:
        print("\n".join(messages))

    if not data_list:
        raise FileNotFoundError("No task modulation analysis CSV files were loaded successfully.")

    # Concatenate all data
    data_table = pd.concat(data_list, ignore_index=True)

    # Normalize neuron_id column
    #if 'neuron_id' not in data_table.columns and 'unit_id' in data_table.columns:
    #    data_table['neuron_id'] = data_table['unit_id']

    data_table = data_table[['mouse_id', 'neuron_id', 'task_modulated']]

    return data_table

def load_roc_analysis_data(nwb_files, experimenter, max_workers=20):
    """
    Load ROC analysis data from NWB files in parallel.
    :param nwb_files: List of NWB file paths.
    :param experimenter: Experimenter identifier ('AB' or 'MH') for path selection.
    :param max_workers:
    :return:
    """
    print('Loading ROC analysis data...')

    def load_single(nwb_file):
        """Load one ROC CSV for a given NWB file."""
        try:
            mouse_id = nwb_reader.get_mouse_id(nwb_file)
            session_id = nwb_reader.get_session_id(nwb_file)

            if experimenter == 'AB':
                data_path = ROOT_PATH_AXEL
            elif experimenter == 'MH':
                data_path = ROOT_PATH_MYRIAM
            else:
                return None, f"[WARN] Unknown experimenter: {experimenter} for {mouse_id}"

            file_path = os.path.join(data_path, mouse_id, 'whisker_0', 'roc_analysis',
                                     f'{mouse_id}_roc_results_new.csv')

            if not os.path.exists(file_path):
                return None, f"[WARN] Missing ROC: {mouse_id} ({file_path})"
            df = pd.read_csv(file_path)
            df['mouse_id'] = mouse_id
            df['session_id'] = session_id

            return df, None
        except Exception as e:
            return None, f"[ERROR] Failed for {nwb_file}: {e}"

    # Parallel load with progress bar
    data_list, messages = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single, nwb_file): nwb_file for nwb_file in nwb_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading ROC files", ncols=100):
            df, msg = future.result()
            if msg:
                messages.append(msg)
            if df is not None and not df.empty:
                data_list.append(df)

    # Print warnings/errors after the progress bar completes
    if messages:
        print("\n".join(messages))

    if not data_list:
        raise FileNotFoundError("No ROC analysis CSV files were loaded successfully.")

    # Concatenate all data
    data_table = pd.concat(data_list, ignore_index=True)

    # Normalize neuron_id column
    if 'neuron_id' not in data_table.columns and 'unit_id' in data_table.columns:
        data_table['neuron_id'] = data_table['unit_id']

    print(f"ROC types available: {data_table['analysis_type'].unique().tolist()}")
    data_table['si_sign'] = np.sign(data_table['selectivity'])
    data_table = data_table[['mouse_id',  'analysis_type', 'auc', 'selectivity', 'si_sign', 'fpr', 'tpr', 'thresholds', 'significant', 'direction', 'p_value', 'p_value_to_show', 'neuron_id']]

    return data_table


def load_wf_analysis_data(nwb_files, experimenter): #TODO: make sure merge is possible
    """
    Load ROC analysis data from NWB files.
    :param nwb_files: List of NWB file paths.
    :return:
    """
    print('Loading waveform classification analysis data...')

    data_list = []
    for nwb_file in nwb_files:
        mouse_id = nwb_reader.get_mouse_id(nwb_file)
        if experimenter=='AB':
            data_path = ROOT_PATH_AXEL
        elif experimenter=='MH':
            data_path = ROOT_PATH_MYRIAM

        # Check if file exists
        file_path = os.path.join(data_path, mouse_id, 'whisker_0', 'waveform_analysis',
                                 f'{mouse_id}_waveform_type.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            data_list.append(df)
        else:
            print(f"[WARN] Waveform type analysis data file not found for {mouse_id} at {file_path}. Skipping.")
            continue
    data_table = pd.concat(data_list, ignore_index=True)

    # Pivot table from long to wide to be able to merge info to each unit
    data_table_wide = data_table.pivot_table(index=['mouse_id', 'session_id', 'unit_id'],  columns='classif_type')
    data_table_wide.columns = ['{}_{}'.format(stat, analysis) for stat, analysis in data_table_wide.columns]
    return data_table_wide