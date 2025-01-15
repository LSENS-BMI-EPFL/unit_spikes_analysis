#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_spikes_analysis
@file: unit_spikes_analysis.py
@time: 10/23/2024 11:300 PM
"""


# Imports
import os
import pandas as pd
import NWB_reader_functions as nwb_reader

from raster_utils import plot_rasters
from roc_utils import roc_analysis
from waveform_utils import assign_rsu_vs_fsu
from xcorr_utils import xcorr_analysis

if __name__ == '__main__':

    single_mouse = False
    multiple_mice = True

    # Set paths
    experimenter = 'Axel_Bisi'

    info_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'mice_info')
    output_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'results')
    root_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'NWBFull')
    proc_data_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'data', 'processed_data')
    all_nwb_names = os.listdir(root_path)
    all_mwb_mice = [name.split('_')[0] for name in all_nwb_names]

    # Load recorded mouse table
    mouse_info_df = pd.read_excel(os.path.join(info_path, 'mouse_reference_weight.xlsx'))
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df = mouse_info_df[mouse_info_df['exclude'] == 0] # excluded mice
    mouse_info_df = mouse_info_df[mouse_info_df['recording'] == 1]
    subject_ids = mouse_info_df['mouse_id'].unique()

    # For each reward group, show the number of mice
    reward_groups = mouse_info_df['reward_group'].unique()
    for reward_group in reward_groups:
        group_subjects = mouse_info_df[mouse_info_df['reward_group'] == reward_group]['mouse_id'].unique()
        print(f"Reward group {reward_group} has {len(mouse_info_df[mouse_info_df['reward_group'] == reward_group])} mice: {group_subjects}.")

    # Select mice to do based on available NWB files
    subject_ids = [mouse for mouse in subject_ids if any(mouse in name for name in all_mwb_mice)]
    subject_ids = [s for s in subject_ids if int(s[2:]) in [50,51,52,54,56,58,59,68,72,73,74,75,76,77,78,79,80,81,82,83,85,86,87,92,93,94,95,96,97,100,101,102,103,104,105,106,107]]
    subject_ids.extend(['AB{}'.format(i) for i in range(82,132)])
    subject_ids.extend(['AB{}'.format(i) for i in range(116,151)])
    subject_ids = ['AB082']

    analyses_to_do = ['unit_raster', 'roc_analysis', 'xcorr_analysis', 'rsu_vs_fsu']
    analyses_to_do = ['roc_analysis']
    analyses_to_do = ['xcorr_analysis']

    # Init. list of NWB files with neural data for analyses requiring multiple mice
    nwb_neural_files = []


    ### -------------------------------
    # Analyses suitable for single mice
    ### -------------------------------
    for subject_id in subject_ids:
        print(" ")
        print(f"Subject ID : {subject_id}")

        # Create mouse results folder if it doesn't exist
        mouse_results_path = os.path.join(output_path, subject_id)
        if not os.path.exists(mouse_results_path):
            os.makedirs(mouse_results_path)

        # Take subset of NWBs with neural data
        nwb_names = [name for name in all_nwb_names if subject_id in name]
        subject_nwb_files = [os.path.join(root_path, name) for name in nwb_names]
        if not subject_nwb_files:
            print(f"No NWB files found for {subject_id}")
            continue

        subject_nwb_neural_files = []

        if experimenter == 'Axel_Bisi':
            for nwb_file in subject_nwb_files: # keep whisker day 0 only
                beh, day = nwb_reader.get_bhv_type_and_training_day_index(nwb_file)
                if beh=='whisker' and day==0:

                    # Collect for aggregated mice analyses
                    nwb_neural_files.append(nwb_file)

                    # Collect for single mouse analyses
                    output_path = os.path.join(mouse_results_path, f'{beh}_{day}')
                    subject_nwb_neural_files.append((nwb_file, output_path))

                    break

        # ----------------------------------------
        # Perform analyses for each mouse NWB file
        # ----------------------------------------

        for nwb_file, output_path in subject_nwb_neural_files:

            if 'unit_raster' in analyses_to_do:
                results_path = os.path.join(output_path, 'unit_raster')
                if not os.path.exists(results_path):
                    os.makedirs(results_path)

                plot_rasters(nwb_file, results_path)

            if 'roc_analysis' in analyses_to_do:
                results_path = os.path.join(output_path, 'roc_analysis')
                if not os.path.exists(results_path):
                    os.makedirs(results_path)

                roc_analysis(nwb_file, results_path)

            if 'xcorr_analysis' in analyses_to_do:
                results_path = os.path.join(output_path, 'xcorr_analysis')
                if not os.path.exists(results_path):
                    os.makedirs(results_path)

                xcorr_analysis(nwb_file, results_path)
                continue

    ### ------------------------------------------
    # Analyses aggregating data from multiple mice
    ### -------------------------------------------
    if 'rsu_vs_fsu' in analyses_to_do:

        assign_rsu_vs_fsu(nwb_neural_files, output_path)

