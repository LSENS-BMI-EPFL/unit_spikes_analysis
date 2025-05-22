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
import allen_utils as allen_utils

from raster_utils import plot_rasters
from roc_utils import roc_analysis
from waveform_utils import assign_rsu_vs_fsu
from unit_label_utils import unit_label_describe
from glm_utils import run_unit_glm_pipeline_with_pool

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
    mouse_info_df = mouse_info_df[mouse_info_df['exclude_ephys'] == 0] # excluded mice
    mouse_info_df = mouse_info_df[mouse_info_df['recording'] == 1]
    included_mice = mouse_info_df['mouse_id'].unique()

    # For each reward group, show the number of mice
    reward_groups = mouse_info_df['reward_group'].unique()
    for reward_group in reward_groups:
        group_subjects = mouse_info_df[mouse_info_df['reward_group'] == reward_group]['mouse_id'].unique()
        print(f"Reward group {reward_group} has {len(mouse_info_df[mouse_info_df['reward_group'] == reward_group])} mice: {group_subjects}.")

    # Select mice to do based on available NWB files
    subject_ids = [mouse for mouse in included_mice if any(mouse in name for name in all_mwb_mice)]
    subject_ids = [s for s in subject_ids if int(s[2:]) in [50,51,52,54,56,58,59,68,72,73,74,75,76,77,78,79,80,81,82,83,85,86,87,92,93,94,95,96,97,100,101,102,103,104,105,106,107]]
    subject_ids = [s for s in subject_ids if int(s[2:]) in [94,95,96,97,100,101,102,103,104,105,106,107]]
#    subject_ids.extend(['AB{}'.format(str(i).zfill(3)) for i in range(80,151)])
    subject_ids = ['AB{}'.format(str(i).zfill(3)) for i in range(80,151)]

    subject_ids = [m for m in subject_ids if m in included_mice]
    subject_ids = [m for m in subject_ids if m not in ['AB104', 'AB107']]

    #subject_ids = ['AB137', 'AB139', 'AB140','AB143']

    ### --------------------
    # Define analyses to do
    ### -------------------

    # Single-mouse analyses
    analyses_to_do_single = ['unit_raster', 'roc_analysis', 'xcorr_analysis']
    analyses_to_do_single = ['unit_glm']

    # Multi-mouse analyses
    analyses_to_do_multi = ['rsu_vs_fsu']
    analyses_to_do_multi = ['unit_labels_processing']

    # Init. list of NWB files with neural data for analyses requiring multiple mice
    nwb_neural_files = []


    ### -------------------------------
    # Analyses suitable for single mice
    ### -------------------------------
    for subject_id in subject_ids:
        print("\n")
        print(f"Subject ID : {subject_id}")

        # Create results folder for the subject
        mouse_results_path = os.path.join(output_path, subject_id)
        os.makedirs(mouse_results_path, exist_ok=True)

        # Get NWB files for the subject
        nwb_names = [name for name in all_nwb_names if subject_id in name]
        subject_nwb_files = [os.path.join(root_path, name) for name in nwb_names]

        if not subject_nwb_files:
            print(f"No NWB files found for {subject_id}")
            continue

        subject_nwb_neural_files = []

        # Keep whisker day 0 files with neural data
        if experimenter == 'Axel_Bisi':
            for nwb_file in subject_nwb_files: # keep whisker day 0 only
                beh, day = nwb_reader.get_bhv_type_and_training_day_index(nwb_file)
                if beh=='whisker' and day==0:
                    unit_table = nwb_reader.get_unit_table(nwb_file)
                    unit_table = allen_utils.create_area_custom_column(unit_table)
                    unit_table = unit_table[~unit_table['ccf_acronym'].isin(allen_utils.get_excluded_areas())]

                    if unit_table is not None:

                        # Collect for aggregated mice analyses
                        nwb_neural_files.append(nwb_file)

                        # Add file and output path for single mouse analyses
                        mouse_output_path = os.path.join(mouse_results_path, f'{beh}_{day}')
                        subject_nwb_neural_files.append((nwb_file, mouse_output_path))
                        break

        # ----------------------------------------
        # Perform analyses for each mouse NWB file
        # ----------------------------------------

        for nwb_file, mouse_output_path in subject_nwb_neural_files:
            for analysis_type in analyses_to_do_single:

                # Define and create results path
                results_path = os.path.join(mouse_output_path, analysis_type)
                os.makedirs(results_path, exist_ok=True)


            if 'unit_raster' in analyses_to_do_single:
                plot_rasters(nwb_file, results_path)

            if 'roc_analysis' in analyses_to_do_single:
                roc_analysis(nwb_file, results_path)

            if 'xcorr_analysis' in analyses_to_do_single:
                #xcorr_analysis(nwb_file, results_path) # on cluster, otherwise adapt xcorr_analysis_mpi for multiprocessing
                pass

            if 'unit_glm' in analyses_to_do_single:
                run_glm_pipeline_with_pool(nwb_file, results_path)

    ### ------------------------------------------
    # Analyses aggregating data from multiple mice
    ### -------------------------------------------

    if 'unit_labels_processing' in analyses_to_do_multi:
        unit_label_describe(nwb_neural_files, output_path)

    if 'rsu_vs_fsu' in analyses_to_do_multi:
        assign_rsu_vs_fsu(nwb_neural_files, output_path)

