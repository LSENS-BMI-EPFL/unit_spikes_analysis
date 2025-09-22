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


ROOT_PATH_AXEL = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Axel_Bisi', 'NWBFull')
ROOT_PATH_MYRIAM = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Myriam_Hamon',
                                'NWBFull')



if __name__ == '__main__':

    single_mouse = True
    multiple_mice = False
    joint_analysis = True
    expert_day = False
    # Set paths
    experimenter = 'Myriam_Hamon'

    proc_data_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'data', 'processed_data')
    all_nwb_names = os.listdir(ROOT_PATH_MYRIAM)
    all_nwb_mice = [name.split('_')[0] for name in all_nwb_names]



    if joint_analysis:
        info_path = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'z_LSENS', 'Share', f'Axel_Bisi_Share',
                                 'dataset_info')
        output_path = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter,
                                   'combined_results')
        myriam_nwb_names = os.listdir(ROOT_PATH_AXEL)
        all_nwb_names.extend(myriam_nwb_names)
        all_nwb_mice.extend([name.split('_')[0] for name in myriam_nwb_names])
        mouse_info_path = os.path.join(info_path, 'joint_mouse_reference_weight.xlsx')

    else:
        info_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'mice_info')
        output_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'results')
        mouse_info_path = pd.read_excel(os.path.join(info_path, 'mouse_reference_weight.xlsx'))


    mouse_info_df = pd.read_excel(mouse_info_path)
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    # Filter for usable mice
    mouse_info_df = mouse_info_df[
        (mouse_info_df['exclude'] == 0) &
        (mouse_info_df['reward_group'].isin(['R+', 'R-'])) &
        (mouse_info_df['recording'] == 1)
        ]

    # Show mouse count per reward group
    for group in mouse_info_df['reward_group'].unique():
        count = len(mouse_info_df[mouse_info_df['reward_group'] == group])
        print(f"Reward group {group} has {count} mice.")

    # Filter by available NWB files
    subject_ids = mouse_info_df['mouse_id'].unique()
    subject_ids = [mouse for mouse in subject_ids if any(mouse in name for name in all_nwb_mice)]

    # Exclude specific mice
    excluded_mice = ['AB073', 'AB152', 'AB158', 'MH006']  # MH026, MH015
    subject_ids = [s for s in subject_ids if s not in excluded_mice]

    print(f"Subject IDs to do: {subject_ids}")

    subject_ids = ['AB131']

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
        if subject_id.startswith('AB'):
            nwb_files = [os.path.join(ROOT_PATH_AXEL, name) for name in nwb_names]
        elif subject_id.startswith('MH'):
            nwb_files = [os.path.join(ROOT_PATH_MYRIAM, name) for name in nwb_names]

        if not nwb_files:
            print(f"No NWB files found for {subject_id}")
            continue

        subject_nwb_neural_files = []

        # Keep whisker day 0 files with neural data
        for nwb_file in nwb_files: # keep whisker day 0 only
            beh, day = nwb_reader.get_bhv_type_and_training_day_index(nwb_file)
            if beh=='whisker' and day!=0 and expert_day == True or expert_day == False and day==0:
                unit_table = nwb_reader.get_unit_table(nwb_file)

                if unit_table is not None:
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
        if single_mouse:
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
                    run_unit_glm_pipeline_with_pool(nwb_file, results_path)

    ### ------------------------------------------
    # Analyses aggregating data from multiple mice
    ### -------------------------------------------
    if multiple_mice:

        print('Multi-mouse analyses')

        if 'unit_labels_processing' in analyses_to_do_multi:
            unit_label_describe(nwb_neural_files, output_path)

        if 'rsu_vs_fsu' in analyses_to_do_multi:
            assign_rsu_vs_fsu(nwb_neural_files, output_path)

