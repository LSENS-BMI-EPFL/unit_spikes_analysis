#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_spikes_analysis
@file: unit_spikes_analysis.py
@time: 10/23/2024 11:300 PM
"""


# Imports
import socket
import pathlib

import sys
sys.path.insert(0, r"M:\analysis\Axel_Bisi\NWB_reader")
sys.path.insert(0, r"M:\analysis\Axel_Bisi\Github\allen_utils")

from load_helpers import load_jaw_onset_data
from roc_analysis.roc_analysis_utils import load_roc_results
#import NWB_reader_functions as nwb_reader

from raster_utils import plot_rasters
from noise_unit_detection import identify_noise_units
from unit_spike_report import generate_unit_spike_report
from roc_utils import roc_analysis
from task_modulation_utils import task_modulation_analysis
from waveform_utils import classify_rsu_vs_fsu, classify_striatal_units
from unit_desc_utils import *

#from glm_utils import run_unit_glm_pipeline_with_pool
from noise_correl_utils import noise_correlation_analysis

from passive_psth_utils import run_passive_psths
#from rastermap_psth.rastermap_clustering_psth import run_rastermap_psth
from single_neuron_shift_test.single_neuron_shift_test_figs_test_new import run_shift_test_analysis
#from neural_inflection.neural_inflection_analysis_figs import load_shift_test_results, get_learning_df, run_analysis, run_figures_only
from rastermap_psth.area_latency_rastermap import run_area_latency_rastermap

if __name__ == '__main__':

    load_tables = True
    single_mouse = False
    multiple_mice = True
    joint_analysis = True
    expert_day = False

    # Set paths
    experimenter = 'Axel_Bisi'

    hostname = socket.gethostname()
    if 'haas' in hostname:
        N_WORKERS = 120
        ROOT_PATH_AXEL = pathlib.Path('/mnt/lsens-analysis/Axel_Bisi/NWB_combined')
        ROOT_PATH_MYRIAM = pathlib.Path('/mnt/lsens-analysis/Myriam_Hamon/NWB')
        INFO_PATH = pathlib.Path('/mnt/share_internal/Axel_Bisi_Share/dataset_info')  # temp before mounted
        OUTPUT_PATH = pathlib.Path(f'/mnt/lsens-analysis/{experimenter}/combined_results')

        sys.path.insert(0, "/home/bisi/code")
        sys.path.insert(0, "/home/bisi/code/NWB_reader")

    else:
        N_WORKERS=30
        ROOT_PATH_AXEL = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Axel_Bisi', 'NWB_combined')
        ROOT_PATH_MYRIAM = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Myriam_Hamon',
                                        'NWB')
        INFO_PATH = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'share_internal', f'Axel_Bisi_Share',
                                 'dataset_info')
        OUTPUT_PATH = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter,
                                   'combined_results')

    #proc_data_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'data', 'processed_data')
    if experimenter == 'Axel_Bisi':
        all_nwb_names = os.listdir(ROOT_PATH_AXEL)
    elif experimenter == 'Myriam_Hamon':
        all_nwb_names = os.listdir(ROOT_PATH_MYRIAM)
    all_nwb_mice = [name.split('_')[0] for name in all_nwb_names]

    if joint_analysis:

        # if experimenter == 'Axel_Bisi':
        #     nwb_names_bis = os.listdir(ROOT_PATH_MYRIAM)
        # elif experimenter == 'Myriam_Hamon':
        #     nwb_names_bis = os.listdir(ROOT_PATH_AXEL)
        # all_nwb_names.extend(nwb_names_bis)
        #all_nwb_mice.extend([name.split('_')[0] for name in myriam_nwb_names])
        mouse_info_path = os.path.join(INFO_PATH, 'joint_mouse_reference_weight.xlsx')

    else:
        INFO_PATH = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'mice_info')
        OUTPUT_PATH = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'results')
        mouse_info_path = os.path.join(INFO_PATH, 'mouse_reference_weight.xlsx')

    mouse_info_df = pd.read_excel(mouse_info_path)
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    # Filter for usable mice
    mouse_info_df = mouse_info_df[
        (mouse_info_df['exclude'] == 0) &
        (mouse_info_df['exclude_ephys'] == 0) &
        (mouse_info_df['reward_group'].isin(['R+', 'R-'])) &
        (mouse_info_df['recording'] == 1)
        ]

    # Show mouse count per reward group
    for group in mouse_info_df['reward_group'].unique():
        count = len(mouse_info_df[mouse_info_df['reward_group'] == group])
        print(f"Reward group {group} has {count} mice.")

    # Filter by available NWB files
    all_nwb_mice = [name.split('_')[0] for name in all_nwb_names]
    subject_ids = mouse_info_df['mouse_id'].unique()
    subject_ids = [mouse for mouse in subject_ids if any(mouse in name for name in all_nwb_mice)]

    # Exclude specific mice
    excluded_mice = ['AB077', 'AB080','AB082','AB085', 'AB092','AB093', 'AB095', 'AB144'] #invalid NWB file 006, 038 ephys_exclude
    excluded_mice = ['AB068', 'AB077','AB144'] #Ab144 ccf labels with Nans
    done_mice = ['AB082','AB086', 'AB120', 'AB162', 'AB134', 'AB141'] #Ab144 ccf labels with Nans
    subject_ids = [s for s in subject_ids if s not in excluded_mice]
    #subject_ids = ['AB144']
    #subject_ids = ['MH062', 'MH064', 'MH065', 'MH068', 'MH069', 'MH070']

    #subject_ids = ['AB131', 'AB133', 'AB082', 'AB151']
    #subject_ids = ['AB162', 'AB131', 'AB164']
    #subject_ids = ['AB162']
    #subject_ids = subject_ids[::15]

    print(f"Subject IDs to do: {subject_ids}")

    #subject_ids = ['AB131']

    ### --------------------
    # Define analyses to do
    ### -------------------

    # Single-mouse analyses
    analyses_to_do_single = ['unit_raster', 'roc_analysis', 'xcorr_analysis']
    analyses_to_do_single = ['noise_correlation']
    analyses_to_do_single = ['roc_analysis']
    analyses_to_do_single = ['unit_spike_report']
    analyses_to_do_single = ['task_modulation']

    # Multi-mouse analyses
    analyses_to_do_multi = ['rsu_vs_fsu', 'striatal_type', 'noise_unit_detection']
    analyses_to_do_multi = ['unit_labels_processing', 'unit_anat_processing']
    analyses_to_do_multi = ['unit_anat_processing', 'area_pairs_describe'] #fix area pairs describe
    analyses_to_do_multi = ['striatal_type']
    analyses_to_do_multi = ['passive_psths_prepost']

    # Analyses to do
    analyses_to_do_single = ['roc_analysis']
    analyses_to_do_multi = ['noise_unit_detection']
    analyses_to_do_multi = ['rastermap_psth']
    #analyses_to_do_multi = ['noise_classification']
    analyses_to_do_multi = ['single_neuron_shift_test']
    #analyses_to_do_multi = ['neural_inflection']
    analyses_to_do_multi = ['area_latency_rastermap']


    # --------------
    # Load NWB files
    # --------------

    nwb_list = [os.path.join(ROOT_PATH_AXEL, name) for name in all_nwb_names if name.startswith('AB')]
    nwb_list.extend([os.path.join(ROOT_PATH_MYRIAM, name) for name in all_nwb_names if name.startswith('MH')])
    nwb_list = [nwb for nwb in nwb_list if any(subj in nwb for subj in subject_ids)]

    #nwb_list = nwb_list[::10]
    #mice = ("AB119", "AB131", "AB132", "AB133")
    #nwb_list = [n for n in nwb_list if any(m in n for m in mice)]

    if load_tables:
        trial_table, unit_table, nwb_neural_files = nutils.combine_ephys_nwb(nwb_list, max_workers=N_WORKERS)
        unit_table = allen_utils.process_allen_labels(unit_table, subdivide_areas=True)

        #subject_ids = [s for s in subject_ids if 'AB13' in s]

        ## Load jaw onset times, then join onto trial table
        jaw_onset_table = load_jaw_onset_data(nwb_neural_files)
        if jaw_onset_table is not None:
            trial_table = trial_table.merge(
                jaw_onset_table[['mouse_id', 'session_id', 'trial_id', 'jaw_dlc_onset', 'piezo_lick_time']],
                on=['mouse_id', 'session_id', 'trial_id'], how='left')
            trial_table['jaw_onset_time'] = trial_table['start_time'] + trial_table['jaw_dlc_onset']

        # ----------------------------------------
    # Perform analyses for each mouse NWB file
    # ----------------------------------------

    if single_mouse:
        for subject_id in subject_ids:
            print(f"Subject ID : {subject_id}")
            # Create results  folders for the subject
            mouse_results_path = os.path.join(OUTPUT_PATH, subject_id)
            os.makedirs(mouse_results_path, exist_ok=True)

            nwb_files = [nwb for nwb in nwb_neural_files if subject_id in nwb]
            if not nwb_files:
                print(f"No NWB files found for {subject_id}")
                continue
            for nwb_file in nwb_files:
                # Create ephys day folder for the session
                beh,day = nwb_reader.get_bhv_type_and_training_day_index(nwb_file)
                mouse_output_path = os.path.join(mouse_results_path, f'{beh}_{day}')
                os.makedirs(mouse_output_path, exist_ok=True)

                for analysis_type in analyses_to_do_single:

                    # Define and create results path
                    results_path = os.path.join(mouse_output_path, analysis_type)
                    os.makedirs(results_path, exist_ok=True)

                    if 'unit_raster' in analyses_to_do_single:
                        plot_rasters(nwb_file, results_path)

                    if 'unit_spike_report' in analyses_to_do_single:
                        generate_unit_spike_report(nwb_file, results_path)

                    if 'roc_analysis' in analyses_to_do_single:
                        roc_analysis(nwb_file, results_path)

                    if 'xcorr_analysis' in analyses_to_do_single:
                        #xcorr_analysis(nwb_file, results_path) # on cluster, otherwise adapt xcorr_analysis_mpi for multiprocessing
                        pass

                    if 'unit_glm' in analyses_to_do_single:
                        run_unit_glm_pipeline_with_pool(nwb_file, results_path)

                    if 'task_modulation' in analyses_to_do_single:
                        task_modulation_analysis(nwb_file, results_path)

                    if 'noise_correlation' in analyses_to_do_single:
                        noise_correlation_analysis(nwb_file, results_path)




    ### ------------------------------------------
    # Analyses aggregating data from multiple mice
    ### -------------------------------------------

    if multiple_mice:

        print('Multi-mouse analyses')

        if 'unit_labels_processing' in analyses_to_do_multi:
            unit_label_describe(unit_table, output_path=OUTPUT_PATH)

        if 'unit_anat_processing' in analyses_to_do_multi:
            unit_anat_describe(unit_table, output_path=OUTPUT_PATH)

        if 'noise_unit_detection' in analyses_to_do_multi:
            identify_noise_units(unit_table, trial_table, output_path=OUTPUT_PATH)

        if 'area_pairs_describe' in analyses_to_do_multi:
            plot_number_area_pairs_heatmap(trial_table, unit_table, output_path=OUTPUT_PATH)

        if 'rsu_vs_fsu' in analyses_to_do_multi:
            classify_rsu_vs_fsu(unit_table, output_path=OUTPUT_PATH)

        if 'striatal_type' in analyses_to_do_multi:
            classify_striatal_units(unit_table, output_path=OUTPUT_PATH)

        if 'passive_psths_prepost' in analyses_to_do_multi:
            roc_df = load_roc_results(OUTPUT_PATH, max_workers=N_WORKERS)
            unit_table_mice = unit_table.mouse_id.unique()
            roc_df = roc_df[roc_df.mouse_id.isin(unit_table_mice)]

            # Fix: correct for choice the direction, positive and negative are inverted
            choice_analyses = [type for type in roc_df.analysis_type if 'choice' in type]
            choice_mask = roc_df['analysis_type'].isin(choice_analyses)
            # Invert direction for these rows (positive becomes negative and vice versa)
            roc_df.loc[choice_mask, 'direction'] = roc_df.loc[choice_mask, 'direction'].replace(
                {'positive': 'negative', 'negative': 'positive'})

            # Merge on mouse_id,session_id,,neuron_id
            roc_cols_to_keep = ['mouse_id', 'session_id', 'neuron_id', 'analysis_type', 'selectivity', 'direction', 'p_value_to_show', 'significant']
            unit_table = unit_table.merge(roc_df[roc_cols_to_keep], on=['mouse_id','session_id','neuron_id'], how='left')

            # Run
            run_passive_psths(unit_table, trial_table, OUTPUT_PATH)

        if 'rastermap_psth' in analyses_to_do_multi:
            run_rastermap_psth(unit_table, trial_table, OUTPUT_PATH)
            #results = run_stats_only(r"M:\analysis\Axel_Bisi\combined_results\rastermap_psth_jaw_new\n_clusters_100\both\zscore_full\whisker_auditory\combined") # give path is here

        if 'area_latency_rastermap' in analyses_to_do_multi:
            run_area_latency_rastermap(unit_table, trial_table, OUTPUT_PATH)

        if 'single_neuron_shift_test' in analyses_to_do_multi:
            run_shift_test_analysis(
                unit_table=unit_table,
                trial_table=trial_table,  #  needed for performance tertiles
                output_path=OUTPUT_PATH,
                figures_only=True,
            )

        if 'neural_inflection' in analyses_to_do_multi:
            path_to_data = r'M:\analysis\Axel_Bisi\combined_results'
            #shift_df = load_shift_test_results(subject_ids)
            #learning_df = get_learning_df(path_to_data, subject_ids)
            #run_analysis(unit_table, trial_table, learning_df, shift_df=shift_df)
            #run_figures_only(learning_df)


        #if 'noise_classification' in analyses_to_do_multi:
        #    from noise_classification import label_gui, train_classifier, apply_classifier
#
        #    output = os.path.join(OUTPUT_PATH, 'noise_classification', 'labels.csv')
        #    #label_gui.run_labeling_gui(unit_table, trial_table, output)
        #    #train_classifier.train(labels_csv = os.path.join(OUTPUT_PATH, 'noise_classification', 'labels.csv'),unit_table = unit_table,trial_table = trial_table,model_dir = os.path.join(OUTPUT_PATH, 'noise_classification', 'model'))
        #    apply_classifier.apply(unit_table = unit_table,model_dir = os.path.join(OUTPUT_PATH, 'noise_classification', 'model'),output_csv = os.path.join(OUTPUT_PATH, 'noise_classification', 'predictions.csv'),bc_labels_to_screen = ("good", "mua"))
