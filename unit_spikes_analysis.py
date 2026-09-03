#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_spikes_analysis
@file: unit_spikes_analysis.py
@time: 10/23/2024 11:300 PM
"""


# Imports
import os
import socket
import pathlib
import sys
import pandas as pd
import numpy as np
import yaml
import matplotlib
import concurrent.futures as cf

from ephys_utilities.helpers import data_utils, load_helpers
from ephys_utilities.neural_utils import neural_utils, unit_metrics_utils
from roc_analysis.roc_analysis_utils import load_roc_results

from raster_utils import plot_rasters
from noise_unit_detection import identify_noise_units
from roc_analysis.roc_utils_new import compute_unit_roc
from task_modulation_utils import task_modulation_analysis
from waveform_utils import plot_waveforms, classify_rsu_vs_fsu, classify_striatal_units
from unit_desc_utils import *
from unit_spike_report_new import generate_unit_spike_report, export_unit_quality_metrics
#from glm_utils import run_unit_glm_pipeline_with_pool
from noise_correl_utils import noise_correlation_analysis
from passive_psth_utils import run_passive_psths
from rastermap_psth.area_latency_rastermap import run_area_latency_rastermap
from single_neuron_shift_test.unit_fr_motion_shift_test_harris import run_motion_shift_test_analysis
#from neural_inflection.neural_inflection_analysis_figs import load_shift_test_results, get_learning_df, run_analysis, run_figures_only


def process_subject(subject_id, nwb_neural_files, mouse_results_path_root, analyses_to_do_single):
    """Runs all single-mouse analyses for one subject. Must be a top-level function (picklable) for spawn."""
    # Must be set before numpy/scipy/pandas import in this process — prevents each
    # worker's BLAS backend from silently grabbing all cores (oversubscription -> OOM/137)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

    matplotlib.use('Agg')  # avoid GUI backend issues across worker processes

    mouse_results_path = os.path.join(mouse_results_path_root, subject_id)
    os.makedirs(mouse_results_path, exist_ok=True)

    nwb_files = [nwb for nwb in nwb_neural_files if subject_id in nwb]
    if not nwb_files:
        print(f"No NWB files found for {subject_id}")
        return subject_id, 'skipped'

    for nwb_file in nwb_files:
        beh, day = nwb_reader.get_bhv_type_and_training_day_index(nwb_file)
        mouse_output_path = os.path.join(mouse_results_path, f'{beh}_{day}')
        os.makedirs(mouse_output_path, exist_ok=True)

        for analysis_type in analyses_to_do_single:
            folder_results_path = os.path.join(mouse_output_path, analysis_type)
            os.makedirs(folder_results_path, exist_ok=True)

            try:
                if analysis_type == 'unit_raster':
                    plot_rasters(nwb_file, folder_results_path)
                elif analysis_type == 'unit_spike_report':
                    generate_unit_spike_report(nwb_file, folder_results_path)
                elif analysis_type == 'unit_drift_qc_metrics':
                    export_unit_quality_metrics(nwb_file, folder_results_path)
                elif analysis_type == 'unit_waveforms':
                    plot_waveforms(nwb_file, folder_results_path)
                elif analysis_type == 'roc_analysis':
                    compute_unit_roc(nwb_file, folder_results_path, make_psth_plots=True)
                elif analysis_type == 'task_modulation':
                    task_modulation_analysis(nwb_file, folder_results_path)
                elif analysis_type == 'noise_correlation':
                    noise_correlation_analysis(nwb_file, folder_results_path)
            except Exception as e:
                print(f"[{subject_id} / {nwb_file} / {analysis_type}] failed: {e}")

    return subject_id, 'done'

if __name__ == '__main__':

    load_tables = True
    single_mouse = True
    multiple_mice = False
    joint_analysis = True
    expert_day = False

    # Set paths
    experimenter = 'Axel_Bisi'
    day_to_analyze = 'learning' # then do for "expert"


    hostname = socket.gethostname()
    if 'haas' in hostname:
        N_WORKERS = 110

        ROOT = pathlib.Path('/mnt/lsens-analysis/Axel_Bisi/unit_spikes_analysis')
        ROOT_PATH_AXEL = pathlib.Path('/mnt/lsens-analysis/Axel_Bisi/NWB_combined')
        ROOT_PATH_AXEL = pathlib.Path('/mnt/lsens-analysis/Axel_Bisi/NWB_ks4')
        ROOT_PATH_MYRIAM = pathlib.Path('/mnt/lsens-analysis/Myriam_Hamon/NWB')
        ROOT_PATH_MYRIAM = pathlib.Path('/mnt/lsens-analysis/Axel_Bisi/NWB_ks4')
        INFO_PATH = pathlib.Path('/mnt/share_internal/Axel_Bisi_Share/dataset_info')  # temp before mounted
        OUTPUT_PATH = pathlib.Path(f'/mnt/lsens-analysis/{experimenter}/combined_results_ks4')
        DATA_PATH = pathlib.Path('/mnt/lsens-analysis/Axel_Bisi/data')
        #sys.path.insert(0, "/home/bisi/code")
        #sys.path.insert(0, "/home/bisi/code/NWB_reader")

    else:
        N_WORKERS=15
        ROOT = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch') / 'Petersen-Lab' / 'analysis' / 'Axel_Bisi' / 'unit_spikes_analysis'
        ROOT_PATH_AXEL = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch') / 'Petersen-Lab' / 'analysis' / 'Axel_Bisi' / 'NWB_combined'
        ROOT_PATH_MYRIAM = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch') / 'Petersen-Lab' / 'analysis' / 'Myriam_Hamon' / 'NWB'
        INFO_PATH = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch') / 'Petersen-Lab' / 'share_internal' / f'Axel_Bisi_Share' / 'dataset_info'
        OUTPUT_PATH = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter,
                                   'combined_results')
        DATA_PATH = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch') / 'Petersen-Lab' / 'analysis' / 'Axel_Bisi' / 'data'


    #proc_data_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'data', 'processed_data')
    if experimenter == 'Axel_Bisi':
        all_nwb_names = os.listdir(ROOT_PATH_AXEL)
    elif experimenter == 'Myriam_Hamon':
        all_nwb_names = os.listdir(ROOT_PATH_MYRIAM)
    all_nwb_mice = [name.split('_')[0] for name in all_nwb_names]

    if joint_analysis:
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
        (mouse_info_df['recording'] == 1) #pertains to day 0
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
    excluded_mice = ['AB068', 'AB077'] #MH065 video stopped midway
    #subject_ids = [s for s in subject_ids if s not in excluded_mice]

    print(f"Subject IDs to do: {subject_ids}")

    #subject_ids = ['AB131', 'AB132', 'AB133', 'AB134']
    #subject_ids = ['AB116', 'AB117', 'AB141']
    #subject_ids = ['AB144', 'AB155']

    ### --------------------
    # Define analyses to do
    ### -------------------

    # Single-mouse analyses
    analyses_to_do_single = ['unit_raster', 'unit_waveforms', 'roc_analysis', 'xcorr_analysis']
    analyses_to_do_single = ['noise_correlation']
    analyses_to_do_single = ['roc_analysis']
    analyses_to_do_single = ['unit_spike_report']
    analyses_to_do_single = ['task_modulation']
    analyses_to_do_single = ['unit_waveforms']

    # Multi-mouse analyses
    analyses_to_do_multi = ['rsu_vs_fsu', 'striatal_type', 'noise_unit_detection']
    analyses_to_do_multi = ['unit_labels_processing', 'unit_anat_processing']
    analyses_to_do_multi = ['unit_anat_processing', 'area_pairs_describe'] #fix area pairs describe
    analyses_to_do_multi = ['striatal_type']
    analyses_to_do_multi = ['passive_psths_prepost']

    # Analyses to do
    analyses_to_do_single = ['task_modulation','roc_analysis', 'unit_waveforms','unit_drift_qc_metrics']
    analyses_to_do_single = ['roc_analysis']

    analyses_to_do_multi = ['noise_unit_detection']
    #analyses_to_do_multi = ['noise_classification']
    #analyses_to_do_multi = ['neural_inflection']
    analyses_to_do_multi = ['rastermap_psth']

    # --------------
    # Load NWB files
    # --------------

    nwb_list = [os.path.join(ROOT_PATH_AXEL, name) for name in all_nwb_names if name.startswith('AB')]
    nwb_list.extend([os.path.join(ROOT_PATH_AXEL, name) for name in all_nwb_names if name.startswith('MH')])
    nwb_list = [nwb for nwb in nwb_list if any(subj in nwb for subj in subject_ids)]

    #nwb_list = nwb_list[::10]
    #mice = ('AB131','AB132', 'AB134', 'AB164', 'AB161'
    #nwb_list = [n for n in nwb_list if any(m in n for m in mice)]

    if load_tables:
        trial_table, unit_table, nwb_neural_files = data_utils.combine_ephys_nwb(nwb_list,
                                                                                 day_to_analyze=day_to_analyze,
                                                                                 max_workers=N_WORKERS)

        show_counts=False
        if show_counts:
            learners = mouse_info_df[mouse_info_df.learning_category.isin(['good', 'moderate'])]
            non_learners = mouse_info_df[~mouse_info_df.learning_category.isin(['good', 'moderate'])]

            for name, mice in [('Learners', learners), ('Non-learners', non_learners)]:
                ephys = unit_table[unit_table.mouse_id.isin(mice.mouse_id)]
                for label, d in [('day 0', ephys[ephys.day == 0]), ('day >0', ephys[ephys.day > 0])]:
                    print(
                        f'{name} {label}: {d.mouse_id.nunique()} mice, {d.session_id.nunique()} sessions, {d[["target_region", "session_id"]].drop_duplicates().shape[0]} insertions')

            for name, mice in [('Learners', learners), ('Non-learners', non_learners)]:
                ephys = unit_table[unit_table.mouse_id.isin(mice.mouse_id)]
                for label, d in [('day 0', ephys[ephys.day == 0]), ('day >0', ephys[ephys.day > 0])]:
                    print(f'\n{name} {label}')
                    print(d.groupby(['target_region', 'reward_group'])[['target_region', 'session_id']].apply(
                        lambda x: x.drop_duplicates().shape[0]))

            d = unit_table[(unit_table.mouse_id.isin(mouse_info_df.mouse_id)) & (unit_table.day > 0)]
            print(d.groupby(['target_region', 'reward_group'])[['target_region', 'session_id']].apply(
                lambda x: x.drop_duplicates().shape[0]))

        ## Apply good/mua classification
        #dredge_df = load_helpers.load_motion_dredge_shift_test_results(nwb_neural_files, day_to_analyze=day_to_analyze, max_workers=N_WORKERS)
        #dredge_df = dredge_df[['mouse_id', 'session_id', 'cluster_id', 'electrode_group', 'p_conservative', 'r']].rename(
        #    columns={'p_conservative': 'drift_shift_test_pval'})
        #dredge_df['drift_abs_r'] = dredge_df['r'].abs()
        #unit_table['cluster_id'] = unit_table['cluster_id'].astype(str)
        #dredge_df['cluster_id'] = dredge_df['cluster_id'].astype(str)
        #unit_table = unit_table.merge(dredge_df, on=['mouse_id', 'session_id', 'cluster_id', 'electrode_group'],
        #                              how='left', validate='one_to_one')
#
        #unit_table = unit_metrics_utils.compute_presence_coverage_metrics(unit_table)
        #print(f"Default labels - number of good bc_label: {sum(unit_table['bc_label'] == 'good')}, Number of mua bc_label: {sum(unit_table['bc_label'] == 'mua')}")
        #unit_table = unit_metrics_utils.classify_units_quality(unit_table, label_col='quality_label')
        #recovered = (unit_table['bc_label'] == 'mua') & (unit_table['quality_label'] == 'good')
        #print(f"Custom labels - number of good labels: {sum(unit_table['quality_label'] == 'good')}, Number of mua bc_label: {sum(unit_table['quality_label'] == 'mua')}")


        # Process Allen labels
        unit_table = allen_utils.process_allen_labels(unit_table)

        # Merge anatomical information
        unit_table = allen_utils.merge_liu_avg_ipsi(unit_table)
        unit_table = allen_utils.merge_hierarchy_columns_from_gao(unit_table)
        hierarchy_df = allen_utils.load_process_hierarchy_columns_from_gao()
        #fig = gao_column_assignment_diagnostics.plot_column_assignment_diagnostics(unit_table, hierarchy_df, merge_key='ccf_atlas_acronym_no_layer')
        #fig.savefig('column_assignment_diagnostics.png', dpi=300)
        unit_table = allen_utils.merge_hierarchy_from_harris(unit_table)


        # Load spontaneous/reward licks
        #lick_times_df = load_helpers.load_spontaneous_reward_lick_times(nwb_neural_files, day_to_analyze=day_to_analyze, max_workers=N_WORKERS, load_summary=False)

        # Load ROC
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
        roc_cols_to_keep = ['mouse_id', 'session_id', 'target_region', 'cluster_id', 'neuron_id',
                            'analysis_type', 'selectivity', 'direction',
                            'p_value_to_show', 'significant']
        #unit_table['cluster_id'] = unit_table['cluster_id'].astype(str)
        roc_df['neuron_id'] = roc_df['neuron_id'].astype(str)
        unit_table['neuron_id'] = unit_table['neuron_id'].astype(str)
        print(len(unit_table), unit_table.columns)
        print(len(roc_df), roc_df.columns)
        print(len(unit_table))


        # Load jaw onset times, then join onto trial table
        #jaw_onset_table = load_helpers.load_jaw_onset_data(nwb_neural_files, day_to_analyze=day_to_analyze, max_workers=N_WORKERS)
        #if jaw_onset_table is not None:
        #    trial_table = trial_table.merge(
        #        jaw_onset_table[['mouse_id', 'session_id', 'trial_id', 'jaw_dlc_onset', 'piezo_lick_time']],
        #        on=['mouse_id', 'session_id', 'trial_id'], how='left')
        #    trial_table['jaw_onset_time'] = trial_table['start_time'] + trial_table['jaw_dlc_onset']
        #    print(' Unique mice after jaw onset merge:', trial_table.mouse_id.unique())

        # ----------------------------------------
    else:
        unit_table = None
        trial_table = None

    # Perform analyses for each mouse NWB file
    # ----------------------------------------
    if single_mouse:
        with cf.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {
                executor.submit(process_subject, sid, nwb_neural_files, OUTPUT_PATH, analyses_to_do_single): sid
                for sid in subject_ids
            }
            for fut in cf.as_completed(futures):
                sid = futures[fut]
                try:
                    _, status = fut.result()
                    print(f"{sid}: {status}")
                except Exception as e:
                    print(f"{sid} raised: {e}")

    #if single_mouse: #delte if not used
    #    for subject_id in subject_ids:
    #        print(f"Subject ID : {subject_id}")
    #        # Create results  folders for the subject
    #        mouse_results_path = os.path.join(OUTPUT_PATH, subject_id)
    #        os.makedirs(mouse_results_path, exist_ok=True)
#
    #        nwb_files = [nwb for nwb in nwb_neural_files if subject_id in nwb]
    #        if not nwb_files:
    #            print(f"No NWB files found for {subject_id}")
    #            continue
    #        for nwb_file in nwb_files:
    #            # Create ephys day folder for the session
    #            beh,day = nwb_reader.get_bhv_type_and_training_day_index(nwb_file)
    #            mouse_output_path = os.path.join(mouse_results_path, f'{beh}_{day}')
    #            os.makedirs(mouse_output_path, exist_ok=True)
#
    #            for analysis_type in analyses_to_do_single:
#
    #                # Define and create results path
    #                folder_results_path = os.path.join(mouse_output_path, analysis_type)
    #                os.makedirs(folder_results_path, exist_ok=True)
#
    #                if 'unit_raster' in analyses_to_do_single:
    #                    plot_rasters(nwb_file, folder_results_path)
#
    #                if 'unit_spike_report' in analyses_to_do_single:
    #                    generate_unit_spike_report(nwb_file, folder_results_path)
#
    #                if 'unit_drift_qc_metrics' in analyses_to_do_single:
    #                    export_unit_quality_metrics(nwb_file, folder_results_path)
#
    #                if 'unit_waveforms' in analyses_to_do_single:
    #                    plot_waveforms(nwb_file, folder_results_path)
#
    #                if 'roc_analysis' in analyses_to_do_single:
    #                    compute_unit_roc(nwb_file, folder_results_path)
#
    #                if 'xcorr_analysis' in analyses_to_do_single:
    #                    #xcorr_analysis(nwb_file, folder_results_path) # on cluster, otherwise adapt xcorr_analysis_mpi for multiprocessing
    #                    pass
#
    #                if 'unit_glm' in analyses_to_do_single:
    #                    run_unit_glm_pipeline_with_pool(nwb_file, folder_results_path)
#
    #                if 'task_modulation' in analyses_to_do_single:
    #                    task_modulation_analysis(nwb_file, folder_results_path)
#
    #                if 'noise_correlation' in analyses_to_do_single:
    #                    noise_correlation_analysis(nwb_file, folder_results_path)



    ### ------------------------------------------
    # Analyses aggregating data from multiple mice
    ### -------------------------------------------

    if multiple_mice:

        print('Multi-mouse analyses: ', analyses_to_do_multi)

        if 'unit_labels_processing' in analyses_to_do_multi:
            unit_label_describe(unit_table, output_path=OUTPUT_PATH)
            unit_label_metric_sankey(unit_table, output_path=OUTPUT_PATH)

        if 'unit_anat_processing' in analyses_to_do_multi:
            unit_anat_describe(unit_table, output_path=OUTPUT_PATH)

        if 'noise_unit_detection' in analyses_to_do_multi:
            identify_noise_units(unit_table, trial_table, output_path=OUTPUT_PATH)

        if 'area_pairs_describe' in analyses_to_do_multi:
            plot_number_area_pairs_heatmap(trial_table, unit_table, output_path=OUTPUT_PATH)

        if 'rsu_vs_fsu' in analyses_to_do_multi:
            for thr_mode in ['single','double']:
                for level in ['isocortex_group', 'area_acronym_custom']:
                    params={'threshold_mode':thr_mode, 'uncertainty_percentile': 80.0, 'level':level}
                    classify_rsu_vs_fsu(unit_table, output_path=OUTPUT_PATH, **params)

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
            # Run rastermap
            #run_rastermap_psth(unit_table, trial_table, lick_df=lick_times_df, out_root=OUTPUT_PATH)

            # Clustering pipeline
            # -----------------------

            from rastermap_psth.build_feature_matrix import run_build_feature_matrix
            path_to_config = pathlib.Path(ROOT, 'rastermap_psth', 'config.yaml')
            cfg = yaml.load(path_to_config.open(), Loader=yaml.FullLoader)
            #result = run_build_feature_matrix(unit_table, trial_table, lick_df=lick_times_df, config_path=path_to_config, out_root=OUTPUT_PATH)
            from rastermap_psth.run_clustering_new import run_rastermap, run_gmm
            #rastermap_result = run_rastermap(data_folder=result["data_folder"], config_path=path_to_config)
            #gmm_result = run_gmm(data_folder=result["data_folder"], config_path=path_to_config)

            #from rastermap_psth.run_clustering_new import run_clustering
            #print(result["data_folder"])
            #cluster_result = run_clustering(data_folder=result["data_folder"], config_path=path_to_config)
            from rastermap_psth.cluster_comparison import run_cluster_comparison
            #run_cluster_comparison(gmm_result["out_folder"], method_a="rastermap", method_b="gmm", cv=True)

            #run_cluster_comparison(cluster_result["out_folder"], cv=True)


            # Stat between reward groups
            # ---------------------------

            rastermap_out_dir = r"/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_clustering/passive_active/whisker_auditory/clustering/n100_passive_active_baseline_whisker_auditory_combined_nobl/rastermap"
            from rastermap_psth.rastermap_cluster_analyses import run_rastermap_analyses
            from rastermap_psth.rastermap_utils import run_stats_only
            #results = run_stats_only(rastermap_out_dir, cfg, unit_table) # give path is here
            #run_rastermap_analyses(rastermap_out_dir, unit_table, trial_table)

            # Area correlations
            # -----------------
            from rastermap_psth.cluster_area_correlations import run_all

            unit_table = unit_table.merge(roc_df[roc_cols_to_keep],
                                          on=['mouse_id', 'session_id', 'neuron_id'], how='left')
            anatomical_cols = ["avg_ipsi", "cc_hierarchy_score_columns", "cc_tc_ct_iterated"]

            results = run_all(
                rastermap_out_folder="/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_clustering/"
                                     "passive_active/whisker_auditory/clustering/"
                                     "n100_passive_active_baseline_whisker_auditory_combined_nobl/rastermap",
                unit_table=unit_table,
                anatomical_cols=anatomical_cols,
                analysis_type_col="analysis_type",  # your unit_table's analysis_type column name
                sig_col="significant",  # your boolean ROC-significance column name
                selectivity_col="selectivity",  # your continuous selectivity column name
                selectivity_analysis_types={"wh_vs_aud_active"},
                # analysis_types where selectivity_index is used instead of significant
                min_neurons=10,
                n_perm=5000,
                specialization_baseline="uniform",
            )

            # Additional analyses
            # ---------------------------

            from rastermap_psth.cluster_postproc_analyses import run_rastermap_analyses
            results = run_rastermap_analyses(rastermap_out_dir, unit_table=unit_table, trial_table=trial_table)


        if 'area_latency_rastermap' in analyses_to_do_multi:
            run_area_latency_rastermap(unit_table, trial_table, OUTPUT_PATH)

        if 'single_neuron_shift_test' in analyses_to_do_multi:
            config = {'data_root': DATA_PATH} #to override default config
            run_motion_shift_test_analysis(
                unit_table=unit_table,
                trial_table=trial_table,  #  needed for performance tertiles
                output_path=OUTPUT_PATH,
                config=config
            )

        if 'neural_inflection' in analyses_to_do_multi:
            path_to_data = r'M:\analysis\Axel_Bisi\combined_results'
            #shift_df = load_shift_test_results(subject_ids)
            #learning_df = get_learning_df(path_to_data, subject_ids)
            #run_analysis(unit_table, trial_table, learning_df, shift_df=shift_df)
            #run_figures_only(learning_df)


        #if 'noise_classification' in analyses_to_do_multi:
        #    from noise_classification import label_gui, train_classifier, apply_classifier

        #    output = os.path.join(OUTPUT_PATH, 'noise_classification', 'labels.csv')
        #    #label_gui.run_labeling_gui(unit_table, trial_table, output)
        #    #train_classifier.train(labels_csv = os.path.join(OUTPUT_PATH, 'noise_classification', 'labels.csv'),unit_table = unit_table,trial_table = trial_table,model_dir = os.path.join(OUTPUT_PATH, 'noise_classification', 'model'))
        #    apply_classifier.apply(unit_table = unit_table,model_dir = os.path.join(OUTPUT_PATH, 'noise_classification', 'model'),output_csv = os.path.join(OUTPUT_PATH, 'noise_classification', 'predictions.csv'),bc_labels_to_screen = ("good", "mua"))
