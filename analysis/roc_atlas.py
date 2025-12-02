#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_spikes_analysis
@file: roc_atlas.py
@time: 28/11/2025 3:19 PM
"""
import sys

# Imports
import os
import argparse
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import cmasher as cmr

from brainrender import Scene
from brainrender.actors import Points

import NWB_reader_functions as nwb_reader
from roc_analysis_utils import load_roc_results

sys.path.append(r'M:\analysis\Axel_Bisi\brain_wide_analysis\neural_utils.py')


DATA_PATH = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis')
ROOT_PATH = os.path.join(DATA_PATH, 'Axel_Bisi', 'combined_results')
FIGURE_PATH = r'M:\analysis\Axel_Bisi\combined_results\roc_atlas'

ROOT_PATH_AXEL = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Axel_Bisi', 'NWBFull_bis')
ROOT_PATH_MYRIAM = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Myriam_Hamon',
                                'NWBFull_new')

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
        sess_meta = nwb_reader.get_session_metadata(nwb)
        reward_group = sess_meta.get('wh_reward')

        trial_table['mouse_id'] = mouse_id
        trial_table['session_id'] = session_id
        trial_table['context'] = trial_table['context'].astype(str)
        trial_table['reward_group'] = reward_group

        if trial_table['context'].str.contains('nan').all():
            trial_table['context'] = 'active'
        else:
            trial_table['context'] = trial_table['context'].fillna('active')
            trial_table['context'] = trial_table['context'].replace('nan','active')


        unit_table['mouse_id'] = mouse_id
        unit_table['session_id'] = mouse_id
        unit_table['day'] = day
        unit_table['behaviour'] = beh_type

        print('Warning: number of root neurons :', mouse_id, len(unit_table[unit_table.ccf_acronym=='root']))
        root_units = unit_table[unit_table.ccf_acronym=='root']
        if not root_units.empty:
            elec_groups = root_units['electrode_group'].unique()
            elec_names = [e.name for e in elec_groups]
            #print(f"Root units found in {mouse_id}: {len(root_units)} with electrode groups: {elec_names}")

        #unit_table = convert_electrode_group_object_to_columns(unit_table)

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
        #unit_table = unit_table[~unit_table['ccf_acronym'].isin(allen.get_excluded_areas())]
        unit_table = unit_table.reset_index(drop=True)
        unit_table['unit_id'] = unit_table.index

    return trial_table, unit_table, ephys_nwb_list

def load_roc_results(root_path):
    files = glob.glob(os.path.join(root_path, '**', '*_roc_results_new.csv'), recursive=True)
    print(f"  Found {len(files)} files in: {root_path}")
    dfs = []
    for f in tqdm(files, desc="  Loading ROC CSV files", unit="file"):
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"    Skipped corrupted file: {f} ({e})")
    if len(dfs) == 0:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def main():

    # Get data information
    # --------------------------
    info_path = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'z_LSENS', 'Share', f'Axel_Bisi_Share',
                             'dataset_info')
    mouse_info_path = os.path.join(info_path, 'joint_mouse_reference_weight.xlsx')
    mouse_info_df = pd.read_excel(mouse_info_path)
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    # Filter for usable mice
    mouse_info_df = mouse_info_df[
        (mouse_info_df['exclude'] == 0) &
        (mouse_info_df['reward_group'].isin(['R+', 'R-'])) &
        (mouse_info_df['recording'] == 1)
        ]

    # Filter by available NWB files
    subject_ids = mouse_info_df['mouse_id'].unique()
    subject_ids = [mouse for mouse in subject_ids if any(mouse in name for name in all_nwb_mice)]
    # subject_ids = ['AB{}'.format(str(i).zfill(3)) for i in range(116,130)]
    # Exclude specific mice
    excluded_mice = ['AB144']  # MH026, MH015
    subject_ids = [s for s in subject_ids if s not in excluded_mice]
    # subject_ids = ['AB116', 'AB117', 'AB118', 'AB119', 'AB120', 'AB121', 'AB122', 'AB123', 'AB124', 'AB125', 'AB126', 'AB127', 'AB128', 'AB129', 'AB130']

    # Get list of NWB files
    # --------------------------
    # Collect NWB files
    all_nwb_names = os.listdir(ROOT_PATH_AXEL)
    all_nwb_mice = [name.split('_')[0] for name in all_nwb_names]

    myriam_nwb_names = os.listdir(ROOT_PATH_MYRIAM)
    all_nwb_names.extend(myriam_nwb_names)
    all_nwb_mice.extend([name.split('_')[0] for name in myriam_nwb_names])


    # Get list of NWB files for each mouse
    nwb_list = [os.path.join(ROOT_PATH_AXEL, name) for name in all_nwb_names if name.startswith('AB')]
    nwb_list.extend([os.path.join(ROOT_PATH_MYRIAM, name) for name in all_nwb_names if name.startswith('MH')])

    # Keep NWB files for specified subject IDs
    # subjecsubt_ids = ['AB131', 'AB132', 'AB133', 'AB134', 'AB135', 'AB136', 'AB137', 'AB138', 'AB139', 'AB145', 'AB146', 'AB147', 'AB148', 'AB149', 'AB150', 'AB151', 'AB152', 'AB153', 'AB154' 'AB155', 'AB156', 'AB157', 'AB158']
    # subject_ids = ['AB131','AB132', 'AB133', 'AB134', 'AB135', 'AB136', 'AB137', 'AB138', 'AB139', 'AB140', 'AB141', 'AB142', 'AB143', 'AB144', 'AB145', 'AB146', 'AB147', 'AB148', 'AB149', 'AB150', 'AB151', 'AB152', 'AB153', 'AB154','AB155','AB156','AB157','AB158',
    #               'MH016','MH017','MH018','MH019','MH020','MH021','MH022','MH023','MH024','MH025']
    # subject_ids = ['AB131', 'AB132', 'AB133', 'AB134', 'AB135', 'AB136', 'AB137', 'AB138', 'AB139']
    # subject_ids = ['AB131', 'AB132']

    excluded_mice = ['AB144']
    subject_ids = [s for s in subject_ids if s not in excluded_mice]
    nwb_list = [nwb_file for nwb_file in nwb_list if any(mouse in nwb_file for mouse in subject_ids)]



    trial_table, unit_table, ephys_nwb_list = combine_ephys_nwb(nwb_list, max_workers=24)

    # ---------
    # LOAD DATA
    # ---------
    print('Loading ROC  data...')
    data_path_axel = os.path.join(DATA_PATH, 'Axel_Bisi', 'combined_results')
    roc_df = load_roc_results(data_path_axel)

    roc_df = roc_df.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')

    # Create unique unit identifier based on index
    roc_df['unit_id'] = roc_df.index.astype(int)

    print('Present mice:', roc_df['mouse_id'].unique(), 'Number of mice', roc_df['mouse_id'].nunique(), 'per reward group',
          roc_df.groupby('reward_group')['mouse_id'].nunique())
    print('ROC analysis types:', roc_df['analysis_type'].unique())

    excluded_mice = []
    roc_df = roc_df[~roc_df['mouse_id'].isin(excluded_mice)]
    print(roc_df.columns)

    # TODO: merge with anatomical data to get CCF coordinates if not already present

    # -----------------------
    # PROCESS AND FILTER DATA
    # -----------------------
    #print('Processing and filtering data...')
    #roc_df = roc_df[~roc_df['area'].isin(allen.get_excluded_areas())]

    # -----------------------
    # MERGE TABLES
    # -----------------------

    print('Merging ROC results with anatomical data...')
    # Merge with unit table to get anatomical data
    roc_df = roc_df.merge(unit_table[['mouse_id','neuron_id','ccf_ap', 'ccf_ml', 'ccf_dv', 'area_acronym_custom']],)

    # -----------------------
    # PLOTTING
    # -----------------------

    # Choose analysis type e.g. whisker stimulus ROC

    analysis_type = 'spontaneous_licks'
    reward_group = 'R+'
    roc_df_subset = roc_df[(roc_df['analysis_type'] == analysis_type) & (roc_df['reward_group'] == reward_group)]


    # Using brainrender from BrainGlobe, create a 3D plot and scene and plot indidivuals neurons using their coordinates
    print('Plotting ROC results on atlas...')
    scene = Scene(title=f'ROC analysis: {analysis_type} - Reward group {reward_group}', inset=False)
    coords_col = ['ccf_ap', 'ccf_ml', 'ccf_dv']

    # Convert p-value as -log p for better visualization
    print(roc_df_subset.columns)
    roc_df_subset['neg_log_pval'] = -np.log10(roc_df_subset['p_value'] + 1e-10)  # add small value to avoid log(0)
    max_neg_log_pval = roc_df_subset['neg_log_pval'].max()

    # Make cmap using range of pvalue fixed across analysies
    cmap = cmr.get_sub_cmap('cmr.amber', 0.15, 0.85)    # Plot each neuron location as a sphere colored by -log p-value
    vmin = 0
    vmax = 50
    # Normalize -log p-values to [0, 1] for colormap
    norm_neg_log_pval = np.clip((roc_df_subset['neg_log_pval'] - vmin) / (vmax - vmin), 0, 1)
    colors = cmap(norm_neg_log_pval.values)

    # Make range of sphere sizes on the log scale for better visibility
    min_radius = 3
    max_radius = 50
    radii = min_radius + (max_radius - min_radius) * norm_neg_log_pval.values


    for idx, row in roc_df_subset.iterrows():
        radius = radii[idx - roc_df_subset.index[0]]  # Adjust index for radii array
        color = colors[idx - roc_df_subset.index[0]]  # Adjust index for colors array
        coords = np.array([[row[coords_col[0]], row[coords_col[1]], row[coords_col[2]]]])


        scene.add(Points(coords, colors=color, radius=radius, alpha=0.6))


    # Save figure
    figure_file = os.path.join(FIGURE_PATH, f'roc_atlas_{analysis_type}_reward_group_{reward_group}.png')
    scene.render()
    scene.screenshot(figure_file)
    print(f'Figure saved to {figure_file}')


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        #parser.add_argument('--input', type=str, nargs='?', required=True)
        #parser.add_argument('--config', type=str, nargs='?', required=False)
        args = parser.parse_args()

        print('- Analysing ROC results...')
        main()
        print('Done.')