
# Imports
import os
import pathlib
import argparse
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

import pandas as pd
import cmasher as cmr

# Brainrender and settings
import brainrender
from brainrender import Scene
from brainrender.actors import Points, Point, PointsDensity

brainrender.settings.SHOW_AXES = False
brainrender.settings.ROOT_ALPHA = 0.1
brainrender.settings.ROOT_COLOR = [0.99, 0.99, 0.99]
brainrender.settings.SHADER_STYLE = 'cartoon'
brainrender.settings.SCREENSHOT_TRANSPARENT_BACKGROUND = True

# Custom imports
import plotting_utils
import NWB_reader_functions
import neural_utils as nutils
import allen_utils as allen

from roc_analysis_utils import load_roc_results


def plot_logp_colorbar(cmap,  logp_min, logp_max, label='−log(p)'):
    """ Standalone colorbar."""
    fig, ax = plt.subplots(figsize=(5,1), dpi=500)
    norm = mpl.colors.Normalize(vmin=logp_min, vmax=logp_max)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig, ax

def value_to_color(value, mode, cmap, logp_min=None, logp_max=None):
    """
    Map either:
      - p-value (mode='pval') using -log10(p)
      - abs(selectivity) (mode='selectivity')
    onto the same colormap.

    Parameters
    ----------
    value : float
        p-value (0–1) OR selectivity (-1–1).
    mode : str
        'pval' or 'selectivity'.
    cmap : Colormap
        Matplotlib colormap.
    logp_min, logp_max : float
        Range for -log10(p) normalization (only used for p-values).

    Returns
    -------
    hex_color : str
    """

    if mode == "pval":
        logp = -np.log10(value)
        logp = np.clip(logp, logp_min, logp_max)
        normed = (logp - logp_min) / (logp_max - logp_min)
    elif mode == "selectivity":
        normed = np.clip(np.abs(value), 0, 1)
    elif mode == "selectivity_signed":
        normed = value

    else:
        raise ValueError("mode must be 'pval' or 'selectivity'")

    rgba = cmap(normed)
    return mpl.colors.to_hex(rgba)


def value_to_size(value, mode,
                  logp_min=None, logp_max=None,
                  size_min=30, size_max=150,
                  gamma=0.6):
    """
    Size scaling for p-value (-log10) or abs(selectivity).

    Parameters
    ----------
    mode : 'pval' or 'selectivity'

    Returns
    -------
    size : float
    """

    if mode == "pval":
        logp = -np.log10(value)
        logp = np.clip(logp, logp_min, logp_max)
        norm = (logp - logp_min) / (logp_max - logp_min)

    elif mode == "selectivity":
        norm = np.clip(np.abs(value), 0, 1)
    elif mode == "selectivity_signed":
        norm = np.abs(value) #note: also absolute for size

    else:
        raise ValueError("mode must be 'pval' or 'selectivity'")

    # gamma gives nicer scaling for visibility
    scaled = norm ** gamma
    return size_min + scaled * (size_max - size_min)

def get_metric_value(unit_row, mode, logp_min, logp_max):
    """
    Returns a metric (p-value log-transformed OR abs(selectivity))
    scaled to the same [logp_min, logp_max] range.
    """
    if mode == "pval":
        pval = unit_row['p_value_to_show']
        metric = -np.log10(pval)
        metric = np.clip(metric, logp_min, logp_max)

    elif mode == "selectivity":
        metric = abs(unit_row['selectivity'])  # 0–1
        # map 0–1 range → [logp_min, logp_max]

    else:
        raise ValueError("mode must be 'pval' or 'selectivity'")

    return metric

def plot_roc_atlas(data, params=None, saving_path=None):
    """
    Plot ROC analysis results on the Allen CCF atlas using brainrender.
    :param data: pd.DataFrame containing ROC results and neuron locations.
    :param params:
    :param saving_path:
    :return:
    """
    # ------------------------
    # PLOTTING
    # -----------------------

    # Params
    mode = 'pval'  # 'pval' or 'selectivity'
    logp_min = 1e-5
    logp_max = 15

    cmap = cmr.get_sub_cmap('cmr.ember_r', 0.1,0.56)

    # Create Scene
    title = ''
    scene = Scene(title=title,
                  inset=False,
                  screenshots_folder=saving_path,
                  title_color='darkgrey',
                  atlas_name='allen_mouse_bluebrain_barrels_10um'
                  )

    # Build combined index
    data['ccf_ap'] = data['ccf_ap'].astype(float)
    data['ccf_dv'] = data['ccf_dv'].astype(float)
    data['ccf_ml'] = data['ccf_ml'].astype(float)
    data['mouse_id_neuron_id'] = data['mouse_id'].astype(str) + '_' + data['neuron_id'].astype(str)
    data = data.set_index('mouse_id_neuron_id')

    # ------------- VECTORIZE METRIC SELECTION -----------------
    if mode == "pval":
        vals = data["p_value_to_show"].values
    elif mode == 'selectivity':
        vals = data["selectivity"].values
    elif mode =='selectivity_signed':
        vals = data["selectivity"].values

    # Neuron coordinates
    coords = data[["ccf_ap", "ccf_dv", "ccf_ml"]].to_numpy()

    # Neuron color and sizes
    colors = np.array([value_to_color(v, mode=mode, cmap=cmap, logp_min=logp_min, logp_max=logp_max) for v in vals])
    sizes = np.array([value_to_size(v, mode=mode, logp_min=logp_min, logp_max=logp_max) for v in vals])

    #  Metric to show
    metrics = np.array([get_metric_value(row, 'selectivity', logp_min, logp_max) for _, row in data.iterrows()])

    # Create points
    points_to_add = [
        Point(
            pos=coords[i],
            color=colors[i],
            radius=sizes[i],
            res=100,
            alpha=0.05
        )
        for i in range(len(data))
    ]
    # Add all points at once
    scene.add(*points_to_add)


    # Save figure
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    figure_path = os.path.join(saving_path, f'roc_atlas_{analysis_type}_reward_group_{reward_group}_{mode}.png')
    camera_angled = { "pos": (13425.5, -31640.2, -29477.5),
            "focal_point": (6345.28, 3646.06, -7080.42),
            "viewup": (-0.562399, -0.520597, 0.642406),
            "roll": 159.964,
            "distance": 42389.7,
            "clipping_range": (29001.6, 61298.6),
      }
    scene.render(interactive=False, camera=camera_angled, zoom=1.5)
    scene.screenshot(figure_path, scale=5)
    scene.close()  # dont't forget to close the scene!

    # Save colorbar for reference
    fig, ax = plot_logp_colorbar(cmap, logp_min=logp_min, logp_max=logp_max, label='−log(p)')
    figname = 'logp_colorbar'
    plotting_utils.save_figure_with_options(fig,
                                            ['png','pdf','svg'],
                                            figname,
                                            FIGURE_PATH,
                                            dark_background=False
                                            )

    return


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        args = parser.parse_args()

        print('Plotting ROC results on CCF atlas...')

        # Set paths
        experimenter = 'Axel_Bisi'
        ROOT_PATH_AXEL = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Axel_Bisi', 'NWBFull_bis')
        ROOT_PATH_MYRIAM = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Myriam_Hamon',
                                        'NWBFull_new')
        INFO_PATH = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'z_LSENS', 'Share', 'Axel_Bisi_Share',
                                 'dataset_info')
        DATA_PATH = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', '', experimenter, 'combined_results')
        FIGURE_PATH = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', '', experimenter,
                                   'combined_results', 'roc_atlas')

        # Get mouse information
        # --------------------------
        mouse_info_df = pd.read_excel(os.path.join(INFO_PATH, 'joint_mouse_reference_weight.xlsx'))
        mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
        # Filter for usable mice
        mouse_info_df = mouse_info_df[
            (mouse_info_df['exclude'] == 0) &
            (mouse_info_df['reward_group'].isin(['R+', 'R-'])) &
            (mouse_info_df['recording'] == 1) &
            (mouse_info_df['exclude_ephys'] == 0)
            ]

        # -------------------------------
        # LOAD ROC / NEURON METRICS DATA
        #---------------------------------
        print('Loading ROC data...')
        roc_df = load_roc_results(DATA_PATH)
        roc_df = roc_df.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')

        print('Present mice: \n', roc_df['mouse_id'].unique(), 'Number of mice', roc_df['mouse_id'].nunique(),
              'per reward group',
              roc_df.groupby('reward_group')['mouse_id'].nunique())
        print('ROC analysis types: \n' , roc_df['analysis_type'].unique())


        # -------------------------------
        # LOAD NEURON INFO FROM NWB FILES
        # ------------------------------
        # Collect NWB files
        all_nwb_names = os.listdir(ROOT_PATH_AXEL)
        all_nwb_mice = [name.split('_')[0] for name in all_nwb_names]
        myriam_nwb_names = os.listdir(ROOT_PATH_MYRIAM)
        all_nwb_names.extend(myriam_nwb_names)
        all_nwb_mice.extend([name.split('_')[0] for name in myriam_nwb_names])

        # Filter subjects
        subject_ids = mouse_info_df['mouse_id'].unique()
        subject_ids = [mouse for mouse in subject_ids if any(mouse in name for name in all_nwb_mice)]

        # Exclude specific mice
        excluded_mice = ['AB144']  # MH026, MH015
        subject_ids = [s for s in subject_ids if s not in excluded_mice]
        #subject_ids = ['AB131','AB132']


        # Get list of NWB paths
        nwb_list = [os.path.join(ROOT_PATH_AXEL, name) for name in all_nwb_names if name.startswith('AB')]
        nwb_list.extend([os.path.join(ROOT_PATH_MYRIAM, name) for name in all_nwb_names if name.startswith('MH')])
        nwb_list = [nwb_file for nwb_file in nwb_list if any(mouse in nwb_file for mouse in subject_ids)]

        # Load NWB files and combine data tables
        _, unit_table, ephys_nwb_list = nutils.combine_ephys_nwb(nwb_list)
        print(f'Loaded data from {len(ephys_nwb_list)} NWB files.')

        # test
        test=False
        if test:
          unit_table = NWB_reader_functions.get_unit_table(r"M:\analysis\Axel_Bisi\NWBFull_bis\AB164_20250422_115457.nwb")
          unit_table = nutils.convert_electrode_group_object_to_columns(unit_table)
          unit_table = allen.process_allen_labels(unit_table, subdivide_areas=False)

        # Filter and process units
        unit_table = unit_table[unit_table['bc_label']=='good']
        unit_table = unit_table[~unit_table['ccf_acronym'].isin(allen.get_excluded_areas())]

        # Merge onto roc
        cols_to_keep = ['mouse_id', 'neuron_id', 'bc_label', 'ccf_ap', 'ccf_dv', 'ccf_ml']
        data_df = roc_df.merge(unit_table[cols_to_keep], left_on=['mouse_id', 'neuron_id'], right_on=['mouse_id', 'neuron_id'],
                              how='left')
        # Keep only rows with coordinates
        data_df = data_df.dropna(subset=['ccf_ap', 'ccf_dv', 'ccf_ml'])
        print('Data for plotting:', data_df.columns, len(data_df))

        # ------------
        # PLOT TABLES
        # ------------
        analyses_to_plot = data_df['analysis_type'].unique()

        for reward_group in ['R+','R-']:
            saving_path = os.path.join(FIGURE_PATH, f'reward_group_{reward_group}')
            if not os.path.exists(saving_path):
                os.makedirs(saving_path)

            for analysis_type in analyses_to_plot:
                data_subset = data_df[(data_df['analysis_type']==analysis_type)
                                    & (data_df['significant']==True)
                                 & (data_df['reward_group']==reward_group)]
                data_subset.reset_index(drop=True, inplace=True)

                print(f'  Analysis type: {analysis_type}, reward group: {reward_group}, n neurons: {len(data_subset)}')
                if len(data_subset) == 0:
                    continue

                plot_roc_atlas(data=data_subset, params=None, saving_path=saving_path)


        print('Done.')