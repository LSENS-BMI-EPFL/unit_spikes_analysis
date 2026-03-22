"""
Plot ROC analysis results on the Allen CCF atlas using brainrender. Standalone script.
"""
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
import seaborn as sns

import pandas as pd
import cmasher as cmr

import allen_utils
# Brainrender and settings
import brainrender
import neural_utils
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

def keep_shared_areas(data_df, nomenclature, n_min_units=10, n_min_mice=3):
    print(f'Filtering for R+/R- shared areas in {nomenclature} with at least {n_min_units} units '
          f'and {n_min_mice} mice in each reward group...')

    # Intersect areas across reward groups
    areas_rplus = data_df[data_df['reward_group'] == 'R+'][nomenclature].unique()
    areas_rminus = data_df[data_df['reward_group'] == 'R-'][nomenclature].unique()
    areas_intersect = set(areas_rplus).intersection(set(areas_rminus))
    print(f'Found {len(areas_intersect)} R+/R- shared areas: {areas_intersect}')

    # Count unique elements
    if n_min_units > 0 or n_min_mice > 0:
        bc_mask = data_df['bc_label'].isin(['good', 'mua'])

        # Count unique units per area and reward group
        n_units_rplus = (
            data_df[(data_df['reward_group'] == 'R+') & bc_mask]
            .groupby(nomenclature)['unit_id']
            .nunique()
        )
        n_units_rminus = (
            data_df[(data_df['reward_group'] == 'R-') & bc_mask]
            .groupby(nomenclature)['unit_id']
            .nunique()
        )

        # Count unique mice per area and reward group
        n_mice_rplus = (
            data_df[(data_df['reward_group'] == 'R+') & bc_mask]
            .groupby(nomenclature)['mouse_id']
            .nunique()
        )
        n_mice_rminus = (
            data_df[(data_df['reward_group'] == 'R-') & bc_mask]
            .groupby(nomenclature)['mouse_id']
            .nunique()
        )

        # Filter shared areas based on both unit and mouse counts
        shared_areas = []
        for area in areas_intersect:
            units_ok = (
                (area in n_units_rplus.index and n_units_rplus[area] >= n_min_units)
                and (area in n_units_rminus.index and n_units_rminus[area] >= n_min_units)
            )
            mice_ok = (
                (area in n_mice_rplus.index and n_mice_rplus[area] >= n_min_mice)
                and (area in n_mice_rminus.index and n_mice_rminus[area] >= n_min_mice)
            )
            if units_ok and mice_ok:
                shared_areas.append(area)

        removed_areas = areas_intersect - set(shared_areas)
        if len(removed_areas) > 0:
            print(f'Removed {len(removed_areas)} areas with insufficient counts:')
            for area in removed_areas:
                print(f"  {area}: "
                      f"R+ {n_units_rplus.get(area, 0)}u/{n_mice_rplus.get(area, 0)}m, "
                      f"R- {n_units_rminus.get(area, 0)}u/{n_mice_rminus.get(area, 0)}m")
    else:
        shared_areas = list(areas_intersect)

    print(f'Keeping {len(shared_areas)} shared areas meeting both unit and subject thresholds.')
    if len(shared_areas) > 0:
        print("Shared areas:", shared_areas)

    # Filter dataset
    data_df = data_df[data_df[nomenclature].isin(shared_areas)]
    return data_df, shared_areas

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
    elif mode == "delta_test_corr":
        normed = value
    else:

        raise ValueError("mode entered is not known.")

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

def plot_single_neuron_atlas_old(data, params=None, saving_path=None):
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
    analysis_type = params['analysis_type']
    camera = params['camera']
    cam_key = params['cam_key']
    data_type = params['type']
    if data_type == 'roc':
        mode = 'selectivity'  # 'pval' or 'selectivity'
        logp_min = 1e-5
        logp_max = 15
    elif data_type == 'glm':
        mode = 'delta_test_corr'  # 'pval' or 'selectivity'
        logp_min = 1e-5
        logp_max = 10

    cmap = cmr.get_sub_cmap('cmr.ember_r', 0.1,0.56)
    cmap = cmr.get_sub_cmap('cmr.guppy', 0.05, 0.95)

    # Get plasma colormap from matpltolib
    cmap = plt.get_cmap('plasma_r')

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
        vals = data[mode].values
    elif mode =='selectivity_signed':
        vals = data[mode].values
    elif mode == 'delta_test_corr':
        vals = data[mode].values # get negative values, as gain from red to full model - also helps for plotting

    # Neuron coordinates
    coords = data[["ccf_ap", "ccf_dv", "ccf_ml"]].to_numpy()

    # ------------- LINEAR COLOR AND SIZE MAPPING -----------------
    from matplotlib.colors import Normalize, CenteredNorm

    # Determine value range for normalization
    if mode == 'pval':
        vmin, vmax = logp_min, logp_max
    elif mode in ['selectivity', 'selectivity_signed']:
        vmin, vmax = 0, 1  # Adjust if selectivity has different range
        # Or use actual data range:
        # vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    elif mode == 'delta_test_corr':
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        vmin, vmax = -0.15,0.05
        print("Delta test_corr value range:", vmin, vmax)

    # Create linear normalizer
    if data_type == 'roc':
        norm = Normalize(vmin=vmin, vmax=vmax)
    elif data_type == 'glm':
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Map values to colors using linear normalization
    normalized_vals = norm(vals)
    colors = np.array([cmap(nv) for nv in normalized_vals])
    colors= np.array([mpl.colors.to_hex(c) for c in colors])

    # Linear size mapping
    if data_type == 'roc':
        size_min, size_max = 50, 150  # Adjust these values as needed
    elif data_type == 'glm':
        size_min, size_max = 5, 130  # Adjust these values as needed
        #sizes = size_min + (size_max - size_min) * -normalized_vals
        # Linearly interpolate sizes between size_min and size_max based on min/max of vals
        # Make size proportional to delta_test_corr (larger size for more negative delta)
        sizes = size_min + (size_max - size_min) * (1 - (vals - vmin) / (vmax - vmin))

    #  Metric to show (if needed for other purposes)
    metrics = vals  # Use raw values or normalized values as needed

    # Neuron color and sizes
    #colors = np.array([value_to_color(v, mode=mode, cmap=cmap, logp_min=logp_min, logp_max=logp_max) for v in vals])
    #sizes = np.array([value_to_size(v, mode=mode, logp_min=logp_min, logp_max=logp_max) for v in vals])

    #  Metric to show
    #metrics = np.array([get_metric_value(row, 'selectivity', logp_min, logp_max) for _, row in data.iterrows()])

    # Create points
    points_to_add = [
        Point(
            pos=coords[i],
            color=colors[i],
            radius=sizes[i],
            res=100,
            alpha=0.4
        )
        for i in range(len(data))
    ]
    # Add all points at once
    scene.add(*points_to_add)

    # Save figure
    saving_path = os.path.join(saving_path)
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    figure_name = f'atlas_{data_type}_{analysis_type}_reward_group_{reward_group}_{mode}_{cam_key}'
    figure_path = os.path.join(saving_path, figure_name)
    print('Saving', figure_path)

    scene.render(interactive=False, camera=camera, zoom=1.5)
    scene.screenshot(figure_path, scale=2)
    scene.close()  # don't forget to close the scene!

    if data_type == 'roc':
        # Save colorbar for reference
        fig, ax = plot_logp_colorbar(cmap, logp_min=logp_min, logp_max=logp_max, label='−log(p)')
        figname = 'logp_colorbar'
        plotting_utils.save_figure_with_options(fig,
                                                ['png','pdf','svg'],
                                                figname,
                                                figure_path,
                                                dark_background=False
                                                )
    elif data_type == 'glm':
        # Plot raw colorbar for delta_test_corr
        fig, ax = plt.subplots(figsize=(5,1), dpi=500)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
        cb.set_label(r'Test $\Delta$r = r$_\text{red}$-r$_\text{full}$', fontsize=15)
        cb.ax.tick_params(labelsize=15)
        plt.tight_layout()
        figname = 'delta_test_corr_colorbar'
        plotting_utils.save_figure_with_options(fig,
                                                ['png','pdf','svg'],
                                                figname,
                                                saving_path,
                                                dark_background=False
                                                )


    return


def plot_single_neuron_atlas(data, params=None, saving_path=None):
    """
    Plot ROC/GLM analysis results on the Allen CCF atlas using brainrender.
    Optimized for speed.
    """
    # Params
    analysis_type = params['analysis_type']
    camera = params['camera']
    cam_key = params['cam_key']
    data_type = params['type']

    # Configuration
    if data_type == 'roc':
        mode = 'selectivity'
        vmin, vmax = 0, 1
        size_min, size_max = 50, 150
        cmap = plt.get_cmap('plasma_r')
    elif data_type == 'glm':
        mode = 'delta_test_corr'
        vmin, vmax = -0.05, 0.01
        size_min, size_max = 10, 120
        cmap = plt.get_cmap('plasma_r')

    # Create Scene
    os.makedirs(saving_path, exist_ok=True)
    scene = Scene(
        title='',
        inset=False,
        screenshots_folder=saving_path,
        title_color='darkgrey',
        atlas_name='allen_mouse_bluebrain_barrels_10um'
    )

    # Vectorized coordinate extraction (no copy needed)
    coords = data[["ccf_ap", "ccf_dv", "ccf_ml"]].to_numpy(dtype=np.float32)
    vals = data[mode].to_numpy(dtype=np.float32)

    # Vectorized normalization
    normalized_vals = np.clip((vals - vmin) / (vmax - vmin), 0, 1)

    # Vectorized color mapping (avoid list comprehension)
    colors_rgba = cmap(normalized_vals)
    colors_hex = np.empty(len(data), dtype=object)
    for i in range(len(data)):
        colors_hex[i] = mpl.colors.to_hex(colors_rgba[i])

    # Vectorized size mapping
    if data_type == 'roc':
        sizes = size_min + (size_max - size_min) * normalized_vals
    else:  # glm - invert for delta_test_corr
        sizes = size_min + (size_max - size_min) * (1 - normalized_vals)

    # Batch create points (faster than list comprehension)
    points_to_add = []
    for i in range(len(data)):
        points_to_add.append(
            Point(
                pos=coords[i],
                color=colors_hex[i],
                radius=float(sizes[i]),
                res=100,
                alpha=0.4
            )
        )

    # Add all points at once
    scene.add(*points_to_add)

    # Render and save
    figure_name = f'atlas_{data_type}_{analysis_type}_reward_group_{reward_group}_{mode}_{cam_key}.png'
    figure_path = os.path.join(saving_path, figure_name)
    print(f'Saving {figure_path}')

    scene.render(interactive=False, camera=camera, zoom=1.5)
    scene.screenshot(figure_path, scale=5)
    scene.close()

    # Save colorbar
    fig, ax = plt.subplots(figsize=(5, 1), dpi=500)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')

    if data_type == 'roc':
        cb.set_label('Selectivity', fontsize=15)
        figname = 'selectivity_colorbar'
    else:
        cb.set_label(r'Test $\Delta$r = r$_\text{red}$-r$_\text{full}$', fontsize=15)
        figname = 'delta_test_corr_colorbar'

    cb.ax.tick_params(labelsize=15)
    plt.tight_layout()

    plotting_utils.save_figure_with_options(fig,
                                            ['png', 'pdf', 'svg'],
                                            figname,
                                            saving_path,
                                            dark_background=False
                                            )
    plt.close(fig)

    return


def plot_significant_fraction_heatmap(data_df, per_mouse, saving_path):
    """
    Plot fraction of significant units across areas and models.
    """
    # Filter out full model
    data_df = data_df[data_df['model_name'] != 'full'].copy()

    if per_mouse is None:
        per_mouse = False
    if per_mouse:
        # Compute fraction significant per mouse/model/area
        frac_per_mouse = data_df.groupby(['mouse_id', 'reward_group', 'model_name', 'area_acronym_custom']).agg(
            frac_significant=('significant', 'mean'),
            n_units=('significant', 'size')
        ).reset_index()

        # Average across mice
        frac_sig = frac_per_mouse.groupby(['reward_group', 'model_name', 'area_acronym_custom']).agg(
            frac_significant=('frac_significant', 'mean'),
            sem_frac=('frac_significant', 'sem'),
            n_mice=('mouse_id', 'nunique')
        ).reset_index()
    else:
        # Compute fraction significant per area/model/reward_group across all mice
        frac_sig = data_df.groupby(['reward_group', 'model_name', 'area_acronym_custom']).agg(
            frac_significant=('significant', 'mean'),
            n_units=('significant', 'size')
        ).reset_index()
        frac_per_mouse = None  # Not used in this case

    # Plot separately for each reward group
    for reward_group in ['R+', 'R-']:
        subset = frac_sig[frac_sig['reward_group'] == reward_group]

        # Pivot for heatmap
        heatmap_data = subset.pivot(index='model_name', columns='area_acronym_custom', values='frac_significant')
        heatmap_data = heatmap_data.apply(pd.to_numeric, errors='coerce')

        # Make dict to rename variables - choose which to include
        git_version = data_df['git_version'].iloc[0]
        if git_version == 'f849441':
            model_name_dict = {
                'auditory_encoding': 'Auditory stimulus',
                'whisker_encoding': 'Whisker stimulus',
                'jaw_onset_encoding': 'Lick initiation',
                'motor_encoding': 'Orofacial motion',
                #'last_reward': 'Prev. trial rewarded',
                'last_whisker_reward': 'Prev. whisker trial rewarded',
                #'last_false_alarm': 'Prev. trial false alarm',
                'prev_success': 'Previous trial success',
                'block_perf_type': 'High/low performance',
                #'prop_past_whisker_reward':'Prop. past whisker rewards',
                #'session_progress_encoding':'Trial index',
                'sum_rewards':'Cumulative rewards',
                'whisker_reward_rate_5':'Perf. last 5 whisker trials',
            }
        elif git_version== '1b14083':
            model_name_dict = {
                'auditory_encoding': 'Auditory stimulus',
                'whisker_encoding': 'Whisker stimulus',
                'jaw_onset_encoding': 'Lick initiation',
                'reward_encoding': 'Reward time',
                'motor_encoding': 'Orofacial motion',
                'pupil_area': 'Pupil area',
                'time_since_whisker_reward': 'Whisker reward recency',
                'block_perf_type': 'High/low performance',
                'session_progress_encoding': 'Trial index',
            }

        # Rename rows
        heatmap_data = heatmap_data.rename(index=model_name_dict)
        # Order like in dict
        heatmap_data = heatmap_data.reindex(model_name_dict.values())

        # Order areas using allen_utils function
        area_order = allen_utils.get_custom_area_order()
        areas_present = [area for area in area_order if area in heatmap_data.columns]
        print('Heatmap areas', areas_present)
        heatmap_data = heatmap_data[areas_present]

        # Plot
        fig, ax = plt.subplots(figsize=(24, 6), dpi=500)
        sns.heatmap(heatmap_data,
                    ax=ax,
                    annot=True,
                    fmt='.2f',
                    cmap='PuRd',
                    vmin=0,
                    vmax=0.8,
                    cbar_kws={'label': 'Fraction significant units', 'shrink': 0.5, 'pad': 0.02, 'aspect':20*0.5}, #default aspect is 20
                    linewidths=0,
                    )
        # Update colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('Fraction significant', fontsize=15)

        ax.xaxis.tick_top()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('Encoding variables', fontsize=15)

        # Format y tick labels by removing underscores
        ytick_labels = [label.get_text().replace('_', ' ') for label in ax.get_yticklabels()]
        ax.set_yticklabels(ytick_labels, rotation=0, fontsize=12)

        plt.tight_layout()

        # Save
        if per_mouse:
            figname = f'glm_lrt_significant_fraction_heatmap_{reward_group}_per_mouse'
        else:
            figname = f'glm_lrt_significant_fraction_heatmap_{reward_group}'
        plotting_utils.save_figure_with_options(fig, ['png', 'pdf', 'svg'],
                                                figname,
                                                saving_path,
                                                dark_background=False)

    return frac_per_mouse, frac_sig


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        args = parser.parse_args()

        plot_roc = False
        plot_glm = True

        print('Plotting single-neuron analysis results on CCF atlas...')

        # Set paths
        experimenter = 'Axel_Bisi'
        ROOT_PATH_AXEL = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'NWBFull_bis')
        ROOT_PATH_MYRIAM = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Myriam_Hamon',
                                        'NWBFull_new')
        INFO_PATH = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'z_LSENS', 'Share', 'Axel_Bisi_Share',
                                 'dataset_info')
        if plot_roc:
            DATA_PATH = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'combined_results')
        elif plot_glm:
            DATA_PATH = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Myriam_Hamon', 'combined_results')

        # Make output folder
        if plot_roc:
            folder_name = 'roc_atlas_cmap'
        else:
            folder_name = 'glm_atlas_cmap'
        FIGURE_PATH = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter,
                                   'combined_results', folder_name)

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
        #subject_ids = ['AB131', 'AB132', 'AB133', 'AB134', 'AB135', 'AB136', 'AB137', 'AB138', 'AB139', 'AB140',]


        # Get list of NWB paths
        nwb_list = [os.path.join(ROOT_PATH_AXEL, name) for name in all_nwb_names if name.startswith('AB')]
        nwb_list.extend([os.path.join(ROOT_PATH_MYRIAM, name) for name in all_nwb_names if name.startswith('MH')])
        nwb_list = [nwb_file for nwb_file in nwb_list if any(mouse in nwb_file for mouse in subject_ids)]

        # Load NWB files and combine data tables
        _, unit_table, ephys_nwb_list = nutils.combine_ephys_nwb(nwb_list)
        print(f'Loaded data from {len(ephys_nwb_list)} NWB files.')

        # -------------------------------
        # LOAD NEURON METRICS DATA
        #---------------------------------
        if plot_roc:
            print('Loading ROC data...')
            roc_df = load_roc_results(DATA_PATH)
            roc_df = roc_df.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')

            print('Present mice: \n', roc_df['mouse_id'].unique(), 'Number of mice', roc_df['mouse_id'].nunique(),
                  'per reward group',
                  roc_df.groupby('reward_group')['mouse_id'].nunique())
            print('ROC analysis types: \n' , roc_df['analysis_type'].unique())

        elif plot_glm:

            git_version = 'f849441'
            git_version = '1b14083'
            glm_dfs = []
            for mouse_id in subject_ids:
                mouse_results_path = os.path.join(DATA_PATH, mouse_id, 'whisker_0', 'unit_glm', git_version)
                fpath = os.path.join(mouse_results_path, f"summary_{mouse_id}_unit_glm_{git_version}.parquet")
                if not os.path.exists(fpath):
                    print(f"[WARNING] GLM summary not found: {fpath}")
                    continue
                df = pd.read_parquet(fpath)
                glm_dfs.append(df)

            glm_df = pd.concat(glm_dfs, ignore_index=True)

            # Create new column for significance: test_corr > 0.1 and LRT significant
            glm_df['significant'] = (glm_df['test_corr'] > 0.2) & (glm_df['lrt_significant'])

            #glm_df = glm_df.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')
            print('Present mice: \n', glm_df['mouse_id'].unique(), 'Number of mice', glm_df['mouse_id'].nunique(),)
            print('GLM model names: \n' , glm_df['model_name'].unique())


        else:
            raise ValueError('Either plot_roc or plot_glm must be True.')


        # debug
        debug=False
        if debug:
          unit_table = NWB_reader_functions.get_unit_table(r"M:\analysis\Axel_Bisi\NWBFull_bis\AB164_20250422_115457.nwb")
          unit_table = nutils.convert_electrode_group_object_to_columns(unit_table)
          unit_table = allen.process_allen_labels(unit_table, subdivide_areas=False)

        # Filter and process units
        # -------------------------------
        unit_table = unit_table[unit_table['bc_label']=='good']
        unit_table = unit_table[~unit_table['ccf_acronym'].isin(allen.get_excluded_areas())]
        #unit_table = nutils.convert_electrode_group_object_to_columns(unit_table)
        unit_table = allen.process_allen_labels(unit_table, subdivide_areas=True)

        # Merge unit table onto metric table
        cols_to_keep = ['mouse_id', 'neuron_id', 'unit_id', 'bc_label', 'ccf_ap', 'ccf_dv', 'ccf_ml', 'area_acronym_custom']
        if plot_roc:
            roc_df = roc_df.drop(columns=['area_acronym_custom'], errors='ignore')
            data_df = roc_df.merge(unit_table[cols_to_keep], left_on=['mouse_id', 'neuron_id'], right_on=['mouse_id', 'neuron_id'],
                                  how='left')
        elif plot_glm:
            glm_df = glm_df.drop(columns=['area_acronym_custom'], errors='ignore')
            data_df = glm_df.merge(unit_table[cols_to_keep], left_on=['mouse_id', 'neuron_id'], right_on=['mouse_id', 'neuron_id'],
                                  how='left')
        data_df['reward_group'] = data_df['reward_group'].map({1: 'R+', 0: 'R-'})

        # Keep only rows with coordinates, but that should be all
        data_df = data_df.dropna(subset=['ccf_ap', 'ccf_dv', 'ccf_ml'])
        print('Data for plotting:', data_df.columns, len(data_df))

        # -------------------------------
        # PLOT FRACTION SIGNIFICANT HEATMAP
        #---------------------------------

        # After merging data_df with unit_table and before plotting atlas
        data_df, _ = keep_shared_areas(data_df, nomenclature='area_acronym_custom', n_min_units=15, n_min_mice=2)
        sig_frac_path = os.path.join(FIGURE_PATH, 'significant_fraction_heatmaps', git_version)
        if not os.path.exists(sig_frac_path):
            os.makedirs(sig_frac_path)
            
        plot_fractions = False
        if plot_fractions:

            for per_mouse in [False, True]:
                frac_per_mouse, frac_sig = plot_significant_fraction_heatmap(data_df, per_mouse, sig_frac_path)

                # Save if not None
                if frac_per_mouse is not None:
                    frac_per_mouse.to_csv(os.path.join(sig_frac_path, 'glm_significant_fraction_per_mouse.csv'), index=False)
                if frac_sig is not None:
                    frac_sig.to_csv(os.path.join(sig_frac_path, 'glm_significant_fraction_global.csv'), index=False)

        # ----------------------------
        # FOCALITY INDICES
        # ----------------------------

        from focality_analysis_simple import analyze_focality, analyze_focality_with_bca, analyze_focality_neurons
        focality_path = os.path.join(FIGURE_PATH, 'focality_analysis', git_version)
        if not os.path.exists(focality_path):
            os.makedirs(focality_path)

        run_focality=False
        plot_focality=True
        if run_focality:
            #results_df = analyze_focality(data_df=data_df, saving_path=focality_path)
            results_df = analyze_focality_neurons(data_df=data_df, saving_path=focality_path) # function saves results
            #foc_results_df = analyze_focality_with_bca(data_df=data_df, saving_path=focality_path)
        if plot_focality:

            # Load focality results
            results_df = pd.read_csv(os.path.join(focality_path, 'focality_results_neurons.csv'))
            exclude_model = ['time_since_whisker_reward']

            n_unique_areas = data_df['area_acronym_custom'].nunique()
            print('Number of unique areas:', n_unique_areas)
            uniform_level = 1 / n_unique_areas

            # if time_since_whisker_reward in git version, exclude it, decide to plot with and without
            if 'time_since_whisker_reward' in results_df['model_name'].values:
                # Then iterate and do one version with, one without
                for exclude_whisker_reward in [True, False]:
                    # FIX: Define results_sub_df for both cases
                    if exclude_whisker_reward:
                        results_sub_df = results_df[~results_df['model_name'].isin(exclude_model)].copy()
                    else:
                        results_sub_df = results_df.copy()

                    fig, ax = plt.subplots(figsize=(6, 4), dpi=400)
                    plotting_utils.remove_top_right_frame(ax)
                    model_name_dict = {
                        'auditory_encoding': 'Auditory stimulus',
                        'whisker_encoding': 'Whisker stimulus',
                        'jaw_onset_encoding': 'Lick initiation',
                        'reward_encoding': 'Reward time',
                        'motor_encoding': 'Orofacial motion',
                        'pupil_area': 'Pupil area',
                        'time_since_whisker_reward': 'Whisker reward recency',
                        'block_perf_type': 'High/low performance',
                        'session_progress_encoding': 'Trial index',
                    }
                    order = list(model_name_dict.keys())
                    if exclude_whisker_reward:
                        order.remove('time_since_whisker_reward')

                    # Reorder dataframe based on order
                    results_sub_df['model_name'] = pd.Categorical(results_sub_df['model_name'], categories=order,
                                                                  ordered=True)
                    results_sub_df = results_sub_df.sort_values('model_name')
                    results_sub_df = results_sub_df.set_index('model_name').loc[order].reset_index()

                    # Update column name: focality_mean -> bootstrap_mean
                    sns.pointplot(data=results_sub_df,
                                  x='model_name',
                                  order=order,
                                  y='bootstrap_mean',
                                  hue='reward_group',
                                  hue_order=['R+', 'R-'],
                                  dodge=0.3,
                                  join=False,
                                  capsize=0.1,
                                  errwidth=1.5,
                                  palette=['forestgreen', 'crimson'],
                                  legend=False,
                                  ax=ax)

                    # Add errorbars using ci_lower and ci_upper
                    for model in order:
                        print(model)
                        # FIX: Handle both reward groups
                        for reward_group in ['R+', 'R-']:
                            row = results_sub_df[(results_sub_df['model_name'] == model) &
                                                 (results_sub_df['reward_group'] == reward_group)]
                            if len(row) == 0:
                                continue
                            row = row.iloc[0]

                            x = order.index(model)
                            if row['reward_group'] == 'R+':
                                x -= 0.15
                            else:
                                x += 0.15

                            # Calculate error bar lengths
                            lower_err = row['bootstrap_mean'] - row['ci_lower']
                            upper_err = row['ci_upper'] - row['bootstrap_mean']

                            yerr = [[lower_err], [upper_err]]
                            color = 'forestgreen' if row['reward_group'] == 'R+' else 'crimson'
                            ax.errorbar(x, row['bootstrap_mean'],
                                        yerr=yerr,
                                        fmt='none', c=color, capsize=5, elinewidth=1.5, zorder=-1)

                    ax.axhline(uniform_level, color='k', linestyle='--', label='Uniform distribution')
                    ax.set_ylabel('Focality index')
                    ax.set_xlabel('Encoding variable')

                    # Rename x labels
                    git_version = data_df['git_version'].iloc[0]
                    if git_version == 'f849441':
                        model_name_dict = {
                            'auditory_encoding': 'Auditory stimulus',
                            'whisker_encoding': 'Whisker stimulus',
                            'jaw_onset_encoding': 'Lick initiation',
                            'motor_encoding': 'Orofacial motion',
                            'last_whisker_reward': 'Prev. whisker trial rewarded',
                            'prev_success': 'Previous trial success',
                            'block_perf_type': 'High/low performance',
                            'sum_rewards': 'Cumulative rewards',
                            'whisker_reward_rate_5': 'Perf. last 5 whisker trials',
                        }
                    elif git_version == '1b14083':
                        if exclude_whisker_reward:
                            model_name_dict = {
                                'auditory_encoding': 'Auditory stimulus',
                                'whisker_encoding': 'Whisker stimulus',
                                'jaw_onset_encoding': 'Lick initiation',
                                'motor_encoding': 'Orofacial motion',
                                'pupil_area': 'Pupil area',
                                'reward_encoding': 'Reward time',
                                'block_perf_type': 'High/low performance',
                                'session_progress_encoding': 'Trial index',
                            }
                        else:
                            model_name_dict = {
                                'auditory_encoding': 'Auditory stimulus',
                                'whisker_encoding': 'Whisker stimulus',
                                'jaw_onset_encoding': 'Lick initiation',
                                'reward_encoding': 'Reward time',
                                'motor_encoding': 'Orofacial motion',
                                'pupil_area': 'Pupil area',
                                'time_since_whisker_reward': 'Whisker reward recency',
                                'block_perf_type': 'High/low performance',
                                'session_progress_encoding': 'Trial index',
                            }

                    order_labels = [model_name_dict.get(model, model) for model in order]
                    ax.set_xticklabels(order_labels, rotation=30, ha='right')
                    plt.tight_layout()

                    # Save
                    if exclude_whisker_reward:
                        figname = 'glm_focality_index_pointplot_exclude_whisker_reward'
                    else:
                        figname = 'glm_focality_index_pointplot'
                    fig_path = os.path.join(focality_path, figname)
                    plotting_utils.save_figure_with_options(fig,
                                                            ['png', 'pdf', 'svg'],
                                                            figname,
                                                            focality_path,
                                                            dark_background=False)
            else:
                pass

        # --------------------------------------
        # CHECK WHICH REGION TRULY ENCODE VARIABLE #TODO: improve this, not great
        # ---------------------------------------
        region_encoding_path = os.path.join(FIGURE_PATH, 'region_encoding_analysis')
        if not os.path.exists(region_encoding_path):
            os.makedirs(region_encoding_path)
        from regional_encoding_analysis import (
            analyze_regional_encoding_complete,
            save_regional_encoding_results
        )

        run_region_stat = False
        if run_region_stat:
            # Run complete analysis
            results = analyze_regional_encoding_complete(
                data_df,
                area_column='area_acronym_custom',
                baseline=0.05,
                n_bootstrap=100,
                n_permutations=100,
                min_mice_per_region=2,
                progress_bar=True
            )

            # Save all results
            save_regional_encoding_results(results, saving_path=region_encoding_path)

            # Access specific results
            whisker_r_plus = results[
                (results['model_name'] == 'whisker_encoding') &
                (results['reward_group'] == 'R+') &
                (results['encodes'] == True)
                ]

            print(whisker_r_plus[['region', 'observed_proportion', 'p_fdr', 'z_score']])



        # -------------------------------------------------
        # PLOT DISTRIBUTION OF ALL UNITS TEST SCORE ON TOP OF DISTRIBUTION OF SIGNIFICANT UNITS FOR WHISKER ENCODING
        #--------------------------------------------------

        plot_dist=True
        if plot_dist:
            fig, ax = plt.subplots(figsize=(4,4), dpi=400)
            data_subset = data_df[data_df['model_name']=='whisker_encoding']
            #Histgoram of neuron counts per test_corr bin
            bins = np.linspace(-0.5, 1.0, 150)
            sns.histplot(data=data_subset, x='test_corr', bins=bins, ax=ax, color='k')
            sns.histplot(data=data_subset[data_subset['significant']==True], x='test_corr', bins=bins, ax=ax, color='purple')
            ax.set_xlabel('Test correlation')
            ax.set_ylabel('Neurons')
            # Save
            figname = 'glm_whisker_encoding_test_corr_distribution_with_significant'
            test_dist_path = os.path.join(FIGURE_PATH, 'test_corr_dist', git_version)
            if not os.path.exists(test_dist_path):
                os.makedirs(test_dist_path)
            plotting_utils.save_figure_with_options(fig,
                                                    ['png','pdf','svg'],
                                                    figname,
                                                    test_dist_path,
                                                    dark_background=False
                                                    )



        # ------------
        # PLOT TABLES
        # ------------
        if plot_roc:
            analyses_to_plot = data_df['analysis_type'].unique()

        elif plot_glm:
            # For GLM, compute delta test_corr between full and all other combinations
            data_df['delta_test_corr'] = data_df.groupby(['mouse_id', 'neuron_id'])['test_corr'].transform(
                lambda x: x - x[data_df['model_name'] == 'full'].values[0]
            )
            # For full model, set delta as test_corr
            data_df.loc[data_df['model_name'] == 'full', 'delta_test_corr'] = data_df.loc[
                data_df['model_name'] == 'full', 'test_corr']
            analyses_to_plot = data_df['model_name'].unique()

            #analyses_to_plot = ['last_whisker_reward',
            #                    'prev_success', 'block_perf_type', 'whisker_reward_rate_5'
            #                    ]

            plot_dist=True
            # For testing, show distributions of test_corr and delta_test_corr
            if plot_dist:
                fig, ax = plt.subplots(figsize=(6,4), dpi=200)
                sns.histplot(data=data_df, x='test_corr', ax=ax, bins=30, kde=True)
                ax.set_title('GLM test_corr distribution')
                # Save figure
                figname = 'glm_test_corr_distribution'
                plotting_utils.save_figure_with_options(fig,
                                                        ['png'],
                                                        figname,
                                                        test_dist_path,
                                                        dark_background=False)



                fig, ax = plt.subplots(figsize=(6,4), dpi=200)
                data_df_sub = data_df[data_df['model_name'] != 'full']
                sns.histplot(data=data_df_sub, x='delta_test_corr', ax=ax, bins=30, kde=True)
                ax.set_title('GLM delta_test_corr distribution')
                # Save figure
                figname = 'glm_delta_test_corr_distribution'
                plotting_utils.save_figure_with_options(fig,
                                                        ['png'],
                                                        figname,
                                                        test_dist_path,
                                                        dark_background=False)


        # Define cameras
        camera_angled = dict(
            pos=(13425.5, -31640.2, -29477.5),
            focal_point=(6345.28, 3646.06, -7080.42),
            viewup=(-0.562399, -0.520597, 0.642406),
            roll=159.964,
            distance=42389.7,
            clipping_range=(29001.6, 61298.6),
        )
        camera_sagittal = dict(
            pos=(6310.00, -8498.56, -59302.7),
            focal_point=(6227.12, 3249.23, -7068.25),
            viewup=(-0.0969644, -0.971065, 0.218244),
            roll=174.456,
            distance=53539.3,
            clipping_range=(42723.5, 70624.2),
        )
        cam_top = dict(
            pos=(26926.4, -43265.8, -4528.68),
            focal_point=(6345.26, 3646.14, -7080.37),
            viewup=(-0.915902, -0.400002, 0.0334856),
            roll=-171.609,
            distance=51291.5,
            clipping_range=(38246.7, 67879.1),
        )
        cameras = {'angled': camera_angled,
                   'sagittal': camera_sagittal,
                   'top': cam_top
                   }
        params = {}

        for cam_key, camera in cameras.items():
            #for reward_group in ['R+','R-']:
            for reward_group in ['R-']:
                saving_path = os.path.join(FIGURE_PATH, git_version, f'reward_group_{reward_group}', cam_key)
                if not os.path.exists(saving_path):
                    os.makedirs(saving_path)

                for analysis_type in analyses_to_plot:
                    analysis_col  = 'analysis_type' if plot_roc else 'model_name'

                    data_subset = data_df[(data_df[analysis_col]==analysis_type)
                                        #& (data_df['significant']==True)
                                     & (data_df['reward_group']==reward_group)
                    #& (data_df['area_acronym_custom'].str.contains('MO-'))
                    ]
                    print('Unique areas', data_subset['area_acronym_custom'].unique())
                    data_subset.reset_index(drop=True, inplace=True)


                    print(f'  Analysis type: {analysis_type}, reward group: {reward_group}, n neurons: {len(data_subset)}')
                    if len(data_subset) == 0:
                        continue

                    params.update({'type': 'roc' if plot_roc else 'glm',
                                      'analysis_type': analysis_type,
                                      'reward_group': reward_group,
                                      'cam_key': cam_key,
                                   'camera': camera
                                      })
                    plot_single_neuron_atlas(data=data_subset, params=params, saving_path=saving_path)


        print('Done.')