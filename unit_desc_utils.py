#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_spikes_analysis
@file: unit_desc_utils.py
@time: 4/13/2025 8:37 PM
"""


# Imports
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import plotly.graph_objects as go
from itertools import combinations, combinations_with_replacement

# Import custom modules
import neural_utils as nutils
import NWB_reader_functions as nwb_reader
import allen_utils as allen_utils
from matplotlib.font_manager import font_scalings

import plotting_utils
from plotting_utils import save_figure_with_options, adjust_lightness

def unit_label_describe(unit_data, output_path):
    """
    Describe unit labels conversion from Kilosort to Bombcell.
    :param unit_data: unit table
    :param output_path: path to save the output
    :return:
    """

    # Load neural data
    #_, unit_data, _ = nutils.combine_ephys_nwb(nwb_files, max_workers=24)

    # Load data
    #unit_data = []
    #for nwb_file in nwb_files:
    #    try:
    #        unit_table = nwb_reader.get_unit_table(nwb_file)
    #        mouse_id = nwb_reader.get_mouse_id(nwb_file)
    #        beh, day = nwb_reader.get_bhv_type_and_training_day_index(nwb_file)
    #        unit_table['mouse_id'] = mouse_id
    #        unit_data.append(unit_table)
    #    except:
    #        continue
    #unit_data = pd.concat(unit_data)

    print('Number of good units after Kilosort', len(unit_data[unit_data['ks_label'] == 'good']))
    print('Number of good units after Bombcell', len(unit_data[unit_data['bc_label'] == 'good']))

    # Order of categories
    ks_order = ['good', 'mua']
    bc_order = ['good', 'mua', 'non-soma', 'noise']

    # Create internal node labels to force vertical, non-circular flow
    ks_nodes = [f"{label}_ks" for label in ks_order]
    bc_nodes = [f"{label}_bc" for label in bc_order]
    all_nodes = ks_nodes + bc_nodes

    # Mapping from actual label to internal node name
    node_map = {label: f"{label}_ks" for label in ks_order}
    node_map.update({label: f"{label}_bc" for label in bc_order})

    # Index map for plotly
    index_map = {name: i for i, name in enumerate(all_nodes)}

    # Count links
    grouped = unit_data.groupby(['ks_label', 'bc_label']).size().reset_index(name='weight')


    # Sankey link values
    sources = grouped['ks_label'].map(lambda x: index_map[f"{x}_ks"])
    targets = grouped['bc_label'].map(lambda x: index_map[f"{x}_bc"])
    weights = grouped['weight']

    # Custom colors
    color_dict = {
        'good': 'mediumvioletred',
        'mua': 'orchid',
        'non-soma': 'slateblue',
        'noise': 'dimgrey', # Note: NOISE neurons not included in NWB files
    }
    node_colors = [color_dict[label.replace('_ks', '').replace('_bc', '')] for label in all_nodes]
    link_colors = [color_dict[label] for label in grouped['bc_label']]

    # Clean labels for display
    visible_labels = [label.replace('_ks', '').replace('_bc', '') for label in all_nodes]

    # Create the Sankey plot
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",  # helps enforce the left-to-right layout
        node=dict(
            pad=30,
            thickness=20,
            line=dict(color='black', width=0.7),
            label=visible_labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=weights,
            color=link_colors

        )
    )])

    #fig.update_layout(title_text="Neuron label classification: ks_label → bc_label", font_size=14)
    #fig.show()

    # Save figure
    file_name = 'unit_label_sankey_diagram'
    file_formats = ['png', 'pdf', 'eps']
    save_path = os.path.join(output_path, file_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save plotly Figure object
    for format in file_formats:
        fig.write_image(os.path.join(save_path, f"{file_name}.{format}"), engine='orca', scale=5)

    return

def unit_anat_describe(unit_data, output_path):
    """
    Describe unit area change before after alignment of histology to ephys data.
    :param unit_data: unit table
    :param output_path: path to save the output
    :return:
    """
    # Load neural data
    #_, unit_data, _ = nutils.combine_ephys_nwb(nwb_files, max_workers=24)

    # Assign neurons to broad categories based on ccf_acronym
    def categorize_area(acronym):
        exclude_areas = allen_utils.get_excluded_areas()
        if acronym in exclude_areas:
            return 'white matter / ventricular system'
        elif acronym in ['root', 'void']:
            return 'outside brain'
        else:
            return 'grey matter'
    unit_data['pre_align_category'] = unit_data['ccf_acronym'].apply(categorize_area)
    unit_data['post_align_category'] = unit_data['ccf_atlas_acronym'].apply(categorize_area)

    unit_data_good = unit_data[unit_data['bc_label'] == 'good']
    unit_data_good_mua = unit_data[unit_data['bc_label'].isin(['good', 'mua'])]
    # Compare number of good units before and after ephys-atlas alignment
    print('Number of good units before ephys-atlas alignment:', len(unit_data_good[unit_data_good['pre_align_category'] == 'grey matter']))
    print('Number of good units after ephys-atlas alignment:', len(unit_data_good[unit_data_good['post_align_category'] == 'grey matter']))
    print('Number of good/mua units before ephys-atlas alignment:', len(unit_data_good_mua[unit_data_good_mua['pre_align_category'] == 'grey matter']))
    print('Number of good/mua units after ephys-atlas alignment:', len(unit_data_good_mua[unit_data_good_mua['post_align_category'] == 'grey matter']))


    # Order of categories
    pre_align_order= ['grey matter', 'white matter / ventricular system', 'outside brain']
    post_align_order = ['grey matter', 'white matter / ventricular system', 'outside brain']

    # Create internal node labels to force vertical, non-circular flow
    pre_nodes = [f"{label}_pre" for label in pre_align_order]
    post_nodes = [f"{label}_post" for label in post_align_order]
    all_nodes = pre_nodes + post_nodes

    # Mapping from actual label to internal node name
    node_map = {label: f"{label}_pre" for label in pre_align_order}
    node_map.update({label: f"{label}_post" for label in post_align_order})

    # Index map for plotly
    index_map = {name: i for i, name in enumerate(all_nodes)}

    # Count links
    grouped = unit_data.groupby(['pre_align_category', 'post_align_category']).size().reset_index(name='weight')

    # Sankey link values
    sources = grouped['pre_align_category'].map(lambda x: index_map[f"{x}_pre"])
    targets = grouped['post_align_category'].map(lambda x: index_map[f"{x}_post"])
    weights = grouped['weight']

    # Custom colors
    color_dict = {
        'grey matter': 'mediumvioletred',
        'white matter / ventricular system': 'dimgrey',
        'outside brain': 'silver',
    }
    node_colors = [color_dict[label.replace('_pre', '').replace('_post', '')] for label in all_nodes]
    link_colors = [color_dict[label] for label in grouped['post_align_category']]

    # Clean labels for display
    visible_labels = [label.replace('_pre', '').replace('_post', '') for label in all_nodes]

    # Create the Sankey plot
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",  # helps enforce the left-to-right layout
        node=dict(
            pad=30,
            thickness=20,
            line=dict(color='black', width=0.7),
            label=visible_labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=weights,
            color=link_colors

        ),

    )])
    fig.update_layout(
        font_color="black",
        font_size=20,
    )
    #fig.update_layout(title_text="Neuron label classification: ks_label → bc_label", font_size=14)
    fig.show()

    # Save figure
    file_name = 'unit_anat_sankey_diagram'
    file_formats = ['png', 'pdf', 'eps']
    save_path = os.path.join(output_path, file_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save plotly Figure object
    for format in file_formats:
        fig.write_image(os.path.join(save_path, f"{file_name}.{format}"), engine='orca', scale=5)

    return


from itertools import combinations

def plot_number_area_pairs_heatmap(trial_table, unit_table, output_path):

    # Load neural data
    #trial_table, unit_table, _ = nutils.combine_ephys_nwb(nwb_files, max_workers=24)
    unit_table = allen_utils.create_area_custom_column(unit_table)
    unit_table = unit_table[~unit_table['area_acronym_custom'].isin(allen_utils.get_excluded_areas())]

    # Add reward group info from trial table onto mouse_id
    mouse_reward_group = trial_table[['mouse_id', 'reward_group']].drop_duplicates()
    unit_table = unit_table.merge(mouse_reward_group, on='mouse_id', how='left')


    MIN_NEURONS = 30


    # Count neurons per area per mouse
    counts = unit_table.groupby(['mouse_id', 'reward_group','area_acronym_custom']).size().reset_index(name='n_neurons')
    area_order_count = counts.groupby('area_acronym_custom')['n_neurons'].sum().sort_values(ascending=True).index
    # Keep only areas with at least min_neurons for each mouse
    counts = counts[counts['n_neurons'] >= MIN_NEURONS]
    all_areas_valid = counts['area_acronym_custom'].unique()
    combined_matrix = pd.DataFrame(0, index=all_areas_valid, columns=all_areas_valid, dtype=int)

    # Build dictionairy: mouse, reward_group -> set of qualifying areas
    mouse_areas_dict = counts.groupby(['mouse_id', 'reward_group'])['area_acronym_custom'].apply(set).to_dict()

    # Count pairs of qualifying areas per mouse
    for (mouse, reward_group), areas in mouse_areas_dict.items():
        for a1, a2 in combinations_with_replacement(areas, 2):
            # Upper triangle
            if reward_group==1:
                combined_matrix.loc[a1, a2] += 1
            else:
                combined_matrix.loc[a2, a1] += 1


    # Sort areas by total number of mice with at least min_neurons
    order_rplus = combined_matrix.iloc[0].sort_values(ascending=False).index
    combined_matrix = combined_matrix.loc[order_rplus, order_rplus]

    # Mask creation for colors
    mask_upper = np.tril(np.ones_like(combined_matrix, dtype=bool), k=-1)  # hide lower triangle
    mask_lower = np.triu(np.ones_like(combined_matrix, dtype=bool), k=1)  # hide upper triangle

    # Diagonal: number of mice with at least min_neurons for this area
    for area in all_areas_valid:
        combined_matrix.loc[area, area] = np.nan


    mask_min_mice = combined_matrix == 0
    combined_matrix[mask_min_mice] = np.nan

    # Plot the heatmap
    fig, ax = plt.subplots(1,1, figsize=(10, 10), dpi=500)

    sns.heatmap(
        data=combined_matrix,
        ax=ax,
        mask=mask_upper,
        annot=True,
        annot_kws={"size": 6},
        cmap=sns.color_palette('light:forestgreen', as_cmap=True),
        vmax=15,
        cbar=None,
        cbar_kws={'label': 'R+ mice, area pair counts', 'orientation': 'vertical', 'shrink': 0.5, 'ticks':range(0,16,2)}
    )
    sns.heatmap(
        data=combined_matrix,
        ax=ax,
        mask=mask_lower,
        annot=True,
        annot_kws={"size": 6},
        cmap=sns.color_palette('light:crimson', as_cmap=True),
        vmax=15,
        cbar=None,
        cbar_kws={'label': 'R- mice, area pair counts', 'orientation': 'vertical', 'shrink': 0.5, 'ticks':range(0,16,2)}
    )
    plt.title( f'Number of area pairs (neurons ≥ {MIN_NEURONS})\n')
    ax.tick_params(top=True, bottom=True, right=True, labeltop=True, labelbottom=True, labelright=True, labelleft=True, rotation=0)
    ax.set_xticks(np.arange(len(all_areas_valid)) + 0.5)
    ax.set_yticks(np.arange(len(all_areas_valid)) + 0.5)
    ax.set_xticklabels(all_areas_valid, rotation=45, fontsize=5)
    ax.set_yticklabels(all_areas_valid, rotation=0, fontsize=5)

    # Save and show
    figname = 'unit_area_pairs_count_heatmap'
    save_path = os.path.join(output_path, figname)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plotting_utils.save_figure_with_options(
                    figure=fig,
                    file_formats=['png', 'pdf', 'eps'],
                    filename=figname, output_dir=save_path,
                    dark_background=False
    )

    # Same figure but greying out when below 3 mice
    fig, ax = plt.subplots(1,1, figsize=(10, 10), dpi=500)

    combined_matrix_filtered = combined_matrix.copy()
    mask_min_mice = combined_matrix_filtered < 3
    combined_matrix_filtered[mask_min_mice] = np.nan


    rplus_cmap = sns.color_palette('light:forestgreen', as_cmap=True)
    #rplus_cmap.set_bad(color='#d3d3d3')  # light grey hex code
    sns.heatmap(
        data=combined_matrix_filtered,
        ax=ax,
        mask=mask_upper,
        annot=True,
        annot_kws={"size": 6},
        cmap=rplus_cmap,
        vmax=15,
        cbar=None,
        cbar_kws={'label': 'R+ mice', 'orientation': 'horizontal', 'pad': 0.02, 'shrink': 0.5}
    )
    rminus_cmap = sns.color_palette('light:crimson', as_cmap=True)
    #rminus_cmap.set_bad(color='#d3d3d3')  # light grey hex code
    sns.heatmap(
        data=combined_matrix_filtered,
        ax=ax,
        mask=mask_lower,
        annot=True,
        annot_kws={"size": 6},
        cmap=rminus_cmap,
        vmax=15,
        cbar=None,
        cbar_kws={'label': 'R- mice', 'orientation': 'horizontal', 'pad': 0.02, 'shrink': 0.5}
    )
    plt.title(f'Number of area pairs (neurons ≥ {MIN_NEURONS}, mice ≥ 3)')
    ax.tick_params(top=True, bottom=True, right=True, labeltop=True, labelbottom=True, labelright=True, rotation=0)
    ax.set_xticks(np.arange(len(all_areas_valid)) + 0.5)
    ax.set_yticks(np.arange(len(all_areas_valid)) + 0.5)
    ax.set_xticklabels(all_areas_valid, rotation=45, fontsize=5)
    ax.set_yticklabels(all_areas_valid, rotation=0, fontsize=5)
    plotting_utils.save_figure_with_options(
                    figure=fig,
                    file_formats=['png', 'pdf', 'eps'],
                    filename=figname+'_min3mice', output_dir=save_path,
                    dark_background=False
    )


    return combined_matrix
