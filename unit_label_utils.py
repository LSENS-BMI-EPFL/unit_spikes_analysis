#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_spikes_analysis
@file: unit_label_utils.py
@time: 4/13/2025 8:37 PM
"""


# Imports
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Import custom modules
import NWB_reader_functions as nwb_reader
import allen_utils as allen_utils
from plotting_utils import save_figure_with_options

def unit_label_describe(nwb_files, output_path):
    """
    Describe unit labels conversion from Kilosort to Bombcell.
    :param nwb_files: list of nwb files
    :param output_path: path to save the output
    :return:
    """
    # Load data
    unit_data = []
    for nwb_file in nwb_files:
        try:
            unit_table = nwb_reader.get_unit_table(nwb_file)
            mouse_id = nwb_reader.get_mouse_id(nwb_file)
            beh, day = nwb_reader.get_bhv_type_and_training_day_index(nwb_file)
            unit_table['mouse_id'] = mouse_id
            unit_data.append(unit_table)
        except:
            continue
    unit_data = pd.concat(unit_data)

    print('Number of good units after Kilosort', len(unit_data[unit_data['ks_label'] == 'good']))
    print('Number of good units after Bombcell', len(unit_data[unit_data['bc_label'] == 'good']))
    return

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

    fig.update_layout(title_text="Neuron label classification: ks_label → bc_label", font_size=14)
    fig.show()

    # Save figure
    file_name = 'unit_label_sankey_diagram'
    file_formats = ['png', 'pdf', 'eps']
    save_path = os.path.join(output_path, file_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save plotly Figure object #TODO: fix saving
    for format in file_formats:
        fig.write_image(os.path.join(save_path, f"{file_name}.{format}"), engine='orca')

    return