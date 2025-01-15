#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_spikes_analysis
@file: plotting_utils.py
@time: 10/24/2024 4:38 PM
"""


# Imports
import os


def remove_top_right_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return


def save_figure_with_options(figure, file_formats, filename, output_dir='', dark_background=False):
    # Make transparent for dark background
    if dark_background:
        figure.patch.set_alpha(0)
        figure.set_facecolor('#f4f4ec')
        for ax in figure.get_axes():
            ax.set_facecolor('#f4f4ec')
        #plt.rcParams.update({'axes.facecolor': '#f4f4ec',  # very pale beige
        #                        'figure.facecolor': '#f4f4ec'})
        transparent = True
        filename = filename + '_transparent'
    else:
        transparent = False

    # Save the figure in each specified file format
    for file_format in file_formats:
        file_path = os.path.join(output_dir, f"{filename}.{file_format}")
        figure.savefig(file_path, transparent=transparent, bbox_inches='tight', dpi='figure')

    return

def get_cortical_areas():
    """
    Retrieve a list of cortical area acronyms.
    :return: List of cortical area acronyms
    """
    return [
        'FRP', 'MOp', 'MOs', 'SSp-bfd', 'SSp-m', 'SSp-ul', 'SSp-ll', 'SSp-un', 'SSp-n', 'SSp-tr',
        'SSs', 'AUDp', 'AUDd', 'AUDv', 'ACA', 'ACAv', 'ACAd', 'VISa', 'VISp', 'VISam', 'VISl',
        'VISpm', 'VISrl', 'VISal', 'PL', 'ILA', 'ORB', 'RSP', 'RSPv', 'RSPd','RSPagl', 'TT', 'SCm',
        'SCsg', 'SCzo', 'SCiw', 'SCop', 'SCs', 'ORBm', 'ORBl', 'ORBvl', 'AId',
        'AIv', 'AIp', 'FRP', 'VISC'
    ]

def get_allen_color_dict():
    """
    Get Allen atlas colors formatted as dictionary of RGB arrays.
    :return:
    """
    # Get Allen atlas colors
    path_to_atlas = r'C:\Users\bisi\.brainglobe\allen_mouse_25um_v1.2'

    with open(os.path.join(path_to_atlas, 'structures.json')) as f:
        structures_dict = json.load(f)

    area_colors = {area['acronym']: np.array(area['rgb_triplet']) / 255 for area in structures_dict}
    return area_colors

def get_excluded_areas():
    """
    Retrieve a list of excluded area acronyms.
    :return: List of excluded area acronyms
    """
    return [
        'root', 'fiber tracts', 'grey', 'nan', 'fxs', 'lfbst', 'cc', 'mfbc', 'cst', 'fa',
        'VS', 'ar', 'ccb', 'int', 'or', 'ccs', 'cing', 'ec', 'em', 'fi', 'scwm', 'alv', 'chpl', 'opt',
        'VL', 'SEZ', 'st', 'ccg', 'cpd'
    ]