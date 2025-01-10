#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_spikes_analysis
@file: plotting_utils.py
@time: 10/24/2024 4:38 PM
"""


# Imports
import os

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