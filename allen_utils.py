#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_analysis
@file: allen_utils.py
@time: 1/21/2025 3:36 PM
"""

# Imports
import os
import json
import numpy as np
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
    PATH_TO_ATLAS = r'C:\Users\bisi\.brainglobe\allen_mouse_25um_v1.2'

    with open(os.path.join(PATH_TO_ATLAS, 'structures.json')) as f:
        structures_dict = json.load(f)

    area_colors = {area['acronym']: np.array(area['rgb_triplet']) / 255 for area in structures_dict}
    return area_colors

def get_excluded_areas():
    """
    Retrieve a list of excluded area acronyms.
    :return: List of excluded area acronyms
    """
    excluded_areas = ["alv", "amc", "aco", "act", "arb", "ar", "bic", "bsc", "c", "cpd", "cbc", "cbp", "cbf", "AQ",
                      "epsc", "mfbc", "cett", "chpl", "cing", "cVIIIn", "fx", "stc", "cc", "fa", "ccb", "ee", "fp",
                      "ccs", "cst", "cm", "tspc", "cuf", "tspd", "dtd", "das", "dc", "df", "dhc", "lotd", "drt", "sctd",
                      "mfbse", "ec", "em", "eps", "VIIn", "fr", "fiber tracts", "fi", "fxs", "V4", "ccg", "gVIIn",
                      "hbc", "hc", "mfsbshy", "icp", "cic", "int", "lfbs", "ll", "lot", "lotg", "V4r", "mp",
                      "mfbsma", "mtg", "mtt", "mct", "mfb", "mfbs", "ml", "mlf", "mcp", "moV", "nst", "IIIn", "In",
                      "onl", "och", "IIn", "or", "opt", "fxpo", "pc", "pm", "py", "pyd", "root", "rust", "sV", "ts",
                      "sptV", "sm", "st", "SEZ", "scp", "dscp", "csc", "scwm", "sup", "tsp", "lfbst", "V3", "tb", "Vn",
                      "IVn", "uf", "Xn", "vhc", "sctv", "vtd", "VS", "vVIIIn", "VIIIn", "von",
                      'VL', 'I'
                      ]
    return excluded_areas

def create_area_custom_column(df):
    """
    Create a new column 'area_custom' based on 'ccf_acronym' and 'ccf_parent_acronym'.
    - If ccf_acronym contains a layer number, use ccf_parent_acronym unless the region is CA1, CA2, or CA3.
    - Simplifies visual areas (e.g., VISpm, VISa, VISal) to "VIS".
    - Simplifies auditory areas (e.g., AUDd, AUDpo, AUDp, AUDv) to "AUD".
    - Simplifies ORBv to "ORB".
    - Handles specific cases like SSp-bfd barrel indications (e.g., SSp-bfd-C4 -> SSp-bfd).

    :param df: A pandas DataFrame containing 'ccf_acronym' and 'ccf_parent_acronym' columns.
    :return: DataFrame with a new column 'area_custom'.
    """
    import re

    def simplify_area(ccf_acronym, ccf_parent_acronym):
        # Helper to check if the region contains a layer number
        def contains_layer(region):
            return bool(re.search(r'\d+[a-zA-Z]*', region))  # e.g., "6a", "6b", "4c", etc.

        # Helper to generalize regions (e.g., VISpm -> VIS, AUDd -> AUD, ORBv -> ORB)
        def generalize_region(region):
            if region.startswith("ACA"):
                return "ACA"
            elif region.startswith("AI"):
                return "AI"
            elif region.startswith("AUD"):
                return "AUD"
            elif region.startswith("CEA"):
                return "CEA"
            elif region.startswith("EPd") or region.startswith("EPv"):
                return "EP"
            elif region.startswith("LGd") or region.startswith("LGv"):
                return "LGN"
            elif region.startswith("LS"):
                return "LS"
            elif region.startswith("ORB"):
                return "ORB"
            elif region.startswith("PAL"):
                return "PAL"
            elif region.startswith("RSP"):
                return "RSP"
            elif region.startswith("SC"):
                if region in ["SCdg", "SCdw", "SCig", "SCiw"]:
                    return "SCm"
                elif region in ["SCop", "SCsg", "SCzo"]:
                    return "SCs"
            elif region.startswith("SSp-bfd"):
                return "SSp-bfd"
            elif region.startswith("STR"):
                return "STR"
            elif region.startswith("TEa"):  # although cortical, TEa has a different hierarchy
                return "TEa"
            elif region.startswith("VIS"):
                if region.startswith("VISC"):
                    return "VISC"
                else:
                    return "VIS"
            elif region.startswith("VPM"):
                return "VPL"
            elif region.startswith("VPM"):
                return "VPM"
            else:
                return region

        # Special case: Handle SSp-bfd barrels (e.g., "SSp-bfd-C4" -> "SSp-bfd")
        def handle_ssp_bfd(region):
            if "SSp-bfd" in region:
                return re.sub(r'SSp-bfd-[A-Z]\d+', 'SSp-bfd', region)
            return region

        # Step 1: If ccf_acronym contains a layer number, decide between ccf_acronym and ccf_parent_acronym
        if contains_layer(ccf_acronym):
            if ccf_acronym in ['CA1', 'CA2', 'CA3']:  # For CA1, CA2, CA3
                base_region = ccf_acronym
            else:
                base_region = ccf_parent_acronym
        else:
            base_region = ccf_acronym

        # Step 2: Generalize the region and handle special cases
        generalized_region = generalize_region(base_region)
        generalized_region = handle_ssp_bfd(generalized_region)

        return generalized_region

    # Apply the function to each row and create the new column
    df['area_custom'] = df.apply(lambda row: simplify_area(row['ccf_acronym'], row['ccf_parent_acronym']), axis=1)

    return df


def get_custom_area_order():
    """
    Get the order of brain areas for plotting.
    """
    area_order = ['MOp', 'MOs', 'FRP', 'ACA', 'PL', 'ORB', 'AI',
                  'SSp-bfd', 'SSs', 'SSp-m', 'SSp-n', 'SSp-ul', 'SSp-ll', 'SSp-tr', 'SSp-un',
                  'AUD', 'RSP',
                  'CLA', 'EP', 'LA',
                  'CA1', 'CA2', 'CA3',
                  'CP', 'STR', 'ACB', 'CEA', 'LS', 'SF', 'GPe', 'PAL', 'MS',
                  'VPL', 'VPM', 'AV', 'LD', 'RT', 'PO', 'LGN',
                  'MRN', 'PAG', 'SCs', 'SCm',
                  'AON', 'OLF', 'PIR',
                  'ZI']
    return area_order

def get_custom_area_groups():
    """
    Get the custom area groups for plotting.
    """

    area_groups = {
        'Motor and frontal areas': ['MOp', 'MOs', 'FRP', 'ACA', 'PL', 'ORB', 'AI'],
        'Somatosensory areas': ['SSp-bfd', 'SSs', 'SSp-m', 'SSp-n', 'SSp-ul', 'SSp-ll', 'SSp-tr', 'SSp-un'],
        'Auditory areas': ['AUD'],
        'Retrosplenial areas': ['RSP'],
        'Cortical subplate': ['CLA', 'EP', 'LA'],
        'Hippocampus': ['CA1', 'CA2', 'CA3'],
        'Striatal and pallidum': ['CP', 'STR', 'ACB', 'CEA', 'LS', 'SF', 'GPe', 'PAL', 'MS'],
        'Thalamus': ['VPL', 'VPM', 'AV', 'LD', 'RT', 'PO', 'LGN'],
        'Midbrain': ['MRN', 'PAG', 'SCs', 'SCm'],
        'Olfactory areas': ['AON', 'OLF', 'PIR'],
        'Hypothalamus': ['ZI']
    }
    return area_groups

def get_custom_area_groups_colors():
    """Get custom area group colors for plotting, here Allen colors."""
    area_group_colors = {
        'Motor and frontal areas': '#1f9d5a',
        'Somatosensory areas': '#188064',
        'Auditory areas': '#019399',
        'Retrosplenial areas': '#1aa698',
        'Cortical subplate': '#8ada87',
        'Hippocampus': '#7ed04b',
        'Striatal and pallidum': '#98d6f9',
        'Thalamus': '#ff7080',
        'Midbrain': '#ff64ff',
        'Olfactory areas': '#9ad2bd',
        'Hypothalamus': '#f2483b'

    }
    return area_group_colors
def get_custom_area_color_per_group():
    """
    Using custom area group above, return a dictionary and list of single-area colors.

    """

    # Make cmap with as many colors as number of area groups
    group_color_palette = get_custom_area_groups_colors() # colors from Allen atlas
    area_groups = get_custom_area_groups() # potentially not all groups are present
    colors = [group_color_palette[i % len(group_color_palette)] for i in range(len(area_groups))]

    # Create a dictionary mapping each single area to its group color
    area_color_dict = {}
    for (group_name, areas), color in zip(area_groups.items(), colors):
        for area in areas:
            area_color_dict[area] = color

    # Make it also a list
    area_color_list = list(area_color_dict.values())
    return area_color_dict, area_color_list
