#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_analysis
@file: allen_utils_old.py
@time: 1/21/2025 3:36 PM
"""

# Imports
import os
import json
import numpy as np
import re


def get_cortical_areas():
    """
    Retrieve a list of cortical(++) area acronyms.
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
                      'VL', 'I',
                      'nan'
                      ]
    return excluded_areas


def create_area_custom_column_old(df): # TODO: old version, to delete
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

    def simplify_area(ccf_acronym, ccf_parent_acronym):
        # Helper to check if the region contains a layer number
        def contains_layer(region):
            return bool(re.search(r'\d+[a-zA-Z]*', region))  # e.g., "6a", "6b"

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
            elif region.startswith("DG"):
                return "DG"
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
            elif region.startswith("VPL"):
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


def contains_layer(region):
    """Check if a region name contains a layer number excluding CA1, CA2, and CA3."""
    if region in ['CA1', 'CA2', 'CA3']:
        return False
    else:
        return bool(re.search(r'\d+[a-zA-Z]*', region))  # e.g., "6a", "6b"

def generalize_region(region):
    """Generalize region names based on predefined rules."""
    region_map = {
        "ACA": "ACA",
        "AI": "AI",
        "AUD": "AUD",
        "BST":"PAL",
        "CEA": "CEA",
        "DG": "DG",
        "Eth": "LAT", # confirm
        "EPd": "EP",
        "EPv": "EP",
        "IGL":"LGN", # confirm
        "IntG":"LGN", # confirm
        "INC":"PAG", # confirm
        "LD":"ATN", # confirm
        "LGd": "LGN",
        "LGv": "LGN",
        "LS": "LS",
        "LT": "MY", # confirm
        "MS":"LGN", #confirm
        "MGm":"MGN", # confirm
        "ORB": "ORB",
        "PAL": "PAL",
        "POL":"LAT", # confirm
        "PRNr":"Pons", # confirm
        "RR":"MRN",     # confirm
        "RSP": "RSP",
        "SI":"PAL",     # confirm
        "STR": "STR",
        "TEa": "TEa",
        "VPL": "VPL",
        "VPM": "VPM",
    }
    for key in region_map:
        if region.startswith(key):
            return region_map[key]

    if region.startswith("LA"):
        return "LAT" if region.startswith("LAT") else "LA"

    if region.startswith("SC"):
        return "SCm" if region in ["SCdg", "SCdw", "SCig", "SCiw"] else "SCs"

    if region.startswith("SSp-bfd"):
        return "SSp-bfd"

    if region.startswith("VIS"):
        return "VISC" if region.startswith("VISC") else "VIS"

    return region  # Default: no change


def handle_ssp_bfd(region):
    """Special case: handle SSp-bfd barrels (e.g., "SSp-bfd-C4" -> "SSp-bfd")."""
    return re.sub(r'SSp-bfd-[A-Z]\d+', 'SSp-bfd', region) if "SSp-bfd" in region else region



def simplify_area(ccf_acronym, ccf_parent_acronym):
    """Decide and return the simplified area name."""
    base_region = ccf_acronym if not contains_layer(ccf_acronym) or ccf_acronym in ['CA1', 'CA2',
                                                                                    'CA3'] else ccf_parent_acronym
    return handle_ssp_bfd(generalize_region(base_region))


def create_area_custom_column(df):
    """
    Using helps, create a new column 'area_acronym_custom' based on 'ccf_acronym' and 'ccf_parent_acronym'.
    - If ccf_acronym contains a layer number, use ccf_parent_acronym unless the region is CA1, CA2, or CA3.
    - Simplifies visual areas (e.g., VISpm, VISa, VISal) to "VIS".
    - Simplifies auditory areas (e.g., AUDd, AUDpo, AUDp, AUDv) to "AUD".
    - Simplifies ORBv to "ORB".
    - Handles specific cases like SSp-bfd barrel indications (e.g., SSp-bfd-C4 -> SSp-bfd).

    :param df: A pandas DataFrame containing 'ccf_acronym' and 'ccf_parent_acronym' columns.
    :return: DataFrame with a new column 'area_acronym_custom'.
    """
    df['area_acronym_custom'] = df.apply(lambda row: simplify_area(row['ccf_acronym'], row['ccf_parent_acronym']), axis=1)
    return df

def extract_layer_info(ccf_acronym):
    """Extract and return layer information from a region name."""
    match = re.search(r'(\d+[a-zA-Z]*)', ccf_acronym)
    if match:
        layer = match.group(0)
        return "2/3" if layer == "2" else layer
    return None


def create_layer_number_column(df):
    """Create a column 'layer_number' that only extracts layer information."""
    df['layer_number'] = df['ccf_acronym'].apply(extract_layer_info)
    return df


def create_ccf_acronym_no_layer_column(df):
    """Create a column 'ccf_acronym_no_layer' that keeps the original ccf_acronym unless it contains a layer number, in which case it uses the parent acronym."""
    df['ccf_acronym_no_layer'] = df.apply(
        lambda row: handle_ssp_bfd(row['ccf_parent_acronym']) if contains_layer(row['ccf_acronym']) else row['ccf_acronym'], axis=1)
    return df

def get_custom_area_order():
    """
    Get the order of brain areas for plotting.
    """
    area_order = ['MOp', 'MOs', 'FRP', 'ACA', 'PL', 'ORB', 'AI',
                  'SSp-bfd', 'SSs', 'SSp-m', 'SSp-n', 'SSp-ul', 'SSp-ll', 'SSp-tr', 'SSp-un',
                  'AUD', 'RSP',
                  'CLA', 'EP', 'LA',
                  'CA1', 'CA2', 'CA3', 'DG',
                  'CP', 'STR', 'ACB', 'CEA', 'LS', 'SF', 'GPe', 'PAL', 'MS',
                  'VPL', 'VPM', 'AV', 'LD', 'RT', 'PO', 'LGN', 'LP', 'ATN', 'LAT', 'MGN',
                  'SCs', 'SCm', 'MB', 'VTA', 'MRN', 'PAG', 'RN', 'SNr',
                  'Pons',
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
        'Hippocampus': ['CA1', 'CA2', 'CA3', 'DG'],
        'Striatal and pallidum': ['CP', 'STR', 'ACB', 'CEA', 'LS', 'SF', 'GPe', 'PAL', 'MS'],
        'Thalamus': ['VPL', 'VPM', 'AV', 'LD', 'RT', 'PO', 'LGN', 'LP', 'ATN', 'LAT', 'MGN'],
        'Midbrain': ['SCs', 'SCm', 'MB', 'VTA', 'MRN', 'PAG', 'RN', 'SNr'],
        'Pons': ['Pons'],
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
        'Pons': '#ffc395',
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

def apply_target_region_filters(peth_table, area):
    """
    Apply specific area filters based on the area name.

    :param peth_area: Subset of PETH table for a specific area
    :param area: Specifically-named brain area
    :return: Filtered PETH table
    """
    specific_filters = {
        'wS1': ['SSp-bfd'],
        'wM1': ['MOp', 'MOs'],
        'wS2': ['SSs'],
        'wM2': ['MOp', 'MOs'],
        'mPFC': ['PL', 'ILA'],
        'tjM1': ['MOp', 'MOs'],
        'A1': ['AUD', 'AUDd', 'AUDp', 'AUDv', 'AUDpo'],
        'DLS': ['STRd', 'CP'],
        'SC': ['SC', 'SCs', 'SCiw', 'SCop', 'SCm', 'SCzo', 'SCsg'],
        'OFC': ['ORB', 'ORBm', 'ORBl', 'ORBvl'],
        'ACA': ['ACA', 'ACAd', 'ACAv'],
        'ALM': ['MOp', 'MOs'],
    }

    if area in specific_filters.keys():
        # For wS1, keep only wS1/C2-targeted recordings and exclude SSPp-bfd obtained from other targets
        peth_area = peth_table[(peth_table['area_acronym_custom'].isin(specific_filters[area]))
                                & (peth_table['target_region'] == area)]
    else:
        print(f'{area} is not part of the target area dict. Skipping.')
        peth_area = None

    return peth_area