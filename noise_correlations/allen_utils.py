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
import pathlib

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def get_cortical_areas():
    """
    Retrieve a list of cortical(++) area acronyms.
    :return: List of cortical area acronyms
    """
    return [
        'FRP', 'MOp',
        'MOs', 'MOs-a', 'MOs-m', 'MOs-p',
        'SSp-bfd', 'SSp-m', 'SSp-ul', 'SSp-ll', 'SSp-un', 'SSp-n', 'SSp-tr',
        'SSs', 'AUD', 'AUDp', 'AUDd', 'AUDv',
        'ACA', 'ACAv', 'ACAd',
        'VIS','VISa', 'VISp', 'VISam', 'VISl', 'VISpm', 'VISrl', 'VISal',
        'PL', 'ILA', 'IL',
        'RSP', 'RSPv', 'RSPd','RSPagl',
        'SCm', 'SCsg', 'SCzo', 'SCiw', 'SCop', 'SCs',
        'ORB', 'ORBm', 'ORBl', 'ORBvl',
        'AId', 'AIv', 'AIp',
        'FRP', 'VISC'
    ]

def get_allen_color_dict():
    """
    Get Allen atlas colors formatted as dictionary of RGB arrays.
    :return:
    """
    # Get Allen atlas colors
    PATH_TO_ATLAS = r'C:\Users\bisi\.brainglobe\allen_mouse_bluebrain_barrels_10um_v1.0'

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



def contains_layer(region):
    """Check if a region name contains a layer number excluding CA1, CA2, and CA3."""
    if region in ['CA1', 'CA2', 'CA3']:
        return False
    else:
        return bool(re.search(r'\d+[a-zA-Z]*', str(region)))  # e.g., "6a", "6b"

def generalize_region(region):
    """Generalize region names based on predefined rules."""
    region_map = {
        "ACA": "ACA",
        "AD":"ATN",
        "AI": "AI",
        "Ai": "AI",
        "AMd":"ATN",
        "AMv":"ATN",
        "AON":"OLF",
        "APN":"APN",
        "AUD": "AUD",
        "AV":"ATN",
        "BLAp":"BLA",
        "BLAa":"BLA",
        "BST":"PAL",
        "CEA": "CEA",
        "CL": "ILM",
        "CM":"ILM",
        "CTXsp": "EP",
        "DG": "DG",
        "Eth": "LAT",
        "EPd": "EP",
        "EPv": "EP",
        'HPF':'HPF',
        "HY":"HY",
        "IG":"HPF",
        "IGL":"LGN",
        "IntG":"LGN",
        "INC":"PAG",
        "LD":"LGN",
        "LGd": "LGN",
        "LGv": "LGN",
        "LH":"HA",
        "LS": "LS",
        "LT": "MY",
        "MD":"MED",
        "MH":"HA",
        "MS":"PAL",
        "MGm":"MGN",
        "MGv":"MGN",
        "MGd":"MGN",
        "MMd":"HY",
        "MMme":"HY",
        "NLL":"Pons",
        "NPC":"MB",
        "ORB": "ORB",
        "PAL": "PAL",
        "PIL":"ILM", # confirm, also, use DORpm vs DORsm ? PIL-PP in lit.
        "PIR":"OLF",
        "POL":"LAT",
        "POST":"HPF",
        "PoT":"VP",
        "PP":"ILM",# PIL-PP in lit., ILMN: intralaminar nuclei
        "PPN":"MB",
        "PR":"MED",
        "PRC":"PAG",
        "PRNr":"Pons",
        "RE":"MTN",
        "RPF":"MB",
        "RR":"MB",
        "RSP": "RSP",
        "SAG":"MB",
        "SGN":"LAT",
        "SI":"PAL",
        "SMT":"MED",
        #"STR": "STR",
        "SUM":"HY",
        "TEa": "TEa",
        "TRS":"PAL",
        "VPL": "VPL",
        "VPM": "VPM",
        "VM":"VM",
        "Xi":"MTN"
    }
    for key in region_map:
        try:
            if region.startswith(key):
                return region_map[key]
        except AttributeError as err:
            print(err, region)

    if region.startswith("LA"):
        return "LAT" if region.startswith("LAT") else "LA"

    if region.startswith("MY"): #medulla
        return "MY"

    if region.startswith("SC"):
        return "SCm" if region in ["SCdg", "SCdw", "SCig", "SCiw"] else "SCs"

    if region.startswith("SSp-bfd"):
        return "SSp-bfd"

    if region.startswith("VIS"):
        if region=='VISC':
            return 'VISC'
        else:
            return 'VIS'

    return region  # Default: no change


def handle_ssp_bfd(region):
    """Special case: handle SSp-bfd barrels (e.g., "SSp-bfd-C4"/"SSp-bfd-Gamma" -> "SSp-bfd")."""
    return re.sub(r'SSp-bfd-[A-Za-z0-9]+', 'SSp-bfd', region) if "SSp-bfd" in region else region

def handle_ppc(region, row=None):
    """Special case: unify PPC subregions to 'PPC'."""
    pcc_areas = ['VIS', 'VISa', 'VISam', 'VISl', 'VISpm', 'VISrl', 'VISal', 'SSp-tr', 'SSp-un', 'SSp-bfd']
    if row['ccf_atlas_parent_acronym'] in pcc_areas and row['target_region']=='PPC':
        return 'PPC'
    else:
        return region


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
    def simplify_per_nomenclature(row):
        # Prefer ephys-align atlas fields if they exist and are not NaN
        if ('ccf_atlas_acronym' in row) and ('ccf_atlas_parent_acronym' in row):
            region = simplify_area(row['ccf_atlas_acronym'], row.get('ccf_atlas_parent_acronym', None))
        else:
            region = simplify_area(row['ccf_acronym'], row.get('ccf_parent_acronym', None))

        # Apply PPC unification (row context available here)
        try:
            region = handle_ppc(region, row)
        except KeyError as err: # If no row['target_region']
            region = region
        return region

    df['area_acronym_custom'] = df.apply(simplify_per_nomenclature, axis=1)
    return df

def extract_layer_info_original(ccf_acronym):
    """Extract and return layer information from a regionF name."""
    match = re.search(r'(\d+[a-zA-Z]*)', ccf_acronym)
    if match:
        layer = match.group(0)
        return "2/3" if layer == "2" else layer
    return None

def extract_layer_info(ccf_acronym):
    """
    Robustly extract layer info from a region name like:
    'MOp2/3', 'VISp2', 'S1-2', ...
    Returns strings like '2/3', '4', '5a', or None if not found / input is NaN.
    """
    # handle None, NaN, non-string inputs
    if pd.isna(ccf_acronym):
        return None

    s = str(ccf_acronym)
    _layer_re = re.compile(r'(?<!\w)([1-6](?:/[23])?(?:[a-zA-Z])?)(?!\w)')
    m = _layer_re.search(s)
    if not m:
        return None

    layer = m.group(1)
    # map plain '2' to '2/3' per your rule
    if layer == '2':
        return '2/3'
    return layer


def create_layer_number_column(df):
    """Create a column 'layer_number' that only extracts layer information."""
    if 'ccf_atlas_acronym' in df.columns:
        col='ccf_atlas_acronym'
    else:
        col='ccf_acronym'
    df['layer_number'] = df[col].apply(extract_layer_info) #Check which is nans
    return df


def create_ccf_acronym_no_layer_column(df):
    """Create a column 'ccf_acronym_no_layer' that keeps the original ccf_acronym unless it contains a layer number, in which case it uses the parent acronym."""
    if 'ccf_atlas_acronym' in df.columns and 'ccf_atlas_parent_acronym' in df.columns:
        col='ccf_atlas_acronym'
        col_parent='ccf_atlas_parent_acronym'
    else:
        col='ccf_acronym'
        col_parent='ccf_parent_acronym'
    df['ccf_acronym_no_layer'] = df.apply(
        lambda row: handle_ssp_bfd(row[col_parent]) if contains_layer(row[col]) else row[col], axis=1)
    return df

def get_target_region_order():
    """
    Get a set order of target regions for plotting.
    :return:
    """
    return ['wS1', 'wS2', 'A1', 'PPC', 'DLS', 'wM1', 'wM2', 'tjM1', 'ALM', 'OFC', 'SC']

def get_custom_area_order():
    """
    Get the order of brain areas for plotting.
    """
    area_order = ['MOp', 'MOs', 'MO-tjM1', 'MO-ALM', 'MO-wM1', 'MO-wM2', 'mPFC', 'FRP', 'ACA', 'PL', 'ORB', 'AI',
                  'SSp-bfd', 'SSs', 'SSp-m', 'SSp-n', 'SSp-ul', 'SSp-ll', 'SSp-tr', 'SSp-un', 'VISC', 'GU',
                  'AUD', 'TEa', 'RSP', 'PPC', 'VIS', 'VISa', 'VISp', 'VISam', 'VISl', 'VISpm', 'VISrl', 'VISal',
                  'CLA', 'EP', 'CTXsp',
                  'CA1', 'CA2', 'CA3', 'DG', 'HPF',
                  'CP', 'DMS', 'DLS', 'TS', 'STR', 'ACB', 'VS', 'FS', 'LS', 'SF', 'GPe', 'GPi', 'PAL', 'MS',
                  'TH', 'VPL', 'VPM', 'VP', 'LD', 'RT', 'PO', 'LGN', 'LP', 'ATN', 'LAT', 'MGN', 'MED', 'MTN', 'ILM', 'HA', 'CL',
                  'SCs', 'SCm', 'MB', 'VTA', 'MRN', 'PAG', 'RN', 'SNr', 'APN',
                  'Pons', 'MY',
                  'AON', 'OLF', 'PIR',
                  'BLA', 'LA', 'CEA','HY', 'ZI']
    return area_order

def get_custom_area_groups():
    """
    Get the custom area groups for plotting.
    """

    area_groups = {
        'Motor and frontal areas': ['MOp', 'MOs', 'MO-tjM1', 'MO-ALM', 'MO-wM1', 'MO-wM2', 'mPFC', 'FRP', 'ACA', 'PL', 'ORB', 'AI'],
        'Somatosensory areas': ['SSp-bfd', 'SSs', 'SSp-m', 'SSp-n', 'SSp-ul', 'SSp-ll', 'SSp-tr', 'SSp-un', 'VISC', 'GU'],
        'Auditory areas': ['AUD', 'TEa'],
        'Retrosplenial areas': ['RSP'],
        'Visual areas': ['PPC', 'VIS', 'VISa', 'VISp', 'VISam', 'VISl', 'VISpm', 'VISrl', 'VISal'],
        'Cortical subplate': ['CLA', 'EP'],
        'Hippocampus': ['CA1', 'CA2', 'CA3', 'DG', 'HPF'],
        'Striatum and pallidum': ['CP', 'DMS', 'DLS', 'TS', 'STR', 'VS', 'ACB', 'FS', 'LS', 'SF', 'GPe', 'GPi', 'PAL', 'MS'],
        'Thalamus': ['TH', 'VPL', 'VPM', 'VP', 'LD', 'RT', 'PO', 'LGN', 'LP', 'ATN', 'LAT', 'MGN', 'MED', 'MTN', 'ILM', 'HA', 'CL'],
        'Midbrain': ['SCs', 'SCm', 'MB', 'VTA', 'MRN', 'PAG', 'RN', 'SNr', 'APN'],
        'Pons and medulla': ['Pons', 'MY'],
        'Olfactory areas': ['AON', 'OLF', 'PIR'],
        'Amygdala and hypothalamus': ['BLA', 'BLAa', 'LA', 'CEA', 'HY', 'ZI']
    }
    return area_groups

# Now a function that outputs the revert dict mapping
def get_custom_area_group_revert_dict():
    """
    Get a dictionary mapping each area acronym to its corresponding group name.
    :return: Dictionary with area acronyms as keys and group names as values
    """
    area_groups = get_custom_area_groups()
    revert_dict = {area: group for group, areas in area_groups.items() for area in areas}
    return revert_dict

def get_custom_group_from_area(area):
    """
    Get the custom area group for a given area acronym.

    :param area: Area acronym (e.g., 'MOp', 'SSp-bfd')
    :return: Group name (e.g., 'Motor and frontal areas') or None if not found
    """
    area_groups = get_custom_area_groups()
    for group, areas in area_groups.items():
        if area in areas:
            return group
    return None


def get_custom_area_groups_from_name():
    """
    Get custom area groups from a predefined dictionary based on area names.

    :return: Dictionary of area groups with area acronyms as keys
    """
    area_groups = get_custom_area_groups()
    area_groups_from_name = {area: group for group, areas in area_groups.items() for area in areas}
    return area_groups_from_name

def get_custom_area_groups_colors():
    """Get custom area group colors for plotting, here Allen colors."""
    area_group_colors = {
        'Motor and frontal areas': '#1f9d5a',
        'Somatosensory areas': '#188064',
        'Auditory areas': '#019399',
        'Retrosplenial areas': '#1aa698',
        'Visual areas': '#1aa698',
        'Cortical subplate': '#8ada87',
        'Hippocampus': '#7ed04b',
        'Striatum and pallidum': '#98d6f9',
        'Thalamus': '#ff7080',
        'Midbrain': '#ff64ff',
        'Pons and medulla': '#ffc395',
        'Olfactory areas': '#9ad2bd',
        'Amygdala and hypothalamus': '#f2483b'

    }
    return area_group_colors

def get_custom_area_color_per_group():
    """
    Using custom area group above, return a dictionary and list of single-area colors.

    """

    # Make cmap with as many colors as number of area groups
    group_color_palette = get_custom_area_groups_colors() # colors from Allen atlas
    area_groups = get_custom_area_groups() # potentially not all groups are present
    default_color = '#888888'
    colors = [group_color_palette.get(group, default_color) for group in area_groups]
    #colors = [group_color_palette[i % len(group_color_palette)] for i in range(len(area_groups))] # Note: keep in case

    # Create a dictionary mapping each single area to its group color
    area_color_dict = {}
    for (group_name, areas), color in zip(area_groups.items(), colors):
        for area in areas:
            area_color_dict[area] = color

    # Make it also a list
    area_color_list = list(area_color_dict.values())
    return area_color_dict, area_color_list

def get_color_per_area(area_list):
    """
    Using the above functions and for a given list of areas, make a palette for seaborn to associate each area to its color.
    :param area_list:
    :return:
    """
    area_color_dict, _ = get_custom_area_color_per_group()
    color_palette = {area: area_color_dict.get(area, '#000000') for area in area_list}  # default to black if not found
    print('Areas not found:', [area for area in area_list if area not in area_color_dict])
    return color_palette

def create_legend_figure(color_dict, rectangles=True, title='Legend'):
    """
    Create a legend figure for the areas with their corresponding colors.
    :param color_dict: A dictionary mapping strings to colors (e.g., {"Label": "#ff0000"}).
    :param rectangles:  If True, use colored rectangles; otherwise, color the text directly.
    :param title: Title for the legend figure.
    :return: a figure containing the legend.
    """

    #color_dict = get_custom_area_groups_colors()

    fig, ax = plt.subplots(figsize=(2, 3), dpi=300)
    ax.axis('off')  # Turn off the axes for a cleaner look

    legend_elements = [Patch(facecolor=color, label=label) for label, color in color_dict.items()]
    ax.legend(handles=legend_elements, loc='upper left', frameon=False, title=title)
    fig.tight_layout()


    return fig


def apply_target_region_filters(peth_table, area):
    """
    Apply specific area filters based on the area name.

    :param peth_area: Subset of PETH table for a specific area
    :param area: Specifically-named brain area
    :return: Filtered PETH table
    """
    specific_filters = {
        'wS1': ['SSp-bfd'],
        'wM1': ['MOp', 'MOs', 'MOs-a', 'MOs-m','MOs-p','SSp-ll'],
        'wS2': ['SSs', 'SSp-bfd'],
        'wM2': ['MOp', 'MOs', 'MOs-a', 'MOs-m', 'MOs-p', 'SSp-ll'],
        'mPFC': ['PL', 'ILA', 'ACA', 'ACAd', 'ACAv'],
        'tjM1': ['MOp', 'MOs', 'SSp-m', 'MOs-a', 'MOs-m', 'MOs-p'],
        'A1': ['AUD', 'AUDd', 'AUDp', 'AUDv', 'AUDpo'],
        'DLS': ['STRd', 'CP', 'DLS'],
        'DMS': ['STRd', 'CP', 'DMS'],
        'VS': ['STRv', 'STR', 'ACB', 'CP', 'VS'],
        'TS': ['STRd', 'STR', 'CP', 'TS'],
        'SC': ['SC', 'SCs', 'SCiw', 'SCop', 'SCm', 'SCzo', 'SCsg'],
        'OFC': ['ORB', 'ORBm', 'ORBl', 'ORBvl'],
        'ALM': ['MOp', 'MOs', 'MOS-a', 'MOs-m', 'MOs-p'],
        'PPC': ['VIS', 'VISa', 'VISam', 'VISl', 'VISpm', 'VISrl', 'VISal', 'SSp-tr', 'SSp-un', 'SSp-bfd'],
    }

    if area in specific_filters.keys():
        # Keep only areas specified in filter and actually targeted e.g. SSp-bfd
        peth_area = peth_table[(peth_table['area_acronym_custom'].isin(specific_filters[area]))
                                & (peth_table['target_region'] == area)]
    else:
        print(f'{area} is not part of the target area dict. Skipping.')
        peth_area = None

    return peth_area

def compute_physical_coordinates_from_df(df, target_coords):
    """
    Compute approximate AP, ML, DV coordinates from entry points and trajectory angles in physical space for each recording site.
    :param df: pd.DataFrame with columns:
    :param target_coords: dict with target names as keys and (AP_entry, ML_entry) tuples as values.
    :return:
    """
    def _compute_row(row):
        # Get entry point
        AP_entry, ML_entry = target_coords[row['target']]
        depth = row['depth']
        az = np.deg2rad(row['azimuth'])
        el = np.deg2rad(row['elevation'])

        # Direction cosines
        dAP = np.cos(el) * np.sin(az)
        dML = np.cos(el) * np.cos(az)
        dDV = np.sin(el)

        # Compute coordinates
        AP = AP_entry + depth * dAP
        ML = ML_entry + depth * dML
        DV = depth * dDV

        return pd.Series({'ap_sample': AP, 'ml_sample': ML, 'dv_sample': DV})

    coords = df.apply(_compute_row, axis=1)
    return pd.concat([df, coords], axis=1)


def create_bregma_centric_coords_from_ccf(df):
    """
    Convert CCF coordinates in BrainGlobe space into bregma-centric coordinates.
    i.e. from (0,0,0)=(A,S,R) anterior top right corner to (0,0,0)=bregma.
    Using IBL bregma estimate:
    https://docs.internationalbrainlab.org/_autosummary/iblatlas.atlas.ALLEN_CCF_LANDMARKS_MLAPDV_UM.html
    :param df: unit_table pd.DataFrame with columns 'ccf_ap', 'ccf_ml', 'ccf_dv', and 'mouse_id'.
    :return:
    """
    # Convert columns to numeric
    df[['ccf_ap', 'ccf_ml', 'ccf_dv']] = df[['ccf_ap', 'ccf_ml', 'ccf_dv']].astype(float)

    # TODO: update fcn after new NWBs
    new_nwb_mice = ['AB077', 'AB080', 'AB082', 'AB085', 'AB086', 'AB087', 'AB092', 'AB093', 'AB094', 'AB095',
                    'AB102', 'AB104', 'AB107', #AB105
                    'AB116', 'AB117', 'AB119', 'AB120', 'AB121', 'AB122', 'AB123', 'AB124', 'AB125', 'AB126', 'AB127', 'AB128', 'AB129',
                    'AB130', 'AB131', 'AB132', 'AB133', 'AB134', #AB135
                    'AB136', 'AB137', 'AB138', 'AB139', 'AB140', 'AB141', 'AB142', 'AB143', 'AB144', 'AB145', 'AB146', 'AB147', 'AB148', 'AB149', #AB144
                    'AB150', 'AB151', 'AB152', 'AB153', 'AB154', 'AB155', 'AB156', 'AB157', 'AB158', 'AB159',
                    'AB161', 'AB162', 'AB163', 'AB164'
                    ]
    mh_mice = [f'MH{str(i).zfill(3)}' for i in range(40)]
    new_nwb_mice.extend(mh_mice)


    # Define conversion functions (all in um)
    #ml = (self.channels['x'] * 1e6) + 5739
    # Define conversion functions
    def func_to_ml(row):
        to_ml = lambda x: x - 5739
        if row['mouse_id'] in new_nwb_mice:
            return to_ml(row['ccf_ml'])
        else:
            return to_ml(row['ccf_dv']) #TODO: note, DV-ML inverted in NWB_Conversion -> update after new NWBs!

    #ap = (-self.channels['y'] * 1e6) + 5400
    def func_to_ap(row):
        to_ap = lambda x: -x + 5400 # AP positive is anterior relative to bregma
        if row['mouse_id'] in new_nwb_mice:
            return to_ap(row['ccf_ap'])
        else:
            return to_ap(row['ccf_ap'])
    #dv = (abs(self.channels['z'] * 1e6)) + 332
    def func_to_dv(row):
        to_dv = lambda x: x - 332
        if row['mouse_id'] in new_nwb_mice:
            return to_dv(row['ccf_dv'])
        else:
            return to_dv(row['ccf_ml']) #TODO: note, DV-ML inverted in NWB_Conversion -> update after new NWBs!

    # Apply conversions
    df['ap'] = df.apply(func_to_ap, axis=1)
    df['ml'] = df.apply(func_to_ml, axis=1)
    df['dv'] = df.apply(func_to_dv, axis=1)

    return df


def create_areas_subdivisions(df, verbose=False):
    """
    Divide large areas into smaller subdivisions for better visualization.
    For MOs/MOp, use only target-based mapping (no coordinate-based subdivision).

    :param df: pd.DataFrame with columns:
               - 'area_acronym_custom'
               - 'ap', 'ml', 'dv'
               - optionally 'target_region'
    :return: Updated DataFrame with new area assignments.
    """
    if verbose:
        print('Creating area subdivisions...')

    # Parent-child definitions (coordinate-based for CP/STR only)
    parent_child_dict = {
        'CP': ['VS', 'DLS', 'DMS', 'TS'], # VS includes STRv, ACB
        'STRd': ['VS', 'DLS', 'DMS', 'TS'],
        'STRv': ['VS', 'DLS', 'DMS', 'TS'],
        'STR': ['VS', 'DLS', 'DMS', 'TS'],
        'ACB': ['VS', 'DLS', 'DMS', 'TS'],
        'FS': ['VS', 'DLS', 'DMS', 'TS'],
    }

    # Coordinate boundaries for striatum subdivisions
    coord_boundaries = {
        'DMS': {'ap': (-1500, 6000), 'ml': (0, 3000), 'dv': (0, 7000)},
        'DLS': {'ap': (-1500, 500), 'ml': (2400, 6000), 'dv': (0, 7000)},
        'TS': {'ap': (-6000, -1500), 'ml': (0, 6000), 'dv': (0, 7000)},
        'VS': {'ap': (0, 6000), 'ml': (0, 6000), 'dv': (4000, 7000)},
    }

    # Target-region → label mapping for MOs/MOp
    target_region_map = {
        'tjM1': 'MO-tjM1',
        'ALM': 'MO-ALM',
        'OFC': 'MO-ALM',
        'wS2': 'MO-wM1', # TODO: DEAL WITH AB127
        'DLS': 'MO-wM1', #TODO: DEAL WITH AB156
        'wM1': 'MO-wM1',
        'PPC': 'MO-wM1',
        'wM2': 'MO-wM2'
    }

    df = df.copy()
    df['__assigned'] = False

    # --- Handle MOs / MOp separately (target-based only) ---
    if 'target_region' in df.columns:
        mos_mask = df['area_acronym_custom'].isin(['MOs', 'MOp'])
        for target_val, new_label in target_region_map.items():
            mask = mos_mask & (df['target_region'].astype(str) == str(target_val))
            df.loc[mask, 'area_acronym_custom'] = new_label
            df.loc[mask, '__assigned'] = True

        remaining = df[mos_mask & ~df['__assigned']]
        if len(remaining):
            if verbose:
                print(f"Unassigned MOs/MOp: {len(remaining)} (targets: {remaining['target_region'].unique()})")

    # --- Handle other parent areas (like CP) using coordinates ---
    all_striatum_aliases = ['CP', 'STR', 'STRd', 'STRv', 'ACB']

    for parent_area, subdivisions in parent_child_dict.items():
        if verbose:
           print(f'- Subdividing {parent_area} → {subdivisions}')

        for sub_area in subdivisions:
            bounds = coord_boundaries[sub_area]

            # VS comes from all striatal aliases
            if sub_area == 'VS':
                parent_mask = df['area_acronym_custom'].isin(all_striatum_aliases) & (~df['__assigned'])
            else:
                parent_mask = (df['area_acronym_custom'] == parent_area) & (~df['__assigned'])

            ap_mask = df['ap'].between(*bounds['ap'])
            ml_mask = df['ml'].between(*bounds['ml'])
            dv_mask = df['dv'].between(*bounds['dv'])
            mask = parent_mask & ap_mask & ml_mask & dv_mask

            df.loc[mask, 'area_acronym_custom'] = sub_area
            df.loc[mask, '__assigned'] = True
            #print(f"{sub_area}: {mask.sum()} units assigned")

        remaining = df[(df['area_acronym_custom'] == parent_area) & (~df['__assigned'])]
        remaining_coords = remaining[['ap', 'ml', 'dv']].values
        if verbose:
            print(f"Unassigned {parent_area}: {len(remaining)} units at coords:", remaining_coords)

    df.drop(columns=['__assigned'], inplace=True)
    return df

def create_area_groupings(df, verbose=False):
    """
    Create area groupings based on custom area acronyms.
    :param df:
    :param verbose:
    :return:
    """
    if verbose:
        print('Creating area groupings...')
    medial_pfc = {"PL", "ILA", "IL", "ACA", "ACAd", "ACAv"} #ORBm for mPFC?
    ppc = {"VIS", "VISa", "VISam", "VISl", "VISpm", "VISrl", "VISal", "SSp-tr", "SSp-un", "SSp-bfd"}

    def classify(acronym):
        if acronym in medial_pfc:
            return "mPFC"
        elif acronym in ppc:
            return "PPC"

    df = df.copy()
    # Override area_acronym_custom
    df['area_acronym_custom'] = df.apply(lambda row: classify(row['area_acronym_custom']) if classify(row['area_acronym_custom']) else row['area_acronym_custom'], axis=1)
    return df

def process_allen_labels(df, subdivide_areas=False):
    """
    Process the DataFrame to create custom area acronyms, layer numbers, and bregma-centric coordinates.
    :param df: unit_table pd.DataFrame from NWB files
    :param params: dictionary of parameters
    :return:
    """

    # Check if processing can be performed
    required_cols = ['target_region', 'ccf_acronym', 'ccf_parent_acronym', 'ccf_atlas_acronym', 'ccf_atlas_parent_acronym', 'ccf_ap', 'ccf_ml', 'ccf_dv', 'mouse_id']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        print(f'allen_utils.process_allen_labels: Warning - missing required columns for processing. Available columns: {df.columns}, Missing: {missing_cols}')
        text = f"Missing required columns for processing Allen labels: {', '.join(missing_cols)}. Available columns: {', '.join(df.columns)}"
        raise ValueError(text)

    # Remove unwanted areas
    try:
        df = df[~df['ccf_atlas_acronym'].isin(get_excluded_areas())]
    except KeyError as err: #TODO: fix these mice
        mouse_id = df['mouse_id'].unique()[0]
        print(f'ALLEN_UTILS [Warning]: issue with {mouse_id} CCF label processing: {err}')

    # Create custom area acronyms simplifying ccf areas acronyms
    df = create_area_custom_column(df)

    # Create layer number column
    df = create_layer_number_column(df)

    # Create a ccf_acronym_no_layer column, copy of ccf_acronym just without layer info
    df = create_ccf_acronym_no_layer_column(df)

    # Create bregma-centric coordinates, going from CCF (BrainGlobe) to bregma-centric coordinates using IBL bregma estimate
    df = create_bregma_centric_coords_from_ccf(df)

    # Create areas subdivisions for specific areas using custom boundaries
    if subdivide_areas:
        df = create_areas_subdivisions(df, verbose=False)
        df = create_area_groupings(df, verbose=False)

    return df

def keep_shared_areas(data_df, nomenclature, n_min_units=10, n_min_mice=3):
    """
    Keep set of shared areas across reward groups with (at least n_min_units units) and n_min_mice mice in each group.
    :param data_df: dataframe containing at least columns for reward_group, mouse_id, unit_id, and the specified nomenclature (e.g., area_acronym_custom).
    :param nomenclature: the column name in data_df to use for area labels (e.g., 'area_acronym_custom').
    :param n_min_units: minimum number of unique units required in each reward group for an area to be kept, across mice
    :param n_min_mice: minimum number of unique mice required in each reward group for an area to be kept.
    :return:
    """
    print(f'Filtering for R+/R- shared areas in {nomenclature} with at least '
          f'{n_min_units} units and {n_min_mice} mice in each reward group...')

    if nomenclature not in data_df.columns:
        raise ValueError(f"Specified nomenclature column '{nomenclature}' not found.")

    # Intersect areas across reward groups
    areas_rplus = data_df.loc[data_df['reward_group'] == 1, nomenclature].unique()
    areas_rminus = data_df.loc[data_df['reward_group'] == 0, nomenclature].unique()
    areas_intersect = set(areas_rplus).intersection(set(areas_rminus))

    print(f'Found {len(areas_intersect)} R+/R- shared areas: {areas_intersect}')

    # Count mice (always required)
    n_mice = (
        data_df
        .groupby(['reward_group', nomenclature])['mouse_id']
        .nunique()
        .unstack(fill_value=0)
    )

    # Count units ONLY if bc_label exists i.e. unit-level data present
    use_unit_threshold = 'bc_label' in data_df.columns

    if use_unit_threshold:
        print("Using unit counts (filtered by bc_label).")
        df_units = data_df[data_df['bc_label'].isin(['good', 'mua'])]

        n_units = (
            df_units
            .groupby(['reward_group', nomenclature])['unit_id']
            .nunique()
            .unstack(fill_value=0)
        )
    else:
        print("No 'bc_label' column found — skipping unit threshold.")
        n_units = None  # not used

    # Apply thresholds
    shared_areas = []

    for area in areas_intersect:

        # ---- Mouse condition (always enforced)
        mice_rplus = n_mice.get(area, {}).get(1, 0) if area in n_mice.columns else 0
        mice_rminus = n_mice.get(area, {}).get(0, 0) if area in n_mice.columns else 0
        mice_ok = (mice_rplus >= n_min_mice) and (mice_rminus >= n_min_mice)

        # ---- Unit condition (only if bc_label exists)
        if use_unit_threshold and n_min_units > 0:
            units_rplus = n_units.get(area, {}).get(1, 0) if area in n_units.columns else 0
            units_rminus = n_units.get(area, {}).get(0, 0) if area in n_units.columns else 0
            units_ok = (units_rplus >= n_min_units) and (units_rminus >= n_min_units)
        else:
            units_ok = True  # ignore unit threshold

        if mice_ok and units_ok:
            shared_areas.append(area)

    print(f'Keeping {len(shared_areas)} shared areas meeting thresholds.')
    if len(shared_areas) > 0:
        print("Shared areas:", shared_areas)

    # Final filtering
    data_df = data_df[data_df[nomenclature].isin(shared_areas)]

    return data_df, shared_areas


def load_process_hierarchy_from_harris():
    """
    Load the Allen atlas hierarchy from the Harris et al. 2019 paper, which provides a simplified hierarchy of brain regions.
    Cortex and thalamus only.
    :return: DataFrame with hierarchy summary scores and area acronyms.
    """

    # Get relative path from here to data file
    filename = 'hierarchy_summary_CreConf.xlsx'
    path_to_data = pathlib.Path(__file__).parent.parent / 'allen_utils' / 'data'
    path_to_file = path_to_data / filename

    print(path_to_file, os.getcwd())
    if not path_to_file.is_file():
        raise FileNotFoundError(f"Hierarchy file not found at {path_to_file}. Get data from: \n https://github.com/AllenInstitute/MouseBrainHierarchy/tree/master")

    hierarchy_df = pd.read_excel(path_to_file, sheet_name='hierarchy_all_regions')

    # Rename columns
    hierarchy_df.rename(columns={'CC+TC+CT iterated': 'cc_tc_ct_iterated',
                                 'areas':'ccf_acronym'}, inplace=True)
    hierarchy_df['ccf_atlas_acronym'] = hierarchy_df['ccf_acronym'] # for compatibility with create_area_custom_column
    print('Areas in raw hierarchy summary:', hierarchy_df.ccf_acronym.nunique())

    # Using "CC+TC+CT iterated" column, create a another column which adapted to use the area_acronym_custom
    hierarchy_df = create_area_custom_column(hierarchy_df)
    print('Areas in processed hierarchy summary:', hierarchy_df.area_acronym_custom.nunique())

    # Remove duplicates areas resulting from merging, keeping the mean hierarchy score for each area_acronym_custom
    print('Merging areas and averaging hierarchy scores for area duplicates...')
    hierarchy_df = hierarchy_df.groupby('area_acronym_custom').agg({'cc_tc_ct_iterated': 'mean', 'ccf_acronym': lambda x: ','.join(x.unique())}).reset_index()

    return hierarchy_df

def merge_hierarchy_from_harris(df):
    """
    Merge the hierarchy summary scores from Harris et al. onto the DataFrame based on area_acronym_custom.
    :param df: DataFrame with 'area_acronym_custom' column.
    :return: DataFrame with a new 'cc_tc_ct_iterated' column containing hierarchy scores.
    """
    hierarchy_df = load_process_hierarchy_from_harris()
    df = df.merge(hierarchy_df[['area_acronym_custom', 'cc_tc_ct_iterated']], on='area_acronym_custom', how='left')
    return df


def load_liu_et_al_avg_ipsi():
    """
    Load the Liu et al. group averages data and return a mapping from area acronym to avg_ipsi.
    :return: Dictionary mapping area acronym to avg_ipsi value.
    """
    liu_path = r'M:\analysis\Myriam_Hamon\combined_data\processed_data\Liu_et_al_Group_averages_ranked.xlsx'
    liu_df = pd.read_excel(liu_path)
    print(liu_df.keys())
    # First two rows are headers, actual data starts from row index 2
    liu_df = liu_df.iloc[2:].reset_index(drop=True)
    liu_df = liu_df.rename(columns={'Unnamed: 0': 'acronym'})
    liu_df = liu_df[['acronym', 'avg_ipsi']].dropna(subset=['acronym'])
    liu_df['avg_ipsi'] = pd.to_numeric(liu_df['avg_ipsi'], errors='coerce')
    return liu_df.set_index('acronym')['avg_ipsi'].to_dict()


def merge_liu_avg_ipsi(df, col_parent):
    """
    Merge Liu et al. avg_ipsi values onto the DataFrame.
    First tries matching on 'area_acronym_custom', then falls back to the parent acronym column.

    :param df: DataFrame with 'area_acronym_custom' column.
    :param col_parent: Name of the parent acronym column to use as fallback.
    :return: DataFrame with a new 'avg_ipsi' column.
    """
    liu_avg_ipsi = load_liu_et_al_avg_ipsi()

    # First try matching on area_acronym_custom
    df['avg_ipsi_corr'] = df['area_acronym_custom'].map(liu_avg_ipsi)

    # For rows without a match, fall back to parent acronym
    missing_mask = df['avg_ipsi_corr'].isna()
    df.loc[missing_mask, 'avg_ipsi_corr'] = df.loc[missing_mask, col_parent].map(liu_avg_ipsi)

    return df