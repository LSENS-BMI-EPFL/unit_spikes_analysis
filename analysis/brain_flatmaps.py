#imports
import os
import glob 
import warnings 
import argparse
import matplotlib
import numpy as np
import pandas as pd
import cmasher as cmr
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from iblatlas.atlas import BrainRegions
from iblatlas.plots import plot_swanson_vector
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize, to_rgb, TwoSlopeNorm

#stop pop ups when creating plots
matplotlib.use('Agg') 

# nan slice warning
# is expected since we intentionally set some brain regions to nan for plotting
warnings.filterwarnings('ignore')

PROJECT_COLORS = {
    #single analyses and one colorbar 
    'whisker': 'forestgreen',
    'auditory': 'mediumblue',
    'spontaneous': '#FF8C42',
    'choice': 'darkorchid',
    'baseline_choice': 'chocolate',
    'learning':'mediumvioletred',

    # general reward groups two colorbars 
    'rewarded': 'forestgreen',
    'non_rewarded': 'crimson',

    # analyses x reward groups two colorbars 
    'whisker_rewarded': 'forestgreen',
    'whisker_nonrewarded': 'mediumseagreen',

    'auditory_rewarded': 'mediumblue',
    'auditory_nonrewarded':'cornflowerblue',

    'learning_rewarded': 'darkmagenta',
    'learning_nonrewarded':'indigo',

    'spontaneous_rewarded': 'teal',
    'spontaneous_nonrewarded': 'darkturquoise',

    'choice_rewarded': 'darkorchid',
    'choice_nonrewarded': 'mediumorchid',

    'baseline_choice_rewarded': 'chocolate',
    'baseline_choice_nonrewarded': 'sandybrown',

    'base': 'blues'
}

LIGHT_GRAY = '#D3D3D3'
DARK_GRAY = "#7E7D7D"
MISSING_DATA_DARK = -2.0


DATA_PATH = os.path.join("/Volumes", "Petersen-Lab", "z_LSENS", "Share", "Dana_Shayakhmetova", "new_roc_csv")
FIGURE_PATH = os.path.join("/Volumes", "Petersen-Lab", "z_LSENS", "Share", "Dana_Shayakhmetova", "flatmap_figures")

single_analyses = ['auditory_active', 'auditory_passive_pre', 'auditory_passive_post', 'baseline_choice', 'baseline_whisker_choice', 'choice', 'spontaneous_licks', 
                   'wh_vs_aud_pre_vs_post_learning', 'whisker_active', 'whisker_passive_pre', 'whisker_passive_post', 'whisker_choice']
reward_groups = ['R+', 'R-']
pair_analyses = [['whisker_active', 'auditory_active'], ['whisker_passive_pre', 'whisker_passive_post'], ['choice', 'whisker_choice'], ['baseline_whisker_choice', 'whisker_choice'], ['auditory_passive_pre', 'auditory_passive_post'], ['baseline_choice', 'choice']]
delta_pairs = [['whisker_passive_pre', 'whisker_passive_post'], ['auditory_passive_pre', 'auditory_passive_post']]


## General helper functions
def get_all_brain_region_names():
    """Get all Swanson atlas region acronyms from BrainRegions."""
    br = BrainRegions()
    swanson_indices = np.unique(br.mappings['Swanson'])
    swanson_ac = np.sort(br.acronym[swanson_indices])
    return swanson_ac

def swanson_conversion(roc_df):
    """
    Convert CCF acronyms to Swanson regions.
    """
    br = BrainRegions()
    roc_df = roc_df.copy()
    roc_df['ccf_acronym'] = roc_df['ccf_acronym'].astype(str).str.strip()

    # Manual mapping for specific missing acronyms
    manual_mapping_dict = {
        'STR': 'STRv', 'HPF': 'CA1', 'OLF': 'AON', 'FRP6a': 'FRP', 'FRP5': 'FRP',
        'FRP4': 'FRP', 'FRP2/3': 'FRP', 'FRP1': 'FRP', 'MB': 'MRN', 'P': 'PRNr',
        'SSp-tr6a': 'SSp-tr', 'SSp-tr5': 'SSp-tr', 'SSp-tr4': 'SSp-tr',
        'SSp-tr2/3': 'SSp-tr', 'SSp-tr1': 'SSp-tr', 'ORBm6a': 'ORBm',
        'RSPagl6a': 'RSPd', 'RSPagl6b': 'RSPd', 'RSPagl5': 'RSPd',
        'RSPagl4': 'RSPd', 'RSPagl2/3': 'RSPd', 'RSPagl1': 'RSPd',
    }
    roc_df['ccf_acronym_mapped'] = roc_df['ccf_acronym'].replace(manual_mapping_dict)

    #just in case again
    ssp_bfd_mask = roc_df['ccf_acronym_mapped'].str.startswith('SSp-bfd')
    roc_df.loc[ssp_bfd_mask, 'ccf_acronym_mapped'] = 'SSp-bfd'

    unique_acronyms = roc_df['ccf_acronym_mapped'].dropna().unique()
    swanson_mapping = {}
    for acronym in unique_acronyms:
        if pd.isna(acronym) or acronym == 'nan':
            swanson_mapping[acronym] = np.nan
        else:
            try:
                mapped_acronym = br.acronym2acronym(acronym, mapping='Swanson')
                swanson_mapping[acronym] = mapped_acronym[0] if mapped_acronym.size > 0 else np.nan
            except Exception:
                swanson_mapping[acronym] = np.nan
    roc_df['swanson_region'] = roc_df['ccf_acronym_mapped'].map(swanson_mapping)

    # Filter out unwanted regions
    regions_to_remove = ['root', 'void', '','nan', 'CTXsp', 'HY']
    roc_df_filtered = roc_df.dropna(subset=['swanson_region']).copy()
    roc_df_filtered = roc_df_filtered[~roc_df_filtered['swanson_region'].isin(regions_to_remove)].copy()
    # cleanup
    roc_df_filtered = roc_df_filtered.drop('ccf_acronym_mapped', axis=1)

    return roc_df_filtered

def filter_number_of_neurons(df, thres=15, mouse_thres=3):
    """
    Filter out brain areas based on anatomical, data-size, and subject-count criteria.

    This function removes:
      a) Pons and adjacent brainstem areas (defined in `excluded_areas`)
      b) Areas that have fewer than `thres` neurons in both reward groups
      c) Areas that have data from fewer than `mouse_thres` unique mice
    """
    # pons and adjacent brain stem areas
    excluded_areas = {"PRNr", "PRNc", "RM", "PPN", "V", "PSV", "PG", "LAV", "NLL", "SUT"}
    df = df[~df['swanson_region'].isin(excluded_areas)]
    
    # neuron count
    counts = (
        df.groupby(['reward_group', 'swanson_region'])['unit_id']
        .nunique()
        .reset_index(name='count')
    )
    pivot_counts = counts.pivot(index='swanson_region', columns='reward_group', values='count').fillna(0)
    areas_low_neurons = pivot_counts[(pivot_counts < thres).all(axis=1)].index.tolist()
    df = df[~df['swanson_region'].isin(areas_low_neurons)]

    # mouse count
    if mouse_thres != 0:
        mouse_counts = (
            df.groupby('swanson_region')['mouse_id']
            .nunique()
            .reset_index(name='num_mice')
        )
        areas_low_mice = mouse_counts[mouse_counts['num_mice'] < mouse_thres]['swanson_region'].tolist()

        df = df[~df['swanson_region'].isin(areas_low_mice)]
        print(f"Areas removed (mice < {mouse_thres}): {areas_low_mice}")

    print(f"Areas removed (count < {thres} neurons in both groups): {areas_low_neurons}")
    
    return df

def save_flatmap_figure(fig, filename, output_dir='./flatmaps/', formats=['pdf', 'svg', 'png'], dpi=500):
    """Save flatmap figure in multiple formats."""
    os.makedirs(output_dir, exist_ok=True)

    for fmt in formats:
        filepath = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        # print(f"Saved: {filepath}")

def generate_template_atlas(annotate=True, hemisphere='both', figsize=(15, 10), dpi=500):
    """Create blank Swanson flatmap template with optional region labels."""
    br = BrainRegions()
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    plot_swanson_vector(
        br=br,
        annotate=annotate,
        orientation='portrait',
        ax=ax,
        fontsize=8,
        hemisphere=hemisphere
    )
    ax.axis('off')
    return fig, ax

def generate_all_flatmaps(df, color_bar_type, output_dir):
    """
    Generate all single and dual hemisphere flatmaps.
    """

    single_hemisphere_output = os.path.join(output_dir, "single_hemisphere_flatmaps")
    dual_hemisphere_output = os.path.join(output_dir, "dual_hemisphere_flatmaps")
    os.makedirs(single_hemisphere_output, exist_ok=True)
    os.makedirs(dual_hemisphere_output, exist_ok=True)

    #types of analyses
    analyses = [
        'whisker_passive_pre', 'whisker_passive_post', 'whisker_active',
        'auditory_passive_pre', 'auditory_passive_post', 'auditory_active',
        'wh_vs_aud_pre_vs_post_learning', 'spontaneous_licks',
        'choice', 'whisker_choice', 'baseline_choice', 'baseline_whisker_choice'
    ]

    metrics = ['absolute', 'fraction', 'sign_positive', 'sign_negative']

    print(f"Single hemisphere metrics: {metrics}")
    reward_groups = ['R+', 'R-']
    annotation_states = [True, False]

    # SINGLE HEMISPHERE
    fig, ax = generate_template_atlas(hemisphere='left', annotate=True)
    save_flatmap_figure(fig, "template_annotated", output_dir=single_hemisphere_output, dpi=500)
    plt.close(fig)
    fig, ax = generate_template_atlas(hemisphere='left', annotate=False)
    save_flatmap_figure(fig, "template_not_annotated", output_dir=single_hemisphere_output, dpi=500)
    plt.close(fig)

    for analysis in analyses:
        print(f"Processing single hemisphere analyses for {analysis}")
        for reward_group in reward_groups:
            for metric in metrics:
                for annotate in annotation_states:
                    try:
                        fig, ax, text = generate_single_hemisphere(
                            df, analysis, reward_group=reward_group, metric=metric, annotate=annotate
                        )
                        if fig is not None:
                            save_flatmap_figure(fig, text, output_dir=single_hemisphere_output, dpi=500)
                            plt.close(fig)
                    except Exception as e:
                        print(f"Error generating single map for {analysis}, {reward_group}, {metric}, annotate={annotate}: {e}")


    # BOTH HEMISPHERES
    fig, ax = generate_template_atlas(hemisphere='both', annotate=True)
    save_flatmap_figure(fig, "both_template_annotated", output_dir=dual_hemisphere_output, dpi=500)
    plt.close(fig)
    fig, ax = generate_template_atlas(hemisphere='both', annotate=False)
    save_flatmap_figure(fig, "both_template_not_annotated", output_dir=dual_hemisphere_output, dpi=500)
    plt.close(fig)

    dual_metrics = ['absolute', 'fraction', 'sign']
    annotation_states = [True, False]

    #one analysis vs the other
    comparison_pairs = [
        (['whisker_passive_pre', 'whisker_passive_post'], 'R+'),
        (['whisker_passive_pre', 'whisker_passive_post'], 'R-'),
        (['auditory_passive_pre', 'auditory_passive_post'], 'R+'),
        (['auditory_passive_pre', 'auditory_passive_post'], 'R-'),
        (['whisker_active', 'auditory_active'], 'R+'),
        (['whisker_active', 'auditory_active'], 'R-'),
        (['baseline_choice', 'choice'], 'R+'),
        (['baseline_choice', 'choice'], 'R-'),
        (['baseline_whisker_choice', 'whisker_choice'], 'R+'),
        (['baseline_whisker_choice', 'whisker_choice'], 'R-'),
        (['choice', 'whisker_choice'], 'R+'),
        (['choice', 'whisker_choice'], 'R-'),
    ]

    print(f"\nDual hemispheres metrics: {dual_metrics}")

    for analyses_pair, reward_group in comparison_pairs:
        print(f"Processing dual map: {analyses_pair[0]} vs {analyses_pair[1]} for {reward_group}")
        for metric in [m for m in dual_metrics if m != 'sign']: #no sign
            for annotate in annotation_states:
                try:
                    fig, ax, text = generate_dual_hemispheres(df, analyses_pair, color_bar_type, reward_group=reward_group, metric=metric, annotate=annotate)
                    if fig is not None:
                        save_flatmap_figure(fig, text, output_dir=dual_hemisphere_output, dpi=500)
                        plt.close(fig)
                except Exception as e:
                    print(f"Error generating dual map analysis 1 vs 2: {analyses_pair}, {reward_group}, {metric}, annotate={annotate}: {e}")


    # R+ vs R-
    for analysis in analyses:
        print(f"Processing dual map: {analysis} R+ vs R-")
        for metric in [m for m in dual_metrics if m != 'sign']:
            for annotate in annotation_states:
                try:
                    # reward_group=None
                    fig, ax, text = generate_dual_hemispheres(df, [analysis], color_bar_type, reward_group=None, metric=metric, annotate=annotate)
                    if fig is not None:
                        save_flatmap_figure(fig, text, output_dir=dual_hemisphere_output, dpi=500)
                        plt.close(fig)
                except Exception as e:
                    print(f"Error generating dual map R+ vs R-: {analysis}, {metric}, annotate={annotate}: {e}")

    # Sign Comparison
    for analysis in analyses:
        for reward_group in reward_groups:
            print(f"Processing dual map: {analysis} {reward_group} Directional")
            for annotate in annotation_states:
                try:
                    # metric='sign' so that we do Positive vs Negative logic
                    fig, ax, text = generate_dual_hemispheres(df, [analysis], color_bar_type, reward_group=reward_group, metric='sign', annotate=annotate)
                    if fig is not None:
                        save_flatmap_figure(fig, text, output_dir=dual_hemisphere_output, dpi=500)
                        plt.close(fig)
                except Exception as e:
                    print(f"Error generating dual map Sign: {analysis}, {reward_group}, annotate={annotate}: {e}")


## Single hemisphere helper functions 
def get_single_analysis_colormap(analysis_type, intensity_range=(0.2, 1.0)):
    """ 
    Get colormap for a specific analysis and reward group.
    Used for single hemisphere plots."""

    base_color = 'Blues' #default for now

    if 'whisker' in analysis_type.lower():
        if '+' in analysis_type.lower():
            base_color = PROJECT_COLORS['whisker_rewarded']
        elif '-' in analysis_type.lower():
            base_color = PROJECT_COLORS['whisker_nonrewarded']
        elif 'active' in analysis_type.lower():
            base_color = PROJECT_COLORS['whisker_rewarded']
        else:
            base_color = PROJECT_COLORS['whisker']

    elif 'auditory' in analysis_type.lower():
        if '+' in analysis_type.lower():
            base_color = PROJECT_COLORS['auditory_rewarded']
        elif '-' in analysis_type.lower():
            base_color = PROJECT_COLORS['auditory_nonrewarded']
        elif 'active' in analysis_type.lower():
            base_color = PROJECT_COLORS['auditory_rewarded']
        else:
            base_color = PROJECT_COLORS['auditory']

    elif 'spontaneous' in analysis_type.lower():
        if '+' in analysis_type.lower():
            base_color = PROJECT_COLORS['spontaneous_rewarded']
        elif '-' in analysis_type.lower():
            base_color = PROJECT_COLORS['spontaneous_nonrewarded']
        else:
            base_color = PROJECT_COLORS['spontaneous']
    elif 'learning' in analysis_type.lower():
        if '+' in analysis_type.lower():
            base_color = PROJECT_COLORS['learning_rewarded']
        elif '-' in analysis_type.lower():
            base_color = PROJECT_COLORS['learning_nonrewarded']
        else:
            base_color = PROJECT_COLORS['learning']
    elif 'choice' in analysis_type.lower():
        if '+' in analysis_type.lower():
            base_color = PROJECT_COLORS['choice_rewarded'] if 'baseline' not in analysis_type.lower() else PROJECT_COLORS['baseline_choice_rewarded']
        elif '-' in analysis_type.lower():
            base_color = PROJECT_COLORS['choice_nonrewarded'] if 'baseline' not in analysis_type.lower() else PROJECT_COLORS['baseline_choice_nonrewarded']
        else:
            base_color = PROJECT_COLORS['choice'] if 'baseline' not in analysis_type.lower() else PROJECT_COLORS['baseline_choice']


    if isinstance(base_color, str) and base_color in ['Blues', 'Reds', 'Greens']:
        base_cmap = matplotlib.colormaps.get_cmap(base_color)
        colors = base_cmap(np.linspace(intensity_range[0], intensity_range[1], 256))
    else:
        # gradient from white to color
        color_rgb = to_rgb(base_color)
        colors = []
        for i in np.linspace(0, 1, 256):
            weight_factor = intensity_range[0] + i * (intensity_range[1] - intensity_range[0])
            white_weight = 1 - weight_factor
            color_weight = weight_factor

            rgb = tuple(white_weight * 1.0 + color_weight * c for c in color_rgb)
            colors.append(rgb + (1.0,))

    return LinearSegmentedColormap.from_list(f'custom_{analysis_type}', colors)


## Dual hemispheres helper functions 
def get_analysis_colormap(analysis_type, intensity_range=(0.2, 1.0)):
    """
    Get colormap for analysis type only (dual hemisphere version).
    Similar to get_single_analysis_colormap but doesn't consider reward group.
    """

    base_color = 'Blues' #default for now

    if 'whisker' in analysis_type.lower():
        base_color = PROJECT_COLORS['whisker']
    elif 'auditory' in analysis_type.lower():
        base_color = PROJECT_COLORS['auditory']
    elif 'spontaneous' in analysis_type.lower():
        base_color = PROJECT_COLORS['spontaneous']
    elif 'learning' in analysis_type.lower():
        base_color = PROJECT_COLORS['learning']
    elif 'choice' in analysis_type.lower() and 'baseline' not in analysis_type.lower():
        base_color = PROJECT_COLORS['choice']
    else: #baseline choice 
        base_color = PROJECT_COLORS['baseline_choice']

    # gradient from white to color
    color_rgb = to_rgb(base_color)
    colors = []
    for i in np.linspace(0, 1, 256):
        weight_factor = intensity_range[0] + i * (intensity_range[1] - intensity_range[0])
        white_weight = 1 - weight_factor
        color_weight = weight_factor

        rgb = tuple(white_weight * 1.0 + color_weight * c for c in color_rgb)
        colors.append(rgb + (1.0,))

    return LinearSegmentedColormap.from_list(f'custom_{analysis_type}', colors)

def get_base_colormap(intensity_range=(0.2, 1.0)):
    """Get default blue colormap when no specific analysis type is specified."""
    base_color = 'Blues'
    base_cmap = matplotlib.colormaps.get_cmap(base_color)
    colors = base_cmap(np.linspace(intensity_range[0], intensity_range[1], 256))
    return LinearSegmentedColormap.from_list(f'custom_base', colors)

def get_reward_group_colormap(reward_group, intensity_range=(0.2, 1.0)):
    """
    Get colormap based on reward group only.
    Green gradient for R+, red gradient for R-.
    """

    if reward_group == 'R+':
        base_color = PROJECT_COLORS['rewarded']
    else:
        base_color = PROJECT_COLORS['non_rewarded']
    # gradient from white to color
    color_rgb = to_rgb(base_color)
    colors = []
    for i in np.linspace(0, 1, 256):
        weight_factor = intensity_range[0] + i * (intensity_range[1] - intensity_range[0])
        white_weight = 1 - weight_factor
        color_weight = weight_factor

        rgb = tuple(white_weight * 1.0 + color_weight * c for c in color_rgb)
        colors.append(rgb + (1.0,))

    return LinearSegmentedColormap.from_list(f'custom_base', colors)

def get_analysis_and_reward_colormap(analysis_type, intensity_range=(0.2, 1.0)):
    """
    Get colormap for specific analysis and reward group combination.
    Dual hemisphere version.
    """

    base_color = 'Blues' #default for now

    if 'whisker' in analysis_type.lower():
        if '+' in analysis_type.lower():
            base_color = PROJECT_COLORS['whisker_rewarded']
        elif '-' in analysis_type.lower():
            base_color = PROJECT_COLORS['whisker_nonrewarded']
        else: #active
            base_color = PROJECT_COLORS['whisker']

    elif 'auditory' in analysis_type.lower():
        if '+' in analysis_type.lower():
            base_color = PROJECT_COLORS['auditory_rewarded']
        elif '-' in analysis_type.lower():
            base_color = PROJECT_COLORS['auditory_nonrewarded']
        else: #active 
            base_color = PROJECT_COLORS['auditory']

    elif 'spontaneous' in analysis_type.lower():
        if '+' in analysis_type.lower():
            base_color = PROJECT_COLORS['spontaneous_rewarded']
        elif '-' in analysis_type.lower():
            base_color = PROJECT_COLORS['spontaneous_nonrewarded']
        else:
            base_color = PROJECT_COLORS['spontaneous']

    elif 'learning' in analysis_type.lower():
        if '+' in analysis_type.lower():
            base_color = PROJECT_COLORS['learning_rewarded']
        elif '-' in analysis_type.lower():
            base_color = PROJECT_COLORS['learning_nonrewarded']
        else:
            base_color = PROJECT_COLORS['learning']

    elif 'choice' in analysis_type.lower() and 'baseline' not in analysis_type.lower():
        if '+' in analysis_type.lower():
            base_color = PROJECT_COLORS['choice_rewarded'] 
        elif '-' in analysis_type.lower():
            base_color = PROJECT_COLORS['choice_nonrewarded'] 
        else:
            base_color = PROJECT_COLORS['choice'] 

    else: #baseline 
        if '+' in analysis_type.lower():
            base_color = PROJECT_COLORS['baseline_choice_rewarded']
        elif '-' in analysis_type.lower():
            base_color = PROJECT_COLORS['baseline_choice_nonrewarded']
        else:
            base_color = PROJECT_COLORS['baseline_choice']


    # gradient from white to color
    color_rgb = to_rgb(base_color)
    colors = []
    for i in np.linspace(0, 1, 256):
        weight_factor = intensity_range[0] + i * (intensity_range[1] - intensity_range[0])
        white_weight = 1 - weight_factor
        color_weight = weight_factor

        rgb = tuple(white_weight * 1.0 + color_weight * c for c in color_rgb)
        colors.append(rgb + (1.0,))

    return LinearSegmentedColormap.from_list(f'custom_{analysis_type}', colors)

def get_diverging_colormap():
    """Get PRGn diverging colormap for directional selectivity comparisons."""

    return matplotlib.colormaps.get_cmap('PRGn')

def create_combined_colormap(cmap_base, vmin_metric, vmax_metric):
    """
    Create colormap with gray region for missing data.
    Made with help from ChatGPT.
    """
    
    # Real color range
    v_min_range = MISSING_DATA_DARK
    v_max_range = vmax_metric

    # Zero one space
    def normalize_value(v, v_min, v_max):
        if v_max == v_min:
            return 0.5
        return (v - v_min) / (v_max - v_min)

    # Normalize positions
    pos_dark = normalize_value(MISSING_DATA_DARK, v_min_range, v_max_range)
    pos_metric_start = normalize_value(vmin_metric, v_min_range, v_max_range)

    # Anchors for dark nodes only
    all_colors_nodes = [
        (pos_dark, DARK_GRAY),
        (pos_dark + 1e-6, DARK_GRAY),
    ]

    # Quick transition to metric colormap
    if pos_metric_start > pos_dark + 1e-6:
        base_cmap_norm_start = 0.0 if vmax_metric == vmin_metric else 0.0
        start_color = cmap_base(base_cmap_norm_start) if not isinstance(cmap_base, str) else to_rgb(cmap_base)
        all_colors_nodes.append((pos_metric_start - 1e-6, start_color))
        all_colors_nodes.append((pos_metric_start, start_color))

    # Smooth colormap nodes
    if vmax_metric > vmin_metric:
        metric_nodes = np.linspace(0.0, 1.0, 256)
        for m_node in metric_nodes:
            metric_value = vmin_metric + m_node * (vmax_metric - vmin_metric)
            pos = normalize_value(metric_value, v_min_range, v_max_range)
            all_colors_nodes.append((pos, cmap_base(m_node)))

    # Sort normalized nodes and remove duplicates
    all_colors_nodes.sort(key=lambda x: x[0])
    unique_nodes = []
    seen_pos = set()
    for pos, color in all_colors_nodes:
        if not any(abs(pos - p) < 1e-5 for p in seen_pos):
            unique_nodes.append((pos, color))
            seen_pos.add(pos)
    
    cmap_final = LinearSegmentedColormap.from_list('CustomGrayMetricMap', unique_nodes)
    norm = Normalize(vmin=v_min_range, vmax=v_max_range)

    return cmap_final, norm, v_min_range, v_max_range, vmin_metric, vmax_metric

def get_data_and_regions(df_filtered, metric, reward_group=None):
    """
    Computes the requested metric (absolute, fraction, sign) for each
    Swanson region in the filtered data.
    """
    
    df_mapped = swanson_conversion(df_filtered)
    swanson_regions = df_mapped.swanson_region.unique()
    
    # default
    data = pd.Series(dtype=float)
    cmap = get_base_colormap()
    vmin_metric, vmax_metric = 0, 1
    label = 'Value'
    
    # to compute vmin and vmax
    def compute_vlims(series, default=(0, 1), diverging=False):
        series = series[(series != MISSING_DATA_DARK) & series.notna()]
        if series.empty:
            return default
        if diverging:
            max_abs = max(abs(series.min()), abs(series.max()))
            return -round(max_abs, 1), round(max_abs, 1)
        return round(series.min(), 1), round(series.max(), 1)

    total_counts = df_mapped.groupby('swanson_region').size()
    
    if metric == 'absolute':
        data = df_mapped['selectivity'].abs().groupby(df_mapped['swanson_region']).mean()
        label = 'Mean absolute selectivity'
        vmin_metric, vmax_metric = compute_vlims(data)

    elif metric == 'fraction':
        sig_counts = df_mapped[df_mapped['significant']].groupby('swanson_region').size()
        counts_df = pd.DataFrame({
            'total': total_counts,
            'selective': sig_counts.reindex(total_counts.index, fill_value=0)
        })
        data = (counts_df['selective'] / counts_df['total'] * 100).fillna(0)
        data[(counts_df['total'] > 0) & (counts_df['selective'] == 0)] = MISSING_DATA_DARK
        label = 'Selective units %'
        vmin_metric, vmax_metric = compute_vlims(data, default=(0, 100))

    elif metric == 'sign':
        pass  # no changes needed
    else:
        print(f"Unknown metric: {metric} in get_data_and_regions")
        return pd.Series(dtype=float), np.array([]), cmap, vmin_metric, vmax_metric, label

    try:
        sample_type = df_filtered['analysis_type'].iloc[0]
        if reward_group:
            sample_type += f'_{reward_group}'

        cmap = get_analysis_and_reward_colormap(sample_type)
    except Exception:
        cmap = get_base_colormap()

    return data, swanson_regions, cmap, vmin_metric, vmax_metric, label


## Single hemisphere plots 
def generate_single_hemisphere(df, analysis_type, reward_group, metric='absolute', annotate=True, figsize=(10, 12), dpi=500, cmap=None):
    """
    Generate single hemisphere flatmap for one analysis and reward group.
    Plots left hemisphere only. Regions with no data are light gray,
    regions with units but no selectivity are dark gray.
    """
    #data prep
    br = BrainRegions()
    df_all_swanson = swanson_conversion(df).swanson_region.unique()
    filtered_df = df[df['analysis_type'] == analysis_type].copy()
    if reward_group is not None:
        filtered_df = filtered_df[filtered_df.reward_group == reward_group]
    filtered_df_mapped = swanson_conversion(filtered_df)
    
    #defaults
    data = pd.Series(dtype=float)
    vmin_metric, vmax_metric = 0, 1
    label = 'Value'
    
    if not filtered_df_mapped.empty:
        total_counts = filtered_df_mapped.groupby('swanson_region').size()
    else:
        total_counts = pd.Series(dtype=float)
    
    #calculating values
    if metric == 'absolute':
        if not filtered_df_mapped.empty:
            data = (filtered_df_mapped
                    .assign(selectivity_abs=filtered_df_mapped['selectivity'].abs())
                    .groupby('swanson_region')['selectivity_abs']
                    .mean())
        # Keep NaN as is
        metric_values = data.dropna()
        if not metric_values.empty:
            vmin_metric = round(metric_values.min(), 1)
            vmax_metric = round(metric_values.max(), 1)
        label = 'Mean absolute selectivity'
        
    elif metric == 'fraction':
        if not filtered_df_mapped.empty:
            significant_counts = (filtered_df_mapped[filtered_df_mapped['significant'] == True]
                                .groupby('swanson_region').size())
            counts_df = pd.DataFrame({
                'total': total_counts,
                'selective': significant_counts.reindex(total_counts.index, fill_value=0)
            })
            data = (counts_df['selective'] / counts_df['total'] * 100)
            # Regions with units but zero selective = dark gray
            counts_df = counts_df.dropna() 
            zero_selective_mask = (counts_df['total'] > 0) & (counts_df['selective'] == 0)
            data.loc[zero_selective_mask] = MISSING_DATA_DARK
            
            metric_values = data[(data != MISSING_DATA_DARK) & data.notna()]
            if not metric_values.empty:
                vmin_metric = round(metric_values.min(), 1)
                vmax_metric = round(metric_values.max(), 1)
            else:
                vmin_metric, vmax_metric = 0, 100
        else:
            vmin_metric, vmax_metric = 0, 100
        label = 'Selective unit %'
        
    elif metric == 'sign_positive':
        sign_filtered = filtered_df_mapped[filtered_df_mapped['selectivity'] > 0]
        if len(sign_filtered) > 0:
            data = (sign_filtered
                    .groupby('swanson_region')['selectivity']
                    .mean())
        metric_values = data.dropna()
        if not metric_values.empty:
            vmin_metric = round(metric_values.min(), 1)
            vmax_metric = round(metric_values.max(), 1)
        label = 'Mean selectivity (positive)'
        
    elif metric == 'sign_negative':
        sign_filtered = filtered_df_mapped[filtered_df_mapped['selectivity'] < 0]
        if len(sign_filtered) > 0:
            data = (sign_filtered
                    .groupby('swanson_region')['selectivity']
                    .mean())
            data = data * -1  # positive for bar
        metric_values = data.dropna()
        if not metric_values.empty:
            vmin_metric = round(metric_values.min(), 1)
            vmax_metric = round(metric_values.max(), 1)
        label = 'Mean selectivity (negative)'
                
    else:
        print(f"Unknown metric: {metric}")
        return None, None, None
    
    #cmap stuff
    if reward_group is not None:
        type_str = analysis_type + f'_{reward_group}'
        cmap_base = get_single_analysis_colormap(type_str)
    else:
        cmap_base = get_single_analysis_colormap(analysis_type)
        
    data_vector = data.reindex(br.acronym)
    cmap_final = ListedColormap(cmap_base(np.linspace(0, 1, 256)))
    cmap_final.set_under(DARK_GRAY)
    data_to_plot = data_vector.dropna()
    region_ids_to_plot = br.acronym2id(data_to_plot.index.values)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    plot_swanson_vector(
        region_ids_to_plot,
        data_to_plot.values,
        cmap=cmap_final,
        orientation='portrait',
        vmin=vmin_metric,
        vmax=vmax_metric,
        br=br,
        fontsize=6,
        ax=ax,
        empty_color=LIGHT_GRAY,
        annotate=annotate,
        annotate_list=df_all_swanson if annotate else []
    )
        
    # Colorbar 
    cbar_ax = fig.add_axes([0.3, 0.95, 0.15, 0.01])
    norm_cbar = Normalize(vmin=vmin_metric, vmax=vmax_metric)
    sm = cm.ScalarMappable(cmap=cmap_base, norm=norm_cbar)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(label, fontsize=7)
    metric_ticks = np.linspace(vmin_metric, vmax_metric, 5) if (vmax_metric > vmin_metric) else [vmin_metric]
    metric_ticks = metric_ticks[(metric_ticks >= vmin_metric) & (metric_ticks <= vmax_metric)]
    cbar.set_ticks(metric_ticks)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.xaxis.set_label_position('top')
    
    ax.axis('off')
    plt.tight_layout()
    
    reward_part = reward_group if reward_group is not None else 'R_combined'
    text = f"{analysis_type}_{reward_part}_{metric}_annotated" if annotate else f"{analysis_type}_{reward_part}_{metric}_not_annotated"
    
    return fig, ax, text


## Dual hemispheres 
def generate_dual_hemispheres(df, analysis_types, color_bar_type, reward_group=None, metric='absolute', annotate=True, figsize=(15, 10), dpi=500):
    """
    Generate dual hemisphere comparison flatmaps.
    
    Three comparison modes:
    1. Two analyses, same reward group 
    2. Same analysis, R+ vs R- 
    3. Same analysis, positive vs negative selectivity 

    """
    br = BrainRegions()
    all_swanson_regions = get_all_brain_region_names()
    df_all_swanson = swanson_conversion(df).swanson_region.unique()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white') 
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.03, hspace=0.2, wspace=0.2)
    ax.set_facecolor('white')     
    
    # two different analyses 
    if len(analysis_types) == 2:
        left_analysis, right_analysis = analysis_types[0], analysis_types[1]
        
        if reward_group is None:
            plt.close(fig)
            return None, None, None
            
        left_df = df[(df['analysis_type'] == left_analysis) & (df['reward_group'] == reward_group)].copy()
        right_df = df[(df['analysis_type'] == right_analysis) & (df['reward_group'] == reward_group)].copy()
        left_data, _, cmap_left, vmin_left, vmax_left, label = get_data_and_regions(left_df, metric,  reward_group)
        right_data, _, cmap_right, vmin_right, vmax_right, _ = get_data_and_regions(right_df, metric, reward_group)
        
        # Shared vmin/vmax
        vmin = min(vmin_left, vmin_right)
        vmax = max(vmax_left, vmax_right)
        
        left_title = f'{left_analysis.replace("_", " ").capitalize()} - {reward_group}'
        right_title = f'{right_analysis.replace("_", " ").capitalize()} - {reward_group}'
        text = f"{left_analysis}_vs_{right_analysis}_{reward_group}_{metric}"

        if color_bar_type == 'one_color_bar':
            #if same type like whisker_passive_pre and whisker_passive_post
            if any(cond in left_analysis and cond in right_analysis for cond in ["whisker", "auditory", "baseline_choice",'wh_vs_aud']):
                cmap = get_analysis_colormap(left_analysis)
            else: #else 
                cmap = get_base_colormap()
            
            plot_single_colorbar(
                fig, ax, br, all_swanson_regions, df_all_swanson,
                left_data, right_data, cmap, vmin, vmax, label,
                left_title, right_title, annotate
            )

        elif color_bar_type == 'reward_group_colorbar':
            cmap = get_reward_group_colormap(reward_group)
            plot_single_colorbar(
                fig, ax, br, all_swanson_regions, df_all_swanson,
                left_data, right_data, cmap, vmin, vmax, label,
                left_title, right_title, annotate
            )
        
        else: # by analysis type x reward group 
            plot_dual_colorbar(
                fig, ax, br, all_swanson_regions, df_all_swanson,
                left_data, right_data,
                cmap_left, vmin_left, vmax_left,
                cmap_right, vmin_right, vmax_right,
                label,
                left_title, right_title, annotate
            )

    # one analysis by reward groups or signs
    elif len(analysis_types) == 1:
        analysis_type = analysis_types[0]
        
        # R+ vs R- comparison
        if reward_group is None and metric != 'sign':
            left_df = df[(df['analysis_type'] == analysis_type) & (df['reward_group'] == 'R+')].copy()
            right_df = df[(df['analysis_type'] == analysis_type) & (df['reward_group'] == 'R-')].copy()
            
            left_data, _, cmap_left, vmin_left, vmax_left, label = get_data_and_regions(left_df, metric, 'R+')
            right_data, _, cmap_right, vmin_right, vmax_right, _ = get_data_and_regions(right_df, metric, 'R-')
            
            left_title = f'{analysis_type.replace("_", " ").capitalize()} R+'
            right_title = f'{analysis_type.replace("_", " ").capitalize()} R-'
            text = f"{analysis_type}_R+_vs_R-_{metric}"

            if color_bar_type == 'one_color_bar':
                cmap = cmap_left #use rewarded as base color 
                vmin = min(vmin_left, vmin_right)
                vmax = max(vmax_left, vmax_right)
                plot_single_colorbar(
                fig, ax, br, all_swanson_regions, df_all_swanson,
                left_data, right_data, cmap, vmin, vmax, label,
                left_title, right_title, annotate)

            elif color_bar_type == 'reward_group_colorbar':
                cmap_left = get_reward_group_colormap('R+')
                cmap_right = get_reward_group_colormap('R-')
                plot_dual_colorbar(fig, ax, br, all_swanson_regions, df_all_swanson, 
                            left_data, right_data, cmap_left, vmin_left, vmax_left, 
                            cmap_right, vmin_right, vmax_right, label, 
                            left_title, right_title, annotate)
                
            else: 
                plot_dual_colorbar(fig, ax, br, all_swanson_regions, df_all_swanson, 
                            left_data, right_data, cmap_left, vmin_left, vmax_left, 
                            cmap_right, vmin_right, vmax_right, label, 
                            left_title, right_title, annotate)

        
        # Positive vs. Negative Sign (ONE COLORBAR)
        elif reward_group is not None and metric == 'sign':
            data_df = df[(df['analysis_type'] == analysis_type) & (df['reward_group'] == reward_group)].copy()
            positive_df = data_df[data_df['selectivity'] > 0].copy()
            negative_df = data_df[data_df['selectivity'] < 0].copy()
            negative_df['selectivity'] = negative_df['selectivity'].abs()
            left_data, _, cmap, vmin_left, vmax_left, label = get_data_and_regions(positive_df, 'absolute', reward_group)
            right_data, _, _, vmin_right, vmax_right, _ = get_data_and_regions(negative_df, 'absolute', reward_group)
            
            # Use shared scale
            vmin = min(vmin_left, vmin_right) if not (np.isnan(vmin_left) or np.isnan(vmin_right)) else 0
            vmax = max(vmax_left, vmax_right) if not (np.isnan(vmax_left) or np.isnan(vmax_right)) else 1
            
            left_title = f'{analysis_type.replace("_", " ").capitalize()} {reward_group} (positive)'
            right_title = f'{analysis_type.replace("_", " ").capitalize()} {reward_group} (negative)'
            text = f"{analysis_type}_{reward_group}_sign_comparison"
            label = 'Mean absolute selectivity by sign'
            
            plot_single_colorbar(fig, ax, br, all_swanson_regions, df_all_swanson, 
                               left_data, right_data, cmap, vmin, vmax, label, 
                               left_title, right_title, annotate)
        else:
            plt.close(fig)
            return None, None, None
    else:
        plt.close(fig)
        return None, None, None
    
    ax.axis('off')
    text_final = f"{text}_annotated" if annotate else f"{text}_not_annotated"
    
    return fig, ax, text_final

def plot_single_colorbar(fig, ax, br, all_swanson_regions, df_all_swanson, left_data, right_data, cmap, vmin, vmax, label, left_title, right_title, annotate):
    """Helper function to plot data with one shared colorbar, using NaN for missing data."""

    left_data_vector = pd.Series(np.nan, index=all_swanson_regions)
    right_data_vector = pd.Series(np.nan, index=all_swanson_regions)
    left_data_vector.update(left_data)
    right_data_vector.update(right_data)

    # acronyms in the atlas
    left_data_vector_valid = left_data_vector[left_data_vector.index.isin(br.acronym)].copy()
    right_data_vector_valid = right_data_vector[right_data_vector.index.isin(br.acronym)].copy()

    # create_combined_colormap can handle DARK_GRAY va
    cmap_final, _, v_min_range, v_max_range, _, _ = create_combined_colormap(cmap, vmin, vmax)
    combined_regions = np.concatenate([-br.acronym2id(left_data_vector_valid.index.values), br.acronym2id(right_data_vector_valid.index.values)])
    combined_values = np.concatenate([left_data_vector_valid.values, right_data_vector_valid.values])

    # Filter out NaN values 
    valid_mask = ~np.isnan(combined_values)

    plot_swanson_vector(combined_regions[valid_mask], combined_values[valid_mask], 
                        hemisphere='both', cmap=cmap_final, orientation='portrait',
                        vmin=v_min_range, vmax=v_max_range, br=br, fontsize=8, ax=ax, annotate=annotate,
                        annotate_list=df_all_swanson if annotate else [],
                        empty_color=LIGHT_GRAY) 

    # colorbar
    norm_cbar = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm_cbar)
    cbar_ax = fig.add_axes([0.42, 0.97, 0.16, 0.008])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(label, fontsize=7, labelpad=5)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks(np.linspace(vmin, vmax, 5))
    ax.text(0.03, 1, left_title, transform=ax.transAxes, fontsize=7, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.97, 1, right_title, transform=ax.transAxes, fontsize=7, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def plot_dual_colorbar(fig, ax, br, all_swanson_regions, df_all_swanson, left_data, right_data, 
                       cmap_left, vmin_left, vmax_left, cmap_right, vmin_right, vmax_right, 
                       label, left_title, right_title, annotate):
    """Helper function to plot data with two separate colorbars, using NaN for missing data."""

    
    vmin_shared = min(vmin_left, vmin_right)
    vmax_shared = max(vmax_left, vmax_right)
    norm_on_shared = mcolors.Normalize(vmin=vmin_shared, vmax=vmax_shared)
    
    offset = vmax_shared + 10
    
    left_data_vector = pd.Series(np.nan, index=all_swanson_regions)
    right_data_vector = pd.Series(np.nan, index=all_swanson_regions)
    left_data_vector.update(left_data)
    right_data_vector.update(right_data)

    # offset for right data 
    right_data_offset = right_data_vector.copy()
    mask_to_offset = right_data_offset.notna() & (right_data_offset != MISSING_DATA_DARK)
    right_data_offset[mask_to_offset] += offset
    
    # acronyms only in br.acronyms
    left_data_vector_valid = left_data_vector[left_data_vector.index.isin(br.acronym)].copy()
    right_data_vector_valid = right_data_offset[right_data_offset.index.isin(br.acronym)].copy()
    
    combined_regions = np.concatenate([
        -br.acronym2id(left_data_vector_valid.index.values), 
        br.acronym2id(right_data_vector_valid.index.values)
    ])
    combined_values = np.concatenate([
        left_data_vector_valid.values, 
        right_data_vector_valid.values
    ])
    
    vmin_norm = MISSING_DATA_DARK
    vmax_norm = offset + vmax_shared
    norm_temp = mcolors.Normalize(vmin=vmin_norm, vmax=vmax_norm)
    
    # nodes for colormap
    nodes = []
    nodes.append((norm_temp(MISSING_DATA_DARK), DARK_GRAY))
    #left colormap
    for val in np.linspace(vmin_shared, vmax_shared, 128):
        nodes.append((norm_temp(val), cmap_left(norm_on_shared(val))))
    gap_start = vmax_shared + 1e-6
    gap_end = offset + vmin_shared - 1e-6
    nodes.append((norm_temp(gap_start), LIGHT_GRAY))
    nodes.append((norm_temp(gap_end), LIGHT_GRAY))
    
    # right colormap
    for val in np.linspace(vmin_shared, vmax_shared, 128):
        nodes.append((norm_temp(offset + val), cmap_right(norm_on_shared(val))))
    
    nodes.sort(key=lambda x: x[0])
    cmap_final = mcolors.LinearSegmentedColormap.from_list("stitched_cmap", nodes)
    
    valid_mask = ~np.isnan(combined_values)
    plot_swanson_vector(combined_regions[valid_mask], combined_values[valid_mask], 
                        hemisphere='both', cmap=cmap_final,
                        orientation='portrait', vmin=vmin_norm, vmax=vmax_norm, br=br, fontsize=8,
                        ax=ax, annotate=annotate, annotate_list=df_all_swanson if annotate else [],
                        empty_color=LIGHT_GRAY) 
    
    # colorbar
    norm_cbar_shared = Normalize(vmin=vmin_shared, vmax=vmax_shared)
    sm_left = cm.ScalarMappable(cmap=cmap_left, norm=norm_cbar_shared)
    cbar_ax_left = fig.add_axes([0.2, 0.97, 0.2, 0.01])
    cbar_left = fig.colorbar(sm_left, cax=cbar_ax_left, orientation='horizontal')
    cbar_left.set_label(label, fontsize=8, labelpad=5)
    cbar_left.ax.xaxis.set_label_position('top')
    cbar_left.ax.tick_params(labelsize=8)
    cbar_left.set_ticks(np.linspace(vmin_shared, vmax_shared, 5))
    sm_right = cm.ScalarMappable(cmap=cmap_right, norm=norm_cbar_shared)
    cbar_ax_right = fig.add_axes([0.6, 0.97, 0.2, 0.01])
    cbar_right = fig.colorbar(sm_right, cax=cbar_ax_right, orientation='horizontal')
    cbar_right.set_label(label, fontsize=8, labelpad=5)
    cbar_right.ax.xaxis.set_label_position('top')
    cbar_right.ax.tick_params(labelsize=8)
    cbar_right.set_ticks(np.linspace(vmin_shared, vmax_shared, 5))
    ax.text(0.03, 0.97, left_title, transform=ax.transAxes, fontsize=7, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.97, 0.97, right_title, transform=ax.transAxes, fontsize=7, verticalalignment='top', 
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

## Delta hemispheres plot
def plot_dual_hemispheres_delta(df,analysis_type_1,analysis_type_2,metric, annotate=True,figsize=(10, 12),dpi=500):
    """
    Plot change (delta) between two analyses for both reward groups.
    """
    br = BrainRegions()
    df = swanson_conversion(df)
    print(f"Generating delta flatmap for {analysis_type_1} and {analysis_type_2} with metric {metric}")

    def delta_fraction(df, analysis_type_1, analysis_type_2, reward_group):
        f1 = df[(df["analysis_type"] == analysis_type_1) & (df["reward_group"] == reward_group)].copy()
        f2 = df[(df["analysis_type"] == analysis_type_2) & (df["reward_group"] == reward_group)].copy()

        frac1 = f1.groupby("swanson_region")["significant"].apply(lambda x: 100 * x.sum() / len(x))
        frac2 = f2.groupby("swanson_region")["significant"].apply(lambda x: 100 * x.sum() / len(x))
        all_regions = set(frac1.index) | set(frac2.index)

        delta = pd.Series({r: frac2.get(r, 0) - frac1.get(r, 0) for r in all_regions})
        return delta
    
    def delta_absolute(df, analysis_type_1, analysis_type_2, reward_group):
        f1 = df[(df["analysis_type"] == analysis_type_1) & (df["reward_group"] == reward_group)].copy()
        f2 = df[(df["analysis_type"] == analysis_type_2) & (df["reward_group"] == reward_group)].copy()

        abs1 = f1.groupby("swanson_region")["selectivity"].apply(lambda x: x.abs().mean())
        abs2 = f2.groupby("swanson_region")["selectivity"].apply(lambda x: x.abs().mean())
        all_regions = set(abs1.index) | set(abs2.index)

        delta = pd.Series({r: abs2.get(r, 0) - abs1.get(r, 0) for r in all_regions})
        return delta

    if metric=='absolute':
        delta_rplus = delta_absolute(df, analysis_type_1, analysis_type_2, "R+")
        delta_rminus = delta_absolute(df, analysis_type_1, analysis_type_2, "R-")
    else:
        delta_rplus = delta_fraction(df, analysis_type_1, analysis_type_2, "R+")
        delta_rminus = delta_fraction(df, analysis_type_1, analysis_type_2, "R-")

    left_regions = delta_rplus.index
    right_regions = delta_rminus.index
    plot_regions = []
    for i in left_regions:
        if i not in plot_regions:
            plot_regions.append(i)

    for i in right_regions:
        if i not in plot_regions:
            plot_regions.append(i)

    all_swanson_regions = get_all_brain_region_names()

    left_data_vector = pd.Series(np.nan, index=all_swanson_regions)
    right_data_vector = pd.Series(np.nan, index=all_swanson_regions)

    left_data_vector.loc[delta_rplus.index] = delta_rplus.values
    right_data_vector.loc[delta_rminus.index] = delta_rminus.values

    left_data_vector_valid = left_data_vector[left_data_vector.index.isin(br.acronym)].copy()
    right_data_vector_valid = right_data_vector[right_data_vector.index.isin(br.acronym)].copy()

    if metric == 'absolute':
        vmin_metric, vmax_metric = -0.1, 0.1
    else:
        vmin_metric, vmax_metric = -15, 15

    metric_colors = cmr.viola(np.linspace(0.1, 0.9, 256))
    cmap_final = ListedColormap(metric_colors)

    cmap_final.set_under(cmr.viola(0.1))   
    cmap_final.set_bad(LIGHT_GRAY)   
    cmap_final.set_over(cmr.viola(0.9))   
    
    norm_final = TwoSlopeNorm(vmin=vmin_metric, vcenter=0, vmax=vmax_metric)

    combined_regions = np.concatenate([
        -br.acronym2id(left_data_vector_valid.index.values),   
         br.acronym2id(right_data_vector_valid.index.values) 
    ])
    combined_values_raw = np.concatenate([
        left_data_vector_valid.values,
        right_data_vector_valid.values
    ])
    
    # TwoSlopeNorm to get normalized values for plotting
    # Did this with the help of chat gpt
    combined_values_normalized = norm_final(combined_values_raw)
    # normalized values back to vmin_metric to vmax_metric range
    # so plot_swanson_vector can use vmin=vmin_metric, vmax=vmax_metric
    combined_values_for_plot = combined_values_normalized * (vmax_metric - vmin_metric) + vmin_metric

    #nan for gray 
    combined_values_for_plot = np.where(np.isneginf(combined_values_for_plot), np.nan, combined_values_for_plot)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    plot_swanson_vector(
        combined_regions,
        combined_values_for_plot, 
        hemisphere='both',
        cmap=cmap_final,
        vmin=vmin_metric,
        vmax=vmax_metric,
        orientation='portrait',
        br=br,
        fontsize=7,
        ax=ax,
        annotate=annotate,
        annotate_list= plot_regions if annotate else [],
        empty_color=LIGHT_GRAY
    )
    # ax.set_title("Δ Fraction Selective Neurons\nR+ (Left) vs R− (Right)", fontsize=10)
    ax.axis('off')

    cbar_ax = fig.add_axes([0.4, 0.95, 0.2, 0.02])
    sm = plt.cm.ScalarMappable(cmap=cmap_final, norm=norm_final)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    if metric == 'absolute':
        cbar.set_label('Delta mean absolute selectivity', fontsize=8)
    else:
        cbar.set_label('Delta fraction selective (%)', fontsize=8)

    cbar.set_ticks(np.linspace(vmin_metric, vmax_metric, 5))
    cbar.ax.tick_params(labelsize=7)

    cbar.ax.xaxis.set_label_position('top')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if 'whisker_passive'in analysis_type_1:
        topic = 'Delta whisker passive'
    elif 'auditory_passive' in analysis_type_1:
        topic = 'Delta auditory passive'
    else:
        topic = analysis_type_2+ "-" + analysis_type_1

    ax.text(0.03, 0.99, f'{topic} R+', transform=ax.transAxes, fontsize=7, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.97, 0.99, f'{topic} R-', transform=ax.transAxes, fontsize=7, verticalalignment='top', 
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))  

    text = f"{analysis_type_1}_to_{analysis_type_2}_both_reward_groups_{metric}"
    return fig, ax, text


## Full run 
def main(args):
    thres = args.thres 
    mouse_thres = args.mouse_thres
    color_bar_type = args.color_bar_type 

    #pulling mice info 
    print("\nPulling mice information...")
    mouse_info_path = os.path.join("/Volumes", "Petersen-Lab", "z_LSENS", "Share", "Dana_Shayakhmetova", "mouse_info", "joint_mouse_reference_weight.xlsx")
    mouse_info_df = pd.read_excel(mouse_info_path)
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df = mouse_info_df[
        (mouse_info_df['exclude'] == 0) &
        (mouse_info_df['reward_group'].isin(['R+', 'R-'])) &
        (mouse_info_df['recording'] == 1)]
    print("Done.")

    #pulling ROC data 
    data_path_ab = os.path.join(DATA_PATH, "new_roc_csv_AB")
    data_path_mh = os.path.join(DATA_PATH, "new_roc_csv_MH")

    print("\nPulling all AB mice csv files...")
    roc_results_files = glob.glob(os.path.join(data_path_ab, '**', '*_roc_results_new.csv'), recursive=True) 
    roc_df_axel = pd.concat([pd.read_csv(f) for f in roc_results_files], ignore_index=True)
    print("Done.")

    print("\nPulling all MH mice csv files...")
    roc_results_files = glob.glob(os.path.join(data_path_mh, '**', '*_roc_results_new.csv'),recursive=True)
    roc_df_myriam = pd.concat([pd.read_csv(f) for f in roc_results_files], ignore_index=True)
    print("Done.")

    print("\nCombining data...")
    roc_df = pd.concat([roc_df_axel, roc_df_myriam], ignore_index=True)
    roc_df = roc_df.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')
    roc_df['neuron_id'] = roc_df.index.astype(int)
    print('Present mice:', roc_df['mouse_id'].unique(), 'Number of mice', roc_df['mouse_id'].nunique(), 'per reward group',
          roc_df.groupby('reward_group')['mouse_id'].nunique())
    print('ROC analysis types:', roc_df['analysis_type'].unique())
    excluded_mice = []
    roc_df = roc_df[~roc_df['mouse_id'].isin(excluded_mice)]
    print("Done.")


    #Creating area columns 
    print("\nCreating swanson area column...")
    roc_df = swanson_conversion(roc_df)
    #else defaults to ccf_parent_acronym 
    print("Done.")


    #Filtering data 
    print("\nFiltering roc data...")
    roc_df = filter_number_of_neurons(roc_df, thres, mouse_thres)
    print("Done.")

    #Running dual flatmaps
    print("\nGenerating single and dual hemisphere flatmaps...")
    generate_all_flatmaps(roc_df, color_bar_type, FIGURE_PATH)
    print("Done.")

    #Running delta flatmaps
    print("\nGenerating dual flatmaps...")
    FIGURE_PATH_DELTA_ABS = os.path.join(FIGURE_PATH, "delta_flatmaps", "mean_absolute")
    FIGURE_PATH_DELTA_FRAC = os.path.join(FIGURE_PATH, "delta_flatmaps", "fraction")

    for pair in delta_pairs:
        fig, ax, label = plot_dual_hemispheres_delta(roc_df, pair[0], pair[1], metric='absolute')
        save_flatmap_figure(fig, label, FIGURE_PATH_DELTA_ABS)
        fig, ax, label = plot_dual_hemispheres_delta(roc_df, pair[0], pair[1], metric='fraction')
        save_flatmap_figure(fig, label, FIGURE_PATH_DELTA_FRAC)

    print("Done.")

    print("\nAll flatmaps generate.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot brain flatmaps from ROC results.")

    parser.add_argument(
        '--thres',
        type=int,
        default=15,
        help='Minimum number of neurons required for filtering areas.'
    )

    parser.add_argument(
        '--mouse_thres',
        type=int,
        default=3,
        help='Minimum number of mice required for filtering areas.'
    ) #make 0 if no mouse-area filtering 

    parser.add_argument(
        '--color_bar_type',
        type=str,
        default= 'one_color_bar',
        choices=['one_color_bar', 'reward_group_colorbar', 'analysis_colorbar'],
        help='Number of color bars displayed in dual flatmaps.'
    ) 

    ## one_color_bar: always a single colorbar for all plots

    # reward_group_colorbar: 
    #   - two colorbars only when comparing R+ vs R- within the same analysis
    #   - otherwise one colorbar
    #   - color depends on reward group (red for R-, green for R+)

    # analysis_colorbar: 
    #   - use two colorbars for all analyses 
    #   - except for sign comparison analyses, where only one colorbar is used


    args = parser.parse_args()

    print('Creating Flatmap Plots...')
    main(args)
    print('Done.')


#  python brain_flatmaps.py --thres 3 --mouse_thres 0 --thres 15 --color_bar_type one_color_bar 
