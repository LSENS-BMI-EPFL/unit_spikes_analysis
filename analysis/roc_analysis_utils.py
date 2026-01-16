#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_spikes_analysis
@file: roc_analysis_utils.py
@time: 9/13/2025 2:10 PM
"""
import os, glob
import numpy as np
import pandas as pd
from tqdm import tqdm

import allen_utils as allen

def load_roc_results(root_path):
    files = glob.glob(os.path.join(root_path, '**', '*_roc_results_new.csv'), recursive=True)
    print(f"  Found {len(files)} files in: {root_path}")
    dfs = []
    for f in tqdm(files, desc="  Loading ROC CSV files", unit="file"):
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"    Skipped corrupted file: {f} ({e})")
    if len(dfs) == 0:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def remove_subjects_without_passive(df):
    """Remove subjects that do not have any passive trials from the dataframe."""
    df['mouse_id_int'] = df['mouse_id'].str.extract(r'(\d+)$').astype(int)

    mask1 = df.mouse_id.str.startswith('AB')
    mask2 = df.mouse_id_int < 116
    filtered_df = df[~(mask1 & mask2)].copy()
    return filtered_df


def fdr_bh(pvals, fdr=0.05):
    """
    Benjamini-Hochberg FDR correction for multiple comparisons.
    :param pvals: list or array of p-values
    :param fdr: desired false discovery rate (default 0.05)
    :return: array of boolean values indicating which hypotheses are rejected,
                array of adjusted p-values
    """
    pvals = np.array(pvals)
    n = len(pvals)
    if n == 0:
        return np.array([]), np.array([])

    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(pvals)
    sorted_pvals = pvals[sorted_indices]

    # Compute adjusted p-values
    adjusted = sorted_pvals * n / (np.arange(1, n + 1))

    # Ensure monotonicity
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)  # Bound between 0 and 1

    # Map back to original order
    pvals_corrected = np.empty_like(adjusted)
    pvals_corrected[sorted_indices] = adjusted

    # Determine which are significant
    reject = pvals_corrected <= fdr

    return reject, pvals_corrected


def filter_process_data(data_df, n_units_min=10, n_mice_per_area_min=3, keep_shared=True):
    """
    Filter input dataframe of results based on criteria.
    :param data_df: input dataframe with ROC results
    :param n_units_min: minimum number of units per area to keep (over whole dataset)
    :param n_mice_per_area_min: minimum number of mice per area to keep
    :param keep_shared: if True, only keep areas that are present in both reward groups
    """

    data_df['selectivity_abs'] = data_df['selectivity'].abs()

    # Step 1: Add custom area column
    data_df = allen.create_area_custom_column(data_df)

    # Step 2: Count occurrences and filter by threshold
    area_counts_by_analysis = data_df.groupby(['analysis_type', 'area_acronym_custom']).size()
    valid_areas = area_counts_by_analysis[area_counts_by_analysis >= n_units_min].index
    data_df = data_df[data_df.set_index(['analysis_type', 'area_acronym_custom']).index.isin(valid_areas)]

    # Step 3: Minimum number of mice per area
    mouse_counts_by_area = data_df.groupby(['analysis_type', 'area_acronym_custom'])['mouse_id'].nunique()
    valid_areas = mouse_counts_by_area[mouse_counts_by_area >= n_mice_per_area_min].index
    filtered_df = data_df[data_df.set_index(['analysis_type', 'area_acronym_custom']).index.isin(valid_areas)]

    # Step 4: Identify shared areas between R+ and R-
    if keep_shared:
        rplus_areas = filtered_df[filtered_df.reward_group == "R+"]['area_acronym_custom'].unique()
        rmins_areas = filtered_df[filtered_df.reward_group == "R-"]['area_acronym_custom'].unique()
        shared_areas = list(set(rplus_areas).intersection(rmins_areas))
    else:
        shared_areas = filtered_df['area_acronym_custom'].unique()

    # Number of areas per reward group
    n_rplus_areas = filtered_df[filtered_df.reward_group == "R+"]['area_acronym_custom'].nunique()
    n_rmins_areas = filtered_df[filtered_df.reward_group == "R-"]['area_acronym_custom'].nunique()

    # Print summary to console for quick inspection
    print(f"Number of areas in R+: {n_rplus_areas}, R-: {n_rmins_areas}, Shared: {len(shared_areas)}")
    print("Shared areas:", shared_areas)

    return filtered_df


def compute_prop_significant(roc_df, per_subject=True):
    """
    Compute proportions of significant neurons per area, analysis type, reward group, - and direction,
    i.e. over the entire dataset aggregated over mice.
    """
    if per_subject:
        default_groups = ['mouse_id', 'analysis_type', 'reward_group', 'area_acronym_custom']
        print("Computing proportions per subject...")
    else:
        default_groups = ['analysis_type', 'reward_group', 'area_acronym_custom']
        print("Computing proportions over all subjects...")

    # Step 1: Total neuron counts per group
    total_neurons_per_group = (
        roc_df.groupby(default_groups)
        .size()
        .reset_index(name='total_count')
    )

    # Step 2: Count selective neurons (significant == True) by direction
    selective_counts = (
        roc_df[roc_df['significant']]
        .groupby(default_groups + ['direction'])
        .size()
        .reset_index(name='count')
    )

    # Build all possible group combinations based on unique values in the data
    if per_subject:
        all_combinations = pd.MultiIndex.from_product(
            [
                roc_df['mouse_id'].unique(),
                roc_df['analysis_type'].unique(),
                roc_df['reward_group'].unique(),
                roc_df['area_acronym_custom'].unique(),
                roc_df['direction'].unique()
            ],
            names=default_groups + ['direction']
        )
    else:
        all_combinations = pd.MultiIndex.from_product(
            [
                roc_df['analysis_type'].unique(),
                roc_df['reward_group'].unique(),
                roc_df['area_acronym_custom'].unique(),
                roc_df['direction'].unique()
            ],
            names=default_groups + ['direction']
        )

    # Reindex the grouped data to include missing combinations (e.g. 0% positively-mod. units), fill them with 0
    selective_counts = (
        selective_counts.set_index(default_groups + ['direction'])
        .reindex(all_combinations, fill_value=0)
        .reset_index()
    )

    # Step 3: Count non-selective neurons (significant == False)
    non_selective_counts = (
        roc_df[~roc_df['significant']]
        .groupby(default_groups)
        .size()
        .reset_index(name='count')
    )
    non_selective_counts['direction'] = 'non-selective' # add own label

    # Step 4: Combine selective and non-selective counts
    roc_df_perc = pd.concat([selective_counts, non_selective_counts], ignore_index=True)

    # Step 5: Merge with total neuron counts to calculate proportions
    roc_df_perc = roc_df_perc.merge(
        total_neurons_per_group,
        on=default_groups
    )
    roc_df_perc['proportion'] = np.where(
        (roc_df_perc['count'] > 0),
        (roc_df_perc['count'] / roc_df_perc['total_count']) * 100,
        0       # 0% if no significant units
    )

    # Step 6: Make a total-significant column, summing both directions positive/negative, or auditory/whisker
    roc_df_perc['proportion_all'] = roc_df_perc['proportion']  # init. with current proportions for non-slective

    # Subset for ROC significance is positive or negative modulation
    mask_direction = roc_df_perc['direction'].isin(['positive', 'negative'])
    roc_df_perc.loc[mask_direction, 'proportion_all'] =roc_df_perc.groupby(default_groups
    )['proportion'].transform(lambda x: x[mask_direction].sum())
    mask_direction_non = roc_df_perc['direction'].isin(['non-selective'])
    roc_df_perc.loc[mask_direction_non, 'proportion_all'] = roc_df_perc.groupby(default_groups
    )['proportion'].transform(lambda x: x[mask_direction_non].sum())

    # Subset for ROC significance is auditory or whisker trials
    mask_modality = roc_df_perc['direction'].isin(['whisker', 'auditory']) & roc_df_perc['analysis_type'].str.contains('wh_vs_aud')
    roc_df_perc.loc[mask_modality, 'proportion_all'] = roc_df_perc.groupby(default_groups
    )['proportion'].transform(lambda x: x[mask_modality].sum())
    mask_modality_non = roc_df_perc['direction'].isin(['non-selective']) & roc_df_perc['analysis_type'].str.contains('wh_vs_aud')
    roc_df_perc.loc[mask_modality_non, 'proportion_all'] = roc_df_perc.groupby(default_groups
    )['proportion'].transform(lambda x: x[mask_modality_non].sum())

    # Step 7: Create signed proportions for plotting (bar above/below x-axis)
    roc_df_perc['proportion_signed'] = roc_df_perc['proportion']
    roc_df_perc.loc[roc_df_perc['direction'] == 'negative', 'proportion_signed'] *= -1
    roc_df_perc.loc[roc_df_perc['direction'] == 'auditory', 'proportion_signed'] *= -1

    return roc_df_perc

def compute_si_differences(roc_df):
    """Compute difference in selectivity index (SI) between pre- and post-learning passive trials."""

    results = roc_df.copy()

    # Keep analysis types relevant i.e. those with pre and post during passive
    results = results[results['analysis_type'].str.contains('passive')]

    # Some mice lack passive data (either pre or post, or both): drop these nans
    mice_with_incomplete_passive = ['MH013', 'MH038'] #TODO: potentially update with AB155
    results = results[~results['mouse_id'].isin(mice_with_incomplete_passive)]

    for anal_type in ['whisker_passive', 'auditory_passive', 'wh_vs_aud_passive']:
        # Filter for only the relevant analysis type
        df = roc_df[roc_df['analysis_type'].isin([f'{anal_type}_pre', f'{anal_type}_post'])].copy()

        # Pivot so each neuron has pre and post SI in separate columns
        pivot_df = df.pivot_table(
            index=['mouse_id', 'unit_id', 'area_acronym_custom', 'reward_group'],
            columns='analysis_type',
            values='selectivity'
        ).reset_index()

        # Ensure columns exist even if missing
        pivot_df = pivot_df.rename(
            columns={
                f'{anal_type}_pre': 'si_pre',
                f'{anal_type}_post': 'si_post'
            }
        )

        # Compute delta SI with different signs based on the SI post sign:
        # The reason for using signed differences like this is to normalize the direction of
        # selectivity so that the computed ΔSI always reflects changes in the preferred direction,
        # regardless of whether the neuron prefers condition A or condition B.
        delta_si = np.where(
            pivot_df['si_post'] >= 0,
            pivot_df['si_post'] - pivot_df['si_pre'],  # Positive SI post
            -(pivot_df['si_post'] - pivot_df['si_pre'])  # Negative SI post
        )

        # Create a unique column name for this analysis type
        delta_col = f"delta_si_{anal_type}"
        pivot_df[delta_col] = delta_si

        # Merge back into the main dataframe
        results = results.merge(
            pivot_df[['mouse_id', 'unit_id', delta_col]],
            on=['mouse_id', 'unit_id'],
            how='left'
        )

    return results
