#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_spikes_analysis
@file: run_roc_analysis.py
@time: 9/8/2025 4:06 PM
"""


# Imports
import os
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mc
from scipy.stats import wilcoxon, ttest_rel, pearsonr
#from statannotations.Annotator import Annotator


import allen_utils as allen
import plotting_utils as putils

DATA_PATH = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis')
FIGURE_PATH = r'M:\analysis\Axel_Bisi\combined_results\roc_analysis'

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
    else:
        default_groups = ['analysis_type', 'reward_group', 'area_acronym_custom']

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
    #roc_df_perc['proportion'] = (roc_df_perc['count'] / roc_df_perc['total_count']) * 100
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

def plot_proportion_across_areas(data_df, area_order, area_color_list, output_path):
    """
    Plot proportions of significant neurons per area, analysis type for each reward group separately.
    Plots for all significant neurons and per directions of significance.
    :param data_df: dataframe with proportions of significant neurons per area
    :param area_order: list of areas in desired order
    :param area_color_list: list of colors corresponding to areas
    :param output_path: path to save figures
    """
    for reward_group in ['R+', 'R-']:
        for anal_type in data_df['analysis_type'].unique():
            subset_df = data_df[
                (data_df['reward_group'] == reward_group) &
                (data_df['analysis_type'] == anal_type)
                ]

            for dir in ['all', 'dir']:
                if dir == 'all':
                    if 'wh_vs_aud' in anal_type:
                        dir_list = ['whisker']
                    else:
                        dir_list = ['positive']
                    palette = area_color_list
                elif dir == 'dir':
                    if 'wh_vs_aud' in anal_type:
                        dir_list = ['whisker', 'auditory']
                        hue_order= dir_list
                        palette = ['forestgreen', 'mediumblue']
                    else:
                        dir_list = ['positive', 'negative']
                        hue_order= dir_list
                        palette = ['tomato', 'dodgerblue']
                subset_df_dir = subset_df[subset_df['direction'].isin(dir_list)] # subset directions to plot

                y_val = 'proportion_all' if dir == 'all' else 'proportion_signed'
                hue = None if dir == 'all' else 'direction'
                suffix = dir


                g = sns.catplot(
                    data=subset_df_dir,
                    kind='bar',
                    x='area_acronym_custom',
                    y=y_val,
                    hue=hue,
                    hue_order=hue_order if dir == 'dir' else None,
                    order=area_order,
                    palette=palette,
                    height=2.5,
                    aspect=4,
                    errorbar=None,
                    legend=False,
                    dodge=False
                )
                g.figure.suptitle(anal_type)
                g.despine(left=False)
                g.set_axis_labels('', 'Proportion (%)')
                if dir == 'all':
                    #g.ax.set_ylim(0, 100)
                    g.set(ylim=(0, 70))
                else:
                    #g.ax.set_ylim(-100, 100)
                    g.set(ylim=(-50, 50))
                g.tight_layout()
                g.set_xticklabels(rotation=90)
                g.tick_params(labelsize=12)

                # Save
                figname = f'brainwide_roc_{reward_group}_{anal_type}_{suffix}'
                putils.save_figure_with_options(plt.gcf(), ['png', 'pdf', 'svg'], figname, output_dir=output_path)
                plt.close()
    return

def plot_pop_selectivity_across_areas(data_df, per_subject, area_order, area_color_list, output_path):
    """
    Plot proportions of significant neurons per area, analysis type for each reward group separately.
    Plots for all significant neurons and per directions of significance.
    :param data_df: dataframe with proportions of significant neurons per area
    :param area_order: list of areas in desired order
    :param area_color_list: list of colors corresponding to areas
    :param output_path: path to save figures
    """
    if per_subject:
        # Average values per subject first
        data_df = data_df.groupby(['mouse_id', 'analysis_type', 'reward_group', 'area_acronym_custom'], as_index=False)[
            'selectivity_abs'].mean()
        errorbar='se'
        errwidth=0.7
    else:
        errorbar=None
        errwidth=None


    for reward_group in ['R+', 'R-']:
        for anal_type in data_df['analysis_type'].unique():
            subset_df = data_df[
                (data_df['reward_group'] == reward_group) &
                (data_df['analysis_type'] == anal_type)
                ]

            y_val = 'selectivity_abs'

            g = sns.catplot(
                data=subset_df,
                kind='bar',
                x='area_acronym_custom',
                y=y_val,
                hue=None,
                hue_order=None,
                order=area_order,
                palette=area_color_list,
                height=2.5,
                aspect=4,
                errorbar=errorbar,
                errwidth=errwidth,
                legend=False,
                dodge=False
            )
            g.figure.suptitle(anal_type)
            g.despine(left=False)
            g.set_axis_labels('', 'Absolute selectivity')
            g.set(ylim=(0, 0.4))
            g.tight_layout()
            g.set_xticklabels(rotation=90)
            g.tick_params(labelsize=12)

            # Save
            figname = f'brainwide_roc_abs_si_{reward_group}_{anal_type}'
            putils.save_figure_with_options(plt.gcf(), ['png', 'pdf', 'svg'], figname, output_dir=output_path)
            plt.close()
    return

def plot_proportion_across_areas_reward_group(data_df, area_order, area_color_list, output_path):
    """
    Plot proportions of significant neurons per area, analysis type, and per reward group.
    Plots for all significant neurons and per directions of significance.
    :param data_df: dataframe with proportions of significant neurons per area
    :param area_order: list of areas in desired order
    :param area_color_list: list of colors corresponding to areas
    :param output_path: path to save figures
    """
    for anal_type in data_df['analysis_type'].unique():
        subset_df = data_df[(data_df['analysis_type'] == anal_type)]

        for dir in ['all', 'dir_1', 'dir_2']:
            if dir == 'all':
                if 'wh_vs_aud' in anal_type:
                    dir_list = ['whisker']
                else:
                    dir_list = ['positive']
            elif dir == 'dir_1':
                if 'wh_vs_aud' in anal_type:
                    dir_list = ['whisker']
                else:
                    dir_list = ['positive']
            elif dir == 'dir_2':
                if 'wh_vs_aud' in anal_type:
                    dir_list = ['auditory']
                else:
                    dir_list = ['negative']
            subset_df_dir = subset_df[subset_df['direction'].isin(dir_list)] # subset directions to plot

            y_val = 'proportion_all' if dir == 'all' else 'proportion'
            suffix = dir

            g = sns.catplot(
                data=subset_df_dir,
                kind='bar',
                x='area_acronym_custom',
                y=y_val,
                hue='reward_group',
                hue_order=['R+','R-'],
                order=area_order,
                palette=['forestgreen', 'crimson'],
                height=2.5,
                aspect=4,
                errorbar=None,
                legend=False
            )
            g.figure.suptitle(anal_type)
            g.despine(left=False)
            g.set_axis_labels('', 'Proportion (%)')
            g.set(ylim=(0, 70))
            g.tight_layout()
            g.set_xticklabels(rotation=90)
            g.tick_params(labelsize=12)

            # Save
            figname = f'brainwide_roc_{anal_type}_{suffix}'
            putils.save_figure_with_options(plt.gcf(), ['png', 'pdf', 'svg'], figname, output_dir=output_path)
            plt.close()
    return

def plot_pop_selectivity_across_areas_reward_group(data_df, per_subject, area_order, area_color_list, output_path):
    """
    Plot proportions of significant neurons per area, analysis type, and per reward group.
    Plots for all significant neurons and per directions of significance.
    :param data_df: dataframe with proportions of significant neurons per area
    :param area_order: list of areas in desired order
    :param area_color_list: list of colors corresponding to areas
    :param output_path: path to save figures
    """
    if per_subject:
        # Average values per subject first
        data_df = data_df.groupby(['mouse_id', 'analysis_type', 'reward_group', 'area_acronym_custom'], as_index=False)[
            'selectivity_abs'].mean()
        errorbar='se'
        errwidth=0.7
    else:
        errorbar=None
        errwidth=None


    for anal_type in data_df['analysis_type'].unique():
        subset_df = data_df[(data_df['analysis_type'] == anal_type)]

        y_val = 'selectivity_abs'

        g = sns.catplot(
            data=subset_df,
            kind='bar',
            x='area_acronym_custom',
            y=y_val,
            hue='reward_group',
            hue_order=['R+','R-'],
            order=area_order,
            palette=['forestgreen', 'crimson'],
            height=2.5,
            aspect=4,
            errorbar=errorbar,
            errwidth=errwidth,
            legend=False
        )
        g.figure.suptitle(anal_type)
        g.despine(left=False)
        g.set_axis_labels('', 'Absolute selectivity')
        g.set(ylim=(0, 0.4))
        g.tight_layout()
        g.set_xticklabels(rotation=90)
        g.tick_params(labelsize=12)

        # Save
        figname = f'brainwide_roc_abs_si_{anal_type}'
        putils.save_figure_with_options(plt.gcf(), ['png', 'pdf', 'svg'], figname, output_dir=output_path)
        plt.close()
    return

def plot_prop_before_vs_after_across_areas(data_df, area_order, area_color_list, output_path):
    """
    Plot proportions of significant neurons per area, analysis type, for each reward separately found in
    pre-learning passive trials vs. proportions in the post-learning passive trials.
    Plots for all significant neurons only.
    :param data_df: dataframe with proportions of significant neurons per area
    :param area_order: list of areas in desired order
    :param area_color_list: list of colors corresponding to areas
    :param output_path: path to save figures
    """
    data_df = data_df[data_df['analysis_type'].str.contains('passive')]
    for reward_group in ['R+', 'R-']:
        data_df_group = data_df[data_df['reward_group'] == reward_group]

        if reward_group == 'R+':
            palette = [putils.adjust_lightness('forestgreen', 1.3),
                       putils.adjust_lightness('forestgreen', 0.7)]
        else:
            palette = [putils.adjust_lightness('crimson', 1.3),
                       putils.adjust_lightness('crimson', 0.7)]

        for stim_type in ['auditory', 'whisker']:

            # Pre-learning ROC proportions
            roc_df_npre = data_df_group[
                data_df_group.analysis_type == '{}_passive_pre'.format(stim_type)]
            roc_df_npre['direction'] = roc_df_npre['direction'].replace(
                {'positive': 'both', 'negative': 'both'})  # replace cat name to 'both'
            roc_df_npre = \
            roc_df_npre.groupby(['mouse_id', 'analysis_type', 'reward_group', 'area_acronym_custom', 'direction'], as_index=False)[
                'proportion'].sum()  # sum 'both' values
            roc_df_npre['period'] = 'pre-learning'

            # Post-learning ROC proportions
            roc_df_npost = data_df_group[
                data_df_group.analysis_type == '{}_passive_post'.format(stim_type)]
            roc_df_npost['direction'] = roc_df_npost['direction'].replace(
                {'positive': 'both', 'negative': 'both'})  # replace cat name to 'both'
            roc_df_npost = \
            roc_df_npost.groupby(['mouse_id', 'analysis_type', 'reward_group', 'area_acronym_custom', 'direction'], as_index=False)[
                'proportion'].sum()  # sum 'both' values
            roc_df_npost['period'] = 'post-learning'

            # Combine
            roc_pre_vs_post_df = pd.concat([roc_df_npre, roc_df_npost])
            roc_pre_vs_post_df.reset_index(inplace=True)

            # Plot pre. vs post
            df = roc_pre_vs_post_df[roc_pre_vs_post_df.direction == 'both']
            g = sns.catplot(
                data=df,
                kind='bar',
                x='area_acronym_custom',
                y='proportion',
                order=area_order,
                hue='period',
                hue_order=['pre-learning', 'post-learning'],
                palette=palette,
                height=2.5,
                aspect=4,
                errorbar='se',
                errwidth=0.7,
                legend=True,
                legend_out=True,
                seed=42
            )
            g.despine(left=False)
            g.set_axis_labels('', 'Proportion (%)')
            g.set(ylim=(0, 100))
            g.tight_layout()
            g.set_xticklabels(rotation=90)

            # Run statistical test for each area
            results = []
            for area in area_order:
                sub_df = df[df['area_acronym_custom'] == area]

                # Pivot so each row is a mouse, columns are pre and post
                pivot_df = sub_df.pivot_table(
                    index='mouse_id',
                    columns='period',
                    values='proportion' #Note: includes all directions of significance
                )

                # Skip if not enough mice
                if pivot_df.shape[0] < 3:
                    continue
                if 'pre-learning' not in pivot_df.columns or 'post-learning' not in pivot_df.columns:
                    continue

                # Test (paired)
                #stat, p = wilcoxon(pivot_df['pre-learning'], pivot_df['post-learning'], alternative="two-sided")
                stat, p = ttest_rel(pivot_df['pre-learning'], pivot_df['post-learning'], alternative="two-sided", nan_policy='omit')
                results.append({'area': area, 'statistic': stat, 'p_value': p})

            results_df = pd.DataFrame(results)
            if not results_df.empty:
                reject, p_corrected = fdr_bh(results_df['p_value'], fdr=0.05)
                results_df['p_corrected'] = p_corrected
                results_df['significant'] = reject
            else:
                results_df['p_corrected'] = []
                results_df['significant'] = []

            print('ROC statistical tests: \n', results_df)

            # Add annotations for significant areas
            ax = g.ax
            y_max = 80
            offset = y_max * 0.05  # spacing for annotation

            for i, area in enumerate(area_order):
                if results_df.loc[results_df['area'] == area, 'significant'].any():
                    p_val = results_df.loc[results_df['area'] == area, 'p_corrected'].values[0]

                    # Mark significance level
                    if p_val < 0.001:
                        text = '***'
                    elif p_val < 0.01:
                        text = '**'
                    elif p_val < 0.05:
                        text = '*'
                    else:
                        text = 'ns'

                    # Position annotation above the bars
                    ax.text(
                        i,
                        y_max + offset,
                        text,
                        ha='center',
                        va='top',
                        fontsize=12,
                        color='black'
                    )

            # Save
            figname = f'brainwide_roc_{reward_group}_{stim_type}_prop_pre_vs_prop_post'
            putils.save_figure_with_options(plt.gcf(), ['png', 'pdf', 'svg'], figname, output_dir=output_path)

    return

def plot_pop_selectivity_before_vs_after_across_areas(data_df, per_subject, area_order, area_color_list, output_path):
    """
    Plot proportions of significant neurons per area, analysis type, for each reward separately found in
    pre-learning passive trials vs. proportions in the post-learning passive trials.
    Plots for all significant neurons only.
    :param data_df: dataframe with proportions of significant neurons per area
    :param area_order: list of areas in desired order
    :param area_color_list: list of colors corresponding to areas
    :param output_path: path to save figures
    """

    if per_subject:
        # Average values per subject first
        data_df = data_df.groupby(['mouse_id', 'analysis_type', 'reward_group', 'area_acronym_custom'], as_index=False)[
            'selectivity_abs'].mean()
        errorbar='se'
        errwidth=0.7
    else:
        errorbar=None
        errwidth=None

    data_df = data_df[data_df['analysis_type'].str.contains('passive')]
    for reward_group in ['R+', 'R-']:
        data_df_group = data_df[data_df['reward_group'] == reward_group]

        if reward_group == 'R+':
            palette = [putils.adjust_lightness('forestgreen', 1.3),
                       putils.adjust_lightness('forestgreen', 0.7)]
        else:
            palette = [putils.adjust_lightness('crimson', 1.3),
                       putils.adjust_lightness('crimson', 0.7)]

        for stim_type in ['auditory', 'whisker']:

            # Pre-learning ROC proportions
            roc_df_npre = data_df_group[
                data_df_group.analysis_type == '{}_passive_pre'.format(stim_type)]
            roc_df_npre['period'] = 'pre-learning'

            # Post-learning ROC proportions
            roc_df_npost = data_df_group[
                data_df_group.analysis_type == '{}_passive_post'.format(stim_type)]
            roc_df_npost['period'] = 'post-learning'

            # Combine
            df = pd.concat([roc_df_npre, roc_df_npost])
            df.reset_index(inplace=True)

            # Plot pre. vs post
            g = sns.catplot(
                data=df,
                kind='bar',
                x='area_acronym_custom',
                y='selectivity_abs',
                order=area_order,
                hue='period',
                hue_order=['pre-learning', 'post-learning'],
                palette=palette,
                height=2.5,
                aspect=4,
                errorbar=errorbar,
                errwidth=errwidth,
                legend=True,
                legend_out=True,
                seed=42
            )
            # Connect each bar value with a line
            for i, area in enumerate(area_order):
                mask_pre = (df['area_acronym_custom'] == area) & (df['period'] == 'pre-learning')
                mask_post = (df['area_acronym_custom'] == area) & (df['period'] == 'post-learning')
                pre_val = df.loc[mask_pre, 'selectivity_abs'].mean()
                post_val = df.loc[mask_post, 'selectivity_abs'].mean()
                plt.plot([i - 0.2, i + 0.2], [pre_val, post_val], color='gray', linewidth=0.5, alpha=1)

            g.despine(left=False)
            g.set_axis_labels('', 'Absolute selectivity')
            g.set(ylim=(0, 0.4))
            g.tight_layout()
            g.set_xticklabels(rotation=90)

            # Run statistical test for each area
            results = []
            for area in area_order:
                sub_df = df[df['area_acronym_custom'] == area]

                # Pivot so each row is a mouse, columns are pre and post
                pivot_df = sub_df.pivot_table(
                    index='mouse_id',
                    columns='period',
                    values='selectivity_abs'
                )

                # Skip if not enough mice
                if pivot_df.shape[0] < 3:
                    continue
                if 'pre-learning' not in pivot_df.columns or 'post-learning' not in pivot_df.columns:
                    continue

                # Test (paired)
                #stat, p = wilcoxon(pivot_df['pre-learning'], pivot_df['post-learning'], alternative="two-sided")
                stat, p = ttest_rel(pivot_df['pre-learning'], pivot_df['post-learning'], alternative="two-sided", nan_policy='omit')
                results.append({'area': area, 'statistic': stat, 'p_value': p})

            results_df = pd.DataFrame(results)
            if not results_df.empty:
                reject, p_corrected = fdr_bh(results_df['p_value'], fdr=0.05)
                results_df['p_corrected'] = p_corrected
                results_df['significant'] = reject
            else:
                results_df['p_corrected'] = []
                results_df['significant'] = []

            print('ROC statistical tests: \n', results_df)

            # Add annotations for significant areas
            ax = g.ax
            y_max = 0.4
            offset = y_max * 0.05  # spacing for annotation

            for i, area in enumerate(area_order):
                if results_df.loc[results_df['area'] == area, 'significant'].any():
                    p_val = results_df.loc[results_df['area'] == area, 'p_corrected'].values[0]

                    # Mark significance level
                    if p_val < 0.001:
                        text = '***'
                    elif p_val < 0.01:
                        text = '**'
                    elif p_val < 0.05:
                        text = '*'
                    else:
                        text = 'ns'

                    # Position annotation above the bars
                    ax.text(
                        i,
                        y_max + offset,
                        text,
                        ha='center',
                        va='top',
                        fontsize=12,
                        color='black'
                    )

            # Save
            figname = f'brainwide_roc_abs_si_{reward_group}_{stim_type}_prop_pre_vs_prop_post'
            putils.save_figure_with_options(plt.gcf(), ['png', 'pdf', 'svg'], figname, output_dir=output_path)

    return

def plot_proportion_across_areas_pre_vs_post(data_df, area_order, area_color_list, output_path):
    """
    Plot proportions of significant neurons per area, analysis type, reward group that are modulated comparing pre and post passive trials.
    Plots for per directions of significance.
    :param data_df: dataframe with proportions of significant neurons per area
    :param area_order: list of areas in desired order
    :param area_color_list: list of colors corresponding to areas
    :param output_path: path to save figures
    """
    for reward_group in ['R+', 'R-']:
        for stim_type in ['auditory', 'whisker']:
            data_df_group = data_df[(data_df['reward_group'] == reward_group)
                            &
                (data_df['analysis_type'].str.contains(f'{stim_type}_pre_vs_post_learning'))]

            g = sns.catplot(
                data=data_df_group,
                kind='bar',
                x='area_acronym_custom',
                y='proportion_signed',
                order=area_order,
                hue='direction',
                hue_order=['positive', 'negative'],
                palette=['tomato', 'dodgerblue'],
                height=2.5,
                aspect=4,
                errorbar='se',
                errwidth=0.7,
                legend=False,
                #legend_out=True,
                dodge=False,
                seed=42
            )
            g.despine(left=False)
            g.set_axis_labels('', 'Proportion (%)')
            g.set(ylim=(-70, 70))
            g.tight_layout()
            g.set_xticklabels(rotation=90)

            # Save
            figname = f'brainwide_roc_{reward_group}_{stim_type}_pre_vs_post'
            putils.save_figure_with_options(plt.gcf(), ['png', 'pdf', 'svg'], figname, output_dir=output_path)

    return


def plot_si_delta_correlation_grid_across_areas(diff_df, cond1, cond2, output_path):
    """
    Plot correlation of delta SI between two conditions across areas in a grid layout.
    """
    # Plotting params
    nrows, ncols = 6, 10
    point_alpha = 0.5

    # Keep data with passive trials
    diff_df = remove_subjects_without_passive(diff_df)

    for reward_group in ['R+', 'R-']:
        subset_main = diff_df[
            (diff_df['reward_group'] == reward_group)
        ]

        # Get areas present in this subset
        areas = sorted(subset_main['area_acronym_custom'].unique())
        total_plots = len(areas)
        total_slots = nrows * ncols

        if total_plots > total_slots:
            print(f"Warning: More areas ({total_plots}) than slots ({total_slots}). "
                  "Some areas will not be shown.")

        # Create figure
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows), dpi=120)
        axes = axes.flatten()

        # Plot per area
        for idx, area in enumerate(areas):
            ax = axes[idx]
            putils.remove_top_right_frame(ax)
            subset_area = subset_main[subset_main['area_acronym_custom'] == area]

            if subset_area.empty:
                ax.axis('off')
                continue

            # Scatter plot of SI values for two condition
            x_val = f'delta_si_{cond1}'
            y_val = f'delta_si_{cond2}'
            sns.scatterplot(data=subset_area,
                            ax=ax,
                            x=x_val,
                            y=y_val,
                            color='gray',
                            alpha=point_alpha,
                            edgecolor=None,
                            s=2
                            )

            # Add reference lines
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
            ax.plot([-2, 2], [-2, 2], color='grey', linestyle='--', linewidth=0.8)
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_xticks(range(-2, 3, 1))
            ax.set_yticks(range(-2, 3, 1))
            ax.set_xlabel(r'$\Delta$SI: whisker', fontsize=10)
            ax.set_ylabel(r'$\Delta$SI: auditory', fontsize=10)

            # Compute Pearson correlation
            r_val, p_val = pearsonr(subset_area[x_val].values, subset_area[y_val].values)
            title = f"{area}\nr={r_val:.2f}, p={p_val:.2f}"
            ax.set_title(title, fontsize=10)


        # Hide extra axes
        for j in range(total_plots, total_slots):
            axes[j].axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save figure
        figname = f'si_delta_correlation_{reward_group}_{cond1}_vs_{cond2}'
        putils.save_figure_with_options(plt.gcf(), ['png', 'pdf', 'svg'], figname, output_dir=output_path)

    return

def plot_si_correlation_grid_across_areas(data_df, cond1, cond2, output_path):

    # Plotting params
    nrows, ncols = 6, 10
    point_alpha = 0.5

    # Keep only data with passive trials
    data_df = remove_subjects_without_passive(data_df)

    for reward_group in ['R+', 'R-']:

        subset_main = data_df[
            (data_df['reward_group'] == reward_group)
        ]
        # Pivot subset_area on analysis_type such that cond1 and cond2 are columns
        subset_main = subset_main.pivot_table(
            index=['mouse_id', 'unit_id', 'area_acronym_custom', 'reward_group'],
            columns='analysis_type',
            values='selectivity'
        ).reset_index()

        # Remove any NaNs
        subset_main = subset_main.dropna(subset=[cond1, cond2])

        # Get areas present in this subset
        areas = sorted(subset_main['area_acronym_custom'].unique())
        total_plots = len(areas)
        total_slots = nrows * ncols

        if total_plots > total_slots:
            print(f"Warning: More areas ({total_plots}) than slots ({total_slots}). "
                  "Some areas will not be shown.")

        # Create figure
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows), dpi=120)
        axes = axes.flatten()

        # Plot per area
        for idx, area in enumerate(areas):
            ax = axes[idx]
            putils.remove_top_right_frame(ax)
            subset_area = subset_main[subset_main['area_acronym_custom'] == area]

            if subset_area.empty:
                ax.axis('off')
                continue

            # Scatter plot of SI values for two condition
            x_val = f'{cond1}'
            y_val = f'{cond2}'
            sns.scatterplot(data=subset_area,
                            ax=ax,
                            x=x_val,
                            y=y_val,
                            color='gray',
                            alpha=point_alpha,
                            edgecolor=None,
                            s=2
                            )

            # Add reference lines
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
            ax.plot([-2, 2], [-2, 2], color='grey', linestyle='--', linewidth=0.8)
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_xticks(range(-2, 3, 1))
            ax.set_yticks(range(-2, 3, 1))
            txt_x = ' '.join(cond1.split('_'))
            txt_y = ' '.join(cond2.split('_'))
            ax.set_xlabel(f'SI: {txt_x}', fontsize=10)
            ax.set_ylabel(f'SI: {txt_y}', fontsize=10)

            # Compute Pearson correlation
            r_val, p_val = pearsonr(subset_area[x_val].values, subset_area[y_val].values)
            title = f"{area}\nr={r_val:.2f}, p={p_val:.2f}"
            ax.set_title(title, fontsize=10)

        # Hide extra axes
        for j in range(total_plots, total_slots):
            axes[j].axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save figure
        figname = f'si_delta_correlation_{reward_group}_{cond1}_vs_{cond2}'
        putils.save_figure_with_options(plt.gcf(), ['png', 'pdf', 'svg'], figname, output_dir=output_path)

    return


def main():

    info_path = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'z_LSENS', 'Share', f'Axel_Bisi_Share',
                             'dataset_info')
    mouse_info_path = os.path.join(info_path, 'joint_mouse_reference_weight.xlsx')
    mouse_info_df = pd.read_excel(mouse_info_path)
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)

    # Filter for usable mice
    mouse_info_df = mouse_info_df[
        (mouse_info_df['exclude'] == 0) &
        (mouse_info_df['reward_group'].isin(['R+', 'R-'])) &
        (mouse_info_df['recording'] == 1)
        ]

    # ---------
    # LOAD DATA
    # ---------

    data_path_axel = os.path.join(DATA_PATH, 'Axel_Bisi', 'results') #TODO: update when change
    roc_results_files = glob.glob(os.path.join(data_path_axel, '**', '*_roc_results.csv'),
                                  recursive=True)  # find all roc results files
    roc_df_axel = pd.concat([pd.read_csv(f) for f in roc_results_files], ignore_index=True)
    data_path_myriam = os.path.join(DATA_PATH, 'Myriam_Hamon', 'combined_results')
    roc_results_files = glob.glob(os.path.join(data_path_myriam, '**', '*_roc_results.csv'),
                                            recursive=True)
    roc_df_myriam = pd.concat([pd.read_csv(f) for f in roc_results_files], ignore_index=True)

    roc_df = pd.concat([roc_df_axel, roc_df_myriam], ignore_index=True)
    roc_df = roc_df.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')

    # Create unique unit identifier based on index
    roc_df['neuron_id'] = roc_df.index.astype(int)

    print('Present mice:', roc_df['mouse_id'].unique(), 'Number of mice', roc_df['mouse_id'].nunique(), 'per reward group',
          roc_df.groupby('reward_group')['mouse_id'].nunique())
    print('ROC analysis types:', roc_df['analysis_type'].unique())

    excluded_mice = ['AB077', 'AB082']
    excluded_mice  = []
    roc_df = roc_df[~roc_df['mouse_id'].isin(excluded_mice)]

    # -----------------------
    # PROCESS AND FILTER DATA
    # -----------------------
    roc_df = filter_process_data(roc_df, n_units_min=10, n_mice_per_area_min=3, keep_shared=True)

    # Create color list based on areas that are present
    shared_areas = roc_df['area_acronym_custom'].unique()
    area_order = allen.get_custom_area_order()
    area_order_shared = [a for a in area_order if a in shared_areas]

    # Make a color dict for the group of areas
    area_groups = allen.get_custom_area_groups()

    # Keep areas that present in dataset
    area_groups = {k: [i for i in v if i in area_order_shared] for k, v in area_groups.items()}

    # Generate a colormap with as many colors as the number of area groups
    color_palette_dict = allen.get_custom_area_groups_colors()
    color_palette = list(color_palette_dict.values())

    colors = [color_palette[i % len(color_palette)] for i in range(len(area_groups))]  # hex
    print('Equality ensured', len(colors) == len(area_groups.keys()))

    # Create a dictionary mapping each area to its group color
    area_color_dict = {}
    for (group_name, areas), color in zip(area_groups.items(), colors):
        for area in areas:
            area_color_dict[area] = color
    area_color_list = list(area_color_dict.values())

    # -------------------------------------------------
    # COMPUTE PROPORTIONS OF SIGNIFICANT UNITS PER AREA
    # -------------------------------------------------

    roc_df_perc = compute_prop_significant(roc_df, per_subject=False)
    roc_df_perc_subjects = compute_prop_significant(roc_df, per_subject=True)

    # ----------------------------------------
    # PERFORM COMPARISONS ON SIGNIFICANT UNITS
    # ----------------------------------------

    figures_to_do =[
        #'prop_across_areas',
        #'prop_across_areas_reward_group',
        #'prop_before_vs_after_across_areas',
        #'prop_across_areas_passive_pre_vs_post'
    ]

    if 'prop_across_areas' in figures_to_do:
        output_path = os.path.join(FIGURE_PATH, 'across_areas')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Compare proportions of significant units across areas, for each condition separately
        plot_proportion_across_areas(roc_df_perc, area_order=area_order_shared, area_color_list=area_color_list,
                                     output_path=output_path)

        output_path = os.path.join(FIGURE_PATH, 'across_areas_per_subjects')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Compare proportions of significant units across areas and per subjects, for each condition separately
        plot_proportion_across_areas(roc_df_perc_subjects, area_order=area_order_shared, area_color_list=area_color_list,
                                     output_path=output_path)

    elif 'prop_across_areas_reward_group' in figures_to_do:
        output_path = os.path.join(FIGURE_PATH, 'across_areas_reward_group')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Compare proportions of significant units across areas and per reward group, for each condition separately
        plot_proportion_across_areas_reward_group(roc_df_perc, area_order=area_order_shared, area_color_list=area_color_list,
                                     output_path=output_path)

        output_path = os.path.join(FIGURE_PATH, 'across_areas_reward_group_per_subjects')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Compare proportions of significant units across areas and per reward group and subjects, for each condition separately
        plot_proportion_across_areas_reward_group(roc_df_perc_subjects, area_order=area_order_shared,
                                                  area_color_list=area_color_list,
                                                  output_path=output_path)

    elif 'prop_before_vs_after_across_areas' in figures_to_do:
        output_path = os.path.join(FIGURE_PATH, 'passive_before_vs_after_across_areas')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Compare proportions of significant units across areas before and after learning, for each reward group separately
        plot_prop_before_vs_after_across_areas(roc_df_perc_subjects, area_order=area_order_shared, area_color_list=area_color_list,
                                     output_path=output_path)

    elif 'prop_across_areas_passive_pre_vs_post' in figures_to_do:
        output_path = os.path.join(FIGURE_PATH, 'passive_pre_vs_post_across_areas')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Compare proportions of significant units pre vs post learning, across areas, for each reward group separately
        plot_proportion_across_areas_pre_vs_post(roc_df_perc_subjects, area_order=area_order_shared, area_color_list=area_color_list,
                                     output_path=output_path)

    else:
        print('Functions not implemented.')

    # ---------------------------------------------
    # PERFORM COMPARISONS ON POPULATION SELECTIVITY
    # --------------------------------------------

    figures_to_do = [
        #'abs_si_across_areas',
        #'abs_si_across_areas_reward_group',
        #'abs_si_before_vs_after_across_areas',
    ]

    if 'abs_si_across_areas' in figures_to_do:
        output_path = os.path.join(FIGURE_PATH, 'across_areas_abs_si')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Compare absolute population selectivity, for each condition separately
        plot_pop_selectivity_across_areas(roc_df, per_subject=False, area_order=area_order_shared, area_color_list=area_color_list,
                                     output_path=output_path)

        output_path = os.path.join(FIGURE_PATH, 'across_areas_per_subjects_abs_si')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Compare absolute population selectivity across areas and per subjects, for each condition separately
        plot_pop_selectivity_across_areas(roc_df, per_subject=True, area_order=area_order_shared, area_color_list=area_color_list,
                                     output_path=output_path)

    elif 'abs_si_across_areas_reward_group' in figures_to_do:
        output_path = os.path.join(FIGURE_PATH, 'across_areas_reward_group_abs_si')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Compare absolute population selectivity across areas and per reward group, for each condition separately
        plot_pop_selectivity_across_areas_reward_group(roc_df, per_subject=False, area_order=area_order_shared, area_color_list=area_color_list,
                                     output_path=output_path)

        output_path = os.path.join(FIGURE_PATH, 'across_areas_reward_group_per_subjects_abs_si')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Compare absolute population selectivity across areas and per reward group and subjects, for each condition separately
        plot_pop_selectivity_across_areas_reward_group(roc_df, per_subject=True, area_order=area_order_shared,
                                                  area_color_list=area_color_list,
                                                  output_path=output_path)

    elif 'abs_si_before_vs_after_across_areas' in figures_to_do:
        output_path = os.path.join(FIGURE_PATH, 'passive_before_vs_after_across_areas_abs_si')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Compare absolute population selectivity across areas before and after learning, for each reward group separately
        #plot_pop_selectivity_before_vs_after_across_areas(roc_df, per_subject=False, area_order=area_order_shared, area_color_list=area_color_list,
        #                             output_path=output_path)

        output_path = os.path.join(FIGURE_PATH, 'passive_before_vs_after_across_areas_per_subjects_abs_si')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Compare absolute population selectivity across areas before and after learning, for each reward group separately
        plot_pop_selectivity_before_vs_after_across_areas(roc_df, per_subject=True, area_order=area_order_shared,
                                                          area_color_list=area_color_list,
                                                          output_path=output_path)
    else:
        print('Functions not implemented.')


    # ---------------------------------------------
    # PERFORM CORRELATIONS ON SINGLE-UNITS
    # ---------------------------------------------

    roc_df_delta = compute_si_differences(roc_df)

    figures_to_do = [
        #'si_delta_correlation_grid_across_areas',
        #'si_delta_correlation_grid_across_areas_multimodal',
        'si_correlation_grid_across_areas'
    ]

    if 'si_delta_correlation_grid_across_areas' in figures_to_do:
        output_path = os.path.join(FIGURE_PATH, 'si_delta_correlation_grid_across_areas')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Correlation of single-unit selectivity changes (delta) across areas, for each modality
        plot_si_delta_correlation_grid_across_areas(roc_df_delta, cond1='whisker_passive', cond2='auditory_passive', output_path=output_path)

        # Correlation of single-unit selectivity changes (delta) across areas, for whisker vs. whisker/auditory selectivity
        plot_si_delta_correlation_grid_across_areas(roc_df_delta, cond1='whisker_passive', cond2='wh_vs_aud_passive', output_path=output_path)

    if 'si_delta_correlation_grid_across_areas_multimodal' in figures_to_do:
        output_path = os.path.join(FIGURE_PATH, 'si_delta_correlation_grid_across_areas_multimodal')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Keep multimodal neurons: neurons that are significantly selective in both modalities in at least one condition
        # -----------------
        si_threshold = 0.1
        whisker_mask_pre = (roc_df['analysis_type'] == 'whisker_passive_pre') & (roc_df['significant']==True)
        #whisker_mask_post = (roc_df['analysis_type'] == 'whisker_passive_post') & (roc_df['significant']==True)

        # Step 2: Identify auditory-tuned neurons
        auditory_mask_pre = (roc_df['analysis_type'] == 'auditory_passive_pre') & (roc_df['significant']==True)
        #auditory_mask_post = (roc_df['analysis_type'] == 'auditory_passive_post') & (roc_df['significant']==True)
        #whisker_units = set(roc_df.loc[whisker_mask_pre & whisker_mask_post, 'unit_id'])
        whisker_units = set(roc_df.loc[whisker_mask_pre, 'neuron_id']) #global neuronal index
        #auditory_units = set(roc_df.loc[auditory_mask_pre & auditory_mask_post, 'unit_id'])
        auditory_units = set(roc_df.loc[auditory_mask_pre, 'neuron_id'])

        multimodal_units = whisker_units.intersection(auditory_units)
        roc_df_delta_sub = roc_df_delta[roc_df_delta['neuron_id'].isin(multimodal_units)]

        # Correlation of multimodal single-unit selectivity changes (delta) across areas, for each modality
        plot_si_delta_correlation_grid_across_areas(roc_df_delta_sub, cond1='whisker_passive', cond2='auditory_passive',
                                                    output_path=output_path)

    elif 'si_correlation_grid_across_areas' in figures_to_do:
        output_path = os.path.join(FIGURE_PATH, 'si_correlation_grid_across_areas')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Correlation of single-unit selectivity (delta) across areas, for each modality
        plot_si_correlation_grid_across_areas(roc_df, cond1='whisker_passive_pre', cond2='whisker_passive_post',
                                              output_path=output_path)
        plot_si_correlation_grid_across_areas(roc_df, cond1='whisker_passive_pre', cond2='whisker_active',
                                              output_path=output_path)
        plot_si_correlation_grid_across_areas(roc_df, cond1='whisker_passive_post', cond2='whisker_active',
                                              output_path=output_path)

        plot_si_correlation_grid_across_areas(roc_df, cond1='auditory_passive_pre', cond2='auditory_passive_post',
                                              output_path=output_path)
        plot_si_correlation_grid_across_areas(roc_df, cond1='auditory_passive_pre', cond2='auditory_active',
                                              output_path=output_path)
        plot_si_correlation_grid_across_areas(roc_df, cond1='auditory_passive_post', cond2='auditory_active',
                                              output_path=output_path)

        plot_si_correlation_grid_across_areas(roc_df, cond1='wh_vs_aud_passive_pre', cond2='wh_vs_aud_passive_post',
                                              output_path=output_path)
        plot_si_correlation_grid_across_areas(roc_df, cond1='wh_vs_aud_passive_pre', cond2='wh_vs_aud_active',
                                              output_path=output_path)
        plot_si_correlation_grid_across_areas(roc_df, cond1='wh_vs_aud_passive_post', cond2='wh_vs_aud_active',
                                              output_path=output_path)

    else:
        print('Functions not implemented.')

    return




if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        #parser.add_argument('--input', type=str, nargs='?', required=True)
        #parser.add_argument('--config', type=str, nargs='?', required=False)
        args = parser.parse_args()


        main()