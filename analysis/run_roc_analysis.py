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
from roc_analysis_utils import *

DATA_PATH = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis')
FIGURE_PATH = r'M:\analysis\Axel_Bisi\combined_results\roc_analysis'

def plot_proportion_across_areas(data_df, area_order, area_color_list, output_path):
    """
    Plot proportions of significant neurons per area, analysis type for each reward group separately.
    Plots for all significant neurons and per directions of significance.
    :param data_df: dataframe with proportions of significant neurons per area
    :param area_order: list of areas in desired order
    :param area_color_list: list of colors corresponding to areas
    :param output_path: path to save figures
    """
    if 'mouse_id' in data_df.columns:
        errorbar = 'se'
        errwidth = 0.7
    else:
        errorbar = None
        errwidth = None
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
                    errorbar=errorbar,
                    errwidth=errwidth,
                    legend=False,
                    dodge=False
                )
                g.figure.suptitle(anal_type)
                g.despine(left=False)
                g.set_axis_labels('', 'Proportion (%)')
                if dir == 'all':
                    g.set(ylim=(0, 70))
                else:
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
        df = data_df[(data_df['analysis_type'] == anal_type)]

        y_val = 'selectivity_abs'

        g = sns.catplot(
            data=df,
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
                seed=42,
            )

            for bar0, bar1 in zip(g.ax.containers[0], g.ax.containers[1]):
                bar0.set_alpha(0.2)
                bar1.set_alpha(0.2)
            for i, area in enumerate(area_order):
                mask_pre = (df['area_acronym_custom'] == area) & (df['period'] == 'pre-learning')
                mask_post = (df['area_acronym_custom'] == area) & (df['period'] == 'post-learning')
                pre_val = df.loc[mask_pre, 'proportion'].mean()
                post_val = df.loc[mask_post, 'proportion'].mean()
                plt.plot([i - 0.2, i + 0.2], [pre_val, post_val], color='gray', linewidth=0.8, alpha=1)
            g.despine(left=False)
            g.set_axis_labels('', 'Proportion (%)')
            g.set(ylim=(0, 80))
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
            for bar0, bar1 in zip(g.ax.containers[0], g.ax.containers[1]):
                bar0.set_alpha(0.2)
                bar1.set_alpha(0.2)
            # Connect each bar value with a line
            for i, area in enumerate(area_order):
                mask_pre = (df['area_acronym_custom'] == area) & (df['period'] == 'pre-learning')
                mask_post = (df['area_acronym_custom'] == area) & (df['period'] == 'post-learning')
                pre_val = df.loc[mask_pre, 'selectivity_abs'].mean()
                post_val = df.loc[mask_post, 'selectivity_abs'].mean()
                plt.plot([i - 0.2, i + 0.2], [pre_val, post_val], color='gray', linewidth=0.8, alpha=1)

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
            df = data_df[(data_df['reward_group'] == reward_group)
                         &
                         (data_df['analysis_type'].str.contains(f'{stim_type}_pre_vs_post_learning'))]

            g = sns.catplot(
                data=df,
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
                legend_out=True,
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

    # Get data information

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
    print('Loading data...')
    data_path_axel = os.path.join(DATA_PATH, 'Axel_Bisi', 'combined_results') #TODO: double check
    roc_results_files = glob.glob(os.path.join(data_path_axel, '**', '*_roc_results_new.csv'),
                                  recursive=True)  # find all roc results files
    roc_df_axel = pd.concat([pd.read_csv(f) for f in roc_results_files], ignore_index=True)
    data_path_myriam = os.path.join(DATA_PATH, 'Myriam_Hamon', 'results')
    roc_results_files = glob.glob(os.path.join(data_path_myriam, '**', '*_roc_results_new.csv'),
                                            recursive=True)
    roc_df_myriam = pd.concat([pd.read_csv(f) for f in roc_results_files], ignore_index=True)

    roc_df = pd.concat([roc_df_axel, roc_df_myriam], ignore_index=True)
    roc_df = roc_df.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')

    # Create unique unit identifier based on index
    roc_df['unit_id'] = roc_df.index.astype(int)

    print('Present mice:', roc_df['mouse_id'].unique(), 'Number of mice', roc_df['mouse_id'].nunique(), 'per reward group',
          roc_df.groupby('reward_group')['mouse_id'].nunique())
    print('ROC analysis types:', roc_df['analysis_type'].unique())

    excluded_mice = []
    roc_df = roc_df[~roc_df['mouse_id'].isin(excluded_mice)]

    # -----------------------
    # PROCESS AND FILTER DATA
    # -----------------------
    print('Processing and filtering data...')
    N_UNITS_MIN = 10                # minimum units per area (whole-dataset)
    N_MICE_PER_AREA_MIN = 3         # minimum mice per area
    KEEP_SHARED_AREAS = True        # keep only areas that are shared between reward groups

    roc_df = filter_process_data(roc_df, n_units_min=N_UNITS_MIN, n_mice_per_area_min=N_MICE_PER_AREA_MIN, keep_shared=KEEP_SHARED_AREAS)

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
    print('Plotting...')

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
        output_path = os.path.join(FIGURE_PATH, 'passive_before_vs_after_across_areas_per_subjects')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Compare proportions of significant units across areas before and after learning, for each reward group separately
        plot_prop_before_vs_after_across_areas(roc_df_perc_subjects, area_order=area_order_shared, area_color_list=area_color_list,
                                     output_path=output_path)


    elif 'prop_across_areas_passive_pre_vs_post' in figures_to_do:
        output_path = os.path.join(FIGURE_PATH, 'passive_pre_vs_post_across_areas_per_subjects')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Compare proportions of significant units pre vs post learning, across areas, for each reward group separately
        plot_proportion_across_areas_pre_vs_post(roc_df_perc_subjects, area_order=area_order_shared, area_color_list=area_color_list,
                                     output_path=output_path)

    else:
        print('Functions not implemented.')

    # ---------------------------------------------
    # PERFORM COMPARISONS ON POPULATION SELECTIVITY
    # ---------------------------------------------

    figures_to_do = [
        #'abs_si_across_areas',
        #'abs_si_across_areas_reward_group',
        'abs_si_before_vs_after_across_areas',
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
        plot_pop_selectivity_before_vs_after_across_areas(roc_df, per_subject=False, area_order=area_order_shared, area_color_list=area_color_list,
                                     output_path=output_path)

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
        'si_delta_correlation_grid_across_areas',
        'si_delta_correlation_grid_across_areas_multimodal',
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
        whisker_units = set(roc_df.loc[whisker_mask_pre, 'unit_id']) #global neuronal index
        #auditory_units = set(roc_df.loc[auditory_mask_pre & auditory_mask_post, 'unit_id'])
        auditory_units = set(roc_df.loc[auditory_mask_pre, 'unit_id'])

        multimodal_units = whisker_units.intersection(auditory_units)
        roc_df_delta_sub = roc_df_delta[roc_df_delta['unit_id'].isin(multimodal_units)]

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

        print('- Analysing ROC results...')
        main()
        print('Done.')