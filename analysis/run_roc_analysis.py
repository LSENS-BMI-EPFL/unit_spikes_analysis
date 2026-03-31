#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_spikes_analysis
@file: run_roc_analysis.py
@time: 9/8/2025 4:06 PM
"""


# Imports
import os
import socket
import argparse
import pathlib
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mc
from scipy.stats import wilcoxon, ttest_rel, pearsonr
import pprint

import allen_utils
#from statannotations.Annotator import Annotator


import allen_utils as allen
import neural_utils
import plotting_utils
import plotting_utils as putils
from roc_analysis_utils import *
from selectivity_grid import plot_selectivity


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
                    g.set(ylim=(0, 80))
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



            # Run statistical debug for each area
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

            # Run statistical debug for each area
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
    hostname = socket.gethostname()
    if 'haas' in hostname:
        DATA_PATH = pathlib.Path('/mnt/lsens-analysis/')
        NWB_PATH = pathlib.Path('/mnt/lsens-analysis/Axel_Bisi/NWB_combined')
        FIGURE_PATH =  pathlib.Path('/mnt/lsens-analysis/Axel_Bisi/combined_results/roc_analysis')
        INFO_PATH = pathlib.Path('/mnt/share_internal/Axel_Bisi_Share/dataset_info')

    else:
        DATA_PATH = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis')
        NWB_PATH = r'M:\analysis\Axel_Bisi\NWB_combined'
        FIGURE_PATH = r'M:\analysis\Axel_Bisi\combined_results\roc_analysis'
        INFO_PATH = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'share_internal', f'Axel_Bisi_Share',
                             'dataset_info')

    mouse_info_path = os.path.join(INFO_PATH, 'joint_mouse_reference_weight.xlsx')
    mouse_info_df = pd.read_excel(mouse_info_path)
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)

    # Filter for usable mice
    mouse_info_df = mouse_info_df[
        (mouse_info_df['exclude'] == 0) &
        (mouse_info_df['exclude_ephys'] == 0) &
        (mouse_info_df['reward_group'].isin(['R+', 'R-'])) &
        (mouse_info_df['recording'] == 1)
        ]
    valid_mice = mouse_info_df['mouse_id'].unique()

    # ---------
    # LOAD DATA
    # ---------
    n_workers = 100
    nwb_list = [os.path.join(NWB_PATH, f) for f in os.listdir(NWB_PATH) if any(m in f for m in valid_mice)]
    #nwb_list = nwb_list[:200]
    _, unit_table, _ = neural_utils.combine_ephys_nwb(nwb_list, max_workers=n_workers)
    unit_table = allen.process_allen_labels(unit_table, subdivide_areas=True)

    print('Loading ROC data...')
    data_path_axel = os.path.join(DATA_PATH, 'Axel_Bisi', 'combined_results')
    roc_df = load_roc_results(data_path_axel, max_workers=n_workers)
    unit_table_mice = unit_table.mouse_id.unique()
    roc_df = roc_df[roc_df.mouse_id.isin(unit_table_mice)]

    # Fix: correct for choice the direction, positive and negative are inverted
    choice_analyses = [type for type in roc_df.analysis_type if 'choice' in type]
    choice_mask = roc_df['analysis_type'].isin(choice_analyses)
    # Invert direction for these rows (positive becomes negative and vice versa)
    roc_df.loc[choice_mask, 'direction'] = roc_df.loc[choice_mask, 'direction'].replace({'positive': 'negative', 'negative': 'positive'})


    # --- Load Myriam data ---
    #data_path_myriam = os.path.join(DATA_PATH, 'Axel_Bisi', 'combined_results')
    #roc_df_myriam = load_roc_results(data_path_myriam)

    #data_path_axel = os.path.join(DATA_PATH, 'Axel_Bisi', 'combined_results') #TODO: double check
    #roc_results_files = glob.glob(os.path.join(data_path_axel, '**', '*_roc_results_new.csv'),
    #                              recursive=True)  # find all roc results files
    #roc_df_axel = pd.concat([pd.read_csv(f) for f in roc_results_files], ignore_index=True)
    #data_path_myriam = os.path.join(DATA_PATH, 'Myriam_Hamon', 'results')
    #roc_results_files = glob.glob(os.path.join(data_path_myriam, '**', '*_roc_results_new.csv'),
    #                                        recursive=True)
    #roc_df_myriam = pd.concat([pd.read_csv(f) for f in roc_results_files], ignore_index=True)

    #roc_df = pd.concat([roc_df_axel, roc_df_myriam], ignore_index=True)
    roc_df = roc_df.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')

    #roc_df['neuron_id'] = roc_df['neuron_id'].astype(int)
    #unit_table['neuron_id'] = unit_table['neuron_id'].astype(int)
    #roc_df = roc_df.merge(unit_table[['mouse_id', 'session_id', 'neuron_id', 'area_acronym_custom']],
    #                      on=['mouse_id', 'session_id', 'neuron_id'], how='right')

    # Create unique unit identifier based on index
    #roc_df['unit_id'] = roc_df.index.astype(int)

    print('Present mice:', roc_df['mouse_id'].unique(), 'Number of mice', roc_df['mouse_id'].nunique(), 'per reward group',
          roc_df.groupby('reward_group')['mouse_id'].nunique())
    print('ROC analysis types:', roc_df['analysis_type'].unique())

    excluded_mice = []
    roc_df = roc_df[~roc_df['mouse_id'].isin(excluded_mice)]
    print(roc_df.columns)

    # -----------------------
    # PROCESS AND FILTER DATA
    # -----------------------
    print('Processing and filtering data...')
    N_UNITS_MIN = 30                # minimum units per area (whole-dataset)
    N_MICE_PER_AREA_MIN = 3         # minimum mice per area
    KEEP_SHARED_AREAS = True        # keep only areas that are shared between reward groups


    # Remove excluded areas
    roc_df = roc_df[~roc_df['area'].isin(allen.get_excluded_areas())]
    roc_df = allen_utils.process_allen_labels(roc_df, subdivide_areas=True)

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
    print('Equality ensured #colors vs. #area groups: ', len(colors) == len(area_groups.keys()))

    # Create a dictionary mapping each area to its group color
    area_color_dict = {}
    for (group_name, areas), color in zip(area_groups.items(), colors):
        for area in areas:
            area_color_dict[area] = color
    area_color_list = list(area_color_dict.values())

    # -------------------------------------------------
    # COMPUTE PROPORTIONS OF SIGNIFICANT UNITS PER AREA
    # -------------------------------------------------
    #roc_df_perc = compute_prop_significant(roc_df, area_col='area_acronym_custom', per_subject=False)
    #roc_df_perc_custom = roc_df_perc[roc_df_perc.area_level=='area_acronym_custom']
    #roc_df_perc_ccf = roc_df_perc[roc_df_perc.area_level=='ccf_atlas_parent_acronym']

    #roc_df_perc_subjects = compute_prop_significant(roc_df, area_col='ccf_acronym_no_layer', per_subject=True)
    #roc_df_perc_subjects_custom =  roc_df_perc_subjects[roc_df_perc_subjects.area_level=='area_acronym_custom']
    #roc_df_perc_subjects_ccf =  roc_df_perc_subjects[roc_df_perc_subjects.area_level=='ccf_atlas_parent_acronym']

    # --------
    # SUMMARY
    # --------
    plot_roc_grid=False
    if plot_roc_grid:
        output_path = os.path.join(FIGURE_PATH, 'roc_summary_grid')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        area_to_group_dict = allen_utils.get_custom_area_groups_from_name()
        # use this dict else put in other
        roc_df['area_group']= None
        roc_df['area_group'] = roc_df['area_acronym_custom'].apply(lambda x: area_to_group_dict.get(x, 'Other'))
        roc_df_perc_coarse = compute_prop_significant(roc_df, area_col='area_group', per_subject=True)
        print('Coarse areas', roc_df_perc_coarse.area_group.unique())
        area_groups = roc_df_perc_coarse.area_group.unique()

        for reward_group in roc_df_perc_coarse.reward_group.unique():
            suffix = 'rplus' if reward_group=='R+' else 'rminus'
            roc_df_sub = roc_df_perc_coarse[roc_df_perc_coarse.reward_group==reward_group]

            ROW_CONFIG = [
                ('Whisker\nresponsive', 'Positive', 'Negative', 'whisker_active', 'positive', 'negative', '#E74C3C',
                 '#3498DB'),
                ('Auditory\nresponsive', 'Positive', 'Negative', 'auditory_active', 'positive', 'negative', '#E74C3C',
                 '#3498DB'),
                ('Modality\nselective', 'Whisker', 'Auditory', 'wh_vs_aud_active', 'whisker', 'auditory', '#ebb134',
                 '#3127c2'),
                ('Choice\nselective', 'Positive', 'Negative', 'choice', 'positive', 'negative', '#E74C3C', '#3498DB'),
                ('Baseline\nchoice selective', 'Positive', 'Negative', 'baseline_choice', 'positive', 'negative', '#E74C3C', '#3498DB'),
                ('Spontaneous\nlicks', 'Positive', 'Negative', 'spontaneous_licks', 'positive', 'negative', '#E74C3C',
                 '#3498DB'),
            ]
            plot_selectivity(roc_df_sub, row_config=ROW_CONFIG, brain_areas=area_groups, area_col='area_group', style='donut', savepath=os.path.join(output_path, f'roc_grid_donuts_{suffix}.pdf'))  # ring charts
            plot_selectivity(roc_df_sub, row_config=ROW_CONFIG, brain_areas=area_groups, area_col='area_group', style='bar', savepath=os.path.join(output_path, f'roc_grid_bars_{suffix}.pdf'))  # grouped vertical bars

            ROW_CONFIG_PASSIVE = [
                ('Whisker\npre.', 'Positive', 'Negative', 'whisker_passive_pre', 'positive', 'negative', '#E74C3C',
                 '#3498DB'),
                ('Whisker\npost', 'Positive', 'Negative', 'whisker_passive_post', 'positive', 'negative', '#E74C3C',
                 '#3498DB'),
                ('Auditory\npre', 'Positive', 'Negative', 'auditory_passive_pre', 'positive', 'negative', '#E74C3C',
                 '#3498DB'),
                ('Auditory\npost', 'Positive', 'Negative', 'auditory_passive_post', 'positive', 'negative', '#E74C3C', '#3498DB'),
                ('Whisker\npre-to-post', 'Positive', 'Negative', 'whisker_pre_vs_post_learning', 'positive', 'negative', '#E74C3C',
                 '#3498DB'),
                ('Auditory\npre-to-post', 'Positive', 'Negative', 'auditory_pre_vs_post_learning', 'positive', 'negative', '#E74C3C',
                 '#3498DB'),
                ('Baseline\npre-to-post', 'Positive', 'Negative', 'baseline_pre_vs_post_learning', 'positive',
                 'negative', '#E74C3C',
                 '#3498DB'),
            ]
            plot_selectivity(roc_df_sub, row_config=ROW_CONFIG_PASSIVE, brain_areas=area_groups, area_col='area_group', style='donut', savepath=os.path.join(output_path, f'roc_grid_donuts_passive_{suffix}.pdf'))  # ring charts
            plot_selectivity(roc_df_sub, row_config=ROW_CONFIG_PASSIVE, brain_areas=area_groups, area_col='area_group', style='bar', savepath=os.path.join(output_path, f'roc_grid_bars_passive_{suffix}.pdf'))  # grouped vertical bars


    # --------
    # HEATMAPS
    # --------
    plot_heatmaps = True #TODO: could do a double heatmap for positive and negative directions (on same celle)
    if plot_heatmaps:
        output_path = os.path.join(FIGURE_PATH, 'roc_heatmaps')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        print(len(roc_df))
        roc_df_perc_subjects = compute_prop_significant(roc_df, area_col='area_acronym_custom', per_subject=True)
        print(len(roc_df_perc_subjects))
        print(roc_df_perc_subjects.columns)

        # Plot separately for each reward group
        for reward_group in roc_df_perc_subjects.reward_group.unique():
            for dir in ['positive','negative']:

                active_analyses = {
                    'whisker_active': 'Whisker resp.',
                    'auditory_active': 'Auditory resp.',
                    'spontaneous_licks': 'Lick resp.',
                    'choice': 'Choice',
                    'whisker_choice': 'Whisker choice',
                    'baseline_choice': 'Baseline choice',
                    'baseline_whisker_choice': 'Baseline choice, whisker',
                }
                passive_analyses = {
                    'whisker_passive_pre': 'Whisker resp. pre-learning',
                    'whisker_passive_post': 'Whisker resp. post-learning',
                    'whisker_pre_vs_post_learning': 'Whisker modulated pre- vs. post-learning',
                    'auditory_passive_pre': 'Auditory resp. pre-learning',
                    'auditory_passive_post': 'Auditory resp. post-learning',
                    'auditory_pre_vs_post_learning': 'Auditory modulated pre- vs. post-learning',
                    'baseline_pre_vs_post_learning': 'Baseline modulated pre- vs. post-learning',
                }
                anal_collect = [active_analyses, passive_analyses]
                for model_name_dict, period in zip(anal_collect, ['active', 'passive']):
                    subset = roc_df_perc_subjects[(roc_df_perc_subjects['reward_group'] == reward_group)
                                                  & (roc_df_perc_subjects['direction']==dir)
                                & (roc_df_perc_subjects['analysis_type'].isin(model_name_dict.keys()))]
                    print(len(subset))

                    # Averaged across mouse_id
                    subset = subset.groupby(['analysis_type', 'area_acronym_custom'], as_index=False)['proportion_signed'].mean()

                    dups = subset.duplicated(
                        subset=['analysis_type', 'area_acronym_custom'],
                        keep=False
                    )
                    print(subset[dups].sort_values(['analysis_type', 'area_acronym_custom']).head(100))

                    # Pivot for heatmap
                    #heatmap_data = subset.pivot_table(index='analysis_type', columns='area_acronym_custom', values='proportion_signed',
                    #                            aggfunc='mean')
                    heatmap_data = subset.pivot(index='analysis_type', columns='area_acronym_custom', values='proportion_signed').abs()

                    heatmap_data = heatmap_data.apply(pd.to_numeric, errors='coerce')

                    # Rename rows
                    heatmap_data = heatmap_data.rename(index=model_name_dict)
                    heatmap_data = heatmap_data.reindex(model_name_dict.values())

                    # Order areas using allen_utils function
                    area_order = allen_utils.get_custom_area_order()
                    areas_present = [area for area in area_order if area in heatmap_data.columns]
                    heatmap_data = heatmap_data[areas_present]

                    # Plot
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    fig, ax = plt.subplots(figsize=(24, 6), dpi=500)
                    if dir == 'positive':
                        cmap = sns.light_palette('#E74C3C', as_cmap=True)
                    elif dir == 'negative':
                        cmap = sns.light_palette('#3498DB', as_cmap=True)

                    sns.heatmap(heatmap_data,
                                ax=ax,
                                annot=True,
                                annot_kws={'fontsize':8},
                                fmt='.1f',
                                cmap=cmap,
                                vmin=5,
                                vmax=60,
                                cbar_kws={'label': 'Fraction significant units', 'shrink': 0.5, 'pad': 0.02,
                                          'aspect': 20 * 0.5},  # default aspect is 20
                                linewidths=0,
                                )
                    # Update colorbar
                    cbar = ax.collections[0].colorbar
                    cbar.ax.tick_params(labelsize=12)
                    cbar.set_label('Percentage significant', fontsize=15)

                    ax.xaxis.tick_top()
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, fontsize=12)
                    ax.set_xlabel('')
                    ax.set_ylabel('')

                    # Format y tick labels by removing underscores
                    ytick_labels = [label.get_text().replace('_', ' ') for label in ax.get_yticklabels()]
                    ax.set_yticklabels(ytick_labels, rotation=0, fontsize=12)

                    plt.tight_layout()

                    # Save
                    figname = f'roc_significant_fraction_heatmap_{reward_group}_per_mouse_{dir}'
                    output_path_dir = os.path.join(output_path, period)
                    if not os.path.exists(output_path_dir):
                        os.makedirs(output_path_dir)
                    putils.save_figure_with_options(fig, ['png', 'pdf', 'eps'],
                                                            figname,
                                                            output_path_dir,
                                                            dark_background=False)


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

    #roc_df_delta = compute_si_differences(roc_df)

    figures_to_do = [
        #'si_delta_correlation_grid_across_areas',
        #'si_delta_correlation_grid_across_areas_multimodal',
        #'si_correlation_grid_across_areas'
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

    # -----------------------------------
    # CORRELATION WITH HIERARCHY MEASURES
    # -----------------------------------
    plot_correlation = False

    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    import numpy as np

    if plot_correlation:

        # Load and merge anatomical metrics
        # 1. Axonal innervation from whisker SSp from Liu et al.

        liu_areas = allen_utils.load_liu_et_al_avg_ipsi()
        liu_areas = liu_areas.keys()
        area_column = 'ccf_acronym_no_layer'
        roc_df_perc_subjects_for_anat = compute_prop_significant(roc_df, area_col=area_column, per_subject=True)

        data_areas = roc_df_perc_subjects_for_anat['ccf_acronym_no_layer'].unique()
        print('liu areas', len(liu_areas))
        print('data areas', len(data_areas))
        n_intersec = set(liu_areas).intersection(set(data_areas))
        print('intersection areas', len(n_intersec))
        roc_df_perc_subjects_for_anat = allen_utils.merge_liu_avg_ipsi_opt(roc_df_perc_subjects_for_anat, cols_priority=area_column)
        roc_df_perc_subjects_for_anat['avg_ipsi_corr'] = np.log(roc_df_perc_subjects_for_anat['avg_ipsi_corr'] + 1e-5)  # log-transform and avoid log(0)

       # 2. Anatomical CT hierarchy from Harris et al. 2019
        harris_areas = allen_utils.load_process_hierarchy_from_harris()
        harris_areas = harris_areas[area_column].unique()
        roc_df_perc_subjects_for_anat = allen_utils.merge_hierarchy_from_harris(roc_df_perc_subjects_for_anat, merge_on=area_column)
        print('harris areas', len(harris_areas))
        print('data areas', len(data_areas))
        n_intersec = set(harris_areas).intersection(set(data_areas))
        print('intersection areas', len(n_intersec))

        palette = {'R+': 'forestgreen', 'R-': 'crimson'}
        print('columns', roc_df_perc_subjects_for_anat.columns)
        assert 'avg_ipsi_corr' in roc_df_perc_subjects_for_anat.columns, f"Expected 'avg_ipsi_corr' column not found. Available columns: {roc_df_perc_subjects_for_anat.columns}"
        assert 'cc_tc_ct_iterated' in roc_df_perc_subjects_for_anat.columns, f"Expected 'cc_tc_ct_iterated' column not found. Available columns: {roc_df_perc_subjects_for_anat.columns}"
        for area_level in [area_column]:
        #for area_level in [area_column, 'area_custom_acronym', 'ccf_atlas_parent_acronym']:

            output_path = os.path.join(FIGURE_PATH, 'roc_corr_anatomy', area_level)
            os.makedirs(output_path, exist_ok=True)

            for anat_var in ['avg_ipsi_corr', 'cc_tc_ct_iterated']:
                # --- Prepare THREE datasets ---
                dfs = {}

                # 1️⃣ Positive
                dfs['positive'] = (
                    roc_df_perc_subjects_for_anat
                    .query("direction == 'positive'")
                    .dropna(subset=[anat_var, 'proportion_signed'])
                    .groupby(['analysis_type', 'reward_group', area_level], as_index=False)
                    [[anat_var, 'proportion_signed']]
                    .mean()
                    .rename(columns={'proportion_signed': 'y'})
                )

                # 2️⃣ Negative
                dfs['negative'] = (
                    roc_df_perc_subjects_for_anat
                    .query("direction == 'negative'")
                    .dropna(subset=[anat_var, 'proportion_signed'])
                    .groupby(['analysis_type', 'reward_group', area_level], as_index=False)
                    [[anat_var, 'proportion_signed']]
                    .mean()
                    .rename(columns={'proportion_signed': 'y'})
                )

                # 3️⃣ All (⚠️ collapse duplicates!)
                dfs['all'] = (
                    roc_df_perc_subjects_for_anat
                    .dropna(subset=[anat_var, 'proportion_all'])
                    .groupby(['analysis_type', 'reward_group', area_level], as_index=False)
                    [[anat_var, 'proportion_all']]
                    .mean()
                    .rename(columns={'proportion_all': 'y'})
                )

                # --- Loop over modes ---
                for mode, df in dfs.items():

                    analysis_types = df['analysis_type'].unique()

                    n_cols = 4
                    n_rows = int(np.ceil(len(analysis_types) / n_cols))

                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

                    plot_idx = 0
                    all_areas = set()

                    for analysis_type in analysis_types:

                        ax = axes[plot_idx // n_cols, plot_idx % n_cols]
                        subset = df[df['analysis_type'] == analysis_type]

                        for group, color in palette.items():
                            sub = subset[subset['reward_group'] == group]

                            if len(sub) < 2:
                                continue

                            all_areas.update(sub[area_level])

                            # Scatter
                            sns.scatterplot(
                                data=sub,
                                x=anat_var,
                                y='y',
                                ax=ax,
                                color=color,
                                label=group,
                                s=50,
                                legend=False
                            )

                            # Regression
                            sns.regplot(
                                data=sub,
                                x=anat_var,
                                y='y',
                                ax=ax,
                                scatter=False,
                                color=color,
                                line_kws={'linewidth': 2},
                            )

                            # Correlation
                            r, pval = pearsonr(sub[anat_var], sub['y'])

                            ax.text(
                                0.05, 0.95 - (0.08 if group == 'R-' else 0),
                                f"{group}: r={r:.2f}, p={pval:.3f}",
                                transform=ax.transAxes,
                                color=color,
                                fontsize=9,
                                verticalalignment='top'
                            )

                            # Labels
                            for _, row in sub.iterrows():
                                ax.text(
                                    row[anat_var],
                                    row['y'],
                                    row[area_level],
                                    fontsize=7,
                                    alpha=0.8
                                )

                        ax.set_title(f"{analysis_type}")
                        if anat_var == 'avg_ipsi_corr':
                            ax.set_xlabel('Connectivity from whisker-SSp')
                        elif anat_var == 'cc_tc_ct_iterated':
                            ax.set_xlabel('Hierarchy score')

                        ax.set_ylabel('Fraction significant (%)')
                        plotting_utils.remove_top_right_frame(ax)

                        plot_idx += 1

                    # Remove empty axes
                    for j in range(plot_idx, n_rows * n_cols):
                        fig.delaxes(axes[j // n_cols, j % n_cols])

                    # Legend
                    handles, labels = ax.get_legend_handles_labels()
                    fig.legend(handles, labels, loc='upper right')

                    # --- Add number of areas ---
                    n_areas = len(all_areas)
                    fig.suptitle(f"{mode} | N areas = {n_areas}", fontsize=14)
                    plt.tight_layout()
                    figname = f'roc_{anat_var}_{mode}_correlation'
                    putils.save_figure_with_options(fig, ['png', 'pdf', 'eps'], figname, output_path)



    # --------------------------------
    # PERFORM NESTED PERMUTATION TESTS
    # --------------------------------
    reward_group_mapper = {'R+': 1, 'R-': 0}
    roc_df['reward_group'] = roc_df['reward_group'].map(reward_group_mapper)

    tests_to_do = [
        #'increased_si_across_reward_group',
        #'change_si_across_reward_group',
        #'crossmodal_change_si_across_reward_group',
        #'run_reward_group_hypotheses'
        ]

    if 'increased_si_across_reward_group' in tests_to_do:
        run_permutation_test_increase_reward_group(roc_df=roc_df)

    if 'change_si_across_reward_group' in tests_to_do:
        run_permutation_test_change_reward_group(roc_df=roc_df)

    if 'crossmodal_change_si_across_reward_group' in tests_to_do:
        test_cross_modality_reward_effect(roc_df=roc_df)

    if 'run_reward_group_hypotheses' in tests_to_do:
        run_reward_group_hypotheses(roc_df=roc_df)


    return


def run_permutation_test_increase_reward_group(roc_df):
    print('Running permutation debug for reward group comparison...')

    # Column names for SI pre/post (change here to match your dataframe)
    col_pre = 'whisker_passive_pre'
    col_post = 'whisker_passive_post'
    #col_pre = 'auditory_passive_pre'
    #col_post = 'auditory_passive_post'
    mouse_col = 'mouse_id'
    area_col = 'area' #TODO: check consistency across ROC runs
    group_col = 'reward_group'  # must be 0/1 or two distinct labels

    # Parameters
    min_units_per_mouse_area = 5  # minimum neurons per mouse-area to include mouse-area cell
    min_mice_per_area = 3  # minimum number of mice that have that area to include area in omnibus/post-hoc
    n_perms = 2000  # number of permutations (increase in production)

    np.random.seed(42)

    # --- 1. Filter and pivot ---
    df_w = roc_df[roc_df['analysis_type'].isin([col_pre, col_post])].copy()
    df_w = df_w.pivot_table(
        index=[mouse_col, area_col, 'neuron_id', group_col],
        columns='analysis_type',
        values='selectivity'
    ).reset_index()

    df_w['delta_absSI'] = df_w[col_post].abs() - df_w[col_pre].abs()

    # --- 2. Aggregate to mouse × area ---
    mouse_area = (
        df_w.groupby([mouse_col, area_col])
        .agg(mean_delta=('delta_absSI', 'mean'),
             n_units=('delta_absSI', 'size'),
             reward_group=(group_col, 'first'))
        .reset_index()
    )

    # Filter low-sampled mouse × area entries
    mouse_area = mouse_area[mouse_area['n_units'] >= min_units_per_mouse_area]

    # Keep only areas recorded in enough mice
    area_counts = mouse_area.groupby(area_col)[mouse_col].nunique()
    valid_areas = area_counts[area_counts >= min_mice_per_area].index
    mouse_area = mouse_area[mouse_area[area_col].isin(valid_areas)].copy()

    # --- 3. Omnibus debug (all areas together, two-sided) ---
    mouse_to_rows = {m: sub.copy() for m, sub in mouse_area.groupby(mouse_col)}
    unique_mice = list(mouse_to_rows.keys())
    orig_labels = mouse_area.groupby(mouse_col)[group_col].first().to_dict()

    # Observed overall difference
    group_means_obs = mouse_area.groupby(group_col)['mean_delta'].mean().to_dict()
    stat_obs = group_means_obs.get(1, 0) - group_means_obs.get(0, 0)

    # Permutations
    perm_stats = []
    for _ in tqdm(range(n_perms), desc='Omnibus permutations'):
        shuffled_labels = dict(zip(unique_mice, np.random.permutation([orig_labels[m] for m in unique_mice])))
        perm_rows = pd.concat([sub.assign(**{group_col: shuffled_labels[m]}) for m, sub in mouse_to_rows.items()],
                              ignore_index=True)
        gm = perm_rows.groupby(group_col)['mean_delta'].mean().to_dict()
        perm_stats.append(gm.get(1, 0) - gm.get(0, 0))
    perm_stats = np.array(perm_stats)

    # Two-sided p-value
    pval_omnibus = (np.sum(np.abs(perm_stats) >= np.abs(stat_obs)) + 1) / (n_perms + 1)

    omnibus_results = {
        'stat_obs': stat_obs,
        'pval': pval_omnibus,
        'perm_stats': perm_stats
    }

    # --- 4. Post-hoc area-level permutation tests (two-sided) ---
    area_results = []
    for area, sub in tqdm(mouse_area.groupby(area_col), desc='Post-hoc per-area'):
        mice_here = list(sub[mouse_col].unique())
        if len(mice_here) < 2:
            continue
        obs_means = sub.groupby(group_col)['mean_delta'].mean().to_dict()
        obs_diff = obs_means.get(1, 0) - obs_means.get(0, 0)
        orig_area_labels = sub.groupby(mouse_col)[group_col].first().to_dict()

        perm_diffs = []
        for _ in range(n_perms):
            shuffled = dict(zip(mice_here, np.random.permutation([orig_area_labels[m] for m in mice_here])))
            tmp = sub.copy()
            tmp[group_col] = tmp[mouse_col].map(shuffled)
            pm = tmp.groupby(group_col)['mean_delta'].mean().to_dict()
            perm_diffs.append(pm.get(1, 0) - pm.get(0, 0))
        perm_diffs = np.array(perm_diffs)

        # Two-sided p-value
        p_two_sided = (np.sum(np.abs(perm_diffs) >= np.abs(obs_diff)) + 1) / (n_perms + 1)
        area_results.append({'area': area, 'obs_diff': obs_diff, 'p_two_sided': p_two_sided})

    area_posthoc = pd.DataFrame(area_results).sort_values('p_two_sided')

    # --- 5. Benjamini-Hochberg FDR ---
    pvals = area_posthoc['p_two_sided'].values
    n = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    thresholds = (np.arange(1, n + 1) / n) * 0.05
    below = sorted_p <= thresholds
    reject = np.zeros(n, dtype=bool)
    if np.any(below):
        max_i = np.max(np.where(below)[0])
        reject[order[:max_i + 1]] = True
    # Adjusted p-values
    cummin = 1.0
    adj_p = np.empty(n)
    for i in range(n - 1, -1, -1):
        pi = sorted_p[i]
        adj = min(cummin, pi * n / (i + 1))
        cummin = adj
        adj_p[i] = adj
    adj_p_orig = np.empty(n)
    adj_p_orig[order] = adj_p
    area_posthoc['reject_H0_FDR'] = reject
    area_posthoc['p_adjusted_BH'] = adj_p_orig

    # --- 6. Prepare results ---
    results = {
        'mouse_area': mouse_area,
        'omnibus': omnibus_results,
        'area_posthoc': area_posthoc
    }

    pp = pprint.PrettyPrinter(indent=4, width=120)
    print("\n=== Omnibus Test Results (Two-sided) ===")
    pp.pprint({
        'Observed difference (reward - control)': stat_obs,
        'Permutation p-value (two-sided)': pval_omnibus
    })

    print("\n=== Post-hoc Area-level Results (Two-sided, first 20 rows) ===")
    pp.pprint(area_posthoc.head(1000).to_dict(orient='records'))

    return


def run_permutation_test_change_reward_group(roc_df):

    # Column names for SI pre/post (change here to match your dataframe)
    col_pre = 'whisker_passive_pre'
    col_post = 'whisker_passive_post'
    #col_pre = 'auditory_passive_pre'
    #col_post = 'auditory_passive_post'
    mouse_col = 'mouse_id'
    area_col = 'area' #TODO: check consistency across ROC runs
    group_col = 'reward_group'  # must be 0/1 or two distinct labels

    np.random.seed(42)

    # Parameters
    min_units_per_mouse_area = 5  # minimum neurons per mouse-area to include mouse-area cell
    min_mice_per_area = 3  # minimum number of mice that have that area to include area in omnibus/post-hoc
    n_perms = 2000  # number of permutations (increase in production)


    # --- 1. Filter and pivot ---
    df_w = roc_df[roc_df['analysis_type'].isin([col_pre, col_post])].copy()
    df_w = df_w.pivot_table(
        index=[mouse_col, area_col, 'neuron_id', group_col],
        columns='analysis_type',
        values='selectivity'
    ).reset_index()

    # Compute neuron-level delta (change in magnitude of selectivity)
    df_w['delta_absSI'] = df_w[col_post].abs() - df_w[col_pre].abs()

    # --- 2. Aggregate to mouse × area ---
    mouse_area = (
        df_w.groupby([mouse_col, area_col])
        .agg(mean_delta=('delta_absSI', 'mean'),
             n_units=('delta_absSI', 'size'),
             reward_group=(group_col, 'first'))
        .reset_index()
    )

    # Filter low-sampled mouse × area entries
    mouse_area = mouse_area[mouse_area['n_units'] >= min_units_per_mouse_area]

    # Keep only areas recorded in enough mice
    area_counts = mouse_area.groupby(area_col)[mouse_col].nunique()
    valid_areas = area_counts[area_counts >= min_mice_per_area].index
    mouse_area = mouse_area[mouse_area[area_col].isin(valid_areas)].copy()

    # --- 3. Omnibus debug (all areas together) ---
    mouse_to_rows = {m: sub.copy() for m, sub in mouse_area.groupby(mouse_col)}
    unique_mice = list(mouse_to_rows.keys())
    orig_labels = mouse_area.groupby(mouse_col)[group_col].first().to_dict()

    # Observed overall difference
    group_means_obs = mouse_area.groupby(group_col)['mean_delta'].mean().to_dict()
    stat_obs = group_means_obs.get(1, 0) - group_means_obs.get(0, 0)

    # Permutations
    perm_stats = []
    for _ in tqdm(range(n_perms), desc='Omnibus permutations'):
        shuffled_labels = dict(zip(unique_mice, np.random.permutation([orig_labels[m] for m in unique_mice])))
        perm_rows = pd.concat([sub.assign(**{group_col: shuffled_labels[m]}) for m, sub in mouse_to_rows.items()],
                              ignore_index=True)
        gm = perm_rows.groupby(group_col)['mean_delta'].mean().to_dict()
        perm_stats.append(gm.get(1, 0) - gm.get(0, 0))
    perm_stats = np.array(perm_stats)

    # One-sided p-value
    pval_omnibus = (np.sum(perm_stats >= stat_obs) + 1) / (n_perms + 1)

    omnibus_results = {
        'stat_obs': stat_obs,
        'pval': pval_omnibus,
        'perm_stats': perm_stats
    }

    # --- 4. Post-hoc area-level permutation tests ---
    area_results = []
    for area, sub in tqdm(mouse_area.groupby(area_col), desc='Post-hoc per-area'):
        mice_here = list(sub[mouse_col].unique())
        if len(mice_here) < 2:
            continue
        obs_means = sub.groupby(group_col)['mean_delta'].mean().to_dict()
        obs_diff = obs_means.get(1, 0) - obs_means.get(0, 0)
        orig_area_labels = sub.groupby(mouse_col)[group_col].first().to_dict()

        perm_diffs = []
        for _ in range(n_perms):
            shuffled = dict(zip(mice_here, np.random.permutation([orig_area_labels[m] for m in mice_here])))
            tmp = sub.copy()
            tmp[group_col] = tmp[mouse_col].map(shuffled)
            pm = tmp.groupby(group_col)['mean_delta'].mean().to_dict()
            perm_diffs.append(pm.get(1, 0) - pm.get(0, 0))
        perm_diffs = np.array(perm_diffs)
        p_one_sided = (np.sum(perm_diffs >= obs_diff) + 1) / (n_perms + 1)
        area_results.append({'area': area, 'obs_diff': obs_diff, 'p_one_sided': p_one_sided})

    area_posthoc = pd.DataFrame(area_results).sort_values('p_one_sided')

    # --- 5. Benjamini-Hochberg FDR ---
    pvals = area_posthoc['p_one_sided'].values
    n = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    thresholds = (np.arange(1, n + 1) / n) * 0.05
    below = sorted_p <= thresholds
    reject = np.zeros(n, dtype=bool)
    if np.any(below):
        max_i = np.max(np.where(below)[0])
        reject[order[:max_i + 1]] = True
    # Adjusted p-values (optional)
    cummin = 1.0
    adj_p = np.empty(n)
    for i in range(n - 1, -1, -1):
        pi = sorted_p[i]
        adj = min(cummin, pi * n / (i + 1))
        cummin = adj
        adj_p[i] = adj
    adj_p_orig = np.empty(n)
    adj_p_orig[order] = adj_p
    area_posthoc['reject_H0_FDR'] = reject
    area_posthoc['p_adjusted_BH'] = adj_p_orig

    # --- Return ---
    results = {
        'mouse_area': mouse_area,
        'omnibus': omnibus_results,
        'area_posthoc': area_posthoc
    }

    pp = pprint.PrettyPrinter(indent=4)
    print('Omnibus debug results:')
    pp.pprint(results)

    return results


def test_cross_modality_reward_effect(roc_df,
                                      whisker_cols=('whisker_passive_pre', 'whisker_passive_post'),
                                      auditory_cols=('auditory_passive_pre', 'auditory_passive_post'),
                                      mouse_col='mouse_id',
                                      area_col='area',
                                      group_col='reward_group',
                                      min_units_per_mouse_area=3,
                                      min_mice_per_area=4,
                                      n_perms=2000,
                                      seed=42,
                                      print_results=True):
    """
    Test the null hypothesis:
    The reward effect on whisker ΔSI is the same as the reward effect on auditory ΔSI.
    Alternative: whisker changes more across reward groups than auditory.
    """
    np.random.seed(seed)

    # --- 1. Compute ΔSI for whisker ---
    pre_w, post_w = whisker_cols
    df_w = roc_df[roc_df['analysis_type'].isin([pre_w, post_w])].copy()
    df_w = df_w.pivot_table(
        index=[mouse_col, area_col, 'neuron_id', group_col],
        columns='analysis_type',
        values='selectivity'
    ).reset_index()
    df_w['delta_whisker'] = df_w[post_w].abs() - df_w[pre_w].abs()

    # --- 2. Compute ΔSI for auditory ---
    pre_a, post_a = auditory_cols
    df_a = roc_df[roc_df['analysis_type'].isin([pre_a, post_a])].copy()
    df_a = df_a.pivot_table(
        index=[mouse_col, area_col, 'neuron_id', group_col],
        columns='analysis_type',
        values='selectivity'
    ).reset_index()
    df_a['delta_auditory'] = df_a[post_a].abs() - df_a[pre_a].abs()

    # --- 3. Aggregate to mouse × area ---
    agg_w = df_w.groupby([mouse_col, area_col]).agg(
        mean_delta_whisker=('delta_whisker', 'mean'),
        n_units=('delta_whisker', 'size'),
        reward_group=(group_col, 'first')
    ).reset_index()
    agg_w = agg_w[agg_w['n_units'] >= min_units_per_mouse_area]

    agg_a = df_a.groupby([mouse_col, area_col]).agg(
        mean_delta_auditory=('delta_auditory', 'mean'),
        n_units=('delta_auditory', 'size'),
        reward_group=(group_col, 'first')
    ).reset_index()
    agg_a = agg_a[agg_a['n_units'] >= min_units_per_mouse_area]

    # Keep only areas present in enough mice
    valid_areas_w = agg_w.groupby(area_col)[mouse_col].nunique()
    valid_areas_w = valid_areas_w[valid_areas_w >= min_mice_per_area].index
    agg_w = agg_w[agg_w[area_col].isin(valid_areas_w)]

    valid_areas_a = agg_a.groupby(area_col)[mouse_col].nunique()
    valid_areas_a = valid_areas_a[valid_areas_a >= min_mice_per_area].index
    agg_a = agg_a[agg_a[area_col].isin(valid_areas_a)]

    # Merge whisker and auditory by mouse × area (inner join)
    merged = pd.merge(agg_w[[mouse_col, area_col, 'mean_delta_whisker', 'reward_group']],
                      agg_a[[mouse_col, area_col, 'mean_delta_auditory']],
                      on=[mouse_col, area_col], how='inner')

    # --- 4. Observed statistic: difference of differences ---
    group_means = merged.groupby(group_col)[['mean_delta_whisker', 'mean_delta_auditory']].mean()
    stat_obs = (group_means.loc[1, 'mean_delta_whisker'] - group_means.loc[0, 'mean_delta_whisker']) - \
               (group_means.loc[1, 'mean_delta_auditory'] - group_means.loc[0, 'mean_delta_auditory'])

    # --- 5. Permutations ---
    mouse_to_rows = {m: sub.copy() for m, sub in merged.groupby(mouse_col)}
    unique_mice = list(mouse_to_rows.keys())
    orig_labels = merged.groupby(mouse_col)['reward_group'].first().to_dict()

    perm_stats = []
    for _ in tqdm(range(n_perms), desc='Cross-modality permutations'):
        shuffled_labels = dict(zip(unique_mice, np.random.permutation([orig_labels[m] for m in unique_mice])))
        perm_rows = pd.concat([sub.assign(reward_group=shuffled_labels[m]) for m, sub in mouse_to_rows.items()],
                              ignore_index=True)
        group_means_perm = perm_rows.groupby(group_col)[['mean_delta_whisker', 'mean_delta_auditory']].mean()
        perm_stat = (group_means_perm.loc[1, 'mean_delta_whisker'] - group_means_perm.loc[0, 'mean_delta_whisker']) - \
                    (group_means_perm.loc[1, 'mean_delta_auditory'] - group_means_perm.loc[0, 'mean_delta_auditory'])
        perm_stats.append(perm_stat)
    perm_stats = np.array(perm_stats)

    # Two-sided p-value
    pval = (np.sum(np.abs(perm_stats) >= np.abs(stat_obs)) + 1) / (n_perms + 1)

    # --- 6. Print results ---
    pp = pprint.PrettyPrinter(indent=4, width=120)
    print("\n=== Cross-modality Reward Effect Test ===")
    pp.pprint({
        'Observed statistic (ΔSI_whisker_reward-control - ΔSI_auditory_reward-control)': stat_obs,
        'Permutation p-value (two-sided)': pval
    })

    return {'obs_stat': stat_obs, 'pval': pval, 'perm_stats': perm_stats, 'merged': merged}


def run_reward_group_hypotheses(roc_df,
                                conditions=[('whisker_passive_pre', 'whisker_passive_post'),
                                            ('auditory_passive_pre', 'auditory_passive_post')],
                                mouse_col='mouse_id',
                                area_col='area',
                                group_col='reward_group',
                                min_units_per_mouse_area=3,
                                min_mice_per_area=4,
                                n_perms=2000,
                                seed=42,
                                print_results=True):
    """
    Run a battery of reward-group-based hypotheses:
    1. Whisker ΔSI reward vs control
    2. Auditory ΔSI reward vs control
    3. Cross-modality difference (ΔSI_whisker - ΔSI_auditory) reward effect
    Includes omnibus tests and area-level post-hoc tests with BH-FDR correction.
    """
    np.random.seed(seed)
    results = {}

    # --- Step 1: Compute ΔSI for each condition ---
    delta_dfs = {}
    for pre_col, post_col in conditions:
        cond_name = pre_col.replace('_pre', '')
        df_cond = roc_df[roc_df['analysis_type'].isin([pre_col, post_col])].copy()
        df_cond = df_cond.pivot_table(
            index=[mouse_col, area_col, 'neuron_id', group_col],
            columns='analysis_type',
            values='selectivity'
        ).reset_index()
        df_cond['delta'] = df_cond[post_col].abs() - df_cond[pre_col].abs()
        delta_dfs[cond_name] = df_cond

    # --- Step 2: Aggregate to mouse × area ---
    agg_dfs = {}
    for cond_name, df_cond in delta_dfs.items():
        agg = df_cond.groupby([mouse_col, area_col]).agg(
            mean_delta=('delta', 'mean'),
            n_units=('delta', 'size'),
            reward_group=(group_col, 'first')
        ).reset_index()
        agg = agg[agg['n_units'] >= min_units_per_mouse_area]
        # Keep areas with enough mice
        area_counts = agg.groupby(area_col)[mouse_col].nunique()
        valid_areas = area_counts[area_counts >= min_mice_per_area].index
        agg = agg[agg[area_col].isin(valid_areas)]
        agg_dfs[cond_name] = agg

    # --- Step 3: Omnibus tests per condition ---
    for cond_name, agg in agg_dfs.items():
        mouse_to_rows = {m: sub.copy() for m, sub in agg.groupby(mouse_col)}
        unique_mice = list(mouse_to_rows.keys())
        orig_labels = agg.groupby(mouse_col)['reward_group'].first().to_dict()

        group_means_obs = agg.groupby(group_col)['mean_delta'].mean().to_dict()
        stat_obs = group_means_obs.get(1, 0) - group_means_obs.get(0, 0)

        # Permutations
        perm_stats = []
        for _ in tqdm(range(n_perms), desc=f"Omnibus permutations ({cond_name})"):
            shuffled_labels = dict(zip(unique_mice, np.random.permutation([orig_labels[m] for m in unique_mice])))
            perm_rows = pd.concat([sub.assign(reward_group=shuffled_labels[m]) for m, sub in mouse_to_rows.items()],
                                  ignore_index=True)
            gm = perm_rows.groupby(group_col)['mean_delta'].mean().to_dict()
            perm_stats.append(gm.get(1, 0) - gm.get(0, 0))
        perm_stats = np.array(perm_stats)
        pval = (np.sum(np.abs(perm_stats) >= np.abs(stat_obs)) + 1) / (n_perms + 1)

        # Store omnibus results
        results[cond_name] = {'omnibus': {'stat_obs': stat_obs, 'pval': pval, 'perm_stats': perm_stats}}

        # Post-hoc area-level debug
        area_results = []
        for area, sub in agg.groupby(area_col):
            mice_here = list(sub[mouse_col].unique())
            if len(mice_here) < 2:
                continue
            obs_diff = sub.groupby(group_col)['mean_delta'].mean().diff().iloc[-1]  # reward - control
            orig_area_labels = sub.groupby(mouse_col)['reward_group'].first().to_dict()

            perm_diffs = []
            for _ in range(n_perms):
                shuffled = dict(zip(mice_here, np.random.permutation([orig_area_labels[m] for m in mice_here])))
                tmp = sub.copy()
                tmp[group_col] = tmp[mouse_col].map(shuffled)
                pm = tmp.groupby(group_col)['mean_delta'].mean().diff().iloc[-1]
                perm_diffs.append(pm)
            perm_diffs = np.array(perm_diffs)
            p_two_sided = (np.sum(np.abs(perm_diffs) >= np.abs(obs_diff)) + 1) / (n_perms + 1)
            area_results.append({'area': area, 'obs_diff': obs_diff, 'p_two_sided': p_two_sided})

        area_posthoc = pd.DataFrame(area_results).sort_values('p_two_sided')

        # BH-FDR correction
        if len(area_posthoc) > 0:
            pvals = area_posthoc['p_two_sided'].values
            n = len(pvals)
            order = np.argsort(pvals)
            sorted_p = pvals[order]
            thresholds = (np.arange(1, n + 1) / n) * 0.05
            below = sorted_p <= thresholds
            reject = np.zeros(n, dtype=bool)
            if np.any(below):
                max_i = np.max(np.where(below)[0])
                reject[order[:max_i + 1]] = True
            # Adjusted p-values
            cummin = 1.0
            adj_p = np.empty(n)
            for i in range(n - 1, -1, -1):
                pi = sorted_p[i]
                adj = min(cummin, pi * n / (i + 1))
                cummin = adj
                adj_p[i] = adj
            adj_p_orig = np.empty(n)
            adj_p_orig[order] = adj_p
            area_posthoc['reject_H0_FDR'] = reject
            area_posthoc['p_adjusted_BH'] = adj_p_orig

        results[cond_name]['area_posthoc'] = area_posthoc

    # --- Step 4: Cross-modality reward effect (difference-of-differences) ---
    # Merge whisker and auditory mouse × area
    merged = pd.merge(agg_dfs['whisker_passive'][['mouse_id', 'area', 'mean_delta', 'reward_group']],
                      agg_dfs['auditory_passive'][['mouse_id', 'area', 'mean_delta']],
                      on=['mouse_id', 'area'], suffixes=('_whisker', '_auditory'))
    # Observed difference of differences
    group_means = merged.groupby(group_col)[['mean_delta_whisker', 'mean_delta_auditory']].mean()
    stat_obs_diff = (group_means.loc[1, 'mean_delta_whisker'] - group_means.loc[0, 'mean_delta_whisker']) - \
                    (group_means.loc[1, 'mean_delta_auditory'] - group_means.loc[0, 'mean_delta_auditory'])

    # Permutations
    mouse_to_rows = {m: sub.copy() for m, sub in merged.groupby(mouse_col)}
    unique_mice = list(mouse_to_rows.keys())
    orig_labels = merged.groupby(mouse_col)['reward_group'].first().to_dict()

    perm_stats = []
    for _ in tqdm(range(n_perms), desc='Cross-modality permutations'):
        shuffled_labels = dict(zip(unique_mice, np.random.permutation([orig_labels[m] for m in unique_mice])))
        perm_rows = pd.concat([sub.assign(reward_group=shuffled_labels[m]) for m, sub in mouse_to_rows.items()],
                              ignore_index=True)
        gm = perm_rows.groupby(group_col)[['mean_delta_whisker', 'mean_delta_auditory']].mean()
        perm_stat = (gm.loc[1, 'mean_delta_whisker'] - gm.loc[0, 'mean_delta_whisker']) - \
                    (gm.loc[1, 'mean_delta_auditory'] - gm.loc[0, 'mean_delta_auditory'])
        perm_stats.append(perm_stat)
    perm_stats = np.array(perm_stats)
    pval_cross = (np.sum(np.abs(perm_stats) >= np.abs(stat_obs_diff)) + 1) / (n_perms + 1)

    results['cross_modality'] = {'stat_obs': stat_obs_diff, 'pval': pval_cross, 'perm_stats': perm_stats,
                                 'merged': merged}

    # --- Step 5: Cross-modality area-level post-hoc ---
    area_results = []
    for area, sub in merged.groupby(area_col):
        mice_here = list(sub[mouse_col].unique())
        if len(mice_here) < 2:
            continue
        # difference-of-differences per area
        group_means_area = sub.groupby(group_col)[['mean_delta_whisker', 'mean_delta_auditory']].mean()
        obs_diff_area = (group_means_area.loc[1, 'mean_delta_whisker'] - group_means_area.loc[
            0, 'mean_delta_whisker']) - \
                        (group_means_area.loc[1, 'mean_delta_auditory'] - group_means_area.loc[
                            0, 'mean_delta_auditory'])

        orig_area_labels = sub.groupby(mouse_col)['reward_group'].first().to_dict()

        perm_diffs = []
        for _ in range(n_perms):
            shuffled = dict(zip(mice_here, np.random.permutation([orig_area_labels[m] for m in mice_here])))
            tmp = sub.copy()
            tmp[group_col] = tmp[mouse_col].map(shuffled)
            gm = tmp.groupby(group_col)[['mean_delta_whisker', 'mean_delta_auditory']].mean()
            perm_stat_area = (gm.loc[1, 'mean_delta_whisker'] - gm.loc[0, 'mean_delta_whisker']) - \
                             (gm.loc[1, 'mean_delta_auditory'] - gm.loc[0, 'mean_delta_auditory'])
            perm_diffs.append(perm_stat_area)
        perm_diffs = np.array(perm_diffs)

        p_two_sided = (np.sum(np.abs(perm_diffs) >= np.abs(obs_diff_area)) + 1) / (n_perms + 1)
        area_results.append({'area': area, 'obs_diff': obs_diff_area, 'p_two_sided': p_two_sided})

    # BH-FDR correction for cross-modality area-level
    area_posthoc_cross = pd.DataFrame(area_results).sort_values('p_two_sided')
    if len(area_posthoc_cross) > 0:
        pvals = area_posthoc_cross['p_two_sided'].values
        n = len(pvals)
        order = np.argsort(pvals)
        sorted_p = pvals[order]
        thresholds = (np.arange(1, n + 1) / n) * 0.05
        below = sorted_p <= thresholds
        reject = np.zeros(n, dtype=bool)
        if np.any(below):
            max_i = np.max(np.where(below)[0])
            reject[order[:max_i + 1]] = True
        cummin = 1.0
        adj_p = np.empty(n)
        for i in range(n - 1, -1, -1):
            pi = sorted_p[i]
            adj = min(cummin, pi * n / (i + 1))
            cummin = adj
            adj_p[i] = adj
        adj_p_orig = np.empty(n)
        adj_p_orig[order] = adj_p
        area_posthoc_cross['reject_H0_FDR'] = reject
        area_posthoc_cross['p_adjusted_BH'] = adj_p_orig

    results['cross_modality']['area_posthoc'] = area_posthoc_cross

    # --- Step 5: Print nicely ---
    if print_results:
        pp = pprint.PrettyPrinter(indent=4, width=120)
        for cond_name, res in results.items():
            print(f"\n=== Hypothesis: {cond_name} ===")
            if cond_name == 'cross_modality':
                pp.pprint({'Observed diff-of-diffs': res['stat_obs'], 'Permutation p-value': res['pval']})
                if not res['area_posthoc'].empty:
                    print("Post-hoc area-level :")
                    pp.pprint(res['area_posthoc'].to_dict(orient='records'))
            else:
                pp.pprint({'Omnibus stat_obs': res['omnibus']['stat_obs'], 'Omnibus pval': res['omnibus']['pval']})
                if not res['area_posthoc'].empty:
                    print("Post-hoc area-level :")
                    pp.pprint(res['area_posthoc'].to_dict(orient='records'))


    return results

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        #parser.add_argument('--input', type=str, nargs='?', required=True)
        #parser.add_argument('--config', type=str, nargs='?', required=False)
        args = parser.parse_args()

        print('- Analysing ROC results...')
        main()
        print('Done.')