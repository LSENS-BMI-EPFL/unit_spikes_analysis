#! /usr/bin/env/python3
"""
@author: Myriam Hamon
@project: unit_spikes_analysis
@file: run_roc_analysis_experts_vs_learners.py

ROC analysis: expert vs learner comparison across area groups.
Key differences from run_roc_analysis.py:
  - x-axis: area_group (coarse) instead of area_acronym_custom
  - learning_group determined from folder: whisker_0 = learner, whisker_1+ = expert
  - All statistics computed over sessions (session_id as the statistical unit)
  - Additional overlap/co-occurrence analysis across selectivity classes
"""

import os
import glob
import socket
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_rel, pearsonr
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import ephys_utilities.allen_utils.allen_utils as allen
import ephys_utilities.neural_utils.neural_utils as neural_utils
import ephys_utilities.plotting_utils.plotting_utils as putils
from roc_analysis_utils import (
    filter_process_data,
    fdr_bh,
)
from selectivity_grid import plot_selectivity


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

AREA_GROUP_ORDER = [
    'Motor and frontal areas',
    'Somatosensory areas',
    'Auditory areas',
    'Visual areas',
    'Retrosplenial areas',
    'Striatum and pallidum',
    'Thalamus',
    'Hippocampus',
    'Midbrain',
    'Olfactory areas',
    'Amygdala and hypothalamus',
]

ACTIVE_ANALYSIS_TYPES = [
    'whisker_active',
    'auditory_active',
    'wh_vs_aud_active',
    'choice',
    'whisker_choice',
    'baseline_choice',
    'spontaneous_licks',
]

LEARNING_ORDER = ['learner', 'expert']

# (type_a, type_b, label_a, label_b, color_a, color_b)
OVERLAP_PAIRS = [
    ('whisker_passive_post',  'auditory_passive_post',  'Wh post',    'Aud post',   '#C0392B', '#2980B9'),
    ('whisker_passive_pre',   'auditory_passive_pre',   'Wh pre',     'Aud pre',    '#E74C3C', '#3498DB'),
    ('whisker_passive_pre',   'whisker_passive_post',   'Wh pre',     'Wh post',    '#F39C12', '#C0392B'),
    ('auditory_passive_pre',  'auditory_passive_post',  'Aud pre',    'Aud post',   '#85C1E9', '#1A5276'),
    ('whisker_choice',        'baseline_choice',        'Wh lick',    'Aud lick',   '#27AE60', '#8E44AD'),
    ('whisker_active',        'auditory_active',        'Wh active',  'Aud active', '#E74C3C', '#3498DB'),
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

LEARNING_COLORS = {
    ('learner', 'R+'): '#45624F',
    ('expert',  'R+'): '#238443',
    ('learner', 'R-'): '#8E6565',
    ('expert',  'R-'): '#BF3435',
}


def _get_learning_palette(reward_group):
    """Return {'learner': color, 'expert': color} for a given reward group."""
    return {
        'learner': LEARNING_COLORS[('learner', reward_group)],
        'expert':  LEARNING_COLORS[('expert',  reward_group)],
    }


def _read_roc_file_with_meta(f):
    """
    Module-level helper so ProcessPoolExecutor can pickle it.
    Adds whisker_folder, learning_group, and guarantees mouse_id and session_id columns.
    mouse_id  → extracted from filename (e.g. AB001_roc_results_new.csv → AB001)
    session_id → {mouse_id}_{whisker_folder}  (one CSV = one session)
    """
    try:
        df = pd.read_csv(f)
        parts = Path(f).parts

        # Whisker folder determines learning group
        wfolder = next((p for p in parts if p.startswith('whisker_')), 'unknown')
        df['whisker_folder'] = wfolder
        df['learning_group'] = 'learner' if wfolder == 'whisker_0' else 'expert'

        # Guarantee mouse_id (extract from filename if the CSV omits it)
        if 'mouse_id' not in df.columns:
            stem = Path(f).stem                         # e.g. 'AB001_roc_results_new'
            df['mouse_id'] = stem.split('_roc_results')[0]   # → 'AB001'

        # Guarantee session_id (one file = one session)
        if 'session_id' not in df.columns:
            mouse_id_val = df['mouse_id'].iloc[0] if len(df) > 0 else 'unknown'
            df['session_id'] = f"{mouse_id_val}_{wfolder}"

        # Make unit_id globally unique across sessions.
        # CSV unit_id is only unique within one file; without the session prefix,
        # neuron 0 from AB001 and neuron 0 from AB002 collide in the pivot table.
        if 'unit_id' in df.columns:
            df['unit_id'] = df['session_id'].astype(str) + '_' + df['unit_id'].astype(str)
        elif 'neuron_id' in df.columns:
            df['unit_id'] = df['session_id'].astype(str) + '_' + df['neuron_id'].astype(str)
        else:
            print(f"  [unit_id] columns in CSV: {list(df.columns)}")
            df['unit_id'] = pd.NA

        return df
    except Exception as e:
        print(f"  Skipped {f}: {e}")
        return None


def load_roc_results_all_sessions(root_path, max_workers=20):
    """
    Load ROC results from ALL whisker folders under root_path.
    Assigns learning_group from folder name:
      whisker_0 → 'learner'
      whisker_1, whisker_2, … → 'expert'
    """
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    # Targeted discovery: exploit the known 3-level structure
    # root / {AB*|MH*} / whisker_* / roc_analysis / *_roc_results_new.csv
    # Much faster than recursive glob on a network drive.
    all_files = []
    for mouse_folder in os.listdir(root_path):
        if not (mouse_folder.startswith('AB') or mouse_folder.startswith('MH')):
            continue
        mouse_path = os.path.join(root_path, mouse_folder)
        if not os.path.isdir(mouse_path):
            continue
        for whisker_folder in os.listdir(mouse_path):
            if not whisker_folder.startswith('whisker_'):
                continue
            roc_dir = os.path.join(mouse_path, whisker_folder, 'roc_analysis')
            if not os.path.isdir(roc_dir):
                continue
            all_files.extend(glob.glob(os.path.join(roc_dir, '*_roc_results_new.csv')))
    print(f"  Found {len(all_files)} ROC files across all whisker folders in: {root_path}")

    if not all_files:
        return pd.DataFrame()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        dfs = list(tqdm(executor.map(_read_roc_file_with_meta, all_files),
                        total=len(all_files),
                        desc="  Loading ROC CSV files", unit="file"))

    dfs = [df for df in dfs if df is not None]
    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    n_learner = (combined['learning_group'] == 'learner').sum()
    n_expert  = (combined['learning_group'] == 'expert').sum()
    print(f"  Loaded {n_learner} learner rows, {n_expert} expert rows")
    return combined


def _compute_prop_significant_per_session(roc_df, area_col):
    """
    Compute proportions of significant neurons per session (session_id as statistical unit).
    learning_group is included in the groupby as it is session-level metadata.
    Mirrors compute_prop_significant(per_subject=True) from roc_analysis_utils but uses session_id.
    """
    subj_col = 'session_id'
    default_groups = [subj_col, 'analysis_type', 'reward_group', 'learning_group', area_col]

    # Total units per (session, analysis_type, reward_group, learning_group, area)
    total = roc_df.groupby(default_groups).size().reset_index(name='total_count')

    # Significant units by direction — expand to all directions per valid group
    sig = (roc_df[roc_df['significant']]
           .groupby(default_groups + ['direction'])
           .size()
           .reset_index(name='count'))

    valid_groups = total[default_groups].copy()
    directions   = roc_df['direction'].dropna().unique()
    expanded     = valid_groups.assign(_k=1).merge(
        pd.DataFrame({'direction': directions, '_k': 1}), on='_k'
    ).drop(columns='_k')
    sig = expanded.merge(sig, on=default_groups + ['direction'], how='left').fillna({'count': 0})
    sig['count'] = sig['count'].astype(int)

    # Non-selective units
    non_sel = (roc_df[~roc_df['significant']]
               .groupby(default_groups)
               .size()
               .reset_index(name='count'))
    non_sel['direction'] = 'non-selective'

    perc = pd.concat([sig, non_sel], ignore_index=True)
    perc = perc.merge(total, on=default_groups, how='left')
    perc['proportion'] = np.where(
        perc['count'] > 0,
        perc['count'] / perc['total_count'] * 100,
        0.0
    )

    # proportion_all: sum of both directions per group
    is_wh_vs_aud = perc['analysis_type'].str.contains('wh_vs_aud', na=False)

    pn_sum = (perc[perc['direction'].isin(['positive', 'negative'])]
              .groupby(default_groups)['proportion'].sum()
              .rename('pn_sum').reset_index())
    aw_sum = (perc[perc['direction'].isin(['whisker', 'auditory'])]
              .groupby(default_groups)['proportion'].sum()
              .rename('aw_sum').reset_index())
    ns_sum = (perc[perc['direction'] == 'non-selective']
              .groupby(default_groups)['proportion'].sum()
              .rename('ns_sum').reset_index())

    perc = perc.merge(pn_sum, on=default_groups, how='left')
    perc = perc.merge(aw_sum, on=default_groups, how='left')
    perc = perc.merge(ns_sum, on=default_groups, how='left')

    perc['proportion_all'] = perc['proportion'].copy()
    pn_mask = perc['direction'].isin(['positive', 'negative']) & ~is_wh_vs_aud
    aw_mask = perc['direction'].isin(['whisker', 'auditory'])  &  is_wh_vs_aud
    ns_mask = perc['direction'] == 'non-selective'
    perc.loc[pn_mask, 'proportion_all'] = perc.loc[pn_mask, 'pn_sum']
    perc.loc[aw_mask, 'proportion_all'] = perc.loc[aw_mask, 'aw_sum']
    perc.loc[ns_mask, 'proportion_all'] = perc.loc[ns_mask, 'ns_sum']
    perc.drop(columns=['pn_sum', 'aw_sum', 'ns_sum'], inplace=True)

    # Signed proportions
    perc['proportion_signed'] = perc['proportion'].copy()
    perc.loc[perc['direction'] == 'negative', 'proportion_signed'] *= -1
    perc.loc[perc['direction'] == 'auditory',  'proportion_signed'] *= -1

    return perc


def _annotate_mwu(ax, data_df, area_group_order, area_col, y_val, annot_y,
                  group_col='learning_group', group_a='expert', group_b='learner'):
    """Mann-Whitney U per area_group (group_a vs group_b), BH-FDR, annotate ax."""
    results = []
    for area in area_group_order:
        sub = data_df[data_df[area_col] == area]
        vals_a = sub[sub[group_col] == group_a][y_val].dropna()
        vals_b = sub[sub[group_col] == group_b][y_val].dropna()
        if len(vals_a) < 3 or len(vals_b) < 3:
            continue
        _, p = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
        results.append({'area': area, 'p_value': p})

    if not results:
        return None

    results_df = pd.DataFrame(results)
    reject, p_corrected = fdr_bh(results_df['p_value'].values, fdr=0.05)
    results_df['p_corrected'] = p_corrected
    results_df['significant'] = reject
    print(results_df.to_string())

    for i, area in enumerate(area_group_order):
        row = results_df[results_df['area'] == area]
        if row.empty or not row['significant'].values[0]:
            continue
        p_val = row['p_corrected'].values[0]
        text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
        ax.text(i, annot_y, text, ha='center', va='bottom', fontsize=12, color='black')

    return results_df


def _filter_area_group_order(area_group_order, data_df, area_col='area_group'):
    """Return only area_groups present in data_df, preserving order."""
    present = data_df[area_col].unique()
    return [a for a in area_group_order if a in present]


# ─────────────────────────────────────────────────────────────────────────────
# PROPORTION PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_proportion_learning(data_df, area_group_order, output_path, direction=None, area_col='area_group'):
    """
    For each reward_group: bar plot of proportion of significant neurons per area_group,
    comparing learners vs experts. Stats: MWU per area_group, BH-FDR.
    Mirrors plot_proportion_across_areas_reward_group but with learning_group as hue.
    """
    area_group_order = _filter_area_group_order(area_group_order, data_df, area_col)

    for reward_group in ['R+', 'R-']:
        palette = _get_learning_palette(reward_group)
        subset_rg = data_df[data_df['reward_group'] == reward_group]

        for anal_type in subset_rg['analysis_type'].unique():
            subset_df = subset_rg[subset_rg['analysis_type'] == anal_type]

            if direction is not None:
                if 'wh_vs_aud' in anal_type:
                    dir_list = ['whisker'] if direction == 'positive' else ['auditory']
                else:
                    dir_list = [direction]
                dirs_to_loop = [(direction, dir_list, 'proportion', direction)]
            else:
                dirs_to_loop = [
                    ('all',   ['whisker'] if 'wh_vs_aud' in anal_type else ['positive'], 'proportion_all', 'all'),
                    ('dir_1', ['whisker'] if 'wh_vs_aud' in anal_type else ['positive'], 'proportion',     'dir_1'),
                    ('dir_2', ['auditory'] if 'wh_vs_aud' in anal_type else ['negative'], 'proportion',    'dir_2'),
                ]

            for _, dir_list, y_val, suffix in dirs_to_loop:
                subset_dir = subset_df[subset_df['direction'].isin(dir_list)]
                if subset_dir.empty:
                    continue

                data_max = subset_dir[y_val].max()
                ylim_top = max(data_max * 1.2, 1)

                g = sns.catplot(
                    data=subset_dir,
                    kind='bar',
                    x=area_col,
                    y=y_val,
                    hue='learning_group',
                    hue_order=LEARNING_ORDER,
                    order=area_group_order,
                    palette=[palette[k] for k in LEARNING_ORDER],
                    height=3,
                    aspect=2.5,
                    errorbar='se',
                    errwidth=0.7,
                    legend=True,
                )
                g.figure.suptitle(f'{anal_type} | {reward_group}')
                g.despine(left=False)
                g.set_axis_labels('', 'Proportion (%)')
                g.set(ylim=(0, ylim_top))
                g.tight_layout()
                g.set_xticklabels(rotation=45, ha='right')
                g.tick_params(labelsize=10)

                _annotate_mwu(g.ax, subset_dir, area_group_order, area_col, y_val,
                              annot_y=data_max * 1.08)

                rg_str = 'rplus' if reward_group == 'R+' else 'rminus'
                figname = f'prop_learning_{rg_str}_{anal_type}_{suffix}'
                putils.save_figure_with_options(plt.gcf(), ['png', 'pdf', 'svg'], figname, output_dir=output_path)
                plt.close()
    return


def plot_proportion_per_reward_group(data_df, area_group_order, output_path, direction=None, area_col='area_group'):
    """
    Per reward_group: bar plot of proportion of selective neurons per area_group,
    with learner vs expert as hue. Sanity-check overview before deeper comparisons.
    """
    area_group_order = _filter_area_group_order(area_group_order, data_df, area_col)

    for reward_group in ['R+', 'R-']:
        palette = _get_learning_palette(reward_group)
        subset_rg = data_df[data_df['reward_group'] == reward_group]

        for anal_type in subset_rg['analysis_type'].unique():
            subset_df = subset_rg[subset_rg['analysis_type'] == anal_type]

            for dir_name, dir_list, y_val, suffix in [
                ('all',   ['whisker'] if 'wh_vs_aud' in anal_type else ['positive'], 'proportion_all', 'all'),
                ('dir_1', ['whisker'] if 'wh_vs_aud' in anal_type else ['positive'], 'proportion',     'dir_1'),
                ('dir_2', ['auditory'] if 'wh_vs_aud' in anal_type else ['negative'], 'proportion',    'dir_2'),
            ]:
                if direction is not None and dir_name != direction:
                    continue
                subset_dir = subset_df[subset_df['direction'].isin(dir_list)]
                if subset_dir.empty:
                    continue

                g = sns.catplot(
                    data=subset_dir,
                    kind='bar',
                    x=area_col,
                    y=y_val,
                    hue='learning_group',
                    hue_order=LEARNING_ORDER,
                    order=area_group_order,
                    palette=[palette[k] for k in LEARNING_ORDER],
                    height=3,
                    aspect=2.5,
                    errorbar='se',
                    errwidth=0.7,
                    legend=True,
                )
                _annotate_mwu(g.ax, subset_dir, area_group_order, area_col, y_val,
                              annot_y=subset_dir[y_val].max() * 1.05)

                g.figure.suptitle(f'{anal_type} | {reward_group}')
                g.despine(left=False)
                g.set_axis_labels('', 'Proportion (%)')
                g.set(ylim=(0, max(subset_dir[y_val].max() * 1.3, 1)))
                g.tight_layout()
                g.set_xticklabels(rotation=45, ha='right')

                rg_str = 'rplus' if reward_group == 'R+' else 'rminus'
                figname = f'prop_{rg_str}_{anal_type}_{suffix}'
                putils.save_figure_with_options(plt.gcf(), ['png', 'pdf', 'svg'], figname, output_dir=output_path)
                plt.close()
    return


# ─────────────────────────────────────────────────────────────────────────────
# SELECTIVITY (SI) PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_pop_selectivity_learning(data_df, area_group_order, output_path, direction=None, area_col='area_group'):
    """
    For each reward_group: bar plot of population selectivity per area_group,
    comparing learners vs experts. Mirrors plot_pop_selectivity_across_areas_reward_group.
    """
    area_group_order = _filter_area_group_order(area_group_order, data_df, area_col)

    if direction is not None:
        data_df = data_df[data_df['direction'] == direction].copy()
        y_col = 'selectivity'
        y_label = f'Selectivity ({direction})'
        negative_dir = (direction == 'negative')
    else:
        y_col = 'selectivity_abs'
        y_label = 'Absolute selectivity'
        negative_dir = False

    # Average per session first
    avg_df = data_df.groupby(
        ['session_id', 'analysis_type', 'reward_group', area_col, 'learning_group'], as_index=False
    )[y_col].mean()

    for reward_group in ['R+', 'R-']:
        palette = _get_learning_palette(reward_group)
        subset_rg = avg_df[avg_df['reward_group'] == reward_group]

        for anal_type in subset_rg['analysis_type'].unique():
            subset_df = subset_rg[subset_rg['analysis_type'] == anal_type]
            if subset_df.empty:
                continue

            data_ext = subset_df[y_col].min() if negative_dir else subset_df[y_col].max()
            pad = abs(data_ext) * 0.2 if data_ext != 0 else 0.05
            y_lim = (data_ext - pad, pad) if negative_dir else (0, data_ext + pad)

            g = sns.catplot(
                data=subset_df,
                kind='bar',
                x=area_col,
                y=y_col,
                hue='learning_group',
                hue_order=LEARNING_ORDER,
                order=area_group_order,
                palette=[palette[k] for k in LEARNING_ORDER],
                height=3,
                aspect=2.5,
                errorbar='se',
                errwidth=0.7,
                legend=True,
            )
            g.figure.suptitle(f'{anal_type} | {reward_group}')
            g.despine(left=False)
            g.set_axis_labels('', y_label)
            g.set(ylim=y_lim)
            g.tight_layout()
            g.set_xticklabels(rotation=45, ha='right')

            _annotate_mwu(g.ax, subset_df, area_group_order, area_col, y_col,
                          annot_y=data_ext * 1.05 if not negative_dir else data_ext * 0.95)

            rg_str = 'rplus' if reward_group == 'R+' else 'rminus'
            dir_suffix = f'_{direction}' if direction else ''
            figname = f'si_learning_{rg_str}_{anal_type}{dir_suffix}'
            putils.save_figure_with_options(plt.gcf(), ['png', 'pdf', 'svg'], figname, output_dir=output_path)
            plt.close()
    return


def plot_mean_metric_significant_learning(roc_df, area_group_order, output_path,
                                          y_col, y_label, figname_suffix, area_col='area_group'):
    """
    For significant units per direction: bidirectional bar plot of mean y_col per area_group,
    comparing learners vs experts per reward_group.
    Mirrors plot_mean_metric_significant_across_areas.
    """
    area_group_order = _filter_area_group_order(area_group_order, roc_df, area_col)

    sig_df = roc_df[roc_df['significant'] & roc_df['direction'].isin(['positive', 'negative'])].copy()

    for reward_group in ['R+', 'R-']:
        palette = _get_learning_palette(reward_group)
        subset_rg = sig_df[sig_df['reward_group'] == reward_group]

        for anal_type in subset_rg['analysis_type'].unique():
            subset_df = subset_rg[subset_rg['analysis_type'] == anal_type]
            if subset_df.empty or subset_df[y_col].isna().all():
                continue

            avg_df = subset_df.groupby(
                ['session_id', area_col, 'direction', 'learning_group'], as_index=False
            )[y_col].mean().dropna(subset=[y_col])
            if avg_df.empty:
                continue

            ylim_range = avg_df.groupby([area_col, 'direction'])[y_col].mean().abs().max() * 1.5

            fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
            for ax, lg in zip(axes, LEARNING_ORDER):
                sub = avg_df[avg_df['learning_group'] == lg]
                if sub.empty:
                    ax.set_title(f'{lg} (no data)')
                    continue
                sns.barplot(
                    data=sub, x=area_col, y=y_col,
                    hue='direction', hue_order=['positive', 'negative'],
                    order=area_group_order,
                    palette=['tomato', 'dodgerblue'],
                    errorbar='se', errwidth=0.7,
                    dodge=False, ax=ax, legend=(lg == LEARNING_ORDER[-1])
                )
                ax.set_title(lg)
                ax.set_xlabel('')
                ax.set_ylabel(y_label if lg == LEARNING_ORDER[0] else '')
                ax.set_ylim(-ylim_range, ylim_range)
                ax.axhline(0, color='black', linewidth=0.6)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                putils.remove_top_right_frame(ax)

            fig.suptitle(f'{anal_type} | {reward_group}')
            plt.tight_layout()

            rg_str = 'rplus' if reward_group == 'R+' else 'rminus'
            figname = f'{figname_suffix}_{rg_str}_{anal_type}'
            putils.save_figure_with_options(fig, ['png', 'pdf', 'svg'], figname, output_dir=output_path)
            plt.close()
    return


# ─────────────────────────────────────────────────────────────────────────────
# PASSIVE / PRE vs POST PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_prop_before_vs_after_learning(data_df, area_group_order, output_path, direction=None, area_col='area_group'):
    """
    For each reward_group × stim_type: two subplots (learner | expert) showing
    pre vs post proportion per area_group with paired t-test within each learning group.
    Mirrors plot_prop_before_vs_after_across_areas with expert/learner split.
    """
    area_group_order = _filter_area_group_order(area_group_order, data_df, area_col)
    data_df = data_df[data_df['analysis_type'].str.contains('passive')]

    for reward_group in ['R+', 'R-']:
        palette = _get_learning_palette(reward_group)
        subset_rg = data_df[data_df['reward_group'] == reward_group]

        for stim_type in ['auditory', 'whisker']:
            df_pre  = subset_rg[subset_rg['analysis_type'] == f'{stim_type}_passive_pre'].copy()
            df_post = subset_rg[subset_rg['analysis_type'] == f'{stim_type}_passive_post'].copy()
            if df_pre.empty or df_post.empty:
                continue

            if direction is None:
                for df in [df_pre, df_post]:
                    df['direction'] = df['direction'].replace({'positive': 'both', 'negative': 'both'})
                df_pre  = df_pre.groupby(
                    ['session_id', 'analysis_type', 'reward_group', area_col, 'direction', 'learning_group'],
                    as_index=False)['proportion'].sum()
                df_post = df_post.groupby(
                    ['session_id', 'analysis_type', 'reward_group', area_col, 'direction', 'learning_group'],
                    as_index=False)['proportion'].sum()
                filter_dir = 'both'
            else:
                df_pre  = df_pre[df_pre['direction'] == direction]
                df_post = df_post[df_post['direction'] == direction]
                filter_dir = direction

            df_pre['period']  = 'pre-learning'
            df_post['period'] = 'post-learning'
            combined = pd.concat([df_pre, df_post]).reset_index(drop=True)
            combined = combined[combined['direction'] == filter_dir]

            fig, axes = plt.subplots(1, 2, figsize=(14, 3.5), sharey=True)
            for ax, lg in zip(axes, LEARNING_ORDER):
                lg_palette = [
                    putils.adjust_lightness(palette[lg], 1.3),
                    putils.adjust_lightness(palette[lg], 0.7),
                ]
                sub = combined[combined['learning_group'] == lg]
                if sub.empty:
                    ax.set_title(f'{lg} (no data)')
                    continue

                sns.barplot(
                    data=sub, x=area_col, y='proportion',
                    hue='period', hue_order=['pre-learning', 'post-learning'],
                    order=area_group_order,
                    palette=lg_palette, errorbar='se', errwidth=0.7,
                    ax=ax, legend=True, alpha=0.5,
                )
                # Connect mean pre to mean post per area with a line
                for i, area in enumerate(area_group_order):
                    area_sub = sub[sub[area_col] == area]
                    pre_val  = area_sub[area_sub['period'] == 'pre-learning']['proportion'].mean()
                    post_val = area_sub[area_sub['period'] == 'post-learning']['proportion'].mean()
                    ax.plot([i - 0.2, i + 0.2], [pre_val, post_val], color='gray', linewidth=0.8)

                # Paired t-test per area (paired on session_id — both periods come from same session)
                results = []
                for area in area_group_order:
                    pivot = sub[sub[area_col] == area].pivot_table(
                        index='session_id', columns='period', values='proportion')
                    if pivot.shape[0] < 3 or 'pre-learning' not in pivot.columns or 'post-learning' not in pivot.columns:
                        continue
                    stat, p = ttest_rel(pivot['pre-learning'], pivot['post-learning'],
                                        alternative='two-sided', nan_policy='omit')
                    results.append({'area': area, 'p_value': p})

                if results:
                    res_df = pd.DataFrame(results)
                    reject, p_corr = fdr_bh(res_df['p_value'].values, fdr=0.05)
                    res_df['p_corrected'] = p_corr
                    data_max = sub['proportion'].max()
                    for i, area in enumerate(area_group_order):
                        row = res_df[res_df['area'] == area]
                        if row.empty or not reject[res_df.index[res_df['area'] == area][0]]:
                            continue
                        p_val = row['p_corrected'].values[0]
                        text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                        ax.text(i, data_max * 1.05, text, ha='center', va='bottom', fontsize=11)

                ax.set_title(f'{lg}')
                ax.set_xlabel('')
                ax.set_ylabel('Proportion (%)' if lg == LEARNING_ORDER[0] else '')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                putils.remove_top_right_frame(ax)

            fig.suptitle(f'{stim_type} passive pre vs post | {reward_group}')
            plt.tight_layout()

            rg_str = 'rplus' if reward_group == 'R+' else 'rminus'
            dir_suffix = f'_{direction}' if direction else ''
            figname = f'prop_pre_vs_post_learning_{rg_str}_{stim_type}{dir_suffix}'
            putils.save_figure_with_options(fig, ['png', 'pdf', 'svg'], figname, output_dir=output_path)
            plt.close()
    return


def plot_pop_selectivity_before_vs_after_learning(roc_df, area_group_order, output_path, direction=None, area_col='area_group'):
    """
    For each reward_group × stim_type: two subplots (learner | expert) showing
    population SI pre vs post per area_group with paired t-test.
    Mirrors plot_pop_selectivity_before_vs_after_across_areas with expert/learner split.
    """
    area_group_order = _filter_area_group_order(area_group_order, roc_df, area_col)

    if direction is not None:
        roc_df = roc_df[roc_df['direction'] == direction].copy()
        y_col = 'selectivity'
        y_label = f'Selectivity ({direction})'
        negative_dir = (direction == 'negative')
    else:
        y_col = 'selectivity_abs'
        y_label = 'Absolute selectivity'
        negative_dir = False

    # Average per session × area_group × analysis_type × learning_group
    avg_df = roc_df.groupby(
        ['session_id', 'analysis_type', 'reward_group', area_col, 'learning_group'], as_index=False
    )[y_col].mean()
    avg_df = avg_df[avg_df['analysis_type'].str.contains('passive')]

    for reward_group in ['R+', 'R-']:
        palette = _get_learning_palette(reward_group)
        subset_rg = avg_df[avg_df['reward_group'] == reward_group]

        for stim_type in ['auditory', 'whisker']:
            df_pre  = subset_rg[subset_rg['analysis_type'] == f'{stim_type}_passive_pre'].copy()
            df_post = subset_rg[subset_rg['analysis_type'] == f'{stim_type}_passive_post'].copy()
            if df_pre.empty or df_post.empty:
                continue

            df_pre['period']  = 'pre-learning'
            df_post['period'] = 'post-learning'
            combined = pd.concat([df_pre, df_post]).reset_index(drop=True)

            y_min = combined[y_col].min()
            y_max = combined[y_col].max()
            margin = (y_max - y_min) * 0.2 if y_max > y_min else 0.1
            y_lim = (y_min - margin, y_max + margin)

            fig, axes = plt.subplots(1, 2, figsize=(14, 3.5), sharey=True)
            for ax, lg in zip(axes, LEARNING_ORDER):
                lg_palette = [
                    putils.adjust_lightness(palette[lg], 1.3),
                    putils.adjust_lightness(palette[lg], 0.7),
                ]
                sub = combined[combined['learning_group'] == lg]
                if sub.empty:
                    ax.set_title(f'{lg} (no data)')
                    continue

                sns.barplot(
                    data=sub, x=area_col, y=y_col,
                    hue='period', hue_order=['pre-learning', 'post-learning'],
                    order=area_group_order, palette=lg_palette,
                    errorbar='se', errwidth=0.7, ax=ax, alpha=0.5,
                )
                for i, area in enumerate(area_group_order):
                    area_sub = sub[sub[area_col] == area]
                    pre_val  = area_sub[area_sub['period'] == 'pre-learning'][y_col].mean()
                    post_val = area_sub[area_sub['period'] == 'post-learning'][y_col].mean()
                    ax.plot([i - 0.2, i + 0.2], [pre_val, post_val], color='gray', linewidth=0.8)

                results = []
                for area in area_group_order:
                    pivot = sub[sub[area_col] == area].pivot_table(
                        index='session_id', columns='period', values=y_col)
                    if pivot.shape[0] < 3 or 'pre-learning' not in pivot.columns or 'post-learning' not in pivot.columns:
                        continue
                    stat, p = ttest_rel(pivot['pre-learning'], pivot['post-learning'],
                                        alternative='two-sided', nan_policy='omit')
                    results.append({'area': area, 'p_value': p})

                if results:
                    res_df = pd.DataFrame(results)
                    reject, p_corr = fdr_bh(res_df['p_value'].values, fdr=0.05)
                    res_df['p_corrected'] = p_corr
                    offset = abs(y_lim[1] - y_lim[0]) * 0.05
                    for idx_r, row in res_df.iterrows():
                        if not reject[idx_r]:
                            continue
                        i = area_group_order.index(row['area']) if row['area'] in area_group_order else -1
                        if i < 0:
                            continue
                        p_val = row['p_corrected']
                        text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                        ax.text(i, y_lim[1] + offset, text, ha='center', va='bottom', fontsize=11)

                ax.set_title(lg)
                ax.set_xlabel('')
                ax.set_ylabel(y_label if lg == LEARNING_ORDER[0] else '')
                ax.set_ylim(y_lim)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                putils.remove_top_right_frame(ax)

            fig.suptitle(f'{stim_type} SI pre vs post | {reward_group}')
            plt.tight_layout()

            rg_str = 'rplus' if reward_group == 'R+' else 'rminus'
            dir_suffix = f'_{direction}' if direction else ''
            figname = f'si_pre_vs_post_learning_{rg_str}_{stim_type}{dir_suffix}'
            putils.save_figure_with_options(fig, ['png', 'pdf', 'svg'], figname, output_dir=output_path)
            plt.close()
    return


def plot_proportion_pre_vs_post_learning(data_df, area_group_order, output_path, area_col='area_group'):
    """
    For each reward_group: signed proportion of pre_vs_post_learning analysis per area_group,
    comparing learner vs expert. Mirrors plot_proportion_across_areas_pre_vs_post.
    """
    area_group_order = _filter_area_group_order(area_group_order, data_df, area_col)

    for reward_group in ['R+', 'R-']:
        palette = _get_learning_palette(reward_group)
        subset_rg = data_df[data_df['reward_group'] == reward_group]

        for stim_type in ['auditory', 'whisker']:
            df = subset_rg[subset_rg['analysis_type'].str.contains(f'{stim_type}_pre_vs_post_learning')]
            if df.empty:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 3.5), sharey=True)
            for ax, lg in zip(axes, LEARNING_ORDER):
                sub = df[df['learning_group'] == lg]
                if sub.empty:
                    ax.set_title(f'{lg} (no data)')
                    continue
                sns.barplot(
                    data=sub, x=area_col, y='proportion_signed',
                    hue='direction', hue_order=['positive', 'negative'],
                    order=area_group_order,
                    palette=['tomato', 'dodgerblue'],
                    errorbar='se', errwidth=0.7, dodge=False,
                    ax=ax, legend=(lg == LEARNING_ORDER[-1])
                )
                ax.axhline(0, color='black', linewidth=0.6)
                ax.set_title(lg)
                ax.set_xlabel('')
                ax.set_ylabel('Proportion (%)' if lg == LEARNING_ORDER[0] else '')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                putils.remove_top_right_frame(ax)

            fig.suptitle(f'{stim_type} pre vs post change | {reward_group}')
            plt.tight_layout()

            rg_str = 'rplus' if reward_group == 'R+' else 'rminus'
            figname = f'prop_pre_vs_post_signed_learning_{rg_str}_{stim_type}'
            putils.save_figure_with_options(fig, ['png', 'pdf', 'svg'], figname, output_dir=output_path)
            plt.close()
    return


# ─────────────────────────────────────────────────────────────────────────────
# SELECTIVITY GRID
# ─────────────────────────────────────────────────────────────────────────────

def plot_selectivity_grids_learning(roc_df_perc, area_groups, output_path, area_col='area_group'):
    """
    Selectivity donut/bar grids per reward_group × learning_group.
    Mirrors the plot_roc_grid section from run_roc_analysis.py.
    """
    ROW_CONFIG = [
        ('Whisker\nresponsive',    'Positive', 'Negative', 'whisker_active',      'positive', 'negative', '#E74C3C', '#3498DB'),
        ('Auditory\nresponsive',   'Positive', 'Negative', 'auditory_active',     'positive', 'negative', '#E74C3C', '#3498DB'),
        ('Modality\nselective',    'Whisker',  'Auditory', 'wh_vs_aud_active',    'whisker',  'auditory', '#ebb134', '#3127c2'),
        ('Choice\nselective',      'Positive', 'Negative', 'choice',              'positive', 'negative', '#E74C3C', '#3498DB'),
        ('Baseline\nchoice',       'Positive', 'Negative', 'baseline_choice',     'positive', 'negative', '#E74C3C', '#3498DB'),
        ('Spontaneous\nlicks',     'Positive', 'Negative', 'spontaneous_licks',   'positive', 'negative', '#E74C3C', '#3498DB'),
    ]
    ROW_CONFIG_PASSIVE = [
        ('Whisker\npre',           'Positive', 'Negative', 'whisker_passive_pre',             'positive', 'negative', '#E74C3C', '#3498DB'),
        ('Whisker\npost',          'Positive', 'Negative', 'whisker_passive_post',            'positive', 'negative', '#E74C3C', '#3498DB'),
        ('Auditory\npre',          'Positive', 'Negative', 'auditory_passive_pre',            'positive', 'negative', '#E74C3C', '#3498DB'),
        ('Auditory\npost',         'Positive', 'Negative', 'auditory_passive_post',           'positive', 'negative', '#E74C3C', '#3498DB'),
        ('Whisker\npre-to-post',   'Positive', 'Negative', 'whisker_pre_vs_post_learning',    'positive', 'negative', '#E74C3C', '#3498DB'),
        ('Auditory\npre-to-post',  'Positive', 'Negative', 'auditory_pre_vs_post_learning',   'positive', 'negative', '#E74C3C', '#3498DB'),
        ('Baseline\npre-to-post',  'Positive', 'Negative', 'baseline_pre_vs_post_learning',   'positive', 'negative', '#E74C3C', '#3498DB'),
    ]

    for reward_group in ['R+', 'R-']:
        rg_str = 'rplus' if reward_group == 'R+' else 'rminus'
        for learning_group in LEARNING_ORDER:
            subset = roc_df_perc[
                (roc_df_perc['reward_group'] == reward_group) &
                (roc_df_perc['learning_group'] == learning_group)
            ]
            if subset.empty:
                continue
            suffix = f'{rg_str}_{learning_group}'
            # Average proportion across sessions so _lookup gets exactly one row
            # per (analysis_type, area, direction) instead of one per session.
            subset_agg = (
                subset.groupby(['analysis_type', area_col, 'direction'], as_index=False)['proportion']
                .mean()
                .fillna(0.0)
            )
            plot_selectivity(subset_agg, row_config=ROW_CONFIG, brain_areas=area_groups,
                             area_col=area_col, style='donut',
                             savepath=os.path.join(output_path, f'roc_grid_donuts_{suffix}.pdf'))
            plot_selectivity(subset_agg, row_config=ROW_CONFIG, brain_areas=area_groups,
                             area_col=area_col, style='bar',
                             savepath=os.path.join(output_path, f'roc_grid_bars_{suffix}.pdf'))
            plot_selectivity(subset_agg, row_config=ROW_CONFIG_PASSIVE, brain_areas=area_groups,
                             area_col=area_col, style='donut',
                             savepath=os.path.join(output_path, f'roc_grid_donuts_passive_{suffix}.pdf'))
            plot_selectivity(subset_agg, row_config=ROW_CONFIG_PASSIVE, brain_areas=area_groups,
                             area_col=area_col, style='bar',
                             savepath=os.path.join(output_path, f'roc_grid_bars_passive_{suffix}.pdf'))
    return


# ─────────────────────────────────────────────────────────────────────────────
# OVERLAP / CO-OCCURRENCE PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def _compute_overlap_matrix(subset, analysis_types):
    """
    Compute conditional co-occurrence matrix: entry [i,j] = % of units sig. in type_i
    that are also sig. in type_j. Returns DataFrame (analysis_types × analysis_types).
    """
    if 'unit_id' not in subset.columns:
        subset = subset.copy()
        id_base = subset['session_id'] if 'session_id' in subset.columns else subset.get('mouse_id', subset.index.astype(str))
        subset['unit_id'] = id_base.astype(str) + '_' + subset.index.astype(str)

    at_present = [at for at in analysis_types if at in subset['analysis_type'].unique()]
    if len(at_present) < 2:
        return None

    pivot = subset[subset['analysis_type'].isin(at_present)].pivot_table(
        index='unit_id', columns='analysis_type', values='significant', aggfunc='max'
    ).fillna(False).astype(bool)

    # ── diagnostic (remove once confirmed working) ──────────────────────────
    n_units = len(pivot)
    n_sig_per_type = pivot.sum()
    n_appear_multiple = (pivot.sum(axis=1) >= 2).sum()
    print(f"  [overlap diag] {n_units} unique unit_ids in pivot")
    print(f"  [overlap diag] sig per type:\n{n_sig_per_type.to_string()}")
    print(f"  [overlap diag] units sig in ≥2 types: {n_appear_multiple}")
    print(f"  [overlap diag] significant dtype: {subset['significant'].dtype}, "
          f"unique vals: {subset['significant'].unique()[:5]}")
    # ────────────────────────────────────────────────────────────────────────

    n = len(at_present)
    mat = np.zeros((n, n))
    for i, at_a in enumerate(at_present):
        if at_a not in pivot.columns:
            continue
        a_mask = pivot[at_a]
        n_a = a_mask.sum()
        for j, at_b in enumerate(at_present):
            if at_b not in pivot.columns:
                continue
            mat[i, j] = (a_mask & pivot[at_b]).sum() / n_a * 100 if n_a > 0 else 0.0

    return pd.DataFrame(mat, index=at_present, columns=at_present)


def plot_overlap_heatmap(roc_df, area_group_order, output_path, analysis_types=None, area_col='area_group'):
    """
    For each (reward_group, learning_group): heatmap of co-occurrence matrix across
    analysis_types. Each cell [i,j] = % of row-type significant units also significant in col-type.
    Also plots learner vs expert side-by-side for comparison.
    """
    if 'unit_id' not in roc_df.columns or roc_df['unit_id'].isna().all():
        print("  Skipping plot_overlap_heatmap: no stable unit_id in data.")
        return

    if analysis_types is None:
        analysis_types = [at for at in ACTIVE_ANALYSIS_TYPES if at in roc_df['analysis_type'].unique()]

    at_labels = [at.replace('_', '\n') for at in analysis_types]

    for reward_group in ['R+', 'R-']:
        rg_str = 'rplus' if reward_group == 'R+' else 'rminus'
        subset_rg = roc_df[roc_df['reward_group'] == reward_group]

        # Side-by-side: learner vs expert
        fig, axes = plt.subplots(1, 2, figsize=(len(analysis_types) * 1.8, len(analysis_types) * 1.2))
        for ax, lg in zip(axes, LEARNING_ORDER):
            sub = subset_rg[subset_rg['learning_group'] == lg]
            mat = _compute_overlap_matrix(sub, analysis_types)
            if mat is None:
                ax.set_title(f'{lg} (insufficient data)')
                continue
            sns.heatmap(mat, ax=ax, annot=True, fmt='.1f', cmap='YlOrRd', square=True,
                        vmin=0, vmax=100,
                        xticklabels=at_labels, yticklabels=at_labels,
                        cbar_kws={'label': '% row-sig also col-sig', 'shrink': 0.7})
            ax.set_title(f'{lg}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

        fig.suptitle(f'Co-occurrence of selectivity classes | {reward_group}')
        plt.tight_layout()
        figname = f'overlap_heatmap_{rg_str}'
        putils.save_figure_with_options(fig, ['png', 'pdf', 'svg'], figname, output_dir=output_path)
        plt.close()

        # Difference: expert - learner
        mat_expert  = _compute_overlap_matrix(subset_rg[subset_rg['learning_group'] == 'expert'],  analysis_types)
        mat_learner = _compute_overlap_matrix(subset_rg[subset_rg['learning_group'] == 'learner'], analysis_types)
        if mat_expert is not None and mat_learner is not None:
            diff = mat_expert - mat_learner
            fig, ax = plt.subplots(figsize=(len(analysis_types) * 1.3, len(analysis_types) * 1.0))
            vmax = np.nanmax(np.abs(diff.values))
            sns.heatmap(diff, ax=ax, annot=True, fmt='.1f', cmap='RdBu_r', square=True,
                        vmin=-vmax, vmax=vmax,
                        xticklabels=at_labels, yticklabels=at_labels,
                        cbar_kws={'label': 'Expert − Learner (%)', 'shrink': 0.7})
            ax.set_title(f'Expert − Learner co-occurrence | {reward_group}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
            plt.tight_layout()
            figname = f'overlap_heatmap_diff_{rg_str}'
            putils.save_figure_with_options(fig, ['png', 'pdf', 'svg'], figname, output_dir=output_path)
            plt.close()

        # Per area_group: heatmaps for each area
        for area in area_group_order:
            subset_area = subset_rg[subset_rg[area_col] == area]
            if subset_area.empty:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(len(analysis_types) * 1.5, len(analysis_types) * 1.0))
            for ax, lg in zip(axes, LEARNING_ORDER):
                sub = subset_area[subset_area['learning_group'] == lg]
                mat = _compute_overlap_matrix(sub, analysis_types)
                if mat is None:
                    ax.set_title(f'{lg} (insufficient data)')
                    continue
                sns.heatmap(mat, ax=ax, annot=True, fmt='.1f', cmap='YlOrRd', square=True,
                            vmin=0, vmax=100,
                            xticklabels=at_labels, yticklabels=at_labels,
                            cbar=False)
                ax.set_title(f'{lg}')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
            fig.suptitle(f'Co-occurrence | {reward_group} | {area}')
            plt.tight_layout()
            area_safe = area.replace(' ', '_').replace('/', '_')
            figname = f'overlap_heatmap_{rg_str}_{area_safe}'
            putils.save_figure_with_options(fig, ['png', 'pdf', 'svg'], figname, output_dir=output_path)
            plt.close()
    return


def plot_overlap_multimembership(roc_df, area_group_order, output_path, analysis_types=None, area_col='area_group'):
    """
    For each reward_group: stacked bar per area_group showing fraction of units significant
    in 0, 1, 2, 3+ analysis_types, comparing learner vs expert side-by-side.
    """
    if 'unit_id' not in roc_df.columns or roc_df['unit_id'].isna().all():
        print("  Skipping plot_overlap_multimembership: no stable unit_id in data.")
        return

    if analysis_types is None:
        analysis_types = [at for at in ACTIVE_ANALYSIS_TYPES if at in roc_df['analysis_type'].unique()]

    area_group_order_filt = _filter_area_group_order(area_group_order, roc_df, area_col)
    membership_order = ['0', '1', '2', '3+']
    membership_colors = ['#d9d9d9', '#fc8d59', '#d7301f', '#7f0000']

    subset_at = roc_df[roc_df['analysis_type'].isin(analysis_types)].copy()

    for reward_group in ['R+', 'R-']:
        rg_str = 'rplus' if reward_group == 'R+' else 'rminus'
        subset_rg = subset_at[subset_at['reward_group'] == reward_group]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)

        for ax, lg in zip(axes, LEARNING_ORDER):
            sub = subset_rg[subset_rg['learning_group'] == lg]
            if sub.empty:
                ax.set_title(f'{lg} (no data)')
                continue

            # Count significant analysis_types per unit per area_group
            unit_counts = sub.groupby(['unit_id', area_col])['significant'].sum().reset_index()
            unit_counts.columns = ['unit_id', area_col, 'n_sig']
            unit_counts['membership'] = unit_counts['n_sig'].apply(
                lambda x: '0' if x == 0 else '1' if x == 1 else '2' if x == 2 else '3+'
            )

            total_per_area = unit_counts.groupby(area_col).size().rename('total').reset_index()
            counts = unit_counts.groupby([area_col, 'membership']).size().rename('count').reset_index()
            counts = counts.merge(total_per_area, on=area_col)
            counts['proportion'] = counts['count'] / counts['total'] * 100

            pivot = counts.pivot_table(index=area_col, columns='membership', values='proportion').fillna(0)
            pivot = pivot.reindex(index=[a for a in area_group_order_filt if a in pivot.index])
            cols_present = [m for m in membership_order if m in pivot.columns]
            pivot = pivot[cols_present]
            colors = [membership_colors[membership_order.index(m)] for m in cols_present]

            pivot.plot(kind='bar', stacked=True, ax=ax, color=colors,
                       legend=False, edgecolor='white', width=0.7)
            ax.set_title(lg)
            ax.set_xlabel('')
            ax.set_ylabel('% units' if lg == LEARNING_ORDER[0] else '')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax.set_ylim(0, 100)
            putils.remove_top_right_frame(ax)

        handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in membership_colors[:len(membership_order)]]
        fig.legend(handles, membership_order, title='# sig. types',
                   loc='upper right', frameon=False, fontsize=9)
        fig.suptitle(f'Multi-class membership | {reward_group}')
        plt.tight_layout()

        figname = f'overlap_multimembership_{rg_str}'
        putils.save_figure_with_options(fig, ['png', 'pdf', 'svg'], figname, output_dir=output_path)
        plt.close()

        # Per-session version: mean number of significant analysis types per unit
        unit_counts_rg = subset_rg.groupby(['session_id', 'unit_id', area_col, 'learning_group'])['significant'].sum().reset_index()
        unit_counts_rg.columns = ['session_id', 'unit_id', area_col, 'learning_group', 'n_sig']
        mouse_mean = unit_counts_rg.groupby(['session_id', area_col, 'learning_group'], as_index=False)['n_sig'].mean()

        palette = _get_learning_palette(reward_group)
        g = sns.catplot(
            data=mouse_mean, kind='bar',
            x=area_col, y='n_sig',
            hue='learning_group', hue_order=LEARNING_ORDER,
            order=area_group_order_filt,
            palette=[palette[k] for k in LEARNING_ORDER],
            height=3.5, aspect=2.5,
            errorbar='se', errwidth=0.7,
        )
        g.figure.suptitle(f'Mean # sig. analysis types per unit | {reward_group}')
        g.despine(left=False)
        g.set_axis_labels('', 'Mean # sig. types per unit')
        g.set_xticklabels(rotation=45, ha='right')

        data_max = mouse_mean['n_sig'].max()
        _annotate_mwu(g.ax, mouse_mean, area_group_order_filt, area_col, 'n_sig',
                      annot_y=data_max * 1.05)

        plt.tight_layout()
        figname = f'overlap_mean_nsig_{rg_str}'
        putils.save_figure_with_options(plt.gcf(), ['png', 'pdf', 'svg'], figname, output_dir=output_path)
        plt.close()
    return


def plot_overlap_donuts_pairs(roc_df, area_group_order, output_path, pairs=None, area_col='area_group'):
    """
    Grid of donut charts: rows = selected analysis-type pairs, columns = area_groups.
    One figure per (reward_group × learning_group).

    Each donut shows 4 slices:
      dark  = significant in both
      col_a = significant only in type A
      col_b = significant only in type B
      gray  = neither
    """
    if 'unit_id' not in roc_df.columns or roc_df['unit_id'].isna().all():
        print("  Skipping plot_overlap_donuts_pairs: no stable unit_id in data.")
        return

    if pairs is None:
        pairs = OVERLAP_PAIRS

    area_group_order_filt = _filter_area_group_order(area_group_order, roc_df, area_col)
    at_in_data = set(roc_df['analysis_type'].unique())
    print(f"  [overlap pairs] analysis_types in data: {sorted(at_in_data)}")
    pairs = [p for p in pairs if p[0] in at_in_data and p[1] in at_in_data]
    if not pairs:
        print("  Skipping plot_overlap_donuts_pairs: none of the requested pairs found in data.")
        return

    NONSIG_COLOR = '#CCCCCC'
    BOTH_COLOR   = '#2C2C2C'
    LWIDTH = 1.8

    for reward_group in ['R+', 'R-']:
        rg_str = 'rplus' if reward_group == 'R+' else 'rminus'

        for lg in LEARNING_ORDER:
            sub = roc_df[(roc_df['reward_group'] == reward_group) &
                         (roc_df['learning_group'] == lg)]
            if sub.empty:
                continue

            # Pivot with session_id in index so we can average fracs per session
            # (consistent denominator philosophy with og donuts).
            all_types = list({t for p in pairs for t in (p[0], p[1])})
            pivot = (
                sub[sub['analysis_type'].isin(all_types)]
                .pivot_table(index=['session_id', 'unit_id', area_col],
                             columns='analysis_type',
                             values='significant',
                             aggfunc='max')
                .fillna(False)
                .astype(bool)
                .reset_index()
            )
            # Total neurons per (session, area) — same denominator as og donut
            # (all neurons, not just those present in this pair's analysis types).
            total_per_sess_area = (
                sub.groupby(['session_id', area_col])['unit_id'].nunique()
                .reset_index(name='_n_total')
            )

            n_rows_grid = len(pairs)
            n_cols_grid = len(area_group_order_filt)
            fig = plt.figure(figsize=(n_cols_grid * 0.9 + LWIDTH, n_rows_grid * 1.0 + 0.65))
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(n_rows_grid, n_cols_grid + 1,
                          width_ratios=[LWIDTH] + [1] * n_cols_grid,
                          wspace=0.04, hspace=0.08,
                          left=0.01, right=0.995, top=0.91, bottom=0.07)

            for i, (at_a, at_b, lbl_a, lbl_b, col_a, col_b) in enumerate(pairs):
                # Row label axis
                ax_lbl = fig.add_subplot(gs[i, 0])
                ax_lbl.axis('off')
                ax_lbl.text(0.97, 0.72, lbl_a, ha='right', va='center',
                            transform=ax_lbl.transAxes, fontsize=6.5,
                            color=col_a, fontweight='bold')
                ax_lbl.text(0.97, 0.28, lbl_b, ha='right', va='center',
                            transform=ax_lbl.transAxes, fontsize=6.5,
                            color=col_b, fontweight='bold')

                # Guard: both types must be in the pivot columns
                if at_a not in pivot.columns or at_b not in pivot.columns:
                    for j in range(n_cols_grid):
                        ax = fig.add_subplot(gs[i, j + 1])
                        ax.axis('off')
                    continue

                for j, area in enumerate(area_group_order_filt):
                    ax = fig.add_subplot(gs[i, j + 1])
                    ax.axis('off')

                    area_piv = pivot[pivot[area_col] == area]
                    if area_piv.empty:
                        continue

                    # Per-session fracs, then average — same as og donuts.
                    # Denominator = all neurons in area (not just those in this pair's types),
                    # so that both+only_a sums to the og donut's whisker proportion.
                    session_fracs = []
                    for sess_id, sess_piv in area_piv.groupby('session_id'):
                        row = total_per_sess_area[
                            (total_per_sess_area['session_id'] == sess_id) &
                            (total_per_sess_area[area_col] == area)
                        ]
                        n_total = int(row['_n_total'].iloc[0]) if len(row) else len(sess_piv)
                        if n_total == 0:
                            continue
                        sig_a = sess_piv[at_a].fillna(False).astype(bool)
                        sig_b = sess_piv[at_b].fillna(False).astype(bool)
                        n_both    = int((sig_a & sig_b).sum())
                        n_only_a  = int((sig_a & ~sig_b).sum())
                        n_only_b  = int((~sig_a & sig_b).sum())
                        n_neither = int((~sig_a & ~sig_b).sum())
                        session_fracs.append([n_both / n_total, n_only_a / n_total,
                                              n_only_b / n_total, n_neither / n_total])
                    if not session_fracs:
                        continue
                    fracs = np.mean(session_fracs, axis=0).tolist()

                    clrs = [BOTH_COLOR, col_a, col_b, NONSIG_COLOR]

                    wedges = [(f, c) for f, c in zip(fracs, clrs) if f > 5e-3]
                    if wedges:
                        fs, cs = zip(*wedges)
                        ax.pie(list(fs), colors=list(cs), startangle=90,
                               wedgeprops={'width': 0.46, 'edgecolor': 'white',
                                           'linewidth': 0.35},
                               counterclock=False)
                    ax.set_aspect('equal')

                    # Percentage annotations — safe even when fracs are 0 or 1
                    ann_labels  = [f'{f*100:.1f}%' if not np.isnan(f) else 'N/A'
                                   for f in fracs]
                    ann_colors  = [BOTH_COLOR, col_a, col_b, '#888888']
                    ann_weights = ['bold', 'bold', 'bold', 'normal']
                    ann_ys      = [0.85, 0.62, 0.38, 0.15]
                    for lbl_ann, c_ann, w_ann, y_ann in zip(ann_labels, ann_colors,
                                                             ann_weights, ann_ys):
                        ax.text(0.92, y_ann, lbl_ann,
                                transform=ax.transAxes,
                                ha='left', va='center',
                                fontsize=3.5, color=c_ann, fontweight=w_ann)

                    if i == 0:
                        lbl = area if len(area) <= 12 else area.replace(' ', '\n', 1)
                        ax.set_title(lbl, fontsize=5.5, pad=1.5, color='#444')

            import matplotlib.patches as mpatches
            legend_handles = [
                mpatches.Patch(color=BOTH_COLOR,   label='Both'),
                mpatches.Patch(color='#888888',    label='Only row'),
                mpatches.Patch(color='#aaaaaa',    label='Only col'),
                mpatches.Patch(color=NONSIG_COLOR, label='Neither'),
            ]
            fig.legend(handles=legend_handles, loc='lower center', ncol=4,
                       fontsize=7, frameon=False, bbox_to_anchor=(0.56, -0.01))
            fig.suptitle(f'Pairwise overlap | {reward_group} | {lg}', y=0.99, fontsize=9)

            figname = f'overlap_pairs_donuts_{rg_str}_{lg}'
            putils.save_figure_with_options(fig, ['png', 'pdf', 'svg'], figname,
                                            output_dir=output_path)
            plt.close()
    return


def plot_overlap_class_proportions(roc_df, area_group_order, output_path, analysis_types=None, area_col='area_group'):
    """
    For each reward_group × area_group: bar plot showing proportion of units in each
    selectivity class, comparing learner vs expert. Each analysis_type is a separate bar
    grouped by learning_group.
    """
    if analysis_types is None:
        analysis_types = [at for at in ACTIVE_ANALYSIS_TYPES if at in roc_df['analysis_type'].unique()]

    area_group_order_filt = _filter_area_group_order(area_group_order, roc_df, area_col)

    for reward_group in ['R+', 'R-']:
        rg_str = 'rplus' if reward_group == 'R+' else 'rminus'
        palette = _get_learning_palette(reward_group)
        subset_rg = roc_df[
            (roc_df['reward_group'] == reward_group) &
            (roc_df['analysis_type'].isin(analysis_types))
        ]

        for area in area_group_order_filt:
            subset_area = subset_rg[subset_rg[area_col] == area]
            if subset_area.empty:
                continue

            # Per session proportion of significant units per analysis_type per learning_group
            per_mouse = subset_area.groupby(
                ['session_id', 'analysis_type', 'learning_group']
            ).agg(prop=('significant', 'mean')).reset_index()
            per_mouse['prop'] *= 100

            g = sns.catplot(
                data=per_mouse, kind='bar',
                x='analysis_type', y='prop',
                hue='learning_group', hue_order=LEARNING_ORDER,
                order=[at for at in analysis_types if at in per_mouse['analysis_type'].unique()],
                palette=[palette[k] for k in LEARNING_ORDER],
                height=3.5, aspect=2,
                errorbar='se', errwidth=0.7,
            )
            g.figure.suptitle(f'{area} | {reward_group}')
            g.despine(left=False)
            g.set_axis_labels('', 'Proportion significant (%)')
            g.set(ylim=(0, min(per_mouse['prop'].max() * 1.3, 100)))
            g.set_xticklabels(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()

            area_safe = area.replace(' ', '_').replace('/', '_')
            figname = f'overlap_class_props_{rg_str}_{area_safe}'
            putils.save_figure_with_options(plt.gcf(), ['png', 'pdf', 'svg'], figname, output_dir=output_path)
            plt.close()
    return


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():

    hostname = socket.gethostname()
    if 'haas' in hostname:
        DATA_PATH   = pathlib.Path('/mnt/lsens-analysis/')
        NWB_PATH    = pathlib.Path('/mnt/lsens-analysis/Axel_Bisi/NWB_combined')
        FIGURE_PATH = pathlib.Path('/mnt/lsens-analysis/Axel_Bisi/combined_results/roc_analysis_experts_vs_learners')
        INFO_PATH   = pathlib.Path('/mnt/share_internal/Axel_Bisi_Share/dataset_info')
    else:
        DATA_PATH   = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis')
        NWB_PATH    = r'M:\analysis\Axel_Bisi\NWB_ks4'
        FIGURE_PATH = r'M:\analysis\Myriam_Hamon\combined_results_ks4\roc_analysis_experts_vs_learners'
        INFO_PATH   = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'share_internal',
                                   'Axel_Bisi_Share', 'dataset_info')

    os.makedirs(FIGURE_PATH, exist_ok=True)

    # ------------------------------------------------------------------
    # LOAD MOUSE INFO  (reward_group only — learning_group comes from file path)
    # ------------------------------------------------------------------
    mouse_info_path = os.path.join(INFO_PATH, 'joint_mouse_reference_weight.xlsx')
    mouse_info_df = pd.read_excel(mouse_info_path)
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df = mouse_info_df[mouse_info_df['reward_group'].isin(['R+', 'R-'])]
    valid_mice = mouse_info_df['mouse_id'].unique()

    # ------------------------------------------------------------------
    # LOAD NWB DATA
    # ------------------------------------------------------------------
    n_workers = 24
    nwb_list = [os.path.join(NWB_PATH, f) for f in os.listdir(NWB_PATH)
                if any(m in f for m in valid_mice)]
    _, unit_table, _ = neural_utils.combine_ephys_nwb(nwb_list, max_workers=n_workers)
    unit_table = allen.process_allen_labels(unit_table, split_merge_areas=True)

    # ------------------------------------------------------------------
    # LOAD ROC DATA FROM ALL SESSIONS
    # learning_group is assigned from the whisker folder:
    #   whisker_0 → 'learner'   (learning-day sessions)
    #   whisker_1+ → 'expert'   (expert-day sessions)
    # ------------------------------------------------------------------
    data_path = os.path.join(DATA_PATH, 'Myriam_Hamon', 'combined_results_ks4')
    roc_df = load_roc_results_all_sessions(data_path, max_workers=n_workers)

    # Keep only mice that passed NWB QC
    unit_table_mice = unit_table['mouse_id'].unique()
    roc_df = roc_df[roc_df['mouse_id'].isin(unit_table_mice)]

    # Fix choice direction (positive and negative are inverted in the raw output)
    choice_mask = roc_df['analysis_type'].str.contains('choice', na=False)
    roc_df.loc[choice_mask, 'direction'] = roc_df.loc[choice_mask, 'direction'].replace(
        {'positive': 'negative', 'negative': 'positive'})

    # Merge reward_group from mouse info (learning_group already in roc_df from file path)
    roc_df = roc_df.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')
    roc_df = roc_df.dropna(subset=['reward_group', 'learning_group'])

    print('Sessions in ROC data:', roc_df['session_id'].nunique())
    print('Mice in ROC data:',     roc_df['mouse_id'].nunique())
    print(roc_df.groupby(['reward_group', 'learning_group'])['session_id'].nunique())

    # ------------------------------------------------------------------
    # MERGE PRESENCE RATIO (Bombcell QC metric) FROM UNIT TABLE
    # ------------------------------------------------------------------
    # unit_table's 'neuron_id' is the per-session positional index into that
    # session's NWB units table -- the same index the ROC pipeline recorded
    # as 'neuron_id' in each CSV row. Recover the whisker_N folder from
    # unit_table's day/behaviour (whisker_N sessions have behaviour=='whisker'
    # and day==N); unit_table's own 'session_id' column can't be used for
    # this join since neural_utils.process_single_nwb sets it to mouse_id.
    unit_table_whisker = unit_table[unit_table['behaviour'] == 'whisker'].copy()
    unit_table_whisker['whisker_folder'] = 'whisker_' + unit_table_whisker['day'].astype(int).astype(str)
    presence_lookup = (
        unit_table_whisker[['mouse_id', 'whisker_folder', 'neuron_id', 'presenceRatio']]
        .drop_duplicates(subset=['mouse_id', 'whisker_folder', 'neuron_id'])
    )
    roc_df = roc_df.merge(presence_lookup, on=['mouse_id', 'whisker_folder', 'neuron_id'], how='left')
    n_matched = roc_df['presenceRatio'].notna().sum()
    print(f'Presence ratio matched for {n_matched}/{len(roc_df)} ROC rows')

    PRESENCE_RATIO_MIN = 0.9

    # ------------------------------------------------------------------
    # PROCESS, COMPUTE PROPORTIONS, AND PLOT
    # Everything below runs once unfiltered and once restricted to units
    # passing the presence-ratio QC threshold (see PRESENCE_RUNS at the
    # bottom), each pass writing into its own output folder.
    # ------------------------------------------------------------------
    def _process_and_plot(roc_df_in, figure_path):

        # --------------------------------------------------------------
        # PROCESS AND FILTER DATA  (filter at fine area level, then aggregate to area_group)
        # --------------------------------------------------------------
        N_UNITS_MIN           = 20
        N_MICE_PER_AREA_MIN   = 5
        N_UNITS_PER_MOUSE_MIN = 10
        KEEP_SHARED_AREAS     = True

        roc_df = roc_df_in[~roc_df_in['area'].isin(allen.get_excluded_areas())]
        roc_df = allen.process_allen_labels(roc_df, split_merge_areas=True)
        # filter_process_data uses area_acronym_custom + mouse_id internally
        roc_df = filter_process_data(
            roc_df,
            n_units_min=N_UNITS_MIN,
            n_mice_per_area_min=N_MICE_PER_AREA_MIN,
            n_units_per_mouse_min=N_UNITS_PER_MOUSE_MIN,
            keep_shared=KEEP_SHARED_AREAS,
        )

        # Map fine areas → coarse area_group for all subsequent plots
        area_to_group = allen.get_custom_area_groups_from_name()
        roc_df['area_group'] = roc_df['area_acronym_custom'].apply(lambda x: area_to_group.get(x, 'Other'))

        # Build area_group order restricted to what's in the data
        area_groups_present = roc_df['area_group'].unique()
        area_group_order = [a for a in AREA_GROUP_ORDER if a in area_groups_present]
        if not area_group_order:
            area_group_order = list(area_groups_present)

        # Build fine-grained (area_acronym_custom) order restricted to what's in the data
        area_acronym_present = roc_df['area_acronym_custom'].unique()
        area_acronym_order = [a for a in allen.get_custom_area_order() if a in area_acronym_present]
        if not area_acronym_order:
            area_acronym_order = list(area_acronym_present)

        roc_df['selectivity_abs'] = roc_df['selectivity'].abs()

        print('Area groups:', area_group_order)
        print('Area acronyms:', area_acronym_order)

        # --------------------------------------------------------------
        # COMPUTE PROPORTIONS (per session, at area_group level and at
        # area_acronym_custom level — learning_group is already in roc_df
        # from the file-path metadata)
        # --------------------------------------------------------------
        print('Computing proportions per session...')
        roc_df_perc = _compute_prop_significant_per_session(roc_df, area_col='area_group')
        roc_df_perc_fine = _compute_prop_significant_per_session(roc_df, area_col='area_acronym_custom')

        # BL-corrected variant
        bl_cols = ['significant_bl_corrected', 'direction_bl_corrected', 'selectivity_bl_corrected']
        has_bl = all(c in roc_df.columns for c in bl_cols)
        if has_bl:
            roc_df_bl = roc_df.copy()
            roc_df_bl['significant']     = roc_df_bl['significant_bl_corrected'].astype(bool)
            roc_df_bl['direction']        = roc_df_bl['direction_bl_corrected']
            roc_df_bl['selectivity']      = roc_df_bl['selectivity_bl_corrected']
            roc_df_bl['selectivity_abs']  = roc_df_bl['selectivity_bl_corrected'].abs()
            roc_df_bl = roc_df_bl.dropna(subset=['significant_bl_corrected'])
            roc_df_perc_bl = _compute_prop_significant_per_session(roc_df_bl, area_col='area_group')
            roc_df_perc_bl_fine = _compute_prop_significant_per_session(roc_df_bl, area_col='area_acronym_custom')

        # --------------------------------------------------------------
        # RUN PLOTS
        # --------------------------------------------------------------

        def _run_all_plots(perc_df, unit_df, path_prefix, label='', area_col='area_group', area_order=None):
            """Run the full set of plots for a given proportion df + unit-level df,
            at whatever area granularity area_col/area_order describe
            (coarse 'area_group' or fine 'area_acronym_custom')."""

            # 1. Proportions per area per reward_group (no expert/learner split)
            out = os.path.join(path_prefix, 'prop_per_group')
            os.makedirs(out, exist_ok=True)
            plot_proportion_per_reward_group(perc_df, area_order, out, area_col=area_col)

            # 2. Proportions: learner vs expert comparison
            out = os.path.join(path_prefix, 'prop_learning')
            os.makedirs(out, exist_ok=True)
            plot_proportion_learning(perc_df, area_order, out, area_col=area_col)
            for d in ['positive', 'negative']:
                out_d = os.path.join(path_prefix, f'prop_learning_{d}')
                os.makedirs(out_d, exist_ok=True)
                plot_proportion_learning(perc_df, area_order, out_d, direction=d, area_col=area_col)

            # 3. Population selectivity: learner vs expert
            out = os.path.join(path_prefix, 'si_learning')
            os.makedirs(out, exist_ok=True)
            plot_pop_selectivity_learning(unit_df, area_order, out, area_col=area_col)
            for d in ['positive', 'negative']:
                out_d = os.path.join(path_prefix, f'si_learning_{d}')
                os.makedirs(out_d, exist_ok=True)
                plot_pop_selectivity_learning(unit_df, area_order, out_d, direction=d, area_col=area_col)

            # 4. Mean metric for significant units
            out = os.path.join(path_prefix, 'mean_si_significant')
            os.makedirs(out, exist_ok=True)
            plot_mean_metric_significant_learning(unit_df, area_order, out,
                                                  y_col='selectivity', y_label='Mean selectivity',
                                                  figname_suffix='mean_si', area_col=area_col)
            out = os.path.join(path_prefix, 'mean_delta_spikes_significant')
            os.makedirs(out, exist_ok=True)
            if 'delta_spikes' in unit_df.columns:
                plot_mean_metric_significant_learning(unit_df, area_order, out,
                                                      y_col='delta_spikes', y_label='Mean Δ spikes',
                                                      figname_suffix='mean_delta_spikes', area_col=area_col)
            if 'delta_spikes_bl_corrected' in unit_df.columns:
                plot_mean_metric_significant_learning(unit_df, area_order, out,
                                                      y_col='delta_spikes_bl_corrected',
                                                      y_label='Mean Δ spikes (BL corr.)',
                                                      figname_suffix='mean_delta_spikes_bl', area_col=area_col)

            # 5. Passive pre vs post
            out = os.path.join(path_prefix, 'passive_pre_vs_post')
            os.makedirs(out, exist_ok=True)
            plot_prop_before_vs_after_learning(perc_df, area_order, out, area_col=area_col)
            for d in ['positive', 'negative']:
                out_d = os.path.join(path_prefix, f'passive_pre_vs_post_{d}')
                os.makedirs(out_d, exist_ok=True)
                plot_prop_before_vs_after_learning(perc_df, area_order, out_d, direction=d, area_col=area_col)

            # 6. SI passive pre vs post
            out = os.path.join(path_prefix, 'si_passive_pre_vs_post')
            os.makedirs(out, exist_ok=True)
            plot_pop_selectivity_before_vs_after_learning(unit_df, area_order, out, area_col=area_col)
            for d in ['positive', 'negative']:
                out_d = os.path.join(path_prefix, f'si_passive_pre_vs_post_{d}')
                os.makedirs(out_d, exist_ok=True)
                plot_pop_selectivity_before_vs_after_learning(unit_df, area_order, out_d, direction=d, area_col=area_col)

            # 7. Signed proportion pre_vs_post learning
            out = os.path.join(path_prefix, 'prop_signed_pre_vs_post')
            os.makedirs(out, exist_ok=True)
            plot_proportion_pre_vs_post_learning(perc_df, area_order, out, area_col=area_col)

            # 8. Selectivity grid (donuts + bars)
            out = os.path.join(path_prefix, 'selectivity_grid')
            os.makedirs(out, exist_ok=True)
            plot_selectivity_grids_learning(perc_df, area_order, out, area_col=area_col)

            # 9. Overlap / co-occurrence
            out = os.path.join(path_prefix, 'overlap')
            os.makedirs(out, exist_ok=True)
            plot_overlap_heatmap(unit_df, area_order, out, area_col=area_col)
            plot_overlap_multimembership(unit_df, area_order, out, area_col=area_col)
            plot_overlap_class_proportions(unit_df, area_order, out, area_col=area_col)
            plot_overlap_donuts_pairs(unit_df, area_order, out, area_col=area_col)

        # Area granularities to run the full plot suite at: (area_col, area_order, output_subdir)
        AREA_RUNS = [
            ('area_group',         area_group_order,    ''),
            ('area_acronym_custom', area_acronym_order, 'fine_areas'),
        ]

        for area_col, area_order, subdir in AREA_RUNS:
            # Standard analysis
            print(f'Running plots (standard, {area_col})...')
            std_path = os.path.join(figure_path, subdir) if subdir else figure_path
            os.makedirs(std_path, exist_ok=True)
            perc_df = roc_df_perc if area_col == 'area_group' else roc_df_perc_fine
            _run_all_plots(perc_df, roc_df, std_path, label='', area_col=area_col, area_order=area_order)

            # BL-corrected analysis
            if has_bl:
                print(f'Running plots (BL corrected, {area_col})...')
                bl_path = os.path.join(std_path, 'bl_corrected')
                os.makedirs(bl_path, exist_ok=True)
                perc_df_bl = roc_df_perc_bl if area_col == 'area_group' else roc_df_perc_bl_fine
                _run_all_plots(perc_df_bl, roc_df_bl, bl_path, label='bl', area_col=area_col, area_order=area_order)

        return

    # Presence-ratio QC runs: unfiltered, and restricted to units with
    # presenceRatio >= PRESENCE_RATIO_MIN, each into its own output folder.
    PRESENCE_RUNS = [
        (roc_df, FIGURE_PATH),
        (roc_df[roc_df['presenceRatio'] >= PRESENCE_RATIO_MIN].copy(),
         os.path.join(FIGURE_PATH, f'presence_ratio_ge_{PRESENCE_RATIO_MIN}')),
    ]

    for roc_df_run, figure_path_run in PRESENCE_RUNS:
        print(f'--- Running full pipeline: {figure_path_run} ({len(roc_df_run)} rows) ---')
        os.makedirs(figure_path_run, exist_ok=True)
        _process_and_plot(roc_df_run, figure_path_run)

    print('Done.')
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    print('- Analysing ROC results: experts vs learners...')
    main()
