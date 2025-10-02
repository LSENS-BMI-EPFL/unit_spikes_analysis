import os
import pandas as pd
import glm_utils
import re
import matplotlib.pyplot as plt
import seaborn as sns
import allen_utils
import NWB_reader_functions as nwb_reader
import numpy as np
import pathlib
import allen_utils as allen
import pickle
import plotting_utils as putils
import ast
import math
from scipy.stats import chi2
import os, re
from joblib import Parallel, delayed
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg') # 'TkAgg' 'Agg' 'Qt5Agg'
from multiprocessing import Pool
from scipy.stats import gaussian_kde


def post_hoc_load_model_results(filename, output_dir):

    file_path = os.path.join(output_dir, '{}_results.parquet'.format(filename))
    try:
        result_df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print('No model results found in:', file_path)
        return None
    return result_df
def load_model_input_output(output_dir):
    with open(os.path.join(output_dir), 'rb') as f:
        data = pickle.load(f)
    X = data['input']
    print(data.keys())
    spikes = data['output']
    feature_names = data['feature_names']
    commit_hash = data['commit_hash']
    neurons_id = data['neurons_id']
    return X, spikes, feature_names, neurons_id


def mouse_glm_results(nwb_list, model_path, plots, output_path, git_version, day_to_analyze = 0):

    # Load and combine NWB files
    trial_table, unit_table, ephys_nwb_list = combine_ephys_nwb(nwb_list, day_to_analyze=day_to_analyze, max_workers=8)



    # Load all models
    df_models = load_models(unit_table['mouse_id'].unique()[0], model_path)
    df_git = df_models[df_models['git_version'] == git_version] # only get the current git version

    if df_git.empty:
        print('Poisson GLMs not fit with that git version for mouse :', unit_table['mouse_id'].unique()[0])
        return None

    merged_df = pd.merge(df_git, unit_table, how='inner', on=["mouse_id", "neuron_id"])

    area_groups = allen.get_custom_area_groups()
    area_colors = allen.get_custom_area_groups_colors()
    merged_df = allen.create_area_custom_column(merged_df)

    if 'metrics' in plots :

        output_folder = os.path.join(output_path, 'metrics')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for model_name in df_git['model_name'].unique():
            if model_name == 'full':
                continue
            plot_full_vs_reduced_per_area(merged_df, model_name, area_groups, area_colors, output_folder)

        plot_kde_full_vs_reduced(merged_df, output_folder)
        # plot_kde_per_trial_type( merged_df[merged_df['model_name'] == 'full'], trial_table, output_folder)
        plot_kde_per_trial_type(merged_df[merged_df['model_name'] == 'full'], trial_table, output_folder)
        plot_corr_per_area_by_trialtype(merged_df[merged_df['model_name'] == 'full'], trial_table, area_groups, output_folder)

    if 'per_unit_kernel_plots' in plots :
        output_folder = os.path.join(output_path, 'per_unit_kernel_plots')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for neuron in merged_df['neuron_id'].unique():
            plot_neuron_kernels_avg_with_responses(
                neuron, merged_df[merged_df['model_name'] == 'full'], ['whisker_stim', 'auditory_stim', 'dlc_lick'], trial_table, output_folder)

    if 'average_predictions_per_trial_types' in plots :
        output_folder = os.path.join(output_path, 'average_predictions_per_trial_types')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        df_git['y_test_array'] = df_git['y_test'].apply(lambda x: np.array(ast.literal_eval(x)))
        df_git['y_pred_array'] = df_git['y_pred'].apply(lambda x: np.array(ast.literal_eval(x)))
        merged_df = pd.merge(df_git, unit_table, on="neuron_id", how="inner")
        plot_predictions_with_reduced_models_parallel(merged_df[merged_df['model_name'] == 'full'], merged_df[merged_df['model_type'] == 'reduced'], trial_table, output_folder)

    if 'average_kernels_by_region' in plots :
        output_folder = os.path.join(output_path, 'average_kernels_by_region')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        lags = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
        plot_average_kernels_by_region( merged_df, output_folder, ['whisker_stim', 'auditory_stim', 'dlc_lick'],
            lags=lags, area_groups=area_groups, area_colors=area_colors, n_cols=3)

    return



def over_mouse_glm_results(nwb_list, plots,info_path, output_path, git_version, day_to_analyze = 0):

    # Load and combine NWB files
    trial_table, unit_table, ephys_nwb_list = combine_ephys_nwb(nwb_list, day_to_analyze=day_to_analyze, max_workers=20)

    all_models = [] #TODO Change to already filter per git_version here
    for mouse in tqdm(unit_table['mouse_id'].unique()):
        models_path = os.path.join(output_path, mouse, "whisker_0", "unit_glm", "models")
        all_models.append(load_models(mouse, models_path))
    df_models = pd.concat(all_models, ignore_index=True)

    mouse_info_path = os.path.join(info_path, 'joint_mouse_reference_weight.xlsx')
    mouse_info_df = pd.read_excel(mouse_info_path)
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df['reward_group'] = mouse_info_df['reward_group'].map({'R+': 1,
                                                                       'R-': 0,
                                                                       'R+proba': 2})
    mouse_info_df = mouse_info_df[(mouse_info_df['exclude'] == 0)
                                  & (mouse_info_df['recording'] == 1)
                                  & (mouse_info_df['reward_group'].isin([0, 1]))]
    mouse_info_df['reward_group'] = mouse_info_df['reward_group'].astype(int)
    unit_table = unit_table.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')

    # Load all models
    df_git = df_models[df_models['git_version'] == git_version] # only get the current git version

    merged_df = pd.merge(df_git, unit_table, how='inner', on=["mouse_id", "neuron_id"])

    area_groups = allen.get_custom_area_groups()
    area_colors = allen.get_custom_area_groups_colors()
    merged_df = allen.create_area_custom_column(merged_df)

    output_path = os.path.join(output_path, 'unit_glm')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if 'metrics' in plots :

        output_folder = os.path.join(output_path, 'metrics')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for model_name in merged_df['model_name'].unique():
            if model_name == 'full':
                continue
            plot_full_vs_reduced_per_area(merged_df, model_name, area_groups, area_colors, output_folder, threshold = None)

        plot_test_corr_vs_firing_rate(merged_df[merged_df['model_name'] == 'full'], output_folder)
        plot_testcorr_per_mouse_reward( merged_df[merged_df['model_name'] == 'full'], output_folder)
        plot_avg_kde_per_trial_type_with_sem(merged_df[merged_df['model_name'] == 'full'], trial_table, output_folder)


    if 'average_kernels_by_region' in plots :
        output_folder = os.path.join(output_path, 'average_kernels_by_region')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        lags = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
        plot_average_kernels_by_region(  merged_df[merged_df['model_name'] == 'full'], output_folder, ['whisker_stim', 'auditory_stim', 'dlc_lick'],
            lags=lags, area_groups=area_groups, area_colors=area_colors, n_cols=3, threshold = 0.2)






def plot_kde_full_vs_reduced(df,output_folder):
    """
    Plot KDEs of test correlations for full and all reduced models.

    :param df: pd.DataFrame with columns ['model_type', 'model_name', 'test_corr']
    :param title: str, figure title
    :param ax: matplotlib.axes.Axes or None
    """

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    # Plot reduced models
    df_reduced = df[df['model_type'] == 'reduced']
    reduced_model_names = df_reduced['model_name'].unique()
    colors = sns.color_palette("husl", len(reduced_model_names))

    for color, model_name in zip(colors, reduced_model_names):
        sub_df = df_reduced[df_reduced['model_name'] == model_name]
        if not sub_df.empty:
            sns.kdeplot(sub_df['test_corr'], ax=ax, color=color, linewidth=1.5,
                        label=f'{model_name} (mean={sub_df["test_corr"].mean():.2f})')

    # Plot full model
    df_full = df[df['model_type'] == 'full']
    sns.kdeplot(df_full['test_corr'], ax=ax, color='black', linewidth=2,
                label=f'Full (mean={df_full["test_corr"].mean():.2f})')

    ax.set_xlabel('Test Score')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(False, linestyle='--', alpha=0.4)
    ax.set_title('Kde_full_vs_reduced')
    putils.save_figure_with_options(fig, file_formats=[ 'png'], filename= 'Kde_full_vs_reduced', output_dir=output_folder)

    return

def plot_kde_per_trial_type(merged, trial_table, output_folder):

    trialtype_corrs = compute_trialtype_correlations(merged,
                                                     trials_df=trial_table
                                                     )


    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    for trial_type, grp in trialtype_corrs.groupby("trial_type"):
        grp["corr"].plot(kind="kde", lw=2, label=f"{trial_type}", ax =ax )


    ax.set_xlabel('Test Score')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.set_title("Test correlation distribution by trial type")
    ax.grid(False, linestyle='--', alpha=0.4)
    ax.set_title('Kde_full_vs_reduced')
    putils.save_figure_with_options(fig, file_formats=[ 'png'], filename= 'Kde_per_trial_type_full_model', output_dir=output_folder)

    return



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
from scipy.stats import gaussian_kde

def plot_avg_kde_per_trial_type_with_sem(merged, trial_table, output_folder):
    """
    Plot average KDE of test correlations per trial type.
    SEM is over folds if only one mouse, or across mice if multiple mice.
    Shaded area represents SEM.
    """
    trialtype_corrs = compute_trialtype_correlations(merged, trials_df=trial_table)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    trial_types = trialtype_corrs["trial_type"].unique()
    mouse_ids = trialtype_corrs["mouse_id"].unique()

    x_grid = np.linspace(-1, 1, 500)  # KDE evaluation points (test_corr range)

    for trial_type in trial_types:
        grp = trialtype_corrs[trialtype_corrs["trial_type"] == trial_type]

        # collect KDEs per mouse
        kde_vals = []

        for mouse in grp["mouse_id"].unique():
            mouse_grp = grp[grp["mouse_id"] == mouse]
            # mean across folds per neuron
            neuron_means = mouse_grp.groupby("neuron_id")["corr"].mean().values
            if len(neuron_means) < 2:
                continue  # skip if not enough neurons
            kde = gaussian_kde(neuron_means)
            kde_vals.append(kde(x_grid))

        if len(kde_vals) == 0:
            continue

        # Convert to array for mean/SEM
        kde_vals_arr = np.array(kde_vals)
        mean_kde = np.mean(kde_vals_arr, axis=0)
        sem_kde = sem(kde_vals_arr, axis=0, nan_policy='omit')

        # Plot mean KDE and shaded SEM
        ax.plot(x_grid, mean_kde, lw=2, label=f"{trial_type}")
        ax.fill_between(x_grid, mean_kde - sem_kde, mean_kde + sem_kde, alpha=0.2)

    ax.set_xlabel("Test Score")
    ax.set_ylabel("Density")
    ax.set_title("Average KDE of Test Correlations per Trial Type")
    ax.legend(fontsize=8)
    ax.grid(False, linestyle='--', alpha=0.4)
    plt.tight_layout()

    # Save figure
    putils.save_figure_with_options(fig, file_formats=['png'],
                                    filename='Avg_KDE_per_trial_type_with_SEM',
                                    output_dir=output_folder)
    plt.close(fig)
    return

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_testcorr_per_mouse_reward(df, output_folder):
    """
    Plot mean ± SEM test correlation per mouse, colored by reward group.
    Rewarded mice on the left, non-rewarded on the right.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'mouse_id', 'neuron_id', 'test_corr', 'reward_group',  'fold'.
        reward_group: 1 = rewarded, 0 = non-rewarded
    output_folder : str
        Path to save figure.
    """

    # Average across folds per neuron
    df_neuron = df.groupby(['mouse_id', 'neuron_id'])['test_corr'].mean().reset_index()


    # Merge reward_group info (assumes one entry per mouse)
    mouse_rewards = df[['mouse_id', 'reward_group']].drop_duplicates()
    df_mouse = df_neuron.groupby('mouse_id')['test_corr'].agg(['mean', 'sem']).reset_index()
    df_mouse = df_mouse.merge(mouse_rewards, on='mouse_id', how='left')

    # Order mice: rewarded first
    df_mouse = df_mouse.sort_values('reward_group', ascending=False).reset_index(drop=True)

    # Colors
    df_mouse['color'] = df_mouse['reward_group'].apply(lambda x: 'forestgreen' if x==1 else 'crimson')

    # Plot
    fig, ax = plt.subplots(figsize=(max(6, len(df_mouse)*0.5), 5), dpi=300)
    x = np.arange(len(df_mouse))
    ax.bar(x, df_mouse['mean'], yerr=df_mouse['sem'], capsize=4, color=df_mouse['color'])
    ax.set_xticks(x)
    ax.set_xticklabels(df_mouse['mouse_id'], rotation=45, ha='right')
    ax.set_ylabel('Test Correlation')
    ax.set_title('Mean ± SEM Test Correlation per Mouse (Reward Group)')
    ax.grid(False)
    plt.tight_layout()

    # Save figure
    filename = 'test_corr_per_mouse_by_reward'
    putils.save_figure_with_options(fig, file_formats=['png'], filename=filename, output_dir=output_folder)
    plt.close(fig)
    return

def plot_test_corr_vs_firing_rate (merged, output_folder):

    merged = merged.groupby(['mouse_id', 'neuron_id', 'reward_group', 'firing_rate'])['test_corr'].mean().reset_index()
    merged['firing_rate'] = pd.to_numeric(merged['firing_rate'], errors='coerce')
    merged['test_corr'] = pd.to_numeric(merged['test_corr'], errors='coerce')

    color_map = {1: 'forestgreen', 0: 'crimson'}

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    for group, subdf in merged.groupby('reward_group'):
        ax.scatter(subdf['firing_rate'], subdf['test_corr'],
                   alpha=0.6, s=30, edgecolor='k',
                   color=color_map[group], label=f'Group {group}')

        # Add regression line per group
        if len(subdf) > 1:
            m, b = np.polyfit(subdf['firing_rate'], subdf['test_corr'], 1)
            ax.plot(subdf['firing_rate'], m * subdf['firing_rate'] + b,
                    color=color_map[group], lw=2,
                    label=f'Group {group} (R={subdf["firing_rate"].corr(subdf["test_corr"]):.2f})')

    ax.set_xlabel('Firing rate (Hz)')
    ax.set_ylabel('Test correlation')
    ax.set_title('Test correlation vs. Firing rate by Reward Group')
    ax.legend()
    plt.tight_layout()
    # Save
    filename = 'test_corr_vs_firing_rate_by_reward_group'
    putils.save_figure_with_options(fig, file_formats=['png'], filename=filename, output_dir=output_folder)
    plt.close(fig)
    return


def plot_corr_per_area_by_trialtype(merged, trial_table, area_groups, output_folder):
    """
    Plot mean ± SEM test correlations per area, with one bar per trial type.

    Parameters
    ----------
    merged : pd.DataFrame
        Must contain 'neuron_id', 'fold', 'test_corr', 'area_acronym_custom', 'trial_type'
    trial_table : pd.DataFrame
        Trial info, used to assign trial types to each neuron/fold if needed
    area_groups : dict
        Mapping from group_name -> list of areas
    area_colors : dict
        Mapping from group_name -> color
    output_folder : str
        Path to save figure
    """

    trial_types_order = ['auditory_hit','auditory_miss', 'whisker_hit', 'whisker_miss', 'catch', 'correct_rejection']

    palettes = {
        'whisker_hit': 'forestgreen',
        'auditory_hit': 'mediumblue',
        'catch': 'k',
        'whisker_miss': 'crimson',  # light green
        'auditory_miss': '#A0C0FF',  # light blue
        'correct_rejection': '#BBBBBB',  # light gray
    }
    # Compute correlation per neuron per trial type
    trialtype_corrs = compute_trialtype_correlations(merged, trials_df=trial_table)

    # Build ordered areas and area colors
    ordered_areas = []
    for group_name, areas in area_groups.items():
        for area in areas:
            if area in trialtype_corrs['area_acronym_custom'].values:
                ordered_areas.append(area)

    # Compute mean & SEM per area and trial type
    trial_types = trialtype_corrs['trial_type'].unique()
    n_areas = len(ordered_areas)
    width = 0.8 / len(trial_types)  # total width divided across trial types
    x = np.arange(n_areas)

    fig, ax = plt.subplots(figsize=(max(12, n_areas * 0.5), 6), dpi=300)

    for i, trial_type in enumerate(trial_types_order):
        if trial_type not in trial_types :
            continue
        means, sems = [], []
        for area in ordered_areas:
            grp = trialtype_corrs[(trialtype_corrs['area_acronym_custom'] == area) &
                                  (trialtype_corrs['trial_type'] == trial_type)]
            values = grp['corr'].values
            means.append(values.mean() if len(values) > 0 else np.nan)
            sems.append(values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0)

        ax.bar(x + i * width - (len(trial_types) - 1) * width / 2,
               means,
               width,
               yerr=sems,
               label=trial_type,
               color=palettes.get(trial_type, 'gray'),  # ← apply color here
               capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(ordered_areas, rotation=45, ha='right')
    ax.set_ylabel("Test Correlation")
    ax.set_title("Mean ± SEM correlation per area by trial type")
    ax.legend()
    ax.grid(False, linestyle='--', alpha=0.4)
    plt.tight_layout()

    putils.save_figure_with_options(fig, file_formats=['png'],
                                    filename='test_correlations_per_area_by_trialtype',
                                    output_dir=output_folder)
    plt.close(fig)
    return

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import pandas as pd

def plot_full_vs_reduced_per_area(df, selected_reduced, area_groups, area_colors, output_folder, threshold=None):
    """
    Plot mean ± SEM test correlations per area for full and one reduced model,
    including significance stars (paired t-test) between models per area.

    :param df: pd.DataFrame with columns ['model_type','model_name','test_corr','area_acronym_custom','mouse_id','neuron_id']
    :param selected_reduced: str, reduced model name
    :param area_groups: dict, group_name -> list of area names
    :param area_colors: dict, group_name -> color
    :param output_folder: str path
    :param threshold: float or None, minimum test_corr_mean for neurons to be included
    """

    # Filter data
    df_full = df[df['model_type'] == 'full'].copy()
    df_reduced = df[(df['model_type'] == 'reduced') & (df['model_name'] == selected_reduced)].copy()

    # Build ordered areas and colors
    ordered_areas = []
    area_to_color = {}
    for group_name, areas in area_groups.items():
        for area in areas:
            if area in df_full['area_acronym_custom'].values or area in df_reduced['area_acronym_custom'].values:
                ordered_areas.append(area)
                area_to_color[area] = area_colors[group_name]

    # Initialize lists
    means_full, sems_full, means_reduced, sems_reduced, bar_colors = [], [], [], [], []

    # Plot preparation
    fig, ax = plt.subplots(figsize=(max(12, len(ordered_areas) * 0.5), 6), dpi=300)
    x = np.arange(len(ordered_areas))
    width = 0.35

    for i, area in enumerate(ordered_areas):
        # --- Full model ---
        full_grp = df_full[df_full['area_acronym_custom'] == area]

        fold_means_full = (
            full_grp.groupby(['mouse_id', 'neuron_id'], as_index=False)['test_corr']
            .mean()
        )

        # Apply threshold if specified
        if threshold is not None:
            neurons_to_keep = fold_means_full[fold_means_full['test_corr'] >= threshold]
        else:
            neurons_to_keep = fold_means_full

        # Keep only neurons passing threshold for full & reduced
        full_values = fold_means_full.merge(
            neurons_to_keep[['mouse_id', 'neuron_id']],
            on=['mouse_id', 'neuron_id'],
            how='inner'
        )['test_corr'].to_numpy()

        # --- Reduced model ---
        reduced_grp = df_reduced[df_reduced['area_acronym_custom'] == area]
        fold_means_reduced = (
            reduced_grp.groupby(['mouse_id', 'neuron_id'], as_index=False)['test_corr']
            .mean()
        )

        reduced_values = fold_means_reduced.merge(
            neurons_to_keep[['mouse_id', 'neuron_id']],
            on=['mouse_id', 'neuron_id'],
            how='inner'
        )['test_corr'].to_numpy()

        # Compute means & SEMs
        means_full.append(full_values.mean() if len(full_values) > 0 else np.nan)
        sems_full.append(full_values.std(ddof=1) / np.sqrt(len(full_values)) if len(full_values) > 1 else 0)

        means_reduced.append(reduced_values.mean() if len(reduced_values) > 0 else np.nan)
        sems_reduced.append(reduced_values.std(ddof=1) / np.sqrt(len(reduced_values)) if len(reduced_values) > 1 else 0)

        bar_colors.append(area_to_color.get(area, 'gray'))

        # --- Significance test and star annotation ---
        if len(full_values) > 1 and len(reduced_values) > 1:
            stat, pval = ttest_rel(full_values, reduced_values)
        else:
            pval = np.nan


        if pval < 0.05:
            star = '*'
        else:
            star = ''

        # Annotate above bars
        if star:
            y = max(means_full[-1] + sems_full[-1], means_reduced[-1] + sems_reduced[-1])
            ax.text(x[i], y + 0.01, star, ha='center', va='bottom', fontsize=12, color='red')

    # --- Plot bars ---
    ax.bar(x - width / 2, means_full, width, yerr=sems_full, label='Full', color='black', capsize=4)
    ax.bar(x + width / 2, means_reduced, width, yerr=sems_reduced, label=f'Reduced: {selected_reduced}',
           color=bar_colors, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(ordered_areas, rotation=45, ha='right')
    ax.set_ylabel('Test Score')
    ax.set_title(f'Full vs {selected_reduced} per area')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    # Save figure
    name = f'Full vs {selected_reduced} per area'
    if threshold is not None:
        name += f' threshold {threshold}'
    putils.save_figure_with_options(fig, file_formats=['png'], filename=name, output_dir=output_folder)
    plt.close(fig)
    return

def compute_lrt_from_model_results(model_results_df, alpha=0.05, ll_field='test_ll'):
    # Extract only full model
    full_df = model_results_df[model_results_df['model_name'] == 'full']

    # Identify all reduced models
    reduced_models = model_results_df['model_name'].unique()
    reduced_models = [m for m in reduced_models if m != 'full']

    results = []

    for reduced_model in reduced_models:
        reduced_df = model_results_df[model_results_df['model_name'] == reduced_model]

        # Merge on neuron and fold
        merged = pd.merge(
            full_df[['neuron_id', 'fold', ll_field, 'predictors']],
            reduced_df[['neuron_id', 'fold', ll_field, 'predictors']],
            on=['neuron_id', 'fold'],
            suffixes=('_full', '_reduced')
        )

        # Compute LRT statistic per fold
        merged['lrt_stat'] = 2 * (merged[f'{ll_field}_full'] - merged[f'{ll_field}_reduced'])

        # Compute degrees of freedom difference
        merged['df_diff'] = merged['predictors_full'].apply(len) - merged['predictors_reduced'].apply(len)

        # Aggregate per neuron
        grouped = merged.groupby('neuron_id').agg(
            mean_lrt_stat=('lrt_stat', 'sum'),
            df_diff=('df_diff', 'first')  # assume same across folds
        ).reset_index()

        # Compute p-values
        grouped['p_value'] = 1 - chi2.cdf(grouped['mean_lrt_stat'], df=grouped['df_diff'])
        grouped['significant'] = grouped['p_value'] < alpha
        grouped['reduced_model'] = reduced_model

        results.append(grouped)

    lrt_df = pd.concat(results, ignore_index=True)
    return lrt_df

def classify_trial(row):
    if row["trial_type"] == "whisker_trial":
        if row["lick_flag"] == 1 :
            return "whisker_hit"
        else:
            return "whisker_miss"

    elif row["trial_type"] == "auditory_trial":
        if row["lick_flag"] == 1 :
            return "auditory_hit"
        else:
            return "auditory_miss"

    elif row["trial_type"] == "no_stim_trial":
        if row["lick_flag"] == 1:
            return "catch"
        else:
            return "correct_rejection"

    else:
        return "other"
def zscore_f(arr):
    return (arr[0:80] - np.mean(arr[0:80])) / (np.std(arr[0:80]) + 1e-8)

from scipy.stats import pearsonr

def compute_trialtype_correlations(merged, trials_df):
    """
    Compute test Pearson correlation per neuron, per fold, per trial type, including mouse_id.
    """
    rows = []
    for _, row in merged.iterrows():
        neuron_id = row["neuron_id"]
        fold = row["fold"]
        mouse_id = row.get("mouse_id", "unknown")  # assume merged has mouse_id column
        area_custom = row.get("area_acronym_custom", None)

        # decode arrays
        y_test = np.array(ast.literal_eval(row["y_test"]))
        y_pred = np.array(ast.literal_eval(row["y_pred"]))
        n_bins = row["n_bins"]

        # reshape into trials x bins
        n_trials = y_pred.shape[0] // n_bins
        y_test = y_test.reshape(n_trials, n_bins)
        y_pred = y_pred.reshape(n_trials, n_bins)

        # align with trials dataframe
        test_trial_ids = np.array(ast.literal_eval(row["test_trials"]))
        trials_test_df = trials_df.iloc[test_trial_ids, :]

        for trial_type in trials_test_df["behav_type"].unique():
            idx = np.where(trials_test_df["behav_type"].values == trial_type)[0]
            if len(idx) < 2:
                continue  # not enough trials to compute corr

            y_true_t = y_test[idx, :].ravel()
            y_pred_t = y_pred[idx, :].ravel()

            if len(np.unique(y_true_t)) > 1:
                r, _ = pearsonr(y_true_t, y_pred_t)
                rows.append({
                    "mouse_id": mouse_id,
                    "neuron_id": neuron_id,
                    "fold": fold,
                    "trial_type": trial_type,
                    "corr": r,
                    "area_acronym_custom": area_custom
                })

    return pd.DataFrame(rows)


def plot_model_comparison(
    neuron_ids, df_full, df_reduced, trials_df, output_folder, name, reduced_model="whisker_encoding",
    bin_size=0.1, zscore=False
):
    """
    Plot average neural data, full model, and reduced model predictions
    across multiple neurons and trial types, with SEM across folds.
    """

    # ------------------------
    # FULL MODEL
    # ------------------------

    all_y_test, all_y_pred_full = {}, {}

    for nid in neuron_ids:
        res_all = df_full[df_full["neuron_id"] == nid]
        if res_all.empty:
            print("No data for neuron", nid)
            continue

        for res in res_all.itertuples(index=False):
            y_test = res.y_test_array
            y_pred = res.y_pred_array
            n_bins = res.n_bins

            n_trials = y_pred.shape[0] // n_bins
            y_test = y_test.reshape(n_trials, n_bins)
            y_pred = y_pred.reshape(n_trials, n_bins)

            test_trial_ids = np.array(ast.literal_eval(res.test_trials))
            order = np.argsort(test_trial_ids)
            y_test = y_test[order, :]
            y_pred = y_pred[order, :]
            trials_test_df = trials_df.iloc[test_trial_ids[order], :]

            for trial_type in trials_test_df["behav_type"].unique():
                idx = np.where(trials_test_df["behav_type"] == trial_type)[0]
                if len(idx) == 0:
                    continue

                test_mean = np.mean(y_test[idx], axis=0)
                pred_mean = np.mean(y_pred[idx], axis=0)

                if zscore:
                    test_mean = zscore_f(test_mean)
                    pred_mean = zscore_f(pred_mean)

                all_y_test.setdefault(trial_type, []).append(test_mean)
                all_y_pred_full.setdefault(trial_type, []).append(pred_mean)

    # ------------------------
    # REDUCED MODEL
    # ------------------------

    all_y_pred_reduced = {}

    for nid in neuron_ids:
        res_all = df_reduced[df_reduced["neuron_id"] == nid]
        if res_all.empty:
            continue

        for res in res_all.itertuples(index=False):
            y_pred = res.y_pred_array
            n_bins = res.n_bins
            n_trials = y_pred.shape[0] // n_bins
            y_pred = y_pred.reshape(n_trials, n_bins)

            test_trial_ids = np.array(ast.literal_eval(res.test_trials))
            order = np.argsort(test_trial_ids)
            y_pred = y_pred[order, :]
            trials_test_df = trials_df.iloc[test_trial_ids[order], :]

            for trial_type in trials_test_df["behav_type"].unique():
                idx = np.where(trials_test_df["behav_type"] == trial_type)[0]
                if len(idx) == 0:
                    continue

                pred_mean = np.mean(y_pred[idx], axis=0)
                if zscore:
                    pred_mean = zscore_f(pred_mean)

                all_y_pred_reduced.setdefault(trial_type, []).append(pred_mean)

    # ------------------------
    # Fits (average correlation)
    # ------------------------
    fits_full = (
        df_full[df_full["neuron_id"].isin(neuron_ids)]["test_corr"]
        .astype(float).mean()
    )
    fits_reduced = (
        df_reduced[df_reduced["neuron_id"].isin(neuron_ids)]["test_corr"]
        .astype(float).mean()
    )

    # ------------------------
    # PLOTTING
    # ------------------------
    trial_types = sorted(all_y_test.keys())
    plt.ioff()
    fig, axes = plt.subplots(1, len(trial_types), figsize=(15, 5), sharey=True)
    window_bounds_sec = (-1, 2)
    time_stim = 0
    time = np.linspace(window_bounds_sec[0] + bin_size/2,
                       window_bounds_sec[1] - bin_size/2,
                       n_bins)

    if len(neuron_ids) == 1:
        fig.suptitle(
            f"Reduced model {reduced_model}, neuron {neuron_ids[0]}\n"
            f"full fit={fits_full:.3f}, reduced fit={fits_reduced:.3f}"
        )
    else:
        fig.suptitle(
            f"Reduced model {reduced_model}, {len(neuron_ids)} neurons\n"
            f"full fit={fits_full:.3f}, reduced fit={fits_reduced:.3f}"
        )

    if len(trial_types) == 1:
        axes = [axes]

    for ax, trial_type in zip(axes, trial_types):
        if trial_type not in all_y_pred_reduced:
            continue

        putils.remove_top_right_frame(ax)
        ax.set_ylabel("Spikes", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=10)


        # Stim marker
        if trial_type == "whisker_hit":
            ax.axvline(time_stim, color="forestgreen", linestyle="-", linewidth=1)
        elif trial_type == "whisker_miss":
            ax.axvline(time_stim, color="orange", linestyle="-", linewidth=1)
        elif trial_type == "auditory_hit":
            ax.axvline(time_stim, color="mediumblue", linestyle="-", linewidth=1)
        elif trial_type == "auditory_miss":
            ax.axvline(time_stim, color="skyblue", linestyle="-", linewidth=1)
        elif trial_type == "catch":
            ax.axvline(time_stim, color="gray", linestyle="-", linewidth=1)
        elif trial_type == "correct_rejection":
            ax.axvline(time_stim, color="black", linestyle="-", linewidth=1)

        # --- Data
        data_stack = np.stack(all_y_test[trial_type])
        mean_data = data_stack.mean(axis=0)[:80]
        sem_data = data_stack.std(axis=0, ddof=1) / np.sqrt(data_stack.shape[0])
        ax.plot(time, mean_data, color="black", label="data")
        ax.fill_between(time, mean_data - sem_data[:80], mean_data + sem_data[:80], color="black", alpha=0.3)

        # --- Full model
        full_stack = np.stack(all_y_pred_full[trial_type])
        mean_full = full_stack.mean(axis=0)[:80]
        sem_full = full_stack.std(axis=0, ddof=1) / np.sqrt(full_stack.shape[0])
        ax.plot(time, mean_full, color="green", label="full")
        ax.fill_between(time, mean_full - sem_full[:80], mean_full + sem_full[:80], color="green", alpha=0.3)

        # --- Reduced model
        reduced_stack = np.stack(all_y_pred_reduced[trial_type])
        mean_reduced = reduced_stack.mean(axis=0)[:80]
        sem_reduced = reduced_stack.std(axis=0, ddof=1) / np.sqrt(reduced_stack.shape[0])
        ax.plot(time, mean_reduced, color="red", label="reduced")
        ax.fill_between(time, mean_reduced - sem_reduced[:80], mean_reduced + sem_reduced[:80], color="red", alpha=0.3)

        ax.set_title(trial_type, fontsize=14)

    axes[min(2, len(axes)-1)].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_folder + f'/{name}.png')
    plt.close(fig)
    return

def plot_models_comparison(
    neuron_ids, df, unit_table, trials_df, output_folder, name, reduced_models=["whisker_encoding"], bin_size=0.1, do_zscore=False
):
    """
    Plot average neural data, full model, and reduced model predictions
    across multiple neurons, trial types, and reduced models.
    """

    def zscore_f(arr):
        return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)

    xticklabels = [-3, -2, -1, 0, 1, 2, 3, 4, 5]
    time_stim = 3

    # ------------------------
    # FULL MODEL
    # ------------------------
    df_full = df[(df["model_type"] == "full")]
    merged_full = pd.merge(df_full, unit_table, on="neuron_id", how="inner")

    all_y_test, all_y_pred_full = {}, {}

    for nid in neuron_ids:
        res = merged_full[merged_full["neuron_id"] == nid]
        if res.empty:
            continue

        y_test = np.array(ast.literal_eval(res["y_test"].values[0]))
        y_pred = np.array(ast.literal_eval(res["y_pred"].values[0]))
        n_bins = res["n_bins"].values[0]

        n_trials = y_pred.shape[0] // n_bins
        y_test = y_test.reshape(n_trials, n_bins)
        y_pred = y_pred.reshape(n_trials, n_bins)

        test_trial_ids = np.array(ast.literal_eval(res["test_trials"].values[0]))
        order = np.argsort(test_trial_ids)
        y_test = y_test[order, :]
        y_pred = y_pred[order, :]

        trials_test_df = trials_df.iloc[test_trial_ids[order], :]

        for trial_type in trials_test_df["trial_type"].unique():
            idx = np.where(trials_test_df["trial_type"] == trial_type)[0]
            y_test_mean = np.mean(y_test[idx], axis=0)
            y_pred_mean = np.mean(y_pred[idx], axis=0)

            if do_zscore:
                y_test_mean = zscore_f(y_test_mean)
                y_pred_mean = zscore_f(y_pred_mean)

            all_y_test.setdefault(trial_type, []).append(y_test_mean)
            all_y_pred_full.setdefault(trial_type, []).append(y_pred_mean)

    # Trial types for plotting
    trial_types = sorted(all_y_test.keys())
    time = np.arange(n_bins) * bin_size
    xticks = np.linspace(0, max(time), len(xticklabels))

    # ------------------------
    # FIGURE SETUP
    # ------------------------
    n_models = len(reduced_models)
    fig, axes = plt.subplots(
        n_models, len(trial_types),
        figsize=(5 * len(trial_types), 4 * n_models),
        sharey=True
    )

    if n_models == 1:
        axes = np.expand_dims(axes, 0)  # make 2D for consistency
    if len(trial_types) == 1:
        axes = np.expand_dims(axes, 1)

    # ------------------------
    # LOOP OVER MODELS
    # ------------------------
    for m_idx, reduced_model in enumerate(reduced_models):

        df_reduced = df[(df["model_type"] == "reduced") ]
        merged_reduced = pd.merge(df_reduced, unit_table, on="neuron_id", how="inner")
        merged_reduced = merged_reduced[merged_reduced["model_name"] == reduced_model]

        all_y_pred_reduced = {}

        for nid in neuron_ids:
            res_all = merged_reduced[merged_reduced["neuron_id"] == nid]
            if res_all.empty:
                continue

            rows = []
            for fold in res_all["fold"].unique():
                res = res_all[res_all["fold"] == fold]

                y_pred = np.array(ast.literal_eval(res["y_pred"].values[0]))
                n_bins = res["n_bins"].values[0]

                n_trials = y_pred.shape[0] // n_bins
                y_pred = y_pred.reshape(n_trials, n_bins)

                test_trial_ids = np.array(ast.literal_eval(res["test_trials"].values[0]))
                order = np.argsort(test_trial_ids)
                y_pred = y_pred[order, :]

                trials_test_df = trials_df.iloc[test_trial_ids[order], :]

                for trial_type in trials_test_df["trial_type"].unique():
                    idx = np.where(trials_test_df["trial_type"] == trial_type)[0]
                    y_pred_mean = np.mean(y_pred[idx], axis=0)
                    if do_zscore:
                        y_pred_mean = zscore_f(y_pred_mean)
                    rows.append({"trial_type": trial_type, "fold": fold, "y_pred_mean": y_pred_mean})

            df_avg = pd.DataFrame(rows).groupby("trial_type")["y_pred_mean"].mean().reset_index()
            for _, row in df_avg.iterrows():
                all_y_pred_reduced.setdefault(row["trial_type"], []).append(row["y_pred_mean"])

        fits_full = df_full[df_full["neuron_id"].isin(neuron_ids)]["test_corr"].astype(float).mean()
        fits_reduced = merged_reduced[merged_reduced["neuron_id"].isin(neuron_ids)]["test_corr"].astype(float).mean()

        # ------------------------
        # PLOT
        # ------------------------
        for t_idx, trial_type in enumerate(trial_types):
            ax = axes[m_idx, t_idx]
            putils.remove_top_right_frame(ax)

            ax.set_ylabel("Spikes (z)" if do_zscore else "Spikes/s", fontsize=10)
            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, fontsize=10)

            if trial_type == "whisker_trial":
                ax.axvline(time_stim, color="forestgreen", linestyle="-", linewidth=1)
            elif trial_type == "auditory_trial":
                ax.axvline(time_stim, color="mediumblue", linestyle="-", linewidth=1)
            elif trial_type == "no_stim_trial":
                ax.axvline(time_stim, color="k", linestyle="-", linewidth=1)

            # Plot averages
            ax.plot(time, np.mean(all_y_test[trial_type], axis=0),
                    color="black", label="data")
            ax.plot(time, np.mean(all_y_pred_full[trial_type], axis=0),
                    color="green", label=f"full (fit={fits_full:.3f})")
            if trial_type in all_y_pred_reduced:
                ax.plot(time, np.mean(all_y_pred_reduced[trial_type], axis=0),
                        color="red", label=f"reduced (fit={fits_reduced:.3f})")

            if m_idx == 0:
                ax.set_title(trial_type, fontsize=14)
            if t_idx == len(trial_types) - 1:
                ax.legend()

        # Row label for model
        axes[m_idx, 0].set_ylabel(
            f"{reduced_model}\nSpikes (z)" if do_zscore else f"{reduced_model}\nSpikes/s",
            fontsize=12
        )

    fig.suptitle(f"Model comparison: {len(neuron_ids)} neurons", fontsize=16)
    plt.tight_layout()

    plt.savefig(output_folder + f'/{name}.png')
    plt.close('all')
    return



def make_pickleable_df(df):
    """
    Keep only the columns required for plotting and convert arrays to numpy.
    """
    df_slim = df[['neuron_id', 'y_test_array', 'y_pred_array', 'n_bins', 'test_trials', 'fold', 'test_corr', 'model_name']].copy()

    # Convert any h5py or list objects to numpy arrays
    df_slim['y_test_array'] = df_slim['y_test_array'].apply(lambda x: np.array(x))
    df_slim['y_pred_array'] = df_slim['y_pred_array'].apply(lambda x: np.array(x))

    return df_slim



def process_neuron(neuron_id, model, trials_df, output_folder, df_full_slim, df_reduced_slim):
    """
    Worker function to plot a single neuron.
    """
    df_full_neuron = df_full_slim[df_full_slim['neuron_id'] == neuron_id]
    df_reduced_neuron = df_reduced_slim[df_reduced_slim['neuron_id'] == neuron_id]

    if df_full_neuron.empty or df_reduced_neuron.empty:
        return

    plt.ioff()  # non-interactive
    plot_model_comparison(
        [neuron_id],
        df_full_neuron,
        df_reduced_neuron,
        trials_df,
        output_folder,
        name=str(neuron_id),
        reduced_model=model,
        bin_size=0.1,
        zscore=False
    )


import os
import multiprocessing

def plot_predictions_with_reduced_models_parallel(df_full_slim, df_reduced_slim, trials_df, output_folder_base):
    n_jobs = max(1, multiprocessing.cpu_count() - 1)

    for model in df_reduced_slim['model_name'].unique():
        if model == 'full':
            continue
        print(model)
        df_full_slim_model = df_reduced_slim[df_reduced_slim['model_name'] == model]

        output_folder = os.path.join(output_folder_base, model)
        os.makedirs(output_folder, exist_ok=True)
        neuron_ids = df_full_slim['neuron_id'].unique()

        Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
            delayed(process_neuron)(
                neuron_id, model, trials_df, output_folder, df_full_slim, df_full_slim_model
            )
            for neuron_id in neuron_ids
        )


def plot_neuron_kernels_avg_with_responses(neuron_id, glm_full_df, kernels, trials_df, output_folder, bin_size=0.1):
    """
    Plot kernels for one neuron alongside average responses and predictions.
    Uses SEM across folds (not across trials).
    """
    print('CAREFUL NOW THE KERNEL GET PLOTTED FOR -0.2 to 0.4')
    # -------------------
    # KERNELS (per fold)
    # -------------------
    coefs_full_str = glm_full_df.loc[glm_full_df['neuron_id'] == neuron_id, 'coef'].tolist()
    lags = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
    coefs_full = [np.array(ast.literal_eval(c)) for c in coefs_full_str]
    coefs_full = np.stack(coefs_full, axis=0)   # shape (n_folds, n_predictors)

    # Predictors list for indexing kernels
    predictors = ast.literal_eval(
        glm_full_df.loc[glm_full_df['neuron_id'] == neuron_id, 'predictors'].iloc[0]
    )

    # -------------------
    # RESPONSES & PREDICTIONS (all folds)
    # -------------------

    res_all = glm_full_df[glm_full_df["neuron_id"] == neuron_id]
    if res_all.empty:
        raise ValueError(f"No full model data found for neuron {neuron_id}")

    time_stim = 3
    xticklabels = [-3, -2, -1, 0, 1, 2, 3, 4, 5]

    # -------------------
    # FIGURE
    # -------------------
    n_kernels = len(kernels)
    fig, axes = plt.subplots(n_kernels, 2, figsize=(10, 3 * n_kernels), sharex=False)
    if n_kernels == 1:
        axes = np.array([axes])  # ensure 2D array

    for i, predictor in enumerate(kernels):
        # --- Kernel panel
        ax_k = axes[i, 0]
        indices = np.where([p.startswith(predictor) for p in predictors])[0]
        if len(indices) == 0:
            continue
        # Stack kernel values for this predictor across folds
        kernel_stack = coefs_full[:, indices].reshape(coefs_full.shape[0], -1)
        mean_kernel = kernel_stack.mean(axis=0)
        sem_kernel = kernel_stack.std(axis=0, ddof=1) / np.sqrt(kernel_stack.shape[0])

        ax_k.plot(lags, mean_kernel, color="blue")
        ax_k.fill_between(lags, mean_kernel - sem_kernel, mean_kernel + sem_kernel, color="blue", alpha=0.3)
        ax_k.set_title(f"{predictor} kernel")
        ax_k.set_xlabel("Lag (s)")
        ax_k.set_ylabel("Coef")

        # --- Response panel
        ax_r = axes[i, 1]
        fold_means_test, fold_means_pred = {}, {}

        for _, res in res_all.iterrows():
            y_test = np.array(ast.literal_eval(res["y_test"]))
            y_pred = np.array(ast.literal_eval(res["y_pred"]))
            n_bins = res["n_bins"]

            n_trials = y_pred.shape[0] // n_bins
            y_test = y_test.reshape(n_trials, n_bins)
            y_pred = y_pred.reshape(n_trials, n_bins)

            test_trial_ids = np.array(ast.literal_eval(res["test_trials"]))
            order = np.argsort(test_trial_ids)
            y_test = y_test[order, :]
            y_pred = y_pred[order, :]
            trials_test_df = trials_df.iloc[test_trial_ids[order], :]

            # Compute fold means for trial types
            for t in trials_test_df["behav_type"].unique():
                idx = np.where(trials_test_df["behav_type"] == t)[0]
                if len(idx) == 0:
                    continue
                fold_means_test.setdefault(t, []).append(np.mean(y_test[idx], axis=0))
                fold_means_pred.setdefault(t, []).append(np.mean(y_pred[idx], axis=0))

        window_bounds_sec = (-1, 2)
        time_stim = 0
        time = np.linspace(window_bounds_sec[0] + bin_size/2,
                           window_bounds_sec[1] - bin_size/2,
                           n_bins)
        # Plot depending on predictor
        if predictor == "whisker_stim":
            for t, col in zip(["whisker_hit", "whisker_miss"], ["green", "orange"]):
                if t not in fold_means_test: continue
                test_stack = np.stack(fold_means_test[t])
                pred_stack = np.stack(fold_means_pred[t])
                mean_test = test_stack.mean(axis=0)
                sem_test = test_stack.std(axis=0, ddof=1) / np.sqrt(test_stack.shape[0])
                mean_pred = pred_stack.mean(axis=0)
                sem_pred = pred_stack.std(axis=0, ddof=1) / np.sqrt(pred_stack.shape[0])

                ax_r.plot(time, mean_test, color=col, label=f"{t} data")
                ax_r.fill_between(time, mean_test - sem_test, mean_test + sem_test, color=col, alpha=0.3)
                ax_r.plot(time, mean_pred, color=col, linestyle="--", label=f"{t} pred")
                ax_r.fill_between(time, mean_pred - sem_pred, mean_pred + sem_pred, color=col, alpha=0.2)

            ax_r.set_title("Whisker hits vs misses")

        elif predictor == "auditory_stim":
            t = "auditory_hit"
            if t in fold_means_test:
                test_stack = np.stack(fold_means_test[t])
                pred_stack = np.stack(fold_means_pred[t])
                mean_test = test_stack.mean(axis=0)
                sem_test = test_stack.std(axis=0, ddof=1) / np.sqrt(test_stack.shape[0])
                mean_pred = pred_stack.mean(axis=0)
                sem_pred = pred_stack.std(axis=0, ddof=1) / np.sqrt(pred_stack.shape[0])

                ax_r.plot(time, mean_test, color="black", label="auditory_hit data")
                ax_r.fill_between(time, mean_test - sem_test, mean_test + sem_test, color="black", alpha=0.3)
                ax_r.plot(time, mean_pred, color="blue", linestyle="--", label="auditory_hit pred")
                ax_r.fill_between(time, mean_pred - sem_pred, mean_pred + sem_pred, color="blue", alpha=0.2)

            ax_r.set_title("Auditory hits")

        elif predictor == "dlc_lick":
            t = "catch"
            if t in fold_means_test:
                test_stack = np.stack(fold_means_test[t])
                pred_stack = np.stack(fold_means_pred[t])
                mean_test = test_stack.mean(axis=0)
                sem_test = test_stack.std(axis=0, ddof=1) / np.sqrt(test_stack.shape[0])
                mean_pred = pred_stack.mean(axis=0)
                sem_pred = pred_stack.std(axis=0, ddof=1) / np.sqrt(pred_stack.shape[0])

                ax_r.plot(time, mean_test, color="black", label="false_alarm data")
                ax_r.fill_between(time, mean_test - sem_test, mean_test + sem_test, color="black", alpha=0.3)
                ax_r.plot(time, mean_pred, color="red", linestyle="--", label="false_alarm pred")
                ax_r.fill_between(time, mean_pred - sem_pred, mean_pred + sem_pred, color="red", alpha=0.2)

            ax_r.set_title("False alarms")

        # Formatting
        ax_r.set_xlabel("Time (s)")
        ax_r.set_ylabel("Spikes")
        ax_r.axvline(time_stim, color="k", linestyle=":")
        ax_r.legend(fontsize=8)


    plt.tight_layout()
    plt.savefig(output_folder + f'/{neuron_id}.png')
    plt.close('all')
    return

def plot_average_kernels_by_region(df, output_folder, kernels_to_plot,
                                   lags=None, area_groups=None, area_colors=None, n_cols=3,  threshold = None):
    """
    Plot average kernels across neurons grouped by area_acronym_custom (regions),
    one figure per kernel. Regions are colored by area group, ordered by area_groups.

    For each neuron (per mouse), coefficients are averaged across folds first,
    then mean ± SEM is computed across neurons in the region.

    Parameters
    ----------
    df : pd.DataFrame
        Must include columns: ['mouse_id','neuron_id','area_acronym_custom','coef','predictors','fold']
    output_folder : str
    kernels_to_plot : list of str
    lags : np.ndarray, optional
    area_groups : dict, optional
    area_colors : dict, optional
    n_cols : int, number of subplot columns
    """

    if lags is None:
        lags = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])

    # Map region -> color
    region_to_color = {}
    for group_name, areas in area_groups.items():
        for area in areas:
            region_to_color[area] = area_colors[group_name]

    def get_region_color(region_name):
        return region_to_color.get(region_name, 'gray')

    # Order regions
    ordered_regions = []
    for group_name, areas in area_groups.items():
        for area in areas:
            if area in df['area_acronym_custom'].values:
                ordered_regions.append(area)

    n_rows = math.ceil(len(ordered_regions) / n_cols)

    for kernel in kernels_to_plot:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
        axes = np.array(axes).reshape(-1)

        for ax, region in zip(axes, ordered_regions):
            region_df = df[df['area_acronym_custom'] == region]

            # Group by mouse_id and neuron_id
            neuron_groups = region_df.groupby(['mouse_id', 'neuron_id'])
            kernels_list = []

            for (_, _), grp in neuron_groups:

                mean_test_corr = grp['test_corr'].mean()
                if threshold is not None and mean_test_corr < threshold:
                    continue  # skip neuron below threshold

                # Collect coefficients for this neuron across folds
                neuron_kernels = []

                for _, row in grp.iterrows():
                    coefs_list = row['coef']
                    if isinstance(coefs_list, list) and isinstance(coefs_list[0], str):
                        coefs_list = [np.array(ast.literal_eval(c)) for c in coefs_list]
                    else:
                        coefs_list = [np.array(ast.literal_eval(coefs_list))]

                    predictors = ast.literal_eval(row['predictors'])

                    indices = [i for i, p in enumerate(predictors) if p.startswith(kernel)]

                    for c in coefs_list:
                        neuron_kernels.append(c[indices].ravel())

                if neuron_kernels:
                    # Average across folds for this neuron
                    neuron_kernels = np.stack(neuron_kernels)
                    kernels_list.append(neuron_kernels.mean(axis=0))

            if len(kernels_list) == 0:
                ax.set_visible(False)
                continue

            kernels_stack = np.stack(kernels_list)
            mean_kernel = kernels_stack.mean(axis=0)
            sem_kernel = kernels_stack.std(axis=0, ddof=1) / np.sqrt(kernels_stack.shape[0])

            color = get_region_color(region)
            ax.plot(lags, mean_kernel, color=color)
            ax.fill_between(lags, mean_kernel - sem_kernel, mean_kernel + sem_kernel,
                            color=color, alpha=0.3)

            ax.set_title(f"{region} (n={len(kernels_list)})", fontsize=10)
            ax.set_xlabel("Lag (s)")
            ax.set_ylabel("Coef")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_visible(True)

        # Hide unused axes
        for ax in axes[len(ordered_regions):]:
            ax.set_visible(False)

        plt.suptitle(f"{kernel} average kernels", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        name = f'average_kernel_{kernel}'
        if threshold:
            name += f'_threshold_{threshold}'
        putils.save_figure_with_options(fig, file_formats=['png'],
                                        filename=name,
                                        output_dir=output_folder)
        plt.close(fig)

    return


import ast
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_1samp


def compute_residual_correlations(glm_full_results: pd.DataFrame,
                                  glm_reduced_results: pd.DataFrame,
                                  full_corr_thresh: float = 0.2,
                                  resid_pval_thresh: float = 0.01):
    """
    Compute residual correlations per neuron and identify neurons with significant whisker kernel effect.

    Parameters
    ----------
    glm_full_results : pd.DataFrame
        Full model results, must contain columns ['neuron_id', 'fold', 'y_test', 'y_pred'].
    glm_reduced_results : pd.DataFrame
        Reduced model results, must contain columns ['neuron_id', 'fold', 'y_pred'].
    full_corr_thresh : float, optional
        Threshold for mean full model correlation to consider neuron significant.
    resid_pval_thresh : float, optional
        Threshold p-value for residual correlation t-test to consider neuron significant.

    Returns
    -------
    results : pd.DataFrame
        DataFrame with per-neuron metrics: mean full correlation, residual mean correlation, p-value, and significance flag.
    residuals_neurons : np.ndarray
        Array of neuron_ids passing the significance criteria.
    """

    # Compute mean full correlation per neuron
    full_corr = (
        glm_full_results.groupby('neuron_id')['test_corr']
        .mean()
        .rename('mean_full_corr')
    )

    residual_corrs = []

    for nid, full_grp in glm_full_results.groupby('neuron_id'):
        red_grp = glm_reduced_results[glm_reduced_results['neuron_id'] == nid]

        fold_corrs = []
        for fold, row_full in full_grp.groupby('fold'):
            row_red = red_grp[red_grp['fold'] == fold]
            if row_red.empty:
                continue  # skip missing fold

            # extract arrays safely
            y_true = np.array(ast.literal_eval(row_full['y_test'].iloc[0]))
            y_full = np.array(ast.literal_eval(row_full['y_pred'].iloc[0]))
            y_red = np.array(ast.literal_eval(row_red['y_pred'].iloc[0]))

            # compute residual correlation
            residual = y_full - y_red
            if len(residual) == len(y_true) and len(y_true) > 1:
                r, _ = pearsonr(residual, y_true)
                fold_corrs.append(r)

        if len(fold_corrs) > 0:
            tstat, pval = ttest_1samp(fold_corrs, 0)
            residual_corrs.append({
                'neuron_id': nid,
                'resid_mean_corr': np.mean(fold_corrs),
                'resid_pval': pval
            })

    resid_df = pd.DataFrame(residual_corrs)

    # Merge with full correlations
    results = (
        full_corr.reset_index()
        .merge(resid_df, on='neuron_id', how='left')
    )

    # Identify significant neurons
    results['whisker_kernel_sig'] = (
            (results['mean_full_corr'] > full_corr_thresh) &
            (results['resid_pval'] < resid_pval_thresh)
    )

    residuals_neurons = results.loc[results['whisker_kernel_sig'], 'neuron_id'].to_numpy()

    return results, residuals_neurons
def load_models(mouse, models_path):
    files = [f for f in os.listdir(models_path) if f.endswith('_results.parquet')]

    def _load(file):
        match = re.match(r'([a-f0-9]+)_model_(full|reduced)_fold(\d+)_results\.parquet', file)
        if not match:
            return None
        git_version, model_type, fold = match.group(1), match.group(2), int(match.group(3))

        df = post_hoc_load_model_results(file.split("_results")[0], models_path)
        df['git_version'] = git_version
        df['fold'] = fold
        df['model_type'] = model_type
        df['mouse_id'] = mouse
        return df

    dfs = Parallel(n_jobs=-1)(
        delayed(_load)(file) for file in files
    )
    all_results = [df for df in dfs if df is not None]
    df_all = pd.concat(all_results, ignore_index=True)
    return df_all

def combine_ephys_nwb(nwb_list,day_to_analyze =0, max_workers=24):
    """
    Combine neural and behavioural data from multiple NWB files using multiprocessing and tqdm.
    :param nwb_list: list of NWB file paths.
    :param max_workers: number of parallel processes.
    :return: (trial_table, unit_table, ephys_nwb_list)
    """
    ephys_nwb_list = []
    trial_table_list = []
    unit_table_list = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_nwb, nwb, day_to_analyze = day_to_analyze): nwb for nwb in nwb_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading NWB files"):
            result = future.result()
            if result is None:
                continue
            ephys_nwb_list.append(result['nwb'])
            trial_table_list.append(result['trial_table'])
            unit_table_list.append(result['unit_table'])

    print(f"Found {len(ephys_nwb_list)} NWB files with ephys data.")
    print(f"Available NWB files {len(ephys_nwb_list)}:", sorted([os.path.basename(nwb) for nwb in ephys_nwb_list]))

    trial_table = pd.concat(trial_table_list, ignore_index=True) if trial_table_list else pd.DataFrame()
    unit_table = pd.concat(unit_table_list, ignore_index=True) if unit_table_list else pd.DataFrame()

    if not unit_table.empty:
        mouse_id = trial_table['mouse_id'].unique()[0]
        print('Warning: number of root neurons :', mouse_id, len(unit_table[unit_table.ccf_acronym=='root']))
        unit_table = unit_table[~unit_table['ccf_acronym'].isin(allen.get_excluded_areas())]
        unit_table = unit_table.reset_index(drop=True)
        unit_table['unit_id'] = unit_table.index

    return trial_table, unit_table, ephys_nwb_list

def convert_electrode_group_object_to_columns(data):
    """
    Convert electrode group object to dictionary.
    Creates a new column in the dataframe.
    :param data: pd.DataFrame containing the NWB electrode group field.
    :return:
    """
    elec_group_list = data['electrode_group'].values
    #elec_group_name = [e.name for e in elec_group_list]
    elec_group_name = [e.name for e in elec_group_list]
    data['electrode_group'] = elec_group_name

    elec_group_location = [e.location.replace('nan', 'None') for e in elec_group_list]
    elec_group_location_dict = [eval(e) for e in elec_group_location]
    data['location'] = elec_group_location_dict
    data['target_region'] = [e.get('area') for e in elec_group_location_dict]

    return data

def process_single_nwb(nwb, day_to_analyze = 0):
    try:
        beh_type, day = nwb_reader.get_bhv_type_and_training_day_index(nwb)
        if day_to_analyze == 0 and day !=0:
            return None
        elif day_to_analyze > 0 and day == 0:
            return None

        unit_table = nwb_reader.get_unit_table(nwb)
        if unit_table is None or 'bc_label' not in unit_table.columns:
            return None

        trial_table = nwb_reader.get_trial_table(nwb)
        trial_table['trial_id'] = trial_table.index

        mouse_id = nwb_reader.get_mouse_id(nwb)
        session_id = nwb_reader.get_session_id(nwb)

        trial_table['mouse_id'] = mouse_id
        trial_table['session_id'] = session_id
        trial_table['context'] = trial_table['context'].astype(str)
        trial_table['day'] = day

        if trial_table['context'].str.contains('nan').all():
            trial_table['context'] = 'active'
        else:
            trial_table['context'] = trial_table['context'].fillna('active')
            trial_table['context'] = trial_table['context'].replace('nan','active')

        # passive trials were not modeled so we drop them
        trial_table["behav_type"] = trial_table.apply(classify_trial, axis=1)
        trial_table = trial_table[(trial_table['context'] == 'active') & (trial_table['perf'] != 6)].copy()
        trial_table = trial_table.reset_index(drop=True)

        unit_table['mouse_id'] = mouse_id
        unit_table = convert_electrode_group_object_to_columns(unit_table)

        # Only keep the neurons fitted for the glms
        unit_table = allen.process_allen_labels(unit_table, subdivide_areas=False)
        unit_table = unit_table[unit_table['bc_label'] == 'good']
        unit_table = unit_table[unit_table['firing_rate'].astype(float).ge(2.0)]
        unit_table = unit_table[~unit_table['ccf_acronym'].isin(allen_utils.get_excluded_areas())]
        unit_table['original_unit_id'] = unit_table.index
        unit_table = unit_table.reset_index(drop=True)
        unit_table['neuron_id'] = unit_table.index

        return {
            'nwb': nwb,
            'trial_table': trial_table,
            'unit_table': unit_table
        }

    except Exception as e:
        print(f"Error processing {nwb}: {e}")
        return None


import plotting_utils as putils
def plot_trial_grid_predictions(results_df, trial_table, neuron_id, bin_size):
    """
    Plot predictions for a single neuron across trials in a grid format.
    :param results_df: DataFrame with model results
    :param trial_table: DataFrame with trial information
    :param neuron_id: int, ID of the neuron to plot
    :param bin_size: float, size of time bin in seconds
    """

    # Plotting params
    n_rows, n_cols = 5, 5
    trials_to_plot = min(n_rows*n_cols, len(trial_table))

    # Get neuron results
    results_df_sub = results_df[results_df['neuron_id'] == neuron_id]
    y_test = results_df_sub['y_test'].values[0]
    y_pred = results_df_sub['y_pred'].values[0]
    n_bins = results_df_sub['n_bins'].values[0]
    y_test = np.array(ast.literal_eval(y_test))
    y_pred = np.array(ast.literal_eval(y_pred))

    # Format data into (n_trials, n_bins)
    n_trials = y_pred.shape[0] // n_bins
    y_test = y_test.reshape(n_trials, n_bins)
    y_pred = y_pred.reshape(n_trials, n_bins)

    # Order test trial temporally
    test_trial_ids =  np.array(ast.literal_eval(results_df_sub['test_trials'].values[0]))
    test_trial_id_order =  np.argsort(test_trial_ids)
    y_test = y_test[test_trial_id_order,:]
    y_pred = y_pred[test_trial_id_order,:]

    trials_test_df = trial_table[trial_table['trial_id'].isin(test_trial_ids)]
    trials_test_df = trials_test_df.sort_values(by='trial_id', ascending=True)
    trials_test_df = trials_test_df.reset_index(drop=True)
    trials_test_df = trials_test_df.iloc[:trials_to_plot]

    # Create figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 12), sharex=False, sharey=False)
    axs = axs.flatten()


    window_bounds_sec = (-1, 2)
    time_stim = 0
    time = np.linspace(window_bounds_sec[0] + bin_size/2,
                   window_bounds_sec[1] - bin_size/2,
                   n_bins)

    for idx, row in trials_test_df.iterrows():
        ax = axs[idx]
        ax.set_title('Trial {}'.format(row['trial_id']), fontsize=10)
        putils.remove_top_right_frame(ax)
        ax.set_ylim(0, 10)
        ax.set_ylabel('Spikes', fontsize=10)
        ax.set_yticks([0, 10])
        ax.set_yticklabels([0, 10], fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)


        trial_type = row['trial_type']
        if trial_type == 'whisker_trial':
            ax.axvline(time_stim, color='forestgreen', linestyle='-', linewidth=1)
        elif trial_type == 'auditory_trial':
            ax.axvline(time_stim, color='mediumblue', linestyle='-', linewidth=1)
        elif trial_type == 'no_stim_trial':
            ax.axvline(time_stim, color='k', linestyle='-', linewidth=1)

        # Plot target and predictions
        ax.plot(time, y_pred[idx], color='red', linewidth=1.5)
        ax.step(time, y_test[idx], where='mid', color='black', alpha=0.9, linewidth=1.5)


    title = (f'GLM predictions on test trials - unit {neuron_id}, '
             f'$R$= {results_df_sub["test_corr"].values[0]:.2f}')
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.align_ylabels()
    plt.show()

    return
