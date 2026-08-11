import os
import pandas as pd
from pathlib import Path
import sys
# Determine project root dynamically
project_root = Path(__file__).resolve().parent.parent

# Add helper repos to sys.path
sys.path.append(str(project_root / "NWB_reader"))
sys.path.append(str(project_root / "allen_utils"))
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
import json
from scipy.stats import gaussian_kde

ROOT_PATH = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Axel_Bisi', 'combined_results')

def compute_density_safe(x, y):
    """
    Compute density for scatter plot using KDE, with fallback to histogram if KDE fails.

    Parameters
    ----------
    x, y : array-like
        2D coordinates of points

    Returns
    -------
    x, y, z : arrays
        Sorted coordinates and density values
    """
    try:
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        return x[idx], y[idx], z[idx]
    except np.linalg.LinAlgError:
        # Fallback: use 2D histogram to compute density
        H, xedges, yedges = np.histogram2d(x, y, bins=50)
        # Get bin indices for each point
        xi = np.digitize(x, xedges) - 1
        yi = np.digitize(y, yedges) - 1
        # Clip to valid range
        xi = np.clip(xi, 0, H.shape[0] - 1)
        yi = np.clip(yi, 0, H.shape[1] - 1)
        # Assign density based on histogram
        z = H[xi, yi]
        idx = z.argsort()
        return x[idx], y[idx], z[idx]

def post_hoc_load_model_results(filename, output_dir):
    file_path = os.path.join(output_dir, f"{filename}_results.parquet")
    try:
        return pd.read_parquet(file_path, engine = 'fastparquet')
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[ERROR] reading {file_path}: {e}")
        return None

def load_models(mouse, models_path, git_version):
    pattern = re.compile(rf'^{git_version}_model_(full|reduced|added)_fold(\d+)_results\.parquet$')

    files = [f for f in os.listdir(models_path) if f.endswith('_results.parquet')]
    valid_files = [(f, *pattern.match(f).groups()) for f in files if pattern.match(f)]

    if not valid_files:
        print(f"[WARNING] No valid model results found for mouse {mouse}.")
        return None

    dfs = []
    for file, model_type, fold in valid_files:
        df = post_hoc_load_model_results(file.split("_results")[0], models_path)
        if df is None or df.empty:
            continue
        df['git_version'] = git_version
        df['fold'] = fold
        df['model_type'] = model_type
        df['mouse_id'] = mouse
        dfs.append(df)

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)
from joblib import Parallel, delayed

def parse_json_array(s):
    """Parse JSON string to numpy array."""
    return np.array(json.loads(s))

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


def compute_kernel_consistency(df_model, kernel_pairs, consistency_threshold=0.75):
    """
    For each neuron, compute whether the average kernel weight (across all bins) is
    consistently increasing or decreasing from kernel_0 to kernel_1 across folds.

    For each fold the mean weight across all bins is computed for each kernel, then
    consistency is assessed on that single per-fold average rather than on individual bins.

    Parameters
    ----------
    df_model : pd.DataFrame
        Dataframe containing model results for multiple folds
    kernel_pairs : list of tuples
        List of (kernel_0, kernel_1) pairs to compare
    consistency_threshold : float
        Fraction of folds that must show a change in the same direction (default: 0.75)

    Returns
    -------
    dict
        {(mouse_id, neuron_id): {kernel_pair: {0: consistency_score}}}
        where consistency_score is positive for consistent increases and negative for
        consistent decreases in the across-bin average
    """
    import re

    def extract_sorted_kernel_indices(predictors, kernel_name):
        pattern = re.compile(fr"^{kernel_name}_t([+-]\d+\.\d+)s$")
        matches = []
        for i, p in enumerate(predictors):
            m = pattern.match(p)
            if m:
                lag = float(m.group(1))
                matches.append((lag, i))
        matches_sorted = sorted(matches, key=lambda x: x[0])
        idx_sorted = [i for (_, i) in matches_sorted]
        lags_sorted = [lag for (lag, _) in matches_sorted]
        return idx_sorted, lags_sorted

    neuron_consistency = {}

    for (mouse_id, neuron_id), grp in df_model.groupby(['mouse_id', 'neuron_id']):
        if len(grp) < 2:  # Need at least 2 folds
            continue

        neuron_consistency[(mouse_id, neuron_id)] = {}

        for kernel_0, kernel_1 in kernel_pairs:
            # Per-fold average weight across all bins
            avg_0_by_fold = []
            avg_1_by_fold = []

            for _, row in grp.iterrows():
                predictors = row["predictors"]
                coefs = np.array(row["coef_array"])

                idx_0, _ = extract_sorted_kernel_indices(predictors, kernel_0)
                if idx_0:
                    avg_0_by_fold.append(np.mean(coefs[idx_0]))

                idx_1, _ = extract_sorted_kernel_indices(predictors, kernel_1)
                if idx_1:
                    avg_1_by_fold.append(np.mean(coefs[idx_1]))

            if len(avg_0_by_fold) == 0 or len(avg_1_by_fold) == 0:
                continue

            if len(avg_0_by_fold) != len(avg_1_by_fold):
                continue

            avg_0 = np.array(avg_0_by_fold)
            avg_1 = np.array(avg_1_by_fold)

            # Check if the across-bin average is consistently higher or lower
            increase_score = np.mean(avg_1 > avg_0)
            decrease_score = np.mean(avg_1 < avg_0)

            if increase_score >= consistency_threshold:
                pair_name = f"{kernel_1}_vs_{kernel_0}"
                neuron_consistency[(mouse_id, neuron_id)][pair_name] = increase_score
            elif decrease_score >= consistency_threshold:
                pair_name = f"{kernel_1}_vs_{kernel_0}"
                neuron_consistency[(mouse_id, neuron_id)][pair_name] = -decrease_score

    return neuron_consistency



def plot_average_real_vs_predicted_per_trialtype_per_area(df_model, trial_table, area_groups, area_colors, output_folder):
    """
    Plot average real vs predicted activity per trial type for each brain area.

    Parameters
    ----------
    df_model : pd.DataFrame
        DataFrame with model results (should be filtered to 'full' model)
    trial_table : pd.DataFrame
        Trial table with trial_type information
    area_groups : dict
        Dictionary mapping group names to lists of area acronyms
    area_colors : dict
        Dictionary mapping group names to colors
    output_folder : str
        Path to save output figures
    """
    # Get unique trial types
    trial_types = trial_table['trial_type'].unique()

    # Get ordered regions
    ordered_regions = []
    for group_name, areas in area_groups.items():
        for area in areas:
            if area in df_model['area_acronym_custom'].values:
                ordered_regions.append(area)
    ordered_regions = sorted(set(ordered_regions))

    # Create figure with subplots per area
    n_areas = len(ordered_regions)
    n_cols = min(4, n_areas)
    n_rows = int(np.ceil(n_areas / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_areas == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Color map for trial types
    trial_type_colors = {
        'whisker_trial': 'green',
        'auditory_trial': 'blue',
        'catch': 'gray',
        'whisker_hit': 'darkgreen',
        'whisker_miss': 'lightgreen',
        'auditory_hit': 'darkblue',
        'auditory_miss': 'lightblue'
    }

    for area_idx, area in enumerate(ordered_regions):
        ax = axes[area_idx]
        df_area = df_model[df_model['area_acronym_custom'] == area]

        if len(df_area) == 0:
            ax.set_visible(False)
            continue

        # For each trial type, compute average real and predicted activity
        for trial_type in trial_types:
            # Get trials of this type
            trial_indices = trial_table[trial_table['trial_type'] == trial_type].index.values

            # Collect real and predicted activity across all neurons in this area
            real_activity_all = []
            pred_activity_all = []

            for _, row in df_area.iterrows():
                y_test = row['y_test_array']
                y_pred = row['y_pred_array']

                # Get activity for this trial type
                # Assuming y_test and y_pred have same length as trials
                if len(y_test) > 0 and len(trial_indices) > 0:
                    # Handle case where trial indices might be out of bounds
                    valid_indices = trial_indices[trial_indices < len(y_test)]
                    if len(valid_indices) > 0:
                        real_activity_all.extend(y_test[valid_indices])
                        pred_activity_all.extend(y_pred[valid_indices])

            if len(real_activity_all) > 0:
                # Compute mean and SEM
                mean_real = np.mean(real_activity_all)
                sem_real = np.std(real_activity_all) / np.sqrt(len(real_activity_all))
                mean_pred = np.mean(pred_activity_all)
                sem_pred = np.std(pred_activity_all) / np.sqrt(len(pred_activity_all))

                color = trial_type_colors.get(trial_type, 'black')

                # Plot as bars
                x_pos = list(trial_types).index(trial_type)
                ax.bar(x_pos - 0.2, mean_real, 0.4, yerr=sem_real,
                      color=color, alpha=0.5, label=f'{trial_type} (real)' if area_idx == 0 else None,
                      capsize=3)
                ax.bar(x_pos + 0.2, mean_pred, 0.4, yerr=sem_pred,
                      color=color, alpha=1.0, label=f'{trial_type} (pred)' if area_idx == 0 else None,
                      capsize=3, edgecolor='black', linewidth=1)

        ax.set_xticks(range(len(trial_types)))
        ax.set_xticklabels([tt.replace('_', '\n') for tt in trial_types], fontsize=8, rotation=0)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=10)
        ax.set_title(f'{area} (n={len(df_area)} neurons)', fontsize=11, fontweight='bold')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

        if area_idx == 0:
            ax.legend(fontsize=7, loc='upper right', ncol=1)

    # Hide unused subplots
    for idx in range(n_areas, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Average Real vs Predicted Activity per Trial Type',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    fname = "avg_real_vs_pred_per_trialtype_per_area"
    putils.save_figure_with_options(fig, file_formats=["png", "pdf"],
                                  filename=fname,
                                  output_dir=output_folder)
    plt.close(fig)


def mouse_glm_results(nwb_list, model_path, plots, output_path, git_version, info_path, day_to_analyze = 0):

    # Load and combine NWB files
    trial_table, unit_table, ephys_nwb_list = combine_ephys_nwb(nwb_list, day_to_analyze=day_to_analyze, max_workers=8, git_version=git_version)
    if git_version in ['2ce0ecd', 'a6b5c56'] :
        trial_table = trial_table[trial_table['trial_type'] =='whisker_trial']
    # if git_version in ['4227ca6', 'b394470', '74987e2', '935b6e1', '15127ae', ]:
    #     trial_table = load_perf_blocks(trial_table, trial_table['mouse_id'].unique()[0])
    #     trial_table = trial_table.reset_index(drop=True)

    # Load all models
    df_models = load_models_one_mouse(unit_table['mouse_id'].unique()[0], model_path, git_version)  # only get the current git version
    if df_models is None or df_models.empty:
        print('Poisson GLMs not fit with that git version for mouse :', unit_table['mouse_id'].unique()[0])
        return None
    df_git = df_models[df_models['git_version'] == git_version]

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

    df_git['y_test_array'] = df_git['y_test'].map(lambda s: np.array(json.loads(s)))
    df_git['y_pred_array'] = df_git['y_pred'].map(lambda s: np.array(json.loads(s)))
    print(df_git['predictors'])
    df_git['predictors'] = df_git['predictors'].apply(lambda s: np.array(json.loads(s)))

    merged_df = pd.merge(df_git, unit_table, how='inner', on=["mouse_id", "neuron_id"])
    

    area_groups = allen.get_custom_area_groups()
    area_colors = allen.get_custom_area_groups_colors()
    merged_df = allen.create_area_custom_column(merged_df)

    if 'metrics' in plots :

        output_folder = os.path.join(output_path, 'metrics')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plot_kde_per_trial_type(merged_df[merged_df['model_name'] == 'full'], trial_table, output_folder)
        plot_box_per_trial_type(merged_df[merged_df['model_name'] == 'full'], trial_table, output_folder, time_stim=0.0)
        plot_kde_full_vs_reduced(merged_df, output_folder)
        plot_box_full_vs_reduced(merged_df, output_folder, alpha=0.05)
        for model_name in df_git['model_name'].unique():
            if model_name == 'full':
                continue
            # plot_full_vs_reduced_per_area(merged_df, model_name, area_groups, area_colors, output_folder)
            plot_full_vs_reduced_per_area(merged_df, model_name, area_groups, area_colors, output_folder)

            # # Step 1: compute trial-type correlations for both models
            # corr_full = compute_trialtype_correlations(merged_df[merged_df['model_name'] == 'full'], trial_table)
            # corr_reduced = compute_trialtype_correlations(merged_df[merged_df['model_name'] == model_name], trial_table)
            # corr_all = pd.concat([corr_full, corr_reduced])
            #
            # # Step 2: plot
            # plot_full_vs_reduced_per_area_and_trialtype(
            #     corr_all,
            #     selected_reduced=model_name,
            #     area_groups=area_groups,
            #     area_colors=area_colors,
            #     output_folder=output_folder,
            #     threshold=None
            # )
        plot_kde_full_vs_reduced(merged_df, output_folder)
        plot_box_full_vs_reduced(merged_df, output_folder, alpha=0.05)
        plot_kde_per_trial_type(merged_df[merged_df['model_name'] == 'full'], trial_table, output_folder)
        plot_corr_per_area_by_trialtype(merged_df[merged_df['model_name'] == 'full'], trial_table, area_groups, output_folder)
        lrt = compute_lrt_from_model_results(merged_df, alpha=0.05)
        # lrt_subset = lrt[lrt['model_name'].isin(['auditory_encoding', 'jaw_onset_encoding', 'motor_encoding', 'session_progress_encoding', 'sum_rewards', 'whisker_encoding'])]
        merged_lrt = pd.merge(
            lrt,
            unit_table[["mouse_id", "neuron_id", "area_acronym_custom"]],
            how="left",  # or "inner" if you want to keep only matching rows
            on=["mouse_id", "neuron_id"]
        )

        plot_lrt_significance_overlap(merged_lrt, output_folder)
        plot_lrt_significance_per_area_per_model(
            merged_lrt,
            area_groups=area_groups,
            area_colors=area_colors,
            output_folder=output_folder
        )
        plot_lrt_significance_overlap_per_area(merged_lrt, output_folder)

    if git_version == '1cce900':
        lags =  np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    else:
        lags = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])

    if 'per_unit_kernel_plots' in plots :
        output_folder = os.path.join(output_path, 'per_unit_kernel_plots')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for neuron in merged_df['neuron_id'].unique():
            plot_neuron_kernels_avg_with_responses(
                neuron, merged_df[merged_df['model_name'] == 'full'], ['whisker_hits_stim_0', 'whisker_hits_stim_1', 'whisker_misses_stim0', 'whisker_misses_stim1', 'auditory_stim0', 'auditory_stim1', 'jaw_onset', 'piezo_reward'], trial_table, output_folder, lags = lags, git_handle=git_version)

    if 'average_predictions_per_trial_types' in plots :
        output_folder = os.path.join(output_path, 'average_predictions_per_trial_types')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # first80_ids = merged_df['neuron_id'].unique()[:80]

        # merged_df = merged_df[merged_df['neuron_id'].isin(first80_ids)]
        # plot_predictions_with_reduced_models_parallel(merged_df[merged_df['model_name'] == 'full'], merged_df[merged_df['model_name'] == 'last_rewards_whisker'], trial_table,type = 'previous_whisker',output_folder_base= output_folder)
        # plot_predictions_with_reduced_models_parallel(merged_df[merged_df['model_name'] == 'full'], merged_df[merged_df['model_name'] == 'jaw_onset_encoding'], trial_table,type = 'Normal',output_folder_base= output_folder)
        # plot_predictions_with_reduced_models_parallel(merged_df[merged_df['model_name'] == 'full'], merged_df[merged_df['model_name'] == '2whisker_kernels'], trial_table,type = 'Normal',output_folder_base= output_folder)

        # plot_predictions_with_reduced_models_parallel(merged_df[merged_df['model_name'] == 'full'], merged_df[merged_df['model_name'] != 'full'], trial_table,type = 'Normal',output_folder_base= output_folder)
        #
        # decreased_neurons, _ = neurons_with_consistent_decrease(merged_df, reduced_name='last_rewards_whisker')
        # print(f"{len(decreased_neurons)} neurons show consistent decrease across folds.")
        # merged_df_sig = merged_df[merged_df['neuron_id'].isin(decreased_neurons['neuron_id'])]
        # output_folder = os.path.join(output_path, 'average_predictions_per_trial_types_per_blocks')
        # if not os.path.exists(output_folder):
        #     os.makedirs(output_folder)
        #
        # decreased_neurons, _ = neurons_with_consistent_decrease(merged_df, reduced_name='prop_last_5_whisker')
        # print(f"{len(decreased_neurons)} neurons show consistent decrease across folds.")
        # merged_df_sig = merged_df[merged_df['neuron_id'].isin(decreased_neurons['neuron_id'])]
        # plot_predictions_with_reduced_models_parallel(merged_df_sig[merged_df_sig['model_name'] == 'full'], merged_df_sig[merged_df_sig['model_name'] == 'prop_last_5_whisker'], trial_table,type = 'last_5',output_folder_base= output_folder)

        # decreased_neurons, _ = neurons_with_consistent_decrease(merged_df, reduced_name='all_whisker_progression_but_local')
        # print(f"{len(decreased_neurons)} neurons show consistent decrease across folds.")
        # merged_df_sig = merged_df[merged_df['neuron_id'].isin(decreased_neurons['neuron_id'])]

        plot_predictions_with_reduced_models_parallel(merged_df[merged_df['model_name'] == 'full'], merged_df[merged_df['model_name'] == '1whisker_kernel'], trial_table,type = 'session_progression',output_folder_base= output_folder)
        plot_predictions_with_reduced_models_parallel(merged_df[merged_df['model_name'] == 'full'], merged_df[merged_df['model_name'] == 'full'], trial_table,type = 'session_progression',output_folder_base= output_folder)
        plot_predictions_with_reduced_models_parallel(merged_df[merged_df['model_name'] == 'full'], merged_df[merged_df['model_name'] == 'random_split'], trial_table,type = 'session_progression',output_folder_base= output_folder)
        # plot_predictions_with_reduced_models_parallel(merged_df[merged_df['model_name'] == 'full'], merged_df[merged_df['model_name'] == '4whisker_kernels'], trial_table,type = 'session_progression',output_folder_base= output_folder)
        # plot_predictions_with_reduced_models_parallel(merged_df[merged_df['model_name'] == 'full'], merged_df[merged_df['model_name'] == '5whisker_kernels'], trial_table,type = 'session_progression',output_folder_base= output_folder)
        # plot_predictions_with_reduced_models_parallel(merged_df[merged_df['model_name'] == 'full'], merged_df[merged_df['model_name'] == '6whisker_kernels'], trial_table,type = 'session_progression',output_folder_base= output_folder)
        # plot_predictions_with_reduced_models_parallel(merged_df[merged_df['model_name'] == 'full'], merged_df[merged_df['model_name'] == '7whisker_kernels'], trial_table,type = 'session_progression',output_folder_base= output_folder)
        # plot_predictions_with_reduced_models_parallel(merged_df[merged_df['model_name'] == 'full'], merged_df[merged_df['model_name'] == '8whisker_kernels'], trial_table,type = 'session_progression',output_folder_base= output_folder)
        # plot_predictions_with_reduced_models_parallel(merged_df[merged_df['model_name'] == 'full'], merged_df[merged_df['model_name'] == '9whisker_kernels'], trial_table,type = 'session_progression',output_folder_base= output_folder)

        # plot_predictions_with_reduced_models_parallel(merged_df[merged_df['model_name'] == 'full'], merged_df[merged_df['model_name'] == '2_reward_kernels'], trial_table,type = 'session_progression',output_folder_base= output_folder)

        # plot_predictions_with_reduced_models_parallel(merged_df[merged_df['model_name'] == 'full'], merged_df[merged_df['model_name'] == 'block_perf_type'], trial_table,type = 'session_progression',output_folder_base= output_folder)


        #
        # print(f"{len(decreased_neurons)} neurons show consistent decrease across folds.")
        # merged_df_sig = merged_df[merged_df['neuron_id'].isin(decreased_neurons['neuron_id'])]
        # plot_predictions_with_reduced_models_parallel(merged_df_sig[merged_df_sig['model_name'] == 'full'], merged_df_sig[merged_df_sig['model_name'] == 'all_whisker_progression_but_local'], trial_table,type = 'session_progression',output_folder_base= output_folder + str('test'))
        #

        # lrt = compute_lrt_from_model_results(merged_df, trial_table, alpha=0.05)
        # lrt_subset = lrt[lrt['model_name'].isin(['auditory_encoding', 'jaw_onset_encoding', 'motor_encoding', 'session_progress_encoding', 'sum_rewards', 'whisker_encoding'])]
        # merged_lrt = pd.merge(
        #     lrt_subset,
        #     unit_table[["mouse_id", "neuron_id", "area_acronym_custom"]],
        #     how="left",  # or "inner" if you want to keep only matching rows
        #     on=["mouse_id", "neuron_id"]
        # )
        # sig_neurons = merged_lrt[
        #     (merged_lrt['model_name'] == 'jaw_onset_encoding') &
        #     (merged_lrt['lrt_significant'] == True)
        # ][['mouse_id', 'neuron_id']]

        # df_sig_full = merged_lrt.merge(
        #     sig_neurons,
        #     on=['mouse_id','neuron_id'],
        #     how='inner'
        # )
        # df_sig_full = df_sig_full[df_sig_full['model_name'] == 'full']

    if 'average_activity_per_trial_type' in plots:
        output_folder = os.path.join(output_path, 'average_activity_per_trial_type')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plot_average_real_vs_predicted_per_trialtype_per_area(
            merged_df[merged_df['model_name'] == 'full'],
            trial_table,
            area_groups,
            area_colors,
            output_folder
        )

    if 'average_kernels_by_region' in plots :
        output_folder = os.path.join(output_path, 'average_kernels_by_region')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if git_version == '1cce900':
            lags =  np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        else:
            lags = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
        plot_average_kernels_by_region( merged_df[merged_df['model_name'] == 'full'], output_folder, ['whisker_stim', 'auditory_stim', 'jaw_onset', 'piezo_reward'],
            lags=lags, area_groups=area_groups, area_colors=area_colors, n_cols=3, git_handle=git_version)
        
        output_folder_red = os.path.join(output_folder, 'reduced_2kernels')
        if not os.path.exists(output_folder_red):
            os.makedirs(output_folder_red)

        plot_average_kernels_by_region( merged_df[merged_df['model_name'] == '2whisker_kernels'], output_folder_red, ['whisker_stim_0','whisker_stim_1',  'jaw_onset', 'piezo_reward'],
            lags=lags, area_groups=area_groups, area_colors=area_colors, n_cols=3, git_handle=git_version)
                
        output_folder_red = os.path.join(output_folder, 'reduced_3kernels')
        if not os.path.exists(output_folder_red):
            os.makedirs(output_folder_red)
        
        plot_average_kernels_by_region( merged_df[merged_df['model_name'] == '3whisker_kernels'], output_folder_red, ['whisker_stim_0','whisker_stim_1','whisker_stim_2', 'jaw_onset', 'piezo_reward'],
            lags=lags, area_groups=area_groups, area_colors=area_colors, n_cols=3, git_handle=git_version)



        decreased_neurons, _ = neurons_with_consistent_decrease(merged_df, reduced_name='whisker_encoding')
        print(f"{len(decreased_neurons)} neurons show consistent decrease across folds.")
        merged_df_sig = merged_df[merged_df['neuron_id'].isin(decreased_neurons['neuron_id'])]
        output_folder = os.path.join(output_path, 'average_kernels_by_region_sign_whisker')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plot_average_kernels_by_region(  merged_df_sig[merged_df_sig['model_name'] == 'full'], output_folder, ['whisker_stim',],
            lags=lags, area_groups=area_groups, area_colors=area_colors, n_cols=3, threshold = None)
    
    if 'individual_trials' in plots:
        output_folder = os.path.join(output_path, 'indiv_trial_prediction')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for neuron_id in merged_df['neuron_id'].unique():
            plot_trial_grid_predictions(merged_df, trial_table, neuron_id, 0.1, output_folder)

    if 'individual_trials_concat' in plots:
        output_folder = os.path.join(output_path, 'indiv_trial_prediction_concat')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for neuron_id in merged_df['neuron_id'].unique():
            plot_trial_concatenated_predictions(merged_df, trial_table, neuron_id, 0.1, output_folder)
    


    if 'individual_trials_with_weights' in plots:
        output_folder = os.path.join(output_path, 'indiv_trial_prediction_with_weight')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for neuron_id in tqdm(merged_df['neuron_id'].unique()):
            neuron_output =  os.path.join(output_folder, str(neuron_id))
            if not os.path.exists(neuron_output):
                os.makedirs(neuron_output)
            plot_trial_with_design_matrix_and_weights_predictions(merged_df[merged_df['model_name'] == 'full'], trial_table, neuron_id, 0.1, neuron_output)

    if 'create_summary' in plots :
        testcorr_long = (
        df_git.groupby(['mouse_id','neuron_id','model_name'])
              .agg(test_corr=('test_corr','mean'))
              .reset_index()
        )
        df_git_with_id = df_git.merge(
            unit_table[['mouse_id', 'neuron_id', 'og_unit_table_id', 'area_acronym_custom', 'reward_group']],
            on=['mouse_id', 'neuron_id'],
            how='left'
        )
        lrt = compute_lrt_from_model_results(merged_df,trial_table, alpha=0.05)

        lrt_long = (
            lrt[['mouse_id','neuron_id','model_name','p_value','lrt_significant', 'lrt_sig_whisker']]
            .rename(columns={
                'model_name':'model_name',
                'p_value':'lrt_p_value',
                'lrt_significant':'lrt_significant'
            })
        )
        df_git['coef_array'] = df_git['coef'].apply(lambda s: np.array(json.loads(s)))

        df_full = df_git[df_git['model_name'] == 'full']

        coef_summary = (
            df_full.groupby(['mouse_id','neuron_id'])
                .agg(
                    coef_full_mean=('coef_array', lambda arrs: np.mean(np.stack(arrs), axis=0)),
                    coef_full_std =('coef_array', lambda arrs: np.std(np.stack(arrs), axis=0)),
                    predictors_full=('predictors', lambda arrs: arrs.iloc[0])
                )
                .reset_index()
        )

        # Analyze multi-kernel models - compute kernel consistency across folds
        kernel_consistency_data = []

        df_model = df_git[df_git['model_name'] == 'full']

        # Identify kernel pairs for this model
        sample_predictors = df_model['predictors'].iloc[0]
        kernels = []
        for pred in sample_predictors:
            if any(x in pred for x in ['whisker', 'auditory']) and '_t' in pred:
                kernel_base = pred.split('_t')[0]
                if kernel_base not in kernels:
                    kernels.append(kernel_base)

        # Group kernels by base name
        kernel_groups = {}
        for kernel in kernels:
            import re
            match = re.match(r'^(.+?)_?(\d+)$', kernel)
            if match:
                base_name = match.group(1)
                if base_name not in kernel_groups:
                    kernel_groups[base_name] = []
                kernel_groups[base_name].append(kernel)

        # Create pairs
        kernel_pairs = []
        for base_name, group_kernels in kernel_groups.items():
            if len(group_kernels) >= 2:
                sorted_kernels = sorted(group_kernels)
                for i in range(len(sorted_kernels) - 1):
                    kernel_pairs.append((sorted_kernels[i], sorted_kernels[i + 1]))

        # Compute consistency
        neuron_consistency = compute_kernel_consistency(df_model, kernel_pairs)

        # Convert to dataframe format with og_unit_table_id
        neuron_to_og_id = df_model[['mouse_id', 'neuron_id', 'og_unit_table_id']].drop_duplicates().set_index(['mouse_id', 'neuron_id'])['og_unit_table_id'].to_dict()

        for (mouse_id, neuron_id), pairs_data in neuron_consistency.items():
            og_unit_table_id = neuron_to_og_id.get((mouse_id, neuron_id))
            for pair_name, consistency_score in pairs_data.items():
                kernel_consistency_data.append({
                    'mouse_id': mouse_id,
                    'og_unit_table_id': og_unit_table_id,
                    'model_name': model_name,
                    'kernel_pair': pair_name,
                    'is_consistent': True,
                    'consistency_score': float(consistency_score),
                })

        # Create consistency dataframe
        if kernel_consistency_data:
            consistency_df = pd.DataFrame(kernel_consistency_data)
            # Merge into summary
            summary_with_consistency = summary.merge(
                consistency_df,
                on=['mouse_id', 'og_unit_table_id', 'model_name'],
                how='left'
            )
        else:
            summary_with_consistency = summary.copy()
            summary_with_consistency['kernel_pair'] = None
            summary_with_consistency['is_consistent'] = False
            summary_with_consistency['consistency_score'] = None

        summary = summary_with_consistency


        mask_nonfull = summary['model_name'] != 'full'
        summary.loc[mask_nonfull, ['coef_full_mean','coef_full_std','predictors_full']] = None

        summary['git_version'] = git_version

        for mouse_id in summary['mouse_id'].unique():
            outdf = summary[summary['mouse_id'] == mouse_id]
            outpath = f"{output_path}/summary_{mouse_id}_unit_glm_{git_version}.parquet"
            outdf.to_parquet(outpath)
            print(f"Saved summary for {mouse_id} → {outpath}")

        # Save predictions summary: neuron_id, per-trial y_pred and y_test (n_trials x n_bins) for full model
        full_df = df_git_with_id[df_git_with_id['model_name'] == 'full']
        pred_summary_rows = []
        # Build lookups from trial index → trial_type and lick_flag using trial_table
        trial_type_lookup = trial_table['trial_type'].to_dict() if 'trial_type' in trial_table.columns else {}
        lick_flag_lookup  = trial_table['lick_flag'].to_dict()  if 'lick_flag'  in trial_table.columns else {}
        for (mouse_id, og_id), grp in full_df.groupby(['mouse_id', 'og_unit_table_id']):
            n_bins = int(grp['n_bins'].iloc[0])
            trial_preds = {}
            trial_tests = {}
            for _, row in grp.iterrows():
                test_trial_ids = np.array(json.loads(row['test_trials']))
                y_pred_fold = row['y_pred_array'].reshape(len(test_trial_ids), n_bins)
                y_test_fold = row['y_test_array'].reshape(len(test_trial_ids), n_bins)
                for i, trial_idx in enumerate(test_trial_ids):
                    trial_preds[int(trial_idx)] = y_pred_fold[i]
                    trial_tests[int(trial_idx)] = y_test_fold[i]
            sorted_trial_ids = sorted(trial_preds.keys())
            y_pred_matrix = np.stack([trial_preds[t] for t in sorted_trial_ids])
            y_test_matrix = np.stack([trial_tests[t] for t in sorted_trial_ids])
            sorted_trial_types = [trial_type_lookup.get(t, 'unknown') for t in sorted_trial_ids]
            sorted_lick_flags  = [int(lick_flag_lookup.get(t, -1))   for t in sorted_trial_ids]
            pred_summary_rows.append({
                'mouse_id': mouse_id,
                'neuron_id': og_id,
                'area_acronym_custom': grp['area_acronym_custom'].iloc[0],
                'reward_group': grp['reward_group'].iloc[0],
                'trial_ids': json.dumps(sorted_trial_ids),
                'trial_types': json.dumps(sorted_trial_types),
                'lick_flags': json.dumps(sorted_lick_flags),
                'y_pred': json.dumps(y_pred_matrix.tolist()),
                'y_test': json.dumps(y_test_matrix.tolist()),
            })
        pred_summary_df = pd.DataFrame(pred_summary_rows)
        for mouse_id in pred_summary_df['mouse_id'].unique():
            outdf = pred_summary_df[pred_summary_df['mouse_id'] == mouse_id]
            outpath = f"{output_path}/predictions_summary_{mouse_id}_unit_glm_{git_version}.parquet"
            outdf.to_parquet(outpath, index=False)
            print(f"Saved predictions summary for {mouse_id} → {outpath}")



    return



def over_mouse_glm_results(nwb_list, plots,info_path, output_path, git_version, day_to_analyze = 0):

    # Load and combine NWB files
    trial_table, unit_table, ephys_nwb_list = combine_ephys_nwb(nwb_list, day_to_analyze=day_to_analyze, max_workers=20, git_version =git_version)


    mice = unit_table['mouse_id'].unique()
    df_models = load_models_optimized(mice, output_path, git_version)

    # Add a safety check before using df_models
    if df_models.empty:
        print("[CRITICAL] No model data loaded. Check the error messages above.")

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

    df_git['predictors'] = Parallel(n_jobs=-1, batch_size=1000)(delayed(parse_json_array)(s) for s in df_git['predictors'])
    df_git['y_test_array'] = Parallel(n_jobs=-1, batch_size=1000)(delayed(parse_json_array)(s) for s in df_git['y_test'])

    if 'y_pred' in df_git.columns:
        ypred_col = 'y_pred'
    elif 'y_test_pred' in df_git.columns:
        ypred_col = 'y_test_pred'
    else:
        raise KeyError("No y_pred or y_test_pred column found in df_git")

    df_git['y_pred_array'] = Parallel(n_jobs=-1, batch_size=1000)(
        delayed(parse_json_array)(s) for s in df_git[ypred_col]
    )
    merged_df = pd.merge(df_git, unit_table, how='inner', on=["mouse_id", "neuron_id"])

    area_groups = allen.get_custom_area_groups()
    area_colors = allen.get_custom_area_groups_colors()
    merged_df = allen.create_area_custom_column(merged_df)

    output_path = os.path.join(output_path, 'unit_glm', git_version)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if 'metrics' in plots :
        for reward_group in [1,0]:

            merged_df_reward = merged_df[merged_df['reward_group'] == reward_group]

            output_folder = os.path.join(output_path, 'metrics', str(reward_group))
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            plot_kde_full_vs_reduced(merged_df_reward, output_folder)

            # corr_full = compute_trialtype_correlations(merged_df[merged_df['model_name'] == 'full'], trial_table)
            for model_name in merged_df_reward['model_name'].unique():
                if model_name == 'full':
                    continue
                plot_full_vs_reduced_per_area(merged_df_reward, model_name, area_groups, area_colors, output_folder, threshold = 0.1)
                plot_full_vs_reduced_per_area(merged_df_reward, model_name, area_groups, area_colors, output_folder, threshold = None)

                # # Step 1: compute trial-type correlations for both models
                # corr_reduced = compute_trialtype_correlations(merged_df[merged_df['model_name'] == model_name], trial_table)
                # corr_all = pd.concat([corr_full, corr_reduced])
                #
                # # Step 2: plot
                # plot_full_vs_reduced_per_area_and_trialtype(
                #     corr_all,
                #     selected_reduced=model_name,
                #     area_groups=area_groups,
                #     area_colors=area_colors,
                #     output_folder=output_folder,
                #     threshold=None
                # )

            plot_kde_full_vs_reduced(merged_df_reward, output_folder)
            plot_test_corr_vs_firing_rate(merged_df_reward[merged_df_reward['model_name'] == 'full'], output_folder)
            plot_testcorr_per_mouse_reward( merged_df_reward[merged_df_reward['model_name'] == 'full'], output_folder)
            lrt = compute_lrt_from_model_results_old(merged_df_reward, alpha=0.05, ll_field='test_ll')
            lrt  =  lrt[~lrt['reduced_model'].isin(['whisker_reward_encoding'])]
            plot_lrt_significance_overlap(lrt, output_folder)
            plot_lrt_significance_per_area_per_model(
                lrt,
                area_groups=area_groups,
                area_colors=area_colors,
                output_folder=output_folder
            )
            plot_lrt_significance_heatmap(lrt, area_groups, area_colors,
                                  output_folder, annotate=False)
            
            output_folder_sub = os.path.join(output_folder, 'subset')
            if not os.path.exists(output_folder_sub):
                os.makedirs(output_folder_sub)
            lrt_subset = lrt[lrt['reduced_model'].isin(['auditory_encoding', 'block_perf_type', 'jaw_onset_encoding', 'motor_encoding', 'session_progress_encoding', 'sum_rewards', 'whisker_encoding'])]
            plot_lrt_significance_overlap(lrt_subset, output_folder_sub)
            plot_lrt_significance_per_area_per_model(
                lrt_subset,
                area_groups=area_groups,
                area_colors=area_colors,
                output_folder=output_folder_sub
            )
            plot_lrt_significance_heatmap(lrt_subset, area_groups, area_colors,
                                  output_folder_sub, annotate=False)
            output_folder_sub = os.path.join(output_folder_sub, 'per_area')
            if not os.path.exists(output_folder_sub):
                os.makedirs(output_folder_sub)
            plot_lrt_significance_overlap_per_area(lrt_subset, output_folder_sub)
            plot_lrt_significance_per_model_per_area(lrt_subset, area_groups, area_colors, output_folder_sub)

            output_folder = os.path.join(output_folder, 'per_area')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            plot_lrt_significance_overlap_per_area(lrt, output_folder)
            plot_lrt_significance_per_model_per_area(lrt, area_groups, area_colors, output_folder)

        # plot_two_reduced_per_area(merged_df, 'all_whisker_progression', 'all_whisker_progression_but_local', area_groups, area_colors, output_folder, threshold=None)

    if 'average_kernels_by_region' in plots :
        output_folder = os.path.join(output_path, 'average_kernels_by_region')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if git_version == '1cce900':
            lags =  np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        else:
            lags = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
        plot_average_kernels_by_region(  merged_df[merged_df['model_name'] == 'full'], output_folder, ['whisker_stim', 'auditory_stim', 'jaw_onset', 'piezo_reward'],
            lags=lags, area_groups=area_groups, area_colors=area_colors, n_cols=3, threshold = None, git_handle=git_version)
        plot_average_kernels_by_region(  merged_df[merged_df['model_name'] == 'full'], output_folder, ['whisker_stim', 'auditory_stim', 'jaw_onset', 'piezo_reward'],
            lags=lags, area_groups=area_groups, area_colors=area_colors, n_cols=3, threshold = 0.2, git_handle=git_version)

        decreased_neurons, _ = neurons_with_consistent_decrease(merged_df, reduced_name='whisker_encoding')
        print(f"{len(decreased_neurons)} neurons show consistent decrease across folds.")
        merged_df_sig = merged_df[merged_df['neuron_id'].isin(decreased_neurons['neuron_id'])]
        output_folder = os.path.join(output_path, 'average_kernels_by_region_sign_whisker')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plot_average_kernels_by_region(  merged_df_sig[merged_df_sig['model_name'] == 'full'], output_folder, ['whisker_stim'],
            lags=lags, area_groups=area_groups, area_colors=area_colors, n_cols=3, threshold = None, git_handle=git_version)





def _discover_kernels(dfs):
    """Scan predictor names across all dataframes and return discovered kernel groups.

    Returns
    -------
    all_kernels : dict[str, list[str]]
        Kernel bases grouped by type, e.g. {'whisker_stim': ['whisker_stim_0', 'whisker_stim_1'], ...}
    whisker_kernels : list[str]
        Flat sorted list of all discovered kernel bases (union of all_kernels values).
    """
    _PREFIX_TO_TYPE = [
        ('whisker_hits',    'whisker_hits'),
        ('whisker_misses',  'whisker_misses'),
        ('whisker_stim',    'whisker_stim'),
        ('whisker',         'whisker_other'),
        ('auditory_stim',   'auditory_stim'),
        ('piezo_reward', 'piezo_reward'),
        ('piezo_reward_au', 'piezo_reward_au')
    ]

    all_kernels = {}  # {kernel_type: set of kernel_base names}

    for df in dfs.values():
        if df.empty:
            continue
        for idx in range(min(10, len(df))):
            for pred in df.iloc[idx]['predictors_full']:
                if '_t' not in pred:
                    continue
                kernel_base = pred.split('_t')[0]

                kernel_type = None
                for prefix, ktype in _PREFIX_TO_TYPE:
                    if kernel_base.startswith(prefix):
                        kernel_type = ktype
                        break
                if kernel_type is None:
                    continue

                all_kernels.setdefault(kernel_type, set()).add(kernel_base)

    # Convert sets to sorted lists and build flat kernel list
    all_kernels = {kt: sorted(bases) for kt, bases in all_kernels.items()}
    whisker_kernels = [base for kt in sorted(all_kernels) for base in all_kernels[kt]]

    return all_kernels, whisker_kernels


def _discover_models(merged_df):
    """Map known model_name values in *merged_df* to (key, label) pairs.

    Returns
    -------
    dfs : dict[str, DataFrame]
        Filtered dataframes keyed by short model key (e.g. "2k", "2k_random").
    model_labels : dict[str, str]
        Human-readable labels keyed the same way.
    """
    dfs = {}
    model_labels = {}

    for model_name in merged_df['model_name'].unique():
        if model_name == 'full':
            key, label = '2k', '2 whisker kernel'
        elif model_name == 'random_split':
            key, label = '2k_random', '2 whisker kernel (random)'
        elif model_name == 'optimal_split':
            key, label = '2k_optimal', '2 whisker kernel (optimal split)'
        elif 'whisker_kernel' in model_name:
            match = re.search(r'(\d+)whisker_kernel', model_name)
            if not match:
                continue
            n = int(match.group(1))
            key, label = f'{n}k', f'{n} whisker kernels'
        else:
            continue

        dfs[key] = merged_df[merged_df['model_name'] == model_name]
        model_labels[key] = label

    return dfs, model_labels


def over_mouse_glm_results_new(subject_ids,nwb_list, plots, output_path, git_version, day_to_analyze = 0):

    dfs = []
    for mouse in subject_ids:
        mouse_results_path = os.path.join(output_path, mouse, 'whisker_0', 'unit_glm', git_version) #, 'test'
        fpath = os.path.join(mouse_results_path, f"summary_{mouse}_unit_glm_{git_version}.parquet")
        if not os.path.exists(fpath):
            print(f"[WARNING] Summary not found: {fpath}")
            continue
        df = pd.read_parquet(fpath)
        dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)

    output_path = os.path.join(output_path, 'unit_glm', git_version, 'test2')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    area_groups = allen.get_custom_area_groups()
    area_colors = allen.get_custom_area_groups_colors()

    area_group_from_name = allen.get_custom_area_groups_from_name()
    merged_df_grouped = merged_df.copy()
    merged_df_grouped['area_acronym_custom'] = merged_df_grouped['area_acronym_custom'].map(area_group_from_name)
    merged_df_grouped = merged_df_grouped[merged_df_grouped['area_acronym_custom'].notna()]
    area_groups_flat = {g: [g] for g in area_groups.keys()}

    # trial_table, unit_table, ephys_nwb_list = combine_ephys_nwb(nwb_list, day_to_analyze=day_to_analyze, max_workers=20, git_version =git_version)
    # merged_df = pd.merge(merged_df, unit_table, how='inner', on=["mouse_id", "neuron_id", "area_acronym_custom"])

    if 'metrics' in plots :
        output_folder = os.path.join(output_path, 'metrics')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        compare_full_vs_reduced_models(merged_df, output_folder)
        # plot_kde_full_vs_reduced(merged_df, output_folder)
        # plot_test_corr_vs_firing_rate(merged_df[merged_df['model_name'] == 'full'], output_folder)
        # plot_testcorr_per_mouse_reward( merged_df[merged_df['model_name'] == 'full'], output_folder)
        for reward_group in [1,0]:

            merged_df_reward = merged_df[merged_df['reward_group'] == reward_group]

            if reward_group == 1:

                output_folder = os.path.join(output_path, 'metrics', str('r+'))
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
            else:
                
                output_folder = os.path.join(output_path, 'metrics', str('r-'))
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

            # corr_full = compute_trialtype_correlations(merged_df[merged_df['model_name'] == 'full'], trial_table)
            for model_name in merged_df_reward['model_name'].unique():
                if model_name == 'whisker_reward_encoding':
                    continue
                if model_name == 'full':
                    continue
                print(model_name)
                # plot_full_vs_reduced_per_area(merged_df_reward, model_name, area_groups, area_colors, output_folder, threshold = 0.1)
                # plot_full_vs_reduced_per_area(merged_df_reward, model_name, area_groups, area_colors, output_folder, threshold = None)
                # plot_full_vs_reduced_barplot(merged_df_reward, model_name, output_folder)

            # plot_kde_full_vs_reduced(merged_df_reward, output_folder)
            # plot_test_corr_vs_firing_rate(merged_df_reward[merged_df_reward['model_name'] == 'full'], output_folder)
            # plot_testcorr_per_mouse_reward( merged_df_reward[merged_df_reward['model_name'] == 'full'], output_folder)
            merged_df_reward_lrt = merged_df_reward[merged_df_reward['model_name'] != 'full']
            merged_df_reward_lrt = allen.merge_liu_avg_ipsi(merged_df_reward_lrt)
            merged_df_reward_lrt = allen.merge_harris_hierarchy(merged_df_reward_lrt)
            # plot_lrt_significance_overlap(merged_df_reward_lrt, output_folder)
            # print(merged_df_reward_lrt.keys())
            # plot_lrt_significance_per_area_per_model(
            #     merged_df_reward_lrt,
            #     area_groups=area_groups,
            #     area_colors=area_colors,
            #     output_folder=output_folder
            # )
            plot_lrt_ipsi_correlation(merged_df_reward_lrt, output_folder)
            plot_lrt_harris_correlation(merged_df_reward_lrt, output_folder)
            plot_lrt_significance_heatmap(merged_df_reward_lrt, area_groups, area_colors,
                                  output_folder, annotate=False)
            
            output_folder_sub = os.path.join(output_folder, 'subset')
            if not os.path.exists(output_folder_sub):
                os.makedirs(output_folder_sub)
            lrt_subset = merged_df_reward_lrt[merged_df_reward_lrt['model_name'].isin(['auditory_encoding', 'jaw_onset_encoding', 'motor_encoding', 'session_progress_encoding', 'sum_rewards', 'whisker_encoding'])]
            plot_lrt_significance_overlap(lrt_subset, output_folder_sub)
            plot_lrt_significance_per_area_per_model(
                lrt_subset,
                area_groups=area_groups,
                area_colors=area_colors,
                output_folder=output_folder_sub
            )
            plot_lrt_ipsi_correlation(lrt_subset, output_folder_sub)
            plot_lrt_harris_correlation(lrt_subset, output_folder_sub)
            plot_lrt_significance_heatmap(lrt_subset, area_groups, area_colors,
                                  output_folder_sub, annotate=False)
            output_folder_sub = os.path.join(output_folder_sub, 'per_area')
            if not os.path.exists(output_folder_sub):
                os.makedirs(output_folder_sub)
            plot_lrt_significance_overlap_per_area(lrt_subset, output_folder_sub)
            plot_lrt_significance_per_model_per_area(lrt_subset, area_groups, area_colors, output_folder_sub)

            output_folder = os.path.join(output_folder, 'per_area')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            plot_lrt_significance_overlap_per_area(merged_df_reward_lrt, output_folder)
            plot_lrt_significance_per_model_per_area(merged_df_reward_lrt, area_groups, area_colors, output_folder)

        # plot_two_reduced_per_area(merged_df, 'all_whisker_progression', 'all_whisker_progression_but_local', area_groups, area_colors, output_folder, threshold=None)

    if 'average_kernels_by_region' in plots :
        output_folder = os.path.join(output_path, 'average_kernels_by_region')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if git_version == '1cce900':
            lags =  np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        else:
            lags = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
        # plot_average_kernels_by_region_new(  merged_df[merged_df['model_name'] == 'full'], output_folder, ['whisker_stim', 'auditory_stim', 'jaw_onset', 'piezo_reward'],
        #     lags=lags, area_groups=area_groups, area_colors=area_colors, n_cols=3, threshold = None, git_handle=git_version)
        for reward_group in [0,1]:

            merged_df_reward = merged_df[merged_df['reward_group'] == reward_group]
            # print(merged_df_reward['predictors_full'].iloc[0])
            output_folder_reward_group = os.path.join(output_folder, str(reward_group))
            if not os.path.exists(output_folder_reward_group):
                os.makedirs(output_folder_reward_group)
            plot_average_kernels_by_region_new(  merged_df_reward[merged_df_reward['model_name'] == 'full'], output_folder_reward_group, ['whisker_hits_stim_0','whisker_hits_stim_1', 'auditory_stim_0','auditory_stim_1','whisker_misses_stim_0','whisker_misses_stim_1','jaw_onset', 'piezo_reward_0','piezo_reward_1','piezo_reward_au_0','piezo_reward_au_1'],
                lags=None, area_groups=area_groups, area_colors=area_colors, n_cols=3, threshold = None, git_handle=git_version)
            plot_average_kernels_by_region_new(  merged_df[merged_df['model_name'] == 'full'], output_folder_reward_group, ['whisker_stim','auditory_stim','jaw_onset','piezo_reward'],
                lags=None, area_groups=area_groups, area_colors=area_colors, n_cols=3, threshold = None, git_handle=git_version)
            # plot_average_kernels_by_region_new(  merged_df[merged_df['model_name'] == 'full'], output_folder, ['whisker_hits_stim_0','whisker_hits_stim_1','auditory_stim_0','auditory_stim_1','whisker_misses_stim_0','whisker_misses_stim_1','jaw_onset','piezo_reward_0','piezo_reward_1','piezo_reward_au_0','piezo_reward_au_1'],
            #     lags=None, area_groups=area_groups, area_colors=area_colors, n_cols=3, threshold = None, git_handle=git_version)

            plot_all_kernels_by_region(
                df= merged_df_reward[merged_df_reward['model_name'] == 'full'],
                output_folder=output_folder_reward_group,
                kernels_to_plot= ['whisker_hits_stim_0','whisker_hits_stim_1', 'whisker_misses_stim_0','whisker_misses_stim_1', 'auditory_stim_0','auditory_stim_1'],
                area_groups=area_groups,
                area_colors=area_colors,
                kernel_colors={
                    'whisker_hits_stim_0': 'lightgreen',
                    'whisker_hits_stim_1': 'green',
                    'whisker_misses_stim_0': 'salmon',
                    'whisker_misses_stim_1': 'red',
                    'auditory_stim_0': 'lightblue',
                    'auditory_stim_1': 'blue',
                },
                n_cols=3,
                threshold=None
            )

    if 'kernel_consistency' in plots:
        output_folder = os.path.join(output_path, 'kernel_consistency')
        os.makedirs(output_folder, exist_ok=True)

        # Kernel pairs to compare – add / remove entries to taste.
        kernel_pairs_to_compare = [
            ('whisker_hits_stim_0',   'whisker_hits_stim_1'),
            ('whisker_misses_stim0', 'whisker_misses_stim1'),
            ('whisker_misses_stim_0', 'whisker_misses_stim_1'),
            ('auditory_stim_0',       'auditory_stim_1'),
            ('auditory_stim0',       'auditory_stim1'),
            ('piezo_reward_0', 'piezo_reward_1'),
            ('piezo_reward_au_0', 'piezo_reward_au_1')
        ]

        # Use the full-model rows only to check which kernels are present
        df_full = merged_df[merged_df['model_name'] == 'full']

        for k0, k1 in kernel_pairs_to_compare:
            # Check that both kernels are present in this dataset
            sample_preds = df_full.iloc[0]['predictors_full'] if len(df_full) > 0 else []
            has_k0 = any(p.startswith(k0 + '_t') for p in sample_preds)
            has_k1 = any(p.startswith(k1 + '_t') for p in sample_preds)
            if not has_k0 or not has_k1:
                print(f"[kernel_consistency] Skipping {k0} vs {k1}: not found in predictors.")
                continue

            pair_folder = os.path.join(output_folder, f"{k1}_vs_{k0}")
            # Pass the full merged_df (all model_names) so that the LRT encoding
            # profile can access lrt_significant from the reduced-model rows.
            # _build_neuron_change_table filters to model_name=='full' internally.
            plot_kernel_consistency_by_area(
                df=merged_df,
                kernel_0=k0,
                kernel_1=k1,
                area_groups=area_groups,
                area_colors=area_colors,
                output_folder=pair_folder,
                reward_groups=[1, 0],
                thresholds=[1, 5, 10],
                n_cols=3,
                performance_csv=r'M:\analysis\Myriam_Hamon\combined_results\whisker_session_rates\whisker_session_rates.csv',
            )

            # Grouped-by-area-group version
            pair_folder_grouped = os.path.join(pair_folder, 'by_area_group')
            plot_kernel_consistency_by_area(
                df=merged_df_grouped,
                kernel_0=k0,
                kernel_1=k1,
                area_groups=area_groups_flat,
                area_colors=area_colors,
                output_folder=pair_folder_grouped,
                reward_groups=[1, 0],
                thresholds=[1, 5, 10],
                n_cols=3,
                performance_csv=r'M:\analysis\Myriam_Hamon\combined_results\whisker_session_rates\whisker_session_rates.csv',
            )

            # Repeat for whisker-encoding-only and auditory-encoding-only neurons
            if 'lrt_significant' in merged_df.columns:
                for encoding_model, encoding_label in [
                    ('whisker_encoding', 'wsk'),
                    ('auditory_encoding', 'aud'),
                ]:
                    sig_ids = merged_df[
                        (merged_df['model_name'] == encoding_model) &
                        (merged_df['lrt_significant'] == True)
                    ][['mouse_id', 'neuron_id']].drop_duplicates()
                    if sig_ids.empty:
                        print(f"[kernel_consistency] No {encoding_model} neurons found, skipping {encoding_label}.")
                        continue
                    merged_df_encoding = merged_df.merge(sig_ids, on=['mouse_id', 'neuron_id'])
                    pair_folder_encoding = os.path.join(pair_folder, encoding_label)
                    plot_kernel_consistency_by_area(
                        df=merged_df_encoding,
                        kernel_0=k0,
                        kernel_1=k1,
                        area_groups=area_groups,
                        area_colors=area_colors,
                        output_folder=pair_folder_encoding,
                        reward_groups=[1, 0],
                        thresholds=[1, 5, 10],
                        n_cols=3,
                    )

                    # Grouped-by-area-group version for encoding subsets
                    merged_df_grouped_encoding = merged_df_grouped.merge(sig_ids, on=['mouse_id', 'neuron_id'])
                    if not merged_df_grouped_encoding.empty:
                        pair_folder_encoding_grouped = os.path.join(pair_folder_grouped, encoding_label)
                        plot_kernel_consistency_by_area(
                            df=merged_df_grouped_encoding,
                            kernel_0=k0,
                            kernel_1=k1,
                            area_groups=area_groups_flat,
                            area_colors=area_colors,
                            output_folder=pair_folder_encoding_grouped,
                            reward_groups=[1, 0],
                            thresholds=[1, 5, 10],
                            n_cols=3,
                        )


                
    if 'compare_kernels' in plots:
        output_folder = os.path.join(output_path, 'compare_kernels_claude')
        os.makedirs(output_folder, exist_ok=True)


        # dfs, model_labels = _discover_models(merged_df)
        # analyze_kernel_amplitude_differences_2(
        #     dfs=dfs,
        #     model_labels=model_labels,
        #     output_folder=output_folder_random,
        #     whisker_kernels=whisker_kernels_random,
        #     area_groups=area_groups,
        #     area_colors=area_colors,
        #     n_cols=3
        # )
        

        # Per-reward-group analyses
        for reward_group in [1, 0]:
            merged_df_reward = merged_df[merged_df['reward_group'] == reward_group]

            output_folder_reward = os.path.join(output_folder, str(reward_group))
            os.makedirs(output_folder_reward, exist_ok=True)

            dfs, model_labels = _discover_models(merged_df_reward)
            if not dfs:
                print(f"[WARNING] No whisker kernel models found for reward_group {reward_group}")
                continue

            print(f"[INFO] Found {len(dfs)} whisker kernel models for reward_group {reward_group}: {list(model_labels.values())}")

            all_kernels, whisker_kernels = _discover_kernels(dfs)
            print(f"[INFO] Discovered kernel types and counts:")
            for kernel_type, kernels in sorted(all_kernels.items()):
                print(f"  - {kernel_type}: {len(kernels)} kernels {kernels}")
            print(f"[INFO] Total whisker kernels: {len(whisker_kernels)}")

            # 1. Model fit comparison
            # output_folder_fit = os.path.join(output_folder_reward, 'model_fit_comparison')
            # os.makedirs(output_folder_fit, exist_ok=True)
            # compare_model_fit_metrics(
            #     dfs=dfs,
            #     model_labels=model_labels,
            #     output_folder=output_folder_fit,
            #     area_groups=area_groups,
            #     area_colors=area_colors,
            #     metrics=['test_corr'],
            #     n_cols=3
            # )

            # 2. Kernel amplitude evolution
            output_folder_amplitude = os.path.join(output_folder_reward, 'kernel_amplitude_evolution')
            os.makedirs(output_folder_amplitude, exist_ok=True)
            compare_kernel_amplitude_evolution(
                dfs=dfs,
                model_labels=model_labels,
                output_folder=output_folder_amplitude,
                whisker_kernels=whisker_kernels,
                area_groups=area_groups,
                area_colors=area_colors
            )

            # 3. First-weight amplitude evolution + per-kernel amplitude differences
            output_folder_first_weight = os.path.join(output_folder_reward, 'first_weight_amplitude_evolution')
            os.makedirs(output_folder_first_weight, exist_ok=True)
            # compare_kernel_amplitude_evolution_2(
            #     dfs=dfs,
            #     model_labels=model_labels,
            #     output_folder=output_folder_first_weight,
            #     whisker_kernels=whisker_kernels,
            #     area_groups=area_groups,
            #     area_colors=area_colors
            # )
            analyze_kernel_amplitude_differences_2(
                dfs=dfs,
                model_labels=model_labels,
                output_folder=output_folder_first_weight,
                whisker_kernels=whisker_kernels,
                area_groups=area_groups,
                area_colors=area_colors,
                n_cols=3
            )

            # 4. Kernel consistency analysis
            output_folder_consistency = os.path.join(output_folder_reward, 'kernel_consistency')
            os.makedirs(output_folder_consistency, exist_ok=True)
            analyze_kernel_consistency(
                dfs=dfs,
                model_labels=model_labels,
                output_folder=output_folder_consistency,
                whisker_kernels=whisker_kernels,
                area_groups=area_groups,
                area_colors=area_colors
            )

            # 5. Top-changing neurons (across and within models)
            # output_folder_top_neurons = os.path.join(output_folder_reward, 'top_changing_neurons')
            # os.makedirs(output_folder_top_neurons, exist_ok=True)
            # identify_neurons_with_kernel_changes(
            #     dfs=dfs,
            #     model_labels=model_labels,
            #     output_folder=output_folder_top_neurons,
            #     whisker_kernels=whisker_kernels,
            #     area_groups=area_groups,
            #     area_colors=area_colors,
            #     top_n=20
            # )

            # 6. Paired kernel comparison per area (stim_0 vs stim_1 with per-bin test)
            output_folder_paired = os.path.join(output_folder_reward, 'paired_kernel_comparison')
            os.makedirs(output_folder_paired, exist_ok=True)
            plot_paired_kernel_comparison_per_area(
                dfs=dfs,
                model_labels=model_labels,
                output_folder=output_folder_paired,
                area_groups=area_groups,
                area_colors=area_colors,
                n_cols=3,
                alpha=0.05
            )

            # 7. Full vs optimal_split kernel comparison per area
            if '2k' in dfs and '2k_optimal' in dfs:
                output_folder_optimal = os.path.join(output_folder_reward, 'full_vs_optimal_split')
                os.makedirs(output_folder_optimal, exist_ok=True)
                plot_kernels_full_vs_optimal_split_per_area(
                    df_full=dfs['2k'],
                    df_optimal=dfs['2k_optimal'],
                    output_folder=output_folder_optimal,
                    area_groups=area_groups,
                    area_colors=area_colors,
                    n_cols=3,
                    alpha=0.05
                )

                # 8. split_idx distribution
                output_folder_split = os.path.join(output_folder_reward, 'split_idx_distribution')
                os.makedirs(output_folder_split, exist_ok=True)
                plot_split_idx_distribution(
                    df_optimal=dfs['2k_optimal'],
                    output_folder=output_folder_split,
                    area_groups=area_groups,
                    area_colors=area_colors,
                    n_cols=3
                )

            # 9. Encoding-specific neuron subsets (whisker-only, auditory-only)
            if 'lrt_significant' in merged_df_reward.columns:
                for encoding_model, encoding_label in [
                    ('whisker_encoding', 'wsk'),
                    ('auditory_encoding', 'aud'),
                ]:
                    sig_ids = merged_df_reward[
                        (merged_df_reward['model_name'] == encoding_model) &
                        (merged_df_reward['lrt_significant'] == True)
                    ][['mouse_id', 'neuron_id']].drop_duplicates()
                    if sig_ids.empty:
                        print(f"[compare_kernels] No {encoding_model} neurons found for reward_group {reward_group}, skipping.")
                        continue
                    merged_df_encoding = merged_df_reward.merge(sig_ids, on=['mouse_id', 'neuron_id'])
                    dfs_enc, model_labels_enc = _discover_models(merged_df_encoding)
                    if not dfs_enc:
                        print(f"[compare_kernels] No models after filtering for {encoding_label}, reward_group {reward_group}, skipping.")
                        continue
                    all_kernels_enc, whisker_kernels_enc = _discover_kernels(dfs_enc)

                    output_folder_enc = os.path.join(output_folder_reward, encoding_label)
                    os.makedirs(output_folder_enc, exist_ok=True)

                    output_folder_enc_amplitude = os.path.join(output_folder_enc, 'kernel_amplitude_evolution')
                    os.makedirs(output_folder_enc_amplitude, exist_ok=True)
                    compare_kernel_amplitude_evolution(
                        dfs=dfs_enc,
                        model_labels=model_labels_enc,
                        output_folder=output_folder_enc_amplitude,
                        whisker_kernels=whisker_kernels_enc,
                        area_groups=area_groups,
                        area_colors=area_colors,
                    )

                    output_folder_enc_consistency = os.path.join(output_folder_enc, 'kernel_consistency')
                    os.makedirs(output_folder_enc_consistency, exist_ok=True)
                    analyze_kernel_consistency(
                        dfs=dfs_enc,
                        model_labels=model_labels_enc,
                        output_folder=output_folder_enc_consistency,
                        whisker_kernels=whisker_kernels_enc,
                        area_groups=area_groups,
                        area_colors=area_colors,
                    )

        # Identical plots grouped by area_group
        for reward_group in [1, 0]:
            merged_df_reward_grouped = merged_df_grouped[merged_df_grouped['reward_group'] == reward_group]

            output_folder_reward_grouped = os.path.join(output_folder, str(reward_group), 'by_area_group')
            os.makedirs(output_folder_reward_grouped, exist_ok=True)

            dfs_grp, model_labels_grp = _discover_models(merged_df_reward_grouped)
            if not dfs_grp:
                print(f"[WARNING] No whisker kernel models found for reward_group {reward_group} (grouped)")
                continue

            all_kernels_grp, whisker_kernels_grp = _discover_kernels(dfs_grp)

            output_folder_amplitude_grp = os.path.join(output_folder_reward_grouped, 'kernel_amplitude_evolution')
            os.makedirs(output_folder_amplitude_grp, exist_ok=True)
            compare_kernel_amplitude_evolution(
                dfs=dfs_grp,
                model_labels=model_labels_grp,
                output_folder=output_folder_amplitude_grp,
                whisker_kernels=whisker_kernels_grp,
                area_groups=area_groups_flat,
                area_colors=area_colors
            )

            output_folder_first_weight_grp = os.path.join(output_folder_reward_grouped, 'evolution_weights')
            os.makedirs(output_folder_first_weight_grp, exist_ok=True)
            analyze_kernel_amplitude_differences_2(
                dfs=dfs_grp,
                model_labels=model_labels_grp,
                output_folder=output_folder_first_weight_grp,
                whisker_kernels=whisker_kernels_grp,
                area_groups=area_groups_flat,
                area_colors=area_colors,
                n_cols=3
            )

            output_folder_consistency_grp = os.path.join(output_folder_reward_grouped, 'kernel_consistency')
            os.makedirs(output_folder_consistency_grp, exist_ok=True)
            analyze_kernel_consistency(
                dfs=dfs_grp,
                model_labels=model_labels_grp,
                output_folder=output_folder_consistency_grp,
                whisker_kernels=whisker_kernels_grp,
                area_groups=area_groups_flat,
                area_colors=area_colors
            )

            output_folder_paired_grp = os.path.join(output_folder_reward_grouped, 'paired_kernel_comparison')
            os.makedirs(output_folder_paired_grp, exist_ok=True)
            plot_paired_kernel_comparison_per_area(
                dfs=dfs_grp,
                model_labels=model_labels_grp,
                output_folder=output_folder_paired_grp,
                area_groups=area_groups_flat,
                area_colors=area_colors,
                n_cols=3,
                alpha=0.05
            )

            if '2k' in dfs_grp and '2k_optimal' in dfs_grp:
                output_folder_optimal_grp = os.path.join(output_folder_reward_grouped, 'full_vs_optimal_split')
                os.makedirs(output_folder_optimal_grp, exist_ok=True)
                plot_kernels_full_vs_optimal_split_per_area(
                    df_full=dfs_grp['2k'],
                    df_optimal=dfs_grp['2k_optimal'],
                    output_folder=output_folder_optimal_grp,
                    area_groups=area_groups_flat,
                    area_colors=area_colors,
                    n_cols=3,
                    alpha=0.05
                )

                output_folder_split_grp = os.path.join(output_folder_reward_grouped, 'split_idx_distribution')
                os.makedirs(output_folder_split_grp, exist_ok=True)
                plot_split_idx_distribution(
                    df_optimal=dfs_grp['2k_optimal'],
                    output_folder=output_folder_split_grp,
                    area_groups=area_groups_flat,
                    area_colors=area_colors,
                    n_cols=3
                )

            merged_df_reward_ungrouped = merged_df[merged_df['reward_group'] == reward_group]
            if 'lrt_significant' in merged_df_reward_ungrouped.columns:
                for encoding_model, encoding_label in [
                    ('whisker_encoding', 'wsk'),
                    ('auditory_encoding', 'aud'),
                ]:
                    sig_ids = merged_df_reward_ungrouped[
                        (merged_df_reward_ungrouped['model_name'] == encoding_model) &
                        (merged_df_reward_ungrouped['lrt_significant'] == True)
                    ][['mouse_id', 'neuron_id']].drop_duplicates()
                    if sig_ids.empty:
                        print(f"[compare_kernels grouped] No {encoding_model} neurons found for reward_group {reward_group}, skipping.")
                        continue
                    merged_df_grouped_encoding = merged_df_reward_grouped.merge(sig_ids, on=['mouse_id', 'neuron_id'])
                    dfs_enc_grp, model_labels_enc_grp = _discover_models(merged_df_grouped_encoding)
                    if not dfs_enc_grp:
                        print(f"[compare_kernels grouped] No models after filtering for {encoding_label}, reward_group {reward_group}, skipping.")
                        continue
                    all_kernels_enc_grp, whisker_kernels_enc_grp = _discover_kernels(dfs_enc_grp)

                    output_folder_enc_grp = os.path.join(output_folder_reward_grouped, encoding_label)
                    os.makedirs(output_folder_enc_grp, exist_ok=True)

                    output_folder_enc_amplitude_grp = os.path.join(output_folder_enc_grp, 'kernel_amplitude_evolution')
                    os.makedirs(output_folder_enc_amplitude_grp, exist_ok=True)
                    compare_kernel_amplitude_evolution(
                        dfs=dfs_enc_grp,
                        model_labels=model_labels_enc_grp,
                        output_folder=output_folder_enc_amplitude_grp,
                        whisker_kernels=whisker_kernels_enc_grp,
                        area_groups=area_groups_flat,
                        area_colors=area_colors,
                    )

                    output_folder_enc_consistency_grp = os.path.join(output_folder_enc_grp, 'kernel_consistency')
                    os.makedirs(output_folder_enc_consistency_grp, exist_ok=True)
                    analyze_kernel_consistency(
                        dfs=dfs_enc_grp,
                        model_labels=model_labels_enc_grp,
                        output_folder=output_folder_enc_consistency_grp,
                        whisker_kernels=whisker_kernels_enc_grp,
                        area_groups=area_groups_flat,
                        area_colors=area_colors,
                    )

    if 'average_activity_predictions' in plots:
        output_folder_pred = os.path.join(output_path, 'average_activity_predictions')
        os.makedirs(output_folder_pred, exist_ok=True)

        pred_dfs = []
        # output_path was reassigned to {base}/unit_glm/{git_version}; go up two levels
        base_output_path = os.path.dirname(os.path.dirname(os.path.dirname(output_path)))
        for mouse in subject_ids:
            mouse_results_path = os.path.join(
                base_output_path, mouse, 'whisker_0', 'unit_glm', git_version, 'test'
            )
            pred_fpath = os.path.join(mouse_results_path,
                                      f"predictions_summary_{mouse}_unit_glm_{git_version}.parquet")
            if not os.path.exists(pred_fpath):
                print(f"[WARNING] Predictions summary not found: {pred_fpath}")
                continue
            pred_df = pd.read_parquet(pred_fpath)
            pred_dfs.append(pred_df)

        if pred_dfs:
            merged_pred_df = pd.concat(pred_dfs, ignore_index=True)

            # Attach reward_group from the main merged_df if not already in the file
            if 'reward_group' not in merged_pred_df.columns:
                mouse_reward = merged_df[['mouse_id', 'reward_group']].drop_duplicates()
                merged_pred_df = merged_pred_df.merge(mouse_reward, on='mouse_id', how='left')

            missing = [c for c in ('trial_types', 'lick_flags') if c not in merged_pred_df.columns]
            if missing:
                print(f"[WARNING] Columns {missing} missing from predictions_summary. "
                      "Re-run mouse_glm_results with 'create_summary' to regenerate the files.")
            else:
                def _run_pred_plots(df_pred_subset, folder):
                    os.makedirs(folder, exist_ok=True)
                    plot_psth_predictions_per_trialtype_per_area(
                        df_pred=df_pred_subset,
                        area_groups=area_groups,
                        area_colors=area_colors,
                        output_folder=folder,
                    )
                    if 'trial_ids' in df_pred_subset.columns:
                        plot_psth_whisker_session_halves_per_area(
                            df_pred=df_pred_subset,
                            area_groups=area_groups,
                            area_colors=area_colors,
                            output_folder=folder,
                        )

                # ── All neurons ───────────────────────────────────────────────
                _run_pred_plots(merged_pred_df, output_folder_pred)

                # ── Per LRT-significant group ─────────────────────────────────
                # Build a lookup: (mouse_id, neuron_id, model_name) → lrt_significant
                lrt_cols = ['mouse_id', 'neuron_id', 'model_name', 'lrt_significant']
                if all(c in merged_df.columns for c in lrt_cols):
                    lrt_lookup = (
                        merged_df[merged_df['model_name'] != 'full'][lrt_cols]
                        .dropna(subset=['lrt_significant'])
                        .groupby(['mouse_id', 'neuron_id', 'model_name'])['lrt_significant']
                        .first()
                        .reset_index()
                    )

                    for model_name in sorted(lrt_lookup['model_name'].unique()):
                        model_lrt = lrt_lookup[lrt_lookup['model_name'] == model_name]
                        model_folder = os.path.join(
                            output_folder_pred, 'lrt_groups',
                            model_name.replace(' ', '_'))

                        sig_ids = model_lrt.loc[
                            model_lrt['lrt_significant'] == True,
                            ['mouse_id', 'neuron_id']]
                        not_sig_ids = model_lrt.loc[
                            model_lrt['lrt_significant'] == False,
                            ['mouse_id', 'neuron_id']]

                        for label, id_df in [('lrt_significant', sig_ids),
                                             ('lrt_not_significant', not_sig_ids)]:
                            if id_df.empty:
                                continue
                            subset = merged_pred_df.merge(
                                id_df, on=['mouse_id', 'neuron_id'], how='inner')
                            if subset.empty:
                                continue
                            folder = os.path.join(model_folder, label)
                            print(f"[average_activity_predictions] {model_name} / {label}: "
                                  f"{len(subset)} neurons")
                            _run_pred_plots(subset, folder)
                else:
                    print("[WARNING] 'lrt_significant' not found in merged_df – "
                          "skipping per-LRT-group prediction plots.")

                # ── Per kernel-pair consistency group ─────────────────────────
                cons_cols = ['mouse_id', 'neuron_id', 'kernel_pair',
                             'is_consistent', 'consistency_score']
                if all(c in merged_df.columns for c in cons_cols):
                    df_full_cons = (
                        merged_df[merged_df['model_name'] == 'full'][cons_cols]
                        .dropna(subset=['kernel_pair'])
                        .groupby(['mouse_id', 'neuron_id', 'kernel_pair'])
                        .first()
                        .reset_index()
                    )

                    for pair_name in sorted(df_full_cons['kernel_pair'].dropna().unique()):
                        pair_df = df_full_cons[df_full_cons['kernel_pair'] == pair_name]
                        pair_folder = os.path.join(
                            output_folder_pred, 'consistency_groups',
                            pair_name.replace(' ', '_'))

                        is_cons = pair_df['is_consistent'].fillna(False).astype(bool)
                        score   = pair_df['consistency_score'].fillna(0)

                        up_ids   = pair_df.loc[is_cons & (score > 0), ['mouse_id', 'neuron_id']]
                        down_ids = pair_df.loc[is_cons & (score < 0), ['mouse_id', 'neuron_id']]
                        non_ids  = pair_df.loc[~is_cons,               ['mouse_id', 'neuron_id']]

                        for label, id_df in [('consistent_up',   up_ids),
                                             ('consistent_down', down_ids),
                                             ('non_consistent',  non_ids)]:
                            if id_df.empty:
                                continue
                            subset = merged_pred_df.merge(
                                id_df, on=['mouse_id', 'neuron_id'], how='inner')
                            if subset.empty:
                                continue
                            folder = os.path.join(pair_folder, label)
                            print(f"[average_activity_predictions] {pair_name} / {label}: "
                                  f"{len(subset)} neurons")
                            _run_pred_plots(subset, folder)
                else:
                    print("[WARNING] Consistency columns not found in merged_df – "
                          "skipping per-consistency-group prediction plots.")
        else:
            print("[WARNING] No predictions summary files found for any mouse.")

    if 'activity_correlation_matrices' in plots:
        output_folder_corr = os.path.join(output_path, 'activity_correlation_matrices')
        os.makedirs(output_folder_corr, exist_ok=True)

        pred_dfs = []
        base_output_path = os.path.dirname(os.path.dirname(output_path))
        for mouse in subject_ids:
            mouse_results_path = os.path.join(
                base_output_path, mouse, 'whisker_0', 'unit_glm', git_version)
            pred_fpath = os.path.join(mouse_results_path,
                                      f"predictions_summary_{mouse}_unit_glm_{git_version}.parquet")
            if not os.path.exists(pred_fpath):
                print(f"[WARNING] Predictions summary not found: {pred_fpath}")
                continue
            pred_dfs.append(pd.read_parquet(pred_fpath))

        if pred_dfs:
            merged_pred_df = pd.concat(pred_dfs, ignore_index=True)
            if 'reward_group' not in merged_pred_df.columns:
                mouse_reward = merged_df[['mouse_id', 'reward_group']].drop_duplicates()
                merged_pred_df = merged_pred_df.merge(mouse_reward, on='mouse_id', how='left')

            plot_activity_trial_correlation_matrices(
                merged_pred_df=merged_pred_df,
                merged_df=merged_df,
                output_folder=output_folder_corr,
                area_groups=area_groups,
                area_colors=area_colors,
            )
        else:
            print("[WARNING] No predictions summary files found for any mouse.")


def over_mouse_compare_git_results_new(subject_ids, plots,info_path, output_path, git_versions, day_to_analyze = 0):

    dfs = []
    for git_version in git_versions:
        for mouse in subject_ids:
            mouse_results_path = os.path.join(output_path, mouse, 'whisker_0', 'unit_glm', git_version)
            fpath = os.path.join(mouse_results_path, f"summary_{mouse}_unit_glm_{git_version}.parquet")
            if not os.path.exists(fpath):
                print(f"[WARNING] Summary not found: {fpath}")
                continue
            df = pd.read_parquet(fpath)
            dfs.append(df)
    df_models = pd.concat(dfs, ignore_index=True)

    git_v1, git_v2 = git_versions[:2]

    comparison_folder_name = f'comparison_{git_v1}_{git_v2}'
    output_path_comparison = os.path.join(output_path, 'unit_glm', comparison_folder_name)
    os.makedirs(output_path_comparison, exist_ok=True)


    required = (
        df_models
        .groupby(['mouse_id', 'neuron_id', 'model_name'])['git_version']
        .nunique()
        .reset_index()
    )

    # We want only model_type="full" (or all types if needed)
    required_full = required[required['model_name'] == 'full']

    # Need at least 2 git versions for this neuron
    valid_pairs = required_full[required_full['git_version'] == len(git_versions)][['mouse_id', 'neuron_id']]

    # Filter df_models (or merged_df) using inner merge
    df_models = df_models.merge(valid_pairs, on=['mouse_id', 'neuron_id'], how='inner')

    # Example: select the two git versions to compare
    git_v1, git_v2 = git_versions[:2]

    df_v1 = df_models[(df_models['git_version'] == git_v1)]
    df_v2 = df_models[(df_models['git_version'] == git_v2) ]


    if 'metrics' in plots :

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        # Loop over all model_names
        model_name = 'full'
        # Select only this model_name and full type for each git version
        df_v1 = df_models[(df_models['git_version'] == git_v1) &
                        (df_models['model_name'] == model_name)]
        df_v2 = df_models[(df_models['git_version'] == git_v2) &
                        (df_models['model_name'] == model_name)]
        
        # Merge by neuron
        df_compare = pd.merge(
            df_v1[['mouse_id','neuron_id','test_corr']],
            df_v2[['mouse_id','neuron_id','test_corr']],
            on=['mouse_id','neuron_id'],
            suffixes=(f'_{git_v1}', f'_{git_v2}')
        )

        # Compute differences
        df_compare['test_corr_diff'] = df_compare[f'test_corr_{git_v1}'] - df_compare[f'test_corr_{git_v2}']

        # --- Density scatter plot: test_corr git_v1 vs git_v2 ---
        x = df_compare[f'test_corr_{git_v1}'].values
        y = df_compare[f'test_corr_{git_v2}'].values
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        x, y, z = compute_density_safe(x, y)

        plt.figure(figsize=(6,6))
        scatter = plt.scatter(x, y, c=z, s=30, cmap='viridis', edgecolor='none')
        plt.plot([0,1],[0,1],'r--', label='unity line')
        plt.xlabel(f'Test corr {git_v1}')
        plt.ylabel(f'Test corr {git_v2}')
        plt.title(f'Density scatter: {model_name}')
        plt.colorbar(scatter, label='Point density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path_comparison,
                                f'test_corr_density_{model_name}_{git_v1}_vs_{git_v2}.png'), dpi=300)
        plt.close()
        
        # Merge by neuron
        df_compare = pd.merge(
            df_v1[['mouse_id','neuron_id','test_ll']],
            df_v2[['mouse_id','neuron_id','test_ll']],
            on=['mouse_id','neuron_id'],
            suffixes=(f'_{git_v1}', f'_{git_v2}')
        )

        # Compute differences
        df_compare['test_ll_diff'] = df_compare[f'test_ll_{git_v1}'] - df_compare[f'test_ll_{git_v2}']

        # --- Density scatter plot: test_corr git_v1 vs git_v2 ---
        x = df_compare[f'test_ll_{git_v1}'].values
        y = df_compare[f'test_ll_{git_v2}'].values
        mask = np.isfinite(x) & np.isfinite(y)

        x = x[mask]
        y = y[mask]
        # Clip LL values at the 1st and 99th percentile
        x_clip = np.clip(x, np.percentile(x, 0), np.percentile(x, 90))
        y_clip = np.clip(y, np.percentile(y, 0), np.percentile(y, 90))

        xy = np.vstack([x_clip, y_clip])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x_plot, y_plot, z_plot = x_clip[idx], y_clip[idx], z[idx]


        plt.figure(figsize=(6,6))
        scatter = plt.scatter(x_plot, y_plot, c=z_plot, s=30, cmap='viridis')
        plt.plot([-30000,1],[-30000,1],'r--', label='unity line')
        plt.xlabel(f'Test ll {git_v1}')
        plt.ylabel(f'Test ll {git_v2}')
        plt.title(f'Density scatter: {model_name}')
        plt.colorbar(scatter, label='Point density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path_comparison,
                                f'test_ll_density_{model_name}_{git_v1}_vs_{git_v2}.png'), dpi=300)
        plt.close()
        
        plt.figure(figsize=(6,6))
        scatter = plt.scatter(x_plot, y_plot, c=z_plot, s=30, cmap='viridis')
        plt.plot([-30000,1],[-30000,1],'r--', label='unity line')
        plt.xlabel(f'Test ll {git_v1}')
        plt.ylabel(f'Test ll {git_v2}')
        plt.title(f'Density scatter: {model_name}')
        plt.colorbar(scatter, label='Point density')
        plt.legend()
        plt.xlim(-5000, 1)
        plt.ylim(-5000, 1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path_comparison,
                                f'test_ll_density_zoom_{model_name}_{git_v1}_vs_{git_v2}.png'), dpi=300)
        plt.close()
        

        ll1 = df_compare[f"test_ll_{git_v1}"].values
        ll2 = df_compare[f"test_ll_{git_v2}"].values

        # Valid values
        mask = np.isfinite(ll1) & np.isfinite(ll2)
        ll1, ll2 = ll1[mask], ll2[mask]

        # Compute delta
        delta_ll = ll1 - ll2

        plt.figure(figsize=(7,5))
        plt.hist(delta_ll, bins=60, alpha=0.8)
        plt.axvline(0, color='red', linestyle='--', label="Equal performance")
        plt.xlabel(f"ΔLL = LL({git_v1}) - LL({git_v2})")
        plt.ylabel("Count")
        plt.title(f"Δ Log-Likelihood Histogram — {model_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_path_comparison, f"delta_ll_hist_{model_name}_{git_v1}_vs_{git_v2}.png"),
            dpi=300
        )
        plt.close()

            
        area_groups = allen.get_custom_area_groups()
        area_colors = allen.get_custom_area_groups_colors()
        lrt_merged = build_lrt_merged(df_models, git_v1, git_v2)

        compare_lrt_between_versions(
            lrt_merged,
            output_path_comparison,
            git_v1,
            git_v2,
            area_groups,
            area_colors
        )


def over_mouse_compare_git_results(nwb_list, plots,info_path, output_path, git_versions, day_to_analyze = 0):

    # Load and combine NWB files
    trial_table, unit_table, ephys_nwb_list = combine_ephys_nwb(nwb_list, day_to_analyze=day_to_analyze, max_workers=20, git_version =git_versions[0])

    mice = unit_table['mouse_id'].unique()
    df_models = load_models_multiple_versions(mice, output_path, git_versions)
    # Add a safety check before using df_models
    if df_models.empty:
        print("[CRITICAL] No model data loaded. Check the error messages above.")

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





    # Parse JSON arrays for all models
    df_models['predictors'] = Parallel(n_jobs=-1, batch_size=1000)(
        delayed(parse_json_array)(s) for s in df_models['predictors']
    )
    df_models['y_test_array'] = Parallel(n_jobs=-1, batch_size=1000)(
        delayed(parse_json_array)(s) for s in df_models['y_test']
    )

    df_models['y_pred_array'] = Parallel(n_jobs=-1, batch_size=1000)(
        delayed(parse_json_array)(s) for s in df_models['y_pred']
    )

    area_groups = allen.get_custom_area_groups()
    area_colors = allen.get_custom_area_groups_colors()
    merged_df = pd.merge(df_models, unit_table, on=['mouse_id', 'neuron_id'], how='left')
    merged_df = allen.create_area_custom_column(merged_df)
    # Keep only neuron_id + mouse_id pairs that appear in both git versions AND both model types

    required = (
        df_models
        .groupby(['mouse_id', 'neuron_id', 'model_name'])['git_version']
        .nunique()
        .reset_index()
    )

    # We want only model_type="full" (or all types if needed)
    required_full = required[required['model_name'] == 'full']

    # Need at least 2 git versions for this neuron
    valid_pairs = required_full[required_full['git_version'] == len(git_versions)][['mouse_id', 'neuron_id']]

    # Filter df_models (or merged_df) using inner merge
    df_models = df_models.merge(valid_pairs, on=['mouse_id', 'neuron_id'], how='inner')

    # Example: select the two git versions to compare
    git_v1, git_v2 = git_versions[:2]

    df_v1 = merged_df[(merged_df['git_version'] == git_v1)]
    df_v2 = merged_df[(merged_df['git_version'] == git_v2) ]

    # Merge by neuron
    df_compare = pd.merge(
        df_v1[['mouse_id','neuron_id','train_corr','test_corr']],
        df_v2[['mouse_id','neuron_id','train_corr','test_corr']],
        on=['mouse_id','neuron_id'],
        suffixes=(f'_{git_v1}', f'_{git_v2}')
    )

    # Compute differences
    df_compare['train_corr_diff'] = df_compare[f'train_corr_{git_v1}'] - df_compare[f'train_corr_{git_v2}']
    df_compare['test_corr_diff'] = df_compare[f'test_corr_{git_v1}'] - df_compare[f'test_corr_{git_v2}']

    comparison_folder_name = f'comparison_{git_v1}_{git_v2}'
    output_path_comparison = os.path.join(output_path, 'unit_glm', comparison_folder_name)
    os.makedirs(output_path_comparison, exist_ok=True)


    if 'metrics' in plots :

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        # Loop over all model_names
        for model_name in merged_df['model_name'].unique():

            # Select only this model_name and full type for each git version
            df_v1 = merged_df[(merged_df['git_version'] == git_v1) &
                            (merged_df['model_name'] == model_name)]
            df_v2 = merged_df[(merged_df['git_version'] == git_v2) &
                            (merged_df['model_name'] == model_name)]

            if df_v1.empty or df_v2.empty:
                print(f"[INFO] No data for model {model_name}, skipping plot.")
                continue

            # Merge by neuron
            df_compare = pd.merge(
                df_v1[['mouse_id','neuron_id','train_corr','test_corr']],
                df_v2[['mouse_id','neuron_id','train_corr','test_corr']],
                on=['mouse_id','neuron_id'],
                suffixes=(f'_{git_v1}', f'_{git_v2}')
            )

            # Compute differences
            df_compare['train_corr_diff'] = df_compare[f'train_corr_{git_v1}'] - df_compare[f'train_corr_{git_v2}']
            df_compare['test_corr_diff'] = df_compare[f'test_corr_{git_v1}'] - df_compare[f'test_corr_{git_v2}']

            # --- Density scatter plot: test_corr git_v1 vs git_v2 ---
            x = df_compare[f'test_corr_{git_v1}'].values
            y = df_compare[f'test_corr_{git_v2}'].values
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            x, y, z = compute_density_safe(x, y)

            plt.figure(figsize=(6,6))
            scatter = plt.scatter(x, y, c=z, s=30, cmap='viridis', edgecolor='none')
            plt.plot([0,1],[0,1],'r--', label='unity line')
            plt.xlabel(f'Test corr {git_v1}')
            plt.ylabel(f'Test corr {git_v2}')
            plt.title(f'Density scatter: {model_name}')
            plt.colorbar(scatter, label='Point density')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path_comparison,
                                    f'test_corr_density_{model_name}_{git_v1}_vs_{git_v2}.png'), dpi=300)
            plt.close()
        
        # --- Distribution of train-test difference for each git version ---
        df_compare['train_test_diff_' + git_v1] = df_compare[f'train_corr_{git_v1}'] - df_compare[f'test_corr_{git_v1}']
        df_compare['train_test_diff_' + git_v2] = df_compare[f'train_corr_{git_v2}'] - df_compare[f'test_corr_{git_v2}']

        plt.figure(figsize=(7,5))
        sns.histplot(df_compare[f'train_test_diff_{git_v1}'], bins=30, color='skyblue', label=git_v1, kde=True)
        sns.histplot(df_compare[f'train_test_diff_{git_v2}'], bins=30, color='orange', label=git_v2, kde=True)
        plt.xlabel('Train - Test correlation')
        plt.ylabel('Neuron count')
        plt.title('Distribution of train-test correlation difference')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path_comparison, f'train_test_diff_distribution_{git_v1}_vs_{git_v2}.png'), dpi=300)
        plt.close()

        


    if 'predictions' in plots:
 
        output_folder = os.path.join(output_path_comparison, 'indiv_trial')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        results_df1 = merged_df[(merged_df['git_version'] == git_v1) &(merged_df['model_name'] == 'full') &(merged_df['fold'] == '0') ]
        results_df2 = merged_df[(merged_df['git_version'] == git_v2) &(merged_df['model_name'] == 'full') &(merged_df['fold'] == '0')]
        for neuron_id in results_df2['neuron_id'].unique():
            # plot_trial_grid_predictions_two_models(results_df1, results_df2, trial_table, neuron_id, 0.1,output_folder, model_names=git_versions)
            plot_trial_grid_predictions_two_models(results_df1, results_df2, trial_table, neuron_id,
                                           (0.1, 0.1), output_folder, model_names=git_versions)


        # output_folder = os.path.join(output_path_comparison, 'average_predictions_per_trial_types')
        # if not os.path.exists(output_folder):
        #     os.makedirs(output_folder)
        # results_df1 = merged_df[(merged_df['git_version'] == git_v1) &(merged_df['model_name'] == 'full')]
        # results_df2 = merged_df[(merged_df['git_version'] == git_v2) &(merged_df['model_name'] == 'full' )]
        # plot_predictions_with_reduced_models_parallel(results_df1, results_df2, trial_table, 'Normal', output_folder, bin_sizes = (0.05, 0.1))




def get_prediction_array_bu(row):
    """
    Return the prediction array for a row, handling both y_pred and y_test_pred columns.
    If both are missing, return empty list.
    """
    val = row.get('y_pred') or row.get('y_test_pred')
    if isinstance(val, str):
        return parse_json_array(val)
    else:
        return []
    
def get_prediction_array(row):
    """Select y_pred (if available) or y_test_pred, safely."""
    
    # Try y_pred first
    val = row.get('y_pred')
    
    # If NaN float from merge → switch to y_test_pred
    if pd.isna(val):
        val = row.get('y_test_pred')

    # If still NaN or missing → no prediction available
    if pd.isna(val) or not isinstance(val, str):
        return None

    # Clean invalid strings
    val_clean = val.strip().lower()
    if val_clean in ("nan", "none", "", "null"):
        return None

    # Try parsing normally
    try:
        return json.loads(val)
    except Exception:
        pass
    
    # Try literal_eval fallback
    try:
        return ast.literal_eval(val)
    except Exception:
        return None


def plot_kde_full_vs_reduced(df,output_folder):
    """
    Plot KDEs of test correlations for full and all reduced models.

    :param df: pd.DataFrame with columns ['model_type', 'model_name', 'test_corr']
    :param title: str, figure title
    :param ax: matplotlib.axes.Axes or None
    """
    

    # Plot reduced models
    df_reduced = df[df['model_name'] != 'full']
    reduced_model_names = df_reduced['model_name'].unique()
    colors = sns.color_palette("husl", len(reduced_model_names))

    for color, model_name in zip(colors, reduced_model_names):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

        sub_df = df_reduced[df_reduced['model_name'] == model_name]
        if not sub_df.empty:
            sns.kdeplot(sub_df['test_corr'], ax=ax, color=color, linewidth=1.5,
                        label=f'Test (mean={sub_df["test_corr"].mean():.2f})')

        sns.kdeplot(sub_df['train_corr'], ax=ax, color='black', linewidth=2,
                    label=f'Train (mean={sub_df["train_corr"].mean():.2f})')

        ax.set_xlabel('Test Score')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.tight_layout()
        ax.grid(False, linestyle='--', alpha=0.4)
        ax.set_title(f'Kde_train_vs_test_{model_name}')
        putils.save_figure_with_options(fig, file_formats=[ 'png', 'pdf', 'svg'], filename= f'Kde_test_train_{model_name}', output_dir=output_folder)

    return

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def compare_full_vs_reduced_models(df_models, output_folder):
    """
    Compare FULL model to all reduced models using:
        - Density scatter of test_corr (full vs reduced)
        - Histogram of ΔLL = full - reduced

    Parameters
    ----------
    df_models : DataFrame
        Must contain columns:
            ['mouse_id','neuron_id','model_name','test_corr','test_ll']
    output_folder : str or Path
        Where to save comparison plots

    Produces
    --------
    - test_corr_density_full_vs_{reduced}.png
    - delta_ll_full_vs_{reduced}.png
    """

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Identify reduced models
    all_models = sorted(df_models["model_name"].unique())
    reduced_models = [m for m in all_models if m != "full"]

    # Base full model
    df_full = df_models[df_models["model_name"] == "full"]

    if df_full.empty:
        raise ValueError("No rows found for model_name == 'full'")

    for red_model in reduced_models:
        print(f"🔍 Comparing FULL vs {red_model}")

        df_red = df_models[df_models["model_name"] == red_model]
        if df_red.empty:
            print(f"⚠️ No rows for reduced model {red_model}, skipping.")
            continue

        # Merge neuron-by-neuron
        df_compare = pd.merge(
            df_full[['mouse_id','neuron_id','test_corr','reward_group']].rename(
                columns={'test_corr': 'test_corr_full'}
            ),
            df_red[['mouse_id','neuron_id','test_corr']].rename(
                columns={'test_corr': f'test_corr_{red_model}'}
            ),
            on=['mouse_id','neuron_id']
        )

        if df_compare.empty:
            print(f"⚠️ No overlapping neurons for model {red_model}, skipping.")
            continue

        # Δ metrics
        df_compare["delta_corr"] = df_compare["test_corr_full"] - df_compare[f"test_corr_{red_model}"]
        # df_compare["delta_ll"]   = df_compare["test_ll_full"]   - df_compare[f"test_ll_{red_model}"]

        # Extract for plotting
        x = df_compare["test_corr_full"].values
        y = df_compare[f"test_corr_{red_model}"].values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        # Density
        x, y, z = compute_density_safe(x, y)

        # ======================
        #  Density scatter plot
        # ======================
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)

        sc = ax.scatter(x, y, c=z, cmap='viridis', s=20, edgecolor='none')

        # Attach colorbar to figure explicitly
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Density")

        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, 'r--', label="unity line")

        ax.set_xlabel("Full model: test_corr")
        ax.set_ylabel(f"{red_model}: test_corr")
        ax.set_title(f"Density scatter: FULL vs {red_model}")
        ax.legend()
        ax.grid(linestyle="--", alpha=0.3)

        # Zoom inset around the main data cluster
        axins = ax.inset_axes([0.48, 0.05, 0.50, 0.50])
        axins.scatter(x, y, c=z, cmap='viridis', s=5, edgecolor='none')
        _zoom = [0, min(0.20, float(np.percentile(np.concatenate([x, y]), 75)))]
        axins.plot(_zoom, _zoom, 'r--', linewidth=0.8)
        axins.set_xlim(_zoom)
        axins.set_ylim(_zoom)
        axins.tick_params(labelsize=7)
        axins.grid(linestyle='--', alpha=0.3)
        ax.indicate_inset_zoom(axins, edgecolor='gray', linewidth=0.8)

        plt.tight_layout()
        putils.save_figure_with_options(
            fig,
            file_formats=["png", "pdf", "svg"],
            filename=f"test_corr_density_full_vs_{red_model}",
            output_dir=output_folder,
            dark_background=True
        )

        # ======================
        #  Delta distribution
        # ======================
        from scipy.stats import wilcoxon as _wilcoxon
        delta = df_compare["delta_corr"].dropna().values
        delta = delta[np.isfinite(delta)]
        if len(delta) >= 10:
            fig_d, ax_d = plt.subplots(figsize=(7, 5))

            _RCOLORS = {1: 'forestgreen', 0: 'crimson'}
            _RLABELS = {1: 'R+', 0: 'R−'}
            bins = np.linspace(delta.min(), delta.max(), 61)
            _annot_lines = []
            for rg in [1, 0]:
                mask_rg = df_compare['reward_group'].values[np.isfinite(df_compare['delta_corr'].values)] == rg
                d_rg = delta[mask_rg]
                if len(d_rg) < 5:
                    continue
                ax_d.hist(d_rg, bins=bins, histtype='step',
                          edgecolor=_RCOLORS[rg], linewidth=2.2,
                          label=f'{_RLABELS[rg]} (n={len(d_rg)})')
                _, p_rg = _wilcoxon(d_rg)
                pct_rg = 100 * np.mean(d_rg > 0)
                p_str = f'p = {p_rg:.2e}'
                _annot_lines.append(f'{_RLABELS[rg]}: {p_str},  {pct_rg:.0f}% improved')

            ax_d.axvline(0, color='black', linewidth=2, linestyle='--')
            ax_d.set_xlabel('Δ test correlation  (full − reduced)', fontsize=16)
            ax_d.set_ylabel('Neuron count', fontsize=16)
            ax_d.set_title(f'Full vs {red_model}', fontsize=17, fontweight='bold')
            ax_d.tick_params(axis='both', labelsize=14)
            ax_d.legend(fontsize=13, frameon=False)
            ax_d.text(0.97, 0.97, '\n'.join(_annot_lines),
                      transform=ax_d.transAxes, fontsize=12,
                      va='top', ha='right',
                      bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9))
            ax_d.spines['top'].set_visible(False)
            ax_d.spines['right'].set_visible(False)
            ax_d.spines['left'].set_linewidth(1.5)
            ax_d.spines['bottom'].set_linewidth(1.5)
            plt.tight_layout()
            putils.save_figure_with_options(
                fig_d,
                file_formats=["png", "pdf"],
                filename=f"delta_test_corr_full_vs_{red_model}",
                output_dir=output_folder,
            )
            plt.close(fig_d)

            # CDF of delta
            fig_c, ax_c = plt.subplots(figsize=(7, 5))
            for rg in [1, 0]:
                mask_rg = df_compare['reward_group'].values[np.isfinite(df_compare['delta_corr'].values)] == rg
                d_rg = delta[mask_rg]
                if len(d_rg) < 5:
                    continue
                sorted_d = np.sort(d_rg)
                cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
                ax_c.plot(sorted_d, cdf, color=_RCOLORS[rg], linewidth=2.2,
                          label=f'{_RLABELS[rg]} (n={len(d_rg)})')
            ax_c.axvline(0, color='black', linewidth=2, linestyle='--')
            ax_c.axhline(0.5, color='gray', linewidth=1, linestyle=':', alpha=0.6)
            ax_c.set_xlabel('Δ test correlation  (full − reduced)', fontsize=16)
            ax_c.set_ylabel('Cumulative fraction', fontsize=16)
            ax_c.set_title(f'Full vs {red_model}', fontsize=17, fontweight='bold')
            ax_c.tick_params(axis='both', labelsize=14)
            ax_c.legend(fontsize=13, frameon=False)
            ax_c.spines['top'].set_visible(False)
            ax_c.spines['right'].set_visible(False)
            ax_c.spines['left'].set_linewidth(1.5)
            ax_c.spines['bottom'].set_linewidth(1.5)
            plt.tight_layout()
            putils.save_figure_with_options(
                fig_c,
                file_formats=["png", "pdf"],
                filename=f"delta_test_corr_cdf_full_vs_{red_model}",
                output_dir=output_folder,
            )
            plt.close(fig_c)

 # Merge neuron-by-neuron
        df_compare = pd.merge(
            df_full[['mouse_id','neuron_id','explained_var']].rename(
                columns={'explained_var': 'test_corr_full'
                        }
            ),
            df_red[['mouse_id','neuron_id','explained_var']].rename(
                columns={'explained_var': f'test_corr_{red_model}'
                }
            ),
            on=['mouse_id','neuron_id']
        )
        print(df_full.keys())
        if df_compare.empty:
            print(f"⚠️ No overlapping neurons for model {red_model}, skipping.")
            continue

        # Δ metrics
        df_compare["delta_corr"] = df_compare["test_corr_full"] - df_compare[f"test_corr_{red_model}"]
        # df_compare["delta_ll"]   = df_compare["test_ll_full"]   - df_compare[f"test_ll_{red_model}"]

        # Extract for plotting
        x = df_compare["test_corr_full"].values
        y = df_compare[f"test_corr_{red_model}"].values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        # Density
        x, y, z = compute_density_safe(x, y)

        # ======================
        #  Density scatter plot
        # ======================
        fig, ax2 = plt.subplots(figsize=(6, 6))
        sc = ax2.scatter(x, y, c=z, cmap='viridis', s=20, edgecolor='none')

        # Add colorbar
        cbar = fig.colorbar(sc, ax=ax2)
        cbar.set_label("Density", rotation=90)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax2.plot(lims, lims, 'r--', label="unity line")
        ax2.set_xlim(-0.2, 0.5)
        ax2.set_ylim(-0.2, 0.5)
        ax2.set_xlabel("Full model: explained variance")
        ax2.set_ylabel(f"{red_model}: explained variance")
        ax2.set_title(f"Density scatter: FULL vs {red_model}")
        ax2.legend()
        ax2.grid(linestyle="--", alpha=0.3)

        # Zoom inset
        axins2 = ax2.inset_axes([0.55, 0.05, 0.42, 0.42])
        axins2.scatter(x, y, c=z, cmap='viridis', s=5, edgecolor='none')
        _zoom2 = [0, min(0.35, float(np.percentile(np.concatenate([x, y]), 95)))]
        axins2.plot(_zoom2, _zoom2, 'r--', linewidth=0.8)
        axins2.set_xlim(_zoom2)
        axins2.set_ylim(_zoom2)
        axins2.tick_params(labelsize=7)
        axins2.grid(linestyle='--', alpha=0.3)
        ax2.indicate_inset_zoom(axins2, edgecolor='gray', linewidth=0.8)

        plt.tight_layout()
        putils.save_figure_with_options(
        fig,
        file_formats=["png", "pdf", "svg"],
        filename= f"test_variance_explained_density_full_vs_{red_model}",
        output_dir=output_folder,
        dark_background=True
        )
        plt.close()





import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import wilcoxon

def plot_box_full_vs_reduced(df, output_folder, alpha=0.05):
    """
    Plot boxplots of test correlations for full and reduced models,
    run paired Wilcoxon tests (per neuron) between full and each reduced model,
    and mark reduced models that perform significantly worse than full.
    """
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # --- Split full vs reduced ---
    df_full = df[df["model_name"] == "full"]
    df_reduced = df[df["model_name"]  != "full"]

    # Get list of reduced models in order of appearance
    reduced_model_names = sorted(df_reduced["model_name"].unique())
    ordered_models = ["full"] + reduced_model_names

    # --- Create column for display ---
    df_plot = df.copy()
    df_plot["model_display"] = np.where(df_plot["model_name"] == "full", "full", df_plot["model_name"])

    # --- Boxplot ---
    sns.boxplot(
        data=df_plot,
        x="model_display",
        y="test_corr",
        order=ordered_models,
        palette="husl",
        width=0.6,
        fliersize=2,
        ax=ax
    )

    # --- Styling ---
    ax.set_xlabel("")
    ax.set_ylabel("Test correlation")
    ax.set_title("Full vs Reduced Models (paired Wilcoxon)")
    ax.grid(False, linestyle="--", alpha=0.4)

    # 🔧 Fix label overlap
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    plt.tight_layout(pad=1.5)

    # --- Save ---
    putils.save_figure_with_options(
        fig,
        file_formats=["png", 'pdf'],
        filename="Box_full_vs_reduced_significance",
        output_dir=output_folder,
        dark_background=True
    )

    plt.close(fig)


def plot_kde_per_trial_type(merged, trial_table, output_folder, time_stim=0.0):
    trialtype_corrs = compute_trialtype_correlations(merged, trials_df=trial_table)
    # Define a consistent color scheme
    color_map = {
        "whisker_hit": "forestgreen",
        "whisker_miss": "crimson",
        "auditory_hit": "mediumblue",
        "auditory_miss": "skyblue",
        "catch": "gray",
        "correct_rejection": "black"
    }

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    for trial_type, grp in trialtype_corrs.groupby("trial_type"):
        color = color_map.get(trial_type, "gray")  # fallback color
        grp["test_corr"].plot(kind="kde", lw=2, label=f"{trial_type}", ax=ax, color=color)



    ax.set_xlabel('Test Score')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.set_title("KDE of test correlation by trial type")
    ax.grid(False, linestyle='--', alpha=0.4)

    putils.save_figure_with_options(
        fig,
        file_formats=['png', 'pdf'],
        filename='Kde_per_trial_type_full_model',
        output_dir=output_folder,
        dark_background= True
    )

    plt.close(fig)
    return

def plot_box_per_trial_type(merged, trial_table, output_folder, time_stim=0.0):
    trialtype_corrs = compute_trialtype_correlations(merged, trials_df=trial_table)
    color_map = {
        "whisker_hit": "forestgreen",
        "whisker_miss": "crimson",
        "auditory_hit": "mediumblue",
        "auditory_miss": "skyblue",
        "catch": "gray",
        "correct_rejection": "black"
    }

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ordered_types = [t for t in color_map.keys() if t in trialtype_corrs["trial_type"].unique()]
    box_colors = [color_map[t] for t in ordered_types]

    data = [trialtype_corrs.loc[trialtype_corrs["trial_type"] == t, "test_corr"] for t in ordered_types]
    data_clean = [d.dropna().values for d in data]
    bp = ax.boxplot(data_clean, patch_artist=True, tick_labels=ordered_types)

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel("Test Score")
    ax.set_title("Boxplot of test correlation by trial type")
    ax.grid(False, linestyle="--", alpha=0.4)
    plt.xticks(rotation=45)
    for median in bp['medians']:
        median.set_color('black')
    putils.save_figure_with_options(
        fig,
        file_formats=["png", 'pdf'],
        filename="Box_per_trial_type_full_model",
        output_dir=output_folder,
        dark_background=True
    )

    plt.close(fig)
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
            neuron_means = mouse_grp.groupby("neuron_id")["test_corr"].mean().values
            neuron_means = np.nan_to_num(neuron_means, nan=0.0)
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
            values = grp['test_corr'].values
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

def label_by_last_whisker_outcome(trials_df):
    """
    Label each trial based on whether the last *whisker trial* before it
    was a hit or miss.

    Adds a new column:
        last_whisker_outcome ∈ {"last_hit", "last_miss", np.nan}
    """
    df = trials_df.copy()
    df["last_whisker_outcome"] = np.nan

    last_outcome = "last_miss"
    for i, row in df.iterrows():
        behav_type = row["behav_type"]

        # Assign current label based on most recent whisker outcome
        if last_outcome is not None:
            df.at[i, "last_whisker_outcome"] = last_outcome

        # Update last_outcome if this trial is a whisker trial
        if behav_type == "whisker_hit":
            last_outcome = "last_hit"
        elif behav_type == "whisker_miss":
            last_outcome = "last_miss"

    return df

def plot_by_last_whisker_outcome(
    neuron_ids, df_full, df_reduced, trials_df, output_folder, name,
    reduced_model="whisker_encoding", bin_size=0.1, zscore=False
):
    """
    Compare model fits for trials grouped by last whisker outcome:
    - Row 0: last whisker was hit
    - Row 1: last whisker was miss
    Columns: trial types
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import zscore as zscore_f
    import ast
    import os

    os.makedirs(output_folder, exist_ok=True)
    trials_df = label_by_last_whisker_outcome(trials_df)

    last_outcomes = ["last_hit", "last_miss"]
    trial_types = sorted(trials_df["behav_type"].unique())

    # Containers
    all_y_test = {lo: {tt: [] for tt in trial_types} for lo in last_outcomes}
    all_y_pred_full = {lo: {tt: [] for tt in trial_types} for lo in last_outcomes}
    all_y_pred_reduced = {lo: {tt: [] for tt in trial_types} for lo in last_outcomes}

    for nid in neuron_ids:
        full_rows = df_full[df_full["neuron_id"] == nid]
        reduced_rows = df_reduced[df_reduced["neuron_id"] == nid]

        for res in full_rows.itertuples(index=False):
            y_test = res.y_test_array
            y_pred = res.y_pred_array
            n_bins = res.n_bins
            n_trials = y_pred.shape[0] // n_bins

            y_test = y_test.reshape(n_trials, n_bins)
            y_pred = y_pred.reshape(n_trials, n_bins)
            test_trial_ids = np.array(ast.literal_eval(res.test_trials))
            order = np.argsort(test_trial_ids)
            y_test, y_pred = y_test[order], y_pred[order]
            trials_test_df = trials_df.iloc[test_trial_ids[order]]

            for lo in last_outcomes:
                idx_lo = np.where(trials_test_df["last_whisker_outcome"] == lo)[0]
                for tt in trial_types:
                    idx_tt = np.where(trials_test_df["behav_type"] == tt)[0]
                    idx = np.intersect1d(idx_lo, idx_tt)
                    if len(idx) == 0:
                        continue
                    test_mean = y_test[idx].mean(axis=0)
                    pred_mean = y_pred[idx].mean(axis=0)
                    if zscore:
                        test_mean = zscore_f(test_mean)
                        pred_mean = zscore_f(pred_mean)
                    all_y_test[lo][tt].append(test_mean)
                    all_y_pred_full[lo][tt].append(pred_mean)

        for res in reduced_rows.itertuples(index=False):
            y_pred = res.y_pred_array
            n_bins = res.n_bins
            n_trials = y_pred.shape[0] // n_bins
            y_pred = y_pred.reshape(n_trials, n_bins)
            test_trial_ids = np.array(ast.literal_eval(res.test_trials))
            order = np.argsort(test_trial_ids)
            y_pred = y_pred[order]
            trials_test_df = trials_df.iloc[test_trial_ids[order]]

            for lo in last_outcomes:
                idx_lo = np.where(trials_test_df["last_whisker_outcome"] == lo)[0]
                for tt in trial_types:
                    idx_tt = np.where(trials_test_df["behav_type"] == tt)[0]
                    idx = np.intersect1d(idx_lo, idx_tt)
                    if len(idx) == 0:
                        continue
                    pred_mean = y_pred[idx].mean(axis=0)
                    if zscore:
                        pred_mean = zscore_f(pred_mean)
                    all_y_pred_reduced[lo][tt].append(pred_mean)

    # ------------------------ PLOTTING ------------------------
    plt.ioff()
    n_rows = len(last_outcomes)
    n_cols = len(trial_types)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), sharey=True)
    time = np.linspace(-1 + bin_size/2, 2 - bin_size/2, n_bins)
    colors = {"full": "green", "reduced": "red", "data": "black"}

    for r, lo in enumerate(last_outcomes):
        for c, tt in enumerate(trial_types):
            ax = axes[r, c] if n_rows > 1 and n_cols > 1 else axes[max(r,c)]
            ax.set_title(tt if r == 0 else "")
            y_data = all_y_test[lo][tt]
            y_full = all_y_pred_full[lo][tt]
            y_reduced = all_y_pred_reduced[lo][tt]

            if len(y_data) == 0:
                ax.text(0.5, 0.5, "No trials", ha="center", va="center")
                continue

            # Data
            y_data = np.stack(y_data)
            m_data = y_data.mean(axis=0)
            s_data = y_data.std(axis=0, ddof=1) / np.sqrt(y_data.shape[0])
            ax.plot(time, m_data, color=colors["data"], label="data")
            ax.fill_between(time, m_data - s_data, m_data + s_data, color=colors["data"], alpha=0.3)

            # Full
            y_full = np.stack(y_full)
            m_full = y_full.mean(axis=0)
            s_full = y_full.std(axis=0, ddof=1) / np.sqrt(y_full.shape[0])
            ax.plot(time, m_full, color=colors["full"], label="full")
            ax.fill_between(time, m_full - s_full, m_full + s_full, color=colors["full"], alpha=0.3)

            # Reduced
            y_reduced = np.stack(y_reduced)
            m_red = y_reduced.mean(axis=0)
            s_red = y_reduced.std(axis=0, ddof=1) / np.sqrt(y_reduced.shape[0])
            ax.plot(time, m_red, color=colors["reduced"], label="reduced")
            ax.fill_between(time, m_red - s_red, m_red + s_red, color=colors["reduced"], alpha=0.3)
            if c == 0:
                # Set row label (leftmost column)
                ax.set_ylabel("Last whisker: Hit" if lo == "last_hit" else "Last whisker: Miss", fontsize=10)
            ax.axvline(0, color="gray", linestyle="--")
            if r == n_rows-1:
                ax.set_xlabel("Time (s)")
            if r == 0 and c == n_cols-1:
                ax.legend(fontsize=8)

    fig.suptitle(
        f"Reduced model {reduced_model}, neuron {neuron_ids[0]}\n"
        f"full fit={df_full['test_corr'].mean():.3f}, reduced fit={df_reduced['test_corr'].mean():.3f}"
    )
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{name}_by_last_whisker_outcome.png", dpi=300)
    plt.close(fig)

def plot_by_session_quartiles(
    neuron_ids, df_full, df_reduced, trials_df, output_folder, name,
    reduced_model="whisker_encoding", bin_size=0.1, zscore=False
):
    """
    Compare model fits across session halves (early vs. late trials).
    Rows: halves (1st, 2nd)
    Columns: trial types (e.g., whisker, no-stim, etc.)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from scipy.stats import zscore as zscore_f
    import ast, os

    os.makedirs(output_folder, exist_ok=True)

    # ------------------------------
    # Assign session halves
    # ------------------------------
    df = trials_df.copy().reset_index(drop=True)
    n_trials = len(df)
    df["quartile"] = pd.qcut(np.arange(n_trials), 2, labels=["H1", "H2"])

    halves = ["H1", "H2"]
    half_labels = {"H1": "First half", "H2": "Second half"}
    trial_types = sorted(df["behav_type"].unique())

    # Get area acronym for the primary neuron
    neuron_row = df_full[df_full["neuron_id"] == neuron_ids[0]]
    area_acronym = (
        neuron_row["area_acronym_custom"].iloc[0]
        if len(neuron_row) > 0 and "area_acronym_custom" in neuron_row.columns
        else ""
    )

    # ------------------------------
    # Compute per-trial-type & half correlations
    # ------------------------------
    trialtype_q_corrs_full = compute_trialtype_quartile_correlations(
        df_full[df_full["neuron_id"].isin(neuron_ids)], df
    )
    trialtype_q_corrs_reduced = compute_trialtype_quartile_correlations(
        df_reduced[df_reduced["neuron_id"].isin(neuron_ids)], df
    )

    # Aggregate per half x trial type
    corr_full = (
        trialtype_q_corrs_full.groupby(["quartile", "trial_type"])["test_corr"]
        .mean().to_dict()
    )
    corr_reduced = (
        trialtype_q_corrs_reduced.groupby(["quartile", "trial_type"])["test_corr"]
        .mean().to_dict()
    )

    # Storage
    all_y_test = {h: {tt: [] for tt in trial_types} for h in halves}
    all_y_pred_full = {h: {tt: [] for tt in trial_types} for h in halves}
    all_y_pred_reduced = {h: {tt: [] for tt in trial_types} for h in halves}

    # ------------------------------
    # Gather model data
    # ------------------------------
    for nid in neuron_ids:
        full_rows = df_full[df_full["neuron_id"] == nid]
        reduced_rows = df_reduced[df_reduced["neuron_id"] == nid]

        for res in full_rows.itertuples(index=False):
            y_test = res.y_test_array
            y_pred = res.y_pred_array
            n_bins = res.n_bins
            n_trials = y_pred.shape[0] // n_bins

            y_test = y_test.reshape(n_trials, n_bins)
            y_pred = y_pred.reshape(n_trials, n_bins)

            test_trial_ids = np.array(ast.literal_eval(res.test_trials))
            order = np.argsort(test_trial_ids)
            y_test, y_pred = y_test[order], y_pred[order]
            trials_test_df = df.iloc[test_trial_ids[order]]

            for h in halves:
                for tt in trial_types:
                    idx = np.where(
                        (trials_test_df["quartile"] == h) &
                        (trials_test_df["behav_type"] == tt)
                    )[0]
                    if len(idx) == 0:
                        continue
                    test_mean = y_test[idx].mean(axis=0)
                    pred_mean = y_pred[idx].mean(axis=0)
                    if zscore:
                        test_mean = zscore_f(test_mean)
                        pred_mean = zscore_f(pred_mean)
                    all_y_test[h][tt].append(test_mean)
                    all_y_pred_full[h][tt].append(pred_mean)

        for res in reduced_rows.itertuples(index=False):
            y_pred = res.y_pred_array
            n_bins = res.n_bins
            n_trials = y_pred.shape[0] // n_bins
            y_pred = y_pred.reshape(n_trials, n_bins)

            test_trial_ids = np.array(ast.literal_eval(res.test_trials))
            order = np.argsort(test_trial_ids)
            y_pred = y_pred[order]
            trials_test_df = df.iloc[test_trial_ids[order]]

            for h in halves:
                for tt in trial_types:
                    idx = np.where(
                        (trials_test_df["quartile"] == h) &
                        (trials_test_df["behav_type"] == tt)
                    )[0]
                    if len(idx) == 0:
                        continue
                    pred_mean = y_pred[idx].mean(axis=0)
                    if zscore:
                        pred_mean = zscore_f(pred_mean)
                    all_y_pred_reduced[h][tt].append(pred_mean)

    # ------------------------------
    # Publication-quality plotting
    # ------------------------------
    plt.ioff()

    FONT_SIZE_LABEL    = 9
    FONT_SIZE_TICK     = 8
    FONT_SIZE_TITLE    = 9
    FONT_SIZE_SUPTITLE = 10
    LW         = 1.2
    ALPHA_FILL = 0.25

    colors = {
        "data":    "#2c2c2c",
        "full":    "#2166ac",
        "reduced": "#d6604d",
    }
    line_labels = {
        "data":    "Data",
        "full":    "Full model",
        "reduced": reduced_model.replace("_", " ").title(),
    }

    n_rows, n_cols = len(halves), len(trial_types)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.8 * n_cols, 2.4 * n_rows),
        sharey=True, sharex=True,
        squeeze=False,
    )
    time = np.linspace(-1 + bin_size / 2, 2 - bin_size / 2, n_bins)

    for r, h in enumerate(halves):
        for c, tt in enumerate(trial_types):
            ax = axes[r, c]
            corr_f = corr_full.get((h, tt), np.nan)
            corr_r = corr_reduced.get((h, tt), np.nan)
            y_data = all_y_test[h][tt]
            y_full = all_y_pred_full[h][tt]
            y_red  = all_y_pred_reduced[h][tt]

            if len(y_data) == 0:
                ax.text(0.5, 0.5, "No trials", ha="center", va="center",
                        fontsize=FONT_SIZE_TICK, transform=ax.transAxes)
            else:
                y_data = np.stack(y_data)
                y_full = np.stack(y_full)
                y_red  = np.stack(y_red)

                def _plot(y, key):
                    m = y.mean(axis=0)
                    s = y.std(axis=0, ddof=1) / np.sqrt(y.shape[0])
                    ax.plot(time, m, color=colors[key], lw=LW, label=line_labels[key])
                    ax.fill_between(time, m - s, m + s, color=colors[key], alpha=ALPHA_FILL, lw=0)

                _plot(y_data, "data")
                _plot(y_full, "full")
                _plot(y_red,  "reduced")

            # Reference lines
            ax.axvline(0, color="#888888", linestyle="--", lw=0.8, zorder=0)
            ax.axhline(0, color="#cccccc", linestyle="-",  lw=0.5, zorder=0)

            # Column header — top row only
            if r == 0:
                ax.set_title(
                    tt.replace("_", " "),
                    fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=4,
                )

            # Correlation annotation
            if not (np.isnan(corr_f) and np.isnan(corr_r)):
                ax.text(
                    0.03, 0.97,
                    f"r\u2099={corr_f:.2f}  r\u1d63={corr_r:.2f}",
                    transform=ax.transAxes, fontsize=6.5,
                    va="top", ha="left", color="#444444",
                )

            # Row label — left column only
            if c == 0:
                y_unit = "z-score" if zscore else "spikes / bin"
                ax.set_ylabel(f"{half_labels[h]}\n{y_unit}", fontsize=FONT_SIZE_LABEL)

            # x-axis label — bottom row only
            if r == n_rows - 1:
                ax.set_xlabel("Time from stimulus (s)", fontsize=FONT_SIZE_LABEL)

            # Clean spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=FONT_SIZE_TICK, length=3)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.5))

    # Shared legend at the bottom
    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels_leg,
            loc="lower center",
            ncol=3,
            fontsize=FONT_SIZE_TICK,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )

    area_str = f" [{area_acronym}]" if area_acronym else ""
    fig.suptitle(
        f"Neuron {neuron_ids[0]}{area_str}  \u00b7  {reduced_model.replace('_', ' ')} model\n"
        f"Full r={df_full['test_corr'].mean():.3f}  |  Reduced r={df_reduced['test_corr'].mean():.3f}",
        fontsize=FONT_SIZE_SUPTITLE,
        y=1.01,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(f"{output_folder}/{name}_by_session_halves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_by_recent_whisker_history(
    neuron_ids, df_full, df_reduced, trials_df, output_folder, name,
    reduced_model="whisker_encoding", bin_size=0.1, zscore=False, history_len=5
):
    """
    Compare model fits for trials grouped by recent whisker history
    (e.g., last 5 whisker trials were mostly hits vs mostly misses).
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import zscore as zscore_f
    import ast, os

    os.makedirs(output_folder, exist_ok=True)

    # ------------------------------
    # Compute recent whisker history
    # ------------------------------
    df = trials_df.copy()
    df["recent_whisker_history"] = np.nan

    whisker_outcomes = []
    for i, row in df.iterrows():
        if len(whisker_outcomes) >= history_len:
            frac_hits = np.mean([o == "hit" for o in whisker_outcomes[-history_len:]])
            if frac_hits >= 0.8:
                df.at[i, "recent_whisker_history"] = "mostly_hits"
            elif frac_hits <= 0.4:
                df.at[i, "recent_whisker_history"] = "mostly_misses"
        # update history if current trial is a whisker trial
        if "whisker" in row["behav_type"]:
            if row["behav_type"] == "whisker_hit":
                whisker_outcomes.append("hit")
            elif row["behav_type"] == "whisker_miss":
                whisker_outcomes.append("miss")

    groups = ["mostly_hits", "mostly_misses"]
    trial_types = sorted(df["behav_type"].unique())

    # Storage dicts
    all_y_test = {g: {tt: [] for tt in trial_types} for g in groups}
    all_y_pred_full = {g: {tt: [] for tt in trial_types} for g in groups}
    all_y_pred_reduced = {g: {tt: [] for tt in trial_types} for g in groups}

    # ------------------------------
    # Collect model data
    # ------------------------------
    for nid in neuron_ids:
        full_rows = df_full[df_full["neuron_id"] == nid]
        reduced_rows = df_reduced[df_reduced["neuron_id"] == nid]

        for res in full_rows.itertuples(index=False):
            y_test = res.y_test_array
            y_pred = res.y_pred_array
            n_bins = res.n_bins
            n_trials = y_pred.shape[0] // n_bins

            y_test = y_test.reshape(n_trials, n_bins)
            y_pred = y_pred.reshape(n_trials, n_bins)

            test_trial_ids = np.array(ast.literal_eval(res.test_trials))
            order = np.argsort(test_trial_ids)
            y_test, y_pred = y_test[order], y_pred[order]
            trials_test_df = df.iloc[test_trial_ids[order]]

            for g in groups:
                for tt in trial_types:
                    idx = np.where(
                        (trials_test_df["recent_whisker_history"] == g) &
                        (trials_test_df["behav_type"] == tt)
                    )[0]
                    if len(idx) == 0:
                        continue
                    test_mean = y_test[idx].mean(axis=0)
                    pred_mean = y_pred[idx].mean(axis=0)
                    if zscore:
                        test_mean = zscore_f(test_mean)
                        pred_mean = zscore_f(pred_mean)
                    all_y_test[g][tt].append(test_mean)
                    all_y_pred_full[g][tt].append(pred_mean)

        for res in reduced_rows.itertuples(index=False):
            y_pred = res.y_pred_array
            n_bins = res.n_bins
            n_trials = y_pred.shape[0] // n_bins
            y_pred = y_pred.reshape(n_trials, n_bins)

            test_trial_ids = np.array(ast.literal_eval(res.test_trials))
            order = np.argsort(test_trial_ids)
            y_pred = y_pred[order]
            trials_test_df = df.iloc[test_trial_ids[order]]

            for g in groups:
                for tt in trial_types:
                    idx = np.where(
                        (trials_test_df["recent_whisker_history"] == g) &
                        (trials_test_df["behav_type"] == tt)
                    )[0]
                    if len(idx) == 0:
                        continue
                    pred_mean = y_pred[idx].mean(axis=0)
                    if zscore:
                        pred_mean = zscore_f(pred_mean)
                    all_y_pred_reduced[g][tt].append(pred_mean)

    # ------------------------------
    # Plotting
    # ------------------------------
    plt.ioff()
    n_rows, n_cols = len(groups), len(trial_types)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.5 * n_rows), sharey=True)
    time = np.linspace(-1 + bin_size/2, 2 - bin_size/2, n_bins)
    colors = {"full": "green", "reduced": "red", "data": "black"}

    for r, g in enumerate(groups):
        for c, tt in enumerate(trial_types):
            ax = axes[r, c] if n_rows > 1 else axes[c]
            if r == 0:
                ax.set_title(tt)
            y_data = all_y_test[g][tt]
            y_full = all_y_pred_full[g][tt]
            y_red = all_y_pred_reduced[g][tt]

            if len(y_data) == 0:
                ax.text(0.5, 0.5, "No trials", ha="center", va="center")
                continue

            y_data = np.stack(y_data)
            y_full = np.stack(y_full)
            y_red = np.stack(y_red)

            def plot_with_error(y, color, label):
                m = y.mean(axis=0)
                s = y.std(axis=0, ddof=1) / np.sqrt(y.shape[0])
                ax.plot(time, m, color=color, label=label)
                ax.fill_between(time, m - s, m + s, color=color, alpha=0.3)

            plot_with_error(y_data, colors["data"], "data")
            plot_with_error(y_full, colors["full"], "full")
            plot_with_error(y_red, colors["reduced"], "reduced")

            ax.axvline(0, color="gray", linestyle="--")
            if c == 0:
                ax.set_ylabel("Recent whiskers:\nMostly hits" if g == "mostly_hits" else "Recent whiskers:\nMostly misses")
            if r == n_rows - 1:
                ax.set_xlabel("Time (s)")

    fig.suptitle(
        f"Reduced model {reduced_model}, neuron {neuron_ids[0]}\n"
        f"full fit={df_full['test_corr'].mean():.3f}, reduced fit={df_reduced['test_corr'].mean():.3f}"
    )
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{name}_by_recent_whisker_history.png", dpi=300)
    plt.close(fig)



def plot_full_vs_reduced_per_area(df, selected_reduced, area_groups, area_colors, output_folder, threshold=None):
    """
    Plot mean ± SEM test correlations per area for full and one reduced model,
    including significance stars (paired t-test across neurons) between models per area.

    :param df: pd.DataFrame with columns ['model_type','model_name','test_corr','area_acronym_custom','mouse_id','neuron_id']
    :param selected_reduced: str, reduced model name
    :param area_groups: dict, group_name -> list of area names
    :param area_colors: dict, group_name -> color
    :param output_folder: str path
    :param threshold: float or None, minimum test_corr_mean for neurons to be included
    """
    from scipy.stats import ttest_rel

    key = 'test_corr'

    # Filter data
    df_full = df[df['model_name'] == 'full'].copy()
    df_reduced = df[(df['model_name'] == selected_reduced)].copy()

    # Build ordered areas and colors
    ordered_areas = []
    area_to_color = {}
    for group_name, areas in area_groups.items():
        for area in areas:
            if area in df_full['area_acronym_custom'].values or area in df_reduced['area_acronym_custom'].values:
                ordered_areas.append(area)
                area_to_color[area] = area_colors[group_name]

    # Initialize lists
    means_full, sems_full, means_reduced, sems_reduced, bar_colors, pvals = [], [], [], [], [], []

    # Plot preparation
    fig, ax = plt.subplots(figsize=(max(12, len(ordered_areas) * 0.5), 6), dpi=300)
    x = np.arange(len(ordered_areas))
    width = 0.35

    for i, area in enumerate(ordered_areas):
        # --- Full model ---
        full_grp = df_full[df_full['area_acronym_custom'] == area]

        fold_means_full = (
            full_grp.groupby(['mouse_id', 'neuron_id'], as_index=False)[key]
            .mean()
        )

        # Apply threshold if specified
        if threshold is not None:
            neurons_to_keep = fold_means_full[fold_means_full[key] >= threshold]
        else:
            neurons_to_keep = fold_means_full

        # Keep only neurons passing threshold
        fold_means_full = fold_means_full.merge(
            neurons_to_keep[['mouse_id', 'neuron_id']],
            on=['mouse_id', 'neuron_id'],
            how='inner'
        )

        # --- Reduced model ---
        reduced_grp = df_reduced[df_reduced['area_acronym_custom'] == area]
        fold_means_reduced = (
            reduced_grp.groupby(['mouse_id', 'neuron_id'], as_index=False)[key]
            .mean()
        )
        fold_means_reduced = fold_means_reduced.merge(
            neurons_to_keep[['mouse_id', 'neuron_id']],
            on=['mouse_id', 'neuron_id'],
            how='inner'
        )

        # --- Paired merge at neuron level ---
        paired = pd.merge(
            fold_means_full[['mouse_id', 'neuron_id', key]].rename(columns={key: 'full'}),
            fold_means_reduced[['mouse_id', 'neuron_id', key]].rename(columns={key: 'reduced'}),
            on=['mouse_id', 'neuron_id'],
            how='inner'
        ).dropna(subset=['full', 'reduced'])

        full_values = paired['full'].values
        reduced_values = paired['reduced'].values

        # Compute means & SEMs
        means_full.append(full_values.mean() if len(full_values) > 0 else np.nan)
        sems_full.append(full_values.std(ddof=1) / np.sqrt(len(full_values)) if len(full_values) > 1 else 0)

        means_reduced.append(reduced_values.mean() if len(reduced_values) > 0 else np.nan)
        sems_reduced.append(reduced_values.std(ddof=1) / np.sqrt(len(reduced_values)) if len(reduced_values) > 1 else 0)

        bar_colors.append(area_to_color.get(area, 'gray'))

        # --- Paired t-test across neurons ---
        if len(full_values) >= 5:
            try:
                _, pval = ttest_rel(full_values, reduced_values)
            except Exception:
                pval = np.nan
        else:
            pval = np.nan
        pvals.append(pval)

    # --- Plot bars ---
    ax.bar(x - width / 2, means_full, width, yerr=sems_full, label='Full', color='black', capsize=4)
    ax.bar(x + width / 2, means_reduced, width, yerr=sems_reduced, label=f'Reduced: {selected_reduced}',
           color=bar_colors, capsize=4)

    # --- Significance annotations ---
    for i, pval in enumerate(pvals):
        if np.isnan(pval):
            continue
        if pval < 0.001:
            star = '***'
        elif pval < 0.01:
            star = '**'
        elif pval < 0.05:
            star = '*'
        else:
            continue
        y_top = max(
            (means_full[i] or 0) + (sems_full[i] or 0),
            (means_reduced[i] or 0) + (sems_reduced[i] or 0)
        )
        ax.text(x[i], y_top + 0.005, star, ha='center', va='bottom', fontsize=11, color='red')

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
    name += f' {key}'
    putils.save_figure_with_options(fig, file_formats=['png', 'pdf'], filename=name, output_dir=output_folder)
    plt.close(fig)
    return


def plot_full_vs_reduced_barplot(df, selected_reduced, output_folder):
    """Bar plot: full model (left) vs reduced model (right) test_corr pooled across all neurons.
    Wilcoxon signed-rank paired test across neurons."""
    from scipy.stats import wilcoxon

    key = 'test_corr'

    df_full = df[df['model_name'] == 'full']
    df_reduced = df[df['model_name'] == selected_reduced]

    full_means = (
        df_full.groupby(['mouse_id', 'neuron_id'])[key].mean()
        .reset_index().rename(columns={key: 'full'})
    )
    reduced_means = (
        df_reduced.groupby(['mouse_id', 'neuron_id'])[key].mean()
        .reset_index().rename(columns={key: 'reduced'})
    )

    paired = pd.merge(full_means, reduced_means, on=['mouse_id', 'neuron_id']).dropna()
    if paired.empty:
        return

    full_vals = paired['full'].values
    reduced_vals = paired['reduced'].values
    n = len(full_vals)

    means = [full_vals.mean(), reduced_vals.mean()]
    sems = [full_vals.std(ddof=1) / np.sqrt(n), reduced_vals.std(ddof=1) / np.sqrt(n)]

    pval = np.nan
    if n >= 10:
        try:
            _, pval = wilcoxon(full_vals, reduced_vals)
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(3, 5), dpi=300)
    ax.bar([0, 1], means, yerr=sems, color=['black', 'steelblue'], capsize=5, width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Full', selected_reduced], rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Test correlation (mean ± SEM)')
    ax.set_title(f'Full vs {selected_reduced}\nn={n} neurons')

    if not np.isnan(pval):
        if pval < 0.001:
            star = '***'
        elif pval < 0.01:
            star = '**'
        elif pval < 0.05:
            star = '*'
        else:
            star = 'ns'
        y_top = max(m + s for m, s in zip(means, sems))
        y_line = y_top * 1.05
        ax.plot([0, 1], [y_line, y_line], 'k-', linewidth=1)
        ax.text(0.5, y_line * 1.01, star, ha='center', va='bottom', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    safe_name = selected_reduced.replace(' ', '_')
    putils.save_figure_with_options(fig, file_formats=['png', 'pdf'],
                                    filename=f'barplot_full_vs_{safe_name}',
                                    output_dir=output_folder)
    plt.close(fig)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_lrt_significance_per_area_per_model(df, area_groups, area_colors, output_folder):
    """
    Plot one figure per reduced model showing the proportion of significant neurons per area.

    Parameters
    ----------
    lrt_df : pd.DataFrame
        Output from compute_lrt_from_model_results().
        Must include ['neuron_id','mouse_id','reduced_model','significant'].

    area_groups : dict
        Mapping {group_name: [area1, area2, ...]} for area ordering and grouping.
    area_colors : dict
        Mapping {group_name: color} for bar colors.
    output_folder : str
        Directory where plots will be saved.
    """


    # Compute proportion of significant neurons per area and reduced model
    proportions = (
        df.groupby(['model_name','area_acronym_custom'])
        .agg(prop_sig=('lrt_significant','mean'), n=('lrt_significant','size'))
        .reset_index()
    )

    # Build ordered list of areas and corresponding colors
    ordered_areas = []
    area_to_color = {}
    for group_name, areas in area_groups.items():
        for area in areas:
            if area in proportions['area_acronym_custom'].values:
                ordered_areas.append(area)
                area_to_color[area] = area_colors.get(group_name, 'gray')

    # --- Plot one figure per reduced model ---
    for reduced_model, subdf in proportions.groupby('model_name'):

        subdf = subdf.set_index('area_acronym_custom').reindex(ordered_areas)
        values = subdf['prop_sig'].fillna(0).values
        colors = [area_to_color.get(a, 'gray') for a in ordered_areas]

        # Plot setup
        fig, ax = plt.subplots(figsize=(max(10, len(ordered_areas)*0.5), 5), dpi=300)
        x = np.arange(len(ordered_areas))

        bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(ordered_areas, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Proportion of significant neurons')
        ax.set_title(f'Significant neurons per area – {reduced_model}')

        # Annotate bar values
        for i, val in enumerate(values):
            if not np.isnan(val):
                ax.text(i, val + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=8)

        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()

        # Save each figure
        filename = f"LRT_significance_per_area_{reduced_model}"
        putils.save_figure_with_options(fig, file_formats=['png', 'pdf', 'svg'],dark_background=True, filename=filename, output_dir=output_folder)
        plt.close(fig)


def plot_lrt_connectivity_correlation(df, metric_col, metric_label, filename_prefix,
                                      output_folder, log_scale=False, invert_x=False,
                                      area_collapse=None):
    """
    For each reduced model, scatter-plot avg % significant neurons per area (averaged over
    mice) vs an area-level connectivity / hierarchy metric.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'area_acronym_custom', 'mouse_id', 'model_name', 'lrt_significant'
        and the column named by metric_col.
    metric_col : str
        Column in df with the per-row metric value (same value for all rows of an area).
    metric_label : str
        X-axis label.
    filename_prefix : str
        Prefix for saved filenames (model name appended).
    output_folder : str
        Directory where plots will be saved.
    log_scale : bool
        Whether to use a log x-axis (and fit regression in log space).
    invert_x : bool
        Whether to invert the x-axis.
    area_collapse : dict or None
        Optional mapping {area -> canonical_area} to collapse sub-regions into one point
        (e.g. {'DLS': 'CP', 'DMS': 'CP', ...}).
    """
    from scipy.stats import pearsonr, spearmanr

    if area_collapse is None:
        area_collapse = {}

    df = df.copy()
    df['plot_area'] = df['area_acronym_custom'].map(lambda a: area_collapse.get(a, a))

    per_area = (
        df.groupby(['model_name', 'plot_area'])
        .agg(mean_prop_sig=('lrt_significant', 'mean'), metric=(metric_col, 'first'))
        .reset_index()
        .rename(columns={'plot_area': 'area_acronym_custom'})
        .dropna(subset=['metric'])
    )

    for model_name, subdf in per_area.groupby('model_name'):
        subdf = subdf.copy()
        subdf['metric'] = pd.to_numeric(subdf['metric'], errors='coerce')
        subdf['mean_prop_sig'] = pd.to_numeric(subdf['mean_prop_sig'], errors='coerce')
        subdf = subdf.dropna(subset=['metric', 'mean_prop_sig'])

        x = subdf['metric'].values.astype(float)
        y = subdf['mean_prop_sig'].values.astype(float)
        areas = subdf['area_acronym_custom'].values

        if len(x) < 3:
            continue

        r, p = pearsonr(x, y)
        rho, p_sp = spearmanr(x, y)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        ax.scatter(x, y, color='steelblue', edgecolor='black', linewidth=0.5, s=60, zorder=3)

        for xi, yi, area in zip(x, y, areas):
            ax.text(xi, yi, area, fontsize=7, ha='left', va='bottom')

        if log_scale:
            log_x = np.log10(x)
            m, b = np.polyfit(log_x, y, 1)
            x_line = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
            ax.plot(x_line, m * np.log10(x_line) + b, color='firebrick', linewidth=1.2, zorder=2)
            ax.set_xscale('log')
        else:
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, m * x_line + b, color='firebrick', linewidth=1.2, zorder=2)

        if invert_x:
            ax.invert_xaxis()

        ax.set_xlabel(metric_label)
        ax.set_ylabel('Avg % significant neurons')
        ax.set_title(f'{model_name}\nr={r:.2f}, p={p:.3f}  |  ρ={rho:.2f}, p={p_sp:.3f}')
        ax.grid(linestyle='--', alpha=0.4)
        plt.tight_layout()

        filename = f"{filename_prefix}_{model_name}"
        putils.save_figure_with_options(fig, file_formats=['png', 'pdf', 'svg'], dark_background=True,
                                        filename=filename, output_dir=output_folder)
        plt.close(fig)


def plot_lrt_ipsi_correlation(df, output_folder):
    _collapse = {
        'DLS': 'CP', 'DMS': 'CP', 'TS': 'CP', 'VS': 'CP',
        'MO-tjM1': 'MO', 'MO-ALM': 'MO', 'MO-wM1': 'MO', 'MO-wM2': 'MO',
    }
    plot_lrt_connectivity_correlation(
        df, metric_col='avg_ipsi_corr',
        metric_label='Avg ipsi connectivity (Liu et al., log scale)',
        filename_prefix='LRT_ipsi_correlation',
        output_folder=output_folder,
        log_scale=True, invert_x=True,
        area_collapse=_collapse,
    )


def plot_lrt_harris_correlation(df, output_folder):
    plot_lrt_connectivity_correlation(
        df, metric_col='harris_hierarchy',
        metric_label='Hierarchy score (Harris et al. 2019)',
        filename_prefix='LRT_harris_correlation',
        output_folder=output_folder,
        log_scale=False, invert_x=False,
    )


def plot_lrt_significance_per_model_per_area(df, area_groups, area_colors, output_folder):
    """
    Plot one figure per area showing the proportion of significant neurons per reduced model.

    Parameters
    ----------
    lrt_df : pd.DataFrame
        Must include ['neuron_id','mouse_id','reduced_model','significant'].
    area_groups : dict
        Mapping {group_name: [area1, area2, ...]} for area ordering.
    area_colors : dict
        Mapping {group_name: color} for group-based coloring.
    output_folder : str
        Directory where plots will be saved.
    """

    # Compute proportion per area × model
    proportions = (
        df.groupby(['area_acronym_custom', 'model_name'])
          .agg(prop_sig=('lrt_significant','mean'), n=('lrt_significant','size'))
          .reset_index()
    )

    # Ordered list of areas (same as in your other function)
    ordered_areas = []
    for group_name, areas in area_groups.items():
        for area in areas:
            if area in proportions['area_acronym_custom'].values:
                ordered_areas.append(area)

    # --- One plot per AREA ---
    for area in ordered_areas:

        subdf = (
            proportions[proportions['area_acronym_custom'] == area]
            .set_index('model_name')
            .sort_index()
        )

        models = subdf.index.tolist()
        values = subdf['prop_sig'].fillna(0).values

        # Compute number of neurons and mice in this area
        df_area = df[df['area_acronym_custom'] == area]

        # Unique neurons = unique (mouse_id, neuron_id) pairs
        unique_neurons = df_area[['mouse_id', 'neuron_id']].drop_duplicates()
        n_neurons = len(unique_neurons)

        # Unique mice
        n_mice = df_area['mouse_id'].nunique()

        # assign color based on group
        area_group = next(
            (g for g, a in area_groups.items() if area in a),
            None
        )
        bar_color = area_colors.get(area_group, 'gray')

        # Plot
        fig, ax = plt.subplots(figsize=(max(8, len(models)*0.7), 5), dpi=300)
        x = np.arange(len(models))

        bars = ax.bar(x, values, color=bar_color, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Proportion of significant neurons')
        ax.set_title(
            f"LRT significance across models – {area}\n"
            f"{n_neurons} neurons from {n_mice} mice"
        )

        # Annotate each bar
        for i, val in enumerate(values):
            ax.text(i, val + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=8)

        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()

        # Save output
        filename = f"LRT_significance_per_model_{area}"
        putils.save_figure_with_options(
            fig,
            file_formats=['png', 'pdf', 'svg'],
            dark_background=True,
            filename=filename,
            output_dir=output_folder
        )
        plt.close(fig)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_lrt_significance_heatmap(df, area_groups, area_colors,
                                  output_folder, annotate=False):
    """
    Plot a heatmap showing the proportion of significant neurons
    for each area × reduced model.

    Parameters
    ----------
    lrt_df : pd.DataFrame
        Must include ['neuron_id','mouse_id','reduced_model','significant'].
    area_groups : dict
        Mapping {group_name: [area1, area2, ...]} defining area ordering.
    area_colors : dict
        Mapping {group_name: color} for side bar colors.
    output_folder : str
    annotate : bool
        If True, writes the value inside each heatmap cell.
    """



    # Compute proportion significant →
    proportions = (
        df.groupby(['area_acronym_custom','model_name'])
        .agg(prop_sig=('lrt_significant','mean'),
             n=('lrt_significant','size'))
        .reset_index()
    )

    # Ordered list of areas based on groups
    ordered_areas = [
        area
        for group, areas in area_groups.items()
        for area in areas
        if area in proportions['area_acronym_custom'].unique()
    ]

    # Pivot to matrix form: areas × models
    heatmap_data = proportions.pivot(
        index="area_acronym_custom",
        columns="model_name",
        values="prop_sig"
    ).reindex(ordered_areas)
    # Ensure all values are numeric (convert object -> float)
    heatmap_data = heatmap_data.astype(float)

    # Fill NaNs with 0 (or whatever makes sense)
    heatmap_data = heatmap_data.fillna(0)
    # --- Create heatmap ---
    fig, ax = plt.subplots(figsize=(12, max(6, len(ordered_areas)*0.4)), dpi=300)

    sns.heatmap(
        heatmap_data,
        cmap="viridis",
        vmin=0, vmax=1,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={"label": "Proportion significant"},
        annot=annotate,
        fmt=".2f" if annotate else ""
    )

    ax.set_ylabel("Area")
    ax.set_xlabel("Reduced model")
    ax.set_title("Encoding selectivity per area (proportion significant)")

    # Optionally add area-group color bars on the left
    # ------------------------------------------------
    import matplotlib.patches as patches

    for group_name, areas in area_groups.items():
        color = area_colors.get(group_name, "gray")
        for area in areas:
            if area in heatmap_data.index:
                y = heatmap_data.index.tolist().index(area)
                ax.add_patch(
                    patches.Rectangle(
                        (-0.6, y), 0.3, 1,
                        fill=True, color=color, transform=ax.transData,
                        clip_on=False, linewidth=0
                    )
                )

    plt.tight_layout()

    # Save figure
    filename = "LRT_significance_heatmap"
    putils.save_figure_with_options(fig, file_formats=['png', 'pdf'], filename=filename, output_dir=output_folder)

    plt.close(fig)


def plot_lrt_significance_overlap_per_area(df, output_folder):
    """
    Plot Jaccard index of significant neurons per area across reduced models.
    One figure per area.

    Parameters
    ----------
    lrt_df : pd.DataFrame
        Output of compute_lrt_from_model_results(), must contain
        ['neuron_id','mouse_id','reduced_model','significant']
    output_folder : str
        Path to save the plots
    """


    # List of areas with at least one neuron
    areas = df['area_acronym_custom'].dropna().unique()

    for area in areas:
        subdf = df[df['area_acronym_custom'] == area]

        # Build dictionary: model_name -> set of significant neuron_ids
        sig_sets = {
            model: set(d['neuron_id'][d['lrt_significant']])
            for model, d in subdf.groupby('model_name')
        }

        models = list(sig_sets.keys())
        n_models = len(models)
        overlap_matrix = np.zeros((n_models, n_models))

        # Compute Jaccard index
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                inter = len(sig_sets[m1] & sig_sets[m2])
                union = len(sig_sets[m1] | sig_sets[m2])
                overlap_matrix[i, j] = inter / union if union > 0 else np.nan

        overlap_df = pd.DataFrame(overlap_matrix, index=models, columns=models)

        # --- Plot ---
        plt.figure(figsize=(8,6), dpi=300)
        sns.heatmap(overlap_df, annot=True, cmap='viridis', vmin=0, vmax=1)
        plt.title(f"Overlap of significant neurons – {area} (Jaccard index)")
        plt.tight_layout()

        filename = f"LRT_overlap_{area}"
        putils.save_figure_with_options(plt.gcf(), file_formats=['png', 'pdf'], filename=filename, output_dir=output_folder)
        plt.close()

    return True


import seaborn as sns

def plot_lrt_significance_overlap(lrt_df, output_folder):
    """
    Plot overlap (Jaccard index) between significant neuron sets across reduced models.

    Parameters
    ----------
    lrt_df : pd.DataFrame
        Output of compute_lrt_from_model_results(), must contain ['neuron_id','reduced_model','significant']
    output_folder : str
        Path to save the overlap plot
    """
    # Build dictionary: model_name -> set of significant neuron_ids
    sig_sets = {
        model: set(df['neuron_id'][df['lrt_significant']])
        for model, df in lrt_df.groupby('model_name')
    }

    models = list(sig_sets.keys())
    n_models = len(models)
    overlap_matrix = np.zeros((n_models, n_models))

    # Compute Jaccard index (intersection / union)
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            inter = len(sig_sets[m1] & sig_sets[m2])
            union = len(sig_sets[m1] | sig_sets[m2])
            overlap_matrix[i, j] = inter / union if union > 0 else np.nan

    overlap_df = pd.DataFrame(overlap_matrix, index=models, columns=models)

    # --- Plot ---
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(overlap_df, annot=True, cmap='viridis', vmin=0, vmax=1)
    plt.title("Overlap of significant neurons between reduced models (Jaccard index)")
    plt.tight_layout()

    name = 'LRT_significance_overlap'
    putils.save_figure_with_options(plt.gcf(), file_formats=['png', 'pdf'], filename=name, output_dir=output_folder)
    plt.close()
    return overlap_df


def plot_two_reduced_per_area(df, reduced1, reduced2, area_groups, area_colors, output_folder, threshold=None):
    """
    Plot mean ± SEM test correlations per area for two reduced models,
    including significance stars (paired t-test) between models per area.

    :param df: pd.DataFrame with columns ['model_type','model_name','test_corr','area_acronym_custom','mouse_id','neuron_id']
    :param reduced1: str, name of first reduced model
    :param reduced2: str, name of second reduced model
    :param area_groups: dict, group_name -> list of area names
    :param area_colors: dict, group_name -> color
    :param output_folder: str path
    :param threshold: float or None, minimum test_corr_mean for neurons to be included
    """

    # Filter for reduced models of interest
    df_r1 = df[(df['model_name'] != 'full') & (df['model_name'] == reduced1)].copy()
    df_r2 = df[(df['model_name'] != 'full') & (df['model_name'] == reduced2)].copy()

    # Build ordered areas and colors
    ordered_areas = []
    area_to_color = {}
    for group_name, areas in area_groups.items():
        for area in areas:
            if area in df_r1['area_acronym_custom'].values or area in df_r2['area_acronym_custom'].values:
                ordered_areas.append(area)
                area_to_color[area] = area_colors[group_name]

    # Initialize lists
    means_r1, sems_r1, means_r2, sems_r2, bar_colors = [], [], [], [], []

    # Plot setup
    fig, ax = plt.subplots(figsize=(max(12, len(ordered_areas) * 0.5), 6), dpi=300)
    x = np.arange(len(ordered_areas))
    width = 0.35

    for i, area in enumerate(ordered_areas):
        # --- Model 1 ---
        grp1 = df_r1[df_r1['area_acronym_custom'] == area]
        fold_means_r1 = grp1.groupby(['mouse_id', 'neuron_id'], as_index=False)['test_corr'].mean()

        # --- Model 2 ---
        grp2 = df_r2[df_r2['area_acronym_custom'] == area]
        fold_means_r2 = grp2.groupby(['mouse_id', 'neuron_id'], as_index=False)['test_corr'].mean()

        # Apply threshold if given
        if threshold is not None:
            neurons_to_keep = fold_means_r1[fold_means_r1['test_corr'] >= threshold]
        else:
            neurons_to_keep = fold_means_r1

        # Keep only neurons that exist in both models and pass threshold
        merged = fold_means_r1.merge(
            fold_means_r2, on=['mouse_id', 'neuron_id'], suffixes=('_r1', '_r2')
        )
        merged = merged.merge(neurons_to_keep[['mouse_id', 'neuron_id']], on=['mouse_id', 'neuron_id'], how='inner')

        vals1 = merged['test_corr_r1'].to_numpy()
        vals2 = merged['test_corr_r2'].to_numpy()

        # Compute mean ± SEM
        means_r1.append(vals1.mean() if len(vals1) > 0 else np.nan)
        sems_r1.append(vals1.std(ddof=1) / np.sqrt(len(vals1)) if len(vals1) > 1 else 0)

        means_r2.append(vals2.mean() if len(vals2) > 0 else np.nan)
        sems_r2.append(vals2.std(ddof=1) / np.sqrt(len(vals2)) if len(vals2) > 1 else 0)

        bar_colors.append(area_to_color.get(area, 'gray'))

        # --- Significance test ---
        if len(vals1) > 1 and len(vals2) > 1:
            _, pval = ttest_rel(vals1, vals2)
        else:
            pval = np.nan

        if pval < 0.05:
            ax.text(x[i], max(means_r1[-1] + sems_r1[-1], means_r2[-1] + sems_r2[-1]) + 0.01,
                    '*', ha='center', va='bottom', fontsize=12, color='red')

    # --- Plot bars ---
    ax.bar(x - width / 2, means_r1, width, yerr=sems_r1, label=reduced1, color='k', capsize=4)
    ax.bar(x + width / 2, means_r2, width, yerr=sems_r2, label=reduced2, color=bar_colors, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(ordered_areas, rotation=45, ha='right')
    ax.set_ylabel('Test Score')
    ax.set_title(f'{reduced1} vs {reduced2} per area')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    # Save
    name = f'{reduced1} vs {reduced2} per area'
    if threshold is not None:
        name += f' threshold {threshold}'
    putils.save_figure_with_options(fig, file_formats=['png', 'pdf'], filename=name, output_dir=output_folder)
    plt.close(fig)


def compute_lrt_from_model_results_weird(
    merged_df,
    trials_df,
    alpha=0.05,
):
    """
    Computes LRT significance flags for:
        - all trials
        - whisker-only trials
    For ALL reduced models relative to the full model.

    merged_df must contain:
        neuron_id, fold, model_name,
        y_test_array, y_pred_array, n_bins, test_trials
    """

    # Full model df
    red_df = merged_df[merged_df["model_name"] == 'full']

    # All reduced models
    reduced_models = merged_df["model_name"].unique()
    reduced_models = [m for m in reduced_models if m != 'full']

    results = []

    for reduced_model in reduced_models:

        full_df = merged_df[merged_df["model_name"] == reduced_model]

        # Merge per neuron + fold
        merged = full_df.merge(
            red_df,
            on=["neuron_id", "fold", "mouse_id"],
            suffixes=("_full", "_red")
        )

        # Degrees of freedom difference
        df_diff = int(
            np.abs(
                merged["predictors_full"].apply(len).iloc[0]
                - merged["predictors_red"].apply(len).iloc[0]
            )
        )

        # Collect per-neuron results
        rows = []

        for neuron_id, sub in merged.groupby("neuron_id"):

            dll_all = 0
            dll_whisker = 0

            for _, row in sub.iterrows():

                # Load arrays
                y_true = row["y_test_array_full"]
                mu_f    = row["y_pred_array_full"]
                mu_r    = row["y_pred_array_red"]

                n_bins  = row["n_bins_full"]
                n_trials = y_true.shape[0] // n_bins

                # Reshape to (trials, bins)
                yt = y_true.reshape(n_trials, n_bins)
                mf = mu_f.reshape(n_trials, n_bins)
                mr = mu_r.reshape(n_trials, n_bins)

                # Find trial types
                test_trials = np.array(ast.literal_eval(row["test_trials_full"]))
                tt = trials_df.iloc[test_trials]["whisker_stim"].values

                # ---- ALL TRIALS ----
                yt_all = yt.ravel()
                mf_all = np.clip(mf.ravel(), 1e-12, None)
                mr_all = np.clip(mr.ravel(), 1e-12, None)

                llf_all = np.sum(yt_all * np.log(mf_all) - mf_all)
                llr_all = np.sum(yt_all * np.log(mr_all) - mr_all)
                dll_all += (llf_all - llr_all)

                # ---- WHISKER TRIALS ONLY ----
                idx = np.where(tt == 1)[0]
                if len(idx) > 0:
                    yt_w = yt[idx].ravel()
                    mf_w = np.clip(mf[idx].ravel(), 1e-12, None)
                    mr_w = np.clip(mr[idx].ravel(), 1e-12, None)

                    llf_w = np.sum(yt_w * np.log(mf_w) - mf_w)
                    llr_w = np.sum(yt_w * np.log(mr_w) - mr_w)
                    dll_whisker += (llf_w - llr_w)

            # Chi2 values
            chi2_all = 2 * dll_all
            chi2_whisker = 2 * dll_whisker

            # p-values
            p_all = 1 - chi2.cdf(chi2_all, df=df_diff)
            p_whisker = 1 - chi2.cdf(chi2_whisker, df=df_diff)

            rows.append({
                "mouse_id" :row['mouse_id'],
                "neuron_id": neuron_id,
                "model_name": reduced_model,
                "df_diff": df_diff,
                "chi2_alltrials": chi2_all,
                "chi2_whisker": chi2_whisker,
                "p_value": p_all,
                "p_whisker": p_whisker,
                "lrt_significant": p_all < alpha,
                "lrt_sig_whisker": p_whisker < alpha
            })

        results.append(pd.DataFrame(rows))

    return pd.concat(results, ignore_index=True)

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
            full_df[['neuron_id','mouse_id', 'fold', ll_field, 'predictors']],
            reduced_df[['neuron_id', 'mouse_id','fold', ll_field, 'predictors']],
            on=['neuron_id', 'fold','mouse_id'],
            suffixes=('_full', '_reduced')
        )

        # Compute LRT statistic per fold
        merged['lrt_stat'] = 2 * (merged[f'{ll_field}_full'] - merged[f'{ll_field}_reduced'])

        # Compute degrees of freedom difference
        merged['df_diff'] = np.abs(merged['predictors_full'].apply(len) - merged['predictors_reduced'].apply(len))

        # Aggregate per neuron and mouse
        grouped = merged.groupby(['neuron_id', 'mouse_id']).agg(
            mean_lrt_stat=('lrt_stat', 'sum'),
            df_diff=('df_diff', 'first')  # assume same across folds
        ).reset_index()

        # Compute p-values
        grouped['p_value'] = 1 - chi2.cdf(grouped['mean_lrt_stat'], df=grouped['df_diff'])
        grouped['lrt_significant'] = grouped['p_value'] < alpha
        grouped['model_name'] = reduced_model

        results.append(grouped)

    lrt_df = pd.concat(results, ignore_index=True)
    return lrt_df


def compute_trialtype_metrics(merged, trials_df):
    """
    Parallel computation of:
        - trial-type Pearson correlations
        - trial-type mean log-likelihood
    per neuron, per fold, per model.

    Uses the same parallel pattern as your original compute_trialtype_correlations.
    """

    def process_row(row_data):
        """
        Process a single row (optimized for parallel execution).
        Extracts:
            y_test, y_pred → reshape into trials × bins
            trial types → compute metrics per trial_type
        """

        (neuron_id, fold, mouse_id, area_custom, model_name,
         y_test, y_pred, n_bins, test_trials_str) = row_data

        # Number of trials
        n_trials = y_pred.shape[0] // n_bins
        y_test = y_test.reshape(n_trials, n_bins)
        y_pred = y_pred.reshape(n_trials, n_bins)

        # Trial type labels for test trials
        test_trial_ids = np.array(ast.literal_eval(test_trials_str))
        behav_types = trials_df.iloc[test_trial_ids]["behav_type"].values
        unique_types = np.unique(behav_types)

        results = []

        for trial_type in unique_types:
            idx = np.where(behav_types == trial_type)[0]
            if len(idx) < 2:
                continue

            # Extract responses
            y_true_t = y_test[idx].ravel()
            y_pred_t = y_pred[idx].ravel()

            if y_true_t.std() == 0:
                continue

            # -------- correlation --------
            r = np.corrcoef(y_true_t, y_pred_t)[0, 1]

            # -------- Poisson LL --------
            # y*log(mu) - mu   (ignoring constant term)
            mu = np.clip(y_pred_t, 1e-12, None)
            ll = np.mean(y_true_t * np.log(mu) - mu)

            results.append({
                "mouse_id": mouse_id,
                "neuron_id": neuron_id,
                "fold": fold,
                "trial_type": trial_type,
                "test_corr": r,
                "test_ll": ll,
                "area_acronym_custom": area_custom,
                "model_name": model_name
            })

        return results

    # --- Prepare data for parallel execution ---
    row_data = [
        (
            row["neuron_id"],
            row["fold"],
            row.get("mouse_id", "unknown"),
            row.get("area_acronym_custom", None),
            row["model_name"],
            row["y_test_array"],
            row["y_pred_array"],
            row["n_bins"],
            row["test_trials"],
        )
        for _, row in merged.iterrows()
    ]

    # --- Parallel compute ---
    results = Parallel(n_jobs=-1, batch_size=10)(
        delayed(process_row)(data)
        for data in tqdm(row_data, desc="Computing trial-type corr + ll")
    )

    # Flatten
    flat = [x for group in results for x in group]

    return pd.DataFrame(flat)

from scipy.stats import chi2
import numpy as np
import ast
import pandas as pd

def compute_lrt_significance_flags(
    merged_df, trials_df,
    full_model="full",
    reduced_model="2whisker",
    whisker_label="whisker",
    alpha=0.05,
):
    """
    Adds two LRT significance fields:
        - lrt_sig_alltrials
        - lrt_sig_whisker

    Returns dataframe indexed per neuron.
    """


    # Extract dataframes
    fdf = merged_df[merged_df["model_name"] == full_model]
    rdf = merged_df[merged_df["model_name"] == reduced_model]
    df_params = np.abs(fdf['predictors'].apply(len).iloc[0] - rdf['predictors'].apply(len).iloc[0])

    # Align full/reduced by neuron + fold
    merged = fdf.merge(
        rdf,
        on=["neuron_id", "fold"],
        suffixes=("_full", "_red")
    )

    rows = []

    for neuron_id, sub in merged.groupby("neuron_id"):

        dll_all = 0
        dll_whisker = 0
        dll_full = []
        dll_reduced = []
        for _, row in sub.iterrows():

            # Load data
            y_true_full = row["y_test_array_full"]
            y_pred_full = row["y_pred_array_full"]
            y_pred_red  = row["y_pred_array_red"]
            n_bins = row["n_bins_full"]

            # reshape trials × bins
            n_trials = y_true_full.shape[0] // n_bins
            yt = y_true_full.reshape(n_trials, n_bins)
            mu_f = y_pred_full.reshape(n_trials, n_bins)
            mu_r = y_pred_red.reshape(n_trials, n_bins)

            # trial types from table
            test_trials = np.array(ast.literal_eval(row["test_trials_full"]))
            tt = trials_df.iloc[test_trials]["whisker_stim"].values

            # --- ALL TRIALS ---
            yt_all = yt.ravel()

            mu_f_all = np.clip(mu_f.ravel(), 1e-12, None)
            mu_r_all = np.clip(mu_r.ravel(), 1e-12, None)

            ll_f_all = np.sum(yt_all * np.log(mu_f_all) - mu_f_all)
            ll_r_all = np.sum(yt_all * np.log(mu_r_all) - mu_r_all)
            dll_all += (ll_f_all - ll_r_all)
            dll_full.append(ll_f_all)
            dll_reduced.append(ll_r_all)
            # --- WHISKER ONLY ---
            idx = np.where(tt == 1)[0]
            if len(idx) > 0:
                yt_w = yt[idx].ravel()
                mu_f_w = np.clip(mu_f[idx].ravel(), 1e-12, None)
                mu_r_w = np.clip(mu_r[idx].ravel(), 1e-12, None)

                ll_f_w = np.sum(yt_w * np.log(mu_f_w) - mu_f_w)
                ll_r_w = np.sum(yt_w * np.log(mu_r_w) - mu_r_w)
                dll_whisker += (ll_f_w - ll_r_w)

        chi2_all = 2 * dll_all
        chi2_whisker = 2 * dll_whisker
        p_all =  1 - chi2.cdf(chi2_all, df=df_params)
        p_whisker =  1 - chi2.cdf(chi2_whisker, df=df_params)

        rows.append({
            "neuron_id": neuron_id,
            "lrt_sig_alltrials": p_all < alpha,
            "lrt_sig_whisker": p_whisker < alpha,
            "chi2_alltrials": chi2_all,
            "chi2_whisker": chi2_whisker
        })

    return pd.DataFrame(rows)


def collapse_trialtype_metrics(df_trial):
    """
    Average per neuron × trial_type × model_name across folds.
    """

    agg = (
        df_trial.groupby(
            ["mouse_id", "neuron_id", "model_name",
             "area_acronym_custom", "trial_type"]
        )
        .agg(
            test_corr_mean=("test_corr", "mean"),
            test_corr_sem=("test_corr", lambda x: x.std() / np.sqrt(len(x))),
            test_ll_mean=("test_ll", "mean"),
            test_ll_sem=("test_ll", lambda x: x.std() / np.sqrt(len(x)))
        )
        .reset_index()
    )

    return agg





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
    Optimized for vectorization and parallel processing.
    """

    def process_row(row_data):
        """Process a single row - designed for parallelization."""
        neuron_id, fold, mouse_id, area_custom, model_type, model_name, \
            y_test, y_pred, n_bins, test_trials_str = row_data

        # Reshape into trials x bins
        n_trials = y_pred.shape[0] // n_bins
        y_test = y_test.reshape(n_trials, n_bins)
        y_pred = y_pred.reshape(n_trials, n_bins)

        # Get test trial indices
        test_trial_ids = np.array(ast.literal_eval(test_trials_str))
        behav_types = trials_df.iloc[test_trial_ids]["behav_type"].values

        results = []
        unique_types = np.unique(behav_types)

        for trial_type in unique_types:
            idx = np.where(behav_types == trial_type)[0]
            if len(idx) < 2:
                continue

            y_true_t = y_test[idx].ravel()
            y_pred_t = y_pred[idx].ravel()

            # Quick check for variance
            if y_true_t.std() == 0:
                continue

            r = np.corrcoef(y_true_t, y_pred_t)[0, 1]  # Faster than pearsonr

            results.append({
                "mouse_id": mouse_id,
                "neuron_id": neuron_id,
                "fold": fold,
                "trial_type": trial_type,
                "test_corr": r,
                "area_acronym_custom": area_custom,
                "model_type": model_type,
                "model_name": model_name
            })

        return results

    # Prepare data for parallel processing
    row_data = [
        (
            row["neuron_id"],
            row["fold"],
            row.get("mouse_id", "unknown"),
            row.get("area_acronym_custom", None),
            row["model_type"],
            row["model_name"],
            row['y_test_array'],
            row['y_pred_array'],
            row["n_bins"],
            row["test_trials"]
        )
        for _, row in merged.iterrows()
    ]

    # Parallel processing
    results = Parallel(n_jobs=-1, batch_size=10)(
        delayed(process_row)(data)
        for data in tqdm(row_data, desc="Computing correlations")
    )

    # Flatten results
    all_rows = [item for sublist in results for item in sublist]

    return pd.DataFrame(all_rows)


def compute_ev(merged):
    """
    Compute explained variance per neuron, per fold, and keeps mouse_id.
    """
    rows = []
    for _, row in merged.iterrows():
        neuron_id = row["neuron_id"]
        fold = row["fold"]
        mouse_id = row.get("mouse_id", "unknown")  # assume merged has mouse_id column
        area_custom = row.get("area_acronym_custom", None)
        model_type = row["model_type"]
        model_name = row["model_name"]

        # decode arrays
        y_test =row['y_test_array']
        y_pred = row['y_pred_array']
        n_bins = row["n_bins"]

        # reshape into trials x bins
        n_trials = y_pred.shape[0] // n_bins

        # Only compute metrics if variance > 0
        if np.var(y_test) > 0:
            # Explained variance
            residual = y_test - y_pred
            ev = 1 - np.var(residual) / np.var(y_test)

            rows.append({
                "mouse_id": mouse_id,
                "neuron_id": neuron_id,
                "fold": fold,
                "explained_variance": ev,
                "area_acronym_custom": area_custom,
                "model_type": model_type,
                "model_name": model_name
            })

    return pd.DataFrame(rows)




def compute_trialtype_quartile_correlations(merged, trials_df):
    """
    Compute test Pearson correlation per neuron, per fold, per trial type, per quartile.
    Requires 'quartile' column in trials_df.
    """
    rows = []
    for _, row in merged.iterrows():
        neuron_id = row["neuron_id"]
        fold = row["fold"]
        mouse_id = row.get("mouse_id", "unknown")
        area_custom = row.get("area_acronym_custom", None)
        model_type = row["model_type"]
        model_name = row["model_name"]
        y_test =row['y_test_array']
        y_pred = row['y_pred_array']
        n_bins = row["n_bins"]
        n_trials = y_pred.shape[0] // n_bins
        y_test = y_test.reshape(n_trials, n_bins)
        y_pred = y_pred.reshape(n_trials, n_bins)

        test_trial_ids = np.array(ast.literal_eval(row["test_trials"]))
        trials_test_df = trials_df.iloc[test_trial_ids, :]

        # Must have quartile column already in trials_df
        if "quartile" not in trials_test_df.columns:
            raise ValueError("trials_df must contain 'quartile' column")

        for q in trials_test_df["quartile"].unique():
            for trial_type in trials_test_df["behav_type"].unique():
                idx = np.where(
                    (trials_test_df["quartile"].values == q) &
                    (trials_test_df["behav_type"].values == trial_type)
                )[0]
                if len(idx) < 2:
                    continue
                y_true_t = y_test[idx, :].ravel()
                y_pred_t = y_pred[idx, :].ravel()
                if len(np.unique(y_true_t)) > 1:
                    r, _ = pearsonr(y_true_t, y_pred_t)
                    rows.append({
                        "mouse_id": mouse_id,
                        "neuron_id": neuron_id,
                        "fold": fold,
                        "quartile": q,
                        "trial_type": trial_type,
                        "test_corr": r,
                        "area_acronym_custom": area_custom,
                        "model_type": model_type,
                        "model_name": model_name
                    })
    return pd.DataFrame(rows)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


def plot_full_vs_reduced_per_area_and_trialtype(
    df,
    selected_reduced,
    area_groups,
    area_colors,
    output_folder,
    threshold=None
):
    """
    Plot mean ± SEM test correlations per area and trial type for full vs reduced model.

    Parameters
    ----------
    df : DataFrame
        Must include ['model_type','model_name','test_corr','area_acronym_custom',
                      'mouse_id','neuron_id','trial_type'].
    selected_reduced : str
        Name of the reduced model to compare against full.
    area_groups : dict
        {group_name: [list of area names]}.
    area_colors : dict
        {group_name: color}.
    output_folder : str
        Directory to save figure.
    threshold : float or None
        Minimum test_corr (full model) per neuron to be included.
    """

    # --- Separate models ---
    df_full = df[df['model_name'] == 'full'].copy()
    df_reduced = df[
        (df['model_name'] != 'full') &
        (df['model_name'] == selected_reduced)
    ].copy()

    # --- Determine trial types and ordered areas ---
    trial_types = sorted(df['trial_type'].unique())
    ordered_areas = []
    area_to_color = {}

    for group_name, areas in area_groups.items():
        for area in areas:
            if area in df['area_acronym_custom'].values:
                ordered_areas.append(area)
                area_to_color[area] = area_colors[group_name]
    # --- Create figure ---
    fig, axes = plt.subplots(
        nrows=len(trial_types),
        ncols= 1,
        figsize=(max(14, len(ordered_areas) * 0.6), 6*len(trial_types)),
        dpi=300,
        sharey=True
    )
    if len(trial_types) == 1:
        axes = [axes]

    # --- Loop over trial types ---
    for t_idx, trial_type in enumerate(trial_types):
        ax = axes[t_idx]
        means_full, sems_full, means_reduced, sems_reduced, bar_colors = [], [], [], [], []

        for area in ordered_areas:
            # select subset
            fgrp = df_full[
                (df_full['trial_type'] == trial_type) &
                (df_full['area_acronym_custom'] == area)
            ]
            rgrp = df_reduced[
                (df_reduced['trial_type'] == trial_type) &
                (df_reduced['area_acronym_custom'] == area)
            ]

            if fgrp.empty or rgrp.empty:
                means_full.append(np.nan)
                sems_full.append(0)
                means_reduced.append(np.nan)
                sems_reduced.append(0)
                bar_colors.append(area_to_color.get(area, 'gray'))
                continue

            # average across folds per neuron
            fmeans = (
                fgrp.groupby(['mouse_id', 'neuron_id'], as_index=False)['test_corr']
                .mean()
                .rename(columns={'test_corr': 'test_corr_full'})
            )
            rmeans = (
                rgrp.groupby(['mouse_id', 'neuron_id'], as_index=False)['test_corr']
                .mean()
                .rename(columns={'test_corr': 'test_corr_reduced'})
            )

            # apply threshold on full model
            if threshold is not None:
                valid_ids = fmeans.loc[fmeans['test_corr_full'] >= threshold, ['mouse_id', 'neuron_id']]
            else:
                valid_ids = fmeans[['mouse_id', 'neuron_id']]

            # restrict both full and reduced
            fvals = fmeans.merge(valid_ids, on=['mouse_id', 'neuron_id'], how='inner')['test_corr_full']
            rvals = rmeans.merge(valid_ids, on=['mouse_id', 'neuron_id'], how='inner')['test_corr_reduced']

            # compute mean ± sem
            means_full.append(fvals.mean() if len(fvals) else np.nan)
            sems_full.append(fvals.std(ddof=1) / np.sqrt(len(fvals)) if len(fvals) > 1 else 0)
            means_reduced.append(rvals.mean() if len(rvals) else np.nan)
            sems_reduced.append(rvals.std(ddof=1) / np.sqrt(len(rvals)) if len(rvals) > 1 else 0)
            bar_colors.append(area_to_color.get(area, 'gray'))

            # paired t-test and star annotation
            if len(fvals) > 1 and len(rvals) > 1:
                stat, p = ttest_rel(fvals, rvals)
                if p < 0.05:
                    y = max(means_full[-1] + sems_full[-1], means_reduced[-1] + sems_reduced[-1])
                    ax.text(ordered_areas.index(area), y + 0.01, '*', ha='center', va='bottom', color='red')

        # --- Plot bars ---
        x = np.arange(len(ordered_areas))
        width = 0.35
        ax.bar(x - width / 2, means_full, width, yerr=sems_full,
               label='Full', color='black', capsize=4)
        ax.bar(x + width / 2, means_reduced, width, yerr=sems_reduced,
               label=f'Reduced ({selected_reduced})', color=bar_colors, capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(ordered_areas, rotation=45, ha='right')
        ax.set_title(trial_type)
        ax.grid(True, linestyle='--', alpha=0.4)
        if t_idx == 0:
            ax.set_ylabel('Test correlation')

    axes[0].legend()
    plt.suptitle(f'Full vs {selected_reduced} per area and trial type', y=1.02)
    plt.savefig(f"{output_folder}/full_vs_{selected_reduced}_per_area_and_trialtype.png", bbox_inches='tight')
    plt.close(fig)
    return


def plot_model_comparison(
    neuron_ids, df_full, df_reduced, trials_df, output_folder, name,
    reduced_model="whisker_encoding", bin_sizes=(0.1, 0.1), zscore=False
):
    """
    Plot average neural data, full model predictions, reduced model predictions,
    AND reduced-model y_test, across multiple neurons and trial types,
    with SEM across folds.
    """

    bin_size_full, bin_size_reduced = bin_sizes

    # ------------------------
    # FULL MODEL
    # ------------------------
    all_y_test_full = {}
    all_y_pred_full = {}

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

                all_y_test_full.setdefault(trial_type, []).append(test_mean)
                all_y_pred_full.setdefault(trial_type, []).append(pred_mean)

    # ------------------------
    # REDUCED MODEL
    # ------------------------
    all_y_test_reduced = {}
    all_y_pred_reduced = {}

    for nid in neuron_ids:
        res_all = df_reduced[df_reduced["neuron_id"] == nid]
        if res_all.empty:
            continue

        for res in res_all.itertuples(index=False):
            y_test = res.y_test_array
            y_pred = res.y_pred_array
            n_bins = res.n_bins

            n_trials = y_pred.shape[0] // n_bins
            y_pred = y_pred.reshape(n_trials, n_bins)
            y_test = y_test.reshape(n_trials, n_bins)

            test_trial_ids = np.array(ast.literal_eval(res.test_trials))
            order = np.argsort(test_trial_ids)
            y_pred = y_pred[order, :]
            y_test = y_test[order, :]

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

                all_y_test_reduced.setdefault(trial_type, []).append(test_mean)
                all_y_pred_reduced.setdefault(trial_type, []).append(pred_mean)

    # ------------------------
    # Fits
    # ------------------------
    fits_full = df_full[df_full["neuron_id"].isin(neuron_ids)]["test_corr"].astype(float).mean()
    fits_reduced = df_reduced[df_reduced["neuron_id"].isin(neuron_ids)]["test_corr"].astype(float).mean()

    trialtype_corrs_full = compute_trialtype_correlations(df_full[df_full["neuron_id"].isin(neuron_ids)], trials_df)
    trialtype_corrs_reduced = compute_trialtype_correlations(df_reduced[df_reduced["neuron_id"].isin(neuron_ids)], trials_df)

    corr_summary_full = trialtype_corrs_full.groupby("trial_type")["test_corr"].mean().to_dict()
    corr_summary_reduced = trialtype_corrs_reduced.groupby("trial_type")["test_corr"].mean().to_dict()

    # ------------------------
    # Plotting
    # ------------------------
    trial_types = sorted(all_y_test_full.keys())
    plt.ioff()

    # Create figure with 2 rows if single neuron (for coefficients), otherwise 1 row
    if len(neuron_ids) == 1:
        # Row 1: trial-type predictions, Row 2: full and reduced coefficients
        fig = plt.figure(figsize=(5 * len(trial_types), 9))

        # Use a finer grid to allow better centering of coefficient plots
        # Make bottom row use double the grid resolution for precise positioning
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2],
                             hspace=0.3, wspace=0.3)

        # Top row: trial type predictions (span both columns to use full width)
        gs_top = gs[0, :].subgridspec(1, len(trial_types), wspace=0.3)
        axes_pred = [fig.add_subplot(gs_top[0, i]) for i in range(len(trial_types))]

        # Bottom row: coefficients (each takes one column, equal width)
        ax_coef_full = fig.add_subplot(gs[1, 0])
        ax_coef_reduced = fig.add_subplot(gs[1, 1])  # No sharey - each has its own labels

        axes = axes_pred  # For compatibility with existing code
    else:
        fig, axes = plt.subplots(1, len(trial_types), figsize=(15, 5), sharey=True)
        if len(trial_types) == 1:
            axes = [axes]

    window_bounds_sec = (-1, 2)
    time_stim = 0

    if len(neuron_ids) == 1:
        title = f"Reduced model {reduced_model}, neuron {neuron_ids[0]}, {df_full['area_acronym_custom'].iloc[0]}\nfull fit={fits_full:.3f}, reduced fit={fits_reduced:.3f}"
    else:
        title = f"Reduced model {reduced_model}, {len(neuron_ids)} neurons\nfull fit={fits_full:.3f}, reduced fit={fits_reduced:.3f}"
    fig.suptitle(title)

    if not isinstance(axes, list):
        axes = [axes]

    # stimulus colors
    stim_colors = {
        "whisker_hit": "forestgreen",
        "whisker_miss": "orange",
        "auditory_hit": "mediumblue",
        "auditory_miss": "skyblue",
        "catch": "gray",
        "correct_rejection": "black"
    }

    # ------------------------
    # Loop over trial types
    # ------------------------
    for ax, trial_type in zip(axes, trial_types):

        if trial_type not in all_y_pred_reduced:
            continue

        putils.remove_top_right_frame(ax)
        ax.set_ylabel("Spikes", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=10)

        ax.axvline(time_stim, color=stim_colors.get(trial_type, "k"),
                   linestyle="-", linewidth=1)

        # === FULL y_test (black) ===
        data_stack = np.stack(all_y_test_full[trial_type])
        mean_data = data_stack.mean(axis=0)
        sem_data = data_stack.std(axis=0, ddof=1) / np.sqrt(data_stack.shape[0])
        time_data = np.linspace(
            window_bounds_sec[0] + bin_size_full / 2,
            window_bounds_sec[1] - bin_size_full / 2,
            mean_data.shape[0]
        )
        ax.plot(time_data, mean_data, color="black", label="data (full)")
        ax.fill_between(time_data, mean_data - sem_data,
                        mean_data + sem_data, color="black", alpha=0.3)

        # # === REDUCED y_test (grey) ===
        # if trial_type in all_y_test_reduced:
        #     data_stack_r = np.stack(all_y_test_reduced[trial_type])
        #     mean_data_r = data_stack_r.mean(axis=0)
        #     sem_data_r = data_stack_r.std(axis=0, ddof=1) / np.sqrt(data_stack_r.shape[0])
        #     time_data_r = np.linspace(
        #         window_bounds_sec[0] + bin_size_reduced / 2,
        #         window_bounds_sec[1] - bin_size_reduced / 2,
        #         mean_data_r.shape[0]
        #     )
        #     ax.plot(time_data_r, mean_data_r, color="gray", label="data (reduced)")
        #     ax.fill_between(time_data_r, mean_data_r - sem_data_r,
        #                     mean_data_r + sem_data_r, color="gray", alpha=0.3)

        # === FULL prediction (green) ===
        full_stack = np.stack(all_y_pred_full[trial_type])
        mean_full = full_stack.mean(axis=0)
        sem_full = full_stack.std(axis=0, ddof=1) / np.sqrt(full_stack.shape[0])
        time_full = np.linspace(
            window_bounds_sec[0] + bin_size_full / 2,
            window_bounds_sec[1] - bin_size_full / 2,
            mean_full.shape[0]
        )
        ax.plot(time_full, mean_full, color="green", label="full")
        ax.fill_between(time_full, mean_full - sem_full,
                        mean_full + sem_full, color="green", alpha=0.3)

        # === REDUCED prediction (red) ===
        reduced_stack = np.stack(all_y_pred_reduced[trial_type])
        mean_reduced = reduced_stack.mean(axis=0)
        sem_reduced = reduced_stack.std(axis=0, ddof=1) / np.sqrt(reduced_stack.shape[0])
        time_reduced = np.linspace(
            window_bounds_sec[0] + bin_size_reduced / 2,
            window_bounds_sec[1] - bin_size_reduced / 2,
            mean_reduced.shape[0]
        )
        ax.plot(time_reduced, mean_reduced, color="red", label="reduced")
        ax.fill_between(time_reduced, mean_reduced - sem_reduced,
                        mean_reduced + sem_reduced, color="red", alpha=0.3)

        corr_f = corr_summary_full.get(trial_type, np.nan)
        corr_r = corr_summary_reduced.get(trial_type, np.nan)
        ax.set_title(f"{trial_type}\nfull={corr_f:.2f}, reduced={corr_r:.2f}", fontsize=12) #"

    # ------------------------
    # Add coefficient plots for single neuron
    # ------------------------
    if len(neuron_ids) == 1:
        nid = neuron_ids[0]

        # ---- FULL MODEL COEFFICIENTS ----
        # Get coefficients from full model
        res_full = df_full[df_full["neuron_id"] == nid].iloc[0]
        predictors_full = res_full.predictors
        coef_full = res_full.coef
        coef_full = np.array(ast.literal_eval(coef_full))

        # Plot using the same style as plot_coefficients_into_axis
        n_features_full = len(coef_full)

        # Reverse order to match design matrix (top to bottom) - same as plot_coefficients_into_axis
        coef_vector_full = coef_full[::-1]
        feature_names_full = predictors_full[::-1]

        # Create y-positions from 0 to 1 (normalized) to match plot_coefficients_into_axis spacing
        spacing_fraction = 0.02
        total_spacing_full = spacing_fraction * (n_features_full - 1)
        available_height = 1.0 - total_spacing_full
        subplot_height_full = available_height / n_features_full

        y_positions_full = []
        for i in range(n_features_full):
            y_top = 1.0 - i * (subplot_height_full + spacing_fraction)
            y_center = y_top - subplot_height_full / 2
            y_positions_full.append(y_center)
        y_positions_full = np.array(y_positions_full)

        bar_height_full = subplot_height_full * 0.8

        # Color code by predictor type
        colors_full = []
        for fname in feature_names_full:
            if 'whisker' in fname:
                colors_full.append('forestgreen')
            elif 'auditory' in fname:
                colors_full.append('mediumblue')
            elif 'jaw' in fname or 'lick' in fname:
                colors_full.append('orange')
            elif 'reward' in fname:
                colors_full.append('purple')
            else:
                colors_full.append('gray')

        ax_coef_full.barh(y_positions_full, coef_vector_full, height=bar_height_full, color=colors_full, alpha=0.7,
                          edgecolor='white', linewidth=0.5)

        ax_coef_full.set_yticks(y_positions_full)
        ax_coef_full.set_yticklabels(feature_names_full, fontsize=7)
        ax_coef_full.set_xlabel("Coefficient", fontsize=10)
        ax_coef_full.set_title("GLM Coefficients\n(Full Model)", fontsize=10)
        ax_coef_full.axvline(0, color='white', lw=1, linestyle='--', alpha=0.7)
        ax_coef_full.spines["top"].set_visible(False)
        ax_coef_full.spines["right"].set_visible(False)
        ax_coef_full.set_ylim(1, 0)  # Reversed: 1 at bottom, 0 at top (matches design matrix)

        # ---- REDUCED MODEL COEFFICIENTS ----
        # Get coefficients from reduced model
        res_reduced = df_reduced[df_reduced["neuron_id"] == nid].iloc[0]
        predictors_reduced = res_reduced.predictors
        coef_reduced = res_reduced.coef
        coef_reduced = np.array(ast.literal_eval(coef_reduced))

        # Plot using the same style
        n_features_reduced = len(coef_reduced)

        # Reverse order to match design matrix
        coef_vector_reduced = coef_reduced[::-1]
        feature_names_reduced = predictors_reduced[::-1]

        # Create y-positions (same logic as full model)
        total_spacing_reduced = spacing_fraction * (n_features_reduced - 1)
        available_height = 1.0 - total_spacing_reduced
        subplot_height_reduced = available_height / n_features_reduced

        y_positions_reduced = []
        for i in range(n_features_reduced):
            y_top = 1.0 - i * (subplot_height_reduced + spacing_fraction)
            y_center = y_top - subplot_height_reduced / 2
            y_positions_reduced.append(y_center)
        y_positions_reduced = np.array(y_positions_reduced)

        bar_height_reduced = subplot_height_reduced * 0.8

        # Color code by predictor type
        colors_reduced = []
        for fname in feature_names_reduced:
            if 'whisker' in fname:
                colors_reduced.append('forestgreen')
            elif 'auditory' in fname:
                colors_reduced.append('mediumblue')
            elif 'jaw' in fname or 'lick' in fname:
                colors_reduced.append('orange')
            elif 'reward' in fname:
                colors_reduced.append('purple')
            else:
                colors_reduced.append('gray')

        ax_coef_reduced.barh(y_positions_reduced, coef_vector_reduced, height=bar_height_reduced,
                             color=colors_reduced, alpha=0.7, edgecolor='white', linewidth=0.5)

        ax_coef_reduced.set_yticks(y_positions_reduced)
        ax_coef_reduced.set_yticklabels(feature_names_reduced, fontsize=7)
        ax_coef_reduced.set_xlabel("Coefficient", fontsize=10)
        ax_coef_reduced.set_title("GLM Coefficients\n(Reduced Model)", fontsize=10)
        ax_coef_reduced.axvline(0, color='white', lw=1, linestyle='--', alpha=0.7)
        ax_coef_reduced.spines["top"].set_visible(False)
        ax_coef_reduced.spines["right"].set_visible(False)
        ax_coef_reduced.set_ylim(1, 0)  # Reversed: 1 at bottom, 0 at top (matches design matrix)

    axes[min(2, len(trial_types)-1)].legend(fontsize=8)
    plt.tight_layout()
    # plt.savefig(os.path.join(output_folder, f"{name}.png"))
    putils.save_figure_with_options(fig, file_formats=['png', 'pdf'],
                                        filename=name,
                                        output_dir=output_folder, dark_background=True)
    plt.close(fig)




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
    df_full = df[(df["model_name"] == "full")]
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

        df_reduced = df[(df["model_name"] != "full") ]
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



def process_neuron(neuron_id, model, trials_df, output_folder, df_full_slim, df_reduced_slim, type = 'Normal', bin_sizes = (0.1,0.1)):
    """
    Worker function to plot a single neuron.
    """
    df_full_neuron = df_full_slim[df_full_slim['neuron_id'] == neuron_id]
    df_reduced_neuron = df_reduced_slim[df_reduced_slim['neuron_id'] == neuron_id]

    if df_full_neuron.empty or df_reduced_neuron.empty:
        return

    plt.ioff()  # non-interactive
    if type == 'Normal':
        plot_model_comparison(
            [neuron_id],
            df_full_neuron,
            df_reduced_neuron,
            trials_df,
            output_folder,
            name=str(neuron_id),
            reduced_model=model,
            bin_sizes= bin_sizes,
            zscore=False
        )
    elif type == 'previous_whisker':
        plot_by_last_whisker_outcome(
            [neuron_id], df_full_neuron, df_reduced_neuron, trials_df, output_folder, name=str(neuron_id),
            reduced_model=model, bin_size=0.1, zscore=False
        )
    elif type =='last_5':
        plot_by_recent_whisker_history(
            [neuron_id], df_full_neuron, df_reduced_neuron, trials_df, output_folder, name=str(neuron_id),
            reduced_model=model, bin_size=0.1, zscore=False, history_len=5
        )
    elif type == 'session_progression':
        plot_by_session_quartiles(
        [neuron_id], df_full_neuron, df_reduced_neuron, trials_df, output_folder,  name=str(neuron_id),
        reduced_model=model, bin_size=0.1, zscore=False
        )

import os
import multiprocessing

def plot_predictions_with_reduced_models_parallel(df_full_slim, df_reduced_slim, trials_df, type, output_folder_base, bin_sizes = (0.1,0.1)):
    n_jobs = max(1, multiprocessing.cpu_count() - 1)

    for model in df_reduced_slim['model_name'].unique():
        # if model == 'full':
        #     continue
        print(model)
        df_full_slim_model = df_reduced_slim[df_reduced_slim['model_name'] == model]

        output_folder = os.path.join(output_folder_base, model)
        os.makedirs(output_folder, exist_ok=True)
        neuron_ids = df_full_slim['neuron_id'].unique()
        Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
            delayed(process_neuron)(
                neuron_id, model, trials_df, output_folder, df_full_slim, df_full_slim_model, type = type, bin_sizes = bin_sizes
            )
            for neuron_id in neuron_ids
        )


def plot_neuron_kernels_avg_with_responses(neuron_id, glm_full_df, kernels, trials_df, output_folder,lags, bin_size=0.1, git_handle = None):
    """
    Plot kernels for one neuron alongside average responses and predictions.
    Uses SEM across folds (not across trials).
    """
    # -------------------
    # KERNELS (per fold)
    # -------------------
    coefs_full_str = glm_full_df.loc[glm_full_df['neuron_id'] == neuron_id, 'coef'].tolist()
    coefs_full = [np.array(ast.literal_eval(c)) for c in coefs_full_str]
    coefs_full = np.stack(coefs_full, axis=0)   # shape (n_folds, n_predictors)

    # Predictors list for indexing kernels
    predictors = glm_full_df.loc[glm_full_df['neuron_id'] == neuron_id, 'predictors'].iloc[0]
    

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
        if git_handle is None:
            ax_k.plot(lags, mean_kernel, color="blue")
            ax_k.fill_between(lags, mean_kernel - sem_kernel, mean_kernel + sem_kernel, color="blue", alpha=0.3)
        else:
            if predictor in ['whisker_stim','auditory_stim', 'whisker_hits_stim_0', 'whisker_hits_stim_1', 'whisker_misses_stim0', 'whisker_misses_stim1', 'auditory_stim0', 'auditory_stim1']:

                lags = [-0.1, -0.05, 0, 0.05, 0.1,0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
                ax_k.plot(lags, mean_kernel, color="blue")
                ax_k.fill_between(lags, mean_kernel - sem_kernel, mean_kernel + sem_kernel, color="blue", alpha=0.3)
            if predictor == 'piezo_reward':
                lags = np.arange(0,1,0.05)
                ax_k.plot(lags, mean_kernel, color="blue")
                ax_k.fill_between(lags, mean_kernel - sem_kernel, mean_kernel + sem_kernel, color="blue", alpha=0.3)
            if predictor == 'jaw_onset':
                lags = [-0.5,-0.45, -0.4,-0.35, -0.3,-0.25, -0.2,-0.15, -0.1, -0.05]
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
        if predictor in [ "whisker_stim", 'whisker_hits_stim_0', 'whisker_hits_stim_1', 'whisker_misses_stim0', 'whisker_misses_stim1' ]:
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

        elif predictor in ["auditory_stim",'auditory_stim0', 'auditory_stim1' ]:
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

        elif predictor == "jaw_onset":
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
                                   lags=None, area_groups=None, area_colors=None, n_cols=3,  threshold = None, git_handle = None):
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
        print(kernel)
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

                    predictors =row['predictors']
                    print(predictors)
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
            if git_handle is None:
                ax.plot(lags, mean_kernel, color=color)
                ax.fill_between(lags, mean_kernel - sem_kernel, mean_kernel + sem_kernel, color=color, alpha=0.3)
            elif git_handle in ['74987e2', 'b394470', 'a784830']:
                if kernel.startswith('whisker_stim') or kernel == 'auditory_stim':
                    lags = [-0.1, 0, 0.1, 0.2, 0.3]
                    ax.plot(lags, mean_kernel, color = color)
                    ax.fill_between(lags, mean_kernel - sem_kernel, mean_kernel + sem_kernel, color=color, alpha=0.3)
                if kernel == 'piezo_reward':
                    lags = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
                    ax.plot(lags, mean_kernel, color=color)
                    ax.fill_between(lags, mean_kernel - sem_kernel, mean_kernel + sem_kernel, color=color, alpha=0.3)
                if kernel == 'jaw_onset':
                    lags = [-0.5, -0.4, -0.3, -0.2, -0.1]
                    ax.plot(lags, mean_kernel, color=color)
                    ax.fill_between(lags, mean_kernel - sem_kernel, mean_kernel + sem_kernel, color=color, alpha=0.3)

            else:
                if kernel == 'whisker_stim' or kernel == 'auditory_stim':
                    lags = [-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
                    ax.plot(lags, mean_kernel, color = color)
                    ax.fill_between(lags, mean_kernel - sem_kernel, mean_kernel + sem_kernel, color=color, alpha=0.3)
                if kernel == 'piezo_reward':
                    lags = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
                    ax.plot(lags, mean_kernel, color=color)
                    ax.fill_between(lags, mean_kernel - sem_kernel, mean_kernel + sem_kernel, color=color, alpha=0.3)
                if kernel == 'jaw_onset':
                    lags = [-0.5, -0.4, -0.3, -0.2, -0.1]
                    ax.plot(lags, mean_kernel, color=color)
                    ax.fill_between(lags, mean_kernel - sem_kernel, mean_kernel + sem_kernel, color=color, alpha=0.3)

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
        putils.save_figure_with_options(fig, file_formats=['png', 'pdf'],
                                        filename=name,
                                        output_dir=output_folder)
        plt.close(fig)

    return


def plot_average_kernels_by_region_new(df, output_folder, kernels_to_plot,
                                   lags=None, area_groups=None, area_colors=None, n_cols=3,  threshold = None, git_handle = None):
    """
    Plot average kernels across neurons grouped by area_acronym_custom (regions),
    one figure per kernel. Regions are colored by area group, ordered by area_groups.

    For each neuron (per mouse), coefficients are averaged across folds first,
    then mean ± SEM is computed across neurons in the region.

    Kernel lengths are automatically detected from predictor names in the format:
    'kernel_name_t+0.00s', 'kernel_name_t-0.50s', etc.

    Parameters
    ----------
    df : pd.DataFrame
        Must include columns: ['mouse_id','neuron_id','area_acronym_custom','coef','predictors','fold']
    output_folder : str
    kernels_to_plot : list of str
    lags : np.ndarray, optional
        If provided, overrides auto-detected lags. Otherwise lags are inferred from predictor names.
    area_groups : dict, optional
    area_colors : dict, optional
    n_cols : int, number of subplot columns
    git_handle : str, optional
        Deprecated parameter, kept for backward compatibility but no longer used for lag detection.
    """

    def extract_lags_from_predictors(predictors, kernel_name):
        """
        Extract lag values from predictor names like 'kernel_name_t+0.00s', 'kernel_name_t-0.50s'.

        Parameters
        ----------
        predictors : list of str
            List of predictor names
        kernel_name : str
            The kernel name to search for

        Returns
        -------
        lags : np.ndarray or None
            Array of unique lag values in seconds, sorted in ascending order. None if no matching predictors found.
        """
        import re
        lag_values = []

        # Pattern to match exact kernel name followed by _t and time
        pattern = re.compile(rf"^{re.escape(kernel_name)}_t([+-])(\d+\.\d+)s$")

        for pred in predictors:
            match = pattern.match(pred)
            if match:
                sign = 1 if match.group(1) == '+' else -1
                value = float(match.group(2))
                lag_values.append(sign * value)

        if not lag_values:
            return None

        # Sort, remove duplicates, and return as numpy array
        return np.array(sorted(set(lag_values)))

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
        # Auto-detect kernel lags – search one row per mouse until the kernel is found,
        # because different sessions may have different predictor sets.
        if lags is None:
            kernel_lags = None
            for _, probe_row in df.drop_duplicates(subset=['mouse_id']).iterrows():
                kernel_lags = extract_lags_from_predictors(probe_row['predictors_full'], kernel)
                if kernel_lags is not None:
                    break
            if kernel_lags is None:
                print(f"[plot_average_kernels_by_region_new] Skipping '{kernel}': not present in any session.")
                plt.close(fig)
                continue
        else:
            kernel_lags = lags
        for ax, region in zip(axes, ordered_regions):
            region_df = df[df['area_acronym_custom'] == region]

            # Group by mouse_id and neuron_id
            neuron_groups = region_df.groupby(['mouse_id', 'neuron_id'])
            kernels_list = []

            for (_, _), grp in neuron_groups:

                # Collect coefficients for this neuron across folds
                neuron_kernels = []

                for _, row in grp.iterrows():
                    coefs_list = row['coef_full_mean']
                    predictors = row['predictors_full']

                    # Extract predictor-coefficient pairs for this kernel, with lag values
                    # Use exact matching pattern to avoid matching similar kernel names
                    import re
                    pattern = re.compile(rf"^{re.escape(kernel)}_t([+-])(\d+\.\d+)s$")
                    kernel_data = []

                    for i, p in enumerate(predictors):
                        match = pattern.match(p)
                        if match:
                            sign = 1 if match.group(1) == '+' else -1
                            lag_value = sign * float(match.group(2))
                            kernel_data.append((lag_value, coefs_list[i]))

                    # Sort by lag value and extract coefficients in order
                    if kernel_data:
                        kernel_data.sort(key=lambda x: x[0])
                        sorted_coefs = np.array([coef for _, coef in kernel_data])
                        neuron_kernels.append(sorted_coefs)

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

            # Ensure lags match kernel length
            if len(kernel_lags) != len(mean_kernel):
                print(f"Warning: lag length ({len(kernel_lags)}) doesn't match kernel length ({len(mean_kernel)}) for {kernel} in {region}")
                kernel_lags = np.linspace(kernel_lags[0], kernel_lags[-1], len(mean_kernel))

            ax.plot(kernel_lags, mean_kernel, color=color)
            ax.fill_between(kernel_lags, mean_kernel - sem_kernel, mean_kernel + sem_kernel, color=color, alpha=0.3)

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
        putils.save_figure_with_options(fig, file_formats=['png', 'pdf'],
                                        filename=name,
                                        output_dir=output_folder)
        plt.close(fig)

    return


def plot_all_kernels_by_region(df, output_folder, kernels_to_plot,
                                lags=None, area_groups=None, area_colors=None,
                                kernel_colors=None, n_cols=3, threshold=None):
    """
    Plot all kernels for each region in a single subplot.
    One figure with multiple subplots (one per region), each showing all kernels overlaid.

    Parameters
    ----------
    df : pd.DataFrame
        Must include columns: ['mouse_id','neuron_id','area_acronym_custom','coef','predictors','fold']
    output_folder : str
    kernels_to_plot : list of str
        List of kernel names to plot
    lags : np.ndarray, optional
        If provided, overrides auto-detected lags. Otherwise lags are inferred from predictor names.
    area_groups : dict, optional
        Grouping of brain areas
    area_colors : dict, optional
        Colors for area groups (used for subplot titles/borders)
    kernel_colors : dict, optional
        Colors for each kernel. If None, will use default color cycle.
    n_cols : int
        Number of subplot columns
    threshold : float, optional
        Threshold value to include in filename
    """

    def extract_lags_from_predictors(predictors, kernel_name):
        """Extract lag values from predictor names."""
        import re
        lag_values = []
        pattern = re.compile(rf"^{re.escape(kernel_name)}_t([+-])(\d+\.\d+)s$")

        for pred in predictors:
            match = pattern.match(pred)
            if match:
                sign = 1 if match.group(1) == '+' else -1
                value = float(match.group(2))
                lag_values.append(sign * value)

        if not lag_values:
            return None

        return np.array(sorted(set(lag_values)))

    # Set default kernel colors if not provided
    if kernel_colors is None:
        default_colors = plt.cm.tab10(np.linspace(0, 1, len(kernels_to_plot)))
        kernel_colors = {k: default_colors[i] for i, k in enumerate(kernels_to_plot)}

    # Map region -> color for borders/titles
    region_to_color = {}
    if area_groups and area_colors:
        for group_name, areas in area_groups.items():
            for area in areas:
                region_to_color[area] = area_colors[group_name]

    def get_region_color(region_name):
        return region_to_color.get(region_name, 'gray')

    # Order regions
    ordered_regions = []
    if area_groups:
        for group_name, areas in area_groups.items():
            for area in areas:
                if area in df['area_acronym_custom'].values:
                    ordered_regions.append(area)
    else:
        ordered_regions = sorted(df['area_acronym_custom'].unique())

    # Create figure with subplots (one per region)
    n_rows = math.ceil(len(ordered_regions) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)

    for ax, region in zip(axes, ordered_regions):
        region_df = df[df['area_acronym_custom'] == region]

        # Plot each kernel on this subplot
        for kernel in kernels_to_plot:
            # Auto-detect kernel lags
            kernel_lags = None
            if lags is None:
                first_row = region_df.iloc[0]
                predictors = first_row['predictors_full']
                kernel_lags = extract_lags_from_predictors(predictors, kernel)

                if kernel_lags is None:
                    continue
            else:
                kernel_lags = lags

            # Group by mouse_id and neuron_id
            neuron_groups = region_df.groupby(['mouse_id', 'neuron_id'])
            kernels_list = []

            for (_, _), grp in neuron_groups:
                neuron_kernels = []

                for _, row in grp.iterrows():
                    coefs_list = row['coef_full_mean']
                    predictors = row['predictors_full']

                    # Extract predictor-coefficient pairs for this kernel
                    import re
                    pattern = re.compile(rf"^{re.escape(kernel)}_t([+-])(\d+\.\d+)s$")
                    kernel_data = []

                    for i, p in enumerate(predictors):
                        match = pattern.match(p)
                        if match:
                            sign = 1 if match.group(1) == '+' else -1
                            lag_value = sign * float(match.group(2))
                            kernel_data.append((lag_value, coefs_list[i]))

                    if kernel_data:
                        kernel_data.sort(key=lambda x: x[0])
                        sorted_coefs = np.array([coef for _, coef in kernel_data])
                        neuron_kernels.append(sorted_coefs)

                if neuron_kernels:
                    neuron_kernels = np.stack(neuron_kernels)
                    kernels_list.append(neuron_kernels.mean(axis=0))

            if len(kernels_list) == 0:
                continue

            kernels_stack = np.stack(kernels_list)
            mean_kernel = kernels_stack.mean(axis=0)
            sem_kernel = kernels_stack.std(axis=0, ddof=1) / np.sqrt(kernels_stack.shape[0])

            # Ensure lags match kernel length
            if len(kernel_lags) != len(mean_kernel):
                kernel_lags = np.linspace(kernel_lags[0], kernel_lags[-1], len(mean_kernel))

            # Plot this kernel
            color = kernel_colors.get(kernel, 'gray')
            ax.plot(kernel_lags, mean_kernel, color=color, label=kernel, linewidth=2)
            ax.fill_between(kernel_lags, mean_kernel - sem_kernel, mean_kernel + sem_kernel,
                          color=color, alpha=0.2)

        # Style the subplot
        n_neurons = len(region_df.groupby(['mouse_id', 'neuron_id']))
        ax.set_title(f"{region} (n={n_neurons})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Lag (s)", fontsize=9)
        ax.set_ylabel("Coefficient", fontsize=9)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, loc='best')

    # Hide unused axes
    for ax in axes[len(ordered_regions):]:
        ax.set_visible(False)

    plt.suptitle("All kernels by region", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    name = 'all_kernels_by_region'
    if threshold:
        name += f'_threshold_{threshold}'
    putils.save_figure_with_options(fig, file_formats=['png', 'pdf'],
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

def load_models_one_mouse(mouse, models_path, git_version):
    try:
        files = [f for f in os.listdir(models_path) if f.endswith('_results.parquet')]
        pattern = rf'^{git_version}_model_(full|reduced|added)_fold(\d+)_results\.parquet'

        def _load(file):
            match = re.match(pattern, file)
            if not match:
                return None
            model_type, fold = match.group(1), match.group(2)

            try:
                df = post_hoc_load_model_results(file.split("_results")[0], models_path)
                if df is None or df.empty:
                    print(f"[WARNING] Empty DataFrame in file {file} for mouse {mouse}. Skipping.")
                    return None
                df['git_version'] = git_version
                df['fold'] = fold
                df['model_type'] = model_type
                df['mouse_id'] = mouse
                        # Normalize prediction column names:
                has_y_pred = 'y_pred' in df.columns
                has_y_test_pred = 'y_test_pred' in df.columns

                if not has_y_pred and has_y_test_pred:
                    # Only y_test_pred is present: rename to y_pred
                    df = df.rename(columns={'y_test_pred': 'y_pred'})
                return df
            except Exception as e:
                print(f"[ERROR] Failed to load {file} for mouse {mouse}: {e}")
                return None

        dfs = Parallel(n_jobs=-1)(
            delayed(_load)(file) for file in files
        )

        all_results = [df for df in dfs if df is not None]

        if not all_results:
            print(f"[WARNING] No valid model results found for mouse {mouse}. Skipping.")
            return None

        df_all = pd.concat(all_results, ignore_index=True)
        return df_all

    except Exception as e:
        print(f"[CRITICAL] load_models failed for mouse {mouse}: {e}")
        return None


def load_single_file(file_path, mouse, git_version, model_type, fold):
    """Load a single parquet file with minimal overhead."""
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
        if df.empty:
            print(f"[WARNING] Empty DataFrame in {file_path}")
            return None
        df['git_version'] = git_version
        df['fold'] = fold
        df['model_type'] = model_type
        df['mouse_id'] = mouse
        # Normalize prediction column names:
        has_y_pred = 'y_pred' in df.columns
        has_y_test_pred = 'y_test_pred' in df.columns

        if not has_y_pred and has_y_test_pred:
            # Only y_test_pred is present: rename to y_pred
            df = df.rename(columns={'y_test_pred': 'y_pred'})
        return df
    except Exception as e:
        print(f"[ERROR] reading {file_path}: {e}")
        return None


def load_models_optimized(mice, output_path, git_version):
    """Collect all file paths first, then parallelize the I/O."""
    pattern = re.compile(rf'^{git_version}_model_(full|reduced|added)_fold(\d+)_results\.parquet$')

    # Collect all file paths upfront (fast, no I/O)
    tasks = []
    for mouse in mice:
        models_path = os.path.join(output_path, mouse, "whisker_0", "unit_glm", "models")
        try:
            files = os.listdir(models_path)
        except FileNotFoundError:
            print(f"[WARNING] Path not found for mouse {mouse}: {models_path}")
            continue
        except Exception as e:
            print(f"[ERROR] accessing {models_path}: {e}")
            continue

        for file in files:
            match = pattern.match(file)
            if match:
                model_type, fold = match.groups()
                file_path = os.path.join(models_path, file)
                tasks.append((file_path, mouse, git_version, model_type, fold))

    if not tasks:
        print(f"[ERROR] No files found matching pattern for git_version: {git_version}")
        print(f"Pattern: ^{git_version}_model_(full|reduced|added)_fold(\\d+)_results\\.parquet$")
        # Debug: show what files exist for first mouse
        if len(mice) > 0:
            sample_path = os.path.join(output_path, mice[0], "whisker_0", "unit_glm", "models")
            if os.path.exists(sample_path):
                sample_files = [f for f in os.listdir(sample_path) if f.endswith('.parquet')]
                print(f"Sample files in {mice[0]}: {sample_files[:5]}")
        return pd.DataFrame()

    print(f"Found {len(tasks)} files to load")

    # # Parallelize all file reading at once
    # results = Parallel(n_jobs=20, batch_size=50)(
    #     delayed(load_single_file)(*task)
    #     for task in tqdm(tasks, desc="Loading all models")
    # )
    results = []

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(load_single_file, *task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading all models"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                task = futures[future]
                print(f"[ERROR] Failed loading {task}: {e}")
                results.append(None)

    valid = [r for r in results if r is not None]

    if not valid:
        print(f"[ERROR] No valid dataframes loaded from {len(tasks)} files")
        return pd.DataFrame()

    print(f"Successfully loaded {len(valid)} out of {len(tasks)} files")
    return pd.concat(valid, ignore_index=True)


def load_models_multiple_versions(mice, output_path, git_versions):
    """
    Load models for multiple git versions.

    :param mice: list of mouse IDs
    :param output_path: base path to model directories
    :param git_versions: list of git versions to load
    :return: concatenated pd.DataFrame with all git versions
    """
    all_results = []

    for git_version in git_versions:
        print(f"\n[INFO] Loading models for git version: {git_version}")
        pattern = re.compile(rf'^{git_version}_model_(full|reduced|added)_fold(\d+)_results\.parquet$')
        tasks = []

        for mouse in mice:
            models_path = os.path.join(output_path, mouse, "whisker_0", "unit_glm", "models")
            try:
                files = os.listdir(models_path)
            except FileNotFoundError:
                print(f"[WARNING] Path not found for mouse {mouse}: {models_path}")
                continue
            except Exception as e:
                print(f"[ERROR] accessing {models_path}: {e}")
                continue

            for file in files:
                match = pattern.match(file)
                if match:
                    model_type, fold = match.groups()
                    file_path = os.path.join(models_path, file)
                    tasks.append((file_path, mouse, git_version, model_type, fold))

        if not tasks:
            print(f"[WARNING] No files found matching pattern for git_version: {git_version}")
            continue

        print(f"[INFO] Found {len(tasks)} files for git version {git_version}")

        # results = Parallel(n_jobs=24, batch_size=50)(
        #     delayed(load_single_file)(*task)
        #     for task in tqdm(tasks, desc=f"Loading models {git_version}")
        # )

        results = []

        with ProcessPoolExecutor(max_workers=24) as executor:
            futures = {executor.submit(load_single_file, *task): task for task in tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading all models"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    task = futures[future]
                    print(f"[ERROR] Failed loading {task}: {e}")
                    results.append(None)

        valid = [r for r in results if r is not None]

        if not valid:
            print(f"[WARNING] No valid dataframes loaded for git_version: {git_version}")
            continue

        df_git = pd.concat(valid, ignore_index=True)
        df_git['git_version'] = git_version  # ensure git_version column exists
        all_results.append(df_git)

    if not all_results:
        print("[ERROR] No models loaded for any git version")
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)




def combine_ephys_nwb(nwb_list,day_to_analyze =0, max_workers=24, git_version = None):
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
        futures = {executor.submit(process_single_nwb, nwb, day_to_analyze = day_to_analyze, git_version= git_version): nwb for nwb in nwb_list}

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

def process_single_nwb(nwb, day_to_analyze = 0, git_version = None):
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

        trial_table = trial_table[(trial_table['context'] != 'passif') & (trial_table['perf'] != 6)].copy()

        if git_version in ['4227ca6', 'b394470', '74987e2', '15127ae',  '935b6e1', '4802e47', 'a784830', 'c2eb670', 'f849441', '64beadc', '4465999', '55a7b9a', 'acbce87', '1b14083']:
            trial_table = load_perf_blocks(trial_table,mouse_id)
            trial_table = trial_table.reset_index(drop=True)
        # passive trials were not modeled so we drop them
        trial_table["behav_type"] = trial_table.apply(classify_trial, axis=1)
        trial_table = trial_table.reset_index(drop=True)
        unit_table['mouse_id'] = mouse_id
        unit_table = convert_electrode_group_object_to_columns(unit_table)

        # Only keep the neurons fitted for the glms
        unit_table = allen.process_allen_labels(unit_table, subdivide_areas=False)
        unit_table = unit_table[unit_table['bc_label'] == 'good']
        unit_table = unit_table[unit_table['firing_rate'].astype(float).ge(2.0)]
        unit_table = unit_table[~unit_table['ccf_acronym'].isin(allen_utils.get_excluded_areas())]
        unit_table['og_unit_table_id'] = unit_table.index
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
def plot_trial_grid_predictions(results_df, trial_table, neuron_id, bin_size, output_folder):
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
        ax.set_title('Trial {} {}'.format(row['trial_id'], row['behav_type']), fontsize=10)
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

    title = (f'GLM predictions on test trials - unit {neuron_id}, {results_df_sub["area_acronym_custom"].iloc[0]}, '
             f'$R$= {results_df_sub["test_corr"].values[0]:.2f}')
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.align_ylabels()
    fig.tight_layout()
    fig.align_ylabels()
    name = f'Neuron_{neuron_id}'

    putils.save_figure_with_options(fig, file_formats=['png', 'pdf', 'svg'],
                                    filename=name,
                                    output_dir=output_folder, dark_background=True)
    plt.close('all')
    return


def plot_trial_concatenated_predictions(results_df, trial_table, neuron_id, bin_size, output_folder,
                                        trials_per_page=50, window_bounds_sec=(-1, 2)):
    """
    Plot all test trials concatenated on a single time axis per page.
    Vertical lines at each stimulus onset are colored by trial type; thin grey dashed lines
    mark trial boundaries. Observed spikes (black step) and GLM prediction (red line) are
    overlaid on the same axis. Splits across multiple figures when there are more trials than
    trials_per_page.
    """
    TRIAL_COLORS = {
        'whisker_trial': 'darkorange',
        'auditory_trial': 'mediumblue',
        'no_stim_trial': 'dimgray',
    }
    trial_duration = window_bounds_sec[1] - window_bounds_sec[0]
    stim_offset_bins = int(round(-window_bounds_sec[0] / bin_size))  # bins from trial start to t=0

    results_df_sub = results_df[results_df['neuron_id'] == neuron_id]
    y_test = np.array(ast.literal_eval(results_df_sub['y_test'].values[0]))
    y_pred = np.array(ast.literal_eval(results_df_sub['y_pred'].values[0]))
    n_bins = results_df_sub['n_bins'].values[0]

    n_trials = y_pred.shape[0] // n_bins
    y_test = y_test.reshape(n_trials, n_bins)
    y_pred = y_pred.reshape(n_trials, n_bins)

    test_trial_ids = np.array(ast.literal_eval(results_df_sub['test_trials'].values[0]))
    order = np.argsort(test_trial_ids)
    y_test = y_test[order]
    y_pred = y_pred[order]

    trials_test_df = trial_table[trial_table['trial_id'].isin(test_trial_ids)]
    trials_test_df = trials_test_df.sort_values('trial_id').reset_index(drop=True)

    neuron_label = (f'unit {neuron_id}, {results_df_sub["area_acronym_custom"].iloc[0]}, '
                    f'$R$= {results_df_sub["test_corr"].values[0]:.2f}')

    n_pages = math.ceil(n_trials / trials_per_page)

    for page in range(n_pages):
        sl = slice(page * trials_per_page, (page + 1) * trials_per_page)
        page_trials = trials_test_df.iloc[sl]
        yt_page = y_test[sl]
        yp_page = y_pred[sl]
        n_page = len(page_trials)

        # Build concatenated traces
        concat_test = yt_page.ravel()
        concat_pred = yp_page.ravel()
        total_bins = n_page * n_bins

        # x-axis in seconds from the start of the first trial on this page
        time_concat = (np.arange(total_bins) + 0.5) * bin_size

        fig, ax = plt.subplots(figsize=(max(12, n_page * 0.4), 4))
        putils.remove_top_right_frame(ax)

        ax.step(time_concat, concat_test, where='mid', color='black', alpha=0.85, linewidth=0.5, label='Observed')
        ax.plot(time_concat, concat_pred, color='forestgreen', linewidth=0.8, label='Predicted')

        legend_handles = {}
        for i, (_, row) in enumerate(page_trials.iterrows()):
            t_start = i * n_bins * bin_size
            t_boundary = t_start
            t_stim = t_start + stim_offset_bins * bin_size

            # Grey dashed line at trial boundary (skip first — it's the axis edge)
            if i > 0:
                ax.axvline(t_boundary, color='lightgray', linewidth=0.6, linestyle='--', zorder=0)

            # Colored line at stimulus onset
            trial_type = row.get('trial_type', 'no_stim_trial')
            color = TRIAL_COLORS.get(trial_type, 'dimgray')
            ax.axvline(t_stim, color=color, linewidth=1.0, linestyle='-', alpha=0.7, zorder=1)

            if trial_type not in legend_handles:
                legend_handles[trial_type] = plt.Line2D([], [], color=color, linewidth=1.5,
                                                        label=trial_type.replace('_trial', ''))

        # x-ticks every 10 trials
        tick_trials = np.arange(0, n_page, max(1, n_page // 10))
        ax.set_xticks(tick_trials * n_bins * bin_size + stim_offset_bins * bin_size)
        ax.set_xticklabels([str(page_trials.iloc[i]['trial_id']) for i in tick_trials
                            if i < len(page_trials)], fontsize=7, rotation=45)
        ax.set_xlabel('Trial ID', fontsize=10)
        ax.set_ylabel('Spikes / bin', fontsize=10)
        ax.set_xlim(0, total_bins * bin_size)

        trace_handles = [
            plt.Line2D([], [], color='black', linewidth=1.0, label='Observed'),
            plt.Line2D([], [], color='forestgreen', linewidth=1.0, label='Predicted'),
        ]
        all_handles = trace_handles + list(legend_handles.values())
        ax.legend(handles=all_handles, fontsize=8, loc='upper right', framealpha=0.7)

        page_suffix = f' (page {page + 1}/{n_pages})' if n_pages > 1 else ''
        ax.set_title(f'GLM predictions — {neuron_label}{page_suffix}', fontsize=11)

        fig.tight_layout()
        name = f'Neuron_{neuron_id}_concat' if n_pages == 1 else f'Neuron_{neuron_id}_concat_p{page + 1}'
        putils.save_figure_with_options(fig, file_formats=['png', 'pdf', 'svg'],
                                        filename=name,
                                        output_dir=output_folder, dark_background=True)
        plt.close('all')


def plot_trial_with_design_matrix_and_weights_predictions(results_df, trial_table, neuron_id, bin_size, output_folder):
    """
    Plot predictions for a single neuron across trials in a grid format.
    :param results_df: DataFrame with model results
    :param trial_table: DataFrame with trial information
    :param neuron_id: int, ID of the neuron to plot
    :param bin_size: float, size of time bin in seconds
    """

    git = results_df['git_version'].iloc[0]
    data_path = r"M:\analysis\Myriam_Hamon\combined_results\AB131\whisker_0\unit_glm/"
    data_path = os.path.join(data_path, str(git), 'data.pkl')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    X = data['input']
    spikes = data['output']
    feature_names = data['feature_names']

    # Get neuron results
    results_df_sub = results_df[results_df['neuron_id'] == neuron_id]
    y_test = results_df_sub['y_test'].values[0]
    y_pred = results_df_sub['y_pred'].values[0]
    n_bins = results_df_sub['n_bins'].values[0]
    coef = results_df_sub['coef'].values[0]
    y_test = np.array(ast.literal_eval(y_test))
    y_pred = np.array(ast.literal_eval(y_pred))
    coef = np.array(ast.literal_eval(coef))

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
    trials_test_df = trials_test_df.iloc[:n_trials]


    for idx, row in trials_test_df.iterrows():

        # Create figure
        fig, axs = plt.subplots(1, 3, figsize=(18, 12), sharex=False, sharey=False)
        axs = axs.flatten()

        ax = axs[0]
        X_trial = X[:,row['trial_id'],:]
        X_trial = X_trial.reshape((-1, n_bins))


        window_bounds_sec = (-1, 2)
        time_stim = 0
        time = np.linspace(window_bounds_sec[0] + bin_size/2,
                    window_bounds_sec[1] - bin_size/2,
                    n_bins)
        
        plot_design_matrix_into_axis(ax, X_trial, feature_names, time)


        ax = axs[1]

        plot_coefficients_into_axis(ax, coef, feature_names)

        ax = axs[2]
        ax.set_title('Trial {} {}'.format(row['trial_id'], row['behav_type']), fontsize=10)
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

        title = (f'GLM predictions on test trials - unit {neuron_id}, {results_df_sub["area_acronym_custom"].iloc[0]}, '
                f'$R$= {results_df_sub["test_corr"].values[0]:.2f}')
        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        fig.align_ylabels()
        fig.tight_layout()
        fig.align_ylabels()
        name = f'Neuron_{neuron_id}_trial_{row["trial_id"]}'

        putils.save_figure_with_options(fig, file_formats=['png', 'pdf'],
                                        filename=name,
                                        output_dir=output_folder, dark_background=True)
        plt.close('all')
    return
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_coefficients_into_axis(ax, coef_vector, feature_names):
    """
    Plot GLM coefficients in a single axis as *horizontal* bars,
    with spacing that matches plot_design_matrix_into_axis.
    """
    n_features = len(coef_vector)
    
    # Reverse order to match design matrix (top to bottom)
    coef_vector = coef_vector[::-1]
    feature_names = feature_names[::-1]
    
    # Match the spacing from design matrix plot
    spacing_fraction = 0.02
    total_spacing = spacing_fraction * (n_features - 1)
    available_height = 1.0 - total_spacing
    subplot_height = available_height / n_features
    
    # Calculate y-positions (centers of each row) - same order as design matrix
    y_positions = []
    for i in range(n_features):
        y_top = 1.0 - i * (subplot_height + spacing_fraction)
        y_center = y_top - subplot_height / 2
        y_positions.append(y_center)
    
    y_positions = np.array(y_positions)
    
    # Basic cleanup
    putils.remove_top_right_frame(ax)
    
    # Horizontal bar plot at specific positions
    bar_height = subplot_height * 0.8  # 80% of row height for bars
    ax.barh(y_positions, coef_vector, height=bar_height, color='black', alpha=0.8)
    
    # Labels - same order as design matrix (top to bottom)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_xlabel("Coefficient value", fontsize=10)
    ax.set_title("GLM coefficients", fontsize=10)
    
    # Set y-axis limits to match design matrix
    ax.set_ylim(1, 0)  # Reversed: 1 at bottom, 0 at top (matches design matrix top-to-bottom)
    
    # Zero line (vertical)
    ax.axvline(0, color='gray', lw=1)

def plot_design_matrix_into_axis(parent_ax, X_trial, feature_names, time):
    n_features = X_trial.shape[0]

    parent_ax.axis("off")

    # Control spacing: smaller = less gap between subplots
    spacing_fraction = 0.02  # 2% spacing (reduce this for less gap, 0 for no gap)
    
    # Calculate heights
    total_spacing = spacing_fraction * (n_features - 1)
    available_height = 1.0 - total_spacing
    subplot_height = available_height / n_features
    
    axes = []

    for i in range(n_features):
        # Calculate position from top
        y_top = 1.0 - i * (subplot_height + spacing_fraction)
        
        ax_i = inset_axes(
            parent_ax,
            width="100%",
            height="100%",  # Use 100% and control via bbox_to_anchor
            loc="upper left",
            bbox_to_anchor=(0, y_top - subplot_height, 1, subplot_height),
            bbox_transform=parent_ax.transAxes,
            borderpad=0,
        )
        axes.append(ax_i)

    for i, ax in enumerate(axes):
        values = X_trial[i]
        name = feature_names[i]

        putils.remove_top_right_frame(ax)

        # --------------------------
        # Plot binary or analog trace
        # --------------------------
        uniq = np.unique(values)

        if np.array_equal(uniq, [0]) or np.array_equal(uniq, [0, 1]):
            ax.step(time, values, where='mid', color='black', lw=1.2)
            ax.set_ylim(-0.3, 1.3)
        else:
            ax.plot(time, values, color='black', lw=1.2)
            # Autoscale analog values with small padding
            vmin, vmax = values.min(), values.max()
            if vmin == vmax:
                vmax = vmin + 1e-6  # avoid zero range
            ax.set_ylim(vmin - 0.1 * abs(vmin), vmax + 0.1 * abs(vmax))

        # --------------------------
        # Clean ticks
        # --------------------------
        if i < n_features - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Time (s)", fontsize=10)

        is_binary = np.array_equal(uniq, [0]) or np.array_equal(uniq, [0, 1])

        ax.yaxis.set_ticks_position('right')
        ax.tick_params(axis='y', labelsize=7)

        # --------------------------
        # Regressor name, well outside
        # --------------------------
        ax.text(
            -0.02, 0.5, name,
            transform=ax.transAxes,
            va="center", ha="right",
            fontsize=8
        )

    return axes
def plot_trial_grid_predictions_two_models(results_df1, results_df2, trial_table, neuron_id,
                                           bin_sizes=(0.01, 0.01), output_folder=None, model_names=None):
    """
    Plot predictions for a single neuron across trials in a grid format for two different models.
    
    :param results_df1: DataFrame with results from model 1
    :param results_df2: DataFrame with results from model 2
    :param trial_table: DataFrame with trial information
    :param neuron_id: int, neuron ID
    :param bin_sizes: tuple of float, size of time bin in seconds for each model
    :param model_names: tuple of strings (optional), names for the two models for labeling
    """
    n_rows, n_cols = 5, 5
    trials_to_plot = min(n_rows * n_cols, len(trial_table))
    
    def get_predictions(results_df):
        sub = results_df[results_df['neuron_id'] == neuron_id]
        y_test = np.array(sub['y_test_array'].values[0])
        y_pred = np.array(sub['y_pred_array'].values[0])
        n_bins = sub['n_bins'].values[0]
        n_trials = y_pred.shape[0] // n_bins
        y_test = y_test.reshape(n_trials, n_bins)
        y_pred = y_pred.reshape(n_trials, n_bins)
        test_trial_ids = np.array(ast.literal_eval(sub['test_trials'].values[0]))
        order = np.argsort(test_trial_ids)
        y_test = y_test[order, :]
        y_pred = y_pred[order, :]
        return y_test, y_pred, test_trial_ids, sub['test_corr'].values[0], n_bins
    
    y_test1, y_pred1, test_ids1, corr1, n_bins1 = get_predictions(results_df1)
    y_test2, y_pred2, test_ids2, corr2, n_bins2 = get_predictions(results_df2)
    
    # Align trials
    common_ids = np.intersect1d(test_ids1, test_ids2)
    idx1 = [np.where(test_ids1 == tid)[0][0] for tid in common_ids]
    idx2 = [np.where(test_ids2 == tid)[0][0] for tid in common_ids]

    y_test1, y_pred1 = y_test1[idx1], y_pred1[idx1]
    y_test2, y_pred2 = y_test2[idx2], y_pred2[idx2]
    trials_test_df = trial_table[trial_table['trial_id'].isin(common_ids)].sort_values('trial_id').reset_index(drop=True)
    trials_test_df = trials_test_df.iloc[:trials_to_plot]
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 12), sharex=False, sharey=False)
    axs = axs.flatten()
    
    window_bounds_sec = (-1, 2)
    time_stim = 0

    time1 = np.linspace(window_bounds_sec[0] + bin_sizes[0]/2,
                        window_bounds_sec[1] - bin_sizes[0]/2, n_bins1)
    time2 = np.linspace(window_bounds_sec[0] + bin_sizes[1]/2,
                        window_bounds_sec[1] - bin_sizes[1]/2, n_bins2)
    
    for idx, row in trials_test_df.iterrows():
        ax = axs[idx]
        ax.set_title(f"Trial {row['trial_id']} {row['behav_type']}", fontsize=9)
        putils.remove_top_right_frame(ax)
        ax.set_ylim(0, 10)
        ax.set_ylabel('Spikes', fontsize=10)
        ax.set_yticks([0, 10])
        ax.set_xlabel('Time (s)', fontsize=10)
        
        trial_type = row['trial_type']
        color_map = {'whisker_trial':'forestgreen', 'auditory_trial':'mediumblue', 'no_stim_trial':'k'}
        ax.axvline(time_stim, color=color_map.get(trial_type,'k'), linestyle='-', linewidth=1)
        
        # Plot predictions with their own bin_sizes
        ax.plot(time1, y_pred1[idx], color='red', linewidth=1.5, label=model_names[0] if model_names else 'Model 1')
        ax.plot(time2, y_pred2[idx], color='orange', linewidth=1.5, label=model_names[1] if model_names else 'Model 2')
        ax.step(time1, y_test1[idx], where='mid', color='black', alpha=0.8, linewidth=1.5, label='Actual')
    
    title = f'Neuron {neuron_id} predictions – R1={corr1:.2f}, R2={corr2:.2f}'
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.align_ylabels()
    name = f'Neuron_{neuron_id}_predictions_{model_names}'

    putils.save_figure_with_options(fig, file_formats=['png', 'pdf', 'svg'],
                                    filename=name,
                                    output_dir=output_folder, dark_background=True)
    plt.close()



def neurons_with_consistent_decrease(df, reduced_name, alpha=0.05):
    """
    Identify neurons showing a consistent decrease in test_corr
    from the full model to a specific reduced model across all folds.

    :param df: DataFrame with columns ['neuron_id', 'fold', 'model_type', 'model_name', 'test_corr']
    :param reduced_name: str, name of the reduced model to compare
    :return: DataFrame with neurons that show consistent decrease
    """

    # Filter data for relevant models
    df_full = df[df["model_type"] == "full"][["neuron_id", "fold", "test_corr"]].rename(columns={"test_corr": "full_corr"})
    df_reduced = df[(df["model_type"] == "reduced") & (df["model_name"] == reduced_name)][
        ["neuron_id", "fold", "test_corr"]
    ].rename(columns={"test_corr": "reduced_corr"})

    # Merge by neuron_id + fold
    merged = pd.merge(df_full, df_reduced, on=["neuron_id", "fold"], how="inner")

    # Compute fold-level difference
    merged["diff"] = merged["full_corr"] - merged["reduced_corr"]

    # Per-neuron t-test
    results = []
    for neuron_id, group in merged.groupby("neuron_id"):
        diffs = group["diff"].dropna()
        if len(diffs) > 1:
            t_stat, p_val = ttest_1samp(diffs, popmean=0, alternative="less")
        else:
            t_stat, p_val = np.nan, np.nan

        results.append({
            "neuron_id": neuron_id,
            "mean": diffs.mean() if len(diffs) > 0 else np.nan,
            "min": diffs.min() if len(diffs) > 0 else np.nan,
            "count": len(diffs),
            "p_val": p_val
        })

    summary = pd.DataFrame(results)

    # Consistent decrease flag based on significance
    summary["consistent_decrease"] = summary["p_val"] < alpha

    # Match previous return variable naming
    decreased_neurons = summary[summary["consistent_decrease"]].sort_values("mean", ascending=False)

    return decreased_neurons, merged

def assign_expertise_blocks(trial_table, n_consecutive=5):
    """
    Assign 'expert' or 'naive' labels to each trial for each mouse based on
    p_low vs p_chance, separately for reward_group, identifying blocks of at
    least `n_consecutive` trials.

    Parameters
    ----------
    trial_table : pd.DataFrame
        Must contain ['mouse_id', 'reward_group', 'trial_id', 'p_low', 'p_chance'].
        Should be sorted by trial_id within each mouse.
    n_consecutive : int
        Minimum number of consecutive trials to label as expert.

    Returns
    -------
    pd.DataFrame
        Original dataframe with added column 'block_perf_type' = 'naive' or 'expert'.
    """
    trial_table = trial_table.copy()
    trial_table['block_perf_type'] = 0

    def process_mouse(df_mouse):
        reward_group = df_mouse['reward_group'].iloc[0]

        # Criterion per trial
        if reward_group == 1:
            criterion = df_mouse['p_low'] > df_mouse['p_chance']
        elif reward_group == 0:
            criterion = df_mouse['p_low'] < df_mouse['p_chance']
        else:
            raise ValueError(f"Unexpected reward_group: {reward_group}")

        # Convert boolean series to runs of consecutive True/False
        vals = criterion.values
        expert_mask = np.zeros(len(vals), dtype=bool)

        start_idx = 0
        while start_idx < len(vals):
            if vals[start_idx]:
                # find run of consecutive True
                end_idx = start_idx
                while end_idx < len(vals) and vals[end_idx]:
                    end_idx += 1
                run_length = end_idx - start_idx
                if run_length >= n_consecutive:
                    expert_mask[start_idx:end_idx] = True
                start_idx = end_idx
            else:
                start_idx += 1

        df_mouse.loc[expert_mask, 'block_perf_type'] = 1
        return df_mouse

    trial_table = trial_table.groupby('mouse_id', group_keys=False).apply(process_mouse)
    return trial_table

def propagate_expertise_inplace(trial_table):
    """
    Propagate 'block_perf_type' (expert/naive) from whisker trials
    to other trial types in-place based on closest start_time per mouse.

    Parameters
    ----------
    trial_table : pd.DataFrame
        Must contain ['mouse_id', trial_type_col, time_col, 'block_perf_type'].
        Only whisker trials will have non-null 'block_perf_type' initially.
    time_col : str
        Column name for trial start time (used to find nearest whisker trial).
    trial_type_col : str
        Column name that identifies trial type (e.g., 'whisker', 'auditory', 'no_stim').

    Returns
    -------
    pd.DataFrame
        The same table with 'block_perf_type' updated for all trials.
    """
    trial_table = trial_table.sort_values(['start_time'])
    updated_trials = []

    for mouse_id, df_mouse in trial_table.groupby('mouse_id', group_keys=False):
        df_mouse = df_mouse.sort_values('start_time').copy()
        whisker_df = df_mouse[df_mouse['trial_type'] == 'whisker_trial']

        if whisker_df.empty:
            df_mouse['block_perf_type'] = np.nan
        else:
            # Use merge_asof for nearest whisker match
            merged = pd.merge_asof(
                df_mouse,
                whisker_df[['start_time', 'block_perf_type']].sort_values('start_time'),
                on='start_time',
                direction='nearest',
                suffixes=('', '_whisker')
            )
            # Update block_perf_type with nearest whisker label where missing
            df_mouse['block_perf_type'] = np.where(
                df_mouse['block_perf_type'].isna(),
                merged['block_perf_type_whisker'],
                df_mouse['block_perf_type']
            )

        updated_trials.append(df_mouse)

    # Combine back into one table
    updated_table = pd.concat(updated_trials, ignore_index=True)
    return updated_table

def keep_active_from_whisker_onset(trial_df):
    """
    Remove auditory blocks at onset of session, where mice were not yet engaged in the task, before whisker introduction
    :param trial_df: trial table dataframe with active trials only
    :return:
    """
    print('Keeping active trials and removing auditory onset blocks. Getting whisker trial indices...')

    # Keep active trials
    trial_df = trial_df[
        (~trial_df['context'].isin(['passive']))
        & (trial_df['perf'] != 6)
        & (trial_df['early_lick'] == 0)]

    df = trial_df.copy()

    # Find first whisker trial per mouse
    first_whisker_id = (
        df[df['trial_type'] == 'whisker_trial']
        .groupby('mouse_id')['trial_id']
        .min()
        .rename('first_whisker_id')
    )

    # Merge to get first whisker trial per mouse
    df = df.merge(first_whisker_id, on='mouse_id', how='left')

    # Keep only trials >= first whisker trial
    df = df[df['trial_id'] >= df['first_whisker_id']].copy()

    # Reindex trial_id to start at 0 from first whisker trial
    df['trial_id'] = df['trial_id'] - df['first_whisker_id']

    # Define also a whisker_trial_id, just for whisker trials
    df['whisker_trial_id'] = np.nan
    whisker_mask = df['trial_type'] == 'whisker_trial'
    df.loc[whisker_mask, 'whisker_trial_id'] = df.loc[whisker_mask].groupby('mouse_id').cumcount()
    df['whisker_trial_id'] = df['whisker_trial_id'].astype('Int64') # keep as nullable integer

    # Drop helper column
    df.drop(columns='first_whisker_id', inplace=True)

    return df




def load_perf_blocks(trial_table, mouse_id):
    path_to_data = os.path.join(r"/mnt/lsens-analysis/", 'Axel_Bisi',
                             'combined_results')
    # curves_df = load_helpers.load_learning_curves_data(path_to_data=path_to_data, subject_ids=subject_ids)

    file_name = f'{mouse_id}_whisker_0_whisker_trial_learning_curve_interp.h5'
    path_to_file = os.path.join(ROOT_PATH, mouse_id,  'whisker_0', 'learning_curve',file_name)

    # df_w = pd.read_hdf(path_to_file)

    # df_w = pd.read_hdf(path_to_file, key=store.keys()[0], columns=['p_mean', 'p_low', 'p_high', 'p_chance'])
    df_w = pd.read_hdf(path_to_file, key='/df')  # read everything


    trial_curves = []
    array_cols = ['p_mean', 'p_low', 'p_high', 'p_chance']
    for _, row in df_w.iterrows():
        n_trials = len(row[array_cols[0]])
        for t in range(n_trials):
            trial_dict = {}
            for col in df_w.columns:
                if col in array_cols:
                    trial_dict[col] = row[col][t]
                else:
                    trial_dict[col] = row[col]
            trial_dict['whisker_trial_id'] = t
            trial_curves.append(trial_dict)
    trial_curves_df = pd.DataFrame(trial_curves)
    trial_curves_df = assign_expertise_blocks(trial_curves_df, n_consecutive=5)

    # Merge learning curve data into trial table onto trial_id, for each mouse and onto whisker trials only
    trial_table = keep_active_from_whisker_onset(trial_table)  # get whisker trial index

    trial_table = trial_table.merge(
        trial_curves_df[
            ['mouse_id', 'block_perf_type', 'whisker_trial_id', 'p_mean', 'p_low', 'p_high', 'p_chance', 'mouse_cat',
             'learning_trial']],
        on=['mouse_id', 'whisker_trial_id'], how='left'
    )

    # Assign block_perf_typ to auditory_trial and no_stim_trial depending on the closest previous whisker trial
    trial_table = propagate_expertise_inplace(trial_table)

    return trial_table




def build_lrt_merged(df_models, git_v1, git_v2):
    # Extract only necessary fields
    cols = ['mouse_id','neuron_id','model_name','area_acronym_custom',
            'git_version','lrt_significant']

    df_lrt = df_models[cols].copy()

    # Split versions
    lrt_v1 = df_lrt[df_lrt['git_version'] == git_v1][
        ['mouse_id','neuron_id','model_name','area_acronym_custom','lrt_significant']
    ].rename(columns={'lrt_significant':'lrt_v1'})

    lrt_v2 = df_lrt[df_lrt['git_version'] == git_v2][
        ['mouse_id','neuron_id','model_name','area_acronym_custom','lrt_significant']
    ].rename(columns={'lrt_significant':'lrt_v2'})

    # Merge
    lrt_merged = pd.merge(
        lrt_v1,
        lrt_v2,
        on=['mouse_id','neuron_id','model_name','area_acronym_custom'],
        how='inner'
    )

    # Remove full model (LRT not meaningful for it)
    lrt_merged = lrt_merged[lrt_merged['model_name'] != 'full']

    return lrt_merged


def compare_lrt_between_versions(
    lrt_merged,
    output_path,
    git_v1,
    git_v2,
    area_groups,
    area_colors
):
    reduced_models = lrt_merged['model_name'].unique()

    for model in reduced_models:

        subset = lrt_merged[lrt_merged['model_name'] == model].copy()

        # Rename for plotting
        subset = subset.rename(columns={
            'lrt_v1': f"lrt_significant_{git_v1}",
            'lrt_v2': f"lrt_significant_{git_v2}"
        })

        # Output folder for this model
        outdir = os.path.join(output_path, f"{model}_compare_{git_v1}_vs_{git_v2}")
        os.makedirs(outdir, exist_ok=True)

        # --- Your existing plots ---
        plot_lrt_significance_overlap(subset, outdir)

        plot_lrt_significance_per_area_per_model(
            subset,
            area_groups=area_groups,
            area_colors=area_colors,
            output_folder=outdir
        )

        plot_lrt_significance_heatmap(
            subset,
            area_groups,
            area_colors,
            outdir,
            annotate=False
        )

        # Per-area versions
        outdir_area = os.path.join(outdir, "per_area")
        os.makedirs(outdir_area, exist_ok=True)

        plot_lrt_significance_overlap_per_area(subset, outdir_area)
        plot_lrt_significance_per_model_per_area(
            subset,
            area_groups,
            area_colors,
            outdir_area
        )


def compare_whisker_kernel_models(
        dfs,                     # dict: {"1k": df1, "2k": df2, "3k": df3}
        model_labels,            # dict: {"1k": "1 whisker kernel", ...}
        output_folder,
        whisker_kernels,         # e.g. ["whisker_stim", "whisker_stim_1", ...]
        lags,
        area_groups,
        area_colors,
        n_cols=3,
        git_handle=None):
    """
    Compare 1-, 2-, and 3-whisker-kernel models by plotting their average kernels
    overlaid on the same axes for each region and each whisker kernel.
    """

    # Map each region to a color (your existing logic)
    region_to_color = {}
    for group_name, areas in area_groups.items():
        for area in areas:
            region_to_color[area] = area_colors[group_name]

    def get_region_color(region):
        return region_to_color.get(region, 'gray')

    # Ordered regions
    ordered_regions = []
    for group_name, areas in area_groups.items():
        for area in areas:
            for df in dfs.values():
                if area in df["area_acronym_custom"].values:
                    ordered_regions.append(area)
                    break

    n_rows = math.ceil(len(ordered_regions)/n_cols)

    # Loop over each whisker kernel name (e.g., whisker_stim, whisker_stim_1...)
    for kernel in whisker_kernels:

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True
        )
        axes = np.array(axes).reshape(-1)

        for ax, region in zip(axes, ordered_regions):

            region_color = get_region_color(region)

            # Loop over model types
            for model_key, df in dfs.items():
                region_df = df[df["area_acronym_custom"] == region]
                neuron_groups = region_df.groupby(["mouse_id", "neuron_id"])
                kernels_list = []

                for (_, _), grp in neuron_groups:

                    this_neuron = []

                    for _, row in grp.iterrows():
                        predictors = row["predictors_full"]
                        coefs = row["coef_full_mean"]

                        # find indices matching this kernel
                        idx = [i for i,p in enumerate(predictors) if p == kernel]
                        import re
                        pattern = re.compile(fr"^{kernel}_t[+-]\d+\.\d+s$")
                        idx = [i for i,p in enumerate(predictors) if pattern.match(p)]

                        if len(idx) > 0:
                            coefs_array = np.array(coefs)
                            this_neuron.append(coefs_array[idx])

                    if len(this_neuron) > 0:
                        this_neuron = np.stack(this_neuron)
                        kernels_list.append(this_neuron.mean(axis=0))

                if len(kernels_list) == 0:
                    continue

                kernels_stack = np.stack(kernels_list)
                mean_kernel = np.nanmean(kernels_stack, axis=0)
                sem_kernel = np.nanstd(kernels_stack, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(kernels_stack), axis=0))
                # plot
                ax.plot(lags, mean_kernel, label=model_labels[model_key], lw=2)
                ax.fill_between(lags,
                                mean_kernel - sem_kernel,
                                mean_kernel + sem_kernel,
                                alpha=0.2)

            ax.set_title(f"{region}", fontsize=10)
            ax.set_xlabel("Lag (s)")
            ax.set_ylabel("Coef")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.legend(fontsize=8)

        # hide empty axes
        for ax in axes[len(ordered_regions):]:
            ax.set_visible(False)

        plt.suptitle(f"Comparison across models: {kernel}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        name = f"compare_models_{kernel}"
        putils.save_figure_with_options(fig, file_formats=["png", 'pdf'],
                                        filename=name,
                                        output_dir=output_folder)
        plt.close(fig)

def compare_kernels_within_model(
        dfs,                     # {"1k": df1, "2k": df2, ...}
        model_labels,            # {"1k": "1 whisker kernel", ...}
        output_folder,
        whisker_kernels,         # ["whisker_stim_0", "whisker_stim_1", ...]
        area_groups,
        area_colors,
        n_cols=3,
        git_handle=None):
    """
    For each model (1k, 2k, 3k...), plot ALL whisker kernels together
    within each region. This shows how the model's separate kernels differ.
    """

    import re

    # --------------------------
    # Helpers
    # --------------------------
    def extract_sorted_kernel_indices(predictors, kernel_name):
        """
        Returns (indices_sorted, lags_sorted)
        for entries like kernel_name_t+0.20s, kernel_name_t-0.10s, etc.
        """
        pattern = re.compile(fr"^{kernel_name}_t([+-]\d+\.\d+)s$")
        matches = []

        for i, p in enumerate(predictors):
            m = pattern.match(p)
            if m:
                lag = float(m.group(1))
                matches.append((lag, i))

        matches_sorted = sorted(matches, key=lambda x: x[0])
        idx_sorted = [i for (_, i) in matches_sorted]
        lags_sorted = [lag for (lag, _) in matches_sorted]
        return idx_sorted, lags_sorted

    region_to_color = {}
    for group_name, areas in area_groups.items():
        for area in areas:
            region_to_color[area] = area_colors[group_name]

    ordered_regions = []
    for group_name, areas in area_groups.items():
        for area in areas:
            for df in dfs.values():
                if area in df["area_acronym_custom"].values:
                    ordered_regions.append(area)
                    break

    # --------------------------
    # Loop per model
    # --------------------------
    for model_key, df in dfs.items():

        model_label = model_labels.get(model_key, model_key)
        
        n_rows = math.ceil(len(ordered_regions)/n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=False
        )
        axes = np.array(axes).reshape(-1)

        for ax, region in zip(axes, ordered_regions):

            region_df = df[df["area_acronym_custom"] == region]
            neuron_groups = region_df.groupby(["mouse_id", "neuron_id"])

            # For each whisker kernel: collect average kernel
            kernel_curves = {}
            kernel_lags = {}

            for kernel in whisker_kernels:

                kernels_list = []
                lags_for_kernel = None

                for (_, _), grp in neuron_groups:
                    this_neuron = []

                    for _, row in grp.iterrows():
                        predictors = row["predictors_full"]
                        coefs = row["coef_full_mean"]

                        idx, lag_list = extract_sorted_kernel_indices(predictors, kernel)

                        if idx:
                            if lags_for_kernel is None:
                                lags_for_kernel = lag_list
                            coefs_array = np.array(coefs)
                            this_neuron.append(coefs_array[idx])

                    if this_neuron:
                        this_neuron = np.stack(this_neuron)
                        kernels_list.append(this_neuron.mean(axis=0))

                if kernels_list:
                    kernels_stack = np.stack(kernels_list)
                    mean_kernel = np.nanmean(kernels_stack, axis=0)
                    sem_kernel = np.nanstd(kernels_stack, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(kernels_stack), axis=0))

                    kernel_curves[kernel] = (mean_kernel, sem_kernel)
                    kernel_lags[kernel] = lags_for_kernel

            # --------------------------
            # Plot all kernels on this region axis
            # --------------------------
            for kernel, (mean_kernel, sem_kernel) in kernel_curves.items():
                lags = kernel_lags[kernel]
                ax.plot(lags, mean_kernel, lw=2, label=kernel)
                ax.fill_between(lags, mean_kernel - sem_kernel, mean_kernel + sem_kernel, alpha=0.2)

            ax.set_title(f"{region} — {model_label}", fontsize=10)
            ax.set_xlabel("Lag (s)")
            ax.set_ylabel("Coef")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=7)

        for ax in axes[len(ordered_regions):]:
            ax.set_visible(False)

        plt.suptitle(f"Kernel comparison WITHIN model: {model_label}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        name = f"compare_kernels_within_model_{model_key}"
        putils.save_figure_with_options(fig, file_formats=["png", 'pdf'],
                                        filename=name,
                                        output_dir=output_folder)
        plt.close(fig)


import numpy as np
import pandas as pd
import re

def summarize_kernels_per_area(
        df,                 # single model dataframe
        whisker_kernels,    # ["whisker_stim_0", "whisker_stim_1", ...]
        area_groups         # dict: {"group": [area1, area2, ...]}
    ):
    """
    Returns a DataFrame with, per area and per whisker kernel:
    - mean coefficient across neurons
    - lag of peak absolute coefficient
    """

    def extract_sorted_kernel_indices(predictors, kernel_name):
        pattern = re.compile(fr"^{kernel_name}_t([+-]\d+\.\d+)s$")
        matches = []
        for i, p in enumerate(predictors):
            m = pattern.match(p)
            if m:
                lag = float(m.group(1))
                matches.append((lag, i))
        matches_sorted = sorted(matches, key=lambda x: x[0])
        idx_sorted = [i for (_, i) in matches_sorted]
        lags_sorted = [lag for (lag, _) in matches_sorted]
        return idx_sorted, lags_sorted

    results = []

    # Get all areas in the dataframe
    areas = df['area_acronym_custom'].unique()

    for area in areas:

        region_df = df[df['area_acronym_custom'] == area]
        neuron_groups = region_df.groupby(['mouse_id','neuron_id'])

        for kernel in whisker_kernels:

            kernels_list = []
            lags_for_kernel = None

            for (_, _), grp in neuron_groups:

                this_neuron = []

                for _, row in grp.iterrows():
                    predictors = row["predictors_full"]
                    coefs = row["coef_full_mean"]

                    idx, lag_list = extract_sorted_kernel_indices(predictors, kernel)
                    if idx:
                        if lags_for_kernel is None:
                            lags_for_kernel = lag_list
                        coefs_array = np.array(coefs)
                        this_neuron.append(coefs_array[idx])

                if this_neuron:
                    this_neuron = np.stack(this_neuron)
                    kernels_list.append(this_neuron.mean(axis=0))

            if kernels_list:
                kernels_stack = np.stack(kernels_list)
                mean_kernel = np.nanmean(kernels_stack, axis=0)

                # Mean coefficient across all lags
                mean_coef = np.nanmean(mean_kernel)

                # Lag of peak absolute coefficient
                peak_lag = lags_for_kernel[np.argmax(np.abs(mean_kernel))]

                results.append({
                    "area": area,
                    "kernel": kernel,
                    "mean_coef": mean_coef,
                    "peak_lag": peak_lag
                })

    return pd.DataFrame(results)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_bar_coef_per_area_per_model(summary_dfs,       # dict: {"1k": df1_summary, ...}
                                     model_labels,      # {"1k": "1 whisker kernel", ...}
                                     output_folder, 
                                     whisker_kernels=None,   # optional, list of kernels to include
                                     figsize=(6,4),
                                     save_formats=["png"]):
    """
    For each model and each area, plots a barplot of average coefficients per whisker kernel.
    """

    os.makedirs(output_folder, exist_ok=True)

    for model_key, df in summary_dfs.items():
        if df.empty:
            print(f"Skipping model {model_key}: summary is empty")
            continue
        model_label = model_labels.get(model_key, model_key)
        areas = df["area"].unique()
        for area in areas:
            df_area = df[df["area"] == area].copy()
            if whisker_kernels is not None:
                df_area = df_area[df_area["kernel"].isin(whisker_kernels)]

            # Optional: sort kernels
            df_area = df_area.sort_values("kernel")

            # Barplot
            plt.figure(figsize=figsize)
            sns.barplot(
                data=df_area,
                x="kernel",
                y="mean_coef",
                ci=None,
                palette="tab10"
            )

            # Add error bars for SEM if you have it (not included in current summary)
            # plt.errorbar(x=np.arange(len(df_area)),
            #              y=df_area["mean_coef"],
            #              yerr=df_area["sem_coef"], fmt='none', color='k', capsize=3)

            plt.title(f"{model_label} — {area}")
            plt.ylabel("Mean coefficient")
            plt.xlabel("Whisker kernel")
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save figure
            for fmt in save_formats:
                fname = f"{model_key}_{area}_barcoef.{fmt}"
                plt.savefig(os.path.join(output_folder, fname), dpi=200)
            plt.close()


def compare_model_fit_metrics(
        dfs,                     # dict: {"1k": df1, "2k": df2, "3k": df3, "4k": df4}
        model_labels,            # dict: {"1k": "1 whisker kernel", ...}
        output_folder,
        area_groups,
        area_colors,
        metrics=['test_corr'],   # list of metrics to compare
        n_cols=3):
    """
    Compare model fit quality across models with different numbers of whisker kernels.
    Plots test_corr, deviance, or other metrics per area and across all areas.
    """

    os.makedirs(output_folder, exist_ok=True)

    # Map regions to colors
    region_to_color = {}
    for group_name, areas in area_groups.items():
        for area in areas:
            region_to_color[area] = area_colors[group_name]

    # Get ordered regions
    ordered_regions = []
    for group_name, areas in area_groups.items():
        for area in areas:
            for df in dfs.values():
                if area in df["area_acronym_custom"].values:
                    ordered_regions.append(area)
                    break

    for metric in metrics:
        # --- Overall comparison across all neurons ---
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        plot_data = []
        for model_key, df in dfs.items():
            if metric not in df.columns:
                print(f"[WARNING] Metric '{metric}' not found in model '{model_key}'")
                continue

            # Group by neuron and take mean across folds
            neuron_metrics = df.groupby(['mouse_id', 'neuron_id'])[metric].mean().values

            for val in neuron_metrics:
                plot_data.append({
                    'model': model_labels[model_key],
                    'value': val
                })

        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            sns.violinplot(data=plot_df, x='model', y='value', ax=ax, inner='box')
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} comparison across models')
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.xticks(rotation=45)
            plt.tight_layout()

            fname = f"model_fit_comparison_{metric}_overall"
            putils.save_figure_with_options(fig, file_formats=["png", 'pdf'],
                                          filename=fname,
                                          output_dir=output_folder)
            plt.close(fig)

        # --- Per-area comparison ---
        n_rows = math.ceil(len(ordered_regions) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(-1)

        for ax, region in zip(axes, ordered_regions):
            plot_data = []

            for model_key, df in dfs.items():
                if metric not in df.columns:
                    continue

                region_df = df[df["area_acronym_custom"] == region]
                neuron_metrics = region_df.groupby(['mouse_id', 'neuron_id'])[metric].mean().values

                for val in neuron_metrics:
                    plot_data.append({
                        'model': model_labels[model_key],
                        'value': val
                    })

            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                sns.violinplot(data=plot_df, x='model', y='value', ax=ax, inner='box')
                ax.set_title(f'{region}')
                ax.set_xlabel('')
                ax.set_ylabel(metric)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(axis='x', rotation=45)

        # Hide empty axes
        for ax in axes[len(ordered_regions):]:
            ax.set_visible(False)

        plt.suptitle(f'{metric} comparison per area', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        fname = f"model_fit_comparison_{metric}_per_area"
        putils.save_figure_with_options(fig, file_formats=["png", 'pdf'],
                                      filename=fname,
                                      output_dir=output_folder)
        plt.close(fig)

        # --- Summary statistics: mean improvement ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        summary_data = []
        model_keys_sorted = sorted(dfs.keys())

        for region in ordered_regions:
            for model_key in model_keys_sorted:
                df = dfs[model_key]
                if metric not in df.columns:
                    continue

                region_df = df[df["area_acronym_custom"] == region]
                neuron_metrics = region_df.groupby(['mouse_id', 'neuron_id'])[metric].mean()

                summary_data.append({
                    'region': region,
                    'model': model_labels[model_key],
                    'mean': neuron_metrics.mean(),
                    'sem': neuron_metrics.sem()
                })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)

            # Plot grouped bar chart
            x_pos = np.arange(len(ordered_regions))
            width = 0.8 / len(model_keys_sorted)

            for i, model_key in enumerate(model_keys_sorted):
                model_data = summary_df[summary_df['model'] == model_labels[model_key]]
                means = model_data['mean'].values
                sems = model_data['sem'].values

                ax.bar(x_pos + i * width, means, width,
                      label=model_labels[model_key], yerr=sems, capsize=3)

            ax.set_xlabel('Region')
            ax.set_ylabel(f'Mean {metric}')
            ax.set_title(f'Mean {metric} per region across models')
            ax.set_xticks(x_pos + width * (len(model_keys_sorted) - 1) / 2)
            ax.set_xticklabels(ordered_regions, rotation=45, ha='right')
            ax.legend()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()

            fname = f"model_fit_comparison_{metric}_mean_per_region"
            putils.save_figure_with_options(fig, file_formats=["png", 'pdf'],
                                          filename=fname,
                                          output_dir=output_folder)
            plt.close(fig)


def compare_kernel_shape_evolution(
        dfs,                     # dict: {"1k": df1, "2k": df2, "3k": df3, "4k": df4}
        model_labels,            # dict: {"1k": "1 whisker kernel", ...}
        output_folder,
        whisker_kernels,         # ["whisker_stim", "whisker_stim_0", "whisker_stim_1", ...]
        area_groups,
        area_colors,
        n_cols=3):
    """
    Visualize how the shape of each whisker kernel evolves as we add more kernels to the model.
    Shows whether kernels remain stable or change shape when additional kernels are added.
    """
    print(model_labels)
    print(whisker_kernels)
    os.makedirs(output_folder, exist_ok=True)

    def extract_sorted_kernel_indices(predictors, kernel_name):
        pattern = re.compile(fr"^{kernel_name}_t([+-]\d+\.\d+)s$")
        matches = []
        for i, p in enumerate(predictors):
            m = pattern.match(p)
            if m:
                lag = float(m.group(1))
                matches.append((lag, i))
        matches_sorted = sorted(matches, key=lambda x: x[0])
        idx_sorted = [i for (_, i) in matches_sorted]
        lags_sorted = [lag for (lag, _) in matches_sorted]
        return idx_sorted, lags_sorted

    # Map regions to colors
    region_to_color = {}
    for group_name, areas in area_groups.items():
        for area in areas:
            region_to_color[area] = area_colors[group_name]

    # Get ordered regions
    ordered_regions = []
    for group_name, areas in area_groups.items():
        for area in areas:
            for df in dfs.values():
                if area in df["area_acronym_custom"].values:
                    ordered_regions.append(area)
                    break

    # For each kernel, show its evolution across models
    for kernel in whisker_kernels:
        n_rows = math.ceil(len(ordered_regions) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True)
        axes = np.array(axes).reshape(-1)

        for ax, region in zip(axes, ordered_regions):
            # Collect kernel for each model
            for model_key, df in dfs.items():
                region_df = df[df["area_acronym_custom"] == region]
                neuron_groups = region_df.groupby(["mouse_id", "neuron_id"])

                kernels_list = []
                lags_for_kernel = None

                for (_, _), grp in neuron_groups:
                    this_neuron = []

                    for _, row in grp.iterrows():
                        predictors = row["predictors_full"]
                        coefs = row["coef_full_mean"]
                        print(predictors)
                        idx, lag_list = extract_sorted_kernel_indices(predictors, kernel)
                        print(idx)
                        if idx:
                            if lags_for_kernel is None:
                                lags_for_kernel = lag_list
                            coefs_array = np.array(coefs)
                            this_neuron.append(coefs_array[idx])

                    if this_neuron:
                        this_neuron = np.stack(this_neuron)
                        kernels_list.append(this_neuron.mean(axis=0))

                if kernels_list:
                    kernels_stack = np.stack(kernels_list)
                    mean_kernel = np.nanmean(kernels_stack, axis=0)
                    sem_kernel = np.nanstd(kernels_stack, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(kernels_stack), axis=0))

                    ax.plot(lags_for_kernel, mean_kernel, label=model_labels[model_key], lw=2)
                    ax.fill_between(lags_for_kernel,
                                   mean_kernel - sem_kernel,
                                   mean_kernel + sem_kernel,
                                   alpha=0.2)

            ax.set_title(f'{region}')
            ax.set_xlabel('Lag (s)')
            ax.set_ylabel('Coefficient')
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=7)

        # Hide empty axes
        for ax in axes[len(ordered_regions):]:
            ax.set_visible(False)

        plt.suptitle(f'Shape evolution of {kernel} across models', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        fname = f"kernel_shape_evolution_{kernel}"
        putils.save_figure_with_options(fig, file_formats=["png", 'pdf'],
                                      filename=fname,
                                      output_dir=output_folder)
        plt.close(fig)


def compare_kernel_amplitude_evolution(
        dfs,
        model_labels,
        output_folder,
        whisker_kernels,
        area_groups,
        area_colors,
        n_cols=3):
    """
    Within each multi-kernel model, compare the amplitudes of different whisker kernels.
    """
    from scipy.stats import ttest_rel, ttest_1samp, spearmanr

    PLANNED_PAIRS = [
        ('auditory_stim0',       'auditory_stim1'),
        ('whisker_hits_stim_0',   'whisker_hits_stim_1'),
        ('whisker_misses_stim0', 'whisker_misses_stim1'),
        ('piezo_reward_0',        'piezo_reward_1'),
        ('piezo_reward_au_0',     'piezo_reward_au_1'),
    ]

    os.makedirs(output_folder, exist_ok=True)

    def extract_peak_amplitude(predictors, coefs, kernel_name):
        """Return peak |coef| for a given kernel, or None."""
        pattern = re.compile(fr"^{kernel_name}_t([+-]\d+\.\d+)s$")
        matches = sorted(
            [(float(m.group(1)), i) for i, p in enumerate(predictors) if (m := pattern.match(p))]
        )
        if not matches:
            return None
        idx = [i for _, i in matches]
        kernel_coefs = coefs[idx]
        peak = np.nanmax(np.abs(kernel_coefs))
        return None if np.isnan(peak) else peak

    def build_neuron_map(region_df, kernel_name):
        """Return {(mouse_id, neuron_id): peak_amp} for neurons that have the kernel."""
        nmap = {}
        for (mouse_id, neuron_id), grp in region_df.groupby(["mouse_id", "neuron_id"]):
            peaks = [
                p for _, row in grp.iterrows()
                if (p := extract_peak_amplitude(row["predictors_full"], row["coef_full_mean"], kernel_name)) is not None
            ]
            if peaks:
                nmap[(mouse_id, neuron_id)] = np.nanmax(peaks)
        return nmap

    def draw_planned_brackets(ax, kernel_order, nmap_per_kernel, y_max, bracket_step,
                              lw=1, fontsize=9, min_n=10):
        """Draw paired t-test significance brackets for the planned pairs."""
        current_y = y_max + bracket_step
        for ka, kb in PLANNED_PAIRS:
            if ka not in nmap_per_kernel or kb not in nmap_per_kernel:
                print(f"  [brackets] DROP {ka} vs {kb}: not in nmap_per_kernel "
                      f"(ka_present={ka in nmap_per_kernel}, kb_present={kb in nmap_per_kernel})")
                continue
            if ka not in kernel_order or kb not in kernel_order:
                print(f"  [brackets] DROP {ka} vs {kb}: not in kernel_order "
                      f"(ka_present={ka in kernel_order}, kb_present={kb in kernel_order})")
                continue
            i, j = kernel_order.index(ka), kernel_order.index(kb)
            shared = set(nmap_per_kernel[ka]) & set(nmap_per_kernel[kb])
            if len(shared) < min_n:
                print(f"  [brackets] DROP {ka} vs {kb}: shared={len(shared)} < min_n={min_n}")
                continue
            print(f"  [brackets] TEST {ka} vs {kb}: shared={len(shared)} neurons")
            v_a = np.array([nmap_per_kernel[ka][n] for n in shared])
            v_b = np.array([nmap_per_kernel[kb][n] for n in shared])
            _, p = ttest_rel(v_a, v_b)
            stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            ax.plot([i, i], [current_y - bracket_step * 0.4, current_y], color='k', linewidth=lw)
            ax.plot([j, j], [current_y - bracket_step * 0.4, current_y], color='k', linewidth=lw)
            ax.plot([i, j], [current_y, current_y], color='k', linewidth=lw)
            ax.text((i + j) / 2, current_y + bracket_step * 0.1, stars,
                    ha='center', va='bottom', fontsize=fontsize, fontweight='bold')
            current_y += bracket_step

    def nmap_to_mouse_map(nmap):
        """Collapse {(mouse_id, neuron_id): amp} → {mouse_id: mean_amp across neurons}."""
        from collections import defaultdict
        by_mouse = defaultdict(list)
        for (mouse_id, _neuron_id), amp in nmap.items():
            by_mouse[mouse_id].append(amp)
        return {mouse_id: np.mean(amps) for mouse_id, amps in by_mouse.items()}

    def compute_trend_label(nmap_per_kernel, kernel_order):
        """
        Per-neuron Spearman ρ across kernels; one-sample Wilcoxon vs 0 to test
        whether the amplitude trend is consistent across the population.
        """
        neuron_trends = []
        all_neurons = set.union(*(set(nmap_per_kernel[k]) for k in kernel_order))
        for neuron in all_neurons:
            k_idx = [i for i, k in enumerate(kernel_order) if neuron in nmap_per_kernel[k]]
            k_amp = [nmap_per_kernel[kernel_order[i]][neuron] for i in k_idx]
            if len(k_idx) >= 2:
                rho, _ = spearmanr(k_idx, k_amp)
                if not np.isnan(rho):
                    neuron_trends.append(rho)
        if len(neuron_trends) < 10:
            return None
        _, p = ttest_1samp(neuron_trends, 0)
        mean_rho = np.mean(neuron_trends)
        direction = "incr." if mean_rho > 0 else "decr."
        return f"Trend ({direction}): mean ρ={mean_rho:.3f}, t-test p={p:.4f} (n={len(neuron_trends)})"

    # Infer reward group color from the data (each call is pre-filtered to one group)
    _rg_vals = pd.concat(list(dfs.values()))['reward_group'].dropna().unique()
    _REWARD_COLORS = {1: 'forestgreen', 0: 'crimson'}
    _bar_color = _REWARD_COLORS.get(int(_rg_vals[0]), 'steelblue') if len(_rg_vals) == 1 else 'steelblue'

    # Ordered regions following area_groups structure
    ordered_regions = [
        area
        for areas in area_groups.values()
        for area in areas
        if any(area in df["area_acronym_custom"].values for df in dfs.values())
    ]

    for model_key, df in dfs.items():
        if model_key == "1k":
            continue

        model_label = model_labels[model_key]
        print(model_label)

        # Collect amplitude data per kernel × region
        amplitude_data = []
        for kernel in whisker_kernels:
            for region in ordered_regions:
                nmap = build_neuron_map(df[df["area_acronym_custom"] == region], kernel)
                if not nmap:
                    continue
                amps = list(nmap.values())
                amplitude_data.append({
                    'kernel': kernel,
                    'region': region,
                    'mean_amplitude': np.mean(amps),
                    'sem_amplitude': np.std(amps, ddof=1) / np.sqrt(len(amps)),
                    'n_neurons': len(amps),
                    'neuron_map': nmap,
                })

        if not amplitude_data:
            print(f"[WARNING] No amplitude data for model {model_key}")
            continue

        amplitude_df = pd.DataFrame(amplitude_data)
        available_kernels = amplitude_df['kernel'].unique()

        # Merge neuron maps across regions per kernel
        nmap_per_kernel = {}
        for kernel in available_kernels:
            merged = {}
            for nmap in amplitude_df[amplitude_df['kernel'] == kernel]['neuron_map']:
                merged.update(nmap)
            nmap_per_kernel[kernel] = merged
        kernel_order = list(available_kernels)
        print(f"[amplitude_evolution] {model_key} kernel_order: {kernel_order}")

        # Build mean-coefficient-over-time data (signed mean, not |peak|)
        mean_amp_data = []
        for kernel in whisker_kernels:
            _pat = re.compile(fr"^{re.escape(kernel)}_t([+-]?\d+\.\d+)s$")
            for region in ordered_regions:
                nmap_m = {}
                for (mouse_id, neuron_id), grp in df[df["area_acronym_custom"] == region].groupby(["mouse_id", "neuron_id"]):
                    fold_means = []
                    for _, row in grp.iterrows():
                        matches = sorted(
                            [(float(_m.group(1)), i)
                             for i, p in enumerate(row["predictors_full"])
                             if (_m := _pat.match(p))]
                        )
                        if matches:
                            idx = [i for _, i in matches]
                            fold_means.append(np.mean(np.array(row["coef_full_mean"])[idx]))
                    if fold_means:
                        nmap_m[(mouse_id, neuron_id)] = np.nanmean(fold_means)
                if not nmap_m:
                    continue
                amps_m = list(nmap_m.values())
                mean_amp_data.append({
                    'kernel': kernel, 'region': region,
                    'mean_amplitude': np.mean(amps_m),
                    'sem_amplitude': np.std(amps_m, ddof=1) / np.sqrt(len(amps_m)),
                    'n_neurons': len(amps_m),
                    'neuron_map': nmap_m,
                })
        mean_amp_df = pd.DataFrame(mean_amp_data) if mean_amp_data else None
        if mean_amp_df is not None:
            nmap_per_kernel_mean = {}
            for kernel in mean_amp_df['kernel'].unique():
                _merged = {}
                for nmap in mean_amp_df[mean_amp_df['kernel'] == kernel]['neuron_map']:
                    _merged.update(nmap)
                nmap_per_kernel_mean[kernel] = _merged

        # --- Plot 1: Overall bar plot ---
        fig, ax = plt.subplots(figsize=(10, 6))
        kernel_means, kernel_sems, kernel_labels = [], [], []
        for kernel in available_kernels:
            kd = amplitude_df[amplitude_df['kernel'] == kernel]
            n = kd['n_neurons'].sum()
            kernel_means.append((kd['mean_amplitude'] * kd['n_neurons']).sum() / n)
            kernel_sems.append(np.sqrt((kd['sem_amplitude']**2 * kd['n_neurons']).sum()) / np.sqrt(n))
            kernel_labels.append(kernel)

        x_pos = np.arange(len(kernel_labels))
        ax.bar(x_pos, kernel_means, yerr=kernel_sems, capsize=5, alpha=0.7, color=_bar_color)
        ax.set_xlabel('Whisker Kernel')
        ax.set_ylabel('Mean Peak Amplitude (|coef|)')
        ax.set_title(f'Kernel amplitudes within {model_label}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(kernel_labels, rotation=45, ha='right')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        y_max = max(m + s for m, s in zip(kernel_means, kernel_sems))
        draw_planned_brackets(ax, kernel_order, nmap_per_kernel, y_max, y_max * 0.06)

        trend_lbl = compute_trend_label(nmap_per_kernel, kernel_order)
        if trend_lbl:
            ax.text(0.02, 0.97, trend_lbl, transform=ax.transAxes, fontsize=8,
                    va='top', ha='left', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()
        putils.save_figure_with_options(fig, file_formats=["png", "pdf"],
                                        filename=f"kernel_amplitude_within_{model_key}_overall",
                                        output_dir=output_folder)
        plt.close(fig)

        # --- Plot 1b: Overall bar plot, averaged per mouse ---
        mouse_map_per_kernel = {k: nmap_to_mouse_map(nmap_per_kernel[k]) for k in kernel_order}

        fig, ax = plt.subplots(figsize=(10, 6))
        mouse_means, mouse_sems, mouse_labels = [], [], []
        for kernel in available_kernels:
            amps = list(mouse_map_per_kernel[kernel].values())
            mouse_means.append(np.mean(amps) if amps else 0)
            mouse_sems.append(np.std(amps, ddof=1) / np.sqrt(len(amps)) if len(amps) >= 2 else 0)
            mouse_labels.append(kernel)

        x_pos = np.arange(len(mouse_labels))
        ax.bar(x_pos, mouse_means, yerr=mouse_sems, capsize=5, alpha=0.7, color=_bar_color)
        ax.set_xlabel('Whisker Kernel')
        ax.set_ylabel('Mean Peak Amplitude (|coef|)')
        ax.set_title(f'Kernel amplitudes within {model_label} (per mouse)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(mouse_labels, rotation=45, ha='right')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        y_max_m = max(m + s for m, s in zip(mouse_means, mouse_sems))
        draw_planned_brackets(ax, kernel_order, mouse_map_per_kernel, y_max_m, y_max_m * 0.06, min_n=3)

        trend_lbl_m = compute_trend_label(mouse_map_per_kernel, kernel_order)
        if trend_lbl_m:
            ax.text(0.02, 0.97, trend_lbl_m, transform=ax.transAxes, fontsize=8,
                    va='top', ha='left', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()
        putils.save_figure_with_options(fig, file_formats=["png", "pdf"],
                                        filename=f"kernel_amplitude_within_{model_key}_overall_per_mouse",
                                        output_dir=output_folder)
        plt.close(fig)

        # --- Plot 2: Per-region bar plots ---
        n_rows = math.ceil(len(ordered_regions) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(-1)

        for ax, region in zip(axes, ordered_regions):
            region_data = amplitude_df[amplitude_df['region'] == region]
            if region_data.empty:
                ax.set_visible(False)
                continue

            kernels_r = list(region_data['kernel'])
            means_r   = region_data['mean_amplitude'].values
            sems_r    = region_data['sem_amplitude'].values
            nmaps_r   = dict(zip(kernels_r, region_data['neuron_map']))

            x_pos = np.arange(len(kernels_r))
            ax.bar(x_pos, means_r, yerr=sems_r, capsize=3, alpha=0.7, color=_bar_color)
            ax.set_title(region)
            ax.set_xlabel('Kernel')
            ax.set_ylabel('Peak Amplitude')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(kernels_r, rotation=45, ha='right', fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            y_max_r = float(max(means_r + sems_r))
            draw_planned_brackets(ax, kernels_r, nmaps_r, y_max_r, y_max_r * 0.08, lw=0.8, fontsize=7)

        for ax in axes[len(ordered_regions):]:
            ax.set_visible(False)

        plt.suptitle(f'Kernel amplitudes per region: {model_label}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        putils.save_figure_with_options(fig, file_formats=["png", "pdf"],
                                        filename=f"kernel_amplitude_within_{model_key}_per_region",
                                        output_dir=output_folder)
        plt.close(fig)

        # --- Plot 2b: Per-region bar plots, averaged per mouse ---
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(-1)

        for ax, region in zip(axes, ordered_regions):
            region_data = amplitude_df[amplitude_df['region'] == region]
            if region_data.empty:
                ax.set_visible(False)
                continue

            kernels_r = list(region_data['kernel'])
            nmaps_r   = dict(zip(kernels_r, region_data['neuron_map']))
            mmaps_r   = {k: nmap_to_mouse_map(v) for k, v in nmaps_r.items()}

            means_r = [np.mean(list(mmaps_r[k].values())) if mmaps_r[k] else 0 for k in kernels_r]
            sems_r  = [np.std(list(mmaps_r[k].values()), ddof=1) / np.sqrt(len(mmaps_r[k]))
                       if len(mmaps_r[k]) >= 2 else 0 for k in kernels_r]

            x_pos = np.arange(len(kernels_r))
            ax.bar(x_pos, means_r, yerr=sems_r, capsize=3, alpha=0.7, color=_bar_color)
            ax.set_title(region)
            ax.set_xlabel('Kernel')
            ax.set_ylabel('Peak Amplitude')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(kernels_r, rotation=45, ha='right', fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            y_max_r = float(max(m + s for m, s in zip(means_r, sems_r))) if any(means_r) else 1
            draw_planned_brackets(ax, kernels_r, mmaps_r, y_max_r, y_max_r * 0.08,
                                  lw=0.8, fontsize=7, min_n=3)

        for ax in axes[len(ordered_regions):]:
            ax.set_visible(False)

        plt.suptitle(f'Kernel amplitudes per region: {model_label} (per mouse)', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        putils.save_figure_with_options(fig, file_formats=["png", "pdf"],
                                        filename=f"kernel_amplitude_within_{model_key}_per_region_per_mouse",
                                        output_dir=output_folder)
        plt.close(fig)

        # --- Plot 2c: Per-region mean coef over time ---
        if mean_amp_df is not None:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes = np.array(axes).reshape(-1)

            for ax, region in zip(axes, ordered_regions):
                region_data = mean_amp_df[mean_amp_df['region'] == region]
                if region_data.empty:
                    ax.set_visible(False)
                    continue

                kernels_r = list(region_data['kernel'])
                means_r   = region_data['mean_amplitude'].values
                sems_r    = region_data['sem_amplitude'].values
                nmaps_r   = dict(zip(kernels_r, region_data['neuron_map']))

                x_pos = np.arange(len(kernels_r))
                ax.bar(x_pos, means_r, yerr=sems_r, capsize=3, alpha=0.7, color=_bar_color)
                ax.axhline(0, color='k', linewidth=0.6, linestyle='--', alpha=0.4)
                ax.set_title(region)
                ax.set_xlabel('Kernel')
                ax.set_ylabel('Mean Coef. (over time)')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(kernels_r, rotation=45, ha='right', fontsize=8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                y_max_r = float(max(means_r + sems_r))
                draw_planned_brackets(ax, kernels_r, nmaps_r, y_max_r, abs(y_max_r) * 0.08, lw=0.8, fontsize=7)

            for ax in axes[len(ordered_regions):]:
                ax.set_visible(False)

            plt.suptitle(f'Kernel mean coef per region: {model_label}', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            putils.save_figure_with_options(fig, file_formats=["png", "pdf"],
                                            filename=f"kernel_mean_coef_within_{model_key}_per_region",
                                            output_dir=output_folder)
            plt.close(fig)

            # --- Plot 2d: Per-region mean coef, averaged per mouse ---
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes = np.array(axes).reshape(-1)

            for ax, region in zip(axes, ordered_regions):
                region_data = mean_amp_df[mean_amp_df['region'] == region]
                if region_data.empty:
                    ax.set_visible(False)
                    continue

                kernels_r = list(region_data['kernel'])
                nmaps_r   = dict(zip(kernels_r, region_data['neuron_map']))
                mmaps_r   = {k: nmap_to_mouse_map(v) for k, v in nmaps_r.items()}

                means_r = [np.mean(list(mmaps_r[k].values())) if mmaps_r[k] else 0 for k in kernels_r]
                sems_r  = [np.std(list(mmaps_r[k].values()), ddof=1) / np.sqrt(len(mmaps_r[k]))
                           if len(mmaps_r[k]) >= 2 else 0 for k in kernels_r]

                x_pos = np.arange(len(kernels_r))
                ax.bar(x_pos, means_r, yerr=sems_r, capsize=3, alpha=0.7, color=_bar_color)
                ax.axhline(0, color='k', linewidth=0.6, linestyle='--', alpha=0.4)
                ax.set_title(region)
                ax.set_xlabel('Kernel')
                ax.set_ylabel('Mean Coef. (over time)')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(kernels_r, rotation=45, ha='right', fontsize=8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                y_max_r = float(max(m + s for m, s in zip(means_r, sems_r))) if any(means_r) else 1
                draw_planned_brackets(ax, kernels_r, mmaps_r, y_max_r, abs(y_max_r) * 0.08,
                                      lw=0.8, fontsize=7, min_n=3)

            for ax in axes[len(ordered_regions):]:
                ax.set_visible(False)

            plt.suptitle(f'Kernel mean coef per region: {model_label} (per mouse)', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            putils.save_figure_with_options(fig, file_formats=["png", "pdf"],
                                            filename=f"kernel_mean_coef_within_{model_key}_per_region_per_mouse",
                                            output_dir=output_folder)
            plt.close(fig)

        # --- Plot 3: Heatmap ---
        pivot_data = amplitude_df.pivot_table(
            index='region', columns='kernel', values='mean_amplitude', aggfunc='mean'
        )
        pivot_data = pivot_data.loc[
            [r for r in ordered_regions if r in pivot_data.index],
            [k for k in available_kernels if k in pivot_data.columns],
        ]
        fig, ax = plt.subplots(figsize=(max(8, len(pivot_data.columns) * 1.5),
                                        max(6, len(pivot_data) * 0.4)))
        _cmap_amp = matplotlib.colors.LinearSegmentedColormap.from_list('rw_amp', ['white', 'forestgreen'])
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap=_cmap_amp,
                    ax=ax, cbar_kws={'label': 'Peak Amplitude'})
        ax.set_title(f'Kernel amplitudes heatmap: {model_label}')
        ax.set_xlabel('Whisker Kernel')
        ax.set_ylabel('Region')
        plt.tight_layout()
        putils.save_figure_with_options(fig, file_formats=["png", "pdf"],
                                        filename=f"kernel_amplitude_within_{model_key}_heatmap",
                                        output_dir=output_folder)
        plt.close(fig)

        # --- Plot 4: Average kernel waveforms for planned pairs ---
        pat_cache = {}
        for ka, kb in PLANNED_PAIRS:
            if ka not in available_kernels or kb not in available_kernels:
                continue

            n_regions = len(ordered_regions)
            n_cols = min(5, n_regions)
            n_rows = int(np.ceil(n_regions / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(n_cols * 3, n_rows * 2.5), squeeze=False)

            for ax_idx, region in enumerate(ordered_regions):
                ax = axes[ax_idx // n_cols][ax_idx % n_cols]
                region_df = df[df["area_acronym_custom"] == region]

                for kernel_name, linestyle in [(ka, '--'), (kb, '-')]:
                    pat = pat_cache.setdefault(
                        kernel_name,
                        re.compile(fr"^{re.escape(kernel_name)}_t([+-]?\d+\.\d+)s$")
                    )
                    waveform_per_neuron = {}
                    lags_ref = None
                    for (mouse_id, neuron_id), grp in region_df.groupby(["mouse_id", "neuron_id"]):
                        per_fold = []
                        for _, row in grp.iterrows():
                            matches = sorted(
                                [(float(m.group(1)), i)
                                 for i, p in enumerate(row["predictors_full"])
                                 if (m := pat.match(p))]
                            )
                            if not matches:
                                continue
                            if lags_ref is None:
                                lags_ref = np.array([lag for lag, _ in matches])
                            idx = [i for _, i in matches]
                            per_fold.append(np.array(row["coef_full_mean"])[idx])
                        if per_fold:
                            waveform_per_neuron[(mouse_id, neuron_id)] = np.nanmean(per_fold, axis=0)

                    if not waveform_per_neuron or lags_ref is None:
                        continue

                    mat = np.array(list(waveform_per_neuron.values()))
                    mean_wvf = np.nanmean(mat, axis=0)
                    sem_wvf = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(len(mat))
                    ax.plot(lags_ref, mean_wvf, color=_bar_color, linestyle=linestyle,
                            linewidth=1.5, label=kernel_name)
                    ax.fill_between(lags_ref, mean_wvf - sem_wvf, mean_wvf + sem_wvf,
                                    color=_bar_color, alpha=0.15)

                ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
                ax.set_title(region, fontsize=9)
                ax.tick_params(labelsize=7)

            for ax_idx in range(len(ordered_regions), n_rows * n_cols):
                axes[ax_idx // n_cols][ax_idx % n_cols].set_visible(False)

            axes[0][0].legend(fontsize=7, frameon=False)
            plt.suptitle(f'Kernel waveforms: {model_label}  ({ka} vs {kb})', fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pair_tag = f"{ka}_vs_{kb}"
            putils.save_figure_with_options(fig, file_formats=["png", "pdf"],
                                            filename=f"kernel_waveform_{model_key}_{pair_tag}",
                                            output_dir=output_folder)
            plt.close(fig)


def compare_kernel_amplitude_evolution_2(
        dfs,                     # dict: {"1k": df1, "2k": df2, "3k": df3, "4k": df4}
        model_labels,            # dict: {"1k": "1 whisker kernel", ...}
        output_folder,
        whisker_kernels,         # ["whisker_stim", "whisker_stim_0", "whisker_stim_1", ...]
        area_groups,
        area_colors,
        n_cols=3):
    """
    Within each multi-kernel model, compare the amplitudes of different whisker kernels across all lags.
    Creates heatmaps and time-series plots showing weight evolution across time lags.
    """

    os.makedirs(output_folder, exist_ok=True)

    def extract_sorted_kernel_indices(predictors, kernel_name):
        pattern = re.compile(fr"^{kernel_name}_t([+-]\d+\.\d+)s$")
        matches = []
        for i, p in enumerate(predictors):
            m = pattern.match(p)
            if m:
                lag = float(m.group(1))
                matches.append((lag, i))
        matches_sorted = sorted(matches, key=lambda x: x[0])
        idx_sorted = [i for (_, i) in matches_sorted]
        lags_sorted = [lag for (lag, _) in matches_sorted]
        return idx_sorted, lags_sorted

    # Get ordered regions
    ordered_regions = []
    for group_name, areas in area_groups.items():
        for area in areas:
            for df in dfs.values():
                if area in df["area_acronym_custom"].values:
                    ordered_regions.append(area)
                    break

    # For each model (except 1k), compare kernel amplitudes within that model
    for model_key, df in dfs.items():
        if model_key == "1k":  # Skip single kernel model
            continue

        model_label = model_labels[model_key]
        print(model_label)
        # Collect kernel data across all lags per region
        kernel_data_by_region = {}  # {region: {kernel: {'lags': [...], 'mean': [...], 'sem': [...]}}}

        for region in ordered_regions:
            kernel_data_by_region[region] = {}
            region_df = df[df["area_acronym_custom"] == region]

            for kernel in whisker_kernels:
                neuron_groups = region_df.groupby(["mouse_id", "neuron_id"])
                kernels_list = []
                lags_for_kernel = None

                for (_, _), grp in neuron_groups:
                    neuron_kernels = []

                    for _, row in grp.iterrows():
                        predictors = row["predictors_full"]
                        coefs = row["coef_full_mean"]
                        idx, lag_list = extract_sorted_kernel_indices(predictors, kernel)

                        if idx:
                            if lags_for_kernel is None:
                                lags_for_kernel = lag_list
                            coefs_array = np.array(coefs)
                            neuron_kernels.append(coefs_array[idx])

                    if neuron_kernels:
                        mean_kernel = np.nanmean(np.stack(neuron_kernels), axis=0)
                        kernels_list.append(mean_kernel)

                if kernels_list:
                    kernels_stack = np.stack(kernels_list)
                    mean_across_neurons = np.nanmean(kernels_stack, axis=0)
                    sem_across_neurons = np.nanstd(kernels_stack, axis=0, ddof=1) / np.sqrt(kernels_stack.shape[0])

                    kernel_data_by_region[region][kernel] = {
                        'lags': np.array(lags_for_kernel),
                        'mean': mean_across_neurons,
                        'sem': sem_across_neurons,
                        'n_neurons': len(kernels_list)
                    }

        # Check if we have data
        if not any(kernel_data_by_region[r] for r in ordered_regions):
            print(f"[WARNING] No kernel data for model {model_key}")
            continue

        # Get all available kernels in this model
        available_kernels = []
        for region_data in kernel_data_by_region.values():
            available_kernels.extend(region_data.keys())
        available_kernels = sorted(set(available_kernels))

        # --- Plot 1: Time-series plot showing all kernels across lags for each region ---
        n_rows = math.ceil(len(ordered_regions) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(-1)

        # Create color mapping based on stimulus type and kernel number
        def get_kernel_color(kernel_name):
            """
            Assign colors based on stimulus type and shade based on kernel number.
            Blue for auditory, Green for whisker_hits, Red for whisker_misses.
            Lighter shades for _0, darker for _1, etc.
            """
            import re

            # Extract kernel number if present (e.g., whisker_hits_0 -> 0, auditory_stim0 -> 0)
            number_match = re.search(r'_?(\d+)$', kernel_name)
            kernel_number = int(number_match.group(1)) if number_match else 0

            # Base colors for each stimulus type
            if 'auditory' in kernel_name.lower():
                # Blue shades
                base_color = np.array([0.2, 0.4, 0.8])  # Base blue
            elif 'whisker_hit' in kernel_name.lower():
                # Green shades
                base_color = np.array([0.2, 0.7, 0.3])  # Base green
            elif 'whisker_miss' in kernel_name.lower():
                # Red shades
                base_color = np.array([0.8, 0.2, 0.2])  # Base red
            else:
                # Default gray for other kernels
                base_color = np.array([0.5, 0.5, 0.5])

            # Adjust brightness based on kernel number
            # kernel 0 -> lighter (multiply by 1.3, cap at 1.0)
            # kernel 1 -> darker (multiply by 0.7)
            # kernel 2+ -> even darker (multiply by 0.5)
            if kernel_number == 0:
                color = np.minimum(base_color * 1.3, 1.0)
            elif kernel_number == 1:
                color = base_color * 0.7
            else:
                color = base_color * 0.5

            return tuple(color)

        color_map = {k: get_kernel_color(k) for k in available_kernels}

        for ax, region in zip(axes, ordered_regions):
            region_data = kernel_data_by_region[region]

            if not region_data:
                ax.set_visible(False)
                continue

            for kernel, kdata in region_data.items():
                lags = kdata['lags']
                mean = kdata['mean']
                sem = kdata['sem']

                color = color_map[kernel]
                ax.plot(lags, mean, color=color, label=kernel, linewidth=2)
                ax.fill_between(lags, mean - sem, mean + sem, color=color, alpha=0.2)

            ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
            ax.set_title(f'{region}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Lag (s)', fontsize=9)
            ax.set_ylabel('Coefficient', fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=7, loc='best')

        for ax in axes[len(ordered_regions):]:
            ax.set_visible(False)

        plt.suptitle(f'Kernel weights across all lags: {model_label}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        fname = f"kernel_weights_all_lags_{model_key}_per_region"
        putils.save_figure_with_options(fig, file_formats=["png", 'pdf'],
                                      filename=fname,
                                      output_dir=output_folder)
        plt.close(fig)

        # --- Plot 2: Heatmap showing kernel weights across lags and regions for each kernel ---
        for kernel in available_kernels:
            # Collect data for this kernel across all regions
            heatmap_data = []
            region_labels = []

            for region in ordered_regions:
                if kernel in kernel_data_by_region[region]:
                    kdata = kernel_data_by_region[region][kernel]
                    heatmap_data.append(kdata['mean'])
                    region_labels.append(f"{region}\n(n={kdata['n_neurons']})")

            if not heatmap_data:
                continue

            # Get lags (should be same for all regions for this kernel)
            lags = None
            for region in ordered_regions:
                if kernel in kernel_data_by_region[region]:
                    lags = kernel_data_by_region[region][kernel]['lags']
                    break

            if lags is None:
                continue

            heatmap_matrix = np.array(heatmap_data)

            # Create heatmap
            fig, ax = plt.subplots(figsize=(max(10, len(lags) * 0.5), max(6, len(region_labels) * 0.4)))

            # Determine color scale limits symmetrically around zero
            vmax = np.abs(heatmap_matrix).max()
            vmin = -vmax

            im = ax.imshow(heatmap_matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest',
                          vmin=vmin, vmax=vmax)

            # Set ticks
            ax.set_xticks(np.arange(len(lags)))
            ax.set_xticklabels([f'{lag:.2f}' for lag in lags], rotation=45, ha='right', fontsize=8)
            ax.set_yticks(np.arange(len(region_labels)))
            ax.set_yticklabels(region_labels, fontsize=9)

            ax.set_xlabel('Lag (s)', fontsize=11)
            ax.set_ylabel('Brain Region', fontsize=11)
            ax.set_title(f'{kernel} weights across lags and regions: {model_label}',
                        fontsize=13, fontweight='bold')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Coefficient', rotation=270, labelpad=20, fontsize=10)

            # Add value annotations (only if not too many cells)
            if len(region_labels) * len(lags) < 200:
                for i in range(len(region_labels)):
                    for j in range(len(lags)):
                        text = ax.text(j, i, f'{heatmap_matrix[i, j]:.2f}',
                                     ha="center", va="center",
                                     color="white" if abs(heatmap_matrix[i, j]) > vmax*0.5 else "black",
                                     fontsize=6)

            plt.tight_layout()

            fname = f"kernel_weights_heatmap_{kernel}_{model_key}"
            putils.save_figure_with_options(fig, file_formats=["png", 'pdf'],
                                          filename=fname,
                                          output_dir=output_folder)
            plt.close(fig)

        # --- Plot 3: Direct comparison of kernel amplitudes at each lag ---
        # For each region, create a plot comparing kernel amplitudes across all lags
        n_rows = math.ceil(len(ordered_regions) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(-1)

        for ax, region in zip(axes, ordered_regions):
            region_data = kernel_data_by_region[region]

            if not region_data:
                ax.set_visible(False)
                continue

            # Get all lags (should be same for all kernels in this region)
            all_lags = None
            for kdata in region_data.values():
                if all_lags is None:
                    all_lags = kdata['lags']
                    break

            if all_lags is None:
                ax.set_visible(False)
                continue

            # For each lag, compare amplitudes across kernels
            lag_positions = np.arange(len(all_lags))
            bar_width = 0.8 / len(available_kernels)  # Divide space among kernels

            for k_idx, kernel in enumerate(available_kernels):
                if kernel not in region_data:
                    continue

                kdata = region_data[kernel]
                mean = kdata['mean']
                sem = kdata['sem']

                # Position bars for this kernel
                positions = lag_positions + (k_idx - len(available_kernels)/2 + 0.5) * bar_width

                color = color_map[kernel]
                ax.bar(positions, mean, bar_width, yerr=sem, label=kernel,
                      color=color, alpha=0.7, capsize=2)

            ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
            ax.set_title(f'{region}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Lag (s)', fontsize=9)
            ax.set_ylabel('Coefficient', fontsize=9)
            ax.set_xticks(lag_positions)
            ax.set_xticklabels([f'{lag:.2f}' for lag in all_lags], rotation=45, ha='right', fontsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=7, loc='best')

        for ax in axes[len(ordered_regions):]:
            ax.set_visible(False)

        plt.suptitle(f'Kernel amplitude comparison at each lag: {model_label}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        fname = f"kernel_amplitude_comparison_per_lag_{model_key}"
        putils.save_figure_with_options(fig, file_formats=["png", 'pdf'],
                                      filename=fname,
                                      output_dir=output_folder)
        plt.close(fig)

        # --- Plot 4: Heatmap comparing kernel amplitudes across lags (aggregated across regions) ---
        # Create a matrix: rows = kernels, columns = lags
        # First, determine common lags across all kernels
        common_lags = None
        for region_data in kernel_data_by_region.values():
            for kdata in region_data.values():
                if common_lags is None:
                    common_lags = kdata['lags']
                break
            if common_lags is not None:
                break

        if common_lags is not None:
            comparison_matrix = []
            kernel_labels_for_heatmap = []

            for kernel in available_kernels:
                # Aggregate across regions (weighted by number of neurons)
                kernel_means_across_lags = None
                total_neurons = 0

                for region in ordered_regions:
                    if kernel in kernel_data_by_region[region]:
                        kdata = kernel_data_by_region[region][kernel]
                        n = kdata['n_neurons']

                        if kernel_means_across_lags is None:
                            kernel_means_across_lags = kdata['mean'] * n
                        else:
                            kernel_means_across_lags += kdata['mean'] * n
                        total_neurons += n

                if kernel_means_across_lags is not None and total_neurons > 0:
                    kernel_means_across_lags /= total_neurons
                    comparison_matrix.append(kernel_means_across_lags)
                    kernel_labels_for_heatmap.append(kernel)

            if comparison_matrix:
                comparison_matrix = np.array(comparison_matrix)

                fig, ax = plt.subplots(figsize=(max(10, len(common_lags) * 0.5), max(6, len(kernel_labels_for_heatmap) * 0.5)))

                # Symmetric color scale
                vmax = np.abs(comparison_matrix).max()
                vmin = -vmax

                im = ax.imshow(comparison_matrix, aspect='auto', cmap='RdBu_r',
                             interpolation='nearest', vmin=vmin, vmax=vmax)

                ax.set_xticks(np.arange(len(common_lags)))
                ax.set_xticklabels([f'{lag:.2f}' for lag in common_lags], rotation=45, ha='right', fontsize=9)
                ax.set_yticks(np.arange(len(kernel_labels_for_heatmap)))
                ax.set_yticklabels(kernel_labels_for_heatmap, fontsize=10)

                ax.set_xlabel('Lag (s)', fontsize=11)
                ax.set_ylabel('Kernel', fontsize=11)
                ax.set_title(f'Kernel comparison across lags (aggregated): {model_label}',
                           fontsize=13, fontweight='bold')

                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Mean Coefficient', rotation=270, labelpad=20, fontsize=10)

                # Add annotations
                if len(kernel_labels_for_heatmap) * len(common_lags) < 150:
                    for i in range(len(kernel_labels_for_heatmap)):
                        for j in range(len(common_lags)):
                            text = ax.text(j, i, f'{comparison_matrix[i, j]:.2f}',
                                         ha="center", va="center",
                                         color="white" if abs(comparison_matrix[i, j]) > vmax*0.5 else "black",
                                         fontsize=7)

                plt.tight_layout()

                fname = f"kernel_comparison_heatmap_aggregated_{model_key}"
                putils.save_figure_with_options(fig, file_formats=["png", 'pdf'],
                                              filename=fname,
                                              output_dir=output_folder)
                plt.close(fig)


def analyze_kernel_amplitude_differences_2(
        dfs,
        model_labels,
        output_folder,
        whisker_kernels,
        area_groups,
        area_colors,
        n_cols=3):
    """
    Analyze the distribution of amplitude differences between kernels (e.g., kernel_1 - kernel_0)
    across neurons at each lag. This shows how heterogeneous the kernel differences are across
    the neuron population.

    Parameters
    ----------
    dfs : dict
        Dictionary of dataframes {model_key: df}
    model_labels : dict
        Dictionary of model labels {model_key: label}
    output_folder : str
        Output directory
    whisker_kernels : list of str
        List of kernel names
    area_groups : dict
        Brain region groupings
    area_colors : dict
        Colors for brain regions
    n_cols : int
        Number of subplot columns
    """

    os.makedirs(output_folder, exist_ok=True)

    def extract_sorted_kernel_indices(predictors, kernel_name):
        pattern = re.compile(fr"^{kernel_name}_t([+-]\d+\.\d+)s$")
        matches = []
        for i, p in enumerate(predictors):
            m = pattern.match(p)
            if m:
                lag = float(m.group(1))
                matches.append((lag, i))
        matches_sorted = sorted(matches, key=lambda x: x[0])
        idx_sorted = [i for (_, i) in matches_sorted]
        lags_sorted = [lag for (lag, _) in matches_sorted]
        return idx_sorted, lags_sorted

    # Get ordered regions
    ordered_regions = []
    for group_name, areas in area_groups.items():
        for area in areas:
            for df in dfs.values():
                if area in df["area_acronym_custom"].values:
                    ordered_regions.append(area)
                    break

    # For each model with multiple kernels, compute differences
    for model_key, df in dfs.items():
        if model_key == "1k":  # Skip single kernel model
            continue

        model_label = model_labels[model_key]
        print(f"Analyzing amplitude differences for {model_label}...")

        # Identify kernel pairs (e.g., kernel_0 and kernel_1 for same stimulus type)
        # Group kernels by base name
        kernel_groups = {}
        for kernel in whisker_kernels:
            # Extract base name - handles patterns like:
            # 'whisker_hits_stim_0' -> 'whisker_hits_stim'
            # 'whisker_misses_stim_0' -> 'whisker_misses_stim'
            # 'auditory_stim_0' -> 'auditory_stim'
            # 'whisker_stim_0' -> 'whisker_stim'
            import re
            # Remove trailing digit(s) with optional underscore before them
            match = re.match(r'^(.+?)_?(\d+)$', kernel)
            if match:
                base_name = match.group(1)
                if base_name not in kernel_groups:
                    kernel_groups[base_name] = []
                kernel_groups[base_name].append(kernel)
            else:
                # No number at end, treat whole name as base
                if kernel not in kernel_groups:
                    kernel_groups[kernel] = []
                kernel_groups[kernel].append(kernel)

        # For each kernel group with multiple kernels, compute differences
        for base_name, kernels_in_group in kernel_groups.items():
            if len(kernels_in_group) < 2:
                continue

            # Sort kernels by number (0, 1, 2, ...)
            kernels_sorted = sorted(kernels_in_group)

            # Compute differences between consecutive kernels
            for i in range(len(kernels_sorted) - 1):
                kernel_0 = kernels_sorted[i]
                kernel_1 = kernels_sorted[i + 1]

                print(f"  Computing {kernel_1} - {kernel_0}...")

                # Collect amplitude differences per neuron per region
                amplitude_diffs_by_region = {}  # {region: {lag_idx: [diff values]}}

                for region in ordered_regions:
                    region_df = df[df["area_acronym_custom"] == region]
                    neuron_groups = region_df.groupby(["mouse_id", "neuron_id"])

                    amplitude_diffs_by_region[region] = {}

                    for (mouse_id, neuron_id), grp in neuron_groups:
                        # Get coefficients for both kernels
                        kernel_0_coefs = None
                        kernel_1_coefs = None
                        lags = None

                        for _, row in grp.iterrows():
                            predictors = row["predictors_full"]
                            coefs = row["coef_full_mean"]

                            # Extract kernel_0
                            idx_0, lags_0 = extract_sorted_kernel_indices(predictors, kernel_0)
                            if idx_0:
                                coefs_array = np.array(coefs)
                                kernel_0_coefs = coefs_array[idx_0]
                                lags = lags_0

                            # Extract kernel_1
                            idx_1, lags_1 = extract_sorted_kernel_indices(predictors, kernel_1)
                            if idx_1:
                                coefs_array = np.array(coefs)
                                kernel_1_coefs = coefs_array[idx_1]

                            break  # Only need first row per neuron

                        # Compute difference
                        if kernel_0_coefs is not None and kernel_1_coefs is not None:
                            if len(kernel_0_coefs) == len(kernel_1_coefs):
                                diff = kernel_1_coefs - kernel_0_coefs

                                # Store by lag
                                for lag_idx, diff_val in enumerate(diff):
                                    if lag_idx not in amplitude_diffs_by_region[region]:
                                        amplitude_diffs_by_region[region][lag_idx] = []
                                    amplitude_diffs_by_region[region][lag_idx].append(diff_val)

                # Also collect differences by reward group (R+ = 1, R- = 0)
                amplitude_diffs_by_reward = {}  # {reward_group: {lag_idx: [diff values]}}
                amplitude_diffs_by_region_and_reward = {}  # {region: {reward_group: {lag_idx: [diff values]}}}
                amplitude_diffs_lag_avg = {}  # {region: {reward_group: [mean-across-lags per neuron]}}

                for region in ordered_regions:
                    region_df = df[df["area_acronym_custom"] == region]
                    neuron_groups = region_df.groupby(["mouse_id", "neuron_id"])

                    amplitude_diffs_by_region_and_reward[region] = {}
                    amplitude_diffs_lag_avg[region] = {}

                    for (mouse_id, neuron_id), grp in neuron_groups:
                        # Get reward group for this neuron
                        reward_group = grp['reward_group'].iloc[0]

                        # Initialize reward group dict if needed
                        if reward_group not in amplitude_diffs_by_reward:
                            amplitude_diffs_by_reward[reward_group] = {}
                            for lag_idx in range(len(lags) if lags else 0):
                                amplitude_diffs_by_reward[reward_group][lag_idx] = []

                        if reward_group not in amplitude_diffs_by_region_and_reward[region]:
                            amplitude_diffs_by_region_and_reward[region][reward_group] = {}
                            for lag_idx in range(len(lags) if lags else 0):
                                amplitude_diffs_by_region_and_reward[region][reward_group][lag_idx] = []

                        # Get coefficients for both kernels
                        kernel_0_coefs = None
                        kernel_1_coefs = None

                        for _, row in grp.iterrows():
                            predictors = row["predictors_full"]
                            coefs = row["coef_full_mean"]

                            # Extract kernel_0
                            idx_0, lags_0 = extract_sorted_kernel_indices(predictors, kernel_0)
                            if idx_0:
                                coefs_array = np.array(coefs)
                                kernel_0_coefs = coefs_array[idx_0]

                            # Extract kernel_1
                            idx_1, lags_1 = extract_sorted_kernel_indices(predictors, kernel_1)
                            if idx_1:
                                coefs_array = np.array(coefs)
                                kernel_1_coefs = coefs_array[idx_1]

                            break  # Only need first row per neuron

                        # Compute difference and store by reward group
                        if kernel_0_coefs is not None and kernel_1_coefs is not None:
                            if len(kernel_0_coefs) == len(kernel_1_coefs):
                                diff = kernel_1_coefs - kernel_0_coefs

                                # Store by lag and reward group (across all regions)
                                for lag_idx, diff_val in enumerate(diff):
                                    if lag_idx not in amplitude_diffs_by_reward[reward_group]:
                                        amplitude_diffs_by_reward[reward_group][lag_idx] = []
                                    amplitude_diffs_by_reward[reward_group][lag_idx].append(diff_val)

                                    # Also store by region and reward group
                                    if lag_idx not in amplitude_diffs_by_region_and_reward[region][reward_group]:
                                        amplitude_diffs_by_region_and_reward[region][reward_group][lag_idx] = []
                                    amplitude_diffs_by_region_and_reward[region][reward_group][lag_idx].append(diff_val)

                                # Store mean across lags for this neuron
                                if reward_group not in amplitude_diffs_lag_avg[region]:
                                    amplitude_diffs_lag_avg[region][reward_group] = []
                                amplitude_diffs_lag_avg[region][reward_group].append(float(np.mean(diff)))

                # Plot distributions for each region
                n_rows = math.ceil(len(ordered_regions) / n_cols)

                if lags is None:
                    continue

                n_lags = len(lags)



                # --- Compare distributions across reward groups ---
                # Create plots comparing distributions between reward groups at each lag
                from scipy import stats

                # Map reward group values to labels
                reward_group_labels = {1: 'R+', 0: 'R-'}
                reward_group_colors = {1: 'forestgreen', 0: 'crimson'}

                # Get available reward groups
                available_reward_groups = sorted(amplitude_diffs_by_reward.keys())

                # # Plot 1: Overlaid histograms for each reward group at each lag, one subplot per region
                # for lag_idx, lag in enumerate(lags):
                #     n_rows = math.ceil(len(ordered_regions) / n_cols)
                #     fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                #     axes = np.array(axes).reshape(-1)

                #     for ax, region in zip(axes, ordered_regions):
                #         # Check if region has data for this lag
                #         has_data = False
                #         all_diffs_region = []
                #         region_data_by_group = {}

                #         for reward_group in available_reward_groups:
                #             if (region in amplitude_diffs_by_region_and_reward and
                #                 reward_group in amplitude_diffs_by_region_and_reward[region] and
                #                 lag_idx in amplitude_diffs_by_region_and_reward[region][reward_group]):
                #                 diffs = amplitude_diffs_by_region_and_reward[region][reward_group][lag_idx]
                #                 if len(diffs) > 0:
                #                     has_data = True
                #                     all_diffs_region.extend(diffs)
                #                     region_data_by_group[reward_group] = np.array(diffs)

                #         if not has_data:
                #             ax.set_visible(False)
                #             continue

                #         # Determine shared bin edges for this region
                #         bins = np.linspace(np.min(all_diffs_region), np.max(all_diffs_region), 41)  # 40 bins

                #         # Plot histograms with shared bins
                #         for reward_group in available_reward_groups:
                #             if reward_group in region_data_by_group:
                #                 diffs = region_data_by_group[reward_group]
                #                 label = reward_group_labels.get(reward_group, f'Group {reward_group}')
                #                 color = reward_group_colors.get(reward_group, 'gray')
                #                 ax.hist(diffs, bins=bins, alpha=0.5, label=f'{label} (n={len(diffs)})', density=True,
                #                        color=color, edgecolor='black', linewidth=0.5)

                #         # Add statistical test if we have both groups
                #         if len(region_data_by_group) >= 2:
                #             group_keys = sorted(region_data_by_group.keys())
                #             data1 = region_data_by_group[group_keys[0]]
                #             data2 = region_data_by_group[group_keys[1]]

                #             # Kolmogorov-Smirnov test (tests if distributions are different)
                #             ks_stat, ks_pvalue = stats.ks_2samp(data1, data2)

                #             # Effect size: Cliff's Delta (non-parametric effect size)
                #             n1, n2 = len(data1), len(data2)
                #             comparisons = np.sum(data1[:, None] > data2[None, :]) - np.sum(data1[:, None] < data2[None, :])
                #             cliffs_delta = comparisons / (n1 * n2)

                #             # Determine significance level
                #             if ks_pvalue < 0.001:
                #                 sig_str = '***'
                #             elif ks_pvalue < 0.01:
                #                 sig_str = '**'
                #             elif ks_pvalue < 0.05:
                #                 sig_str = '*'
                #             else:
                #                 sig_str = 'n.s.'

                #             test_text = f"KS: p={ks_pvalue:.3f} {sig_str}\nδ={cliffs_delta:.3f}"
                #             ax.text(0.98, 0.98, test_text, transform=ax.transAxes,
                #                    fontsize=6, verticalalignment='top', horizontalalignment='right',
                #                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                #         ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
                #         ax.set_title(f'{region}', fontsize=10, fontweight='bold')
                #         ax.set_xlabel(f'Amp. Diff. ({kernel_1} - {kernel_0})', fontsize=8)
                #         ax.set_ylabel('Density', fontsize=8)
                #         ax.legend(fontsize=7, loc='upper left')
                #         ax.spines["top"].set_visible(False)
                #         ax.spines["right"].set_visible(False)
                #         ax.tick_params(labelsize=7)

                #     # Hide unused subplots
                #     for ax in axes[len(ordered_regions):]:
                #         ax.set_visible(False)

                #     plt.suptitle(f'Reward Group Comparison by Region at Lag {lag:.3f}s\n{model_label}',
                #                fontsize=14, fontweight='bold')
                #     plt.tight_layout(rect=[0, 0, 1, 0.96])

                #     fname = f"amplitude_diff_by_reward_per_area_lag{lag:.3f}s_{kernel_1}_minus_{kernel_0}_{model_key}"
                #     putils.save_figure_with_options(fig, file_formats=["png"],
                #                                   filename=fname,
                #                                   output_dir=output_folder)
                #     plt.close(fig)

                # Plot 2: Box plots comparing reward groups across all lags
                # fig, axes = plt.subplots(1, n_lags, figsize=(n_lags * 3, 5), sharey=True)
                # if n_lags == 1:
                #     axes = [axes]

                # for lag_idx, (lag, ax) in enumerate(zip(lags, axes)):
                #     group_data = []
                #     group_labels = []
                #     group_colors = []

                #     for reward_group in available_reward_groups:
                #         if lag_idx in amplitude_diffs_by_reward[reward_group]:
                #             diffs = amplitude_diffs_by_reward[reward_group][lag_idx]
                #             if len(diffs) > 0:
                #                 group_data.append(diffs)
                #                 group_labels.append(reward_group_labels.get(reward_group, f'Group {reward_group}'))
                #                 group_colors.append(reward_group_colors.get(reward_group, 'gray'))

                #     if group_data:
                #         # Create box plot
                #         bp = ax.boxplot(group_data, labels=group_labels, patch_artist=True)

                #         # Color boxes by reward group
                #         for patch, color in zip(bp['boxes'], group_colors):
                #             patch.set_facecolor(color)
                #             patch.set_alpha(0.7)

                #         ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
                #         ax.set_title(f'Lag {lag:.2f}s', fontsize=10)
                #         ax.set_xlabel('Reward Group', fontsize=9)
                #         if lag_idx == 0:
                #             ax.set_ylabel(f'Amplitude Difference\n({kernel_1} - {kernel_0})', fontsize=9)
                #         ax.tick_params(axis='x', rotation=45)
                #         ax.spines["top"].set_visible(False)
                #         ax.spines["right"].set_visible(False)

                # plt.suptitle(f'Reward Group Comparison Across Lags\n{kernel_1} - {kernel_0} | {model_label}',
                #            fontsize=14, fontweight='bold')
                # plt.tight_layout(rect=[0, 0, 1, 0.96])

                # fname = f"amplitude_diff_boxplot_by_reward_{kernel_1}_minus_{kernel_0}_{model_key}"
                # putils.save_figure_with_options(fig, file_formats=["png"],
                #                               filename=fname,
                #                               output_dir=output_folder)
                # plt.close(fig)

                # Plot 3: Summary heatmap showing effect direction and significance
                # Create a matrix: regions × lags showing which group has higher mean and if significant
                # n_lags = len(lags)
                # effect_matrix = np.zeros((len(ordered_regions), n_lags))  # Will store effect direction
                # pvalue_matrix = np.ones((len(ordered_regions), n_lags))   # Will store p-values

                # for region_idx, region in enumerate(ordered_regions):
                #     for lag_idx, lag in enumerate(lags):
                #         if (region in amplitude_diffs_by_region_and_reward and
                #             len(amplitude_diffs_by_region_and_reward[region]) >= 2):

                #             # Get data for both reward groups
                #             group_keys = sorted(amplitude_diffs_by_region_and_reward[region].keys())

                #             if (len(group_keys) >= 2 and
                #                 lag_idx in amplitude_diffs_by_region_and_reward[region].get(group_keys[0], {}) and
                #                 lag_idx in amplitude_diffs_by_region_and_reward[region].get(group_keys[1], {})):

                #                 data1 = np.array(amplitude_diffs_by_region_and_reward[region][group_keys[0]][lag_idx])
                #                 data2 = np.array(amplitude_diffs_by_region_and_reward[region][group_keys[1]][lag_idx])

                #                 if len(data1) > 0 and len(data2) > 0:
                #                     # KS test
                #                     ks_stat, ks_pvalue = stats.ks_2samp(data1, data2)
                #                     pvalue_matrix[region_idx, lag_idx] = ks_pvalue

                #                     # Effect direction: positive if R- (group_keys[0]=0) > R+ (group_keys[1]=1)
                #                     # negative if R+ > R-
                #                     mean_diff = np.mean(data1) - np.mean(data2)

                #                     # Store signed effect size (Cliff's Delta with sign)
                #                     n1, n2 = len(data1), len(data2)
                #                     comparisons = np.sum(data1[:, None] > data2[None, :]) - np.sum(data1[:, None] < data2[None, :])
                #                     cliffs_delta = comparisons / (n1 * n2)

                #                     effect_matrix[region_idx, lag_idx] = cliffs_delta

                # # Create heatmap
                # fig, ax = plt.subplots(1, 1, figsize=(max(10, n_lags * 1.5), max(8, len(ordered_regions) * 0.4)))

                # # Create custom colormap: green for R+ > R-, white for 0, red for R- > R+
                # # Note: Cliff's Delta is positive when data1 (R-) > data2 (R+)
                # from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

                # # Mask non-significant effects
                # masked_effect = effect_matrix.copy()
                # masked_effect[pvalue_matrix >= 0.05] = 0  # Set non-significant to 0

                # # Create custom green-white-red colormap
                # # Colormap goes from min to max value:
                # # Most negative (R+ > R-) → green
                # # Zero → white
                # # Most positive (R- > R+) → red
                # colors_rgb = ['forestgreen', 'white', 'crimson']
                # n_bins = 256
                # cmap_custom = LinearSegmentedColormap.from_list('custom_gwr', colors_rgb, N=n_bins)

                # # Plot heatmap
                # vmax = max(abs(np.min(masked_effect)), abs(np.max(masked_effect)))
                # if vmax == 0:
                #     vmax = 1  # Avoid division by zero if all non-significant

                # norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                # im = ax.imshow(masked_effect, cmap=cmap_custom, aspect='auto', norm=norm)

                # # Add significance markers
                # for region_idx in range(len(ordered_regions)):
                #     for lag_idx in range(n_lags):
                #         pval = pvalue_matrix[region_idx, lag_idx]
                #         effect = effect_matrix[region_idx, lag_idx]

                #         # Add stars for significance
                #         if pval < 0.001:
                #             marker = '***'
                #         elif pval < 0.01:
                #             marker = '**'
                #         elif pval < 0.05:
                #             marker = '*'
                #         else:
                #             marker = ''

                #         if marker:
                #             ax.text(lag_idx, region_idx, marker, ha='center', va='center',
                #                    color='black', fontsize=8, fontweight='bold')

                # # Set ticks and labels
                # ax.set_xticks(np.arange(n_lags))
                # ax.set_xticklabels([f'{lag:.2f}s' for lag in lags], rotation=45, ha='right')
                # ax.set_yticks(np.arange(len(ordered_regions)))
                # ax.set_yticklabels(ordered_regions)

                # ax.set_xlabel('Lag', fontsize=11)
                # ax.set_ylabel('Brain Region', fontsize=11)
                # ax.set_title(f'Reward Group Effect Summary: {kernel_1} - {kernel_0}\n'
                #            f'{model_label}\n'
                #            f'(Red: R- > R+, Green: R+ > R-, * p<0.05, ** p<0.01, *** p<0.001)',
                #            fontsize=12, fontweight='bold', pad=20)

                # # Add colorbar
                # cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                # cbar.set_label("Cliff's Delta\n(Effect Size & Direction)", rotation=270, labelpad=20, fontsize=10)

                # plt.tight_layout()

                # fname = f"amplitude_diff_summary_heatmap_{kernel_1}_minus_{kernel_0}_{model_key}"
                # putils.save_figure_with_options(fig, file_formats=["png"],
                #                               filename=fname,
                #                               output_dir=output_folder)
                # plt.close(fig)

                # Plot 4: Statistical comparison - compute p-values between reward groups
                # Create a summary table showing mean, std, and statistical tests
                print(f"\n  Statistical Comparison Across Reward Groups:")
                print(f"  Kernel Pair: {kernel_1} - {kernel_0}")

                for lag_idx, lag in enumerate(lags):
                    print(f"\n  Lag {lag:.3f}s:")

                    # Collect data for all reward groups at this lag
                    reward_data_dict = {}
                    for reward_group in available_reward_groups:
                        if lag_idx in amplitude_diffs_by_reward[reward_group]:
                            diffs = amplitude_diffs_by_reward[reward_group][lag_idx]
                            if len(diffs) > 0:
                                label = reward_group_labels.get(reward_group, f'Group {reward_group}')
                                reward_data_dict[label] = diffs
                                print(f"    {label}: mean={np.mean(diffs):.4f}, std={np.std(diffs):.4f}, n={len(diffs)}")

                    # Perform pairwise comparisons (Mann-Whitney U test)
                    if len(reward_data_dict) >= 2:
                        print(f"    Pairwise comparisons (Mann-Whitney U test):")
                        group_list = list(reward_data_dict.keys())
                        for i in range(len(group_list)):
                            for j in range(i + 1, len(group_list)):
                                group1 = group_list[i]
                                group2 = group_list[j]
                                data1 = reward_data_dict[group1]
                                data2 = reward_data_dict[group2]

                                # Mann-Whitney U test (non-parametric)
                                statistic, pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                                print(f"      {group1} vs {group2}: p={pvalue:.4f}")

                def _sig_str(p):
                    if p < 0.001: return '***'
                    if p < 0.01:  return '**'
                    if p < 0.05:  return '*'
                    return 'n.s.'

                # Plot 5: Lag-averaged amplitude difference distributions, R+ and R- on the same plot
                n_rows = math.ceil(len(ordered_regions) / n_cols)
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                axes = np.array(axes).reshape(-1)

                for ax, region in zip(axes, ordered_regions):
                    region_data_by_group = {}
                    all_diffs_region = []

                    for reward_group in available_reward_groups:
                        if (region in amplitude_diffs_lag_avg and
                                reward_group in amplitude_diffs_lag_avg[region]):
                            vals = amplitude_diffs_lag_avg[region][reward_group]
                            if len(vals) > 0:
                                region_data_by_group[reward_group] = np.array(vals)
                                all_diffs_region.extend(vals)

                    if not all_diffs_region:
                        ax.set_visible(False)
                        continue

                    bins = np.linspace(np.min(all_diffs_region), np.max(all_diffs_region), 41)

                    for reward_group in available_reward_groups:
                        if reward_group in region_data_by_group:
                            diffs = region_data_by_group[reward_group]
                            label = reward_group_labels.get(reward_group, f'Group {reward_group}')
                            color = reward_group_colors.get(reward_group, 'gray')
                            ax.hist(diffs, bins=bins, alpha=0.5, label=f'{label} (n={len(diffs)})',
                                    density=True, color=color, edgecolor='black', linewidth=0.5)

                    # KS test + Mann-Whitney U + Cliff's delta if both groups present
                    if len(region_data_by_group) >= 2:
                        group_keys = sorted(region_data_by_group.keys())
                        data1 = region_data_by_group[group_keys[0]]
                        data2 = region_data_by_group[group_keys[1]]

                        ks_stat, ks_pvalue = stats.ks_2samp(data1, data2)
                        mw_stat, mw_pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')

                        n1, n2 = len(data1), len(data2)
                        comparisons = np.sum(data1[:, None] > data2[None, :]) - np.sum(data1[:, None] < data2[None, :])
                        cliffs_delta = comparisons / (n1 * n2)

                        test_text = (f"KS: p={ks_pvalue:.3f} {_sig_str(ks_pvalue)}\n"
                                     f"MW: p={mw_pvalue:.3f} {_sig_str(mw_pvalue)}\n"
                                     f"δ={cliffs_delta:.3f}")
                        ax.text(0.98, 0.98, test_text, transform=ax.transAxes,
                                fontsize=6, verticalalignment='top', horizontalalignment='right',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax.set_title(f'{region}', fontsize=10, fontweight='bold')
                    ax.set_xlabel(f'Mean Amp. Diff. across lags ({kernel_1} - {kernel_0})', fontsize=8)
                    ax.set_ylabel('Density', fontsize=8)
                    ax.legend(fontsize=7, loc='upper left')
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.tick_params(labelsize=7)

                for ax in axes[len(ordered_regions):]:
                    ax.set_visible(False)

                plt.suptitle(f'Lag-Averaged Weight Change by Region (R+ vs R-)\n{model_label}',
                             fontsize=14, fontweight='bold')
                plt.tight_layout(rect=[0, 0, 1, 0.96])

                fname = f"amplitude_diff_lag_avg_by_reward_per_area_{kernel_1}_minus_{kernel_0}_{model_key}"
                putils.save_figure_with_options(fig, file_formats=["png", "pdf"],
                                               filename=fname,
                                               output_dir=output_folder)
                plt.close(fig)

                # Plot 6: Lag-averaged amplitude difference, pooled across all areas (single panel)
                pooled_by_group = {}
                for region in ordered_regions:
                    if region not in amplitude_diffs_lag_avg:
                        continue
                    for reward_group in available_reward_groups:
                        if reward_group in amplitude_diffs_lag_avg[region]:
                            pooled_by_group.setdefault(reward_group, []).extend(
                                amplitude_diffs_lag_avg[region][reward_group]
                            )

                if len(pooled_by_group) >= 1:
                    all_pooled = []
                    for vals in pooled_by_group.values():
                        all_pooled.extend(vals)
                    bins = np.linspace(np.min(all_pooled), np.max(all_pooled), 41)

                    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                    for reward_group in available_reward_groups:
                        if reward_group in pooled_by_group:
                            diffs = np.array(pooled_by_group[reward_group])
                            label = reward_group_labels.get(reward_group, f'Group {reward_group}')
                            color = reward_group_colors.get(reward_group, 'gray')
                            ax.hist(diffs, bins=bins, alpha=0.5, label=f'{label} (n={len(diffs)})',
                                    density=True, color=color, edgecolor='black', linewidth=0.5)

                    # KS + Mann-Whitney U + Cliff's delta
                    if len(pooled_by_group) >= 2:
                        group_keys = sorted(pooled_by_group.keys())
                        data1 = np.array(pooled_by_group[group_keys[0]])
                        data2 = np.array(pooled_by_group[group_keys[1]])

                        ks_stat, ks_pvalue = stats.ks_2samp(data1, data2)
                        mw_stat, mw_pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')

                        n1, n2 = len(data1), len(data2)
                        comparisons = np.sum(data1[:, None] > data2[None, :]) - np.sum(data1[:, None] < data2[None, :])
                        cliffs_delta = comparisons / (n1 * n2)

                        test_text = (f"KS: p={ks_pvalue:.3f} {_sig_str(ks_pvalue)}\n"
                                     f"MW: p={mw_pvalue:.3f} {_sig_str(mw_pvalue)}\n"
                                     f"δ={cliffs_delta:.3f}")
                        ax.text(0.98, 0.98, test_text, transform=ax.transAxes,
                                fontsize=9, verticalalignment='top', horizontalalignment='right',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

                    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax.set_xlabel(f'Mean Amp. Diff. across lags ({kernel_1} - {kernel_0})', fontsize=11)
                    ax.set_ylabel('Density', fontsize=11)
                    ax.set_title(f'Lag-Averaged Weight Change – All Areas (R+ vs R-)\n{model_label}',
                                 fontsize=13, fontweight='bold')
                    ax.legend(fontsize=9)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    plt.tight_layout()

                    fname = f"amplitude_diff_lag_avg_by_reward_all_areas_{kernel_1}_minus_{kernel_0}_{model_key}"
                    putils.save_figure_with_options(fig, file_formats=["png", "pdf"],
                                                   filename=fname,
                                                   output_dir=output_folder)
                    plt.close(fig)


def plot_paired_kernel_comparison_per_area(
        dfs,
        model_labels,
        output_folder,
        area_groups,
        area_colors,
        n_cols=3,
        alpha=0.05):
    """
    For each model and brain area, plot two kernels of the same stimulus type
    overlaid (mean ± SEM across neurons), with a paired Wilcoxon signed-rank test
    at each time bin (pairing = same neuron). A colored significance bar is drawn
    above the traces for bins where p < alpha.

    Three figures per model:
      - Whisker hits:   whisker_hits_stim_0 vs whisker_hits_stim_1
      - Whisker misses: whisker_misses_stim_0 vs whisker_misses_stim_1
      - Auditory:       auditory_stim_0       vs auditory_stim_1
    """
    from scipy.stats import ttest_rel

    os.makedirs(output_folder, exist_ok=True)

    # Infer reward group color (dfs are pre-filtered to one reward group)
    _rg_vals = pd.concat(list(dfs.values()))['reward_group'].dropna().unique()
    if len(_rg_vals) == 1 and int(_rg_vals[0]) == 1:
        _col0, _col1, _sig_color = '#90EE90', 'forestgreen', 'darkgreen'
    elif len(_rg_vals) == 1 and int(_rg_vals[0]) == 0:
        _col0, _col1, _sig_color = '#FF9999', 'crimson', 'darkred'
    else:
        _col0, _col1, _sig_color = 'steelblue', 'navy', 'navy'

    # (kernel_0, kernel_1, descriptive_title, color_0, color_1, sig_bar_color)
    KERNEL_PAIR_SPECS = [
        ('whisker_hits_stim_0',  'whisker_hits_stim_1',
         'Whisker Hits',
         _col0, _col1, _sig_color),
        ('whisker_misses_stim_0', 'whisker_misses_stim_1',
         'Whisker Misses',
         _col0, _col1, _sig_color),
        ('auditory_stim_0',       'auditory_stim_1',
         'Auditory',
         _col0, _col1, _sig_color),
        ('piezo_reward_0',        'piezo_reward_1',
         'Reward (whisker)',
         _col0, _col1, _sig_color),
        ('piezo_reward_au_0',     'piezo_reward_au_1',
         'Reward (auditory)',
         _col0, _col1, _sig_color),
    ]

    # ── Helper: extract one kernel's (lags, coefs) from a single model row ──
    def _extract_kernel(predictors, coefs, kernel_name):
        pattern = re.compile(fr"^{re.escape(kernel_name)}_t([+-]\d+\.\d+)s$")
        matches = sorted(
            [(float(m.group(1)), i)
             for i, p in enumerate(predictors)
             if (m := pattern.match(p))]
        )
        if not matches:
            return None, None
        lags = np.array([lag for lag, _ in matches])
        idx  = [i for _, i in matches]
        return lags, np.array(coefs)[idx]

    # ── Helper: collect paired matrices across neurons in a region ───────────
    def _collect_paired(region_df, k0_name, k1_name):
        """
        Returns (lags, mat0, mat1) where mat0/mat1 are (n_sessions, n_lags).
        Each row is one session; only rows that have BOTH kernels are included.
        Returns (None, None, None) if fewer than 3 paired rows are found.
        """
        lags_ref = None
        paired0, paired1 = [], []
        for _, row in region_df.iterrows():
            l0, c0 = _extract_kernel(
                row["predictors_full"], row["coef_full_mean"], k0_name)
            l1, c1 = _extract_kernel(
                row["predictors_full"], row["coef_full_mean"], k1_name)
            if l0 is not None and l1 is not None and len(c0) == len(c1):
                if lags_ref is None:
                    lags_ref = l0
                paired0.append(c0)
                paired1.append(c1)
        if len(paired0) < 3 or lags_ref is None:
            return None, None, None
        return lags_ref, np.stack(paired0), np.stack(paired1)

    # ── Helper: paired t-test at every time bin ──────────────────────────────
    def _paired_wilcoxon_per_bin(mat0, mat1):
        """Returns p-value array of length n_lags (paired t-test)."""
        n_lags = mat0.shape[1]
        pvals  = np.ones(n_lags)
        for t in range(n_lags):
            if len(mat0[:, t]) >= 3:
                try:
                    _, pvals[t] = ttest_rel(mat0[:, t], mat1[:, t])
                except Exception:
                    pass
        return pvals

    # ── Ordered regions following area_groups structure ───────────────────────
    ordered_regions = [
        area
        for areas in area_groups.values()
        for area in areas
        if any(area in df["area_acronym_custom"].values for df in dfs.values())
    ]

    # ── Main loop ─────────────────────────────────────────────────────────────
    for model_key, df in dfs.items():
        model_label = model_labels.get(model_key, model_key)

        for k0_name, k1_name, pair_title, col0, col1, sig_color in KERNEL_PAIR_SPECS:

            n_rows = math.ceil(len(ordered_regions) / n_cols)
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(5 * n_cols, 3.5 * n_rows),
                sharex=False
            )
            axes = np.array(axes).reshape(-1)
            any_plotted = False
            # Accumulate per-area p-value arrays for the heatmap
            heatmap_regions = []
            heatmap_pvals   = []   # list of 1-D arrays (one per area)
            heatmap_lags    = None

            for ax, region in zip(axes, ordered_regions):
                region_df = df[df["area_acronym_custom"] == region]
                lags, mat0, mat1 = _collect_paired(region_df, k0_name, k1_name)
                if lags is None:
                    ax.set_visible(False)
                    continue
                any_plotted = True
                if heatmap_lags is None:
                    heatmap_lags = lags

                n_neurons = mat0.shape[0]
                mean0 = np.mean(mat0, axis=0)
                sem0  = np.std(mat0,  axis=0, ddof=1) / np.sqrt(n_neurons)
                mean1 = np.mean(mat1, axis=0)
                sem1  = np.std(mat1,  axis=0, ddof=1) / np.sqrt(n_neurons)

                # Plot both kernels
                ax.plot(lags, mean0, color=col0, lw=2,
                        label=f'stim 0  (n={n_neurons})')
                ax.fill_between(lags, mean0 - sem0, mean0 + sem0,
                                color=col0, alpha=0.25, lw=0)
                ax.plot(lags, mean1, color=col1, lw=2,
                        label='stim 1')
                ax.fill_between(lags, mean1 - sem1, mean1 + sem1,
                                color=col1, alpha=0.25, lw=0)
                ax.axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.4)

                # Paired t-test per time bin
                pvals    = _paired_wilcoxon_per_bin(mat0, mat1)
                sig_mask = pvals < alpha

                # Accumulate for heatmap
                heatmap_regions.append(region)
                heatmap_pvals.append(pvals)

                # Compute y range and set ylim before drawing the significance bar
                all_vals = np.concatenate([
                    mean0 + sem0, mean0 - sem0,
                    mean1 + sem1, mean1 - sem1
                ])
                y_bot  = np.nanmin(all_vals)
                y_top  = np.nanmax(all_vals)
                y_span = max(y_top - y_bot, 1e-9)
                y_bar  = y_top + y_span * 0.08
                bar_h  = y_span * 0.05
                ax.set_ylim(y_bot - y_span * 0.05,
                            y_bar + bar_h + y_span * 0.08)

                # Significance bar above the traces
                if np.any(sig_mask):
                    ax.fill_between(lags, y_bar, y_bar + bar_h,
                                    where=sig_mask,
                                    color=sig_color, alpha=0.75,
                                    linewidth=0, zorder=5)

                ax.set_title(f'{region}', fontsize=10, fontweight='bold')
                ax.set_xlabel('Lag (s)', fontsize=8)
                ax.set_ylabel('Coefficient', fontsize=8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.legend(fontsize=6, loc='best')

            if not any_plotted:
                plt.close(fig)
                continue

            for ax in axes[len(ordered_regions):]:
                ax.set_visible(False)

            pair_clean = f'{k0_name}_vs_{k1_name}'
            plt.suptitle(
                f'{pair_title}: stim_0 vs stim_1 — {model_label}',
                fontsize=13, fontweight='bold'
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            fname = f'paired_kernel_comparison_{pair_clean}_{model_key}'
            putils.save_figure_with_options(
                fig, file_formats=["png", "pdf"],
                filename=fname,
                output_dir=output_folder
            )
            plt.close(fig)

            # ── Heatmap: areas × time bins, cells show p-value + star ────────
            if heatmap_regions and heatmap_lags is not None:
                pval_matrix = np.stack(heatmap_pvals)   # (n_areas, n_lags)
                n_areas, n_lags = pval_matrix.shape

                fig_h, ax_h = plt.subplots(
                    figsize=(max(6, n_lags * 0.6), max(3, n_areas * 0.5))
                )

                # Color by -log10(p); cap at 4 for display
                log_p = np.clip(-np.log10(pval_matrix), 0, 4)
                _cmap_pval = matplotlib.colors.LinearSegmentedColormap.from_list('rw_sig', ['white', 'crimson'])
                im = ax_h.imshow(log_p, aspect='auto', cmap=_cmap_pval,
                                 vmin=0, vmax=4)

                # Annotate each cell with the p-value and a star if significant
                for r_idx in range(n_areas):
                    for c_idx in range(n_lags):
                        p = pval_matrix[r_idx, c_idx]
                        star = '*' if p < alpha else ''
                        txt = f'{p:.3f}{star}' if p >= 0.001 else f'<.001{star}'
                        # White text on dark cells, black on light cells
                        text_color = 'white' if log_p[r_idx, c_idx] > 2 else 'black'
                        ax_h.text(c_idx, r_idx, txt,
                                  ha='center', va='center',
                                  fontsize=6, color=text_color)

                ax_h.set_xticks(np.arange(n_lags))
                ax_h.set_xticklabels([f'{l:.2f}' for l in heatmap_lags],
                                     rotation=45, ha='right', fontsize=7)
                ax_h.set_yticks(np.arange(n_areas))
                ax_h.set_yticklabels(heatmap_regions, fontsize=8)
                ax_h.set_xlabel('Lag (s)', fontsize=9)

                cbar = plt.colorbar(im, ax=ax_h, fraction=0.03, pad=0.02)
                cbar.set_label('−log₁₀(p)', fontsize=8)
                cbar.set_ticks([0, 1, 2, 3, 4])
                cbar.set_ticklabels(['1', '0.1', '0.01', '0.001', '≤0.0001'],
                                    fontsize=7)

                plt.suptitle(
                    f'{pair_title}: paired t-test p-values per bin — {model_label}',
                    fontsize=11, fontweight='bold'
                )
                plt.tight_layout(rect=[0, 0, 1, 0.96])

                fname_hm = f'paired_kernel_pvalue_heatmap_{pair_clean}_{model_key}'
                putils.save_figure_with_options(
                    fig_h, file_formats=["png", "pdf"],
                    filename=fname_hm,
                    output_dir=output_folder
                )
                plt.close(fig_h)


def identify_consistently_increasing_neurons(
        dfs,                     # dict: {"1k": df1, "2k": df2, "3k": df3, "4k": df4}
        model_labels,            # dict: {"1k": "1 whisker kernel", ...}
        output_folder,
        whisker_kernels,         # ["whisker_stim_0", "whisker_stim_1", ...]
        area_groups,
        area_colors,
        consistency_threshold=0.75):  # Fraction of folds that must show increase
    """
    Identify neurons whose kernel weights consistently increase from kernel_0 to kernel_1
    across most folds of cross-validation. Reports which lags show consistent increases.

    Parameters
    ----------
    consistency_threshold : float
        Fraction of folds that must show an increase for it to be considered consistent
        (default: 0.75, i.e., 75% of folds)
    """

    os.makedirs(output_folder, exist_ok=True)

    def extract_sorted_kernel_indices(predictors, kernel_name):
        pattern = re.compile(fr"^{kernel_name}_t([+-]\d+\.\d+)s$")
        matches = []
        for i, p in enumerate(predictors):
            m = pattern.match(p)
            if m:
                lag = float(m.group(1))
                matches.append((lag, i))
        matches_sorted = sorted(matches, key=lambda x: x[0])
        idx_sorted = [i for (_, i) in matches_sorted]
        lags_sorted = [lag for (lag, _) in matches_sorted]
        return idx_sorted, lags_sorted

    # Get ordered regions
    ordered_regions = []
    for group_name, areas in area_groups.items():
        for area in areas:
            for df in dfs.values():
                if area in df["area_acronym_custom"].values:
                    ordered_regions.append(area)
                    break

    # For each model with multiple kernels, find consistently increasing neurons
    for model_key, df in dfs.items():
        if model_key == "1k":  # Skip single kernel model
            continue

        model_label = model_labels[model_key]
        print(f"\nIdentifying consistently increasing neurons for {model_label}...")

        # Identify kernel pairs
        kernel_groups = {}
        for kernel in whisker_kernels:
            import re
            match = re.match(r'^(.+?)_?(\d+)$', kernel)
            if match:
                base_name = match.group(1)
                if base_name not in kernel_groups:
                    kernel_groups[base_name] = []
                kernel_groups[base_name].append(kernel)
            else:
                if kernel not in kernel_groups:
                    kernel_groups[kernel] = []
                kernel_groups[kernel].append(kernel)

        # For each kernel pair
        for base_name, kernels_in_group in kernel_groups.items():
            if len(kernels_in_group) < 2:
                continue

            kernels_sorted = sorted(kernels_in_group)

            for i in range(len(kernels_sorted) - 1):
                kernel_0 = kernels_sorted[i]
                kernel_1 = kernels_sorted[i + 1]

                print(f"\n  Analyzing {kernel_1} vs {kernel_0}...")

                # Store results per region
                consistent_neurons_by_region = {}  # {region: {neuron_id: {lag_idx: consistency_score}}}

                for region in ordered_regions:
                    region_df = df[df["area_acronym_custom"] == region]

                    if len(region_df) == 0:
                        continue

                    consistent_neurons_by_region[region] = {}

                    # Group by neuron (across folds)
                    neuron_groups = region_df.groupby(["mouse_id", "neuron_id"])

                    for (mouse_id, neuron_id), grp in neuron_groups:
                        n_folds = len(grp)

                        if n_folds < 2:  # Need at least 2 folds
                            continue

                        # Collect kernel weights across folds
                        kernel_0_weights_by_fold = []
                        kernel_1_weights_by_fold = []
                        lags = None

                        for _, row in grp.iterrows():
                            predictors = row["predictors_full"]
                            coefs = np.array(row["coef_full_mean"])

                            # Extract kernel_0
                            idx_0, lags_0 = extract_sorted_kernel_indices(predictors, kernel_0)
                            if idx_0:
                                kernel_0_weights_by_fold.append(coefs[idx_0])
                                if lags is None:
                                    lags = lags_0

                            # Extract kernel_1
                            idx_1, lags_1 = extract_sorted_kernel_indices(predictors, kernel_1)
                            if idx_1:
                                kernel_1_weights_by_fold.append(coefs[idx_1])

                        if len(kernel_0_weights_by_fold) == 0 or len(kernel_1_weights_by_fold) == 0:
                            continue

                        kernel_0_weights_by_fold = np.array(kernel_0_weights_by_fold)
                        kernel_1_weights_by_fold = np.array(kernel_1_weights_by_fold)

                        if kernel_0_weights_by_fold.shape != kernel_1_weights_by_fold.shape:
                            continue

                        # For each lag, check consistency of increase across folds
                        n_lags = kernel_0_weights_by_fold.shape[1]
                        neuron_consistency = {}

                        for lag_idx in range(n_lags):
                            # Check how many folds show kernel_1 > kernel_0 at this lag
                            increases = kernel_1_weights_by_fold[:, lag_idx] > kernel_0_weights_by_fold[:, lag_idx]
                            consistency_score = np.mean(increases)

                            if consistency_score >= consistency_threshold:
                                neuron_consistency[lag_idx] = consistency_score

                        if neuron_consistency:
                            neuron_key = f"{mouse_id}_{neuron_id}"
                            consistent_neurons_by_region[region][neuron_key] = neuron_consistency

                # Create summary plots
                if lags is None:
                    continue

                # Plot 1: Heatmap showing number of consistent neurons per region per lag
                n_consistent_matrix = np.zeros((len(ordered_regions), len(lags)))

                for region_idx, region in enumerate(ordered_regions):
                    if region in consistent_neurons_by_region:
                        for neuron_key, lag_consistency in consistent_neurons_by_region[region].items():
                            for lag_idx in lag_consistency.keys():
                                n_consistent_matrix[region_idx, lag_idx] += 1

                fig, ax = plt.subplots(1, 1, figsize=(max(10, len(lags) * 1.5), max(8, len(ordered_regions) * 0.4)))

                im = ax.imshow(n_consistent_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')

                # Add text annotations
                for region_idx in range(len(ordered_regions)):
                    for lag_idx in range(len(lags)):
                        count = int(n_consistent_matrix[region_idx, lag_idx])
                        if count > 0:
                            ax.text(lag_idx, region_idx, str(count), ha='center', va='center',
                                   color='black' if count < n_consistent_matrix.max() / 2 else 'white',
                                   fontsize=8, fontweight='bold')

                ax.set_xticks(np.arange(len(lags)))
                ax.set_xticklabels([f'{lag:.2f}s' for lag in lags], rotation=45, ha='right')
                ax.set_yticks(np.arange(len(ordered_regions)))
                ax.set_yticklabels(ordered_regions)

                ax.set_xlabel('Lag', fontsize=11)
                ax.set_ylabel('Brain Region', fontsize=11)
                ax.set_title(f'Number of Neurons with Consistent Increase\n'
                           f'{kernel_1} > {kernel_0} in ≥{consistency_threshold*100:.0f}% of folds | {model_label}',
                           fontsize=12, fontweight='bold', pad=20)

                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Number of Neurons', rotation=270, labelpad=20, fontsize=10)

                plt.tight_layout()

                fname = f"consistent_increase_heatmap_{kernel_1}_vs_{kernel_0}_{model_key}"
                putils.save_figure_with_options(fig, file_formats=["png", 'pdf'],
                                              filename=fname,
                                              output_dir=output_folder)
                plt.close(fig)

                # Print summary statistics
                print(f"\n  Summary for {kernel_1} > {kernel_0}:")
                for region in ordered_regions:
                    if region in consistent_neurons_by_region:
                        n_neurons = len(consistent_neurons_by_region[region])
                        if n_neurons > 0:
                            # Find most consistent lags
                            lag_counts = {}
                            for neuron_key, lag_consistency in consistent_neurons_by_region[region].items():
                                for lag_idx in lag_consistency.keys():
                                    if lag_idx not in lag_counts:
                                        lag_counts[lag_idx] = 0
                                    lag_counts[lag_idx] += 1

                            if lag_counts:
                                most_consistent_lag_idx = max(lag_counts, key=lag_counts.get)
                                most_consistent_lag = lags[most_consistent_lag_idx]
                                most_consistent_count = lag_counts[most_consistent_lag_idx]

                                print(f"    {region}: {n_neurons} neurons, most consistent at lag {most_consistent_lag:.2f}s ({most_consistent_count} neurons)")


def analyze_kernel_consistency(
        dfs,                     # dict: {"1k": df1, "2k": df2, "3k": df3, "4k": df4}
        model_labels,            # dict: {"1k": "1 whisker kernel", ...}
        output_folder,
        whisker_kernels,         # ["whisker_stim_0", "whisker_stim_1", "whisker_stim_2", ...]
        area_groups,
        area_colors):
    """
    Analyze whether all whisker kernels within a model change similarly across neurons.
    This helps understand if the kernels capture different aspects of the stimulus
    or if they're redundant.
    """

    os.makedirs(output_folder, exist_ok=True)

    def extract_sorted_kernel_indices(predictors, kernel_name):
        pattern = re.compile(fr"^{kernel_name}_t([+-]\d+\.\d+)s$")
        matches = []
        for i, p in enumerate(predictors):
            m = pattern.match(p)
            if m:
                lag = float(m.group(1))
                matches.append((lag, i))
        matches_sorted = sorted(matches, key=lambda x: x[0])
        idx_sorted = [i for (_, i) in matches_sorted]
        lags_sorted = [lag for (lag, _) in matches_sorted]
        return idx_sorted, lags_sorted

    # Get ordered regions
    ordered_regions = []
    for group_name, areas in area_groups.items():
        for area in areas:
            for df in dfs.values():
                if area in df["area_acronym_custom"].values:
                    ordered_regions.append(area)
                    break

    # For each model with multiple kernels, compute correlations between kernels
    for model_key, df in dfs.items():
        if model_key == "1k":  # Skip single kernel model
            continue

        model_label = model_labels[model_key]

        # Collect kernel pairs correlations per region
        correlation_data = []

        for region in ordered_regions:
            region_df = df[df["area_acronym_custom"] == region]
            neuron_groups = region_df.groupby(["mouse_id", "neuron_id"])

            # For each neuron, extract all whisker kernels
            neuron_kernel_dict = {}

            for (mouse_id, neuron_id), grp in neuron_groups:
                kernel_curves = {}

                for kernel in whisker_kernels:
                    kernels_list = []
                    lags_for_kernel = None

                    for _, row in grp.iterrows():
                        predictors = row["predictors_full"]
                        coefs = row["coef_full_mean"]

                        idx, lag_list = extract_sorted_kernel_indices(predictors, kernel)

                        if idx:
                            if lags_for_kernel is None:
                                lags_for_kernel = lag_list
                            kernels_list.append(coefs[idx])

                    if kernels_list:
                        mean_kernel = np.mean(np.stack(kernels_list), axis=0)
                        kernel_curves[kernel] = mean_kernel

                if len(kernel_curves) >= 2:  # Need at least 2 kernels to correlate
                    neuron_kernel_dict[(mouse_id, neuron_id)] = kernel_curves

            # Compute pairwise correlations between kernels
            if neuron_kernel_dict:
                available_kernels = list(next(iter(neuron_kernel_dict.values())).keys())

                for i, kernel1 in enumerate(available_kernels):
                    for kernel2 in available_kernels[i+1:]:
                        correlations = []

                        for neuron_id, kernel_curves in neuron_kernel_dict.items():
                            if kernel1 in kernel_curves and kernel2 in kernel_curves:
                                k1 = kernel_curves[kernel1]
                                k2 = kernel_curves[kernel2]

                                if len(k1) != len(k2):
                                    continue

                                # Compute correlation
                                corr = np.corrcoef(k1, k2)[0, 1]
                                if not np.isnan(corr):
                                    correlations.append(corr)

                        if correlations:
                            correlation_data.append({
                                'region': region,
                                'kernel1': kernel1,
                                'kernel2': kernel2,
                                'mean_corr': np.mean(correlations),
                                'std_corr': np.std(correlations, ddof=1),
                                'n_neurons': len(correlations)
                            })

        if not correlation_data:
            print(f"[WARNING] No correlation data for model {model_key}")
            continue

        corr_df = pd.DataFrame(correlation_data)

        # --- Plot 1: Heatmap of pairwise kernel correlations averaged across regions ---
        # Get unique kernel pairs
        kernel_pairs = corr_df[['kernel1', 'kernel2']].drop_duplicates()

        # Create matrix for heatmap
        unique_kernels = sorted(set(corr_df['kernel1'].tolist() + corr_df['kernel2'].tolist()))
        n_kernels = len(unique_kernels)

        corr_matrix = np.zeros((n_kernels, n_kernels))
        count_matrix = np.zeros((n_kernels, n_kernels))

        for _, row in corr_df.iterrows():
            i = unique_kernels.index(row['kernel1'])
            j = unique_kernels.index(row['kernel2'])
            corr_matrix[i, j] += row['mean_corr']
            corr_matrix[j, i] += row['mean_corr']
            count_matrix[i, j] += 1
            count_matrix[j, i] += 1

        # Average
        corr_matrix = np.divide(corr_matrix, count_matrix, where=count_matrix>0)
        np.fill_diagonal(corr_matrix, 1.0)

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

        _cmap_corr = matplotlib.colors.LinearSegmentedColormap.from_list('rw_corr', ['white', 'forestgreen'])
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=_cmap_corr,
                   center=None, vmin=0, vmax=1,
                   xticklabels=unique_kernels, yticklabels=unique_kernels,
                   ax=ax, cbar_kws={'label': 'Correlation'})

        ax.set_title(f'Pairwise kernel correlations: {model_label}')
        plt.tight_layout()

        fname = f"kernel_consistency_corr_matrix_{model_key}"
        putils.save_figure_with_options(fig, file_formats=["png", "pdf"],
                                      filename=fname,
                                      output_dir=output_folder)
        plt.close(fig)

        # --- Plot 2: Distribution of correlations per region ---
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        region_corr_data = []
        for region in ordered_regions:
            region_data = corr_df[corr_df['region'] == region]
            if not region_data.empty:
                for _, row in region_data.iterrows():
                    region_corr_data.append({
                        'region': region,
                        'correlation': row['mean_corr']
                    })

        if region_corr_data:
            region_corr_df = pd.DataFrame(region_corr_data)

            sns.boxplot(data=region_corr_df, x='region', y='correlation', ax=ax)
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax.set_xlabel('Region')
            ax.set_ylabel('Pairwise Kernel Correlation')
            ax.set_title(f'Distribution of kernel correlations per region: {model_label}')
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            fname = f"kernel_consistency_per_region_{model_key}"
            putils.save_figure_with_options(fig, file_formats=["png", "pdf"],
                                          filename=fname,
                                          output_dir=output_folder)
            plt.close(fig)


def identify_neurons_with_kernel_changes(
        dfs,                     # dict: {"1k": df1, "2k": df2, "3k": df3, "4k": df4}
        model_labels,            # dict: {"1k": "1 whisker kernel", ...}
        output_folder,
        whisker_kernels,         # ["whisker_stim", "whisker_stim_0", "whisker_stim_1", ...]
        area_groups,
        area_colors,
        top_n=20):              # Number of top neurons to identify
    """
    Identify neurons with the most changes in their whisker kernels in TWO ways:
    1. Changes ACROSS models (1k → 2k → 3k → 4k): how does the kernel change as we add more kernels?
    2. Changes WITHIN multi-kernel models: how different are kernel_0, kernel_1, kernel_2 from each other?

    Plots metrics for top neurons with most changes.
    """

    os.makedirs(output_folder, exist_ok=True)

    def extract_sorted_kernel_indices(predictors, kernel_name):
        pattern = re.compile(fr"^{kernel_name}_t([+-]\d+\.\d+)s$")
        matches = []
        for i, p in enumerate(predictors):
            m = pattern.match(p)
            if m:
                lag = float(m.group(1))
                matches.append((lag, i))
        matches_sorted = sorted(matches, key=lambda x: x[0])
        idx_sorted = [i for (_, i) in matches_sorted]
        lags_sorted = [lag for (lag, _) in matches_sorted]
        return idx_sorted, lags_sorted

    # ==================== ANALYSIS 1: Changes ACROSS models ====================
    print("[INFO] Analyzing changes across models...")

    across_model_metrics = []

    # Get all unique neurons
    all_neurons = set()
    for df in dfs.values():
        for _, row in df.iterrows():
            all_neurons.add((row['mouse_id'], row['neuron_id'], row['area_acronym_custom']))

    for mouse_id, neuron_id, area in all_neurons:
        neuron_data = {
            'mouse_id': mouse_id,
            'neuron_id': neuron_id,
            'area': area,
            'amplitudes': [],
            'test_corrs': [],
            'kernel_shapes': {},
            'models': []
        }

        # For each model, extract this neuron's first whisker kernel
        for model_key, df in dfs.items():
            neuron_df = df[(df['mouse_id'] == mouse_id) & (df['neuron_id'] == neuron_id)]

            if neuron_df.empty:
                continue

            neuron_data['models'].append(model_key)

            # Get test_corr
            test_corr = neuron_df['test_corr'].mean()
            neuron_data['test_corrs'].append(test_corr)

            # Extract first whisker kernel
            for _, row in neuron_df.iterrows():
                predictors = row["predictors_full"]
                coefs = row["coef_full_mean"]

                for kernel in whisker_kernels:
                    idx, lag_list = extract_sorted_kernel_indices(predictors, kernel)
                    if idx:
                        kernel_coefs = coefs[idx]
                        if not np.all(np.isnan(kernel_coefs)):
                            peak_amp = np.nanmax(np.abs(kernel_coefs))
                            if not np.isnan(peak_amp):
                                neuron_data['amplitudes'].append(peak_amp)
                            neuron_data['kernel_shapes'][model_key] = kernel_coefs
                            break
                break

        # Skip neurons with insufficient data
        if len(neuron_data['models']) < 2:
            continue

        # Compute variability metrics
        amplitude_variance = np.var(neuron_data['amplitudes']) if len(neuron_data['amplitudes']) > 1 else 0
        test_corr_variance = np.var(neuron_data['test_corrs']) if len(neuron_data['test_corrs']) > 1 else 0

        # Compute shape variability (average pairwise correlation distance)
        shape_distances = []
        model_keys = list(neuron_data['kernel_shapes'].keys())
        for i, mk1 in enumerate(model_keys):
            for mk2 in model_keys[i+1:]:
                k1 = neuron_data['kernel_shapes'][mk1]
                k2 = neuron_data['kernel_shapes'][mk2]
                min_len = min(len(k1), len(k2))
                if min_len > 0:
                    corr = np.corrcoef(k1[:min_len], k2[:min_len])[0, 1]
                    if not np.isnan(corr):
                        shape_distances.append(1 - corr)

        shape_variance = np.mean(shape_distances) if shape_distances else 0

        # Compute composite change score
        change_score = amplitude_variance + test_corr_variance + shape_variance

        across_model_metrics.append({
            'mouse_id': mouse_id,
            'neuron_id': neuron_id,
            'area': area,
            'amplitude_variance': amplitude_variance,
            'test_corr_variance': test_corr_variance,
            'shape_variance': shape_variance,
            'change_score': change_score,
            'n_models': len(neuron_data['models']),
            'mean_amplitude': np.mean(neuron_data['amplitudes']) if neuron_data['amplitudes'] else np.nan,
            'mean_test_corr': np.mean(neuron_data['test_corrs']) if neuron_data['test_corrs'] else np.nan
        })

    # ==================== ANALYSIS 2: Changes WITHIN models ====================
    print("[INFO] Analyzing changes within multi-kernel models...")

    within_model_metrics = []

    for model_key, df in dfs.items():
        if model_key == "1k":  # Skip single kernel model
            continue

        print(f"  Processing model: {model_key}")

        # Get unique neurons in this model
        neurons_in_model = df[['mouse_id', 'neuron_id', 'area_acronym_custom']].drop_duplicates()

        for _, neuron_row in neurons_in_model.iterrows():
            mouse_id = neuron_row['mouse_id']
            neuron_id = neuron_row['neuron_id']
            area = neuron_row['area_acronym_custom']

            neuron_df = df[(df['mouse_id'] == mouse_id) & (df['neuron_id'] == neuron_id)]

            # Extract all whisker kernels for this neuron
            kernel_amplitudes = []
            kernel_curves = {}

            for _, row in neuron_df.iterrows():
                predictors = row["predictors_full"]
                coefs = row["coef_full_mean"]

                for kernel in whisker_kernels:
                    idx, lag_list = extract_sorted_kernel_indices(predictors, kernel)
                    if idx:
                        kernel_coefs = coefs[idx]
                        if not np.all(np.isnan(kernel_coefs)):
                            peak_amp = np.nanmax(np.abs(kernel_coefs))
                            if not np.isnan(peak_amp):
                                kernel_amplitudes.append(peak_amp)
                                kernel_curves[kernel] = kernel_coefs
                break

            # Need at least 2 kernels to measure within-model variability
            if len(kernel_curves) < 2:
                continue

            # Compute amplitude variability across kernels
            amplitude_cv = np.std(kernel_amplitudes) / np.mean(kernel_amplitudes) if np.mean(kernel_amplitudes) > 0 else 0

            # Compute shape dissimilarity between kernels
            shape_distances = []
            kernel_names = list(kernel_curves.keys())
            for i, k1_name in enumerate(kernel_names):
                for k2_name in kernel_names[i+1:]:
                    k1 = kernel_curves[k1_name]
                    k2 = kernel_curves[k2_name]
                    min_len = min(len(k1), len(k2))
                    if min_len > 0:
                        corr = np.corrcoef(k1[:min_len], k2[:min_len])[0, 1]
                        if not np.isnan(corr):
                            shape_distances.append(1 - corr)

            shape_dissimilarity = np.mean(shape_distances) if shape_distances else 0

            # Composite within-model change score
            within_change_score = amplitude_cv + shape_dissimilarity

            within_model_metrics.append({
                'mouse_id': mouse_id,
                'neuron_id': neuron_id,
                'area': area,
                'model': model_key,
                'amplitude_cv': amplitude_cv,
                'shape_dissimilarity': shape_dissimilarity,
                'within_change_score': within_change_score,
                'n_kernels': len(kernel_curves),
                'mean_amplitude': np.mean(kernel_amplitudes),
                'test_corr': neuron_df['test_corr'].mean()
            })

    # ==================== SAVE AND PLOT RESULTS ====================

    if not across_model_metrics:
        print("[WARNING] No across-model metrics collected")
    else:
        across_df = pd.DataFrame(across_model_metrics)
        across_df = across_df.sort_values('change_score', ascending=False)
        top_across = across_df.head(top_n)
        top_across.to_csv(os.path.join(output_folder, 'top_neurons_across_models.csv'), index=False)
        print(f"[INFO] Top {top_n} neurons with changes ACROSS models saved")

        # Plot across-model changes
        _plot_across_model_changes(across_df, top_across, dfs, model_labels, whisker_kernels,
                                   area_groups, output_folder, top_n)

    if not within_model_metrics:
        print("[WARNING] No within-model metrics collected")
    else:
        within_df = pd.DataFrame(within_model_metrics)
        within_df = within_df.sort_values('within_change_score', ascending=False)
        top_within = within_df.head(top_n)
        top_within.to_csv(os.path.join(output_folder, 'top_neurons_within_models.csv'), index=False)
        print(f"[INFO] Top {top_n} neurons with changes WITHIN models saved")

        # Plot within-model changes
        _plot_within_model_changes(within_df, top_within, dfs, model_labels, whisker_kernels,
                                   output_folder, top_n)

    return across_df, within_df


def _plot_across_model_changes(metrics_df, top_neurons, dfs, model_labels, whisker_kernels,
                               area_groups, output_folder, top_n):
    """Helper function to plot neurons with changes across models"""

    def extract_sorted_kernel_indices(predictors, kernel_name):
        pattern = re.compile(fr"^{kernel_name}_t([+-]\d+\.\d+)s$")
        matches = []
        for i, p in enumerate(predictors):
            m = pattern.match(p)
            if m:
                lag = float(m.group(1))
                matches.append((lag, i))
        matches_sorted = sorted(matches, key=lambda x: x[0])
        idx_sorted = [i for (_, i) in matches_sorted]
        lags_sorted = [lag for (lag, _) in matches_sorted]
        return idx_sorted, lags_sorted

    # Plot 1: Scatter plot of variance components
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(metrics_df['amplitude_variance'], metrics_df['test_corr_variance'],
                    alpha=0.5, s=20, c='gray')
    axes[0].scatter(top_neurons['amplitude_variance'], top_neurons['test_corr_variance'],
                    alpha=0.8, s=50, c='red', label=f'Top {top_n}')
    axes[0].set_xlabel('Amplitude Variance Across Models')
    axes[0].set_ylabel('Test Corr Variance Across Models')
    axes[0].legend()
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].scatter(metrics_df['amplitude_variance'], metrics_df['shape_variance'],
                    alpha=0.5, s=20, c='gray')
    axes[1].scatter(top_neurons['amplitude_variance'], top_neurons['shape_variance'],
                    alpha=0.8, s=50, c='red', label=f'Top {top_n}')
    axes[1].set_xlabel('Amplitude Variance Across Models')
    axes[1].set_ylabel('Shape Variance Across Models')
    axes[1].legend()
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    axes[2].scatter(metrics_df['test_corr_variance'], metrics_df['shape_variance'],
                    alpha=0.5, s=20, c='gray')
    axes[2].scatter(top_neurons['test_corr_variance'], top_neurons['shape_variance'],
                    alpha=0.8, s=50, c='red', label=f'Top {top_n}')
    axes[2].set_xlabel('Test Corr Variance Across Models')
    axes[2].set_ylabel('Shape Variance Across Models')
    axes[2].legend()
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    plt.tight_layout()
    fname = "across_models_variability_scatter"
    putils.save_figure_with_options(fig, file_formats=["png", 'pdf'],
                                  filename=fname,
                                  output_dir=output_folder)
    plt.close(fig)

    # Plot 2: Kernel trajectories for top 12 neurons
    top_12 = top_neurons.head(12)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (_, neuron_row) in enumerate(top_12.iterrows()):
        if idx >= 12:
            break

        ax = axes[idx]
        mouse_id = neuron_row['mouse_id']
        neuron_id = neuron_row['neuron_id']
        area = neuron_row['area']

        for model_key, df in dfs.items():
            neuron_df = df[(df['mouse_id'] == mouse_id) & (df['neuron_id'] == neuron_id)]
            if neuron_df.empty:
                continue

            for _, row in neuron_df.iterrows():
                predictors = row["predictors_full"]
                coefs = row["coef_full_mean"]

                for kernel in whisker_kernels:
                    idx_k, lag_list = extract_sorted_kernel_indices(predictors, kernel)
                    if idx_k:
                        kernel_coefs = coefs[idx_k]
                        if not np.all(np.isnan(kernel_coefs)):
                            ax.plot(lag_list, kernel_coefs, label=model_labels[model_key], lw=2)
                            break
                break

        ax.set_title(f'{area} - {mouse_id}:{neuron_id}\nScore={neuron_row["change_score"]:.3f}',
                    fontsize=8)
        ax.set_xlabel('Lag (s)', fontsize=7)
        ax.set_ylabel('Coef', fontsize=7)
        ax.legend(fontsize=5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle('Kernel evolution ACROSS models - Top 12 neurons', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fname = "across_models_top_neurons_trajectories"
    putils.save_figure_with_options(fig, file_formats=["png", 'pdf'],
                                  filename=fname,
                                  output_dir=output_folder)
    plt.close(fig)


def _plot_within_model_changes(metrics_df, top_neurons, dfs, model_labels, whisker_kernels,
                               output_folder, top_n):
    """Helper function to plot neurons with changes within models"""

    def extract_sorted_kernel_indices(predictors, kernel_name):
        pattern = re.compile(fr"^{kernel_name}_t([+-]\d+\.\d+)s$")
        matches = []
        for i, p in enumerate(predictors):
            m = pattern.match(p)
            if m:
                lag = float(m.group(1))
                matches.append((lag, i))
        matches_sorted = sorted(matches, key=lambda x: x[0])
        idx_sorted = [i for (_, i) in matches_sorted]
        lags_sorted = [lag for (lag, _) in matches_sorted]
        return idx_sorted, lags_sorted

    # Plot 1: Scatter of amplitude CV vs shape dissimilarity
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.scatter(metrics_df['amplitude_cv'], metrics_df['shape_dissimilarity'],
               alpha=0.5, s=20, c='gray')
    ax.scatter(top_neurons['amplitude_cv'], top_neurons['shape_dissimilarity'],
               alpha=0.8, s=50, c='red', label=f'Top {top_n}')
    ax.set_xlabel('Amplitude CV Within Model')
    ax.set_ylabel('Shape Dissimilarity Within Model')
    ax.set_title('Within-model kernel variability')
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    fname = "within_model_variability_scatter"
    putils.save_figure_with_options(fig, file_formats=["png", 'pdf'],
                                  filename=fname,
                                  output_dir=output_folder)
    plt.close(fig)

    # Plot 2: Show all kernels for top 12 neurons
    top_12 = top_neurons.head(12)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (_, neuron_row) in enumerate(top_12.iterrows()):
        if idx >= 12:
            break

        ax = axes[idx]
        mouse_id = neuron_row['mouse_id']
        neuron_id = neuron_row['neuron_id']
        area = neuron_row['area']
        model_key = neuron_row['model']

        df = dfs[model_key]
        neuron_df = df[(df['mouse_id'] == mouse_id) & (df['neuron_id'] == neuron_id)]

        if neuron_df.empty:
            continue

        for _, row in neuron_df.iterrows():
            predictors = row["predictors_full"]
            coefs = row["coef_full_mean"]

            # Plot all whisker kernels
            for kernel in whisker_kernels:
                idx_k, lag_list = extract_sorted_kernel_indices(predictors, kernel)
                if idx_k:
                    kernel_coefs = coefs[idx_k]
                    if not np.all(np.isnan(kernel_coefs)):
                        ax.plot(lag_list, kernel_coefs, label=kernel, lw=2)

            break

        ax.set_title(f'{area} - {mouse_id}:{neuron_id}\n{model_labels[model_key]} - Score={neuron_row["within_change_score"]:.3f}',
                    fontsize=8)
        ax.set_xlabel('Lag (s)', fontsize=7)
        ax.set_ylabel('Coef', fontsize=7)
        ax.legend(fontsize=5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle('Kernel diversity WITHIN models - Top 12 neurons', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fname = "within_model_top_neurons_kernels"
    putils.save_figure_with_options(fig, file_formats=["png", 'pdf'],
                                  filename=fname,
                                  output_dir=output_folder)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# KERNEL CONSISTENCY PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def _extract_kernel_vector(coef_full_mean, predictors_full, kernel_name):
    """Extract sorted (by lag) coefficient vector for a named kernel.

    Parameters
    ----------
    coef_full_mean : array-like
        Mean coefficient vector for the neuron (across folds).
    predictors_full : list of str
        Predictor names matching positions in coef_full_mean.
    kernel_name : str
        Exact kernel base name (e.g. 'whisker_hits_stim_0').

    Returns
    -------
    lags : np.ndarray or None
    coefs : np.ndarray or None
    """
    pattern = re.compile(rf"^{re.escape(kernel_name)}_t([+-])(\d+\.\d+)s$")
    kernel_data = []
    for i, p in enumerate(predictors_full):
        m = pattern.match(p)
        if m:
            sign = 1 if m.group(1) == '+' else -1
            lag = sign * float(m.group(2))
            kernel_data.append((lag, coef_full_mean[i]))
    if not kernel_data:
        return None, None
    kernel_data.sort(key=lambda x: x[0])
    lags = np.array([x[0] for x in kernel_data])
    coefs = np.array([x[1] for x in kernel_data])
    return lags, coefs


def _build_neuron_change_table(df, kernel_0, kernel_1):
    """For each neuron, compute directional % weight change from kernel_0 to kernel_1.

    Direction consistency is read from the pre-computed columns produced by
    compute_kernel_consistency during summary creation:
      - ``kernel_pair``       : e.g. 'whisker_hits_stim_1_vs_whisker_hits_stim_0'
      - ``is_consistent``     : True if the neuron passed the fold-consistency threshold
      - ``consistency_score`` : float, positive = consistent increase,
                                negative = consistent decrease

    ``all_positive`` is True when is_consistent=True and consistency_score > 0.
    ``all_negative`` is True when is_consistent=True and consistency_score < 0.

    If those columns are absent the function warns and falls back to checking
    the sign of every lag bin in the mean-coefficient delta.

    The percent change is defined as::

        pct_change = mean(w1 - w0) / (mean(|w0|) + mean(|w1|) + ε) * 100

    Bounded in [-100 %, +100 %] by the triangle inequality.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'model_name', 'coef_full_mean', 'predictors_full',
        'area_acronym_custom', 'reward_group'.
        Should also contain 'kernel_pair', 'is_consistent', 'consistency_score'.
    kernel_0, kernel_1 : str
        Kernel names to compare.

    Returns
    -------
    pd.DataFrame with columns:
        mouse_id, neuron_id, area_acronym_custom, reward_group,
        pct_change, all_positive, all_negative, w0, w1, lags
    """
    pair_name = f"{kernel_1}_vs_{kernel_0}"
    df_full = df[df['model_name'] == 'full']
    has_consistency = ('kernel_pair' in df_full.columns
                       and 'is_consistent' in df_full.columns
                       and 'consistency_score' in df_full.columns)
    if not has_consistency:
        print("[kernel_consistency] WARNING: consistency columns not found in df. "
              "Falling back to all-bins delta direction. "
              "Re-run create_summary to generate them.")

    rows = []
    for (mouse_id, neuron_id), grp in df_full.groupby(['mouse_id', 'neuron_id']):
        # coef_full_mean and predictors_full are the same across all fold-rows
        row = grp.iloc[0]
        coef = np.array(row['coef_full_mean'])
        preds = row['predictors_full']
        area = row['area_acronym_custom']
        reward_group = row['reward_group']

        lags, w0 = _extract_kernel_vector(coef, preds, kernel_0)
        _, w1 = _extract_kernel_vector(coef, preds, kernel_1)

        if w0 is None or w1 is None or len(w0) != len(w1):
            continue

        # Peak-based % change: symmetric, bounded in [-100 %, +100 %]
        amp0 = np.max(np.abs(w0))
        amp1 = np.max(np.abs(w1))
        pct_change = (amp1 - amp0) / (amp0 + amp1 + 1e-10) * 100

        # Mean-coef % change: same formula but using signed mean over time bins
        mean0 = np.mean(w0)
        mean1 = np.mean(w1)
        mean_pct_change = (mean1 - mean0) / (np.abs(mean0) + np.abs(mean1) + 1e-10) * 100

        if has_consistency:
            pair_rows = grp[grp['kernel_pair'] == pair_name]
            if pair_rows.empty:
                all_positive = False
                all_negative = False
            else:
                r = pair_rows.iloc[0]
                is_con = bool(r['is_consistent']) if not pd.isna(r['is_consistent']) else False
                score = float(r['consistency_score']) if not pd.isna(r['consistency_score']) else 0.0
                all_positive = is_con and score > 0
                all_negative = is_con and score < 0
        else:
            all_positive = bool(np.all(delta > 0))
            all_negative = bool(np.all(delta < 0))

        rows.append({
            'mouse_id': mouse_id,
            'neuron_id': neuron_id,
            'area_acronym_custom': area,
            'reward_group': reward_group,
            'pct_change': pct_change,
            'mean_pct_change': mean_pct_change,
            'all_positive': all_positive,
            'all_negative': all_negative,
            'w0': w0,
            'w1': w1,
            'lags': lags,
        })
    return pd.DataFrame(rows)


def _plot_consistent_kernels_by_area(
        neuron_df, ordered_regions, region_to_color,
        threshold, direction, output_folder,
        kernel_0, kernel_1, reward_label, n_cols=3):
    """Plot average kernel_0 (light) and kernel_1 (full colour) by brain area
    for a subset of neurons that consistently change in the given direction.

    Parameters
    ----------
    neuron_df : pd.DataFrame
        Already-filtered rows from _build_neuron_change_table.
    ordered_regions : list of str
    region_to_color : dict  {region: hex_color}
    threshold : float       % change threshold (used in title/filename only here).
    direction : str         'up' or 'down'.
    output_folder : str
    kernel_0, kernel_1 : str
    reward_label : str      e.g. 'R+' or 'R-'.
    n_cols : int
    """
    if neuron_df.empty:
        return

    present_regions = [r for r in ordered_regions
                       if r in neuron_df['area_acronym_custom'].values]
    if not present_regions:
        return

    n_rows = math.ceil(len(present_regions) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    legend_added = False
    for ax, region in zip(axes, present_regions):
        region_rows = neuron_df[neuron_df['area_acronym_custom'] == region]
        if region_rows.empty:
            ax.set_visible(False)
            continue

        w0_stack = np.stack(region_rows['w0'].values)
        w1_stack = np.stack(region_rows['w1'].values)
        lags = region_rows.iloc[0]['lags']
        n = len(region_rows)

        mean_w0 = w0_stack.mean(axis=0)
        sem_w0 = (w0_stack.std(axis=0, ddof=1) / np.sqrt(n)) if n > 1 else np.zeros_like(mean_w0)
        mean_w1 = w1_stack.mean(axis=0)
        sem_w1 = (w1_stack.std(axis=0, ddof=1) / np.sqrt(n)) if n > 1 else np.zeros_like(mean_w1)

        base_color = region_to_color.get(region, 'gray')

        # kernel_0: lighter / kernel_1: full colour
        ax.plot(lags, mean_w0, color=base_color, alpha=0.45, lw=1.5,
                label=kernel_0)
        ax.fill_between(lags, mean_w0 - sem_w0, mean_w0 + sem_w0,
                        color=base_color, alpha=0.12)
        ax.plot(lags, mean_w1, color=base_color, alpha=1.0, lw=2.0,
                label=kernel_1)
        ax.fill_between(lags, mean_w1 - sem_w1, mean_w1 + sem_w1,
                        color=base_color, alpha=0.28)

        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.set_title(f"{region} (n={n})", fontsize=9)
        ax.set_xlabel("Lag (s)", fontsize=8)
        ax.set_ylabel("Coef", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if not legend_added:
            ax.legend(fontsize=7)
            legend_added = True

    for ax in axes[len(present_regions):]:
        ax.set_visible(False)

    dir_label = 'increase' if direction == 'up' else 'decrease'
    thr_str = f"{threshold:.0f}"
    plt.suptitle(
        f"Avg kernels – consistent {dir_label} >|{thr_str}%| "
        f"[{kernel_1} vs {kernel_0}] – {reward_label}",
        fontsize=11
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    thr_label = str(threshold).replace('.', 'p')
    fname = (f"cst_{direction}_gt{thr_label}pct")
    putils.save_figure_with_options(fig, file_formats=['png', 'pdf'], filename=fname,
                                    output_dir=output_folder)
    plt.close(fig)


def _plot_consistency_fraction_by_area(
        neuron_df, reward_groups, reward_group_labels,
        ordered_regions, threshold,
        output_folder, kernel_0, kernel_1):
    """Grouped bar chart: % of neurons with consistent weight increase / decrease
    per brain area, one bar cluster per reward group.

    Produces two figures: one for 'up', one for 'down'.

    Parameters
    ----------
    neuron_df : pd.DataFrame
        Full table from _build_neuron_change_table (all reward groups).
    reward_groups : list of int
    reward_group_labels : dict  {int: str}
    ordered_regions : list of str
    threshold : float
    output_folder : str
    kernel_0, kernel_1 : str
    """
    reward_colors = {1: 'forestgreen', 0: 'crimson'}
    thr_label = str(threshold).replace('.', 'p')

    present_regions = [r for r in ordered_regions
                       if r in neuron_df['area_acronym_custom'].values]
    if not present_regions:
        return

    from scipy.stats import mannwhitneyu as _mwu

    def _per_mouse_fracs(rg_df, region, direction, threshold):
        per_mouse = []
        for _, mouse_df in rg_df[rg_df['area_acronym_custom'] == region].groupby('mouse_id'):
            total_m = len(mouse_df)
            if total_m == 0:
                continue
            if direction == 'up':
                n_sel = (mouse_df['all_positive'] & (mouse_df['pct_change'] > threshold)).sum()
            else:
                n_sel = (mouse_df['all_negative'] & (mouse_df['pct_change'] < -threshold)).sum()
            per_mouse.append(100.0 * n_sel / total_m)
        return per_mouse

    n_rg = len(reward_groups)
    bar_width = 0.8 / n_rg

    for direction in ['up', 'down']:
        # Sort areas by R+ mean fraction (descending) for this direction
        rp_df = neuron_df[neuron_df['reward_group'] == 1]
        def _region_rp_frac(region):
            total = len(rp_df[rp_df['area_acronym_custom'] == region])
            if total == 0:
                return 0.0
            if direction == 'up':
                n_sel = (rp_df[rp_df['area_acronym_custom'] == region]['all_positive']
                         & (rp_df[rp_df['area_acronym_custom'] == region]['pct_change'] > threshold)).sum()
            else:
                n_sel = (rp_df[rp_df['area_acronym_custom'] == region]['all_negative']
                         & (rp_df[rp_df['area_acronym_custom'] == region]['pct_change'] < -threshold)).sum()
            return 100.0 * n_sel / total
        sorted_regions = sorted(present_regions, key=_region_rp_frac, reverse=True)
        x = np.arange(len(sorted_regions))
        fig, ax = plt.subplots(figsize=(max(10, len(sorted_regions) * 0.75), 5))

        # Collect per-mouse fractions for all groups first (needed for the MWU test)
        all_fracs = {}   # reward_group -> list-of-lists (one per region)
        for reward_group in reward_groups:
            rg_df = neuron_df[neuron_df['reward_group'] == reward_group]
            all_fracs[reward_group] = [
                _per_mouse_fracs(rg_df, region, direction, threshold)
                for region in sorted_regions
            ]

        # Plot bars + dots
        bar_tops = np.zeros(len(sorted_regions))
        for rg_idx, reward_group in enumerate(reward_groups):
            rg_label = reward_group_labels.get(reward_group, str(reward_group))
            fracs = all_fracs[reward_group]
            means = [np.mean(f) if f else 0.0 for f in fracs]
            sems  = [np.std(f, ddof=1) / np.sqrt(len(f)) if len(f) >= 2 else 0.0 for f in fracs]

            offset = (rg_idx - (n_rg - 1) / 2) * bar_width
            ax.bar(x + offset, means, yerr=sems, width=bar_width,
                   color=reward_colors.get(reward_group, 'gray'),
                   label=rg_label, alpha=0.85, edgecolor='white',
                   capsize=3, error_kw={'linewidth': 1.2})
            for xi, per_mouse in zip(x + offset, fracs):
                ax.scatter(np.full(len(per_mouse), xi), per_mouse,
                           color='k', s=12, zorder=5, alpha=0.6, linewidths=0)
            bar_tops = np.maximum(bar_tops,
                                  [m + s for m, s in zip(means, sems)])

        # Mann-Whitney U (R+ vs R-) per area, annotate stars
        if len(reward_groups) == 2:
            rg_a, rg_b = reward_groups[0], reward_groups[1]
            y_star = bar_tops * 1.05
            step = bar_tops.max() * 0.06 if bar_tops.max() > 0 else 1.0
            for xi, fa, fb, yt in zip(x, all_fracs[rg_a], all_fracs[rg_b], y_star):
                if len(fa) < 2 or len(fb) < 2:
                    continue
                _, p = _mwu(fa, fb, alternative='two-sided')
                stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                if stars:
                    ax.text(xi, yt + step * 0.1, stars, ha='center', va='bottom',
                            fontsize=10, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(sorted_regions, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('% neurons (mean ± SEM across mice)', fontsize=10)
        dir_label = 'increase' if direction == 'up' else 'decrease'
        thr_title = 'no % threshold' if threshold == 0 else f'>|{threshold:.0f}%|'
        ax.set_title(
            f"% neurons – consistent {dir_label} {thr_title} per area"
            f"\n[{kernel_1} vs {kernel_0}]",
            fontsize=11
        )
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        fname_thr = 'no_thr' if threshold == 0 else f'gt{thr_label}pct'
        fname = (f"f_{direction}_{fname_thr}_by_area")
        putils.save_figure_with_options(fig, file_formats=['png', 'pdf'], filename=fname,
                                        output_dir=output_folder)
        plt.close(fig)


def _plot_pct_change_vs_performance(
        neuron_df, reward_groups, reward_group_labels,
        ordered_regions, threshold,
        output_folder, kernel_0, kernel_1,
        perf_df, change_col='pct_change'):
    """Scatter: per-mouse mean pct_change vs behavioural performance per area."""
    from scipy.stats import spearmanr

    reward_colors = {1: 'forestgreen', 0: 'crimson'}
    present_regions = [r for r in ordered_regions
                       if r in neuron_df['area_acronym_custom'].values]
    if not present_regions:
        return

    thr_label = 'no_thr' if threshold == 0 else f"gt{str(threshold).replace('.', 'p')}pct"

    # Per-mouse mean of change_col per area
    records = []
    for region in present_regions:
        region_df = neuron_df[neuron_df['area_acronym_custom'] == region]
        for (mouse_id, rg), grp in region_df.groupby(['mouse_id', 'reward_group']):
            records.append({
                'mouse_id': mouse_id,
                'reward_group': int(rg),
                'area': region,
                'mean_pct_change': grp[change_col].mean(),
            })
    if not records:
        return

    mouse_area_df = pd.DataFrame(records).merge(
        perf_df[['mouse_id', 'outcome_w', 'd_prime_w']], on='mouse_id', how='inner')
    if mouse_area_df.empty:
        return

    perf_folder = os.path.join(output_folder, 'vs_performance')
    os.makedirs(perf_folder, exist_ok=True)

    n_regions = len(present_regions)
    n_cols = min(5, n_regions)
    n_rows = math.ceil(n_regions / n_cols)

    for metric in ['outcome_w', 'd_prime_w']:
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
        for ax_idx, region in enumerate(present_regions):
            ax = axes[ax_idx // n_cols][ax_idx % n_cols]
            region_data = mouse_area_df[mouse_area_df['area'] == region]
            any_plotted = False
            for rg in reward_groups:
                rg_data = region_data[region_data['reward_group'] == rg].dropna(
                    subset=[metric, 'mean_pct_change'])
                if len(rg_data) < 3:
                    continue
                x = rg_data['mean_pct_change'].values
                y = rg_data[metric].values
                color = reward_colors.get(rg, 'gray')
                label = reward_group_labels.get(rg, str(rg))
                ax.scatter(x, y, color=color, s=30, alpha=0.8, label=label, zorder=3)
                m, b = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 50)
                ax.plot(x_line, m * x_line + b, color=color, linewidth=1.2, alpha=0.7)
                r, p = spearmanr(x, y)
                p_str = f'p={p:.2e}' if p >= 0.001 else 'p<0.001'
                ax.annotate(f'{label}: r={r:.2f}, {p_str}',
                            xy=(0.05, 0.95 - reward_groups.index(rg) * 0.13),
                            xycoords='axes fraction', fontsize=7,
                            color=color, va='top')
                any_plotted = True
            if not any_plotted:
                ax.set_visible(False)
                continue
            ax.axvline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.4)
            change_label = 'Mean % change (mean coef)' if change_col == 'mean_pct_change' else 'Mean % change (peak amp)'
            ax.set_title(region, fontsize=9, fontweight='bold')
            ax.set_xlabel(change_label, fontsize=8)
            ax.set_ylabel(metric, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for ax_idx in range(len(present_regions), n_rows * n_cols):
            axes[ax_idx // n_cols][ax_idx % n_cols].set_visible(False)

        col_tag = 'mean_coef' if change_col == 'mean_pct_change' else 'peak_amp'
        plt.suptitle(f'Mean % {col_tag} change vs {metric}\n[{kernel_1} vs {kernel_0}]',
                     fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fname = f"pct_change_{col_tag}_vs_{metric}"
        putils.save_figure_with_options(fig, file_formats=['png', 'pdf'],
                                        filename=fname, output_dir=perf_folder)
        plt.close(fig)


def _plot_pct_change_by_area(
        neuron_df, reward_groups, reward_group_labels,
        ordered_regions, threshold,
        output_folder, kernel_0, kernel_1):
    """Mean pct_change per area, R+ vs R−, with Mann-Whitney U stars.

    Four figures: {all, consistent} x {per_neuron, per_mouse}.
    """
    from scipy.stats import mannwhitneyu as _mwu

    reward_colors = {1: 'forestgreen', 0: 'crimson'}
    present_regions = [r for r in ordered_regions
                       if r in neuron_df['area_acronym_custom'].values]
    if not present_regions:
        return

    n_rg = len(reward_groups)
    bar_width = 0.8 / n_rg
    x = np.arange(len(present_regions))
    thr_label = 'no_thr' if threshold == 0 else f"gt{str(threshold).replace('.', 'p')}pct"

    for subset_label in ['all', 'consistent']:
        if subset_label == 'consistent':
            sub_df = neuron_df[neuron_df['all_positive'] | neuron_df['all_negative']]
            if threshold > 0:
                sub_df = sub_df[np.abs(sub_df['pct_change']) > threshold]
        else:
            sub_df = neuron_df

        for avg_method in ['per_neuron', 'per_mouse']:
            # Collect values first so the sort key matches the plotted aggregation
            rp_df = sub_df[sub_df['reward_group'] == 1]
            all_vals = {}
            for reward_group in reward_groups:
                rg_df = sub_df[sub_df['reward_group'] == reward_group]
                region_vals = []
                for region in present_regions:
                    region_rg = rg_df[rg_df['area_acronym_custom'] == region]
                    if avg_method == 'per_mouse':
                        v = [mouse_df['pct_change'].mean()
                             for _, mouse_df in region_rg.groupby('mouse_id')
                             if len(mouse_df) > 0]
                    else:
                        v = list(region_rg['pct_change'].dropna().values)
                    region_vals.append(v)
                all_vals[reward_group] = region_vals

            # Sort by R+ aggregated mean (matches exactly what is plotted)
            rp_vals = dict(zip(present_regions, all_vals.get(1, [[] for _ in present_regions])))
            region_sort_key = {r: (np.mean(rp_vals[r]) if rp_vals[r] else 0.0)
                               for r in present_regions}
            sorted_regions = sorted(present_regions,
                                    key=lambda r: region_sort_key.get(r, 0.0),
                                    reverse=True)
            # Reorder all_vals to match sorted_regions
            region_idx = {r: i for i, r in enumerate(present_regions)}
            all_vals = {rg: [vals[region_idx[r]] for r in sorted_regions]
                        for rg, vals in all_vals.items()}

            n_reg = len(sorted_regions)
            fig, ax = plt.subplots(figsize=(6, max(4, n_reg * 0.55 + 1.5)))
            y = np.arange(n_reg)
            bar_rights = np.zeros(n_reg)
            for rg_idx, reward_group in enumerate(reward_groups):
                rg_label = reward_group_labels.get(reward_group, str(reward_group))
                vals = all_vals[reward_group]
                means = [np.mean(v) if v else 0.0 for v in vals]
                sems  = [np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) >= 2 else 0.0 for v in vals]
                offset = (rg_idx - (n_rg - 1) / 2) * bar_width
                ax.barh(y + offset, means, xerr=sems, height=bar_width,
                        color=reward_colors.get(reward_group, 'gray'),
                        label=rg_label, alpha=0.85, edgecolor='white',
                        capsize=3, error_kw={'linewidth': 1.2})
                if avg_method == 'per_mouse':
                    for yi, v in zip(y + offset, vals):
                        ax.scatter(v, np.full(len(v), yi),
                                   color='k', s=12, zorder=5, alpha=0.6, linewidths=0)
                bar_rights = np.maximum(bar_rights,
                                        [abs(m) + s for m, s in zip(means, sems)])

            if len(reward_groups) == 2:
                rg_a, rg_b = reward_groups[0], reward_groups[1]
                step = bar_rights.max() * 0.06 if bar_rights.max() > 0 else 1.0
                for yi, fa, fb, xr in zip(y, all_vals[rg_a], all_vals[rg_b], bar_rights * 1.05):
                    if len(fa) < 2 or len(fb) < 2:
                        continue
                    _, p = _mwu(fa, fb, alternative='two-sided')
                    stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                    if stars:
                        ax.text(xr + step * 0.1, yi, stars, ha='left', va='center',
                                fontsize=10, fontweight='bold')

            ax.axvline(0, color='k', linewidth=0.6, linestyle='--', alpha=0.4)
            ax.set_yticks(y)
            ax.set_yticklabels(sorted_regions, fontsize=9)
            avg_lbl = 'per mouse' if avg_method == 'per_mouse' else 'per neuron'
            ax.set_xlabel(f'Mean % amplitude change ({avg_lbl})', fontsize=10)
            subset_title = 'consistent neurons' if subset_label == 'consistent' else 'all neurons'
            thr_title = '' if threshold == 0 else f'  (>|{threshold:.0f}%|)'
            ax.set_title(f'Δ amplitude – {subset_title}{thr_title} per area', fontsize=11)
            ax.legend(fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            fname = (f"pct_change_{subset_label}_{avg_method}_{thr_label}")
            putils.save_figure_with_options(fig, file_formats=['png', 'pdf'],
                                            filename=fname, output_dir=output_folder)
            plt.close(fig)



def _plot_consistency_fraction_combined(
        neuron_df, reward_groups, reward_group_labels,
        ordered_regions, threshold,
        output_folder, kernel_0, kernel_1):
    """Three figures showing fraction of neurons with ANY consistent change
    (up OR down combined) at a given threshold.

    Figure 1 – by area, both reward groups on the same axes (grouped bars).
    Figure 2 – all areas pooled, one bar per reward group.
    Figure 3 – one figure per reward group: fraction per area
                (to compare area contributions within each reward group).

    Parameters
    ----------
    neuron_df : pd.DataFrame  (output of _build_neuron_change_table)
    reward_groups : list of int
    reward_group_labels : dict {int: str}
    ordered_regions : list of str
    threshold : float   (0 = no threshold)
    output_folder : str
    kernel_0, kernel_1 : str
    """
    reward_colors = {1: 'forestgreen', 0: 'crimson'}
    fname_thr = 'no_thr' if threshold == 0 else f'gt{str(threshold).replace(".", "p")}pct'
    thr_title = 'no % threshold' if threshold == 0 else f'>|{threshold:.0f}%|'

    present_regions = [r for r in ordered_regions
                       if r in neuron_df['area_acronym_custom'].values]
    if not present_regions:
        return

    def _count_consistent(sub_df, thr):
        """Count neurons with any consistent change at threshold thr."""
        if thr == 0:
            return (sub_df['all_positive'] | sub_df['all_negative']).sum()
        return (
            (sub_df['all_positive'] & (sub_df['pct_change'] > thr)) |
            (sub_df['all_negative'] & (sub_df['pct_change'] < -thr))
        ).sum()

    # ── Figure 1: by area, grouped bars (all reward groups on same axes) ──────
    n_rg = len(reward_groups)
    bar_width = 0.8 / n_rg
    x = np.arange(len(present_regions))

    fig1, ax1 = plt.subplots(figsize=(max(10, len(present_regions) * 0.75), 5))
    for rg_idx, reward_group in enumerate(reward_groups):
        rg_label = reward_group_labels.get(reward_group, str(reward_group))
        rg_df = neuron_df[neuron_df['reward_group'] == reward_group]
        fractions = []
        for region in present_regions:
            region_rg = rg_df[rg_df['area_acronym_custom'] == region]
            total = len(region_rg)
            if total == 0:
                fractions.append(0.0)
            else:
                fractions.append(100.0 * _count_consistent(region_rg, threshold) / total)
        offset = (rg_idx - (n_rg - 1) / 2) * bar_width
        ax1.bar(x + offset, fractions, width=bar_width,
                color=reward_colors.get(reward_group, 'gray'),
                label=rg_label, alpha=0.85, edgecolor='white')

    ax1.set_xticks(x)
    ax1.set_xticklabels(present_regions, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('% neurons', fontsize=10)
    ax1.set_title(
        f"% neurons – consistent changes (up or down) {thr_title} per area"
        f"\n[{kernel_1} vs {kernel_0}]",
        fontsize=11
    )
    ax1.legend(fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    plt.tight_layout()
    putils.save_figure_with_options(
        fig1, file_formats=['png', 'pdf'],
        filename=f"frac_cst_{fname_thr}_areas",
        output_dir=output_folder)
    plt.close(fig1)

    # ── Figure 2: all areas pooled, one bar per reward group ──────────────────
    fig2, ax2 = plt.subplots(figsize=(4, 5))
    overall_fractions = []
    overall_labels = []
    overall_colors = []
    for reward_group in reward_groups:
        rg_label = reward_group_labels.get(reward_group, str(reward_group))
        rg_df = neuron_df[neuron_df['reward_group'] == reward_group]
        total = len(rg_df)
        frac = (100.0 * _count_consistent(rg_df, threshold) / total) if total > 0 else 0.0
        overall_fractions.append(frac)
        overall_labels.append(rg_label)
        overall_colors.append(reward_colors.get(reward_group, 'gray'))

    ax2.bar(overall_labels, overall_fractions, color=overall_colors,
            alpha=0.85, edgecolor='white', width=0.5)
    ax2.set_ylabel('% neurons', fontsize=10)
    ax2.set_title(
        f"% neurons – consistent changes (all areas pooled)\n{thr_title}"
        f"\n[{kernel_1} vs {kernel_0}]",
        fontsize=11
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    putils.save_figure_with_options(
        fig2, file_formats=['png', 'pdf'],
        filename=f"frac_cst_{fname_thr}_areas",
        output_dir=output_folder)
    plt.close(fig2)

    # ── Figure 3: per reward group, fraction per area (area contribution) ─────
    for reward_group in reward_groups:
        rg_label = reward_group_labels.get(reward_group, str(reward_group))
        rg_df = neuron_df[neuron_df['reward_group'] == reward_group]
        color = reward_colors.get(reward_group, 'gray')

        fractions = []
        for region in present_regions:
            region_rg = rg_df[rg_df['area_acronym_custom'] == region]
            total = len(region_rg)
            if total == 0:
                fractions.append(0.0)
            else:
                fractions.append(100.0 * _count_consistent(region_rg, threshold) / total)

        fig3, ax3 = plt.subplots(figsize=(max(10, len(present_regions) * 0.75), 5))
        ax3.bar(present_regions, fractions, color=color, alpha=0.85, edgecolor='white')
        ax3.set_xticks(range(len(present_regions)))
        ax3.set_xticklabels(present_regions, rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('% neurons', fontsize=10)
        ax3.set_title(
            f"Area contribution – consistent changes – {rg_label} {thr_title}"
            f"\n[{kernel_1} vs {kernel_0}]",
            fontsize=11
        )
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        plt.tight_layout()
        putils.save_figure_with_options(
            fig3, file_formats=['png', 'pdf'],
            filename=f"area_contrib_cst__{rg_label}",
            output_dir=output_folder)
        plt.close(fig3)


def _plot_pct_change_distribution(
        neuron_df, reward_groups, reward_group_labels,
        output_folder, kernel_0, kernel_1):
    """Histogram of per-neuron % weight change, overlaid by reward group.

    Plots ALL neurons (no threshold filter) to help choose thresholds.
    When exactly two reward groups are present, a Mann-Whitney U test is
    run and the p-value is annotated on the figure.

    Parameters
    ----------
    neuron_df : pd.DataFrame
        Full table from _build_neuron_change_table.
    reward_groups : list of int
    reward_group_labels : dict  {int: str}
    output_folder : str
    kernel_0, kernel_1 : str
    """
    from scipy import stats

    if neuron_df.empty:
        return

    reward_colors = {1: 'forestgreen', 0: 'crimson'}

    fig, ax = plt.subplots(figsize=(7, 4))

    group_values = {}
    for reward_group in reward_groups:
        rg_label = reward_group_labels.get(reward_group, str(reward_group))
        values = neuron_df.loc[neuron_df['reward_group'] == reward_group, 'pct_change'].values
        if len(values) == 0:
            continue
        group_values[reward_group] = values
        color = reward_colors.get(reward_group, 'gray')
        ax.hist(values, bins=50, color=color, alpha=0.5,
                label=f"{rg_label} (n={len(values)})",
                density=True, edgecolor='none')

    ax.axvline(0, color='black', lw=1, ls='--')
    ax.set_xlabel('% amplitude change  (peak|w₁| − peak|w₀|) / (peak|w₁| + peak|w₀|)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)

    # Mann-Whitney U test when exactly two groups are available
    stat_text = ''
    keys = list(group_values.keys())
    if len(keys) == 2:
        v1, v2 = group_values[keys[0]], group_values[keys[1]]
        _, p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
        if p < 0.001:
            p_str = 'p < 0.001'
        elif p < 0.01:
            p_str = f'p = {p:.3f}'
        else:
            p_str = f'p = {p:.2f}'
        lbl0 = reward_group_labels.get(keys[0], str(keys[0]))
        lbl1 = reward_group_labels.get(keys[1], str(keys[1]))
        stat_text = f'Mann-Whitney U\n{lbl0} vs {lbl1}: {p_str}'
        ax.text(0.97, 0.95, stat_text,
                transform=ax.transAxes, fontsize=8,
                va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='lightgray'))

    ax.set_title(
        f"Distribution of weight change\n{kernel_1} vs {kernel_0}",
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    fname = f"pct_change_distribution_{kernel_1}_vs_{kernel_0}"
    putils.save_figure_with_options(fig, file_formats=['png', 'pdf'], filename=fname,
                                    output_dir=output_folder)
    plt.close(fig)


def _plot_consistent_neurons_encoding_profile(
        neuron_df, df, reward_groups, reward_group_labels,
        output_folder, kernel_0, kernel_1, encoding_models=None,
        area_groups=None, ordered_regions=None,
        excluded_models=None):
    """Horizontal bar chart: % consistently-up / consistently-down neurons that
    are LRT-significant for each encoding model, vs all-neuron baseline.

    Produces three sets of figures per direction ('up'/'down'):
      1. Pooled (all areas).
      2. Per area_group  → per_area_group/ subfolder.
      3. Per area        → per_area/ subfolder.

    Bars    : % consistent neurons that are LRT-significant.
    Dashed  : % of ALL neurons in that reward group (baseline).
    Colours : forestgreen = R+, crimson = R-.
    """
    from scipy.stats import fisher_exact

    lrt_cols = ['mouse_id', 'neuron_id', 'model_name', 'lrt_significant']
    if not all(c in df.columns for c in lrt_cols):
        print("[kernel_consistency] 'lrt_significant' column missing – "
              "skipping encoding profile plot.")
        return

    _EXCLUDE = excluded_models if excluded_models is not None else [
        'random_split', '1whisker_kernel'
    ]

    if encoding_models is None:
        encoding_models = sorted(
            m for m in df['model_name'].dropna().unique() if m != 'full'
        )
    encoding_models = [m for m in encoding_models
                       if not any(ex in m for ex in _EXCLUDE)]

    if not encoding_models:
        print("[kernel_consistency] No reduced models found for profile plot.")
        return

    # One LRT row per (mouse_id, neuron_id, model_name) — avoid fold duplication
    lrt_df = (
        df[df['model_name'].isin(encoding_models)][lrt_cols]
        .dropna(subset=['lrt_significant'])
        .groupby(['mouse_id', 'neuron_id', 'model_name'])['lrt_significant']
        .first()
        .reset_index()
    )

    model_short = {m: m.replace('_encoding', '').replace('_', ' ')
                   for m in encoding_models}
    reward_colors = {1: 'forestgreen', 0: 'crimson'}
    n_models = len(encoding_models)
    n_rg = len(reward_groups)
    bar_height = 0.7 / n_rg
    y_pos = np.arange(n_models)

    def _sig_label(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return ''

    def _draw_profile(ax, neuron_sub, direction, avg_method='per_neuron'):
        """Draw encoding-profile barh into ax for a subset of neurons.

        avg_method='per_neuron' : fraction pooled across all neurons.
        avg_method='per_mouse'  : mean ± SEM of per-mouse fractions.
        Fisher's exact test is always per-neuron (count-based).
        """
        for rg_idx, reward_group in enumerate(reward_groups):
            rg_label = reward_group_labels.get(reward_group, str(reward_group))
            rg_df = neuron_sub[neuron_sub['reward_group'] == reward_group]
            color = reward_colors.get(reward_group, 'gray')

            all_ids = rg_df[['mouse_id', 'neuron_id']].drop_duplicates()
            if direction == 'up':
                cons_ids = rg_df.loc[rg_df['all_positive'],
                                     ['mouse_id', 'neuron_id']].drop_duplicates()
            else:
                cons_ids = rg_df.loc[rg_df['all_negative'],
                                     ['mouse_id', 'neuron_id']].drop_duplicates()

            n_cons = len(cons_ids)
            fracs_cons, fracs_base, sems_cons, pvals = [], [], [], []

            for model in encoding_models:
                model_lrt = lrt_df[lrt_df['model_name'] == model]
                all_m  = all_ids.merge(model_lrt, on=['mouse_id', 'neuron_id'])
                cons_m = cons_ids.merge(model_lrt, on=['mouse_id', 'neuron_id'])

                if avg_method == 'per_mouse':
                    mc = [g['lrt_significant'].mean() * 100
                          for _, g in cons_m.groupby('mouse_id')]
                    mb = [g['lrt_significant'].mean() * 100
                          for _, g in all_m.groupby('mouse_id')]
                    fracs_cons.append(np.mean(mc) if mc else 0.0)
                    sems_cons.append(
                        np.std(mc, ddof=1) / np.sqrt(len(mc)) if len(mc) > 1 else 0.0)
                    fracs_base.append(np.mean(mb) if mb else 0.0)
                else:
                    fracs_cons.append(
                        cons_m['lrt_significant'].mean() * 100 if len(cons_m) > 0 else 0.0)
                    sems_cons.append(0.0)
                    fracs_base.append(
                        all_m['lrt_significant'].mean() * 100 if len(all_m) > 0 else 0.0)

                # Fisher's exact (per-neuron counts)
                cons_sig     = int(cons_m['lrt_significant'].sum())
                cons_not_sig = len(cons_m) - cons_sig
                non_cons_m   = all_m.merge(
                    cons_ids, on=['mouse_id', 'neuron_id'], how='left', indicator=True)
                non_cons_m       = non_cons_m[non_cons_m['_merge'] == 'left_only']
                non_cons_sig     = int(non_cons_m['lrt_significant'].sum())
                non_cons_not_sig = len(non_cons_m) - non_cons_sig
                contingency = [[cons_sig, cons_not_sig],
                               [non_cons_sig, non_cons_not_sig]]
                if cons_not_sig + non_cons_not_sig > 0 and cons_sig + non_cons_sig > 0:
                    _, p = fisher_exact(contingency, alternative='two-sided')
                else:
                    p = 1.0
                pvals.append(p)

            offset = (rg_idx - (n_rg - 1) / 2) * bar_height
            xerr = sems_cons if avg_method == 'per_mouse' else None
            bars = ax.barh(y_pos + offset, fracs_cons, height=bar_height, color=color,
                           alpha=0.85, label=f"{rg_label} consistent (n={n_cons})",
                           edgecolor='white', xerr=xerr,
                           error_kw={'ecolor': color, 'capsize': 2, 'elinewidth': 1.2})

            for yi, b in zip(y_pos, fracs_base):
                ax.vlines(b, yi + offset - bar_height / 2, yi + offset + bar_height / 2,
                          colors=color, linestyles='--', linewidth=1.5)

            for bar, p in zip(bars, pvals):
                lbl = _sig_label(p)
                if lbl:
                    ax.text(bar.get_width() + 0.8,
                            bar.get_y() + bar.get_height() / 2,
                            lbl, ha='left', va='center',
                            fontsize=10, color=color, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels([model_short[m] for m in encoding_models], fontsize=9)
        xlabel = ('% LRT-significant (mean ± SEM per mouse)'
                  if avg_method == 'per_mouse' else '% LRT-significant neurons')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    panel_h = max(4, n_models * 0.55 + 1.5)

    for avg_method in ['per_neuron', 'per_mouse']:
        _avg_suffix = f'_{avg_method}'

        # ── 1. Pooled (all areas) ─────────────────────────────────────────────
        for direction in ['up', 'down']:
            fig, ax = plt.subplots(figsize=(6, panel_h))
            _draw_profile(ax, neuron_df, direction, avg_method)
            dir_label = 'increase' if direction == 'up' else 'decrease'
            ax.set_title(
                f"Encoding profile – consistently {dir_label}ing neurons\n"
                f"[{kernel_1} vs {kernel_0}]  "
                f"(dashed = baseline, * Fisher's exact p<0.05)",
                fontsize=10)
            ax.legend(fontsize=8)
            plt.tight_layout()
            fname = (f"{direction}_{_avg_suffix}")
            putils.save_figure_with_options(fig, file_formats=['png', 'pdf'],
                                            filename=fname, output_dir=output_folder)
            plt.close(fig)

        # ── 2. Per area_group ─────────────────────────────────────────────────
        if area_groups is not None:
            ag_folder = os.path.join(output_folder, 'per_area_group')
            os.makedirs(ag_folder, exist_ok=True)
            groups_present = {gn: areas for gn, areas in area_groups.items()
                              if any(a in neuron_df['area_acronym_custom'].values
                                     for a in areas)}
            if groups_present:
                for direction in ['up', 'down']:
                    n_g = len(groups_present)
                    n_cols_g = min(3, n_g)
                    n_rows_g = math.ceil(n_g / n_cols_g)
                    fig, axes = plt.subplots(
                        n_rows_g, n_cols_g,
                        figsize=(6 * n_cols_g, panel_h * n_rows_g),
                        squeeze=False)
                    for g_idx, (gn, areas) in enumerate(groups_present.items()):
                        ax = axes[g_idx // n_cols_g][g_idx % n_cols_g]
                        sub = neuron_df[neuron_df['area_acronym_custom'].isin(areas)]
                        _draw_profile(ax, sub, direction, avg_method)
                        ax.set_title(gn, fontsize=10, fontweight='bold')
                        if g_idx == 0:
                            ax.legend(fontsize=7)
                    for g_idx in range(len(groups_present), n_rows_g * n_cols_g):
                        axes[g_idx // n_cols_g][g_idx % n_cols_g].set_visible(False)
                    dir_label = 'increase' if direction == 'up' else 'decrease'
                    plt.suptitle(
                        f"Encoding profile per area group – consistently {dir_label}ing\n"
                        f"[{kernel_1} vs {kernel_0}]  (dashed = baseline, * Fisher p<0.05)",
                        fontsize=11)
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    fname = (f"{direction}_{_avg_suffix}")
                    putils.save_figure_with_options(fig, file_formats=['png', 'pdf'],
                                                    filename=fname, output_dir=ag_folder)
                    plt.close(fig)

        # ── 3. Per area ───────────────────────────────────────────────────────
        if ordered_regions is not None:
            area_folder = os.path.join(output_folder, 'per_area')
            os.makedirs(area_folder, exist_ok=True)
            regions_present = [r for r in ordered_regions
                               if r in neuron_df['area_acronym_custom'].values]
            if regions_present:
                for direction in ['up', 'down']:
                    n_r = len(regions_present)
                    n_cols_r = min(5, n_r)
                    n_rows_r = math.ceil(n_r / n_cols_r)
                    fig, axes = plt.subplots(
                        n_rows_r, n_cols_r,
                        figsize=(6 * n_cols_r, panel_h * n_rows_r),
                        squeeze=False)
                    for r_idx, region in enumerate(regions_present):
                        ax = axes[r_idx // n_cols_r][r_idx % n_cols_r]
                        sub = neuron_df[neuron_df['area_acronym_custom'] == region]
                        _draw_profile(ax, sub, direction, avg_method)
                        ax.set_title(region, fontsize=10, fontweight='bold')
                        if r_idx == 0:
                            ax.legend(fontsize=7)
                    for r_idx in range(len(regions_present), n_rows_r * n_cols_r):
                        axes[r_idx // n_cols_r][r_idx % n_cols_r].set_visible(False)
                    dir_label = 'increase' if direction == 'up' else 'decrease'
                    plt.suptitle(
                        f"Encoding profile per area – consistently {dir_label}ing\n"
                        f"[{kernel_1} vs {kernel_0}]  (dashed = baseline, * Fisher p<0.05)",
                        fontsize=11)
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    fname = (f"{direction}_{_avg_suffix}")
                    putils.save_figure_with_options(fig, file_formats=['png', 'pdf'],
                                                    filename=fname, output_dir=area_folder)
                    plt.close(fig)


def _plot_kernel_correlation_matrices(
        neuron_df, df, reward_groups, reward_group_labels,
        output_folder, kernel_0, kernel_1):
    """
    For each reward group, compute and plot the average pairwise correlation
    matrix between all kernels, separately for:
      - consistent-up neurons  (all_positive == True)
      - consistent-down neurons (all_negative == True)
      - non-consistent neurons

    For each neuron, a (n_kernels × n_kernels) correlation matrix is computed
    by correlating the time-series vectors of each pair of kernels (Pearson r
    between the two coefficient vectors across time lags).  These per-neuron
    matrices are then averaged across neurons in each group.

    One figure per reward group with 3 subplots side-by-side.
    """
    import re as _re
    from scipy.stats import pearsonr

    df_full = df[df['model_name'] == 'full'].copy()
    if df_full.empty:
        return

    # ── auto-detect kernel names from a sample row ────────────────────────────
    sample_preds = df_full.iloc[0]['predictors_full']
    kernel_pattern = _re.compile(r'^(.+)_t[+-]\d+\.\d+s$')
    seen = set()
    detected = []
    for p in sample_preds:
        p_str = p.decode() if isinstance(p, bytes) else p
        m = kernel_pattern.match(p_str)
        if m:
            kb = m.group(1)
            if kb not in seen:
                seen.add(kb)
                detected.append(kb)

    # ── order kernels by biological grouping ─────────────────────────────────
    # Group order: whisker hits, whisker misses, auditory, piezo reward (whisker),
    # piezo reward (auditory), then anything remaining (motor, jaw, etc.)
    GROUP_ORDER = [
        'whisker_hits_stim_',
        'whisker_misses_stim_',
        'auditory_stim_',
        'piezo_reward_au_',   # au before plain so prefix match doesn't swallow it
        'piezo_reward_',
    ]

    def _sort_key(kname):
        for rank, prefix in enumerate(GROUP_ORDER):
            if kname.startswith(prefix):
                # secondary sort: numeric suffix (0 before 1 before 2 …)
                suffix_match = _re.search(r'(\d+)$', kname)
                secondary = int(suffix_match.group(1)) if suffix_match else 0
                return (rank, secondary, kname)
        return (len(GROUP_ORDER), 0, kname)   # unknown → end

    kernel_names_ordered = sorted(detected, key=_sort_key)

    if len(kernel_names_ordered) < 2:
        print("[_plot_kernel_correlation_matrices] fewer than 2 kernels found – skipping.")
        return

    n_k = len(kernel_names_ordered)

    def _extract_vec(coef, preds, kernel_name):
        pat = _re.compile(fr"^{_re.escape(kernel_name)}_t([+-]\d+\.\d+)s$")
        matches = sorted(
            [(float(m.group(1)), i)
             for i, p in enumerate(preds)
             if (m := pat.match(p.decode() if isinstance(p, bytes) else p))]
        )
        if not matches:
            return None
        return np.array(coef)[[i for _, i in matches]]

    # ── per-neuron correlation matrix (n_kernels × n_kernels) ─────────────────
    # Entry (i, j) = Pearson r between time series of kernel_i and kernel_j
    # for this neuron.
    neuron_corr_mats = {}  # (mouse_id, neuron_id) -> np.array (n_k, n_k)
    for (mouse_id, neuron_id), grp in df_full.groupby(['mouse_id', 'neuron_id']):
        row   = grp.iloc[0]
        coef  = np.array(row['coef_full_mean'])
        preds = row['predictors_full']
        vecs  = [_extract_vec(coef, preds, k) for k in kernel_names_ordered]
        if any(v is None for v in vecs):
            continue

        corr_mat = np.full((n_k, n_k), np.nan)
        for i in range(n_k):
            for j in range(n_k):
                vi, vj = vecs[i], vecs[j]
                if i == j:
                    corr_mat[i, j] = 1.0
                elif len(vi) == len(vj) and len(vi) >= 2:
                    r, _ = pearsonr(vi, vj)
                    corr_mat[i, j] = r
        neuron_corr_mats[(mouse_id, neuron_id)] = corr_mat

    if not neuron_corr_mats:
        print("[_plot_kernel_correlation_matrices] no valid neuron records – skipping.")
        return

    def _group_avg_corr(uid_set):
        """Mean correlation matrix across all neurons in uid_set."""
        mats = [neuron_corr_mats[uid]
                for uid in uid_set if uid in neuron_corr_mats]
        if len(mats) < 2:
            return None, 0
        return np.nanmean(np.stack(mats), axis=0), len(mats)

    short_labels = [k.replace('_stim', '').replace('_', '\n') for k in kernel_names_ordered]

    for reward_group in reward_groups:
        rg_label = reward_group_labels.get(reward_group, str(reward_group))
        rg_df    = neuron_df[neuron_df['reward_group'] == reward_group]
        if rg_df.empty:
            continue

        all_ids  = set(zip(rg_df['mouse_id'], rg_df['neuron_id']))
        up_ids   = set(zip(rg_df.loc[rg_df['all_positive'],  'mouse_id'],
                           rg_df.loc[rg_df['all_positive'],  'neuron_id']))
        down_ids = set(zip(rg_df.loc[rg_df['all_negative'],  'mouse_id'],
                           rg_df.loc[rg_df['all_negative'],  'neuron_id']))
        non_ids  = all_ids - up_ids - down_ids

        groups = [
            ('consistent up',   up_ids,   '#2ca02c'),
            ('consistent down', down_ids, '#d62728'),
            ('non-consistent',  non_ids,  '#7f7f7f'),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(6 * 3, 5.5))

        for ax, (group_label, uid_set, color) in zip(axes, groups):
            corr, n = _group_avg_corr(uid_set)
            if corr is None:
                ax.set_visible(False)
                continue

            im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1,
                           aspect='auto', interpolation='nearest')
            ax.set_xticks(range(n_k))
            ax.set_yticks(range(n_k))
            ax.set_xticklabels(short_labels, fontsize=7, rotation=45, ha='right')
            ax.set_yticklabels(short_labels, fontsize=7)
            ax.set_title(f'{group_label}  (n={n})', fontsize=10,
                         color=color, fontweight='bold')

            for i in range(n_k):
                for j in range(n_k):
                    ax.text(j, i, f'{corr[i, j]:.2f}',
                            ha='center', va='center', fontsize=6,
                            color='black' if abs(corr[i, j]) < 0.6 else 'white')

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='mean Pearson r')

        fig.suptitle(
            f'Average per-neuron kernel correlation matrices  [{kernel_1} vs {kernel_0}]  |  {rg_label}',
            fontsize=12)
        fig.tight_layout()
        fname = f"kernel_corr_matrix_{kernel_1}_vs_{kernel_0}_{rg_label}"
        putils.save_figure_with_options(fig, file_formats=['png', 'pdf'],
                                        filename=fname, output_dir=output_folder)
        plt.close(fig)


def plot_kernel_consistency_by_area(
        df,
        kernel_0,
        kernel_1,
        area_groups,
        area_colors,
        output_folder,
        reward_groups=None,
        thresholds=None,
        n_cols=3,
        performance_csv=None):
    """Plot kernel consistency analysis by brain area and reward group.

    For each reward group, identifies neurons whose weight changes consistently
    in the SAME direction across ALL lag bins when going from kernel_0 to
    kernel_1, and optionally exceeds a % change threshold.

    The percent change is::

        pct_change = mean(w1 - w0) / (mean(|w0|) + ε) × 100

    Produces four plot types
    ------------------------
    **Plot 1** (threshold = thresholds[0], default 1 %)
        Average kernel_0 (light) and kernel_1 (bold) by brain area for neurons
        with all-positive / all-negative delta **and** |pct_change| > 1 %.
        One figure for 'increase', one for 'decrease', per reward group.

    **Plot 2** (threshold = thresholds[1], default 10 %)
        Same as Plot 1 but with the stricter 10 % threshold.

    **Plot 3** (per threshold)
        Grouped bar chart: percentage of neurons with consistent increase /
        decrease per brain area, one bar cluster per reward group.

    **Plot 4** (no threshold filter)
        Violin distribution of per-neuron pct_change split by reward group –
        to help choose an appropriate threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe with 'model_name', 'coef_full_mean', 'predictors_full',
        'area_acronym_custom', 'reward_group' columns.
    kernel_0 : str
        Reference kernel (e.g. ``'whisker_hits_stim_0'``).
    kernel_1 : str
        Comparison kernel (e.g. ``'whisker_hits_stim_1'``).
    area_groups : dict   {group_name: [area, ...]}
    area_colors : dict   {group_name: hex_color}
    output_folder : str
    reward_groups : list of int, optional   Default ``[1, 0]``.
    thresholds : list of float, optional    Default ``[1.0, 10.0]``.
    n_cols : int                             Default ``3``.
    """
    if reward_groups is None:
        reward_groups = [1, 0]
    if thresholds is None:
        thresholds = [1.0, 10.0]

    reward_group_labels = {1: 'R+', 0: 'R-'}
    os.makedirs(output_folder, exist_ok=True)

    perf_df = None
    if performance_csv is not None and os.path.exists(performance_csv):
        perf_df = pd.read_csv(performance_csv)[['mouse_id', 'outcome_w', 'd_prime_w']]
    elif performance_csv is not None:
        print(f"[kernel_consistency] WARNING: performance_csv not found: {performance_csv}")

    # Build region ordering and colour map
    region_to_color = {}
    for group_name, areas in area_groups.items():
        for area in areas:
            region_to_color[area] = area_colors[group_name]

    ordered_regions = []
    for group_name, areas in area_groups.items():
        for area in areas:
            if area in df['area_acronym_custom'].values:
                ordered_regions.append(area)

    # ── per-neuron change statistics ──────────────────────────────────────────
    print(f"[kernel_consistency] Building change table: {kernel_0} → {kernel_1} ...")
    neuron_df = _build_neuron_change_table(df, kernel_0, kernel_1)

    if neuron_df.empty:
        print(f"[WARNING] No neurons found with both '{kernel_0}' and '{kernel_1}'.")
        return

    print(f"[kernel_consistency] {len(neuron_df)} neurons analysed.")

    # ── Plot 4: distribution of % change per reward group ─────────────────────
    _plot_pct_change_distribution(
        neuron_df=neuron_df,
        reward_groups=reward_groups,
        reward_group_labels=reward_group_labels,
        output_folder=output_folder,
        kernel_0=kernel_0,
        kernel_1=kernel_1,
    )

    # ── Encoding profile (LRT) for consistently-changing neurons ─────────────
    _plot_consistent_neurons_encoding_profile(
        neuron_df=neuron_df,
        df=df,
        reward_groups=reward_groups,
        reward_group_labels=reward_group_labels,
        output_folder=output_folder,
        kernel_0=kernel_0,
        kernel_1=kernel_1,
        area_groups=area_groups,
        ordered_regions=ordered_regions,
    )

    # ── Kernel correlation matrices per group ─────────────────────────────────
    # _plot_kernel_correlation_matrices(
    #     neuron_df=neuron_df,
    #     df=df,
    #     reward_groups=reward_groups,
    #     reward_group_labels=reward_group_labels,
    #     output_folder=output_folder,
    #     kernel_0=kernel_0,
    #     kernel_1=kernel_1,
    # )

    # ── No % threshold: fraction chart + average kernel plots ─────────────────
    no_thr_folder = os.path.join(output_folder, 'thr_none')
    os.makedirs(no_thr_folder, exist_ok=True)
    _plot_consistency_fraction_by_area(
        neuron_df=neuron_df,
        reward_groups=reward_groups,
        reward_group_labels=reward_group_labels,
        ordered_regions=ordered_regions,
        threshold=0,
        output_folder=no_thr_folder,
        kernel_0=kernel_0,
        kernel_1=kernel_1,
    )
    _plot_consistency_fraction_combined(
        neuron_df=neuron_df,
        reward_groups=reward_groups,
        reward_group_labels=reward_group_labels,
        ordered_regions=ordered_regions,
        threshold=0,
        output_folder=no_thr_folder,
        kernel_0=kernel_0,
        kernel_1=kernel_1,
    )
    _plot_pct_change_by_area(
        neuron_df=neuron_df,
        reward_groups=reward_groups,
        reward_group_labels=reward_group_labels,
        ordered_regions=ordered_regions,
        threshold=0,
        output_folder=no_thr_folder,
        kernel_0=kernel_0,
        kernel_1=kernel_1,
    )
    neuron_df_mean = neuron_df.copy()
    neuron_df_mean['pct_change'] = neuron_df_mean['mean_pct_change']
    _no_thr_mean_folder = os.path.join(no_thr_folder, 'mean_coef')
    os.makedirs(_no_thr_mean_folder, exist_ok=True)
    _plot_consistency_fraction_by_area(
        neuron_df=neuron_df_mean,
        reward_groups=reward_groups,
        reward_group_labels=reward_group_labels,
        ordered_regions=ordered_regions,
        threshold=0,
        output_folder=_no_thr_mean_folder,
        kernel_0=kernel_0,
        kernel_1=kernel_1,
    )
    _plot_pct_change_by_area(
        neuron_df=neuron_df_mean,
        reward_groups=reward_groups,
        reward_group_labels=reward_group_labels,
        ordered_regions=ordered_regions,
        threshold=0,
        output_folder=_no_thr_mean_folder,
        kernel_0=kernel_0,
        kernel_1=kernel_1,
    )
    if perf_df is not None:
        _plot_pct_change_vs_performance(
            neuron_df=neuron_df,
            reward_groups=reward_groups,
            reward_group_labels=reward_group_labels,
            ordered_regions=ordered_regions,
            threshold=0,
            output_folder=no_thr_folder,
            kernel_0=kernel_0,
            kernel_1=kernel_1,
            perf_df=perf_df,
            change_col='pct_change',
        )
        _plot_pct_change_vs_performance(
            neuron_df=neuron_df_mean,
            reward_groups=reward_groups,
            reward_group_labels=reward_group_labels,
            ordered_regions=ordered_regions,
            threshold=0,
            output_folder=_no_thr_mean_folder,
            kernel_0=kernel_0,
            kernel_1=kernel_1,
            perf_df=perf_df,
            change_col='mean_pct_change',
        )
    for reward_group in reward_groups:
        rg_label = reward_group_labels.get(reward_group, str(reward_group))
        rg_df = neuron_df[neuron_df['reward_group'] == reward_group]
        rg_folder = os.path.join(no_thr_folder, rg_label)
        os.makedirs(rg_folder, exist_ok=True)

        up_df = rg_df[rg_df['all_positive']]
        _plot_consistent_kernels_by_area(
            neuron_df=up_df,
            ordered_regions=ordered_regions,
            region_to_color=region_to_color,
            threshold=0,
            direction='up',
            output_folder=rg_folder,
            kernel_0=kernel_0,
            kernel_1=kernel_1,
            reward_label=rg_label,
            n_cols=n_cols,
        )

        down_df = rg_df[rg_df['all_negative']]
        _plot_consistent_kernels_by_area(
            neuron_df=down_df,
            ordered_regions=ordered_regions,
            region_to_color=region_to_color,
            threshold=0,
            direction='down',
            output_folder=rg_folder,
            kernel_0=kernel_0,
            kernel_1=kernel_1,
            reward_label=rg_label,
            n_cols=n_cols,
        )

    # ── Plots 1, 2, 3 per threshold ───────────────────────────────────────────
    for threshold in thresholds:
        thr_folder = os.path.join(output_folder,
                                  f"thr_{str(threshold).replace('.', 'p')}pct")
        os.makedirs(thr_folder, exist_ok=True)

        # Plot 3 – fraction per area (all reward groups on same figure)
        _plot_consistency_fraction_by_area(
            neuron_df=neuron_df,
            reward_groups=reward_groups,
            reward_group_labels=reward_group_labels,
            ordered_regions=ordered_regions,
            threshold=threshold,
            output_folder=thr_folder,
            kernel_0=kernel_0,
            kernel_1=kernel_1,
        )
        _plot_consistency_fraction_combined(
            neuron_df=neuron_df,
            reward_groups=reward_groups,
            reward_group_labels=reward_group_labels,
            ordered_regions=ordered_regions,
            threshold=threshold,
            output_folder=thr_folder,
            kernel_0=kernel_0,
            kernel_1=kernel_1,
        )
        _plot_pct_change_by_area(
            neuron_df=neuron_df,
            reward_groups=reward_groups,
            reward_group_labels=reward_group_labels,
            ordered_regions=ordered_regions,
            threshold=threshold,
            output_folder=thr_folder,
            kernel_0=kernel_0,
            kernel_1=kernel_1,
        )

        # Plots 1/2 – average kernels per reward group
        for reward_group in reward_groups:
            rg_label = reward_group_labels.get(reward_group, str(reward_group))
            rg_df = neuron_df[neuron_df['reward_group'] == reward_group]

            rg_folder = os.path.join(thr_folder, rg_label)
            os.makedirs(rg_folder, exist_ok=True)

            # Consistently UP
            up_df = rg_df[rg_df['all_positive'] & (rg_df['pct_change'] > threshold)]
            _plot_consistent_kernels_by_area(
                neuron_df=up_df,
                ordered_regions=ordered_regions,
                region_to_color=region_to_color,
                threshold=threshold,
                direction='up',
                output_folder=rg_folder,
                kernel_0=kernel_0,
                kernel_1=kernel_1,
                reward_label=rg_label,
                n_cols=n_cols,
            )

            # Consistently DOWN
            down_df = rg_df[rg_df['all_negative'] & (rg_df['pct_change'] < -threshold)]
            _plot_consistent_kernels_by_area(
                neuron_df=down_df,
                ordered_regions=ordered_regions,
                region_to_color=region_to_color,
                threshold=threshold,
                direction='down',
                output_folder=rg_folder,
                kernel_0=kernel_0,
                kernel_1=kernel_1,
                reward_label=rg_label,
                n_cols=n_cols,
            )

        # Mean-coef versions of fraction + pct_change plots
        _thr_mean_folder = os.path.join(thr_folder, 'mean_coef')
        os.makedirs(_thr_mean_folder, exist_ok=True)
        neuron_df_mean = neuron_df.copy()
        neuron_df_mean['pct_change'] = neuron_df_mean['mean_pct_change']
        _plot_consistency_fraction_by_area(
            neuron_df=neuron_df_mean,
            reward_groups=reward_groups,
            reward_group_labels=reward_group_labels,
            ordered_regions=ordered_regions,
            threshold=threshold,
            output_folder=_thr_mean_folder,
            kernel_0=kernel_0,
            kernel_1=kernel_1,
        )
        _plot_pct_change_by_area(
            neuron_df=neuron_df_mean,
            reward_groups=reward_groups,
            reward_group_labels=reward_group_labels,
            ordered_regions=ordered_regions,
            threshold=threshold,
            output_folder=_thr_mean_folder,
            kernel_0=kernel_0,
            kernel_1=kernel_1,
        )
        if perf_df is not None:
            _plot_pct_change_vs_performance(
                neuron_df=neuron_df,
                reward_groups=reward_groups,
                reward_group_labels=reward_group_labels,
                ordered_regions=ordered_regions,
                threshold=threshold,
                output_folder=thr_folder,
                kernel_0=kernel_0,
                kernel_1=kernel_1,
                perf_df=perf_df,
                change_col='pct_change',
            )
            _plot_pct_change_vs_performance(
                neuron_df=neuron_df_mean,
                reward_groups=reward_groups,
                reward_group_labels=reward_group_labels,
                ordered_regions=ordered_regions,
                threshold=threshold,
                output_folder=_thr_mean_folder,
                kernel_0=kernel_0,
                kernel_1=kernel_1,
                perf_df=perf_df,
                change_col='mean_pct_change',
            )


# Bin size used during GLM fitting — converts spikes/bin → Hz
_BIN_SIZE_S    = 0.050          # 50 ms
_SPIKES_TO_HZ  = 1.0 / _BIN_SIZE_S   # multiply y_test / y_pred by this factor

# Trial window (seconds relative to stimulus onset)
_WINDOW_START_S = -1.0
_WINDOW_END_S   =  2.0

# Colors for lick vs no-lick conditions
_LICK_COLOR   = '#2ca02c'   # green  — lick trials
_NOLICK_COLOR = '#d62728'   # red    — no-lick trials


def _make_time_axis(n_bins):
    """Return time axis in seconds from trial start (stimulus at t=0)."""
    return np.arange(n_bins) * _BIN_SIZE_S + _WINDOW_START_S


def _draw_psth_panel(ax, real_mat, pred_mat, color, lw=1.5):
    """Draw mean ± SEM for real (solid) and predicted (dashed) on ax."""
    n_bins = real_mat.shape[1]
    x = _make_time_axis(n_bins)
    n = real_mat.shape[0]
    mean_r = real_mat.mean(axis=0)
    sem_r  = real_mat.std(axis=0) / np.sqrt(n)
    mean_p = pred_mat.mean(axis=0)
    sem_p  = pred_mat.std(axis=0) / np.sqrt(n)
    ax.plot(x, mean_r, color=color, lw=lw)
    ax.fill_between(x, mean_r - sem_r, mean_r + sem_r, color=color, alpha=0.18)
    ax.plot(x, mean_p, color=color, lw=lw, linestyle='--')
    ax.fill_between(x, mean_p - sem_p, mean_p + sem_p, color=color, alpha=0.08)


def _draw_residual_panel(ax, real_mat, pred_mat, color, lw=1.5):
    """Draw mean ± SEM residual (real − predicted) on ax."""
    resid = real_mat - pred_mat
    x     = _make_time_axis(resid.shape[1])
    n     = resid.shape[0]
    mean_res = resid.mean(axis=0)
    sem_res  = resid.std(axis=0) / np.sqrt(n)
    ax.plot(x, mean_res, color=color, lw=lw)
    ax.fill_between(x, mean_res - sem_res, mean_res + sem_res, color=color, alpha=0.18)


def _make_area_grid(n_areas, n_cols, panel_w=4.5, panel_h=3.5, dpi=150):
    n_rows = max(1, int(np.ceil(n_areas / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(panel_w * n_cols, panel_h * n_rows),
                             dpi=dpi)
    axes = np.array(axes).flatten()
    return fig, axes


def plot_psth_predictions_per_trialtype_per_area(df_pred, area_groups, area_colors,
                                                  output_folder, n_cols=4):
    """
    Plot PSTHs of real activity and model predictions split by lick / no-lick,
    one figure per trial type and per reward group.

    Layout per figure
    -----------------
    Grid of panels, one per brain area.
    Each panel shows two PSTH pairs:
      • Lick trials   — solid (real) / dashed (predicted), green
      • No-lick trials— solid (real) / dashed (predicted), red
    A companion residual figure (real − predicted) is also saved.

    Parameters
    ----------
    df_pred : pd.DataFrame
        One row per neuron. Required columns:
          mouse_id, neuron_id, area_acronym_custom, reward_group,
          trial_types – JSON list[str], one entry per trial,
          lick_flags  – JSON list[int] (1=lick, 0=no-lick, -1=unknown),
          y_pred      – JSON 2-D list (n_trials × n_bins),
          y_test      – JSON 2-D list (n_trials × n_bins).
    area_groups : dict  {group_name: [area, ...]}
    area_colors : dict  {group_name: color}
    output_folder : str
    n_cols : int
    """
    os.makedirs(output_folder, exist_ok=True)

    # ── decode helper ─────────────────────────────────────────────────────────
    def _decode(row):
        def _j(v):
            return json.loads(v) if isinstance(v, str) else v
        tts   = np.array(_j(row['trial_types']))
        licks = np.array(_j(row['lick_flags']), dtype=int)
        y_p   = np.array(_j(row['y_pred']), dtype=float) * _SPIKES_TO_HZ
        y_t   = np.array(_j(row['y_test']), dtype=float) * _SPIKES_TO_HZ
        return tts, licks, y_p, y_t

    # ── ordered areas ─────────────────────────────────────────────────────────
    all_areas = set(df_pred['area_acronym_custom'].unique())
    ordered_areas = []
    area_to_color = {}
    for grp_name, areas in area_groups.items():
        for area in areas:
            if area in all_areas:
                ordered_areas.append(area)
                area_to_color[area] = area_colors.get(grp_name, 'gray')

    reward_labels = {1: 'r+', 0: 'r-'}

    # ── loop over reward groups ────────────────────────────────────────────────
    for reward_group, rg_label in reward_labels.items():
        rg_df = df_pred[df_pred['reward_group'] == reward_group]
        if rg_df.empty:
            continue

        # Collect per-(area, trial_type, lick) arrays
        # data[area][trial_type][lick_key] = {'real': [psth, ...], 'pred': [psth, ...]}
        # lick_key: 'lick' or 'nolick'
        data = {area: {} for area in ordered_areas}

        for _, row in rg_df.iterrows():
            area = row['area_acronym_custom']
            if area not in data:
                continue
            tts, licks, y_p, y_t = _decode(row)

            for tt in np.unique(tts):
                if tt not in data[area]:
                    data[area][tt] = {'lick':   {'real': [], 'pred': []},
                                      'nolick': {'real': [], 'pred': []}}
                for lick_key, flag in [('lick', 1), ('nolick', 0)]:
                    mask = (tts == tt) & (licks == flag)
                    if mask.sum() == 0:
                        continue
                    # Average over trials of this (type, lick) for this neuron
                    data[area][tt][lick_key]['real'].append(y_t[mask].mean(axis=0))
                    data[area][tt][lick_key]['pred'].append(y_p[mask].mean(axis=0))

        # All trial types seen
        all_tts = sorted({tt for area_d in data.values() for tt in area_d})

        # ── one figure pair per trial type ────────────────────────────────────
        for tt in all_tts:
            n_areas = len(ordered_areas)

            # Figure A: real vs predicted
            fig_a, axes_a = _make_area_grid(n_areas, n_cols)
            # Figure B: residuals
            fig_b, axes_b = _make_area_grid(n_areas, n_cols)

            for ai, area in enumerate(ordered_areas):
                ax_a = axes_a[ai]
                ax_b = axes_b[ai]
                n_neurons = len(rg_df[rg_df['area_acronym_custom'] == area])
                title = f'{area}  (n={n_neurons})'

                tt_data = data[area].get(tt, {})

                for lick_key, color, label in [('lick',   _LICK_COLOR,   'lick'),
                                                ('nolick', _NOLICK_COLOR, 'no lick')]:
                    lk = tt_data.get(lick_key, {'real': [], 'pred': []})
                    if not lk['real']:
                        continue
                    real_mat = np.stack(lk['real'])
                    pred_mat = np.stack(lk['pred'])

                    _draw_psth_panel(ax_a, real_mat, pred_mat, color)
                    _draw_residual_panel(ax_b, real_mat, pred_mat, color)

                area_color = area_to_color.get(area, 'black')
                for ax, ylabel in [(ax_a, 'Firing rate (Hz)'), (ax_b, 'Residual (Hz)')]:
                    ax.set_title(title, fontsize=8, color=area_color, fontweight='bold')
                    ax.set_xlabel('Time from trial start (s)', fontsize=7)
                    ax.set_ylabel(ylabel, fontsize=7)
                    ax.tick_params(labelsize=6)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color(area_color)
                    ax.spines['left'].set_linewidth(2.0)
                    ax.spines['bottom'].set_color(area_color)
                    ax.spines['bottom'].set_linewidth(2.0)
                    ax.tick_params(color=area_color)
                    ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)

            # Hide unused panels
            for ai in range(n_areas, len(axes_a)):
                axes_a[ai].set_visible(False)
                axes_b[ai].set_visible(False)

            # Shared legend (in first visible panel)
            legend_handles = [
                plt.Line2D([0], [0], color=_LICK_COLOR,   lw=1.5, ls='-',  label='lick — real'),
                plt.Line2D([0], [0], color=_LICK_COLOR,   lw=1.5, ls='--', label='lick — pred'),
                plt.Line2D([0], [0], color=_NOLICK_COLOR, lw=1.5, ls='-',  label='no lick — real'),
                plt.Line2D([0], [0], color=_NOLICK_COLOR, lw=1.5, ls='--', label='no lick — pred'),
            ]
            resid_legend = [
                plt.Line2D([0], [0], color=_LICK_COLOR,   lw=1.5, label='lick'),
                plt.Line2D([0], [0], color=_NOLICK_COLOR, lw=1.5, label='no lick'),
            ]
            if n_areas > 0:
                axes_a[0].legend(handles=legend_handles, fontsize=6, loc='upper right')
                axes_b[0].legend(handles=resid_legend,   fontsize=6, loc='upper right')

            safe_tt = tt.replace(' ', '_')
            fig_a.suptitle(f'{tt}  |  real (solid) vs predicted (dashed)  |  {rg_label}',
                           fontsize=11)
            fig_a.tight_layout()
            putils.save_figure_with_options(
                fig_a, file_formats=['png', 'pdf'],
                filename=f'psth_{safe_tt}_{rg_label}',
                output_dir=output_folder)
            plt.close(fig_a)

            fig_b.suptitle(f'{tt}  |  residual PSTHs (real − predicted)  |  {rg_label}',
                           fontsize=11)
            fig_b.tight_layout()
            putils.save_figure_with_options(
                fig_b, file_formats=['png', 'pdf'],
                filename=f'psth_residuals_{safe_tt}_{rg_label}',
                output_dir=output_folder)
            plt.close(fig_b)


# Colors for session-half comparison
_FIRST_HALF_COLOR  = '#74c476'   # light green — first half of session
_SECOND_HALF_COLOR = '#00441b'   # dark green  — second half of session


def plot_psth_whisker_session_halves_per_area(df_pred, area_groups, area_colors,
                                               output_folder, n_cols=4):
    """
    For whisker hits and whisker misses separately, plot PSTHs split by the
    first vs second half of the session (ranked by trial order).

    One figure per (lick condition × reward group):
      - whisker_hits_{rg_label}   : first half (light green) vs second half (dark green)
      - whisker_misses_{rg_label} : same colour scheme
    Each panel = one brain area; solid = real, dashed = predicted.
    A companion residual figure is also saved per combination.

    Parameters
    ----------
    df_pred : pd.DataFrame
        One row per neuron. Required columns:
          mouse_id, neuron_id, area_acronym_custom, reward_group,
          trial_ids   – JSON list[int], sorted ascending (= temporal order),
          trial_types – JSON list[str],
          lick_flags  – JSON list[int] (1=lick, 0=no-lick),
          y_pred      – JSON 2-D list (n_trials × n_bins),
          y_test      – JSON 2-D list (n_trials × n_bins).
    area_groups : dict  {group_name: [area, ...]}
    area_colors : dict  {group_name: color}
    output_folder : str
    n_cols : int
    """
    os.makedirs(output_folder, exist_ok=True)

    # ── decode helper ─────────────────────────────────────────────────────────
    def _decode(row):
        def _j(v):
            return json.loads(v) if isinstance(v, str) else v
        ids   = np.array(_j(row['trial_ids']),   dtype=int)
        tts   = np.array(_j(row['trial_types']))
        licks = np.array(_j(row['lick_flags']),  dtype=int)
        y_p   = np.array(_j(row['y_pred']),      dtype=float) * _SPIKES_TO_HZ
        y_t   = np.array(_j(row['y_test']),      dtype=float) * _SPIKES_TO_HZ
        return ids, tts, licks, y_p, y_t

    # ── ordered areas ─────────────────────────────────────────────────────────
    all_areas = set(df_pred['area_acronym_custom'].unique())
    ordered_areas = []
    area_to_color = {}
    for grp_name, areas in area_groups.items():
        for area in areas:
            if area in all_areas:
                ordered_areas.append(area)
                area_to_color[area] = area_colors.get(grp_name, 'gray')

    reward_labels  = {1: 'r+', 0: 'r-'}
    # (label, trial_type, lick_flag_value)  lick_val=None means all lick outcomes
    lick_conditions = [
        ('whisker_hits',   'whisker_trial',  1),
        ('whisker_misses', 'whisker_trial',  0),
        ('auditory_hits',  'auditory_trial', 1),
        ('auditory_misses','auditory_trial', 0),
    ]

    for reward_group, rg_label in reward_labels.items():
        rg_df = df_pred[df_pred['reward_group'] == reward_group]
        if rg_df.empty:
            continue

        for cond_label, trial_type, lick_val in lick_conditions:

            # data[area][half] = {'real': [psth_per_neuron, ...], 'pred': [...]}
            # half ∈ {'first', 'second'}
            data = {area: {'first':  {'real': [], 'pred': []},
                           'second': {'real': [], 'pred': []}}
                    for area in ordered_areas}

            for _, row in rg_df.iterrows():
                area = row['area_acronym_custom']
                if area not in data:
                    continue
                ids, tts, licks, y_p, y_t = _decode(row)

                mask = (tts == trial_type) & (licks == lick_val)
                if mask.sum() < 2:
                    continue

                # trial_ids are already sorted ascending → temporal order preserved
                sel_positions = np.where(mask)[0]   # positions in the arrays
                n_sel = len(sel_positions)
                mid   = n_sel // 2

                first_pos  = sel_positions[:mid]
                second_pos = sel_positions[mid:]

                # Average over trials within each half → one PSTH per neuron per half
                data[area]['first']['real'].append(y_t[first_pos].mean(axis=0))
                data[area]['first']['pred'].append(y_p[first_pos].mean(axis=0))
                data[area]['second']['real'].append(y_t[second_pos].mean(axis=0))
                data[area]['second']['pred'].append(y_p[second_pos].mean(axis=0))

            n_areas = len(ordered_areas)
            fig_a, axes_a = _make_area_grid(n_areas, n_cols)
            fig_b, axes_b = _make_area_grid(n_areas, n_cols)

            half_specs = [
                ('first',  _FIRST_HALF_COLOR,  '1st half'),
                ('second', _SECOND_HALF_COLOR, '2nd half'),
            ]

            for ai, area in enumerate(ordered_areas):
                ax_a = axes_a[ai]
                ax_b = axes_b[ai]
                n_neurons = len(rg_df[rg_df['area_acronym_custom'] == area])
                title = f'{area}  (n={n_neurons})'

                for half_key, color, _ in half_specs:
                    half_d = data[area][half_key]
                    if not half_d['real']:
                        continue
                    real_mat = np.stack(half_d['real'])
                    pred_mat = np.stack(half_d['pred'])
                    _draw_psth_panel(ax_a, real_mat, pred_mat, color)
                    _draw_residual_panel(ax_b, real_mat, pred_mat, color)

                area_color = area_to_color.get(area, 'black')
                for ax, ylabel in [(ax_a, 'Firing rate (Hz)'), (ax_b, 'Residual (Hz)')]:
                    ax.set_title(title, fontsize=8, color=area_color, fontweight='bold')
                    ax.set_xlabel('Time from trial start (s)', fontsize=7)
                    ax.set_ylabel(ylabel, fontsize=7)
                    ax.tick_params(labelsize=6)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color(area_color)
                    ax.spines['left'].set_linewidth(2.0)
                    ax.spines['bottom'].set_color(area_color)
                    ax.spines['bottom'].set_linewidth(2.0)
                    ax.tick_params(color=area_color)
                    ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)

            for ai in range(n_areas, len(axes_a)):
                axes_a[ai].set_visible(False)
                axes_b[ai].set_visible(False)

            legend_handles = [
                plt.Line2D([0], [0], color=_FIRST_HALF_COLOR,  lw=1.5, ls='-',  label='1st half — real'),
                plt.Line2D([0], [0], color=_FIRST_HALF_COLOR,  lw=1.5, ls='--', label='1st half — pred'),
                plt.Line2D([0], [0], color=_SECOND_HALF_COLOR, lw=1.5, ls='-',  label='2nd half — real'),
                plt.Line2D([0], [0], color=_SECOND_HALF_COLOR, lw=1.5, ls='--', label='2nd half — pred'),
            ]
            resid_legend = [
                plt.Line2D([0], [0], color=_FIRST_HALF_COLOR,  lw=1.5, label='1st half'),
                plt.Line2D([0], [0], color=_SECOND_HALF_COLOR, lw=1.5, label='2nd half'),
            ]
            if n_areas > 0:
                axes_a[0].legend(handles=legend_handles, fontsize=6, loc='upper right')
                axes_b[0].legend(handles=resid_legend,   fontsize=6, loc='upper right')

            human_label = cond_label.replace('_', ' ')
            fig_a.suptitle(
                f'{human_label}  |  1st vs 2nd half  |  real (solid) / pred (dashed)  |  {rg_label}',
                fontsize=11)
            fig_a.tight_layout()
            putils.save_figure_with_options(
                fig_a, file_formats=['png', 'pdf'],
                filename=f'psth_halves_{cond_label}_{rg_label}',
                output_dir=output_folder)
            plt.close(fig_a)

            fig_b.suptitle(
                f'{human_label}  |  residuals 1st vs 2nd half  |  {rg_label}',
                fontsize=11)
            fig_b.tight_layout()
            putils.save_figure_with_options(
                fig_b, file_formats=['png', 'pdf'],
                filename=f'psth_halves_residuals_{cond_label}_{rg_label}',
                output_dir=output_folder)
            plt.close(fig_b)


def plot_kernels_full_vs_optimal_split_per_area(
        df_full,
        df_optimal,
        output_folder,
        area_groups,
        area_colors,
        n_cols=3,
        alpha=0.05):
    """
    For each kernel pair (kernel_0 and kernel_1 of each stimulus type), overlay
    the average kernel from the *full* model and the *optimal_split* model on the
    same axis, per brain area.  A per-bin paired t-test is run across neurons
    to highlight bins where the two models differ significantly.

    One figure per stimulus-type pair (whisker hits, whisker misses, auditory,
    reward …).  Each panel = one brain area.

    Parameters
    ----------
    df_full : pd.DataFrame
        Rows from the full model (model_name == 'full').
    df_optimal : pd.DataFrame
        Rows from the optimal_split model (model_name == 'optimal_split').
    output_folder : str
    area_groups : dict  {group_name: [area, ...]}
    area_colors : dict  {group_name: color}
    n_cols : int
    alpha : float  significance threshold for per-bin t-test
    """
    from scipy.stats import ttest_rel

    os.makedirs(output_folder, exist_ok=True)

    # Each entry: (kernel_name, title, color_full, color_optimal)
    KERNEL_SPECS = [
        ('whisker_hits_stim_0',   'Whisker Hits – kernel 0',   '#4daf4a', '#a6d96a'),
        ('whisker_hits_stim_1',   'Whisker Hits – kernel 1',   '#1a7416', '#66bb6a'),
        ('whisker_misses_stim_0', 'Whisker Misses – kernel 0', '#e41a1c', '#f4a582'),
        ('whisker_misses_stim_1', 'Whisker Misses – kernel 1', '#8b0000', '#d6604d'),
        ('auditory_stim_0',       'Auditory – kernel 0',       '#377eb8', '#74b9ff'),
        ('auditory_stim_1',       'Auditory – kernel 1',       '#08306b', '#2171b5'),
        ('piezo_reward_0',        'Reward whisker – kernel 0', '#ff7f00', '#fdbf6f'),
        ('piezo_reward_1',        'Reward whisker – kernel 1', '#b35806', '#e08214'),
        ('piezo_reward_au_0',     'Reward auditory – kernel 0','#984ea3', '#c994c7'),
        ('piezo_reward_au_1',     'Reward auditory – kernel 1','#6a3d9a', '#9970ab'),
    ]

    def _extract_kernel(predictors, coefs, kernel_name):
        pattern = re.compile(fr"^{re.escape(kernel_name)}_t([+-]\d+\.\d+)s$")
        matches = sorted(
            [(float(m.group(1)), i)
             for i, p in enumerate(predictors)
             if (m := pattern.match(p.decode() if isinstance(p, bytes) else p))]
        )
        if not matches:
            return None, None
        lags = np.array([lag for lag, _ in matches])
        idx  = [i for _, i in matches]
        return lags, np.array(coefs)[idx]

    def _collect_neuron_matrix(region_df, kernel_name):
        """Return (lags, n_neurons x n_lags matrix) for neurons that have the kernel."""
        lags_ref, rows = None, []
        for _, row in region_df.iterrows():
            lags, coefs = _extract_kernel(
                row['predictors_full'], row['coef_full_mean'], kernel_name)
            if lags is not None:
                if lags_ref is None:
                    lags_ref = lags
                rows.append(coefs)
        if not rows or lags_ref is None:
            return None, None
        return lags_ref, np.stack(rows)

    ordered_regions = [
        area
        for areas in area_groups.values()
        for area in areas
        if area in df_full['area_acronym_custom'].values
        or area in df_optimal['area_acronym_custom'].values
    ]

    def _to_str(p):
        return p.decode() if isinstance(p, bytes) else p

    for kernel_name, title, color_full, color_optimal in KERNEL_SPECS:
        # Quick check: kernel present in at least one model
        sample_full    = df_full.iloc[0]['predictors_full'] if len(df_full) > 0 else []
        sample_optimal = df_optimal.iloc[0]['predictors_full'] if len(df_optimal) > 0 else []
        has_kernel = (
            any(_to_str(p).startswith(kernel_name + '_t') for p in sample_full) or
            any(_to_str(p).startswith(kernel_name + '_t') for p in sample_optimal)
        )
        if not has_kernel:
            continue

        n_rows = math.ceil(len(ordered_regions) / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), sharex=False)
        axes = np.array(axes).reshape(-1)

        for ax, region in zip(axes, ordered_regions):
            reg_full    = df_full[df_full['area_acronym_custom'] == region]
            reg_optimal = df_optimal[df_optimal['area_acronym_custom'] == region]

            # Restrict to neurons present in both models.
            # Cast to str to avoid silent drops from dtype mismatches (int vs str, bytes vs str).
            ids_full    = set(zip(reg_full['mouse_id'].astype(str),    reg_full['neuron_id'].astype(str)))
            ids_optimal = set(zip(reg_optimal['mouse_id'].astype(str), reg_optimal['neuron_id'].astype(str)))
            common_ids  = ids_full & ids_optimal

            mask_f = reg_full.apply(
                lambda r: (str(r['mouse_id']), str(r['neuron_id'])) in common_ids, axis=1)
            mask_o = reg_optimal.apply(
                lambda r: (str(r['mouse_id']), str(r['neuron_id'])) in common_ids, axis=1)
            reg_full    = reg_full[mask_f]
            reg_optimal = reg_optimal[mask_o]

            lags_f, mat_f = _collect_neuron_matrix(reg_full, kernel_name)
            lags_o, mat_o = _collect_neuron_matrix(reg_optimal, kernel_name)

            n_common = len(common_ids)
            plotted = False
            if mat_f is not None and len(mat_f) > 0:
                mean_f = np.nanmean(mat_f, axis=0)
                sem_f  = np.nanstd(mat_f, axis=0, ddof=1) / np.sqrt(len(mat_f))
                ax.plot(lags_f, mean_f, color=color_full, lw=2,
                        label=f'full (n={n_common})')
                ax.fill_between(lags_f, mean_f - sem_f, mean_f + sem_f,
                                color=color_full, alpha=0.2)
                plotted = True

            if mat_o is not None and len(mat_o) > 0:
                mean_o = np.nanmean(mat_o, axis=0)
                sem_o  = np.nanstd(mat_o, axis=0, ddof=1) / np.sqrt(len(mat_o))
                ax.plot(lags_o, mean_o, color=color_optimal, lw=2, linestyle='--',
                        label=f'optimal split (n={n_common})')
                ax.fill_between(lags_o, mean_o - sem_o, mean_o + sem_o,
                                color=color_optimal, alpha=0.2)
                plotted = True

            # Paired t-test on the matched neurons
            if mat_f is not None and mat_o is not None and lags_f is not None and n_common >= 3:
                pairs_f, pairs_o = [], []
                for uid in common_ids:
                    rows_f = reg_full[
                        (reg_full['mouse_id'].astype(str) == uid[0]) &
                        (reg_full['neuron_id'].astype(str) == uid[1])]
                    rows_o = reg_optimal[
                        (reg_optimal['mouse_id'].astype(str) == uid[0]) &
                        (reg_optimal['neuron_id'].astype(str) == uid[1])]
                    if rows_f.empty or rows_o.empty:
                        continue
                    _, cf = _extract_kernel(
                        rows_f.iloc[0]['predictors_full'],
                        rows_f.iloc[0]['coef_full_mean'], kernel_name)
                    _, co = _extract_kernel(
                        rows_o.iloc[0]['predictors_full'],
                        rows_o.iloc[0]['coef_full_mean'], kernel_name)
                    if cf is not None and co is not None and len(cf) == len(co):
                        pairs_f.append(cf)
                        pairs_o.append(co)

                if len(pairs_f) >= 3:
                    mat_pf = np.stack(pairs_f)
                    mat_po = np.stack(pairs_o)
                    n_lags = mat_pf.shape[1]
                    pvals  = np.ones(n_lags)
                    for t in range(n_lags):
                        try:
                            _, pvals[t] = ttest_rel(mat_pf[:, t], mat_po[:, t])
                        except Exception:
                            pass
                    sig_bins = pvals < alpha
                    if sig_bins.any():
                        y_top = ax.get_ylim()[1]
                        ax.scatter(lags_f[sig_bins],
                                   np.full(sig_bins.sum(), y_top * 0.95),
                                   marker='*', color='black', s=20, zorder=5)

            color = area_colors.get(
                next((g for g, areas in area_groups.items() if region in areas), None),
                'gray')
            ax.set_title(region, fontsize=9, color=color, fontweight='bold')
            ax.set_xlabel('Lag (s)', fontsize=8)
            ax.set_ylabel('Coefficient', fontsize=8)
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if plotted:
                ax.legend(fontsize=7)

        for ax in axes[len(ordered_regions):]:
            ax.set_visible(False)

        fig.suptitle(f'{title}  |  full vs optimal split', fontsize=12)
        fig.tight_layout()
        safe_name = kernel_name.replace('/', '_')
        putils.save_figure_with_options(
            fig, file_formats=['png', 'pdf'],
            filename=f'full_vs_optimal_split_{safe_name}',
            output_dir=output_folder)
        plt.close(fig)


def plot_split_idx_distribution(df_optimal, output_folder, area_groups, area_colors, n_cols=3):
    """
    Plot the distribution of split_idx values for the optimal_split model.

    split_idx is the trial index chosen as the boundary between kernel_0 and
    kernel_1 for each neuron.  A uniform distribution across trial positions
    would mean no preferred split point; a peaked distribution indicates a
    preferred moment in the session where the neural response changes.

    Produces:
      - One figure with a histogram per brain area (deduplicated per neuron,
        taking the value from the first fold).
      - One overall histogram across all areas.
      - One figure showing median split_idx per area as a bar plot.
    """
    os.makedirs(output_folder, exist_ok=True)

    if 'split_idx' not in df_optimal.columns:
        print('[plot_split_idx_distribution] No split_idx column found — skipping.')
        return

    # One row per neuron (first fold is enough; split_idx is neuron-level)
    df_dedup = (df_optimal
                .drop_duplicates(subset=['mouse_id', 'neuron_id'], keep='first')
                .copy())
    df_dedup['split_idx'] = pd.to_numeric(df_dedup['split_idx'], errors='coerce')
    df_dedup = df_dedup.dropna(subset=['split_idx'])

    if df_dedup.empty:
        print('[plot_split_idx_distribution] split_idx is empty after cleaning — skipping.')
        return

    ordered_regions = [
        area
        for areas in area_groups.values()
        for area in areas
        if area in df_dedup['area_acronym_custom'].values
    ]

    # ── 1. Histogram per brain area ───────────────────────────────────────────
    n_rows = math.ceil(len(ordered_regions) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    global_min = df_dedup['split_idx'].min()
    global_max = df_dedup['split_idx'].max()
    bins = np.linspace(global_min, global_max, 30)

    for ax, region in zip(axes, ordered_regions):
        reg_df = df_dedup[df_dedup['area_acronym_custom'] == region]
        vals = reg_df['split_idx'].values

        color = area_colors.get(
            next((g for g, areas in area_groups.items() if region in areas), None),
            'steelblue')

        ax.hist(vals, bins=bins, color=color, alpha=0.75, edgecolor='white', lw=0.4)
        median_val = np.median(vals)
        ax.axvline(median_val, color='black', lw=1.5, ls='--',
                   label=f'median={median_val:.0f}')
        ax.set_title(f'{region}  (n={len(vals)})', fontsize=9,
                     color=color, fontweight='bold')
        ax.set_xlabel('split_idx (trial)', fontsize=8)
        ax.set_ylabel('# neurons', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=7)

    for ax in axes[len(ordered_regions):]:
        ax.set_visible(False)

    fig.suptitle('split_idx distribution per area  |  optimal split', fontsize=12)
    fig.tight_layout()
    putils.save_figure_with_options(fig, file_formats=['png', 'pdf'],
                                    filename='split_idx_distribution_per_area',
                                    output_dir=output_folder)
    plt.close(fig)

    # ── 2. Overall histogram (all areas combined) ─────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    all_vals = df_dedup['split_idx'].values
    ax2.hist(all_vals, bins=30, color='steelblue', alpha=0.75,
             edgecolor='white', lw=0.4)
    ax2.axvline(np.median(all_vals), color='black', lw=1.5, ls='--',
                label=f'median={np.median(all_vals):.0f}')
    ax2.axvline(np.mean(all_vals), color='red', lw=1.5, ls=':',
                label=f'mean={np.mean(all_vals):.0f}')
    ax2.set_xlabel('split_idx (trial)', fontsize=10)
    ax2.set_ylabel('# neurons', fontsize=10)
    ax2.set_title(f'split_idx distribution — all areas  (n={len(all_vals)})', fontsize=11)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    putils.save_figure_with_options(fig2, file_formats=['png', 'pdf'],
                                    filename='split_idx_distribution_overall',
                                    output_dir=output_folder)
    plt.close(fig2)

    # ── 3. Median split_idx per area (bar plot) ───────────────────────────────
    medians, stds, ns, colors_bar, region_labels = [], [], [], [], []
    for region in ordered_regions:
        vals = df_dedup[df_dedup['area_acronym_custom'] == region]['split_idx'].values
        if len(vals) == 0:
            continue
        medians.append(np.median(vals))
        stds.append(np.std(vals))
        ns.append(len(vals))
        colors_bar.append(area_colors.get(
            next((g for g, areas in area_groups.items() if region in areas), None),
            'steelblue'))
        region_labels.append(f'{region}\n(n={len(vals)})')

    fig3, ax3 = plt.subplots(figsize=(max(6, len(medians) * 0.8), 4))
    x = np.arange(len(medians))
    bars = ax3.bar(x, medians, color=colors_bar, alpha=0.8, edgecolor='white')
    ax3.errorbar(x, medians, yerr=stds, fmt='none', color='black',
                 capsize=3, lw=1.2)
    ax3.set_xticks(x)
    ax3.set_xticklabels(region_labels, fontsize=8, rotation=45, ha='right')
    ax3.set_ylabel('median split_idx (trial)', fontsize=10)
    ax3.set_title('Median optimal split index per area', fontsize=11)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    fig3.tight_layout()
    putils.save_figure_with_options(fig3, file_formats=['png', 'pdf'],
                                    filename='split_idx_median_per_area',
                                    output_dir=output_folder)
    plt.close(fig3)


def plot_activity_trial_correlation_matrices(
        merged_pred_df, merged_df, output_folder, area_groups, area_colors, n_cols=4):
    """
    For each consistency group (consistent up, consistent down, non-consistent),
    and for each kernel pair found in merged_df, compute and plot the average
    trial × trial correlation matrix of neural activity per area per mouse.

    For each neuron, a (n_trials × n_trials) Pearson correlation matrix is computed
    from the trial responses (each trial = PSTH vector of n_bins).  These matrices
    are then averaged across neurons sharing the same (mouse, area, group).

    Three trial subsets per figure:
      - whisker hits only  (trial_type == 'whisker', lick_flag == 1)
      - whisker misses only (trial_type == 'whisker', lick_flag == 0)
      - both mixed          (trial_type == 'whisker')

    Each saved figure: 2 rows (real | predicted) × n_areas cols.
    """
    os.makedirs(output_folder, exist_ok=True)

    # ── decode helper ─────────────────────────────────────────────────────────
    def _decode(row):
        def _j(v):
            return json.loads(v) if isinstance(v, str) else v
        tts   = np.array(_j(row['trial_types']))
        licks = np.array(_j(row['lick_flags']), dtype=int)
        y_t   = np.array(_j(row['y_test']), dtype=float) * _SPIKES_TO_HZ
        y_p   = np.array(_j(row['y_pred']), dtype=float) * _SPIKES_TO_HZ
        return tts, licks, y_t, y_p

    def _trial_corr(mat):
        """Return n_trials × n_trials Pearson correlation matrix, or None."""
        if mat.shape[0] < 3:
            return None
        # Avoid degenerate rows (constant activity)
        stds = mat.std(axis=1)
        if (stds == 0).all():
            return None
        try:
            return np.corrcoef(mat)
        except Exception:
            return None

    # ── build consistency lookup from merged_df ───────────────────────────────
    cons_cols = ['mouse_id', 'neuron_id', 'kernel_pair', 'is_consistent', 'consistency_score']
    if not all(c in merged_df.columns for c in cons_cols):
        print("[plot_activity_trial_correlation_matrices] Consistency columns missing – skipping.")
        return

    cons_lookup = (
        merged_df[merged_df['model_name'] == 'full'][cons_cols]
        .dropna(subset=['kernel_pair'])
        .groupby(['mouse_id', 'neuron_id', 'kernel_pair'])
        .first()
        .reset_index()
    )

    # Ordered areas
    all_areas_in_pred = set(merged_pred_df['area_acronym_custom'].unique())
    ordered_areas = [
        area for areas in area_groups.values()
        for area in areas if area in all_areas_in_pred
    ]
    area_to_color = {
        area: area_colors.get(grp, 'gray')
        for grp, areas in area_groups.items()
        for area in areas
    }

    reward_labels = {1: 'r+', 0: 'r-'}

    TRIAL_SUBSETS = [
        ('hits',   lambda tts, licks: (tts == 'whisker_trial') & (licks == 1)),
        ('misses', lambda tts, licks: (tts == 'whisker_trial') & (licks == 0)),
        ('both',   lambda tts, licks:  tts == 'whisker_trial'),
    ]

    # ── shared helper: run corr plots for one group ──────────────────────────
    def _run_corr_group(group_ids, group_label, group_folder, pair_name=''):
        group_pred = merged_pred_df.merge(
            group_ids.drop_duplicates(), on=['mouse_id', 'neuron_id'], how='inner')
        if group_pred.empty:
            return
        os.makedirs(group_folder, exist_ok=True)

        for reward_group, rg_label in reward_labels.items():
            rg_pred = group_pred[group_pred['reward_group'] == reward_group]
            if rg_pred.empty:
                continue
            rg_folder = os.path.join(group_folder, rg_label)
            os.makedirs(rg_folder, exist_ok=True)

            for trial_label, trial_mask_fn in TRIAL_SUBSETS:

                def _compute_mouse_corr(args, _tmf=trial_mask_fn):
                    mid, mpred = args
                    real_by_area, pred_by_area, n_by = {}, {}, {}
                    for area in ordered_areas:
                        area_rows = mpred[mpred['area_acronym_custom'] == area]
                        if area_rows.empty:
                            continue
                        real_mats, pred_mats = [], []
                        for _, row in area_rows.iterrows():
                            tts, licks, y_t, y_p = _decode(row)
                            mask = _tmf(tts, licks)
                            if mask.sum() < 3:
                                continue
                            cr = _trial_corr(y_t[mask])
                            cp = _trial_corr(y_p[mask])
                            if cr is not None:
                                real_mats.append(cr)
                            if cp is not None:
                                pred_mats.append(cp)
                        if real_mats:
                            real_by_area[area] = np.nanmean(np.stack(real_mats), axis=0)
                            n_by[area] = len(real_mats)
                        if pred_mats:
                            pred_by_area[area] = np.nanmean(np.stack(pred_mats), axis=0)
                    return mid, real_by_area, pred_by_area, n_by

                mouse_groups = list(rg_pred.groupby('mouse_id'))
                from concurrent.futures import ThreadPoolExecutor as _TPE
                with _TPE(max_workers=min(8, len(mouse_groups))) as ex:
                    mouse_results = list(ex.map(_compute_mouse_corr, mouse_groups))

                for mouse_id, real_corr_by_area, pred_corr_by_area, n_by_area \
                        in mouse_results:

                    areas_to_plot = [a for a in ordered_areas if a in real_corr_by_area]
                    if not areas_to_plot:
                        continue

                    n_areas_plot = len(areas_to_plot)
                    fig, axes = plt.subplots(2, n_areas_plot,
                                             figsize=(3.5 * n_areas_plot, 7),
                                             squeeze=False)

                    for ai, area in enumerate(areas_to_plot):
                        color = area_to_color.get(area, 'gray')
                        n_neu = n_by_area.get(area, 0)
                        for row_idx, (label_row, corr_dict) in enumerate([
                                ('real', real_corr_by_area),
                                ('predicted', pred_corr_by_area)]):
                            ax = axes[row_idx, ai]
                            mat = corr_dict.get(area)
                            if mat is None:
                                ax.set_visible(False)
                                continue
                            # Scale on off-diagonal only so the diagonal (=1) doesn't dominate
                            off_diag = mat[~np.eye(mat.shape[0], dtype=bool)]
                            abs_max = np.nanmax(np.abs(off_diag)) if off_diag.size else 1.0
                            im = ax.imshow(mat, cmap='RdBu_r',
                                           vmin=-abs_max, vmax=abs_max,
                                           aspect='equal',
                                           interpolation='nearest')
                            ax.set_xticks([])
                            ax.set_yticks([])
                            n_trials = mat.shape[0]
                            if row_idx == 0:
                                ax.set_title(f'{area}\n(n={n_neu} neu, {n_trials} trials)',
                                             fontsize=8, color=color, fontweight='bold')
                            else:
                                ax.set_xlabel(f'{n_trials} trials', fontsize=7)
                            if ai == 0:
                                ax.set_ylabel(label_row, fontsize=8)
                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='r')

                    title_parts = [p for p in [group_label, pair_name, trial_label,
                                               rg_label, mouse_id] if p]
                    fig.suptitle('  |  '.join(title_parts), fontsize=9)
                    fig.tight_layout()
                    putils.save_figure_with_options(
                        fig, file_formats=['png', 'pdf'],
                        filename=f'{mouse_id}_{trial_label}', output_dir=rg_folder)
                    plt.close(fig)

    # ── all_neurons: independent of kernel pair ───────────────────────────────
    all_neu_ids = cons_lookup[['mouse_id', 'neuron_id']].drop_duplicates()
    _run_corr_group(all_neu_ids, 'all_neurons',
                    os.path.join(output_folder, 'all_neurons'))

    for pair_name in sorted(cons_lookup['kernel_pair'].dropna().unique()):
        pair_cons = cons_lookup[cons_lookup['kernel_pair'] == pair_name]

        is_cons = pair_cons['is_consistent'].fillna(False).astype(bool)
        score   = pair_cons['consistency_score'].fillna(0)

        groups = {
            'consistent_up':   pair_cons.loc[is_cons & (score > 0),  ['mouse_id', 'neuron_id']],
            'consistent_down': pair_cons.loc[is_cons & (score < 0),  ['mouse_id', 'neuron_id']],
            'non_consistent':  pair_cons.loc[~is_cons,               ['mouse_id', 'neuron_id']],
        }

        for group_label, group_ids in groups.items():
            _run_corr_group(group_ids, group_label,
                            os.path.join(output_folder,
                                         pair_name.replace(' ', '_'), group_label),
                            pair_name=pair_name)
