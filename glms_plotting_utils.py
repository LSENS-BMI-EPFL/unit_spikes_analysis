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
    import gc
    del all_y_test, all_y_pred_full, all_y_pred_reduced
    gc.collect()

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



def plot_neuron_kernels_avg_with_responses(neuron_id, glm_full_df, kernels, trials_df, output_folder,
    git_hash="0b29b9b", bin_size=0.1):
    """
    Plot kernels for one neuron alongside average responses and predictions.
    Uses SEM across folds (not across trials).
    """

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


def plot_average_kernels_by_region_per_kernel(
            neuron_ids, merged_df, output_folder, kernels_to_plot,
            lags=None, area_groups=None, area_colors=None, n_cols=3):
        """
        Plot average kernels across neurons grouped by ccf_parent_acronym,
        one figure per kernel, regions colored by area group,
        ordered by area_groups.
        """
        if lags is None:
            lags = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])

        # neuron_ids is now an array of (mouse_id, neuron_id) tuples
        df = merged_df[
            merged_df.set_index(['mouse_id', 'neuron_id']).index.isin(neuron_ids)
        ]

        # Build mapping from region -> color
        region_to_color = {}
        for group_name, areas in area_groups.items():
            for area in areas:
                region_to_color[area] = area_colors[group_name]

        def get_region_color(region_name):
            return region_to_color.get(region_name, 'gray')

        # Order regions by area_groups
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
                kernels_list = []

                for _, row in region_df.iterrows():
                    coefs_list = row['coef']
                    if isinstance(coefs_list, list) and isinstance(coefs_list[0], str):
                        coefs_list = [np.array(ast.literal_eval(c)) for c in coefs_list]
                    else:
                        coefs_list = [np.array(ast.literal_eval(coefs_list))]

                    predictors = ast.literal_eval(row['predictors'])
                    indices = [i for i, p in enumerate(predictors) if p.startswith(kernel)]
                    if len(indices) == 0:
                        continue

                    for c in coefs_list:
                        kernels_list.append(c[indices].ravel())

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

                ax.axvline(0, color='k', linestyle=':')  # vertical line at stimulus onset

                ax.set_title(f"{region} (n={len(region_df['neuron_id'].unique())})", fontsize=10)
                ax.set_xlabel("Lag (s)")
                ax.set_ylabel("Coef")

            # Hide unused axes
            for ax in axes[len(ordered_regions):]:
                ax.set_visible(False)

            plt.suptitle(f"{kernel} average kernels", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f"{output_folder}/average_kernel_{kernel}.png", transparent=True)
            plt.close(fig)


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
