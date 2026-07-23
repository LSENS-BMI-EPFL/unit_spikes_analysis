import os
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
from scipy import signal 
import matplotlib.pyplot as plt
from helper import save_figure_multiple_formats, parse_array


DATA_PATH = os.path.join('/Volumes', 'Petersen-Lab', 'z_LSENS', 'Share', 'Dana_Shayakhmetova', 'dynamic_analysis_dec16')
AUTOCORR_OUTPUT_DIR = os.path.join(DATA_PATH, 'auto_correlations')


#Helpers
def calculate_autocorr(series, lags_range=np.arange(0, 11)):
    """Calculate normalized autocorrelation at specified time lags for a single time series."""
    N = len(series)
    if N < 2 or np.std(series) == 0:
        print(f"Series length: {N}, std: {np.std(series) if N > 0 else 'N/A'}")
        print("Autocorrelation is skipped.")
        return None

    series_norm = (series - np.mean(series)) / np.std(series)
    raw_corr = signal.correlate(series_norm, series_norm, mode='full')
    all_lags = signal.correlation_lags(N, N, mode='full')
    normalized_corr = raw_corr / N 
    lag_mask = np.isin(all_lags, lags_range)
    final_lags = all_lags[lag_mask]
    final_corrs = normalized_corr[lag_mask]
    return pd.DataFrame({'lag': final_lags, 'autocorr': final_corrs})


#Behavioral Autocorr
def plot_behavioral_autocorr_by_reward_group(bhv_data, mouse_info_df, output_path, bhv_metric):
    """Plot mean behavioral autocorrelation grouped by reward groups."""

    os.makedirs(output_path, exist_ok=True)
    
    if bhv_metric == 'performance':
        title= '(HR - FA)'
    elif bhv_metric == 'p_mean':
        title = 'Hit Rate'
    else:
        title = 'False Alarm Rate'
    
    mouse_group_map = mouse_info_df[['mouse_id', 'reward_group']].drop_duplicates()
    bhv_data_merged = pd.merge(bhv_data, mouse_group_map, on='mouse_id', how='inner')


    all_autocorrs = []    
    for mouse, mouse_trials_df in bhv_data_merged.groupby('mouse_id'):
        reward_group = mouse_trials_df['reward_group'].iloc[0]

        if bhv_metric == 'performance':
            bhv_series = mouse_trials_df['p_mean'].values - mouse_trials_df['p_chance'].values
        else:
            bhv_series = mouse_trials_df[bhv_metric].values
                
        autocorr_df = calculate_autocorr(bhv_series)
        if autocorr_df is not None:
            autocorr_df['mouse_id'] = mouse
            autocorr_df['reward_group'] = reward_group
            all_autocorrs.append(autocorr_df)

    full_autocorr_df = pd.concat(all_autocorrs, ignore_index=True)
    full_autocorr_df.to_pickle(os.path.join(output_path, f'bhv_autocorr_{bhv_metric}.pkl'))
    full_autocorr_df.to_csv(os.path.join(output_path, f'bhv_autocorr_{bhv_metric}.csv'))
    print("Saved the data.")

    mean_sem_df = (
        full_autocorr_df.groupby(['reward_group', 'lag'])['autocorr']
        .agg(['mean', 'sem'])
        .reset_index()
    )    
    fig, ax = plt.subplots(figsize=(8, 6))
    lags = mean_sem_df['lag'].unique()
    
    print("Starting to plot...")
    for group in list(mean_sem_df.reward_group.unique()):
        group_data = mean_sem_df[mean_sem_df['reward_group'] == group]
        color = 'forestgreen' if group == 'R+' else 'crimson'
        ax.plot(group_data['lag'], group_data['mean'], marker='o', linestyle='-', linewidth=2, color=color, label=f'{group} Mean')
        ax.fill_between(
                    group_data['lag'], 
                    group_data['mean'] - group_data['sem'], 
                    group_data['mean'] + group_data['sem'], 
                    color=color, 
                    alpha=0.15
                )
        
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(-0.5,1.05)
    ax.set_xlim(0, 10)
    ax.set_xticks(lags)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_title(f'Behavioral Autocorrelation: {title}', fontsize=14, fontweight='bold')
    ax.legend(title='Reward Group', loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    filename_base = f"autocorr_behavior_{bhv_metric}_by_reward_group"
    save_figure_multiple_formats(fig, os.path.join(output_path, filename_base))
    plt.close(fig)
    print("Done.")

def plot_behavioral_autocorr_per_mouse(bhv_data, mouse_info_df, output_path, bhv_metric):
    """Plot behavioral autocorrelation for all individual mice colored by reward group."""
    mice_auto_corr_data = os.path.join(output_path, "mice")
    os.makedirs(mice_auto_corr_data, exist_ok=True)

    if bhv_metric == "performance":
        title_metric = "(HR minus FA)"
    elif bhv_metric == "p_mean":
        title_metric = "Hit Rate"
    else:
        title_metric = "False Alarm Rate"

    mouse_group_map = mouse_info_df[["mouse_id", "reward_group"]].drop_duplicates()
    bhv_data_merged = pd.merge(bhv_data, mouse_group_map, on="mouse_id", how="inner")

    for mouse, mouse_trials_df in bhv_data_merged.groupby("mouse_id"):
        reward_group = mouse_trials_df["reward_group"].iloc[0]
        if bhv_metric == "performance":
            bhv_series = (mouse_trials_df["p_mean"].values - mouse_trials_df["p_chance"].values)
        else:
            bhv_series = mouse_trials_df[bhv_metric].values

        autocorr_df = calculate_autocorr(bhv_series)
        if autocorr_df is None:
            print(f"No valid autocorrelation for mouse {mouse}")
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        color = "forestgreen" if reward_group == "R+" else "crimson"
        ax.plot(
            autocorr_df["lag"],
            autocorr_df["autocorr"],
            marker="o",
            linewidth=1.5,
            alpha=0.8,
            color=color
        )

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"Mouse {mouse} Autocorrelation {title_metric} ({reward_group})")
        ax.set_ylim(-0.5, 1.05)
        ax.set_xlim(0, 10)
        ax.set_xticks(sorted(autocorr_df["lag"].unique()))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        filename_base = f"autocorr_behavior_{bhv_metric}_{mouse}"
        save_figure_multiple_formats(fig, os.path.join(mice_auto_corr_data, filename_base))
        plt.close(fig)



#Neural Autocorr
def plot_neural_autocorr_by_reward_group(neural_data, mouse_info_df, output_path, neural_metric):
    """Plot mean neural autocorrelation by brain area, grouped by reward group."""

    os.makedirs(output_path, exist_ok=True)

    if neural_metric == 'baseline':
        title = 'Pre stimulus Firing Rate'
    else:
        title = 'Evoked Response Firing Rate'

    mouse_group_map = mouse_info_df[['mouse_id', 'reward_group']].drop_duplicates()
    neural_df = pd.merge(neural_data, mouse_group_map, on='mouse_id', how='inner')

    all_autocorrs = []


    for (mouse, area, neuron), neuron_df in neural_df.groupby(['mouse_id', 'area_custom', 'neuron_id']):
        reward_group = mouse_info_df[mouse_info_df.mouse_id==mouse].iloc[0].reward_group
        if neural_metric == 'baseline':
            neural_series = neuron_df['pre_firing_rate'].iloc[0]
        else:
            neural_series = neuron_df['rate_evoked_response'].iloc[0]

        autocorr_df = calculate_autocorr(neural_series)
        if autocorr_df is not None:
            autocorr_df['mouse_id'] = mouse
            autocorr_df['area_custom'] = area
            autocorr_df['neuron_id'] = neuron
            autocorr_df['reward_group'] = reward_group
            all_autocorrs.append(autocorr_df)

    full_autocorr_df = pd.concat(all_autocorrs, ignore_index=True)
    full_autocorr_df.to_pickle(os.path.join(output_path, f'neural_autocorr_{neural_metric}.pkl'))
    full_autocorr_df.to_csv(os.path.join(output_path, f'neural_autocorr_{neural_metric}.csv'))

    mouse_area_mean_df = (
        full_autocorr_df
        .groupby(['mouse_id', 'area_custom', 'reward_group', 'lag'])['autocorr']
        .mean()
        .reset_index()
    )
    mean_sem_df = (
        mouse_area_mean_df
        .groupby(['area_custom', 'reward_group', 'lag'])['autocorr']
        .agg(['mean', 'sem'])
        .reset_index()
    )

    for area in mean_sem_df['area_custom'].unique():
        area_df = mean_sem_df[mean_sem_df['area_custom'] == area]
        fig, ax = plt.subplots(figsize=(8, 6))
        lags = area_df['lag'].unique()

        for group in area_df['reward_group'].unique():
            group_data = area_df[area_df['reward_group'] == group]
            color = 'forestgreen' if group == 'R+' else 'crimson'
            ax.plot(
                group_data['lag'],
                group_data['mean'],
                marker='o',
                linewidth=2,
                color=color,
                label=f'{group}'
            )

            ax.fill_between(
                group_data['lag'],
                group_data['mean'] - group_data['sem'],
                group_data['mean'] + group_data['sem'],
                color=color,
                alpha=0.15
            )
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(-0.5, 1.05)
        ax.set_xticks(lags)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_title(f'{area} Neural Autocorrelation {title}')
        ax.legend(title='Reward Group')
        ax.grid(True, alpha=0.3)
        filename_base = f'autocorr_neural_{neural_metric}_{area}_by_reward_group'
        save_figure_multiple_formats(fig, os.path.join(output_path, filename_base))
        plt.close(fig)

def plot_neural_autocorr_by_area(neural_data, mouse_info_df, output_path, neural_metric):
    """Plot neural autocorrelation for all mice within each brain area colored by reward group."""

    os.makedirs(output_path, exist_ok=True)

    if neural_metric == 'baseline':
        title = 'Pre stimulus Firing Rate'
    else:
        title = 'Evoked Response Firing Rate'

    mouse_group_map = mouse_info_df[['mouse_id', 'reward_group']].drop_duplicates()
    neural_df = pd.merge(neural_data, mouse_group_map, on='mouse_id', how='inner')

    all_autocorrs = []


    for (mouse, area, neuron), neuron_df in neural_df.groupby(['mouse_id', 'area_custom', 'neuron_id']):
        reward_group = mouse_info_df[mouse_info_df.mouse_id==mouse].iloc[0].reward_group


        if neural_metric == 'baseline':
            neural_series = neuron_df['pre_firing_rate'].iloc[0]
        else:
            neural_series = neuron_df['rate_evoked_response'].iloc[0]

        autocorr_df = calculate_autocorr(neural_series)

        if autocorr_df is not None:
            autocorr_df['mouse_id'] = mouse
            autocorr_df['area_custom'] = area
            autocorr_df['neuron_id'] = neuron
            autocorr_df['reward_group'] = reward_group
            all_autocorrs.append(autocorr_df)

    full_autocorr_df = pd.concat(all_autocorrs, ignore_index=True)
    full_autocorr_df.to_pickle(os.path.join(output_path, f'neural_autocorr_{neural_metric}.pkl'))
    full_autocorr_df.to_csv(os.path.join(output_path, f'neural_autocorr_{neural_metric}.csv'))

    mouse_area_mean_df = (
        full_autocorr_df
        .groupby(['mouse_id', 'area_custom', 'reward_group', 'lag'])['autocorr']
        .mean()
        .reset_index()
    )

    for area in mouse_area_mean_df['area_custom'].unique():
        area_df = mouse_area_mean_df[mouse_area_mean_df['area_custom'] == area]
        fig, ax = plt.subplots(figsize=(8, 6))
        lags = sorted(area_df['lag'].unique())
        for mouse in area_df['mouse_id'].unique():
            mouse_df = area_df[area_df['mouse_id'] == mouse]
            reward_group = mouse_df['reward_group'].iloc[0]

            color = 'forestgreen' if reward_group == 'R+' else 'crimson'

            ax.plot(
                mouse_df['lag'],
                mouse_df['autocorr'],
                # marker='o',
                linewidth=1.5,
                alpha=0.8,
                color=color,
                label=reward_group
            )

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(-0.5, 1.05)
        ax.set_xlim(0, 10)
        ax.set_xticks(lags)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_title(f'{title} Autocorrelation in {area}')
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title='Reward Group')

        filename_base = f'autocorr_neural_{neural_metric}_{area}_per_mouse'
        save_figure_multiple_formats(fig, os.path.join(output_path, filename_base))
        plt.close(fig)


def main():
    bhv_data_path = os.path.join(DATA_PATH, 'behavior_data', 'merged_bhv_trials.pkl')
    mouse_info_path = os.path.join("/Volumes", "Petersen-Lab", "z_LSENS", "Share", "Dana_Shayakhmetova", "mouse_info", "joint_mouse_reference_weight.xlsx")
    
    print("Loading mouse info...")
    mouse_info_df = pd.read_excel(mouse_info_path)
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df = mouse_info_df[
        (mouse_info_df['exclude'] == 0) &
        (mouse_info_df['reward_group'].isin(['R+', 'R-'])) &
        (mouse_info_df['recording'] == 1)]
    print("Done.")

    print("\nLoading neural data...")
    neural_path = os.path.join(DATA_PATH, 'neuronal_data','neuronal_data_mh_ab.csv')
    neural_data = pd.read_csv(neural_path)
    neural_data = neural_data[~neural_data.mouse_id.isin(['AB068','AB164', 'AB142', 'AB107', 'AB149'])]
    array_cols = ["pre_firing_rate", "post_firing_rate", "rate_evoked_response"]
    for col in array_cols:
        neural_data[col] = neural_data[col].apply(parse_array)
    print(neural_data.head())
    print("done.")

    print("Loading behavioral data...")
    bhv_data = pd.read_pickle(bhv_data_path)
    bhv_data = bhv_data[bhv_data.whisker_stim == 1]
    print("Done.")
    
    print("Plotting mean behavioral autocorr by reward group...")
    plot_behavioral_autocorr_by_reward_group(bhv_data, mouse_info_df, os.path.join(AUTOCORR_OUTPUT_DIR, 'behavioral_autocorr', 'performance'), 'performance')
    plot_behavioral_autocorr_by_reward_group(bhv_data, mouse_info_df, os.path.join(AUTOCORR_OUTPUT_DIR,'behavioral_autocorr', 'hit_rate'), 'p_mean')
    plot_behavioral_autocorr_by_reward_group(bhv_data, mouse_info_df, os.path.join(AUTOCORR_OUTPUT_DIR, 'behavioral_autocorr','false_alarm'), 'p_chance')
    print("Done.")

    print("Plotting mean behavioral autocorr by mouse...")
    plot_behavioral_autocorr_per_mouse(bhv_data, mouse_info_df, os.path.join(AUTOCORR_OUTPUT_DIR, 'behavioral_autocorr','performance'), 'performance')
    plot_behavioral_autocorr_per_mouse(bhv_data, mouse_info_df, os.path.join(AUTOCORR_OUTPUT_DIR, 'behavioral_autocorr','hit_rate'), 'p_mean')
    plot_behavioral_autocorr_per_mouse(bhv_data, mouse_info_df, os.path.join(AUTOCORR_OUTPUT_DIR, 'behavioral_autocorr','false_alarm'), 'p_chance')
    print("Done.")


    print("Plotting mean neural autocorr by reward group...")
    plot_neural_autocorr_by_reward_group(neural_data, mouse_info_df, os.path.join(AUTOCORR_OUTPUT_DIR, 'neural_autocorr', 'baseline', 'mean_by_reward_group'), 'baseline')
    plot_neural_autocorr_by_reward_group(neural_data, mouse_info_df, os.path.join(AUTOCORR_OUTPUT_DIR, 'neural_autocorr', 'evoked_response','mean_by_reward_group'), 'evoked_response')
    print("Done.")

    print("Plotting mean neural autocorr by area...")
    plot_neural_autocorr_by_area(neural_data, mouse_info_df, os.path.join(AUTOCORR_OUTPUT_DIR,'neural_autocorr', 'baseline', 'all_mice_in_area'), 'baseline')
    plot_neural_autocorr_by_area(neural_data, mouse_info_df, os.path.join(AUTOCORR_OUTPUT_DIR, 'neural_autocorr', 'evoked_response', 'all_mice_in_area'), 'evoked_response')
    print("Done.")

if __name__ == '__main__':
    main()