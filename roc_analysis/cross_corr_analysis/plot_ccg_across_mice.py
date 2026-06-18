import os
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

DATA_PATH = os.path.join('/Volumes', 'Petersen-Lab', 'z_LSENS', 'Share', 'Dana_Shayakhmetova', 'dynamic_analysis_dec16')
BASELINE_DATA_DIR = os.path.join(DATA_PATH, 'baseline_analysis', 'cross_corr_data')
LAG_PLOTS_BASELINE = os.path.join(DATA_PATH, 'baseline_analysis', 'all_lags_plots')
EVOKED_RESPONSE_DATA_DIR = os.path.join(DATA_PATH, 'evoked_response_analysis', 'cross_corr_data')
LAG_PLOTS_EVOKED_RESPONSE = os.path.join(DATA_PATH, 'evoked_response_analysis', 'all_lags_plots')


#helpers 
def parse_array(x):
    """Convert string representation of array to numpy array."""
    if isinstance(x, str):
        x = x.replace('\n', ' ').replace('  ', ' ')
        x = x.replace('[', '').replace(']', '')
        if x.strip() == '':
            return np.array([])
        return np.array([float(v) for v in x.split() if v != ''])
    return x

def save_figure_multiple_formats(fig, filepath_base, dpi=300):
    """Save matplotlib figure as PNG, PDF, and SVG files."""
    directory = os.path.dirname(filepath_base)
    if directory:
        os.makedirs(directory, exist_ok=True)

    for ext in ['png', 'pdf', 'svg']:
        filepath = f"{filepath_base}.{ext}"
        try:
            if ext == 'svg':
                fig.savefig(filepath, format='svg', bbox_inches='tight')
            else:
                fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        except OSError as e:
            print(f"Failed to save {filepath}: {e}")
    plt.close(fig)



# Significant proportions 
def calculate_mean_sig_proportion_per_reward_group(corr_neuronal_data,  mouse_info_df):
    """Calculate mean proportion of significant neurons per reward group and area across lags."""
    total_units = (
        corr_neuronal_data.groupby(['mouse_id', 'area_custom'])['neuron_id']
        .nunique()
        .reset_index(name='total_units')
    )

    significant_df = corr_neuronal_data[corr_neuronal_data['significant_fdr'] == True]
    sig_counts = (
        significant_df.groupby(['mouse_id', 'area_custom', 'lag'])['neuron_id']
        .nunique()
        .reset_index(name='significant_counts')
    )
    
    merged_df = pd.merge(sig_counts, total_units, on=['mouse_id', 'area_custom'], how='left')
    merged_df['significant_proportion'] = merged_df['significant_counts'] / merged_df['total_units']
    mouse_group_map = mouse_info_df[['mouse_id', 'reward_group']].drop_duplicates()
    group_df = pd.merge(merged_df, mouse_group_map, on='mouse_id', how='left')
    
    # mean per reward group
    final_proportions = (
        group_df.groupby(['reward_group', 'area_custom', 'lag'])['significant_proportion']
        .mean()
        .reset_index(name='mean_significant_proportion')
    )
    
    # pivot
    final_df = final_proportions.pivot_table(
        index=['reward_group', 'area_custom'], 
        columns='lag', 
        values='mean_significant_proportion',
        fill_value=0
    ).reset_index()
    
    final_df.columns.name = None
    return final_df

def plot_mean_sig_proportion_by_reward_group(mean_group_df, output_path):
    """Plot mean proportion of significant neurons by reward group for each brain area across lags."""
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    reward_groups = mean_group_df['reward_group'].unique()
    areas = mean_group_df['area_custom'].unique()
    
    lag_cols = [col for col in mean_group_df.columns 
                if col not in ['reward_group', 'area_custom']]
    lags = np.array([int(col) for col in lag_cols])

    max_prop = mean_group_df[lag_cols].max().max()

    for area in areas:
        fig, ax = plt.subplots(figsize=(6, 5))  

        area_df = mean_group_df[mean_group_df['area_custom'] == area]
        
        for group in reward_groups:
            group_data = area_df[area_df['reward_group'] == group]
            if group_data.empty:
                continue

            proportions = group_data[lag_cols].iloc[0].values
            color = 'forestgreen' if group == 'R+' else 'crimson'

            ax.plot(
                lags,
                proportions,
                marker='o',
                linestyle='-',
                linewidth=2,
                color=color,
                label=group
            )

        ax.axvline(0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.axhline(0, color='black', linewidth=1)

        ax.set_ylim(0, max_prop * 1.1 if max_prop > 0 else 0.2)
        ax.set_xlim(lags.min() - 0.5, lags.max() + 0.5)
        ax.set_xticks(lags)

        ax.set_title(f'{area}', fontsize=13)
        ax.set_xlabel('Lag', fontsize=11)
        ax.set_ylabel('Mean Proportion', fontsize=11)
        ax.legend(title='Reward Group', loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        if output_path:
            safe_area = area.replace('/', '_').replace('\\', '_')
            filename = f"mean_sig_prop_by_reward_group_{safe_area}"
            # output path/area mean 
            save_figure_multiple_formats(fig, os.path.join(output_path, filename))
        else:
            plt.show()


# Grid view of beahvior and mean CCGs of several mice, ordered by number of neurons 
def plot_mean_ccg_grid(df, bhv_data, shuffled_bhv_data,bhv_metric, mouse_info_df, area_to_plot, output_dir, figsize_per_row=(12, 3)):
    """ Create grid showing behavioral trace (left panel) and mean CCG (right panel) for each mouse in a brain area."""

    area_df = df[df['area_custom'] == area_to_plot]
    mice_with_area = area_df['mouse_id'].unique()
    
    if len(mice_with_area) == 0:
        print(f"No data found for Area: {area_to_plot}")
        return
    
    neuron_counts = area_df.groupby('mouse_id')['neuron_id'].nunique()
    mice_sorted = neuron_counts.sort_values(ascending=False).index.tolist()
    n_mice = len(mice_sorted)
    
    # left = behavioral trace, right = mean CCG
    n_cols = 2
    n_rows = n_mice
    fig_width = figsize_per_row[0]
    fig_height = figsize_per_row[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharex=False, constrained_layout=True)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0) 
    reward_color_map = {'R-': 'crimson', 'R+': 'forestgreen'}
    
    for i, mouse_id in enumerate(mice_sorted):
        reward_group = mouse_info_df.loc[mouse_info_df['mouse_id'] == mouse_id, 'reward_group']
        color = reward_color_map.get(reward_group.values[0], 'gray') if not reward_group.empty else 'gray'
        ax_p = axes[i, 0]
        ax_ccg = axes[i, 1]
        mouse_bhv = bhv_data[bhv_data['mouse_id'] == mouse_id].sort_values('whisker_trial_id')
        bhv_trials = mouse_bhv['whisker_trial_id'].values
        mouse_shuff = shuffled_bhv_data[shuffled_bhv_data['mouse_id'] == mouse_id]

        if not mouse_shuff.empty:
            if bhv_metric == 'performance':
                bhv_title = '(HR-FA)'
                hr_matrix = np.array(mouse_shuff["p_mean_shuffled"].values.tolist())
                fa_matrix = np.array(mouse_shuff["p_chance_shuffled"].values.tolist())
                perf_matrix = hr_matrix - fa_matrix
                mean_perf = np.nanmean(perf_matrix, axis=1)
                sem_perf = np.nanstd(perf_matrix, axis=1, ddof=1) / np.sqrt(perf_matrix.shape[1])
                ax_p.fill_between(bhv_trials, mean_perf - sem_perf, mean_perf + sem_perf, color='black', alpha=0.25, label='Shuffled (HR-FA)')
                mouse_bhv['perf'] = mouse_bhv['p_mean'] - mouse_bhv['p_chance']
                ax_p.plot(mouse_bhv['whisker_trial_id'], mouse_bhv['perf'].values, linestyle='-', color='black')

                
            elif bhv_metric == 'p_mean': #HR
                bhv_title = 'Hit Rate'
                hr_matrix = np.array(mouse_shuff["p_mean_shuffled"].values.tolist())
                mean_hr = np.nanmean(hr_matrix, axis=1)
                sem_hr = np.nanstd(hr_matrix, axis=1, ddof=1) / np.sqrt(hr_matrix.shape[1])
                ax_p.fill_between(bhv_trials, mean_hr - sem_hr, mean_hr + sem_hr, color=color, alpha=0.25, label='Shuffled HR')
                ax_p.plot(mouse_bhv['whisker_trial_id'], mouse_bhv['p_mean'].values, linestyle='-', color=color)


            else: #FA
                bhv_title = 'False Alarm Rate'
                fa_matrix = np.array(mouse_shuff["p_chance_shuffled"].values.tolist())
                mean_fa = np.nanmean(fa_matrix, axis=1)
                sem_fa = np.nanstd(fa_matrix, axis=1, ddof=1) / np.sqrt(fa_matrix.shape[1])
                ax_p.fill_between(bhv_trials, mean_fa - sem_fa, mean_fa + sem_fa, color='gray', alpha=0.25, label='Shuffled FA') 
                ax_p.plot(mouse_bhv['whisker_trial_id'], mouse_bhv['p_chance'].values, linestyle='-', color='black')

        
        ax_p.set_title(f"{bhv_title} of {mouse_id}")
        ax_p.set_xlim(0, max(bhv_trials))
        ax_p.set_xlabel("Whisker Trial ID")
        ax_p.set_ylabel("Response Rate")
        ax_p.xaxis.set_major_locator(MultipleLocator(10))
        ax_p.grid(True, linestyle='--', alpha=0.3)
        ax_p.legend(loc='upper left', fontsize=8)
        plt.setp(ax_p.get_xticklabels(), rotation=45, ha='center')

    
        mouse_df = area_df[area_df['mouse_id'] == mouse_id]
        num_neurons = mouse_df['neuron_id'].nunique()
        ccg_summary = mouse_df.groupby('lag')['cross_corr'].agg(['mean', 'sem']).reset_index()
        ax_ccg.plot(ccg_summary['lag'], ccg_summary['mean'], color=color, linewidth=2)
        ax_ccg.fill_between(
            ccg_summary['lag'],
            ccg_summary['mean'] - ccg_summary['sem'],
            ccg_summary['mean'] + ccg_summary['sem'],
            color=color, alpha=0.3
        )
        ax_ccg.axhline(0, color='gray', linestyle='--', linewidth=1, alpha = 0.4)
        ax_ccg.set_title(f"Mean CCG of {mouse_id} in {area_to_plot} (n={num_neurons})")
        ax_ccg.set_xlabel("Lag")
        ax_ccg.set_xlim(-10,10)
        ax_ccg.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax_ccg.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        max_y = np.max(np.abs(ccg_summary['mean']))
        ax_ccg.set_ylim(-1.1 * max_y, 1.1 * max_y)
        ax_ccg.set_xticks(np.arange(-10, 11, 1))
        ax_ccg.set_ylabel("Mean Cross-Correlation")
        ax_ccg.grid(True, linestyle='--', alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    area_safe = area_to_plot.replace('-', '_')
    filename = f"all_mice_{bhv_metric}_and_ccg_with_shuffled_{area_safe}"
    save_figure_multiple_formats(fig, os.path.join(output_dir, filename))



# Newly added functions from meeting 16/12/2025
# Means across all mice by reward group 
#    1. General Mean CCG 
#       - All neurons 
#       - Filter out mice that have less than 10 neurons 
#    2. Mean CCG separating anti-correlated vs correlated neurons
#        - This is determined by the sign of the peak correlation 
#        - NO filtering 
#    3. Mean CCG of only significant neurons
#        - FDR corrected p-value at its peak
#        - NO filtering 


def plot_mean_ccg_all_neurons_by_reward_group(df, mouse_info_df, area_to_plot, output_dir, min_neurons=10):
    """Plot mean cross-correlogram by reward group using all neurons from mice with ≥min_neurons."""

    area_df = df[df['area_custom'] == area_to_plot]
    neuron_counts = area_df.groupby('mouse_id')['neuron_id'].nunique()
    valid_mice = neuron_counts[neuron_counts >= min_neurons].index
    area_df = area_df[area_df['mouse_id'].isin(valid_mice)]
    if len(valid_mice) == 0:
        print(f"No mice with >={min_neurons} neurons in {area_to_plot}")
        return
    area_df = area_df.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')
    ccg_by_group = area_df.groupby(['reward_group', 'lag'])['cross_corr'].agg(['mean', 'sem']).reset_index()

    group_stats = area_df.groupby('reward_group').agg(
        n_mice=('mouse_id', 'nunique'),
        n_neurons=('neuron_id', 'nunique')
    ).reset_index()

    
    fig, ax = plt.subplots(figsize=(8, 6))
    reward_color_map = {'R-': 'crimson', 'R+': 'forestgreen'}
    
    for group in ccg_by_group['reward_group'].unique():
        group_data = ccg_by_group[ccg_by_group['reward_group'] == group]
        stats = group_stats[group_stats['reward_group'] == group].iloc[0]
        n_mice = int(stats['n_mice'])
        n_neurons = int(stats['n_neurons'])

        color = reward_color_map.get(group, 'gray')
        ax.plot(group_data['lag'], group_data['mean'], color=color, linewidth=2.5,
             label=f"{group} (n={n_mice} mice, {n_neurons} neurons)")
        ax.fill_between(
            group_data['lag'],
            group_data['mean'] - group_data['sem'],
            group_data['mean'] + group_data['sem'],
            color=color, alpha=0.3
        )

    y_max = max(abs(ccg_by_group['mean'] + ccg_by_group['sem']).max(), abs(ccg_by_group['mean'] - ccg_by_group['sem']).max())
    y_max = y_max * 1.05
    ax.set_ylim(-y_max, y_max)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel("Lag", fontsize=12)
    ax.set_xticks(np.arange(-10, 11, 1))
    ax.set_ylabel("Mean Cross-Correlation", fontsize=12)
    ax.set_title(f"Mean CCG by Reward Group in {area_to_plot}\n(mice with ≥{min_neurons} neurons)", fontsize=13)
    ax.legend(title='Reward Group', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    area_safe = area_to_plot.replace('-', '_')
    filename = f"mean_ccg_all_neurons_by_reward_{area_safe}"
    save_figure_multiple_formats(fig, os.path.join(output_dir, filename))


def plot_mean_ccg_by_corr_sign(df, mouse_info_df, area_to_plot, output_dir):
    """ Plot mean CCGs separated by correlation sign (correlated vs anti-correlated neurons).
    Sign determined by whether peak correlation value is positive or negative."""

    area_df = df[df['area_custom'] == area_to_plot].copy()
    area_df = area_df.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')
    
    # correlation sign per neuron based on peak correlation
    peak_idx = area_df.groupby(['mouse_id', 'neuron_id'])['cross_corr'].apply(lambda x: x.abs().idxmax())
    peak_df = area_df.loc[peak_idx, ['mouse_id', 'neuron_id', 'cross_corr']]
    peak_df['corr_type'] = peak_df['cross_corr'].apply(lambda x: 'Correlated' if x > 0 else 'Anti-correlated')
    
    area_df = area_df.merge(peak_df[['mouse_id', 'neuron_id', 'corr_type']], on=['mouse_id', 'neuron_id'], how='left')
    ccg_summary = area_df.groupby(['reward_group', 'corr_type', 'lag'])['cross_corr'].agg(['mean', 'sem']).reset_index()
    reward_color_map = {'R+': 'forestgreen', 'R-': 'crimson'}

    group_stats = area_df.groupby(['reward_group', 'corr_type']).agg(
        n_mice=('mouse_id', 'nunique'),
        n_neurons=('neuron_id', 'nunique')
    ).reset_index()


    y_max = max(abs(ccg_summary['mean'] + ccg_summary['sem']).max(),abs(ccg_summary['mean'] - ccg_summary['sem']).max())
    y_max = y_max * 1.05

    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    
    
    for idx, corr_type in enumerate(['Correlated', 'Anti-correlated']):
        ax = axes[idx]
        type_data = ccg_summary[ccg_summary['corr_type'] == corr_type]
        
        for group in type_data['reward_group'].unique():
            group_data = type_data[type_data['reward_group'] == group]
            color = reward_color_map.get(group, 'gray')
            stats = group_stats[(group_stats['reward_group'] == group) &   
                       (group_stats['corr_type'] == corr_type)].iloc[0]
            n_mice = int(stats['n_mice'])
            n_neurons = int(stats['n_neurons'])

            
            ax.plot(group_data['lag'], group_data['mean'], color=color, linewidth=2.5, 
                label=f"{group} (n={n_mice} mice, {n_neurons} neurons)")
            ax.fill_between(
                group_data['lag'],
                group_data['mean'] - group_data['sem'],
                group_data['mean'] + group_data['sem'],
                color=color, alpha=0.3
            )


        ax.set_ylim(-y_max, y_max)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.set_xticks(np.arange(-10, 11, 1))
        ax.set_xlabel("Lag", fontsize=12)
        ax.set_ylabel("Mean Cross-Correlation", fontsize=12)
        ax.set_title(f"{corr_type} Neurons", fontsize=13)
        ax.legend(title='Reward Group', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
    
    fig.suptitle(f"Mean CCG by Correlation Sign in {area_to_plot}", fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    area_safe = area_to_plot.replace('/', '_').replace('\\', '_')
    filename = f"mean_ccg_{area_safe}_corr_sign"
    save_figure_multiple_formats(fig, os.path.join(output_dir, filename))


def plot_mean_ccg_by_peak_significance(df, mouse_info_df, area_to_plot, output_dir):
    """Plot mean CCG by reward group using only neurons with FDR-significant peak correlation."""

    area_df = df[df['area_custom'] == area_to_plot].copy()
    area_df = area_df.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')
    peak_idx = area_df.groupby(['mouse_id', 'neuron_id'])['cross_corr'].apply(lambda x: x.abs().idxmax())
    peak_df = area_df.loc[peak_idx, ['mouse_id', 'neuron_id', 'significant_fdr']]
    sig_neurons = peak_df[peak_df['significant_fdr']]
    area_df = area_df.merge(sig_neurons[['mouse_id', 'neuron_id']], on=['mouse_id', 'neuron_id'], how='inner')
    ccg_summary = area_df.groupby(['reward_group', 'lag'])['cross_corr'].agg(['mean', 'sem']).reset_index()
    reward_color_map = {'R+': 'forestgreen', 'R-': 'crimson'}

    group_stats = area_df.groupby('reward_group').agg(
        n_mice=('mouse_id', 'nunique'),
        n_neurons=('neuron_id', 'nunique')
    ).reset_index()


    

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for group in ccg_summary['reward_group'].unique():
        group_data = ccg_summary[ccg_summary['reward_group'] == group]
        color = reward_color_map.get(group, 'gray')
        
        stats = group_stats[group_stats['reward_group'] == group].iloc[0]  # <-- ADDED
        n_mice = int(stats['n_mice'])
        n_neurons = int(stats['n_neurons'])

        ax.plot(group_data['lag'], group_data['mean'], color=color, linewidth=2.5, 
            label=f"{group} (n={n_mice} mice, {n_neurons} neurons)")
        ax.fill_between(
            group_data['lag'],
            group_data['mean'] - group_data['sem'],
            group_data['mean'] + group_data['sem'],
            color=color,
            alpha=0.3
        )

    y_max = max(abs(ccg_summary['mean'] + ccg_summary['sem']).max(),abs(ccg_summary['mean'] - ccg_summary['sem']).max())
    y_max = y_max * 1.05
    ax.set_ylim(-y_max, y_max)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_xticks(np.arange(-10, 11, 1))
    ax.set_ylabel('Mean Cross-Correlation', fontsize=12)
    ax.set_title(f'Mean CCG of Significant Neurons in {area_to_plot} by Reward Group', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(title='Reward Group')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    safe_area = area_to_plot.replace('/', '_').replace('\\', '_')
    filename = f"mean_ccg_{safe_area}_all_mice_significant"
    save_figure_multiple_formats(fig, os.path.join(output_dir, filename))


def main():    
    print("\nLoading mouse info...")
    mouse_info_path = os.path.join("/Volumes", "Petersen-Lab", "z_LSENS", "Share", "Dana_Shayakhmetova", "mouse_info", "joint_mouse_reference_weight.xlsx")
    mouse_info_df = pd.read_excel(mouse_info_path)
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df = mouse_info_df[
        (mouse_info_df['exclude'] == 0) &
        (mouse_info_df['reward_group'].isin(['R+', 'R-'])) &
        (mouse_info_df['recording'] == 1)
    ]
    print("Done.")
    

    print("\nLoading behavioral info...")
    bhv_path = os.path.join(DATA_PATH, 'behavior_data', 'merged_bhv_trials.pkl')
    bhv_data = pd.read_pickle(bhv_path)
    bhv_data = bhv_data[bhv_data.whisker_stim == 1]
    excluded_mice = ['AB068', 'AB164', 'AB142', 'AB107', 'AB149']
    bhv_data = bhv_data[~bhv_data.mouse_id.isin(excluded_mice)]
    bhv_shuffled_data = pd.read_pickle(os.path.join(DATA_PATH, 'behavior_data', 'shuffled_bhv_data.pkl'))
    bhv_shuffled_data = bhv_shuffled_data[~bhv_shuffled_data.mouse_id.isin(excluded_mice)]
    print(f"Done.")


    interesting_areas = ['SSs', 'SSp-bfd', 'CP', 'MOs', 'CA1', 'SCm', 'AI','SCs', 'MOp', 'CA2', 'CA3', 'CL', 'MRN']
    min_neurons_threshold = 10
    print(f"Minimum neuron threshold: {min_neurons_threshold}")


    print("PROCESSING BASELINE DATA")
    baseline_perf_corr = pd.read_pickle(os.path.join(BASELINE_DATA_DIR, 'baseline_perf_cross_corr_light.pkl'))
    baseline_hr_corr = pd.read_pickle(os.path.join(BASELINE_DATA_DIR, 'baseline_hr_cross_corr_light.pkl'))
    baseline_fa_corr = pd.read_pickle(os.path.join(BASELINE_DATA_DIR, 'baseline_fa_cross_corr_light.pkl'))
    

    #PERF
    print("\nStarting baseline performance.")
    print("Calculating mean significant proportions by reward group...")
    mean_group_perf = calculate_mean_sig_proportion_per_reward_group(
        baseline_perf_corr, mouse_info_df)
    plot_mean_sig_proportion_by_reward_group(
        mean_group_perf, 
        os.path.join(LAG_PLOTS_BASELINE, 'performance', 'across_mice', 'mean_sig_prop_by_area_reward'))
    
    print("Creating grid view of mean CCGs by area...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_grid(
            baseline_perf_corr, bhv_data, bhv_shuffled_data, 'performance',
            mouse_info_df, area, 
            os.path.join(LAG_PLOTS_BASELINE, 'performance', 'across_mice','mean_ccg_grid_view'))
    
    print("Plotting mean CCG across all neurons by reward group...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_all_neurons_by_reward_group(
            baseline_perf_corr, mouse_info_df, area,
            os.path.join(LAG_PLOTS_BASELINE, 'performance', 'across_mice', 'mean_ccg_by_reward'),
            min_neurons=min_neurons_threshold)
    
    print("Plotting CCGs separated by correlation sign...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_by_corr_sign(
            baseline_perf_corr, mouse_info_df, area,
            os.path.join(LAG_PLOTS_BASELINE, 'performance', 'across_mice', 'mean_ccg_by_corr_sign'))
    
    print("Plotting CCGs for significant neurons only...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_by_peak_significance(
            baseline_perf_corr, mouse_info_df, area,
            os.path.join(LAG_PLOTS_BASELINE, 'performance', 'across_mice', 'mean_significant_neurons'))


    #HR
    print("\nStarting baseline hit rate.")
    print("Calculating mean significant proportions by reward group...")
    mean_group_hr = calculate_mean_sig_proportion_per_reward_group(baseline_hr_corr, mouse_info_df)
    plot_mean_sig_proportion_by_reward_group(
        mean_group_hr,
        os.path.join(LAG_PLOTS_BASELINE, 'hit_rate', 'across_mice', 'mean_sig_prop_by_area_reward'))
    
    print("Creating grid view of mean CCGs by area...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_grid(
            baseline_hr_corr, bhv_data, bhv_shuffled_data, 'p_mean',
            mouse_info_df, area,
            os.path.join(LAG_PLOTS_BASELINE, 'hit_rate', 'across_mice', 'mean_ccg_grid_view'))
    
    print("Plotting mean CCG across all neurons by reward group...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_all_neurons_by_reward_group(
            baseline_hr_corr, mouse_info_df, area,
            os.path.join(LAG_PLOTS_BASELINE, 'hit_rate', 'across_mice', 'mean_ccg_by_reward'),
            min_neurons=min_neurons_threshold)
    
    print("Plotting CCGs separated by correlation sign...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_by_corr_sign(
            baseline_hr_corr, mouse_info_df, area,
            os.path.join(LAG_PLOTS_BASELINE, 'hit_rate', 'mean_ccg_by_corr_sign'))
    
    print("Plotting CCGs for significant neurons only...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_by_peak_significance(
            baseline_hr_corr, mouse_info_df, area,
            os.path.join(LAG_PLOTS_BASELINE, 'hit_rate', 'across_mice', 'mean_significant_neurons'))
    
    #FA
    print("\nStarting baseline false alarm.")
    print("Calculating mean significant proportions by reward group...")
    mean_group_fa = calculate_mean_sig_proportion_per_reward_group(
        baseline_fa_corr, mouse_info_df)
    plot_mean_sig_proportion_by_reward_group(
        mean_group_fa,
        os.path.join(LAG_PLOTS_BASELINE, 'false_alarm', 'across_mice', 'mean_sig_prop_by_area_reward'))
        
    print("Creating grid view of mean CCGs by area...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_grid(
            baseline_fa_corr, bhv_data, bhv_shuffled_data, 'p_chance',
            mouse_info_df, area,
            os.path.join(LAG_PLOTS_BASELINE, 'false_alarm', 'across_mice', 'mean_ccg_grid_view'))
    
    print("Plotting mean CCG across all neurons by reward group...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_all_neurons_by_reward_group(
            baseline_fa_corr, mouse_info_df, area,
            os.path.join(LAG_PLOTS_BASELINE, 'false_alarm', 'across_mice', 'mean_ccg_by_reward'),
            min_neurons=min_neurons_threshold)
    
    print("Plotting CCGs separated by correlation sign...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_by_corr_sign(
            baseline_fa_corr, mouse_info_df, area,
            os.path.join(LAG_PLOTS_BASELINE, 'false_alarm', 'across_mice', 'mean_ccg_by_corr_sign'))
    
    print("Plotting CCGs for significant neurons only...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_by_peak_significance(
            baseline_fa_corr, mouse_info_df, area,
            os.path.join(LAG_PLOTS_BASELINE, 'false_alarm', 'across_mice', 'mean_significant_neurons'))
    




    print("PROCESSING EVOKED RESPONSE DATA")
    evoked_perf_corr = pd.read_pickle(os.path.join(EVOKED_RESPONSE_DATA_DIR,'evoked_response_perf_cross_corr_light.pkl'))
    evoked_hr_corr = pd.read_pickle(os.path.join(EVOKED_RESPONSE_DATA_DIR,'evoked_response_hr_cross_corr_light.pkl'))
    evoked_fa_corr = pd.read_pickle(os.path.join(EVOKED_RESPONSE_DATA_DIR,'evoked_response_fa_cross_corr_light.pkl'))
    

    #PERF
    print("\nStarting evoked response performance.")    
    print("Calculating mean significant proportions by reward group...")
    mean_group_evoked_perf = calculate_mean_sig_proportion_per_reward_group(evoked_perf_corr, mouse_info_df)
    plot_mean_sig_proportion_by_reward_group(
        mean_group_evoked_perf,
        os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'performance', 'across_mice', 'mean_sig_prop_by_area_reward'))
        
    print("Creating grid view of mean CCGs by area...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_grid(
            evoked_perf_corr, bhv_data, bhv_shuffled_data, 'performance',
            mouse_info_df, area,
            os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'performance', 'across_mice', 'mean_ccg_grid_view'))
    
    print("Plotting mean CCG across all neurons by reward group...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_all_neurons_by_reward_group(
            evoked_perf_corr, mouse_info_df, area,
            os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'performance', 'across_mice', 'mean_ccg_by_reward'),
            min_neurons=min_neurons_threshold)
    
    print("Plotting CCGs separated by correlation sign...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_by_corr_sign(
            evoked_perf_corr, mouse_info_df, area,
            os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'performance', 'across_mice', 'mean_ccg_by_corr_sign'))
    
    print("Plotting CCGs for significant neurons only...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_by_peak_significance(
            evoked_perf_corr, mouse_info_df, area,
            os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'performance', 'across_mice', 'mean_significant_neurons'))
    

    #HR
    print("\nStarting evoked response hit rate.")
    print("Calculating mean significant proportions by reward group...")
    mean_group_evoked_hr = calculate_mean_sig_proportion_per_reward_group(evoked_hr_corr, mouse_info_df)
    plot_mean_sig_proportion_by_reward_group(
        mean_group_evoked_hr,
        os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'hit_rate', 'across_mice', 'mean_sig_prop_by_area_reward'))


    print("Creating grid view of mean CCGs by area...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_grid(
            evoked_hr_corr, bhv_data, bhv_shuffled_data, 'p_mean',
            mouse_info_df, area,
            os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'hit_rate', 'across_mice','mean_ccg_grid_view'))
    
    print("Plotting mean CCG across all neurons by reward group...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_all_neurons_by_reward_group(
            evoked_hr_corr, mouse_info_df, area,
            os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'hit_rate', 'across_mice', 'mean_ccg_by_reward'),
            min_neurons=min_neurons_threshold)
    
    print("Plotting CCGs separated by correlation sign...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_by_corr_sign(
            evoked_hr_corr, mouse_info_df, area,
            os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'hit_rate', 'across_mice', 'mean_ccg_by_corr_sign'))
    
    print("Plotting CCGs for significant neurons only...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_by_peak_significance(
            evoked_hr_corr, mouse_info_df, area,
            os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'hit_rate', 'across_mice', 'mean_significant_neurons'))
    

    #FA
    print("\nStarting evoked respononse false alarm.")
    print("Calculating mean significant proportions by reward group...")
    mean_group_evoked_fa = calculate_mean_sig_proportion_per_reward_group(evoked_fa_corr, mouse_info_df)
    plot_mean_sig_proportion_by_reward_group(
        mean_group_evoked_fa,
        os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'false_alarm', 'across_mice', 'mean_sig_prop_by_area_reward'))
        
    print("Creating grid view of mean CCGs by area...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_grid(
            evoked_fa_corr, bhv_data, bhv_shuffled_data, 'p_chance',
            mouse_info_df, area,
            os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'false_alarm', 'across_mice', 'mean_ccg_grid_view'))
    
    print("Plotting mean CCG across all neurons by reward group...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_all_neurons_by_reward_group(
            evoked_fa_corr, mouse_info_df, area,
            os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'false_alarm', 'across_mice', 'mean_ccg_by_reward'),
            min_neurons=min_neurons_threshold)
    
    print("Plotting CCGs separated by correlation sign...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_by_corr_sign(
            evoked_fa_corr, mouse_info_df, area,
            os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'false_alarm', 'across_mice','mean_ccg_by_corr_sign'))
    
    print("Plotting CCGs for significant neurons only...")
    for area in interesting_areas:
        print(f"   Processing {area}...")
        plot_mean_ccg_by_peak_significance(
            evoked_fa_corr, mouse_info_df, area,
            os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'false_alarm', 'across_mice', 'mean_significant_neurons'))
    print("Done Script.")


if __name__ == '__main__':
    main()