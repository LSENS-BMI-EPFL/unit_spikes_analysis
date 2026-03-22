import os
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
from helper import save_figure_multiple_formats, parse_array


DATA_PATH = os.path.join('/Volumes', 'Petersen-Lab', 'z_LSENS', 'Share', 'Dana_Shayakhmetova', 'dynamic_analysis_dec16')
BASELINE_DATA_DIR = os.path.join(DATA_PATH, 'baseline_analysis', 'cross_corr_data')
LAG_PLOTS_BASELINE = os.path.join(DATA_PATH, 'baseline_analysis', 'all_lags_plots')
EVOKED_RESPONSE_DATA_DIR = os.path.join(DATA_PATH, 'evoked_response_analysis', 'cross_corr_data')
LAG_PLOTS_EVOKED_RESPONSE = os.path.join(DATA_PATH, 'evoked_response_analysis', 'all_lags_plots')


# Cross corrs
def plot_single_neuron_correlogram_with_sem(corr_neuronal_data, output_path=None):
    """Plot individual cross-correlograms with shuffled baseline for each neuron.
    Shows correlation with FDR-significant lags highlighted in red."""

    mice = corr_neuronal_data.mouse_id.unique()
    lags = np.arange(-10, 11)

    for mouse in mice:
        mouse_df = corr_neuronal_data[corr_neuronal_data.mouse_id == mouse]
        areas = mouse_df.area_custom.unique()
        
        for area in areas:
            area_df = mouse_df[mouse_df.area_custom == area]
            neurons = area_df.neuron_id.unique()
            
            for unit in neurons:
                unit_df = area_df[area_df.neuron_id == unit]
                if unit_df.empty:
                    print(f"No data found for {mouse}/{area}/{unit}")
                    continue

                corrs = unit_df['cross_corr'].values
                significant = unit_df['significant_fdr'].values
                
                #N_lags x N_shuffles
                shuffles_matrix = np.stack(unit_df['shuffled_cross_corrs'].values)
                shuffle_mean = np.mean(shuffles_matrix, axis=1)
                shuffle_std = np.std(shuffles_matrix, axis=1)
                upper_bound = shuffle_mean + shuffle_std 
                floor_bound = shuffle_mean - shuffle_std
                
                colors = ['red' if sig else 'steelblue' for sig in significant]
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.fill_between(
                    unit_df['lag'], 
                    floor_bound, 
                    upper_bound, 
                    color='gray', 
                    alpha=0.2, 
                    label='Shuffled Mean ± SD'
                )
                ax.plot(unit_df['lag'], shuffle_mean, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Shuffle Mean')
                
                # true cross corr
                ax.plot(unit_df['lag'], corrs, linewidth=2, color='steelblue', alpha=0.7)
                ax.scatter(unit_df['lag'], corrs, color=colors, s=80, edgecolor='black', zorder=3)
                ax.axhline(0, color='black', linewidth=1.5)
                ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
                ax.set_ylim(-1, 1)
                ax.set_xlim(-10.5, 10.5)
                ax.set_xlabel('Lag', fontsize=13)
                ax.set_ylabel('Cross Correlation', fontsize=13)
                ax.set_title(f'Neuron {unit} in {area} of {mouse} ', fontsize=15, fontweight='bold')
                ax.set_xticks(range(min(lags), max(lags) + 1))
                ax.grid(True, alpha=0.3, axis='y')
                
                # legend
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                           markersize=8, label='Significant', markeredgecolor='black'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', 
                           markersize=8, label='Not Significant', markeredgecolor='black'),
                    Line2D([0], [0], color='gray', linewidth=10, alpha=0.2, label='Shuffled Mean ± SD')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
                
                plt.tight_layout()
                
                if output_path:
                    # output_path/mouse/area/
                    save_dir = os.path.join(output_path, mouse, area)
                    os.makedirs(save_dir, exist_ok=True)
                    filename_base = f"neuron_{unit}_cross_corr_with_sem"
                    save_figure_multiple_formats(fig, os.path.join(save_dir, filename_base), dpi=300)
                else:
                    plt.show()

def plot_area_correlogram(corr_neuronal_data, output_path=None):
    """Plot all neuron correlograms overlaid for each brain area within each mouse."""

    mice = corr_neuronal_data.mouse_id.unique()
    lags = np.arange(-10, 11)

    for mouse in mice:
        mouse_df = corr_neuronal_data[corr_neuronal_data.mouse_id == mouse]
        areas = mouse_df.area_custom.unique()

        for area in areas:
            area_df = mouse_df[mouse_df.area_custom == area]
            neurons = area_df.neuron_id.unique()

            if len(neurons) == 0:
                print(f"No neurons found for {mouse} {area}")
                continue

            fig, ax = plt.subplots(figsize=(12, 6))

            for unit in neurons:
                unit_df = area_df[area_df.neuron_id == unit]
                if unit_df.empty:
                    continue
                corrs = unit_df["cross_corr"].values
                ax.plot(unit_df["lag"], corrs, linewidth=1.6, alpha=0.6, label=f'Neuron {unit}')

            ax.axhline(0, color='black', linewidth=1.3)
            ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
            ax.set_ylim(-1, 1)
            ax.set_xlim(-10.5, 10.5)
            ax.set_xlabel("Lag ", fontsize=13)
            ax.set_ylabel("Cross Correlation", fontsize=13)
            ax.set_title(f"All Neurons in {area} of {mouse} ", fontsize=15, fontweight='bold')
            ax.set_xticks(range(min(lags), max(lags) + 1))
            ax.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()

            if output_path:
                # output_path/mouse/
                save_dir = os.path.join(output_path, mouse)
                os.makedirs(save_dir, exist_ok=True)
                safe_area = area.replace('/', '_').replace('\\', '_')
                filename_base = f"{safe_area}_all_neurons"
                save_figure_multiple_formats(fig, os.path.join(save_dir, filename_base), dpi=300)
            else:
                plt.show()


#6 pannel
def plot_single_area_summary(df, mouse_info_df, mouse_id, area, save_dir, figsize=(21, 12)):
    """ Generates 6-panel summary figure for one mouse and one brain area.
    
    Top row: (1) mean CCG all neurons, (2) proportion significant by lag, (3) peak lag distribution

    Bottom row: same three plots but only for neurons with FDR-significant peaks.

    Color-coded by reward group (R+ = green, R- = red).

    Note: Row 1, Plot 2 slightly redundant.
    """

    reward_group = mouse_info_df[mouse_info_df.mouse_id==mouse_id].iloc[0].reward_group
    if reward_group=='R+':
        plot_color = 'forestgreen'
    elif reward_group =='R-':
        plot_color = 'crimson'
    else:
        plot_color = 'gray'

    area_df = df[(df['mouse_id'] == mouse_id) & (df['area_custom'] == area)].copy()

    #not all mice have the same areas recorded
    if area_df.empty:
        print(f"No data for mouse {mouse_id}, area {area}")
        return

    num_neurons = area_df['neuron_id'].nunique()

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    fig.suptitle(f"           {mouse_id}, Area: {area} (n={num_neurons} neurons) \n", fontsize=16)

    #Plot 1
    ax1 = axes[0][0]
    ccg_summary = area_df.groupby('lag')['cross_corr'].agg(['mean', 'sem']).reset_index()
    ax1.plot(ccg_summary['lag'], ccg_summary['mean'], color=plot_color, linewidth=2)
    ax1.fill_between(
        ccg_summary['lag'],
        ccg_summary['mean'] - ccg_summary['sem'],
        ccg_summary['mean'] + ccg_summary['sem'],
        color=plot_color,
        alpha=0.2,
        label='SEM'
    )
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_title(f'Mean Cross-Correlogram \nfor All Neurons')
    ax1.set_xlabel('Lag')
    ax1.set_xticks(np.arange(-10,11, 1))
    max_value_for_y = max(abs(ccg_summary['mean'].min()-ccg_summary['sem'].max()), abs(ccg_summary['mean'].max()+ccg_summary['sem'].max()))
    if np.isnan(max_value_for_y) or np.isinf(max_value_for_y):
        print(f"Skipping plotting for {mouse_id}, area {area} because max_value_for_y is invalid")
    else:
        ax1.set_ylim(-max_value_for_y * 1.1, max_value_for_y * 1.1)
    ax1.set_ylabel('Mean Cross-Correlation')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend()

    # Plot 2
    ax2 = axes[0][1]
    total_neurons = area_df['neuron_id'].nunique()
    sig_df = area_df[area_df['significant_fdr'] == True]
    sig_prop = pd.DataFrame({'lag': range(-10, 11)})

    if not sig_df.empty:
        pos_counts = (
            sig_df[sig_df['cross_corr'] > 0]
            .groupby('lag')['neuron_id']
            .nunique()
        )
        neg_counts = (
            sig_df[sig_df['cross_corr'] < 0]
            .groupby('lag')['neuron_id']
            .nunique()
        )

        pos_prop = (pos_counts / total_neurons).rename('pos_prop')
        neg_prop = -(neg_counts / total_neurons).rename('neg_prop')

        sig_prop = sig_prop.merge(pos_prop, on='lag', how='left')
        sig_prop = sig_prop.merge(neg_prop, on='lag', how='left')
        sig_prop = sig_prop.fillna(0)

    else:
        sig_prop['pos_prop'] = 0
        sig_prop['neg_prop'] = 0


    ax2.bar(sig_prop['lag'], sig_prop['pos_prop'],color='skyblue', edgecolor='black', label='Positive significant')
    ax2.bar(sig_prop['lag'], sig_prop['neg_prop'],color='lightcoral', edgecolor='black', label='Negative significant')
    ax2.set_title(f'Proportion of Significant Cross-Correlations at Each Lag \n for All Neurons')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Proportion')
    max_val = max(sig_prop['pos_prop'].max(), abs(sig_prop['neg_prop'].min()))
    ax2.set_ylim(-1.1 * max_val, 1.1 * max_val)
    ax2.set_xticks(np.arange(-10, 11, 1))
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend()



    # Plot 3
    ax3 = axes[0][2]
    all_neurons = area_df['neuron_id'].unique()

    if len(all_neurons) > 0:
        peak_idx = area_df.groupby('neuron_id')['cross_corr'].apply(lambda x: x.abs().idxmax())
        peak_df = area_df.loc[peak_idx, ['neuron_id', 'lag', 'cross_corr']]
        pos_counts = (
            peak_df[peak_df['cross_corr'] > 0]
            .groupby('lag')['neuron_id']
            .nunique()
            .reindex(range(-10, 11), fill_value=0)
        )
        neg_counts = (
            peak_df[peak_df['cross_corr'] < 0]
            .groupby('lag')['neuron_id']
            .nunique()
            .reindex(range(-10, 11), fill_value=0)
        )
        neg_counts = -neg_counts
        ax3.bar(pos_counts.index, pos_counts.values, color='skyblue', edgecolor='black', label='Positive')
        ax3.bar(neg_counts.index, neg_counts.values, color='lightcoral', edgecolor='black', label='Negative')

        ax3.set_title("Peak Lag Distribution for All Neurons \n")
    else:
        ax3.set_title("Peak Lag Distribution for All Neurons \n")
        ax3.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=20, alpha=0.5)

    ax3.set_ylabel('Number of Neurons')
    ax3.set_xlabel('Lag')
    ax3.set_xticks(np.arange(-10, 11, 1))

    max_val = max(pos_counts.max(), abs(neg_counts.min()))
    ax3.set_ylim(-1.1 * max_val, 1.1 * max_val)

    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend()


    #row two 
    #Plot 4
    ax4 = axes[1][0]
    peak_idx = area_df.groupby('neuron_id')['cross_corr'].apply(lambda x: x.abs().idxmax())
    peak_df = area_df.loc[peak_idx, ['neuron_id', 'cross_corr', 'significant_fdr']]
    sig_neurons = peak_df[peak_df['significant_fdr'] == True]['neuron_id'].unique()
    sig_area_df = area_df[area_df['neuron_id'].isin(sig_neurons)]

    if not sig_area_df.empty:
        ccg_summary = sig_area_df.groupby('lag')['cross_corr'].agg(['mean', 'sem']).reset_index()
        num_neurons_plot4 = len(sig_neurons)

        ax4.plot(ccg_summary['lag'], ccg_summary['mean'], color=plot_color, linewidth=2)
        ax4.fill_between(
            ccg_summary['lag'],
            ccg_summary['mean'] - ccg_summary['sem'],
            ccg_summary['mean'] + ccg_summary['sem'],
            color=plot_color,
            alpha=0.2,
            label='SEM'
        )
        ax4.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax4.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        max_value_for_y = max(abs(ccg_summary['mean'].min() - ccg_summary['sem'].max()), abs(ccg_summary['mean'].max() + ccg_summary['sem'].max()))
        if np.isnan(max_value_for_y) or np.isinf(max_value_for_y):
            print(f"Skipping plotting for {mouse_id}, area {area} because max_value_for_y is invalid")
        else:
            ax4.set_ylim(max_value_for_y * -1.1, max_value_for_y * 1.1)

        ax4.set_title(f'Mean Cross-Correlogram \n for Neurons with Sig. Peaks (n={num_neurons_plot4})')
        ax4.legend()
    else:
        ax4.set_title('Mean Cross-Correlogram \n for Neurons with Sig. Peaks (n=0)')
        ax4.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=20, alpha=0.5)
        ax4.set_ylim(-1, 1)

    ax4.set_xlabel('Lag')
    ax4.set_xticks(np.arange(-10, 11, 1))
    ax4.set_ylabel('Mean Cross-Correlation')
    ax4.grid(True, linestyle='--', alpha=0.3)

    



    # Plot 5
    ax5 = axes[1][1]
    peak_idx = area_df.groupby('neuron_id')['cross_corr'].apply(lambda x: x.abs().idxmax())
    peak_df = area_df.loc[peak_idx, ['neuron_id', 'cross_corr', 'significant_fdr']]
    sig_neurons = peak_df[peak_df['significant_fdr'] == True]['neuron_id'].unique()
    sig_lag_df = area_df[(area_df['neuron_id'].isin(sig_neurons)) & (area_df['significant_fdr'] == True)]
    sig_prop = pd.DataFrame({'lag': range(-10, 11)})
    total_sig_neurons = len(sig_neurons)

    if total_sig_neurons > 0:
        pos_counts = sig_lag_df[sig_lag_df['cross_corr'] > 0].groupby('lag')['neuron_id'].nunique()
        neg_counts = sig_lag_df[sig_lag_df['cross_corr'] < 0].groupby('lag')['neuron_id'].nunique()

        pos_prop = (pos_counts / total_sig_neurons).rename('pos_prop')
        neg_prop = -(neg_counts / total_sig_neurons).rename('neg_prop')

        sig_prop = sig_prop.merge(pos_prop, on='lag', how='left')
        sig_prop = sig_prop.merge(neg_prop, on='lag', how='left')
        sig_prop = sig_prop.fillna(0)
    else:
        sig_prop['pos_prop'] = 0
        sig_prop['neg_prop'] = 0

    ax5.bar(sig_prop['lag'], sig_prop['pos_prop'], color='skyblue', edgecolor='black', label='Positive Significant')
    ax5.bar(sig_prop['lag'], sig_prop['neg_prop'], color='lightcoral', edgecolor='black', label='Negative Significant')
    ax5.set_title(f'Proportion of Significant Cross-Correlations at Each Lag \n for Neurons with Sig. Peaks (n={total_sig_neurons})')
    ax5.set_xlabel('Lag')
    ax5.set_ylabel('Proportion')
    max_val = max(sig_prop['pos_prop'].max(), abs(sig_prop['neg_prop'].min()))
    ax5.set_ylim(-1.1 * max_val, 1.1 * max_val)
    ax5.set_xticks(np.arange(-10, 11, 1))
    ax5.grid(True, linestyle='--', alpha=0.3)
    ax5.legend()




    # Plot 6
    ax6 = axes[1][2]
    sig_neurons = area_df[area_df['significant_fdr'] == True]['neuron_id'].unique()
    if len(sig_neurons) > 0:
        peak_idx = (
            area_df[area_df['neuron_id'].isin(sig_neurons)]
            .groupby('neuron_id')['cross_corr']
            .apply(lambda x: x.abs().idxmax())
        )
        peak_df = area_df.loc[peak_idx, ['neuron_id', 'lag', 'cross_corr', 'significant_fdr']]
        peak_sig_df = peak_df[peak_df['significant_fdr'] == True]

        # positive and negative counts
        pos_counts = (
            peak_sig_df[peak_sig_df['cross_corr'] > 0]
            .groupby('lag')['neuron_id']
            .nunique()
            .reindex(range(-10, 11), fill_value=0)
        )

        neg_counts = (
            peak_sig_df[peak_sig_df['cross_corr'] < 0]
            .groupby('lag')['neuron_id']
            .nunique()
            .reindex(range(-10, 11), fill_value=0)
        )

        neg_counts = -neg_counts
        ax6.bar(pos_counts.index, pos_counts.values, color='skyblue', edgecolor='black', label='Positive Significant')
        ax6.bar(neg_counts.index, neg_counts.values,color='lightcoral', edgecolor='black', label='Negative Significant')
        ax6.set_title(f"Significant Peak Lag Distribution (n={len(peak_sig_df)})\n")
    else:
        ax6.set_title("No Significant Neurons Found")
        ax6.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=20, alpha=0.5)

    ax6.set_ylabel('Number of Neurons')
    ax6.yaxis.set_major_locator(MaxNLocator(integer=True))

    max_val = max(pos_counts.max(), abs(neg_counts.min()))
    ax6.set_ylim(-1.1 * max_val, 1.1 * max_val)

    ax6.set_xlabel('Lag')
    ax6.grid(True, linestyle='--', alpha=0.3)
    ax6.set_xticks(np.arange(-10, 11, 1))
    ax6.legend()
    fname_base = f"{mouse_id}_{area.replace('-', '_')}_summary"
    save_path = os.path.join(save_dir, fname_base)
    save_figure_multiple_formats(fig, save_path)

def plot_all_mice_areas_summary(df, mouse_info_df, output_dir, interesting_areas=True):
    """Generate 6-panel summary figures for all mice and areas.
    If interesting_areas=True, only plots predefined areas of interest.
    Otherwise plots all areas present in each mouse's data.
    """

    mice = df['mouse_id'].unique()

    for mouse in mice:

        if interesting_areas:
            areas = ['SSs', 'SSp-bfd', 'CP', 'MOs', 'CA1', 'SCm', 'AI', 'SCs', 'MOp', 'CA2', 'CA3', 'CL', 'MRN']
        else:
            mouse_df = df[df['mouse_id'] == mouse]
            areas = mouse_df['area_custom'].unique()


        mouse_dir = os.path.join(output_dir, mouse)
        os.makedirs(mouse_dir, exist_ok=True)


        for area in areas:
            print(f"Plotting mouse {mouse}, area {area}")
            plot_single_area_summary(
                df=df,
                mouse_info_df =mouse_info_df,
                mouse_id=mouse,
                area=area,
                save_dir=mouse_dir
            )





# Newly added functions from meeting 16/12/2025
# Means per mouse and per area
#     1. General Mean CCG 
#     2. Mean CCG separating anti-correlated vs correlated neurons
#          - This is determined by the sign of the peak correlation 
#     3. Mean CCG of only significant neurons
#          - FDR corrected p-value at its peak


def plot_mean_ccg_per_area(df, mouse_info_df, output_dir):
    """Plot mean cross-correlogram across all neurons for each mouse and brain area."""
    mice = df['mouse_id'].unique()
    
    for mouse in mice:
        mouse_df = df[df['mouse_id'] == mouse]
        areas = mouse_df['area_custom'].unique()
        reward_group = mouse_info_df[mouse_info_df.mouse_id == mouse].iloc[0].reward_group
        if reward_group == 'R+':
            plot_color = 'forestgreen'
        elif reward_group == 'R-':
            plot_color = 'crimson'
        else:
            plot_color = 'gray'
        
        for area in areas:
            area_df = mouse_df[mouse_df.area_custom == area]
            
            if area_df.empty:
                continue
            
            num_neurons = area_df['neuron_id'].nunique()
            ccg_summary = area_df.groupby('lag')['cross_corr'].agg(['mean', 'sem']).reset_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(ccg_summary['lag'], ccg_summary['mean'], color=plot_color, linewidth=2.5)
            ax.fill_between(
                ccg_summary['lag'],
                ccg_summary['mean'] - ccg_summary['sem'],
                ccg_summary['mean'] + ccg_summary['sem'],
                color=plot_color,
                alpha=0.2,
                label='SEM'
            )
            
            ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
            ax.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
            ax.set_title(f'Mean CCG of {mouse} in {area} (n={num_neurons} neurons)', fontsize=14, fontweight='bold')
            max_value_for_y = max(abs(ccg_summary['mean'].min() - ccg_summary['sem'].max()), abs(ccg_summary['mean'].max() + ccg_summary['sem'].max()))
            if np.isnan(max_value_for_y) or np.isinf(max_value_for_y):
                print(f"Skipping y-axis limits for {mouse}, area {area} because max_value_for_y is invalid")
            else:
                ax.set_ylim(-max_value_for_y * 1.1, max_value_for_y * 1.1)
            ax.set_xlabel('Lag', fontsize=12)
            ax.set_ylabel('Mean Cross-Correlation', fontsize=12)
            ax.set_xticks(np.arange(-10, 11, 2))
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(fontsize=10)
            plt.tight_layout()
            mouse_dir = os.path.join(output_dir,'mean_ccg_all_neurons', mouse)
            os.makedirs(mouse_dir, exist_ok=True)
            safe_area = area.replace('/', '_').replace('\\', '_')
            fname_base = f"{mouse}_{safe_area}_mean_ccg_all_neurons"
            save_figure_multiple_formats(fig, os.path.join(mouse_dir, fname_base))

def plot_mean_ccg_by_correlation_sign(df, mouse_info_df, output_dir):
    """ Plot mean CCG separated by correlation sign for each mouse and area.
    Creates side-by-side plots for positively vs negatively correlated neurons.
    Sign determined by peak correlation value."""

    mice = df['mouse_id'].unique()
    for mouse in mice:
        mouse_df = df[df['mouse_id'] == mouse]
        areas = mouse_df['area_custom'].unique()
        
        for area in areas:
            area_df = mouse_df[mouse_df.area_custom == area]
            
            if area_df.empty:
                continue
            
            # peak corr 
            peak_idx = area_df.groupby('neuron_id')['cross_corr'].apply(lambda x: x.abs().idxmax())
            peak_df = area_df.loc[peak_idx, ['neuron_id', 'cross_corr']]
            pos_neurons = peak_df[peak_df['cross_corr'] > 0]['neuron_id'].values
            neg_neurons = peak_df[peak_df['cross_corr'] < 0]['neuron_id'].values

            # two subplots 
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            max_y = 0
            if len(pos_neurons) > 0:
                pos_df = area_df[area_df['neuron_id'].isin(pos_neurons)]
                ccg_pos = pos_df.groupby('lag')['cross_corr'].agg(['mean', 'sem']).reset_index()
                max_y = max(
                max_y,
                abs(ccg_pos['mean'].min() - ccg_pos['sem'].max()),
                abs(ccg_pos['mean'].max() + ccg_pos['sem'].max())
                )

                
                axes[0].plot(ccg_pos['lag'], ccg_pos['mean'], color='steelblue', linewidth=2.5)
                axes[0].fill_between(
                    ccg_pos['lag'],
                    ccg_pos['mean'] - ccg_pos['sem'],
                    ccg_pos['mean'] + ccg_pos['sem'],
                    color='steelblue',
                    alpha=0.2,
                    label='SEM'
                )
            
            axes[0].axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
            axes[0].axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
            axes[0].set_title(f'Positively Correlated (n={len(pos_neurons)})', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Lag', fontsize=11)
            axes[0].set_ylabel('Mean Cross-Correlation', fontsize=11)
            axes[0].set_xticks(np.arange(-10, 11, 2))
            axes[0].grid(True, linestyle='--', alpha=0.3)
            axes[0].legend()
            
            if len(neg_neurons) > 0:
                neg_df = area_df[area_df['neuron_id'].isin(neg_neurons)]
                ccg_neg = neg_df.groupby('lag')['cross_corr'].agg(['mean', 'sem']).reset_index()
                max_y = max(max_y,abs(ccg_neg['mean'].min() - ccg_neg['sem'].max()),abs(ccg_neg['mean'].max() + ccg_neg['sem'].max()))
                axes[1].plot(ccg_neg['lag'], ccg_neg['mean'], color='coral', linewidth=2.5)
                axes[1].fill_between(
                    ccg_neg['lag'],
                    ccg_neg['mean'] - ccg_neg['sem'],
                    ccg_neg['mean'] + ccg_neg['sem'],
                    color='coral',
                    alpha=0.2,
                    label='SEM'
                )
            
            axes[1].axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
            axes[1].axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
            axes[1].set_title(f'Negatively Correlated (n={len(neg_neurons)})', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Lag', fontsize=11)
            axes[1].set_ylabel('Mean Cross-Correlation', fontsize=11)
            axes[1].set_xticks(np.arange(-10, 11, 2))
            axes[1].grid(True, linestyle='--', alpha=0.3)
            axes[1].legend()
            if np.isnan(max_y) or np.isinf(max_y):
                print(f"Skipping y-axis limits for {mouse}, area {area} because max_value_for_y is invalid")
            else:
                y_lim = max_y * 1.1
                axes[0].set_ylim(-y_lim, y_lim)
                axes[1].set_ylim(-y_lim, y_lim)


            fig.suptitle(f'Mean CCG by Correlation Sign \n{mouse}, {area} ', fontsize=12, fontweight='bold')
            plt.tight_layout()
            mouse_dir = os.path.join(output_dir, 'mean_ccg_by_sign', mouse)
            os.makedirs(mouse_dir, exist_ok=True)
            safe_area = area.replace('/', '_').replace('\\', '_')
            fname_base = f"{mouse}_{safe_area}_mean_ccg_by_sign"
            save_figure_multiple_formats(fig, os.path.join(mouse_dir, fname_base))

def plot_mean_ccg_by_significance(df, mouse_info_df, output_dir):
    """Plot mean CCG using only neurons with FDR-significant peak correlations."""

    mice = df['mouse_id'].unique()
    
    for mouse in mice:
        mouse_df = df[df['mouse_id'] == mouse]
        areas = mouse_df['area_custom'].unique()
        reward_group = mouse_info_df[mouse_info_df.mouse_id == mouse].iloc[0].reward_group
        if reward_group == 'R+':
            plot_color = 'forestgreen'
        elif reward_group == 'R-':
            plot_color = 'crimson'
        else:
            plot_color = 'gray'
        
        for area in areas:
            area_df = mouse_df[mouse_df.area_custom == area]
            
            if area_df.empty:
                continue
            
            # peak
            peak_idx = area_df.groupby('neuron_id')['cross_corr'].apply(lambda x: x.abs().idxmax())
            peak_df = area_df.loc[peak_idx, ['neuron_id', 'cross_corr', 'significant_fdr']]
            sig_neurons = peak_df[peak_df['significant_fdr'] == True]['neuron_id'].values
            
            #plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            if len(sig_neurons) > 0:
                sig_df = area_df[area_df['neuron_id'].isin(sig_neurons)]
                ccg_sig = sig_df.groupby('lag')['cross_corr'].agg(['mean', 'sem']).reset_index()
                
                ax.plot(ccg_sig['lag'], ccg_sig['mean'], color=plot_color, linewidth=2.5)
                ax.fill_between(
                    ccg_sig['lag'],
                    ccg_sig['mean'] - ccg_sig['sem'],
                    ccg_sig['mean'] + ccg_sig['sem'],
                    color=plot_color,
                    alpha=0.2,
                    label='SEM'
                )
            ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
            ax.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
            max_value_for_y = max(abs(ccg_sig['mean'].min() - ccg_sig['sem'].max()), abs(ccg_sig['mean'].max() + ccg_sig['sem'].max()))
            if np.isnan(max_value_for_y) or np.isinf(max_value_for_y):
                print(f"Skipping y-axis limits for {mouse}, area {area} because max_value_for_y is invalid")
            else:
                ax.set_ylim(-max_value_for_y * 1.1, max_value_for_y * 1.1)

            ax.set_title(f'Mean Cross-Correlation of Neurons with Significant Peak (FDR) \n {mouse}, {area}\nn = {len(sig_neurons)} neurons', fontsize=12, fontweight='bold')
            ax.set_xlabel('Lag', fontsize=11)
            ax.set_ylabel('Mean Cross-Correlation', fontsize=11)
            ax.set_xticks(np.arange(-10, 11, 2))
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend()
            plt.tight_layout()
            mouse_dir = os.path.join(output_dir, 'mean_ccg_significant_only', mouse)
            os.makedirs(mouse_dir, exist_ok=True)
            safe_area = area.replace('/', '_').replace('\\', '_')
            fname_base = f"{mouse}_{safe_area}_mean_ccg_significant_neurons_only"
            save_figure_multiple_formats(fig, os.path.join(mouse_dir, fname_base))



def main():    
    print("\nLoading mouse info...")
    mouse_info_path = os.path.join("/Volumes", "Petersen-Lab", "z_LSENS", "Share",  "Dana_Shayakhmetova", "mouse_info",  "joint_mouse_reference_weight.xlsx")
    mouse_info_df = pd.read_excel(mouse_info_path)
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df = mouse_info_df[
        (mouse_info_df['exclude'] == 0) &
        (mouse_info_df['reward_group'].isin(['R+', 'R-'])) &
        (mouse_info_df['recording'] == 1)
    ]
    print("Done.")

    #manually defined in plot_all_mice_areas_summary()
    # interesting_areas = ['SSs', 'SSp-bfd', 'CP', 'MOs', 'CA1', 'SCm', 'AI', 'SCs', 'MOp', 'CA2', 'CA3', 'CL', 'MRN']

    print("PROCESSING BASELINE DATA")    
    baseline_perf_corr = pd.read_pickle(os.path.join(BASELINE_DATA_DIR, 'baseline_perf_cross_corr.pkl'))
    baseline_hr_corr = pd.read_pickle(os.path.join(BASELINE_DATA_DIR, 'baseline_hr_cross_corr.pkl'))
    baseline_fa_corr = pd.read_pickle(os.path.join(BASELINE_DATA_DIR, 'baseline_fa_cross_corr.pkl'))

    interesting_mice = ['MH022', 'AB085', 'AB162', 'AB131']
    baseline_perf_corr_filtered = baseline_perf_corr[baseline_perf_corr['mouse_id'].isin(interesting_mice)]
    baseline_hr_corr_filtered = baseline_hr_corr[baseline_hr_corr['mouse_id'].isin(interesting_mice)]
    baseline_fa_corr_filtered = baseline_fa_corr[baseline_fa_corr['mouse_id'].isin(interesting_mice)]


    # PERF
    print("\nStarting baseline performance.")
    print("Plotting individual neuron correlograms (baseline + performance)...")
    plot_single_neuron_correlogram_with_sem(baseline_perf_corr_filtered, os.path.join(LAG_PLOTS_BASELINE, 'performance', 'indv_mice'))
    
    print("Plotting area correlograms (baseline + performance)...")
    plot_area_correlogram(baseline_perf_corr_filtered, os.path.join(LAG_PLOTS_BASELINE, 'performance', 'indv_mice'))

    print("Generating 3-panel summary (mean ccg, prop sig, peak lag dist for one mouse and area)...")
    plot_all_mice_areas_summary(baseline_perf_corr, mouse_info_df, os.path.join(LAG_PLOTS_BASELINE, 'performance', 'indv_mice','three_panel_summaries'))

    print("Plotting mean CCG per area...")
    plot_mean_ccg_per_area(baseline_perf_corr, mouse_info_df, os.path.join(LAG_PLOTS_BASELINE, 'performance', 'indv_mice'))
    
    print("Plotting mean CCG by correlation sign...")
    plot_mean_ccg_by_correlation_sign(baseline_perf_corr, mouse_info_df, os.path.join(LAG_PLOTS_BASELINE, 'performance', 'indv_mice'))
    
    print("Plotting mean CCG for significant neurons only...")
    plot_mean_ccg_by_significance(baseline_perf_corr, mouse_info_df, os.path.join(LAG_PLOTS_BASELINE, 'performance', 'indv_mice'))

    # HIT RATE 
    print("\nStarting baseline hit rate.")
    print("Plotting individual neuron correlograms (baseline + hit rate)...")
    plot_single_neuron_correlogram_with_sem(baseline_hr_corr_filtered, os.path.join(LAG_PLOTS_BASELINE, 'hit_rate', 'indv_mice'))
    
    print("Plotting area correlograms (baseline + hit rate)...")
    plot_area_correlogram(baseline_hr_corr_filtered, os.path.join(LAG_PLOTS_BASELINE, 'hit_rate'))

    print("Generating 3-panel summary (mean ccg, prop sig, peak lag dist for one mouse and area)...")
    plot_all_mice_areas_summary(baseline_hr_corr, mouse_info_df, os.path.join(LAG_PLOTS_BASELINE, 'hit_rate', 'indv_mice', 'three_panel_summaries'))

    print("Plotting mean CCG per area...")
    plot_mean_ccg_per_area(baseline_hr_corr, mouse_info_df, os.path.join(LAG_PLOTS_BASELINE, 'hit_rate', 'indv_mice'))
    
    print("Plotting mean CCG by correlation sign...")
    plot_mean_ccg_by_correlation_sign(baseline_hr_corr, mouse_info_df, os.path.join(LAG_PLOTS_BASELINE, 'hit_rate', 'indv_mice'))
    
    print("Plotting mean CCG for significant neurons only...")
    plot_mean_ccg_by_significance(baseline_hr_corr, mouse_info_df, os.path.join(LAG_PLOTS_BASELINE, 'hit_rate', 'indv_mice'))

    # FALSE ALARM
    print("\nStarting baseline false alarm.")
    print("Plotting individual neuron correlograms (baseline + false alarm rate)...")
    plot_single_neuron_correlogram_with_sem(baseline_fa_corr_filtered, os.path.join(LAG_PLOTS_BASELINE, 'false_alarm', 'indv_mice'))
    
    print("Plotting area correlograms (baseline + false alarm)...")
    plot_area_correlogram(baseline_fa_corr_filtered, os.path.join(LAG_PLOTS_BASELINE, 'false_alarm', 'indv_mice'))

    print("Generating 3-panel summary (mean ccg, prop sig, peak lag dist for one mouse and area)...")
    plot_all_mice_areas_summary(baseline_fa_corr, mouse_info_df, os.path.join(LAG_PLOTS_BASELINE, 'false_alarm', 'indv_mice', 'three_panel_summaries'))

    print("Plotting mean CCG per area...")
    plot_mean_ccg_per_area(baseline_fa_corr, mouse_info_df, os.path.join(LAG_PLOTS_BASELINE, 'false_alarm', 'indv_mice'))
    
    print("Plotting mean CCG by correlation sign...")
    plot_mean_ccg_by_correlation_sign(baseline_fa_corr, mouse_info_df, os.path.join(LAG_PLOTS_BASELINE, 'false_alarm', 'indv_mice'))
    
    print("Plotting mean CCG for significant neurons only...")
    plot_mean_ccg_by_significance(baseline_fa_corr, mouse_info_df, os.path.join(LAG_PLOTS_BASELINE, 'false_alarm', 'indv_mice'))




    print("PROCESSING EVOKED RESPONSE DATA")
    evoked_perf_corr = pd.read_pickle(os.path.join(EVOKED_RESPONSE_DATA_DIR, 'evoked_response_perf_cross_corr.pkl'))
    evoked_hr_corr = pd.read_pickle(os.path.join(EVOKED_RESPONSE_DATA_DIR, 'evoked_response_hr_cross_corr.pkl'))
    evoked_fa_corr = pd.read_pickle(os.path.join(EVOKED_RESPONSE_DATA_DIR, 'evoked_response_fa_cross_corr.pkl'))


    evoked_perf_corr_filtered = evoked_perf_corr[evoked_perf_corr['mouse_id'].isin(interesting_mice)]
    evoked_hr_corr_filtered = evoked_hr_corr[evoked_hr_corr['mouse_id'].isin(interesting_mice)]
    evoked_fa_corr_filtered = evoked_fa_corr[evoked_fa_corr['mouse_id'].isin(interesting_mice)]
    

    # PERF
    print("\nStarting evoked response performance.")
    print("Plotting individual neuron correlograms (evoked + performance)...")
    plot_single_neuron_correlogram_with_sem(evoked_perf_corr_filtered, os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'performance', 'indv_mice'))
    
    print("Plotting area correlograms (evoked + performance)...")
    plot_area_correlogram(evoked_perf_corr_filtered, os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'performance', 'indv_mice'))

    print("Generating 3-panel summary (mean ccg, prop sig, peak lag dist for one mouse and area)...")
    plot_all_mice_areas_summary(evoked_perf_corr, mouse_info_df, os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'performance', 'indv_mice', 'three_panel_summaries'))

    print("Plotting mean CCG per area...")
    plot_mean_ccg_per_area(evoked_perf_corr, mouse_info_df, os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'performance', 'indv_mice'))
    
    print("Plotting mean CCG by correlation sign...")
    plot_mean_ccg_by_correlation_sign(evoked_perf_corr, mouse_info_df, os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'performance', 'indv_mice'))
    
    print("Plotting mean CCG for significant neurons only...")
    plot_mean_ccg_by_significance(evoked_perf_corr, mouse_info_df, os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'performance', 'indv_mice'))

    # HIT RATE 
    print("\nStarting evoked response hit rate.")
    print("Plotting individual neuron correlograms (evoked + hit rate)...")
    plot_single_neuron_correlogram_with_sem(evoked_hr_corr_filtered, os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'hit_rate', 'indv_mice'))
    
    print("Plotting area correlograms (evoked + hit rate)...")
    plot_area_correlogram(evoked_hr_corr_filtered, os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'hit_rate'))

    print("Generating 3-panel summary (mean ccg, prop sig, peak lag dist for one mouse and area)...")
    plot_all_mice_areas_summary(evoked_hr_corr, mouse_info_df, os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'hit_rate', 'indv_mice','three_panel_summaries'))

    print("Plotting mean CCG per area...")
    plot_mean_ccg_per_area(evoked_hr_corr, mouse_info_df, os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'hit_rate', 'indv_mice'))
    
    print("Plotting mean CCG by correlation sign...")
    plot_mean_ccg_by_correlation_sign(evoked_hr_corr, mouse_info_df, os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'hit_rate', 'indv_mice'))
    
    print("Plotting mean CCG for significant neurons only...")
    plot_mean_ccg_by_significance(evoked_hr_corr, mouse_info_df, os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'hit_rate', 'indv_mice'))

    # FALSE ALARM 
    print("\nStarting evoked response false alarm.")
    print("Plotting individual neuron correlograms (evoked + false alarm)...")
    plot_single_neuron_correlogram_with_sem(evoked_fa_corr_filtered, os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'false_alarm', 'indv_mice'))
    
    print("Plotting area correlograms (evoked + false alarm)...")
    plot_area_correlogram(evoked_fa_corr_filtered, os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'false_alarm', 'indv_mice'))

    print("Generating 3-panel summary (mean ccg, prop sig, peak lag dist for one mouse and area)...")
    plot_all_mice_areas_summary(evoked_fa_corr, mouse_info_df, os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'false_alarm', 'indv_mice','three_panel_summaries'))

    print("Plotting mean CCG per area...")
    plot_mean_ccg_per_area(evoked_fa_corr, mouse_info_df, os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'false_alarm', 'indv_mice'))
    
    print("Plotting mean CCG by correlation sign...")
    plot_mean_ccg_by_correlation_sign(evoked_fa_corr, mouse_info_df, os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'false_alarm', 'indv_mice'))
    
    print("Plotting mean CCG for significant neurons only...")
    plot_mean_ccg_by_significance(evoked_fa_corr, mouse_info_df, os.path.join(LAG_PLOTS_EVOKED_RESPONSE, 'false_alarm', 'indv_mice'))

    print("\n\nScript done.")


if __name__ == '__main__':
    main()