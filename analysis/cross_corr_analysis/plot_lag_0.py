import os
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from helper import save_figure_multiple_formats, parse_array


DATA_PATH = os.path.join('/Volumes', 'Petersen-Lab', 'z_LSENS', 'Share', 'Dana_Shayakhmetova', 'dynamic_analysis_dec16')
BASELINE_DATA_PATH = os.path.join(DATA_PATH, 'baseline_analysis', 'cross_corr_data')
EVOKED_DATA_PATH = os.path.join(DATA_PATH, 'evoked_response_analysis', 'cross_corr_data')
OUTPUT_DIR_BASELINE = os.path.join(DATA_PATH, 'baseline_analysis', 'lag_0_plots')
OUTPUT_DIR_EVOKED_RESPONSE = os.path.join(DATA_PATH, 'evoked_response_analysis', 'lag_0_plots')


def plot_behavior_and_neural_trialwise(behavioral_df, neural_data, cross_corr_df, bhv_shuffled_df, mouse_info_df, output_dir, neural_metric='rate_evoked_response', show_bhv_shuffled=True):
    """Generate trial by trial plots for individual neurons showing behavioral dynamics (HR and FA), task performance (HR minus FA), and neural activity with lag 0 cross correlation statistics. 
    Separate three panel figures are created for each neuron, brain area, and mouse.
    """
    os.makedirs(output_dir, exist_ok=True)

    mice = neural_data['mouse_id'].unique()
    brain_areas = neural_data["area_custom"].unique()
    lag0_corr_df = cross_corr_df[cross_corr_df['lag'] == 0].copy()

    for mouse in mice:
        behav_mouse = behavioral_df[behavioral_df.mouse_id == mouse]
        mouse_lag0_corr = lag0_corr_df[lag0_corr_df.mouse_id == mouse]

        if behav_mouse.empty:
            print(f"Skipping {mouse}: no behavioral data.")
            continue

        reward_group = mouse_info_df.loc[mouse_info_df['mouse_id'] == mouse, 'reward_group'].iloc[0]
        mouse_df = neural_data[neural_data['mouse_id'] == mouse]
        bhv_shuffled_mouse = bhv_shuffled_df[bhv_shuffled_df.mouse_id == mouse]
        for area in brain_areas:
            area_df = mouse_df[mouse_df["area_custom"] == area]
            area_lag0_corr = mouse_lag0_corr[mouse_lag0_corr.area_custom == area]

            if area_df.empty:
                continue

            neuron_ids = area_df['neuron_id'].unique()
            save_dir = os.path.join(output_dir, f"{mouse}", f"{area}")
            os.makedirs(save_dir, exist_ok=True)

            for neuron in neuron_ids:
                neuron_df = area_df[area_df['neuron_id'] == neuron].copy()
                neuron_lag0_corr = area_lag0_corr[area_lag0_corr.neuron_id == neuron]

                # Lag 0 correlation
                if len(neuron_lag0_corr) == 0:
                    corr, p_val = 'N/A', 'N/A'
                else:
                    corr_row = neuron_lag0_corr.iloc[0]
                    corr, p_val = corr_row['cross_corr'], corr_row['p_value_corrected']

                # Set up figure
                fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
                fig.suptitle(
                    f"Mouse {mouse} ({reward_group}) - Area {area} Neuron {neuron}",
                    fontsize=16, fontweight='bold', ha='center', y=1.03
                )
                ax_behav, ax_disc, ax_neural = axes

                bhv_trials = behav_mouse['whisker_trial_id'].values
                hit_rate = behav_mouse['p_mean'].values
                fa_rate = behav_mouse['p_chance'].values
                disc_rate = hit_rate - fa_rate
                x = bhv_trials
                hr_colour = 'forestgreen' if reward_group == 'R+' else 'crimson'

                # Plot behavior
                ax_behav.plot(x, hit_rate, label='Hit Rate', color=hr_colour, linewidth=2)
                ax_behav.plot(x, fa_rate, label='False Alarm Rate', color='black', linewidth=2)
                ax_behav.set_ylim(0, 1)
                ax_behav.set_xlim(0, x.max())
                ax_behav.set_ylabel('Response Rate', fontsize=12)
                ax_behav.set_title("Behavioral Dynamics", fontsize=13)
                ax_behav.legend(fontsize=9)
                ax_behav.grid(alpha=0.3)
                ax_behav.tick_params(labelbottom=False)

                # Behavioral shuffled
                if show_bhv_shuffled and not bhv_shuffled_mouse.empty:
                    hr_matrix = np.array(bhv_shuffled_mouse["p_mean_shuffled"].values.tolist())
                    fa_matrix = np.array(bhv_shuffled_mouse["p_chance_shuffled"].values.tolist())
                    mean_hr = np.nanmean(hr_matrix, axis=1)
                    sem_hr = np.nanstd(hr_matrix, axis=1, ddof=1) / np.sqrt(hr_matrix.shape[1])
                    ax_behav.fill_between(x, mean_hr - sem_hr, mean_hr + sem_hr, color=hr_colour, alpha=0.25, label='Shuffled HR')
                    mean_fa = np.nanmean(fa_matrix, axis=1)
                    sem_fa = np.nanstd(fa_matrix, axis=1, ddof=1) / np.sqrt(fa_matrix.shape[1])
                    ax_behav.fill_between(x, mean_fa - sem_fa, mean_fa + sem_fa, color='gray', alpha=0.25, label='Shuffled FA') 

                # HR minus FA
                ax_disc.plot(x, disc_rate, label='HR minus FA', color='black', linewidth=2)
                ax_disc.set_ylim(-1, 1)
                ax_disc.set_xlim(0, x.max())
                ax_disc.set_ylabel('Response Rate', fontsize=12)
                ax_disc.grid(alpha=0.3)
                ax_disc.set_title("HR minus FA", fontsize=13)
                ax_disc.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax_disc.tick_params(labelbottom=False)

                if show_bhv_shuffled and not bhv_shuffled_mouse.empty:
                    perf_matrix = hr_matrix - fa_matrix
                    mean_perf = np.nanmean(perf_matrix, axis=1)
                    sem_perf = np.nanstd(perf_matrix, axis=1, ddof=1) / np.sqrt(perf_matrix.shape[1])
                    ax_disc.fill_between(x, mean_perf - sem_perf, mean_perf + sem_perf, color='black', alpha=0.25, label='Shuffled (HR-FA)')

                # Neuron trace
                y = np.array(neuron_df[neural_metric].iloc[0])
                
                # Check significance from cross_corr data
                if len(neuron_lag0_corr) > 0 and 'significant_fdr' in neuron_lag0_corr.columns:
                    is_significant = neuron_lag0_corr.iloc[0]['significant_fdr']
                    neuron_color = 'red' if is_significant else 'steelblue'
                else:
                    neuron_color = 'steelblue'
                
                if isinstance(corr, (int, float)):
                    neuron_label = f"Neuron (Cross corr={corr:.3f}, p={p_val:.3f})"
                else:
                    neuron_label = f"Neuron (Cross corr={corr}, p={p_val})"
                
                ax_neural.plot(x, y, color=neuron_color, lw=1.8, alpha=0.8, label=neuron_label)
                if neural_metric == 'pre_firing_rate':
                    ax_neural.set_ylabel('Pre-stimulus Firing Rate', fontsize=12)
                else:
                    ax_neural.set_ylabel('Evoked Response', fontsize=12)
                max_y = max(abs(y.min()), abs(y.max())) if len(y) > 0 else 1
                ax_neural.set_ylim(-max_y, max_y)
                ax_neural.set_xlim(0, x.max())
                ax_neural.set_title(f"Neuronal Dynamics for {area}: Neuron {neuron}", fontsize=13)
                ax_neural.grid(True, linestyle=':', alpha=0.3)
                ax_neural.set_xlabel("Trial ID", fontsize=12)
                ax_neural.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax_neural.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax_neural.xaxis.set_major_locator(MultipleLocator(10))
                ax_neural.legend(fontsize=9)

                plt.tight_layout()
                safe_area = area.replace('/', '_').replace('\\', '_')
                filename_base = f"{mouse}_{safe_area}_neuron{neuron}"
                filepath_base = os.path.join(save_dir, filename_base)
                save_figure_multiple_formats(fig, filepath_base, dpi=300)

    print(f"\nDone.")

def plot_behavior_and_neural_trialwise_by_area(behavioral_df, neural_data, bhv_shuffled_df, mouse_info_df, output_dir, neural_metric='rate_evoked_response', show_bhv_shuffled=True):
    """Generate trial by trial plots that aggregate all neurons within a given brain area for each mouse, showing behavioral dynamics (HR and FA),
    task performance (HR minus FA), and overlaid neural activity traces from all neurons in that area.
    """
    os.makedirs(output_dir, exist_ok=True)
    mice = neural_data['mouse_id'].unique()
    brain_areas = neural_data['area_custom'].unique()

    for mouse in mice:
        behav_mouse = behavioral_df[behavioral_df.mouse_id == mouse]
        if behav_mouse.empty:
            print(f"Skipping {mouse}: no behavioral data")
            continue

        reward_group = mouse_info_df.loc[
            mouse_info_df['mouse_id'] == mouse, 
            'reward_group'
        ].iloc[0]

        mouse_df = neural_data[neural_data['mouse_id'] == mouse]
        bhv_shuffled_mouse = bhv_shuffled_df[bhv_shuffled_df.mouse_id == mouse]

        for area in brain_areas:
            area_df = mouse_df[mouse_df["area_custom"] == area]
            if area_df.empty:
                continue

            neuron_ids = area_df['neuron_id'].unique()

            save_dir = os.path.join(output_dir, f"{mouse}", f"{area}")
            os.makedirs(save_dir, exist_ok=True)

            # create one figure per area
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            fig.suptitle(
                f"Mouse {mouse} ({reward_group}) - Area {area}",
                fontsize=16, fontweight='bold', ha='center', y=1.03
            )

            ax_behav, ax_disc, ax_neural = axes

            # Behavioral panel
            bhv_trials = behav_mouse['whisker_trial_id'].values
            hit_rate = behav_mouse['p_mean'].values
            fa_rate = behav_mouse['p_chance'].values
            disc_rate = hit_rate - fa_rate
            x = bhv_trials

            hr_colour = 'forestgreen' if reward_group == 'R+' else 'crimson'

            ax_behav.plot(x, hit_rate, label='Hit Rate', color=hr_colour, linewidth=2)
            ax_behav.plot(x, fa_rate, label='False Alarm Rate', color='black', linewidth=2)
            ax_behav.set_ylim(0, 1)
            ax_behav.set_xlim(0, x.max())
            ax_behav.set_ylabel('Response Rate', fontsize=12)
            ax_behav.set_title("Behavioral Dynamics", fontsize=13)
            ax_behav.legend(fontsize=9)
            ax_behav.grid(True, linestyle=':', alpha=0.3)
            ax_behav.tick_params(labelbottom=False)

            # Shuffled behavior
            if show_bhv_shuffled and not bhv_shuffled_mouse.empty:
                hr_matrix = np.array(bhv_shuffled_mouse["p_mean_shuffled"].values.tolist())
                fa_matrix = np.array(bhv_shuffled_mouse["p_chance_shuffled"].values.tolist())
                mean_hr = np.nanmean(hr_matrix, axis=1)
                sem_hr = np.nanstd(hr_matrix, axis=1, ddof=1) / np.sqrt(hr_matrix.shape[1])
                ax_behav.fill_between(x, mean_hr - sem_hr, mean_hr + sem_hr, color=hr_colour, alpha=0.25, label='Shuffled HR')
                mean_fa = np.nanmean(fa_matrix, axis=1)
                sem_fa = np.nanstd(fa_matrix, axis=1, ddof=1) / np.sqrt(fa_matrix.shape[1])
                ax_behav.fill_between(x, mean_fa - sem_fa, mean_fa + sem_fa, color='gray', alpha=0.25, label='Shuffled FA')   

            ax_disc.plot(x, disc_rate, label='HR minus FA', color='black', linewidth=2)
            ax_disc.set_ylim(-1, 1)
            ax_disc.set_xlim(0, x.max())
            ax_disc.set_ylabel('Response Rate', fontsize=12)
            ax_disc.grid(True, linestyle=':', alpha=0.3)
            ax_disc.set_title("HR minus FA", fontsize=13)
            ax_disc.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax_disc.tick_params(labelbottom=False)

            if show_bhv_shuffled and not bhv_shuffled_mouse.empty:
                perf_matrix = hr_matrix - fa_matrix
                mean_perf = np.nanmean(perf_matrix, axis=1)
                sem_perf = np.nanstd(perf_matrix, axis=1, ddof=1) / np.sqrt(perf_matrix.shape[1])
                ax_disc.fill_between(x, mean_perf - sem_perf, mean_perf + sem_perf, color='black', alpha=0.25)

            # Neural panel: all neurons in area
            for neuron in neuron_ids:
                neuron_df = area_df[area_df['neuron_id'] == neuron]
                y = np.array(neuron_df[neural_metric].iloc[0])
                ax_neural.plot(x, y, lw=1.8, alpha=0.8, label=f'Neuron {neuron}')

            if neural_metric == 'pre_firing_rate':
                ax_neural.set_ylabel('Pre-stimulus Firing Rate', fontsize=12)
            else:
                ax_neural.set_ylabel('Evoked Response', fontsize=12)
            
            # Get max across all neurons
            all_y = [np.array(area_df[area_df['neuron_id'] == n][neural_metric].iloc[0]) for n in neuron_ids]
            max_y = max([max(abs(y.min()), abs(y.max())) for y in all_y if len(y) > 0])
            ax_neural.set_ylim(-max_y, max_y)
            ax_neural.set_xlim(0, x.max())
            ax_neural.set_title(f"Neuronal Dynamics for {area}", fontsize=13)
            ax_neural.grid(True, linestyle=':', alpha=0.3)
            ax_neural.set_xlabel("Trial ID", fontsize=12)
            ax_neural.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax_neural.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_neural.xaxis.set_major_locator(MultipleLocator(10))
            # ax_neural.legend(fontsize=8, ncol=2, loc='upper right')

            plt.tight_layout()

            # save figure
            safe_area = area.replace('/', '_').replace('\\', '_')
            filename_base = f"{mouse}_{safe_area}_all_neurons"
            filepath_base = os.path.join(save_dir, filename_base)
            save_figure_multiple_formats(fig, filepath_base, dpi=300)

    print("\nDone")


def main():
    print("\nLoading behavioral data...")
    bhv_path = os.path.join(DATA_PATH, 'behavior_data', 'merged_bhv_trials.pkl')
    bhv_data = pd.read_pickle(bhv_path)
    bhv_data = bhv_data[bhv_data.whisker_stim == 1]
    bhv_data = bhv_data[~bhv_data.mouse_id.isin(['AB068','AB164', 'AB142', 'AB107', 'AB149'])]
    bhv_shuffled_data = pd.read_pickle(os.path.join(DATA_PATH, 'behavior_data', 'shuffled_bhv_data.pkl'))
    bhv_shuffled_data = bhv_shuffled_data[~bhv_shuffled_data.mouse_id.isin(['AB068','AB164', 'AB142', 'AB107', 'AB149'])]
    print("done.")

    print("\nLoading neural data...")
    neural_path = os.path.join(DATA_PATH, 'neuronal_data','neuronal_data_mh_ab.csv')
    neural_data = pd.read_csv(neural_path)
    neural_data = neural_data[~neural_data.mouse_id.isin(['AB068','AB164', 'AB142', 'AB107', 'AB149'])]
    array_cols = ["pre_firing_rate", "post_firing_rate", "rate_evoked_response"]
    for col in array_cols:
        neural_data[col] = neural_data[col].apply(parse_array)
    print("done.")

    print("\nGrabbing mouse info...")
    mouse_info_path = os.path.join("/Volumes", "Petersen-Lab", "z_LSENS", "Share", "Dana_Shayakhmetova", "mouse_info", "joint_mouse_reference_weight.xlsx")
    mouse_info_df = pd.read_excel(mouse_info_path)
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df = mouse_info_df[
        (mouse_info_df['exclude'] == 0) &
        (mouse_info_df['reward_group'].isin(['R+', 'R-'])) &
        (mouse_info_df['recording'] == 1)]
    print("Done.")




    # baseline
    print("\nPulling baseline info...")
    baseline_perf_path = os.path.join(BASELINE_DATA_PATH, 'baseline_perf_cross_corr_light.pkl')
    baseline_hr_path = os.path.join(BASELINE_DATA_PATH, 'baseline_hr_cross_corr_light.pkl')
    baseline_fa_path = os.path.join(BASELINE_DATA_PATH, 'baseline_fa_cross_corr_light.pkl')
    baseline_perf_corr = pd.read_pickle(baseline_perf_path)
    baseline_hr_corr = pd.read_pickle(baseline_hr_path)
    baseline_fa_corr = pd.read_pickle(baseline_fa_path)
    print("Done.")

    interesting_mice = ['MH022', 'AB085', 'AB162', 'AB131']

    #filter for intersting mice 
    bhv_filtered = bhv_data[bhv_data.mouse_id.isin(interesting_mice)]
    neural_filtered = neural_data[neural_data.mouse_id.isin(interesting_mice)]
    bhv_shuffled_filtered = bhv_shuffled_data[bhv_shuffled_data.mouse_id.isin(interesting_mice)]
    baseline_perf_corr_filtered = baseline_perf_corr[baseline_perf_corr.mouse_id.isin(interesting_mice)]
    baseline_hr_corr_filtered = baseline_hr_corr[baseline_hr_corr.mouse_id.isin(interesting_mice)]
    baseline_fa_corr_filtered = baseline_fa_corr[baseline_fa_corr.mouse_id.isin(interesting_mice)]

    print("\nPlotting singular neurons (baseline and perf)...")
    plot_behavior_and_neural_trialwise(
        bhv_filtered, 
        neural_filtered, 
        baseline_perf_corr_filtered, 
        bhv_shuffled_filtered, 
        mouse_info_df, 
        os.path.join(OUTPUT_DIR_BASELINE, 'perf_corr_plots'), 
        neural_metric='pre_firing_rate', 
        show_bhv_shuffled=True)
    print("\nDone.")

    print("\nPlotting singular neurons (baseline and hr)...")
    plot_behavior_and_neural_trialwise(
        bhv_filtered, 
        neural_filtered, 
        baseline_hr_corr_filtered, 
        bhv_shuffled_filtered, 
        mouse_info_df, 
        os.path.join(OUTPUT_DIR_BASELINE, 'hr_corr_plots'), 
        neural_metric='pre_firing_rate', 
        show_bhv_shuffled=True)
    print("\nDone.")

    print("\nPlotting singular neurons (baseline and fa)...")
    plot_behavior_and_neural_trialwise(
        bhv_filtered, 
        neural_filtered, 
        baseline_fa_corr_filtered, 
        bhv_shuffled_filtered, 
        mouse_info_df, 
        os.path.join(OUTPUT_DIR_BASELINE, 'fa_corr_plots'), 
        neural_metric='pre_firing_rate', 
        show_bhv_shuffled=True)
    print("\nDone.")

    print("\nStarting by area plots...")
    plot_behavior_and_neural_trialwise_by_area(
        bhv_filtered, 
        neural_filtered, 
        bhv_shuffled_filtered, 
        mouse_info_df, 
        os.path.join(OUTPUT_DIR_BASELINE, 'all_neurons_plotted'), 
        neural_metric='pre_firing_rate', 
        show_bhv_shuffled=True)
    print("Done.")









    #evoked response 
    print("\nPulling evoked response info...")
    evoked_perf_path = os.path.join(EVOKED_DATA_PATH, 'evoked_response_perf_cross_corr_light.pkl')
    evoked_hr_path = os.path.join(EVOKED_DATA_PATH, 'evoked_response_hr_cross_corr_light.pkl')
    evoked_fa_path = os.path.join(EVOKED_DATA_PATH, 'evoked_response_fa_cross_corr_light.pkl')
    evoked_perf_corr = pd.read_pickle(evoked_perf_path)
    evoked_hr_corr = pd.read_pickle(evoked_hr_path)
    evoked_fa_corr = pd.read_pickle(evoked_fa_path)
    print("Done.")

    #filter for interesting mice 
    evoked_perf_corr_filtered = evoked_perf_corr[evoked_perf_corr.mouse_id.isin(interesting_mice)]
    evoked_hr_corr_filtered = evoked_hr_corr[evoked_hr_corr.mouse_id.isin(interesting_mice)]
    evoked_fa_corr_filtered = evoked_fa_corr[evoked_fa_corr.mouse_id.isin(interesting_mice)]

    print("\nPlotting singular neurons (evoked response and perf)...")
    plot_behavior_and_neural_trialwise(
        bhv_filtered, 
        neural_filtered, 
        evoked_perf_corr_filtered, 
        bhv_shuffled_filtered, 
        mouse_info_df, 
        os.path.join(OUTPUT_DIR_EVOKED_RESPONSE, 'perf_corr_plots'), 
        neural_metric='rate_evoked_response', 
        show_bhv_shuffled=True)
    print("\nDone.")

    print("\nPlotting singular neurons (evoked response and hr)...")
    plot_behavior_and_neural_trialwise(
        bhv_filtered, 
        neural_filtered, 
        evoked_hr_corr_filtered, 
        bhv_shuffled_filtered, 
        mouse_info_df, 
        os.path.join(OUTPUT_DIR_EVOKED_RESPONSE, 'hr_corr_plots'), 
        neural_metric='rate_evoked_response', 
        show_bhv_shuffled=True)
    print("\nDone.")

    print("\nPlotting singular neurons (evoked response and fa)...")
    plot_behavior_and_neural_trialwise(
        bhv_filtered, 
        neural_filtered, 
        evoked_fa_corr_filtered, 
        bhv_shuffled_filtered, 
        mouse_info_df, 
        os.path.join(OUTPUT_DIR_EVOKED_RESPONSE, 'fa_corr_plots'), 
        neural_metric='rate_evoked_response', 
        show_bhv_shuffled=True)
    print("\nDone.")

    print("\nStarting by area plots...")
    plot_behavior_and_neural_trialwise_by_area(
        bhv_filtered, 
        neural_filtered, 
        bhv_shuffled_filtered, 
        mouse_info_df, 
        os.path.join(OUTPUT_DIR_EVOKED_RESPONSE, 'all_neurons_plotted'), 
        neural_metric='rate_evoked_response', 
        show_bhv_shuffled=True)
    print("\nDone.")


    print("\n\nScript done.")




if __name__ == "__main__":
    main()


