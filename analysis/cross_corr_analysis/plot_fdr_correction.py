import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
from helper import save_figure_multiple_formats, parse_array


DATA_PATH = os.path.join('/Volumes', 'Petersen-Lab', 'z_LSENS', 'Share', 'Dana_Shayakhmetova', 'dynamic_analysis_dec16')
BASELINE_DATA_PATH = os.path.join(DATA_PATH, 'baseline_analysis', 'cross_corr_data')
EVOKED_DATA_PATH = os.path.join(DATA_PATH, 'evoked_response_analysis', 'cross_corr_data')
OUTPUT_DIR_BASELINE = os.path.join(DATA_PATH, 'baseline_analysis', 'before_after_fdr')
OUTPUT_DIR_EVOKED_RESPONSE = os.path.join(DATA_PATH, 'evoked_response_analysis', 'before_after_fdr')
os.makedirs(OUTPUT_DIR_BASELINE, exist_ok=True)
os.makedirs(OUTPUT_DIR_EVOKED_RESPONSE, exist_ok=True)


def plot_raw_count_significant_peaks_by_area(df,mouse_info_df,title,output_dir, interesting_areas=True):
    """ Generate grouped bar plots comparing significant peak counts before (light color) and after FDR correction (dark color).
    Plots created per brain area by reward group."""

    peak_df = df.merge(mouse_info_df[['mouse_id', 'reward_group']],on='mouse_id',how='left')
    colors = {'R+': ('forestgreen', 'darkgreen'),'R-': ('crimson', 'darkred')}

    if interesting_areas:  
        areas = ['SSs', 'SSp-bfd', 'CP', 'MOs', 'CA1', 'SCm', 'AI','SCs', 'MOp', 'CA2', 'CA3', 'CL', 'MRN']
        title = title + "_specific_areas"
    else:
        areas = df.area_custom.unique()
        title = title + "_all_areas"


    x = np.arange(len(areas))
    width = 0.4

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 8),
        sharex=True
    )

    for ax, rg in zip(axes, ['R+', 'R-']):
        group_df = peak_df[peak_df['reward_group'] == rg]
        counts_before = []
        counts_after = []

        for area in areas:
            area_df = group_df[group_df['area_custom'] == area]
            counts_before.append(area_df['significant'].sum())
            counts_after.append(area_df['significant_fdr'].sum())

        ax.bar(
            x - width / 2,
            counts_before,
            width=width,
            color=colors[rg][0],
            label='Before FDR'
        )

        ax.bar(
            x + width / 2,
            counts_after,
            width=width,
            color=colors[rg][1],
            label='After FDR'
        )

        ax.set_ylabel('Number of significant peaks')
        ax.set_title(f'Reward group {rg}',fontsize=13,fontweight='bold')
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.legend(loc='upper right')
        ax.set_xticks(x)
        ax.set_xticklabels(areas, rotation=45, ha='right')
        ax.tick_params(axis='x', labelbottom=True)

    fig.suptitle('Raw count of neurons with significant peaks per area',fontsize=15,fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    filebase = os.path.join(output_dir, title)
    save_figure_multiple_formats(fig, filebase)
    plt.close()



def main():
    print("\nGrabbing mouse info...")
    mouse_info_path = os.path.join("/Volumes", "Petersen-Lab", "z_LSENS", "Share", "Dana_Shayakhmetova", "mouse_info", "joint_mouse_reference_weight.xlsx")
    mouse_info_df = pd.read_excel(mouse_info_path)
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df = mouse_info_df[
        (mouse_info_df['exclude'] == 0) &
        (mouse_info_df['reward_group'].isin(['R+', 'R-'])) &
        (mouse_info_df['recording'] == 1)]
    print("Done.")

    # Baseline data
    print("\nPulling baseline info...")
    baseline_perf_path = os.path.join(BASELINE_DATA_PATH, 'baseline_perf_cross_corr_light.pkl')
    baseline_hr_path = os.path.join(BASELINE_DATA_PATH, 'baseline_hr_cross_corr_light.pkl')
    baseline_fa_path = os.path.join(BASELINE_DATA_PATH, 'baseline_fa_cross_corr_light.pkl')
    baseline_perf_corr = pd.read_pickle(baseline_perf_path)
    baseline_hr_corr = pd.read_pickle(baseline_hr_path)
    baseline_fa_corr = pd.read_pickle(baseline_fa_path)
    print("Done.")

    print("GENERATING BASELINE PLOTS")
    
    # Performance
    print("\nBaseline Performance")
    print("Plotting significant neurons (all lags)...")
    plot_raw_count_significant_peaks_by_area(
        baseline_perf_corr, 
        mouse_info_df, 
        'baseline_performance_significant_peak_count',
        OUTPUT_DIR_BASELINE,
        interesting_areas=True
    )

    # Hit Rate
    print("\nBaseline Hit Rate")
    print("Plotting significant neurons (all lags)...")
    plot_raw_count_significant_peaks_by_area(
        baseline_hr_corr,
        mouse_info_df,
        'baseline_hr_significant_peak_count',
        OUTPUT_DIR_BASELINE,
        interesting_areas=True
    )

    # False Alarm
    print("\nBaseline False Alarm")
    print("Plotting significant neurons (all lags)...")
    plot_raw_count_significant_peaks_by_area(
        baseline_fa_corr,
        mouse_info_df,
        'baseline_fa_significant_peak_count',
        OUTPUT_DIR_BASELINE,
        interesting_areas=True
    )




    # Evoked response data
    print("\nPulling evoked response info...")
    evoked_perf_path = os.path.join(EVOKED_DATA_PATH, 'evoked_response_perf_cross_corr_light.pkl')
    evoked_hr_path = os.path.join(EVOKED_DATA_PATH, 'evoked_response_hr_cross_corr_light.pkl')
    evoked_fa_path = os.path.join(EVOKED_DATA_PATH, 'evoked_response_fa_cross_corr_light.pkl')
    evoked_perf_corr = pd.read_pickle(evoked_perf_path)
    evoked_hr_corr = pd.read_pickle(evoked_hr_path)
    evoked_fa_corr = pd.read_pickle(evoked_fa_path)
    print("Done.")

    print("GENERATING EVOKED RESPONSE PLOTS")
    
    # Performance
    print("\nEvoked Performance")
    print("Plotting significant neurons (all lags)...")
    plot_raw_count_significant_peaks_by_area(
        evoked_perf_corr,
        mouse_info_df,
        'evoked_performance_significant_peak_count',
        OUTPUT_DIR_EVOKED_RESPONSE,
        interesting_areas=True
    )
    
    # Hit Rate
    print("\nEvoked Hit Rate")
    print("Plotting significant neurons (all lags)...")
    plot_raw_count_significant_peaks_by_area(
        evoked_hr_corr,
        mouse_info_df,
        'evoked_hr_significant_peak_count',
        OUTPUT_DIR_EVOKED_RESPONSE,
        interesting_areas=True
    )

    # False Alarm
    print("\nEvoked False Alarm")
    print("Plotting significant neurons (all lags)...")
    plot_raw_count_significant_peaks_by_area(
        evoked_fa_corr,
        mouse_info_df,
        'evoked_fa_significant_peak_count',
        OUTPUT_DIR_EVOKED_RESPONSE,
        interesting_areas=True
    )
    



    print("\n\nScript done.")


if __name__ == '__main__':
    main()