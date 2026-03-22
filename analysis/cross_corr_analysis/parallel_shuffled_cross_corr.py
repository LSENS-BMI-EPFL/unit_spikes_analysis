import os
import warnings
import numpy as np
import pandas as pd
from scipy import signal, stats
warnings.filterwarnings('ignore')
from multiprocessing import Pool, cpu_count
from helper import save_figure_multiple_formats, parse_array


DATA_PATH = os.path.join('/Volumes', 'Petersen-Lab', 'z_LSENS', 'Share', 'Dana_Shayakhmetova', 'dynamic_analysis_dec16')
OUTPUT_DIR_BASELINE = os.path.join(DATA_PATH, 'baseline_analysis', 'cross_corr_data')
OUTPUT_DIR_INDV_TRIALS = os.path.join(DATA_PATH, 'evoked_response_analysis', 'cross_corr_data')
os.makedirs(OUTPUT_DIR_BASELINE, exist_ok=True)
os.makedirs(OUTPUT_DIR_INDV_TRIALS, exist_ok=True)


def parallel_correlogram_worker(args):
    """Compute cross-correlation with shuffles for a single neuron-behavior pair."""

    neural_series, bhv_series, mouse, area, neuron, n_shuffles = args

    if len(neural_series) < 2 or np.std(neural_series) == 0:
        return None

    corr_df = calculate_single_correlogram_shuffled(neural_series, bhv_series, n_shuffles=n_shuffles)

    if corr_df is not None:
        corr_df['mouse_id'] = mouse
        corr_df['area_custom'] = area
        corr_df['neuron_id'] = neuron
        corr_df = add_significance_to_correlogram(corr_df)
        return corr_df
    return None

def calculate_single_correlogram_shuffled(neuronal_data, bhv_data, lags_range=np.arange(-10, 11), n_shuffles=100):
    """Calculate cross-correlation at multiple time lags with shuffle-based null distribution."""

    #checking if we have enough valid data 
    N = len(neuronal_data)
    if N < 2 or np.std(neuronal_data) == 0 or np.std(bhv_data) == 0:
        return None 
    try:
        neural_norm = (neuronal_data - np.mean(neuronal_data)) / np.std(neuronal_data)
        bhv_norm = (bhv_data - np.mean(bhv_data)) / np.std(bhv_data)
    except Exception as e:
        warnings.warn(f"Normalization failed: {e}", RuntimeWarning)
        return None
    

    #normalized cross corr 
    raw_corr = signal.correlate(neural_norm, bhv_norm, mode='full')
    all_lags = signal.correlation_lags(N, N, mode='full')
    normalized_corr = raw_corr / N 
    lag_mask = np.isin(all_lags, lags_range)
    final_lags = all_lags[lag_mask]
    final_corrs = normalized_corr[lag_mask]
    shuffle_results = {lag: [] for lag in final_lags}
    
    #creating null distribution 
    for shuffle_idx in range(n_shuffles):
        shuffled_bhv = np.random.permutation(bhv_norm)
        shuffled_raw_corr = signal.correlate(neural_norm, shuffled_bhv, mode='full')
        shuffled_normalized_corr = shuffled_raw_corr / N
        shuffled_final_corrs = shuffled_normalized_corr[lag_mask]
        for lag, corr in zip(final_lags, shuffled_final_corrs):
            shuffle_results[lag].append(corr)
    
    correlations = []
    for lag, real_corr in zip(final_lags, final_corrs):
        correlations.append({
            'lag': lag,
            'cross_corr': real_corr,  
            'shuffled_cross_corrs': np.array(shuffle_results[lag]),  
            'n_points': N - abs(lag),
            'n_shuffles': n_shuffles
        })
    
    return pd.DataFrame(correlations)

def compute_time_lag_correlations_with_shuffles_parallel(neuronal_data, bhv_data, neuronal_metric, bhv_metric, n_shuffles=100):
    """Compute time-lagged correlations between neural and behavioral metrics across all neurons in parallel."""

    mice = neuronal_data.mouse_id.unique()
    all_tasks = []
    
    for mouse in mice:
        bhv_mouse_df = bhv_data[bhv_data.mouse_id == mouse].reset_index(drop=True)
        neural_mouse_df = neuronal_data[neuronal_data.mouse_id == mouse]
        areas = neural_mouse_df.area_custom.unique()
        
        if bhv_metric == 'performance':
            bhv_values = bhv_mouse_df['p_mean'] - bhv_mouse_df['p_chance']
        elif bhv_metric == 'hit_rate':
            # Hit Rate 
            bhv_values = bhv_mouse_df['p_mean']
        elif bhv_metric == 'false_alarm':
            # false alarm
            bhv_values = bhv_mouse_df['p_chance']
        else:
            raise ValueError(f"Unknown bhv_metric: {bhv_metric}")
            
        bhv_series = np.array(bhv_values)

        for area in areas:
            neural_area_df = neural_mouse_df[neural_mouse_df.area_custom == area]
            neurons = neural_area_df.neuron_id.unique()

            for neuron in neurons:        
                neuron_row = neural_area_df[neural_area_df.neuron_id == neuron].iloc[0]
                neural_series = np.array(neuron_row[neuronal_metric])
                all_tasks.append((neural_series, bhv_series, mouse, area, neuron, n_shuffles))

    # n_cpus = 5
    n_cpus = cpu_count()-1
    print(f"Starting parallel computation with {len(all_tasks)} tasks on {n_cpus} cores...")
    with Pool(processes=n_cpus) as pool:
        results = pool.map(parallel_correlogram_worker, all_tasks)
    all_corrs = [df for df in results if df is not None]

    #applying fdr correction to p-values 
    if all_corrs:
        combined_df =  pd.concat(all_corrs, ignore_index=True)
        fdr_corrected_df = combined_df.groupby(['mouse_id', 'area_custom', 'neuron_id'], group_keys=False).apply(add_fdr_correction)
        return fdr_corrected_df

    else:
        print("No valid correlations found across all mice/neurons.")
        return pd.DataFrame()


def add_fdr_correction(group):
    """Apply false discovery rate (benjamin-yekutieli) correction to p-values neuron by neuron."""
    p_values = group['p_value']
    alpha = 0.05
    pvals_corrected = stats.false_discovery_control(p_values, method='by')
    rejected = pvals_corrected < alpha
    group['p_value_corrected'] = pvals_corrected
    group['significant_fdr'] = rejected        
    return group



def add_significance_to_correlogram(corr_df):
    """Add p-values and significance flags to correlogram results.
    No FDR correction."""
    significance_df = corr_df.apply(calculate_pval_and_significance, axis=1)
    result_df = pd.concat([corr_df, significance_df], axis=1)
    return result_df

def calculate_pval_and_significance(row):
    """Calculate two-tailed p-value by comparing observed correlation to shuffled distribution.
    Inspired from ROC p-value calculation."""
    observed_corr = row['cross_corr']
    shuffled_corrs = row['shuffled_cross_corrs']
    n_shuffles = len(shuffled_corrs)

    p_value_pos = np.sum(shuffled_corrs >= observed_corr) / n_shuffles
    p_value_neg = np.sum(shuffled_corrs <= observed_corr) / n_shuffles
    
    if p_value_pos < 0.025:
        is_significant = True
        p_value = p_value_pos
    elif p_value_neg < 0.025:
        is_significant = True
        p_value = p_value_neg
    else:
        is_significant = False
        p_value = p_value_pos
    
    return pd.Series({
        'significant': is_significant,
        'p_value': p_value,
        'p_value_pos': p_value_pos,
        'p_value_neg': p_value_neg
    })

def main():
    print("\nLoading behavioral data...")
    bhv_path = os.path.join(DATA_PATH, 'behavior_data', 'merged_bhv_trials.pkl')
    with open(bhv_path, 'rb') as f:
        bhv_data = pd.read_pickle(f)
    bhv_data = bhv_data[bhv_data.whisker_stim==1]
    bhv_data = bhv_data[~bhv_data.mouse_id.isin(['AB164', 'AB142', 'AB107', 'AB149'])]
    print("Done.")
    
    print("\nLoading neural data...")
    neural_path = os.path.join(DATA_PATH, 'neuronal_data', 'neuronal_data_mh_ab.csv')
    neural_data = pd.read_csv(neural_path)
    neural_data = neural_data[~neural_data.mouse_id.isin(['AB068','AB164', 'AB142', 'AB107', 'AB149'])]
    array_cols = ["raw_post_spikes", "raw_pre_spikes", "raw_evoked_response", "pre_firing_rate", "post_firing_rate", "rate_evoked_response"]
    for col in array_cols:
        neural_data[col] = neural_data[col].apply(parse_array)
    print(neural_data.columns)
    print(neural_data.head())
    print("Done.")


    # BASELINE
    #PERF 
    print("\nCalculating cross corr for performance and baseline ...")
    cross_corr_baseline_perf = compute_time_lag_correlations_with_shuffles_parallel(
        neural_data, bhv_data, neuronal_metric='pre_firing_rate', bhv_metric='performance',n_shuffles=1000)
    cross_corr_baseline_perf.to_pickle((os.path.join(OUTPUT_DIR_BASELINE, 'baseline_perf_cross_corr.pkl')))
    cross_corr_baseline_perf_light = cross_corr_baseline_perf.drop(columns=['shuffled_cross_corrs'])
    cross_corr_baseline_perf_light.to_pickle((os.path.join(OUTPUT_DIR_BASELINE, 'baseline_perf_cross_corr_light.pkl')))
    cross_corr_baseline_perf_light.to_csv((os.path.join(OUTPUT_DIR_BASELINE, 'baseline_perf_cross_corr_light.csv')))
    print("Done.")

    #HR
    print("\nCalculating cross corr for hr and baseline ...")
    cross_corr_baseline_hr = compute_time_lag_correlations_with_shuffles_parallel(
        neural_data, bhv_data, neuronal_metric='pre_firing_rate', bhv_metric='hit_rate',n_shuffles=1000)
    cross_corr_baseline_hr.to_pickle((os.path.join(OUTPUT_DIR_BASELINE, 'baseline_hr_cross_corr.pkl')))
    cross_corr_baseline_hr_light = cross_corr_baseline_hr.drop(columns=['shuffled_cross_corrs'])
    cross_corr_baseline_hr_light.to_pickle((os.path.join(OUTPUT_DIR_BASELINE, 'baseline_hr_cross_corr_light.pkl')))
    cross_corr_baseline_hr_light.to_csv((os.path.join(OUTPUT_DIR_BASELINE, 'baseline_hr_cross_corr_light.csv')))
    print("Done.")
    
    #FA
    print("\nCalculating cross corr for false alarm and baseline ...")
    cross_corr_baseline_fa = compute_time_lag_correlations_with_shuffles_parallel(
        neural_data, bhv_data, neuronal_metric='pre_firing_rate', bhv_metric='false_alarm',n_shuffles=1000)
    cross_corr_baseline_fa.to_pickle((os.path.join(OUTPUT_DIR_BASELINE, 'baseline_fa_cross_corr.pkl')))
    cross_corr_baseline_fa_light = cross_corr_baseline_fa.drop(columns=['shuffled_cross_corrs'])
    cross_corr_baseline_fa_light.to_pickle((os.path.join(OUTPUT_DIR_BASELINE, 'baseline_fa_cross_corr_light.pkl')))
    cross_corr_baseline_fa_light.to_csv((os.path.join(OUTPUT_DIR_BASELINE, 'baseline_fa_cross_corr_light.csv')))
    print("Done.")



    # EVOKED RESPONSE RATE 
    #PERF
    print("\nCalculating cross corr for performance and evoked response...")
    cross_corr_evoked_response_perf = compute_time_lag_correlations_with_shuffles_parallel(neural_data, bhv_data, neuronal_metric='rate_evoked_response', bhv_metric='performance',n_shuffles=1000)
    cross_corr_evoked_response_perf.to_pickle((os.path.join(OUTPUT_DIR_INDV_TRIALS, 'evoked_response_perf_cross_corr.pkl')))
    cross_corr_evoked_response_perf_light = cross_corr_evoked_response_perf.drop(columns=['shuffled_cross_corrs'])
    cross_corr_evoked_response_perf_light.to_pickle((os.path.join(OUTPUT_DIR_INDV_TRIALS, 'evoked_response_perf_cross_corr_light.pkl')))
    cross_corr_evoked_response_perf_light.to_csv((os.path.join(OUTPUT_DIR_INDV_TRIALS, 'evoked_response_perf_cross_corr_light.csv')))
    print("Done.")

    #HR
    print("\nCalculating cross corr for hr and evoked response...")
    cross_corr_evoked_response_hr = compute_time_lag_correlations_with_shuffles_parallel(
        neural_data, bhv_data, neuronal_metric='rate_evoked_response', bhv_metric='hit_rate',n_shuffles=1000)
    cross_corr_evoked_response_hr.to_pickle((os.path.join(OUTPUT_DIR_INDV_TRIALS, 'evoked_response_hr_cross_corr.pkl')))
    cross_corr_evoked_response_hr_light = cross_corr_evoked_response_hr.drop(columns=['shuffled_cross_corrs'])
    cross_corr_evoked_response_hr_light.to_pickle((os.path.join(OUTPUT_DIR_INDV_TRIALS, 'evoked_response_hr_cross_corr_light.pkl')))
    cross_corr_evoked_response_hr_light.to_csv((os.path.join(OUTPUT_DIR_INDV_TRIALS, 'evoked_response_hr_cross_corr_light.csv')))
    print("Done.")
    
    #FA
    print("\nCalculating cross corr for false alarm and evoked response...")
    cross_corr_evoked_response_fa = compute_time_lag_correlations_with_shuffles_parallel(
        neural_data, bhv_data, neuronal_metric='rate_evoked_response', bhv_metric='false_alarm',n_shuffles=1000)
    cross_corr_evoked_response_fa.to_pickle((os.path.join(OUTPUT_DIR_INDV_TRIALS, 'evoked_response_fa_cross_corr.pkl')))
    cross_corr_evoked_response_fa_light = cross_corr_evoked_response_fa.drop(columns=['shuffled_cross_corrs'])
    cross_corr_evoked_response_fa_light.to_pickle((os.path.join(OUTPUT_DIR_INDV_TRIALS, 'evoked_response_fa_cross_corr_light.pkl')))
    cross_corr_evoked_response_fa_light.to_csv((os.path.join(OUTPUT_DIR_INDV_TRIALS, 'evoked_response_fa_cross_corr_light.csv')))
    print("Done.")


if __name__ == '__main__':
    main()