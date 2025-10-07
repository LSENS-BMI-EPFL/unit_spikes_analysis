#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: brain_wide_analysis
@file: waveform_utils.py
@time: 10/23/2024 10:59 PM
"""

#Imports
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import signal, optimize, stats


import NWB_reader_functions as nwb_reader
import allen_utils as allen_utils
import plotting_utils as plutils


def process_area_acronyms(unit_table):  # TODO: use this function to combine future areas e.g. ACAv,ACAd -> ACA, etc.
    """
    Process and re-assign area acronyms.
    In particular, groups barrel columns together.
    :param unit_table: PETH table with all mice data
    :param params: Plotting parameters
    :return: Updated PETH table with processed area acronyms
    """
    # Assign to SSp-bfd if ccf parent acronym contains SSp-bfd
    unit_table['ccf_parent_acronym'] = unit_table['ccf_parent_acronym'].astype(str)
    unit_table['ccf_parent_acronym'] = unit_table['ccf_parent_acronym'].apply(
        lambda x: 'SSp-bfd' if 'SSp-bfd' in x else x
    )

    # Decide which area acronym to use
    # Use ccf_parent_acronym if layer or part is in the name
    unit_table['area_acronym'] = unit_table.apply(
        lambda row: row['ccf_parent_acronym']
        if ('layer' in row['ccf_name'].lower() or 'part' in row['ccf_name'].lower())
        else row['ccf_acronym'],
        axis=1
    )

    # Same for area name
    unit_table['area_name'] = unit_table.apply(
        lambda row: row['ccf_parent_name']
        if ('layer' in row['ccf_name'].lower() or 'part' in row['ccf_name'].lower())
        else row['ccf_name'],
        axis=1
    )

    # For cortical areas, use the ccf_parent_acronym
    ctx_areas = allen_utils.get_cortical_areas()
    unit_table['area_acronym'] = [row['ccf_parent_acronym'] if row['ccf_parent_acronym'] in ctx_areas
                                  else row['ccf_acronym'] for idx, row in unit_table.iterrows()]

    # Same for area name
    unit_table['area_name'] = [row['ccf_parent_name'] if row['ccf_parent_name'] in ctx_areas
                                  else row['ccf_name'] for idx, row in unit_table.iterrows()]

    return unit_table



def classify_rsu_vs_fsu(unit_data, output_path):
    """
    This function assigns the unit type (FSU or RSU) to each unit using area-specific
    distributions obtained from a list of NWB files.
    :param unit_data: unit table with all mice data
    :param output_path: Path to save the results
    :return:
    """

    print('Classification of cortical cells: RSU vs FSU...')

    print('Total number of well-isolated "good" neurons', len(unit_data[unit_data['bc_label'] == 'good']))

    # Get unique areas
    unit_data = unit_data[~unit_data['ccf_acronym'].isin(allen_utils.get_excluded_areas())]
    unit_data = allen_utils.create_area_custom_column(unit_data)
    area_list = unit_data['area_acronym_custom'].unique()

    # Iterate over areas
    results = []
    for area in area_list:
        print('Processing area:', area)
        area_data = unit_data[(unit_data['area_acronym_custom'] == area)
                                 & (unit_data['bc_label'] == 'good')]

        if area_data.empty or len(area_data) < 5:
            print(f'Not enough good units found in {area}. Only MUA or non-somatic. Skipping...')
            continue

        area_data['duration'] = area_data['duration'].astype(float)

        # Get distributions AP durations (peak-to-trough)
        ap_durations = np.array(area_data['duration'].values)

        # Fit Gaussian Mixture Model
        means_init_array = np.array([0.2, 0.6]).reshape(-1,1)
        gmm = GaussianMixture(n_components=2,
                              covariance_type='diag',
                              tol=1e-4,
                              reg_covar=1e-6,
                              max_iter=200,
                              init_params='kmeans',
                              #weights_init=[0.15,0.85], # approx. relative proportion of FSU and RSU
                              n_init=5,
                              means_init=means_init_array, # initial means for FSU and RSU
                              random_state=0,
                              warm_start=False,
                              verbose=False)
        gmm.fit(ap_durations.reshape(-1,1))

        def find_intersect_gauss(m1,m2,std1,std2,weight1,weight2):
            """Find intersection(s) of two weighted Gaussians."""
            a = 1/(2*std1**2) - 1/(2*std2**2)
            b = m2/(std2**2) - m1/(std1**2)
            c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1) - np.log(weight1/weight2)
            return np.roots([a,b[0],c[0]])

        # Find all roots
        intersect = find_intersect_gauss(gmm.means_[0],
                                         gmm.means_[1],
                                         math.sqrt(gmm.covariances_[0]),
                                         math.sqrt(gmm.covariances_[1]),
                                         gmm.weights_[0],
                                         gmm.weights_[1])
        #print('GMM intersection roots', intersect)

        # Among all roots, keep the biologically-plausible one
        upper_lim = 0.9 # ms #0.8
        lower_lim = 0.1 #0.2
        intersect_real = intersect[np.where(np.logical_and(intersect>=lower_lim, intersect<=upper_lim))[0]]

        if len(intersect_real) > 1:
            # Take the smallest root
            threshold = np.min(intersect_real)
        elif len(intersect_real) == 1:
            threshold = float(intersect_real[0])
        elif len(intersect_real) == 0:
            threshold = np.nan
            print(f'Could not find root/AP duration threshold in range {lower_lim}-{upper_lim} ms. Skipping...')

        # Plot results
        debug = True
        if debug:
            color_dict = {'rsu':'#ff8783', 'fsu':'#83b1ff'}
            fig,ax=plt.subplots(1,1,figsize=(5,5),dpi=300)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            _, bins, _ = ax.hist(ap_durations, bins=int(np.sqrt(len(ap_durations))),
                    color='dimgrey', histtype='step',
                    density=False, lw=1.5, label='All units')

            # Evaluate densities, ensuring mean FS < mean RS
            def gaussian(x, mu, sig):
                return (1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2))

            x = np.linspace(0,1.3,1000)
            if gmm.means_[0] < gmm.means_[1]:
                params_fs = gmm.weights_[0], gmm.means_[0], np.sqrt(gmm.covariances_[0])
                y_fs =  params_fs[0] * gaussian(x, params_fs[1], params_fs[2])
                params_rs = gmm.weights_[1], gmm.means_[1], np.sqrt(gmm.covariances_[1])
                y_rs = params_rs[0] * gaussian(x, params_rs[1], params_rs[2])
            else:
                params_fs = gmm.weights_[1], gmm.means_[1], np.sqrt(gmm.covariances_[1])
                y_fs =  params_fs[0] * gaussian(x, params_fs[1], params_fs[2])
                params_rs = gmm.weights_[0], gmm.means_[0], np.sqrt(gmm.covariances_[0])
                y_rs = params_rs[0] * gaussian(x, params_rs[1], params_rs[2])

            # Scale densities to count to match histogram
            bin_width = np.diff(bins)[0]
            y_fs = y_fs * bin_width * len(ap_durations)
            y_rs = y_rs * bin_width * len(ap_durations)

            # Plot the two Gaussians from the GMM
            ax.plot(x, y_fs, color=color_dict['fsu'], lw=3, label=r'Fast-spiking' '\n' '$\mathcal{N}_F$'+r'({:.2f},{:.2f})'.format(gmm.means_[0][0], np.sqrt(gmm.covariances_[0][0])))
            ax.plot(x, y_rs, color=color_dict['rsu'], lw=3, label=r'Regular-spiking' '\n' '$\mathcal{N}_R$'+r'({:.2f},{:.2f})'.format(gmm.means_[1][0], np.sqrt(gmm.covariances_[1][0])))
            ax.axvline(x=threshold, color='k', linestyle='--', label=r'threshold$={:.2f}$'.format(threshold))
            ax.set_title(f'{area} AP durations', fontsize=15)
            ax.set_xlabel('AP peak-to-trough duration (ms)', fontsize=15)
            ax.set_xlim(0,1.3)
            ax.set_ylabel('Count', fontsize=15)
            ax.legend(frameon=False, fontsize=8, loc='upper right')
            plt.show()

            # Save figure for illustration
            fig_name = f'all_mice_{area}_ap_durations'
            output_path_all = os.path.join(output_path, 'waveform_analysis', 'cortex', area)
            if not os.path.exists(output_path_all):
                os.makedirs(output_path_all)
            plutils.save_figure_with_options(figure=fig,
                                             file_formats=['png','svg','pdf'],
                                             filename=fig_name,
                                             output_dir=output_path_all,
                                             dark_background=False)

        # Assign RSU vs FSU based on the duration column using apply
        area_data['waveform_type'] = area_data['duration'].apply(lambda x: 'fsu' if x<=threshold else 'rsu')
        results.append(area_data)

    # Recombine the results, keeping only identifiers and the waveform type
    results = pd.concat(results)

    # For each mouse, save the results
    mouse_list = results['mouse_id'].unique()
    for mouse_id in mouse_list:
        mouse_results = results[results['mouse_id'] == mouse_id]
        cols_to_keep = ['mouse_id', 'session_id', 'behaviour', 'day', 'electrode_group', 'cluster_id', 'waveform_type']
        mouse_results = mouse_results[cols_to_keep]
        behaviour = mouse_results['behaviour'].unique()[0]
        day = mouse_results['day'].unique()[0]

        mouse_output_path = os.path.join(output_path, mouse_id, f'{behaviour}_{day}', 'waveform_analysis')

        if not os.path.exists(mouse_output_path):
            os.makedirs(mouse_output_path)

        file_name = '{}_cortical_wf_type.csv'.format(mouse_id)
        mouse_results.to_csv(os.path.join(mouse_output_path, file_name))

    print('RSU vs FSU assignment done and saved for each mouse.')
    return


def compute_isi_properties(spike_times, param):
    """
    Compute inter-spike interval based properties

    Parameters
    ----------
    spike_times : array
        Spike times in seconds
    param : dict
        Parameters dictionary

    Returns
    -------
    isi_props : dict
        Dictionary containing ISI properties
    """
    isi_props = {
        'prop_long_isi': np.nan,
        #'cv': np.nan,
        #'cv2': np.nan,
        #'isi_skewness': np.nan
    }

    if len(spike_times) < 2:
        return isi_props

    # Compute ISIs
    isis = np.diff(spike_times)

    if 'prop_long_isi' in isi_props.keys():
        # Proportion of long ISIs - use MATLAB parameter name
        long_isi_threshold = param.get('longISI', 2.0)  # MATLAB: paramEP.longISI = 2
        long_isis = isis > long_isi_threshold
        isi_props['prop_long_isi'] = np.mean(long_isis)

    if 'cv' in isi_props.keys() or 'isi_skewness' in isi_props.keys():
        # Coefficient of variation
        if len(isis) > 0:
            isi_props['cv'] = np.std(isis) / np.mean(isis)
            isi_props['isi_skewness'] = stats.skew(isis)

    if 'cv2' in isi_props.keys():
        # CV2 (local coefficient of variation) - vectorized for speed
        if len(isis) > 1:
            mean_isis = (isis[:-1] + isis[1:]) / 2
            diff_isis = np.abs(isis[1:] - isis[:-1])
            valid_mask = mean_isis > 0
            if np.any(valid_mask):
                cv2_values = diff_isis[valid_mask] / mean_isis[valid_mask]
                isi_props['cv2'] = np.mean(cv2_values)

    return isi_props

from ccg_fast import acg as compute_acg_fast
def compute_acg(spike_times, bin_size, duration, param=None):
    """
    Wrapper for fast ACG computation - maintains interface compatibility

    Parameters
    ----------
    spike_times : array
        Spike times in seconds
    bin_size : float
        Bin size in seconds
    duration : float
        Total duration in seconds (centered around 0)
    param : dict, optional
        Parameters dictionary containing memory management settings

    Returns
    -------
    acg : array
        Auto-correlogram counts
    lags : array
        Lag times in seconds
    """
    bin_size_ms = bin_size * 1000  # Convert to ms
    duration_ms = duration * 1000  # Convert to ms

    # Get sampling rate from param or use default
    fs = param.get('ephys_sampling_rate', 30000) if param is not None else 30000

    # Convert spike times from seconds to samples
    spike_times_samples = (spike_times * fs).astype(np.uint64)

    # Use acg from ccg_fast
    acg_result = compute_acg_fast(spike_times_samples, cbin=bin_size_ms, cwin=duration_ms,
                                  fs=fs, normalize="counts", cache_results=False)

    # Generate lag times to match expected output
    n_bins = len(acg_result)
    lags_ms = np.linspace(-duration_ms / 2, duration_ms / 2, n_bins)
    lags = lags_ms / 1000  # Convert back to seconds

    return acg_result, lags

def compute_acg_properties(spike_times, param):
    """
    Compute auto-correlogram based properties

    Parameters
    ----------
    spike_times : array
        Spike times in seconds
    param : dict
        Parameters dictionary

    Returns
    -------
    acg_props : dict
        Dictionary containing ACG properties
    """
    # Compute ACG using MATLAB parameter names
    acg_bin_size = param.get('ACGbinSize', 0.001)
    acg_duration = param.get('ACGduration', 1.0)
    acg, lags = compute_acg(spike_times, acg_bin_size, acg_duration, param)

    # Initialize output
    acg_props = {
        'acg': acg,
        #'post_spike_suppression_ratio': np.nan,
        'post_spike_suppression_ms': np.nan,
        #'tau_rise_ms': np.nan,
        #'tau_decay_ms': np.nan
    }

    if len(acg) == 0 or len(spike_times) < 10:
        return acg_props

    # Ensure ACG has some counts
    if np.sum(acg) == 0:
        return acg_props

    # Find center bin
    center_idx = len(acg) // 2

    # Post-spike suppression: ratio of minimum in first 10ms to baseline
    bin_size_sec = acg_bin_size
    post_bins = max(1, int(0.01 / bin_size_sec))  # 10ms in bins, at least 1
    baseline_start_bins = min(max(20, int(0.05 / bin_size_sec)),
                              len(acg) // 4)  # Start baseline at 50ms, but adapt to ACG length

    # Make sure we have enough bins for both regions
    if center_idx + baseline_start_bins + 10 < len(acg):
        # Suppression region: 1-10ms after center (skip center bin to avoid refractory period)
        supp_start = center_idx + 1
        supp_end = min(center_idx + post_bins + 1, len(acg))
        suppression_region = acg[supp_start:supp_end]

        # Baseline region: 50ms to end (use symmetric baseline on both sides)
        baseline_left = acg[max(0, center_idx - baseline_start_bins):center_idx - post_bins]
        baseline_right = acg[center_idx + baseline_start_bins:]
        baseline_region = np.concatenate([baseline_left, baseline_right]) if len(baseline_left) > 0 else baseline_right

        if len(suppression_region) > 0 and len(baseline_region) > 5:
            min_val = np.min(suppression_region)
            baseline_mean = np.mean(baseline_region)

            if baseline_mean > 0:
                postSpikeSuppression_ratio = min_val / baseline_mean
                acg_props['post_spike_suppression_ratio'] = postSpikeSuppression_ratio

                # Compute post-spike suppression EXACTLY as in MATLAB Nature paper
                # MATLAB: postSpikeSup = find(thisACG(500:1000) >= nanmean(thisACG(600:900)));
                # Convert MATLAB indices to Python (MATLAB uses 1-based indexing)

                # Calculate equivalent indices for our ACG
                # MATLAB ACG is 1000 bins (1s duration, 1ms bins), center at 500
                # Our ACG has different binning, so scale appropriately
                acg_duration_sec = acg_duration
                total_bins = len(acg)

                if total_bins >= 100:  # Need enough bins for the computation
                    # Map MATLAB indices to our indices
                    # MATLAB 500:1000 = 0:500ms post-spike (500 bins)
                    # MATLAB 600:900 = 100:400ms post-spike (300 bins for baseline)

                    # Scale to our bin structure
                    post_start_idx = center_idx  # Start at center (0ms)
                    post_end_idx = min(center_idx + int(0.5 / bin_size_sec), len(acg))  # Up to 500ms

                    baseline_start_idx = center_idx + int(0.1 / bin_size_sec)  # 100ms post-spike
                    baseline_end_idx = min(center_idx + int(0.4 / bin_size_sec), len(acg))  # 400ms post-spike

                    if (post_end_idx > post_start_idx and
                            baseline_end_idx > baseline_start_idx and
                            baseline_end_idx <= len(acg)):

                        # Compute baseline as in MATLAB
                        baseline_region = acg[baseline_start_idx:baseline_end_idx]
                        baseline_mean_matlab = np.nanmean(baseline_region)

                        # Find first point where ACG >= baseline mean (as in MATLAB)
                        post_region = acg[post_start_idx:post_end_idx]
                        recovery_indices = np.where(post_region >= baseline_mean_matlab)[0]

                        if len(recovery_indices) > 0:
                            # Time to first recovery point (in ms)
                            recovery_bin = recovery_indices[0]  # First index in post_region
                            postSpikeSup_ms = recovery_bin * bin_size_sec * 1000  # Convert to ms
                            acg_props['post_spike_suppression_ms'] = postSpikeSup_ms
                        else:
                            acg_props['post_spike_suppression_ms'] = np.nan
                    else:
                        acg_props['post_spike_suppression_ms'] = np.nan
                else:
                    acg_props['post_spike_suppression_ms'] = np.nan
            else:
                acg_props['post_spike_suppression_ratio'] = 0.0
                acg_props['post_spike_suppression_ms'] = 50.0  # Default

    # Simple tau estimation based on ACG shape in first 20ms
    try:
        # Use correct parameter name
        acg_bin_size_used = param.get('ACGbinSize', 0.001)
        if center_idx + int(0.02 / acg_bin_size_used) < len(acg):
            fit_region = acg[center_idx + 1:center_idx + int(0.02 / acg_bin_size_used)]  # 0-20ms region

            if len(fit_region) > 5:
                # Find rise time: time to reach half maximum
                max_val = np.max(fit_region)
                half_max = max_val * 0.5

                if max_val > 0:
                    # Find first bin that exceeds half max
                    rise_idx = np.where(fit_region >= half_max)[0]
                    if len(rise_idx) > 0:
                        acg_props['tau_rise_ms'] = rise_idx[0] * acg_bin_size_used * 1000

                # Simple decay estimation: fit exponential to second half
                if len(fit_region) > 10:
                    decay_region = fit_region[len(fit_region) // 2:]
                    x = np.arange(len(decay_region))

                    if len(decay_region) > 3 and np.max(decay_region) > 0:
                        # Simple exponential fit
                        try:
                            # Log-linear fit for exponential decay
                            y_log = np.log(np.maximum(decay_region, np.max(decay_region) * 0.01))
                            coeffs = np.polyfit(x, y_log, 1)
                            tau_bins = -1 / coeffs[0] if coeffs[0] < 0 else np.nan
                            acg_props['tau_decay_ms'] = tau_bins * acg_bin_size_used * 1000
                        except:
                            pass
    except:
        pass

    return acg_props


def plot_striatal_waveforms_and_acgs(results_df, output_path):
    """"
    Plot average waveforms and ACGs for striatal cell types.
    :param results_df: DataFrame with unit data including 'striatal_type', 'waveform_mean', and 'acg'
    :param output_path: Path to save the figures
    """

    # Define striatal areas and cell type order
    striatal_areas = ['CP', 'ACB', 'STR', 'STRd', 'STRv', 'FS', 'OT']
    cell_types = ['msn', 'fsi', 'tan', 'uin']
    cell_type_colors = { #fiery aurora nights palette
        'msn': '#d7263d',
        'fsi': '#2e294e',
        'tan': '#1b998b',
        'uin': '#f46036'
    }
    n_types = len(cell_types)

    # Filter to striatal units only
    results_df = results_df[(results_df['ccf_atlas_acronym'].isin(striatal_areas))
                        & (results_df['bc_label']=='good')]


    # Set up figure
    fig, axs = plt.subplots(2, n_types, figsize=(2*n_types, 4), dpi=600, sharex=False, sharey=False)
    for ax in axs.flat:
        plutils.remove_top_right_frame(ax)
        ax.set_box_aspect(1)
        ax.tick_params(axis='both', which='major', labelsize=6)


    # --- Loop through cell types ---
    for idx, ctype in enumerate(cell_types):
        subset = results_df[results_df['striatal_type'] == ctype]
        color = cell_type_colors[ctype]

        # Skip if no units of this type
        if subset.empty:
            for r in range(2):
                axs[r, idx].set_axis_off()
            continue

        # Waveforms
        gain_to_uV = 0.6 / (2**10/2) / 500 * 1e6
        waveforms = [np.array(w) * gain_to_uV for w in subset['waveform_mean']] #dropna()
        wf_mat = np.vstack(waveforms)
        wf_mean = wf_mat.mean(axis=0)
        #Get standard error of the mean across cells
        wf_sem = wf_mat.std(axis=0) / np.sqrt(wf_mat.shape[0]) #
        wf_x = np.arange(len(wf_mean))

         # Plot waveform
        ax = axs[0, idx]
        ax.plot(wf_x, wf_mean, color=color)
        ax.fill_between(wf_x, wf_mean - 1.96 * wf_sem, wf_mean + 1.96 * wf_sem, color=color, alpha=0.2, lw=0)
        ax.set_title(ctype.upper()+'s' + f' (n={len(subset)})', fontsize=10)
        ax.set_xlabel("Time (ms)", fontsize=8)
        ax.set_ylabel("Amplitude (uV)", fontsize=8)
        #ax.grid(True, alpha=0.2)
        #axs[0, idx].set_ylim(-20,5)
        n_samples = len(wf_mean)
        labels = [0,1,2,3]
        sampling_rate = 30000  # Hz
        xticks = [int(i * sampling_rate / 1000) for i in labels]
        axs[0,idx].set_xticks(xticks)
        axs[0,idx].set_xticklabels(labels)

        # Autocorrelogram
        acgs = [np.array(a) for a in subset['acg'].dropna()]
        acg_mat = np.vstack(acgs)
        acg_mean = acg_mat.mean(axis=0)
        acg_sem = acg_mat.std(axis=0) / np.sqrt(acg_mat.shape[0])
        acg_x = np.arange(len(acg_mean))

        ax = axs[1, idx]
        ax.plot(acg_x, acg_mean, color=color)
        ax.fill_between(acg_x, acg_mean, 0, color=color, alpha=1.0)
        ax.fill_between(acg_x, acg_mean - 1.96 * acg_sem, acg_mean + 1.96 * acg_sem, color=color, alpha=0.2, lw=0)
        ax.set_xlabel("Lag (ms)", fontsize=8)
        ax.set_ylabel("Firing rate (spikes/s)", fontsize=8)
        axs[1, idx].set_xlim(300,700)
        axs[1, idx].set_xticks([300,400,500,600,700])
        axs[1, idx].set_xticklabels([-200,-100,0,100,200])
        #axs[1, idx].set_ylim(-100,2500)
        #ax.grid(True, alpha=0.2)

    # Adjust layout
    plt.tight_layout()
    fig.align_ylabels()

    # Save
    fig_name = 'striatal_cell_types_waveforms_acgs'
    output_path_all = os.path.join(output_path, 'waveform_analysis', 'striatum')
    if not os.path.exists(output_path_all):
        os.makedirs(output_path_all)
    plutils.save_figure_with_options(figure=fig,
                                     file_formats=['png', 'svg', 'pdf'],
                                     filename=fig_name,
                                     output_dir=output_path_all,
                                     dark_background=False)

    return

def classify_striatal_units(unit_data, output_path):
    """
    Vectorized classification of striatal cell types.
    """
    param = {}  # will use default values

    # --- Parameters ---
    templateDuration_CP_threshold = 400  # µs
    postSpikeSup_CP_threshold = 40       # ms
    propISI_CP_threshold = 0.1

    # --- Keep only valid rows (good quality + striatal areas) ---
    valid_mask = (
        (unit_data['bc_label'] == 'good') &
        (unit_data['ccf_atlas_acronym'].isin(['CP', 'ACB', 'STR', 'STRd', 'STRv', 'FS', 'OT']))
        #& (len(unit_data['spike_times'])>300)
    )

    # Initialize result column
    unit_data = unit_data.copy()
    unit_data['striatal_type'] = 'NA'
    unit_data.loc[valid_mask, 'striatal_type'] = 'unknown'

    unit_data['post_spike_suppression_ms'] = np.nan
    unit_data['acg'] = None
    unit_data['prop_long_isi'] = np.nan


    # --- Compute ACG and ISI properties for valid units ---
    # Compute and store both ACG arrays and suppression durations
    acg_results = unit_data.loc[valid_mask, 'spike_times'].apply(
        lambda st: compute_acg_properties(st, param)
    )
    unit_data.loc[valid_mask, 'acg'] = acg_results.apply(lambda d: d['acg'])
    unit_data.loc[valid_mask, 'post_spike_suppression_ms'] = acg_results.apply(
        lambda d: d['post_spike_suppression_ms']
    )

    # Compute and store ISI features
    isi_results = unit_data.loc[valid_mask, 'spike_times'].apply(
        lambda st: compute_isi_properties(st, param)
    )
    unit_data.loc[valid_mask, 'prop_long_isi'] = isi_results.apply(lambda d: d['prop_long_isi'])

    del acg_results
    del isi_results

    # --- Extract arrays for vectorized conditions ---
    unit_data['waveformDuration_peakTrough'] = unit_data['waveformDuration_peakTrough'].astype(float)

    wf_duration = unit_data['waveformDuration_peakTrough'] # in us
    post_spike_suppression_ms = unit_data['post_spike_suppression_ms']
    prop_long_isi = unit_data['prop_long_isi']

    # Invalid feature mask (missing data)
    invalid_mask = valid_mask & (
            wf_duration.isna() | post_spike_suppression_ms.isna() | prop_long_isi.isna()
    )
    unit_data.loc[invalid_mask, 'striatal_type'] = 'unknown'

    # --- Classification masks (vectorized logic) ---
    msn_mask = (
            valid_mask &
            (wf_duration > templateDuration_CP_threshold) &
            (post_spike_suppression_ms < postSpikeSup_CP_threshold)
    )
    fsi_mask = (
            valid_mask &
            (wf_duration <= templateDuration_CP_threshold) &
            (prop_long_isi <= propISI_CP_threshold)
    )
    tan_mask = (
            valid_mask &
            (wf_duration > templateDuration_CP_threshold) &
            (post_spike_suppression_ms >= postSpikeSup_CP_threshold)
    )
    uin_mask = (
            valid_mask &
            (wf_duration <= templateDuration_CP_threshold) &
            (prop_long_isi > propISI_CP_threshold)
    )

    # --- Assign labels ---
    unit_data.loc[msn_mask, 'striatal_type'] = 'msn'
    unit_data.loc[fsi_mask, 'striatal_type'] = 'fsi'
    unit_data.loc[tan_mask, 'striatal_type'] = 'tan'
    unit_data.loc[uin_mask, 'striatal_type'] = 'uin'

    #summary = unit_data['striatal_type'].value_counts()

    # Plot classification results
    plot_striatal_waveforms_and_acgs(unit_data, output_path)

    # Save entire dataframe with results
    output_file = os.path.join(output_path, 'waveform_analysis', 'striatum', 'all_mice_striatal_waveform_classification.csv')
    unit_data.to_csv(output_file, index=False)
    print('Striatal cell type classification done and saved.')


    # For each mouse, save the results
    mouse_list = unit_data['mouse_id'].unique()
    for mouse_id in mouse_list:
        mouse_results = unit_data[unit_data['mouse_id'] == mouse_id]
        cols_to_keep = ['mouse_id', 'session_id', 'behaviour', 'day', 'electrode_group', 'cluster_id', 'striatal_type']
        mouse_results = mouse_results[cols_to_keep]
        behaviour = mouse_results['behaviour'].unique()[0]
        day = mouse_results['day'].unique()[0]

        mouse_output_path = os.path.join(output_path, mouse_id, f'{behaviour}_{day}', 'waveform_analysis')

        if not os.path.exists(mouse_output_path):
            os.makedirs(mouse_output_path)

        file_name = '{}_striatal_wf_type.csv'.format(mouse_id)
        mouse_results.to_csv(os.path.join(mouse_output_path, file_name))

    print('Striatal cell type classification done and saved for each mouse.')
    return

def classify_striatum_units_old(unit_data, output_path):
    """
    Classify striatal cell types exactly matching MATLAB classifyCells.m

    MATLAB Classification rules:
    - MSN: waveformDuration_peakTrough_us > templateDuration_CP_threshold & postSpikeSuppression_ms < postSpikeSup_CP_threshold
    - FSI: waveformDuration_peakTrough_us <= templateDuration_CP_threshold & propLongISI <= propISI_CP_threshold
    - TAN: waveformDuration_peakTrough_us > templateDuration_CP_threshold & postSpikeSuppression_ms >= postSpikeSup_CP_threshold
    - UIN: waveformDuration_peakTrough_us <= templateDuration_CP_threshold & propLongISI > propISI_CP_threshold

    Parameters from MATLAB:
    - templateDuration_CP_threshold = 400 (microseconds)
    - postSpikeSup_CP_threshold = 40 (milliseconds)
    - propISI_CP_threshold = 0.1
    """

    param = {} # will use default values

    n_units = len(unit_data)
    results = pd.DataFrame(index=unit_data.index, columns=['strial_type'])
    cell_types = ['unknown'] * n_units  # Initialize as Unknown

    # MATLAB parameter values
    templateDuration_CP_threshold = 400  # microseconds
    postSpikeSup_CP_threshold = 40  # milliseconds
    propISI_CP_threshold = 0.1

    #for i in range(n_units):
    for i, row in unit_data.iterrows():
        if row['bc_label'] != 'good' or row['ccf_atlas_acronym'] not in ['CP', 'ACB', 'STR', 'STRd', 'STRv', 'FS', 'OT']:
            cell_types[i] = 'NA'
            continue
        # Use exact MATLAB variable names
        waveformDuration_peakTrough_us = row['waveformDuration_peakTrough'] # in us

        spike_times = row['spike_times']

        acg_props = compute_acg_properties(spike_times, param)
        postSpikeSuppression_ms = acg_props['post_spike_suppression_ms']

        isi_props = compute_isi_properties(spike_times, param)
        propLongISI = isi_props['prop_long_isi']

        # If any required property is missing, assign Unknown
        if (np.isnan(waveformDuration_peakTrough_us) or
                np.isnan(postSpikeSuppression_ms) or
                np.isnan(propLongISI)):
            cell_types[i] = 'unknown'
            continue

        # Apply exact MATLAB classification logic
        # Medium spiny neurons: wide waveform AND short post-spike suppression
        if (waveformDuration_peakTrough_us > templateDuration_CP_threshold and
                postSpikeSuppression_ms < postSpikeSup_CP_threshold):
            cell_types[i] = 'msn'

        # Fast-spiking interneurons: narrow waveform AND low proportion of long ISIs
        elif (waveformDuration_peakTrough_us <= templateDuration_CP_threshold and
              propLongISI <= propISI_CP_threshold):
            cell_types[i] = 'fsi'

        # Tonically active neurons: wide waveform AND long post-spike suppression
        elif (waveformDuration_peakTrough_us > templateDuration_CP_threshold and
              postSpikeSuppression_ms >= postSpikeSup_CP_threshold):
            cell_types[i] = 'tan'

        # Undefined interneurons: narrow waveform AND high proportion of long ISIs
        elif (waveformDuration_peakTrough_us <= templateDuration_CP_threshold and
              propLongISI > propISI_CP_threshold):
            cell_types[i] = 'uin'

        else:
            # Should not reach here with valid data
            cell_types[i] = 'unknown'

    # Format dataframe for output
    results['striatal_type'] = cell_types

    # For each mouse, save the results
    mouse_list = results['mouse_id'].unique()
    for mouse_id in mouse_list:
        mouse_results = results[results['mouse_id'] == mouse_id]
        cols_to_keep = ['mouse_id', 'session_id', 'behaviour', 'day', 'electrode_group', 'cluster_id', 'waveform_type']
        mouse_results = mouse_results[cols_to_keep]
        behaviour = mouse_results['behaviour'].unique()[0]
        day = mouse_results['day'].unique()[0]

        mouse_output_path = os.path.join(output_path, mouse_id, f'{behaviour}_{day}', 'waveform_analysis')

        if not os.path.exists(mouse_output_path):
            os.makedirs(mouse_output_path)

        file_name = '{}_striatal_wf_type.csv'.format(mouse_id)
        mouse_results.to_csv(os.path.join(mouse_output_path, file_name))

    print('Striatal cell type classification done and saved for each mouse.')
    return


