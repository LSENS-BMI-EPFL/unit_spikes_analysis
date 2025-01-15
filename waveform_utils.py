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
import NWB_reader_functions as nwb_reader
from plotting_utils import save_figure_with_options

from sklearn.mixture import GaussianMixture

def get_cortical_areas():
    """
    Retrieve a list of cortical area acronyms.
    :return: List of cortical area acronyms
    """
    return [
        'FRP', 'MOp', 'MOs', 'SSp-bfd', 'SSp-m', 'SSp-ul', 'SSp-ll', 'SSp-un', 'SSp-n', 'SSp-tr',
        'SSs', 'AUDp', 'AUDd', 'AUDv', 'ACA', 'ACAv', 'ACAd', 'VISa', 'VISp', 'VISam', 'VISl',
        'VISpm', 'VISrl', 'VISal', 'PL', 'ILA', 'ORB', 'RSP', 'RSPv', 'RSPd','RSPagl', 'TT', 'SCm',
        'SCsg', 'SCzo', 'SCiw', 'SCop', 'SCs', 'ORBm', 'ORBl', 'ORBvl', 'AId',
        'AIv', 'AIp', 'FRP', 'VISC'
    ]

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
    ctx_areas = get_cortical_areas()
    unit_table['area_acronym'] = [row['ccf_parent_acronym'] if row['ccf_parent_acronym'] in ctx_areas
                                  else row['ccf_acronym'] for idx, row in unit_table.iterrows()]

    # Same for area name
    unit_table['area_name'] = [row['ccf_parent_name'] if row['ccf_parent_name'] in ctx_areas
                                  else row['ccf_name'] for idx, row in unit_table.iterrows()]

    return unit_table

def get_filtered_area_list(unit_table, params):
    """
    Filter the list of areas present in the PETH table based on default exclusions.
    :param unit_table: PETH table with all mice data
    :return: List of filtered areas to plot
    """
    if params['area_nomenclature'] == 'area_acronym':
        # Get unique areas present in dataset
        areas_present = unit_table['area_acronym'].unique()

        # Exclude areas that are not brain regions
        excluded_areas = [
            'root', 'fiber tracts', 'grey', 'nan', 'fxs', 'lfbst', 'cc', 'mfbc', 'cst', 'fa',
            'VS','ar','ccb','int','or','ccs','cing','ec','em','fi','scwm','alv','chpl','opt',
            'VL',
        ]
        areas_present = [a for a in areas_present if a.lower() not in excluded_areas]

    elif params['area_nomenclature'] == 'target_region':
        areas_present = unit_table['target_region'].unique()


    return areas_present

def assign_rsu_vs_fsu(nwb_files, output_path):
    """
    This function assigns the unit type (FSU or RSU) to each unit using area-specific
    distributions obtained from a list of NWB files.
    :param nwb_files: List of neural data containing NWB files
    :param output_path: Path to save the results
    :return:
    """

    print('Assigning RSU vs FSU...')

    # Load data
    unit_data = []
    for nwb_file in nwb_files:
        try:
            unit_table = nwb_reader.get_unit_table(nwb_file)
            mouse_id = nwb_reader.get_mouse_id(nwb_file)
            beh, day = nwb_reader.get_bhv_type_and_training_day_index(nwb_file)
            unit_table['mouse_id'] = mouse_id
            unit_table['behaviour'] = beh
            unit_table['day'] = day
            unit_data.append(unit_table)
        except:
            continue
    unit_data = pd.concat(unit_data)

    # Get unique areas
    unit_data = process_area_acronyms(unit_data)
    area_list = get_filtered_area_list(unit_data, params={'area_nomenclature': 'area_acronym'})

    # Iterate over areas
    results = []
    for area in area_list:
        print('Processing area:', area)
        area_data = unit_data[(unit_data['area_acronym'] == area)
                                 & (unit_data['bc_label'] == 'good')]

        if area_data.empty or len(area_data) < 45:
            print(f'No good units found in {area}. Only MUA or non-somatic. Skipping...')
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
        print('GMM intersection roots', intersect)

        # Among all roots, keep the biologically-plausible one
        upper_lim = 0.8 # ms
        lower_lim = 0.2
        intersect_real = intersect[np.where(np.logical_and(intersect>=lower_lim, intersect<=upper_lim))[0]]

        if len(intersect_real) > 1:
            # Take the smallest root
            threshold = np.min(intersect_real)
        elif len(intersect_real) == 1:
            threshold = float(intersect_real[0])
        elif len(intersect_real) == 0:
            threshold = np.nan
            print('Could not find root/AP duration threshold in range 0.2-0.8 ms. Skipping...')

        # Plot results
        debug = False
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
            output_path_all = os.path.join(output_path, 'waveform_analysis')
            save_figure_with_options(figure=fig, file_formats=['png',], filename=fig_name, output_dir=output_path_all, dark_background=False)

        # Assign RSU vs FSU based on the duration column using apply
        area_data['waveform_type'] = area_data['duration'].apply(lambda x: 'fsu' if x<=threshold else 'rsu')
        results.append(area_data)

    # Recombine the results, keeping only identifiers and the waveform type
    results = pd.concat(results)

    # For each mouse, save the results
    mouse_list = results['mouse_id'].unique()
    for mouse_id in mouse_list:
        mouse_results = results[results['mouse_id'] == mouse_id]
        cols_to_keep = ['mouse_id', 'behaviour', 'day', 'electrode_group', 'cluster_id', 'waveform_type']
        mouse_results = mouse_results[cols_to_keep]
        behaviour = mouse_results['behaviour'].unique()[0]
        day = mouse_results['day'].unique()[0]

        mouse_output_path = os.path.join(output_path, mouse_id, f'{behaviour}_{day}', 'waveform_analysis')

        if not os.path.exists(mouse_output_path):
            os.makedirs(mouse_output_path)

        file_name = '{}_waveform_type.csv'.format(mouse_id)
        mouse_results.to_csv(os.path.join(mouse_output_path, file_name))





