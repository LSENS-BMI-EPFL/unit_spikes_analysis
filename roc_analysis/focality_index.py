#imports 
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
import allen_utils_old as allen
from iblatlas.atlas import BrainRegions
#stop pop ups when creating plots
matplotlib.use('Agg') 


#paths 
DATA_PATH = os.path.join("/Volumes", "Petersen-Lab", "z_LSENS", "Share", "Dana_Shayakhmetova", "new_roc_csv")
FIGURE_PATH = os.path.join("/Volumes", "Petersen-Lab", "z_LSENS", "Share", "Dana_Shayakhmetova", "new_focality_indices")

single_analyses = ['auditory_active', 'auditory_passive_pre', 'auditory_passive_post', 'baseline_choice', 'baseline_whisker_choice', 'choice', 'spontaneous_licks', 
                   'wh_vs_aud_pre_vs_post_learning', 'whisker_active', 'whisker_passive_pre', 'whisker_passive_post', 'whisker_choice']
reward_groups = ['R+', 'R-']
pair_analyses = [['whisker_active', 'auditory_active'], ['whisker_passive_pre', 'whisker_passive_post'], ['choice', 'whisker_choice'], ['baseline_whisker_choice', 'whisker_choice'], ['auditory_passive_pre', 'auditory_passive_post'], ['baseline_choice', 'choice']]


def filter_number_of_neurons(df, area_type, thres=15, mouse_thres=3):
    """
    Filter out brain areas based on anatomical, data-size, and subject-count criteria.

    This function removes:
      a) Pons and adjacent brainstem areas (defined in `excluded_areas`)
      b) Areas that have fewer than `thres` neurons in both reward groups
      c) Areas that have data from fewer than `mouse_thres` unique mice
    """
    # pons and adjacent brain stem areas
    excluded_areas = {"PRNr", "PRNc", "RM", "PPN", "V", "PSV", "PG", "LAV", "NLL", "SUT"}
    df = df[~df[area_type].isin(excluded_areas)]
    
    # neuron count
    counts = (
        df.groupby(['reward_group', area_type])['unit_id']
        .nunique()
        .reset_index(name='count')
    )
    pivot_counts = counts.pivot(index=area_type, columns='reward_group', values='count').fillna(0)
    areas_low_neurons = pivot_counts[(pivot_counts < thres).all(axis=1)].index.tolist()
    df = df[~df[area_type].isin(areas_low_neurons)]

    # mouse count
    if mouse_thres != 0:
        mouse_counts = (
            df.groupby(area_type)['mouse_id']
            .nunique()
            .reset_index(name='num_mice')
        )
        areas_low_mice = mouse_counts[mouse_counts['num_mice'] < mouse_thres][area_type].tolist()

        df = df[~df[area_type].isin(areas_low_mice)]
        print(f"Areas removed (mice < {mouse_thres}): {areas_low_mice}")

    print(f"Areas removed (count < {thres} neurons in both groups): {areas_low_neurons}")
    
    
    return df


def swanson_conversion(roc_df):
    """ Maps CCF acronyms to Swanson regions, using manual corrections and iblatlas."""
    br = BrainRegions()
    roc_df = roc_df.copy()
    roc_df['ccf_acronym'] = roc_df['ccf_acronym'].astype(str).str.strip()

    # Manual mapping for specific missing acronyms
    manual_mapping_dict = {
        'STR': 'STRv', 'HPF': 'CA1', 'OLF': 'AON', 'FRP6a': 'FRP', 'FRP5': 'FRP',
        'FRP4': 'FRP', 'FRP2/3': 'FRP', 'FRP1': 'FRP', 'MB': 'MRN', 'P': 'PRNr',
        'SSp-tr6a': 'SSp-tr', 'SSp-tr5': 'SSp-tr', 'SSp-tr4': 'SSp-tr',
        'SSp-tr2/3': 'SSp-tr', 'SSp-tr1': 'SSp-tr', 'ORBm6a': 'ORBm',
        'RSPagl6a': 'RSPd', 'RSPagl6b': 'RSPd', 'RSPagl5': 'RSPd',
        'RSPagl4': 'RSPd', 'RSPagl2/3': 'RSPd', 'RSPagl1': 'RSPd',
    }
    roc_df['ccf_acronym_mapped'] = roc_df['ccf_acronym'].replace(manual_mapping_dict)

    #just in case again
    ssp_bfd_mask = roc_df['ccf_acronym_mapped'].str.startswith('SSp-bfd')
    roc_df.loc[ssp_bfd_mask, 'ccf_acronym_mapped'] = 'SSp-bfd'

    unique_acronyms = roc_df['ccf_acronym_mapped'].dropna().unique()
    swanson_mapping = {}
    for acronym in unique_acronyms:
        if pd.isna(acronym) or acronym == 'nan':
            swanson_mapping[acronym] = np.nan
        else:
            try:
                mapped_acronym = br.acronym2acronym(acronym, mapping='Swanson')
                swanson_mapping[acronym] = mapped_acronym[0] if mapped_acronym.size > 0 else np.nan
            except Exception:
                swanson_mapping[acronym] = np.nan

    roc_df['swanson_region'] = roc_df['ccf_acronym_mapped'].map(swanson_mapping)
    # Filter out unwanted regions
    regions_to_remove = ['root', 'void', '', 'CTXsp','nan', 'HY']
    roc_df_filtered = roc_df.dropna(subset=['swanson_region']).copy()
    roc_df_filtered = roc_df_filtered[~roc_df_filtered['swanson_region'].isin(regions_to_remove)].copy()
    roc_df_filtered = roc_df_filtered.drop('ccf_acronym_mapped', axis=1)
    return roc_df_filtered

def compute_fi(df, area_type):
    """
    Compute the Focality Index (FI) for neuronal activity across brain areas.
    The Focality Index quantifies how selectivity is distributed across areas:
        FI = (Σ P_i²) / (Σ P_i)²
    where P_i is the proportion of significant neurons in area i.
    """
    area_sums = df.groupby(area_type).agg(
        sig_neurons_area=('significant', 'sum'),
        total_neurons_area=('unit_id', 'count')
    ).reset_index()
    area_sums['P_i'] = area_sums['sig_neurons_area'] / area_sums['total_neurons_area']

    if area_sums.empty:
        return np.nan

    sum_of_squares = (area_sums['P_i']**2).sum()
    square_of_sum  = (area_sums['P_i'].sum())**2
    if square_of_sum == 0:
        return np.nan
    fi = sum_of_squares / square_of_sum
    return fi

def plot_focality_index_by_reward_group(roc, analysis_type, area_type, bootstrap_method, save_dir):
    """ Computes and plots per-group focality index with different bootstrapped CIs."""
    os.makedirs(save_dir, exist_ok=True)
    roc = roc[roc.analysis_type == analysis_type]
    roc_rminus = roc[roc.reward_group == 'R-']
    roc_rplus  = roc[roc.reward_group == 'R+']

    common_areas = set(roc_rminus[area_type]) & set(roc_rplus[area_type])
    roc_rminus = roc_rminus[roc_rminus[area_type].isin(common_areas)]
    roc_rplus  = roc_rplus[roc_rplus[area_type].isin(common_areas)]

    N_areas = len(common_areas)
    uniform_fi = 1 / N_areas 
    
    if bootstrap_method == 'hierarchical':
        ci_fi_rminus = hierarchical_bootstrap_fi(roc_rminus, area_type)
        ci_fi_rplus  = hierarchical_bootstrap_fi(roc_rplus, area_type)
    elif bootstrap_method == 'bca':
        ci_fi_rminus = bca_bootstrap_fi(roc_rminus, area_type)
        ci_fi_rplus  = bca_bootstrap_fi(roc_rplus, area_type)
    else:
        #defaults to percentile
        ci_fi_rminus = percentile_bootstrap_fi(roc_rminus, area_type)
        ci_fi_rplus  = percentile_bootstrap_fi(roc_rplus, area_type)

    
    val_fi_rminus = compute_fi(roc_rminus, area_type)
    val_fi_rplus  = compute_fi(roc_rplus, area_type)

    reward_groups = ['R-', 'R+']
    fi_vals = [val_fi_rminus, val_fi_rplus]
    ci_values = [ci_fi_rminus, ci_fi_rplus]
    fi_errors = [[m - ci[0], ci[1] - m] for m, ci in zip(fi_vals, ci_values)]
    fi_errors = np.abs(np.array(fi_errors)).T


    plt.figure(figsize=(3, 5))
    ax = plt.gca()
    ax.errorbar(
        reward_groups, fi_vals, yerr=fi_errors, fmt='o',
        color='black', capsize=5, linestyle='none'
    )
    
    ax.axhline(y=uniform_fi, color='gray', linestyle='--', label=f'Uniform (1/N, N={N_areas})')
    max_ci_upper = max(ci_fi_rminus[1], ci_fi_rplus[1])
    y_offset = max_ci_upper * 0.05
    for i, (val, ci) in enumerate(zip(fi_vals, ci_values)):
        if np.isfinite(val):
            ax.text(i, ci[1] + y_offset, f"FI Score: {val:.3f}\n [{ci[0]:.3f}, {ci[1]:.3f}]",
                    ha='center', va='bottom', fontsize=8)
        
    ax.set_ylabel('Focality index', fontsize=12)
    ax.set_xlabel('Reward group', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.7, 1.7)
    plt.xticks(fontsize=12)
    plt.ylim(0, 0.06)
    plt.title(f"{analysis_type.replace('_', ' ').capitalize()}")
    plt.yticks(fontsize=12)
    plt.tight_layout()
    filename = os.path.join(save_dir, f"focality_index_bootstrapped_{analysis_type}")
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(f"{filename}.{ext}", dpi=500, bbox_inches='tight')

def plot_focality_index_by_analysis(roc, analyses, reward_group, area_type, bootstrap_method,save_dir):
    """Computes and plots the focality index (FI) across multiple analyses for a given reward group."""

    os.makedirs(save_dir, exist_ok=True)

    #filter data 
    roc = roc[roc.reward_group == reward_group]
    roc = roc[roc.analysis_type.isin(analyses)]
    fi_vals, fi_cis = [], []
    N_areas = roc[area_type].nunique() 
    uniform_fi = 1 / N_areas

    for analysis in analyses:
        roc_analysis = roc[roc.analysis_type == analysis]
        if bootstrap_method == 'hierarchical':
            ci_fi = hierarchical_bootstrap_fi(roc_analysis, area_type) 
        elif bootstrap_method == 'bca':
            ci_fi = bca_bootstrap_fi(roc_analysis, area_type) 
        else:
            #defaults to percentile (manual)
            ci_fi = percentile_bootstrap_fi(roc_analysis, area_type) 
        

        val_fi = compute_fi(roc_analysis, area_type)
        fi_vals.append(val_fi)
        fi_cis.append(ci_fi)

    fi_errors = [[m - ci[0], ci[1] - m] for m, ci in zip(fi_vals, fi_cis)]
    fi_errors = np.abs(np.array(fi_errors)).T

    plt.figure(figsize=(4, 5))
    ax = plt.gca()
    ax.errorbar(
        analyses, fi_vals, yerr=fi_errors, fmt='o',
        color='black', capsize=5, linestyle='none'
    )

    ax.axhline(y=uniform_fi, color='gray', linestyle='--', linewidth=1.5, label='Uniform FI (1/N)')
    finite_ci_uppers = [ci[1] for ci in fi_cis if np.isfinite(ci[1])]
    max_ci_upper = max(finite_ci_uppers)
    y_offset = max_ci_upper * 0.2
    for i, (mean, ci) in enumerate(zip(fi_vals, fi_cis)):
        if np.isfinite(mean):
             ax.text(i, ci[1] + y_offset, f"FI Score: {mean:.3f}\n[{ci[0]:.3f}, {ci[1]:.3f}]",
                    ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Focality Index', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.7, 1.7)
    plt.xticks(fontsize=8, rotation=45, ha="right")
    plt.ylim(0, 0.06)
    title = f"{analyses[0].replace('_', ' ').capitalize()} vs {analyses[1].replace('_', ' ').capitalize()} ({reward_group})\n"
    plt.title(title, fontsize=10)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    filename = os.path.join(save_dir, f"focality_index_bootstrapped_{analyses[0]}_{analyses[1]}_{reward_group}")
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(f"{filename}.{ext}", dpi=500, bbox_inches='tight')


## Bootstrap methods ##

### Hierarchical (Two-stage)
def focality_from_df_rows(df, area_col):
    """Compute the focality index (FI) for a given DataFrame and area column."""
    fi = compute_fi(df, area_col)
    return fi

def hierarchical_bootstrap_focality(df, category_col='area', n_boot=2000, seed=None):
    """Perform a hierarchical bootstrap of the focality index (FI) across mice.
    Resamples both mice and their neurons with replacement."""

    rng = np.random.default_rng(seed)
    mice = df['mouse_id'].unique()
    M = len(mice)
    groups = {m: df[df['mouse_id'] == m] for m in mice}
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sampled_mice = rng.choice(mice, size=M, replace=True)
        parts = []
        for m in sampled_mice:
            grp = groups[m]
            # sample rows within mouse with replacement (preserve count)
            sampled = grp.sample(len(grp), replace=True, random_state=rng.bit_generator)
            parts.append(sampled)
        boot_df = pd.concat(parts, ignore_index=True)
        boots[i] = focality_from_df_rows(boot_df, category_col)
    return boots

def mouse_level_bootstrap_focality(df, category_col='area', n_boot=2000, seed=None):
    """Perform mouse-level bootstrapping of the focality index (FI).
    Resamples mice with replacement to compute FI."""

    rng = np.random.default_rng(seed)
    mice = df['mouse_id'].unique()
    groups = {m: df[df['mouse_id'] == m] for m in mice}
    M = len(mice)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(mice, size=M, replace=True)
        parts = [groups[m] for m in sampled]
        boot_df = pd.concat(parts, ignore_index=True)
        boots[i] = focality_from_df_rows(boot_df, category_col)
    return boots

def hierarchical_bootstrap_fi(neuron_df, area_type, n_bootstraps=10000):
    """Compute 95% confidence intervals for the focality index (FI)
    using hierarchical bootstrapping across mice and neurons."""
    fi_bootstrap_values = hierarchical_bootstrap_focality(
        neuron_df, area_type, n_boot=n_bootstraps
    )
    ci_fi = np.percentile(fi_bootstrap_values, [2.5, 97.5])
    return ci_fi

###  Bias-corrected accelerator (BCa)
def bca_bootstrap_fi(neuron_df, area_type,  bootstrap_type='bca', n_bootstraps=10000):
    """Compute bias-corrected and accelerated (BCa) bootstrap confidence intervals
    for the focality index (FI)."""

    def fi_statistic_wrapper(indices):
        resampled_df = neuron_df.iloc[indices]
        return compute_fi(resampled_df,area_type)

    data = (np.arange(len(neuron_df)),)
    # data = (neuron_df,)
    res = bootstrap(
            data,
            statistic=fi_statistic_wrapper,
            n_resamples=n_bootstraps,
            method=bootstrap_type,  
            confidence_level=0.95,
            random_state=np.random.default_rng()
        )
    ci_fi = (res.confidence_interval.low, res.confidence_interval.high)
    return ci_fi

### Percentile (manual)
def percentile_bootstrap_fi(neuron_df, area_type, n_bootstraps=10000):
    """ Compute percentile bootstrap confidence intervals for the focality index (FI).
    Resamples neurons with replacement and estimates the 2.5th and 97.5th
    percentiles of the FI distribution."""
    fi_bootstrap_values = []
    n_neurons = len(neuron_df)

    for _ in range(n_bootstraps):
        resampled_df = neuron_df.sample(n=n_neurons, replace=True, ignore_index=True)
        bootstrap_fi_val= compute_fi(resampled_df, area_type)
        fi_bootstrap_values.append(bootstrap_fi_val)

    ci_fi = np.percentile(fi_bootstrap_values, [2.5, 97.5]) 
    return ci_fi


def main(args):
    # vars
    bootstrap_method = args.bootstrap_method
    area_type = args.area_type
    thres = args.thres
    mouse_thres = args.mouse_thres

    #pulling mice info 
    print("\nPulling mice information...")
    mouse_info_path = os.path.join("/Volumes", "Petersen-Lab", "z_LSENS", "Share", "Dana_Shayakhmetova", "mouse_info", "joint_mouse_reference_weight.xlsx")
    mouse_info_df = pd.read_excel(mouse_info_path)
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df = mouse_info_df[
        (mouse_info_df['exclude'] == 0) &
        (mouse_info_df['reward_group'].isin(['R+', 'R-'])) &
        (mouse_info_df['recording'] == 1)]
    print("Done.")

    #pulling ROC data 
    data_path_ab = os.path.join(DATA_PATH, "new_roc_csv_AB")
    data_path_mh = os.path.join(DATA_PATH, "new_roc_csv_MH")

    print("\nPulling all AB mice csv files...")
    roc_results_files = glob.glob(os.path.join(data_path_ab, '**', '*_roc_results_new.csv'), recursive=True) 
    roc_df_axel = pd.concat([pd.read_csv(f) for f in roc_results_files], ignore_index=True)
    print("Done.")

    print("\nPulling all MH mice csv files...")
    roc_results_files = glob.glob(os.path.join(data_path_mh, '**', '*_roc_results_new.csv'),recursive=True)
    roc_df_myriam = pd.concat([pd.read_csv(f) for f in roc_results_files], ignore_index=True)
    print("Done.")

    print("\nCombining data...")
    roc_df = pd.concat([roc_df_axel, roc_df_myriam], ignore_index=True)
    roc_df = roc_df.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')
    roc_df['neuron_id'] = roc_df.index.astype(int)
    print('Present mice:', roc_df['mouse_id'].unique(), 'Number of mice', roc_df['mouse_id'].nunique(), 'per reward group',
          roc_df.groupby('reward_group')['mouse_id'].nunique())
    print('ROC analysis types:', roc_df['analysis_type'].unique())
    excluded_mice = []
    roc_df = roc_df[~roc_df['mouse_id'].isin(excluded_mice)]
    print("Done.")


    #Creating area columns 
    print("\nCreating area column if needed...")
    if area_type == "swanson_region":
        roc_df = swanson_conversion(roc_df)
    elif area_type == "area_custom":
        roc_df = allen.create_area_custom_column(roc_df)
    #else defaults to ccf_parent_acronym 
    print("Done.")


    #Filtering data 
    print("\nFiltering roc data...")
    roc_df = filter_number_of_neurons(roc_df,area_type, thres, mouse_thres)
    print("Done.")


    # Creating all the FI plots
    print("\nGenerating focality index figures for single analyses per reward group......")
    for analysis in single_analyses:
        print(f"Processing analysis: {analysis}")
        plot_focality_index_by_reward_group(roc_df, analysis, area_type, bootstrap_method, FIGURE_PATH)
    print("Done.")

    print("\nGenerating focality index figures for paired analyses for a given reward group...")
    for r in reward_groups:
        print(f"Reward group: {r}")
        for pair in pair_analyses:
            print(f"Processing analysis pair: {pair}")
            plot_focality_index_by_analysis(roc_df, pair, r, area_type, bootstrap_method, FIGURE_PATH)
    print("Done.")

    print("\nAll FI analyses done.")
    return 

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute and plot focality indices from ROC results.")

    parser.add_argument(
        '--bootstrap_method',
        type=str,
        default='percentile',
        choices=['hierarchical', 'bca', 'percentile'],
        help='Bootstrap method to use for computing confidence intervals.'
    )

    parser.add_argument(
        '--area_type',
        type=str,
        default='swanson_region',
        choices=['swanson_region', 'area_custom', 'ccf_parent_acronym'],
        help='Column name indicating brain area type.'
    )

    parser.add_argument(
        '--thres',
        type=int,
        default=15,
        help='Minimum number of neurons required for filtering areas.'
    )

    parser.add_argument(
        '--mouse_thres',
        type=int,
        default=3,
        help='Minimum number of mice required for filtering areas.'
    ) #make 0 if no mouse-area filtering 


    args = parser.parse_args()

    print('Creating FI figures...')
    main(args)
    print('Done.')


# python focality_index.py --bootstrap_method percentile --area_type swanson_region --thres 15 --mouse_thres 3
