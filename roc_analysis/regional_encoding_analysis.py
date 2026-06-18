"""
Complete Regional Encoding Analysis Pipeline

Tests which brain regions encode each variable, accounting for:
1. Multiple encoding variables (models)
2. Multiple reward groups (R+, R-)
3. Hierarchical structure (neurons within mice)
4. Multiple comparisons (FDR correction)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

def test_regional_encoding_hierarchical(region_data, all_data_in_group,
                                        baseline=0.05, n_bootstrap=100,
                                        n_permutations=100):
    """
    Test if a region encodes above baseline using hierarchical bootstrap and permutation.

    Parameters:
    -----------
    region_data : DataFrame
        Data for neurons in this specific region
    all_data_in_group : DataFrame
        Data for all neurons in this reward group (for permutation null)
    baseline : float
        Chance level (e.g., 0.05 for α=0.05)
    n_bootstrap : int
        Bootstrap iterations
    n_permutations : int
        Permutation iterations

    Returns:
    --------
    dict : Test results
    """

    if len(region_data) == 0:
        return None

    # Get mice in this region
    mice_in_region = region_data['mouse_id'].unique()
    n_mice = len(mice_in_region)

    if n_mice < 2:
        return None

    # Observed proportion
    observed_prop = region_data['significant'].mean()
    n_neurons = len(region_data)
    n_significant = region_data['significant'].sum()

    # ================================================================
    # BOOTSTRAP: Resample mice for confidence interval
    # ================================================================

    boot_props = []

    for _ in range(n_bootstrap):
        # Resample mice with replacement
        boot_mice = np.random.choice(mice_in_region, size=n_mice, replace=True)
        boot_data = region_data[region_data['mouse_id'].isin(boot_mice)]

        if len(boot_data) > 0:
            boot_prop = boot_data['significant'].mean()
            boot_props.append(boot_prop)

    boot_props = np.array(boot_props)

    # Compute CI using SE (to avoid negative error bars)
    boot_mean = np.mean(boot_props)
    boot_se = np.std(boot_props)
    ci_lower = boot_mean - 1.96 * boot_se
    ci_upper = boot_mean + 1.96 * boot_se

    # Test if CI excludes baseline
    encodes_ci = ci_lower > baseline

    # ================================================================
    # PERMUTATION: Shuffle mouse-region assignments
    # ================================================================

    all_mice = all_data_in_group['mouse_id'].unique()

    perm_props = []

    for _ in range(n_permutations):
        # Randomly select n_mice from all mice (permute which mice are "in region")
        perm_mice = np.random.choice(all_mice, size=n_mice, replace=False)
        perm_data = all_data_in_group[all_data_in_group['mouse_id'].isin(perm_mice)]

        if len(perm_data) > 0:
            perm_prop = perm_data['significant'].mean()
            perm_props.append(perm_prop)

    perm_props = np.array(perm_props)

    # One-sided p-value: observed >= permuted?
    p_value_perm = np.mean(perm_props >= observed_prop)

    # Z-score
    z_score = (observed_prop - np.mean(perm_props)) / np.std(perm_props) if np.std(perm_props) > 0 else np.nan

    return {
        'n_mice': n_mice,
        'n_neurons': n_neurons,
        'n_significant': n_significant,
        'observed_proportion': observed_prop,
        'bootstrap_mean': boot_mean,
        'bootstrap_se': boot_se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'perm_null_mean': np.mean(perm_props),
        'perm_null_std': np.std(perm_props),
        'p_value': p_value_perm,
        'z_score': z_score,
        'exceeds_baseline': observed_prop > baseline,
        'ci_excludes_baseline': encodes_ci
    }


# Add this new worker function (before analyze_regional_encoding_complete)
def _test_single_region_worker(args):
    """
    Worker function for multiprocessing. Tests one region.

    Parameters:
    -----------
    args : tuple
        (region, model_name, reward_group, region_data_dict, all_data_dict,
         baseline, n_bootstrap, n_permutations)
    """
    (region, model_name, reward_group, region_data_dict, all_data_dict,
     baseline, n_bootstrap, n_permutations) = args

    # Reconstruct dataframes from dictionaries (needed for multiprocessing)
    region_data = pd.DataFrame(region_data_dict)
    all_data = pd.DataFrame(all_data_dict)

    result = test_regional_encoding_hierarchical(
        region_data=region_data,
        all_data_in_group=all_data,
        baseline=baseline,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations
    )

    if result is not None:
        result['model_name'] = model_name
        result['reward_group'] = reward_group
        result['region'] = region
        return result
    return None

def analyze_regional_encoding_complete(data_df,
                                       area_column='area_acronym_custom',
                                       baseline=0.05,
                                       n_bootstrap=5000,
                                       n_permutations=10000,
                                       min_mice_per_region=2,
                                       progress_bar=True,
                                       n_jobs=None):
    """
    Complete regional encoding analysis across all models and groups.

    Tests which brain regions encode each variable for each reward group.

    Parameters:
    -----------
    data_df : DataFrame
        Must have columns: mouse_id, reward_group, model_name,
        area_acronym_custom (or specify area_column), significant (boolean)
    area_column : str
        Column name for brain areas
    baseline : float
        Chance level for encoding (default 0.05)
    n_bootstrap : int
        Bootstrap iterations (default 5000)
    n_permutations : int
        Permutation iterations (default 10000)
    min_mice_per_region : int
        Minimum mice required per region (default 2)
    progress_bar : bool
        Show progress bar

    Returns:
    --------
    DataFrame : Complete results with columns:
        - model_name
        - reward_group
        - region
        - n_mice, n_neurons, n_significant
        - observed_proportion
        - bootstrap_mean, bootstrap_se
        - ci_lower, ci_upper
        - p_value, p_fdr
        - z_score
        - encodes (boolean, after FDR correction)
    """

    # Filter out full model if present
    if 'full' in data_df['model_name'].values:
        data_df = data_df[data_df['model_name'] != 'full'].copy()

    print("\n" + "=" * 70)
    print("REGIONAL ENCODING ANALYSIS")
    print("=" * 70)
    print(f"Testing which regions encode each variable")
    print(f"Baseline: {baseline:.2%} (chance level)")
    print(f"Bootstrap iterations: {n_bootstrap}")
    print(f"Permutation iterations: {n_permutations}")
    print(f"Minimum mice per region: {min_mice_per_region}")
    print("=" * 70 + "\n")

    all_results = []

    # Get all unique combinations
    models = data_df['model_name'].unique()
    groups = data_df['reward_group'].unique()

    # Prepare tasks for parallel processing
    tasks = []

    for model_name in models:
        for reward_group in groups:

            subset = data_df[(data_df['model_name'] == model_name) &
                             (data_df['reward_group'] == reward_group)]

            if len(subset) == 0:
                continue

            regions = subset[area_column].unique()

            for region in regions:

                region_data = subset[subset[area_column] == region]
                n_mice = region_data['mouse_id'].nunique()

                if n_mice < min_mice_per_region:
                    continue

                # Convert to dict for multiprocessing (can't pickle complex objects)
                region_data_dict = region_data.to_dict('list')
                all_data_dict = subset.to_dict('list')

                tasks.append((
                    region, model_name, reward_group,
                    region_data_dict, all_data_dict,
                    baseline, n_bootstrap, n_permutations
                ))

    print(f"\nPrepared {len(tasks)} region tests to run in parallel...")

    # Determine number of processes
    if n_jobs is None:
        n_jobs = cpu_count() - 5

    print(f"Using {n_jobs} parallel processes")

    # Run in parallel
    if progress_bar:
        try:
            from tqdm import tqdm
            with Pool(processes=n_jobs) as pool:
                all_results = list(tqdm(
                    pool.imap(_test_single_region_worker, tasks),
                    total=len(tasks),
                    desc="Testing regions"
                ))
        except ImportError:
            with Pool(processes=n_jobs) as pool:
                all_results = pool.map(_test_single_region_worker, tasks)
    else:
        with Pool(processes=n_jobs) as pool:
            all_results = pool.map(_test_single_region_worker, tasks)

    # Filter out None results
    all_results = [r for r in all_results if r is not None]

    # Create results dataframe
    results_df = pd.DataFrame(all_results)

    if len(results_df) == 0:
        print("⚠️  No regions met minimum criteria for testing")
        return results_df

    # ================================================================
    # FDR CORRECTION (within each model-group combination)
    # ================================================================

    print("\nApplying FDR correction within each model-group combination...")

    results_df['p_fdr'] = np.nan

    for model_name in results_df['model_name'].unique():
        for reward_group in results_df['reward_group'].unique():

            mask = ((results_df['model_name'] == model_name) &
                    (results_df['reward_group'] == reward_group))

            subset_idx = results_df[mask].index

            if len(subset_idx) > 0:
                p_values = results_df.loc[subset_idx, 'p_value'].values
                _, p_fdr, _, _ = multipletests(p_values, method='fdr_bh')
                results_df.loc[subset_idx, 'p_fdr'] = p_fdr

    # Final encoding decision (after FDR correction)
    results_df['encodes'] = results_df['p_fdr'] < 0.05

    # Sort by model, group, and proportion
    results_df = results_df.sort_values(
        ['model_name', 'reward_group', 'observed_proportion'],
        ascending=[True, True, False]
    )

    # ================================================================
    # SUMMARY STATISTICS
    # ================================================================

    print("\n" + "=" * 70)
    print("SUMMARY BY MODEL AND GROUP")
    print("=" * 70)

    for model_name in sorted(results_df['model_name'].unique()):
        print(f"\n{model_name}:")

        for reward_group in sorted(results_df['reward_group'].unique()):
            model_group_data = results_df[
                (results_df['model_name'] == model_name) &
                (results_df['reward_group'] == reward_group)
                ]

            if len(model_group_data) == 0:
                continue

            n_regions_tested = len(model_group_data)
            n_regions_encoding = model_group_data['encodes'].sum()

            print(f"  {reward_group}: {n_regions_encoding}/{n_regions_tested} regions encode (after FDR)")

            # List encoding regions
            encoding_regions = model_group_data[model_group_data['encodes']].sort_values(
                'observed_proportion', ascending=False
            )

            if len(encoding_regions) > 0:
                for _, row in encoding_regions.iterrows():
                    print(f"    • {row['region']}: {row['observed_proportion']:.1%} "
                          f"[{row['ci_lower']:.1%}, {row['ci_upper']:.1%}], "
                          f"p_FDR={row['p_fdr']:.4f}, z={row['z_score']:.2f}")

    # ================================================================
    # COMPARISON BETWEEN GROUPS
    # ================================================================

    print("\n" + "=" * 70)
    print("GROUP COMPARISONS")
    print("=" * 70)

    for model_name in sorted(results_df['model_name'].unique()):
        model_data = results_df[results_df['model_name'] == model_name]

        # Get regions that encode in either group
        encoding_regions = model_data[model_data['encodes']]['region'].unique()

        if len(encoding_regions) == 0:
            continue

        print(f"\n{model_name}:")

        for region in sorted(encoding_regions):
            region_data = model_data[model_data['region'] == region]

            if len(region_data) == 2:  # Both groups present
                r_plus = region_data[region_data['reward_group'] == 'R+'].iloc[0]
                r_minus = region_data[region_data['reward_group'] == 'R-'].iloc[0]

                both_encode = r_plus['encodes'] and r_minus['encodes']
                only_plus = r_plus['encodes'] and not r_minus['encodes']
                only_minus = not r_plus['encodes'] and r_minus['encodes']

                if both_encode:
                    diff = r_plus['observed_proportion'] - r_minus['observed_proportion']
                    print(f"  {region}: Both groups encode "
                          f"(R+={r_plus['observed_proportion']:.1%}, "
                          f"R-={r_minus['observed_proportion']:.1%}, "
                          f"diff={diff:+.1%})")
                elif only_plus:
                    print(f"  {region}: Only R+ encodes "
                          f"(R+={r_plus['observed_proportion']:.1%}, "
                          f"R-={r_minus['observed_proportion']:.1%} n.s.)")
                elif only_minus:
                    print(f"  {region}: Only R- encodes "
                          f"(R+={r_plus['observed_proportion']:.1%} n.s., "
                          f"R-={r_minus['observed_proportion']:.1%})")

    print("\n" + "=" * 70)
    print(f"Analysis complete. Tested {len(results_df)} region-model-group combinations.")
    print("=" * 70)

    return results_df


def save_regional_encoding_results(results_df, saving_path):
    """
    Save results to CSV files with different views.
    """

    # Full results
    full_path = os.path.join(saving_path, 'regional_encoding_full_results.csv')
    results_df.to_csv(full_path, index=False)
    print(f"\nFull results saved to: {full_path}")

    # Summary: Only encoding regions
    encoding_only = results_df[results_df['encodes']].copy()
    encoding_path = os.path.join(saving_path, 'regional_encoding_significant_only.csv')
    encoding_only.to_csv(encoding_path, index=False)
    print(f"Significant regions saved to: {encoding_path}")

    # Pivot table: Regions x Models (for R+ and R- separately)
    for group in results_df['reward_group'].unique():
        group_data = results_df[results_df['reward_group'] == group]

        pivot = group_data.pivot_table(
            index='region',
            columns='model_name',
            values='observed_proportion',
            aggfunc='first'
        )

        pivot_path = os.path.join(saving_path, f'regional_encoding_pivot_{group}.csv')
        pivot.to_csv(pivot_path)
        print(f"{group} pivot table saved to: {pivot_path}")

    # Summary counts
    summary = results_df.groupby(['model_name', 'reward_group']).agg({
        'region': 'count',
        'encodes': 'sum',
        'observed_proportion': ['mean', 'std', 'max']
    }).round(3)

    summary_path = os.path.join(saving_path, 'regional_encoding_summary.csv')
    summary.to_csv(summary_path)
    print(f"Summary statistics saved to: {summary_path}")


# ================================================================
# USAGE EXAMPLE
# ================================================================

if __name__ == "__main__":
    print("""
    Regional Encoding Analysis - Complete Pipeline

    This code tests which brain regions encode each variable, for each
    reward group, accounting for hierarchical structure and multiple
    comparisons.

    Usage:
    ------
    from regional_encoding_analysis import analyze_regional_encoding_complete

    results = analyze_regional_encoding_complete(
        data_df,
        area_column='area_acronym_custom',
        baseline=0.05,
        n_bootstrap=5000,
        n_permutations=10000,
        min_mice_per_region=2
    )

    # Save results
    save_regional_encoding_results(results, saving_path='/your/path')

    Required columns in data_df:
    ----------------------------
    - mouse_id
    - reward_group ('R+', 'R-')
    - model_name (encoding variable)
    - area_acronym_custom (or your area column)
    - significant (boolean)

    Output:
    -------
    DataFrame with one row per region-model-group combination containing:
    - Test statistics (proportion, CI, p-values)
    - FDR-corrected significance
    - Effect sizes (z-scores)

    Statistical approach:
    --------------------
    1. Hierarchical bootstrap (resample mice) for CIs
    2. Permutation test (shuffle mouse-region assignments) for p-values
    3. FDR correction within each model-group combination
    4. Report regions where p_FDR < 0.05
    """)