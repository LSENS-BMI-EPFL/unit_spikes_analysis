"""
Single function for focality index analysis of GLM encoding variables.

Usage:
    from focality_analysis_simple import analyze_focality
    
    results = analyze_focality(data_df, saving_path='/your/path')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from scipy.stats import ks_2samp, mannwhitneyu


def gini_coefficient(fractions):
    """Compute Gini coefficient (0=distributed, 1=concentrated)."""
    fractions = np.array(fractions)
    fractions = fractions[~np.isnan(fractions)]
    if len(fractions) == 0 or np.sum(fractions) == 0:
        return np.nan
    sorted_fractions = np.sort(fractions)
    n = len(sorted_fractions)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_fractions)) / (n * np.sum(sorted_fractions)) - (n + 1) / n


def focality_index(fractions):
    """
    Compute focality index as sum of squared proportions over squared sum.

    Formula: sum(p_i^2) / (sum(p_i))^2
    """
    fractions = np.array(fractions)
    fractions = fractions[~np.isnan(fractions)]

    if len(fractions) == 0:
        return np.nan

    sum_fractions = np.sum(fractions)

    if sum_fractions == 0:
        return np.nan

    sum_squared = np.sum(fractions ** 2)
    squared_sum = sum_fractions ** 2

    focality = sum_squared / squared_sum

    return focality


import numpy as np
from scipy.stats import norm


def jackknife_estimates(data, area_column, focality_func):
    """
    Compute jackknife estimates by leaving out one neuron at a time.

    Parameters:
    -----------
    data : DataFrame
        Neuron-level data
    area_column : str
        Column name for areas
    focality_func : function
        Function to compute focality (e.g., focality_index)

    Returns:
    --------
    array : Jackknife estimates
    """
    n = len(data)
    jack_estimates = []

    # For computational efficiency, sample if dataset is very large
    if n > 1000:
        # Use systematic sampling for large datasets
        indices = np.arange(0, n, max(1, n // 1000))
    else:
        indices = np.arange(n)

    for i in indices:
        # Leave out observation i
        jack_data = data.drop(data.index[i])

        # Compute focality on jackknife sample
        area_fracs = jack_data.groupby(area_column)['significant'].mean()
        if len(area_fracs) > 0:
            fi = focality_func(area_fracs.values)
            if not np.isnan(fi):
                jack_estimates.append(fi)

    return np.array(jack_estimates)


def bias_corrected_accelerated_ci(boot_values, observed_value, jack_estimates=None, alpha=0.05):
    """
    Compute bias-corrected and accelerated (BCa) bootstrap confidence interval.

    Parameters:
    -----------
    boot_values : array
        Bootstrap distribution
    observed_value : float
        Observed statistic from original data
    jack_estimates : array, optional
        Jackknife estimates for acceleration. If None, uses BC (not BCa)
    alpha : float
        Significance level (default 0.05 for 95% CI)

    Returns:
    --------
    tuple : (ci_lower, ci_upper)
    """
    boot_values = np.array(boot_values)
    n_boot = len(boot_values)

    # Bias correction factor (z0)
    # Proportion of bootstrap estimates less than observed
    prop_less = np.mean(boot_values < observed_value)

    # Avoid edge cases
    prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)

    z0 = norm.ppf(prop_less)

    # Acceleration factor (a)
    if jack_estimates is not None and len(jack_estimates) > 0:
        # Compute acceleration from jackknife
        jack_mean = np.mean(jack_estimates)
        numerator = np.sum((jack_mean - jack_estimates) ** 3)
        denominator = 6 * (np.sum((jack_mean - jack_estimates) ** 2) ** 1.5)

        if denominator > 0:
            a = numerator / denominator
        else:
            a = 0
    else:
        # No jackknife - use bias-corrected (BC) only
        a = 0

    # Adjusted percentiles
    z_alpha = norm.ppf(alpha / 2)
    z_1alpha = norm.ppf(1 - alpha / 2)

    # BCa adjustment
    p_lower = norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    p_upper = norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

    # Clip to valid percentile range
    p_lower = np.clip(p_lower, 0, 1)
    p_upper = np.clip(p_upper, 0, 1)

    # Compute CI
    ci_lower = np.percentile(boot_values, p_lower * 100)
    ci_upper = np.percentile(boot_values, p_upper * 100)

    return ci_lower, ci_upper

def analyze_focality_old(data_df, saving_path=None, n_bootstrap=1000, n_permutations=1000):
    """
    Focality analysis: resample neurons to get distributions, test difference.

    For each group (R+ or R-):
      - Resample neurons with replacement (n_bootstrap times)
      - Compute focality index for each resample
      - Get distribution of focality values

    Statistical tests:
      - Permutation test on distribution means
      - Kolmogorov-Smirnov test for distribution difference
      - Effect size (Cohen's d)

    Parameters:
    -----------
    data_df : DataFrame
        Must have columns: mouse_id, reward_group ('R+' or 'R-'),
        model_name, area_acronym_custom, significant (boolean)
    saving_path : str
        Directory to save figures and results
    n_bootstrap : int
        Bootstrap iterations (resample neurons)
    n_permutations : int
        Permutation test iterations

    Returns:
    --------
    DataFrame with results for each variable
    """

    # Filter out full model
    data_df = data_df[data_df['model_name'] != 'full'].copy()

    print("\nFOCALITY ANALYSIS: RESAMPLING NEURONS")
    print("=" * 60)
    print("Creating FI distributions by resampling neurons")
    print("WARNING: Does not account for hierarchical structure")
    print("=" * 60 + "\n")

    results = []

    for model_name in data_df['model_name'].unique():
        subset = data_df[data_df['model_name'] == model_name]

        # ============================================================
        # 1. COMPUTE OBSERVED FOCALITY (pooled across all neurons)
        # ============================================================

        observed_focality = {}

        for reward_group in ['R+', 'R-']:
            group_data = subset[subset['reward_group'] == reward_group]

            if len(group_data) == 0:
                observed_focality[reward_group] = np.nan
                continue

            # Pool all neurons from all mice in this group
            area_fracs = group_data.groupby('area_acronym_custom')['significant'].mean()

            if len(area_fracs) > 0:
                focality = focality_index(area_fracs.values)
                observed_focality[reward_group] = focality
            else:
                observed_focality[reward_group] = np.nan

        # Skip if either group has no data
        if np.isnan(observed_focality.get('R+', np.nan)) or np.isnan(observed_focality.get('R-', np.nan)):
            continue

        obs_diff = observed_focality['R+'] - observed_focality['R-']

        # ============================================================
        # 2. BOOTSTRAP DISTRIBUTIONS (resample neurons)
        # ============================================================

        boot_focality = {'R+': [], 'R-': []}

        for reward_group in ['R+', 'R-']:
            group_data = subset[subset['reward_group'] == reward_group]
            n_neurons = len(group_data)

            if n_neurons < 10:  # Skip if too few neurons
                continue

            for _ in range(n_bootstrap):
                # Resample neurons with replacement
                boot_indices = np.random.choice(n_neurons, size=n_neurons, replace=True)
                boot_data = group_data.iloc[boot_indices]

                # Compute fraction significant per area
                area_fracs = boot_data.groupby('area_acronym_custom')['significant'].mean()

                if len(area_fracs) > 0:
                    focality = focality_index(area_fracs.values)
                    if not np.isnan(focality):
                        boot_focality[reward_group].append(focality)

        if len(boot_focality['R+']) == 0 or len(boot_focality['R-']) == 0:
            continue

        boot_focality['R+'] = np.array(boot_focality['R+'])
        boot_focality['R-'] = np.array(boot_focality['R-'])

        # Compute statistics from distributions
        r_plus_mean = np.mean(boot_focality['R+'])
        r_plus_std = np.std(boot_focality['R+'])
        r_plus_ci = np.percentile(boot_focality['R+'], [2.5, 97.5])

        r_minus_mean = np.mean(boot_focality['R-'])
        r_minus_std = np.std(boot_focality['R-'])
        r_minus_ci = np.percentile(boot_focality['R-'], [2.5, 97.5])

        # ============================================================
        # 3. STATISTICAL TESTS
        # ============================================================

        # Test 1: Permutation test on means
        boot_diff_mean = r_plus_mean - r_minus_mean

        # Permutation: pool both distributions and randomly split
        combined = np.concatenate([boot_focality['R+'], boot_focality['R-']])
        n_r_plus = len(boot_focality['R+'])

        perm_diffs = []
        for _ in range(n_permutations):
            shuffled = np.random.permutation(combined)
            perm_r_plus = shuffled[:n_r_plus]
            perm_r_minus = shuffled[n_r_plus:]
            perm_diffs.append(np.mean(perm_r_plus) - np.mean(perm_r_minus))

        perm_diffs = np.array(perm_diffs)
        p_value_perm = np.mean(np.abs(perm_diffs) >= np.abs(boot_diff_mean))

        # Test 2: Kolmogorov-Smirnov test (tests if distributions differ)
        ks_stat, p_value_ks = ks_2samp(boot_focality['R+'], boot_focality['R-'])

        # Test 3: Mann-Whitney U test on bootstrap distributions
        mw_stat, p_value_mw = mannwhitneyu(boot_focality['R+'], boot_focality['R-'],
                                           alternative='two-sided')

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((r_plus_std ** 2 + r_minus_std ** 2) / 2)
        cohens_d = boot_diff_mean / pooled_std if pooled_std > 0 else np.nan

        # Distribution overlap
        overlap = np.sum((boot_focality['R+'][:, None] <= boot_focality['R-']).any(axis=1)) / len(boot_focality['R+'])

        # ============================================================
        # 4. CREATE FIGURE
        # ============================================================

        if saving_path:
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

            colors = ['#d62728', '#1f77b4']

            # Panel 1: Distribution comparison (KDE)
            ax1 = fig.add_subplot(gs[0, 0])

            from scipy.stats import gaussian_kde

            # KDE for smooth visualization
            kde_plus = gaussian_kde(boot_focality['R+'])
            kde_minus = gaussian_kde(boot_focality['R-'])

            x_range = np.linspace(
                min(boot_focality['R+'].min(), boot_focality['R-'].min()),
                max(boot_focality['R+'].max(), boot_focality['R-'].max()),
                200
            )

            ax1.plot(x_range, kde_plus(x_range), color=colors[0], linewidth=2, label='R+')
            ax1.plot(x_range, kde_minus(x_range), color=colors[1], linewidth=2, label='R-')
            ax1.fill_between(x_range, kde_plus(x_range), alpha=0.3, color=colors[0])
            ax1.fill_between(x_range, kde_minus(x_range), alpha=0.3, color=colors[1])

            ax1.axvline(r_plus_mean, color=colors[0], linestyle='--', linewidth=1.5)
            ax1.axvline(r_minus_mean, color=colors[1], linestyle='--', linewidth=1.5)

            ax1.set_xlabel('Focality index', fontsize=12)
            ax1.set_ylabel('Density', fontsize=12)
            ax1.set_title(f'Bootstrap distributions\n({n_bootstrap} resamples of neurons)',
                          fontsize=12)
            ax1.legend()

            # Panel 2: Histograms
            ax2 = fig.add_subplot(gs[0, 1])

            ax2.hist(boot_focality['R+'], bins=50, alpha=0.6, color=colors[0],
                     density=True, label=f'R+: {r_plus_mean:.3f}±{r_plus_std:.3f}')
            ax2.hist(boot_focality['R-'], bins=50, alpha=0.6, color=colors[1],
                     density=True, label=f'R-: {r_minus_mean:.3f}±{r_minus_std:.3f}')

            ax2.set_xlabel('Focality index', fontsize=12)
            ax2.set_ylabel('Density', fontsize=12)
            ax2.set_title('Histogram comparison', fontsize=12)
            ax2.legend()

            # Panel 3: Box plots
            ax3 = fig.add_subplot(gs[0, 2])

            bp = ax3.boxplot([boot_focality['R+'], boot_focality['R-']],
                             labels=['R+', 'R-'],
                             patch_artist=True,
                             widths=0.6)

            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax3.set_ylabel('Focality index', fontsize=12)
            ax3.set_title('Box plot comparison', fontsize=12)

            # Add significance stars
            y_max = max(boot_focality['R+'].max(), boot_focality['R-'].max())
            if p_value_perm < 0.001:
                sig_text = '***'
            elif p_value_perm < 0.01:
                sig_text = '**'
            elif p_value_perm < 0.05:
                sig_text = '*'
            else:
                sig_text = 'n.s.'

            ax3.plot([1, 2], [y_max * 1.05, y_max * 1.05], 'k-', linewidth=1.5)
            ax3.text(1.5, y_max * 1.08, sig_text, ha='center', fontsize=16)

            # Panel 4: Permutation distribution
            ax4 = fig.add_subplot(gs[1, 0])

            ax4.hist(perm_diffs, bins=50, alpha=0.7, color='gray', density=True)
            ax4.axvline(boot_diff_mean, color='red', linestyle='--', linewidth=2,
                        label=f'Observed: {boot_diff_mean:.3f}')
            ax4.axvline(0, color='black', linestyle='-', linewidth=1)

            ax4.set_xlabel('Difference (R+ - R-)', fontsize=12)
            ax4.set_ylabel('Density', fontsize=12)
            ax4.set_title(f'Permutation test\n(p={p_value_perm:.4f})', fontsize=12)
            ax4.legend()

            # Panel 5: Spatial distribution
            ax5 = fig.add_subplot(gs[1, 1])
            spatial_data = []

            for group in ['R+', 'R-']:
                group_data = subset[subset['reward_group'] == group]
                area_fracs = group_data.groupby('area_acronym_custom')['significant'].mean()
                for area, frac in area_fracs.items():
                    spatial_data.append({
                        'area': area,
                        'reward_group': group,
                        'fraction': frac
                    })

            if spatial_data:
                spatial_df = pd.DataFrame(spatial_data)
                pivot_df = spatial_df.pivot(index='area', columns='reward_group', values='fraction')
                pivot_df.plot(kind='bar', ax=ax5, color=colors, width=0.8)
                ax5.set_xlabel('Brain area', fontsize=12)
                ax5.set_ylabel('Fraction significant', fontsize=12)
                ax5.set_title('Spatial distribution', fontsize=12)
                ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right', fontsize=10)
                ax5.legend(title='Group')

            # Panel 6: Summary statistics
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.axis('off')

            summary = f"STATISTICAL TESTS\n{'=' * 35}\n\n"
            summary += f"Observed focality:\n"
            summary += f"  R+: {observed_focality['R+']:.3f}\n"
            summary += f"  R-: {observed_focality['R-']:.3f}\n\n"

            summary += f"Bootstrap distributions:\n"
            summary += f"  R+: {r_plus_mean:.3f}±{r_plus_std:.3f}\n"
            summary += f"  R-: {r_minus_mean:.3f}±{r_minus_std:.3f}\n\n"

            summary += f"Permutation test:\n"
            summary += f"  p = {p_value_perm:.4f}\n\n"

            summary += f"KS test:\n"
            summary += f"  stat = {ks_stat:.3f}\n"
            summary += f"  p = {p_value_ks:.4f}\n\n"

            summary += f"Mann-Whitney U:\n"
            summary += f"  p = {p_value_mw:.4f}\n\n"

            summary += f"Effect size:\n"
            summary += f"  Cohen's d = {cohens_d:.2f}\n\n"

            if p_value_perm < 0.05:
                if boot_diff_mean > 0:
                    summary += "→ R+ MORE concentrated"
                else:
                    summary += "→ R- MORE concentrated"
            else:
                summary += "→ No significant difference"

            ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
                     verticalalignment='top', fontsize=10, family='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.suptitle(f'Focality Analysis: {model_name}\n(Resampling neurons)',
                         fontsize=14, fontweight='bold')

            for ext in ['png', 'pdf', 'svg']:
                fig.savefig(f"{saving_path}/focality_neurons_{model_name}.{ext}",
                            dpi=300, bbox_inches='tight')
            plt.close()

        # Store results
        results.append({
            'model_name': model_name,
            'observed_r_plus': observed_focality['R+'],
            'observed_r_minus': observed_focality['R-'],
            'r_plus_mean': r_plus_mean,
            'r_plus_std': r_plus_std,
            'r_plus_ci_lower': r_plus_ci[0],
            'r_plus_ci_upper': r_plus_ci[1],
            'r_minus_mean': r_minus_mean,
            'r_minus_std': r_minus_std,
            'r_minus_ci_lower': r_minus_ci[0],
            'r_minus_ci_upper': r_minus_ci[1],
            'difference': boot_diff_mean,
            'p_permutation': p_value_perm,
            'p_ks': p_value_ks,
            'p_mannwhitney': p_value_mw,
            'ks_statistic': ks_stat,
            'cohens_d': cohens_d,
            'n_r_plus_neurons': len(subset[subset['reward_group'] == 'R+']),
            'n_r_minus_neurons': len(subset[subset['reward_group'] == 'R-']),
            'n_r_plus_mice': subset[subset['reward_group'] == 'R+']['mouse_id'].nunique(),
            'n_r_minus_mice': subset[subset['reward_group'] == 'R-']['mouse_id'].nunique()
        })

    # Create results dataframe
    results_df = pd.DataFrame(results)



    # FDR correction (use permutation test p-values)
    if len(results_df) > 0:
        _, p_fdr, _, _ = multipletests(results_df['p_permutation'], method='fdr_bh')
        results_df['p_fdr'] = p_fdr
        results_df = results_df.sort_values('p_permutation')

        if saving_path:
            results_df.to_csv(f"{saving_path}/focality_neurons_results.csv", index=False)

        print("\n" + "=" * 70)
        print("FOCALITY RESULTS (RESAMPLING NEURONS)")
        print("=" * 70)

        for _, row in results_df.iterrows():
            print(f"\n{row['model_name']}:")
            print(f"  Observed: R+ = {row['observed_r_plus']:.3f}, R- = {row['observed_r_minus']:.3f}")
            print(f"  Bootstrap: R+ = {row['r_plus_mean']:.3f}±{row['r_plus_std']:.3f}, "
                  f"R- = {row['r_minus_mean']:.3f}±{row['r_minus_std']:.3f}")
            print(f"  Difference: {row['difference']:.3f}")
            print(f"  Permutation test: p = {row['p_permutation']:.4f}, p_FDR = {row['p_fdr']:.4f}")
            print(f"  KS test: p = {row['p_ks']:.4f}")
            print(f"  Cohen's d: {row['cohens_d']:.2f}")
            print(f"  Neurons: R+ n={row['n_r_plus_neurons']}, R- n={row['n_r_minus_neurons']}")

            if row['p_fdr'] < 0.05:
                if row['difference'] > 0:
                    print(f"  → R+ MORE concentrated than R- ***")
                else:
                    print(f"  → R- MORE concentrated than R+ ***")

    return results_df


def compute_group_focality(data, area_column='area_acronym_custom'):
    """
    Compute focality index for a group by pooling all neurons.

    Parameters:
    -----------
    data : DataFrame
        Subset of data for one group
    area_column : str
        Column name for brain areas

    Returns:
    --------
    float : Focality index
    """
    if len(data) == 0:
        return np.nan

    # Compute proportion significant per area (pooled across all neurons)
    area_fracs = data.groupby(area_column)['significant'].mean()

    if len(area_fracs) == 0:
        return np.nan

    return focality_index(area_fracs.values)


def bootstrap_group_focality(data, area_column='area_acronym_custom', n_bootstrap=5000):
    """
    Bootstrap with SE-based CIs instead of percentile CIs.
    """
    mice = data['mouse_id'].unique()
    n_mice = len(mice)

    if n_mice < 2:
        return None

    boot_values = []

    for _ in range(n_bootstrap):
        boot_mice = np.random.choice(mice, size=n_mice, replace=True)
        boot_data = data[data['mouse_id'].isin(boot_mice)]
        fi = compute_group_focality(boot_data, area_column)

        if not np.isnan(fi):
            boot_values.append(fi)

    boot_values = np.array(boot_values)

    # Use mean ± 1.96*SE for CI (normal approximation)
    boot_mean = np.mean(boot_values)
    boot_se = np.std(boot_values)

    ci_lower = boot_mean - 1.96 * boot_se
    ci_upper = boot_mean + 1.96 * boot_se

    return {
        'bootstrap_values': boot_values,
        'mean': boot_mean,
        'std': boot_se,
        'ci': (ci_lower, ci_upper)  # Now guaranteed to contain mean!
    }


def permutation_test(data, area_column='area_acronym_custom', n_permutations=10000):
    """
    Permutation test by shuffling mouse group labels.

    Parameters:
    -----------
    data : DataFrame
        Data for both groups (must have 'mouse_id' and 'reward_group')
    area_column : str
        Column name for brain areas
    n_permutations : int
        Number of permutations

    Returns:
    --------
    dict : {
        'observed_diff': observed difference,
        'perm_diffs': array of permuted differences,
        'p_value': two-sided p-value
    }
    """
    # Compute observed difference
    r_plus_data = data[data['reward_group'] == 'R+']
    r_minus_data = data[data['reward_group'] == 'R-']

    fi_plus_obs = compute_group_focality(r_plus_data, area_column)
    fi_minus_obs = compute_group_focality(r_minus_data, area_column)
    observed_diff = fi_plus_obs - fi_minus_obs

    # Create mouse-level mapping
    mouse_to_group = data[['mouse_id', 'reward_group']].drop_duplicates()
    all_mice = mouse_to_group['mouse_id'].values

    perm_diffs = []

    for _ in range(n_permutations):
        # Shuffle group labels at mouse level
        perm_groups = np.random.permutation(mouse_to_group['reward_group'].values)
        perm_mapping = dict(zip(all_mice, perm_groups))

        # Apply permuted labels
        data_perm = data.copy()
        data_perm['reward_group_perm'] = data_perm['mouse_id'].map(perm_mapping)

        # Compute focality for permuted groups
        fi_plus_perm = compute_group_focality(
            data_perm[data_perm['reward_group_perm'] == 'R+'],
            area_column
        )
        fi_minus_perm = compute_group_focality(
            data_perm[data_perm['reward_group_perm'] == 'R-'],
            area_column
        )

        perm_diff = fi_plus_perm - fi_minus_perm

        if not np.isnan(perm_diff):
            perm_diffs.append(perm_diff)

    perm_diffs = np.array(perm_diffs)

    # Two-sided p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    return {
        'observed_diff': observed_diff,
        'perm_diffs': perm_diffs,
        'p_value': p_value
    }

def analyze_focality(data_df, saving_path=None, area_column='area_acronym_custom',
                     n_bootstrap=5000, n_permutations=10000):
    """
    Complete focality analysis using Approach 2: Group-level with mouse-resampling.

    Parameters:
    -----------
    data_df : DataFrame
        Must have columns: mouse_id, reward_group ('R+' or 'R-'),
        model_name, area_acronym_custom (or specify area_column), significant (boolean)
    saving_path : str, optional
        Directory to save results and figures
    area_column : str
        Column name for brain areas
    n_bootstrap : int
        Bootstrap iterations (default 5000)
    n_permutations : int
        Permutation test iterations (default 10000)

    Returns:
    --------
    DataFrame : Results in long format with reward_group column
    """

    # Filter out full model if present
    if 'full' in data_df['model_name'].values:
        data_df = data_df[data_df['model_name'] != 'full'].copy()

    print("\n" + "=" * 70)
    print("FOCALITY ANALYSIS: GROUP-LEVEL WITH MOUSE-RESAMPLING")
    print("=" * 70)
    print("Statistical approach:")
    print("  1. Compute observed focality (pooled neurons per group)")
    print("  2. Bootstrap CIs (resample mice)")
    print("  3. Permutation test (shuffle mouse labels)")
    print("=" * 70 + "\n")

    all_results = []
    test_results = []

    for model_name in data_df['model_name'].unique():
        print(f"\nProcessing {model_name}...")
        subset = data_df[data_df['model_name'] == model_name]

        # Check we have both groups
        groups_present = subset['reward_group'].unique()
        if 'R+' not in groups_present or 'R-' not in groups_present:
            print(f"  ⚠️  Skipping - missing one or both groups")
            continue

        # ================================================================
        # STEP 1: COMPUTE OBSERVED FOCALITY (POOLED)
        # ================================================================

        print(f"  Step 1: Computing observed focality...")

        observed_fi = {}
        for group in ['R+', 'R-']:
            group_data = subset[subset['reward_group'] == group]
            fi = compute_group_focality(group_data, area_column)
            observed_fi[group] = fi

            n_mice = group_data['mouse_id'].nunique()
            n_neurons = len(group_data)
            print(f"    {group}: FI = {fi:.4f} ({n_mice} mice, {n_neurons} neurons)")

        if np.isnan(observed_fi['R+']) or np.isnan(observed_fi['R-']):
            print(f"  ⚠️  Skipping - unable to compute focality")
            continue

        observed_diff = observed_fi['R+'] - observed_fi['R-']
        print(f"    Difference: {observed_diff:.4f}")

        # ================================================================
        # STEP 2: BOOTSTRAP CONFIDENCE INTERVALS
        # ================================================================

        print(f"  Step 2: Bootstrap resampling ({n_bootstrap} iterations)...")

        bootstrap_results = {}
        for group in ['R+', 'R-']:
            group_data = subset[subset['reward_group'] == group]
            boot_result = bootstrap_group_focality(group_data, area_column, n_bootstrap)
            bootstrap_results[group] = boot_result

            if boot_result:
                print(f"    {group}: mean = {boot_result['mean']:.4f} ± {boot_result['std']:.4f}, "
                      f"95% CI = [{boot_result['ci'][0]:.4f}, {boot_result['ci'][1]:.4f}]")

        # Check if observed is in bootstrap CI
        for group in ['R+', 'R-']:
            if bootstrap_results[group]:
                ci = bootstrap_results[group]['ci']
                obs = observed_fi[group]
                in_ci = ci[0] <= obs <= ci[1]
                if not in_ci:
                    print(f"    ⚠️  {group}: Observed value outside bootstrap CI (bias detected)")

        # Compute difference distribution and CI
        if bootstrap_results['R+'] and bootstrap_results['R-']:
            boot_diffs = (bootstrap_results['R+']['bootstrap_values'] -
                          bootstrap_results['R-']['bootstrap_values'])
            diff_ci = tuple(np.percentile(boot_diffs, [2.5, 97.5]))
            print(f"    Difference: 95% CI = [{diff_ci[0]:.4f}, {diff_ci[1]:.4f}]")

        # ================================================================
        # STEP 3: PERMUTATION TEST
        # ================================================================

        print(f"  Step 3: Permutation test ({n_permutations} iterations)...")

        perm_result = permutation_test(subset, area_column, n_permutations)
        p_value = perm_result['p_value']

        print(f"    Observed difference: {perm_result['observed_diff']:.4f}")
        print(f"    Permutation p-value: {p_value:.4f}")

        # ================================================================
        # STEP 4: EFFECT SIZE
        # ================================================================

        if bootstrap_results['R+'] and bootstrap_results['R-']:
            pooled_std = np.sqrt(
                (bootstrap_results['R+']['std'] ** 2 +
                 bootstrap_results['R-']['std'] ** 2) / 2
            )
            cohens_d = observed_diff / pooled_std if pooled_std > 0 else np.nan
            print(f"    Cohen's d: {cohens_d:.2f}")
        else:
            cohens_d = np.nan

        # ================================================================
        # STEP 5: STORE RESULTS
        # ================================================================

        for group in ['R+', 'R-']:
            boot = bootstrap_results[group]

            all_results.append({
                'model_name': model_name,
                'reward_group': group,
                'observed_focality': observed_fi[group],
                'bootstrap_mean': boot['mean'] if boot else np.nan,
                'bootstrap_std': boot['std'] if boot else np.nan,
                'ci_lower': boot['ci'][0] if boot else np.nan,
                'ci_upper': boot['ci'][1] if boot else np.nan,
                'n_mice': subset[subset['reward_group'] == group]['mouse_id'].nunique(),
                'n_neurons': len(subset[subset['reward_group'] == group]),
                'n_areas': subset[subset['reward_group'] == group][area_column].nunique()
            })

        test_results.append({
            'model_name': model_name,
            'difference': observed_diff,
            'p_value': p_value,
            'cohens_d': cohens_d
        })

    # ================================================================
    # CREATE RESULTS DATAFRAMES
    # ================================================================

    results_df = pd.DataFrame(all_results)
    test_df = pd.DataFrame(test_results)

    # Merge test results
    results_df = results_df.merge(test_df, on='model_name', how='left')

    # ================================================================
    # FDR CORRECTION
    # ================================================================

    if len(test_df) > 0:
        print("\n" + "=" * 70)
        print("FDR CORRECTION")
        print("=" * 70)

        _, p_fdr, _, _ = multipletests(test_df['p_value'], method='fdr_bh')

        # Map back to results
        model_to_p_fdr = dict(zip(test_df['model_name'], p_fdr))
        results_df['p_fdr'] = results_df['model_name'].map(model_to_p_fdr)

    # ================================================================
    # SAVE RESULTS
    # ================================================================

    if saving_path:
        results_df.to_csv(f"{saving_path}/focality_results.csv", index=False)
        print(f"\nResults saved to: {saving_path}/focality_results.csv")

    # ================================================================
    # PRINT SUMMARY
    # ================================================================

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for model in results_df['model_name'].unique():
        model_data = results_df[results_df['model_name'] == model]

        print(f"\n{model}:")

        for _, row in model_data.iterrows():
            if row['reward_group'] == 'R+':
                print(f"  R+ (n={row['n_mice']} mice): "
                      f"Observed FI = {row['observed_focality']:.3f}, "
                      f"Bootstrap = {row['bootstrap_mean']:.3f} ± {row['bootstrap_std']:.3f}, "
                      f"95% CI [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
            else:
                print(f"  R- (n={row['n_mice']} mice): "
                      f"Observed FI = {row['observed_focality']:.3f}, "
                      f"Bootstrap = {row['bootstrap_mean']:.3f} ± {row['bootstrap_std']:.3f}, "
                      f"95% CI [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")

        # Get test results
        test_row = model_data.iloc[0]
        print(f"  Difference: {test_row['difference']:.3f}")
        print(f"  p = {test_row['p_value']:.4f}, p_FDR = {test_row['p_fdr']:.4f}, d = {test_row['cohens_d']:.2f}")

        if test_row['p_fdr'] < 0.05:
            if test_row['difference'] > 0:
                print(f"  → R+ significantly MORE concentrated than R- ***")
            else:
                print(f"  → R- significantly MORE concentrated than R+ ***")
        else:
            print(f"  → No significant difference")

    return results_df

def analyze_focality_with_bca(data_df, saving_path=None, n_bootstrap=1000,
                              n_permutations=100, use_jackknife=False):
    """
    Focality analysis with bias-corrected accelerated (BCa) bootstrap CIs.

    Modified version that uses BCa instead of percentile CIs.
    """

    # Filter out full model
    data_df = data_df[data_df['model_name'] != 'full'].copy()

    print("\nFOCALITY ANALYSIS: BCa BOOTSTRAP")
    print("=" * 60)
    print("Using bias-corrected accelerated confidence intervals")
    print("=" * 60 + "\n")

    results = []

    for model_name in data_df['model_name'].unique():
        subset = data_df[data_df['model_name'] == model_name]

        # Get area column (assume 'area_acronym_custom')
        area_column = 'area_acronym_custom'

        # ============================================================
        # 1. COMPUTE OBSERVED FOCALITY
        # ============================================================

        observed_focality = {}

        for reward_group in ['R+', 'R-']:
            group_data = subset[subset['reward_group'] == reward_group]

            if len(group_data) == 0:
                observed_focality[reward_group] = np.nan
                continue

            area_fracs = group_data.groupby(area_column)['significant'].mean()

            if len(area_fracs) > 0:
                focality = focality_index(area_fracs.values)
                observed_focality[reward_group] = focality
            else:
                observed_focality[reward_group] = np.nan

        if np.isnan(observed_focality.get('R+', np.nan)) or np.isnan(observed_focality.get('R-', np.nan)):
            continue

        obs_diff = observed_focality['R+'] - observed_focality['R-']

        # ============================================================
        # 2. BOOTSTRAP DISTRIBUTIONS
        # ============================================================

        boot_focality = {'R+': [], 'R-': []}
        jack_estimates = {'R+': None, 'R-': None}

        for reward_group in ['R+', 'R-']:
            group_data = subset[subset['reward_group'] == reward_group]
            n_neurons = len(group_data)

            if n_neurons < 10:
                continue

            # Jackknife for acceleration (optional - can be slow)
            if use_jackknife:
                print(f"  Computing jackknife for {reward_group} {model_name}...")
                jack_estimates[reward_group] = jackknife_estimates(
                    group_data, area_column, focality_index
                )

            # Bootstrap
            print(f"  Bootstrapping {reward_group} {model_name}...")
            for _ in range(n_bootstrap):
                boot_indices = np.random.choice(n_neurons, size=n_neurons, replace=True)
                boot_data = group_data.iloc[boot_indices]

                area_fracs = boot_data.groupby(area_column)['significant'].mean()

                if len(area_fracs) > 0:
                    focality = focality_index(area_fracs.values)
                    if not np.isnan(focality):
                        boot_focality[reward_group].append(focality)

        if len(boot_focality['R+']) == 0 or len(boot_focality['R-']) == 0:
            continue

        boot_focality['R+'] = np.array(boot_focality['R+'])
        boot_focality['R-'] = np.array(boot_focality['R-'])

        # ============================================================
        # 3. COMPUTE BCa CONFIDENCE INTERVALS
        # ============================================================

        print(f"  Computing BCa CIs for {model_name}...")

        r_plus_mean = np.mean(boot_focality['R+'])
        r_plus_std = np.std(boot_focality['R+'])

        # BCa CI for R+
        r_plus_ci = bias_corrected_accelerated_ci(
            boot_focality['R+'],
            observed_focality['R+'],
            jack_estimates['R+']
        )

        r_minus_mean = np.mean(boot_focality['R-'])
        r_minus_std = np.std(boot_focality['R-'])

        # BCa CI for R-
        r_minus_ci = bias_corrected_accelerated_ci(
            boot_focality['R-'],
            observed_focality['R-'],
            jack_estimates['R-']
        )

        # Check if observed value is in CI
        r_plus_in_ci = r_plus_ci[0] <= observed_focality['R+'] <= r_plus_ci[1]
        r_minus_in_ci = r_minus_ci[0] <= observed_focality['R-'] <= r_minus_ci[1]

        # Compute bootstrap bias
        r_plus_bias = r_plus_mean - observed_focality['R+']
        r_minus_bias = r_minus_mean - observed_focality['R-']

        print(
            f"    R+ observed: {observed_focality['R+']:.4f}, CI: [{r_plus_ci[0]:.4f}, {r_plus_ci[1]:.4f}], in CI: {r_plus_in_ci}")
        print(
            f"    R- observed: {observed_focality['R-']:.4f}, CI: [{r_minus_ci[0]:.4f}, {r_minus_ci[1]:.4f}], in CI: {r_minus_in_ci}")
        print(f"    Bootstrap bias: R+ {r_plus_bias:.4f}, R- {r_minus_bias:.4f}")

        # ============================================================
        # 4. STATISTICAL TESTS (same as before)
        # ============================================================

        boot_diff_mean = r_plus_mean - r_minus_mean

        # Permutation test
        combined = np.concatenate([boot_focality['R+'], boot_focality['R-']])
        n_r_plus = len(boot_focality['R+'])

        perm_diffs = []
        for _ in range(n_permutations):
            shuffled = np.random.permutation(combined)
            perm_r_plus = shuffled[:n_r_plus]
            perm_r_minus = shuffled[n_r_plus:]
            perm_diffs.append(np.mean(perm_r_plus) - np.mean(perm_r_minus))

        perm_diffs = np.array(perm_diffs)
        p_value_perm = np.mean(np.abs(perm_diffs) >= np.abs(boot_diff_mean))

        # KS test
        from scipy.stats import ks_2samp, mannwhitneyu
        ks_stat, p_value_ks = ks_2samp(boot_focality['R+'], boot_focality['R-'])

        # Mann-Whitney
        mw_stat, p_value_mw = mannwhitneyu(boot_focality['R+'], boot_focality['R-'],
                                           alternative='two-sided')

        # Cohen's d
        pooled_std = np.sqrt((r_plus_std ** 2 + r_minus_std ** 2) / 2)
        cohens_d = boot_diff_mean / pooled_std if pooled_std > 0 else np.nan

        # ============================================================
        # 5. STORE RESULTS
        # ============================================================

        results.append({
            'model_name': model_name,
            'reward_group': 'R+',
            'observed_focality': observed_focality['R+'],
            'focality_mean': r_plus_mean,
            'focality_std': r_plus_std,
            'ci_lower': r_plus_ci[0],
            'ci_upper': r_plus_ci[1],
            'ci_type': 'BCa',
            'observed_in_ci': r_plus_in_ci,
            'bootstrap_bias': r_plus_bias,
            'n_neurons': len(subset[subset['reward_group'] == 'R+']),
            'n_mice': subset[subset['reward_group'] == 'R+']['mouse_id'].nunique(),
            'difference': boot_diff_mean,
            'p_permutation': p_value_perm,
            'p_ks': p_value_ks,
            'p_mannwhitney': p_value_mw,
            'ks_statistic': ks_stat,
            'cohens_d': cohens_d
        })

        results.append({
            'model_name': model_name,
            'reward_group': 'R-',
            'observed_focality': observed_focality['R-'],
            'focality_mean': r_minus_mean,
            'focality_std': r_minus_std,
            'ci_lower': r_minus_ci[0],
            'ci_upper': r_minus_ci[1],
            'ci_type': 'BCa',
            'observed_in_ci': r_minus_in_ci,
            'bootstrap_bias': r_minus_bias,
            'n_neurons': len(subset[subset['reward_group'] == 'R-']),
            'n_mice': subset[subset['reward_group'] == 'R-']['mouse_id'].nunique(),
            'difference': boot_diff_mean,
            'p_permutation': p_value_perm,
            'p_ks': p_value_ks,
            'p_mannwhitney': p_value_mw,
            'ks_statistic': ks_stat,
            'cohens_d': cohens_d
        })

    results_df = pd.DataFrame(results)

    # FDR correction
    if len(results_df) > 0:
        from statsmodels.stats.multitest import multipletests

        # Get unique models for FDR correction
        unique_models = results_df['model_name'].unique()
        p_values = []
        for model in unique_models:
            p_val = results_df[results_df['model_name'] == model]['p_permutation'].iloc[0]
            p_values.append(p_val)

        _, p_fdr, _, _ = multipletests(p_values, method='fdr_bh')

        # Map back to results
        model_to_p_fdr = dict(zip(unique_models, p_fdr))
        results_df['p_fdr'] = results_df['model_name'].map(model_to_p_fdr)

        if saving_path:
            results_df.to_csv(f"{saving_path}/focality_bca_results.csv", index=False)

    return results_df


def bootstrap_group_focality_neurons(data, area_column='area_acronym_custom', n_bootstrap=5000):
    """
    Bootstrap confidence interval by resampling NEURONS (not mice).

    WARNING: This ignores hierarchical structure. Neurons within same mouse
    are correlated, so this approach is anti-conservative.

    Parameters:
    -----------
    data : DataFrame
        Data for one group
    area_column : str
        Column name for brain areas
    n_bootstrap : int
        Number of bootstrap iterations

    Returns:
    --------
    dict : {
        'bootstrap_values': array of FI values,
        'mean': bootstrap mean,
        'se': bootstrap standard error,
        'ci': (lower, upper) 95% CI using SE
    }
    """
    n_neurons = len(data)

    if n_neurons < 10:
        return None

    boot_values = []

    for _ in range(n_bootstrap):
        # Resample neurons with replacement
        boot_indices = np.random.choice(n_neurons, size=n_neurons, replace=True)
        boot_data = data.iloc[boot_indices]

        # Compute focality
        area_fracs = boot_data.groupby(area_column)['significant'].mean()

        if len(area_fracs) > 0:
            fi = focality_index(area_fracs.values)
            if not np.isnan(fi):
                boot_values.append(fi)

    boot_values = np.array(boot_values)

    # Use mean ± 1.96*SE for CI (guaranteed to contain mean)
    boot_mean = np.mean(boot_values)
    boot_se = np.std(boot_values)

    ci_lower = boot_mean - 1.96 * boot_se
    ci_upper = boot_mean + 1.96 * boot_se

    return {
        'bootstrap_values': boot_values,
        'mean': boot_mean,
        'se': boot_se,
        'ci': (ci_lower, ci_upper)
    }


def permutation_test_neurons(data, area_column='area_acronym_custom', n_permutations=10000):
    """
    Permutation test by shuffling neuron group labels (not mouse labels).

    WARNING: This ignores hierarchical structure.

    Parameters:
    -----------
    data : DataFrame
        Data for both groups (must have 'reward_group')
    area_column : str
        Column name for brain areas
    n_permutations : int
        Number of permutations

    Returns:
    --------
    dict : {
        'observed_diff': observed difference,
        'perm_diffs': array of permuted differences,
        'p_value': two-sided p-value
    }
    """
    # Compute observed difference
    r_plus_data = data[data['reward_group'] == 'R+']
    r_minus_data = data[data['reward_group'] == 'R-']

    fi_plus_obs = compute_group_focality(r_plus_data, area_column)
    fi_minus_obs = compute_group_focality(r_minus_data, area_column)
    observed_diff = fi_plus_obs - fi_minus_obs

    # Permutation: shuffle neuron labels
    n_neurons = len(data)
    group_labels = data['reward_group'].values

    perm_diffs = []

    for _ in range(n_permutations):
        # Shuffle group labels at neuron level
        perm_labels = np.random.permutation(group_labels)

        # Create permuted data
        data_perm = data.copy()
        data_perm['reward_group_perm'] = perm_labels

        # Compute focality for permuted groups
        fi_plus_perm = compute_group_focality(
            data_perm[data_perm['reward_group_perm'] == 'R+'],
            area_column
        )
        fi_minus_perm = compute_group_focality(
            data_perm[data_perm['reward_group_perm'] == 'R-'],
            area_column
        )

        perm_diff = fi_plus_perm - fi_minus_perm

        if not np.isnan(perm_diff):
            perm_diffs.append(perm_diff)

    perm_diffs = np.array(perm_diffs)

    # Two-sided p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    return {
        'observed_diff': observed_diff,
        'perm_diffs': perm_diffs,
        'p_value': p_value
    }


def analyze_focality_neurons(data_df, saving_path=None, area_column='area_acronym_custom',
                             n_bootstrap=5000, n_permutations=10000):
    """
    Focality analysis by resampling NEURONS (not mice).

    WARNING: This approach ignores hierarchical structure (neurons nested in mice).
    Use for exploratory analysis only. For rigorous inference, use mouse-level resampling.

    Parameters:
    -----------
    data_df : DataFrame
        Must have columns: mouse_id, reward_group ('R+' or 'R-'),
        model_name, area_acronym_custom (or specify area_column), significant (boolean)
    saving_path : str, optional
        Directory to save results and figures
    area_column : str
        Column name for brain areas
    n_bootstrap : int
        Bootstrap iterations (default 5000)
    n_permutations : int
        Permutation test iterations (default 10000)

    Returns:
    --------
    DataFrame : Results in long format with reward_group column
    """

    # Filter out full model if present
    if 'full' in data_df['model_name'].values:
        data_df = data_df[data_df['model_name'] != 'full'].copy()

    print("\n" + "=" * 70)
    print("FOCALITY ANALYSIS: RESAMPLING NEURONS (NOT MICE)")
    print("=" * 70)
    print("⚠️  WARNING: This approach ignores hierarchical structure")
    print("⚠️  Neurons within mice are correlated - not independent")
    print("⚠️  Use for exploratory analysis only")
    print("\nStatistical approach:")
    print("  1. Compute observed focality (pooled neurons per group)")
    print("  2. Bootstrap CIs (resample neurons with replacement)")
    print("  3. Permutation test (shuffle neuron labels)")
    print("  4. SE-based CIs to avoid negative error bars")
    print("=" * 70 + "\n")

    all_results = []
    test_results = []

    for model_name in data_df['model_name'].unique():
        print(f"\nProcessing {model_name}...")
        subset = data_df[data_df['model_name'] == model_name]

        # Check we have both groups
        groups_present = subset['reward_group'].unique()
        if 'R+' not in groups_present or 'R-' not in groups_present:
            print(f"  ⚠️  Skipping - missing one or both groups")
            continue

        # ================================================================
        # STEP 1: COMPUTE OBSERVED FOCALITY (POOLED)
        # ================================================================

        print(f"  Step 1: Computing observed focality...")

        observed_fi = {}
        for group in ['R+', 'R-']:
            group_data = subset[subset['reward_group'] == group]
            fi = compute_group_focality(group_data, area_column)
            observed_fi[group] = fi

            n_mice = group_data['mouse_id'].nunique()
            n_neurons = len(group_data)
            print(f"    {group}: FI = {fi:.4f} ({n_mice} mice, {n_neurons} neurons)")

        if np.isnan(observed_fi['R+']) or np.isnan(observed_fi['R-']):
            print(f"  ⚠️  Skipping - unable to compute focality")
            continue

        observed_diff = observed_fi['R+'] - observed_fi['R-']
        print(f"    Difference: {observed_diff:.4f}")

        # ================================================================
        # STEP 2: BOOTSTRAP CONFIDENCE INTERVALS (RESAMPLE NEURONS)
        # ================================================================

        print(f"  Step 2: Bootstrap resampling neurons ({n_bootstrap} iterations)...")

        bootstrap_results = {}
        for group in ['R+', 'R-']:
            group_data = subset[subset['reward_group'] == group]
            boot_result = bootstrap_group_focality_neurons(group_data, area_column, n_bootstrap)
            bootstrap_results[group] = boot_result

            if boot_result:
                print(f"    {group}: mean = {boot_result['mean']:.4f} ± {boot_result['se']:.4f}, "
                      f"95% CI = [{boot_result['ci'][0]:.4f}, {boot_result['ci'][1]:.4f}]")

                # Check bias
                bias = boot_result['mean'] - observed_fi[group]
                print(f"          bias = {bias:.4f}")

        # Compute difference distribution and CI
        if bootstrap_results['R+'] and bootstrap_results['R-']:
            boot_diffs = (bootstrap_results['R+']['bootstrap_values'] -
                          bootstrap_results['R-']['bootstrap_values'])
            diff_mean = np.mean(boot_diffs)
            diff_se = np.std(boot_diffs)
            diff_ci = (diff_mean - 1.96 * diff_se, diff_mean + 1.96 * diff_se)
            print(f"    Difference: mean = {diff_mean:.4f}, 95% CI = [{diff_ci[0]:.4f}, {diff_ci[1]:.4f}]")

        # ================================================================
        # STEP 3: PERMUTATION TEST (SHUFFLE NEURON LABELS)
        # ================================================================

        print(f"  Step 3: Permutation test ({n_permutations} iterations)...")

        perm_result = permutation_test_neurons(subset, area_column, n_permutations)
        p_value = perm_result['p_value']

        print(f"    Observed difference: {perm_result['observed_diff']:.4f}")
        print(f"    Permutation p-value: {p_value:.4f}")

        # ================================================================
        # STEP 4: EFFECT SIZE
        # ================================================================

        if bootstrap_results['R+'] and bootstrap_results['R-']:
            pooled_se = np.sqrt(
                (bootstrap_results['R+']['se'] ** 2 +
                 bootstrap_results['R-']['se'] ** 2) / 2
            )
            cohens_d = observed_diff / pooled_se if pooled_se > 0 else np.nan
            print(f"    Cohen's d: {cohens_d:.2f}")
        else:
            cohens_d = np.nan

        # ================================================================
        # STEP 5: STORE RESULTS
        # ================================================================

        for group in ['R+', 'R-']:
            boot = bootstrap_results[group]

            all_results.append({
                'model_name': model_name,
                'reward_group': group,
                'observed_focality': observed_fi[group],
                'bootstrap_mean': boot['mean'] if boot else np.nan,
                'bootstrap_se': boot['se'] if boot else np.nan,
                'ci_lower': boot['ci'][0] if boot else np.nan,
                'ci_upper': boot['ci'][1] if boot else np.nan,
                'n_mice': subset[subset['reward_group'] == group]['mouse_id'].nunique(),
                'n_neurons': len(subset[subset['reward_group'] == group]),
                'n_areas': subset[subset['reward_group'] == group][area_column].nunique()
            })

        test_results.append({
            'model_name': model_name,
            'difference': observed_diff,
            'p_value': p_value,
            'cohens_d': cohens_d
        })

    # ================================================================
    # CREATE RESULTS DATAFRAMES
    # ================================================================

    results_df = pd.DataFrame(all_results)
    test_df = pd.DataFrame(test_results)

    # Merge test results
    results_df = results_df.merge(test_df, on='model_name', how='left')

    # ================================================================
    # FDR CORRECTION
    # ================================================================

    if len(test_df) > 0:
        print("\n" + "=" * 70)
        print("FDR CORRECTION")
        print("=" * 70)

        _, p_fdr, _, _ = multipletests(test_df['p_value'], method='fdr_bh')

        # Map back to results
        model_to_p_fdr = dict(zip(test_df['model_name'], p_fdr))
        results_df['p_fdr'] = results_df['model_name'].map(model_to_p_fdr)

    # ================================================================
    # SAVE RESULTS
    # ================================================================

    if saving_path:
        results_df.to_csv(f"{saving_path}/focality_results_neurons.csv", index=False)
        print(f"\nResults saved to: {saving_path}/focality_results_neurons.csv")

    # ================================================================
    # PRINT SUMMARY
    # ================================================================

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for model in results_df['model_name'].unique():
        model_data = results_df[results_df['model_name'] == model]

        print(f"\n{model}:")

        for _, row in model_data.iterrows():
            if row['reward_group'] == 'R+':
                print(f"  R+ (n={row['n_mice']} mice, {row['n_neurons']} neurons): "
                      f"Observed FI = {row['observed_focality']:.3f}, "
                      f"Bootstrap = {row['bootstrap_mean']:.3f} ± {row['bootstrap_se']:.3f}, "
                      f"95% CI [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
            else:
                print(f"  R- (n={row['n_mice']} mice, {row['n_neurons']} neurons): "
                      f"Observed FI = {row['observed_focality']:.3f}, "
                      f"Bootstrap = {row['bootstrap_mean']:.3f} ± {row['bootstrap_se']:.3f}, "
                      f"95% CI [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")

        # Get test results
        test_row = model_data.iloc[0]
        print(f"  Difference: {test_row['difference']:.3f}")
        print(f"  p = {test_row['p_value']:.4f}, p_FDR = {test_row['p_fdr']:.4f}, d = {test_row['cohens_d']:.2f}")

        if test_row['p_fdr'] < 0.05:
            if test_row['difference'] > 0:
                print(f"  → R+ significantly MORE concentrated than R- ***")
            else:
                print(f"  → R- significantly MORE concentrated than R+ ***")
        else:
            print(f"  → No significant difference")

    return results_df