"""
Single function for focality index analysis of GLM encoding variables.

Usage:
    from focality_analysis_simple import analyze_focality
    
    results = analyze_focality(data_df, saving_path='/your/path')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests


def gini_coefficient(fractions):
    """Compute Gini coefficient (0=distributed, 1=concentrated)."""
    fractions = np.array(fractions)
    fractions = fractions[~np.isnan(fractions)]
    if len(fractions) == 0:
        return np.nan
    sorted_fractions = np.sort(fractions)
    n = len(sorted_fractions)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_fractions)) / (n * np.sum(sorted_fractions)) - (n + 1) / n


def analyze_focality(data_df, saving_path=None, n_bootstrap=5000, n_permutations=10000):
    """
    Complete focality analysis: compute, test, bootstrap, and plot.
    
    Parameters:
    -----------
    data_df : DataFrame
        Must have columns: mouse_id, reward_group ('R+' or 'R-'), 
        model_name, area_acronym_custom, significant (boolean)
    saving_path : str
        Directory to save figures and results
    n_bootstrap : int
        Bootstrap iterations for confidence intervals
    n_permutations : int
        Permutation test iterations
    
    Returns:
    --------
    DataFrame with results for each variable
    """
    
    # Filter out full model
    data_df = data_df[data_df['model_name'] != 'full'].copy()
    
    results = []
    
    for model_name in data_df['model_name'].unique():
        subset = data_df[data_df['model_name'] == model_name]
        
        # Compute focality per mouse for each group
        r_plus_focality = []
        r_minus_focality = []
        
        for reward_group, focality_list in [('R+', r_plus_focality), ('R-', r_minus_focality)]:
            group_data = subset[subset['reward_group'] == reward_group]
            
            for mouse_id in group_data['mouse_id'].unique():
                mouse_data = group_data[group_data['mouse_id'] == mouse_id]
                area_fracs = mouse_data.groupby('area_acronym_custom')['significant'].mean()
                
                if len(area_fracs) > 0:
                    gini = gini_coefficient(area_fracs.values)
                    if not np.isnan(gini):
                        focality_list.append(gini)
        
        if len(r_plus_focality) < 2 or len(r_minus_focality) < 2:
            continue
        
        r_plus_focality = np.array(r_plus_focality)
        r_minus_focality = np.array(r_minus_focality)
        
        # Permutation test
        obs_diff = np.mean(r_plus_focality) - np.mean(r_minus_focality)
        combined = np.concatenate([r_plus_focality, r_minus_focality])
        n_r_plus = len(r_plus_focality)
        
        perm_diffs = []
        for _ in range(n_permutations):
            shuffled = np.random.permutation(combined)
            perm_diffs.append(np.mean(shuffled[:n_r_plus]) - np.mean(shuffled[n_r_plus:]))
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
        
        # Bootstrap CI
        boot_r_plus, boot_r_minus = [], []
        mice_r_plus = subset[subset['reward_group'] == 'R+']['mouse_id'].unique()
        mice_r_minus = subset[subset['reward_group'] == 'R-']['mouse_id'].unique()
        
        for _ in range(n_bootstrap):
            # Resample mice
            boot_mice_plus = np.random.choice(mice_r_plus, len(mice_r_plus), replace=True)
            boot_mice_minus = np.random.choice(mice_r_minus, len(mice_r_minus), replace=True)
            
            # Compute focality for resampled mice
            boot_gini_plus = []
            for mouse_id in boot_mice_plus:
                mouse_data = subset[(subset['reward_group'] == 'R+') & 
                                   (subset['mouse_id'] == mouse_id)]
                area_fracs = mouse_data.groupby('area_acronym_custom')['significant'].mean()
                if len(area_fracs) > 0:
                    g = gini_coefficient(area_fracs.values)
                    if not np.isnan(g):
                        boot_gini_plus.append(g)
            
            boot_gini_minus = []
            for mouse_id in boot_mice_minus:
                mouse_data = subset[(subset['reward_group'] == 'R-') & 
                                   (subset['mouse_id'] == mouse_id)]
                area_fracs = mouse_data.groupby('area_acronym_custom')['significant'].mean()
                if len(area_fracs) > 0:
                    g = gini_coefficient(area_fracs.values)
                    if not np.isnan(g):
                        boot_gini_minus.append(g)
            
            if len(boot_gini_plus) > 0:
                boot_r_plus.append(np.mean(boot_gini_plus))
            if len(boot_gini_minus) > 0:
                boot_r_minus.append(np.mean(boot_gini_minus))
        
        ci_plus = np.percentile(boot_r_plus, [2.5, 97.5]) if boot_r_plus else [np.nan, np.nan]
        ci_minus = np.percentile(boot_r_minus, [2.5, 97.5]) if boot_r_minus else [np.nan, np.nan]
        
        # Cohen's d
        pooled_std = np.sqrt(((len(r_plus_focality)-1)*np.var(r_plus_focality, ddof=1) +
                              (len(r_minus_focality)-1)*np.var(r_minus_focality, ddof=1)) /
                             (len(r_plus_focality) + len(r_minus_focality) - 2))
        cohens_d = obs_diff / pooled_std if pooled_std > 0 else np.nan
        
        # Create figure
        if saving_path:
            fig = plt.figure(figsize=(12, 12))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # Panel 1: Main comparison
            ax1 = fig.add_subplot(gs[0, 0])
            x_pos = [0, 1]
            colors = ['#d62728', '#1f77b4']
            
            for i, (vals, color) in enumerate([(r_plus_focality, colors[0]), 
                                                (r_minus_focality, colors[1])]):
                ax1.bar(x_pos[i], np.mean(vals), yerr=np.std(vals)/np.sqrt(len(vals)),
                       color=color, alpha=0.7, capsize=5)
                ax1.scatter(np.ones(len(vals))*x_pos[i] + np.random.normal(0, 0.02, len(vals)),
                           vals, color='black', alpha=0.6, s=50, zorder=3)
            
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(['R+', 'R-'], fontsize=14)
            ax1.set_ylabel('Gini coefficient', fontsize=14)
            sig_text = '***' if p_value<0.001 else '**' if p_value<0.01 else '*' if p_value<0.05 else 'n.s.'
            ax1.set_title(f'{model_name}\np={p_value:.4f}, d={cohens_d:.2f}', fontsize=12, fontweight='bold')
            y_max = ax1.get_ylim()[1]
            ax1.plot([0,1], [y_max*0.95, y_max*0.95], 'k-', linewidth=1.5)
            ax1.text(0.5, y_max*0.97, sig_text, ha='center', fontsize=16)
            
            # Panel 2: Bootstrap distributions
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.hist(boot_r_plus, bins=50, alpha=0.6, color=colors[0], density=True,
                    label=f'R+: [{ci_plus[0]:.3f}, {ci_plus[1]:.3f}]')
            ax2.hist(boot_r_minus, bins=50, alpha=0.6, color=colors[1], density=True,
                    label=f'R-: [{ci_minus[0]:.3f}, {ci_minus[1]:.3f}]')
            ax2.set_xlabel('Gini coefficient', fontsize=12)
            ax2.set_ylabel('Density', fontsize=12)
            ax2.set_title('Bootstrap distributions', fontsize=12)
            ax2.legend(fontsize=10)
            
            # Panel 3: Spatial distribution
            ax3 = fig.add_subplot(gs[1, 0])
            spatial_data = []
            for rg in ['R+', 'R-']:
                group_subset = subset[subset['reward_group'] == rg]
                area_fracs = group_subset.groupby('area_acronym_custom')['significant'].mean()
                for area, frac in area_fracs.items():
                    spatial_data.append({'area': area, 'reward_group': rg, 'fraction': frac})
            
            if spatial_data:
                spatial_df = pd.DataFrame(spatial_data)
                pivot_df = spatial_df.pivot(index='area', columns='reward_group', values='fraction')
                pivot_df.plot(kind='bar', ax=ax3, color=colors, width=0.8)
                ax3.set_xlabel('Brain area', fontsize=12)
                ax3.set_ylabel('Fraction significant', fontsize=12)
                ax3.set_title('Spatial distribution', fontsize=12)
                ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            
            # Panel 4: Summary text
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.axis('off')
            summary = f"SUMMARY\n{'='*30}\n\n"
            summary += f"R+ (n={len(r_plus_focality)}): {np.mean(r_plus_focality):.3f}±{np.std(r_plus_focality):.3f}\n"
            summary += f"R- (n={len(r_minus_focality)}): {np.mean(r_minus_focality):.3f}±{np.std(r_minus_focality):.3f}\n\n"
            summary += f"Difference: {obs_diff:.3f}\n"
            summary += f"p-value: {p_value:.4f}\n"
            summary += f"Cohen's d: {cohens_d:.2f}\n\n"
            if p_value < 0.05:
                summary += "→ " + ("R+ MORE" if obs_diff > 0 else "R- MORE") + " concentrated"
            else:
                summary += "→ No significant difference"
            ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            plt.suptitle(f'Focality Analysis: {model_name}', fontsize=14, fontweight='bold')
            
            for ext in ['png', 'pdf', 'svg']:
                fig.savefig(f"{saving_path}/focality_{model_name}.{ext}", 
                           dpi=300, bbox_inches='tight')
            plt.close()
        
        results.append({
            'model_name': model_name,
            'r_plus_mean': np.mean(r_plus_focality),
            'r_plus_std': np.std(r_plus_focality),
            'r_plus_ci_lower': ci_plus[0],
            'r_plus_ci_upper': ci_plus[1],
            'r_minus_mean': np.mean(r_minus_focality),
            'r_minus_std': np.std(r_minus_focality),
            'r_minus_ci_lower': ci_minus[0],
            'r_minus_ci_upper': ci_minus[1],
            'difference': obs_diff,
            'p_value': p_value,
            'cohens_d': cohens_d
        })
    
    results_df = pd.DataFrame(results)
    
    # FDR correction
    if len(results_df) > 0:
        _, p_fdr, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['p_fdr'] = p_fdr
        results_df = results_df.sort_values('p_value')
        
        if saving_path:
            results_df.to_csv(f"{saving_path}/focality_results.csv", index=False)
        
        print("\nFOCALITY RESULTS:")
        print("="*80)
        for _, row in results_df.iterrows():
            print(f"\n{row['model_name']}:")
            print(f"  R+: {row['r_plus_mean']:.3f} [{row['r_plus_ci_lower']:.3f}, {row['r_plus_ci_upper']:.3f}]")
            print(f"  R-: {row['r_minus_mean']:.3f} [{row['r_minus_ci_lower']:.3f}, {row['r_minus_ci_upper']:.3f}]")
            print(f"  p={row['p_value']:.4f}, p_fdr={row['p_fdr']:.4f}, d={row['cohens_d']:.2f}")
    
    return results_df
