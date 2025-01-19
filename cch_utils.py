import math
from typing import Dict, Optional, List, Union, Tuple
from tqdm import tqdm
import warnings



from itertools import product
from functools import partial
import numpy as np
import neo
import quantities as pq
from elephant import statistics, kernels
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
#from viziphant.rasterplot import rasterplot_rates, eventplot
#from viziphant.spike_train_correlation import plot_cross_correlation_histogram
#from viziphant.statistics import plot_instantaneous_rates_colormesh
#from viziphant.events import add_event
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import cross_correlation_histogram
from elephant.spike_train_surrogates import dither_spikes, jitter_spikes
from elephant.statistics import mean_firing_rate, isi

import plotting_utils
import os




def calculate_cch(bst1: BinnedSpikeTrain, bst2: BinnedSpikeTrain, window: int):
    """
    Calculate the cross-correlation histogram (CCH) between two binned spike trains using Elephant.
    :param bst1: BinnedSpikeTrain object for the first spike train
    :param bst2: BinnedSpikeTrain object for the second spike train
    :param window: Maximum absolute lag in number of bins
    :return:
    """
    window_range = [-window, window]
    cch_results = cross_correlation_histogram(bst1, bst2, window=window_range)
    return np.array(cch_results[0]).flatten(), np.array(cch_results[1]).flatten()


def calculate_jitter_cch(bst1: BinnedSpikeTrain, spiketrain: neo.SpikeTrain, jitter_window: pq.Quantity, lag_window: int, num_surrogates: int) -> np.ndarray:
    """
    Calculate the mean cross-correlation histogram (CCH) between a binned spike train and jittered surrogates.
    :param bst1: BinnedSpikeTrain object for the first spike train
    :param spiketrain: Neo SpikeTrain object for the second spike train
    :param jitter_window: Window for jittering the spike train
    :param lag_window: Maximum absolute lag in number of bins
    :param num_surrogates: Number of jittered surrogates to generate
    :return:
    """

    # Generate jittered surrogate spike trains
    jittered_trains = list(dither_spikes(spiketrain, dither=jitter_window, n_surrogates=num_surrogates))

    if len(jittered_trains) != num_surrogates:
        raise ValueError(f"Expected {num_surrogates} jittered trains, but got {len(jittered_trains)}.")

    # Init. storage
    cch_length = None
    jittered_cchs = []

    # Compute CCH for each jittered train
    for i, jittered_train in enumerate(jittered_trains):
        binned_jittered_train = BinnedSpikeTrain(jittered_train, bin_size=bst1.bin_size, tolerance=None)
        cch, _ = calculate_cch(bst1, binned_jittered_train, lag_window)

        if cch_length is None:
            cch_length = len(cch)  # Determine the length from the first CCH

        jittered_cchs.append(cch)

    # Calculate mean CCH across surrogates
    jittered_cchs = np.array(jittered_cchs, dtype=np.float64)
    mean_jittered_cch = np.mean(jittered_cchs, axis=0)
    return mean_jittered_cch


def find_interactions(cch: np.ndarray, sig_level: float = 7.0, window: int = 10):
    """
    Identify significant interactions in the cross-correlation histogram (CCH).
    :param cch: Cross-correlation histogram, typically the corrected CCH
    :param threshold: Number of standard deviations above the mean to consider as significant
    :param window: Number of bins to serach for significant interactions after zero lag
    :return:
    """

    # Calculate the standard deviation of the flanks, defined as the first and last 50 bins
    flank_sd = np.std(np.concatenate((cch[:50], cch[-50:])))
    sig_threshold = sig_level * flank_sd

    # Define the search window as [1ms, + window ms] after zero lag
    zero_lag_index = len(cch) // 2  # 100
    search_range = slice(zero_lag_index + 2, zero_lag_index + 1 + window)
    cch_window = cch[search_range]

    # Initialize results
    significant_result = {
        'flank_sd': flank_sd,
        'significant': False,
        'lag_index': None,
        'cch_peak_value': None,
        'int_type': None,
    }

    # Look for significant interactions above the upper bound
    above_indices = np.where(cch_window > sig_threshold)[0]
    below_indices = np.where(cch_window < -sig_threshold)[0]

    # Combine both above and below indices, labeling their direction
    significant_indices = []
    if len(above_indices) > 0:
        significant_indices.append((above_indices[0], 'excitatory'))
    if len(below_indices) > 0:
        significant_indices.append((below_indices[0], 'inhibitory'))

    # If no significant interactions exist, return the default result
    if not significant_indices:
        return significant_result
    else:

        # Sort indices to determine the first crossing (whether above or below)
        significant_indices.sort(key=lambda x: x[0])
        first_index, direction = significant_indices[0]

        # Create result with the first crossing
        significant_result = {
            'flank_sd': flank_sd,
            'significant': True,
            'lag_index': zero_lag_index + 1 + first_index,
            'cch_peak_value': cch_window[first_index],
            'int_type': direction
        }

    return significant_result


def plot_cross_correlation_histogram(
    cch_magnitude, 
    sampling_rate=1*pq.ms, 
    axes=None, 
    units=None, 
    maxlag=None,
    legend=None, 
    title='Cross-correlation histogram', 
    normalization='Rate (Hz)'
):
    
    if axes is None:
        _, axes = plt.subplots()

    # Convert single magnitude array to list
    if isinstance(cch_magnitude, np.ndarray):
        cch_magnitude = [cch_magnitude]

    # Handle units
    if units is None:
        units = sampling_rate.units
    elif isinstance(units, str):
        units = pq.Quantity(1, units)

    # Handle legend
    if legend is None:
        legend = [None] * len(cch_magnitude)
    elif isinstance(legend, str):
        legend = [legend]
    if len(legend) != len(cch_magnitude):
        raise ValueError("The length of the input list and legend labels do not match.")

    # Calculate time points for x-axis
    for label, magnitude in zip(legend, cch_magnitude):
        num_points = len(magnitude)
        half_width = (num_points // 2) * sampling_rate
        times = np.linspace(-half_width, half_width, num_points)
        times = times.rescale(units).magnitude
        
        axes.plot(times, magnitude, label=label)

    axes.set_ylabel(normalization)
    axes.set_xlabel(f"Time lag ({units.dimensionality})")
    axes.set_title(title)
    
    if maxlag is not None:
        maxlag = maxlag.rescale(units).magnitude
        axes.set_xlim(-maxlag, maxlag)
    
    if legend[0] is not None:
        axes.legend()
    
    # Remove plt.show() from here
    # plt.show()
    
    return axes


def plot_cch_correction(
    cch: np.ndarray,
    mean_jittered_cch: np.ndarray,
    corrected_cch: np.ndarray,
    flank_sd: float,
    peak_lag: int,
    peak_value: float,
    bin_size: pq.Quantity = 1 * pq.ms,
    maxlag: int = 100,
    save_path: str = None, 
    lag_window: int = 10,
    title: str = 'Cross-Correlation Histogram'
) -> None:
    plt.figure(figsize=(10, 6))
    axes = plt.gca()

    if bin_size.units is not pq.ms:
        bin_size = bin_size.rescale('ms')
    
    units = bin_size.units

    
    # Plot original CCH
    plot_cross_correlation_histogram(
        cch,
        sampling_rate=bin_size,
        axes=axes,
        units=units,
        legend="original",
        title='Cross-Correlation Histogram',
    )

    # Plot mean jittered CCH
    plot_cross_correlation_histogram(
        mean_jittered_cch,
        sampling_rate=bin_size,
        axes=axes,
        units=units,
        legend="jittered",
    )

    # Plot corrected CCH
    plot_cross_correlation_histogram(
        corrected_cch,
        sampling_rate=bin_size,
        axes=axes,
        units=units,
        legend="corrected",
    )

    # Add vertical line at time lag zero
    axes.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    
    SD = flank_sd
    # Plot horizontal lines at ±7*SD
    threshold = 7 * SD
    axes.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label='7*SD threshold')
    axes.axhline(y=-threshold, color='red', linestyle='--', linewidth=1.5)

        
    # Plot scatter point
    axes.scatter(peak_lag, peak_value, 
                color='black', 
                marker='o',
                s=80,
                zorder=5)
        
    axes.annotate(
        f'Peak: {peak_value:.2f}\nLag: {peak_lag} ms',
        xy=(peak_lag, peak_value),
        xytext=(peak_lag + 2, peak_value + (0.1 * peak_value)),  # Offset text slightly
        arrowprops=dict(arrowstyle='->'),  # Removed color parameter
        fontsize=8
    )
        
    # Shade the significance window
    axes.axvspan(-lag_window, lag_window, color='gray', alpha=0.1, 
                 label='Significance window')

    # Set x-limits to maxlag
    maxlag = maxlag * pq.ms
    if maxlag is not None:
        maxlag_ms = maxlag.magnitude
        axes.set_xlim(-maxlag_ms, maxlag_ms)

    plt.legend(fontsize=12, loc='upper right')
    plt.xlabel('Time Lag (ms)')
    plt.ylabel('spike count')
    plt.title(title)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"The plot has been saved to: {save_path}")
    else:
        plt.show()

def plot_jittered_cch(
    jittered_cchs: np.ndarray,
    bin_size: pq.Quantity = 1 * pq.ms,
    maxlag: int = 100,
    alpha: float = 0.2,
    save_path: str = None
) -> None:
    """
    Plot multiple jittered CCHs and their mean on the same graph.
    
    Args:
        jittered_cchs (np.ndarray): Array of shape (n_surrogates, n_bins) containing jittered CCHs
        bin_size (pq.Quantity): Size of bins in time units
        maxlag (int): Maximum lag to display in ms
        alpha (float): Transparency for individual jittered CCHs
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    axes = plt.gca()

    if bin_size.units is not pq.ms:
        bin_size = bin_size.rescale('ms')
    
    # Calculate mean of jittered CCHs
    mean_jittered_cch = np.mean(jittered_cchs, axis=0)
    
    # Plot individual jittered CCHs
    for jittered_cch in jittered_cchs:
        plot_cross_correlation_histogram(
            jittered_cch,
            sampling_rate=bin_size,
            axes=axes,
            units='ms',
            legend=None,
            title='Jittered Cross-Correlation Histograms'
        )
        axes.lines[-1].set_alpha(alpha)
        axes.lines[-1].set_color('gray')
    
    # Plot mean jittered CCH with different color and thicker line
    plot_cross_correlation_histogram(
        mean_jittered_cch,
        sampling_rate=bin_size,
        axes=axes,
        units='ms',
        legend='Mean Jittered CCH'
    )
    axes.lines[-1].set_color('red')
    axes.lines[-1].set_linewidth(2)
    
    # Add vertical line at time lag zero
    axes.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    # Set x-limits to maxlag
    maxlag = maxlag * pq.ms
    if maxlag is not None:
        maxlag_ms = maxlag.magnitude
        axes.set_xlim(-maxlag_ms, maxlag_ms)
    
    plt.legend(fontsize=12)
    plt.xlabel('Time Lag (ms)')
    plt.ylabel('Correlation')
    plt.title('Jittered Cross-Correlation Histograms')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"The plot has been saved to: {save_path}")
    else:
        plt.show()


def plot_significant_interactions_frequency(interaction_df: pd.DataFrame, folder_name: str) -> None:
    """
    Plot the frequency of significant interactions by source parent region and
    decompose significant interactions into inhibitory and excitatory types.
    """

    # Setup output directory
    plots_dir = os.path.join(folder_name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Create figure and subplots only once
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Significant Interactions Analysis', fontsize=16, y=1.02)

    # Calculate Overall Percentages
    total_interactions = interaction_df.groupby('source_ccf_parent').size()
    significant_interactions = interaction_df.groupby('source_ccf_parent')['is_significant'].sum()
    percentage_df = (significant_interactions / total_interactions * 100).reset_index(name='percentage')

    # Calculate Inhibitory vs Excitatory Percentages
    sig_interactions_df = interaction_df[interaction_df['is_significant'] == True].copy()
    sig_interactions_df['interaction_type'] = sig_interactions_df['cch_max_peak_value'].apply(
        lambda x: 'Excitatory' if x > 0 else 'Inhibitory'
    )
    
    interaction_counts = sig_interactions_df.groupby(
        ['source_ccf_parent', 'interaction_type']
    ).size().reset_index(name='count')
    
    total_sig_interactions = sig_interactions_df.groupby('source_ccf_parent').size().reset_index(name='total_sig')
    interaction_counts = interaction_counts.merge(total_sig_interactions, on='source_ccf_parent')
    interaction_counts['proportion'] = (interaction_counts['count'] / interaction_counts['total_sig']) * 100

    # Left Subplot: Overall Significant Interactions
    sns.barplot(
        data=percentage_df,
        x='source_ccf_parent',
        y='percentage',
        hue=None,  # Explicitly set hue to None
        palette='viridis',  # use a palette 
        ax=ax1
    )
    ax1.set_xlabel('Source CCF Parent Region', fontsize=14)
    ax1.set_ylabel('Percentage of Significant Interactions (%)', fontsize=14)
    ax1.set_title('Overall Significant Interactions by Region', fontsize=16)
    ax1.set_ylim(0, percentage_df['percentage'].max() + 10)
    ax1.tick_params(axis='x', rotation=45)

    # Right Subplot: Inhibitory vs Excitatory
    sns.barplot(
        data=interaction_counts,
        x='source_ccf_parent',
        y='proportion',
        hue='interaction_type',
        palette={'Excitatory': 'steelblue', 'Inhibitory': 'salmon'},
        ax=ax2
    )
    ax2.set_xlabel('Source CCF Parent Region', fontsize=14)
    ax2.set_ylabel('Percentage of Significant Interactions (%)', fontsize=14)
    ax2.set_title('Excitatory vs Inhibitory Significant Interactions by Region', fontsize=16)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Interaction Type')

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(plots_dir, 'significant_interactions_frequency.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Significant interactions frequency plot saved to: {plot_path}")
        





def plot_peak_lag_distributions(interaction_df, folder_name):
    """
    Plot the distribution of peak lags for both significant and all interactions.
    
    Args:
        interaction_df (pd.DataFrame): DataFrame containing interaction data
        folder_name (str): Path to save the output plots
    """
    # Setup output directory
    plots_dir = os.path.join(folder_name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Peak Lag Distribution Analysis', fontsize=14, y=1.05)

    # Define common parameters
    bins = np.arange(-10, 11, 1)
    plot_params = {
        'bins': bins,
        'kde': False
    }

    # Left subplot: Significant Interactions
    significant_lags = interaction_df[interaction_df['is_significant']]['cch_max_peak_lag']
    sns.histplot(
        data=significant_lags,
        ax=ax1,
        color='skyblue',
        **plot_params
    )
    
    ax1.set_xlabel('Peak Lag (ms)')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Peak Lags\nfor Significant Interactions')
    ax1.set_xticks(bins)
    ax1.set_xticklabels(bins)
    ax1.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Right subplot: All Interactions
    all_lags = interaction_df['cch_max_peak_lag']
    sns.histplot(
        data=all_lags,
        ax=ax2,
        color='salmon',
        **plot_params
    )
    
    ax2.set_xlabel('Peak Lag (ms)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Peak Lags\nfor All Interactions')
    ax2.set_xticks(bins)
    ax2.set_xticklabels(bins)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'peak_lag_distributions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Peak lag distribution plots saved to: {plot_path}")


def plot_unit_type_interactions(interaction_df: pd.DataFrame, folder_name) -> None:
    """
    Create a scatter plot showing CCH peak values for different unit type interaction pairs.
    Points are jittered based on their lag values.
    
    Args:
        interaction_df (pd.DataFrame): DataFrame containing interaction data
        plots_dir (str): Directory to save the plot
    """
    # Create figure and axis
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), dpi=100)
    
    # Define interaction pairs and their positions on x-axis
    pairs = [
        ('fsu', 'fsu', 1),
        ('fsu', 'rsu', 2),
        ('rsu', 'fsu', 3),
        ('rsu', 'rsu', 4)
    ]
    
    # Define markers and colors
    markers = {
        'excitatory': '^',  # triangle up for excitatory
        'inhibitory': 'v'   # triangle down for inhibitory
    }
    colors = {
        'excitatory': 'red',
        'inhibitory': 'blue'
    }
    
    # Filter for significant interactions only
    interaction_df = interaction_df[interaction_df["is_significant"] == True].copy()

    # Set initial axis limits to ensure consistent scale
    ax.set_xlim(0.5, 4.5)  # Add padding around the x positions
    
    # Plot scatter values for each interaction pair
    for source_type, target_type, x_pos in pairs:
        # Filter interactions for this pair type
        mask = (interaction_df['source_waveform_type'].str.lower() == source_type) & \
               (interaction_df['target_waveform_type'].str.lower() == target_type)
        pair_interactions = interaction_df[mask]
        
        # Add vertical line and label for this position regardless of data
        ax.axvline(x=x_pos, color='gray', linestyle='--', alpha=0.3)
        
        if not pair_interactions.empty:
            # Determine marker type based on peak value
            for _, row in pair_interactions.iterrows():
                marker_type = 'excitatory' if row['cch_max_peak_value'] > 0 else 'inhibitory'
                
                # Add some jitter to x position based on lag
                jitter = row['cch_max_peak_lag'] / 50  # Scale factor for jitter
                x_jittered = x_pos + jitter
                
                ax.scatter(x_jittered, 
                          row['cch_max_peak_value'],
                          marker=markers[marker_type],
                          c=colors[marker_type],
                          s=30,
                          alpha=0.6)
    
    # Customize plot
    ax.set_ylabel('CCH Peak Value')
    
    # Set x-axis ticks
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['FSU-FSU', 'FSU-RSU', 'RSU-FSU', 'RSU-RSU'])

    # Add '0 lag' text for all positions
    ymin, ymax = ax.get_ylim()
    for x in [1, 2, 3, 4]:
        ax.text(x, ymax + 1, '0 lag', 
                color='gray', 
                alpha=0.7,
                ha='center',  # horizontal alignment
                va='bottom'   # vertical alignment
                )


    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                  markersize=7, label='Excitatory'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='blue', 
                  markersize=7, label='Inhibitory')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Setup output directory
    plots_dir = os.path.join(folder_name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'unit_type_interactions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Unit type interactions plot saved to: {plots_dir}")


def filter_positive_lag_interactions(significant_interactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize interactions by ensuring all peak lags are positive.
    For negative peak lags, swap source and target information while preserving the peak value.
    
    Args:
        interaction_df (pd.DataFrame): DataFrame containing interaction data with columns for
            source/target information, significance flag, and CCH peak values/lags
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing:
            - DataFrame with standardized positive-lag significant interactions
            - DataFrame with standardized positive-lag all interactions
    """
    # Create copies for both significant and all interactions
    df_all = significant_interactions_df.copy()
    df_significant = df_all[df_all['is_significant']].copy()
    
    # Process both DataFrames with the same logic
    for df in [df_all, df_significant]:
        # Find rows where peak lag is negative
        negative_lag_mask = df['cch_max_peak_lag'] < 0
        
        # For negative lag rows, swap source and target columns
        columns_to_swap = [
            ('source_id', 'target_id'),
            ('source_ccf_parent', 'target_ccf_parent'),
            ('source_waveform_type', 'target_waveform_type'), 
            ('source_ccf', 'target_ccf'), 
            ('source_fr', 'target_fr')
        ]
        
        for source_col, target_col in columns_to_swap:
            df.loc[negative_lag_mask, [source_col, target_col]] = \
                df.loc[negative_lag_mask, [target_col, source_col]].values
        
        # Make peak lag positive (absolute value)
        df.loc[negative_lag_mask, 'cch_max_peak_lag'] = \
            -df.loc[negative_lag_mask, 'cch_max_peak_lag']
    
    return df_significant, df_all


def plot_cortical_area_interactions(significant_interactions_df: pd.DataFrame, folder_name: str) -> None:
    """
    Create a scatter plot showing different types of interactions between cortical areas.
    
    Args:
        significant_interactions_df (pd.DataFrame): DataFrame containing significant interaction data
        folder_name (str): Directory to save the plot
    """
    # Get cortical areas and filter DataFrame
    cortical_areas = plotting_utils.get_cortical_areas()
    cortical_df = significant_interactions_df[
        (significant_interactions_df['source_ccf_parent'].isin(cortical_areas)) &
        (significant_interactions_df['target_ccf_parent'].isin(cortical_areas))
    ].copy()

    # Define interaction types and their properties
    interaction_types = {
        'FSU-FSU': {'color': 'red', 'marker': 'o'},
        'FSU-RSU': {'color': 'blue', 'marker': 'o'},
        'RSU-FSU': {'color': 'green', 'marker': 'o'},
        'RSU-RSU': {'color': 'purple', 'marker': 'o'}
    }

    # Create figure
    plt.figure(figsize=(12, 8))  # Increased figure size for better readability
    ax = plt.gca()

    # Plot each interaction type
    for interaction_type, properties in interaction_types.items():
        source_type, target_type = interaction_type.split('-')
        
        # Filter for this interaction type
        mask = (
            (cortical_df['source_waveform_type'].str.upper() == source_type) &
            (cortical_df['target_waveform_type'].str.upper() == target_type)
        )
        subset = cortical_df[mask]
        
        if not subset.empty:
            # Plot excitatory interactions (positive peak values)
            excitatory = subset[subset['cch_max_peak_value'] > 0]
            if not excitatory.empty:
                ax.scatter(excitatory['cch_max_peak_lag'], 
                         excitatory['cch_max_peak_value'],
                         color=properties['color'],
                         marker=properties['marker'],
                         s=70,  # Increased marker size
                         alpha=0.7,  # Slightly increased transparency
                         label=f'{interaction_type} (Excitatory)')

            # Plot inhibitory interactions (negative peak values)
            inhibitory = subset[subset['cch_max_peak_value'] < 0]
            if not inhibitory.empty:
                ax.scatter(
                    inhibitory['cch_max_peak_lag'], 
                    inhibitory['cch_max_peak_value'],
                    color=properties['color'],
                    marker=properties['marker'],
                    s=70,  # Increased marker size
                    alpha=0.7,  # Slightly increased transparency
                    label=f'{interaction_type} (Inhibitory)'
                )

    # Add labels and title
    ax.set_xlabel('Peak Lag (ms)', fontsize=14)  # Increased font size
    ax.set_ylabel('CCH Peak Value', fontsize=14)  # Increased font size
    ax.set_title('Cortical Area Interactions by Unit Type', fontsize=16)  # Increased font size

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.5)

    # Add legend with smaller font and outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Setup output directory and save
    plots_dir = os.path.join(folder_name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'cortical_area_interactions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Cortical area interactions plot saved to: {plot_path}")

