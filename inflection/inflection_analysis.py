"""
Single-Neuron Inflection Analysis for Neuropixels Data (NWB)
=============================================================

Analyzes whether single-neuron spiking activity follows the behavioral inflection
trial using the pseudosession method.

NWB Data Structure Expected:
- units table: spike_times, bc_label (filter for 'good'), area columns
- trials table: start_time, stim_time, p_mean, inflection_trial, reward_group

Analysis:
- Pre-inflection window: trials -16 to -6 relative to inflection
- Post-inflection window: trials 0 to +10 relative to inflection
- PSTH: -100ms to 200ms post-stim (for visualization)
- Statistical focus: 50ms post-stim firing rate
"""

import numpy as np
from pathlib import Path
from pynwb import NWBHDF5IO
from scipy import stats
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import cmasher as cmr

import NWB_reader_functions
import allen_utils

import load_hmm_results


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for inflection analysis."""
    # Trial windows relative to inflection trial
    pre_trials: tuple = (-15, -6)   # trials -16 to -6 (inclusive)
    post_trials: tuple = (0, 9)    # trials 0 to +10 (inclusive)

    # Time windows for PSTH (seconds)
    psth_window: tuple = (-0.1, 0.2)      # -100ms to 200ms for visualization
    baseline_window: tuple = (-0.1, 0)     # -100ms to 0 for baseline subtraction
    response_window: tuple = (0.005, 0.05)     # 0 to 50ms for statistics

    # PSTH binning
    bin_size: float = 0.005  # ms bins for PSTH

    # Pseudosession parameters
    n_pseudosessions: int = 10000
    min_edge_trials: int = 10  # Minimum trials from session edges

    # Unit filtering
    bc_label_filter: str = 'good'  # Only use 'good' units


@dataclass
class NeuronResult:
    """Results from single-neuron pseudosession analysis."""
    mouse_id: str
    session_id: str
    unit_id: int
    area: str
    reward_group: int
    inflection_trial: int
    n_total_trials: int

    # Response values (firing rate in spks/s)
    pre_response: float
    post_response: float
    shift_magnitude: float  # post - pre

    # Statistics
    observed_statistic: float  # R²
    null_mean: float
    null_std: float
    p_value: float
    z_score: float

    # Additional info
    n_pre_trials: int
    n_post_trials: int
    mean_firing_rate: float  # Overall firing rate


# =============================================================================
# Plot Style
# =============================================================================

def set_plot_style():
    """Set publication-quality plot style."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.titlesize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.dpi': 150,
    })


# =============================================================================
# NWB Data Loading
# =============================================================================

def load_nwb_session(nwb_path: Path, config: AnalysisConfig) -> Dict[str, Any]:
    """
    Load Neuropixels session data from NWB file.

    Filters trials to include only:
    - trial_type == 'whisker' (whisker trials only)
    - context != 'passive' (active trials only)
    - perf != 6 (exclude perf==6 trials)

    Parameters
    ----------
    nwb_path : Path to NWB file
    config : AnalysisConfig with filtering parameters

    Returns
    -------
    session_data : dict with:
        - 'units': list of dicts with unit info (id, spike_times, area, etc.)
        - 'trials': dict with trial info (stim_times, trial indices, etc.)
        - 'metadata': session metadata

    Note: 'p_mean' and 'inflection_trial' should be added to session_data['trials']
    after loading, as they are computed externally.
    """
    with NWBHDF5IO(str(nwb_path), 'r') as io:
        nwbfile = io.read()

        # =====================================================================
        # Extract metadata
        # =====================================================================
        metadata = {
            'session_id': nwbfile.identifier,
            'mouse_id': nwbfile.subject.subject_id if nwbfile.subject else 'unknown',
            'session_start': str(nwbfile.session_start_time) if nwbfile.session_start_time else None,
        }

        # =====================================================================
        # Extract and filter trials
        # =====================================================================
        trials = nwbfile.trials
        if trials is None:
            raise ValueError(f"No trials table found in {nwb_path}")

        n_total_trials = len(trials)
        colnames = trials.colnames

        # Build trial mask for filtering
        trial_mask = np.ones(n_total_trials, dtype=bool)

        # Filter 1: Only whisker trials (trial_type == 'whisker')
        if 'trial_type' in colnames:
            trial_types = trials['trial_type'][:]
            # Handle bytes encoding
            if len(trial_types) > 0 and isinstance(trial_types[0], bytes):
                trial_types = np.array([t.decode() for t in trial_types])
            whisker_mask = (trial_types == 'whisker_trial')
            trial_mask &= whisker_mask
            n_whisker = np.sum(whisker_mask)
            print(f"  Trial filter: {n_whisker}/{n_total_trials} are whisker trials")
        else:
            warnings.warn("No 'trial_type' column found, using all trials")

        # Filter 2: Only active trials (context != 'passive')
        if 'context' in colnames:
            contexts = trials['context'][:]
            # Handle bytes encoding
            if len(contexts) > 0 and isinstance(contexts[0], bytes):
                contexts = np.array([c.decode() for c in contexts])
            active_mask = (contexts != 'passive')
            trial_mask &= active_mask
            n_active = np.sum(active_mask)
            print(f"  Trial filter: {n_active}/{n_total_trials} are active (non-passive)")
        else:
            warnings.warn("No 'context' column found, not filtering by context")

        # Filter 3: Exclude perf == 6 trials
        if 'perf' in colnames:
            perf = trials['perf'][:]
            perf_mask = (perf != 6)
            trial_mask &= perf_mask
            n_valid_perf = np.sum(perf_mask)
            print(f"  Trial filter: {n_valid_perf}/{n_total_trials} have perf != 6")
        else:
            warnings.warn("No 'perf' column found, not filtering by perf")

        # Get filtered trial indices (original indices that pass all filters)
        valid_trial_indices = np.where(trial_mask)[0]
        n_valid = len(valid_trial_indices)
        print(f"  Final: {n_valid}/{n_total_trials} trials pass all filters")

        if n_valid == 0:
            raise ValueError(f"No valid trials after filtering in {nwb_path}")

        # Extract trial data for valid trials only
        trial_data = {
            'start_time': trials['start_time'][:][valid_trial_indices],
            'stop_time': trials['stop_time'][:][valid_trial_indices],
            'original_trial_indices': valid_trial_indices,  # Keep track of original indices
        }

        # Get stim_time (or use start_time as fallback)
        if 'stim_time' in colnames:
            trial_data['stim_time'] = trials['stim_time'][:][valid_trial_indices]
        else:
            trial_data['stim_time'] = trials['start_time'][:][valid_trial_indices]
            warnings.warn("No 'stim_time' column found, using 'start_time'")

        # Get reward_group if available
        if 'reward_group' in colnames:
            rg = trials['reward_group'][:][valid_trial_indices]
            # Could be per-trial or session-level - take first value
            if np.isscalar(rg):
                trial_data['reward_group'] = int(rg)
            else:
                trial_data['reward_group'] = int(rg[0])  # Assume constant within session
        else:
            trial_data['reward_group'] = 0

        # Store additional trial info that might be useful
        for col in ['perf', 'context', 'trial_type']:
            if col in colnames:
                trial_data[col] = trials[col][:][valid_trial_indices]

        trial_data['n_trials'] = n_valid

        # Placeholder for p_mean and inflection_trial - to be added by user
        # trial_data['p_mean'] = None
        # trial_data['inflection_trial'] = None

        # =====================================================================
        # Extract units
        # =====================================================================
        units_table = nwbfile.units
        if units_table is None:
            raise ValueError(f"No units table found in {nwb_path}")

        n_units = len(units_table)
        units = []

        unit_colnames = units_table.colnames

        for unit_idx in range(n_units):
            # Check bc_label filter
            if 'bc_label' in unit_colnames:
                bc_label = units_table['bc_label'][unit_idx]
                # Handle bytes encoding
                if isinstance(bc_label, bytes):
                    bc_label = bc_label.decode()
                if bc_label != config.bc_label_filter:
                    continue

            # Get spike times
            spike_times = units_table['spike_times'][unit_idx]
            if len(spike_times) == 0:
                continue

            # Get area
            area = 'unknown'
            for area_col in ['ccf_atlas_acronym', 'area_custom_acronym', 'location', 'brain_area', 'area']:
                if area_col in unit_colnames:
                    area = units_table[area_col][unit_idx]
                    if isinstance(area, bytes):
                        area = area.decode()
                    break

            # Get unit ID
            if 'unit_id' in unit_colnames:
                unit_id = units_table['unit_id'][unit_idx]
            else:
                unit_id = unit_idx

            units.append({
                'unit_id': unit_id,
                'unit_index': unit_idx,
                'spike_times': np.array(spike_times),
                'area': area,
                'n_spikes': len(spike_times),
            })

        print(f"  Loaded {len(units)} good units from {n_units} total")

        return {
            'units': units,
            'trials': trial_data,
            'metadata': metadata
        }

def load_multiple_sessions(
    nwb_paths: List[Path],
    config: AnalysisConfig
) -> List[Dict[str, Any]]:
    """Load multiple NWB sessions."""
    sessions = []

    for nwb_path in nwb_paths:
        print(f"Loading {nwb_path.name}...")
        try:
            session = load_nwb_session(nwb_path, config)
            sessions.append(session)
        except Exception as e:
            warnings.warn(f"Failed to load {nwb_path}: {e}")

    return sessions


# =============================================================================
# Spike Processing
# =============================================================================

def compute_psth(
    spike_times: np.ndarray,
    event_times: np.ndarray,
    window: tuple = (-0.1, 0.2),
    bin_size: float = 0.01
) -> tuple[np.ndarray,np.ndarray, np.ndarray]:
    """
    Compute peri-stimulus time histogram (PSTH) for a single unit.

    Parameters
    ----------
    spike_times : array of spike times (seconds)
    event_times : array of stimulus/event times (seconds)
    window : (pre, post) in seconds relative to event
    bin_size : bin size in seconds

    Returns
    -------
    psth : array (n_trials, n_bins) of spike counts per bin
    bin_centers : array of bin center times
    """
    n_bins = int((window[1] - window[0]) / bin_size)
    bin_edges = np.linspace(window[0], window[1], n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    n_trials = len(event_times)
    raster = []
    psth = np.zeros((n_trials, n_bins))

    for i, t_event in enumerate(event_times):
        # Get spikes relative to this event
        rel_spikes = spike_times - t_event

        # Keep only spikes in window
        mask = (rel_spikes >= window[0]) & (rel_spikes < window[1])
        trial_spikes = rel_spikes[mask]
        raster.append(trial_spikes)

        # Histogram
        counts, _ = np.histogram(trial_spikes, bins=bin_edges)
        psth[i, :] = counts

    return raster, psth, bin_centers

# Source - https://stackoverflow.com/a
# Posted by Nick ODell
# Retrieved 2025-11-26, License - CC BY-SA 4.0

import scipy.ndimage

def halfgaussian_kernel1d(sigma, radius):
    """
    Computes a 1-D Half-Gaussian convolution kernel.
    """
    sigma2 = sigma * sigma
    x = np.arange(0, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    return phi_x

def halfgaussian_filter1d(input, sigma, axis=-1, output=None,
                      mode="constant", cval=0.0, truncate=4.0):
    """
    Convolves a 1-D Half-Gaussian convolution kernel.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = halfgaussian_kernel1d(sigma, lw)
    origin = -lw // 2
    return scipy.ndimage.convolve1d(input, weights, axis, output, mode, cval, origin)


def psth_to_firing_rate(
    psth: np.ndarray,
    bin_size: float = 0.001,
    smooth_sigma: Optional[float] = None
) -> np.ndarray:
    """
    Convert spike counts to firing rate (spks/s).

    Parameters
    ----------
    psth : array (n_trials, n_bins) of spike counts
    bin_size : bin size in seconds
    smooth_sigma : Gaussian smoothing sigma in bins (None = no smoothing)

    Returns
    -------
    firing_rate : array (n_trials, n_bins) in spks/s
    """
    fr = psth / bin_size  # Convert to spks/s

    if smooth_sigma is not None and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        # make the filter half-gaussian
        #fr = gaussian_filter1d(fr, sigma=smooth_sigma, axis=1)
        fr = halfgaussian_filter1d(fr, sigma=smooth_sigma)

    return fr


def baseline_subtract(
    firing_rate: np.ndarray,
    bin_centers: np.ndarray,
    baseline_window: tuple = (-0.1, 0)
) -> np.ndarray:
    """Subtract baseline firing rate from each trial."""
    mask = (bin_centers >= baseline_window[0]) & (bin_centers < baseline_window[1])
    baseline = np.mean(firing_rate[:, mask], axis=1, keepdims=True)
    return firing_rate - baseline


def get_response_amplitude(
    firing_rate: np.ndarray,
    bin_centers: np.ndarray,
    response_window: tuple = (0, 0.05)
) -> np.ndarray:
    """Extract mean firing rate in response window."""
    mask = (bin_centers >= response_window[0]) & (bin_centers < response_window[1])
    return np.mean(firing_rate[:, mask], axis=1)


# =============================================================================
# Trial Selection
# =============================================================================

def get_trial_indices(
    inflection_trial: int,
    n_total_trials: int,
    pre_window: tuple = (-16, -6),
    post_window: tuple = (0, 10)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get trial indices for pre and post inflection windows.

    Pre: trials from inflection + pre_window[0] to inflection + pre_window[1]
    Post: trials from inflection + post_window[0] to inflection + post_window[1]
    """
    pre_start = max(0, inflection_trial + pre_window[0])
    pre_end = min(n_total_trials, inflection_trial + pre_window[1] + 1)

    post_start = max(0, inflection_trial + post_window[0])
    post_end = min(n_total_trials, inflection_trial + post_window[1] + 1)

    pre_indices = np.arange(pre_start, pre_end)
    post_indices = np.arange(post_start, post_end)

    return pre_indices, post_indices


# =============================================================================
# Pseudosession Method
# =============================================================================

def compute_test_statistic(
    responses: np.ndarray,
    inflection_trial: int,
    n_total_trials: int,
    pre_window: tuple,
    post_window: tuple
) -> float:
    """
    Compute R² test statistic for a given inflection point.

    R² measures how well neural response predicts trial type (pre vs post).
    """
    pre_idx, post_idx = get_trial_indices(
        inflection_trial, n_total_trials, pre_window, post_window
    )

    if len(pre_idx) < 2 or len(post_idx) < 2:
        return 0.0

    pre_responses = responses[pre_idx]
    post_responses = responses[post_idx]

    # Remove NaNs
    pre_valid = pre_responses[~np.isnan(pre_responses)]
    post_valid = post_responses[~np.isnan(post_responses)]

    if len(pre_valid) < 2 or len(post_valid) < 2:
        return 0.0

    # Build regression data
    X = np.concatenate([pre_valid, post_valid]).reshape(-1, 1)
    y = np.concatenate([np.zeros(len(pre_valid)), np.ones(len(post_valid))])

    if np.std(X) == 0:
        return 0.0

    # Linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return max(0, r_squared)


def generate_pseudo_inflection(
    n_total_trials: int,
    pre_window: tuple,
    post_window: tuple,
    min_edge_trials: int = 10
) -> int:
    """
    Generate random pseudo-inflection trial.

    Ensures enough trials exist for both pre and post windows.
    """
    # Need enough trials before for pre_window
    min_trial = max(min_edge_trials, -pre_window[0])

    # Need enough trials after for post_window
    max_trial = min(n_total_trials - min_edge_trials,
                    n_total_trials - post_window[1] - 1)

    if max_trial <= min_trial:
        return n_total_trials // 2

    return np.random.randint(min_trial, max_trial + 1)


def run_pseudosession_test(
    responses: np.ndarray,
    inflection_trial: int,
    config: AnalysisConfig
) -> tuple[float, float, np.ndarray]:
    """
    Run pseudosession test for a single neuron.

    Returns
    -------
    observed_r2 : R² for true inflection
    p_value : proportion of null R² >= observed
    null_distribution : array of null R² values
    """
    n_trials = len(responses)

    # Observed statistic
    observed_r2 = compute_test_statistic(
        responses, inflection_trial, n_trials,
        config.pre_trials, config.post_trials
    )

    # Generate null distribution
    null_r2 = np.zeros(config.n_pseudosessions)

    for i in range(config.n_pseudosessions):
        pseudo_infl = generate_pseudo_inflection(
            n_trials, config.pre_trials, config.post_trials,
            config.min_edge_trials
        )
        null_r2[i] = compute_test_statistic(
            responses, pseudo_infl, n_trials,
            config.pre_trials, config.post_trials
        )

    # p-value: proportion of null >= observed
    p_value = np.mean(null_r2 >= observed_r2)

    return observed_r2, p_value, null_r2


# =============================================================================
# Single Unit Analysis
# =============================================================================

def analyze_single_unit(
    spike_times: np.ndarray,
    stim_times: np.ndarray,
    inflection_trial: int,
    config: AnalysisConfig,
    mouse_id: str = '',
    session_id: str = '',
    unit_id: int = 0,
    area: str = '',
    reward_group: int = 0
) -> tuple[NeuronResult, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete analysis for a single unit.

    Returns
    -------
    result : NeuronResult
    null_distribution : array of null R²
    firing_rate : (n_trials, n_bins) baseline-subtracted firing rate
    bin_centers : time axis
    """
    n_trials = len(stim_times)

    # Compute raster and PSTH
    raster, psth, bin_centers = compute_psth(spike_times, stim_times, window=config.psth_window, bin_size=config.bin_size)

    # Convert to firing rate and smooth slightly
    firing_rate = psth_to_firing_rate(psth, config.bin_size, smooth_sigma=1)

    # Baseline subtract
    fr_baselined = baseline_subtract(firing_rate, bin_centers, config.baseline_window)

    # Get 50ms response for each trial
    responses = get_response_amplitude(fr_baselined, bin_centers, config.response_window)

    # Get trial indices
    pre_idx, post_idx = get_trial_indices(
        inflection_trial, n_trials, config.pre_trials, config.post_trials
    )

    # Pre/post means
    pre_mean = np.nanmean(responses[pre_idx]) if len(pre_idx) > 0 else np.nan
    post_mean = np.nanmean(responses[post_idx]) if len(post_idx) > 0 else np.nan
    shift = post_mean - pre_mean

    # Overall firing rate
    total_time = spike_times.max() - spike_times.min() if len(spike_times) > 1 else 1
    mean_fr = len(spike_times) / total_time

    # Run pseudosession test
    observed_r2, p_value, null_dist = run_pseudosession_test(
        responses, inflection_trial, config
    )

    null_mean = np.mean(null_dist)
    null_std = np.std(null_dist)
    z_score = (observed_r2 - null_mean) / null_std if null_std > 0 else 0

    result = NeuronResult(
        mouse_id=mouse_id,
        session_id=session_id,
        unit_id=unit_id,
        area=area,
        reward_group=reward_group,
        inflection_trial=inflection_trial,
        n_total_trials=n_trials,
        pre_response=pre_mean,
        post_response=post_mean,
        shift_magnitude=shift,
        observed_statistic=observed_r2,
        null_mean=null_mean,
        null_std=null_std,
        p_value=p_value,
        z_score=z_score,
        n_pre_trials=len(pre_idx),
        n_post_trials=len(post_idx),
        mean_firing_rate=mean_fr
    )

    return result, null_dist, raster, firing_rate, fr_baselined, bin_centers


# =============================================================================
# Visualization
# =============================================================================

def plot_unit_analysis(
    spike_raster: list,
    firing_rate: np.ndarray,
    firing_rate_baselined: np.ndarray,
    bin_centers: np.ndarray,
    inflection_trial: int,
    result: NeuronResult,
    null_distribution: np.ndarray,
    config: AnalysisConfig,
    p_mean: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create comprehensive figure for single-unit analysis.

    5 panels:
    1. PSTH heatmap (raster-style)
    2. Average PSTH pre vs post
    3. Trial-by-trial 50ms response
    4. Behavioral curve (p_mean)
    5. Null distribution
    """
    set_plot_style()

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)

    n_trials = firing_rate.shape[0]
    times_ms = bin_centers * 1000

    # Get trial indices
    pre_idx, post_idx = get_trial_indices(
        inflection_trial, n_trials, config.pre_trials, config.post_trials
    )

    # Get responses
    responses = get_response_amplitude(firing_rate, bin_centers, config.response_window)

    # Colors
    c_whisker = 'forestgreen' if result.reward_group==1 else 'crimson'
    c_pre = '#e240ff'
    c_post = '#5200bd'
    c_null = '#bdc3c7'
    c_obs = '#ff3f14'

    # =========================================================================
    # Panel 1: PSTH Heatmap
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    fr_raster=False
    if fr_raster:
        vmax = np.nanpercentile(np.abs(firing_rate), 95)
        if vmax == 0:
            vmax = 1
        # Make a scatter plot instead
        im = ax1.imshow(firing_rate, aspect='auto', cmap=cmr.sunburst,
                        #vmin=-vmax, vmax=vmax,
                        extent=[times_ms[0], times_ms[-1], n_trials, 0])

        # Mark inflection and events
        ax1.axhline(inflection_trial, color=c_whisker, linewidth=2, linestyle='-')
        #ax1.axvline(0, color='k', linewidth=1, linestyle='--')
        #ax1.axvline(config.response_window[1] * 1000, color='orange',
        #            linewidth=1.5, linestyle=':')
        ax1.axvline(0, color='k', linewidth=1, linestyle='--', alpha=0.5)
        ax1.axvspan(config.response_window[0] * 1000, config.response_window[1] * 1000,
                    alpha=0.2, color='dimgrey')

        cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
        cbar.set_label('Firing rate (spks/s)')

    else:
        # Plot spike raster from early to late (top to bottom)
        for i in range(n_trials):
            xs = spike_raster[i]
            ys = np.full(xs.shape, i)
            ax1.scatter(xs,ys,s=6,marker='o',color='k',edgecolor='none')
        # Mark inflection and events
        ax1.axhline(inflection_trial, color=c_whisker, linewidth=2, linestyle='-')
        ax1.axvline(0, color='k',
                    linewidth=1, linestyle='--', alpha=0.5)
        ax1.axvspan(config.response_window[0], config.response_window[1],
                    alpha=0.2, color='dimgrey')
        ax1.set_xlim(-0.1, 0.2)

        ax1.invert_yaxis()

    # Shade pre/post windows
    if len(pre_idx) > 0:
        ax1.axhspan(pre_idx[0], pre_idx[-1] + 1, alpha=0.15, color=c_pre)
    if len(post_idx) > 0:
        ax1.axhspan(post_idx[0], post_idx[-1] + 1, alpha=0.15, color=c_post)

    ax1.set_xlabel('Time from whisker stimulus (s)')
    ax1.set_ylabel('Trial number')
    ax1.set_title(f'Unit {result.unit_id} | {result.area}\nMean FR: {result.mean_firing_rate:.1f} spks/s')


    # =========================================================================
    # Panel 2: Average PSTH
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    if len(pre_idx) > 0:
        pre_fr = firing_rate_baselined[pre_idx]
        pre_mean = np.nanmean(pre_fr, axis=0)
        pre_sem = np.nanstd(pre_fr, axis=0) / np.sqrt(len(pre_idx))

        ax2.fill_between(times_ms, pre_mean - pre_sem, pre_mean + pre_sem,
                         alpha=0.3, color=c_pre)
        ax2.plot(times_ms, pre_mean, color=c_pre, linewidth=2,
                 label=f'Pre ({config.pre_trials[0]} to {config.pre_trials[1]})')

    if len(post_idx) > 0:
        post_fr = firing_rate_baselined[post_idx]
        post_mean = np.nanmean(post_fr, axis=0)
        post_sem = np.nanstd(post_fr, axis=0) / np.sqrt(len(post_idx))

        ax2.fill_between(times_ms, post_mean - post_sem, post_mean + post_sem,
                         alpha=0.3, color=c_post)
        ax2.plot(times_ms, post_mean, color=c_post, linewidth=2,
                 label=f'Post ({config.post_trials[0]} to {config.post_trials[1]})')

    ax2.axvline(0, color='k', linewidth=1, linestyle='--', alpha=0.5)
    ax2.axvspan(config.response_window[0]*1000, config.response_window[1]*1000,
                alpha=0.2, color='dimgrey')

    ax2.set_xlabel('Time from stimulus (ms)')
    ax2.set_ylabel('ΔFiring rate (spks/s)')
    ax2.set_title('PSTH: pre. vs post-inflection')
    ax2.legend(loc='upper right', frameon=False)

    # =========================================================================
    # Panel 3: Trial-by-trial response
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    trials = np.arange(n_trials)

    # All trials
    ax3.scatter(trials, responses, c='gray', alpha=0.3, s=15)

    # Highlight pre/post
    if len(pre_idx) > 0:
        ax3.scatter(pre_idx, responses[pre_idx], c=c_pre, s=30, alpha=0.8,
                    label=f'Pre (n={len(pre_idx)})')
    if len(post_idx) > 0:
        ax3.scatter(post_idx, responses[post_idx], c=c_post, s=30, alpha=0.8,
                    label=f'Post (n={len(post_idx)})')
    ax3.axvline(inflection_trial, color=c_whisker, linewidth=2, linestyle='--')

    # Means
    if not np.isnan(result.pre_response):
        ax3.axhline(result.pre_response, color=c_pre, linewidth=2, alpha=0.7)
    if not np.isnan(result.post_response):
        ax3.axhline(result.post_response, color=c_post, linewidth=2, alpha=0.7)

    # Shift annotation
    if not np.isnan(result.shift_magnitude):
        mid_y = (result.pre_response + result.post_response) / 2
        ax3.annotate('', xy=(inflection_trial + 3, result.post_response),
                     xytext=(inflection_trial + 3, result.pre_response),
                     arrowprops=dict(arrowstyle='<->', color='k', lw=1.5))
        ax3.text(inflection_trial + 5, mid_y, f'Δ={result.shift_magnitude:.1f} spks/s',
                 fontsize=9, va='center')

    ax3.set_xlabel('Trial number')
    ax3.set_ylabel('Response (0-50ms, spks/s)')
    ax3.set_title('Trial-by-trial response')
    ax3.legend(loc='upper left', frameon=False)

    # =========================================================================
    # Panel 4: Behavioral curve
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    if p_mean is not None and len(p_mean) == n_trials:

        ax4.plot(trials, p_mean, c='k', ls='-', linewidth=2, label='P(lick)')
        ax4.axvline(inflection_trial, color=c_whisker, linewidth=2,
                    linestyle='--', label=f'Inflection (trial {inflection_trial})')
        ax4.set_ylabel('P(lick)')
        ax4.set_ylim(-0.05, 1.05)

        ax4_bis = ax4.twinx()
        ax4_bis.tick_params(axis='y', colors='grey')
        ax4_bis.plot(trials, p_mean-p_chance, c='grey', ls='--', linewidth=2, label=r'$\Delta$P(lick)')
        ax4_bis.set_ylabel(r'$\Delta$P(lick)', color='grey')
        ax4_bis.set_ylim(-1.05, 1.05)
        ax4_bis.spines['right'].set_color('grey')
        ax4_bis.tick_params(axis='y', colors='grey')
        ax4_bis.axhline(y=0, c='grey',lw=1)
    else:
        # Just show inflection marker
        ax4.axvline(inflection_trial, color=c_whisker, linewidth=2, linestyle='-')
        ax4.text(inflection_trial, 0.5, f'Inflection\ntrial {inflection_trial}',
                 ha='center', va='center', fontsize=10)
        ax4.set_ylabel('Behavioral metric')

    ax4.set_xlabel('Trial number')
    ax4.set_title('Learning curve')
    ax4.legend(loc='best', frameon=False)

    # =========================================================================
    # Panel 5: Null distribution
    # =========================================================================
    ax5 = fig.add_subplot(gs[0, 2])

    ax5.hist(null_distribution, bins=50, color=c_null, alpha=0.7,
             edgecolor='white', density=True, label='Null')

    ax5.axvline(result.observed_statistic, color=c_obs, linewidth=3,
                label=f'Observed R² = {result.observed_statistic:.3f}')

    # Significance region
    pct_95 = np.percentile(null_distribution, 95)
    ax5.axvspan(pct_95, max(null_distribution.max(), result.observed_statistic) * 1.1,
                alpha=0.2, color=c_obs)

    # p-value annotation
    sig = '***' if result.p_value < 0.001 else '**' if result.p_value < 0.01 else '*' if result.p_value < 0.05 else 'n.s.'
    ymax = ax5.get_ylim()[1]
    ax5.text(result.observed_statistic, ymax * 0.95,
             f'p = {result.p_value:.4f} {sig}\nz = {result.z_score:.2f}',
             ha='center', va='top', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax5.set_xlabel('R² (test statistic)')
    ax5.set_ylabel('Density')
    ax5.set_title(f'Pseudosession Test (n={config.n_pseudosessions})')
    ax5.legend(loc='upper right', frameon=False)

    # =========================================================================
    # Panel 6: Summary text
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    summary_text = f"""
    Mouse: {result.mouse_id}
    Session: {result.session_id}
    Unit: {result.unit_id}
    Area: {result.area}
    Reward Group: {result.reward_group}
    
    Inflection Trial: {result.inflection_trial}
    Total Trials: {result.n_total_trials}
    
    Pre-inflection (n={result.n_pre_trials}):
      Mean response: {result.pre_response:.2f} spks/s
    
    Post-inflection (n={result.n_post_trials}):
      Mean response: {result.post_response:.2f} spks/s
    
    Shift: {result.shift_magnitude:+.2f} spks/s
    
    Statistics:
      Observed R²: {result.observed_statistic:.4f}
      Null R² (mean±std): {result.null_mean:.4f} ± {result.null_std:.4f}
      Z-score: {result.z_score:.2f}
      p-value: {result.p_value:.4f} {sig}
    """

    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Main title
    reward_group_txt = 'R+' if result.reward_group else 'R'
    fig.suptitle(
        f'Inflection analysis | {result.mouse_id}, {reward_group_txt} | Unit {result.unit_id} | {result.area}',
        fontsize=14, fontweight='normal'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_population_summary(
    results: List[NeuronResult],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Create summary figure across all units."""
    set_plot_style()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Extract data
    areas = sorted(set(r.area for r in results))
    rgs = sorted(set(r.reward_group for r in results))

    shifts = np.array([r.shift_magnitude for r in results])
    r2s = np.array([r.observed_statistic for r in results])
    pvals = np.array([r.p_value for r in results])
    zscores = np.array([r.z_score for r in results])

    c_rg = {0: '#3498db', 1: '#e74c3c'}

    # Panel 1: Shift distribution by area
    ax = axes[0, 0]
    positions = []
    labels = []
    data_list = []
    colors_list = []

    for i, area in enumerate(areas):
        for j, rg in enumerate(rgs):
            mask = np.array([(r.area == area and r.reward_group == rg) for r in results])
            if mask.any():
                data_list.append(shifts[mask])
                positions.append(i + (j - 0.5) * 0.3)
                labels.append(f'{area}\nRG{rg}')
                colors_list.append(c_rg.get(rg, 'gray'))

    if data_list:
        vp = ax.violinplot(data_list, positions=positions, widths=0.25, showmeans=True)
        for i, pc in enumerate(vp['bodies']):
            pc.set_facecolor(colors_list[i])
            pc.set_alpha(0.7)

    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas)
    ax.set_xlabel('Brain Area')
    ax.set_ylabel('Shift (spks/s)')
    ax.set_title('Response Shift Distribution')

    # Panel 2: R² distribution
    ax = axes[0, 1]
    data_list = []
    positions = []
    colors_list = []

    for i, area in enumerate(areas):
        for j, rg in enumerate(rgs):
            mask = np.array([(r.area == area and r.reward_group == rg) for r in results])
            if mask.any():
                data_list.append(r2s[mask])
                positions.append(i + (j - 0.5) * 0.3)
                colors_list.append(c_rg.get(rg, 'gray'))

    if data_list:
        vp = ax.violinplot(data_list, positions=positions, widths=0.25, showmeans=True)
        for i, pc in enumerate(vp['bodies']):
            pc.set_facecolor(colors_list[i])
            pc.set_alpha(0.7)

    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas)
    ax.set_xlabel('Brain Area')
    ax.set_ylabel('R²')
    ax.set_title('Test Statistic Distribution')

    # Panel 3: p-value histogram
    ax = axes[0, 2]
    ax.hist(pvals, bins=20, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(0.05, color='r', linestyle='--', linewidth=2, label='α = 0.05')

    n_sig = np.sum(pvals < 0.05)
    ax.text(0.95, 0.95, f'Significant: {n_sig}/{len(pvals)}\n({100*n_sig/len(pvals):.1f}%)',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white'))

    ax.set_xlabel('p-value')
    ax.set_ylabel('Count')
    ax.set_title('p-value Distribution')
    ax.legend()

    # Panel 4: Shift vs R²
    ax = axes[1, 0]
    for rg in rgs:
        mask = np.array([r.reward_group == rg for r in results])
        ax.scatter(np.abs(shifts[mask]), r2s[mask], c=c_rg.get(rg, 'gray'),
                   alpha=0.6, s=40, label=f'RG {rg}')

    ax.set_xlabel('|Shift| (spks/s)')
    ax.set_ylabel('R²')
    ax.set_title('Effect Size vs Test Statistic')
    ax.legend(frameon=False)

    # Panel 5: Z-score distribution
    ax = axes[1, 1]
    ax.hist(zscores, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(1.96, color='r', linestyle='--', linewidth=1.5, label='z = 1.96')
    ax.axvline(0, color='k', linewidth=0.5)

    ax.set_xlabel('Z-score')
    ax.set_ylabel('Count')
    ax.set_title('Z-score Distribution')
    ax.legend()

    # Panel 6: Proportion significant by area
    ax = axes[1, 2]
    x = np.arange(len(areas))
    width = 0.35

    for j, rg in enumerate(rgs):
        props = []
        ns = []
        for area in areas:
            mask = np.array([(r.area == area and r.reward_group == rg) for r in results])
            area_pvals = pvals[mask]
            if len(area_pvals) > 0:
                props.append(np.mean(area_pvals < 0.05))
                ns.append(len(area_pvals))
            else:
                props.append(0)
                ns.append(0)

        bars = ax.bar(x + (j - 0.5) * width, props, width,
                      label=f'RG {rg}', color=c_rg.get(rg, 'gray'), alpha=0.8)

        # Add count labels
        for xi, (bar, n) in enumerate(zip(bars, ns)):
            if n > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'n={n}', ha='center', va='bottom', fontsize=8)

    ax.axhline(0.05, color='gray', linestyle=':', label='Chance (5%)')
    ax.set_xticks(x)
    ax.set_xticklabels(areas)
    ax.set_xlabel('Brain Area')
    ax.set_ylabel('Proportion Significant')
    ax.set_title('Significant Units by Area')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)

    plt.tight_layout()
    fig.align_xlabels()
    fig.align_ylabels()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Main Analysis Functions
# =============================================================================

def analyze_session(
    session_data: Dict[str, Any],
    config: AnalysisConfig,
    save_dir: Optional[Path] = None,
    plot_individual: bool = True,
    max_units: Optional[int] = None
) -> List[NeuronResult]:
    """
    Analyze all units in a session.

    Parameters
    ----------
    session_data : output from load_nwb_session
    config : AnalysisConfig
    save_dir : directory for saving figures
    plot_individual : whether to plot each unit
    max_units : limit number of units to analyze (for testing)

    Returns
    -------
    results : list of NeuronResult
    """
    units = session_data['units']
    trials = session_data['trials']
    metadata = session_data['metadata']

    stim_times = trials['stim_time']
    inflection_trial = trials['inflection_trial']
    reward_group = trials['reward_group']
    p_mean = trials.get('p_mean', None)

    if max_units is not None:
        units = units[:max_units]

    results = []

    print(f"\nAnalyzing {len(units)} units...")
    print(f"Inflection trial: {inflection_trial}")
    print(f"Total trials: {len(stim_times)}")
    print("-" * 60)

    for unit in units:
        result, null_dist, raster, fr, fr_bas, bin_centers = analyze_single_unit(
            spike_times=unit['spike_times'],
            stim_times=stim_times,
            inflection_trial=inflection_trial,
            config=config,
            mouse_id=metadata['mouse_id'],
            session_id=metadata['session_id'],
            neuron_id=unit['neuron_id'],
            area=unit['area'],
            reward_group=reward_group
        )

        results.append(result)

        sig = '***' if result.p_value < 0.001 else '**' if result.p_value < 0.01 else '*' if result.p_value < 0.05 else ''
        #print(f"Unit {result.neuron_id:4d} ({result.area:>6s}): "
        #      f"Δ={result.shift_magnitude:+6.1f} spks/s, R²={result.observed_statistic:.3f}, "
        #      f"p={result.p_value:.4f} {sig}")

        # Individual plot
        if plot_individual and save_dir:
            fig = plot_unit_analysis(
                raster, fr, fr_bas, bin_centers, inflection_trial, result, null_dist, config, p_mean,
                save_path=save_dir / f'{metadata["mouse_id"]}_unit{result.neuron_id}_analysis.png'
            )
            plt.close(fig)

    return results


def analyze_nwb_files(
    nwb_paths: List[Path],
    config: AnalysisConfig = None,
    save_dir: Optional[Path] = None,
    plot_individual: bool = True,
    max_units_per_session: Optional[int] = None
) -> List[NeuronResult]:
    """
    Main entry point: analyze multiple NWB files.

    Parameters
    ----------
    nwb_paths : list of paths to NWB files
    config : AnalysisConfig (uses defaults if None)
    save_dir : directory for saving figures and results
    plot_individual : whether to plot each unit
    max_units_per_session : limit units per session (for testing)

    Returns
    -------
    all_results : list of NeuronResult from all sessions
    """
    if config is None:
        config = AnalysisConfig()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Neuropixels Inflection Analysis (Pseudosession Method)")
    print("=" * 70)
    print(f"Pre-inflection trials: {config.pre_trials[0]} to {config.pre_trials[1]}")
    print(f"Post-inflection trials: {config.post_trials[0]} to {config.post_trials[1]}")
    print(f"Response window: {config.response_window[0]*1000:.0f}-{config.response_window[1]*1000:.0f} ms")
    print(f"N pseudosessions: {config.n_pseudosessions}")
    print(f"Unit filter: bc_label == '{config.bc_label_filter}'")
    print("=" * 70)

    all_results = []

    for nwb_path in nwb_paths:
        print(f"\n{'='*70}")
        print(f"Processing: {nwb_path.name}")
        print(f"{'='*70}")

        try:
            session_data = load_nwb_session(nwb_path, config)

            session_save_dir = None
            if save_dir:
                session_save_dir = save_dir / nwb_path.stem
                session_save_dir.mkdir(exist_ok=True)

            results = analyze_session(
                session_data, config, session_save_dir,
                plot_individual, max_units_per_session
            )

            all_results.extend(results)

        except Exception as e:
            warnings.warn(f"Failed to process {nwb_path}: {e}")
            import traceback
            traceback.print_exc()

    # Summary statistics
    if all_results:
        pvals = np.array([r.p_value for r in all_results])
        n_sig = np.sum(pvals < 0.05)

        print("\n" + "=" * 70)
        print("OVERALL SUMMARY")
        print("=" * 70)
        print(f"Total units analyzed: {len(all_results)}")
        print(f"Significant (p < 0.05): {n_sig} ({100*n_sig/len(all_results):.1f}%)")
        print(f"Significant (p < 0.01): {np.sum(pvals < 0.01)}")
        print(f"Significant (p < 0.001): {np.sum(pvals < 0.001)}")

        # By area
        areas = sorted(set(r.area for r in all_results))
        print("\nBy area:")
        for area in areas:
            area_pvals = [r.p_value for r in all_results if r.area == area]
            n_area_sig = sum(p < 0.05 for p in area_pvals)
            print(f"  {area}: {n_area_sig}/{len(area_pvals)} significant "
                  f"({100*n_area_sig/len(area_pvals):.1f}%)")

        # Population summary plot
        if save_dir and len(all_results) > 1:
            fig = plot_population_summary(
                all_results, save_path=save_dir / 'population_summary.png'
            )
            plt.close(fig)
            print(f"\nFigures saved to {save_dir}")

    return all_results


def save_results_to_csv(results: List[NeuronResult], save_path: Path):
    """Save results to CSV file."""
    import csv

    fieldnames = [
        'mouse_id', 'session_id', 'neuron_id', 'area', 'reward_group',
        'inflection_trial', 'n_total_trials', 'n_pre_trials', 'n_post_trials',
        'pre_response', 'post_response', 'shift_magnitude',
        'observed_statistic', 'null_mean', 'null_std', 'p_value', 'z_score',
        'mean_firing_rate'
    ]

    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow({
                'mouse_id': r.mouse_id,
                'session_id': r.session_id,
                'neuron_id': r.neuron_id,
                'area': r.area,
                'reward_group': r.reward_group,
                'inflection_trial': r.inflection_trial,
                'n_total_trials': r.n_total_trials,
                'n_pre_trials': r.n_pre_trials,
                'n_post_trials': r.n_post_trials,
                'pre_response': r.pre_response,
                'post_response': r.post_response,
                'shift_magnitude': r.shift_magnitude,
                'observed_statistic': r.observed_statistic,
                'null_mean': r.null_mean,
                'null_std': r.null_std,
                'p_value': r.p_value,
                'z_score': r.z_score,
                'mean_firing_rate': r.mean_firing_rate
            })

    print(f"Results saved to {save_path}")


# =============================================================================
# Example with Synthetic Data
# =============================================================================

def create_synthetic_nwb_data(
    mouse_id: str = 'mouse1',
    session_id: str = 'session1',
    n_units: int = 10,
    n_trials: int = 150,
    inflection_trial: int = 75,
    effect_sizes: Optional[List[float]] = None,
    areas: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create synthetic session data mimicking NWB structure."""

    if effect_sizes is None:
        # Mix of effect sizes
        effect_sizes = [0.8, 0.6, 0.4, 0.2, 0.0] * 2
    if areas is None:
        areas = ['S1', 'S1', 'S1', 'V1', 'V1'] * 2

    # Generate trial times
    trial_duration = 2.0  # seconds
    stim_times = np.arange(n_trials) * trial_duration + 1.0  # Start at 1s

    # Generate behavioral curve (p_mean)
    trials = np.arange(n_trials)
    p_mean = 1 / (1 + np.exp(-0.1 * (trials - inflection_trial)))  # Sigmoid
    p_mean += np.random.randn(n_trials) * 0.05  # Add noise
    p_mean = np.clip(p_mean, 0, 1)

    # Generate units
    units = []
    for i in range(n_units):
        effect = effect_sizes[i % len(effect_sizes)]
        area = areas[i % len(areas)]

        # Generate spike times
        spike_times = []
        baseline_rate = 10  # spks/s

        for trial_idx, t_stim in enumerate(stim_times):
            # Baseline spikes
            n_baseline = np.random.poisson(baseline_rate * trial_duration)
            trial_spikes = np.random.uniform(t_stim - 0.5, t_stim + 1.5, n_baseline)

            # Evoked response (50ms window)
            if trial_idx < inflection_trial:
                evoked_rate = 20  # spks/s
            else:
                evoked_rate = 20 + effect * 50  # Increased rate

            n_evoked = np.random.poisson(evoked_rate * 0.05)  # 50ms
            evoked_spikes = np.random.uniform(t_stim, t_stim + 0.05, n_evoked)

            spike_times.extend(trial_spikes)
            spike_times.extend(evoked_spikes)

        spike_times = np.sort(spike_times)

        units.append({
            'neuron_id': i,
            'unit_index': i,
            'spike_times': spike_times,
            'area': area,
            'n_spikes': len(spike_times)
        })

    return {
        'units': units,
        'trials': {
            'stim_time': stim_times,
            'start_time': stim_times - 0.5,
            'stop_time': stim_times + 1.5,
            'p_mean': p_mean,
            'inflection_trial': inflection_trial,
            'reward_group': 0,
            'n_trials': n_trials
        },
        'metadata': {
            'mouse_id': mouse_id,
            'session_id': session_id,
            'session_start': None
        }
    }


if __name__ == "__main__":

    np.random.seed(42)

    # Configure
    config = AnalysisConfig(
        pre_trials=(-16, -6),
        post_trials=(0, 10),
        response_window=(0.005, 0.05),
        n_pseudosessions=1000,  # Fewer for demo
    )

    # Output directory
    OUTPUT_DIR = r'M:\analysis\Axel_Bisi\combined_results'


    # Load and format NWB file + add HMM results
    # ---------------------------------------
    import glob
    import os
    nwb_file_list = glob.glob(os.path.join(r"M:\analysis\Axel_Bisi\NWBFull_bis", '*.nwb'))
    print('Available files:', nwb_file_list)
    for nwb_file in nwb_file_list[0:2]:
        try:
            unit_table = NWB_reader_functions.get_unit_table(nwb_file)
            if unit_table is None:
                continue
        except Exception as err:
            print(err)


        #nwb_file = Path(r"M:\analysis\Axel_Bisi\NWBFull_bis\AB087_20231017_141901.nwb")
        session_data = load_nwb_session(nwb_path=nwb_file, config=config)


        # Load HMM results
        mouse_id = NWB_reader_functions.get_mouse_id(nwb_file)
        reward_group = NWB_reader_functions.get_session_metadata(nwb_file)['wh_reward']
        curves_df = load_hmm_results.load_mouse_hmm_results([mouse_id])
        inflection_trial = curves_df['learning_trial'].unique()[0]
        if np.isnan(inflection_trial):
            print(f'No inflection trial found for {mouse_id}')
        p_mean = curves_df['p_mean'].values[0]
        p_chance = curves_df['p_chance'].values[0]
        assert len(p_mean)==len(session_data['trials']['stim_time'])

        # Step 2: Add your behavioral data by updating the dict
        session_data['trials'].update({
            'p_mean': p_mean, # optional, for plotting
            'p_chance': p_chance, # optional, for plotting
            'inflection_trial': inflection_trial, # required
            'reward_group': reward_group
        })


        # Analyze
        mouse_output_dir = Path(OUTPUT_DIR, mouse_id, 'whisker_0', 'inflection_analysis')
        mouse_output_dir.mkdir(exist_ok=True)

        results = analyze_session(
            session_data,
            config=config,
            save_dir=mouse_output_dir,
            plot_individual=True
        )

        # Save CSV
        save_results_to_csv(results, mouse_output_dir / 'results.csv')

        # Population summary
        fig = plot_population_summary(results, mouse_output_dir / 'population_summary.png')
        plt.close(fig)

        print(f"\nAll outputs saved to {mouse_output_dir}")
