"""
Population-Level Inflection Analysis for Neuropixels Data (NWB)
================================================================

Analyzes whether POPULATION neural activity follows the behavioral inflection
trial using the pseudosession method. Pools across all neurons (including 
multi-units) to compute a single population-level test.

Key difference from single-neuron analysis:
- Aggregates responses across all neurons per trial
- Computes a single population R² statistic
- Tests whether the population as a whole tracks the behavioral inflection

NWB Data Structure Expected:
- units table: spike_times, bc_label, area columns
- trials table: start_time, stim_time, etc.

Analysis:
- Pre-inflection window: trials -15 to -6 relative to inflection
- Post-inflection window: trials 0 to +9 relative to inflection
- PSTH: -100ms to 200ms post-stim (for visualization)
- Statistical focus: 5-50ms post-stim firing rate
"""

import numpy as np
from pathlib import Path
from pynwb import NWBHDF5IO
from scipy import stats
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any, Tuple
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# Try to import optional dependencies
try:
    import cmasher as cmr
    HAS_CMASHER = True
except ImportError:
    HAS_CMASHER = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PopulationAnalysisConfig:
    """Configuration for population inflection analysis."""
    # Trial windows relative to inflection trial
    pre_trials: tuple = (-15, -6)   # trials -15 to -6 (inclusive), 10 trials
    post_trials: tuple = (0, 9)     # trials 0 to +9 (inclusive), 10 trials

    # Time windows for PSTH (seconds)
    psth_window: tuple = (-0.1, 0.2)      # -100ms to 200ms for visualization
    baseline_window: tuple = (-0.1, 0)     # -100ms to 0 for baseline subtraction
    response_window: tuple = (0.005, 0.05) # 5-50ms for statistics

    # PSTH binning
    bin_size: float = 0.005  # 5ms bins for PSTH

    # Pseudosession parameters
    n_pseudosessions: int = 10000
    min_edge_trials: int = 10  # Minimum trials from session edges

    # Unit filtering - set to None to include ALL units (good + mua)
    bc_label_filter: Optional[str] = None  # None = include all, 'good' = only good
    
    # Population aggregation method
    aggregation_method: str = 'mean'  # 'mean', 'median', or 'sum'
    
    # Normalization per neuron before aggregation
    normalize_neurons: bool = True  # z-score each neuron's responses


@dataclass
class PopulationResult:
    """Results from population-level pseudosession analysis."""
    mouse_id: str
    session_id: str
    n_neurons: int
    areas: List[str]  # List of unique areas
    reward_group: int
    inflection_trial: int
    n_total_trials: int

    # Population response values
    pre_response: float      # Mean population response pre-inflection
    post_response: float     # Mean population response post-inflection
    shift_magnitude: float   # post - pre

    # Statistics
    observed_statistic: float  # Population R²
    null_mean: float
    null_std: float
    p_value: float
    z_score: float

    # Additional info
    n_pre_trials: int
    n_post_trials: int
    neuron_ids: List[int]


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
# NWB Data Loading (modified to include all units)
# =============================================================================

def load_nwb_session_all_units(nwb_path: Path, config: PopulationAnalysisConfig) -> Dict[str, Any]:
    """
    Load Neuropixels session data from NWB file, including ALL units.

    Filters trials to include only:
    - trial_type == 'whisker_trial' (whisker trials only)
    - context != 'passive' (active trials only)
    - perf != 6 (exclude perf==6 trials)

    Parameters
    ----------
    nwb_path : Path to NWB file
    config : PopulationAnalysisConfig with filtering parameters

    Returns
    -------
    session_data : dict with:
        - 'units': list of dicts with unit info (id, spike_times, area, bc_label)
        - 'trials': dict with trial info (stim_times, trial indices, etc.)
        - 'metadata': session metadata

    Note: 'p_mean' and 'inflection_trial' should be added after loading.
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

        # Filter 1: Only whisker trials
        if 'trial_type' in colnames:
            trial_types = trials['trial_type'][:]
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

        # Get filtered trial indices
        valid_trial_indices = np.where(trial_mask)[0]
        n_valid = len(valid_trial_indices)
        print(f"  Final: {n_valid}/{n_total_trials} trials pass all filters")

        if n_valid == 0:
            raise ValueError(f"No valid trials after filtering in {nwb_path}")

        # Extract trial data for valid trials only
        trial_data = {
            'start_time': trials['start_time'][:][valid_trial_indices],
            'stop_time': trials['stop_time'][:][valid_trial_indices],
            'original_trial_indices': valid_trial_indices,
        }

        # Get stim_time
        if 'stim_time' in colnames:
            trial_data['stim_time'] = trials['stim_time'][:][valid_trial_indices]
        else:
            trial_data['stim_time'] = trials['start_time'][:][valid_trial_indices]
            warnings.warn("No 'stim_time' column found, using 'start_time'")

        # Get reward_group if available
        if 'reward_group' in colnames:
            rg = trials['reward_group'][:][valid_trial_indices]
            if np.isscalar(rg):
                trial_data['reward_group'] = int(rg)
            else:
                trial_data['reward_group'] = int(rg[0])
        else:
            trial_data['reward_group'] = 0

        # Store additional trial info
        for col in ['perf', 'context', 'trial_type']:
            if col in colnames:
                trial_data[col] = trials[col][:][valid_trial_indices]

        trial_data['n_trials'] = n_valid

        # =====================================================================
        # Extract ALL units (good + mua)
        # =====================================================================
        units_table = nwbfile.units
        if units_table is None:
            raise ValueError(f"No units table found in {nwb_path}")

        n_units = len(units_table)
        units = []
        n_good = 0
        n_mua = 0

        unit_colnames = units_table.colnames

        for unit_idx in range(n_units):
            # Get bc_label
            bc_label = 'unknown'
            if 'bc_label' in unit_colnames:
                bc_label = units_table['bc_label'][unit_idx]
                if isinstance(bc_label, bytes):
                    bc_label = bc_label.decode()
            
            # Apply filter if specified
            if config.bc_label_filter is not None:
                if bc_label != config.bc_label_filter:
                    continue
            
            # Count by type
            if bc_label == 'good':
                n_good += 1
            elif bc_label == 'mua':
                n_mua += 1

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
            if 'neuron_id' in unit_colnames:
                neuron_id = units_table['neuron_id'][unit_idx]
            else:
                neuron_id = unit_idx

            units.append({
                'neuron_id': neuron_id,
                'neuron_index': unit_idx,
                'spike_times': np.array(spike_times),
                'area': area,
                'bc_label': bc_label,
                'n_spikes': len(spike_times),
            })

        filter_str = f"bc_label=={config.bc_label_filter}" if config.bc_label_filter else "all units"
        print(f"  Loaded {len(units)} units ({filter_str}): {n_good} good, {n_mua} mua")

        return {
            'units': units,
            'trials': trial_data,
            'metadata': metadata
        }


# =============================================================================
# Spike Processing (same as single-neuron version)
# =============================================================================

import scipy.ndimage

def halfgaussian_kernel1d(sigma, radius):
    """Computes a 1-D Half-Gaussian convolution kernel."""
    sigma2 = sigma * sigma
    x = np.arange(0, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    return phi_x

def halfgaussian_filter1d(input, sigma, axis=-1, output=None,
                          mode="constant", cval=0.0, truncate=4.0):
    """Convolves a 1-D Half-Gaussian convolution kernel."""
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    weights = halfgaussian_kernel1d(sigma, lw)
    origin = -lw // 2
    return scipy.ndimage.convolve1d(input, weights, axis, output, mode, cval, origin)


def compute_psth(
    spike_times: np.ndarray,
    event_times: np.ndarray,
    window: tuple = (-0.1, 0.2),
    bin_size: float = 0.005
) -> Tuple[list, np.ndarray, np.ndarray]:
    """
    Compute peri-stimulus time histogram (PSTH) for a single unit.
    """
    n_bins = int((window[1] - window[0]) / bin_size)
    bin_edges = np.linspace(window[0], window[1], n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    n_trials = len(event_times)
    raster = []
    psth = np.zeros((n_trials, n_bins))

    for i, t_event in enumerate(event_times):
        rel_spikes = spike_times - t_event
        mask = (rel_spikes >= window[0]) & (rel_spikes < window[1])
        trial_spikes = rel_spikes[mask]
        raster.append(trial_spikes)
        counts, _ = np.histogram(trial_spikes, bins=bin_edges)
        psth[i, :] = counts

    return raster, psth, bin_centers


def psth_to_firing_rate(
    psth: np.ndarray,
    bin_size: float = 0.005,
    smooth_sigma: Optional[float] = None
) -> np.ndarray:
    """Convert spike counts to firing rate (spks/s)."""
    fr = psth / bin_size

    if smooth_sigma is not None and smooth_sigma > 0:
        fr = halfgaussian_filter1d(fr, sigma=smooth_sigma, axis=1)

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
    response_window: tuple = (0.005, 0.05)
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
    pre_window: tuple = (-15, -6),
    post_window: tuple = (0, 9)
) -> Tuple[np.ndarray, np.ndarray]:
    """Get trial indices for pre and post inflection windows."""
    pre_start = max(0, inflection_trial + pre_window[0])
    pre_end = min(n_total_trials, inflection_trial + pre_window[1] + 1)

    post_start = max(0, inflection_trial + post_window[0])
    post_end = min(n_total_trials, inflection_trial + post_window[1] + 1)

    pre_indices = np.arange(pre_start, pre_end)
    post_indices = np.arange(post_start, post_end)

    return pre_indices, post_indices


# =============================================================================
# Population Response Computation
# =============================================================================

def compute_population_responses(
    units: List[Dict],
    stim_times: np.ndarray,
    config: PopulationAnalysisConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute population-averaged response across all neurons.
    
    Parameters
    ----------
    units : list of unit dicts with spike_times
    stim_times : array of stimulus times
    config : analysis configuration
    
    Returns
    -------
    population_response : (n_trials,) array of population-averaged responses
    all_neuron_responses : (n_neurons, n_trials) array of individual responses
    population_psth : (n_trials, n_bins) population-averaged PSTH
    bin_centers : time axis
    """
    n_trials = len(stim_times)
    n_neurons = len(units)
    
    # Collect responses from all neurons
    all_responses = np.zeros((n_neurons, n_trials))
    all_psths = []
    bin_centers = None
    
    for i, unit in enumerate(units):
        # Compute PSTH
        _, psth, bin_centers = compute_psth(
            unit['spike_times'], stim_times,
            window=config.psth_window, bin_size=config.bin_size
        )
        
        # Convert to firing rate
        fr = psth_to_firing_rate(psth, config.bin_size, smooth_sigma=1)
        
        # Baseline subtract
        fr_baselined = baseline_subtract(fr, bin_centers, config.baseline_window)
        
        # Get response amplitude per trial
        responses = get_response_amplitude(fr_baselined, bin_centers, config.response_window)
        
        all_responses[i, :] = responses
        all_psths.append(fr_baselined)
    
    all_psths = np.array(all_psths)  # (n_neurons, n_trials, n_bins)
    
    # Normalize each neuron's responses if requested
    if config.normalize_neurons:
        # Z-score each neuron's responses across trials
        neuron_means = np.nanmean(all_responses, axis=1, keepdims=True)
        neuron_stds = np.nanstd(all_responses, axis=1, keepdims=True)
        neuron_stds[neuron_stds == 0] = 1  # Avoid division by zero
        all_responses_norm = (all_responses - neuron_means) / neuron_stds
    else:
        all_responses_norm = all_responses
    
    # Aggregate across neurons
    if config.aggregation_method == 'mean':
        population_response = np.nanmean(all_responses_norm, axis=0)
        population_psth = np.nanmean(all_psths, axis=0)
    elif config.aggregation_method == 'median':
        population_response = np.nanmedian(all_responses_norm, axis=0)
        population_psth = np.nanmedian(all_psths, axis=0)
    elif config.aggregation_method == 'sum':
        population_response = np.nansum(all_responses_norm, axis=0)
        population_psth = np.nansum(all_psths, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {config.aggregation_method}")
    
    return population_response, all_responses, population_psth, bin_centers


# =============================================================================
# Pseudosession Method for Population
# =============================================================================

def compute_population_test_statistic(
    population_response: np.ndarray,
    inflection_trial: int,
    n_total_trials: int,
    pre_window: tuple,
    post_window: tuple
) -> float:
    """
    Compute R² test statistic for population response at given inflection point.
    """
    pre_idx, post_idx = get_trial_indices(
        inflection_trial, n_total_trials, pre_window, post_window
    )

    if len(pre_idx) < 2 or len(post_idx) < 2:
        return 0.0

    pre_responses = population_response[pre_idx]
    post_responses = population_response[post_idx]

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
    """Generate random pseudo-inflection trial."""
    min_trial = max(min_edge_trials, -pre_window[0])
    max_trial = min(n_total_trials - min_edge_trials,
                    n_total_trials - post_window[1] - 1)

    if max_trial <= min_trial:
        return n_total_trials // 2

    return np.random.randint(min_trial, max_trial + 1)


def run_population_pseudosession_test(
    population_response: np.ndarray,
    inflection_trial: int,
    config: PopulationAnalysisConfig
) -> Tuple[float, float, np.ndarray]:
    """
    Run pseudosession test for population response.
    
    Returns
    -------
    observed_r2 : R² for true inflection
    p_value : proportion of null R² >= observed
    null_distribution : array of null R² values
    """
    n_trials = len(population_response)

    # Observed statistic
    observed_r2 = compute_population_test_statistic(
        population_response, inflection_trial, n_trials,
        config.pre_trials, config.post_trials
    )

    # Generate null distribution
    null_r2 = np.zeros(config.n_pseudosessions)

    for i in range(config.n_pseudosessions):
        pseudo_infl = generate_pseudo_inflection(
            n_trials, config.pre_trials, config.post_trials,
            config.min_edge_trials
        )
        null_r2[i] = compute_population_test_statistic(
            population_response, pseudo_infl, n_trials,
            config.pre_trials, config.post_trials
        )

    # p-value: proportion of null >= observed
    p_value = np.mean(null_r2 >= observed_r2)

    return observed_r2, p_value, null_r2


# =============================================================================
# Population Analysis
# =============================================================================

def analyze_population(
    session_data: Dict[str, Any],
    config: PopulationAnalysisConfig,
    area_filter: Optional[str] = None
) -> Tuple[PopulationResult, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze population response for a session.
    
    Parameters
    ----------
    session_data : output from load_nwb_session_all_units
    config : PopulationAnalysisConfig
    area_filter : if provided, only include units from this area
    
    Returns
    -------
    result : PopulationResult
    null_distribution : array of null R²
    population_response : (n_trials,) population-averaged responses
    population_psth : (n_trials, n_bins) population-averaged PSTH
    bin_centers : time axis
    """
    units = session_data['units']
    trials = session_data['trials']
    metadata = session_data['metadata']
    
    stim_times = trials['stim_time']
    inflection_trial = int(trials['inflection_trial'])
    reward_group = trials.get('reward_group', 0)
    n_trials = len(stim_times)
    
    # Filter by area if requested
    if area_filter is not None:
        units = [u for u in units if u['area'] == area_filter]
        print(f"  Filtered to {len(units)} units in {area_filter}")
    
    if len(units) == 0:
        raise ValueError("No units to analyze after filtering")
    
    # Get unique areas
    areas = sorted(set(u['area'] for u in units))
    neuron_ids = [u['neuron_id'] for u in units]
    
    print(f"\nAnalyzing population of {len(units)} neurons...")
    print(f"Areas: {areas}")
    print(f"Inflection trial: {inflection_trial}")
    print(f"Total trials: {n_trials}")
    print(f"Aggregation: {config.aggregation_method}, normalize: {config.normalize_neurons}")
    
    # Compute population responses
    population_response, all_responses, population_psth, bin_centers = compute_population_responses(
        units, stim_times, config
    )
    
    # Get trial indices
    pre_idx, post_idx = get_trial_indices(
        inflection_trial, n_trials, config.pre_trials, config.post_trials
    )
    
    # Pre/post means
    pre_mean = np.nanmean(population_response[pre_idx]) if len(pre_idx) > 0 else np.nan
    post_mean = np.nanmean(population_response[post_idx]) if len(post_idx) > 0 else np.nan
    shift = post_mean - pre_mean
    
    # Run pseudosession test
    observed_r2, p_value, null_dist = run_population_pseudosession_test(
        population_response, inflection_trial, config
    )
    
    null_mean = np.mean(null_dist)
    null_std = np.std(null_dist)
    z_score = (observed_r2 - null_mean) / null_std if null_std > 0 else 0
    
    result = PopulationResult(
        mouse_id=metadata['mouse_id'],
        session_id=metadata['session_id'],
        n_neurons=len(units),
        areas=areas,
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
        neuron_ids=neuron_ids
    )
    
    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'
    print(f"\nPopulation result: Δ={shift:+.3f}, R²={observed_r2:.4f}, p={p_value:.4f} {sig}")
    
    return result, null_dist, population_response, population_psth, bin_centers


# =============================================================================
# Visualization
# =============================================================================

def plot_population_analysis(
    population_response: np.ndarray,
    population_psth: np.ndarray,
    bin_centers: np.ndarray,
    inflection_trial: int,
    result: PopulationResult,
    null_distribution: np.ndarray,
    config: PopulationAnalysisConfig,
    all_neuron_responses: Optional[np.ndarray] = None,
    p_mean: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create comprehensive figure for population analysis.
    
    6 panels:
    1. Population PSTH heatmap (trials × time)
    2. Average PSTH pre vs post
    3. Trial-by-trial population response
    4. Individual neuron responses (optional)
    5. Behavioral curve
    6. Null distribution
    """
    set_plot_style()
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    
    n_trials = population_psth.shape[0]
    times_ms = bin_centers * 1000
    
    # Get trial indices
    pre_idx, post_idx = get_trial_indices(
        inflection_trial, n_trials, config.pre_trials, config.post_trials
    )
    
    # Colors
    c_whisker = 'forestgreen' if result.reward_group == 1 else 'crimson'
    c_pre = '#3498db'
    c_post = '#e74c3c'
    c_null = '#bdc3c7'
    c_obs = '#27ae60'
    
    # =========================================================================
    # Panel 1: Population PSTH Heatmap
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    vmax = np.nanpercentile(np.abs(population_psth), 95)
    if vmax == 0:
        vmax = 1
    
    cmap = cmr.sunburst if HAS_CMASHER else 'viridis'
    im = ax1.imshow(population_psth, aspect='auto', cmap=cmap,
                    extent=[times_ms[0], times_ms[-1], n_trials, 0])
    
    # Mark inflection and windows
    ax1.axhline(inflection_trial, color=c_whisker, linewidth=2, linestyle='-')
    if len(pre_idx) > 0:
        ax1.axhspan(pre_idx[0], pre_idx[-1]+1, alpha=0.15, color=c_pre)
    if len(post_idx) > 0:
        ax1.axhspan(post_idx[0], post_idx[-1]+1, alpha=0.15, color=c_post)
    
    ax1.axvline(0, color='k', linewidth=1, linestyle='--')
    ax1.axvline(config.response_window[1]*1000, color='orange', linewidth=1.5, linestyle=':')
    
    ax1.set_xlabel('Time from stimulus (ms)')
    ax1.set_ylabel('Trial number')
    ax1.set_title(f'Population PSTH (n={result.n_neurons} neurons)')
    
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Mean firing rate (spks/s)')
    
    # =========================================================================
    # Panel 2: Average PSTH pre vs post
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    if len(pre_idx) > 0:
        pre_psth = population_psth[pre_idx]
        pre_mean_psth = np.nanmean(pre_psth, axis=0)
        pre_sem_psth = np.nanstd(pre_psth, axis=0) / np.sqrt(len(pre_idx))
        
        ax2.fill_between(times_ms, pre_mean_psth - pre_sem_psth, 
                         pre_mean_psth + pre_sem_psth, alpha=0.3, color=c_pre)
        ax2.plot(times_ms, pre_mean_psth, color=c_pre, linewidth=2,
                 label=f'Pre ({config.pre_trials[0]} to {config.pre_trials[1]})')
    
    if len(post_idx) > 0:
        post_psth = population_psth[post_idx]
        post_mean_psth = np.nanmean(post_psth, axis=0)
        post_sem_psth = np.nanstd(post_psth, axis=0) / np.sqrt(len(post_idx))
        
        ax2.fill_between(times_ms, post_mean_psth - post_sem_psth,
                         post_mean_psth + post_sem_psth, alpha=0.3, color=c_post)
        ax2.plot(times_ms, post_mean_psth, color=c_post, linewidth=2,
                 label=f'Post ({config.post_trials[0]} to {config.post_trials[1]})')
    
    ax2.axvline(0, color='k', linewidth=1, linestyle='--', alpha=0.5)
    ax2.axvspan(config.response_window[0]*1000, config.response_window[1]*1000,
                alpha=0.2, color='yellow')
    
    ax2.set_xlabel('Time from stimulus (ms)')
    ax2.set_ylabel('Population firing rate (spks/s)')
    ax2.set_title('Population PSTH: Pre vs Post')
    ax2.legend(loc='upper right', frameon=False)
    
    # =========================================================================
    # Panel 3: Trial-by-trial population response
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    trials_arr = np.arange(n_trials)
    
    # All trials
    ax3.scatter(trials_arr, population_response, c='gray', alpha=0.3, s=20)
    
    # Highlight pre/post
    if len(pre_idx) > 0:
        ax3.scatter(pre_idx, population_response[pre_idx], c=c_pre, s=40, alpha=0.8,
                    label=f'Pre (n={len(pre_idx)})')
    if len(post_idx) > 0:
        ax3.scatter(post_idx, population_response[post_idx], c=c_post, s=40, alpha=0.8,
                    label=f'Post (n={len(post_idx)})')
    
    ax3.axvline(inflection_trial, color='k', linewidth=2, linestyle='--')
    
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
        ax3.text(inflection_trial + 5, mid_y, f'Δ={result.shift_magnitude:.3f}',
                 fontsize=9, va='center')
    
    norm_str = "z-scored" if config.normalize_neurons else "raw"
    ax3.set_xlabel('Trial number')
    ax3.set_ylabel(f'Population response ({norm_str})')
    ax3.set_title('Trial-by-trial Population Response')
    ax3.legend(loc='upper left', frameon=False)
    
    # =========================================================================
    # Panel 4: Behavioral curve
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    if p_mean is not None and len(p_mean) == n_trials:
        ax4.plot(trials_arr, p_mean, 'k-', linewidth=2, label='P(lick)')
        ax4.axvline(inflection_trial, color=c_whisker, linewidth=2,
                    linestyle='--', label=f'Inflection (trial {inflection_trial})')
        ax4.set_ylabel('P(lick)')
        ax4.set_ylim(-0.05, 1.05)
    else:
        ax4.axvline(inflection_trial, color=c_whisker, linewidth=2, linestyle='--')
        ax4.text(inflection_trial, 0.5, f'Inflection\ntrial {inflection_trial}',
                 ha='center', va='center', fontsize=10)
        ax4.set_ylabel('Behavioral metric')
    
    ax4.set_xlabel('Trial number')
    ax4.set_title('Behavioral Curve')
    ax4.legend(loc='best', frameon=False)
    
    # =========================================================================
    # Panel 5: Null distribution
    # =========================================================================
    ax5 = fig.add_subplot(gs[0, 2])
    
    ax5.hist(null_distribution, bins=50, color=c_null, alpha=0.7,
             edgecolor='white', density=True, label='Null')
    
    ax5.axvline(result.observed_statistic, color=c_obs, linewidth=3,
                label=f'Observed R² = {result.observed_statistic:.4f}')
    
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
    
    areas_str = ', '.join(result.areas[:5])
    if len(result.areas) > 5:
        areas_str += f', ... ({len(result.areas)} total)'
    
    summary_text = f"""
    POPULATION ANALYSIS SUMMARY
    ===========================
    
    Mouse: {result.mouse_id}
    Session: {result.session_id}
    Reward Group: {result.reward_group}
    
    Population:
      N neurons: {result.n_neurons}
      Areas: {areas_str}
      Aggregation: {config.aggregation_method}
      Normalized: {config.normalize_neurons}
    
    Trial Info:
      Inflection Trial: {result.inflection_trial}
      Total Trials: {result.n_total_trials}
      Pre window: {result.n_pre_trials} trials
      Post window: {result.n_post_trials} trials
    
    Response:
      Pre mean: {result.pre_response:.4f}
      Post mean: {result.post_response:.4f}
      Shift: {result.shift_magnitude:+.4f}
    
    Statistics:
      Observed R²: {result.observed_statistic:.4f}
      Null R² (mean±std): {result.null_mean:.4f} ± {result.null_std:.4f}
      Z-score: {result.z_score:.2f}
      p-value: {result.p_value:.4f} {sig}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Main title
    fig.suptitle(
        f'Population Inflection Analysis | {result.mouse_id} | n={result.n_neurons} neurons',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_session_population(
    session_data: Dict[str, Any],
    config: PopulationAnalysisConfig,
    save_dir: Optional[Path] = None,
    area_filter: Optional[str] = None
) -> PopulationResult:
    """
    Analyze population response for a session.
    
    Parameters
    ----------
    session_data : output from load_nwb_session_all_units, with inflection_trial added
    config : PopulationAnalysisConfig
    save_dir : directory for saving figures
    area_filter : if provided, only include units from this area
    
    Returns
    -------
    result : PopulationResult
    """
    trials = session_data['trials']
    metadata = session_data['metadata']
    p_mean = trials.get('p_mean', None)
    
    # Run population analysis
    result, null_dist, pop_response, pop_psth, bin_centers = analyze_population(
        session_data, config, area_filter
    )
    
    # Also get individual neuron responses for optional plotting
    units = session_data['units']
    if area_filter:
        units = [u for u in units if u['area'] == area_filter]
    
    _, all_neuron_responses, _, _ = compute_population_responses(
        units, trials['stim_time'], config
    )
    
    # Plot
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        area_suffix = f"_{area_filter}" if area_filter else ""
        fig = plot_population_analysis(
            pop_response, pop_psth, bin_centers,
            int(trials['inflection_trial']), result, null_dist, config,
            all_neuron_responses, p_mean,
            save_path=save_dir / f'{metadata["mouse_id"]}_population{area_suffix}_analysis.png'
        )
        plt.close(fig)
    
    return result


def save_population_results_to_csv(results: List[PopulationResult], save_path: Path):
    """Save population results to CSV file."""
    import csv
    
    fieldnames = [
        'mouse_id', 'session_id', 'n_neurons', 'areas', 'reward_group',
        'inflection_trial', 'n_total_trials', 'n_pre_trials', 'n_post_trials',
        'pre_response', 'post_response', 'shift_magnitude',
        'observed_statistic', 'null_mean', 'null_std', 'p_value', 'z_score'
    ]
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            writer.writerow({
                'mouse_id': r.mouse_id,
                'session_id': r.session_id,
                'n_neurons': r.n_neurons,
                'areas': ';'.join(r.areas),
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
                'z_score': r.z_score
            })
    
    print(f"Results saved to {save_path}")


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("Population Inflection Analysis")
    print("=" * 70)
    print("""
This script performs population-level inflection analysis.

Usage with your NWB files:
--------------------------

from population_inflection_analysis import (
    load_nwb_session_all_units,
    analyze_session_population,
    PopulationAnalysisConfig,
    save_population_results_to_csv
)

# Configure - bc_label_filter=None includes ALL units (good + mua)
config = PopulationAnalysisConfig(
    pre_trials=(-15, -6),
    post_trials=(0, 9),
    response_window=(0.005, 0.05),
    n_pseudosessions=10000,
    bc_label_filter=None,        # None = all units, 'good' = only good
    normalize_neurons=True,       # z-score each neuron before averaging
    aggregation_method='mean'     # 'mean', 'median', or 'sum'
)

# Load NWB (includes all units)
session_data = load_nwb_session_all_units(nwb_path, config)

# Add behavioral data
session_data['trials']['inflection_trial'] = your_inflection_trial
session_data['trials']['p_mean'] = your_p_mean  # optional

# Run population analysis
result = analyze_session_population(
    session_data, 
    config, 
    save_dir=Path('./output'),
    area_filter=None  # or 'wS1' to analyze only that area
)

# For multiple areas, loop:
for area in ['wS1', 'wS2', 'wM1']:
    result = analyze_session_population(
        session_data, config, save_dir, area_filter=area
    )
    results.append(result)

save_population_results_to_csv(results, Path('./output/population_results.csv'))
""")
