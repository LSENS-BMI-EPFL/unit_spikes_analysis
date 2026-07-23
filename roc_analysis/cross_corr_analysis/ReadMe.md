# Final Cross Correlation Analysis 

These files analyze the trial-by-trial relationship between neural activity (baseline and evoked response) and behavioral metrics (Hit Rate, False Alarm Rate, and (HR-FA)) across different brain areas.

## Workflow

To run the full pipeline, run the scripts in the following order:

1.  **`create_neuronal_df.py`**: Extract and clean data from raw NWB files.
2.  **`parallel_shuffled_cross_corr.py`**: Perform cross-correlation and shuffled validation.
3.  **Visualization Scripts**: Run any of the `plot_` scripts to generate figures.

---

## File Descriptions

### 1. Data Processing
*   **`create_neuronal_df.py`**
    *  Processes raw NWB files from the server.
    * Filters for (whisker, day 0)sessions and calculates pre-stimulus (baseline) and post-stimulus (for evoked) firing rates
    * Assigns custom brain areas using `allen_utils_old.py` helper file.
    * Saves a master dataframe as `.parquet`, `.csv`, and `.pkl`.

### 2. Statistical Analysis
*   **`parallel_shuffled_cross_corr.py`**
    *   Calculates time-lagged correlations between neural metrics and behavior with parallel processing.
    *   Generates a null distribution with 1,000 shuffles per neuron.
        *   Applies FDR (`method='by'`) correction to p-values.

### 3. Visualizations
*   **`plot_lag_0.py`**
    * Creates 3-panel plots showing Hit/FA rates, (HR-FA), and the corresponding neuronal firing rates for individual neurons at lag 0. 

*   **`plot_ccg_per_mouse.py`**
    * Creates cross-correlograms (CCG) for every neuron with shuffle confidence intervals.
    * Also creates 3-Panel summary plots:
       *    Distribution of peak lags, mean CCGs, and significant value distributions by brain area and mouse.
       * The first row includes all neurons, and the second row includes only neurons with significant peaks.

*   **`plot_ccg_across_mice.py`**
    *   Creates aggregated plots across all mice, comparing Reward Groups (R+ vs R-) and identifying trends in specific brain areas (e.g., MOs, SSs, CP).

*   **`plot_auto_correlations.py`**
    *   Creates autocorrelograms for both behavior and neural activity.

*   **`plot_fdr_correction.py`**
    *   Plots bar charts showing the impact of FDR correction on the number of significant neurons identified per area.

---

## Libraries used

- **Python Libraries**: `numpy`, `pandas`, `matplotlib`, `scipy`, `multiprocessing`, `pyarrow`.
- **Helpers**: `roc_utils`, `allen_utils_old`, `nwb_reader`.

---

## Output Directory Structure
Results are organized in the `Petersen-Lab/z_LSENS/Share/Dana_Shayakhmetova/dynamic_analysis_dec16` directory:
- `neuronal_data/`: Cleaned dataframes.
- `baseline_analysis/`: Cross-correlations using pre-stimulus rates.
- `evoked_response_analysis/`: Cross-correlations using evoked rates.
- `auto_correlations/`: Timescale analysis.
