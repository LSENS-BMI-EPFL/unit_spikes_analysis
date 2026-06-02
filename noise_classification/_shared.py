"""_shared.py — minimal shared utilities."""

import numpy as np
import pandas as pd

LABEL_COLS    = ["mouse_id", "session_id", "probe_id", "cluster_id"]
_SKIP_COLS    = {*LABEL_COLS, "unit_id", "spike_times", "waveform_mean", "bc_label"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in _SKIP_COLS and pd.api.types.is_numeric_dtype(df[c])]


def plot_unit(ax_raster, ax_wave, spike_times, trial_starts,
              waveform=None, pre_s: float = 2.0, post_s: float = 5.0) -> None:
    """Raster + waveform. Single scatter call for performance."""
    spikes = np.sort(np.asarray(spike_times))
    all_t, all_tr = [], []
    for i, t0 in enumerate(trial_starts):
        mask = (spikes >= t0 - pre_s) & (spikes <= t0 + post_s)
        all_t.append(spikes[mask] - t0)
        all_tr.append(np.full(mask.sum(), i))

    if all_t:
        ax_raster.scatter(np.concatenate(all_t), np.concatenate(all_tr),
                          s=1.5, c="k", linewidths=0, rasterized=True)
    ax_raster.axvline(0, color="crimson", lw=0.8, ls="--")
    ax_raster.set_xlim(-pre_s, post_s)
    ax_raster.set_xlabel("Time from trial onset (s)", fontsize=8)
    ax_raster.set_ylabel("Trial #", fontsize=8)
    ax_raster.tick_params(labelsize=7)

    if ax_wave is not None:
        if waveform is not None and len(waveform):
            ax_wave.plot(np.asarray(waveform, dtype=float), color="#2166ac", lw=1.2)
            ax_wave.axhline(0, color="grey", lw=0.5, ls=":")
        else:
            ax_wave.text(0.5, 0.5, "no waveform", ha="center", va="center",
                         transform=ax_wave.transAxes, color="grey", fontsize=8)
        ax_wave.set_xlabel("Sample", fontsize=8)
        ax_wave.set_ylabel("µV", fontsize=8)
        ax_wave.tick_params(labelsize=7)
