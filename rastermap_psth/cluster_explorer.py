"""
cluster_explorer.py
--------------------
Standalone script to explore rastermap_psth clusters from a saved embedding.
Loads raw NWB data, matches neurons by unit_id (positional index), and
generates per-neuron and per-cluster figures.

Usage
-----
Set EMBEDDING_PATH and CLUSTERS_TO_EXPLORE at the top, then run.

NOTE on unit_id fragility
--------------------------
unit_ids in embedding_results.npz are positional indices created after
combine_ephys_nwb(). They are valid only if NWB files are loaded in the
same order. A future fix is to save mouse_id, session_id, and original
unit-table row into embedding_results.npz at pipeline time, e.g.:
    save_dict["mouse_ids"]   = unit_table["mouse_id"].values
    save_dict["session_ids"] = unit_table["session_id"].values
    save_dict["unit_table_idx"] = unit_table.index.values
"""

# ── user config ────────────────────────────────────────────────────────────────
EMBEDDING_PATH = r"M:\analysis\Axel_Bisi\combined_results\rastermap_psth_jaw_test\n_clusters_100\both\zscore\whisker_auditory\combined\embedding_results.npz"

CLUSTERS_TO_EXPLORE = [2,10,11,36,45,75,76,99]   # edit this list

# NWB roots
NWB_ROOT_AB = r"M:\analysis\Axel_Bisi\NWB_combined"
NWB_ROOT_MH = r"M:\analysis\Myriam_Hamon\NWB"

# Time windows (seconds)
T_PRE_STIM,  T_POST_STIM  = -0.2,  0.5
T_PRE_JAW,   T_POST_JAW   = -0.35, 0.35

# PSTH binning / smoothing
BIN_SIZE   = 0.01    # seconds
SIGMA_BINS = 2       # gaussian smooth (bins)

# R+/R- colours
COLOR_RPLUS  = "forestgreen"
COLOR_RMINUS = "crimson"
COLOR_NEUTRAL = "steelblue"

# Workers for NWB loading
N_WORKERS = 4
# ──────────────────────────────────────────────────────────────────────────────

#from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

import allen_utils
import neural_utils as nutils
from load_helpers import load_jaw_onset_data


# ── helpers ───────────────────────────────────────────────────────────────────

def discover_nwb_files() -> dict[str, list[Path]]:
    """Return {mouse_id: [Path, ...]} from both NWB roots (no duplicates)."""
    files: dict[str, list[Path]] = {}
    for root in [NWB_ROOT_AB, NWB_ROOT_MH]:
        for p in sorted(Path(root).glob("*.nwb")):
            mouse_id = p.stem.split("_")[0]   # e.g. AB013 or MH007
            files.setdefault(mouse_id, []).append(p)
    # deduplicate: if same mouse appears in both roots keep first seen (AB root first)
    return files


def load_all_data(nwb_paths: list[Path]):
    """Load trial_table and unit_table from a list of NWB files."""
    trial_table, unit_table, nwb_neural_files = nutils.combine_ephys_nwb(
        [str(p) for p in nwb_paths], max_workers=N_WORKERS
    )
    unit_table = allen_utils.process_allen_labels(unit_table, subdivide_areas=True)

    jaw_onset_table = load_jaw_onset_data(nwb_neural_files)
    trial_table = trial_table.merge(
        jaw_onset_table[["mouse_id", "session_id", "trial_id",
                         "jaw_dlc_onset", "piezo_lick_time"]],
        on=["mouse_id", "session_id", "trial_id"], how="left"
    )
    trial_table["jaw_onset_time"] = (
        trial_table["start_time"] + trial_table["jaw_dlc_onset"]
    )
    return trial_table, unit_table


def psth(spike_times: np.ndarray, t_events: np.ndarray,
         t_pre: float, t_post: float, bin_size: float = BIN_SIZE,
         sigma: float = SIGMA_BINS) -> tuple[np.ndarray, np.ndarray]:
    """Return (rate_hz [n_bins], t_centers [n_bins]) smoothed PSTH."""
    bins = np.arange(t_pre, t_post + bin_size, bin_size)
    counts = np.zeros(len(bins) - 1)
    n_valid = 0
    for t0 in t_events:
        if np.isnan(t0):
            continue
        rel = spike_times - t0
        c, _ = np.histogram(rel, bins=bins)
        counts += c
        n_valid += 1
    if n_valid == 0:
        t_ctrs = 0.5 * (bins[:-1] + bins[1:])
        return np.full(len(t_ctrs), np.nan), t_ctrs
    rate = counts / (n_valid * bin_size)
    rate = gaussian_filter1d(rate, sigma=sigma)
    t_ctrs = 0.5 * (bins[:-1] + bins[1:])
    return rate, t_ctrs


def raster_data(spike_times: np.ndarray, t_events: np.ndarray,
                t_pre: float, t_post: float) -> list[np.ndarray]:
    """Return list of spike-time arrays relative to each event."""
    out = []
    for t0 in t_events:
        if np.isnan(t0):
            out.append(np.array([]))
            continue
        rel = spike_times - t0
        out.append(rel[(rel >= t_pre) & (rel <= t_post)])
    return out


def get_reward_group(mouse_id: str, trial_table: pd.DataFrame) -> str:
    """Infer R+/R- from trial table for a given mouse."""
    rows = trial_table[trial_table["mouse_id"] == mouse_id]
    if rows.empty:
        return "unknown"
    # reward column heuristic: use 'reward_group' if present, else infer
    if "reward_group" in rows.columns:
        val = rows["reward_group"].iloc[0]
        return str(val)
    return "unknown"


def reward_color(reward_group: str) -> str:
    rg = str(reward_group).upper()
    if "R+" in rg or "RPLUS" in rg:
        return COLOR_RPLUS
    if "R-" in rg or "RMINUS" in rg:
        return COLOR_RMINUS
    return COLOR_NEUTRAL


# ── trial-type helpers ────────────────────────────────────────────────────────

TRIAL_TYPES = [
    ("whisker_hit",   dict(trial_type="whisker_trial",  lick_flag=1)),
    ("whisker_miss",  dict(trial_type="whisker_trial",  lick_flag=0)),
    ("auditory_hit",  dict(trial_type="auditory_trial", lick_flag=1)),
    ("auditory_miss", dict(trial_type="auditory_trial", lick_flag='False')),
    ("false_alarm",   dict(trial_type="no_stim_trial",  lick_flag=1)),
    ("correct_rejection",   dict(trial_type="no_stim_trial",  lick_flag=0)),
    ("passive_whisker", dict(context="passive",   trial_type="whisker_trial")),
]

TRIAL_COLORS = {
    "whisker_hit":    "#2563eb",
    "whisker_miss":   "#93c5fd",
    "auditory_hit":   "#d97706",
    "auditory_miss":  "#fcd34d",
    "false_alarm":    "#dc2626",
    "false_alarm":    "#7d7d7d",
    "passive_whisker":"#7c3aed",
}


def select_trials(trial_table: pd.DataFrame,
                  mouse_id: str, session_id,
                  trial_type: str = None, lick_flag: bool = None,
                  context: str = None) -> pd.DataFrame:
    mask = (trial_table["mouse_id"] == mouse_id)
    if session_id is not None:
        mask &= (trial_table["session_id"] == session_id)
    if trial_type is not None:
        mask &= (trial_table["trial_type"] == trial_type)
    if lick_flag is not None:
        mask &= (trial_table["lick_flag"] == lick_flag)
    if context is not None:
        mask &= (trial_table["context"] == context)
    return trial_table[mask].copy()


# ── per-neuron figure ─────────────────────────────────────────────────────────

def fig_neuron(unit_row: pd.Series, spike_times: np.ndarray,
               trial_table: pd.DataFrame, out_path: Path,
               unit_id: int, cluster_id: int):
    """Full per-neuron figure: rasters (3 sortings) + PSTHs (stim + jaw)."""

    mouse_id   = unit_row["mouse_id"]
    session_id = unit_row.get("session_id", "?")
    area       = unit_row.get("area_acronym_custom", "?")
    rg         = get_reward_group(mouse_id, trial_table)
    rc         = reward_color(rg)
    wf_type    = unit_row.get("waveform_type", unit_row.get("wf_type", "?"))

    # ── build trial sets ──────────────────────────────────────────────────────
    trial_sets: dict[str, pd.DataFrame] = {}
    for name, kwargs in TRIAL_TYPES:
        df = select_trials(trial_table, mouse_id, session_id, **kwargs)
        if len(df) > 0:
            trial_sets[name] = df

    # ── layout ────────────────────────────────────────────────────────────────
    # rows: 3 raster panels (sorting A/B/C) | 1 PSTH-stim | 1 PSTH-jaw
    n_trial_types = len(trial_sets)
    n_cols_raster = max(n_trial_types, 1)

    fig = plt.figure(figsize=(max(16, n_cols_raster * 2.5), 18))
    fig.suptitle(
        f"Cluster {cluster_id}  |  Unit {unit_id}  |  {mouse_id}  "
        f"|  {area}  |  {wf_type}  |  {rg}",
        fontsize=11, fontweight="bold", color=rc, y=0.99
    )

    outer = gridspec.GridSpec(5, 1, figure=fig,
                              hspace=0.45,
                              height_ratios=[1, 1, 1, 1.2, 1.2])

    raster_labels = [
        "Raster — sorted by trial index",
        "Raster — sorted by trial index × trial type",
        "Raster — sorted by jaw onset latency (hits only)",
    ]

    # ── raster panels (rows 0-2) ──────────────────────────────────────────────
    for row_i, sort_mode in enumerate(["trial_index", "trial_type", "jaw_latency"]):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_cols_raster, subplot_spec=outer[row_i], wspace=0.05)
        axes = [fig.add_subplot(inner[0, c]) for c in range(n_cols_raster)]

        for ax_i, (ttype, tdf) in enumerate(trial_sets.items()):
            ax = axes[ax_i]
            color = TRIAL_COLORS.get(ttype, "gray")

            if sort_mode == "trial_index":
                tdf_sorted = tdf.sort_values("trial_id")
                events = tdf_sorted["start_time"].values
            elif sort_mode == "trial_type":
                # sort by trial type alphabetically then trial_id
                tdf_sorted = tdf.sort_values(["trial_type", "trial_id"])
                events = tdf_sorted["start_time"].values
            else:  # jaw_latency — hits only
                if "hit" in ttype:
                    tdf_sorted = tdf.dropna(subset=["jaw_dlc_onset"])
                    tdf_sorted = tdf_sorted.sort_values("jaw_dlc_onset")
                    events = tdf_sorted["start_time"].values
                else:
                    tdf_sorted = tdf.sort_values("trial_id")
                    events = tdf_sorted["start_time"].values

            spk_lists = raster_data(spike_times, events, T_PRE_STIM, T_POST_STIM)
            for trial_i, spks in enumerate(spk_lists):
                if len(spks):
                    ax.scatter(spks, np.full(len(spks), trial_i),
                               s=1.5, c=color, linewidths=0, rasterized=True)

            ax.axvline(0, color="k", lw=0.8, ls="--")
            ax.set_xlim(T_PRE_STIM, T_POST_STIM)
            ax.set_ylim(-0.5, max(len(events) - 0.5, 0.5))
            ax.set_title(ttype.replace("_", " "), fontsize=7, color=color, pad=2)
            if ax_i == 0:
                ax.set_ylabel(f"trial #\n({sort_mode})", fontsize=7)
                ax.text(-0.18, 0.5, raster_labels[row_i],
                        transform=ax.transAxes, fontsize=8, rotation=90,
                        va="center", ha="center", fontweight="bold")
            else:
                ax.set_yticklabels([])
            if row_i == 2:
                ax.set_xlabel("time from stim (s)", fontsize=7)
            _style_ax(ax)

        # hide unused axes
        for ax_i in range(len(trial_sets), n_cols_raster):
            axes[ax_i].set_visible(False)

    # ── PSTH stim-aligned (row 3) ─────────────────────────────────────────────
    ax_ps = fig.add_subplot(outer[3])
    for ttype, tdf in trial_sets.items():
        events = tdf["start_time"].values
        rate, t = psth(spike_times, events, T_PRE_STIM, T_POST_STIM)
        ax_ps.plot(t, rate, color=TRIAL_COLORS.get(ttype, "gray"),
                   lw=1.2, label=ttype.replace("_", " "))
    ax_ps.axvline(0, color="k", lw=0.8, ls="--")
    ax_ps.set_xlim(T_PRE_STIM, T_POST_STIM)
    ax_ps.set_xlabel("time from stim onset (s)", fontsize=8)
    ax_ps.set_ylabel("firing rate (Hz)", fontsize=8)
    ax_ps.set_title("PSTH — stim aligned", fontsize=9)
    ax_ps.legend(fontsize=6, ncol=3, loc="upper right", frameon=False)
    _style_ax(ax_ps)

    # ── PSTH jaw-aligned (row 4) ──────────────────────────────────────────────
    ax_pj = fig.add_subplot(outer[4])
    for ttype, tdf in trial_sets.items():
        jaw_events = tdf["jaw_onset_time"].dropna().values
        if len(jaw_events) == 0:
            continue
        rate, t = psth(spike_times, jaw_events, T_PRE_JAW, T_POST_JAW)
        ax_pj.plot(t, rate, color=TRIAL_COLORS.get(ttype, "gray"),
                   lw=1.2, label=ttype.replace("_", " "))
    ax_pj.axvline(0, color="k", lw=0.8, ls="--")
    ax_pj.set_xlim(T_PRE_JAW, T_POST_JAW)
    ax_pj.set_xlabel("time from jaw onset (s)", fontsize=8)
    ax_pj.set_ylabel("firing rate (Hz)", fontsize=8)
    ax_pj.set_title("PSTH — jaw-onset aligned", fontsize=9)
    ax_pj.legend(fontsize=6, ncol=3, loc="upper right", frameon=False)
    _style_ax(ax_pj)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── per-cluster summary figure ────────────────────────────────────────────────

def fig_cluster_summary(cluster_id: int,
                        cluster_units: pd.DataFrame,
                        all_spike_times: dict[int, np.ndarray],
                        trial_table: pd.DataFrame,
                        X_cluster: np.ndarray,
                        unit_ids_cluster: np.ndarray,
                        out_path: Path):
    """
    Cluster-level figure:
      Row 0: mean ± SEM PSTH per trial type, R+ vs R- overlay (stim-aligned)
      Row 1: mean ± SEM PSTH per trial type, R+ vs R- overlay (jaw-aligned)
      Row 2: population raster heatmap (neurons × time, rastermap_psth order, stim-aligned)
      Row 3: brain region bar + pie, waveform type bar
    """
    n_neurons = len(cluster_units)
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(f"Cluster {cluster_id}  —  {n_neurons} neurons", fontsize=13,
                 fontweight="bold", y=0.995)

    outer = gridspec.GridSpec(4, 1, figure=fig, hspace=0.5,
                              height_ratios=[1.2, 1.2, 1.5, 1.0])

    # ── rows 0-1: mean ± SEM PSTH R+ vs R- ───────────────────────────────────
    for row_i, (t_pre, t_post, align_col, title_suffix) in enumerate([
        (T_PRE_STIM, T_POST_STIM, "start_time",     "stim aligned"),
        (T_PRE_JAW,  T_POST_JAW,  "jaw_onset_time",  "jaw aligned"),
    ]):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, len(TRIAL_TYPES), subplot_spec=outer[row_i], wspace=0.08)
        axes = [fig.add_subplot(inner[0, c]) for c in range(len(TRIAL_TYPES))]

        for ax_i, (ttype, kwargs) in enumerate(TRIAL_TYPES):
            ax = axes[ax_i]
            color_tt = TRIAL_COLORS.get(ttype, "gray")

            for rg_label, rg_color, rg_ls in [
                ("R+", COLOR_RPLUS,  "-"),
                ("R-", COLOR_RMINUS, "--"),
            ]:
                rates_all = []
                for _, unit_row in cluster_units.iterrows():
                    uid  = unit_row["_unit_id"]
                    mid  = unit_row["mouse_id"]
                    sid  = unit_row.get("session_id")
                    spks = all_spike_times.get(uid)
                    if spks is None:
                        continue
                    if get_reward_group(mid, trial_table).upper() not in rg_label:
                        continue
                    tdf = select_trials(trial_table, mid, sid, **kwargs)
                    events = tdf[align_col].dropna().values
                    if len(events) == 0:
                        continue
                    rate, t = psth(spks, events, t_pre, t_post)
                    rates_all.append(rate)

                if len(rates_all) == 0:
                    continue
                arr = np.vstack(rates_all)
                mean = np.nanmean(arr, axis=0)
                sem  = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
                ax.plot(t, mean, color=rg_color, lw=1.5, ls=rg_ls,
                        label=rg_label)
                ax.fill_between(t, mean - sem, mean + sem,
                                color=rg_color, alpha=0.15)

            ax.axvline(0, color="k", lw=0.7, ls="--")
            ax.set_title(ttype.replace("_", " "), fontsize=7, color=color_tt, pad=2)
            ax.set_xlim(t_pre, t_post)
            if ax_i == 0:
                ax.set_ylabel("rate (Hz)", fontsize=8)
                ax.text(-0.22, 0.5, f"mean±SEM PSTH\n{title_suffix}",
                        transform=ax.transAxes, fontsize=8, rotation=90,
                        va="center", ha="center", fontweight="bold")
            else:
                ax.set_yticklabels([])
            _style_ax(ax)

        # shared legend on last axis
        legend_elements = [
            Line2D([0], [0], color=COLOR_RPLUS,  lw=1.5, ls="-",  label="R+"),
            Line2D([0], [0], color=COLOR_RMINUS, lw=1.5, ls="--", label="R-"),
        ]
        axes[-1].legend(handles=legend_elements, fontsize=7, frameon=False)

    # ── row 2: population raster heatmap ─────────────────────────────────────
    ax_pop = fig.add_subplot(outer[2])
    bins = np.arange(T_PRE_STIM, T_POST_STIM + BIN_SIZE, BIN_SIZE)
    t_ctrs = 0.5 * (bins[:-1] + bins[1:])

    # build heatmap: one row per neuron, whisker_hit events, stim-aligned
    heatmap_rows = []
    for _, unit_row in cluster_units.iterrows():
        uid  = unit_row["_unit_id"]
        mid  = unit_row["mouse_id"]
        sid  = unit_row.get("session_id")
        spks = all_spike_times.get(uid)
        if spks is None:
            heatmap_rows.append(np.full(len(t_ctrs), np.nan))
            continue
        tdf = select_trials(trial_table, mid, sid,
                            trial_type="whisker_trial", lick_flag=True)
        events = tdf["start_time"].dropna().values
        rate, _ = psth(spks, events, T_PRE_STIM, T_POST_STIM)
        # z-score per neuron for display
        mu, sd = np.nanmean(rate), np.nanstd(rate)
        heatmap_rows.append((rate - mu) / sd if sd > 0 else rate - mu)

    if heatmap_rows:
        mat = np.vstack(heatmap_rows)
        vmax = np.nanpercentile(np.abs(mat), 95)
        im = ax_pop.imshow(mat, aspect="auto", origin="upper",
                           extent=[T_PRE_STIM, T_POST_STIM, n_neurons, 0],
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           interpolation="nearest")
        plt.colorbar(im, ax=ax_pop, fraction=0.02, pad=0.01,
                     label="z-score FR")
    ax_pop.axvline(0, color="k", lw=1.0, ls="--")
    ax_pop.set_xlabel("time from stim onset (s)", fontsize=8)
    ax_pop.set_ylabel("neuron (rastermap_psth order)", fontsize=8)
    ax_pop.set_title("Population heatmap — whisker hits, stim aligned", fontsize=9)
    _style_ax(ax_pop)

    # ── row 3: metadata panels ────────────────────────────────────────────────
    inner_meta = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[3], wspace=0.4)
    ax_area = fig.add_subplot(inner_meta[0, 0])
    ax_rg   = fig.add_subplot(inner_meta[0, 1])
    ax_wf   = fig.add_subplot(inner_meta[0, 2])

    # brain area bar
    area_counts = cluster_units["area_acronym_custom"].value_counts()
    ax_area.barh(range(len(area_counts)), area_counts.values,
                 color="slategray")
    ax_area.set_yticks(range(len(area_counts)))
    ax_area.set_yticklabels(area_counts.index, fontsize=7)
    ax_area.set_xlabel("n neurons", fontsize=7)
    ax_area.set_title("Brain area", fontsize=8)
    _style_ax(ax_area)

    # reward group bar
    rg_counts = cluster_units["reward_group_label"].value_counts()
    colors_rg = [reward_color(v) for v in rg_counts.index]
    ax_rg.bar(rg_counts.index, rg_counts.values, color=colors_rg)
    ax_rg.set_xlabel("reward group", fontsize=7)
    ax_rg.set_ylabel("n neurons", fontsize=7)
    ax_rg.set_title("Reward group", fontsize=8)
    _style_ax(ax_rg)

    # waveform type bar
    wf_col = "waveform_type" if "waveform_type" in cluster_units.columns else "wf_type"
    if wf_col in cluster_units.columns:
        wf_counts = cluster_units[wf_col].value_counts()
        ax_wf.bar(wf_counts.index, wf_counts.values, color="mediumpurple")
        ax_wf.set_xlabel("waveform type", fontsize=7)
        ax_wf.set_ylabel("n neurons", fontsize=7)
        ax_wf.set_title("Waveform type (RSU/FSU)", fontsize=8)
        _style_ax(ax_wf)
    else:
        ax_wf.set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    emb_path = Path(EMBEDDING_PATH)
    out_root = emb_path.parent / "cluster_explorer"
    out_root.mkdir(exist_ok=True)
    print(f"Output → {out_root}")

    # ── load embedding ────────────────────────────────────────────────────────
    print("Loading embedding results…")
    emb = np.load(emb_path, allow_pickle=True)
    unit_ids       = emb["unit_ids"]          # positional indices
    cluster_labels = emb["cluster_labels"]
    reward_arr     = emb.get("reward_arr", None)
    mouse_arr      = emb.get("mouse_arr",  None)
    X              = emb["X"]

    # filter to requested clusters
    clusters_needed = [c for c in CLUSTERS_TO_EXPLORE
                       if c in np.unique(cluster_labels)]
    if not clusters_needed:
        print("No matching clusters found in embedding. Check CLUSTERS_TO_EXPLORE.")
        return

    needed_unit_ids = unit_ids[np.isin(cluster_labels, clusters_needed)]
    print(f"Clusters {clusters_needed} → {len(needed_unit_ids)} neurons total")

    # ── discover & load NWB files ─────────────────────────────────────────────
    print("Discovering NWB files…")
    nwb_map = discover_nwb_files()
    all_nwb_paths = []
    seen = set()
    for mouse_id, paths in nwb_map.items():
        for p in paths:
            if p not in seen:
                all_nwb_paths.append(p)
                seen.add(p)

    print(f"Loading {len(all_nwb_paths)} NWB files…")
    trial_table, unit_table = load_all_data(all_nwb_paths)

    # ── reconstruct unit_id as positional index (mirrors pipeline) ────────────
    unit_table = unit_table.reset_index(drop=True)
    unit_table["_unit_id"] = unit_table.index   # must match pipeline

    # attach reward group label to unit_table for convenience
    rg_map = (trial_table.groupby("mouse_id")["reward_group"].first()
              if "reward_group" in trial_table.columns
              else pd.Series(dtype=str))
    unit_table["reward_group_label"] = unit_table["mouse_id"].map(rg_map).fillna("unknown")

    # ── pre-fetch spike times for all needed neurons ──────────────────────────
    print("Fetching spike times…")
    needed_set = set(needed_unit_ids.tolist())
    spike_times_dict: dict[int, np.ndarray] = {}
    for uid in needed_set:
        if uid >= len(unit_table):
            print(f"  WARNING: unit_id {uid} out of range — skipping")
            continue
        row = unit_table.iloc[uid]
        mouse_id   = row["mouse_id"]
        session_id = row.get("session_id")
        # get spike times from unit_table (assumed stored or accessible via nutils)
        if "spike_times" in unit_table.columns:
            spike_times_dict[uid] = np.asarray(row["spike_times"])
        else:
            # fallback: try to load from unit_table index via nutils helper if available
            try:
                spks = nutils.get_spike_times(mouse_id, session_id, uid)
                spike_times_dict[uid] = spks
            except Exception as e:
                print(f"  WARNING: could not get spike times for unit {uid}: {e}")

    # ── generate figures per cluster ──────────────────────────────────────────
    for cluster_id in clusters_needed:
        print(f"\n── Cluster {cluster_id} ──")
        cluster_dir = out_root / f"cluster_{cluster_id:03d}"
        cluster_dir.mkdir(exist_ok=True)

        mask = cluster_labels == cluster_id
        c_unit_ids = unit_ids[mask]
        c_X        = X[mask]

        # gather unit rows for this cluster
        valid_ids  = [uid for uid in c_unit_ids if uid < len(unit_table)]
        cluster_units = unit_table.iloc[valid_ids].copy()
        cluster_units["_unit_id"] = valid_ids

        print(f"  {len(cluster_units)} neurons")

        # ── per-neuron figures ────────────────────────────────────────────────
        for idx, (_, unit_row) in enumerate(cluster_units.iterrows()):
            uid = unit_row["_unit_id"]
            spks = spike_times_dict.get(uid)
            if spks is None:
                print(f"  [skip] unit {uid}: no spike times")
                continue
            out_file = cluster_dir / f"neuron_{idx:04d}_unit{uid}.pdf"
            print(f"  neuron {idx+1}/{len(cluster_units)} (unit {uid})…", end="\r")
            try:
                #fig_neuron(unit_row, spks, trial_table,
                #          out_file, uid, cluster_id)
                pass# print()
            except Exception as e:
                print(f"\n  ERROR neuron {uid}: {e}")

        print(f"  Per-neuron figures done.")

        # ── cluster summary figure ────────────────────────────────────────────
        print(f"  Building cluster summary…")
        summary_path = cluster_dir / f"cluster_{cluster_id:03d}_summary.pdf"
        try:
            fig_cluster_summary(
                cluster_id, cluster_units,
                spike_times_dict, trial_table,
                c_X, np.array(valid_ids), summary_path
            )
            print(f"  Summary saved → {summary_path.name}")
        except Exception as e:
            print(f"  ERROR cluster summary: {e}")

    print(f"\nDone. All outputs in {out_root}")


if __name__ == "__main__":
    main()