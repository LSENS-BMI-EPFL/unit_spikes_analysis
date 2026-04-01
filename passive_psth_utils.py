"""
psth_passive.py — Raster/PSTH + population z-score matrices for passive trials.

Entry point
───────────
    run(units, trials, roc, out_root, **cfg_overrides)

All three DataFrames are pre-loaded externally (e.g. concatenated across NWB
files). The function groups by mouse_id and processes each mouse in parallel.

units  : one row per neuron  — must have (mouse_id, session_id, neuron_id,
         spike_times, [area_acronym_custom], [<roc_order_col>])
trials : one row per trial   — must have (mouse_id, session_id, start_time,
         context, trial_type)
roc    : ROC results          — must have (mouse_id, session_id, neuron_id,
         analysis_type, <roc_sig_col>, <roc_order_col>)

Outputs (under out_root/<mouse_id>/)
─────────────────────────────────────
  neurons/neuron_<uid>.png        per-neuron raster + PSTH, 4 conditions
  population_passive_pre.png      z-score matrix, whisker | auditory
  population_passive_post.png     z-score matrix, whisker | auditory

PSTH details
────────────
• Window   : [-t_pre, +t_post] = [-50 ms, +200 ms] around start_time
• Baseline : per-trial subtraction of mean rate in [-t_pre, 0]
• Whisker artefact: spikes in artifact_win_s=[-5 ms, +3 ms] are removed and
  replaced with Poisson-sampled spikes at that trial's mean rate outside the window.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

from notebooks.roc_analysis_new import values_minus

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed
import pynwb


# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_CFG: dict[str, Any] = dict(
    t_pre       = 0.05,          # s before trial onset  (50 ms baseline)
    t_post      = 0.10,          # s after  trial onset  (200 ms response)
    bin_ms      = 10,             # ms per histogram bin
    sigma_ms    = 5,             # Gaussian smoothing σ (ms)
    align_col   = "start_time",
    # whisker artefact replacement window (seconds, relative to start_time)
    artifact_win_s      = (-0.005, 0.005),   # [-5 ms, +3 ms]
    whisker_trial_label = "whisker_trial",
    fr_threshold_hz  = 1,
    max_neurons  = 4000, # max neurons matrix
    # ROC filter
    roc_analysis_type = None,
    roc_sig_col       = "p-val",
    roc_sig_thr       = 0.05,
    roc_order_col     = "selectivity",
    # contexts / trial types
    contexts       = ["passive_pre", "passive_post"],
    trial_types    = ["whisker_trial", "auditory_trial"],
    context_col    = "context",
    trial_type_col = "trial_type",
    # grouping keys shared between units and trials
    mouse_id_col    = "mouse_id",
    session_id_col  = "session_id",
    neuron_id_col   = "neuron_id",
    reward_group_col = "reward_group",   # e.g. "R+" / "R-"
    n_jobs = -1,
)

TRIAL_TYPE_COLORS = {
    "whisker_trial":  "#ffc43b",
    "auditory_trial": "#0045c4",
}
CONTEXT_LABELS = {
    "passive_pre":  "Passive pre-learning",
    "passive_post": "Passive post-learning",
}

def _rg_label(reward_group: int) -> str:
    return "R+" if reward_group == 1 else "R-"

def _rg_dir(reward_group: int) -> str:
    return "Rplus" if reward_group == 1 else "Rminus"


# ── passive_pre / passive_post splitting ──────────────────────────────────────

def assign_passive_context(trials: pd.DataFrame,
                           mouse_id_col: str = "mouse_id") -> pd.DataFrame:
    """
    For each mouse, split context=="passive" trials in half by index order:
      first half  → "passive_pre"
      second half → "passive_post"
    Non-passive trials are unchanged.
    Returns a copy.
    """
    trials = trials.copy()
    for mouse_id, grp in trials[trials["context"] == "passive"].groupby(mouse_id_col):
        idx_mid = len(grp) // 2
        trials.loc[grp.index[:idx_mid], "context"] = "passive_pre"
        trials.loc[grp.index[idx_mid:], "context"] = "passive_post"
    return trials

def compute_mean_fr(unit_ids: list, units: pd.DataFrame,
                    trials: pd.DataFrame, cfg: dict) -> dict[Any, float]:
    """
    Compute each neuron's mean firing rate (Hz) across all passive_pre and
    passive_post trials combined (both trial types pooled).
    Rate = total spikes across all matching trials / total window duration.
    """
    contexts   = cfg["contexts"]           # ["passive_pre", "passive_post"]
    trial_types = cfg["trial_types"]
    win_dur    = cfg["t_pre"] + cfg["t_post"]
    mouse_col  = cfg["mouse_id_col"]
    fr = {}
    for uid in unit_ids:
        st       = get_spike_times(units.loc[uid])
        mouse_id = units.loc[uid, mouse_col]
        mask = (
            (trials[mouse_col]                == mouse_id) &
            (trials[cfg["context_col"]].isin(contexts))   &
            (trials[cfg["trial_type_col"]].isin(trial_types))
        )
        event_times = trials.loc[trials.index[mask], cfg["align_col"]].to_numpy()
        if len(event_times) == 0:
            fr[uid] = 0.0
            continue
        total_spikes = sum(
            ((st >= t0 - cfg["t_pre"]) & (st <= t0 + cfg["t_post"])).sum()
            for t0 in event_times
        )
        fr[uid] = total_spikes / (len(event_times) * win_dur)
    return fr

# ── ROC filter ────────────────────────────────────────────────────────────────

def filter_roc(units: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Keep units where analysis_type matches and significant == True.
    ROC columns are already part of the unit table.
    """
    mask = (units["analysis_type"] == cfg["roc_analysis_type"]) & (units["significant"] == True)
    units_sig = units[mask].copy()
    print(f"  ROC filter ({cfg['roc_analysis_type']}, significant): "
          f"{len(units)} → {len(units_sig)} units")
    return units_sig


# ── spike helpers ─────────────────────────────────────────────────────────────

def get_spike_times(unit_row: pd.Series) -> np.ndarray:
    return np.asarray(unit_row["spike_times"])


def _replace_artifact(spikes_rel: np.ndarray,
                      t_pre: float, t_post: float,
                      art_lo: float, art_hi: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Drop spikes in [art_lo, art_hi]; replace with Poisson noise at the
    trial's mean rate estimated from spikes outside that window."""
    art_dur   = art_hi - art_lo
    #clean_dur = (t_pre + t_post) - art_dur
    #outside   = spikes_rel[(spikes_rel < art_lo) | (spikes_rel > art_hi)]
    #rate_hz   = len(outside) / clean_dur if clean_dur > 0 else 0.0
    pre_spikes = spikes_rel[spikes_rel < art_lo]  # [-50 ms, -5 ms)
    pre_dur = t_pre + art_lo  # 0.05 - 0.005 = 0.045 s
    rate_hz = len(pre_spikes) / pre_dur if pre_dur > 0 else 0.0
    outside = spikes_rel[(spikes_rel < art_lo) | (spikes_rel > art_hi)]  # kept for replacement placement
    n_replace = rng.poisson(rate_hz * art_dur)
    if n_replace > 0:
        replacement = rng.uniform(art_lo, art_hi, size=n_replace)
        return np.sort(np.concatenate([outside, replacement]))
    return np.sort(outside)


def spikes_around_events_old(spike_times: np.ndarray,
                         event_times: np.ndarray,
                         t_pre: float, t_post: float,
                         is_whisker: bool = False,
                         artifact_win_s: tuple[float, float] = (-0.005, 0.003),
                         rng: np.random.Generator | None = None,
                         ) -> list[np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    art_lo, art_hi = artifact_win_s
    raster = []
    for t0 in event_times:
        rel = spike_times[(spike_times >= t0 - t_pre) & (spike_times <= t0 + t_post)] - t0
        if is_whisker:
            rel = _replace_artifact(rel, t_pre, t_post, art_lo, art_hi, rng)
        raster.append(rel)
    return raster

def spikes_around_events(spike_times, event_times, t_pre, t_post,
                               is_whisker=False, artifact_win_s=(-0.005,0.003), rng=None):
    if rng is None:
        rng = np.random.default_rng()
    lo = event_times - t_pre
    hi = event_times + t_post
    i_lo = np.searchsorted(spike_times, lo)
    i_hi = np.searchsorted(spike_times, hi, side='right')
    raster = [spike_times[i_lo[i]:i_hi[i]] - event_times[i] for i in range(len(event_times))]
    if is_whisker:
        art_lo, art_hi = artifact_win_s
        raster = [_replace_artifact(r, t_pre, t_post, art_lo, art_hi, rng) for r in raster]
    return raster


# ── PSTH ──────────────────────────────────────────────────────────────────────

def _bin_rates(raster: list[np.ndarray], bins: np.ndarray, dt: float) -> np.ndarray:
    """(n_trials × n_bins) firing rate matrix in Hz."""
    return np.vstack([np.histogram(r, bins=bins)[0] for r in raster]).astype(float) / dt


def _causal_gaussian(x: np.ndarray, sigma_bins: float, truncate: float = 4.0) -> np.ndarray:
    """One-sided (causal) Gaussian smoothing."""
    radius = int(truncate * sigma_bins)
    t      = np.arange(0, radius + 1)
    kernel = np.exp(-0.5 * (t / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode="full")[: len(x)]


def compute_psth(raster: list[np.ndarray],
                 t_pre: float, t_post: float,
                 bin_ms: float, sigma_ms: float,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin → per-trial baseline subtract → causal-smooth → average.
    Returns (t_ctr, mean ΔHz, sem ΔHz)."""
    dt        = bin_ms / 1000
    bins      = np.arange(-t_pre, t_post + dt, dt)
    t_ctr     = bins[:-1] + dt / 2
    base_mask = t_ctr < 0

    rates    = _bin_rates(raster, bins, dt)
    rates_bc = rates - rates[:, base_mask].mean(axis=1, keepdims=True)

    mean_sm = _causal_gaussian(rates_bc.mean(axis=0), sigma_bins=sigma_ms / bin_ms)
    sem     = rates_bc.std(axis=0) / np.sqrt(max(len(raster), 1))
    return t_ctr, mean_sm, sem


def zscore_for_matrix(raster: list[np.ndarray],
                      t_pre: float, t_post: float,
                      bin_ms: float, sigma_ms: float) -> np.ndarray:
    """Bin -> per-trial baseline subtract -> smooth -> z-score (independent per neuron per context)."""
    dt        = bin_ms / 1000
    bins      = np.arange(-t_pre, t_post + dt, dt)
    t_ctr     = bins[:-1] + dt / 2
    base_mask = t_ctr < 0

    rates    = _bin_rates(raster, bins, dt)
    rates_bc = rates - rates[:, base_mask].mean(axis=1, keepdims=True)

    mean_sm = _causal_gaussian(rates_bc.mean(axis=0), sigma_bins=sigma_ms / bin_ms)
    base_sd = rates_bc[:, base_mask].std() + 1e-9
    return mean_sm / base_sd


# ── per-neuron raster + PSTH ──────────────────────────────────────────────────

def _plot_neuron(uid: Any, unit_row: pd.Series, trials_mouse: pd.DataFrame,
                 cfg: dict, out_dir: Path) -> None:
    spike_times = get_spike_times(unit_row)
    contexts    = cfg["contexts"]
    trial_types = cfg["trial_types"]
    n_cols      = len(contexts) * len(trial_types)
    rng         = np.random.default_rng()

    fig, axes = plt.subplots(
        2, n_cols, figsize=(3.5 * n_cols, 5),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True,
    )

    col = 0
    for ttype in trial_types:
        for ctx in contexts:
            mask = (
                (trials_mouse[cfg["context_col"]]    == ctx) &
                (trials_mouse[cfg["trial_type_col"]] == ttype)
            )
            tidx       = trials_mouse.index[mask]
            ax_r, ax_p = axes[0, col], axes[1, col]

            if len(tidx) == 0:
                ax_r.set_visible(False); ax_p.set_visible(False)
                col += 1; continue

            event_times = trials_mouse.loc[tidx, cfg["align_col"]].to_numpy()
            is_whisker  = (ttype == cfg["whisker_trial_label"])
            raster = spikes_around_events(
                spike_times, event_times, cfg["t_pre"], cfg["t_post"],
                is_whisker=is_whisker, artifact_win_s=cfg["artifact_win_s"], rng=rng,
            )
            psth, mean, sem = compute_psth(
                raster, cfg["t_pre"], cfg["t_post"], cfg["bin_ms"], cfg["sigma_ms"])
            color = TRIAL_TYPE_COLORS.get(ttype, "gray")

            # raster
            for ti, spks in enumerate(raster):
                ax_r.vlines(spks, ti + 0.5, ti + 1.5, lw=1.2, color=color)
            ax_r.axvline(0, color="k", lw=2, ls="--")
            ax_r.set_title(f"{CONTEXT_LABELS.get(ctx, ctx)}\n{ttype}", fontsize=8)
            ax_r.set_ylim(0.5, max(len(raster), 1) + 0.5)
            if col == 0:
                ax_r.set_ylabel("Trial", fontsize=10)

            # PSTH
            ax_p.plot(psth, mean, color=color, lw=1.5)
            #ax_p.fill_between(psth, mean - sem, mean + sem, color=color, alpha=0.25)
            ax_p.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
            ax_p.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
            ax_p.set_xlabel("Time (s)", fontsize=12)
            if col == 0:
                ax_p.set_ylabel("FR (spks/sec)", fontsize=12)

            col += 1

    # vertical separator between the two contexts
    for row in range(2):
        axes[row, len(trial_types) - 1].spines["right"].set(linewidth=2, color="#888")

    # Get global y-limits across the row
    ymins, ymaxs = [], []

    for ax in axes[row]:
        ymin, ymax = ax.get_ylim()
        ymins.append(ymin)
        ymaxs.append(ymax)

    ymin, ymax = min(ymins), max(ymaxs)

    # Apply same limits to all axes in the row
    for ax in axes[row]:
        ax.set_ylim(ymin, ymax)

    area         = unit_row.get("area_acronym_custom", "")
    rg_txt       = _rg_label(unit_row.get(cfg["reward_group_col"], 0))
    order_val    = unit_row.get(cfg["roc_order_col"], float("nan"))
    mouse_id     = trials_mouse[cfg["mouse_id_col"]].iloc[0]
    title_str    = (f"Unit {uid}  |  {area}  |  {mouse_id}, {rg_txt}  |  "
                    f"{cfg['roc_order_col']}={order_val:.3f} ({cfg['roc_analysis_type']})"
                    if isinstance(order_val, float) and np.isfinite(order_val)
                    else f"Unit {uid}  |  {area}  |  {mouse_id}, {rg_txt}")
    fig.suptitle(title_str, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"neuron_{uid}.png", dpi=150)
    plt.close(fig)
    return


# ── per-mouse processing (neurons only) ───────────────────────────────────────

def _process_mouse(mouse_id: str,
                   reward_group: int,
                   units_mouse: pd.DataFrame,
                   trials_mouse: pd.DataFrame,   # already has passive_pre/post
                   cfg: dict,
                   out_root: Path) -> None:
    out_dir  = out_root / _rg_dir(reward_group) / mouse_id
    neur_dir = out_dir / "neurons"
    neur_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n── {_rg_label(reward_group)} / {mouse_id}  ({len(units_mouse)} units)")
    for ctx in cfg["contexts"]:
        for tt in cfg["trial_types"]:
            n = ((trials_mouse[cfg["context_col"]] == ctx) &
                 (trials_mouse[cfg["trial_type_col"]] == tt)).sum()
            print(f"    {ctx} × {tt}: {n} trials")

    #Parallel(n_jobs=cfg["n_jobs"], prefer="threads")(
    #    delayed(_plot_neuron)(uid, units_mouse.loc[uid], trials_mouse, cfg, neur_dir)
    #    for uid in units_mouse.index.tolist()
    #)
    Parallel(n_jobs=cfg["n_jobs"], prefer="threads")(
        delayed(_plot_neuron)(
            uid,
            units_mouse.loc[units_mouse["neuron_id"] == uid].iloc[0],
            trials_mouse,
            cfg,
            neur_dir
        )
        for uid in units_mouse["neuron_id"].unique()
    )
    print(f"  neuron figures → {neur_dir}")


# ── group population matrix ───────────────────────────────────────────────────

def _zscore_one_neuron(uid: Any, units: pd.DataFrame,
                       event_times: np.ndarray,
                       t_pre: float, t_post: float,
                       bin_ms: float, sigma_ms: float,
                       n_bins: int,
                       is_whisker: bool,
                       artifact_win_s: tuple,
                       ) -> np.ndarray:
    """Compute z-scored PSTH trace for one neuron. Called in parallel."""
    if len(event_times) == 0:
        return np.zeros(n_bins)
    st     = get_spike_times(units.loc[uid])
    rng    = np.random.default_rng()          # independent RNG per thread
    raster = spikes_around_events(
        st, event_times, t_pre, t_post,
        is_whisker=is_whisker, artifact_win_s=artifact_win_s, rng=rng,
    )
    return zscore_for_matrix(raster, t_pre, t_post, bin_ms, sigma_ms)


def _build_context_matrix(unit_ids: list, units: pd.DataFrame,
                          trials: pd.DataFrame, context: str, trial_type: str,
                          cfg: dict, rng: np.random.Generator,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Z-score matrix for one (context, trial_type) across all mice in the group.
    Each neuron is z-scored independently using its own baseline trials.

    Speedups vs naive loop:
    - trial mask computed once outside the neuron loop
    - neurons processed in parallel with joblib threads

    Returns
    -------
    z_matrix : (n_units x n_bins), rows in unit_ids order
    t_ctr    : bin centre times (s)
    """
    dt    = cfg["bin_ms"] / 1000
    bins  = np.arange(-cfg["t_pre"], cfg["t_post"] + dt, dt)
    t_ctr = bins[:-1] + dt / 2
    n_bins     = len(t_ctr)
    is_whisker = (trial_type == cfg["whisker_trial_label"])

    # pre-filter trials once — same set for every neuron
    mask        = (
        (trials[cfg["context_col"]]    == context) &
        (trials[cfg["trial_type_col"]] == trial_type)
    )
    event_times = trials.loc[mask, cfg["align_col"]].to_numpy()

    rows = Parallel(n_jobs=cfg["n_jobs"], prefer="threads")(
        delayed(_zscore_one_neuron)(
            uid, units, event_times,
            cfg["t_pre"], cfg["t_post"], cfg["bin_ms"], cfg["sigma_ms"],
            n_bins, is_whisker, cfg["artifact_win_s"],
        )
        for uid in unit_ids
    )
    return np.vstack(rows), t_ctr

def _plot_group_matrix(z_pre: np.ndarray, z_post: np.ndarray,
                       t_ctr: np.ndarray, sort_idx: np.ndarray,
                       trial_type: str, reward_group: int,
                       cfg: dict, out_dir: Path) -> None:
    """
    passive_pre | passive_post side by side for one trial_type × reward_group.
    Rows sorted by passive_pre selectivity (sort_idx computed externally).
    """
    n_units = len(sort_idx)
    fig_h   = max(3.5, n_units * 0.12 + 1.5)
    fig, (ax_pre, ax_post) = plt.subplots(1, 2, figsize=(6, 10), dpi=400, sharey=True)

    vmax   = np.nanpercentile(np.abs(np.concatenate([z_pre, z_post])), 98)
    extent = [t_ctr[0], t_ctr[-1], n_units, 0]

    for ax, z, ctx in ((ax_pre, z_pre, "passive_pre"), (ax_post, z_post, "passive_post")):
        im = ax.imshow(
            z[sort_idx], aspect="auto", interpolation="none",
            extent=extent, cmap="seismic", vmin=-vmax, vmax=vmax,
        )
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Time (s)")
        ax.set_title(CONTEXT_LABELS.get(ctx, ctx))
        vmax = 0.5
        plt.colorbar(im, ax=ax, label="z-score", shrink=0.2, pad=0.04, aspect=20, vmin=-vmax, vmax=vmax)

        # Make all fonts
        ax.tick_params(labelsize=10)


    #ax_pre.set_ylabel(
    #    f"Neuron ↓ {cfg['roc_order_col']} (passive_pre)  "
    #    f"(n={n_units}, {cfg['roc_analysis_type']})"
    #)
    #fig.suptitle(f"{trial_type}  —  {_rg_label(reward_group)}",
    #             fontsize=11)
    fig.tight_layout()
    fname = f"population_{trial_type}_{_rg_label(reward_group)}.png"
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def build_and_plot_group_matrices(reward_group: int,
                                  units_rg: pd.DataFrame,
                                  trials_rg: pd.DataFrame,
                                  cfg: dict,
                                  out_dir: Path) -> None:
    """
    One figure per trial_type across all mice in the reward group.
    passive_pre | passive_post side by side, each z-scored independently.
    Row order: descending passive_pre selectivity, applied to passive_post.
    """
    assert units_rg.index.is_unique, "units_rg index has duplicates — reset_index() before calling"

    out_dir.mkdir(parents=True, exist_ok=True)
    rng      = np.random.default_rng(42)
    unit_ids = units_rg.index.tolist()
    ocol     = cfg["roc_order_col"]

    # firing rate filter: keep neurons with mean FR >= fr_threshold_hz
    # computed across passive_pre + passive_post trials combined
    thr = cfg["fr_threshold_hz"]
    if thr > 0:
        fr_map = compute_mean_fr(unit_ids, units_rg, trials_rg, cfg)
        unit_ids = [uid for uid in unit_ids if fr_map[uid] >= thr]
        print(f"  FR filter (>={thr} Hz): {len(fr_map)} → {len(unit_ids)} units")
    if not unit_ids:
        print("  [skip matrix] no units above FR threshold");
        return

    # stratified sampling to plot some  neurons
    max_n = cfg["max_neurons"]
    if max_n is not None and len(unit_ids) > max_n:
        ocol = cfg["roc_order_col"]
        sel_vals = np.array([
            float(units_rg.loc[uid, ocol]) if ocol in units_rg.columns else 0.0
            for uid in unit_ids
        ])
        #sorted_idx = np.argsort(np.abs(sel_vals))
        #bins = np.array_split(sorted_idx, max_n)
        #rng_sel = np.random.default_rng(0)
        #unit_ids = [unit_ids[rng_sel.choice(b)] for b in bins if len(b) > 0]
        top_idx = np.argsort(np.abs(sel_vals))[::-1][:max_n]
        unit_ids = [unit_ids[i] for i in top_idx]
        print(f"  selectivity filter: kept top {len(unit_ids)} by |{ocol}|")

    #order_vals = [float(units_rg.loc[uid, ocol]) if ocol in units_rg.columns else 0.0
    #              for uid in unit_ids]
    #sort_idx = np.argsort(order_vals)[::-1]

    order_vals = [float(units_rg.loc[uid, ocol]) if ocol in units_rg.columns else 0.0
                  for uid in unit_ids]
    sort_idx = np.argsort(order_vals)[::-1]

    # build all (trial_type × context) matrices in parallel, then plot
    combos = [(tt, ctx)
              for tt  in cfg["trial_types"]
              for ctx in ("passive_pre", "passive_post")]

    matrices = Parallel(n_jobs=20, prefer="processes")(
        delayed(_build_context_matrix)(
            unit_ids, units_rg, trials_rg, ctx, tt, cfg, np.random.default_rng(42))
        for tt, ctx in combos
    )
    # reorganise results: {trial_type: {context: (z, t_ctr)}}
    results = {}
    for (tt, ctx), (z, t_ctr) in zip(combos, matrices):
        results.setdefault(tt, {})[ctx] = (z, t_ctr)

    for trial_type in cfg["trial_types"]:
        z_pre,  t_ctr = results[trial_type]["passive_pre"]
        z_post, _     = results[trial_type]["passive_post"]
        _plot_group_matrix(z_pre, z_post, t_ctr, sort_idx,
                           trial_type, reward_group, cfg, out_dir)

def run_passive_psths(units: pd.DataFrame,
                      trials: pd.DataFrame,
                      out_root: str | Path = "psth_passive_output",
                      **cfg_overrides) -> None:
    """
    Parameters
    ----------
    units  : unit table with ROC columns already merged in.
             Required: mouse_id, session_id, neuron_id, spike_times,
                       reward_group, area_acronym_custom,
                       analysis_type, significant, selectivity.
    trials : trial table. context=="passive" rows are split into
             passive_pre / passive_post here before any processing.
             Required: mouse_id, session_id, start_time,
                       context, trial_type, reward_group.
    out_root : root output directory.
    **cfg_overrides : override any DEFAULT_CFG key.
               roc_analysis_type may be a str or a list of str.
               Neuron figures use the first entry; matrices are produced
               once per entry without re-running neuron figures.
    """
    cfg      = {**DEFAULT_CFG, **cfg_overrides}
    out_root = Path(out_root) / "psth_passive"
    out_root.mkdir(parents=True, exist_ok=True)

    # Keep good units with min firing rate
    units['firing_rate'] = units['firing_rate'].astype(float)
    #units = units[(units.bc_label=='good') & (units.firing_rate > 2)]

    # roc_analysis_type may be a single str or a list or None
    if cfg["roc_analysis_type"] == None:
        analysis_types = units['analysis_type'].unique()
    else:
        analysis_types = cfg["roc_analysis_type"]

    if isinstance(analysis_types, str):
        analysis_types = [analysis_types]

    # split passive → passive_pre / passive_post once, for all mice
    trials    = assign_passive_context(trials, cfg["mouse_id_col"])
    mouse_col = cfg["mouse_id_col"]
    rg_col    = cfg["reward_group_col"]

    # ── per-neuron figures — run once using the first analysis_type ───────────
    cfg_neurons = {**cfg, "roc_analysis_type": analysis_types[0]}
    units_sig   = filter_roc(units, cfg_neurons)
    if units_sig.empty:
        print("[abort] no significant units for neuron figures"); return
    plot_single_neurons=False
    if plot_single_neurons:

        groups = (units_sig.groupby([rg_col, mouse_col], sort=True)
                           .size()
                           .reset_index(name="_n")[[rg_col, mouse_col]]
                           .values.tolist())
        print(f"\nNeuron figures: {len(groups)} (reward_group, mouse) pairs "
              f"[{analysis_types[0]}]")

        Parallel(n_jobs=min(len(groups), 8), prefer="processes")(
            delayed(_process_mouse)(
                mouse_id,
                reward_group,
                units_sig[(units_sig[rg_col] == reward_group) &
                          (units_sig[mouse_col] == mouse_id)].copy(),
                trials[(trials[rg_col] == reward_group) &
                       (trials[mouse_col] == mouse_id)].copy(),
                cfg_neurons,
                out_root,
            )
            for reward_group, mouse_id in groups
        )

    # ── population matrices — once per analysis_type ──────────────────────────
    for analysis_type in analysis_types:
        print(f"\nBuilding matrices: {analysis_type}")
        cfg_mat   = {**cfg, "roc_analysis_type": analysis_type}
        units_mat = filter_roc(units, cfg_mat)
        if units_mat.empty:
            print(f"  [skip] no significant units for {analysis_type}"); continue

        for rg in sorted(units_mat[rg_col].unique()):
            build_and_plot_group_matrices(
                reward_group = rg,
                units_rg     = units_mat[units_mat[rg_col] == rg],
                trials_rg    = trials[trials[rg_col] == rg],
                cfg          = cfg_mat,
                out_dir      = out_root / analysis_type / _rg_dir(rg),
            )
