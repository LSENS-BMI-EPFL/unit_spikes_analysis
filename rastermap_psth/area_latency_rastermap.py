"""
area_latency_rastermap.py — Area-level population PSTH matrix ordered by latency via Rastermap.

Each row = one brain area (area_acronym_custom).
Ordering = Rastermap applied to the (n_areas × n_bins_total) matrix of
           grand-averaged z-scored PSTHs, sorting areas from early to late firing.

Normalization
─────────────
    Per-neuron z-score (baseline window, per condition) — same logic as
    _neuron_vector_strided in the parent pipeline — then average z-scored
    vectors within each area.  This preserves shape comparability and prevents
    high-FR neurons from dominating the area mean.

Area inclusion filter
─────────────────────
    An area is included only if, in BOTH reward groups, it has:
      • ≥ 3 mice contributing neurons
      • ≥ 5 neurons per mouse (bc_label in {"good","mua"}) from that area
    The area set is the intersection across reward groups, so both figures
    share identical rows.

Cross-validation
────────────────
    Odd/even trial split at the event_map level (same as parent pipeline).
    Per-neuron z-score computed independently on each half.
    Rastermap fitted on odd-trial area averages → isort applied to even-trial
    area averages for the validation panel.

Output variants (per reward group)
───────────────────────────────────
    variant 1 : all conditions concatenated (one figure)
    variant 2 : one figure per (trial_type, context, align_col) combination
    variant 3 : whisker + auditory active, stimulus-aligned only
    variant 4 : whisker + auditory active, jaw-aligned only

Each figure: two-panel matrix (input order | rastermap order) + area profile plot.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import cmasher as cmr
from joblib import Parallel, delayed
from rastermap import Rastermap


#TOOD: different cmaps
#TODO: low and high performance states rastermap

# ── reuse helpers from parent pipeline ────────────────────────────────────────
from rastermap_psth.rastermap_clustering_psth import (
    DEFAULT_CFG,
    get_conditions,
    get_cond_infos,
    get_t_window,
    assign_passive_context,
    assign_active_context,
    precompute_event_map,
    split_event_map,
    apply_fr_filter,
    _neuron_vector_strided,
    _save,
    MOUSE_INFO,
    layer_number_mapper,
)

# ── config ─────────────────────────────────────────────────────────────────────
AREA_CFG: dict[str, Any] = {
    **DEFAULT_CFG,
    "min_mice_per_area"   : 3,    # minimum mice contributing to an area (per reward group)
    "min_neurons_per_mouse": 10,   # minimum neurons per mouse per area (good + mua)
    "bc_labels_area"      : {"good", "mua"},  # labels counted for area inclusion
    "n_jobs"              : 25,
}

# ── area matrix helpers ────────────────────────────────────────────────────────

def diverging_cmap(cmap_left, cmap_right, name='diverging', N=256):
    left = cmr.get_sub_cmap(cmap_left, 0, 1)(np.linspace(1, 0, N//2))
    right = cmr.get_sub_cmap(cmap_right, 0, 1)(np.linspace(0, 1, N//2))
    colors = np.vstack([left, right])
    return mc.LinearSegmentedColormap.from_list(name, colors)

custom_hotcold_cmap = diverging_cmap(cmr.get_sub_cmap('cmr.arctic', 0, 0.85),
                             cmr.get_sub_cmap('cmr.ember', 0, 0.85))

def _build_area_matrix(unit_ids, st_map, mouse_map, session_map, event_map,
                       area_arr, valid_areas, cfg, conds, cond_align_cols,
                       conds_full=None, cond_align_cols_full=None,
                       event_map_full=None):
    """Compute per-neuron z-scored PSTHs, average within each area, then
    re-z-score each area row using its own baseline window.

    Jaw-aligned normalization fix
    ─────────────────────────────
    _neuron_vector_strided borrows the baseline mean/std for jaw-aligned
    conditions from the start_time-aligned sibling of the same
    (trial_type, context).  The sibling is looked up by index in the conds
    list passed to that function.  When a variant contains ONLY jaw-aligned
    conditions, the sibling is absent from conds and the lookup returns None,
    so the jaw data is never z-scored.

    Fix: when conds_full / cond_align_cols_full are provided (the complete
    conds_all list), _neuron_vector_strided is called with the full condition
    set so sibling lookup always succeeds.  Only the columns corresponding to
    the requested subset (conds / cond_align_cols) are then sliced out of the
    resulting full feature vector before averaging.

    conds_full / cond_align_cols_full / event_map_full:
        Set to the complete condition list + event_map when any jaw-aligned
        condition is present in conds; leave as None otherwise (falls back
        to the variant-only path, which is cheaper).

    Pipeline per area row:
      1. Average neuron-level z-scored PSTHs -> area mean vector
      2. Subtract per-area baseline mean and divide by per-area baseline std,
         where the baseline is the pre-stimulus window pooled across all
         requested conditions (each condition's own base_mask, t_ctr < 0).
         This sets the baseline to exactly zero and puts all areas on a
         comparable amplitude scale -- critical for Rastermap to sort by
         latency rather than by response strength.

    Returns
    -------
    mat         : (n_areas, n_bins_total) -- baseline-z-scored area PSTHs
    t_ctrs      : list of per-condition time axes
    n_bins_list : list of per-condition bin counts
    """
    use_full = (conds_full is not None)

    if use_full:
        # Compute per-neuron vectors over the full condition set so that
        # jaw-aligned conditions can find their start_time siblings.
        cond_infos_full = get_cond_infos(cfg, conds_full, cond_align_cols_full)
        em_for_compute  = event_map_full

        # Build column slices: which bins in the full vector belong to each
        # requested condition?
        full_n_bins  = [info[3] for info in cond_infos_full]
        full_offsets = np.concatenate([[0], np.cumsum(full_n_bins)])
        full_key_to_slice = {
            (conds_full[ci][0], conds_full[ci][1], cond_align_cols_full[ci]):
                slice(int(full_offsets[ci]), int(full_offsets[ci + 1]))
            for ci in range(len(conds_full))
        }
        col_slices = [
            full_key_to_slice[(tt, ctx, acol)]
            for (tt, ctx), acol in zip(conds, cond_align_cols)
        ]
    else:
        cond_infos_full = get_cond_infos(cfg, conds, cond_align_cols)
        em_for_compute  = event_map
        col_slices      = None   # use full vector as-is

    # Per-neuron z-scored vectors over the effective condition set
    rows = Parallel(n_jobs=cfg["n_jobs"], prefer="threads")(
        delayed(_neuron_vector_strided)(
            st_map[uid], mouse_map[uid], session_map[uid],
            em_for_compute,
            cond_infos_full, cfg,
            conds_full if use_full else conds,
            cond_align_cols=cond_align_cols_full if use_full else cond_align_cols,
        )
        for uid in unit_ids
    )
    X_full = np.vstack(rows)   # (n_neurons, n_bins_full_or_variant)

    # Slice to the requested columns when using the full condition set
    if use_full and col_slices is not None:
        X = np.hstack([X_full[:, sl] for sl in col_slices])
    else:
        X = X_full

    # cond_infos for the requested (display) conditions -- t_ctrs, n_bins_list,
    # and baseline mask.
    cond_infos  = get_cond_infos(cfg, conds, cond_align_cols)
    t_ctrs      = [info[2] for info in cond_infos]
    n_bins_list = [info[3] for info in cond_infos]

    # Global baseline mask over the requested conditions only.
    # cond_infos[i][4] is the boolean base_mask (t_ctr < 0) for condition i.
    global_base_mask = np.concatenate([info[4] for info in cond_infos])

    # Average within each area, then re-z-score at the area level
    mat = np.full((len(valid_areas), X.shape[1]), np.nan)
    for i, area in enumerate(valid_areas):
        mask   = area_arr == area
        if mask.sum() == 0:
            continue
        sub    = X[mask]
        finite = np.isfinite(sub).all(axis=1)
        if finite.sum() == 0:
            continue
        row = sub[finite].mean(axis=0)   # area-mean z-scored PSTH

        # Per-area baseline z-score: baseline sits at exactly zero,
        # amplitude is in units of baseline std of the area mean.
        bl_vals = row[global_base_mask]
        bl_mean = bl_vals.mean()
        bl_std  = bl_vals.std()
        mat[i]  = (row - bl_mean) / (bl_std + 1e-9)

    return mat, t_ctrs, n_bins_list

def _fit_rastermap_area(mat, n_areas):
    """Fit Rastermap on the (n_areas, n_bins) area matrix.

    n_clusters is set to n_areas so each cluster = one area.
    With very few rows the PCA dimension is capped tightly.
    """
    n_pcs = min(n_areas - 1, mat.shape[1] - 1, 64)
    model = Rastermap(
        n_clusters   = n_areas,
        n_PCs        = n_pcs,
        locality     = 0.95,
        time_lag_window = 25,
        grid_upsample = 0,
        verbose      = False,
    ).fit(mat)
    return model.isort   # (n_areas,) permutation


# ── plotting helpers ───────────────────────────────────────────────────────────

def _draw_area_matrix(ax, mat, n_bins_list, conds, cond_align_cols,
                      cond_labels, cfg, title, area_names=None):
    """Imshow of an area matrix with condition separators and onset lines."""
    n_areas, n_total = mat.shape
    vmax = np.nanpercentile(np.abs(mat), 95)
    vmax = vmax if vmax > 0 else 1.0

    im = ax.imshow(mat, aspect="auto", interpolation="none",
                   cmap=custom_hotcold_cmap, vmin=-vmax, vmax=vmax,
                   extent=[0, n_total, n_areas, 0])

    dt      = cfg["stride_ms"] / 1000
    offsets = np.concatenate([[0], np.cumsum(n_bins_list)])
    for ci, ((tt, ctx), acol) in enumerate(zip(conds, cond_align_cols)):
        t_pre, _ = get_t_window(cfg, ctx, acol)
        onset_bin = int(round(t_pre / dt))
        if ci > 0:
            ax.axvline(offsets[ci], color="k", lw=1.5)
        ax.axvline(offsets[ci] + onset_bin, color="white", lw=0.8, ls="--")

    tick_pos = [(offsets[i] + offsets[i + 1]) / 2 for i in range(len(n_bins_list))]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(
        [l.replace(" ", "\n") for l in cond_labels], fontsize=7)

    if area_names is not None:
        ax.set_yticks(np.arange(n_areas) + 0.5)
        ax.set_yticklabels(area_names, fontsize=7)
    else:
        ax.set_ylabel("Area")

    ax.set_title(title, fontsize=9)
    return im


def _draw_area_profiles(axes_row, mat, isort, n_bins_list, conds, cond_align_cols,
                        cond_labels, cond_colors, cfg, valid_areas, title_prefix=""):
    """One subplot per area (rastermap order), mean PSTH with all conditions overlaid."""
    offsets = np.concatenate([[0], np.cumsum(n_bins_list)]).astype(int)

    for plot_i, area_i in enumerate(isort):
        ax    = axes_row[plot_i]
        vec   = mat[area_i]
        area  = valid_areas[area_i]

        for ci, (label, color) in enumerate(zip(cond_labels, cond_colors)):
            sl = slice(offsets[ci], offsets[ci + 1])
            # Build matching time axis from cond_infos
            t_pre, t_post = get_t_window(cfg, conds[ci][1], cond_align_cols[ci])
            dt = cfg["stride_ms"] / 1000
            n_out = n_bins_list[ci]
            t_ctr = np.linspace(-t_pre, t_post, n_out, endpoint=False)
            ax.plot(t_ctr, vec[sl], color=color, lw=1.2, label=label)

        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_title(f"{title_prefix}{area}", fontsize=7)
        ax.set_xlabel("Time (s)", fontsize=6)
        if plot_i == 0:
            ax.set_ylabel("z-score", fontsize=7)
        ax.legend(fontsize=4, loc="upper right", frameon=False)


def _make_figure(mat_input, mat_ordered, isort, n_bins_list, conds, cond_align_cols,
                 cond_labels, cond_colors, cfg, valid_areas, out_path, suptitle=""):
    """Two-panel matrix figure (input order | rastermap order) + profile subplots.

    Saves three formats: png, pdf, svg.
    """
    n_areas = len(valid_areas)

    # ── Panel A & B: matrices ─────────────────────────────────────────────
    fig_mat, axes_mat = plt.subplots(
        1, 3, figsize=(20, max(4, n_areas * 0.28)),
        gridspec_kw={"width_ratios": [10, 10, 0.4], "wspace": 0.05})

    im1 = _draw_area_matrix(axes_mat[0], mat_input, n_bins_list, conds, cond_align_cols,
                            cond_labels, cfg,
                            f"Input order  (n_areas={n_areas})",
                            area_names=valid_areas)
    im2 = _draw_area_matrix(axes_mat[1], mat_ordered, n_bins_list, conds, cond_align_cols,
                            cond_labels, cfg,
                            f"Rastermap order",
                            area_names=[valid_areas[i] for i in isort])
    for ax, im in zip(axes_mat[:2], [im1, im2]):
        fig_mat.colorbar(im, ax=ax, label="z-score", shrink=0.4, pad=0.01)
    axes_mat[2].axis("off")

    if suptitle:
        fig_mat.suptitle(suptitle, fontsize=10)
    fig_mat.tight_layout()
    _save(fig_mat, out_path.with_name(out_path.name + "_matrix"), dpi=300)

    # ── Panel C: area profiles (one subplot per area, rastermap order) ────
    ncols   = min(8, n_areas)
    nrows   = int(np.ceil(n_areas / ncols))
    fig_prof, axes_prof = plt.subplots(
        nrows, ncols,
        figsize=(3 * ncols, 2.5 * nrows),
        sharey=False, sharex=False)
    axes_prof_flat = np.atleast_1d(axes_prof).ravel()

    _draw_area_profiles(
        axes_prof_flat, mat_input, isort, n_bins_list,
        conds, cond_align_cols, cond_labels, cond_colors, cfg, valid_areas)

    for ax in axes_prof_flat[n_areas:]:
        ax.set_visible(False)

    if suptitle:
        fig_prof.suptitle(suptitle + "  — profiles (rastermap order)", fontsize=9)
    fig_prof.tight_layout()
    _save(fig_prof, out_path.with_name(out_path.name + "_profiles"), dpi=300)


def _run_variant(variant_label, conds, cond_align_cols, cond_labels, cond_colors,
                 unit_ids, st_map, mouse_map, session_map,
                 event_map_odd, event_map_even,
                 area_arr, valid_areas, cfg, out_dir, rg_label,
                 conds_full=None, cond_align_cols_full=None,
                 event_map_odd_full=None, event_map_even_full=None):
    """Full pipeline for one plot variant and one reward group.

    conds_full / cond_align_cols_full / event_map_odd_full / event_map_even_full:
        When the variant contains jaw-aligned conditions, pass the complete
        condition list and corresponding full event maps here so that
        _build_area_matrix can forward them to _neuron_vector_strided for
        correct sibling-based baseline estimation.  None = not needed.

    Steps:
      1. Build area matrix from odd trials -> fit Rastermap -> isort
      2. Build area matrix from even trials -> apply isort (validation)
      3. Save two figures: odd-trial ordered | even-trial same order
    """
    n_areas = len(valid_areas)
    if n_areas == 0:
        print(f"  [{rg_label}] {variant_label}: no valid areas, skipping")
        return

    print(f"  [{rg_label}] {variant_label}: building odd-trial area matrix...")
    mat_odd, t_ctrs, n_bins_list = _build_area_matrix(
        unit_ids, st_map, mouse_map, session_map,
        event_map_odd, area_arr, valid_areas, cfg, conds, cond_align_cols,
        conds_full=conds_full,
        cond_align_cols_full=cond_align_cols_full,
        event_map_full=event_map_odd_full)

    # Drop all-NaN area rows
    valid_rows = np.isfinite(mat_odd).any(axis=1)
    if not valid_rows.all():
        n_drop = (~valid_rows).sum()
        print(f"    dropping {n_drop} all-NaN area rows from odd matrix")
        mat_odd       = mat_odd[valid_rows]
        valid_areas_v = [a for a, ok in zip(valid_areas, valid_rows) if ok]
    else:
        valid_areas_v = list(valid_areas)

    n_areas_v = len(valid_areas_v)
    if n_areas_v < 2:
        print(f"  [{rg_label}] {variant_label}: <2 valid areas after NaN filter, skipping")
        return

    print(f"    fitting Rastermap on {n_areas_v} areas...")
    isort = _fit_rastermap_area(mat_odd, n_areas_v)

    print(f"  [{rg_label}] {variant_label}: building even-trial area matrix...")
    mat_even, _, _ = _build_area_matrix(
        unit_ids, st_map, mouse_map, session_map,
        event_map_even, area_arr, valid_areas_v, cfg, conds, cond_align_cols,
        conds_full=conds_full,
        cond_align_cols_full=cond_align_cols_full,
        event_map_full=event_map_even_full)

    # ── Figure 1: odd-trial embedding ─────────────────────────────────────
    slug     = variant_label.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
    out_base = out_dir / f"{rg_label}_{slug}_odd"
    _make_figure(
        mat_odd, mat_odd[isort], isort,
        n_bins_list, conds, cond_align_cols, cond_labels, cond_colors, cfg,
        valid_areas_v, out_base,
        suptitle=f"{rg_label} | {variant_label} — odd trials (embedding)")

    # ── Figure 2: even-trial validation (same isort) ──────────────────────
    # replace NaN even rows with odd rows so the figure is still informative
    bad_even = ~np.isfinite(mat_even).any(axis=1)
    if bad_even.any():
        mat_even[bad_even] = mat_odd[bad_even]

    out_base_even = out_dir / f"{rg_label}_{slug}_even"
    _make_figure(
        mat_even, mat_even[isort], isort,
        n_bins_list, conds, cond_align_cols, cond_labels, cond_colors, cfg,
        valid_areas_v, out_base_even,
        suptitle=f"{rg_label} | {variant_label} — even trials (cross-validation)")

    print(f"    saved → {out_dir.name}/{rg_label}_{slug}_*")

# ── area inclusion filter ──────────────────────────────────────────────────────

def compute_valid_areas(units_all, area_col, mouse_id_col, reward_group_col,
                        bc_labels, min_mice, min_neurons_per_mouse):
    """Return the intersection of areas passing the filter in every reward group.

    For each reward group:
      - keep only units with bc_label in bc_labels
      - for each area: count how many mice have >= min_neurons_per_mouse neurons there
      - area passes if that count >= min_mice

    Returns
    -------
    valid_areas : sorted list of area strings passing in ALL reward groups
    per_rg_info : dict{rg -> set of areas passing for that group} (for diagnostics)
    """
    rgs = [rg for rg in units_all[reward_group_col].unique()
           if rg in ("R+", "R-")]

    per_rg = {}
    for rg in rgs:
        rg_units = units_all[
            (units_all[reward_group_col] == rg) &
            (units_all["bc_label"].isin(bc_labels))
        ]
        passing = set()
        for area, grp in rg_units.groupby(area_col):
            mice_ok = (
                grp.groupby(mouse_id_col).size() >= min_neurons_per_mouse
            ).sum()
            if mice_ok >= min_mice:
                passing.add(area)
        per_rg[rg] = passing
        print(f"  {rg}: {len(passing)} areas pass filter")

    # Intersection across all reward groups
    if not per_rg:
        return [], per_rg
    shared = set.intersection(*per_rg.values())
    valid  = sorted(shared)
    print(f"  Shared intersection: {len(valid)} areas → {valid}")
    return valid, per_rg


# ── main entry point ───────────────────────────────────────────────────────────

def run_area_latency_rastermap(units: pd.DataFrame,
                                trials: pd.DataFrame,
                                out_root: str | Path = "rastermap_psth_out",
                                **cfg_overrides) -> None:
    """
    Parameters
    ----------
    units  : same DataFrame as parent pipeline (unique index = unit_id)
    trials : same concatenated trial table
    out_root : root output directory (new subfolder created inside)
    """
    cfg = {**AREA_CFG, **cfg_overrides}

    # ── resolve globals from parent (conditions, labels, colors) ──────────
    conds_all, cond_labels_all, cond_colors_all, _, cond_align_cols_all = \
        get_conditions(cfg)

    # ── output folder ─────────────────────────────────────────────────────
    out_folder = Path(out_root) / "area_latency_rastermap_hotcol"
    out_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out_folder}")

    # ── learner filter (same as parent pipeline) ──────────────────────────
    print("Filtering to learners (good/moderate)...")
    mouse_info  = pd.read_excel(MOUSE_INFO)
    valid_mice  = mouse_info[
        mouse_info["learning_category"].isin(["good", "moderate"])
    ]["mouse_id"].unique()
    units  = units[units["mouse_id"].isin(valid_mice)].copy()
    trials = trials[trials["mouse_id"].isin(valid_mice)].copy()

    # ── context assignment ────────────────────────────────────────────────
    print("Assigning trial contexts...")
    period = cfg.get("period", "both")
    if period in ("passive", "both"):
        trials = assign_passive_context(trials, cfg["mouse_id_col"], cfg["session_id_col"])
    if period in ("active", "both"):
        trials = assign_active_context(trials)

    # ── reward group map: mouse -> R+/R- ──────────────────────────────────
    rg_raw     = units[cfg["reward_group_col"]].map({1: "R+", 0: "R-"})
    units      = units.copy()
    units["_rg_str"] = rg_raw.values

    # ── bc_label filter for FR pass (good only, as in parent) ─────────────
    units_good = units[units["bc_label"].isin(["good","mua"])].copy()
    all_ids    = units_good.index.tolist()
    print(f"  {len(units)} total → {len(units_good)} bc=good units")

    # ── area inclusion filter (good + mua, intersection across R+/R-) ─────
    print("Computing valid areas (intersection across reward groups)...")
    # remap reward group for filtering
    units_for_filter = units.copy()
    units_for_filter[cfg["reward_group_col"]] = units_for_filter["_rg_str"]

    valid_areas, per_rg_info = compute_valid_areas(
        units_for_filter,
        area_col          = cfg["area_col"],
        mouse_id_col      = cfg["mouse_id_col"],
        reward_group_col  = "_rg_str",   # already mapped to "R+"/"R-" strings
        bc_labels         = cfg["bc_labels_area"],
        min_mice          = cfg["min_mice_per_area"],
        min_neurons_per_mouse = cfg["min_neurons_per_mouse"],
    )
    if len(valid_areas) < 2:
        print("ERROR: fewer than 2 valid areas after filter. Aborting.")
        return

    # ── spike time maps (good units only, for PSTH computation) ───────────
    print("Pre-extracting spike times and metadata...")
    st_map      = {uid: np.asarray(units_good.loc[uid, "spike_times"]) for uid in all_ids}
    mouse_map   = units_good[cfg["mouse_id_col"]].to_dict()
    session_map = units_good[cfg["session_id_col"]].to_dict()
    area_map    = units_good[cfg["area_col"]].to_dict()
    rg_map      = units_good["_rg_str"].to_dict()

    # ── event map (all conditions) ────────────────────────────────────────
    print("Pre-grouping trial event times...")
    event_map = precompute_event_map(trials, cfg, conds_all, cond_align_cols_all)
    print(f"  {len(event_map)} groups")

    # ── jaw filter (same as parent) ───────────────────────────────────────
    if "jaw_onset_time" in cond_align_cols_all:
        mice_with_jaw = {
            mid for (mid, *_rest, acol) in event_map if acol == "jaw_onset_time"
        }
        all_ids = [uid for uid in all_ids if mouse_map[uid] in mice_with_jaw]
        print(f"  {len(all_ids)} units after jaw filter")

    # ── FR filter ─────────────────────────────────────────────────────────
    print("Applying FR filter...")
    unit_ids, _ = apply_fr_filter(
        all_ids, st_map, mouse_map, session_map, event_map, cfg)

    # Final per-neuron metadata arrays
    area_arr = np.array([area_map.get(uid, "unknown") for uid in unit_ids])
    rg_arr   = np.array([rg_map.get(uid, "unknown")  for uid in unit_ids])

    # Keep only neurons in valid areas
    in_valid = np.isin(area_arr, valid_areas)
    unit_ids  = [uid for uid, ok in zip(unit_ids, in_valid) if ok]
    area_arr  = area_arr[in_valid]
    rg_arr    = rg_arr[in_valid]
    print(f"  {len(unit_ids)} units in valid areas")

    # ── odd / even trial split ────────────────────────────────────────────
    event_map_odd, event_map_even = split_event_map(event_map)

    # ── define the four plot variants ─────────────────────────────────────
    def _sub(conds, acols, labels, colors, keep_fn=None):
        """Filter condition lists by keep_fn(trial_type, context, align_col) -> bool."""
        if keep_fn is None:
            return conds, acols, labels, colors
        out = [(c, a, l, col) for c, a, l, col in zip(conds, acols, labels, colors)
               if keep_fn(c[0], c[1], a)]
        if not out:
            return [], [], [], []
        cs, as_, ls, cols = zip(*out)
        return list(cs), list(as_), list(ls), list(cols)

    variants = []

    # Variant 1: all conditions
    variants.append(dict(
        label       = "all_conditions",
        conds       = conds_all,
        cond_acols  = cond_align_cols_all,
        cond_labels = cond_labels_all,
        cond_colors = cond_colors_all,
    ))

    # Variant 2: one per (trial_type, context, align_col)
    seen = {}
    for (tt, ctx), acol, label, color in zip(
            conds_all, cond_align_cols_all, cond_labels_all, cond_colors_all):
        key = (tt, ctx, acol)
        if key not in seen:
            seen[key] = dict(
                label       = f"tt-{tt}_ctx-{ctx}_align-{acol}",
                conds       = [(tt, ctx)],
                cond_acols  = [acol],
                cond_labels = [label],
                cond_colors = [color],
            )
        variants.append(seen[key])
    # deduplicate (dict preserves insertion order in Python 3.7+)
    seen_labels = set()
    deduped = []
    for v in variants:
        if v["label"] not in seen_labels:
            seen_labels.add(v["label"])
            deduped.append(v)
    variants = deduped

    # Variant 3: whisker + auditory active, start_time-aligned
    cs3, as3, ls3, cols3 = _sub(
        conds_all, cond_align_cols_all, cond_labels_all, cond_colors_all,
        keep_fn=lambda tt, ctx, acol: (
            tt in ("whisker_trial", "auditory_trial") and
            ctx in ("active_lick", "active_nolick") and
            acol == "start_time"
        ))
    if cs3:
        variants.append(dict(
            label       = "active_stim_aligned",
            conds       = cs3, cond_acols=as3,
            cond_labels = ls3, cond_colors=cols3))

    # Variant 4: whisker + auditory active, jaw-aligned
    cs4, as4, ls4, cols4 = _sub(
        conds_all, cond_align_cols_all, cond_labels_all, cond_colors_all,
        keep_fn=lambda tt, ctx, acol: (
            tt in ("whisker_trial", "auditory_trial") and
            ctx in ("active_lick", "active_nolick") and
            acol == "jaw_onset_time"
        ))
    if cs4:
        variants.append(dict(
            label       = "active_jaw_aligned",
            conds       = cs4, cond_acols=as4,
            cond_labels = ls4, cond_colors=cols4))

    # ── loop over reward groups × variants ────────────────────────────────
    for rg_label in ("R+", "R-"):
        rg_mask   = rg_arr == rg_label
        uid_rg    = [uid for uid, ok in zip(unit_ids, rg_mask) if ok]
        area_rg   = area_arr[rg_mask]

        print(f"\n{'='*60}")
        print(f"Reward group: {rg_label}  ({len(uid_rg)} units)")

        rg_dir = out_folder / rg_label.replace("+", "plus").replace("-", "minus")
        rg_dir.mkdir(exist_ok=True)

        for v in variants:
            conds_v  = v["conds"]
            acols_v  = v["cond_acols"]
            labels_v = v["cond_labels"]
            colors_v = v["cond_colors"]

            if not conds_v:
                continue

            # Jaw-aligned normalization fix: _neuron_vector_strided looks up
            # each jaw condition's start_time sibling by index in conds.
            # When a variant contains jaw-aligned conditions, we pass the full
            # condition set (conds_all) so the sibling is always found.
            # _build_area_matrix then slices out only the requested columns
            # after per-neuron vectors have been computed with correct norms.
            has_jaw = "jaw_onset_time" in acols_v
            if has_jaw:
                conds_full_v          = conds_all
                acols_full_v          = cond_align_cols_all
                event_map_odd_full_v  = event_map_odd    # full maps have all keys
                event_map_even_full_v = event_map_even
                event_map_odd_v       = event_map_odd
                event_map_even_v      = event_map_even
            else:
                conds_full_v          = None
                acols_full_v          = None
                event_map_odd_full_v  = None
                event_map_even_full_v = None
                # Filter event maps to only the align_cols this variant needs
                needed_acols = set(acols_v)
                event_map_odd_v  = {k: v2 for k, v2 in event_map_odd.items()
                                    if k[4] in needed_acols}
                event_map_even_v = {k: v2 for k, v2 in event_map_even.items()
                                    if k[4] in needed_acols}

            _run_variant(
                variant_label        = v["label"],
                conds                = conds_v,
                cond_align_cols      = acols_v,
                cond_labels          = labels_v,
                cond_colors          = colors_v,
                unit_ids             = uid_rg,
                st_map               = st_map,
                mouse_map            = mouse_map,
                session_map          = session_map,
                event_map_odd        = event_map_odd_v,
                event_map_even       = event_map_even_v,
                area_arr             = area_rg,
                valid_areas          = valid_areas,
                cfg                  = cfg,
                out_dir              = rg_dir,
                rg_label             = rg_label,
                conds_full           = conds_full_v,
                cond_align_cols_full = acols_full_v,
                event_map_odd_full   = event_map_odd_full_v,
                event_map_even_full  = event_map_even_full_v,
            )
    print(f"\nDone. All outputs → {out_folder}")