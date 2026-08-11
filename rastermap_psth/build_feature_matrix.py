"""
build_feature_matrix.py  —  Step 1 of the rastermap pipeline.
Load spike data, filter units/trials, build z-scored PSTH feature matrices
(all trials + odd/even CV split), extract per-neuron metadata, save everything
needed by run_clustering.py.  No clustering is performed here.

Outputs  (written to <out_root>/input_data/<period>/<modality>/<norm>/)
    feature_matrix.npz     X, X_odd, X_even, unit_ids, n_bins_list,
                           t_ctr_*, n_conds, all metadata arrays
    neuron_metadata.csv    one row per neuron, all metadata columns
    diagnostics/           fig0-fig4

Usage
    from build_feature_matrix import run_build_feature_matrix
    run_build_feature_matrix(units, trials, out_root="...",
                             config_path="config.yaml", lick_df=lick_df)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from rastermap_psth.rastermap_utils import (
    load_cfg, N_WORKERS, MOUSE_INFO, layer_number_mapper,
    get_conditions, get_cond_infos,
    assign_passive_context, assign_active_context,
    get_spike_times, apply_fr_filter,
    precompute_event_map, add_lick_event_map, split_event_map,
    build_feature_matrix_strided,
    load_waveform_classification, load_anatomy_scores,
    order_area_groups, spikes_around_events, _bin_rates_strided,
    _get_events, _save,
)

try:
    from allen_utils import get_custom_area_groups
    _HAS_ALLEN = True
except Exception:
    _HAS_ALLEN = False


# ── diagnostic figures (fig0–fig4) ────────────────────────────────────────────

def fig0_data_summary(units_raw, units_good, unit_ids, trials, cfg, out_dir):
    mid, sid = cfg["mouse_id_col"], cfg["session_id_col"]
    from rastermap_psth.rastermap_utils import CONDITIONS, COND_LABELS
    rows = [
        ("Mice",              trials[mid].nunique()),
        ("Sessions",          trials.groupby([mid, sid]).ngroups),
        ("Neurons (raw)",     len(units_raw)),
        ("Neurons (bc=good)", len(units_good)),
        ("Neurons (FR pass)", len(unit_ids)),
        ("Trials total",      len(trials)),
    ] + [(f"  {lbl}",
          ((trials[cfg["context_col"]] == ctx) &
           (trials[cfg["trial_type_col"]] == tt)).sum())
         for (tt, ctx), lbl in zip(CONDITIONS, COND_LABELS)]
    fig, ax = plt.subplots(figsize=(4, max(3, 0.3 * len(rows) + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=[[r, str(v)] for r, v in rows],
                   colLabels=["Quantity", "Count"], loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    fig.tight_layout()
    _save(fig, out_dir / "fig0_data_summary", dpi=200)


def fig1_trial_counts(trials, cfg, out_dir):
    from rastermap_psth.rastermap_utils import CONDITIONS, COND_LABELS, COND_COLORS
    counts = [((trials[cfg["context_col"]] == ctx) &
               (trials[cfg["trial_type_col"]] == tt)).sum()
              for tt, ctx in CONDITIONS]
    fig, ax = plt.subplots(figsize=(max(5, len(counts) * 0.9), 3))
    bars = ax.bar(COND_LABELS, counts, color=COND_COLORS, edgecolor="none")
    if counts:
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, c + max(counts) * 0.01,
                    str(c), ha="center", va="bottom", fontsize=8, rotation=45)
    ax.set_ylabel("Trial count"); ax.set_title("Trial counts per condition")
    ax.tick_params(axis="x", rotation=30); fig.tight_layout()
    _save(fig, out_dir / "fig1_trial_counts", dpi=200)


def table1b_trial_counts_per_mouse(event_map, cfg, out_dir):
    from rastermap_psth.rastermap_utils import CONDITIONS, COND_ALIGN_COLS, COND_LABELS
    mice      = sorted({k[0] for k in event_map})
    mouse_idx = {m: i for i, m in enumerate(mice)}
    n_mice, n_conds = len(mice), len(CONDITIONS)
    global_c = np.zeros((n_mice, n_conds), int)
    odd_c    = np.zeros((n_mice, n_conds), int)
    even_c   = np.zeros((n_mice, n_conds), int)
    for (mid, sid, ctx, tt, acol), events in event_map.items():
        mi = mouse_idx.get(mid)
        if mi is None: continue
        for ci, ((tt2, ctx2), acol2) in enumerate(zip(CONDITIONS, COND_ALIGN_COLS)):
            if tt == tt2 and ctx == ctx2 and acol == acol2:
                global_c[mi, ci] += len(events)
                odd_c[mi, ci]    += len(events[0::2])
                even_c[mi, ci]   += len(events[1::2])
    rows = [{"mouse_id": m, "condition": lbl,
             "n_global": global_c[mi, ci],
             "n_odd":    odd_c[mi, ci],
             "n_even":   even_c[mi, ci]}
            for mi, m in enumerate(mice)
            for ci, lbl in enumerate(COND_LABELS)]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table1b_trial_counts_per_mouse.csv", index=False)
    print(f"  saved table1b_trial_counts_per_mouse.csv")
    return df


def fig2_fr_distribution(fr_map, unit_ids, thr, out_dir):
    frs = [fr_map[uid] for uid in unit_ids]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(frs, bins=50, color="#4C72B0", edgecolor="none")
    ax.axvline(thr, color="crimson", lw=1.5, ls="--", label=f"threshold={thr} Hz")
    ax.set_xlabel("Firing rate (Hz)"); ax.set_ylabel("Neuron count")
    ax.set_title("FR distribution (post-filter)")
    ax.legend(fontsize=8, frameon=False); fig.tight_layout()
    _save(fig, out_dir / "fig2_fr_distribution", dpi=200)


def fig3_neuron_counts(units_raw, units_good, unit_ids_final, cfg, out_dir):
    mid   = cfg["mouse_id_col"]
    mice  = sorted(units_raw[mid].unique())
    uid_s = set(unit_ids_final)
    x     = np.arange(len(mice))
    fig, ax = plt.subplots(figsize=(max(6, len(mice) * 0.7), 3.5))
    ax.bar(x - 0.25, [len(units_raw[units_raw[mid] == m])   for m in mice],
           0.25, label="raw",     color="#aec6e8")
    ax.bar(x,        [len(units_good[units_good[mid] == m]) for m in mice],
           0.25, label="bc=good", color="#4C72B0")
    ax.bar(x + 0.25, [sum(1 for u in units_good[units_good[mid] == m].index
                          if u in uid_s) for m in mice],
           0.25, label="FR pass", color="#174a8a")
    ax.set_xticks(x); ax.set_xticklabels(mice, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Neuron count"); ax.set_title("Neuron counts per mouse")
    ax.legend(fontsize=7, frameon=False); fig.tight_layout()
    _save(fig, out_dir / "fig3_neuron_counts", dpi=200)


def fig4_sample_neurons(unit_ids, st_map, mouse_map, session_map,
                        event_map, cond_infos, cfg, out_dir, fr_map=None):
    from rastermap_psth.rastermap_utils import (CONDITIONS, COND_ALIGN_COLS, COND_LABELS, COND_COLORS)
    n   = min(cfg.get("n_sample_neurons", 24), len(unit_ids))
    if fr_map is not None:
        fr_vals = np.array([fr_map.get(uid, 0.0) for uid in unit_ids])
        sample  = np.argsort(fr_vals)[::-1][:n]
    else:
        sample = np.random.default_rng(0).choice(len(unit_ids), size=n, replace=False)

    seen, align_types = set(), []
    for acol in COND_ALIGN_COLS:
        if acol not in seen: seen.add(acol); align_types.append(acol)
    align_idx = {a: i for i, a in enumerate(align_types)}
    align_titles = {"start_time": "Stimulus-aligned",
                    "jaw_onset_time": "Jaw-aligned", "lick_time": "Lick-aligned"}

    fig, axes = plt.subplots(n, len(align_types),
                             figsize=(3.5 * len(align_types), 2.8 * n),
                             sharey="row", sharex=False, squeeze=False)
    for row, idx in enumerate(sample):
        uid = unit_ids[idx]; st = st_map[uid]
        for c, ((tt, ctx), (t_pre, t_post, t_ctr_c, n_out_c, bm)) \
                in enumerate(zip(CONDITIONS, cond_infos)):
            acol = COND_ALIGN_COLS[c]; col = align_idx[acol]
            events = _get_events(event_map, mouse_map[uid], session_map[uid], ctx, tt, acol)
            if not len(events): continue
            dt  = cfg["stride_ms"] / 1000
            pad = max(1, int(round(cfg["bin_ms"] / cfg["stride_ms"]))) // 2
            tpe = t_pre + pad * dt; tpe2 = t_post + pad * dt
            ras = spikes_around_events(st, events, tpe, tpe2,
                                       is_whisker=(tt == cfg["whisker_trial_label"]),
                                       artifact_win_s=cfg["artifact_win_s"],
                                       rng=np.random.default_rng(c), align_col=acol)
            ras   = [r[(r >= -tpe) & (r < tpe2)] for r in ras]
            rates = _bin_rates_strided(ras, tpe, tpe2, cfg["bin_ms"], cfg["stride_ms"], n_out_c)
            axes[row, col].plot(t_ctr_c,
                                (rates - rates[:, bm].mean(1, keepdims=True)).mean(0),
                                color=COND_COLORS[c], lw=1.0, label=COND_LABELS[c])
        for col, acol in enumerate(align_types):
            ax = axes[row, col]
            ax.axvline(0, color="k", lw=0.5, ls="--")
            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.legend(fontsize=4, loc="upper left", frameon=False)
            if col == 0: ax.set_ylabel(f"uid={uid}\nΔFR (Hz)", fontsize=7)
            if row == 0: ax.set_title(align_titles.get(acol, acol), fontsize=8)
    fig.suptitle(f"Sample neurons (n={n}, top-FR)", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / "fig4_sample_neurons", dpi=200)


# ── entry point ───────────────────────────────────────────────────────────────

def run_build_feature_matrix(
        units:       pd.DataFrame,
        trials:      pd.DataFrame,
        out_root:    str | Path = "rastermap_psth_out",
        config_path: str | Path = "config.yaml",
        lick_df:     pd.DataFrame | None = None,
        **cfg_overrides,
) -> dict:
    cfg = load_cfg(config_path, **cfg_overrides)

    print('Building PSTH-concatenated feature matrix...')

    # propagate conditions to rastermap_utils globals
    import rastermap_psth.rastermap_utils as _u
    conds = get_conditions(cfg)
    (_u.CONDITIONS, _u.COND_LABELS, _u.COND_COLORS,
     _u.COND_LABELS_MATRIX, _u.COND_ALIGN_COLS) = conds
    CONDITIONS, COND_LABELS, COND_COLORS, COND_LABELS_MATRIX, COND_ALIGN_COLS = conds
    cond_infos = get_cond_infos(cfg, CONDITIONS, COND_ALIGN_COLS)

    # ── output folders ────────────────────────────────────────────────────────
    bl_txt   = "nobl" if cfg.get("baseline_removal", False) else "bl"
    if cfg.get('waveform_type') not in ['rsu','fsu']:
        data_dir = Path(out_root, "rastermap_clustering",
                        cfg["period"], cfg["modality"],
                        f"{bl_txt}_{cfg.get('normalize','zscore')}")
    else:
        data_dir = Path(out_root, "rastermap_clustering",
                        cfg["period"], cfg["modality"],
                        f"{bl_txt}_{cfg.get('normalize', 'zscore')}_{cfg.get('waveform_type')}")
    diag_dir = data_dir / "diagnostics"
    data_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(exist_ok=True)

    if not cfg["recompute_feature_matrix"]:
        return dict(data_folder=data_dir,)

    # save config snapshot
    cfg_snap = {k: (list(v) if isinstance(v, range) else v) for k, v in cfg.items()}
    with open(data_dir / "config_used.yaml", "w") as f:
        yaml.dump(cfg_snap, f, default_flow_style=False, sort_keys=True)

    assert units.index.is_unique, "units index must be unique"

    # ── filters ───────────────────────────────────────────────────────────────
    mouse_info = pd.read_excel(MOUSE_INFO)
    valid_mice = mouse_info[mouse_info["learning_category"].isin(
                    ["good", "moderate", "bad"])]["mouse_id"].unique()
    units["firing_rate"] = units["firing_rate"].astype(float)
    units_raw = units.copy()
    units     = units[units.firing_rate > cfg["global_fr_hz"]]
    trials    = trials[trials.mouse_id.isin(valid_mice)]

    period = cfg["period"]
    if period in ("passive", "passive_active"):
        trials = assign_passive_context(trials, cfg["mouse_id_col"], cfg["session_id_col"])
    if period in ("active", "passive_active"):
        trials = assign_active_context(trials)

    units_good = units[units.bc_label.isin(["good", "mua"])]
    #units_good = units_good[units_good[cfg["area_col"]].isin(["DLS", "DMS", "VS", "TS", "VTA"])]
    all_ids    = units_good.index.tolist()

    st_map      = {uid: get_spike_times(units_good.loc[uid]) for uid in all_ids}
    mouse_map   = units_good[cfg["mouse_id_col"]].to_dict()
    session_map = units_good[cfg["session_id_col"]].to_dict()

    event_map = precompute_event_map(trials, cfg, CONDITIONS, COND_ALIGN_COLS)
    if lick_df is not None and cfg.get("include_lick_conditions", False):
        add_lick_event_map(lick_df, event_map, cfg)

    mid_col, sid_col = cfg["mouse_id_col"], cfg["session_id_col"]
    start_times_map = {
        (mid, sid): np.sort(grp["start_time"].dropna().to_numpy())
        for (mid, sid), grp in trials.groupby([mid_col, sid_col])
    }

    if "jaw_onset_time" in COND_ALIGN_COLS:
        jaw_mice = {k[0] for k in event_map if k[4] == "jaw_onset_time"}
        all_ids  = [u for u in all_ids if mouse_map[u] in jaw_mice]
    if period in ("passive", "passive_active"):
        pas_mice = {k[0] for k in event_map if k[2] in ("passive_pre", "passive_post")}
        all_ids  = [u for u in all_ids if mouse_map[u] in pas_mice]

    unit_ids, fr_map = apply_fr_filter(all_ids, st_map, mouse_map, session_map, event_map, cfg)

    trial_counts_df = table1b_trial_counts_per_mouse(event_map, cfg, diag_dir)
    too_few = trial_counts_df.groupby(["mouse_id", "condition"]).filter(
        lambda x: x["n_odd"].sum() < cfg["n_min_trial_per_condition"]
    )["mouse_id"].unique()
    if len(too_few):
        unit_ids = [u for u in unit_ids if mouse_map[u] not in too_few]
    print(f"  {len(unit_ids)} units pass all filters")

    # ── metadata ──────────────────────────────────────────────────────────────
    reward_map = {uid: ("R+" if r == 1 else "R-" if r == 0 else "unknown")
                  for uid, r in units_good[cfg["reward_group_col"]].to_dict().items()}
    area_map   = units_good[cfg["area_col"]].to_dict()
    reward_arr = np.array([reward_map.get(u, "unknown") for u in unit_ids])
    area_arr   = np.array([area_map.get(u, "unknown")   for u in unit_ids])

    if cfg.get("anatomy_score_cols"):
        am         = load_anatomy_scores(units_good, cfg["anatomy_score_cols"])
        axon_arr   = np.array([am["avg_ipsi"].get(u, np.nan)                   for u in unit_ids], float)
        harris_arr = np.array([am["cc_tc_ct_iterated"].get(u, np.nan)          for u in unit_ids], float)
        gao_arr    = np.array([am["cc_hierarchy_score_columns"].get(u, np.nan) for u in unit_ids], float)
    else:
        axon_arr = harris_arr = gao_arr = None

    if cfg.get("use_wf_classification_csv", True):
        wf_map       = load_waveform_classification(units_good, cfg["mouse_id_col"], cfg["session_id_col"])
        waveform_arr = np.array([wf_map.get(u, "unknown") for u in unit_ids])
    else:
        wd           = units_good["duration"].to_dict()
        wda          = np.array([wd.get(u, np.nan) for u in unit_ids], float)
        waveform_arr = np.where(wda < np.nanpercentile(wda, 30), "NW", "WW")

    units_good["layer_number"] = units_good["layer_number"].map(layer_number_mapper)
    layer_arr = np.array([str(units_good["layer_number"].to_dict().get(u, "unknown"))
                          for u in unit_ids])

    _a2g = ({a: g for g, areas in get_custom_area_groups().items() for a in areas}
            if _HAS_ALLEN else {})
    area_group_arr = np.array([_a2g.get(a, "Other") for a in area_arr])

    # ── diagnostic figures ────────────────────────────────────────────────────
    fig0_data_summary(units_raw, units_good, unit_ids, trials, cfg, diag_dir)
    fig1_trial_counts(trials, cfg, diag_dir)
    fig2_fr_distribution(fr_map, unit_ids, cfg["fr_threshold_hz"], diag_dir)
    fig3_neuron_counts(units_raw, units_good, unit_ids, cfg, diag_dir)

    # ── feature matrices ──────────────────────────────────────────────────────
    def _build(em, tag):
        print(f"Building X ({tag})...")
        return build_feature_matrix_strided(
            unit_ids, st_map, mouse_map, session_map, em, cfg,
            CONDITIONS, COND_LABELS, COND_COLORS, COND_LABELS_MATRIX,
            cond_align_cols=COND_ALIGN_COLS,
            start_times_map=start_times_map)

    X,      t_ctrs, n_bins_list = _build(event_map, "all trials")
    em_odd, em_even             = split_event_map(event_map)
    X_odd,  t_ctrs, n_bins_list = _build(em_odd,    "odd trials")
    X_even, _,      _           = _build(em_even,   "even trials")

    # ── drop degenerate rows (same mask applied to all three) ─────────────────
    bad = (np.isnan(X).any(1) | np.isinf(X).any(1) |
           (X == 0).all(1)    | (X.std(1) < 1e-6))
    if bad.sum():
        print(f"  Dropping {bad.sum()} degenerate rows from X, X_odd, X_even")
        unit_ids = [u for u, b in zip(unit_ids, bad) if not b]
        X = X[~bad]; X_odd = X_odd[~bad]; X_even = X_even[~bad]
        reward_arr = reward_arr[~bad]; area_arr       = area_arr[~bad]
        waveform_arr = waveform_arr[~bad]; layer_arr  = layer_arr[~bad]
        area_group_arr = area_group_arr[~bad]
        if axon_arr is not None:
            axon_arr = axon_arr[~bad]; harris_arr = harris_arr[~bad]
            gao_arr  = gao_arr[~bad]
    print(f"  Final: {len(unit_ids)} neurons  X={X.shape}")

    # ── odd/even reliability filter ───────────────────────────────────────────
    # Per-neuron Pearson r between X_odd[i] and X_even[i]: measures how
    # reproducibly a neuron's PSTH shape replicates across trial splits.
    # Both vectors are already z-scored so Pearson r = normalised dot product.
    # Neurons below the threshold are noisy or trial-sparse and unreliable
    # for clustering.
    def _pearson_r_rows(A, B):
        """Per-row Pearson r between two matrices of the same shape."""
        A = A - A.mean(1, keepdims=True)
        B = B - B.mean(1, keepdims=True)
        num  = (A * B).sum(1)
        denom = (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)) + 1e-12
        return num / denom

    odd_even_r   = _pearson_r_rows(X_odd, X_even)
    r_threshold  = cfg.get("min_odd_even_r", 0.2)
    reliable     = odd_even_r >= r_threshold

    print(f"  Odd/even reliability filter (r >= {r_threshold}): "
          f"{reliable.sum()} / {len(unit_ids)} neurons pass "
          f"(median r={np.median(odd_even_r):.2f}, "
          f"mean r={odd_even_r.mean():.2f})")

    if (~reliable).sum():
        unit_ids       = [u for u, r in zip(unit_ids, reliable) if r]
        X              = X[reliable];    X_odd  = X_odd[reliable];  X_even = X_even[reliable]
        odd_even_r     = odd_even_r[reliable]
        reward_arr     = reward_arr[reliable];    area_arr       = area_arr[reliable]
        waveform_arr   = waveform_arr[reliable];  layer_arr      = layer_arr[reliable]
        area_group_arr = area_group_arr[reliable]
        if axon_arr is not None:
            axon_arr = axon_arr[reliable]; harris_arr = harris_arr[reliable]
            gao_arr  = gao_arr[reliable]

    print(f"  After reliability filter: {len(unit_ids)} neurons")

    fig4_sample_neurons(unit_ids, st_map, mouse_map, session_map,
                        event_map, cond_infos, cfg, diag_dir, fr_map)

    # ── save feature_matrix.npz ───────────────────────────────────────────────
    sd = dict(X=X, X_odd=X_odd, X_even=X_even,
              unit_ids=np.array(unit_ids),
              n_bins_list=np.array(n_bins_list), n_conds=np.array(len(t_ctrs)),
              mouse_arr=np.array([mouse_map[u]   for u in unit_ids]),
              session_arr=np.array([session_map[u] for u in unit_ids]),
              reward_arr=reward_arr, area_arr=area_arr,
              waveform_arr=waveform_arr, layer_arr=layer_arr,
              area_group_arr=area_group_arr,
              odd_even_r=odd_even_r)
    for ci, tc in enumerate(t_ctrs): sd[f"t_ctr_{ci}"] = tc
    if axon_arr is not None:
        sd.update(axon_arr=axon_arr, harris_arr=harris_arr, gao_arr=gao_arr)
    npz_path = data_dir / "feature_matrix.npz"
    np.savez_compressed(npz_path, **sd)
    print(f"  Saved → {npz_path}")

    # ── save neuron_metadata.csv ──────────────────────────────────────────────
    cid_map = units_good["cluster_id"].to_dict()
    egrp_map = units_good["electrode_group"].to_dict()
    meta_df = pd.DataFrame({
        "unit_id":         unit_ids,
        "mouse_id":        [mouse_map[u]   for u in unit_ids],
        "session_id":      [session_map[u] for u in unit_ids],
        "cluster_id":      [cid_map.get(u)  for u in unit_ids],
        "electrode_group": [egrp_map.get(u) for u in unit_ids],
        "reward_group":    reward_arr, "area_acronym": area_arr,
        "area_group":      area_group_arr, "waveform_type": waveform_arr,
        "layer_number":    layer_arr,
        "odd_even_r":      odd_even_r,
    })
    if axon_arr is not None:
        meta_df["avg_ipsi"]                   = axon_arr
        meta_df["cc_tc_ct_iterated"]          = harris_arr
        meta_df["cc_hierarchy_score_columns"] = gao_arr
    meta_df.to_csv(data_dir / "neuron_metadata.csv", index=False)
    print(f"  Saved → neuron_metadata.csv  ({len(meta_df)} neurons)")
    print(f"\nDone.  Data folder → {data_dir}")

    return dict(data_folder=data_dir, unit_ids=unit_ids,
                X=X, X_odd=X_odd, X_even=X_even,
                t_ctrs=t_ctrs, n_bins_list=n_bins_list,
                reward_arr=reward_arr, area_arr=area_arr,
                waveform_arr=waveform_arr, layer_arr=layer_arr,
                area_group_arr=area_group_arr,
                axon_arr=axon_arr, harris_arr=harris_arr, gao_arr=gao_arr)