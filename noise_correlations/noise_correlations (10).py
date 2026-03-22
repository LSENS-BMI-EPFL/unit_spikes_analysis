#!/usr/bin/env python3
"""
Noise correlation analysis — brain-wide, session-level.
Usage: python noise_correlations.py <input_folder> <output_folder>
"""

import ast, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, wilcoxon, pearsonr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pynwb import NWBHDF5IO

import allen_utils
from utils import convert_electrode_group_object_to_columns

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
COLORS      = {0: "crimson", 1: "forestgreen"}
GROUPS      = {0: "R-",      1: "R+"}
WINDOWS     = {"baseline": (-2.0, -0.005), "stimulus": (0.005, 0.05)}
TRIAL_TYPES = ["whisker_trial", "auditory_trial", "no_stim_trial"]
MIN_MICE    = 2
MIN_NEURONS = 10

TC_RANGE   = (-2.5, 1.0)
TC_WIN     = 0.5
TC_STEP    = 0.1
TC_CENTERS = np.arange(TC_RANGE[0] + TC_WIN / 2, TC_RANGE[1], TC_STEP)

EXCLUDED  = set(allen_utils.get_excluded_areas())
GROUP_MAP = allen_utils.get_custom_area_group_revert_dict()
AREA_ORDER = allen_utils.get_custom_area_order()   # canonical ordered list

# ── IO ────────────────────────────────────────────────────────────────────────

def _wh_reward(nwb):
    try:
        raw = nwb.session_metadata["wh_reward"]
    except Exception:
        try:    raw = nwb.scratch["wh_reward"].data
        except Exception: return {}
    return ast.literal_eval(raw) if isinstance(raw, str) else raw


def _fine_areas(units_df):
    df = allen_utils.process_allen_labels(units_df.copy(), subdivide_areas=True)
    for col in ["area_custom_acronym", "area_acronym_custom", "ccf_atlas_parent_acronym"]:
        if col in df.columns:
            return df[col].reindex(units_df.index).fillna("Unknown").astype(str).values
    raise ValueError("allen_utils.process_allen_labels: no recognised area column")


def load_nwb(path):
    with NWBHDF5IO(path, "r") as io:
        nwb = io.read()
        if nwb.units is None:
            raise ValueError("no units table")
        metadata     = ast.literal_eval(nwb.experiment_description)
        reward_group = metadata["wh_reward"]
        try:    mouse_id = nwb.subject.subject_id
        except Exception: mouse_id = path.stem
        return nwb.trials.to_dataframe(), nwb.units.to_dataframe(), reward_group, mouse_id


def filter_trials(trials, trial_type):
    trials = trials[trials["context"] != "passive"]
    m = trials["trial_type"] == trial_type
    if "block_perf_type" in trials.columns:
        m &= trials["block_perf_type"] != 6
    return trials[m]


# ── NC core ───────────────────────────────────────────────────────────────────

def spike_counts(spike_times, t_starts, t0, t1):
    st = np.sort(np.asarray(spike_times, dtype=float))
    return (np.searchsorted(st, t_starts + t1) - np.searchsorted(st, t_starts + t0)).astype(float)


def count_matrix(units, areas, t_starts, t0, t1):
    return np.vstack([spike_counts(r.spike_times, t_starts, t0, t1)
                      for _, r in units.iterrows()])


def pearson_mat(C):
    std = C.std(axis=1)
    with np.errstate(invalid="ignore"):
        z = np.where(std[:, None] < 1e-10, np.nan,
                     (C - C.mean(axis=1, keepdims=True)) / std[:, None])
        r = (z @ z.T) / (C.shape[1] - 1)
    np.fill_diagonal(r, np.nan)
    return r


def nc_by_area(r, areas):
    areas = np.array(areas)
    rows  = []
    for ai in np.unique(areas):
        for aj in np.unique(areas):
            ii   = np.where(areas == ai)[0]
            jj   = np.where(areas == aj)[0]
            blk  = r[np.ix_(ii, jj)]
            vals = blk[np.triu_indices_from(blk, k=1)] if ai == aj else blk.ravel()
            vals = vals[~np.isnan(vals)]
            if len(vals):
                rows.append({"area_i": ai, "area_j": aj, "nc": vals.mean(), "n": len(vals)})
    return pd.DataFrame(rows)


def timecourse(units, areas, t_starts):
    a      = np.array(areas)
    unique = sorted(set(areas))
    w_all, x_all = [], []
    per_area = {ar: [] for ar in unique}

    for c in TC_CENTERS:
        C    = count_matrix(units, areas, t_starts, c - TC_WIN / 2, c + TC_WIN / 2)
        r    = pearson_mat(C)
        uidx = np.triu_indices(len(a), k=1)
        same = a[uidx[0]] == a[uidx[1]]
        vals = r[uidx]
        w_all.append(np.nanmean(vals[same]))
        x_all.append(np.nanmean(vals[~same]))
        for ar in unique:
            aidx = np.where(a == ar)[0]
            if len(aidx) < 2:
                per_area[ar].append(np.nan)
            else:
                sub = r[np.ix_(aidx, aidx)]
                per_area[ar].append(np.nanmean(sub[np.triu_indices_from(sub, k=1)]))

    return (np.array(w_all), np.array(x_all),
            {ar: np.array(v) for ar, v in per_area.items()})


# ── Session processing ────────────────────────────────────────────────────────

def process(path):
    trials_all, units_raw, reward_group, mouse_id = load_nwb(path)
    units_raw["mouse_id"] = mouse_id
    units_raw = convert_electrode_group_object_to_columns(units_raw)
    units_raw = units_raw[units_raw["bc_label"].isin(["good", "mua"])].reset_index(drop=True)

    fine = _fine_areas(units_raw)
    glob = np.array([GROUP_MAP.get(a, "Other") for a in fine])

    # store raw pairwise NC values per session per window (for diagnostic plots)
    raw_nc = {}

    out = {"group": reward_group, "mouse_id": mouse_id, "results": {}, "raw_nc": raw_nc}

    for gran, areas_all in [("fine", fine), ("global", glob)]:
        mask  = ~pd.Series(areas_all).isin(EXCLUDED).values
        units = units_raw[mask].reset_index(drop=True)
        areas = areas_all[mask]

        gran_res = {"neuron_counts": pd.Series(areas).value_counts().to_dict()}
        for tt in TRIAL_TYPES:
            trials = filter_trials(trials_all, tt)
            if len(trials) < 5:
                continue
            t   = trials["start_time"].values
            res = {"areas": sorted(set(areas))}
            for wname, (t0, t1) in WINDOWS.items():
                C           = count_matrix(units, areas, t, t0, t1)
                r           = pearson_mat(C)
                res[f"nc_{wname}"]       = nc_by_area(r, areas)
                res[f"C_{wname}"]        = C          # raw counts (n_units × n_trials)
                res[f"r_mat_{wname}"]    = r          # full NC matrix
                res[f"areas_arr_{wname}"] = np.array(areas)
            res["tc_within"], res["tc_across"], res["tc_per_area"] = timecourse(units, areas, t)
            gran_res[tt] = res

        out["results"][gran] = gran_res
    return out


# ── Area ordering & filtering ─────────────────────────────────────────────────

def ordered_areas(valid, gran):
    """Return valid areas in canonical AREA_ORDER, appending unknowns at end."""
    ordered = [a for a in AREA_ORDER if a in valid]
    extra   = sorted(valid - set(ordered))
    return ordered + extra


def valid_areas(sessions, gran):
    a_groups  = {}
    a_mice    = {}
    a_neurons = {}

    for s in sessions.values():
        gr = s["results"].get(gran, {})
        for a, n in gr.get("neuron_counts", {}).items():
            if a in EXCLUDED: continue
            a_neurons[a] = a_neurons.get(a, 0) + n
        for tt in TRIAL_TYPES:
            if tt not in gr: continue
            for a in gr[tt]["areas"]:
                a_groups.setdefault(a, set()).add(s["group"])
                a_mice.setdefault(a, set()).add(s["mouse_id"])
            break

    all_areas = sorted(set(a_groups) | set(a_neurons))
    print(f"\n── valid_areas [{gran}] ─────────────────────────────────────")
    print(f"  {'area':<30} {'groups':<12} {'mice':>5} {'neurons':>8} {'pass'}")
    for a in all_areas:
        grps   = a_groups.get(a, set())
        n_mice = len(a_mice.get(a, set()))
        n_neu  = a_neurons.get(a, 0)
        ok     = ({0, 1} <= grps) and (n_mice >= MIN_MICE) and (n_neu >= MIN_NEURONS)
        print(f"  {a:<30} {str(grps):<12} {n_mice:>5} {n_neu:>8} {'YES' if ok else 'NO'}")

    return {a for a in a_groups
            if {0, 1} <= a_groups[a]
            and len(a_mice.get(a, set())) >= MIN_MICE
            and a_neurons.get(a, 0)       >= MIN_NEURONS}


# ── Stats helpers ─────────────────────────────────────────────────────────────

def mwu(a, b):
    a, b = np.asarray(a), np.asarray(b)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) > 1 and len(b) > 1:
        return mannwhitneyu(a, b, alternative="two-sided")
    return None, None


def wilcox(a, b):
    """Paired Wilcoxon signed-rank for baseline vs stimulus within a group."""
    d = np.asarray(a) - np.asarray(b)
    d = d[~np.isnan(d)]
    if len(d) > 4:
        return wilcoxon(d)
    return None, None


def pstar(p):
    if p is None:  return ""
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "ns"


def annotate_bracket(ax, x0, x1, y, p, color="k"):
    s = pstar(p)
    if not s or s == "ns": return
    h = np.diff(ax.get_ylim())[0] * 0.03
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], "-", color=color, lw=0.8)
    ax.text((x0 + x1) / 2, y + h, s, ha="center", va="bottom", fontsize=7, color=color)


def _mouse_nc(sessions, gran, tt, wname, area, g, pair_type="within"):
    """Return one NC value per mouse (mean across sessions of same mouse)."""
    per_mouse = {}
    for s in sessions.values():
        if s["group"] != g: continue
        gr = s["results"].get(gran, {})
        if tt not in gr: continue
        df = gr[tt][f"nc_{wname}"]
        if pair_type == "within":
            row = df[(df.area_i == area) & (df.area_j == area)]
        else:
            row = df[(df.area_i == area) & (df.area_j != area) |
                     (df.area_j == area) & (df.area_i != area)]
        if row.empty: continue
        per_mouse.setdefault(s["mouse_id"], []).append(row["nc"].mean())
    return np.array([np.mean(v) for v in per_mouse.values()])


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _grp(sessions, g, gran, tt):
    return [s["results"][gran][tt] for s in sessions.values()
            if s["group"] == g and tt in s["results"].get(gran, {})]


def _mean_mat(sessions, gran, tt, window, g, areas):
    idx  = {a: i for i, a in enumerate(areas)}
    n    = len(areas)
    mats = []
    for res in _grp(sessions, g, gran, tt):
        mat = np.full((n, n), np.nan)
        for _, row in res[f"nc_{window}"].iterrows():
            i, j = idx.get(row.area_i), idx.get(row.area_j)
            if i is not None and j is not None:
                mat[i, j] = mat[j, i] = row.nc
        mats.append(mat)
    return np.nanmean(mats, axis=0) if mats else np.full((n, n), np.nan)


def _safe_vmax(*mats):
    vals = [np.nanmax(np.abs(m)) for m in mats if not np.all(np.isnan(m))]
    return max(vals) if vals else 1.0


def _dedup_legend(ax, **kw):
    h, l = ax.get_legend_handles_labels()
    seen  = {}
    for hi, li in zip(h, l): seen.setdefault(li, hi)
    ax.legend(seen.values(), seen.keys(), **kw)


def _heatmap(ax, mat, areas, vmax, title, cbar=True):
    sns.heatmap(mat, xticklabels=areas, yticklabels=areas,
                cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax, ax=ax, square=True,
                cbar=cbar, cbar_kws={"label": "NC (r)", "shrink": 0.7} if cbar else {})
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


# ── Analysis summary ─────────────────────────────────────────────────────────

def print_analysis_summary(sessions, gran, valid):
    """Print area pairs, neuron counts, and mice per group before analysis."""
    areas = ordered_areas(valid, gran)
    print(f"\n{'='*65}")
    print(f"  ANALYSIS SUMMARY  |  {gran}  |  {len(areas)} areas")
    print(f"{'='*65}")

    # mice per group
    for g in [0, 1]:
        mice = sorted({s["mouse_id"] for s in sessions.values() if s["group"] == g})
        print(f"  {GROUPS[g]}  ({len(mice)} mice): {', '.join(mice)}")

    # neurons per area per group
    print(f"\n  {'Area':<28} {'R- neurons':>12} {'R+ neurons':>12}  {'R- mice':>8}  {'R+ mice':>8}")
    print(f"  {'-'*28} {'-'*12} {'-'*12}  {'-'*8}  {'-'*8}")
    for area in areas:
        n_neu, n_mice = {0: 0, 1: 0}, {0: set(), 1: set()}
        for s in sessions.values():
            g  = s["group"]
            nc = s["results"].get(gran, {}).get("neuron_counts", {})
            if area in nc:
                n_neu[g]  += nc[area]
                n_mice[g].add(s["mouse_id"])
        print(f"  {area:<28} {n_neu[0]:>12} {n_neu[1]:>12}  {len(n_mice[0]):>8}  {len(n_mice[1]):>8}")

    # area pairs
    pairs_within = [(a, a) for a in areas]
    pairs_across = [(a, b) for i, a in enumerate(areas) for b in areas[i+1:]]
    print(f"\n  Within-area pairs : {len(pairs_within)}")
    print(f"  Across-area pairs : {len(pairs_across)}  "
          f"({len(pairs_across)} unique, {len(pairs_across)*2} directed)")
    print(f"{'='*65}\n")


# ── Raw / diagnostic figures ──────────────────────────────────────────────────

def plot_raw_nc_distributions(sessions, gran, tt, valid):
    """Histogram of raw pairwise NC values per group, within-area pairs only."""
    areas = ordered_areas(valid, gran)
    if not areas: return plt.figure()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)
    bins = np.linspace(-1, 1, 50)

    for ax, g in zip(axes, [0, 1]):
        all_vals = []
        for res in _grp(sessions, g, gran, tt):
            r   = res.get(f"r_mat_baseline")
            arr = res.get(f"areas_arr_baseline")
            if r is None or arr is None: continue
            a   = np.array(arr)
            for area in areas:
                aidx = np.where(a == area)[0]
                if len(aidx) < 2: continue
                sub  = r[np.ix_(aidx, aidx)]
                idx2 = np.triu_indices_from(sub, k=1)
                all_vals.extend(sub[idx2][~np.isnan(sub[idx2])].tolist())
        ax.hist(all_vals, bins=bins, color=COLORS[g], alpha=0.7, density=True)
        ax.axvline(np.nanmean(all_vals), color=COLORS[g], lw=1.5, ls="--",
                   label=f"mean={np.nanmean(all_vals):.3f}" if all_vals else "")
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.set(title=f"{GROUPS[g]} — baseline", xlabel="Pearson r", ylabel="Density")
        ax.legend(fontsize=8)

    fig.suptitle(f"Raw NC distribution (within-area pairs)  |  {gran}  |  {tt}")
    fig.tight_layout()
    return fig


def plot_example_nc_computation(sessions, gran, tt, valid):
    """
    For one example session per group: show spike count matrix,
    mean-subtracted (residual) counts, and resulting NC matrix.
    """
    areas = ordered_areas(valid, gran)
    if not areas: return plt.figure()

    fig = plt.figure(figsize=(18, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    for row, g in enumerate([0, 1]):
        res_list = _grp(sessions, g, gran, tt)
        if not res_list: continue
        res = res_list[0]    # first session as example
        C   = res.get("C_baseline")
        r   = res.get("r_mat_baseline")
        arr = res.get("areas_arr_baseline")
        if C is None: continue
        a = np.array(arr)

        # restrict to valid areas
        keep = np.isin(a, list(valid))
        C    = C[keep]
        r    = r[np.ix_(keep, keep)]
        a    = a[keep]

        # sort by canonical area order
        order = sorted(range(len(a)), key=lambda i: AREA_ORDER.index(a[i])
                        if a[i] in AREA_ORDER else len(AREA_ORDER))
        C = C[order]; r = r[np.ix_(order, order)]; a = a[order]
        residuals = C - C.mean(axis=1, keepdims=True)

        # panel 1: spike count matrix (units × trials)
        ax1 = fig.add_subplot(gs[row, 0])
        im  = ax1.imshow(C, aspect="auto", cmap="viridis",
                         interpolation="none", vmin=0, vmax=np.percentile(C, 99))
        plt.colorbar(im, ax=ax1, fraction=0.03, label="spikes")
        ax1.set(title=f"{GROUPS[g]} — spike counts", xlabel="Trial", ylabel="Unit")
        ax1.tick_params(labelsize=7)

        # panel 2: residual counts (mean-subtracted)
        ax2 = fig.add_subplot(gs[row, 1])
        lim = np.percentile(np.abs(residuals), 99)
        im2 = ax2.imshow(residuals, aspect="auto", cmap="RdBu_r",
                         vmin=-lim, vmax=lim, interpolation="none")
        plt.colorbar(im2, ax=ax2, fraction=0.03, label="residual spikes")
        ax2.set(title=f"{GROUPS[g]} — residuals (mean-subtracted)", xlabel="Trial", ylabel="Unit")
        ax2.tick_params(labelsize=7)

        # panel 3: NC matrix
        ax3  = fig.add_subplot(gs[row, 2])
        vmax = _safe_vmax(r)
        sns.heatmap(r, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                    ax=ax3, square=True, cbar_kws={"label": "r", "shrink": 0.7},
                    xticklabels=False, yticklabels=False)
        ax3.set_title(f"{GROUPS[g]} — NC matrix (baseline)", fontsize=9)

        # colour-code unit axis by area
        unique_a = list(dict.fromkeys(a))
        cmap_a   = plt.cm.tab20(np.linspace(0, 1, len(unique_a)))
        col_map  = {ar: cmap_a[i] for i, ar in enumerate(unique_a)}
        for ax in [ax1, ax2]:
            for i, ar in enumerate(a):
                ax.add_patch(plt.Rectangle((-0.5, i - 0.5), 0.5, 1,
                             color=col_map[ar], clip_on=False, transform=ax.get_yaxis_transform()))

    fig.suptitle(f"Example session NC computation  |  {gran}  |  {tt}")
    return fig


def plot_nc_scatter(sessions, gran, tt, valid):
    """Scatter: baseline NC vs stimulus NC per neuron pair, per area, per group."""
    areas = ordered_areas(valid, gran)
    if not areas: return plt.figure()

    ncols = min(5, len(areas))
    nrows = int(np.ceil(len(areas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).ravel()

    for i, area in enumerate(areas):
        ax    = axes[i]
        annot = []                          # (text, color) per group
        for g in [0, 1]:
            bvals, svals = [], []
            for res in _grp(sessions, g, gran, tt):
                arr = res.get("areas_arr_baseline")
                if arr is None: continue
                a    = np.array(arr)
                aidx = np.where(a == area)[0]
                if len(aidx) < 2: continue
                rb   = res["r_mat_baseline"][np.ix_(aidx, aidx)]
                rs   = res["r_mat_stimulus"][np.ix_(aidx, aidx)]
                idx2 = np.triu_indices_from(rb, k=1)
                mask = ~np.isnan(rb[idx2]) & ~np.isnan(rs[idx2])
                bvals.extend(rb[idx2][mask].tolist())
                svals.extend(rs[idx2][mask].tolist())
            if len(bvals) < 2: continue
            ax.scatter(bvals, svals, s=3, alpha=0.3, color=COLORS[g],
                       label=GROUPS[g], rasterized=True)
            r, p = pearsonr(bvals, svals)
            p_str = f"p<0.001" if p < 0.001 else f"p={p:.3f}"
            annot.append((f"{GROUPS[g]}: r={r:.2f}, {p_str}", COLORS[g]))

        lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
                  abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.6)
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.axvline(0, color="k", lw=0.4, ls=":")
        ax.set_title(area, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("NC baseline", fontsize=7)
        ax.set_ylabel("NC stimulus", fontsize=7)

        # annotation box — one line per group, bottom-left corner
        for k, (txt, col) in enumerate(annot):
            ax.text(0.04, 0.04 + k * 0.13, txt,
                    transform=ax.transAxes, fontsize=6.5,
                    color=col, va="bottom",
                    bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))

    for j in range(i + 1, len(axes)): axes[j].set_visible(False)
    axes[0].legend(fontsize=7, markerscale=3)
    fig.suptitle(f"Baseline vs stimulus NC (pairwise)  |  {gran}  |  {tt}")
    fig.tight_layout()
    return fig


# ── Summary figures ───────────────────────────────────────────────────────────

def plot_timecourse(sessions, gran, tt, valid):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, key, title in zip(axes, ["tc_within", "tc_across"], ["Within-area", "Across-area"]):
        for g in [0, 1]:
            data = [res[key] for res in _grp(sessions, g, gran, tt)]
            if not data: continue
            m  = np.nanmean(data, axis=0)
            se = np.nanstd(data,  axis=0) / np.sqrt(len(data))
            ax.plot(TC_CENTERS, m, color=COLORS[g], label=GROUPS[g])
            ax.fill_between(TC_CENTERS, m - se, m + se, color=COLORS[g], alpha=0.2)
        for wname, (t0, t1) in WINDOWS.items():
            ax.axvspan(t0, t1, color="gray", alpha=0.08, label=f"_{wname}")
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.axhline(0, color="k", ls=":",  lw=0.5)
        ax.set(title=title, xlabel="Time from trial start (s)")
    axes[0].set_ylabel("Mean NC (r)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"NC time course  |  {gran}  |  {tt}")
    fig.tight_layout()
    return fig


def plot_tc_per_area(sessions, gran, tt, valid):
    areas = ordered_areas(valid, gran)
    if not areas: return plt.figure()
    ncols = min(5, len(areas))
    nrows = int(np.ceil(len(areas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows), sharey=True)
    axes = np.array(axes).ravel()
    for i, area in enumerate(areas):
        ax = axes[i]
        for g in [0, 1]:
            data = [res["tc_per_area"][area] for res in _grp(sessions, g, gran, tt)
                    if area in res.get("tc_per_area", {})]
            if not data: continue
            m  = np.nanmean(data, axis=0)
            se = np.nanstd(data,  axis=0) / np.sqrt(len(data))
            ax.plot(TC_CENTERS, m, color=COLORS[g], lw=1, label=GROUPS[g])
            ax.fill_between(TC_CENTERS, m - se, m + se, color=COLORS[g], alpha=0.15)
        for wname, (t0, t1) in WINDOWS.items():
            ax.axvspan(t0, t1, color="gray", alpha=0.08)
        ax.axvline(0, color="k", ls="--", lw=0.6)
        ax.axhline(0, color="k", ls=":",  lw=0.4)
        ax.set_title(area, fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(i + 1, len(axes)): axes[j].set_visible(False)
    if len(axes) and axes[0].lines: axes[0].legend(fontsize=7)
    fig.suptitle(f"Within-area NC per area  |  {gran}  |  {tt}")
    fig.tight_layout()
    return fig



def plot_all_timecourses(sessions, gran, valid):
    """
    Rows = trial_types, cols = within / across.
    All panels share a single y-axis.
    """
    n_tt  = len(TRIAL_TYPES)
    fig, axes = plt.subplots(n_tt, 2, figsize=(12, 3.5 * n_tt),
                             sharey=True, sharex=True, squeeze=False)
    for row, tt in enumerate(TRIAL_TYPES):
        for col, (key, title) in enumerate([("tc_within", "Within-area"),
                                            ("tc_across", "Across-area")]):
            ax = axes[row][col]
            for g in [0, 1]:
                data = [res[key] for res in _grp(sessions, g, gran, tt)]
                if not data: continue
                m  = np.nanmean(data, axis=0)
                se = np.nanstd(data,  axis=0) / np.sqrt(len(data))
                ax.plot(TC_CENTERS, m, color=COLORS[g], label=GROUPS[g])
                ax.fill_between(TC_CENTERS, m - se, m + se, color=COLORS[g], alpha=0.2)
            for wname, (t0, t1) in WINDOWS.items():
                ax.axvspan(t0, t1, color="gray", alpha=0.08)
            ax.axvline(0, color="k", ls="--", lw=0.8)
            ax.axhline(0, color="k", ls=":",  lw=0.5)
            if row == 0:
                ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(tt.replace("_trial", ""), fontsize=8)
            if row == n_tt - 1:
                ax.set_xlabel("Time from trial start (s)", fontsize=8)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"NC time course  |  {gran}  (shared y-axis)", y=1.005)
    fig.tight_layout()
    return fig


def plot_all_tc_per_area(sessions, gran, valid):
    """
    Rows = trial_types, cols = areas (canonical order).
    All panels share a single y-axis.
    """
    areas = ordered_areas(valid, gran)
    if not areas: return plt.figure()
    n_tt  = len(TRIAL_TYPES)
    n_a   = len(areas)
    fig, axes = plt.subplots(n_tt, n_a,
                             figsize=(2.8 * n_a, 2.8 * n_tt),
                             sharey=True, sharex=True, squeeze=False)
    for row, tt in enumerate(TRIAL_TYPES):
        for col, area in enumerate(areas):
            ax = axes[row][col]
            for g in [0, 1]:
                data = [res["tc_per_area"][area] for res in _grp(sessions, g, gran, tt)
                        if area in res.get("tc_per_area", {})]
                if not data: continue
                m  = np.nanmean(data, axis=0)
                se = np.nanstd(data,  axis=0) / np.sqrt(len(data))
                ax.plot(TC_CENTERS, m, color=COLORS[g], lw=1, label=GROUPS[g])
                ax.fill_between(TC_CENTERS, m - se, m + se, color=COLORS[g], alpha=0.15)
            for wname, (t0, t1) in WINDOWS.items():
                ax.axvspan(t0, t1, color="gray", alpha=0.08)
            ax.axvline(0, color="k", ls="--", lw=0.6)
            ax.axhline(0, color="k", ls=":",  lw=0.4)
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(area, fontsize=8)
            if col == 0:
                ax.set_ylabel(tt.replace("_trial", ""), fontsize=8)
            if row == n_tt - 1:
                ax.set_xlabel("Time (s)", fontsize=7)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"Within-area NC per area  |  {gran}  (shared y-axis)", y=1.005)
    fig.tight_layout()
    return fig



def plot_tc_single_area(sessions, gran, area):
    """One figure per area: rows = trial_types, cols = within / across. Shared y-axis."""
    n_tt = len(TRIAL_TYPES)
    fig, axes = plt.subplots(n_tt, 2, figsize=(10, 3 * n_tt),
                             sharey=True, sharex=True, squeeze=False)
    for row, tt in enumerate(TRIAL_TYPES):
        for col, (key, title) in enumerate([("tc_per_area", "Within-area"),
                                            ("tc_across",   "Across-area (global)")]):
            ax = axes[row][col]
            for g in [0, 1]:
                if key == "tc_per_area":
                    data = [res[key][area] for res in _grp(sessions, g, gran, tt)
                            if area in res.get(key, {})]
                else:
                    data = [res[key] for res in _grp(sessions, g, gran, tt)]
                if not data: continue
                m  = np.nanmean(data, axis=0)
                se = np.nanstd(data,  axis=0) / np.sqrt(len(data))
                ax.plot(TC_CENTERS, m, color=COLORS[g], label=GROUPS[g])
                ax.fill_between(TC_CENTERS, m - se, m + se, color=COLORS[g], alpha=0.2)
            for wname, (t0, t1) in WINDOWS.items():
                ax.axvspan(t0, t1, color="gray", alpha=0.08)
            ax.axvline(0, color="k", ls="--", lw=0.8)
            ax.axhline(0, color="k", ls=":",  lw=0.5)
            if row == 0:        ax.set_title(title, fontsize=9)
            if col == 0:        ax.set_ylabel(tt.replace("_trial", ""), fontsize=8)
            if row == n_tt - 1: ax.set_xlabel("Time from trial start (s)", fontsize=8)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"{area}  |  {gran}", y=1.005)
    fig.tight_layout()
    return fig


def plot_all_matrices(sessions, gran, valid):
    """
    Rows = trial_types.
    Cols = R+ base | R- base | R+ stim | R- stim
         | R+ stim-base | R- stim-base
         | R+ minus R- (base) | R+ minus R- (stim)
    NC panels share one colormap; both sets of diffs each share their own.
    """
    areas = ordered_areas(valid, gran)
    if not areas: return plt.figure()
    n_tt  = len(TRIAL_TYPES)
    COLS  = ["R+ base", "R- base", "R+ stim", "R- stim",
             "R+ stim−base", "R- stim−base",
             "(R+)−(R-) base", "(R+)−(R-) stim"]
    ncols = len(COLS)

    # pre-compute and collect for shared vmaxes
    all_nc, all_stim_base, all_group_diff = [], [], []
    mats_by_tt = {}
    for tt in TRIAL_TYPES:
        rp_b = _mean_mat(sessions, gran, tt, "baseline", 1, areas)
        rm_b = _mean_mat(sessions, gran, tt, "baseline", 0, areas)
        rp_s = _mean_mat(sessions, gran, tt, "stimulus",  1, areas)
        rm_s = _mean_mat(sessions, gran, tt, "stimulus",  0, areas)
        dp   = rp_s - rp_b          # R+ stim − base
        dm   = rm_s - rm_b          # R- stim − base
        db   = rp_b - rm_b          # R+ minus R- (baseline)
        ds   = rp_s - rm_s          # R+ minus R- (stimulus)
        mats_by_tt[tt] = (rp_b, rm_b, rp_s, rm_s, dp, dm, db, ds)
        all_nc.extend([rp_b, rm_b, rp_s, rm_s])
        all_stim_base.extend([dp, dm])
        all_group_diff.extend([db, ds])

    vmax_nc        = _safe_vmax(*all_nc)
    vmax_stim_base = _safe_vmax(*all_stim_base)
    vmax_group     = _safe_vmax(*all_group_diff)
    vmaxes = [vmax_nc]*4 + [vmax_stim_base]*2 + [vmax_group]*2

    # 3 extra columns for colorbars (one per colorscale group)
    CB_COLS   = [3, 5, 7]          # after col indices 0-3, 4-5, 6-7
    CB_VMAXES = [vmax_nc, vmax_stim_base, vmax_group]
    CB_LABELS = ["NC (r)", "Δ stim−base (r)", "Δ R+−R- (r)"]
    total_cols = ncols + 3         # 8 data + 3 cbar columns

    sz  = max(2.5, 16 / ncols)
    # column widths: data cols = sz, cbar cols = 0.25
    widths = []
    cb_idx = 0
    for c in range(ncols):
        widths.append(sz)
        if c in CB_COLS:
            widths.append(0.25)
    fig, axes_all = plt.subplots(
        n_tt, total_cols,
        figsize=(sum(widths), sz * n_tt),
        gridspec_kw={"width_ratios": widths},
        squeeze=False,
    )

    # map data col index → axes_all column index (skip cbar columns)
    col_map = {}
    ac = 0
    for c in range(ncols):
        col_map[c] = ac
        ac += 1
        if c in CB_COLS:
            ac += 1   # skip cbar slot

    for row, tt in enumerate(TRIAL_TYPES):
        panels = list(zip(mats_by_tt[tt], vmaxes, COLS))
        for col, (mat, vmax, title) in enumerate(panels):
            ax = axes_all[row][col_map[col]]
            _heatmap(ax, mat, areas, vmax, title if row == 0 else "", cbar=False)
            if col == 0:
                ax.set_ylabel(tt.replace("_trial", ""), fontsize=8)

    # draw shared colorbars in the dedicated columns (only once, middle row)
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    mid_row = n_tt // 2
    cb_positions = [CB_COLS[0]+1, CB_COLS[1]+2, CB_COLS[2]+3]   # axes_all col indices
    for ac_col, vmax, label in zip(cb_positions, CB_VMAXES, CB_LABELS):
        cax = axes_all[mid_row][ac_col]
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        sm   = cm.ScalarMappable(cmap="RdBu_r", norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(label, fontsize=7)
        cb.ax.tick_params(labelsize=6)
        # hide cbar axes in other rows
        for r in range(n_tt):
            if r != mid_row:
                axes_all[r][ac_col].set_visible(False)

    fig.suptitle(f"NC matrices  |  {gran}  (shared colorscale per group)", y=1.01)
    fig.tight_layout()
    return fig


def _build_bar_df(sessions, gran, tt, valid, areas):
    """Per-mouse mean NC for within-area bars."""
    records = []
    for s in sessions.values():
        gr = s["results"].get(gran, {})
        if tt not in gr: continue
        for wname in WINDOWS:
            df_nc = gr[tt][f"nc_{wname}"]
            for area in areas:
                row = df_nc[(df_nc.area_i == area) & (df_nc.area_j == area)]
                if not row.empty:
                    records.append({"area": area, "window": wname,
                                    "group": s["group"], "mouse_id": s["mouse_id"],
                                    "nc": row["nc"].iloc[0]})
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    return df.groupby(["area", "window", "group", "mouse_id"], as_index=False)["nc"].mean()


def _draw_bars(ax, df, areas, y_shared_max=None):
    """Draw grouped bars with SEM, scatter, and stats onto ax. Returns data y_max."""
    x      = np.arange(len(areas))
    W      = 0.18
    shifts = {"baseline": {0: -3*W/2, 1: -W/2}, "stimulus": {0: W/2, 1: 3*W/2}}
    hatch  = {"baseline": "//", "stimulus": ""}

    for wname in WINDOWS:
        for g in [0, 1]:
            xi  = x + shifts[wname][g]
            sub = df[(df.window == wname) & (df.group == g)]
            ms  = [sub[sub.area == a]["nc"].mean() for a in areas]
            ses = [sub[sub.area == a]["nc"].sem()  for a in areas]
            ax.bar(xi, ms, yerr=ses, width=W, color=COLORS[g], alpha=0.7,
                   hatch=hatch[wname], label=f"{GROUPS[g]} {wname}",
                   error_kw={"elinewidth": 1}, zorder=2)
            for j, area in enumerate(areas):
                vals = sub[sub.area == area]["nc"].dropna().values
                ax.scatter([xi[j]] * len(vals), vals, c=COLORS[g], s=14, zorder=3)

    if y_shared_max is not None:
        ax.set_ylim(top=y_shared_max * 1.25)
    ax.autoscale(axis="x")
    y_top = ax.get_ylim()[1]

    for j, area in enumerate(areas):
        for k, wname in enumerate(WINDOWS):
            a_v = df[(df.window==wname)&(df.group==0)&(df.area==area)]["nc"].dropna().values
            b_v = df[(df.window==wname)&(df.group==1)&(df.area==area)]["nc"].dropna().values
            _, p = mwu(a_v, b_v)
            offset = y_top + k * abs(y_top) * 0.08
            annotate_bracket(ax, x[j]+shifts[wname][0], x[j]+shifts[wname][1], offset, p)

    ax.set_xticks(x)
    ax.set_xticklabels(areas, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean NC (r)", fontsize=8)
    return df["nc"].max()


def plot_all_barplots(sessions, gran, valid):
    """
    Rows = trial_types. Single within-area barplot per row.
    Y-axis shared across all rows. Stats brackets for MWU R+ vs R-.
    """
    areas = ordered_areas(valid, gran)
    if not areas: return plt.figure()

    # collect all data first to set shared y limits
    dfs   = {tt: _build_bar_df(sessions, gran, tt, valid, areas) for tt in TRIAL_TYPES}
    all_nc = pd.concat([d for d in dfs.values() if not d.empty], ignore_index=True)
    if all_nc.empty: return plt.figure()
    y_max = all_nc["nc"].max()
    y_min = all_nc["nc"].min()

    fig, axes = plt.subplots(len(TRIAL_TYPES), 1,
                              figsize=(max(10, len(areas) * 1.4), 5 * len(TRIAL_TYPES)),
                              sharey=True)
    axes = np.array(axes).ravel()

    for ax, tt in zip(axes, TRIAL_TYPES):
        df = dfs[tt]
        if df.empty: continue
        _draw_bars(ax, df, areas, y_shared_max=y_max)
        ax.set_title(tt.replace("_trial", ""), fontsize=9)

        # Wilcoxon base vs stim within group — print
        for g in [0, 1]:
            base = [df[(df.window=="baseline")&(df.group==g)&(df.area==a)]["nc"].mean() for a in areas]
            stim = [df[(df.window=="stimulus")&(df.group==g)&(df.area==a)]["nc"].mean() for a in areas]
            _, p = wilcox(base, stim)
            if p is not None:
                print(f"  Wilcoxon base vs stim  {GROUPS[g]}  {gran}  {tt}  p={p:.4f} {pstar(p)}")

    axes[-1].set_ylim(bottom=y_min * 1.1 if y_min < 0 else y_min * 0.9)
    _dedup_legend(axes[0], fontsize=8)
    fig.suptitle(f"Within-area NC per area  |  {gran}  (shared y-axis)", y=1.005)
    fig.tight_layout()
    return fig


def plot_all_summaries(sessions, gran, valid):
    """
    Rows = trial_types. Cols = within / across.
    Y-axis shared. Stats: MWU R+ vs R- + Wilcoxon base vs stim.
    """
    # collect per-mouse data across all trial types first
    all_records = []
    for tt in TRIAL_TYPES:
        for s in sessions.values():
            gr = s["results"].get(gran, {})
            if tt not in gr: continue
            for wname in WINDOWS:
                dv = gr[tt][f"nc_{wname}"]
                dv = dv[dv.area_i.isin(valid) & dv.area_j.isin(valid)]
                all_records.append({"tt": tt, "window": wname, "group": s["group"],
                                    "mouse_id": s["mouse_id"],
                                    "within": dv[dv.area_i == dv.area_j]["nc"].mean(),
                                    "across": dv[dv.area_i != dv.area_j]["nc"].mean()})
    if not all_records: return plt.figure()
    all_df = pd.DataFrame(all_records).groupby(
        ["tt", "window", "group", "mouse_id"], as_index=False)[["within", "across"]].mean()

    # shared y limits per nc_type
    ylims = {nc: (all_df[nc].min(), all_df[nc].max()) for nc in ["within", "across"]}

    n_tt  = len(TRIAL_TYPES)
    fig, axes = plt.subplots(n_tt, 2, figsize=(10, 4 * n_tt), sharey="col")
    axes = np.array(axes).reshape(n_tt, 2)
    offsets = {0: -0.2, 1: 0.2}
    x_pos   = {w: i for i, w in enumerate(WINDOWS)}

    print(f"\n── Summary stats  {gran} ─────────────────────────────────────")
    for row, tt in enumerate(TRIAL_TYPES):
        df = all_df[all_df.tt == tt]
        for col, nc_type in enumerate(["within", "across"]):
            ax = axes[row][col]
            for g in [0, 1]:
                sub = df[df.group == g]
                for wname in WINDOWS:
                    vals = sub[sub.window == wname][nc_type].dropna().values
                    xi   = x_pos[wname] + offsets[g]
                    ax.bar(xi, np.nanmean(vals),
                           yerr=np.nanstd(vals) / np.sqrt(max(len(vals), 1)),
                           width=0.35, color=COLORS[g], alpha=0.7,
                           error_kw={"elinewidth": 1},
                           label=GROUPS[g] if wname == list(WINDOWS)[0] else "")
                    ax.scatter([xi] * len(vals), vals, c=COLORS[g], s=25, zorder=3)

            # MWU R+ vs R-
            ax.autoscale()
            y_top = ax.get_ylim()[1]
            for k, wname in enumerate(WINDOWS):
                a = df[(df.group==0)&(df.window==wname)][nc_type].dropna().values
                b = df[(df.group==1)&(df.window==wname)][nc_type].dropna().values
                u, p = mwu(a, b)
                if p is not None:
                    print(f"  MWU {GROUPS[0]} vs {GROUPS[1]}  {tt}  {wname:10}  {nc_type:6}"
                          f"  U={u:.0f}  p={p:.4f} {pstar(p)}  n=({len(a)},{len(b)})")
                annotate_bracket(ax, x_pos[wname]+offsets[0], x_pos[wname]+offsets[1],
                                 y_top + k * abs(y_top) * 0.1, p, color="k")

            # Wilcoxon base vs stim within group
            for g in [0, 1]:
                sub  = df[df.group == g]
                base = sub[sub.window == "baseline"][nc_type].dropna().values
                stim = sub[sub.window == "stimulus"][nc_type].dropna().values
                _, p = wilcox(base, stim)
                if p is not None:
                    print(f"  Wilcoxon base vs stim  {GROUPS[g]}  {tt}  {nc_type:6}"
                          f"  p={p:.4f} {pstar(p)}")
                x0 = x_pos["baseline"] + offsets[g]
                x1 = x_pos["stimulus"] + offsets[g]
                annotate_bracket(ax, x0, x1,
                                 max(np.nanmean(base), np.nanmean(stim)) * 1.15,
                                 p, color=COLORS[g])

            ax.set_xticks(list(x_pos.values()))
            ax.set_xticklabels(list(WINDOWS.keys()), fontsize=8)
            ax.set_ylabel("Mean NC (r)", fontsize=8)
            if row == 0:
                ax.set_title(f"{nc_type.capitalize()}-area NC", fontsize=9)
            if col == 0:
                ax.annotate(tt.replace("_trial",""), xy=(-0.25, 0.5),
                            xycoords="axes fraction", rotation=90,
                            va="center", fontsize=9, fontweight="bold")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle(f"NC summary  |  {gran}  (shared y-axis per type)", y=1.005)
    fig.tight_layout()
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, pickle
    ap = argparse.ArgumentParser()
    ap.add_argument("input_folder",  type=Path)
    ap.add_argument("output_folder", type=Path)
    ap.add_argument("--recompute", action="store_true",
                    help="Ignore cached results and reprocess all NWB files")
    args = ap.parse_args()
    fig_dir = args.output_folder / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir = args.output_folder / "results"; res_dir.mkdir(parents=True, exist_ok=True)

    cache = res_dir / "sessions_cache.pkl"

    if not args.recompute and cache.exists():
        print(f"Loading cached results from {cache}")
        with open(cache, "rb") as f:
            sessions = pickle.load(f)
        print(f"  {len(sessions)} sessions loaded.")
    else:
        files = sorted(args.input_folder.glob("*.nwb"))
        sessions = {}
        for path in files:
            try:
                sessions[path] = process(path)
                s = sessions[path]
                print(f"OK   {path.name}  group={GROUPS[s['group']]}  mouse={s['mouse_id']}")
            except Exception as e:
                print(f"SKIP {path.name}: {e}")
        with open(cache, "wb") as f:
            pickle.dump(sessions, f)
        print(f"Results cached to {cache}")

    def save(fig, subfolder, name):
        d = fig_dir / subfolder; d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_results(sessions, gran, tt, valid):
        rows = []
        for path, s in sessions.items():
            gr = s["results"].get(gran, {})
            if tt not in gr: continue
            for wname in WINDOWS:
                df = gr[tt][f"nc_{wname}"].copy()
                df["window"]   = wname
                df["session"]  = path.stem
                df["mouse_id"] = s["mouse_id"]
                df["group"]    = GROUPS[s["group"]]
                rows.append(df)
        if rows:
            out = pd.concat(rows, ignore_index=True)
            out = out[out.area_i.isin(valid) & out.area_j.isin(valid)]
            d   = res_dir / gran; d.mkdir(parents=True, exist_ok=True)
            out.to_csv(d / f"nc_{tt}.csv", index=False)

    for gran in ["fine", "global"]:
        valid = valid_areas(sessions, gran)
        print(f"\n{gran.upper()} — {len(valid)} valid areas: {sorted(valid)}")
        print_analysis_summary(sessions, gran, valid)
        for tt in TRIAL_TYPES:
            tt_short = tt.replace("_trial", "")
            save_results(sessions, gran, tt, valid)
            # ── raw / diagnostic
            save(plot_raw_nc_distributions(sessions, gran, tt, valid),   f"{gran}/raw/{tt_short}", "dist")
            save(plot_example_nc_computation(sessions, gran, tt, valid), f"{gran}/raw/{tt_short}", "example")
            save(plot_nc_scatter(sessions, gran, tt, valid),             f"{gran}/raw/{tt_short}", "scatter")
            # ── time courses (per trial type)
            save(plot_timecourse(sessions, gran, tt, valid),             f"{gran}/timecourse", tt_short)
            save(plot_tc_per_area(sessions, gran, tt, valid),            f"{gran}/timecourse", f"{tt_short}_per_area")
        # ── consolidated across trial types
        save(plot_all_timecourses(sessions, gran, valid),   f"{gran}/timecourse",  "all_timecourses")
        save(plot_all_tc_per_area(sessions, gran, valid),   f"{gran}/timecourse",  "all_tc_per_area")
        for area in ordered_areas(valid, gran):
            save(plot_tc_single_area(sessions, gran, area), f"{gran}/timecourse/per_area", area)
        save(plot_all_matrices(sessions, gran, valid),      f"{gran}/matrices",    "all_matrices")
        save(plot_all_barplots(sessions, gran, valid),      f"{gran}/barplots",    "all_barplots")
        save(plot_all_summaries(sessions, gran, valid),     f"{gran}/summaries",   "all_summaries")
