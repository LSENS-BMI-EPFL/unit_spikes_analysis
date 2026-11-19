"""
population_matrix_summary.py — extended fig5_population_matrix for rastermap_psth outputs.

Extends run_clustering_new.fig5_population_matrix (same matrix drawing via
rastermap_utils._draw_matrix, same anatomy colormaps via build_anatomy_cmaps, same
stacked-proportion column style as _draw_prop_column) with a user-selectable list of
colour-coded side columns:

    anatomical      avg_ipsi | cc_tc_ct_iterated | cc_hierarchy_score_columns
    categorical     waveform_type | area_group | area_acronym_custom
    metacluster     metacluster (family colour strip, run_meta_clustering definition)
    ROC categories  roc:<category>   (fraction significant or mean selectivity, see ROC_CATEGORIES)
    reward group    reward_fmk       (mouse-level F_mk share R+ vs R-, placed last)
                    or reward_panel="figS4": F_mk mean ± SEM + significance lollipop
                    (columns 1-2 of rastermap_utils._fig_cluster_stats_matrix), far right.

Options
-------
    row_mode    "neuron" | "cluster_mean" | "cluster_mean_uniform"
                  neuron:               one row per neuron (as fig5_population_matrix_cv)
                  cluster_mean:         each fine rastermap cluster's mean PSTH, row height ∝ n neurons
                  cluster_mean_uniform: one row per cluster
    order       "rastermap" | "metacluster"
                  metacluster: clusters reordered by the meta-clustering dendrogram leaves,
                  dendrogram drawn right of the matrix
    roc_mode    "fraction" | "selectivity"
    reward_panel "fmk_share" | "figS4" | "neuron" | None

ROC categories (day-0 ROC, joined on mouse_id/session_id/electrode_group/cluster_id;
all non-'baseline_*' ROC types use single-trial baseline-corrected counts (roc_utils_new._get_counts,
apply_baseline=True: count - baseline[-1.0,-0.015 s] scaled to the window);
NaN AUC -> not significant and excluded from selectivity means; selectivity = 2*(AUC-0.5),
+ = second condition of the pair higher; choice types sign-flipped so + = hit > miss,
matching the direction fix in unit_spikes_analysis.py). Significance counted in either direction.

    whisker_sensory     any of whisker_passive_pre, whisker_passive_post, whisker_active
                        (post- vs pre-stim window)                           + = evoked increase
    auditory_sensory    any of auditory_passive_pre, auditory_passive_post, auditory_active
    modality_pref       any of wh_vs_aud_active, wh_vs_aud_passive_pre, wh_vs_aud_passive_post
                        (whisker vs auditory post-stim); preferred modality = sign of mean
                        selectivity over the significant epochs            + = auditory-pref.
    lick_responsive     motor (lick-related), def. C: spontaneous_licks ([0, 0.2] s vs [-0.4, -0.2] s) AND
                        spontaneous_licks_vs_cr (lick vs correct rejection), both significant, same sign
    learning_modulated  any of whisker_pre_vs_post_learning, auditory_pre_vs_post_learning
                        (baseline-corrected evoked response, passive_post vs passive_pre) + = larger after learning
    decision            per modality, never crossed:
                        (whisker_choice AND whisker_hit_vs_spontaneous) OR
                        (auditory_choice AND auditory_hit_vs_spontaneous)
                        (hit vs miss, choice sign flipped: + = hit > miss; hit vs spontaneous lick)
    gated_decision      per modality, as roc_utils_new.compute_neuron_labels:
                        (whisker decision AND NOT whisker_sensory AND NOT spontaneous_licks_vs_cr) OR
                        (auditory decision AND NOT auditory_sensory AND NOT spontaneous_licks_vs_cr)
                        [M_sensory = M miss vs CR]; column = gated neurons (either modality) as a fraction of
                        the cluster's DECISION neurons (either modality); grey if < GATED_MIN_DECISION
    fraction-mode colour scale: per column, capped at the 95th pct of cluster fractions (rounded up to
    5 %); clusters above the cap are drawn at full colour (colorbar overflow arrow)

Entry point: run_matrix_summary(method_dir, config_path, cfg_overrides, roc_root)
Outputs:     <method_dir>/matrix_summary/
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, kendalltau

import rastermap_psth.rastermap_utils as rmu
from rastermap_psth.rastermap_utils import load_cfg, build_anatomy_cmaps, order_area_groups
from rastermap_psth import cluster_postproc_analyses as cpa
import ephys_utilities.allen_utils.allen_utils as allen_utils
from ephys_utilities.plotting_utils.plotting_utils import GROUP_COLORS

KEY = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
DEFAULT_ROC_ROOT = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
# ROC results folder inside <mouse>/whisker_0/: "roc_analysis_rates" = rerun with rates (per-window spike counts /
# window length), needed because some ROC types compare windows of different lengths (024_roc_rates.py)
ROC_SUBDIR = "roc_analysis"   # TODO: switch to "roc_analysis_rates" once 024_roc_rates.py has finished

ROC_CATEGORIES = {
    # stimulus-responsive: any whisker or auditory, passive (pre/post) or active, evoked response
    "stim_responsive": dict(types=["whisker_passive_pre", "whisker_passive_post", "whisker_active",
                                   "auditory_passive_pre", "auditory_passive_post", "auditory_active"], rule="any",
                            label="Stimulus-responsive"),
    "whisker_sensory": dict(types=["whisker_passive_pre", "whisker_passive_post", "whisker_active"], rule="any",
                            label="Whisker sensory"),
    "auditory_sensory": dict(types=["auditory_passive_pre", "auditory_passive_post", "auditory_active"], rule="any",
                             label="Auditory sensory"),
    "modality_pref": dict(types=["wh_vs_aud_active", "wh_vs_aud_passive_pre", "wh_vs_aud_passive_post"], rule="any",
                          split=True, label="Modality pref."),
    # motor (lick-related), definition C: lick vs pre-lick baseline AND lick vs correct-rejection,
    # both significant and in the same direction (spontaneous_licks: + = post-lick > pre-lick;
    # spontaneous_licks_vs_cr: + = spontaneous lick > CR)
    "lick_responsive": dict(types=["spontaneous_licks", "spontaneous_licks_vs_cr"], rule="all_same_sign",
                            label="Motor (lick-related)"),
    "learning_modulated": dict(types=["whisker_pre_vs_post_learning", "auditory_pre_vs_post_learning"], rule="any",
                               label="Learning-modulated"),
    # decision: (hit vs miss, either modality) AND (hit vs spontaneous lick, either modality)
    # decision, per modality (never crossed): (whisker_choice AND whisker_hit_vs_spontaneous)
    #                                          OR (auditory_choice AND auditory_hit_vs_spontaneous)
    "decision": dict(types=["whisker_choice", "whisker_hit_vs_spontaneous", "auditory_choice",
                            "auditory_hit_vs_spontaneous"], rule="per_modality",
                     modalities=[(["whisker_choice", "whisker_hit_vs_spontaneous"], []),
                                 (["auditory_choice", "auditory_hit_vs_spontaneous"], [])],
                     label="Decision"),
    # gated, per modality (as roc_utils_new.compute_neuron_labels): M_decision AND NOT M_sensory (miss vs CR)
    # AND NOT spontaneous_licks_vs_cr; a neuron is gated if gated in either modality
    "gated_decision": dict(types=["whisker_choice", "whisker_hit_vs_spontaneous", "auditory_choice",
                                  "auditory_hit_vs_spontaneous"], rule="per_modality",
                           modalities=[(["whisker_choice", "whisker_hit_vs_spontaneous"],
                                        ["whisker_sensory", "spontaneous_licks_vs_cr"]),
                                       (["auditory_choice", "auditory_hit_vs_spontaneous"],
                                        ["auditory_sensory", "spontaneous_licks_vs_cr"])],
                           label="Gated decision"),
}
ANATOMY_COLS = ["avg_ipsi", "cc_tc_ct_iterated", "cc_hierarchy_score_columns"]
ANATOMY_TITLES = {"avg_ipsi": "wS1/2 proj. (Liu '24)", "cc_tc_ct_iterated": "Hierarchy (Harris '19)",
                  "cc_hierarchy_score_columns": "Hierarchy (Gao '26)"}
CATEGORICAL_COLS = ["waveform_type", "area_group", "area_acronym_custom"]
# metacluster is always drawn as a strip left of the matrix (next to the dendrogram), not as a column
# default ROC columns (user, 2026-09-27): modality preference, motor (lick-related), decision, gated
DEFAULT_ROC = ["stim_responsive", "modality_pref", "lick_responsive", "decision", "gated_decision"]
DEFAULT_COLUMNS = ANATOMY_COLS + ["area_group"] + [f"roc:{c}" for c in DEFAULT_ROC] + ["reward_fmk"]


def column_range(v, step=0.05, floor0=True):
    """Per-column colour range emphasising that column's own spread: [5th, 95th] percentile of the
    cluster values, rounded outwards to `step`; returns (lo, hi, extend)."""
    v = np.asarray([x for x in v if np.isfinite(x)])
    if not len(v):
        return 0.0, 1.0, "neither"
    lo = np.floor(np.percentile(v, 5) / step) * step
    hi = np.ceil(np.percentile(v, 95) / step) * step
    lo = max(0.0, lo) if floor0 else lo
    if hi <= lo:
        hi = lo + step
    ext = {(False, False): "neither", (True, False): "min", (False, True): "max", (True, True): "both"}[
        (bool(v.min() < lo), bool(v.max() > hi))]
    return float(lo), float(hi), ext
MODALITY_COLORS = {"whisker": "#D4A017", "auditory": "#1f4fbf", "none": "#c8c8c8"}
WAVEFORM_COLORS = {"NW": "#83b1ff", "WW": "#ff8783"}   # as fig5_population_matrix
# fraction mode: one hue per ROC category (white -> hue), all on one shared 0..max scale
ROC_COLORS = {"stim_responsive": "#2e6e8e", "whisker_sensory": "#c98a00", "auditory_sensory": "#1f4fbf", "lick_responsive": "#7b3fa0",
              "learning_modulated": "#0f8a7a", "decision": "#c0392b", "gated_decision": "#7a1030"}
GATED_MIN_DECISION = 10  # gated column: fraction of a cluster's DECISION neurons; NaN (grey) if fewer decision neurons
SEL_COLOR = "#3b0f70"   # selectivity mode: shared white -> SEL_COLOR map for |selectivity| columns
CBAR_W, CBAR_H = 0.011, 0.004   # every small colorbar has this size (figure fraction)


def white_to(color, name=None):
    return mcolors.LinearSegmentedColormap.from_list(name or f"w2{color}", ["#ffffff", color])


def _srgb_to_lab(rgb):
    """sRGB (0-1, N x 3) -> CIELAB (D65)."""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.column_stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])])


_PALETTE_CACHE = {}


def distinct_colors(n):
    """n maximally distinct colours (greedy farthest-point sampling in CIELAB over an RGB grid,
    lightness 35-80 and chroma > 25 so no near-white / near-black / grey). Each new colour is the
    candidate farthest from all colours already chosen, so consecutive colours contrast strongly.
    Deterministic; glasbey-style without extra dependencies."""
    if n in _PALETTE_CACHE:
        return _PALETTE_CACHE[n]
    g = np.linspace(0, 1, 18)
    rgb = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    lab = _srgb_to_lab(rgb)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    keep = (lab[:, 0] > 35) & (lab[:, 0] < 80) & (chroma > 25)
    rgb, lab = rgb[keep], lab[keep]
    first = np.argmin(np.linalg.norm(rgb - np.array([0.12, 0.47, 0.71]), axis=1))   # start at a mid blue
    chosen = [first]
    dmin = np.linalg.norm(lab - lab[first], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(dmin))
        chosen.append(i)
        dmin = np.minimum(dmin, np.linalg.norm(lab - lab[i], axis=1))
    out = [mcolors.to_hex(rgb[i]) for i in chosen]
    _PALETTE_CACHE[n] = out
    return out


def family_palette(families, display_order=None):
    """Distinct colour per metacluster / segment, shared by the dendrogram links and the strip.
    Colours are handed out in DISPLAY order (first appearance along the plotted rows), so families
    that sit next to each other in the figure get maximally contrasting colours."""
    ids = list(dict.fromkeys(display_order)) if display_order is not None else sorted(set(families))
    ids += [f for f in sorted(set(families)) if f not in ids]
    return dict(zip(ids, distinct_colors(len(ids))))


# ══════════════════════════════════════════════════════════════════════════
# data loading
# ══════════════════════════════════════════════════════════════════════════

def find_data_folder(method_dir: Path) -> Path:
    """<data_folder>/../clustering/<run>/rastermap -> the sibling folder holding feature_matrix.npz."""
    cands = [p.parent for p in method_dir.parents[2].glob("*/feature_matrix.npz")]
    assert len(cands) == 1, f"expected one feature_matrix.npz next to {method_dir.parents[1]}, got {cands}"
    return cands[0]


def load_rastermap(method_dir: Path):
    d = np.load(method_dir / "rastermap_results_cv.npz", allow_pickle=True)
    fm = np.load(find_data_folder(method_dir) / "feature_matrix.npz", allow_pickle=True)
    assert np.array_equal(d["unit_ids"], fm["unit_ids"]), "unit order differs between CV results and feature matrix"
    n = len(d["unit_ids"])
    get = lambda k, fill: fm[k] if k in fm.files else np.array([fill] * n)
    return dict(
        X=d["X"], X_odd=fm["X_odd"], isort=d["isort"], labels=d["cluster_labels"].astype(int),
        unit_ids=d["unit_ids"], n_bins_list=list(d["n_bins_list"]),
        t_ctrs=[d[f"t_ctr_{i}"] for i in range(int(d["n_conds"]))],
        mouse_arr=d["mouse_arr"].astype(str), reward_arr=d["reward_arr"].astype(str),
        area_arr=get("area_arr", "unknown").astype(str), waveform_arr=get("waveform_arr", "unknown").astype(str),
        area_group_arr=get("area_group_arr", "Other").astype(str),
        anatomy={c: (fm[k].astype(float) if k in fm.files else np.full(n, np.nan))
                 for c, k in zip(ANATOMY_COLS, ["axon_arr", "harris_arr", "gao_arr"])},
    )


def build_roc_categories(method_dir: Path, unit_ids, roc_root: Path = DEFAULT_ROC_ROOT) -> pd.DataFrame:
    """Per-neuron ROC category table, rows aligned to unit_ids.
    Columns: <cat>__sig (bool, NaN if no ROC match), <cat>__sel (float); modality_pref__whisker,
    modality_pref__auditory (bool); roc_matched."""
    neurons = pd.read_csv(method_dir / "neuron_cluster_labels_cv.csv").rename(columns={"unit_id": "unit_ids"})
    neurons["cluster_id"] = neurons["cluster_id"].astype(str)
    frames = []
    for (m, s), _ in neurons.groupby(["mouse_id", "session_id"]):
        f = Path(roc_root) / m / "whisker_0" / ROC_SUBDIR / f"{m}_roc_results_new.csv"
        if f.exists():
            r = pd.read_csv(f, usecols=KEY + ["analysis_type", "auc", "selectivity", "significant"])
            frames.append(r[r.session_id == s])
        else:
            print(f"  [warn] no ROC file for {m}")
    roc = pd.concat(frames, ignore_index=True)
    roc["cluster_id"] = roc["cluster_id"].astype(str)
    roc["significant"] = roc["significant"].astype(bool) & roc["auc"].notna()
    roc.loc[roc.auc.isna(), "selectivity"] = np.nan
    choice = roc.analysis_type.str.contains("choice")
    roc.loc[choice, "selectivity"] = -roc.loc[choice, "selectivity"]
    assert not roc.duplicated(KEY + ["analysis_type"]).any(), "duplicated ROC rows on unit key"

    sig = roc.pivot_table(index=KEY, columns="analysis_type", values="significant", aggfunc="first")
    sel = roc.pivot_table(index=KEY, columns="analysis_type", values="selectivity", aggfunc="first", dropna=False)
    base = neurons[["unit_ids"] + KEY].set_index(KEY)
    sig = sig.reindex(base.index)
    sel = sel.reindex(base.index)
    matched = sig.notna().any(axis=1).values
    col = lambda df, t, fill: df[t].values if t in df.columns else np.full(len(df), fill)

    out = pd.DataFrame({"unit_ids": base.unit_ids.values, "roc_matched": matched})
    for cat, spec in ROC_CATEGORIES.items():
        types = spec["types"] if "types" in spec else [t for g in spec["groups"] for t in g]
        tobool = lambda t: np.nan_to_num(col(sig, t, False).astype(float), nan=0).astype(bool)
        S = np.column_stack([tobool(t) for t in types])
        V = np.column_stack([col(sel, t, np.nan) for t in types]).astype(float)
        if spec["rule"] == "groups":   # AND over groups of (OR within group)
            is_sig = np.all([np.any([tobool(t) for t in g], axis=0) for g in spec["groups"]], axis=0)
        elif spec["rule"] == "per_modality":
            # OR over modalities of (AND of that modality's required types AND NOT its excluded types):
            # e.g. decision = (wh_choice & wh_hit_vs_spont) | (aud_choice & aud_hit_vs_spont); never crossed
            per_mod = []
            for req, excl in spec["modalities"]:
                m = np.all([tobool(t) for t in req], axis=0)
                for t in excl:
                    m &= ~tobool(t)
                per_mod.append(m)
                out[f"{cat}__{req[0].split('_')[0]}"] = np.where(matched, m, np.nan)   # e.g. decision__whisker
            is_sig = np.any(per_mod, axis=0)
        else:
            is_sig = S.any(1) if spec["rule"] == "any" else S.all(1)
            if spec["rule"] == "all_same_sign":   # all significant AND selectivities share one sign
                with np.errstate(invalid="ignore"):
                    sg = np.sign(V)
                    is_sig = S.all(1) & (np.all(sg > 0, axis=1) | np.all(sg < 0, axis=1))
        for t in spec.get("exclude", []):
            is_sig &= ~np.nan_to_num(col(sig, t, False).astype(float), nan=0).astype(bool)
        out[f"{cat}__sig"] = np.where(matched, is_sig, np.nan)
        with np.errstate(invalid="ignore"):
            out[f"{cat}__sel"] = np.nanmean(V, axis=1) if V.shape[1] > 1 else V[:, 0]      # signed
            out[f"{cat}__abssel"] = np.nanmean(np.abs(V), axis=1)                           # both directions
        if spec.get("split"):
            with np.errstate(invalid="ignore"):
                sig_mean = np.nanmean(np.where(S, V, np.nan), axis=1)  # mean selectivity over significant epochs
            out[f"{cat}__auditory"] = np.where(matched, is_sig & (sig_mean > 0), np.nan)
            out[f"{cat}__whisker"] = np.where(matched, is_sig & (sig_mean < 0), np.nan)
            out[f"{cat}__none"] = np.where(matched, ~is_sig, np.nan)                       # non-preferring
    out = out.set_index("unit_ids").reindex(unit_ids).reset_index()
    return out


META_METHOD = "contiguous"   # default metaclustering (user, 2026-09-27): contiguity-constrained agglomerative
META_CUT = 0.40              # default distance threshold (1 - Pearson r) for the contiguous metaclustering


def _cluster_means(method_dir: Path):
    """run_meta_clustering's input: cluster-mean PSTHs over full-trial neuron PSTHs, rows = clusters."""
    cluster_df = cpa.load_cluster_table(method_dir)
    unit_ids, cond_labels, psth, t_ctr = cpa.load_psth_npz(method_dir)
    X_full, sub_df, _ = cpa._build_neuron_matrix(cluster_df, cond_labels, psth, unit_ids)
    clusters = np.sort(sub_df["cluster_label"].unique())
    means = np.vstack([X_full[sub_df["cluster_label"].values == c].mean(0) for c in clusters])
    return clusters, means


def contiguous_metaclustering(means, cut=META_CUT):
    """Contiguity-constrained agglomerative clustering (sklearn): average linkage on 1 - Pearson r
    (cosine on row-centred cluster means), merges allowed only between clusters adjacent in rastermap
    order (chain connectivity), cut at `cut`. Returns (families 1..F in rastermap order, scipy linkage Z
    whose dendrogram leaf order equals the rastermap order)."""
    from scipy.sparse import diags
    from sklearn.cluster import AgglomerativeClustering
    K = len(means)
    C = means - means.mean(1, keepdims=True)
    conn = diags([np.ones(K - 1), np.ones(K - 1)], [-1, 1], shape=(K, K))
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=cut, metric="cosine", linkage="average",
                                    connectivity=conn, compute_full_tree=True, compute_distances=True).fit(C)
    order = list(dict.fromkeys(model.labels_))
    fam = np.array([order.index(l) + 1 for l in model.labels_])
    assert np.all(np.diff(fam) >= 0), "contiguous metaclusters are not contiguous in rastermap order"
    lo, cnt, Z = {i: i for i in range(K)}, {i: 1 for i in range(K)}, []
    for i, (a, b) in enumerate(model.children_):
        a, b = int(a), int(b)
        if lo[b] < lo[a]:
            a, b = b, a
        lo[K + i], cnt[K + i] = lo[a], cnt[a] + cnt[b]
        Z.append([a, b, float(model.distances_[i]), cnt[K + i]])
    Z = np.array(Z, float)
    assert dendrogram(Z, no_plot=True)["leaves"] == list(range(K)), "leaf order differs from rastermap order"
    return fam, Z


def metaclustering(method_dir: Path, method=None, cut=None):
    """Metaclusters of the rastermap clusters.
    method="contiguous" (default): contiguity-constrained agglomerative clustering (families are contiguous
        segments of the rastermap axis; dendrogram leaves in rastermap order).
    method="hierarchical": run_meta_clustering exactly (unconstrained average linkage on 1 - r, cut 0.35),
        checked against the saved cluster_families.csv if present."""
    method = method or META_METHOD
    clusters, means = _cluster_means(method_dir)
    if method == "contiguous":
        assert np.array_equal(clusters, np.arange(len(clusters))), "cluster labels must be 0..K-1 in rastermap order"
        cut = META_CUT if cut is None else cut
        fam, Z = contiguous_metaclustering(means, cut)
        return dict(Z=Z, clusters=clusters, families=dict(zip(clusters, fam)), cut=cut,
                    method="contiguous (sklearn agglomerative, average, 1-r, chain connectivity)")
    cfg = dict(cpa.DEFAULT_CFG)
    condensed = np.clip(pdist(means, metric="correlation"), 0, None)
    Z = linkage(condensed, method=cfg["meta_linkage_method"])
    fam = fcluster(Z, t=cfg["meta_dendro_cut_dist"], criterion="distance")
    saved = sorted(method_dir.glob("analyses/analyses_*/02_meta_clustering/cluster_families.csv"),
                   key=lambda p: p.stat().st_mtime)
    if saved:
        sv = pd.read_csv(saved[-1]).set_index("cluster_label").family.reindex(clusters).values
        assert np.array_equal(sv, fam), f"recomputed families differ from {saved[-1]}"
    return dict(Z=Z, clusters=clusters, families=dict(zip(clusters, fam)),
                cut=cfg["meta_dendro_cut_dist"], method=cfg["meta_linkage_method"])


def leaf_order(meta):
    """Dendrogram leaf order (cluster ids, top → bottom), globally flipped if needed so it runs
    in the same direction as the rastermap order (a left-right flip of the whole tree is a free
    choice in a dendrogram, so the raw orientation is arbitrary). Returns (order, flipped)."""
    leaves = [meta["clusters"][i] for i in dendrogram(meta["Z"], no_plot=True)["leaves"]]
    rho = spearmanr(np.arange(len(leaves)), np.argsort(leaves))[0]  # leaf position vs cluster index
    return (leaves[::-1], True) if rho < 0 else (leaves, False)


def load_fmk_stats(method_dir: Path):
    fz = np.load(method_dir / "stats" / "f_matrix.npz", allow_pickle=True)
    st = pd.read_csv(method_dir / "stats" / "per_cluster_stats.csv")  # cluster is 1-based
    import re
    txt = (method_dir / "stats" / "permanova_summary.txt").read_text()
    F = re.search(r"F-statistic\s*:\s*([-\d.eE]+)", txt); p = re.search(r"p-value\s*:\s*([-\d.eE]+)", txt)
    both = None
    fb = method_dir / "stats" / "cluster_mean_matrix_fmk_stats.csv"   # MWU + Welch (BH-FDR) per cluster
    if fb.exists():
        b = pd.read_csv(fb).sort_values("cluster")
        both = ((b.p_mannwhitney_fdr < 0.05) & (b.p_welch_t_fdr < 0.05)).values
    out = dict(F=fz["f_matrix"], groups=fz["reward_groups"].astype(str), sig_both=both if both is not None
               else st.significant.values.astype(bool),
               p_fdr=st.p_fdr.values, reject=st.significant.values.astype(bool),
               permanova_F=float(F.group(1)) if F else np.nan, permanova_p=float(p.group(1)) if p else np.nan)
    # optional learning vs expert stage stats (session-level F_sk; written by the learning+expert variant script)
    fs, cs = method_dir / "stats" / "stage_f_matrix.npz", method_dir / "stats" / "stage_cluster_stats.csv"
    if fs.exists() and cs.exists():
        sz = np.load(fs, allow_pickle=True)
        sc = pd.read_csv(cs).sort_values("cluster")
        out["stage"] = dict(F=sz["f_matrix"], groups=sz["stage"].astype(str),
                            sig=((sc.p_mannwhitney_fdr < 0.05) & (sc.p_welch_t_fdr < 0.05)).values)
    return out


# ══════════════════════════════════════════════════════════════════════════
# row layout
# ══════════════════════════════════════════════════════════════════════════

def row_layout(R, cluster_order, row_mode):
    """Returns (matrix, blocks) with blocks = [(lo, hi, cluster_k)] in row coordinates,
    neuron_rows = neuron index per row (neuron mode) or None."""
    X, isort, labels = R["X"], R["isort"], R["labels"]
    rows, blocks, lo = [], [], 0
    neuron_rows = []
    for k in cluster_order:
        idx = isort[labels[isort] == k]          # neurons of cluster k, rastermap order
        if row_mode == "neuron":
            rows.append(X[idx]); neuron_rows.append(idx); h = len(idx)
        elif row_mode == "cluster_mean":
            rows.append(np.repeat(X[idx].mean(0, keepdims=True), len(idx), 0)); h = len(idx)
        elif row_mode == "cluster_mean_uniform":
            rows.append(X[idx].mean(0, keepdims=True)); h = 1
        else:
            raise ValueError(row_mode)
        blocks.append((lo, lo + h, k)); lo += h
    return np.vstack(rows), blocks, (np.concatenate(neuron_rows) if neuron_rows else None)


# ══════════════════════════════════════════════════════════════════════════
# column drawers (same visual language as fig5_population_matrix)
# ══════════════════════════════════════════════════════════════════════════

def _style_col(ax, n_rows, title, fontsize=9):
    for s in ("top", "left", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(n_rows, 0); ax.set_xticks([]); ax.set_yticks([])
    col_title(ax, title, fontsize)


def col_title(ax, title, fontsize=9):
    """Column title slightly rotated (reads better than vertical text)."""
    ax.text(0.5, 1.006, title, rotation=45, ha="left", va="bottom", fontsize=fontsize,
            transform=ax.transAxes, rotation_mode="anchor")


def draw_prop_col(ax, blocks, props, categories, colors, n_rows, title):
    """props: {k: {cat: proportion}} — stacked bar per cluster (as _draw_prop_column)."""
    for lo, hi, k in blocks:
        left = 0.0
        for cat in categories:
            p = props.get(k, {}).get(cat, 0)
            if p > 0:
                ax.barh((lo + hi) / 2, p, left=left, height=(hi - lo) * 0.92, color=colors.get(cat, "#aaaaaa"),
                        edgecolor="none", align="center")
                left += p
    ax.set_xlim(0, 1)
    _style_col(ax, n_rows, title)


NAN_ROC_GREY = "#b5b5b5"   # "not enough data" in ROC columns (clearly distinct from white = 0)


def draw_scalar_col(ax, blocks, vals, cmap, norm, n_rows, title, nan_color="#eeeeee"):
    for lo, hi, k in blocks:
        v = vals.get(k, np.nan)
        ax.barh((lo + hi) / 2, 1.0, left=0, height=(hi - lo) * 0.92, align="center", edgecolor="none",
                color=nan_color if not np.isfinite(v) else cmap(norm(v)))
    ax.set_xlim(0, 1)
    _style_col(ax, n_rows, title)


def draw_family_strip(ax, blocks, families, palette, n_rows, label="Metacluster"):
    for lo, hi, k in blocks:
        ax.barh((lo + hi) / 2, 1.0, left=0, height=(hi - lo), align="center", edgecolor="none",
                color=palette[families[k]])
    ax.set_xlim(0, 1)
    for s in ("top", "left", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(n_rows, 0); ax.set_xticks([]); ax.set_yticks([])
    col_title(ax, label)


def draw_dendrogram(ax, meta, blocks, n_rows, palette, flipped=False, label="Metacluster dendrogram"):
    """Metaclustering dendrogram on the LEFT of the matrix (root on the left, leaves touching
    the metacluster strip), links coloured by their metacluster (unique colour per family);
    links spanning several families are grey."""
    fam = meta["families"]
    leaf_fam = [fam[c] for c in meta["clusters"]]
    lcf = cpa.build_link_color_func(meta["Z"], leaf_fam, palette, default_color="#9a9a9a")
    dn = dendrogram(meta["Z"], no_plot=True, link_color_func=lcf)
    centers = np.array([(lo + hi) / 2 for lo, hi, _ in blocks])
    if flipped:  # blocks follow the flipped leaf order: leaf i sits at block K-1-i
        centers = centers[::-1]
    for ic, dc, c in zip(dn["icoord"], dn["dcoord"], dn["color_list"]):
        y = np.interp((np.asarray(ic) - 5) / 10, np.arange(len(centers)), centers)
        ax.plot(dc, y, color=c, lw=0.6)
    ax.axvline(meta["cut"], color="k", ls="--", lw=0.5, alpha=0.6)
    dmax = max(max(dc) for dc in dn["dcoord"])
    ax.set_ylim(n_rows, 0); ax.set_xlim(dmax * 1.03, 0)   # root on the left, leaves at x=0 (right edge)
    ax.set_yticks([]); ax.tick_params(left=False, labelleft=False)
    for s in ("top", "left", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("1 − r", fontsize=9)
    col_title(ax, label)


def draw_figS4_panels(ax_f, ax_p, blocks, stats, n_rows):
    """Columns 1-2 of rastermap_utils._fig_cluster_stats_matrix, aligned to the blocks."""
    F, g = stats["F"], stats["groups"]
    rp, rm = g == "R+", g == "R-"
    y = {k: (lo + hi) / 2 for lo, hi, k in blocks}
    dy = max(n_rows / len(blocks) * 0.18, 0.18)
    ks = [k for _, _, k in blocks]
    yy = np.array([y[k] for k in ks])
    mean_rp, mean_rm = F[rp][:, ks].mean(0), F[rm][:, ks].mean(0)
    sem_rp = F[rp][:, ks].std(0) / np.sqrt(max(rp.sum(), 1))
    sem_rm = F[rm][:, ks].std(0) / np.sqrt(max(rm.sum(), 1))
    ax_f.errorbar(mean_rp, yy - dy, xerr=sem_rp, fmt="o", color=GROUP_COLORS["rplus"], ms=2, lw=0.6, elinewidth=0.6)
    ax_f.errorbar(mean_rm, yy + dy, xerr=sem_rm, fmt="o", color=GROUP_COLORS["rminus"], ms=2, lw=0.6, elinewidth=0.6)
    ax_f.set_ylim(n_rows, 0); ax_f.set_yticks([])
    ax_f.spines[["top", "right", "left"]].set_visible(False)
    ax_f.set_xlabel(r"$f_{m,k}$ mean ± SEM", fontsize=9); ax_f.tick_params(labelsize=7)
    pF, pp = stats.get("permanova_F", np.nan), stats.get("permanova_p", np.nan)
    col_title(ax_f, f"Fractional rep. (mouse-level)\nPERMANOVA F={pF:.2f}, p={pp:.3g}")
    nlp = -np.log10(np.clip(stats["p_fdr"][ks], 1e-10, 1))
    rej = stats["reject"][ks]
    cols = [GROUP_COLORS["rplus"] if (r and a > b) else GROUP_COLORS["rminus"] if r else "darkgrey"
            for r, a, b in zip(rej, mean_rp, mean_rm)]
    ax_p.hlines(yy, 0, nlp, color=cols, lw=0.8)
    ax_p.scatter(nlp, yy, s=6, color=cols, edgecolors="none", zorder=3)
    ax_p.axvline(-np.log10(0.05), color="k", ls="--", lw=0.6, alpha=0.6)
    ax_p.set_ylim(n_rows, 0); ax_p.set_yticks([])
    ax_p.spines[["top", "right", "left"]].set_visible(False)
    ax_p.set_xlabel("−log₁₀ p (FDR)", fontsize=9); ax_p.tick_params(labelsize=7)
    col_title(ax_p, "Post-hoc MWU")


def small_cbar(fig, ax, cmap, norm, ticks, ticklabels, y_off=0.045, extend="neither"):
    """Uniform small horizontal colorbar centred under a column (same size everywhere)."""
    pos = ax.get_position()
    cx = pos.x0 + pos.width / 2
    cax = fig.add_axes([cx - CBAR_W / 2, pos.y0 - y_off, CBAR_W, CBAR_H])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      extend=extend, extendfrac=0.25)
    cb.set_ticks(ticks); cb.set_ticklabels(ticklabels)
    cb.ax.tick_params(labelsize=5.5, length=1.2, pad=1, labelrotation=90)
    cb.outline.set_linewidth(0.3)
    return cb


def compact_legend(fig, ax, title, entries, anchor="center", ncol=1, y_off=0.14):
    """Ultra-compact legend for one column group, placed under that column."""
    pos = ax.get_position()
    x = {"center": pos.x0 + pos.width / 2, "right": pos.x1, "left": pos.x0}[anchor]
    loc = {"center": "upper center", "right": "upper right", "left": "upper left"}[anchor]
    handles = [Patch(facecolor=c, edgecolor="none", label=l) for l, c in entries.items()]
    lg = fig.legend(handles=handles, title=title, loc=loc, bbox_to_anchor=(x, pos.y0 - y_off), ncol=ncol,
                    fontsize=7, title_fontsize=8, frameon=False, handlelength=0.9, handleheight=0.9,
                    handletextpad=0.3, labelspacing=0.15, columnspacing=0.6, borderaxespad=0)
    lg._legend_box.align = "left"
    return lg


# ══════════════════════════════════════════════════════════════════════════
# per-cluster aggregates
# ══════════════════════════════════════════════════════════════════════════

def cluster_aggregates(R, roc, K):
    labels = R["labels"]
    agg = dict(props={}, scalars={}, frac={}, sel={}, abssel={})
    for name, arr in [("waveform_type", R["waveform_arr"]), ("area_group", R["area_group_arr"]),
                      ("area_acronym_custom", R["area_arr"])]:
        agg["props"][name] = {}
        for k in range(K):
            a = arr[labels == k]
            a = a[~np.isin(a, ["unknown", "None", "nan", ""])]
            vals, cnt = np.unique(a, return_counts=True)
            agg["props"][name][k] = dict(zip(vals, cnt / max(cnt.sum(), 1)))
    for c in ANATOMY_COLS:
        agg["scalars"][c] = {k: np.nanmean(R["anatomy"][c][labels == k]) if np.isfinite(R["anatomy"][c][labels == k]).any()
                             else np.nan for k in range(K)}
    for cat, spec in ROC_CATEGORIES.items():
        s = roc[f"{cat}__sig"].values.astype(float)
        v = roc[f"{cat}__sel"].values.astype(float)
        va = roc[f"{cat}__abssel"].values.astype(float)
        with np.errstate(invalid="ignore"):
            agg["frac"][cat] = {k: np.nanmean(s[labels == k]) if np.isfinite(s[labels == k]).any() else np.nan
                                for k in range(K)}
            agg["sel"][cat] = {k: np.nanmean(v[labels == k]) for k in range(K)}
            agg["abssel"][cat] = {k: np.nanmean(va[labels == k]) for k in range(K)}
        if spec.get("split"):
            cols_ = {n: roc[f"{cat}__{n}"].values.astype(float) for n in ("whisker", "auditory", "none")}
            agg["props"][cat] = {k: {n: np.nanmean(x[labels == k]) for n, x in cols_.items()} for k in range(K)}
    # gated decision: fraction (and mean |sel|) WITHIN each cluster's decision neurons
    if "gated_decision" in ROC_CATEGORIES and "decision" in ROC_CATEGORIES:
        dec = roc["decision__sig"].values == 1
        gat = roc["gated_decision__sig"].values == 1
        va = roc["gated_decision__abssel"].values.astype(float)
        agg["n_decision"] = {k: int((dec & (labels == k)).sum()) for k in range(K)}
        ok = {k: agg["n_decision"][k] >= GATED_MIN_DECISION for k in range(K)}
        with np.errstate(invalid="ignore"):
            agg["frac"]["gated_decision"] = {k: (gat & dec & (labels == k)).sum() / agg["n_decision"][k] if ok[k]
                                             else np.nan for k in range(K)}
            agg["abssel"]["gated_decision"] = {k: np.nanmean(va[dec & (labels == k)]) if ok[k] else np.nan
                                               for k in range(K)}
    return agg


# ══════════════════════════════════════════════════════════════════════════
# main figure
# ══════════════════════════════════════════════════════════════════════════

def fig5_population_matrix_ext(R, roc, meta, fmk, cfg, out_path, columns=DEFAULT_COLUMNS,
                               row_mode="neuron", order="rastermap", roc_mode="fraction",
                               reward_panel="fmk_share", title_prefix="", family_label="Metacluster"):
    K = int(R["labels"].max()) + 1
    flipped = False
    if order == "rastermap":
        cluster_order = list(range(K))
    else:
        cluster_order, flipped = leaf_order(meta)
    M, blocks, _ = row_layout(R, cluster_order, row_mode)
    n_rows = M.shape[0]
    agg = cluster_aggregates(R, roc, K)
    fam = meta["families"]
    fam_pal = family_palette(fam.values(), display_order=[fam[k] for k in cluster_order])
    # the family strip is only shown together with its dendrogram: always in dendrogram order, and in
    # rastermap order only when the families are contiguous along it and the tree's leaves follow it
    # (contiguity-constrained segments); scattered metaclusters in rastermap order get neither.
    seq = [fam[k] for k in cluster_order]
    runs = [f for i, f in enumerate(seq) if i == 0 or f != seq[i - 1]]
    contiguous = len(runs) == len(set(runs))
    tree_in_rastermap_order = meta.get("Z") is not None and \
        [meta["clusters"][i] for i in dendrogram(meta["Z"], no_plot=True)["leaves"]] == list(range(K))
    show_family = order == "metacluster" or (contiguous and tree_in_rastermap_order)

    cols = [c for c in columns if c not in ("reward_fmk", "metacluster")]
    tail = {"fmk_share": ["reward_fmk"] if "reward_fmk" in columns else [], "figS4": ["figS4_f", "figS4_p"],
            "neuron": ["reward_neuron"]}.get(reward_panel, [])
    width = {"avg_ipsi": 0.3, "cc_tc_ct_iterated": 0.3, "cc_hierarchy_score_columns": 0.3, "waveform_type": 0.5,
             "area_group": 0.8, "area_acronym_custom": 0.8, "reward_fmk": 0.45, "reward_neuron": 0.45,
             "figS4_f": 1.8, "figS4_p": 1.4}
    panels = (["dendro", "strip"] if show_family else []) + ["matrix"] + cols + tail
    wr = [{"matrix": 10, "dendro": 1.3, "strip": 0.25}.get(p, width.get(p, 0.38)) for p in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(17 + 0.45 * len(panels) + 3.5 * (reward_panel == "figS4"), 13.5),
                             dpi=300, gridspec_kw={"width_ratios": wr, "wspace": 0.09})
    ax_of = dict(zip(panels, axes))

    # ── matrix: same drawing as fig5_population_matrix; cluster means get a wider
    #    colour range (99th pct instead of vmax_pct) so the averaged rows look less saturated
    if row_mode == "neuron":
        vmax = np.nanpercentile(np.abs(R["X_odd"]), cfg["vmax_pct"])
    else:
        vmax = np.nanpercentile(np.abs(M), 97)
    rows_txt = {"neuron": f"n={len(R['X'])} neurons", "cluster_mean": f"{K} cluster means (height ∝ n neurons)",
                "cluster_mean_uniform": f"{K} cluster means"}[row_mode]
    axm = ax_of["matrix"]
    im = rmu._draw_matrix(axm, M, R["n_bins_list"], [], vmax, cfg,
                          f"{title_prefix}CV (even trials), "
                          f"{'dendrogram' if order == 'metacluster' else order} order — {rows_txt}")
    axm.set_yticks([])
    axm.set_ylabel("")
    axm.title.set_fontsize(13)
    axm.set_xticklabels(cond_ticklabels(list(rmu.COND_LABELS)), fontsize=9, linespacing=0.95)
    for lo, hi, _ in blocks[1:]:
        axm.axhline(lo, color="k", lw=0.15, alpha=0.4)
    cax = axm.inset_axes([0.0, -0.1, 0.2, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Firing rate (z-score)" if cfg["normalize"] == "zscore" else "Firing rate (baseline-norm.)", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    # ── metacluster strip (+ dendrogram on the left)
    if show_family:
        draw_family_strip(ax_of["strip"], blocks, fam, fam_pal, n_rows, label=family_label)
        ax_of["strip"].tick_params(left=False, labelleft=False)
        draw_dendrogram(ax_of["dendro"], meta, blocks, n_rows, fam_pal, flipped=flipped,
                        label=f"{family_label.split(' (')[0]} dendrogram")

    # ── shared ROC scales
    roc_cats = [p[4:] for p in cols if p.startswith("roc:")]
    nonsplit = [c for c in roc_cats if not ROC_CATEGORIES[c].get("split")]
    finite = lambda d: np.array([x for x in d.values() if np.isfinite(x)])
    if roc_mode == "fraction":
        allv = np.concatenate([finite(agg["frac"][c]) for c in nonsplit]) if nonsplit else np.array([1.0])
        roc_vmax = min(1.0, np.ceil(allv.max() * 10) / 10)          # shared, rounded up to 10 %
        roc_norm = mcolors.Normalize(0, roc_vmax)
    else:
        # shared |selectivity| scale over the 2nd-98th pct of cluster values across all |sel| columns:
        # |selectivity| has a noise floor > 0, so a 0-based scale would wash every column out
        allv = np.concatenate([finite(agg["abssel"][c]) for c in nonsplit]) if nonsplit else np.array([0.0, 1.0])
        roc_vmin, roc_vmax = np.floor(np.percentile(allv, 2) * 20) / 20, np.ceil(np.percentile(allv, 98) * 20) / 20
        roc_norm = mcolors.Normalize(roc_vmin, roc_vmax)
        sel_cmap = white_to(SEL_COLOR)
        modv = finite(agg["sel"]["modality_pref"]) if "modality_pref" in roc_cats else np.array([1.0])
        mod_vmax = np.ceil(np.abs(modv).max() * 20) / 20
        mod_cmap = mcolors.LinearSegmentedColormap.from_list(
            "wh_aud", [MODALITY_COLORS["whisker"], "#ffffff", MODALITY_COLORS["auditory"]])

    anatomy_cmaps = build_anatomy_cmaps(*[R["anatomy"][c] for c in ANATOMY_COLS])
    cbar_jobs, legend_jobs = [], []
    for p in cols:
        ax = ax_of[p]
        if p in ANATOMY_COLS:
            cmap, norm = anatomy_cmaps[p]
            draw_scalar_col(ax, blocks, agg["scalars"][p], cmap, norm, n_rows, ANATOMY_TITLES[p])
            v0, v1 = norm.vmin, norm.vmax
            cbar_jobs.append((ax, cmap, norm, [v0, v1], [f"{v0:.2g}", f"{v1:.2g}"]))
        elif p == "waveform_type":
            draw_prop_col(ax, blocks, agg["props"][p], ["NW", "WW"], WAVEFORM_COLORS, n_rows, "Waveform")
            legend_jobs.append((ax, "Waveform", WAVEFORM_COLORS, "center", 1))
        elif p == "area_group":
            order_ref = allen_utils.get_area_group_custom_order()
            present = set(R["area_group_arr"])
            cats = [g for g in order_ref if g in present] + sorted(present - set(order_ref))
            colors = allen_utils.get_custom_area_groups_colors()
            draw_prop_col(ax, blocks, agg["props"][p], cats, colors, n_rows, "Area group")
            legend_jobs.append((ax, "Area group", {c: colors.get(c, "#aaaaaa") for c in cats}, "right", 2))
        elif p == "area_acronym_custom":
            order_ref = allen_utils.get_area_acronym_custom_order()
            present = list(pd.unique(R["area_arr"]))
            cats = sorted(present, key=lambda a: order_ref.index(a) if a in order_ref else len(order_ref))
            colors = acronym_colors(cats)
            draw_prop_col(ax, blocks, agg["props"][p], cats, colors, n_rows, "Area")
        elif p.startswith("roc:"):
            cat = p[4:]
            spec = ROC_CATEGORIES[cat]
            lbl = spec["label"]
            if spec.get("split") and roc_mode == "fraction":
                draw_prop_col(ax, blocks, agg["props"][cat], ["whisker", "auditory", "none"], MODALITY_COLORS,
                              n_rows, f"{lbl} (frac.)")
                legend_jobs.append((ax, "Modality pref.", {"whisker": MODALITY_COLORS["whisker"],
                                                           "auditory": MODALITY_COLORS["auditory"],
                                                           "none": MODALITY_COLORS["none"]}, "center", 1))
            elif spec.get("split"):
                norm = mcolors.Normalize(-mod_vmax, mod_vmax)
                draw_scalar_col(ax, blocks, agg["sel"][cat], mod_cmap, norm, n_rows, f"{lbl} (signed sel.)")
                cbar_jobs.append((ax, mod_cmap, norm, [-mod_vmax, 0, mod_vmax], [f"wh {mod_vmax:.2f}", "0",
                                                                                   f"aud {mod_vmax:.2f}"]))
            elif roc_mode == "fraction":
                cmap = white_to(ROC_COLORS.get(cat, "#444444"))
                ttl = (f"Gated / decision (frac.; grey < {GATED_MIN_DECISION} dec.)" if cat == "gated_decision"
                       else f"{lbl} (frac.)")
                # per-column range: [5th, 95th] pct of this column's cluster fractions (rounded to 5 %)
                lo, hi, ext = column_range(finite(agg["frac"][cat]).tolist())
                hi = min(hi, 1.0)
                norm = mcolors.Normalize(lo, hi, clip=True)
                draw_scalar_col(ax, blocks, agg["frac"][cat], cmap, norm, n_rows, ttl, nan_color=NAN_ROC_GREY)
                cbar_jobs.append((ax, cmap, norm, [lo, hi], [f"{lo:.0%}", f"{hi:.0%}"], ext))
            else:
                ttl = (f"Gated (|sel.|, decision nrns; grey < {GATED_MIN_DECISION})" if cat == "gated_decision"
                       else f"{lbl} (|sel.|)")
                # per-column |sel| range: [5th, 95th] pct of this column's cluster means (rounded to 0.05)
                lo, hi, ext = column_range(finite(agg["abssel"][cat]).tolist())
                norm = mcolors.Normalize(lo, hi, clip=True)
                draw_scalar_col(ax, blocks, agg["abssel"][cat], sel_cmap, norm, n_rows, ttl, nan_color=NAN_ROC_GREY)
                cbar_jobs.append((ax, sel_cmap, norm, [lo, hi], [f"{lo:.2f}", f"{hi:.2f}"], ext))
    if "reward_fmk" in tail:
        F, g = fmk["F"], fmk["groups"]
        mrp, mrm = F[g == "R+"].mean(0), F[g == "R-"].mean(0)
        share = {k: {"R+": mrp[k] / (mrp[k] + mrm[k]), "R-": mrm[k] / (mrp[k] + mrm[k])} if mrp[k] + mrm[k] > 0 else {}
                 for k in range(K)}
        colors = {"R+": GROUP_COLORS["rplus"], "R-": GROUP_COLORS["rminus"]}
        draw_prop_col(ax_of["reward_fmk"], blocks, share, ["R+", "R-"], colors, n_rows, "Group (mouse-level F_mk)")
        for lo, hi, k in blocks:
            if fmk["reject"][k]:
                ax_of["reward_fmk"].text(1.05, (lo + hi) / 2, "*", fontsize=9, va="center",
                                         transform=ax_of["reward_fmk"].get_yaxis_transform())
        legend_jobs.append((ax_of["reward_fmk"], "Reward group", {"R+": colors["R+"], "R−": colors["R-"]},
                            "center", 1))
    if "reward_neuron" in tail:
        props = {k: {"R+": (R["reward_arr"][R["labels"] == k] == "R+").mean(),
                     "R-": (R["reward_arr"][R["labels"] == k] == "R-").mean()} for k in range(K)}
        draw_prop_col(ax_of["reward_neuron"], blocks, props, ["R+", "R-"],
                      {"R+": GROUP_COLORS["rplus"], "R-": GROUP_COLORS["rminus"]}, n_rows, "Group (neurons)")
    if "figS4_f" in tail:
        draw_figS4_panels(ax_of["figS4_f"], ax_of["figS4_p"], blocks, fmk, n_rows)
        legend_jobs.append((ax_of["figS4_f"], "Reward group", {"R+": GROUP_COLORS["rplus"],
                                                               "R−": GROUP_COLORS["rminus"]}, "center", 2))

    # colorbars and legends after layout is final, positioned from the real axes positions
    for i, job in enumerate(cbar_jobs):
        ax, cmap, norm, ticks, tl = job[:5]
        # neighbouring colorbars alternate in height so their tick labels never collide
        small_cbar(fig, ax, cmap, norm, ticks, tl, extend=job[5] if len(job) > 5 else "neither",
                   y_off=0.045 + 0.035 * (i % 2))
    for ax, title, entries, anchor, ncol in legend_jobs:
        compact_legend(fig, ax, title, entries, anchor=anchor, ncol=ncol)
    rmu._save(fig, Path(out_path), dpi=300)

def acronym_colors(acronyms):
    """Allen colour per acronym if available, else its area-group colour."""
    allen = allen_utils.get_allen_color_dict()
    grp, _ = allen_utils.get_custom_area_color_per_group()
    out = {}
    for a in acronyms:
        c = allen.get(a) if isinstance(allen, dict) else None
        if c is not None and not isinstance(c, str):
            c = mcolors.to_hex(np.asarray(c) / (255 if np.max(c) > 1 else 1))
        out[a] = c if c is not None else grp.get(a, "#888888")
    # acronyms sharing one group colour: spread them in lightness so they stay distinguishable
    by_col = {}
    for a in acronyms:
        by_col.setdefault(out[a], []).append(a)
    for base, members in by_col.items():
        if len(members) < 2:
            continue
        rgb = np.array(mcolors.to_rgb(base))
        for i, a in enumerate(members):
            f = np.linspace(-0.45, 0.45, len(members))[i]   # <0 darker, >0 lighter
            out[a] = mcolors.to_hex(rgb * (1 + f) if f < 0 else rgb + (1 - rgb) * f)
    return out



# ══════════════════════════════════════════════════════════════════════════
# publication layout
# ══════════════════════════════════════════════════════════════════════════

# publication figure: categories (fixed order: metaclusters | matrix | anatomy | function | enrichment); each can be
# dropped or reduced; waveform type is always the first anatomy column
PUB_ANATOMY = ["waveform_type", "area_group", "avg_ipsi", "cc_hierarchy_score_columns"]
PUB_FUNCTION = ["roc:stim_responsive", "roc:modality_pref", "roc:lick_responsive", "roc:decision", "roc:gated_decision"]
PUB_ENRICHMENT = {"share": ["reward_dot"],                 # R+ share of mean F_mk (lollipop, filled = post-hoc sig.)
                  "fmk": ["fmk_dist", "fmk_p"]}            # F_mk mean +- SEM per cohort | -log10 p (MWU, BH-FDR)
PUB_COLUMNS = PUB_ANATOMY + PUB_FUNCTION + PUB_ENRICHMENT["share"]      # default layout
PUB_SHORT = {"roc:stim_responsive": "Stim. resp.", "roc:modality_pref": "Modality", "roc:lick_responsive": "Lick-resp.",
             "roc:decision": "Decision", "roc:gated_decision": "Gated / Decision", "area_group": "Area",
             "waveform_type": "Waveform", "reward_dot": "R+ share", "fmk_dist": "F_mk (mean ± SEM)",
             "fmk_p": "R+ vs R− (MWU)", "avg_ipsi": "SS-whisker proj.", "cc_tc_ct_iterated": "Hier. (Harris)",
             "cc_hierarchy_score_columns": "CC hierarchy score", "stage_dot": "Learning share"}
STAGE_COLORS = {"learning": "#E08214", "expert": "#542788"}   # learning (day 0) vs expert (day > 0) sessions
PUB_KEY_SHORT = {"cc_hierarchy_score_columns": "CC hierarchy score"}   # colour-key titles (single line)
PUB_BLOCKS = [("Anatomy", PUB_ANATOMY + ["cc_tc_ct_iterated"]),
              ("Function", [f"roc:{c}" for c in ROC_CATEGORIES]),
              ("Enrichment", PUB_ENRICHMENT["share"] + ["stage_dot"] + PUB_ENRICHMENT["fmk"])]
WAVEFORM_NAMES = {"NW": "narrow waveform", "WW": "wide waveform"}
def cond_ticklabels(labels):
    """Full PSTH-type names for every figure, first word on its own line ('Whisker\\nhit (lick)')."""
    return [l.replace(" ", "\n", 1) for l in labels]


PUB_COND_SHORT = {"Whisker pre": "W pre", "Whisker post": "W post", "Auditory pre": "A pre", "Auditory post": "A post",
                  "Whisker miss": "W miss", "Whisker hit": "W hit", "Auditory hit": "A hit", "Spont. lick": "Spont.",
                  "Whisker hit (lick)": "W hit", "Auditory hit (lick)": "A hit"}
OTHER_AREA_COLOR = "#bdbdbd"
# functional family names: traits enriched >= NAME_MIN_RATIO x vs population and present in >= NAME_MIN_FRAC of
# the family's neurons (gated: >= NAME_MIN_FRAC_GATED); up to 2 traits, strongest first
# traits are ranked by the DIFFERENCE between the family's fraction and the population fraction; a trait
# qualifies if diff >= NAME_MIN_DIFF (gated: NAME_MIN_DIFF_GATED) and ratio >= NAME_MIN_RATIO; up to 2 traits
NAME_MIN_RATIO, NAME_MIN_DIFF, NAME_MIN_DIFF_GATED = 1.3, 0.08, 0.02
TRAIT_NAMES = {"stim": "Stimulus", "whisker": "Whisker", "auditory": "Auditory", "lick_pos": "Lick-activated",
               "lick_neg": "Lick-suppressed", "decision": "Decision", "gated": "Gated"}


def hotcold_white_cmap(name="hotcold_white", sub=0.55, white_frac=0.30, N=256):
    """User's CUSTOM_HOTCOLD_CMAP (cmr.arctic_r | cmr.ember_r, each sub-range [0, sub]) made exactly white at 0.
    Samples the parent cmaps directly (area_latency_rastermap.diverging_cmap re-samples a sub-cmap at 1.0,
    which wraps the top end to blue) and blends to white over the inner `white_frac` of each half
    (ember_r starts at yellow, so without this the centre is yellow on the positive side)."""
    import cmasher as cmr  # noqa: F401  (registers cmr.* colormaps)
    half = N // 2
    left = plt.get_cmap("cmr.arctic_r")(np.linspace(sub, 0, half))[:, :3]    # edge -> centre
    right = plt.get_cmap("cmr.ember_r")(np.linspace(0, sub, half))[:, :3]    # centre -> edge
    x = np.linspace(0, 1, half)                                              # 0 at centre, 1 at edge
    w = np.clip(x / white_frac, 0, 1)[:, None]                               # weight of the cmap colour
    right = w * right + (1 - w)
    left = w[::-1] * left + (1 - w[::-1])
    return mcolors.LinearSegmentedColormap.from_list(name, np.vstack([left, right]), N=N)


MATRIX_CMAPS = {"coolwarm": "coolwarm", "bwr": "bwr", "hotcold_white": None}   # None -> built on demand


def _cond_group(label):
    l = label.lower()
    if "lick" in l:
        return "Lick-aligned"
    if " pre" in l or " post" in l:
        return "Passive"
    return "Active (stimulus-aligned)"


def _cond_header(label):
    """two-level header: (top, sub) = ('Passive', '') or ('Active', 'stimulus-aligned' | 'lick-aligned')"""
    g = _cond_group(label)
    return ("Passive", "") if g == "Passive" else ("Active", "lick-aligned" if g == "Lick-aligned" else "stimulus-aligned")


def draw_cond_headers(fig, col_axes, labels, y_top, y_sub, fs):
    """two-level Passive | Active (stimulus-aligned, lick-aligned) headers over per-condition axes (figure coords)"""
    hdr = [_cond_header(l) for l in labels]
    for key_idx, y, lw, f in ((1, y_sub, 0.6, fs - 0.5), (0, y_top, 0.9, fs + 0.5)):
        for (name, i0, i1) in _spans([h[key_idx] for h in hdr]):
            if not name:
                continue
            a0, a1 = col_axes[i0].get_position(), col_axes[i1 - 1].get_position()
            fig.add_artist(plt.Line2D([a0.x0, a1.x1], [y - 0.003] * 2, color="k", lw=lw, transform=fig.transFigure))
            fig.text((a0.x0 + a1.x1) / 2, y, name, ha="center", va="bottom", fontsize=f)


def _spans(keys):
    """contiguous runs [(key, i0, i1_exclusive)]"""
    out, s = [], 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[s]:
            out.append((keys[s], s, i)); s = i
    return out


# publication figure: single-hue matplotlib cmaps for the ROC fraction columns, darkest 20% dropped
PUB_ROC_CMAPS = {"stim_responsive": "Oranges", "lick_responsive": "Purples", "decision": "RdPu",
                 "gated_decision": "YlGn"}
PUB_CMAP_MAX = 0.8


def truncated_cmap(name, lo=0.0, hi=PUB_CMAP_MAX, n=256):
    return mcolors.LinearSegmentedColormap.from_list(f"{name}_{lo:g}_{hi:g}", plt.get_cmap(name)(np.linspace(lo, hi, n)))


def tint_to(color, light=0.85, dark=0.65):
    """Visible sequential map: light tint of `color` -> `color` -> darker `color` (no pure white end)."""
    rgb = np.array(mcolors.to_rgb(color))
    return mcolors.LinearSegmentedColormap.from_list(
        f"tint_{color}", [rgb + (1 - rgb) * light, rgb, rgb * dark])


def family_names(R, roc, fam):
    """Data-driven functional name per metacluster (see NAME_* thresholds). Returns (dict fam -> name, table)."""
    labels = R["labels"]
    fam_n = np.array([fam[k] for k in labels])
    matched = roc["roc_matched"].values.astype(bool)
    lk_sig = roc["lick_responsive__sig"].values == 1
    lk_sel = roc["lick_responsive__sel"].values
    traits = {"stim": roc["stim_responsive__sig"].values == 1,
              "whisker": roc["modality_pref__whisker"].values == 1,
              "auditory": roc["modality_pref__auditory"].values == 1,
              "lick_pos": lk_sig & (lk_sel > 0), "lick_neg": lk_sig & (lk_sel < 0),
              "decision": roc["decision__sig"].values == 1, "gated": roc["gated_decision__sig"].values == 1}
    base = {t: v[matched].mean() for t, v in traits.items()}
    rows, names = [], {}
    for f in sorted(set(fam_n)):
        m = (fam_n == f) & matched
        row = dict(family=f, n_neurons=int((fam_n == f).sum()))
        cand = []
        for t, v in traits.items():
            fr = v[m].mean() if m.any() else np.nan
            ratio = fr / base[t] if base[t] > 0 else np.nan
            diff = fr - base[t]
            row[f"frac_{t}"], row[f"ratio_{t}"], row[f"diff_{t}"] = fr, ratio, diff
            if np.isfinite(ratio) and ratio >= NAME_MIN_RATIO and \
                    diff >= (NAME_MIN_DIFF_GATED if t == "gated" else NAME_MIN_DIFF):
                cand.append((diff, t))
        cand = [t for _, t in sorted(cand, reverse=True)]
        if "lick_pos" in cand and "lick_neg" in cand:
            cand.remove("lick_neg" if row["diff_lick_pos"] >= row["diff_lick_neg"] else "lick_pos")
        if "stim" in cand and ("whisker" in cand or "auditory" in cand):
            cand.remove("stim")                      # a modality name already implies stimulus responsiveness
        names[f] = " + ".join(TRAIT_NAMES[t] for t in cand[:2]) if cand else "Mixed / weakly selective"
        row["name"] = names[f]
        rows.append(row)
    return names, pd.DataFrame(rows)


PUB_FONT_DIRS = [Path.home() / ".local/share/fonts", Path.home() / ".fonts", Path("C:/Windows/Fonts")]


def use_pub_font(family="Arial"):
    """register `family` TTFs from the user font dirs (haas: ~/.local/share/fonts) and make it the default
    sans-serif; falls back to DejaVu Sans when unavailable. No bold anywhere in the publication figures."""
    from matplotlib import font_manager
    for d in PUB_FONT_DIRS:
        if d.is_dir():
            for f in d.glob(f"{family.lower()}*.ttf"):
                try:
                    font_manager.fontManager.addfont(str(f))
                except Exception:
                    pass
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
                         "font.weight": "normal", "axes.titleweight": "normal", "axes.labelweight": "normal",
                         "figure.titleweight": "normal"})


# family names from the activity itself (not ROC): signed peak of the family-mean PSTH
#   whisker = passive pre/post + whisker miss (10-100 ms), auditory = passive pre/post (0-100 ms): sensory, no lick;
#   lick = spontaneous licks (-100..100 ms: motor, no stimulus)
NAME_THR_HZ = 3.5          # below this for all three features -> weakly responsive
NAME_MOD_RATIO = 2.0       # |W| >= 2|A| -> whisker, |A| >= 2|W| -> auditory, else multisensory
NAME_SM_RATIO = 1.5        # sensory >= 1.5 x motor -> sensory; motor >= 1.5 x sensory -> motor; else sensorimotor
NAME_FEATURES = {"whisker": (["Whisker pre", "Whisker post", "Whisker miss"], 0.01, 0.1),
                 "auditory": (["Auditory pre", "Auditory post"], 0.0, 0.1),     # auditory onset can be < 10 ms
                 "lick": (["Spont. lick"], -0.1, 0.1)}


def family_names_psth(R, fam):
    labels_c = list(rmu.COND_LABELS)
    off = np.r_[0, np.cumsum(R["n_bins_list"])]
    fam_n = np.array([fam[k] for k in R["labels"]])

    def peak(conds, a, b, m):
        tr = []
        for cond in conds:
            if cond not in labels_c:
                continue
            c = labels_c.index(cond); t = R["t_ctrs"][c]; sel = (t >= a) & (t <= b)
            tr.append(np.nanmean(R["X"][m][:, off[c]:off[c + 1]], 0)[sel])
        if not tr:
            return np.nan
        tr = np.mean(tr, 0)
        return float(tr[np.nanargmax(np.abs(tr))])

    names, rows = {}, []
    for f in dict.fromkeys(fam[k] for k in range(int(R["labels"].max()) + 1)):
        m = fam_n == f
        W, A, M = (peak(*NAME_FEATURES[k], m) for k in ("whisker", "auditory", "lick"))
        aW, aA, aM = abs(W), abs(A), abs(M)
        mod = "Whisker" if aW >= NAME_MOD_RATIO * aA else "Auditory" if aA >= NAME_MOD_RATIO * aW else "Multisensory"
        sv = W if aW >= aA else A; S = abs(sv)
        sens = ("Multisensory" if mod == "Multisensory" else f"{mod} sensory") if sv > 0 else f"{mod}-suppressed"
        mot = "Lick-activated" if M > 0 else "Lick-suppressed"
        if max(S, aM) < NAME_THR_HZ:
            kind, name = "weak", "Weakly responsive"
        elif aM >= NAME_SM_RATIO * S:
            kind, name = "motor", mot
        elif S >= NAME_SM_RATIO * aM:
            kind, name = "sensory", sens
        else:
            kind = "sensorimotor"
            name = (f"{mod} sensorimotor" if sv > 0 and M > 0 else "Broadly suppressed" if sv < 0 and M < 0
                    else f"{sens}, {mot.lower()}")
        names[f] = name
        rows.append(dict(family=f, n_neurons=int(m.sum()), peak_whisker_hz=W, peak_auditory_hz=A, peak_lick_hz=M,
                         modality=mod, kind=kind, name=name))
    return names, pd.DataFrame(rows)


# area groups pooled into "Other" in the publication figure (column + legend)
AREA_OTHER_GROUPS = {"Pons and medulla", "Amygdala and hypothalamus", "Olfactory areas", "Cortical subplate",
                     "Insular areas", "Visual areas", "Somatosensory-body"}


def _num(v):
    """compact tick label: integers for |v| >= 100, else 3 significant digits"""
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.3g}"


def _area_categories(R):
    order_ref = allen_utils.get_area_group_custom_order()
    present = set(pd.Series(R["area_group_arr"]).dropna().astype(str))
    major = [g for g in order_ref if g in present and g not in AREA_OTHER_GROUPS]
    colors = allen_utils.get_custom_area_groups_colors()
    return major, {g: colors.get(g, "#aaaaaa") for g in major}


def pub_columns(anatomy=PUB_ANATOMY, function=PUB_FUNCTION, enrichment="share"):
    """Side columns in the fixed category order; waveform type (if present) always first in Anatomy."""
    anatomy = list(anatomy or [])
    if "waveform_type" in anatomy:
        anatomy = ["waveform_type"] + [c for c in anatomy if c != "waveform_type"]
    return anatomy + list(function or []) + (PUB_ENRICHMENT[enrichment] if enrichment else [])


def fig5_population_matrix_publication(R, roc, meta, fmk, cfg, out_path, anatomy=PUB_ANATOMY,
                                       function=PUB_FUNCTION, enrichment="share", meta_style="dendrogram+strip",
                                       row_mode="cluster_mean_uniform", roc_mode="fraction",
                                       width_mm=183, height_mm=165, base_fs=7.5, title=None, matrix_cmap=None,
                                       title_rotation=45, columns=None, witness=None):
    """Flexible publication layout. Fixed order: metaclusters (dendrogram, optional numbered strip — never a
    strip without its dendrogram) | population matrix (row_mode 'neuron' or 'cluster_mean_uniform'; side-column
    row heights follow it) | Anatomy | Function | Enrichment. anatomy / function: lists of columns (subset of
    PUB_ANATOMY / PUB_FUNCTION) or None to drop the category; enrichment: 'share' (R+ share lollipop), 'fmk'
    (F_mk mean ± SEM per cohort + -log10 p MWU) or None. `columns` overrides the three categories."""
    """Publication layout (rastermap order, contiguous metaclusters):
    dendrogram | numbered metacluster strip | matrix (condition-group header bar, thin within-group /
    thick between-group separators, 100 ms scale bar, condition names) | blocks: Anatomy (wS1 projection,
    Gao hierarchy, Area) · Function (ROC) · Enrichment (R+ share of mean F_mk, diverging; post-hoc clusters
    significant by BOTH MWU and Welch emphasised) ; metacluster boundaries across all columns;
    vertical colour keys under their columns; compact legends."""
    import matplotlib.transforms as mtrans
    use_pub_font()
    plt.rcParams.update({"font.size": base_fs, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                         "ytick.major.width": 0.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
    K = int(R["labels"].max()) + 1
    cluster_order = list(range(K))
    M, blocks, _ = row_layout(R, cluster_order, row_mode)
    n_rows = M.shape[0]
    agg = cluster_aggregates(R, roc, K)
    fam = meta["families"]
    pal = family_palette(fam.values(), display_order=[fam[k] for k in cluster_order])
    seq = [fam[k] for k in cluster_order]
    runs = [f for i, f in enumerate(seq) if i == 0 or f != seq[i - 1]]
    show_family = len(runs) == len(set(runs)) and meta.get("Z") is not None

    # witness: optional {label: pd.Series cluster -> value} (e.g. centered reward-free MMD witness per cluster);
    # each becomes an Enrichment column 'wit:<label>' with a shared diverging R- / white / R+ scale
    witness = witness or {}
    blocks_def = [(b, list(cs) + ([f"wit:{k}" for k in witness] if b == "Enrichment" else [])) for b, cs in PUB_BLOCKS]
    short = {**PUB_SHORT, **{f"wit:{k}": f"Witness: {k}" for k in witness}}
    blk_of = {c: b for b, cs in blocks_def for c in cs}
    cols = list(columns) if columns is not None else pub_columns(anatomy, function, enrichment)
    cols += [f"wit:{k}" for k in witness if f"wit:{k}" not in cols]
    if fmk.get("stage") is not None and "stage_dot" not in cols and "reward_dot" in cols:
        cols.insert(cols.index("reward_dot") + 1, "stage_dot")              # learning+expert runs only
    if witness:
        wit_v = float(np.nanpercentile(np.abs(np.concatenate([np.asarray(s, float) for s in witness.values()])), 98))
        wit_cmap = mcolors.LinearSegmentedColormap.from_list("wit", [GROUP_COLORS["rminus"], "white", GROUP_COLORS["rplus"]])
        wit_norm = mcolors.Normalize(-wit_v, wit_v, clip=True)
    show_strip = show_family and "strip" in meta_style
    n_neu, n_mice = len(R["X"]), len(np.unique(R["mouse_arr"]))
    panels, prev = [], None
    for c in cols:
        b = blk_of.get(c, c)
        if prev is not None and b != prev:
            panels.append("_gap_wide" if b == "Enrichment" else "_gap")
        panels.append(c); prev = b
    panels = (["dendro"] + (["strip"] if show_strip else []) if show_family else []) + ["matrix"] + \
        (["_gap"] if cols else []) + panels
    # widths by information content: multi-category composition columns wider, single-value columns narrow
    wmap = {"dendro": 0.9, "strip": 0.18, "matrix": 8.0, "_gap": 0.3, "_gap_wide": 0.55, "area_group": 0.8, "waveform_type": 0.2,
            "roc:modality_pref": 0.3, "reward_dot": 0.95, "stage_dot": 0.95, "fmk_dist": 0.95, "fmk_p": 0.7}
    wr = [wmap.get(p, 0.2) for p in panels]
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=300)
    gs = fig.add_gridspec(1, len(panels), width_ratios=wr, wspace=0.06, left=0.04, right=0.93, top=0.8, bottom=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(panels))]
    ax_of = {}
    for p, a in zip(panels, axes):
        if p.startswith("_gap"):
            a.axis("off")
        else:
            ax_of[p] = a

    # ── matrix
    axm = ax_of["matrix"]
    vmax = np.nanpercentile(np.abs(R["X_odd"] if row_mode == "neuron" else M), 97 if row_mode != "neuron" else cfg["vmax_pct"])
    im = rmu._draw_matrix(axm, M, R["n_bins_list"], [], vmax, cfg, "")
    if matrix_cmap is None:                           # default: coolwarm for single neurons, bwr for cluster means
        matrix_cmap = "coolwarm" if row_mode == "neuron" else "bwr"
    im.set_cmap(hotcold_white_cmap() if matrix_cmap == "hotcold_white" else matrix_cmap)
    white_centred = matrix_cmap in ("bwr", "hotcold_white")
    for ln in list(axm.lines):
        ln.remove()
    nb = R["n_bins_list"]; off = np.r_[0, np.cumsum(nb)]
    labels_c = list(rmu.COND_LABELS)
    grp = [_cond_group(l) for l in labels_c]
    for i in range(1, len(nb)):
        axm.axvline(off[i], color="k", lw=1.2 if grp[i] != grp[i - 1] else 0.35)
    for i, tc in enumerate(R["t_ctrs"]):   # onset marks: grey on white-centred maps, white otherwise
        axm.axvline(off[i] + int(np.searchsorted(tc, 0)), color="0.45" if white_centred else "w", lw=0.45,
                    ls=(0, (2, 2)))
    axm.set_xticks([(off[i] + off[i + 1]) / 2 for i in range(len(nb))])
    axm.set_xticklabels(cond_ticklabels(labels_c), fontsize=base_fs - 1.5, rotation=0, linespacing=0.95)
    axm.tick_params(axis="x", length=0, pad=2)
    axm.set_yticks([]); axm.set_ylabel(""); axm.set_title("")
    tr = mtrans.blended_transform_factory(axm.transData, axm.transAxes)
    hdr = [_cond_header(l) for l in labels_c]
    for (sub, i0, i1) in _spans([h[1] for h in hdr]):                  # level 2: alignment within Active
        if sub:
            x0, x1 = off[i0] + 3, off[i1] - 3
            axm.plot([x0, x1], [1.012, 1.012], color="k", lw=0.6, transform=tr, clip_on=False)
            axm.text((x0 + x1) / 2, 1.018, sub, transform=tr, ha="center", va="bottom", fontsize=base_fs - 1.2)
    for (top_, i0, i1) in _spans([h[0] for h in hdr]):                 # level 1: Passive | Active
        x0, x1 = off[i0] + 3, off[i1] - 3
        axm.plot([x0, x1], [1.062, 1.062], color="k", lw=0.9, transform=tr, clip_on=False)
        axm.text((x0 + x1) / 2, 1.068, top_, transform=tr, ha="center", va="bottom", fontsize=base_fs - 0.3)
    n100 = 100 / cfg["stride_ms"]
    axm.plot([off[0] + 2, off[0] + 2 + n100], [-0.07, -0.07], color="k", lw=1.3, transform=tr, clip_on=False)
    axm.text(off[0] + 2 + n100 / 2, -0.08, "100 ms", transform=tr, ha="center", va="top", fontsize=base_fs - 0.5)
    rows_txt = {"neuron": f"{n_neu:,} neurons, {n_mice} mice",
                "cluster_mean_uniform": f"{K} clusters (cluster means) · {n_neu:,} neurons, {n_mice} mice"}.get(row_mode, "")
    ylab = {"neuron": f"Neurons (n = {n_neu:,}), rastermap order",
            "cluster_mean_uniform": f"Clusters (n = {K}; {n_neu:,} neurons), rastermap order"}.get(row_mode, "")

    # ── dendrogram + numbered strip
    if show_family:
        draw_dendrogram(ax_of["dendro"], meta, blocks, n_rows, pal, label="")
        ad = ax_of["dendro"]; ad.tick_params(labelsize=base_fs - 1); ad.set_xlabel("1 − r", fontsize=base_fs - 0.5)
        for t_ in list(ad.texts):
            t_.remove()
        if show_strip:
            ast = ax_of["strip"]
            fam_rows = {}
            for lo, hi, k in blocks:
                ast.barh((lo + hi) / 2, 1.0, height=hi - lo, color=pal[fam[k]], edgecolor="none", align="center")
                a, b = fam_rows.get(fam[k], (lo, hi)); fam_rows[fam[k]] = (min(a, lo), max(b, hi))
            for f, (a, b) in fam_rows.items():
                if (b - a) / n_rows > 0.028:
                    ast.text(0.5, (a + b) / 2, str(f), ha="center", va="center", fontsize=base_fs - 1.5, color="w")
            ast.set_xlim(0, 1); ast.set_ylim(n_rows, 0); ast.axis("off")
        hdr_ax = ax_of["strip"] if show_strip else ax_of["dendro"]
        hdr_ax.text(0.5 if show_strip else 0.95, 1.01, "Metacluster", transform=hdr_ax.transAxes,
                    rotation=90, rotation_mode="anchor", ha="left", va="center", fontsize=base_fs)

    # ── columns
    keys = []
    anat_cmaps = build_anatomy_cmaps(*[R["anatomy"][c] for c in ANATOMY_COLS])
    major, area_col = _area_categories(R)
    for p in cols:
        ax = ax_of[p]
        for s_ in ("top", "right", "left", "bottom"):
            ax.spines[s_].set_visible(False)
        if p in ANATOMY_COLS:
            cmap, norm = anat_cmaps[p]
            for lo, hi, k in blocks:
                v = agg["scalars"][p].get(k, np.nan)
                ax.barh((lo + hi) / 2, 1, height=(hi - lo) * 0.92, color=cmap(norm(v)) if np.isfinite(v) else "#eeeeee")
            keys.append((ax, cmap, norm, [norm.vmin, norm.vmax], [_num(norm.vmin), _num(norm.vmax)], "neither"))
        elif p == "waveform_type":
            for lo, hi, k in blocks:
                pr = agg["props"]["waveform_type"].get(k, {})
                l0 = 0.0
                for w_ in ("NW", "WW"):
                    v = pr.get(w_, 0)
                    if v > 0:
                        ax.barh((lo + hi) / 2, v, left=l0, height=(hi - lo) * 0.92, color=WAVEFORM_COLORS[w_],
                                edgecolor="none")
                        l0 += v
        elif p.startswith("wit:"):
            vals = witness[p[4:]]
            for lo, hi, k in blocks:
                v = vals.get(k, np.nan)
                ax.barh((lo + hi) / 2, 1, height=(hi - lo) * 0.92,
                        color=wit_cmap(wit_norm(v)) if np.isfinite(v) else NAN_ROC_GREY, edgecolor="none")
            if p == [c for c in cols if c.startswith("wit:")][0]:                 # one shared key
                keys.append((ax, wit_cmap, wit_norm, [-wit_v, 0, wit_v], [f"{-wit_v:.2g}", "0", f"{wit_v:.2g}"], "both"))
        elif p in ("fmk_dist", "fmk_p"):
            F, g = fmk["F"], fmk["groups"]
            if p == "fmk_dist":
                stats_ = {}
                for grp_ in ("R+", "R-"):
                    Fg = F[g == grp_]
                    stats_[grp_] = (Fg.mean(0), Fg.std(0, ddof=1) / np.sqrt(len(Fg)))
                xmax = max(float(np.nanpercentile(mu + se, 98)) for mu, se in stats_.values())   # robust axis
                xmax = np.ceil(xmax * 1.1 * 100) / 100
                for grp_, dy in (("R+", -0.18), ("R-", 0.18)):
                    mu, se = stats_[grp_]
                    col = GROUP_COLORS["rplus" if grp_ == "R+" else "rminus"]
                    for lo, hi, k in blocks:
                        y = (lo + hi) / 2 + dy * (hi - lo)
                        ax.plot([mu[k] - se[k], min(mu[k] + se[k], xmax)], [y, y], color=col, lw=0.6, zorder=2)
                        ax.scatter([min(mu[k], xmax)], [y], s=4, color=col, zorder=3, linewidths=0, clip_on=False)
                ax.set_xlim(0, xmax); ax.set_xticks([0, xmax]); ax.set_xticklabels(["0", f"{xmax:g}"], fontsize=base_fs - 1.5)
                tl = ax.get_xticklabels(); tl[0].set_ha("left"); tl[-1].set_ha("right")
            else:
                lp = -np.log10(np.clip(fmk["p_fdr"], 1e-12, 1))
                for lo, hi, k in blocks:
                    ax.barh((lo + hi) / 2, lp[k], height=(hi - lo) * 0.8, color="0.15" if fmk["reject"][k] else "0.78",
                            edgecolor="none")
                xmax = max(2.0, float(np.ceil(np.nanmax(lp) * 1.05)))
                ax.axvline(-np.log10(0.05), color="0.35", lw=0.6, ls="--", zorder=3)
                ax.set_xlim(0, xmax); ax.set_xticks([0, xmax]); ax.set_xticklabels(["0", f"{xmax:g}"], fontsize=base_fs - 1.5)
                tl = ax.get_xticklabels(); tl[0].set_ha("left"); tl[-1].set_ha("right")
                ax.set_xlabel("−log10 p", fontsize=base_fs - 1.5, labelpad=1)
            ax.spines["bottom"].set_visible(True); ax.tick_params(axis="x", length=2, pad=1)
        elif p == "area_group":
            for lo, hi, k in blocks:
                pr = agg["props"]["area_group"].get(k, {})
                parts = [(g, pr.get(g, 0)) for g in major] + [("Other", sum(v for g, v in pr.items() if g not in major))]
                l0 = 0.0
                for g, v in parts:
                    if v > 0:
                        ax.barh((lo + hi) / 2, v, left=l0, height=(hi - lo) * 0.92,
                                color=area_col.get(g, OTHER_AREA_COLOR), edgecolor="none")
                        l0 += v
        elif p.startswith("roc:"):
            cat = p[4:]
            if ROC_CATEGORIES[cat].get("split"):
                for lo, hi, k in blocks:
                    l0 = 0.0
                    for c in ("whisker", "auditory", "none"):
                        v = agg["props"][cat].get(k, {}).get(c, 0)
                        if np.isfinite(v) and v > 0:
                            ax.barh((lo + hi) / 2, v, left=l0, height=(hi - lo) * 0.92, color=MODALITY_COLORS[c], edgecolor="none")
                            l0 += v
            else:
                vals = agg["frac"][cat] if roc_mode == "fraction" else agg["abssel"][cat]
                lo_, hi_, ext = column_range([v for v in vals.values() if np.isfinite(v)])
                if roc_mode == "fraction":
                    hi_ = min(hi_, 1.0)
                cmap = (truncated_cmap(PUB_ROC_CMAPS[cat]) if roc_mode == "fraction" and cat in PUB_ROC_CMAPS
                        else tint_to(ROC_COLORS.get(cat, "#444444") if roc_mode == "fraction" else SEL_COLOR))
                norm = mcolors.Normalize(lo_, hi_, clip=True)
                for lo, hi, k in blocks:
                    v = vals.get(k, np.nan)
                    ax.barh((lo + hi) / 2, 1, height=(hi - lo) * 0.92,
                            color=cmap(norm(v)) if np.isfinite(v) else NAN_ROC_GREY, edgecolor="none")
                fmt = (lambda x: f"{x:.0%}") if roc_mode == "fraction" else (lambda x: f"{x:.2f}")
                keys.append((ax, cmap, norm, [lo_, hi_], [fmt(lo_), fmt(hi_)], ext))
        elif p in ("reward_dot", "stage_dot"):
            if p == "reward_dot":
                F, g = fmk["F"], fmk["groups"]
                m_hi, m_lo = F[g == "R+"].mean(0), F[g == "R-"].mean(0)
                sig = fmk["reject"]      # post-hoc: Mann-Whitney U, BH-FDR (run_stats_only); Welch deferred
                col_hi, col_lo, lab_lo, lab_hi = GROUP_COLORS["rplus"], GROUP_COLORS["rminus"], "R−", "R+"
            else:                        # learning share of mean session-level F_sk (learning + expert runs)
                F, g = fmk["stage"]["F"], fmk["stage"]["groups"]
                m_hi, m_lo = F[g == "learning"].mean(0), F[g == "expert"].mean(0)
                sig = fmk["stage"]["sig"]   # sessions: Mann-Whitney U AND Welch, BH-FDR
                col_hi, col_lo, lab_lo, lab_hi = STAGE_COLORS["learning"], STAGE_COLORS["expert"], "Expert", "Learning"
            with np.errstate(invalid="ignore", divide="ignore"):
                shares = m_hi / (m_hi + m_lo)
            dev = np.abs(shares - 0.5)
            half = float(np.clip(np.ceil(np.nanpercentile(dev[np.isfinite(dev)], 95) * 1.15 * 20) / 20, 0.15, 0.5))
            ax.axvline(0.5, color="0.55", lw=0.6, zorder=1)
            for lo, hi, k in blocks:
                if not np.isfinite(shares[k]):
                    continue
                s_ = shares[k]; col = col_hi if s_ > 0.5 else col_lo
                y = (lo + hi) / 2
                sc = float(np.clip(s_, 0.5 - half, 0.5 + half))          # out-of-range values sit at the edge
                if sig[k]:                                  # post-hoc significant: filled circle
                    ax.plot([0.5, sc], [y, y], color=col, lw=0.8, zorder=2, clip_on=False)
                    ax.scatter([sc], [y], s=10, marker="o", facecolor=col, edgecolor=col, linewidths=0.5, zorder=4, clip_on=False)
                else:
                    ax.plot([0.5, sc], [y, y], color=col, lw=0.5, alpha=0.55, zorder=2, clip_on=False)
                    ax.scatter([sc], [y], s=5, marker="o", facecolor="white", edgecolor=col, linewidths=0.5,
                               alpha=0.8, zorder=3, clip_on=False)
            ax.set_xlim(0.5 - half, 0.5 + half)
            # cluster numbers (1-based, as in per_cluster_stats.csv) of post-hoc significant clusters: on the
            # opposite side of the 50% line from the marker; vertically adjacent labels alternate x-offset
            # label lane right of the column: labels spread vertically (min gap) with leader lines to markers
            items = [((lo + hi) / 2, k, float(np.clip(shares[k], 0.5 - half, 0.5 + half)))
                     for lo, hi, k in blocks if sig[k] and np.isfinite(shares[k])]
            if items:
                gap = 0.022 * n_rows                                  # minimum label spacing (rows)
                ys = [y for y, _, _ in items]
                lab_y = list(ys)
                for i in range(1, len(lab_y)):                        # push down
                    lab_y[i] = max(lab_y[i], lab_y[i - 1] + gap)
                over = lab_y[-1] - (n_rows - gap / 2)
                if over > 0:                                          # shift up if it runs past the bottom
                    lab_y = [v - over for v in lab_y]
                    for i in range(len(lab_y) - 2, -1, -1):
                        lab_y[i] = min(lab_y[i], lab_y[i + 1] - gap)
                x_lane = 0.5 + half * 1.18
                for (y, k, sc), ly in zip(items, lab_y):
                    ax.plot([sc, 0.5 + half * 1.02, x_lane - 0.04 * half], [y, y, ly], color="0.6", lw=0.3,
                            clip_on=False, zorder=1)
                    ax.text(x_lane, ly, str(k + 1), ha="left", va="center", fontsize=base_fs - 2.5, color="0.15",
                            clip_on=False)
            ax.set_xticks([0.5 - half, 0.5, 0.5 + half])
            ax.set_xticklabels([f"{0.5 - half:.0%}\n{lab_lo}", "50%", f"{0.5 + half:.0%}\n{lab_hi}"], fontsize=base_fs - 1.5)
            ax.spines["bottom"].set_visible(True); ax.tick_params(axis="x", length=2, pad=1)
        ax.set_ylim(n_rows, 0); ax.set_yticks([])
        if p not in ("reward_dot", "stage_dot", "fmk_dist", "fmk_p"):
            ax.set_xlim(0, 1); ax.set_xticks([])
        ax.text(0.5, 1.01, short.get(p, p), transform=ax.transAxes, rotation=title_rotation,
                rotation_mode="anchor", ha="left", va="bottom", fontsize=base_fs - 1.0, linespacing=0.95)

    # ── keep angled column titles / enrichment tick labels inside the figure: shrink the grid by the overflow
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    right_arts = [t for p in cols for t in ax_of[p].texts] + \
                 [t for p in cols for t in ax_of[p].get_xticklabels() if t.get_text()]
    x_right = max([a.get_window_extent(rend).transformed(inv).x1 for a in right_arts] or [0])
    if x_right > 0.99:
        gs.update(right=gs.right - (x_right - 0.99))

    # ── block headers above the tallest column label (measured)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    label_top = max([t.get_window_extent(rend).transformed(fig.transFigure.inverted()).y1
                     for p in cols for t in ax_of[p].texts] or [axm.get_position().y1 + 0.05])
    # PSTH-type labels under the matrix: largest font (<= base_fs - 1.5) at which neighbours don't collide
    cond_labs = [t for t in axm.get_xticklabels() if t.get_text()]
    for fs_ in np.arange(base_fs - 1.5, base_fs - 4.01, -0.25):
        for t in cond_labs:
            t.set_fontsize(fs_)
        fig.canvas.draw()
        bb = [t.get_window_extent(fig.canvas.get_renderer()).expanded(1.04, 1.0) for t in cond_labs]
        if not any(bb[i].overlaps(bb[i + 1]) for i in range(len(bb) - 1)):
            break
    title_room = 0.935                                   # block headers must stay below the figure title
    if label_top + 0.045 > title_room:                   # tall angled titles: move the panel grid down
        gs.update(top=gs.top - (label_top + 0.045 - title_room))
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        label_top = max(t.get_window_extent(rend).transformed(fig.transFigure.inverted()).y1
                        for p in cols for t in ax_of[p].texts)
    for bname, bcols in blocks_def:
        present = [ax_of[c] for c in cols if c in bcols]
        if not present:
            continue
        x0 = present[0].get_position().x0; x1 = present[-1].get_position().x1
        fig.add_artist(plt.Line2D([x0, x1], [label_top + 0.01] * 2, color="k", lw=0.9, transform=fig.transFigure))
        fig.text((x0 + x1) / 2, label_top + 0.016, bname, ha="center", va="bottom", fontsize=base_fs - 0.3)

    # ── colour keys: matrix (horizontal, under the matrix); columns (vertical, staggered in 2 rows)
    pm = axm.get_position()
    cax = fig.add_axes([pm.x0 + pm.width * 0.6, pm.y0 - 0.075, pm.width * 0.32, 0.013])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Firing rate (z-score)", fontsize=base_fs - 0.5, labelpad=1)
    cb.ax.tick_params(labelsize=base_fs - 1, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
    # column keys: compact grid of small horizontal bars (3 per row) under the side columns, each titled with
    # its column name; font = largest size at which no key text overlaps
    col_of = {id(ax_of[c]): c for c in cols}
    side = cols or ["matrix"]
    gx0 = min(ax_of[c].get_position().x0 for c in side); gx1 = max(ax_of[c].get_position().x1 for c in side)
    ncol_k = 3
    cell = (gx1 - gx0) / ncol_k
    key_w, key_h, row_h = cell * 0.55, 0.007, 0.05
    key_axes, key_titles = [], []
    for i, (ax, cmap, norm, ticks, tl, ext) in enumerate(keys):
        r_, c_ = divmod(i, ncol_k)
        x0k = gx0 + c_ * cell + (cell - key_w) / 2
        y0k = pm.y0 - 0.085 - r_ * row_h
        cx = fig.add_axes([x0k, y0k, key_w, key_h])
        c = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cx, orientation="horizontal",
                         extend=ext, extendfrac=0.08)
        c.set_ticks(ticks); c.set_ticklabels(tl)
        c.ax.tick_params(length=1.2, pad=1)
        lab = c.ax.get_xticklabels()
        lab[0].set_ha("left"); lab[-1].set_ha("right")        # end labels stay within the bar's width
        c.outline.set_linewidth(0.35)
        cname = col_of[id(ax)]
        key_titles.append(cx.set_title(PUB_KEY_SHORT.get(cname, "Witness (centered)" if cname.startswith("wit:") else short.get(cname, cname)), pad=2))
        key_axes.append(cx)
    for fs_ in np.arange(base_fs - 1, base_fs - 3.51, -0.25):
        texts = [t for cx in key_axes for t in cx.get_xticklabels()] + key_titles
        for t in texts:
            t.set_fontsize(fs_)
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        bb = [t.get_window_extent(rend).expanded(1.1, 1.0) for t in texts if t.get_text()]
        if not any(bb[a].overlaps(bb[b]) for a in range(len(bb)) for b in range(a + 1, len(bb))):
            break
    y_below = pm.y0 - 0.11
    if "area_group" in cols:
        handles = [Patch(facecolor=area_col[g], label=g) for g in major] + [Patch(facecolor=OTHER_AREA_COLOR, label="Other")]
        lg_area = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(pm.x0, y_below), ncol=3,
                             fontsize=base_fs - 1.5, frameon=False, handlelength=0.9, handleheight=0.9,
                             columnspacing=0.9, labelspacing=0.25, title="Area group", title_fontsize=base_fs - 1)
        fig.canvas.draw()
        y_below = lg_area.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y0 - 0.01
    extra = []
    if "waveform_type" in cols:
        extra += [Patch(facecolor=WAVEFORM_COLORS[w_], label=WAVEFORM_NAMES[w_]) for w_ in ("NW", "WW")]
    if "roc:modality_pref" in cols:
        extra += [Patch(facecolor=MODALITY_COLORS[k], label=l) for k, l in
                  [("whisker", "whisker-pref."), ("auditory", "auditory-pref."), ("none", "no pref.")]]
    if "roc:gated_decision" in cols:
        extra += [Patch(facecolor=NAN_ROC_GREY, label=f"gated: < {GATED_MIN_DECISION} decision neurons")]
    if "reward_dot" in cols:
        extra += [plt.Line2D([], [], marker="o", color="w", markerfacecolor="0.3", markeredgecolor="0.3", ms=4,
                             label="R+/R− post-hoc Mann-Whitney (FDR<0.05)")]
    if "stage_dot" in cols:
        extra += [plt.Line2D([], [], marker="o", color="w", markerfacecolor=STAGE_COLORS[s_], markeredgecolor=STAGE_COLORS[s_],
                             ms=4, label=f"{s_}-enriched (sessions; MWU & Welch FDR<0.05)") for s_ in ("learning", "expert")]
    if "fmk_dist" in cols:
        extra += [plt.Line2D([], [], marker="o", color=GROUP_COLORS["rplus"], ms=3, lw=0.8, label="R+ (mean ± SEM)"),
                  plt.Line2D([], [], marker="o", color=GROUP_COLORS["rminus"], ms=3, lw=0.8, label="R− (mean ± SEM)")]
    if "fmk_p" in cols:
        extra += [Patch(facecolor="0.15", label="Mann-Whitney FDR<0.05"), Patch(facecolor="0.78", label="n.s.")]
    if extra:
        fig.legend(handles=extra, loc="upper left", bbox_to_anchor=(pm.x0, y_below), ncol=3,
                   fontsize=base_fs - 1.5, frameon=False, handlelength=0.9, handleheight=0.9, labelspacing=0.25,
                   columnspacing=1.2)
    # population size: title and a y-axis label left of the leftmost panel (outside the dendrogram labels)
    left_ax = ax_of.get("dendro", axm)
    lp_ = left_ax.get_position()
    fig.text(lp_.x0 - 0.012, pm.y0 + pm.height / 2, ylab, rotation=90, ha="right", va="center", fontsize=base_fs - 0.5)
    fig.text(0.02, 0.985, (f"{title} — " if title else "") + rows_txt, ha="left", va="top", fontsize=base_fs + 1.5)
    rmu._save(fig, Path(out_path), dpi=600)
    plt.rcdefaults()


def fig_metacluster_psth_summary(R, roc, meta, fmk, cfg, out_path, min_frac_neurons=0.01, base_fs=7,
                                 width_mm=183, row_mm=9.0, names=None):
    """One row per metacluster (rastermap order; families with >= min_frac_neurons of neurons), one column per
    PSTH type (condition): mean ± SEM (even trials) of the family's neurons, with R+ and R− means overlaid.
    Rows are labelled with the metacluster number, its data-driven functional name, n neurons and n mice."""
    import matplotlib.transforms as mtrans
    use_pub_font()
    plt.rcParams.update({"font.size": base_fs, "axes.linewidth": 0.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
    labels, X = R["labels"], R["X"]
    fam = meta["families"]
    fam_n = np.array([fam[k] for k in labels])
    order = list(dict.fromkeys(fam[k] for k in range(int(labels.max()) + 1)))
    share = pd.Series(fam_n).value_counts(normalize=True)
    shown = [f for f in order if share.get(f, 0) >= min_frac_neurons]
    if names is None:
        names, _ = family_names_psth(R, fam)
    pal = family_palette(fam.values(), display_order=order)
    nb = R["n_bins_list"]; off = np.r_[0, np.cumsum(nb)]
    labels_c = list(rmu.COND_LABELS); nC = len(nb)
    grp = [_cond_group(l) for l in labels_c]
    height_mm = 26 + row_mm * len(shown)
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=300)
    wr = [3.6] + [1.0] * nC
    gs = fig.add_gridspec(len(shown), nC + 1, width_ratios=wr, wspace=0.12, hspace=0.25,
                          left=0.01, right=0.99, top=1 - 18 / height_mm, bottom=8 / height_mm)
    for r, f in enumerate(shown):
        m = fam_n == f
        axl = fig.add_subplot(gs[r, 0]); axl.axis("off")
        axl.add_patch(plt.Rectangle((0, 0.1), 0.06, 0.8, color=pal[f], transform=axl.transAxes, clip_on=False))
        n_m = len(np.unique(R["mouse_arr"][m]))
        axl.text(0.09, 0.82, f"Metacluster {f}", transform=axl.transAxes, ha="left", va="center", fontsize=base_fs)
        axl.text(0.09, 0.5, names.get(f, ""), transform=axl.transAxes, ha="left", va="center", fontsize=base_fs - 0.5)
        axl.text(0.09, 0.18, f"{m.sum():,} neurons · {n_m} mice", transform=axl.transAxes, ha="left", va="center",
                 fontsize=base_fs - 1, color="0.35")
        seg_vals = [X[m][:, off[c]:off[c + 1]] for c in range(nC)]
        ymax = max(np.nanmax(np.nanmean(v, 0) + np.nanstd(v, 0) / np.sqrt(len(v))) for v in seg_vals)
        ymin = min(np.nanmin(np.nanmean(v, 0) - np.nanstd(v, 0) / np.sqrt(len(v))) for v in seg_vals)
        pad = 0.08 * (ymax - ymin + 1e-9)
        for c in range(nC):
            ax = fig.add_subplot(gs[r, c + 1])
            t = R["t_ctrs"][c]; v = seg_vals[c]
            mu, se = np.nanmean(v, 0), np.nanstd(v, 0) / np.sqrt(len(v))
            ax.fill_between(t, mu - se, mu + se, color="0.75", lw=0)
            ax.plot(t, mu, color="k", lw=0.8)
            ax.axvline(0, color="0.5", lw=0.4, ls=":")
            ax.axhline(0, color="0.8", lw=0.3)
            ax.set_ylim(ymin - pad, ymax + pad)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            ax.tick_params(labelsize=base_fs - 2, length=1.5, pad=1)
            if c > 0:
                ax.set_yticklabels([])
            if r < len(shown) - 1:
                ax.set_xticklabels([])
            if r == 0:
                ax.set_title(cond_ticklabels([labels_c[c]])[0], fontsize=base_fs - 1.5, pad=2, linespacing=0.95)
    # condition-group headers
    fig.canvas.draw()
    top_axes = [a for a in fig.axes if a.get_title()]
    draw_cond_headers(fig, top_axes, labels_c, y_top=1 - 4 / height_mm, y_sub=1 - 8.5 / height_mm, fs=base_fs)
    fig.text(0.01, 2 / height_mm, "Mean ± SEM firing rate across all neurons of both cohorts (z-score, even trials); "
             "time (s) from alignment event", fontsize=base_fs - 1, color="0.3")
    rmu._save(fig, Path(out_path), dpi=600)
    plt.rcdefaults()


# ══════════════════════════════════════════════════════════════════════════
# sanity checks
# ══════════════════════════════════════════════════════════════════════════

def reordering_assessment(meta, K, out_dir):
    raw = [meta["clusters"][i] for i in dendrogram(meta["Z"], no_plot=True)["leaves"]]
    raw_pos = np.empty(K, int); raw_pos[raw] = np.arange(K)
    rho_raw = spearmanr(np.arange(K), raw_pos)[0]
    leaves, flipped = leaf_order(meta)            # oriented to the rastermap direction
    pos = np.empty(K, int); pos[leaves] = np.arange(K)
    fam = meta["families"]
    rho, p_rho = spearmanr(np.arange(K), pos)
    tau, p_tau = kendalltau(np.arange(K), pos)
    disp = np.abs(pos - np.arange(K))
    adj_kept = np.mean(np.abs(np.diff(pos)) == 1)
    same_fam_adj = np.mean([fam[k] == fam[k + 1] for k in range(K - 1)])
    # flipping a subtree is free in a dendrogram, so also report the orientation-invariant
    # neighbourhood measure: rastermap neighbours within 3 positions in leaf order
    near = np.mean(np.abs(np.diff(pos)) <= 3)
    res = dict(leaf_order_flipped_to_match_rastermap=bool(flipped), spearman_rho_raw_orientation=rho_raw,
               spearman_rho=rho, spearman_p=p_rho, kendall_tau=tau, kendall_p=p_tau,
               mean_abs_displacement=float(disp.mean()), median_abs_displacement=float(np.median(disp)),
               max_abs_displacement=int(disp.max()), frac_rastermap_neighbours_adjacent_in_leaf_order=adj_kept,
               frac_rastermap_neighbours_within3_in_leaf_order=near,
               frac_rastermap_neighbours_same_family=same_fam_adj, n_clusters=K,
               n_families=len(set(fam.values())))
    pd.DataFrame(dict(cluster=np.arange(K) + 1, rastermap_position=np.arange(K), leaf_position=pos,
                      family=[fam[k] for k in range(K)], abs_displacement=disp)).to_csv(
        out_dir / "reordering_per_cluster.csv", index=False)
    json.dump({k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in res.items()},
              open(out_dir / "reordering_summary.json", "w"), indent=2)
    colors = family_palette(fam.values())
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), dpi=200)
    axes[0].scatter(np.arange(K), pos, c=[colors[fam[k]] for k in range(K)], s=14, edgecolors="k", linewidths=0.2)
    axes[0].plot([0, K], [0, K], color="0.6", ls="--", lw=0.6)
    axes[0].set_xlabel("Rastermap cluster position"); axes[0].set_ylabel("Metacluster dendrogram leaf position")
    axes[0].set_title(f"Spearman ρ={rho:.2f}, Kendall τ={tau:.2f}", fontsize=10)
    axes[1].hist(disp, bins=20, color="k")
    axes[1].set_xlabel("|displacement| (positions)"); axes[1].set_ylabel("n clusters")
    axes[1].set_title(f"mean {disp.mean():.1f}, median {np.median(disp):.0f}; neighbours adjacent "
                      f"{adj_kept:.0%}, same family {same_fam_adj:.0%}", fontsize=7)
    for ax in axes:
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    rmu._save(fig, out_dir / "reordering_rastermap_vs_metacluster", dpi=200)
    return res


def roc_category_psth_sanity(R, roc, cfg, out_dir, n_examples=12, seed=0):
    """Per ROC category: mean ± SEM PSTH (rastermap even-trial matrix) for significant neurons
    split by selectivity sign vs non-significant neurons, per condition; plus example neurons."""
    rng = np.random.default_rng(seed)
    X, nb, t_ctrs = R["X"], R["n_bins_list"], R["t_ctrs"]
    offs = np.r_[0, np.cumsum(nb)]
    labels_c = list(rmu.COND_LABELS)
    for cat, spec in ROC_CATEGORIES.items():
        s = roc[f"{cat}__sig"].values; v = roc[f"{cat}__sel"].values
        valid = np.isfinite(s.astype(float))
        sig = valid & (s == 1)
        groups = {"non-significant": valid & ~sig}
        if spec.get("split"):
            groups["whisker-pref."] = roc[f"{cat}__whisker"].values == 1
            groups["auditory-pref."] = roc[f"{cat}__auditory"].values == 1
        else:
            groups["significant, sel>0"] = sig & (v > 0)
            groups["significant, sel<0"] = sig & (v < 0)
            if cat == "gated_decision":
                groups["significant, sel>0"] = sig & (v > 0)
        gcol = {"non-significant": "0.55", "significant, sel>0": "crimson", "significant, sel<0": "royalblue",
                "whisker-pref.": MODALITY_COLORS["whisker"], "auditory-pref.": MODALITY_COLORS["auditory"]}
        nC = len(nb)
        fig, axes = plt.subplots(2, nC, figsize=(2.5 * nC, 5.8), dpi=200, sharey="row")
        for ci in range(nC):
            seg = slice(offs[ci], offs[ci + 1]); t = t_ctrs[ci]
            ax = axes[0, ci]
            for gname, m in groups.items():
                if m.sum() < 2:
                    continue
                mu = X[m, seg].mean(0); se = X[m, seg].std(0) / np.sqrt(m.sum())
                ax.plot(t, mu, color=gcol[gname], lw=0.9, label=f"{gname} (n={int(m.sum())})")
                ax.fill_between(t, mu - se, mu + se, color=gcol[gname], alpha=0.25, lw=0)
            ax.axvline(0, color="k", lw=0.4, ls=":")
            ax.set_title(labels_c[ci] if ci < len(labels_c) else f"cond {ci}", fontsize=10)
            ax.tick_params(labelsize=8)
            ex = rng.choice(np.flatnonzero(sig), size=min(n_examples, sig.sum()), replace=False) if sig.sum() else []
            for i in ex:
                axes[1, ci].plot(t, X[i, seg], lw=0.4, alpha=0.7)
            axes[1, ci].axvline(0, color="k", lw=0.4, ls=":")
            axes[1, ci].tick_params(labelsize=8)
            axes[1, ci].set_xlabel("Time (s)", fontsize=9)
        axes[0, 0].set_ylabel("mean ± SEM", fontsize=10)
        axes[1, 0].set_ylabel(f"{n_examples} example\nsignificant neurons", fontsize=10)
        axes[0, -1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        if spec["rule"] == "groups":
            types = " & ".join("(" + " | ".join(g) + ")" for g in spec["groups"])
        elif spec["rule"] == "all_same_sign":
            types = " & ".join(spec["types"]) + " (same sign)"
        else:
            types = " & ".join(spec["types"]) if spec["rule"] == "all" else " | ".join(spec["types"])
        excl = f"  NOT ({' | '.join(spec['exclude'])})" if spec.get("exclude") else ""
        fig.suptitle(f"ROC category '{cat}': {types}{excl}  —  {int(sig.sum())}/{int(valid.sum())} neurons "
                     f"significant ({sig.sum() / max(valid.sum(), 1):.1%})", fontsize=12)
        for a in axes.flat:
            for sp in ("top", "right"): a.spines[sp].set_visible(False)
        fig.tight_layout()
        rmu._save(fig, out_dir / f"sanity_psth_{cat}", dpi=200)


# ══════════════════════════════════════════════════════════════════════════
# entry point
# ══════════════════════════════════════════════════════════════════════════

def run_matrix_summary(method_dir, config_path, cfg_overrides=None, roc_root=DEFAULT_ROC_ROOT,
                       columns=DEFAULT_COLUMNS, title_prefix="", families_csv=None, out_name="matrix_summary",
                       family_label=None, linkage_npz=None, meta_method=None, meta_cut=None):
    """All figure variants for one rastermap output folder → <method_dir>/<out_name>/.

    families_csv: optional csv with columns (cluster [1-based], family) — e.g. contiguity-constrained
    segments. When given, the strip shows these families instead of run_meta_clustering's, only the
    rastermap order is drawn (no dendrogram / reordering), and the reordering assessment is skipped.
    linkage_npz: optional npz with a scipy linkage matrix "Z" (+ "cut") over the same clusters (e.g. the
    contiguity-constrained tree); with it, dendrogram-order figures and the reordering assessment are
    produced for these families as well."""
    method_dir = Path(method_dir)
    out_dir = method_dir / out_name
    out_dir.mkdir(exist_ok=True)
    cfg = load_cfg(config_path, **(cfg_overrides or {}))
    (rmu.CONDITIONS, rmu.COND_LABELS, rmu.COND_COLORS,
     rmu.COND_LABELS_MATRIX, rmu.COND_ALIGN_COLS) = rmu.get_conditions(cfg)

    R = load_rastermap(method_dir)
    K = int(R["labels"].max()) + 1
    roc = build_roc_categories(method_dir, R["unit_ids"], roc_root)
    if families_csv is None:
        meta = metaclustering(method_dir, method=meta_method, cut=meta_cut)
        # contiguous metaclusters: dendrogram order == rastermap order, so only rastermap-order figures
        orders = ["rastermap"] if meta["method"].startswith("contiguous") else ["rastermap", "metacluster"]
        if family_label is None:
            family_label = (f"Metacluster (contiguous, 1-r<={meta['cut']:.2f})" if meta["method"].startswith("contiguous")
                            else "Metacluster")
    else:
        fdf = pd.read_csv(families_csv)
        lz = np.load(linkage_npz) if linkage_npz is not None else None
        meta = dict(Z=None if lz is None else lz["Z"], clusters=np.arange(K),
                    cut=None if lz is None else float(lz["cut"]),
                    families={int(c) - 1: int(f) for c, f in zip(fdf.cluster, fdf.family)})
        orders = ["rastermap"] if lz is None else ["rastermap", "metacluster"]
        family_label = family_label or "Segment"
    fmk = load_fmk_stats(method_dir)
    print(f"  {len(R['X'])} neurons, {K} clusters, {len(set(meta['families'].values()))} metaclusters, "
          f"ROC matched {int(roc.roc_matched.sum())}", flush=True)

    # per-neuron / per-cluster tables (joinable on the unit key)
    keys = pd.read_csv(method_dir / "neuron_cluster_labels_cv.csv").rename(columns={"unit_id": "unit_ids"})
    neuron_tab = keys.merge(roc, on="unit_ids", how="left", validate="one_to_one")
    neuron_tab["metacluster"] = neuron_tab.cluster_label.map(meta["families"])
    neuron_tab.to_csv(out_dir / "neuron_roc_categories.csv", index=False)
    agg = cluster_aggregates(R, roc, K)
    ct = pd.DataFrame({"cluster": np.arange(K) + 1, "n_neurons": np.bincount(R["labels"], minlength=K),
                       "metacluster": [meta["families"][k] for k in range(K)]})
    for cat in ROC_CATEGORIES:
        ct[f"{cat}__frac_sig"] = [agg["frac"][cat][k] for k in range(K)]
        ct[f"{cat}__mean_sel"] = [agg["sel"][cat][k] for k in range(K)]
        ct[f"{cat}__mean_abs_sel"] = [agg["abssel"][cat][k] for k in range(K)]
    if "n_decision" in agg:  # gated columns above are WITHIN decision neurons
        ct["n_decision_neurons"] = [agg["n_decision"][k] for k in range(K)]
        ct = ct.rename(columns={"gated_decision__frac_sig": "gated_decision__frac_of_decision",
                                "gated_decision__mean_abs_sel": "gated_decision__mean_abs_sel_decision_nrns"})
    ct.to_csv(out_dir / "cluster_roc_categories.csv", index=False)
    json.dump({c: {k: v for k, v in s.items()} for c, s in ROC_CATEGORIES.items()},
              open(out_dir / "roc_category_definitions.json", "w"), indent=2)

    if "metacluster" in orders:   # contiguous metaclusters: dendrogram order == rastermap order, nothing to assess
        reordering_assessment(meta, K, out_dir)
    roc_category_psth_sanity(R, roc, cfg, out_dir)

    for row_mode, order, roc_mode, reward_panel in itertools.product(
            ["neuron", "cluster_mean_uniform"], orders,   # no height-proportional cluster rows (user, 2026-09-27)
            ["fraction", "selectivity"], ["fmk_share", "figS4"]):
        name = f"matrix_{row_mode}_{order}_roc-{roc_mode}_{reward_panel}"
        fig5_population_matrix_ext(R, roc, meta, fmk, cfg, out_dir / name, columns=columns, row_mode=row_mode,
                                   order=order, roc_mode=roc_mode, reward_panel=reward_panel,
                                   title_prefix=title_prefix, family_label=family_label)
    # publication layout (rastermap order; needs contiguous families + tree to show the dendrogram)
    names, name_tab = family_names_psth(R, meta["families"])
    name_tab.to_csv(out_dir / "metacluster_family_names.csv", index=False)
    for row_mode in ["cluster_mean_uniform", "neuron"]:
        fig5_population_matrix_publication(R, roc, meta, fmk, cfg, out_dir / f"pub_matrix_{row_mode}_roc-fraction",
                                           row_mode=row_mode, title=title_prefix.rstrip(": ") or None)
    fig_metacluster_psth_summary(R, roc, meta, fmk, cfg, out_dir / "pub_metacluster_psth_summary", names=names)
    return out_dir
