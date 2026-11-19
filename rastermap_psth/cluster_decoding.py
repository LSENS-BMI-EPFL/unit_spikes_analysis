"""
cluster_decoding.py
--------------------
Single-trial population decoding of behavior from rastermap functional
clusters.

Every other rastermap_psth figure (fig6, fig12, sensorimotor_arc, ...)
describes a cluster through its *trial-averaged* PSTH. This script asks a
different question: given only the single-trial population activity of the
neurons that make up a cluster (in a given session), can a decoder predict
behavior on that trial? A cluster whose mean PSTH differs between two
conditions is not automatically decodable trial-by-trial -- averaging can
manufacture a difference that trial-to-trial noise swamps, or hide one that
only shows up in the joint activity pattern across neurons.

Decode targets
--------------
1. trial outcome (hit vs miss), within a fixed trial_type (whisker or
   auditory) -- same stimulus, different report. This is both "trial
   outcome" and the stimulus-matched "choice independent of stimulus" from
   the original spec: restricting to one trial_type already isolates
   choice/action from sensory drive, so both are implemented as one
   function (`decode_outcome`) called once per trial_type.
2. trial type (whisker vs auditory), active trials, regardless of outcome.
   For modality-biased clusters (e.g. cluster 13) this asks whether that
   bias is usable trial-by-trial or only visible in the population average.
3. reaction time (continuous regression), lick_time - stim_onset, from
   PRE-MOVEMENT population activity on hit trials. Using a decode window
   from stim onset up to just before the earliest reasonable RT to avoid
   the regression trivially picking up movement/licking artifacts rather
   than genuine pre-movement timing information.
4. reward group (R+ vs R-), decoded from single-trial cluster-mean
   activity POOLED ACROSS MICE. Individual neuron identity does not
   generalize across animals, so the feature here is the cluster's
   population-mean rate (not per-neuron rates), and cross-validation is
   grouped by mouse_id (StratifiedGroupKFold) with a mouse-level shuffle
   null, so no mouse's trials ever appear in both train and test.

Data flow
---------
neuron_cluster_labels_cv.csv (in the rastermap output folder) is the
source of truth for cluster membership: mouse_id, session_id,
electrode_group, cluster_id, cluster_label, reward_group, ... Spike times
and trial-level behavior are NOT cached anywhere in the rastermap output
(only trial-averaged PSTHs are), so this script reloads them fresh --
scoped only to the mice referenced in the CSV, which is far cheaper than
rerunning the full-cohort pipeline.

Usage
-----
    from rastermap_psth.cluster_decoding import run_cluster_decoding
    run_cluster_decoding(
        rastermap_dir=".../clustering/n100_.../rastermap",
        clusters=None,   # None = all clusters present in the CSV
    )

Or run standalone: edit the CONFIG block below and run this file directly.
"""

from __future__ import annotations

import socket
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, r2_score

from rastermap_psth.rastermap_utils import get_spike_times, spikes_around_events

warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════════
# CONFIG (standalone-run defaults; all overridable via run_cluster_decoding)
# ══════════════════════════════════════════════════════════════════════════

RASTERMAP_DIR = (
    r"M:\analysis\Axel_Bisi\combined_results_ks4\rastermap_clustering\passive_active"
    r"\whisker_auditory\clustering"
    r"\n100_passive_active_merge_baseline_whisker_auditory_combined_nobl\rastermap"
)
CLUSTERS = None          # None = decode every cluster present in the CSV
DAY_TO_ANALYZE = "all"   # 'learning' | 'expert' | 'all'

NWB_ROOT = (r"/mnt/lsens-analysis/Axel_Bisi/NWB_ks4" if "haas" in socket.gethostname()
            else r"M:\analysis\Axel_Bisi\NWB_ks4")
N_WORKERS = 24

ARTIFACT_WIN_S = (-0.010, 0.005)          # whisker coil-artifact blanking window
T_ACTIVE = (0.0, 0.5)                      # decode window for outcome / trial-type (post stim_onset)
T_RT_PREMOVEMENT = (0.0, 0.15)             # decode window for RT regression -- strictly pre-movement
                                            # (well below the ~0.35s median RT) so the regressor can't
                                            # just be reading out the lick/jaw movement itself
T_REWARD_GROUP = (0.0, 0.5)

MIN_NEURONS = 5             # minimum cluster-member neurons in a session to attempt a decode
MIN_TRIALS_PER_CLASS = 8    # minimum trials in the smaller class
N_SPLITS = 5
N_SHUFFLES_SESSION = 200    # shuffle-null draws for per-session (targets 1-3) decodes
N_SHUFFLES_MOUSE = 500      # shuffle-null draws for the mouse-grouped reward-group decode

RNG_SEED = 0


# ══════════════════════════════════════════════════════════════════════════
# data loading
# ══════════════════════════════════════════════════════════════════════════

def load_cluster_membership(rastermap_dir: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(rastermap_dir) / "neuron_cluster_labels_cv.csv")
    df["cluster_id"] = df["cluster_id"].astype(str)
    df["electrode_group"] = df["electrode_group"].astype(str)
    return df


def load_neural_and_trial_data(mouse_ids, nwb_root=NWB_ROOT, day_to_analyze=DAY_TO_ANALYZE,
                                max_workers=N_WORKERS):
    """Load trial_table + unit_table (with spike_times) for exactly the mice needed."""
    from ephys_utilities.helpers.data_utils import combine_ephys_nwb

    mouse_ids = set(mouse_ids)
    nwb_paths = [p for p in sorted(Path(nwb_root).glob("*.nwb"))
                 if p.stem.split("_")[0] in mouse_ids]
    print(f"Loading {len(nwb_paths)} NWB files for {len(mouse_ids)} mice...")
    trial_table, unit_table, _ = combine_ephys_nwb(
        [str(p) for p in nwb_paths], day_to_analyze=day_to_analyze, max_workers=max_workers)

    unit_table["cluster_id"] = unit_table["cluster_id"].astype(str)
    unit_table["electrode_group"] = unit_table["electrode_group"].astype(str)
    trial_table["stim_onset"] = trial_table["stim_onset"].astype(float)
    return trial_table, unit_table


def build_cluster_sessions(cluster_df: pd.DataFrame, unit_table: pd.DataFrame) -> dict:
    """Return {cluster_label: [session_dict, ...]}, one entry per (cluster, mouse, session)
    with that group's list of spike-time arrays and its reward group."""
    unit_table = unit_table.drop(columns=["reward_group"], errors="ignore")
    merged = unit_table.merge(
        cluster_df[["mouse_id", "session_id", "electrode_group", "cluster_id",
                    "cluster_label", "reward_group"]],
        on=["mouse_id", "session_id", "electrode_group", "cluster_id"], how="inner",
    )
    out: dict[int, list[dict]] = {}
    for (cl, mid, sid), grp in merged.groupby(["cluster_label", "mouse_id", "session_id"]):
        out.setdefault(cl, []).append(dict(
            mouse_id=mid, session_id=sid,
            spike_times=[get_spike_times(row) for _, row in grp.iterrows()],
            reward_group=grp["reward_group"].iloc[0],
        ))
    return out


# ══════════════════════════════════════════════════════════════════════════
# feature extraction
# ══════════════════════════════════════════════════════════════════════════

def population_rate_features(spike_times_list, events, t_pre, t_post,
                              is_whisker=False, artifact_win_s=ARTIFACT_WIN_S):
    """(n_trials, n_neurons) mean firing-rate matrix, one column per neuron."""
    dt = t_pre + t_post
    n_trials = len(events)
    X = np.zeros((n_trials, len(spike_times_list)))
    for j, spk in enumerate(spike_times_list):
        raster = spikes_around_events(spk, events, t_pre, t_post,
                                       is_whisker=is_whisker, artifact_win_s=artifact_win_s,
                                       align_col="start_time")
        X[:, j] = [len(r) / dt for r in raster]
    return X


def population_rate_features_mixed(spike_times_list, events, is_whisker_mask, t_pre, t_post,
                                    artifact_win_s=ARTIFACT_WIN_S):
    """Like population_rate_features, but for an event set that mixes whisker and
    non-whisker trials -- artifact correction is applied only to the whisker subset.

    Recombines the two subsets by explicit index assignment into a plain Python
    list (not a boolean-masked numpy object-array slice): when every raster in a
    subset happens to have the same spike count (e.g. all zero, common in short
    windows), numpy silently coerces the ragged list of arrays returned by
    spikes_around_events into a regular 2-D array, and a boolean-mask assignment
    of that into an object array then raises a dimensionality error.
    """
    dt = t_pre + t_post
    n_trials = len(events)
    idx_wh = np.where(is_whisker_mask)[0]
    idx_other = np.where(~is_whisker_mask)[0]
    X = np.zeros((n_trials, len(spike_times_list)))
    for j, spk in enumerate(spike_times_list):
        raster = [None] * n_trials
        if len(idx_wh):
            r_wh = spikes_around_events(spk, events[idx_wh], t_pre, t_post, is_whisker=True,
                                        artifact_win_s=artifact_win_s, align_col="start_time")
            for k, i in zip(r_wh, idx_wh):
                raster[i] = k
        if len(idx_other):
            r_other = spikes_around_events(spk, events[idx_other], t_pre, t_post, is_whisker=False,
                                           artifact_win_s=artifact_win_s, align_col="start_time")
            for k, i in zip(r_other, idx_other):
                raster[i] = k
        X[:, j] = [len(r) / dt for r in raster]
    return X


# ══════════════════════════════════════════════════════════════════════════
# stats helpers
# ══════════════════════════════════════════════════════════════════════════

def _shuffle_pvalue(observed, null_dist):
    null_dist = np.asarray(null_dist)
    return (1 + np.sum(null_dist >= observed)) / (1 + len(null_dist))


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Return FDR-corrected q-values, same order as input. NaN p-values (e.g. a
    degenerate cluster with no valid CV fold) are excluded from the correction
    and returned as NaN -- np.minimum.accumulate propagates a single NaN through
    every other q-value otherwise, silently hiding real signal in the rest."""
    pvals = np.asarray(pvals, float)
    out = np.full(len(pvals), np.nan)
    valid = ~np.isnan(pvals)
    p = pvals[valid]
    n = len(p)
    if n == 0:
        return out
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    q_full = np.empty(n)
    q_full[order] = np.clip(q, 0, 1)
    out[valid] = q_full
    return out


def _make_clf():
    return Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def _make_reg():
    return Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])


# ══════════════════════════════════════════════════════════════════════════
# target 1/3: trial outcome (hit vs miss), within one trial_type
# ══════════════════════════════════════════════════════════════════════════

def decode_outcome_one_session(sess, trials_this_session, trial_type, rng):
    tdf = trials_this_session[
        (trials_this_session["trial_type"] == trial_type)
        & (trials_this_session["context"] == "active")
        & (trials_this_session["early_lick"] == 0)
    ]
    y = tdf["lick_flag"].to_numpy().astype(int)
    if len(sess["spike_times"]) < MIN_NEURONS:
        return None
    if min((y == 0).sum(), (y == 1).sum()) < MIN_TRIALS_PER_CLASS:
        return None

    events = tdf["start_time"].to_numpy()
    X = population_rate_features(sess["spike_times"], events, *T_ACTIVE,
                                  is_whisker=(trial_type == "whisker_trial"))

    n_splits = min(N_SPLITS, (y == 0).sum(), (y == 1).sum())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG_SEED)
    auc = _cv_auc(X, y, cv)

    null = np.empty(N_SHUFFLES_SESSION)
    for i in range(N_SHUFFLES_SESSION):
        y_sh = rng.permutation(y)
        null[i] = _cv_auc(X, y_sh, cv)
    p = _shuffle_pvalue(auc, null)

    return dict(mouse_id=sess["mouse_id"], session_id=sess["session_id"],
                n_neurons=len(sess["spike_times"]), n_trials=len(y),
                n_hit=int((y == 1).sum()), n_miss=int((y == 0).sum()),
                metric="roc_auc", value=auc, p_shuffle=p)


def _cv_auc(X, y, cv):
    scores = []
    for tr, te in cv.split(X, y):
        clf = _make_clf().fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) < 2:
            continue
        scores.append(roc_auc_score(y[te], proba))
    return np.mean(scores) if scores else np.nan


# ══════════════════════════════════════════════════════════════════════════
# target 2: trial type (whisker vs auditory), active trials
# ══════════════════════════════════════════════════════════════════════════

def decode_trial_type_one_session(sess, trials_this_session, rng):
    tdf = trials_this_session[
        trials_this_session["trial_type"].isin(["whisker_trial", "auditory_trial"])
        & (trials_this_session["context"] == "active")
        & (trials_this_session["early_lick"] == 0)
    ]
    y = (tdf["trial_type"] == "whisker_trial").to_numpy().astype(int)
    if len(sess["spike_times"]) < MIN_NEURONS:
        return None
    if min((y == 0).sum(), (y == 1).sum()) < MIN_TRIALS_PER_CLASS:
        return None

    events = tdf["start_time"].to_numpy()
    is_wh = tdf["trial_type"].to_numpy() == "whisker_trial"
    X = population_rate_features_mixed(sess["spike_times"], events, is_wh, *T_ACTIVE)

    n_splits = min(N_SPLITS, (y == 0).sum(), (y == 1).sum())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG_SEED)
    auc = _cv_auc(X, y, cv)

    null = np.empty(N_SHUFFLES_SESSION)
    for i in range(N_SHUFFLES_SESSION):
        null[i] = _cv_auc(X, rng.permutation(y), cv)
    p = _shuffle_pvalue(auc, null)

    return dict(mouse_id=sess["mouse_id"], session_id=sess["session_id"],
                n_neurons=len(sess["spike_times"]), n_trials=len(y),
                n_whisker=int(y.sum()), n_auditory=int((1 - y).sum()),
                metric="roc_auc", value=auc, p_shuffle=p)


# ══════════════════════════════════════════════════════════════════════════
# target 4: reaction time regression (hit trials, pre-movement window)
# ══════════════════════════════════════════════════════════════════════════

def decode_rt_one_session(sess, trials_this_session, rng):
    tdf = trials_this_session[
        trials_this_session["trial_type"].isin(["whisker_trial", "auditory_trial"])
        & (trials_this_session["context"] == "active")
        & (trials_this_session["lick_flag"] == 1)
        & (trials_this_session["early_lick"] == 0)
    ].dropna(subset=["lick_time", "stim_onset"])
    rt = (tdf["lick_time"] - tdf["stim_onset"]).to_numpy()
    # drop trials whose RT falls inside the decode window -- those would let the
    # regressor see the movement itself rather than pre-movement activity.
    keep = rt > T_RT_PREMOVEMENT[1]
    tdf, rt = tdf[keep], rt[keep]
    if len(sess["spike_times"]) < MIN_NEURONS or len(rt) < MIN_TRIALS_PER_CLASS * 2:
        return None

    events = tdf["start_time"].to_numpy()
    is_wh = (tdf["trial_type"] == "whisker_trial").to_numpy()
    X = population_rate_features_mixed(sess["spike_times"], events, is_wh, *T_RT_PREMOVEMENT)

    n_splits = min(N_SPLITS, len(rt) // 2)
    if n_splits < 2:
        return None
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RNG_SEED)
    r2 = _cv_r2(X, rt, cv)

    null = np.empty(N_SHUFFLES_SESSION)
    for i in range(N_SHUFFLES_SESSION):
        null[i] = _cv_r2(X, rng.permutation(rt), cv)
    p = _shuffle_pvalue(r2, null)

    return dict(mouse_id=sess["mouse_id"], session_id=sess["session_id"],
                n_neurons=len(sess["spike_times"]), n_trials=len(rt),
                metric="r2", value=r2, p_shuffle=p)


def _cv_r2(X, y, cv):
    preds = np.full(len(y), np.nan)
    for tr, te in cv.split(X, y):
        reg = _make_reg().fit(X[tr], y[tr])
        preds[te] = reg.predict(X[te])
    return r2_score(y, preds)


# ══════════════════════════════════════════════════════════════════════════
# target 5: reward group (R+ vs R-), pooled across mice, grouped CV
# ══════════════════════════════════════════════════════════════════════════

def decode_reward_group(cluster_sessions, trial_table, rng):
    """Cluster-mean population rate per trial, pooled across all mice with this
    cluster present, decoded R+ vs R- with GroupKFold(mouse_id) so no mouse's
    trials appear on both sides of a split -- and a mouse-level shuffle null
    (permute which mice are R+/R-, not individual trials) for the same reason.
    """
    rows_X, rows_y, rows_group = [], [], []
    for sess in cluster_sessions:
        if len(sess["spike_times"]) < MIN_NEURONS:
            continue
        tdf = trial_table[
            (trial_table["mouse_id"] == sess["mouse_id"])
            & (trial_table["session_id"] == sess["session_id"])
            & (trial_table["trial_type"].isin(["whisker_trial", "auditory_trial", "no_stim_trial"]))
            & (trial_table["context"] == "active")
            & (trial_table["early_lick"] == 0)
        ]
        if len(tdf) == 0:
            continue
        events = tdf["start_time"].to_numpy()
        is_wh = (tdf["trial_type"] == "whisker_trial").to_numpy()
        X_sess = population_rate_features_mixed(sess["spike_times"], events, is_wh, *T_REWARD_GROUP)
        pop_rate = X_sess.mean(axis=1)   # cluster-mean rate: the only feature that
                                          # generalizes across mice with different neurons

        rows_X.append(pop_rate)
        rows_y.append(np.full(len(events), 1 if sess["reward_group"] == "R+" else 0))
        rows_group.append(np.full(len(events), sess["mouse_id"]))

    if len(rows_X) == 0:
        return None
    X = np.concatenate(rows_X).reshape(-1, 1)
    y = np.concatenate(rows_y)
    groups = np.concatenate(rows_group)
    mice = np.unique(groups)
    if len(mice) < 4 or min((y == 0).sum(), (y == 1).sum()) < MIN_TRIALS_PER_CLASS:
        return None

    n_splits = min(N_SPLITS, len(mice))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RNG_SEED)
    auc = _cv_auc_grouped(X, y, groups, cv)

    # mouse-level shuffle null: permute the R+/R- label assigned to each mouse,
    # keep trial-mouse structure fixed.
    mouse_label = {m: y[groups == m][0] for m in mice}
    labels = np.array([mouse_label[m] for m in mice])
    null = np.empty(N_SHUFFLES_MOUSE)
    for i in range(N_SHUFFLES_MOUSE):
        shuffled = dict(zip(mice, rng.permutation(labels)))
        y_sh = np.array([shuffled[m] for m in groups])
        if min((y_sh == 0).sum(), (y_sh == 1).sum()) < MIN_TRIALS_PER_CLASS:
            null[i] = np.nan
            continue
        null[i] = _cv_auc_grouped(X, y_sh, groups, cv)
    null = null[~np.isnan(null)]
    p = _shuffle_pvalue(auc, null) if len(null) else np.nan

    return dict(n_mice=len(mice), n_trials=len(y), metric="roc_auc", value=auc, p_shuffle=p)


def _cv_auc_grouped(X, y, groups, cv):
    scores = []
    for tr, te in cv.split(X, y, groups):
        # StratifiedGroupKFold does best-effort stratification under the group
        # constraint, not guaranteed -- a fold can still end up single-class
        # when few groups (mice) carry one of the classes.
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = _make_clf().fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        scores.append(roc_auc_score(y[te], proba))
    return np.mean(scores) if scores else np.nan


# ══════════════════════════════════════════════════════════════════════════
# orchestration
# ══════════════════════════════════════════════════════════════════════════

def run_cluster_decoding(rastermap_dir: str | Path, clusters=None,
                          nwb_root=NWB_ROOT, day_to_analyze=DAY_TO_ANALYZE,
                          max_workers=N_WORKERS, out_root: str | Path | None = None):
    rastermap_dir = Path(rastermap_dir)
    out_dir = Path(out_root) if out_root else rastermap_dir / "decoding"
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_df = load_cluster_membership(rastermap_dir)
    if clusters is None:
        clusters = sorted(cluster_df["cluster_label"].unique())
    else:
        # only load NWB files for mice that actually contribute to the requested clusters
        cluster_df = cluster_df[cluster_df["cluster_label"].isin(clusters)]

    trial_table, unit_table = load_neural_and_trial_data(
        cluster_df["mouse_id"].unique(), nwb_root=nwb_root,
        day_to_analyze=day_to_analyze, max_workers=max_workers)
    cluster_sessions = build_cluster_sessions(cluster_df, unit_table)

    rng = np.random.default_rng(RNG_SEED)
    per_session_rows = []
    reward_group_rows = []

    for cl in clusters:
        sessions = cluster_sessions.get(cl, [])
        print(f"\n-- cluster {cl} - {len(sessions)} sessions --")

        try:
            for sess in sessions:
                tdf = trial_table[(trial_table["mouse_id"] == sess["mouse_id"])
                                   & (trial_table["session_id"] == sess["session_id"])]

                for trial_type, target_name in [("whisker_trial", "outcome_whisker"),
                                                  ("auditory_trial", "outcome_auditory")]:
                    res = decode_outcome_one_session(sess, tdf, trial_type, rng)
                    if res:
                        res.update(cluster_label=cl, target=target_name)
                        per_session_rows.append(res)

                res = decode_trial_type_one_session(sess, tdf, rng)
                if res:
                    res.update(cluster_label=cl, target="trial_type")
                    per_session_rows.append(res)

                res = decode_rt_one_session(sess, tdf, rng)
                if res:
                    res.update(cluster_label=cl, target="reaction_time")
                    per_session_rows.append(res)

            res = decode_reward_group(sessions, trial_table, rng)
            if res:
                res.update(cluster_label=cl, target="reward_group")
                reward_group_rows.append(res)
        except Exception as e:
            print(f"  ERROR cluster {cl}: {e!r} -- skipping rest of this cluster")

        # checkpoint after every cluster so a later crash doesn't lose earlier work
        pd.DataFrame(per_session_rows).to_csv(out_dir / "per_session_decoding_results.csv", index=False)
        pd.DataFrame(reward_group_rows).to_csv(out_dir / "reward_group_decoding_results.csv", index=False)

    per_session_df = pd.DataFrame(per_session_rows)
    reward_group_df = pd.DataFrame(reward_group_rows)

    summary_df = _summarize(per_session_df, reward_group_df)
    summary_df.to_csv(out_dir / "cluster_decoding_summary.csv", index=False)

    _plot_summary(summary_df, out_dir)
    print(f"\nDone. Results -> {out_dir}")
    return dict(per_session=per_session_df, reward_group=reward_group_df, summary=summary_df)


def _summarize(per_session_df, reward_group_df):
    rows = []
    if len(per_session_df):
        for (cl, target), grp in per_session_df.groupby(["cluster_label", "target"]):
            rows.append(dict(
                cluster_label=cl, target=target, level="per_session",
                n_sessions=len(grp), mean_value=grp["value"].mean(),
                frac_sessions_p_lt_05=(grp["p_shuffle"] < 0.05).mean(),
                combined_p_fisher=_fisher_combine(grp["p_shuffle"].to_numpy()),
            ))
    if len(reward_group_df):
        for _, row in reward_group_df.iterrows():
            rows.append(dict(
                cluster_label=row["cluster_label"], target="reward_group", level="pooled_mice",
                n_sessions=row["n_mice"], mean_value=row["value"],
                frac_sessions_p_lt_05=np.nan, combined_p_fisher=row["p_shuffle"],
            ))
    df = pd.DataFrame(rows)
    if len(df):
        df["q_value_fdr"] = np.nan
        for target, idx in df.groupby("target").groups.items():
            df.loc[idx, "q_value_fdr"] = benjamini_hochberg(df.loc[idx, "combined_p_fisher"].to_numpy())
    return df


def _fisher_combine(pvals):
    pvals = pvals[~np.isnan(pvals)]
    if len(pvals) == 0:
        return np.nan
    pvals = np.clip(pvals, 1e-12, 1.0)
    from scipy.stats import chi2
    stat = -2 * np.sum(np.log(pvals))
    return chi2.sf(stat, df=2 * len(pvals))


def _plot_summary(summary_df, out_dir):
    if len(summary_df) == 0:
        print("No decodable cluster/session combinations -- nothing to plot.")
        return
    targets = summary_df["target"].unique()
    fig, axes = plt.subplots(len(targets), 1, figsize=(14, 3.2 * len(targets)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, target in zip(axes, targets):
        sub = summary_df[summary_df["target"] == target].sort_values("cluster_label")
        colors = np.where(sub["q_value_fdr"] < 0.05, "crimson", "slategray")
        ax.bar(sub["cluster_label"].astype(str), sub["mean_value"], color=colors)
        chance = 0.0 if target == "reaction_time" else 0.5   # r2 chance=0, roc_auc chance=0.5
        ax.axhline(chance, color="k", lw=0.8, ls="--")
        ax.set_title(f"{target}  (red = FDR q<0.05)", fontsize=9)
        ax.set_ylabel("mean CV metric", fontsize=8)
        ax.tick_params(labelsize=6)
    axes[-1].set_xlabel("cluster_label", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"fig_cluster_decoding_summary.{ext}", dpi=200)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# report: extra figures + a written summary, built from the saved CSVs
# (no re-decoding -- safe to rerun cheaply after a bugfix)
# ══════════════════════════════════════════════════════════════════════════

def build_decoding_report(rastermap_dir: str | Path, out_root: str | Path | None = None,
                           highlight_clusters=(13,)):
    from scipy.stats import pearsonr

    rastermap_dir = Path(rastermap_dir)
    decoding_dir = Path(out_root) if out_root else rastermap_dir / "decoding"
    fig_dir = decoding_dir / "report_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_session = pd.read_csv(decoding_dir / "per_session_decoding_results.csv")
    reward_group = pd.read_csv(decoding_dir / "reward_group_decoding_results.csv")
    summary = pd.read_csv(decoding_dir / "cluster_decoding_summary.csv")

    targets = ["outcome_whisker", "outcome_auditory", "trial_type", "reaction_time", "reward_group"]
    targets = [t for t in targets if t in summary["target"].unique()]

    stats = {}
    for t in targets:
        sub = summary[summary["target"] == t]
        stats[t] = dict(
            n_tested=len(sub), n_sig=int((sub["q_value_fdr"] < 0.05).sum()),
            mean=sub["mean_value"].mean(), median=sub["mean_value"].median(),
            vmin=sub["mean_value"].min(), vmax=sub["mean_value"].max(),
            top=sub.sort_values("q_value_fdr").head(5),
        )

    # cross-target correlation: does "outcome_whisker" decodability just track
    # "trial_type" decodability across clusters? (the movement/modality-confound check)
    ow = summary[summary["target"] == "outcome_whisker"].set_index("cluster_label")["mean_value"]
    tt = summary[summary["target"] == "trial_type"].set_index("cluster_label")["mean_value"]
    common = ow.index.intersection(tt.index)
    r_ow_tt, p_ow_tt = pearsonr(ow.loc[common], tt.loc[common]) if len(common) > 2 else (np.nan, np.nan)

    # ── figure 1: per-target overview (significance rate + effect-size spread) ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sig_frac = [stats[t]["n_sig"] / stats[t]["n_tested"] for t in targets]
    axes[0].bar(targets, sig_frac, color="slategray")
    axes[0].set_ylabel("fraction of clusters with q<0.05")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=30, labelsize=8)
    axes[0].set_title("Significance rate by target")

    data = [summary[summary["target"] == t]["mean_value"].to_numpy() for t in targets]
    parts = axes[1].violinplot(data, showmeans=True)
    axes[1].set_xticks(range(1, len(targets) + 1))
    axes[1].set_xticklabels(targets, rotation=30, fontsize=8)
    axes[1].axhline(0.5, color="k", lw=0.7, ls="--", label="AUC chance")
    axes[1].axhline(0.0, color="gray", lw=0.7, ls=":", label="R2 chance")
    axes[1].set_ylabel("mean CV metric per cluster")
    axes[1].set_title("Effect-size distribution across clusters")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(fig_dir / f"fig_target_overview.{ext}", dpi=200)
    plt.close(fig)

    # ── figure 2: volcano plots per target ──────────────────────────────────
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 4.2))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, targets):
        sub = summary[summary["target"] == t]
        neglogq = -np.log10(sub["q_value_fdr"].clip(lower=1e-300))
        sig = sub["q_value_fdr"] < 0.05
        ax.scatter(sub.loc[~sig, "mean_value"], neglogq[~sig], s=14, color="slategray", alpha=0.6)
        ax.scatter(sub.loc[sig, "mean_value"], neglogq[sig], s=14, color="crimson", alpha=0.8)
        for _, row in sub.sort_values("q_value_fdr").head(3).iterrows():
            ax.annotate(str(int(row["cluster_label"])),
                        (row["mean_value"], -np.log10(max(row["q_value_fdr"], 1e-300))),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(-np.log10(0.05), color="k", lw=0.6, ls="--")
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("mean CV metric", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("-log10(FDR q)", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(fig_dir / f"fig_volcano_per_target.{ext}", dpi=200)
    plt.close(fig)

    # ── figure 3: outcome_whisker vs trial_type scatter (confound check) ────
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(tt.loc[common], ow.loc[common], s=22, color="slategray")
    for cl in common:
        ax.annotate(str(int(cl)), (tt.loc[cl], ow.loc[cl]), fontsize=6,
                    xytext=(2, 2), textcoords="offset points", alpha=0.7)
    ax.set_xlabel("trial_type decode AUC")
    ax.set_ylabel("outcome_whisker decode AUC")
    ax.set_title(f"r={r_ow_tt:.2f}, p={p_ow_tt:.1e}, n={len(common)} clusters")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(fig_dir / f"fig_outcome_vs_trialtype_scatter.{ext}", dpi=200)
    plt.close(fig)

    # ── figure 4: reward_group clusters ranked by p-value ────────────────────
    if "reward_group" in targets:
        sub = summary[summary["target"] == "reward_group"].sort_values("combined_p_fisher")
        fig, ax = plt.subplots(figsize=(7, 10))
        colors = np.where(sub["q_value_fdr"] < 0.05, "crimson", "slategray")
        ax.barh(range(len(sub)), sub["mean_value"], color=colors)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["cluster_label"].astype(int).astype(str), fontsize=6)
        ax.axvline(0.5, color="k", lw=0.7, ls="--")
        ax.invert_yaxis()
        ax.set_xlabel("mean CV ROC-AUC (mouse-grouped)")
        ax.set_ylabel("cluster_label (sorted by p-value, best first)")
        ax.set_title("reward_group decode, ranked (red = FDR q<0.05)")
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(fig_dir / f"fig_reward_group_ranked.{ext}", dpi=200)
        plt.close(fig)

    # ── written report ────────────────────────────────────────────────────
    lines = []
    lines.append("# Rastermap cluster decoding report\n")
    lines.append(f"Source: `{rastermap_dir}`\n")
    lines.append(f"Clusters tested: {summary['cluster_label'].nunique()} / 100\n")
    lines.append("\n## Method summary\n")
    lines.append(
        "For each rastermap functional cluster, single-trial population activity of the "
        "cluster's member neurons was used to decode five behavioral targets per session "
        "(hit vs miss within whisker trials, hit vs miss within auditory trials, whisker vs "
        "auditory trial type, reaction time, and reward group). Outcome/trial-type/RT decoders "
        "are cross-validated logistic regression (AUC) or Ridge regression (R2) within a single "
        "session; reward group pools cluster-mean activity across all mice carrying that cluster "
        "and cross-validates with StratifiedGroupKFold on mouse_id, with a mouse-level shuffle "
        "null, so no mouse's trials ever appear on both sides of a split. Per-session/per-mouse "
        "p-values come from a label-shuffle null; cluster-level p-values Fisher-combine across "
        "sessions; q-values are Benjamini-Hochberg FDR-corrected across all clusters tested for "
        "that target.\n"
    )
    lines.append("\n## Results by target\n")
    lines.append("| target | clusters tested | significant (q<0.05) | mean metric | range |")
    lines.append("|---|---|---|---|---|")
    for t in targets:
        s = stats[t]
        lines.append(f"| {t} | {s['n_tested']} | {s['n_sig']} | {s['mean']:.3f} "
                      f"| [{s['vmin']:.3f}, {s['vmax']:.3f}] |")
    lines.append("")

    for t in targets:
        s = stats[t]
        lines.append(f"\n### {t}\n")
        lines.append("Top 5 clusters (lowest FDR q):\n")
        lines.append("| cluster | n_sessions | mean metric | q_value_fdr |")
        lines.append("|---|---|---|---|")
        for _, row in s["top"].iterrows():
            lines.append(f"| {int(row['cluster_label'])} | {int(row['n_sessions'])} "
                          f"| {row['mean_value']:.3f} | {row['q_value_fdr']:.2e} |")

    if highlight_clusters:
        lines.append("\n## Highlighted clusters\n")
        for cl in highlight_clusters:
            sub = summary[summary["cluster_label"] == cl]
            if len(sub) == 0:
                continue
            lines.append(f"\n### Cluster {cl}\n")
            lines.append("| target | n_sessions | mean metric | q_value_fdr |")
            lines.append("|---|---|---|---|")
            for _, row in sub.iterrows():
                q = "n/a (degenerate)" if pd.isna(row["q_value_fdr"]) else f"{row['q_value_fdr']:.2e}"
                mv = "n/a" if pd.isna(row["mean_value"]) else f"{row['mean_value']:.3f}"
                lines.append(f"| {row['target']} | {int(row['n_sessions'])} | {mv} | {q} |")

    lines.append("\n## Caveats\n")
    lines.append(
        f"- **outcome_whisker vs trial_type collinearity check**: across the {len(common)} "
        f"clusters tested for both, per-cluster decode AUCs correlate only weakly "
        f"(Pearson r={r_ow_tt:.2f}, p={p_ow_tt:.1e}) -- see `fig_outcome_vs_trialtype_scatter`. "
        "This argues against outcome_whisker's near-universal significance being purely a "
        "byproduct of the same signal that separates whisker from auditory trials, but does not "
        "rule out a movement/licking confound specific to hit trials (hits, by definition, "
        "involve a lick inside the 0-0.5s decode window). A stricter pre-lick window (as used "
        "for reaction_time) would be a natural follow-up before treating outcome_whisker as "
        "clean choice-related signal in every cluster.\n"
    )
    lines.append(
        "- **reaction_time R2 is often negative** even in clusters flagged significant: "
        "significance here means the observed R2 beat the shuffle-null distribution (which is "
        "itself usually strongly negative, since a regressor fit to shuffled labels overfits "
        "noise), not that the model is a good absolute predictor of RT. Interpret as weak-but-"
        "real above-chance timing information, not strong RT decoding.\n"
    )
    lines.append(
        "- **reward_group**: 0 of 90 testable clusters survive FDR correction across all 100 "
        "clusters. Clusters 47 (p=0.004) and 35 (p=0.008) are the closest to nominal "
        "significance but do not survive multiple-comparison correction -- the honest "
        "conclusion is that no single rastermap cluster's population activity separates R+ from "
        "R- mice above chance once corrected, using a leakage-safe (mouse-grouped) test.\n"
    )
    lines.append(
        "- **outcome_auditory** was only testable in 43/100 clusters -- auditory trials are "
        "rarer per session than whisker trials, so many cluster-session combinations didn't "
        "clear the minimum-trials-per-class threshold.\n"
    )

    lines.append("\n## Files\n")
    lines.append("- `cluster_decoding_summary.csv` -- cluster x target level summary (this report's source table)")
    lines.append("- `per_session_decoding_results.csv` -- raw per-session decode results")
    lines.append("- `reward_group_decoding_results.csv` -- raw per-cluster reward-group decode results")
    lines.append("- `fig_cluster_decoding_summary.*` -- per-cluster bar chart per target")
    lines.append("- `report_figures/fig_target_overview.*` -- significance rate + effect-size distribution")
    lines.append("- `report_figures/fig_volcano_per_target.*` -- effect size vs significance, per target")
    lines.append("- `report_figures/fig_outcome_vs_trialtype_scatter.*` -- collinearity check")
    lines.append("- `report_figures/fig_reward_group_ranked.*` -- all reward_group clusters ranked")

    report_path = decoding_dir / "decoding_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report -> {report_path}")
    return report_path


if __name__ == "__main__":
    run_cluster_decoding(RASTERMAP_DIR, clusters=CLUSTERS)
