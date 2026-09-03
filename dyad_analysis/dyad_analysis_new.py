"""
Dyad CCG connectivity pipeline: naive/learned score-change analysis + figures.

Run with debug=True first (restricts to one/few mice, end-to-end) to confirm
column names / paths before launching the full 82-mouse run.

Performance notes:
- CCG arrays (ccgs*_nostim.npy) are huge (can be 100+ GB) and live on a
  network (M:) drive, where numpy's mmap_mode='r' is unreliable and can
  silently fall back to a full-file load. So they are never loaded as full
  arrays at all: process_session() determines connected pairs and score
  changes purely from the (tiny) score files, and make_session_figures()
  reads only the handful of specific rows it needs to plot directly off
  disk via NpyRowReader (seek + read, no mmap, no full load).
- Sessions are independent, so they're processed in parallel with a process
  pool. Tune N_WORKERS: for a network drive, I/O bandwidth -- not CPU -- is
  often the bottleneck, so test on a handful of sessions before committing
  to the full run.

LOAD_CCGS toggle:
- Set LOAD_CCGS = False to skip all CCG file access. Only score files are
  read; process_session() runs and score CSVs are written as usual, but
  make_session_figures() is never called, so NpyRowReader never opens a CCG
  file. Useful for isolating whether a failure comes from CCG reading or
  from something else (pairs_light.pkl, Excel parsing, etc). Flip to True
  to also produce the CCG figures.
"""
import os
import sys

import neural_utils_old

sys.path.append(r"M:\analysis\Axel_Bisi\Github\allen_utils")
import allen_utils
sys.path.append(r"M:\analysis\Axel_Bisi\NWB_reader")
import NWB_reader_functions

# Must be set before numpy/matplotlib import; avoids each worker process
# spinning up its own BLAS thread pool and oversubscribing CPU cores.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gc
import pickle
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")  # headless backend, required for parallel figure saving
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.format as npy_format
import pandas as pd

# ============================== CONFIG ======================================
DATA_ROOT = Path(r"M:\share_external\Dyad_collaboration\processed_data")
WEIGHT_XLSX = Path(r"M:\share_internal\Axel_Bisi_Share\dataset_info\joint_mouse_reference_weight.xlsx")
PROBE_XLSX = Path(r"M:\share_internal\Axel_Bisi_Share\dataset_info\joint_probe_insertion_info.xlsx")
OUTPUT_ROOT = Path(r"M:\analysis\Axel_Bisi\combined_results")
FIG_SUBDIR = "dyad"

CONNECTED_THRESH = -10.0
SCORE_TAG = {"E": "causal_piriall_v4_10x3",
             "I": "causal_piri_v2_10x3"}  # v4 for E, v2 for I (v4 not available for I)
# Score files are (3, N_pairs) -- one score per pair from each of 3 models.
# 'min' is conservative: if ANY of the 3 models flags a pair as unscoreable
# (sentinel -7788) or gives it a low score, the min preserves that signal
# instead of a mean diluting it toward the other models' real scores.
SCORE_ROW_REDUCE = "min"
SIGNS = ("E", "I")
# Two comparisons, both attempted for every session (whichever files exist
# for that session) -- NOT selected by day_type:
#   passive:    naive0 (passive_pre)      vs learned0 (passive_post)
#   inflection: naive1 (pre-inflection)   vs learned1 (post-inflection)
# day_type (day0 / learning_expert) is kept purely as a grouping label on
# the output rows for later summaries -- it no longer picks which files
# get compared.
COMPARISONS = {"passive": ("naive0", "learned0"), "inflection": ("naive1", "learned1")}

N_WORKERS = min(8, os.cpu_count() or 4)  # EDIT: tune for your machine / drive
LOAD_CCGS = True  # EDIT: True to also read CCG files and produce figures

# NWB-derived Allen-CCF labels, merged onto neurons.pkl (see
# merge_nwb_allen_labels). EDIT NWB_DIR / column names to match your setup.
NWB_DIR = Path(r"M:\analysis\Axel_Bisi\NWB_combined") # one {session_id}.nwb file per session
NEURONS_FIRING_RATE_COL = "pre_firing_rate"  # column in neurons.pkl
NWB_FIRING_RATE_COL = "firing_rate"      # column in the NWB unit table
FR_ROUND_DECIMALS = 15  # precision for the fuzzy firing-rate match key
# Column that allen_utils.process_allen_labels(subdivide_areas=True) is
# expected to ADD to the merged neurons+unit_table data (this is NOT a
# pre-existing neurons.pkl column) -- EDIT if the real output column name
# differs. merge_nwb_allen_labels checks this column actually appears and
# warns loudly if not, since AREA_SRC_CANDIDATES/AREA_TGT_CANDIDATES below
# rely on it to override area_relation.
CUSTOM_AREA_COL = "area_acronym_custom"

# EDIT these to match your actual Excel column headers
WEIGHT_COLS = dict(mouse="mouse_id", reward_group="reward_group")
PROBE_COLS = dict(mouse="mouse_name", date="date", day_of_recording="day_of_recording")

# Area columns used for area_source/area_target throughout (within/across
# classification, area-pair grouping, figure titles, etc). Checked in
# order -- pre_/post_{CUSTOM_AREA_COL} come from neurons.pkl+NWB+Allen
# processing (merged in load_session, see merge_nwb_allen_labels and
# _merge_neuron_metadata) and take priority when present, OVERRIDING
# area_relation to be computed from them instead of the pairs_light.pkl-
# native pre_acronym/post_acronym (used as fallback when unavailable).
AREA_SRC_CANDIDATES = (f"pre_{CUSTOM_AREA_COL}", "pre_acronym", "area_source", "source_area", "area_1")
AREA_TGT_CANDIDATES = (f"post_{CUSTOM_AREA_COL}", "post_acronym", "area_target", "target_area", "area_2")

# Within/across-area classification ignores layer info in the area acronym
# (e.g. Allen CCF style 'SSp-bfd4'/'SSp-bfd5' both count as area 'SSp-bfd').
# The regex requires 2+ letters immediately before a trailing digit suffix,
# so short whole-area codes with no layer info (e.g. 'S1', 'M1') are left
# untouched, while 'SSp-bfd4' -> 'SSp-bfd', 'VISp2/3' -> 'VISp',
# 'RSPagl2/3' -> 'RSPagl' (not 'RSPag') strip correctly.
#
# THIS HAS NOT BEEN VERIFIED AGAINST YOUR REAL pre_acronym/post_acronym
# VALUES -- only tested against plausible Allen CCF-style names. Run
# check_layer_stripping.py against a real session before trusting any
# within/across-area result. Set STRIP_LAYER_SUFFIX = False to fall back to
# plain string equality (no stripping) if this doesn't match your data.
STRIP_LAYER_SUFFIX = True
LAYER_SUFFIX_PATTERN = r"(?<=[A-Za-z]{2})\d+(?:/\d+)?[ab]?$"


def _first_present(row_like, candidates, default="unknown"):
    for c in candidates:
        if c in row_like:
            return row_like[c]
    return default


def _strip_layer_series(area_series):
    """Vectorized: remove a trailing cortical-layer suffix from an area
    acronym Series. See STRIP_LAYER_SUFFIX / LAYER_SUFFIX_PATTERN above for
    the convention assumed and the verification caveat."""
    area_series = area_series.astype(str)
    if not STRIP_LAYER_SUFFIX:
        return area_series
    return area_series.str.replace(LAYER_SUFFIX_PATTERN, "", regex=True)


def _get_area_columns(pairs_df):
    """Vectorized column-level pick (not the per-row _first_present loop):
    for a session-wide pairs_light.pkl (can be ~1M rows), assume the same
    candidate column is present for every row (true in practice -- schema
    doesn't vary row to row within a session) and just grab that whole
    column at once. Needed for computing within/across-area status over
    ALL pairs, not just the small connected subset."""
    def pick(candidates):
        for c in candidates:
            if c in pairs_df.columns:
                return pairs_df[c].astype(str)
        return pd.Series("unknown", index=pairs_df.index)
    return pick(AREA_SRC_CANDIDATES), pick(AREA_TGT_CANDIDATES)


def score_file(session_dir, sign, cond=None):
    suffix = f"_{cond}" if cond else ""
    score_tag = SCORE_TAG[sign]
    return session_dir / "rankings" / f"scores{sign}_{score_tag}{suffix}_nostim.npy"


def _reduce_score_rows(arr, method=SCORE_ROW_REDUCE):
    """Score files are shape (3, N_pairs), not (N_pairs,) as originally
    assumed -- one score per pair from each of 3 models/folds (filename tag
    '10x3' likely refers to this). Reduced to one score per pair via
    SCORE_ROW_REDUCE. Per-pair filtering downstream (CONNECTED_THRESH) is
    applied only to the reduced WHOLE-SESSION score, never to individual
    model rows or to score_pre/score_post directly.

    EDIT SCORE_ROW_REDUCE ('mean', 'median', 'min', or an int row index like
    0/1/2) if you confirm from Dyad's docs what the 3 rows actually represent.
    """
    if isinstance(method, int):
        return arr[method]
    if method == "mean":
        return arr.mean(axis=0)
    if method == "median":
        return np.median(arr, axis=0)
    if method == "min":
        return arr.min(axis=0)
    raise ValueError(f"Unknown SCORE_ROW_REDUCE: {method!r}")


class NpyRowReader:
    """Reads individual rows of a 2D .npy file via direct seeks.

    CCG files here are far too large to load fully (hundreds of GB), and
    mmap_mode='r' is unreliable over network (SMB) drives -- it can silently
    fall back to a full-file read and blow up memory exactly like a plain
    np.load would. This bypasses both problems: parse the .npy header once
    to get shape/dtype/data offset, then seek + read only the requested rows.
    Assumes a C-contiguous (row-major), non-pickled numeric array.

    Not used at all while LOAD_CCGS is False -- kept in place so flipping
    the flag back on doesn't require restoring any code.
    """

    def __init__(self, path):
        self.f = open(path, "rb")
        version = npy_format.read_magic(self.f)
        if version == (1, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_1_0(self.f)
        else:
            shape, fortran_order, dtype = npy_format.read_array_header_2_0(self.f)
        if fortran_order:
            raise ValueError(f"{path} is Fortran-ordered; row-wise reads assume C order")
        self.n_rows, self.n_cols = shape
        self.dtype = dtype
        self.row_bytes = self.n_cols * dtype.itemsize
        self.data_offset = self.f.tell()

    def __getitem__(self, idx):
        self.f.seek(self.data_offset + int(idx) * self.row_bytes)
        return np.fromfile(self.f, dtype=self.dtype, count=self.n_cols)

    def rows(self, indices):
        """Return {pair_id: row_array} for the given indices only."""
        return {int(i): self[i] for i in indices}

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ============================== MANIFEST ====================================
def _reformat_date(date_val):
    """'06.03.2025' (DD.MM.YYYY) -> '20250306'. Also handles real datetime/Timestamp cells."""
    if isinstance(date_val, str):
        d, m, y = date_val.strip().split(".")
        return f"{y}{m}{d}"
    return pd.Timestamp(date_val).strftime("%Y%m%d")


def build_manifest():
    """Cross-reference the two Excel files with sessions actually on disk.

    Session folders are named MOUSE_YYYYMMDD_HHMMSS. The probe file has no
    time component, so sessions are matched on (mouse, date) rather than an
    exact session_id -- this assumes at most one session per mouse per date.
    """
    weight = pd.read_excel(WEIGHT_XLSX)[[WEIGHT_COLS["mouse"], WEIGHT_COLS["reward_group"]]]
    weight.columns = ["mouse", "reward_group"]

    probe = pd.read_excel(PROBE_XLSX)[
        [PROBE_COLS["mouse"], PROBE_COLS["date"], PROBE_COLS["day_of_recording"]]
    ]
    probe.columns = ["mouse", "date", "day_of_recording"]
    probe["date_str"] = probe["date"].apply(_reformat_date)

    ref = probe.merge(weight, on="mouse", how="left")
    ref["day_type"] = np.where(ref.day_of_recording == 0, "day0", "learning_expert")

    on_disk = pd.DataFrame(
        [(p.name[:5], p.name[6:14], p.name, p) for p in DATA_ROOT.iterdir() if p.is_dir()],
        columns=["mouse", "date_str", "session_id", "session_path"],
    )
    ref = ref.merge(on_disk, on=["mouse", "date_str"], how="inner")
    return ref.reset_index(drop=True)


# ============================== LOADING =====================================
def load_neurons(session_dir):
    """Load neurons.pkl: one row per recorded unit, indexed positionally
    (row i = unit i) -- the SAME index space that pairs_light.pkl's
    preIdx/postIdx columns already reference. Returns None if the file
    doesn't exist for this session (e.g. older sessions predating it)."""
    neurons_path = session_dir / "neurons.pkl"
    if not neurons_path.exists():
        return None
    with open(neurons_path, "rb") as f:
        neurons = pickle.load(f)
    return neurons if isinstance(neurons, pd.DataFrame) else pd.DataFrame(neurons)


def _merge_neuron_metadata(pairs, neurons):
    """Merge neurons.pkl metadata onto pairs BY ROW INDEXING: neurons.pkl
    row i = unit i, the same index space pairs['preIdx']/['postIdx']
    already reference -- a positional .iloc lookup, NOT a column-based
    merge/join. Adds pre_{col}/post_{col} for every neurons.pkl column;
    any column already present in pairs (e.g. pre_acronym/pre_firing_rate,
    already there from pairs_light.pkl itself) is left untouched rather
    than silently overwritten by the neurons.pkl version.

    Raises if preIdx/postIdx reference a unit index beyond neurons.pkl's
    length -- that means the 'row i = unit i' assumption doesn't hold for
    this session, and silently proceeding would merge WRONG per-unit data
    onto pairs rather than fail loudly."""
    if neurons is None:
        return pairs

    max_idx = int(max(pairs["preIdx"].max(), pairs["postIdx"].max()))
    if max_idx >= len(neurons):
        raise ValueError(
            f"pairs_light.pkl references unit index {max_idx}, but neurons.pkl "
            f"only has {len(neurons)} rows -- the row-indexed merge assumption "
            f"('they match') doesn't hold for this session."
        )

    pairs = pairs.reset_index(drop=True)
    pre_meta = neurons.iloc[pairs["preIdx"].to_numpy()].reset_index(drop=True)
    post_meta = neurons.iloc[pairs["postIdx"].to_numpy()].reset_index(drop=True)

    for col in neurons.columns:
        pre_col, post_col = f"pre_{col}", f"post_{col}"
        if pre_col not in pairs.columns:
            pairs[pre_col] = pre_meta[col].to_numpy()
        if post_col not in pairs.columns:
            pairs[post_col] = post_meta[col].to_numpy()

    return pairs


def merge_nwb_allen_labels(neurons, session_id):
    """Steps 2+3 of the pipeline: (2) merge neurons.pkl onto the session's
    NWB unit table using firing rate, (3) run allen_utils.process_allen_
    labels on that merged result, which is expected to ADD the
    CUSTOM_AREA_COL column. Returns the enriched neurons DataFrame, still
    one row per unit in the SAME row order pairs_light.pkl's preIdx/postIdx
    index into (see _merge_neuron_metadata) -- this row alignment must
    survive both the merge and the allen_utils call, or every downstream
    preIdx/postIdx lookup silently breaks.

    There's no shared unit ID between neurons.pkl and the NWB unit table,
    so units are matched by ROUNDED firing rate (FR_ROUND_DECIMALS) as an
    imperfect proxy key. This is fragile -- ties/near-ties in firing rate
    can cause wrong or duplicate matches -- so two invariants are enforced
    rather than assumed:
      - the NWB unit table is de-duplicated on the rounded key first
        (keep='first'), so the merge can never inflate row count.
      - neurons.merge(unit_table, how='left') -- LEFT, with neurons as the
        base -- keeps every unit even when unmatched (NaN for the new
        columns) and preserves neurons' row order exactly.
    Row count is asserted unchanged after both the merge and the
    allen_utils call (defensive -- process_allen_labels is an external
    function whose row-dropping behavior, if any, isn't controlled here).

    Also checks that CUSTOM_AREA_COL actually appears in allen_utils'
    output -- step 5 (overriding area_relation) silently does nothing if
    this column never shows up, so that failure mode is made loud instead.

    Returns neurons unchanged (no NWB/Allen columns added) if no NWB file
    exists for this session."""
    nwb_file = NWB_DIR / f"{session_id}.nwb"
    if not nwb_file.exists():
        return neurons

    # Step 2: load neurons, merge onto the NWB unit table using firing rate
    unit_table = NWB_reader_functions.get_unit_table(nwb_file)
    unit_table = neural_utils.convert_electrode_group_object_to_columns(unit_table)
    neurons = neurons.copy()
    neurons['cluster_id'] = neurons.index
    neurons["cluster_id"] = neurons["cluster_id"].astype(str)
    neurons["fr_round"] = neurons["firing_rate"].round(FR_ROUND_DECIMALS).astype(str)
    neurons["ccf_atlas_acronym"] = neurons["structure_acronym"]
    neurons["waveformDuration_peakTrough"] = neurons["waveform_duration"].astype(str)
    neurons["fr_round"] = neurons["fr_round"].astype(str)
    neurons["waveformDuration_peakTrough"] = neurons["waveformDuration_peakTrough"].astype(str)

    unit_table = unit_table.copy()
    unit_table["fr_round"] = unit_table[NWB_FIRING_RATE_COL].round(FR_ROUND_DECIMALS).astype(str)
    unit_table["nspikes"] = unit_table["spike_times"].apply(lambda x: len(x))
    unit_table["waveformDuration_peakTrough"] = unit_table["waveformDuration_peakTrough"].astype(str)
    unit_table['cluster_id']  = unit_table['cluster_id'].astype(str)
    #n_dupe_keys = int(unit_table["fr_round"].duplicated().sum())
    #if n_dupe_keys:
    #    warnings.warn(f"{session_id}: {n_dupe_keys} duplicate rounded-firing-rate values in the "
    #                   f"NWB unit table -- keeping only the first match per value to avoid "
    #                   f"inflating neurons' row count (would break preIdx/postIdx alignment).")
    #unit_table = unit_table.drop_duplicates(subset="fr_round", keep="first")

    n_before = len(neurons)
    merged = neurons.merge(unit_table,
                           on=["cluster_id", "fr_round", "waveformDuration_peakTrough", "ccf_atlas_acronym"],
                           #on=["cluster_id"],
                           how="left",
                           suffixes=("", "_nwb"), indicator=True)
    if len(merged) != n_before:
        raise RuntimeError(
            f"{session_id}: NWB merge changed neurons row count ({n_before} -> {len(merged)}) -- "
            f"this must never happen (breaks preIdx/postIdx row-index alignment)."
        )

    n_matched = int((merged["_merge"] == "both").sum())
    match_rate = n_matched / n_before if n_before else 0.0
    print(f"{session_id}: NWB firing-rate match {n_matched}/{n_before} units ({match_rate:.0%})")
    if match_rate < 1.0:
        warnings.warn(f"{session_id}: only {match_rate:.0%} of units matched an NWB unit by rounded "
                       f"firing rate -- unmatched units keep NaN for NWB-derived columns.")
    merged = merged.drop(columns=["_merge", "fr_round"])

    # Step 3: process_allen_labels on the merged (neurons + NWB) data
    merged = allen_utils.process_allen_labels(merged, split_merge_areas=True)
    if len(merged) != n_before:
        raise RuntimeError(
            f"{session_id}: allen_utils.process_allen_labels changed row count "
            f"({n_before} -> {len(merged)}) -- this must never happen "
            f"(breaks preIdx/postIdx row-index alignment)."
        )
    if CUSTOM_AREA_COL not in merged.columns:
        warnings.warn(
            f"{session_id}: allen_utils.process_allen_labels did not produce a "
            f"'{CUSTOM_AREA_COL}' column (columns present: {list(merged.columns)}) -- "
            f"area_relation will NOT be overridden for this session; it will silently "
            f"fall back to pre_acronym/post_acronym instead. Check CUSTOM_AREA_COL "
            f"matches the actual output column name."
        )
    return merged


def load_session(session_dir):
    """
    1. load pairs (pairs_light.pkl)
    2. load neurons.pkl, merge onto the NWB unit table using firing rate
    3. run allen_utils.process_allen_labels on that merged result
       (steps 2+3: merge_nwb_allen_labels)
    4. merge the enriched neurons data -- including CUSTOM_AREA_COL from
       step 3 -- onto pairs by row index (preIdx/postIdx), alongside every
       other neurons.pkl/NWB column (_merge_neuron_metadata)
    5. area_relation is overridden to use CUSTOM_AREA_COL instead of
       pre_acronym/post_acronym: this isn't a separate step here, it
       happens automatically downstream in process_session() via
       _get_area_columns(), which checks pre_/post_{CUSTOM_AREA_COL}
       FIRST (see AREA_SRC_CANDIDATES/AREA_TGT_CANDIDATES), falling back
       to pre_acronym/post_acronym only if step 3 didn't produce it for
       this session.

    Also loads score arrays for ALL FOUR conditions (naive0, naive1,
    learned0, learned1) plus the whole-session score -- whichever files
    exist for this session. CCG arrays are NOT loaded here, only their
    paths recorded; make_session_figures() pulls specific rows via
    NpyRowReader once it knows which pair_ids it needs (skipped entirely
    when LOAD_CCGS=False)."""
    # Step 1: load pairs
    with open(session_dir / "pairs_light.pkl", "rb") as f:
        pairs = pickle.load(f)
    pairs = pairs if isinstance(pairs, pd.DataFrame) else pd.DataFrame(pairs)


    # Merge
    # Steps 2-3: load neurons, merge onto NWB unit table by firing rate,
    # then run allen_utils processing on the merged result
    neurons = load_neurons(session_dir)

    #pairs = _merge_neuron_metadata(pairs, neurons) #merge neurons and pairs before allen info?
    #mask = pairs.preIdx == pairs.postIdx
    #neurons = pairs[mask]

    if neurons is not None:
        neurons = merge_nwb_allen_labels(neurons, session_dir.name)

    # Step 4: merge the enriched neuron metadata (incl. CUSTOM_AREA_COL)
    # onto pairs by row index
    pairs = _merge_neuron_metadata(pairs, neurons)

    # Step 5 (area_relation override) happens automatically downstream --
    # see AREA_SRC_CANDIDATES/AREA_TGT_CANDIDATES and _get_area_columns().

    # Score files are (3, N_pairs) and small (~25 MB) -- safe to fully load,
    # then reduce 3 rows -> 1 score per pair. See _reduce_score_rows.
    scores = {}
    for sign in SIGNS:
        scores[(sign, "whole")] = _reduce_score_rows(np.load(score_file(session_dir, sign)))
        for cond in ("naive0", "naive1", "learned0", "learned1"):
            f_ = score_file(session_dir, sign, cond)
            if f_.exists():
                scores[(sign, cond)] = _reduce_score_rows(np.load(f_))

    ccg_paths = {"whole": session_dir / "ccgs_nostim.npy"}
    for cond in ("naive0", "naive1", "learned0", "learned1"):
        f_ = session_dir / f"ccgs_{cond}_nostim.npy"
        if f_.exists():
            ccg_paths[cond] = f_

    return dict(pairs=pairs, scores=scores, ccg_paths=ccg_paths)


# ============================== SCORING =====================================
def process_session(loaded, row):
    """Return tidy long-format df: one row per connected pair, per sign, per
    comparison (passive / inflection). Both comparisons are attempted for
    every session; whichever ones have both condition score files present
    for this session are included, the rest are silently skipped. Uses only
    score arrays -- never touches CCG data."""
    pairs, scores = loaded["pairs"], loaded["scores"]
    area_source_all, area_target_all = _get_area_columns(pairs)
    # Computed once per session, reused across signs -- layer info omitted
    # per STRIP_LAYER_SUFFIX (see config comment for the verification caveat).
    within_all = (_strip_layer_series(area_source_all) == _strip_layer_series(area_target_all)).to_numpy()
    records = []

    for sign in SIGNS:
        if (sign, "whole") not in scores:
            continue
        whole = scores[(sign, "whole")]
        connected_idx = np.where(whole > CONNECTED_THRESH)[0]
        if connected_idx.size == 0:
            continue

        area_source = area_source_all.iloc[connected_idx].to_numpy()
        area_target = area_target_all.iloc[connected_idx].to_numpy()
        area_relation = np.where(within_all[connected_idx], "within", "across")

        for comparison, (pre_cond, post_cond) in COMPARISONS.items():
            if (sign, pre_cond) not in scores or (sign, post_cond) not in scores:
                continue

            pre = scores[(sign, pre_cond)][connected_idx]
            post = scores[(sign, post_cond)][connected_idx]
            diff = post - pre

            records.append(pd.DataFrame({
                "mouse": row.mouse, "session_id": row.session_id, "reward_group": row.reward_group,
                "day_type": row.day_type, "sign": sign, "comparison": comparison,
                "pair_id": connected_idx,
                "area_source": area_source, "area_target": area_target, "area_relation": area_relation,
                "whole_score": whole[connected_idx],
                "score_pre": pre, "score_post": post, "score_diff": diff,
            }))

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def session_connectivity_summary(loaded, row):
    """One row per sign: % of pairs that are connected, computed with THREE
    separate denominators (all using TOTAL CANDIDATE pairs, not just the
    already-filtered connected subset that process_session() outputs):
      - overall:     n_connected_total  / n_total_pairs
      - within-area: n_connected_within / n_total_within_area_pairs
      - across-area: n_connected_across / n_total_across_area_pairs
    within/across-area status is computed once over the FULL pairs_light.pkl
    (not just connected pairs), via the vectorized _get_area_columns picker
    since this can be ~1M rows. Layer info is omitted from the comparison
    (see STRIP_LAYER_SUFFIX), matching process_session()'s area_relation
    column so the two stay consistent.

    Alignment assumption (same one process_session() already relies on):
    pair index i in the score arrays corresponds to row i of pairs_light.pkl.
    """
    pairs, scores = loaded["pairs"], loaded["scores"]
    area_source_all, area_target_all = _get_area_columns(pairs)
    is_within_all = (_strip_layer_series(area_source_all) == _strip_layer_series(area_target_all)).to_numpy()
    n_total_within = int(is_within_all.sum())
    n_total_across = int((~is_within_all).sum())

    rows = []
    for sign in SIGNS:
        if (sign, "whole") not in scores:
            continue
        whole = scores[(sign, "whole")]
        connected_mask = whole > CONNECTED_THRESH
        n_total = len(whole)
        n_connected = int(connected_mask.sum())
        n_connected_within = int(np.sum(connected_mask & is_within_all))
        n_connected_across = int(np.sum(connected_mask & ~is_within_all))

        rows.append(dict(
            mouse=row.mouse, session_id=row.session_id, reward_group=row.reward_group,
            day_type=row.day_type, sign=sign,
            n_total_pairs=n_total, n_connected_pairs=n_connected,
            pct_connected_overall=n_connected / n_total * 100 if n_total else np.nan,
            n_total_within_area=n_total_within, n_connected_within_area=n_connected_within,
            pct_connected_within_area=n_connected_within / n_total_within * 100 if n_total_within else np.nan,
            n_total_across_area=n_total_across, n_connected_across_area=n_connected_across,
            pct_connected_across_area=n_connected_across / n_total_across * 100 if n_total_across else np.nan,
        ))
    return pd.DataFrame(rows)


# ============================== FIGURES =====================================
def _style_ax(ax, xlabel="Lag (ms)", ylabel="CCG"):
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)


LEGEND_FONTSIZE = 5  # small: every panel gets its own legend, keep it compact


def _ccg_grid(traces_getter, pair_ids, pairs_meta, out_path, suptitle, legend_labels=None, ncols=6, nrows=5):
    n = min(len(pair_ids), ncols * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.1 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, pair_ids[:n]):
        ax.set_box_aspect(1)
        for label, trace, color in traces_getter(idx):
            ax.plot(trace, lw=1, color=color, label=label)
        src = _first_present(pairs_meta.iloc[idx], AREA_SRC_CANDIDATES)
        tgt = _first_present(pairs_meta.iloc[idx], AREA_TGT_CANDIDATES)
        ax.set_title(f"{src}\u2192{tgt}  id={idx}", fontsize=7)
        _style_ax(ax)
        if legend_labels:
            # Every panel gets its own legend (not just the first) -- each
            # panel plots a different pair with different scores, so a
            # single shared legend would only ever show one pair's values.
            ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, handlelength=1,
                      labelspacing=0.2, borderaxespad=0.2, handletextpad=0.4)
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_session_figures(loaded, row, df, out_dir):
    """Reads only the CCG rows needed for plotting via NpyRowReader. Not
    called at all while LOAD_CCGS=False (see _process_one_session)."""
    pairs, scores, ccg_paths = loaded["pairs"], loaded["scores"], loaded["ccg_paths"]

    # ---- whole-session top-30 grids: one per sign, independent of comparison ----
    top_whole_by_sign = {sign: np.argsort(scores[(sign, "whole")])[::-1][:30]
                          for sign in SIGNS if (sign, "whole") in scores}
    whole_ids_needed = sorted({int(i) for ids in top_whole_by_sign.values() for i in ids})

    if whole_ids_needed and "whole" in ccg_paths:
        with NpyRowReader(ccg_paths["whole"]) as reader:
            whole_rows = reader.rows(whole_ids_needed)
        for sign, top_whole in top_whole_by_sign.items():
            whole_s = scores[(sign, "whole")]
            _ccg_grid(
                lambda idx, s=whole_s, rows=whole_rows: [(f"score={s[idx]:.1f}", rows[idx], "black")],
                top_whole, pairs, out_dir / f"{row.session_id}_top30_whole_{sign}.png",
                f"{row.session_id} \u2014 top 30 whole-session {sign} connections",
                legend_labels=True,
            )

    # ---- condition-comparison grids: one set per comparison actually present ----
    for comparison, (pre_cond, post_cond) in COMPARISONS.items():
        if pre_cond not in ccg_paths or post_cond not in ccg_paths:
            continue

        ids_by_sign_tag = {}
        score_lookup_by_sign = {}
        for sign in SIGNS:
            sub = df[(df.sign == sign) & (df.comparison == comparison)]
            if sub.empty:
                continue
            top_whole_pairs = sub.nlargest(30, "whole_score").pair_id.values
            top_change_pairs = sub.reindex(sub.score_diff.abs().sort_values(ascending=False).index).pair_id.values[:30]
            ids_by_sign_tag[sign] = dict(best_whole=top_whole_pairs, biggest_change=top_change_pairs)
            # pair_id is unique within a (sign, comparison) slice (each connected
            # pair contributes exactly one row per comparison), so this is a
            # clean 1:1 lookup from pair_id -> its score_pre/score_post.
            score_lookup_by_sign[sign] = sub.drop_duplicates(subset="pair_id").set_index("pair_id")[
                ["score_pre", "score_post"]]

        cond_ids_needed = sorted({int(i) for tags in ids_by_sign_tag.values() for arr in tags.values() for i in arr})
        if not cond_ids_needed:
            continue

        with NpyRowReader(ccg_paths[pre_cond]) as r_pre, NpyRowReader(ccg_paths[post_cond]) as r_post:
            pre_rows = r_pre.rows(cond_ids_needed)
            post_rows = r_post.rows(cond_ids_needed)

        for sign, tags in ids_by_sign_tag.items():
            score_lookup = score_lookup_by_sign[sign]
            for tag, pair_ids in tags.items():
                _ccg_grid(
                    lambda idx, sl=score_lookup: [
                        (f"{pre_cond} (score={sl.loc[idx, 'score_pre']:.1f})", pre_rows[idx], "tab:blue"),
                        (f"{post_cond} (score={sl.loc[idx, 'score_post']:.1f})", post_rows[idx], "tab:red"),
                    ],
                    pair_ids, pairs, out_dir / f"{row.session_id}_{comparison}_{tag}_{sign}_condition_compare.png",
                    f"{row.session_id} \u2014 {sign} {comparison} {tag.replace('_', ' ')}: {pre_cond} vs {post_cond}",
                    legend_labels=True,
                )


def plot_score_diff_distribution(df, out_dir):
    """One independent figure per (sign, comparison) -- E and I scores come
    from different underlying models (different SCORE_TAG versions) and are
    never compared to each other; passive and inflection are also kept
    separate since they're different behavioral epochs."""
    for sign, color in zip(SIGNS, ("tab:orange", "tab:purple")):
        for comparison in df.comparison.unique():
            d = df.loc[(df.sign == sign) & (df.comparison == comparison), "score_diff"].dropna()
            if not len(d):
                continue
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(d, bins=40, color=color)
            _style_ax(ax, xlabel="Score change (post \u2212 pre)", ylabel="Count")
            ax.set_title(f"{sign} {comparison}: distribution of CCG score change", fontsize=12, fontweight="bold")
            fig.tight_layout()
            fig.savefig(out_dir / f"score_diff_distribution_{sign}_{comparison}.png", dpi=200)
            plt.close(fig)


def plot_area_summary(df, out_dir):
    for sign in SIGNS:
        sub_sign = df[df.sign == sign]
        for day_type in sub_sign.day_type.unique():
            for comparison in sub_sign.comparison.unique():
                sub = sub_sign[(sub_sign.day_type == day_type) & (sub_sign.comparison == comparison)]
                piv = sub.pivot_table(index="area_source", columns="area_target", values="score_diff", aggfunc="mean")
                if piv.empty:
                    continue
                vmax = np.nanmax(np.abs(piv.values))
                fig, ax = plt.subplots(figsize=(0.5 * piv.shape[1] + 3, 0.5 * piv.shape[0] + 3))
                im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=90, fontsize=7)
                ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=7)
                fig.colorbar(im, ax=ax, label="Mean score change")
                ax.set_title(f"{sign} \u2014 {day_type} \u2014 {comparison}: mean score change by area pair",
                             fontsize=11, fontweight="bold")
                fig.tight_layout()
                fig.savefig(out_dir / f"area_summary_{sign}_{day_type}_{comparison}.png", dpi=200)
                plt.close(fig)


# ============================== PER-SESSION WORKER ==========================
def _process_one_session(row):
    """Runs in a worker process. Returns
    (session_id, df_or_None, connectivity_df_or_None, traceback_or_None).

    Only the small tidy dataframes cross back to the main process. CCG
    figures (and all CCG file access) are skipped entirely while
    LOAD_CCGS=False -- make_session_figures() is simply never called.
    """
    try:
        loaded = load_session(row.session_path)

        connectivity_df = session_connectivity_summary(loaded, row)
        out_dir = OUTPUT_ROOT / row.mouse / f"whisker_{int(row.day_of_recording)}" / FIG_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        if not connectivity_df.empty:
            connectivity_df.to_csv(out_dir / f"{row.session_id}_connectivity_summary.csv", index=False)

        df = process_session(loaded, row)
        if df.empty:
            del loaded
            gc.collect()
            return row.session_id, None, connectivity_df, None

        df.to_csv(out_dir / f"{row.session_id}_score_changes.csv", index=False)

        if LOAD_CCGS:
            make_session_figures(loaded, row, df, out_dir)

        del loaded
        gc.collect()
        return row.session_id, df, connectivity_df, None
    except Exception:  # keep one bad session from killing the whole pool
        return row.session_id, None, None, traceback.format_exc()


# ============================== MAIN LOOP ===================================
def run_pipeline(manifest, n_workers=N_WORKERS):
    # itertuples() returns pandas' dynamically-generated `Pandas` namedtuple,
    # which fails to pickle under Windows' spawn-based multiprocessing.
    # SimpleNamespace is a plain, always-picklable stdlib type with the same
    # row.attr access pattern.
    rows = [SimpleNamespace(**r) for r in manifest.to_dict("records")]
    all_results = []
    all_connectivity = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_one_session, row): row.session_id for row in rows}
        for fut in as_completed(futures):
            session_id, df, connectivity_df, err = fut.result()

            if err is not None:
                print(f"FAILED {session_id}:\n{err}")
                continue
            if connectivity_df is not None and not connectivity_df.empty:
                all_connectivity.append(connectivity_df)
            if df is None:
                print(f"skip (no connected pairs): {session_id}")
            else:
                all_results.append(df)
                print(f"done: {session_id}")

    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_ROOT / "combined_score_changes.csv", index=False)

    combined_connectivity = pd.concat(all_connectivity, ignore_index=True) if all_connectivity else pd.DataFrame()
    combined_connectivity.to_csv(OUTPUT_ROOT / "combined_connectivity_summary.csv", index=False)

    return combined


def main(debug=False):
    manifest = build_manifest()
    if debug:
        manifest = manifest[manifest.mouse.isin(["AB131", "AB132", "AB133"])]
        print(f"[debug] running only mouse(s): {manifest.mouse.unique().tolist()}")

    combined = run_pipeline(manifest)

    summary_dir = OUTPUT_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if not combined.empty and LOAD_CCGS:
        plot_score_diff_distribution(combined, summary_dir)
        plot_area_summary(combined, summary_dir)
    return combined


if __name__ == "__main__":
    main(debug=True)  # flip to False once verified