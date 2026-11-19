"""
cluster_mmd_analysis.py

Maximum Mean Discrepancy (MMD) analysis comparing neuron-level PSTH-space
distributions between reward cohorts (R+ vs R-), independent of any
clustering step (rastermap/k-means/GMM). Complements the PERMANOVA-on-clusters
result with a cluster-free, continuous-embedding test operating directly on
the same feature matrix used for rastermap.

Design decisions (per discussion):
  - Test statistic: unbiased MMD^2_u with an RBF kernel. Bandwidth (sigma^2)
    is set ONCE via the median heuristic on the pooled, label-blind data and
    then frozen for every permutation -- the kernel must not track the
    label assignment being tested.
  - Null distribution: MOUSE-LEVEL permutation of cohort labels, NEVER
    neuron-level. Each mouse's full neuron complement moves together under
    every relabeling. This mirrors the pipeline's existing mouse-level
    correction convention (f_{m,k}) and avoids pseudoreplication.
    Monte Carlo sampling is used throughout (default n_perm=9999, giving a
    p-value resolution floor of 1/10000), rather than exact enumeration of
    all mouse-label arrangements -- exact enumeration is still available
    via exact_if_feasible=True but is not the default.
  - Scope:
      * per-area, using the pipeline's existing area validity filter
        (>=min_mice_per_area mice, >=min_neurons_per_mouse neurons/mouse,
        bc_label in {good, mua}, intersected across cohorts)
      * aggregated across all valid areas pooled, under the explicit
        assumption that areas are similarly sampled. sigma^2 for the
        aggregated test is estimated ONCE, globally, on the pooled
        label-blind aggregate -- NOT per area. Mouse-level permutation
        still shuffles each mouse's ENTIRE cross-area neuron complement
        together, so the mouse (not the mouse-within-area) is the
        exchangeable unit.
  - Feature space: the FULL concatenated z-scored PSTH vector, and
    SEPARATELY per trial type (sub-slices of the same vector). The test
    always runs in full-dimensional feature space -- PCA is never used
    for the statistic, only for visualization.
  - Visualization is illustrative only:
      * PCA scatter + pe0cohort KDE contours (may miss the informative
        direction if it's not in PC1-PC2)
      * MMD witness function evaluated in the ORIGINAL high-dim space,
        overlaid on the same PCA projection for color -- this panel is
        faithful to what the test detected, unlike the raw scatter.
      * Null histogram with observed statistic marked (the actual proof).
      * Cross-area summary/forest plot with BH-FDR stars.

Entry point:
  - run_all(feature_matrix_path, config_path=None, output_dir=None)
    Loads EVERYTHING from the single feature_matrix.npz (X, unit_ids,
    mouse_arr, session_arr, reward_arr, area_arr, area_group_arr,
    waveform_arr, layer_arr, odd_even_r, n_bins_list, n_conds) -- no
    unit_table join, no ROC measures. Runs pe0area + aggregated MMD,
    writes CSVs/figures.
    (run_mmd_per_area / run_mmd_aggregated remain available individually
    if you already have a `neuron_meta` dict of arrays in memory.)

Data source (npz-only, no unit_table):
  - feature_matrix.npz is expected to contain, all positionally aligned
    1:1 with each other (row i of X = neuron i of every metadata array):
        X, unit_ids, mouse_arr, session_arr, reward_arr, area_arr,
        area_group_arr, waveform_arr, layer_arr, odd_even_r,
        n_bins_list, n_conds
  - Quality-label (bc_label good/mua) filtering is NOT available at this
    stage -- that column lives only in unit_table, which this module no
    longer touches. If you need quality filtering, apply it upstream
    when feature_matrix.npz is built.
  - feature_matrix.npz is built BEFORE the odd/even CV reliability filter
    that's later applied when producing results_cv.npz -- so, unlike the
    unit_table-based version of this module, reliability filtering is NOT
    assumed to already be done. Set cfg['min_odd_even_r'] (e.g. 0.2, the
    pipeline's usual threshold) to filter on `odd_even_r` directly here;
    leave it None to skip reliability filtering entirely.
  - `cfg['trial_type_slices']` is auto-built from n_bins_list if not given
    explicitly (one slice per condition, named cond0/cond1/... since
    feature_matrix.npz does not store condition text labels) -- pass your
    own dict of {name: (start_idx, end_idx)} to use real condition names.
  - Column names (area/mouse/cohort field) are configurable via an `mmd:`
    section in config.yaml in case they differ from the defaults below.

Figure types produced:
  - method_overview/  : ONE didactic multi-panel figure per (area x
    feature_version) walking through the method end-to-end using the
    actual numbers from that run: input feature matrix -> pairwise
    distance / kernel-bandwidth selection -> kernel similarity matrix ->
    MMD^2 term decomposition -> mouse-level permutation schematic ->
    resulting null distribution. For explaining/presenting the method
    itself (methods sections, talks), not for pe0area significance
    screening.
  - null_histograms/  : per (area x feature_version), the permutation null
    with the observed statistic marked -- the actual significance evidence.
  - scatter_witness/  : per (area x feature_version), PCA scatter (illustration
    only) + MMD witness function computed in full-dim space (faithful).
  - mmd_summary_per_area.pdf : cross-area forest plot with BH-FDR stars.
"""

import os
import pathlib
import time
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

try:
    from threadpoolctl import threadpool_limits
    _HAS_THREADPOOLCTL = True
except ImportError:
    _HAS_THREADPOOLCTL = False
    from contextlib import contextmanager

    @contextmanager
    def threadpool_limits(limits=None, user_api=None):
        """No-op fallback if threadpoolctl isn't installed. Per-worker BLAS
        thread counts won't be capped when running the per-area loop across
        processes, which risks core oversubscription (n_jobs processes each
        spawning their own multi-threaded BLAS calls). `pip install
        threadpoolctl` to fix; harmless (just slower) otherwise."""
        yield

N_CORES = os.cpu_count() or 1

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

    def tqdm(iterable=None, *args, **kwargs):
        """No-op fallback if tqdm isn't installed: prints a one-line notice
        once, then behaves like a plain iterable (or returns a dummy context
        manager if called with no iterable, matching tqdm's manual-update
        usage as `pbar = tqdm(total=...)`)."""
        if iterable is not None:
            return iterable
        class _DummyBar:
            def update(self, *a, **k): pass
            def set_description(self, *a, **k): pass
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return _DummyBar()

if not _HAS_TQDM:
    print("[MMD] tqdm not installed -- progress bars will fall back to plain "
          "print statements only. `pip install tqdm` for live progress bars.")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DEFAULT_MMD_CFG = {
    'area_col': 'area_group',      # key into neuron_meta -- 'area_group' (coarser,
                                    # broader anatomical grouping) by default for
                                    # area-wise analyses; switch to 'area' for
                                    # fine-grained per-area testing once the
                                    # area_group-level picture is established
    'mouse_col': 'mouse',          # key into neuron_meta
    'cohort_col': 'reward_group',  # key into neuron_meta
    'cohort_labels': ('R+', 'R-'),
    'min_odd_even_r': None,        # e.g. 0.2 to reproduce the pipeline's usual CV
                                    # reliability filter here; None = no filtering
    'min_mice_per_area': 3,
    'min_neurons_per_mouse': 5,
    'mmd_n_perm': 5000,
    'mmd_exact_if_feasible': False,
    'mmd_random_seed': 0,
    'n_jobs': min(N_CORES, 110),   # cores used for the per-area outer loop
                                    # (joblib); the aggregated test instead
                                    # relies on multi-threaded BLAS for its
                                    # one big batched matmul -- see
                                    # mouse_level_permutation_test.
    'output_dir': 'mmd_analysis',
    'fdr_alpha': 0.05,
    'trial_type_slices': None,   # dict {name: (start_idx, end_idx)} or None ->
                                  # auto-built from n_bins_list (see _load_npz_data)
    'cond_labels': None,          # optional list of names for auto-built slices;
                                   # defaults to cond0, cond1, ... if not given
    'pca_test_n_components': None,      # int or None. If set, EVERY feature
                                    # version is ALSO tested after PCA-reducing
                                    # to this many components (fit fresh per
                                    # test, on that test's own pooled
                                    # label-blind neurons -- never globally) --
                                    # run ALONGSIDE the full-dimensional test,
                                    # as an additional '<name>_pca' row, not a
                                    # replacement (mitigates distance
                                    # concentration in high-d/low-N regimes;
                                    # a real difference outside the retained
                                    # components would only show up in the
                                    # full-dim test, so both are kept). At
                                    # most one of this and
                                    # pca_test_variance_retained should be set.
                                    # NOT the same as pca_n_components below,
                                    # which is unrelated (2D scatter plotting).
    'pca_test_variance_retained': 0.99, # float in (0, 1] or None. Alternative
                                    # to pca_test_n_components: keep as many
                                    # components as needed to retain this
                                    # fraction of variance (fit per test, as
                                    # above), instead of a fixed count.
                                    # DEFAULT 0.99: every test now ALSO runs
                                    # in a 99%-variance PCA-reduced space by
                                    # default, alongside (never instead of)
                                    # the full-dimensional test -- both are
                                    # always reported (see 'space' column in
                                    # results_df). Set to None to disable and
                                    # run only the full-dimensional test.
    'pca_n_components': 2,   # unrelated to the two keys above -- this is
                              # ALWAYS 2, purely for the illustrative 2D
                              # scatter/witness figure, never for the test.
    'method_overview_neuron_order': 'similarity',  # 'similarity' or 'mouse';
                              # governs panels A/C of the method_overview
                              # figure only -- see _fig_mmd_method_overview.
    'exclude_bins': None,     # dict {trial_type_name: [(rel_start, rel_end), ...]}
                              # or None. Index ranges LOCAL to that trial
                              # type's own slice (0 = first bin of that
                              # trial type) to drop -- e.g. a baseline
                              # window. Applied everywhere that trial type
                              # appears: its own sub-slice AND 'full' (which
                              # is built by concatenating each trial type's
                              # OWN masked sub-slice, not a raw untouched
                              # copy of the whole matrix -- see
                              # _slice_feature_matrix). Excluded bins are
                              # also never seen by PCA fitting.
    'exclude_from_full': [],  # list of trial-type names to leave OUT of
                              # 'full' entirely (e.g. ['passive_pre']) -- an
                              # excluded trial type is STILL tested as its
                              # own standalone entry, just never
                              # concatenated into 'full'.
    'exclude_from_condition_fdr': None,  # list of trial-type (base) names
                              # to exclude from the WITHIN-AREA,
                              # across-single-condition-tests FDR group
                              # (see 'q_value_within_area' in results_df).
                              # None -> reuses exclude_from_full's list (a
                              # condition you've said isn't a comparison
                              # target for the 'full' vector usually
                              # shouldn't inflate this correction either).
                              # 'full' itself (and '<x>_pca') are always
                              # excluded from this group regardless.
    'cluster_col': None,      # optional key into neuron_meta holding an
                              # existing (e.g. rastermap) cluster id per
                              # neuron. If set AND present in neuron_meta,
                              # extreme/typical neurons are cross-tabulated
                              # against cluster membership as a bridge back
                              # to the cluster-based story. None/missing ->
                              # this cross-reference is silently skipped.
}


def _save(fig, path, dpi=300):
    """Save figure as PDF, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, format='pdf', bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------
# Core MMD statistic
# --------------------------------------------------------------------------

def compute_mmd2_u(X, Y, sigma2=None):
    """
    Unbiased MMD^2 estimator with an RBF kernel.

    MMD^2_u(X,Y) = 1/(m(m-1)) sum_{i!=j} k(x_i,x_j)
                 + 1/(n(n-1)) sum_{i!=j} k(y_i,y_j)
                 - 2/(mn)     sum_{i,j}  k(x_i,y_j)

    Parameters
    ----------
    X, Y : ndarray, shape (m, d) and (n, d)
    sigma2 : float or None
        RBF bandwidth, k(x,y) = exp(-||x-y||^2 / (2*sigma2)). If None,
        estimated via the median heuristic on the POOLED [X;Y] sample
        (label-blind). Callers running a permutation test must estimate
        sigma2 ONCE on the observed pooled data and pass it explicitly to
        every subsequent (permuted) call.

    Returns
    -------
    mmd2 : float
        Unbiased estimate. Can be slightly negative under H0 -- expected,
        not a bug.
    sigma2 : float
        Bandwidth actually used (echoed back so callers can cache it).
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    m, n = X.shape[0], Y.shape[0]
    if m < 2 or n < 2:
        raise ValueError(
            f"Need >=2 samples per group for unbiased MMD (got m={m}, n={n})."
        )

    if sigma2 is None:
        pooled = np.vstack([X, Y])
        sq_d = pdist(pooled, metric='sqeuclidean')
        sigma2 = np.median(sq_d)
        if not np.isfinite(sigma2) or sigma2 <= 0:
            mean_d = np.mean(sq_d)
            sigma2 = mean_d if (np.isfinite(mean_d) and mean_d > 0) else 1.0

    gamma = 1.0 / (2.0 * sigma2)
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)

    term_x = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    term_y = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    term_xy = Kxy.mean()

    mmd2 = term_x + term_y - 2.0 * term_xy
    return float(mmd2), float(sigma2)


# --------------------------------------------------------------------------
# Precomputed-kernel batched permutation machinery
# --------------------------------------------------------------------------
#
# The observation that makes this possible: sigma2 (and therefore every
# pairwise kernel similarity k(neuron_i, neuron_j)) is FROZEN once, on the
# pooled label-blind data, and is identical for the observed split and
# every single permutation. Only which side of the X/Y split a given
# neuron falls on changes across permutations -- the underlying N x N
# similarity structure never does.
#
# So instead of recomputing an O(N^2 * d) kernel matrix from scratch on
# every one of n_perm permutations (the original per-permutation
# `compute_mmd2_u` loop), the full N x N kernel matrix is computed ONCE
# (O(N^2 * d)), and every permutation's MMD^2 is then obtained from that
# single fixed matrix via cheap O(N^2) linear algebra -- batched across ALL
# permutations at once as one matrix multiply, which BLAS parallelizes
# across cores automatically. Verified bit-for-bit identical to the
# original per-permutation `compute_mmd2_u` loop (see module tests / dev
# notes) up to floating-point noise (~1e-16).

def _precompute_pooled_kernel(features, sigma2=None, dtype=np.float32):
    """
    Compute the full N x N RBF kernel matrix ONCE for the pooled sample
    (both cohorts together), plus its row sums (needed to derive the
    complement-group kernel sums for free, see _mmd2_batch_from_kernel).

    sigma2 estimation and the kernel matrix itself share the same pairwise
    squared-distance computation (via squareform(pdist(...))) rather than
    computing distances twice (once for the median heuristic, once again
    inside a separate rbf_kernel(...) call) -- halving the one-time
    O(N^2 * d) cost relative to naively calling compute_mmd2_u's approach.

    Returns
    -------
    K        : (N, N) kernel matrix, dtype as given (float32 by default to
               halve memory vs. float64 -- kernel values live in [0, 1], so
               float32 precision is ample; only the final summed statistics
               use float64 accumulation internally via numpy's default).
    sigma2   : float, the bandwidth actually used.
    rowsum_K : (N,) row sums of K, i.e. rowsum_K[i] = sum_j K[i, j].
    """
    features = np.atleast_2d(features)
    N = features.shape[0]
    sq_d = pdist(features, metric='sqeuclidean')
    if sigma2 is None:
        sigma2 = np.median(sq_d)
        if not np.isfinite(sigma2) or sigma2 <= 0:
            mean_d = np.mean(sq_d)
            sigma2 = mean_d if (np.isfinite(mean_d) and mean_d > 0) else 1.0
    gamma = 1.0 / (2.0 * sigma2)
    D_sq = squareform(sq_d, checks=False)
    K = np.exp(-D_sq * gamma).astype(dtype, copy=False)
    # RBF kernel diagonal is exactly 1.0 (k(x,x) = exp(0) = 1) by
    # construction -- used as an identity below rather than read off K,
    # since it's exact regardless of dtype/rounding.
    rowsum_K = K.sum(axis=1)
    return K, float(sigma2), rowsum_K


def _mmd2_batch_from_kernel(K, rowsum_K, S, blas_threads=None):
    """
    Compute unbiased MMD^2 for EVERY column of S in one batched pass over
    the precomputed pooled kernel matrix K.

    S : (N, P) array, dtype float32/float64. Column p is the group-X
        membership indicator (1.0 if neuron i is in group X for split p,
        else 0.0) for the p-th split (observed split or one permutation).
        Complement (group Y) is implicitly (1 - S) -- every neuron is
        assigned to exactly one of the two groups in every split.

    blas_threads : int or None
        If given, cap BLAS threads to this many for the one big matmul
        below via threadpoolctl (use a small number, e.g. 1, when this is
        called from inside an already-parallel outer loop over many
        smaller tasks, to avoid oversubscribing cores; use a large number,
        or None to leave the default, for a single big call that should
        use many cores itself). None leaves numpy's default BLAS thread
        count untouched.

    Returns
    -------
    mmd2 : (P,) array, one unbiased MMD^2 estimate per column of S.
    """
    S = np.asarray(S, dtype=K.dtype)
    N, P = S.shape
    with threadpool_limits(limits=blas_threads, user_api='blas'):
        A = K @ S                          # (N, P); A[i,p] = sum_j K[i,j] S[j,p]
    B = rowsum_K[:, None] - A              # (N, P); == K @ (1 - S), for free
    diagA = np.einsum('ip,ip->p', S, A, optimize=True)       # full within-X sum incl. diagonal
    T = 1.0 - S
    diagB = np.einsum('ip,ip->p', T, B, optimize=True)       # full within-Y sum incl. diagonal
    sum_XY = np.einsum('ip,ip->p', S, B, optimize=True)      # cross X-Y sum
    m_p = S.sum(axis=0)
    n_p = N - m_p
    # diag(K) == 1.0 exactly (RBF self-similarity), so the "off-diagonal
    # only" within-group sum is just (full sum incl. diagonal) - (group size).
    off_diag_X = diagA - m_p
    off_diag_Y = diagB - n_p
    term_x = off_diag_X / (m_p * (m_p - 1))
    term_y = off_diag_Y / (n_p * (n_p - 1))
    term_xy = sum_XY / (m_p * n_p)
    return term_x + term_y - 2.0 * term_xy


def mmd_witness_function(z, X, Y, sigma2):
    """
    Evaluate the MMD witness function f(z) = mean_x k(x,z) - mean_y k(y,z)
    at query points z, using the SAME sigma2 and high-dim coordinates as
    the test itself. Positive -> z resembles the X (first-group)
    distribution more in kernel space; negative -> resembles Y more.

    NOTE: when a query point z_i is ITSELF a member of X (or Y), this
    includes k(z_i, z_i) = 1 in that group's average -- trivially pulling
    z_i's witness value toward its own group (every point is maximally
    "similar to itself"). Fine for the illustrative scatter/witness plot
    (small, uniform bias, doesn't change the overall spatial pattern), but
    a real bias for any QUANTITATIVE per-neuron statistic (e.g. ranking
    neurons by typicality/atypicality, or an AUC) -- use
    mmd_witness_function_loo for those instead.

    Parameters
    ----------
    z : ndarray, shape (q, d)      query points (typically all neurons)
    X, Y : ndarray                 the two reference samples defining f
    sigma2 : float                 bandwidth used for the original MMD test

    Returns
    -------
    f : ndarray, shape (q,)
    """
    gamma = 1.0 / (2.0 * sigma2)
    Kzx = rbf_kernel(z, X, gamma=gamma)
    Kzy = rbf_kernel(z, Y, gamma=gamma)
    return Kzx.mean(axis=1) - Kzy.mean(axis=1)


def mmd_witness_function_loo(features, cohorts, cohort_values, sigma2):
    """
    Leave-one-out witness function: for every neuron, f(z) is computed the
    same way as mmd_witness_function, EXCEPT a neuron's similarity to its
    own group excludes itself (the diagonal k(z,z)=1 term), so no point is
    trivially pulled toward its own group by self-similarity. This is the
    version to use for any per-neuron ranking or classifier-style summary
    (compute_witness_auc, extract_extreme_neurons) -- the plain
    mmd_witness_function remains fine for the illustrative scatter plot.

    Uses the same precomputed-kernel machinery as the permutation test
    (_precompute_pooled_kernel) rather than three separate rbf_kernel calls,
    so this is cheap even for large N.

    Parameters
    ----------
    features : ndarray, shape (N, d)
    cohorts  : ndarray, shape (N,)   cohort label per neuron (any dtype;
               compared via string cast, consistent with the rest of the
               module -- see _as_str_series)
    cohort_values : tuple(pos_label, neg_label)
    sigma2   : float, bandwidth to use (pass the SAME sigma2 the MMD test
               used, for consistency -- do not re-estimate here)

    Returns
    -------
    witness : (N,) array. Positive -> resembles pos_label group more;
              negative -> resembles neg_label group more. Unlike the plain
              witness function, a neuron's own membership never inflates
              its own score.
    """
    pos_label, neg_label = cohort_values
    cohorts_str = _as_str_series(pd.Series(cohorts)).values
    pos_mask = cohorts_str == pos_label
    neg_mask = cohorts_str == neg_label

    K, _, rowsum_K = _precompute_pooled_kernel(features, sigma2=sigma2)
    s = pos_mask.astype(K.dtype)
    t = neg_mask.astype(K.dtype)
    n_pos, n_neg = s.sum(), t.sum()

    sum_to_pos = K @ s   # (N,), sum_j in pos of K[i,j], including self if i in pos
    sum_to_neg = rowsum_K - sum_to_pos

    # Leave-one-out: subtract self-similarity (K[i,i]=1 exactly) and the
    # own count from the denominator, ONLY for points that are themselves
    # members of that group.
    mean_to_pos = np.where(pos_mask, (sum_to_pos - 1.0) / (n_pos - 1.0), sum_to_pos / n_pos)
    mean_to_neg = np.where(neg_mask, (sum_to_neg - 1.0) / (n_neg - 1.0), sum_to_neg / n_neg)
    return mean_to_pos - mean_to_neg


def compute_witness_auc(features, cohorts, cohort_values, sigma2):
    """
    Treat the leave-one-out witness value as a classifier score and compute
    its ROC-AUC against true cohort membership -- a single, immediately
    interpretable [0.5, 1.0] effect size: 0.5 = witness carries no
    information about cohort at all (chance), 1.0 = witness perfectly
    separates the two cohorts.

    This is DESCRIPTIVE, computed once on the real (observed) split -- it
    is not itself permutation-tested (that would need re-deriving the LOO
    witness under every permutation, a separate and much larger
    computation than the MMD^2 test itself). Report it ALONGSIDE the
    MMD^2 permutation p-value/mmd2_z as an interpretable magnitude, not as
    a replacement for the rigorous significance test.

    Returns
    -------
    auc     : float
    witness : (N,) array, the LOO witness scores used (so callers/figures
              can reuse them without recomputing).
    """
    from sklearn.metrics import roc_auc_score
    pos_label, neg_label = cohort_values
    cohorts_str = _as_str_series(pd.Series(cohorts)).values
    witness = mmd_witness_function_loo(features, cohorts, cohort_values, sigma2)
    y_true = (cohorts_str == pos_label).astype(int)
    auc = float(roc_auc_score(y_true, witness))
    return auc, witness


def _mmd_kernel_terms(X, Y, sigma2):
    """
    Same building blocks as compute_mmd2_u, but also returns the raw
    kernel matrices and the three summed/averaged terms individually
    (term_x = mean within-X similarity, term_y = mean within-Y similarity,
    term_xy = mean cross similarity), for visualization/decomposition.
    Not used inside the permutation loop -- call once on the observed
    split, for the method-overview figure only.
    """
    gamma = 1.0 / (2.0 * sigma2)
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    m, n = X.shape[0], Y.shape[0]
    term_x = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    term_y = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    term_xy = Kxy.mean()
    return {
        'Kxx': Kxx, 'Kyy': Kyy, 'Kxy': Kxy,
        'term_x': term_x, 'term_y': term_y, 'term_xy': term_xy,
        'mmd2': term_x + term_y - 2.0 * term_xy,
    }


# --------------------------------------------------------------------------
# Mouse-level permutation test
# --------------------------------------------------------------------------

def mouse_level_permutation_test(features, mouse_ids, cohort_labels,
                                  cohort_values=('1', '0'),
                                  n_perm=9999, exact_if_feasible=False,
                                  sigma2=None, rng=None,
                                  show_progress=False, progress_desc=None,
                                  blas_threads=None, perm_batch_size=2000):
    """
    MMD^2 permutation test with the null built by shuffling cohort labels
    AT THE MOUSE LEVEL ONLY, preserving each mouse's full neuron complement
    intact under every relabeling. This is deliberate: permuting at the
    neuron level would treat within-mouse neurons as independently and
    freely exchangeable across cohorts, which they are not, and would make
    the null distribution artificially narrow (anti-conservative).

    PERFORMANCE: the pooled N x N kernel matrix is computed ONCE (see
    _precompute_pooled_kernel), then every split (the true observed split
    AND all n_perm permutations) is evaluated as columns of one batched
    matrix multiply against that fixed kernel (see _mmd2_batch_from_kernel)
    -- not by recomputing an O(N^2 * d) kernel from scratch per
    permutation as a naive loop would. This is mathematically identical to
    calling compute_mmd2_u independently for every split (verified
    bit-for-bit up to ~1e-16 floating-point noise) but removes the
    dominant per-permutation cost entirely, and the batched matmul itself
    is parallelized across cores automatically by BLAS.

    Parameters
    ----------
    features : ndarray, shape (N, d)
        Neuron-level feature vectors (full concatenated PSTH, or a
        trial-type sub-slice of it).
    mouse_ids : ndarray, shape (N,)
        Mouse identifier per neuron, aligned with `features` rows.
    cohort_labels : ndarray, shape (N,)
        Pe0neuron cohort label, aligned with `features` rows. Must be
        constant within each mouse (asserted below).
    cohort_values : tuple(str, str)
        (positive_label, negative_label), e.g. ('1', '0'). MMD^2 itself
        is symmetric in the two groups; this only fixes the sign
        convention of the witness function downstream.
    n_perm : int
        Number of Monte Carlo permutations (default 9999, for a p-value
        resolution floor of 1/10000). Each draw randomly selects a
        mouse-subset of size n_mice_pos to relabel as pseudo-1; drawn
        WITHOUT replacement across mice within a single draw, but draws
        themselves are independent, so the same split can in principle
        recur -- this is standard Monte Carlo permutation testing, not
        exact enumeration.
    exact_if_feasible : bool
        Default False: always use Monte Carlo sampling with n_perm draws,
        even when the number of distinct mouse-level label arrangements
        C(n_mice, n_mice_pos) is small enough to enumerate exactly. Set to
        True to switch to exact enumeration instead (recommended only if a
        given area's mouse count is small enough that exact enumeration
        would in fact use FEWER than n_perm evaluations -- otherwise
        Monte Carlo at n_perm=9999 gives strictly finer resolution).
    sigma2 : float or None
        If provided, reuse this bandwidth (recommended when this function
        is called with a bandwidth fixed elsewhere, e.g. for the aggregated
        analysis where sigma2 is estimated globally once). If None,
        estimated here via the median heuristic on the observed pooled
        data, then reused for every permutation.
    rng : np.random.Generator or None
    show_progress : bool
        If True, print a start/end summary and (if tqdm is installed) show
        a progress bar over permutation BATCHES (see perm_batch_size) --
        there's no longer a natural per-permutation loop to show progress
        over, since permutations are evaluated together in batched matrix
        ops. Off by default since this function is usually called many
        times in a row from run_mmd_per_area/run_mmd_aggregated, which show
        their own higher-level progress bar instead. Automatically turned
        on (prints only, no bar) whenever n_loop >= 2000 regardless of this
        flag, so slow calls are never silent.
    progress_desc : str or None
        Label shown in progress prints/bars (e.g. the area name).
    blas_threads : int or None
        Caps BLAS threads (via threadpoolctl) for the batched matmul. Pass
        a small number (e.g. 1) when this function is itself called from
        inside an already-parallel outer loop over many smaller tasks (see
        run_mmd_per_area's joblib loop), to avoid oversubscribing cores.
        Pass a large number, or leave None, for a single big call (e.g.
        run_mmd_aggregated) that should use many/all cores itself.
    perm_batch_size : int
        Number of splits (observed + permutations) processed per batched
        matmul call. Bounds peak memory (each batch allocates two N x
        batch_size temporaries) and gives a meaningful progress readout for
        very large N / n_perm; does not affect the result.

    Returns
    -------
    dict with keys: 'mmd2_obs', 'sigma2', 'null_mmd2' (ndarray), 'p_value',
    'n_perm_used', 'n_exact_possible', 'used_exact', 'n_mice_pos', 'n_mice_neg'
    """
    if rng is None:
        rng = np.random.default_rng(0)

    features = np.atleast_2d(features)
    mouse_ids = np.asarray(mouse_ids)
    cohort_labels = np.asarray(cohort_labels)
    unique_mice = np.unique(mouse_ids)
    N = features.shape[0]

    mouse_cohort = {}
    for m in unique_mice:
        labels_m = np.unique(cohort_labels[mouse_ids == m])
        if len(labels_m) != 1:
            raise ValueError(
                f"Mouse {m!r} has inconsistent cohort labels {labels_m}; "
                f"cohort must be constant within a mouse."
            )
        mouse_cohort[m] = labels_m[0]

    pos_label, neg_label = cohort_values
    pos_mice = [m for m in unique_mice if mouse_cohort[m] == pos_label]
    neg_mice = [m for m in unique_mice if mouse_cohort[m] == neg_label]
    n_mice_pos, n_mice_neg = len(pos_mice), len(neg_mice)
    n_total_mice = n_mice_pos + n_mice_neg

    if n_mice_pos < 1 or n_mice_neg < 1:
        raise ValueError("Need at least 1 mouse per cohort.")

    # Pooled N x N kernel, computed ONCE -- sigma2 estimated here (if not
    # given) via the median heuristic on this same pooled, label-blind
    # data, then frozen for the observed split AND every permutation below.
    t0 = time.time()
    K, sigma2, rowsum_K = _precompute_pooled_kernel(features, sigma2=sigma2)

    n_exact = comb(n_total_mice, n_mice_pos)
    used_exact = exact_if_feasible and (n_exact <= n_perm)
    n_loop = n_exact if used_exact else n_perm

    all_mice = list(unique_mice)

    def _indicator(pos_set):
        return np.isin(mouse_ids, list(pos_set)).astype(np.float32)

    # Column 0 = the TRUE observed split; every remaining column is one
    # permutation (exact enumeration or Monte Carlo draw). Batching them
    # together as one matrix lets a single matmul evaluate all of them at
    # once against the fixed kernel K.
    S_cols = [_indicator(pos_mice)]
    if used_exact:
        for pos_set in combinations(all_mice, n_mice_pos):
            S_cols.append(_indicator(pos_set))
        n_perm_used = n_exact
    else:
        for _ in range(n_perm):
            pos_set = rng.choice(all_mice, size=n_mice_pos, replace=False)
            S_cols.append(_indicator(pos_set))
        n_perm_used = n_perm

    verbose = show_progress or n_loop >= 2000
    if verbose:
        label = f" [{progress_desc}]" if progress_desc else ""
        print(f"[MMD]{label} {n_mice_pos} vs {n_mice_neg} mice, {N} neurons, "
              f"sigma2={sigma2:.4g} -- batched "
              f"{'exact enumeration' if used_exact else 'Monte Carlo'} over "
              f"{n_loop} splits (perm_batch_size={perm_batch_size})...")

    S_all = np.stack(S_cols, axis=1)  # (N, 1 + n_loop)
    n_cols = S_all.shape[1]
    mmd2_all = np.empty(n_cols, dtype=np.float64)
    batch_starts = list(range(0, n_cols, perm_batch_size))
    iterator = tqdm(batch_starts, desc=progress_desc or "MMD batches", leave=False) \
        if show_progress else batch_starts
    for start in iterator:
        stop = min(start + perm_batch_size, n_cols)
        mmd2_all[start:stop] = _mmd2_batch_from_kernel(
            K, rowsum_K, S_all[:, start:stop], blas_threads=blas_threads)

    mmd2_obs = float(mmd2_all[0])
    null_mmd2 = mmd2_all[1:]
    p_value = (1 + np.sum(null_mmd2 >= mmd2_obs)) / (1 + len(null_mmd2))
    null_mean, null_std = float(null_mmd2.mean()), float(null_mmd2.std())
    # Standardized effect size, same convention as specialization_null_z
    # elsewhere in this pipeline: how many null-SDs above the null mean the
    # observed statistic sits. Unlike the raw MMD^2 value, this is roughly
    # comparable across tests with different N/d, since it's already scaled
    # by that test's own null spread.
    mmd2_z = (mmd2_obs - null_mean) / null_std if null_std > 0 else np.nan

    if verbose:
        label = f" [{progress_desc}]" if progress_desc else ""
        print(f"[MMD]{label} done in {time.time() - t0:.1f}s -- "
              f"mmd2_obs={mmd2_obs:.4g}  z={mmd2_z:.3g}  p_value={p_value:.4g} "
              f"(n_perm_used={n_perm_used})")

    return {
        'mmd2_obs': mmd2_obs,
        'mmd2_z': mmd2_z,
        'sigma2': sigma2,
        'null_mmd2': null_mmd2,
        'p_value': p_value,
        'n_perm_used': n_perm_used,
        'n_exact_possible': n_exact,
        'used_exact': used_exact,
        'n_mice_pos': n_mice_pos,
        'n_mice_neg': n_mice_neg,
    }


# --------------------------------------------------------------------------
# Area validity filter (mirrors the pipeline's existing convention)
# --------------------------------------------------------------------------

def _cohort_values(cfg):
    """
    Return cfg['cohort_labels'] normalized to strings. Every comparison
    against a cohort column in this module casts that column to str first
    (see _as_str_series below) and compares against THESE values, so that
    cohort_labels configured as strings (e.g. '1'/'0') correctly match a
    column stored as ints/bools/categoricals, and vice versa -- a raw
    dtype mismatch here (int 0/1 in the column vs configured str '0'/'1')
    would otherwise silently match nothing, since Python's `0 == '0'` is
    False and `{0, 1} & {'0', '1'}` is empty even though they "look" the
    same once printed.
    """
    return tuple(str(c) for c in cfg['cohort_labels'])


def _as_str_series(series):
    """Cast a pandas Series to str for robust, dtype-agnostic comparison
    against string-normalized cohort/quality-label values."""
    return series.astype(str)


def get_valid_areas(neuron_meta, cfg):
    """
    Areas with >=cfg['min_mice_per_area'] mice, each with
    >=cfg['min_neurons_per_mouse'] neurons, intersected ACROSS reward
    cohorts so both cohorts share identical area rows -- mirrors the
    filter used elsewhere in the pipeline.

    Quality-label (bc_label good/mua) filtering is NOT available here --
    that column lives only in unit_table, which this module doesn't load.
    If cfg['min_odd_even_r'] is set, neurons below that odd/even CV
    reliability threshold are excluded first (feature_matrix.npz predates
    the pipeline's usual reliability filter, unlike results_cv.npz).

    neuron_meta : dict of 1D arrays, all aligned 1:1 with X's rows, must
        contain at least cfg['area_col'], cfg['mouse_col'], cfg['cohort_col']
        (and 'odd_even_r' if cfg['min_odd_even_r'] is set).

    All cohort comparisons are done on STRING-cast values (see
    _cohort_values / _as_str_series) so e.g. a reward_group array stored
    as int 0/1 still correctly matches cohort_labels configured as strings.
    """
    area_col = cfg['area_col']
    mouse_col = cfg['mouse_col']
    cohort_col = cfg['cohort_col']
    cohort_values = _cohort_values(cfg)

    required = {'area_col': area_col, 'mouse_col': mouse_col, 'cohort_col': cohort_col}
    missing = {k: v for k, v in required.items() if v not in neuron_meta}
    if missing:
        raise KeyError(
            f"neuron_meta is missing field(s) expected by the MMD config: "
            f"{missing}. Available neuron_meta fields: {list(neuron_meta)}. "
            f"Fix the corresponding key(s) in DEFAULT_MMD_CFG, or override them "
            f"via an `mmd:` section in config.yaml (passed as config_path= to "
            f"run_all())."
        )

    df = pd.DataFrame({
        area_col: neuron_meta[area_col],
        mouse_col: neuron_meta[mouse_col],
        cohort_col: _as_str_series(pd.Series(neuron_meta[cohort_col])).values,
    })

    observed_cohorts = set(df[cohort_col].dropna().unique())
    if not set(cohort_values) & observed_cohorts:
        raise ValueError(
            f"None of the configured cohort_labels {cfg['cohort_labels']} "
            f"(compared as strings: {cohort_values}) were found in "
            f"neuron_meta['{cohort_col}']. Observed values there (as strings): "
            f"{sorted(observed_cohorts)}. Set 'cohort_labels' (and/or "
            f"'cohort_col') via an `mmd:` section in config.yaml to match."
        )

    if cfg.get('min_odd_even_r') is not None:
        if 'odd_even_r' not in neuron_meta:
            raise KeyError(
                "cfg['min_odd_even_r'] is set but neuron_meta has no 'odd_even_r' field.")
        n_before = len(df)
        df = df[neuron_meta['odd_even_r'] >= cfg['min_odd_even_r']]
        print(f"[MMD] odd/even reliability filter (>= {cfg['min_odd_even_r']}): "
              f"{n_before} -> {len(df)} neurons")

    valid_areas_per_cohort = {}
    for cohort in cohort_values:
        sub = df[df[cohort_col] == cohort]
        counts = (
            sub.groupby([area_col, mouse_col]).size()
               .reset_index(name='n_neurons')
        )
        counts = counts[counts['n_neurons'] >= cfg['min_neurons_per_mouse']]
        mice_per_area = counts.groupby(area_col)[mouse_col].nunique()
        valid_areas_per_cohort[cohort] = set(
            mice_per_area[mice_per_area >= cfg['min_mice_per_area']].index
        )
        print(f"[MMD] cohort '{cohort}': {len(sub)} neurons, "
              f"{sub[mouse_col].nunique()} mice total, "
              f"{len(valid_areas_per_cohort[cohort])} areas pass "
              f">= {cfg['min_mice_per_area']} mice with "
              f">= {cfg['min_neurons_per_mouse']} neurons each")

    valid_areas = set.intersection(*valid_areas_per_cohort.values())
    valid_areas = [a for a in valid_areas if a not in ['Olfactory areas', 'Insular areas', ]]
    print(f"[MMD] {len(valid_areas)} areas valid in BOTH cohorts (intersection)")
    return sorted(valid_areas)


# --------------------------------------------------------------------------
# Feature-matrix / neuron-selection helpers
# --------------------------------------------------------------------------

def _slice_feature_matrix(feature_matrix, trial_type_slices, exclude_bins=None,
                           exclude_from_full=None):
    """
    Returns dict {'full': matrix, <trial_type_name>: sub-matrix, ...}.

    trial_type_slices: dict {name: (start_idx, end_idx)} of COLUMN index
    ranges into feature_matrix. See _build_default_trial_type_slices for
    how this is auto-derived from n_bins_list when not given explicitly.

    exclude_bins: dict {name: [(rel_start, rel_end), ...]} or None. Index
    ranges LOCAL to that trial type's own slice (0 = first bin of that
    trial type's window) to drop from it -- e.g. a baseline window right
    after alignment. Applied to that trial type's own sub-matrix AND
    wherever it feeds into 'full' (see below); never seen by any
    downstream PCA fit either, since PCA is fit on these already-masked
    matrices.

    exclude_from_full: list of trial-type names to leave OUT of 'full'
    entirely (e.g. ['passive_pre']) -- excluded trial types are still
    returned as their own standalone dict entries (so they're still
    tested individually), just never concatenated into 'full'.

    NOTE: 'full' is built by concatenating each trial type's OWN
    (possibly bin-masked) sub-matrix, rather than being a raw untouched
    copy of `feature_matrix` -- this is required so that exclude_bins and
    exclude_from_full actually take effect on 'full' too, not just on the
    per-trial-type entries. If trial_type_slices is None, 'full' is just
    `feature_matrix` unchanged (no masking possible without slice
    boundaries).
    """
    exclude_bins = exclude_bins or {}
    exclude_from_full = set(exclude_from_full or [])
    out = {}
    if not trial_type_slices:
        return {'full': feature_matrix}

    full_parts = []
    for name, (start, end) in trial_type_slices.items():
        sub = feature_matrix[:, start:end]
        excl = exclude_bins.get(name)
        if excl:
            n_local = end - start
            keep = np.ones(n_local, dtype=bool)
            for (rs, re) in excl:
                keep[rs:re] = False
            sub = sub[:, keep]
        out[name] = sub
        if name not in exclude_from_full:
            full_parts.append(sub)

    out['full'] = np.concatenate(full_parts, axis=1) if full_parts else feature_matrix
    return out


def _build_default_trial_type_slices(n_bins_list, cond_labels=None):
    """Auto-build {name: (start_idx, end_idx)} column-range slices from
    n_bins_list's per-condition bin counts (cumulative offsets), one slice
    per condition. feature_matrix.npz does not store condition text labels,
    so names default to cond0, cond1, ... unless cond_labels is given
    explicitly (e.g. from get_conditions(cfg) if you have that cfg handy --
    see _get_cond_labels_from_pipeline_cfg, which run_all calls
    automatically)."""
    n_bins_list = np.asarray(n_bins_list, dtype=int)
    offsets = np.concatenate([[0], np.cumsum(n_bins_list)])
    names = (list(cond_labels) if cond_labels is not None
             else [f"cond{i}" for i in range(len(n_bins_list))])
    if len(names) != len(n_bins_list):
        raise ValueError(
            f"cond_labels has {len(names)} entries but n_bins_list has "
            f"{len(n_bins_list)} -- they must match 1:1. This usually means "
            f"the pipeline cfg passed to get_conditions() (period/modality/"
            f"merge_passive/include_lick_conditions) doesn't match the "
            f"config actually used to build this feature_matrix.npz -- set "
            f"cfg['cond_labels'] explicitly under `mmd:` in config.yaml to "
            f"override, or fix the mismatch upstream."
        )
    return {name: (int(offsets[i]), int(offsets[i + 1])) for i, name in enumerate(names)}


def _get_cond_labels_from_pipeline_cfg(pipeline_cfg):
    """
    Derive REAL condition names (in the same order n_bins_list was built,
    per build_feature_matrix.py) from the shared pipeline's own
    get_conditions(cfg) in rastermap_utils.py -- e.g. "Whisker pre",
    "Whisker hit", "Auditory hit" -- instead of falling back to
    placeholder cond0/cond1/... names.

    `pipeline_cfg` must be the FULL pipeline config (as loaded from
    config.yaml), NOT the `mmd:` sub-dict -- get_conditions reads
    pipeline-wide keys (period, modality, merge_passive,
    include_lick_conditions) that live at the top level of config.yaml,
    the same ones build_feature_matrix.py itself reads.

    The import is deliberately LAZY and best-effort: rastermap_utils.py
    pulls in a heavy, lab-specific dependency chain (ephys_utilities,
    rastermap, umap, cmasher, ...) that may not be installed in every
    environment this module runs in. If the import fails for any reason,
    this prints a note and returns None -- callers (run_all) fall back to
    the cond0/cond1/... placeholder names in that case rather than raising.

    Returns
    -------
    list of str, or None if rastermap_utils/get_conditions isn't usable.
    """
    try:
        from rastermap_psth.rastermap_utils import get_conditions
    except Exception as e:
        print(f"[MMD] Could not import get_conditions from rastermap_utils "
              f"({type(e).__name__}: {e}) -- falling back to cond0/cond1/... "
              f"names. Set cfg['cond_labels'] explicitly under `mmd:` in "
              f"config.yaml if you want real names without fixing the import.")
        return None
    try:
        _, cond_labels, _, _, _ = get_conditions(pipeline_cfg)
    except Exception as e:
        print(f"[MMD] get_conditions(cfg) raised ({type(e).__name__}: {e}) -- "
              f"falling back to cond0/cond1/... names.")
        return None
    return list(cond_labels)


def _maybe_pca_reduce(feats, cfg):
    """
    Optionally PCA-reduce `feats` (N x d, ALREADY neuron- and
    condition-selected for one specific test) before it goes into
    mouse_level_permutation_test -- fit fresh on exactly this pooled,
    label-blind selection (never globally across the whole dataset), so
    each area/feature-version test's PCA basis reflects only its own data.

    Controlled by cfg['pca_test_n_components'] (int) or
    cfg['pca_test_variance_retained'] (float in (0, 1]); set at most one.
    Default is pca_test_variance_retained=0.99. Returns (feats, None)
    UNCHANGED if neither is set. Callers add this as an ADDITIONAL feature
    version alongside the untouched original, not a replacement -- see
    run_mmd_aggregated / run_mmd_per_area.

    Returns
    -------
    feats_out : ndarray -- either the original `feats` (if PCA disabled)
        or its projection onto the fitted components.
    pca : fitted sklearn PCA object, or None if PCA disabled. Kept around
        so callers can document the reduction (see _fig_pca_variance) and,
        for cross-validation, transform a SEPARATE held-out matrix into
        this same fitted basis (pca.transform(...)).
    """
    n_comp = cfg.get('pca_test_n_components')
    var_ret = cfg.get('pca_test_variance_retained')
    if n_comp is None and var_ret is None:
        return feats, None
    if n_comp is not None and var_ret is not None:
        raise ValueError(
            "Set only one of cfg['pca_test_n_components'] / "
            "cfg['pca_test_variance_retained'], not both.")
    n_samples, n_features = feats.shape
    max_comp = min(n_samples, n_features) - 1  # PCA needs n_components < min(n,d)
    if max_comp < 1:
        raise ValueError(
            f"Too few samples/features ({feats.shape}) for any PCA reduction.")
    if n_comp is not None:
        k = min(n_comp, max_comp)
        pca = PCA(n_components=k)
    else:
        pca = PCA(n_components=var_ret, svd_solver='full')
    feats_out = pca.fit_transform(feats)
    return feats_out, pca


def _select_neurons(neuron_meta, cfg, area=None, areas=None):
    """
    Boolean mask -> integer positions for neurons matching validity filters
    and (optionally) a single area or a set of areas. Exactly one of
    `area` / `areas` should be given; if neither, only cohort (and, if
    configured, odd/even reliability) filters apply.

    Cohort is compared as strings (see _as_str_series/_cohort_values) so a
    numeric-coded array (e.g. reward_group stored as int 0/1) still matches
    cohort_labels configured as strings, and vice versa.
    """
    area_col = cfg['area_col']
    cohort_col = cfg['cohort_col']
    cohort_values = _cohort_values(cfg)

    cohort_arr = _as_str_series(pd.Series(neuron_meta[cohort_col])).values
    mask = np.isin(cohort_arr, cohort_values)

    if cfg.get('min_odd_even_r') is not None:
        if 'odd_even_r' not in neuron_meta:
            raise KeyError(
                "cfg['min_odd_even_r'] is set but neuron_meta has no 'odd_even_r' field.")
        mask &= (neuron_meta['odd_even_r'] >= cfg['min_odd_even_r'])

    if area is not None:
        mask &= (neuron_meta[area_col] == area)
    elif areas is not None:
        mask &= np.isin(neuron_meta[area_col], list(areas))

    pos = np.where(mask)[0]
    return pos


def _get_neuron_data(neuron_meta, feature_matrix, cfg, area=None, areas=None):
    """
    Returns (features, mouse_ids, cohort_labels, unit_ids) for the
    selection. `cohort_labels` is returned as STRING-cast values (see
    _as_str_series), matching _cohort_values(cfg) -- so callers comparing
    against cfg['cohort_labels'] must use _cohort_values(cfg) rather than
    the raw tuple, to stay dtype-consistent regardless of the original
    array's dtype (int/bool/str all normalize to the same strings).
    """
    pos = _select_neurons(neuron_meta, cfg, area=area, areas=areas)
    feats = feature_matrix[pos]
    mice = np.asarray(neuron_meta[cfg['mouse_col']])[pos]
    cohorts = _as_str_series(pd.Series(neuron_meta[cfg['cohort_col']]))[pos].values
    unit_ids = np.asarray(neuron_meta['unit_ids'])[pos] if 'unit_ids' in neuron_meta else pos
    return feats, mice, cohorts, unit_ids


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _fig_mmd_null_histogram(res, title=''):
    """Null distribution with observed MMD^2 marked -- this is the actual
    evidence for/against significance; scatter/witness panels below are
    interpretive companions, not substitutes."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(res['null_mmd2'], bins=40, color='0.75', edgecolor='0.4',
             label='Permutation null (mouse-level)')
    ax.axvline(res['mmd2_obs'], color='crimson', lw=2, label='Observed MMD$^2$')
    ax.set_xlabel(r'MMD$^2_u$')
    ax.set_ylabel('Count')
    perm_kind = 'exact enumeration' if res['used_exact'] else 'Monte Carlo'
    ax.set_title(
        f"{title}\np={res['p_value']:.4g}  ({perm_kind}, n={res['n_perm_used']}, "
        f"n_mice 1={res['n_mice_pos']}, 0={res['n_mice_neg']})",
        fontsize=8
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _fig_mmd_scatter_witness(feats, cohorts, cohort_values, sigma2,
                              n_components=2, title=''):
    """
    Two-panel illustration:
      (1) PCA scatter + pe0cohort KDE contours -- for intuition only; may
          miss the informative direction if it's not within PC1-PC2.
      (2) Same PCA projection, colored by the MMD witness function
          evaluated in the ORIGINAL high-dimensional feature space using
          the same sigma^2 as the test -- faithful to what MMD detected.
    """
    pos_label, neg_label = cohort_values
    pos_mask = cohorts == pos_label
    neg_mask = cohorts == neg_label

    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(feats)

    X = feats[pos_mask]
    Y = feats[neg_mask]
    witness = mmd_witness_function(feats, X, Y, sigma2)

    fig = plt.figure(figsize=(10, 4.5))
    gs = gridspec.GridSpec(1, 2, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(proj[pos_mask, 0], proj[pos_mask, 1], s=8, alpha=0.5,
                color='tab:blue', label=pos_label)
    ax0.scatter(proj[neg_mask, 0], proj[neg_mask, 1], s=8, alpha=0.5,
                color='tab:orange', label=neg_label)
    try:
        import seaborn as sns
        if pos_mask.sum() > 5:
            sns.kdeplot(x=proj[pos_mask, 0], y=proj[pos_mask, 1], ax=ax0,
                        color='tab:blue', levels=3, linewidths=1)
        if neg_mask.sum() > 5:
            sns.kdeplot(x=proj[neg_mask, 0], y=proj[neg_mask, 1], ax=ax0,
                        color='tab:orange', levels=3, linewidths=1)
    except ImportError:
        pass
    ax0.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax0.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax0.set_title('Neuron scatter (PCA, illustration only)', fontsize=9)
    ax0.legend(fontsize=8)

    ax1 = fig.add_subplot(gs[0, 1])
    vmax = np.abs(witness).max()
    vmax = vmax if vmax > 0 else 1.0
    sc = ax1.scatter(proj[:, 0], proj[:, 1], c=witness, cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax, s=10)
    fig.colorbar(sc, ax=ax1, label=f'witness f(z)  (+{pos_label} / -{neg_label})')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('MMD witness function\n(computed in full-dim feature space)',
                  fontsize=9)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def _spectral_seriation_order(K_block, max_dense_n=2500):
    """
    Order the rows/columns of a similarity matrix K_block by their
    position along the FIEDLER VECTOR (the eigenvector of the second-
    smallest eigenvalue of the graph Laplacian L = I - D^-1/2 K D^-1/2).

    This is the standard spectral seriation solution: it directly
    minimizes the sum of squared index-distances weighted by similarity,
    which is exactly "put similar things next to each other, smoothly" --
    unlike hierarchical-clustering leaf order, which is constrained to
    whatever binary-merge tree the linkage happened to build and can
    place an outlier point in the MIDDLE of an otherwise smooth block
    (producing stray cross-cutting lines in the similarity-matrix figure).

    Falls back to hierarchical-clustering leaf order for blocks larger
    than `max_dense_n` (dense eigendecomposition cost grows steeply);
    prints a note when this happens rather than silently changing method.

    Returns
    -------
    order : ndarray of int, positions into K_block's own rows/columns.
    """
    n = K_block.shape[0]
    if n > max_dense_n:
        print(f"[MMD] block of {n} neurons exceeds max_dense_n={max_dense_n} "
              f"for spectral seriation -- falling back to hierarchical-"
              f"clustering leaf order for this block (still similarity-based, "
              f"just less smooth for large N).")
        D_block = 1.0 - K_block
        np.fill_diagonal(D_block, 0.0)
        D_block = (D_block + D_block.T) / 2.0
        Z = hierarchy.linkage(squareform(D_block, checks=False), method='average')
        return hierarchy.leaves_list(Z)

    d = K_block.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.clip(d, 1e-12, None))
    L = np.eye(n) - (d_inv_sqrt[:, None] * K_block * d_inv_sqrt[None, :])
    L = (L + L.T) / 2.0  # guard against float asymmetry
    # Only the two smallest eigenpairs are needed (index 0 is the trivial
    # ~constant eigenvector with eigenvalue ~0; index 1 is the Fiedler
    # vector) -- subset_by_index is much cheaper than a full decomposition.
    _, eigvecs = eigh(L, subset_by_index=[0, 1])
    fiedler = eigvecs[:, 1]
    return np.argsort(fiedler)


def _similarity_order_within_cohorts(feats, cohorts, cohort_values, sigma2):
    """
    Reorder neurons so that, WITHIN each cohort block, neurons follow a
    SMOOTH similarity gradient (spectral/Fiedler-vector seriation -- see
    _spectral_seriation_order), rather than by mouse or arbitrarily. This
    makes the within-cohort brightness and cross-cohort dimness in the
    kernel similarity matrix visually much clearer, without the stray
    cross-cutting lines hierarchical-clustering leaf order can produce
    when an outlier neuron gets grafted mid-block onto the dendrogram.

    Returns
    -------
    order : ndarray of int
        Positions into the original (feats, cohorts, ...) arrays giving
        the new neuron order: pos_label block first (seriation-ordered),
        then neg_label block (seriation-ordered).
    """
    pos_label, neg_label = cohort_values
    gamma = 1.0 / (2.0 * sigma2)
    order_blocks = []
    for label in (pos_label, neg_label):
        block_idx = np.where(cohorts == label)[0]
        if len(block_idx) < 3:
            order_blocks.append(block_idx)  # too few to seriate meaningfully
            continue
        K_block = rbf_kernel(feats[block_idx], feats[block_idx], gamma=gamma)
        order = _spectral_seriation_order(K_block)
        order_blocks.append(block_idx[order])
    return np.concatenate(order_blocks)



def _fig_mmd_method_overview(feats, mice, cohorts, cohort_values, sigma2,
                              res, title='', neuron_order='similarity'):
    """
    Didactic, six-panel figure walking through the whole method end-to-end
    on ONE example (area x feature_version), using the actual numbers from
    that run rather than a schematic. Intended for methods sections/talks;
    use _fig_mmd_null_histogram and _fig_mmd_scatter_witness for routine
    pe0area diagnostics instead.

    A. Input feature matrix (z-scored PSTH per neuron), neurons ordered per
       `neuron_order` -- what actually goes into the test.
    B. Pairwise squared-distance histogram (pooled, label-blind) with the
       median-heuristic sigma^2 marked -- how the kernel bandwidth is set.
    C. Full kernel similarity matrix, same neuron order as A, with a line
       marking the cohort block boundary -- visualizes the within-cohort
       vs cross-cohort similarity blocks that MMD^2 compares.
    D. Bar chart of the three MMD^2 terms (mean within-X, mean within-Y,
       mean cross) and how they combine into the observed MMD^2.
    E. Mouse-level permutation schematic: true cohort assignment of each
       mouse vs one example random relabeling, showing that a mouse's
       full neuron block always moves together.
    F. Resulting null distribution with the observed MMD^2 marked.

    Parameters
    ----------
    neuron_order : 'similarity' or 'mouse'
        'similarity' (default): within each cohort block, order neurons by
        hierarchical-clustering leaf order on kernel dissimilarity -- makes
        the within/cross-cohort block contrast in panel C much clearer.
        'mouse': order by cohort then mouse ID (useful if you want to see
        mouse-level structure instead of similarity structure).
    """
    pos_label, neg_label = cohort_values

    neuron_order = 'mouse'
    if neuron_order == 'similarity':
        order = _similarity_order_within_cohorts(feats, cohorts, cohort_values, sigma2)
        order_desc = 'cohort, similarity'
    elif neuron_order == 'mouse':
        order = np.lexsort((mice, cohorts != pos_label))
        order_desc = 'cohort, mouse'
    else:
        raise ValueError(f"neuron_order must be 'similarity' or 'mouse', got {neuron_order!r}")

    feats_ord = feats[order]
    cohorts_ord = cohorts[order]
    mice_ord = mice[order]
    n_pos = int((cohorts_ord == pos_label).sum())

    X = feats[cohorts == pos_label]
    Y = feats[cohorts == neg_label]
    terms = _mmd_kernel_terms(X, Y, sigma2)
    gamma = 1.0 / (2.0 * sigma2)

    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    # --- A: input feature matrix ---
    axA = fig.add_subplot(gs[0, 0])
    vmax = np.percentile(np.abs(feats_ord), 99) if feats_ord.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    imA = axA.imshow(feats_ord, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axA.axhline(n_pos - 0.5, color='k', lw=1.5)
    axA.set_ylabel(f'Neurons (sorted: {order_desc})')
    axA.set_xlabel('Concatenated PSTH bins')
    axA.set_title('A. Input feature matrix\n(z-scored PSTH per neuron)', fontsize=9)
    fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.08, label='z-score')

    # --- B: pairwise distance histogram + bandwidth ---
    axB = fig.add_subplot(gs[0, 1])
    pooled = np.vstack([X, Y])
    sq_d = pdist(pooled, metric='sqeuclidean')
    axB.hist(sq_d, bins=50, color='0.75', edgecolor='0.4')
    axB.axvline(sigma2, color='crimson', lw=2, label=r'median heuristic $\sigma^2$')
    axB.set_xlabel(r'pairwise $\|x_i-x_j\|^2$')
    axB.set_ylabel('Count')
    axB.set_title('B. Kernel bandwidth selection\n(pooled, label-blind)', fontsize=9)
    axB.legend(fontsize=7)

    # --- C: kernel similarity matrix, block structure ---
    axC = fig.add_subplot(gs[0, 2])
    K_full = rbf_kernel(feats_ord, feats_ord, gamma=gamma)
    imC = axC.imshow(K_full, cmap='viridis', vmin=0, vmax=1)
    axC.axhline(n_pos - 0.5, color='w', lw=1.2)
    axC.axvline(n_pos - 0.5, color='w', lw=1.2)
    axC.set_title('C. Kernel similarity matrix\n(within/cross-cohort blocks)', fontsize=9)
    axC.set_xlabel('neuron j')
    axC.set_ylabel('neuron i')
    fig.colorbar(imC, ax=axC, fraction=0.046, pad=0.08, label='k(i,j)')

    # --- D: MMD^2 term decomposition ---
    axD = fig.add_subplot(gs[1, 0])
    bar_labels = [f'within {pos_label}\n(term_x)', f'within {neg_label}\n(term_y)',
                  'cross\n(term_xy)']
    vals = [terms['term_x'], terms['term_y'], terms['term_xy']]
    bars = axD.bar(bar_labels, vals, color=['tab:blue', 'tab:orange', '0.5'])
    for b, v in zip(bars, vals):
        axD.text(b.get_x() + b.get_width() / 2, v, f'{v:.3f}',
                  ha='center', va='bottom', fontsize=8)
    axD.set_ylabel('mean kernel similarity')
    axD.set_title(
        f"D. MMD$^2$ term decomposition\n"
        f"term_x + term_y - 2*term_xy = {terms['mmd2']:.4f}",
        fontsize=9
    )

    # --- E: mouse-level permutation schematic ---
    axE = fig.add_subplot(gs[1, 1])
    unique_mice = list(dict.fromkeys(mice_ord))
    mouse_cohort_map = {m: cohorts_ord[mice_ord == m][0] for m in unique_mice}
    colors_true = ['tab:blue' if mouse_cohort_map[m] == pos_label else 'tab:orange'
                   for m in unique_mice]
    rng_demo = np.random.default_rng(1)
    n_pos_mice = sum(1 for m in unique_mice if mouse_cohort_map[m] == pos_label)
    perm_pos_mice = set(rng_demo.choice(unique_mice, size=n_pos_mice, replace=False))
    colors_perm = ['tab:blue' if m in perm_pos_mice else 'tab:orange'
                   for m in unique_mice]

    x = np.arange(len(unique_mice))
    axE.bar(x, np.ones_like(x, dtype=float), color=colors_true,
            edgecolor='k', linewidth=0.5)
    axE.bar(x, -np.ones_like(x, dtype=float), color=colors_perm,
            edgecolor='k', linewidth=0.5)
    axE.axhline(0, color='k', lw=1)
    axE.set_yticks([0.5, -0.5])
    axE.set_yticklabels(['true\ncohort', 'one example\npermutation'], fontsize=8)
    axE.set_xticks(x)
    axE.set_xticklabels([f'm{i}' for i in range(len(unique_mice))],
                         fontsize=6, rotation=90)
    axE.set_title("E. Mouse-level permutation\n"
                  "(each mouse's full neuron block moves together)", fontsize=9)
    handles = [mpatches.Patch(color='tab:blue', label=pos_label),
               mpatches.Patch(color='tab:orange', label=neg_label)]
    axE.legend(handles=handles, fontsize=7, loc='upper right')

    # --- F: resulting null distribution ---
    axF = fig.add_subplot(gs[1, 2])
    axF.hist(res['null_mmd2'], bins=40, color='0.75', edgecolor='0.4')
    axF.axvline(res['mmd2_obs'], color='crimson', lw=2, label='observed MMD$^2$')
    axF.set_xlabel(r'MMD$^2_u$')
    axF.set_ylabel('Count')
    axF.set_title(f"F. Resulting null distribution\np={res['p_value']:.4g}", fontsize=9)
    axF.legend(fontsize=7)

    fig.suptitle(title, fontsize=12)
    return fig


# --------------------------------------------------------------------------
# PCA documentation figures (for the '_pca' test variant)
# --------------------------------------------------------------------------

def _fig_pca_variance(pca, title=''):
    """
    Documents the PCA dimensionality reduction used for a '_pca' test
    variant: scree plot (per-component variance) + cumulative-variance
    curve, with the retained-variance target and number of retained
    components marked. One of these per (area/aggregated, feature_version)
    PCA test, saved under pca_variance/.
    """
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n_keep = pca.n_components_
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    axes[0].plot(np.arange(1, len(evr) + 1), evr, 'o-', ms=3, color='tab:blue')
    axes[0].set_xlabel('Principal component')
    axes[0].set_ylabel('Variance explained')
    axes[0].set_title('Scree plot', fontsize=9)

    axes[1].plot(np.arange(1, len(cum) + 1), cum, 'o-', ms=3, color='tab:green')
    axes[1].axhline(cum[n_keep - 1], color='crimson', ls='--', lw=1)
    axes[1].axvline(n_keep, color='crimson', ls='--', lw=1)
    axes[1].set_xlabel('Principal component')
    axes[1].set_ylabel('Cumulative variance explained')
    axes[1].set_title(
        f'{n_keep} PCs retain {cum[n_keep - 1] * 100:.1f}% variance', fontsize=9
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def _fig_pc_loading_timecourse(pca, feats_pca, cohorts, cohort_values, top_k=3, title=''):
    """
    For a PCA-reduced test: correlates each retained PC's score with
    cohort membership (Pearson correlation against a 0/1 cohort
    indicator -- equivalent to a point-biserial correlation), then plots
    the loading vector (back in the original bin-space this test was run
    on) of the top_k PCs by |correlation| -- shows which temporal features
    the most cohort-discriminative components actually correspond to.
    """
    pos_label, neg_label = cohort_values
    cohorts_str = _as_str_series(pd.Series(cohorts)).values
    y = (cohorts_str == pos_label).astype(float)
    n_pc = feats_pca.shape[1]
    with np.errstate(invalid='ignore'):
        corrs = np.array([
            np.corrcoef(feats_pca[:, i], y)[0, 1] if np.std(feats_pca[:, i]) > 0 else 0.0
            for i in range(n_pc)
        ])
    corrs = np.nan_to_num(corrs)
    order = np.argsort(-np.abs(corrs))[:min(top_k, n_pc)]

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), gridspec_kw={'height_ratios': [1, 2]})
    colors = ['crimson' if i in order else '0.6' for i in range(n_pc)]
    axes[0].bar(np.arange(1, n_pc + 1), corrs, color=colors)
    axes[0].set_xlabel('Principal component')
    axes[0].set_ylabel('corr(PC score, cohort)')
    axes[0].set_title('Which PCs actually track cohort', fontsize=9)

    x = np.arange(pca.components_.shape[1])
    for i in order:
        axes[1].plot(x, pca.components_[i], label=f'PC{i + 1} (r={corrs[i]:.2f})')
    axes[1].axhline(0, color='0.7', lw=0.7)
    axes[1].set_xlabel('Bin (within the tested feature vector)')
    axes[1].set_ylabel('PC loading')
    axes[1].legend(fontsize=7)
    axes[1].set_title(f"Top {len(order)} cohort-correlated PCs' loadings "
                       f"(what temporal shape drives separation)", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Characterization: going back to the data to see what drives the MMD result
# --------------------------------------------------------------------------

def _fig_avg_psth_perbin_sig(features_raw, cohorts, cohort_values, fdr_alpha=0.05, title=''):
    """
    Cohort-average PSTH (mean +/- SEM) overlay, with a per-bin two-sample
    test (Welch's t-test) time-course underneath (BH-FDR corrected across
    bins) -- localizes WHEN in the trace the cohorts differ, complementing
    the omnibus MMD^2 result with a classical, easy-to-explain figure.

    CAVEAT (stated explicitly in the figure title): this only shows the
    part of the distributional difference visible in the MEAN. MMD^2 can
    be sensitive to differences in variance/shape/multimodality that a
    mean-trace overlay will not reveal at all -- treat this as one piece
    of the picture, not a full explanation of a significant MMD^2 result.

    `features_raw` must be the ORIGINAL (non-PCA-reduced) feature vectors
    for this to be interpretable bin-by-bin -- never call this on the
    PCA-space test's features.

    Returns
    -------
    fig, qvals : the figure, and the per-bin FDR q-values (so callers can
        report e.g. the fraction of significant bins alongside the figure).
    """
    from scipy.stats import ttest_ind
    pos_label, neg_label = cohort_values
    cohorts_str = _as_str_series(pd.Series(cohorts)).values
    pos_mask = cohorts_str == pos_label
    neg_mask = cohorts_str == neg_label
    Xp, Xn = features_raw[pos_mask], features_raw[neg_mask]
    mean_p = Xp.mean(axis=0)
    sem_p = Xp.std(axis=0) / np.sqrt(max(int(pos_mask.sum()), 1))
    mean_n = Xn.mean(axis=0)
    sem_n = Xn.std(axis=0) / np.sqrt(max(int(neg_mask.sum()), 1))

    with np.errstate(invalid='ignore'):
        tvals, pvals = ttest_ind(Xp, Xn, axis=0, equal_var=False)
    pvals = np.nan_to_num(pvals, nan=1.0)
    _, qvals, _, _ = multipletests(pvals, alpha=fdr_alpha, method='fdr_bh')
    sig = qvals < fdr_alpha

    x = np.arange(features_raw.shape[1])
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1]})
    axes[0].fill_between(x, mean_p - sem_p, mean_p + sem_p, color='tab:blue', alpha=0.2)
    axes[0].fill_between(x, mean_n - sem_n, mean_n + sem_n, color='tab:orange', alpha=0.2)
    axes[0].plot(x, mean_p, color='tab:blue', label=pos_label)
    axes[0].plot(x, mean_n, color='tab:orange', label=neg_label)
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel('mean +/- SEM (z-score)')
    axes[0].set_title(
        f"{title}\nCohort-average PSTH -- shows only the MEAN-level part "
        f"of the distributional difference", fontsize=9
    )

    neglog10p = -np.log10(np.clip(pvals, 1e-300, 1))
    axes[1].plot(x, neglog10p, color='k', lw=0.8)
    axes[1].fill_between(x, 0, neglog10p, where=sig, color='crimson', alpha=0.4, step='mid')
    axes[1].axhline(-np.log10(fdr_alpha), color='0.5', ls='--', lw=0.8,
                     label=f'FDR alpha={fdr_alpha}')
    axes[1].set_ylabel('-log10(p)\n(red fill = FDR-sig bin)')
    axes[1].set_xlabel('Bin')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    return fig, qvals


def _fig_witness_weighted_trace(features_raw, witness, title=''):
    """
    Synthetic 'representative differential PSTH': each neuron's raw trace
    weighted by its leave-one-out witness value and averaged (normalized
    by the sum of |witness|) -- a single trace showing the temporal SHAPE
    the kernel found most discriminative, as opposed to picking a handful
    of individual extreme neurons (see extract_extreme_neurons /
    _fig_extreme_neuron_traces for that complementary view).
    `features_raw` must be the ORIGINAL (non-PCA-reduced) feature vectors.
    """
    w = witness / (np.sum(np.abs(witness)) + 1e-12)
    diff_trace = (features_raw * w[:, None]).sum(axis=0)
    x = np.arange(features_raw.shape[1])
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, diff_trace, color='k')
    ax.axhline(0, color='0.7', lw=0.7)
    ax.set_xlabel('Bin')
    ax.set_ylabel('witness-weighted amplitude (a.u.)')
    ax.set_title(f"{title}\nWitness-weighted differential trace", fontsize=9)
    fig.tight_layout()
    return fig


def _crosstab_extreme_vs_cluster(extreme_df, neuron_meta, cluster_col):
    """
    OPTIONAL, best-effort: if `cluster_col` is set AND present in
    neuron_meta (e.g. an existing rastermap cluster id carried alongside
    the other per-neuron metadata), cross-tabulate extreme/typical-neuron
    roles (see extract_extreme_neurons) against cluster membership --
    connects this cluster-free MMD result back to the cluster-based story
    (does MMD find the same neurons an existing clustering already
    separated, or something clustering missed?).

    Returns None if cluster_col is falsy or not present in neuron_meta --
    most npz-only runs of this module won't have cluster labels available,
    and that's fine; this is a bonus cross-reference, not a requirement.
    """
    if not cluster_col or cluster_col not in neuron_meta:
        return None
    cluster_arr = np.asarray(neuron_meta[cluster_col])
    unit_ids_all = np.asarray(neuron_meta.get('unit_ids', np.arange(len(cluster_arr))))
    id_to_cluster = dict(zip(unit_ids_all, cluster_arr))
    extreme_df = extreme_df.copy()
    extreme_df['cluster'] = extreme_df['unit_id'].map(id_to_cluster)
    return pd.crosstab(extreme_df['role'], extreme_df['cluster'])



# --------------------------------------------------------------------------
# Downstream: per-neuron extremity ranking, effect-size figure, example traces
# --------------------------------------------------------------------------

def extract_extreme_neurons(features, cohorts, cohort_values, sigma2, unit_ids=None,
                             top_k=10):
    """
    Rank every neuron by its leave-one-out witness value (see
    mmd_witness_function_loo) and pull out the most extreme members of
    each group, in both directions:

      'typical_pos' / 'typical_neg' : the top_k neurons MOST representative
          of their own group (largest witness magnitude in their own
          group's direction) -- canonical examples of what makes each
          cohort distinct in this test.
      'atypical_pos' / 'atypical_neg' : the top_k neurons from EACH group
          that are LEAST like their own group (witness value closest to,
          or across, zero toward the opposite sign) -- candidates for "this
          neuron sits far into the other cohort's distribution", i.e. the
          answer to "can I extract example neurons that sit far from their
          own cohort / close to the other cohort".

    Returns
    -------
    DataFrame with columns: unit_id, pos_idx (row index into `features` /
    `cohorts` -- use this, not unit_id, to look the neuron's raw feature
    vector back up, e.g. in _fig_extreme_neuron_traces), cohort, witness,
    role, rank (1 = most extreme within that role).
    """
    pos_label, neg_label = cohort_values
    cohorts_str = _as_str_series(pd.Series(cohorts)).values
    witness = mmd_witness_function_loo(features, cohorts, cohort_values, sigma2)
    if unit_ids is None:
        unit_ids = np.arange(len(witness))
    unit_ids = np.asarray(unit_ids)

    pos_mask = cohorts_str == pos_label
    neg_mask = cohorts_str == neg_label

    rows = []

    def _add(mask, descending, role, k):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return
        vals = witness[idx]
        order = np.argsort(-vals if descending else vals)
        for rank, oi in enumerate(order[:k], start=1):
            ci = idx[oi]
            rows.append(dict(unit_id=unit_ids[ci], pos_idx=int(ci),
                              cohort=cohorts_str[ci], witness=float(witness[ci]),
                              role=role, rank=rank))

    # typical: pos neurons with the HIGHEST witness (most pos-like);
    #          neg neurons with the LOWEST witness (most neg-like)
    _add(pos_mask, True, 'typical_pos', top_k)
    _add(neg_mask, False, 'typical_neg', top_k)
    # atypical: pos neurons with the LOWEST witness (least pos-like,
    #           furthest INTO the neg cohort's region); neg neurons with
    #           the HIGHEST witness (least neg-like, furthest into pos)
    _add(pos_mask, False, 'atypical_pos', top_k)
    _add(neg_mask, True, 'atypical_neg', top_k)

    return pd.DataFrame(rows)


def _fig_witness_distribution(witness, cohorts, cohort_values, res, auc, title=''):
    """
    THE single-glance "is there a difference between groups" figure:
    strip + violin of per-neuron leave-one-out witness values, split by
    cohort, annotated with the AUC (witness-as-classifier effect size, on
    [0.5, 1.0]), mmd2_z (standardized MMD^2 effect size, comparable across
    tests with different N/d), and the permutation p-value (the actual
    significance evidence -- see _fig_mmd_null_histogram for its own
    dedicated null-distribution plot).

    If there's a real, detectable difference: the two violins should
    visibly separate and pull away from the dashed zero line in opposite
    directions (pos cohort above 0, neg cohort below), and AUC should sit
    clearly above 0.5.
    """
    pos_label, neg_label = cohort_values
    cohorts_str = _as_str_series(pd.Series(cohorts)).values
    pos_mask = cohorts_str == pos_label
    neg_mask = cohorts_str == neg_label

    fig, ax = plt.subplots(figsize=(4, 4.5))
    parts = ax.violinplot([witness[pos_mask], witness[neg_mask]], positions=[0, 1],
                          showmeans=False, showmedians=True, showextrema=False)
    for pc, color in zip(parts['bodies'], ['tab:blue', 'tab:orange']):
        pc.set_facecolor(color)
        pc.set_alpha(0.25)
    parts['cmedians'].set_color('k')

    rng = np.random.default_rng(0)
    for xi, (mask, color) in enumerate([(pos_mask, 'tab:blue'), (neg_mask, 'tab:orange')]):
        jitter = rng.uniform(-0.08, 0.08, size=int(mask.sum()))
        ax.scatter(xi + jitter, witness[mask], s=14, color=color, alpha=0.7,
                  edgecolors='none', zorder=3)

    ax.axhline(0, color='k', ls='--', lw=0.8, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([pos_label, neg_label])
    ax.set_ylabel(f"Leave-one-out witness f(z)\n(+resembles {pos_label} / "
                 f"-resembles {neg_label})", fontsize=8)
    p_str = f"p={res['p_value']:.3g}" if res['p_value'] >= 1e-3 else "p<0.001"
    mmd2_z = res.get('mmd2_z', np.nan)
    ax.set_title(f"{title}\nAUC={auc:.3f}   MMD\u00b2 z={mmd2_z:.2f}   {p_str}",
                fontsize=9)
    fig.tight_layout()
    return fig


def _fig_extreme_neuron_traces(features, cohorts, cohort_values, extreme_df, role,
                                max_neurons=6, title=''):
    """
    Grid of line plots, one per neuron listed under `role` in extreme_df
    (see extract_extreme_neurons), showing that neuron's raw feature
    vector against the mean +/- SEM of BOTH cohorts -- a visual sanity
    check for "does this neuron actually look different from its own
    group / like the other group across most of the trace, or is the
    witness value being driven by one narrow, possibly artifactual bin".
    """
    pos_label, neg_label = cohort_values
    cohorts_str = _as_str_series(pd.Series(cohorts)).values
    pos_mask = cohorts_str == pos_label
    neg_mask = cohorts_str == neg_label
    mean_pos = features[pos_mask].mean(axis=0)
    sem_pos = features[pos_mask].std(axis=0) / np.sqrt(max(int(pos_mask.sum()), 1))
    mean_neg = features[neg_mask].mean(axis=0)
    sem_neg = features[neg_mask].std(axis=0) / np.sqrt(max(int(neg_mask.sum()), 1))

    sub = extreme_df[extreme_df['role'] == role].sort_values('rank').head(max_neurons)
    n = len(sub)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, f"No neurons for role='{role}'", ha='center', va='center')
        ax.axis('off')
        return fig

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows), squeeze=False)
    x = np.arange(features.shape[1])
    for i, (_, row) in enumerate(sub.iterrows()):
        ax = axes[i // ncols, i % ncols]
        ax.fill_between(x, mean_pos - sem_pos, mean_pos + sem_pos, color='tab:blue', alpha=0.15)
        ax.fill_between(x, mean_neg - sem_neg, mean_neg + sem_neg, color='tab:orange', alpha=0.15)
        ax.plot(x, mean_pos, color='tab:blue', lw=1, label=f"{pos_label} mean")
        ax.plot(x, mean_neg, color='tab:orange', lw=1, label=f"{neg_label} mean")
        neuron_vec = features[int(row['pos_idx'])]
        ax.plot(x, neuron_vec, color='k', lw=1.3, label=f"neuron ({row['cohort']})")
        ax.set_title(f"{row['role']} #{int(row['rank'])}  witness={row['witness']:.3f}\n"
                    f"unit={row['unit_id']}", fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6, frameon=False)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig


def _fig_area_condition_heatmap(results_df, cfg, value_col='mmd2_z'):
    """
    #1: one figure showing the WHOLE brain-by-condition landscape at once
    -- rows = areas, columns = conditions (feature_version), color =
    `value_col` (default mmd2_z, the standardized effect size -- comparable
    across tests with different N/d, unlike raw mmd2_obs), cell annotated
    with significance stars from q_value (BH-FDR across areas, within that
    condition -- the same correction mmd_summary_per_area uses).
    Complements the per-condition forest plot with a single at-a-glance
    view across every area and condition together.

    Uses `results_df` as given -- there is no separate full-dim/PCA-space
    duplication to filter out any more (PCA reduction, when configured,
    is the only test that runs; see run_mmd_per_area).

    Returns None if `results_df` is empty.
    """
    sub = results_df
    if sub.empty:
        return None

    areas = sorted(sub['area'].unique())
    conditions = sorted(sub['feature_version'].unique())
    mat = np.full((len(areas), len(conditions)), np.nan)
    qmat = np.full((len(areas), len(conditions)), np.nan)
    area_idx = {a: i for i, a in enumerate(areas)}
    cond_idx = {c: j for j, c in enumerate(conditions)}
    for _, row in sub.iterrows():
        i, j = area_idx[row['area']], cond_idx[row['feature_version']]
        mat[i, j] = row[value_col]
        qmat[i, j] = row.get('q_value', np.nan)

    def _stars(q):
        if pd.isna(q):
            return ''
        if q < 0.001:
            return '***'
        if q < 0.01:
            return '**'
        if q < 0.05:
            return '*'
        return ''

    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    vmax = vmax if vmax > 0 else 1.0
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(conditions) + 2),
                                     max(4, 0.3 * len(areas) + 1)))
    im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(areas)))
    ax.set_yticklabels(areas, fontsize=7)
    for i in range(len(areas)):
        for j in range(len(conditions)):
            star = _stars(qmat[i, j])
            if star:
                ax.text(j, i, star, ha='center', va='center', fontsize=8, color='black')
    fig.colorbar(im, ax=ax, label=value_col)
    ax.set_title(f"Areas x conditions\n"
                 f"color={value_col}, stars=BH-FDR across areas (q_value)", fontsize=10)
    fig.tight_layout()
    return fig


def _fig_whole_brain_perbin_significance(perbin_qvals_by_area, fdr_alpha=0.05, title=''):
    """
    #5: extend the per-area avg-PSTH-with-per-bin-significance figure
    (_fig_avg_psth_perbin_sig) into one whole-brain matrix -- rows =
    areas, columns = time bins (within the 'full' condition), color =
    -log10(q) for bins that are FDR-significant WITHIN that area's own
    per-bin correction (non-significant bins shown in flat gray) --
    showing where in the brain AND when in time effects concentrate,
    instead of paging through dozens of separate per-area figures.

    Areas are sorted by their own number of significant bins (descending)
    so the areas with the most temporally-widespread effects sit at top.

    perbin_qvals_by_area : dict {area: ndarray of per-bin q-values}
        All arrays are expected to be the same length (same 'full'
        condition, same exclude_bins/trial_type_slices config for every
        area) -- mismatched lengths are handled by truncating to the
        shortest, with a printed warning, rather than raising.
    """
    areas = list(perbin_qvals_by_area.keys())
    if not areas:
        return None
    lengths = {a: len(perbin_qvals_by_area[a]) for a in areas}
    n_bins = min(lengths.values())
    if len(set(lengths.values())) > 1:
        print(f"[MMD] whole-brain per-bin figure: mismatched bin counts across "
              f"areas {lengths} -- truncating all to the shortest ({n_bins}).")

    areas = sorted(areas, key=lambda a: -(perbin_qvals_by_area[a][:n_bins] < fdr_alpha).sum())
    mat = np.full((len(areas), n_bins), np.nan)
    for i, a in enumerate(areas):
        q = np.asarray(perbin_qvals_by_area[a][:n_bins])
        mat[i] = -np.log10(np.clip(q, 1e-300, 1))

    thresh = -np.log10(fdr_alpha)
    display = np.ma.masked_where(mat < thresh, mat)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='0.9')

    fig, ax = plt.subplots(figsize=(max(6, n_bins * 0.12), max(4, len(areas) * 0.22 + 1)))
    im = ax.imshow(display, aspect='auto', cmap=cmap)
    ax.set_yticks(np.arange(len(areas)))
    ax.set_yticklabels(areas, fontsize=7)
    ax.set_xlabel('Bin (within concatenated \'full\' vector)')
    ax.set_title(f"{title}\ngray = not FDR-significant within that area "
                 f"(alpha={fdr_alpha})", fontsize=10)
    fig.colorbar(im, ax=ax, label='-log10(q)  (per-bin, FDR within area)')
    fig.tight_layout()
    return fig


def _fig_mmd_summary(results_df, cfg):
    """Forest-plot-style summary: one row per area, faceted by feature
    version, BH-FDR significance stars."""
    feature_versions = list(results_df['feature_version'].unique())
    n_fv = len(feature_versions)
    n_areas = results_df['area'].nunique()
    fig, axes = plt.subplots(
        1, n_fv, figsize=(4.5 * n_fv, max(3, 0.35 * n_areas)), sharex=False
    )
    if n_fv == 1:
        axes = [axes]

    def _stars(q):
        if pd.isna(q):
            return ''
        if q < 0.001:
            return '***'
        if q < 0.01:
            return '**'
        if q < 0.05:
            return '*'
        return ''

    for ax, fv_name in zip(axes, feature_versions):
        sub = results_df[results_df['feature_version'] == fv_name].sort_values('mmd2_obs')
        y_pos = np.arange(len(sub))
        colors = ['crimson' if (not pd.isna(q) and q < cfg['fdr_alpha']) else '0.6'
                  for q in sub['q_value']]
        ax.barh(y_pos, sub['mmd2_obs'], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub['area'])
        for yi, (mmd2, q) in enumerate(zip(sub['mmd2_obs'], sub['q_value'])):
            star = _stars(q)
            if star:
                ax.text(mmd2, yi, f' {star}', va='center', fontsize=9)
        ax.set_xlabel(r'MMD$^2_u$')
        ax.set_title(fv_name, fontsize=9)
        ax.axvline(0, color='k', lw=0.5)

    fig.suptitle('MMD$^2$ by area (mouse-level permutation, BH-FDR within '
                 'each feature version)', fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------

def _within_area_condition_fdr(results_df, cfg):
    """
    Second FDR axis: WITHIN each area, across its own single-condition
    tests (never 'full' or its '_pca' counterpart, and never a trial type
    listed in cfg['exclude_from_condition_fdr'] -- e.g. passive_pre, which
    is still tested and reported, just excluded from inflating this
    within-area correction since it isn't a comparison target).

    This is IN ADDITION TO the existing across-areas-within-feature-version
    correction (results_df['q_value']) -- the two answer different
    questions ("is this area unusual among areas, for this condition" vs
    "is this condition unusual among this area's own conditions") and are
    reported side by side rather than one replacing the other.

    Returns
    -------
    q_within : pd.Series aligned with results_df.index; NaN for rows not
        in the within-area comparison group (i.e. 'full'/'<x>_pca' and any
        excluded trial type).
    """
    exclude = set(cfg.get('exclude_from_condition_fdr')
                  if cfg.get('exclude_from_condition_fdr') is not None
                  else (cfg.get('exclude_from_full') or []))

    def _base_name(fv):
        return fv[:-4] if fv.endswith('_pca') else fv

    base = results_df['feature_version'].apply(_base_name)
    cond_mask = (base != 'full') & (~base.isin(exclude))

    q_within = pd.Series(np.nan, index=results_df.index)
    for area in results_df.loc[cond_mask, 'area'].unique():
        m = cond_mask & (results_df['area'] == area)
        if m.sum() > 0:
            _, qvals, _, _ = multipletests(
                results_df.loc[m, 'p_value'], alpha=cfg['fdr_alpha'], method='fdr_bh'
            )
            q_within.loc[m] = qvals
    return q_within


def run_mmd_aggregated(neuron_meta, feature_matrix, cfg=None, valid_areas=None,
                        make_figures=True):
    """
    Pool neurons across all valid areas (assumed similarly sampled) into a
    single X/Y comparison. sigma^2 is estimated ONCE, globally, on the
    pooled label-blind aggregate -- not per area -- consistent with the
    "similarly sampled" assumption. Mouse-level permutation shuffles each
    mouse's ENTIRE cross-area neuron complement together, so the mouse (not
    mouse-within-area) remains the exchangeable unit.

    This is usually the SLOWEST single test in the module (largest N,
    pooled across every area) so it runs FIRST -- before the per-area loop
    -- and uses cfg['n_jobs'] BLAS threads on its one big batched matmul
    (see mouse_level_permutation_test / _mmd2_batch_from_kernel), rather
    than being split across processes the way run_mmd_per_area is: a
    single large matmul already saturates many cores via BLAS on its own,
    so process-level parallelism here would just add overhead without
    speeding anything up.

    If PCA reduction is configured (cfg['pca_test_n_components'] /
    cfg['pca_test_variance_retained'] -- the module's default), it is the
    ONLY test run per feature version, fit fresh on that test's own
    pooled neurons (see _maybe_pca_reduce); there is no separate
    full-dimensional test run alongside it any more. Characterization
    figures that need raw bin-space (avg-PSTH-with-per-bin-significance,
    witness-weighted trace) still use the untouched feats_raw regardless
    of whether the STATISTIC itself ran in PCA-reduced space.

    Each test's downstream effect-size/diagnostic bundle (mmd2_z,
    witness-AUC, leave-one-out witness scores, extreme-neuron ranking) is
    computed once on the OBSERVED split and attached to fig_payload for
    the figure step below -- see mmd_witness_function_loo,
    compute_witness_auc, extract_extreme_neurons.
    """
    cfg = {**DEFAULT_MMD_CFG, **(cfg or {})}
    rng = np.random.default_rng(cfg['mmd_random_seed'])
    out_dir = cfg['output_dir']
    n_jobs = cfg.get('n_jobs', N_CORES)

    if valid_areas is None:
        valid_areas = get_valid_areas(neuron_meta, cfg)

    if not valid_areas:
        print("[MMD aggregated] No valid areas -- nothing to pool, skipping "
              "aggregated test entirely (0 valid areas usually means too few "
              "mice/neurons in this data, e.g. in test_n_neurons=... test mode, "
              "or min_mice_per_area/min_neurons_per_mouse too strict for the "
              "data given).")
        return pd.DataFrame(columns=[
            'area', 'feature_version', 'space', 'n_neurons', 'n_areas_pooled',
            'n_mice_pos', 'n_mice_neg', 'mmd2_obs', 'mmd2_z', 'witness_auc',
            'sigma2', 'p_value', 'n_perm_used', 'used_exact', 'q_value',
            'q_value_within_area',
        ])

    feature_versions = _slice_feature_matrix(
        feature_matrix, cfg['trial_type_slices'],
        exclude_bins=cfg.get('exclude_bins'), exclude_from_full=cfg.get('exclude_from_full'),
    )
    n_tests = len(feature_versions)
    print(f"[MMD aggregated] {len(valid_areas)} areas pooled x "
          f"{n_tests} feature versions = {n_tests} tests "
          f"(n_perm={cfg['mmd_n_perm']} each, up to {n_jobs} BLAS threads/test)")
    t_start = time.time()

    records = []
    fig_payload = {}

    for fv_name, fv_matrix in tqdm(feature_versions.items(), desc="MMD aggregated tests"):
        feats_raw, mice, cohorts, unit_ids = _get_neuron_data(
            neuron_meta, fv_matrix, cfg, areas=valid_areas
        )
        feats, pca_obj = _maybe_pca_reduce(feats_raw, cfg)
        label = fv_name

        res = mouse_level_permutation_test(
            feats, mice, cohorts,
            cohort_values=_cohort_values(cfg),
            n_perm=cfg['mmd_n_perm'],
            exact_if_feasible=cfg['mmd_exact_if_feasible'],
            sigma2=None,   # global, label-blind, estimated once here
            rng=rng,
            progress_desc=f"aggregated/{label}",
            blas_threads=n_jobs,   # one big matmul -- use many/all cores directly
        )
        auc, witness = compute_witness_auc(feats, cohorts, _cohort_values(cfg), res['sigma2'])
        extreme_df = extract_extreme_neurons(feats, cohorts, _cohort_values(cfg),
                                             res['sigma2'], unit_ids=unit_ids)

        print(f"[MMD aggregated] {label:<16s} n={feats.shape[0]:>5d}  "
              f"mice={res['n_mice_pos']}v{res['n_mice_neg']}  "
              f"mmd2={res['mmd2_obs']:.4g}  z={res['mmd2_z']:.3g}  "
              f"auc={auc:.3f}  p={res['p_value']:.4g}")

        records.append({
            'area': 'ALL_AREAS_AGGREGATED',
            'feature_version': label,
            'space': 'pca' if pca_obj is not None else 'full',
            'n_neurons': feats.shape[0],
            'n_areas_pooled': len(valid_areas),
            'n_mice_pos': res['n_mice_pos'],
            'n_mice_neg': res['n_mice_neg'],
            'mmd2_obs': res['mmd2_obs'],
            'mmd2_z': res['mmd2_z'],
            'witness_auc': auc,
            'sigma2': res['sigma2'],
            'p_value': res['p_value'],
            'n_perm_used': res['n_perm_used'],
            'used_exact': res['used_exact'],
        })
        fig_payload[label] = (res, feats, feats_raw, mice, cohorts, auc, witness,
                               extreme_df, pca_obj, fv_name)

    print(f"[MMD aggregated] all {n_tests} tests done in "
          f"{time.time() - t_start:.1f}s")

    results_df = pd.DataFrame.from_records(records)
    if len(results_df) > 0:
        _, qvals, _, _ = multipletests(
            results_df['p_value'], alpha=cfg['fdr_alpha'], method='fdr_bh'
        )
        results_df['q_value'] = qvals
        results_df['q_value_within_area'] = _within_area_condition_fdr(results_df, cfg)

    os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(os.path.join(out_dir, 'mmd_results_aggregated.csv'), index=False)

    if make_figures and len(results_df) > 0:
        print(f"[MMD aggregated] generating figures for {len(fig_payload)} "
              f"feature version(s)...")
        t_fig = time.time()
        extreme_dir = os.path.join(out_dir, 'extreme_neurons')
        os.makedirs(extreme_dir, exist_ok=True)
        cluster_col = cfg.get('cluster_col')
        for label, (res, feats, feats_raw, mice, cohorts, auc, witness, extreme_df,
                    pca_obj, fv_name) in tqdm(
                fig_payload.items(), desc="MMD aggregated figures", total=len(fig_payload)):
            fig = _fig_mmd_null_histogram(
                res, title=f"MMD null | ALL AREAS AGGREGATED | {label}"
            )
            _save(fig, os.path.join(
                out_dir, 'null_histograms', f"null_hist_AGGREGATED_{label}.pdf"
            ))

            fig = _fig_mmd_scatter_witness(
                feats, cohorts, cohort_values=_cohort_values(cfg),
                sigma2=res['sigma2'], n_components=cfg['pca_n_components'],
                title=f"MMD scatte1witness | ALL AREAS AGGREGATED | {label}"
            )
            _save(fig, os.path.join(
                out_dir, 'scatter_witness', f"scatter_witness_AGGREGATED_{label}.pdf"
            ))

            fig = _fig_mmd_method_overview(
                feats, mice, cohorts, cohort_values=_cohort_values(cfg),
                sigma2=res['sigma2'], res=res,
                title=f"MMD method overview | ALL AREAS AGGREGATED | {label}",
                neuron_order=cfg.get('method_overview_neuron_order', 'similarity'),
            )
            _save(fig, os.path.join(
                out_dir, 'method_overview', f"method_overview_AGGREGATED_{label}.pdf"
            ))

            fig = _fig_witness_distribution(
                witness, cohorts, cohort_values=_cohort_values(cfg), res=res, auc=auc,
                title=f"ALL AREAS AGGREGATED | {label}"
            )
            _save(fig, os.path.join(
                out_dir, 'witness_distribution', f"witness_dist_AGGREGATED_{label}.pdf"
            ))

            fig = _fig_neuron_mouse_point_cloud(
                feats, mice, cohorts, cohort_values=_cohort_values(cfg),
                title=f"ALL AREAS AGGREGATED | {label}"
            )
            _save(fig, os.path.join(
                out_dir, 'point_clouds', f"point_cloud_AGGREGATED_{label}.pdf"
            ))

            extreme_df.to_csv(os.path.join(
                extreme_dir, f"extreme_neurons_AGGREGATED_{label}.csv"), index=False)
            for role in ('typical_pos', 'typical_neg', 'atypical_pos', 'atypical_neg'):
                fig = _fig_extreme_neuron_traces(
                    feats, cohorts, cohort_values=_cohort_values(cfg),
                    extreme_df=extreme_df, role=role,
                    title=f"ALL AREAS AGGREGATED | {label} | {role}"
                )
                _save(fig, os.path.join(
                    extreme_dir, f"traces_AGGREGATED_{label}_{role}.pdf"
                ))

            crosstab = _crosstab_extreme_vs_cluster(extreme_df, neuron_meta, cluster_col)
            if crosstab is not None:
                crosstab.to_csv(os.path.join(
                    extreme_dir, f"cluster_crosstab_AGGREGATED_{label}.csv"))

            # PCA documentation figures (if a PCA basis was actually fit) and
            # raw-bin-space characterization figures are no longer mutually
            # exclusive -- both are generated whenever available, since
            # feats_raw is always computed regardless of test space.
            if pca_obj is not None:
                fig = _fig_pca_variance(
                    pca_obj, title=f"PCA variance | ALL AREAS AGGREGATED | {fv_name}")
                _save(fig, os.path.join(
                    out_dir, 'pca_variance', f"pca_variance_AGGREGATED_{fv_name}.pdf"))

                fig = _fig_pc_loading_timecourse(
                    pca_obj, feats, cohorts, cohort_values=_cohort_values(cfg),
                    title=f"PC loadings | ALL AREAS AGGREGATED | {fv_name}")
                _save(fig, os.path.join(
                    out_dir, 'pc_loading', f"pc_loading_AGGREGATED_{fv_name}.pdf"))

            char_dir = os.path.join(out_dir, 'characterization')
            fig, _ = _fig_avg_psth_perbin_sig(
                feats_raw, cohorts, cohort_values=_cohort_values(cfg),
                fdr_alpha=cfg['fdr_alpha'],
                title=f"ALL AREAS AGGREGATED | {label}")
            _save(fig, os.path.join(char_dir, f"avg_psth_AGGREGATED_{label}.pdf"))

            fig = _fig_witness_weighted_trace(
                feats_raw, witness, title=f"ALL AREAS AGGREGATED | {label}")
            _save(fig, os.path.join(
                char_dir, f"witness_weighted_AGGREGATED_{label}.pdf"))

        print(f"[MMD aggregated] figures done in {time.time() - t_fig:.1f}s")

    return results_df



def _run_one_area_test(neuron_meta, fv_matrix, cfg, area, fv_name, seed):
    """One (area, feature_version) MMD test -- the unit of work
    distributed across processes by run_mmd_per_area's joblib.Parallel
    call below.

    blas_threads=1 here is deliberate: with many of these running at once
    (one per core, via joblib), letting each ALSO spawn a multi-threaded
    BLAS call would oversubscribe cores (n_jobs processes x multiple BLAS
    threads each, fighting over the same 110 cores). One thread per worker,
    many workers, is the efficient split for many SMALL independent tests
    -- the opposite tradeoff from the single big aggregated test above.

    PCA reduction (see _maybe_pca_reduce), if configured, is attempted
    unconditionally -- there is no separate non-PCA combo any more, so
    each (area, feature_version) is exactly one test, not two.
    """
    feats_raw, mice, cohorts, unit_ids = _get_neuron_data(neuron_meta, fv_matrix, cfg, area=area)
    feats, pca_obj = _maybe_pca_reduce(feats_raw, cfg)
    label = fv_name
    rng = np.random.default_rng(seed)
    try:
        res = mouse_level_permutation_test(
            feats, mice, cohorts,
            cohort_values=_cohort_values(cfg),
            n_perm=cfg['mmd_n_perm'],
            exact_if_feasible=cfg['mmd_exact_if_feasible'],
            sigma2=None,
            rng=rng,
            progress_desc=f"{area}/{label}",
            blas_threads=1,
        )
    except ValueError as e:
        return (area, label, None, feats, feats_raw, mice, cohorts, None, None, None,
                pca_obj, fv_name, str(e))

    auc, witness = compute_witness_auc(feats, cohorts, _cohort_values(cfg), res['sigma2'])
    extreme_df = extract_extreme_neurons(feats, cohorts, _cohort_values(cfg),
                                         res['sigma2'], unit_ids=unit_ids)

    print(f"[MMD pe0area] {area:<20s} {label:<16s} "
          f"n={feats.shape[0]:>5d}  mice={res['n_mice_pos']}v{res['n_mice_neg']}  "
          f"mmd2={res['mmd2_obs']:.4g}  z={res['mmd2_z']:.3g}  auc={auc:.3f}  "
          f"p={res['p_value']:.4g}", flush=True)
    return (area, label, res, feats, feats_raw, mice, cohorts, auc, witness, extreme_df,
            pca_obj, fv_name, None)


def run_mmd_per_area(neuron_meta, feature_matrix, cfg=None, valid_areas=None,
                      make_figures=True):
    """
    Run the mouse-level-permutation MMD test independently per valid area,
    for the full concatenated feature vector and (if configured) each
    trial-type sub-slice. BH-FDR corrected across areas, separately within
    each feature version.

    Runs AFTER run_mmd_aggregated (see run_all) -- unlike that single big
    pooled test, this is many SMALL independent tests, so it parallelizes
    across cfg['n_jobs'] processes (one (area, feature_version[, pca])
    combo per task) instead of relying on BLAS threading within a single
    test; each worker caps its own BLAS threads to 1 to avoid
    oversubscribing cores (see _run_one_area_test).

    If cfg['pca_test_n_components'] / cfg['pca_test_variance_retained'] is
    set, EVERY (area, feature_version) combo is ALSO tested after
    PCA-reducing, as an additional '<name>_pca' row (see
    _maybe_pca_reduce) -- doubles the combo count when enabled.
    """
    cfg = {**DEFAULT_MMD_CFG, **(cfg or {})}
    out_dir = cfg['output_dir']
    n_jobs_cfg = cfg.get('n_jobs', N_CORES)

    if valid_areas is None:
        valid_areas = get_valid_areas(neuron_meta, cfg)
    feature_versions = _slice_feature_matrix(
        feature_matrix, cfg['trial_type_slices'],
        exclude_bins=cfg.get('exclude_bins'), exclude_from_full=cfg.get('exclude_from_full'),
    )
    pca_on = cfg.get('pca_test_n_components') is not None or cfg.get('pca_test_variance_retained') is not None
    if not pca_on:
        print("[MMD pe0area] WARNING: no PCA reduction configured "
              "(pca_test_n_components/pca_test_variance_retained both None) -- "
              "tests will run in full-dimensional space. PCA-reduced testing "
              "is the supported default; set pca_test_variance_retained "
              "(e.g. 0.99) unless you specifically intend full-dim testing.")

    combos = [(area, fv_name) for area in valid_areas for fv_name in feature_versions]
    n_jobs = max(1, min(n_jobs_cfg, len(combos))) if combos else 1
    print(f"[MMD pe0area] {len(valid_areas)} areas x {len(feature_versions)} "
          f"feature versions = {len(combos)} "
          f"tests to run (n_perm={cfg['mmd_n_perm']} each, n_jobs={n_jobs} "
          f"parallel workers, 1 BLAS thread/worker)")
    t_start = time.time()

    if not combos:
        results_list = []
    else:
        results_list = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
            delayed(_run_one_area_test)(
                neuron_meta, feature_versions[fv_name], cfg, area, fv_name,
                seed=cfg['mmd_random_seed'] * 1_000_003 + i,
            )
            for i, (area, fv_name) in enumerate(combos)
        )

    print(f"[MMD pe0area] all {len(combos)} tests done in "
          f"{time.time() - t_start:.1f}s")

    records = []
    fig_payload = {}  # (area, label) -> (res, feats, feats_raw, mice, cohorts, auc, witness, extreme_df, pca_obj, fv_name)
    for (area, label, res, feats, feats_raw, mice, cohorts, auc, witness, extreme_df,
         pca_obj, fv_name, err) in results_list:
        if res is None:
            print(f"[MMD pe0area] skipping area={area}, "
                  f"feature_version={label}: {err}")
            continue
        records.append({
            'area': area,
            'feature_version': label,
            'space': 'pca' if pca_obj is not None else 'full',
            'n_neurons': feats.shape[0],
            'n_mice_pos': res['n_mice_pos'],
            'n_mice_neg': res['n_mice_neg'],
            'mmd2_obs': res['mmd2_obs'],
            'mmd2_z': res['mmd2_z'],
            'witness_auc': auc,
            'sigma2': res['sigma2'],
            'p_value': res['p_value'],
            'n_perm_used': res['n_perm_used'],
            'used_exact': res['used_exact'],
        })
        fig_payload[(area, label)] = (res, feats, feats_raw, mice, cohorts, auc, witness,
                                       extreme_df, pca_obj, fv_name)

    results_df = pd.DataFrame.from_records(records)
    if results_df.empty:
        print(
            "[MMD pe0area] No results to report: 0 valid areas or every "
            "area was skipped (e.g. too few mice per cohort after "
            "mouse_level_permutation_test's own checks). Nothing written."
        )
        return pd.DataFrame(columns=[
            'area', 'feature_version', 'space', 'n_neurons', 'n_mice_pos', 'n_mice_neg',
            'mmd2_obs', 'mmd2_z', 'witness_auc', 'sigma2', 'p_value',
            'n_perm_used', 'used_exact', 'q_value', 'q_value_within_area',
        ])

    results_df['q_value'] = np.nan
    for fv_name_ in results_df['feature_version'].unique():
        m = results_df['feature_version'] == fv_name_
        if m.sum() > 0:
            _, qvals, _, _ = multipletests(
                results_df.loc[m, 'p_value'], alpha=cfg['fdr_alpha'], method='fdr_bh'
            )
            results_df.loc[m, 'q_value'] = qvals

    results_df['q_value_within_area'] = _within_area_condition_fdr(results_df, cfg)

    os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(os.path.join(out_dir, 'mmd_results_per_area.csv'), index=False)

    if make_figures and len(results_df) > 0:
        print(f"[MMD pe0area] generating figures for {len(fig_payload)} "
              f"(area, feature_version) combinations...")
        t_fig = time.time()
        extreme_dir = os.path.join(out_dir, 'extreme_neurons')
        os.makedirs(extreme_dir, exist_ok=True)
        cluster_col = cfg.get('cluster_col')
        perbin_qvals_by_area = {}  # area -> per-bin FDR q-values, 'full' condition only (see #5 below)
        for (area, label), (res, feats, feats_raw, mice, cohorts, auc, witness, extreme_df,
                             pca_obj, fv_name) in tqdm(
                fig_payload.items(), desc="MMD per-area figures", total=len(fig_payload)):
            fig = _fig_mmd_null_histogram(
                res, title=f"MMD null | area={area} | {label}"
            )
            _save(fig, os.path.join(
                out_dir, 'null_histograms', f"null_hist_{area}_{label}.pdf"
            ))

            fig = _fig_mmd_scatter_witness(
                feats, cohorts, cohort_values=_cohort_values(cfg),
                sigma2=res['sigma2'], n_components=cfg['pca_n_components'],
                title=f"MMD scatte1witness | area={area} | {label}"
            )
            _save(fig, os.path.join(
                out_dir, 'scatter_witness', f"scatter_witness_{area}_{label}.pdf"
            ))

            fig = _fig_mmd_method_overview(
                feats, mice, cohorts, cohort_values=_cohort_values(cfg),
                sigma2=res['sigma2'], res=res,
                title=f"MMD method overview | area={area} | {label}",
                neuron_order=cfg.get('method_overview_neuron_order', 'similarity'),
            )
            _save(fig, os.path.join(
                out_dir, 'method_overview', f"method_overview_{area}_{label}.pdf"
            ))

            fig = _fig_witness_distribution(
                witness, cohorts, cohort_values=_cohort_values(cfg), res=res, auc=auc,
                title=f"area={area} | {label}"
            )
            _save(fig, os.path.join(
                out_dir, 'witness_distribution', f"witness_dist_{area}_{label}.pdf"
            ))

            fig = _fig_neuron_mouse_point_cloud(
                feats, mice, cohorts, cohort_values=_cohort_values(cfg),
                title=f"area={area} | {label}"
            )
            _save(fig, os.path.join(
                out_dir, 'point_clouds', f"point_cloud_{area}_{label}.pdf"
            ))

            extreme_df.to_csv(os.path.join(
                extreme_dir, f"extreme_neurons_{area}_{label}.csv"), index=False)
            for role in ('typical_pos', 'typical_neg', 'atypical_pos', 'atypical_neg'):
                fig = _fig_extreme_neuron_traces(
                    feats, cohorts, cohort_values=_cohort_values(cfg),
                    extreme_df=extreme_df, role=role,
                    title=f"area={area} | {label} | {role}"
                )
                _save(fig, os.path.join(
                    extreme_dir, f"traces_{area}_{label}_{role}.pdf"
                ))

            crosstab = _crosstab_extreme_vs_cluster(extreme_df, neuron_meta, cluster_col)
            if crosstab is not None:
                crosstab.to_csv(os.path.join(
                    extreme_dir, f"cluster_crosstab_{area}_{label}.csv"))

            # PCA documentation figures (if a PCA basis was actually fit) and
            # raw-bin-space characterization figures are no longer mutually
            # exclusive -- both are generated whenever available, since
            # feats_raw is always computed regardless of test space.
            if pca_obj is not None:
                fig = _fig_pca_variance(
                    pca_obj, title=f"PCA variance | area={area} | {fv_name}")
                _save(fig, os.path.join(
                    out_dir, 'pca_variance', f"pca_variance_{area}_{fv_name}.pdf"))

                fig = _fig_pc_loading_timecourse(
                    pca_obj, feats, cohorts, cohort_values=_cohort_values(cfg),
                    title=f"PC loadings | area={area} | {fv_name}")
                _save(fig, os.path.join(
                    out_dir, 'pc_loading', f"pc_loading_{area}_{fv_name}.pdf"))

            char_dir = os.path.join(out_dir, 'characterization')
            fig, perbin_qvals = _fig_avg_psth_perbin_sig(
                feats_raw, cohorts, cohort_values=_cohort_values(cfg),
                fdr_alpha=cfg['fdr_alpha'], title=f"area={area} | {label}")
            _save(fig, os.path.join(char_dir, f"avg_psth_{area}_{label}.pdf"))
            if fv_name == 'full':
                perbin_qvals_by_area[area] = perbin_qvals

            fig = _fig_witness_weighted_trace(
                feats_raw, witness, title=f"area={area} | {label}")
            _save(fig, os.path.join(char_dir, f"witness_weighted_{area}_{label}.pdf"))

        fig = _fig_mmd_summary(results_df, cfg)
        _save(fig, os.path.join(out_dir, 'mmd_summary_per_area.pdf'))

        # #1: areas x conditions summary heatmap.
        fig = _fig_area_condition_heatmap(results_df, cfg)
        if fig is not None:
            _save(fig, os.path.join(out_dir, 'mmd_summary_heatmap.pdf'))

        # #5: whole-brain per-bin significance, built from the 'full'
        # condition's per-bin q-values collected above (one row per area).
        if perbin_qvals_by_area:
            fig = _fig_whole_brain_perbin_significance(
                perbin_qvals_by_area, fdr_alpha=cfg['fdr_alpha'],
                title="Whole-brain per-bin significance ('full' condition)")
            _save(fig, os.path.join(out_dir, 'whole_brain_perbin_significance.pdf'))

        print(f"[MMD pe0area] figures done in {time.time() - t_fig:.1f}s")

    return results_df



def _load_npz_data(path):
    """Load feature_matrix.npz in full: the feature matrix X plus every
    per-neuron metadata array needed to run this module WITHOUT unit_table
    or a rastermap-cluster join.

    Returns (X, neuron_meta, X_odd, X_even) where neuron_meta is a dict of
    1D arrays, all positionally aligned 1:1 with X's rows:
        unit_ids, mouse (from mouse_arr), session (from session_arr),
        reward_group (from reward_arr), area (from area_arr),
        area_group (from area_group_arr), waveform (from waveform_arr),
        layer (from layer_arr), odd_even_r, plus n_bins_list/n_conds for
        building trial-type slices.
    Arrays not present in the npz are simply omitted from neuron_meta
    (e.g. an older feature_matrix.npz missing waveform_arr/layer_arr still
    works fine as long as the fields you actually use in cfg are present).

    X_odd/X_even are the same feature matrix computed on odd-trial-only
    and even-trial-only PSTHs respectively (same neuron rows/ordering as
    X) -- used ONLY by run_mmd_cv for the odd/even cross-validation
    scheme (fit sigma2/PCA on one split, test on the other, and vice
    versa). Both are None if the npz doesn't contain 'X_odd'/'X_even'
    (run_mmd_cv is simply unavailable in that case; the main run_all
    pipeline doesn't need them at all).
    """
    path = pathlib.Path(path)
    print(f"[MMD] loading {path} ...")
    t0 = time.time()
    field_map = {
        'unit_ids': 'unit_ids',
        'mouse': 'mouse_arr',
        'session': 'session_arr',
        'reward_group': 'reward_arr',
        'area': 'area_arr',
        'area_group': 'area_group_arr',
        'waveform': 'waveform_arr',
        'layer': 'layer_arr',
        'odd_even_r': 'odd_even_r',
    }
    with np.load(path, allow_pickle=True) as npz:
        if 'X' not in npz.files:
            raise KeyError(
                f"{path} has no 'X' array (found: {npz.files}) -- expected the "
                f"feature_matrix.npz produced by build_feature_matrix.py.")
        X = npz['X']
        X_odd = npz['X_odd'] if 'X_odd' in npz.files else None
        X_even = npz['X_even'] if 'X_even' in npz.files else None
        neuron_meta = {}
        found, skipped = [], []
        for meta_key, npz_key in field_map.items():
            if npz_key in npz.files:
                arr = npz[npz_key]
                neuron_meta[meta_key] = arr.astype(str) if arr.dtype.kind in ('U', 'S', 'O') else arr
                found.append(meta_key)
            else:
                skipped.append(meta_key)
        for extra_key in ('n_bins_list', 'n_conds'):
            if extra_key in npz.files:
                neuron_meta[extra_key] = npz[extra_key]

    n_neurons = X.shape[0]
    for key, arr in neuron_meta.items():
        if key in ('n_bins_list', 'n_conds'):
            continue
        if len(arr) != n_neurons:
            raise ValueError(
                f"{path}: X has {n_neurons} rows (neurons) but neuron_meta['{key}'] "
                f"has {len(arr)} entries -- these must be positionally aligned 1:1.")
    for split_name, split_arr in (('X_odd', X_odd), ('X_even', X_even)):
        if split_arr is not None and split_arr.shape != X.shape:
            raise ValueError(
                f"{path}: {split_name} has shape {split_arr.shape} but X has shape "
                f"{X.shape} -- these must be the same shape, positionally aligned.")

    print(f"[MMD] loaded in {time.time() - t0:.1f}s -- "
          f"X shape={X.shape} ({n_neurons} neurons, {X.shape[1]} feature bins)")
    print(f"[MMD] fields found: {found}" + (f"  (missing: {skipped})" if skipped else ""))
    print(f"[MMD] odd/even CV matrices: "
          f"{'found' if X_odd is not None and X_even is not None else 'NOT found (run_mmd_cv unavailable)'}")

    return X, neuron_meta, X_odd, X_even


def _subsample_npz_data(feature_matrix, neuron_meta, n, random_subset=False, seed=0):
    """
    Restrict feature_matrix/neuron_meta to the first `n` neurons (default,
    matching the literal "first small subset"), or a random `n`-neuron
    subset if random_subset=True. Used by run_all's test_n_neurons=... for
    a fast smoke test of the whole pipeline (loading -> valid-area
    filtering -> both MMD tests -> figures) before committing to a full,
    many-hour run on the real neuron count.

    Only per-neuron fields are sliced -- n_bins_list/n_conds (not
    per-neuron) pass through unchanged.
    """
    N = feature_matrix.shape[0]
    n = min(n, N)
    if random_subset:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(N, size=n, replace=False))
    else:
        idx = np.arange(n)

    sub_feature_matrix = feature_matrix[idx]
    sub_neuron_meta = {}
    for key, arr in neuron_meta.items():
        if key in ('n_bins_list', 'n_conds'):
            sub_neuron_meta[key] = arr
        else:
            sub_neuron_meta[key] = np.asarray(arr)[idx]
    return sub_feature_matrix, sub_neuron_meta


# --------------------------------------------------------------------------
# Odd/even cross-validation: fit on one half-split, test on the other
# --------------------------------------------------------------------------

def _fit_sigma2_and_pca(fit_feats, cfg):
    """
    Estimate the kernel bandwidth (and, if cfg has PCA reduction
    configured -- the module's default -- fit a PCA basis) from a FIT
    split only (e.g. X_odd for one direction of the CV scheme), entirely
    label-blind -- exactly the same discipline as the main pipeline's
    sigma2/PCA fitting, just using the fit-split's feature VALUES instead
    of the full dataset's. Neuron/mouse/cohort identity is identical
    between X_odd and X_even (same neurons, just different trial-subset
    PSTHs), so nothing else needs to change for this to be a valid
    application of the existing sigma2/PCA-then-test machinery.

    PCA reduction is attempted unconditionally here (via
    _maybe_pca_reduce, which itself no-ops -- returning fit_feats
    unchanged and pca=None -- if cfg['pca_test_n_components'] /
    cfg['pca_test_variance_retained'] aren't set); there is no separate
    non-PCA test variant to choose between any more.

    Returns
    -------
    sigma2 : float, estimated on (possibly PCA-projected) fit_feats.
    pca : fitted PCA object, or None if PCA reduction isn't configured.
    """
    fit_proj, pca = _maybe_pca_reduce(fit_feats, cfg)
    sq_d = pdist(fit_proj, metric='sqeuclidean')
    sigma2 = np.median(sq_d)
    if not np.isfinite(sigma2) or sigma2 <= 0:
        mean_d = np.mean(sq_d)
        sigma2 = mean_d if (np.isfinite(mean_d) and mean_d > 0) else 1.0
    return float(sigma2), pca


def _run_one_cv_direction(fit_feats_raw, test_feats_raw, mice, cohorts, cfg,
                           seed, progress_desc):
    """One direction of the odd/even CV scheme: fit sigma2 (+PCA, if
    configured) on `fit_feats_raw`, then run the full mouse-level
    permutation test on `test_feats_raw` (PCA-projected into the FIT
    basis, if a PCA basis was fit) with that frozen sigma2. Returns the
    same dict mouse_level_permutation_test returns, plus the fitted pca
    (or None)."""
    sigma2, pca = _fit_sigma2_and_pca(fit_feats_raw, cfg)
    test_feats = pca.transform(test_feats_raw) if pca is not None else test_feats_raw
    rng = np.random.default_rng(seed)
    res = mouse_level_permutation_test(
        test_feats, mice, cohorts,
        cohort_values=_cohort_values(cfg),
        n_perm=cfg['mmd_n_perm'],
        exact_if_feasible=cfg['mmd_exact_if_feasible'],
        sigma2=sigma2,   # FROZEN from the fit split, not re-estimated here
        rng=rng,
        progress_desc=progress_desc,
        blas_threads=1,
    )
    return res, pca


def _fig_cv_directions(res_A, res_B, title=''):
    """
    Two-panel null-histogram figure for ONE (area, feature_version, space)
    CV combo: direction A (fit odd, test even) and direction B (fit even,
    test odd) side by side, each showing its own permutation null with the
    observed MMD^2 marked -- the CV analogue of _fig_mmd_null_histogram,
    saved immediately per combo (see run_mmd_cv) rather than only as part
    of the final cross-combo consistency scatter.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, res, direction_title in zip(
            axes, (res_A, res_B), ('fit odd, test even', 'fit even, test odd')):
        ax.hist(res['null_mmd2'], bins=40, color='0.75', edgecolor='0.4')
        ax.axvline(res['mmd2_obs'], color='crimson', lw=2, label='Observed MMD$^2$')
        ax.set_xlabel(r'MMD$^2_u$')
        ax.set_ylabel('Count')
        ax.set_title(f"{direction_title}\np={res['p_value']:.4g}", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def _run_cv_combos(combos, neuron_meta, fv_odd, fv_even, cfg, valid_areas, out_dir,
                    make_figures, seed_base, i_start=0):
    """
    Run one batch of CV combos (list of (area, is_agg, fv_name) tuples --
    PCA reduction, if configured, is always attempted internally; there is
    no separate non-PCA combo any more), saving each combo's
    cv_null_histograms/ figure immediately (if make_figures), and
    returning its per-direction record dicts plus a null_store dict (see
    below, used by the report's null-histogram-grid figure so it doesn't
    have to recompute anything). Shared by run_mmd_cv's aggregated-first
    pass and per-area pass so the per-combo loop body (and its seed
    indexing, via i_start) is identical for both -- only which combos are
    passed in differs.

    Returns
    -------
    records : list of per-direction record dicts.
    null_store : dict {(area, label): {'null_A', 'obs_A', 'null_B', 'obs_B',
        'is_agg'}} -- the full null distributions and observed statistics
        for both directions of every combo, kept so a later report step
        can build a null-histogram grid without re-running any tests.
    """
    records = []
    null_store = {}
    for j, (area, is_agg, fv_name) in enumerate(combos):
        i = i_start + j
        sel_kwargs = dict(areas=valid_areas) if is_agg else dict(area=area)
        feats_odd_raw, mice, cohorts, _ = _get_neuron_data(neuron_meta, fv_odd[fv_name], cfg, **sel_kwargs)
        feats_even_raw, mice2, cohorts2, _ = _get_neuron_data(neuron_meta, fv_even[fv_name], cfg, **sel_kwargs)
        if not (np.array_equal(mice, mice2) and np.array_equal(cohorts, cohorts2)):
            print(f"[MMD cv] skipping area={area}, {fv_name}: neuron ordering "
                  f"differs between X_odd/X_even selections (unexpected).")
            continue
        label = fv_name
        try:
            res_A, _ = _run_one_cv_direction(
                feats_odd_raw, feats_even_raw, mice, cohorts, cfg,
                seed=seed_base + 2 * i, progress_desc=f"{area}/{label}/oddto_even")
            res_B, _ = _run_one_cv_direction(
                feats_even_raw, feats_odd_raw, mice, cohorts, cfg,
                seed=seed_base + 2 * i + 1, progress_desc=f"{area}/{label}/eventoodd")
        except ValueError as e:
            print(f"[MMD cv] skipping area={area}, {label}: {e}")
            continue

        if make_figures:
            fig = _fig_cv_directions(
                res_A, res_B, title=f"MMD CV | area={area} | {label}")
            area_tag = 'AGGREGATED' if is_agg else area
            _save(fig, os.path.join(
                out_dir, 'cv_null_histograms', f"cv_null_hist_{area_tag}_{label}.pdf"))

        null_store[(area, label)] = {
            'null_A': res_A['null_mmd2'], 'obs_A': res_A['mmd2_obs'],
            'null_B': res_B['null_mmd2'], 'obs_B': res_B['mmd2_obs'],
            'is_agg': is_agg,
        }

        pca_used = cfg.get('pca_test_n_components') is not None or cfg.get('pca_test_variance_retained') is not None
        for direction, res in (('fit_odd_test_even', res_A), ('fit_even_test_odd', res_B)):
            records.append({
                'area': area, 'feature_version': label,
                'space': 'pca' if pca_used else 'full', 'direction': direction,
                'n_neurons': len(mice), 'n_mice_pos': res['n_mice_pos'],
                'n_mice_neg': res['n_mice_neg'], 'mmd2_obs': res['mmd2_obs'],
                'mmd2_z': res['mmd2_z'], 'sigma2': res['sigma2'],
                'p_value': res['p_value'], 'n_perm_used': res['n_perm_used'],
            })
    return records, null_store


def _cv_consistency_fig_and_summary(records, cfg, fdr_correct, title_prefix=''):
    """
    Build the odd/even CV consistency scatter + wide-form summary table
    from a list of per-direction CV record dicts (same shape as
    _run_cv_combos' output).

    fdr_correct : bool
        True (used for the FINAL combined summary, across every area):
        BH-FDR the p-values (grouped by feature_version x direction) into
        a 'q_value' column, and significant_both = (q < alpha) in BOTH
        directions.
        False (used for the EARLY aggregated-only figure, before any
        per-area result exists to correct alongside): no FDR column;
        significant_both = RAW p < alpha in BOTH directions instead. The
        figure title says "raw p" explicitly so this is never confused
        with the properly-corrected final summary's significant_both.

    Returns
    -------
    df : DataFrame with (and without, if fdr_correct=False) a 'q_value' column.
    summary_df : wide-form, one row per (area, feature_version, space).
    fig : matplotlib Figure, or None if there was nothing to plot.
    """
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df, df.copy(), None

    if fdr_correct:
        df['q_value'] = np.nan
        for (fv, direction), m in df.groupby(['feature_version', 'direction']).groups.items():
            _, qvals, _, _ = multipletests(df.loc[m, 'p_value'], alpha=cfg['fdr_alpha'], method='fdr_bh')
            df.loc[m, 'q_value'] = qvals
        sig_value_col = 'q_value'
    else:
        sig_value_col = 'p_value'

    wide = df.pivot_table(
        index=['area', 'feature_version', 'space'],
        columns='direction', values=['p_value', 'mmd2_obs'] + (['q_value'] if fdr_correct else []),
    )
    wide.columns = [f"{stat}__{direction}" for stat, direction in wide.columns]
    wide = wide.reset_index()
    a = wide.get(f'{sig_value_col}__fit_odd_test_even')
    b = wide.get(f'{sig_value_col}__fit_even_test_odd')
    if a is not None and b is not None:
        wide['significant_both'] = (a < cfg['fdr_alpha']) & (b < cfg['fdr_alpha'])
    summary_df = wide

    n_both = int(summary_df['significant_both'].sum()) if 'significant_both' in summary_df else 0
    sig_desc = 'BH-FDR' if fdr_correct else 'RAW p, uncorrected'
    print(f"[MMD cv] {title_prefix}{n_both}/{len(summary_df)} (area, feature_version, space) "
          f"combinations significant in BOTH CV directions ({sig_desc} alpha={cfg['fdr_alpha']})")

    fig = None
    if 'mmd2_obs__fit_odd_test_even' in wide.columns and 'mmd2_obs__fit_even_test_odd' in wide.columns:
        fig, ax = plt.subplots(figsize=(5, 5))
        colors = np.where(summary_df.get('significant_both', False), 'crimson', '0.6')
        ax.scatter(wide['mmd2_obs__fit_odd_test_even'], wide['mmd2_obs__fit_even_test_odd'],
                   c=colors, s=20, alpha=0.8)
        lim = max(wide['mmd2_obs__fit_odd_test_even'].abs().max(),
                  wide['mmd2_obs__fit_even_test_odd'].abs().max(), 1e-6)
        ax.plot([-lim * 0.1, lim], [-lim * 0.1, lim], color='0.7', lw=0.8, ls='--')
        ax.set_xlabel('MMD$^2$ (fit odd, test even)')
        ax.set_ylabel('MMD$^2$ (fit even, test odd)')
        ax.set_title(f'{title_prefix}Odd/even CV consistency ({sig_desc})\n'
                     f'{n_both}/{len(summary_df)} significant both directions (red)', fontsize=9)
        fig.tight_layout()

    return df, summary_df, fig


def run_mmd_cv(neuron_meta, X_odd, X_even, cfg=None, valid_areas=None, make_figures=True):
    """
    Odd/even cross-validated MMD: for every (area incl. 'ALL_AREAS_AGGREGATED',
    feature_version, space) combination, run BOTH directions --
        A: fit sigma2/PCA on ODD, test on EVEN
        B: fit sigma2/PCA on EVEN, test on ODD
    -- IN ADDITION TO the main (non-CV, full-data) test in run_all. Neuron/
    mouse/cohort identity is identical between the two splits; only the
    feature VALUES (and therefore sigma2/PCA basis) differ, so each
    direction is a legitimate held-out test of the other direction's
    fitted scale/subspace.

    Requires X_odd/X_even (see _load_npz_data) -- prints a message and
    returns two empty DataFrames if either is None.

    Runs in TWO PASSES -- aggregated combos, then per-area combos -- so
    everything aggregated-scoped is fully on disk before any per-area
    combo even starts, matching run_mmd_aggregated running before
    run_mmd_per_area elsewhere in this module:
      1. Aggregated combos: each combo's cv_null_histograms/ figure saves
         immediately (see make_figures), then -- once ALL aggregated
         combos are done -- a DEDICATED aggregated-only consistency
         scatter saves to mmd_cv_consistency_AGGREGATED.pdf (raw p,
         uncorrected: too few aggregated combos for a meaningful FDR
         correction on their own, and the properly-corrected version
         needs the per-area results anyway -- see below).
      2. Per-area combos: same per-combo figure saving.
      3. Finally, the FULL combined result (aggregated + every area) is
         BH-FDR corrected together and written to mmd_results_cv.csv /
         mmd_results_cv_summary.csv / mmd_cv_consistency.pdf, exactly as
         before -- this is still the authoritative, properly-corrected
         significant_both.

    Returns
    -------
    cv_df : one row per (area, feature_version, space, direction) -- the
        full detail, analogous to mmd_results_per_area.csv.
    summary_df : one row per (area, feature_version, space) with both
        directions' p/q-values side by side and a `significant_both`
        flag (q < fdr_alpha in BOTH directions) -- the "does this effect
        replicate" table.
    null_store : dict {(area, feature_version): {'null_A', 'obs_A',
        'null_B', 'obs_B', 'is_agg'}} -- every combo's full null
        distributions and observed statistics, for the report's
        null-histogram-grid figure (see _fig_cv_null_grid) to reuse
        without recomputation.
    """
    cfg = {**DEFAULT_MMD_CFG, **(cfg or {})}
    out_dir = cfg['output_dir']

    if X_odd is None or X_even is None:
        print("[MMD cv] X_odd/X_even not available in this feature_matrix.npz "
              "-- skipping odd/even cross-validation entirely.")
        empty = pd.DataFrame(columns=[
            'area', 'feature_version', 'space', 'direction', 'n_neurons',
            'n_mice_pos', 'n_mice_neg', 'mmd2_obs', 'mmd2_z', 'sigma2',
            'p_value', 'n_perm_used', 'q_value',
        ])
        return empty, empty.copy(), {}

    if valid_areas is None:
        valid_areas = get_valid_areas(neuron_meta, cfg)

    fv_odd = _slice_feature_matrix(X_odd, cfg['trial_type_slices'],
                                    exclude_bins=cfg.get('exclude_bins'),
                                    exclude_from_full=cfg.get('exclude_from_full'))
    fv_even = _slice_feature_matrix(X_even, cfg['trial_type_slices'],
                                     exclude_bins=cfg.get('exclude_bins'),
                                     exclude_from_full=cfg.get('exclude_from_full'))
    # PCA reduction (if configured -- the module's default) is attempted
    # unconditionally inside _run_one_cv_direction; there is no separate
    # non-PCA combo any more, so each (area, feature_version) is exactly
    # ONE combo, not two.
    agg_combos = ([('ALL_AREAS_AGGREGATED', True, fv_name) for fv_name in fv_odd]
                  if valid_areas else [])
    area_combos = [(area, False, fv_name) for area in valid_areas for fv_name in fv_odd]
    print(f"[MMD cv] {len(valid_areas)} areas (+aggregated) x {len(fv_odd)} "
          f"feature versions x 2 directions "
          f"= {(len(agg_combos) + len(area_combos)) * 2} CV tests")
    t_start = time.time()
    seed_base = cfg['mmd_random_seed'] * 7_000_003

    print(f"[MMD cv] === pass 1/2: aggregated ({len(agg_combos)} combos) ===")
    agg_records, agg_null_store = _run_cv_combos(
        tqdm(agg_combos, desc="MMD CV tests (aggregated)"), neuron_meta, fv_odd, fv_even,
        cfg, valid_areas, out_dir, make_figures, seed_base, i_start=0,
    )
    if make_figures and agg_records:
        _, _, agg_fig = _cv_consistency_fig_and_summary(
            agg_records, cfg, fdr_correct=False, title_prefix='AGGREGATED | ')
        if agg_fig is not None:
            _save(agg_fig, os.path.join(out_dir, 'mmd_cv_consistency_AGGREGATED.pdf'))

    print(f"[MMD cv] === pass 2/2: per-area ({len(area_combos)} combos) ===")
    area_records, area_null_store = _run_cv_combos(
        tqdm(area_combos, desc="MMD CV tests (per-area)"), neuron_meta, fv_odd, fv_even,
        cfg, valid_areas, out_dir, make_figures, seed_base, i_start=len(agg_combos),
    )

    print(f"[MMD cv] all CV tests done in {time.time() - t_start:.1f}s")
    records = agg_records + area_records
    null_store = {**agg_null_store, **area_null_store}
    if not records:
        print("[MMD cv] No CV results to report.")
        empty = pd.DataFrame()
        return empty, empty.copy(), null_store

    cv_df, summary_df, fig = _cv_consistency_fig_and_summary(records, cfg, fdr_correct=True)

    os.makedirs(out_dir, exist_ok=True)
    cv_df.to_csv(os.path.join(out_dir, 'mmd_results_cv.csv'), index=False)
    summary_df.to_csv(os.path.join(out_dir, 'mmd_results_cv_summary.csv'), index=False)
    if make_figures and fig is not None:
        _save(fig, os.path.join(out_dir, 'mmd_cv_consistency.pdf'))

    if make_figures and null_store:
        grid_fig = _fig_cv_null_grid(
            null_store, summary_df, cfg,
            title="Odd/even CV null distributions, every area x condition")
        if grid_fig is not None:
            _save(grid_fig, os.path.join(out_dir, 'cv_null_grid.pdf'))

    return cv_df, summary_df, null_store


def _fig_evidence_concordance(concordance_df, method_cols, title=''):
    """
    #6: one row per area, one column per method (e.g. 'main_test',
    'cv_both_directions', and optionally 'permanova' if the caller
    supplied it) -- green = significant, gray = not -- sorted so areas
    where every available method agrees sit at the top. This is the
    "does independent evidence converge on this area" summary: an area
    green across every column is the strongest claim; an area green in
    only one column is worth scrutiny before trusting it.
    """
    df = concordance_df.sort_values('n_methods_significant', ascending=False)
    mat = df[method_cols].to_numpy(dtype=float)  # True/False/NaN -> 1/0/nan
    fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(method_cols) + 1),
                                     max(4, 0.3 * len(df) + 1)))
    cmap = plt.get_cmap('RdYlGn')
    cmap.set_bad(color='0.9')
    display = np.ma.masked_invalid(mat)
    ax.imshow(display, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(method_cols)))
    ax.set_xticklabels(method_cols, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['area'], fontsize=7)
    ax.set_title(f"{title}\ngreen=significant, gray=not/unavailable, "
                 f"sorted by agreement count", fontsize=10)
    fig.tight_layout()
    return fig


def _build_evidence_concordance(per_area_df, cv_summary_df, cfg, condition='full',
                                 permanova_significant=None):
    """
    Build the per-area evidence-concordance table (see #6):
        - 'main_test'          : q_value < alpha for `condition` in per_area_df
        - 'cv_both_directions' : significant_both for `condition` in cv_summary_df
                                  (NaN if CV wasn't run)
        - 'permanova'          : OPTIONAL -- pass a {area: bool} dict/Series if you
                                  have an existing PERMANOVA-on-clusters result to
                                  cross-reference; omitted entirely if not given,
                                  since this module has no access to that result on
                                  its own.
    Adds 'n_methods_significant' (count of True across whichever columns are
    present, ignoring NaN) for sorting.

    Returns (concordance_df, method_cols) -- method_cols is the list of
    columns actually present, for _fig_evidence_concordance to use.
    """
    main_sub = per_area_df[per_area_df.get('feature_version') == condition] \
        if len(per_area_df) else per_area_df
    areas = sorted(main_sub['area'].unique()) if len(main_sub) else []

    rows = []
    for area in areas:
        main_row = main_sub[main_sub['area'] == area]
        main_sig = bool((main_row['q_value'] < cfg['fdr_alpha']).iloc[0]) if len(main_row) else np.nan

        cv_sig = np.nan
        if len(cv_summary_df) and 'significant_both' in cv_summary_df.columns:
            cv_row = cv_summary_df[
                (cv_summary_df['area'] == area) &
                (cv_summary_df['feature_version'] == condition)
            ]
            if len(cv_row):
                cv_sig = bool(cv_row['significant_both'].iloc[0])

        row = {'area': area, 'main_test': main_sig, 'cv_both_directions': cv_sig}
        if permanova_significant is not None:
            row['permanova'] = bool(permanova_significant.get(area, np.nan)) \
                if area in permanova_significant else np.nan
        rows.append(row)

    concordance_df = pd.DataFrame(rows)
    method_cols = [c for c in ('main_test', 'cv_both_directions', 'permanova')
                   if c in concordance_df.columns]
    if method_cols:
        concordance_df['n_methods_significant'] = concordance_df[method_cols].sum(axis=1, skipna=True)
    return concordance_df, method_cols


# --------------------------------------------------------------------------
# Report: neuron/mouse point clouds, metric schematics, effects table,
# and a CV null-histogram grid -- assembled into one multi-page PDF by
# generate_peth_report() below.
# --------------------------------------------------------------------------

def _fig_neuron_mouse_point_cloud(feats, mice, cohorts, cohort_values, title=''):
    """
    Point-cloud visualization of the mouse-level geometry the permutation
    test actually operates on: each neuron as a small, faint dot (colored
    by cohort), each MOUSE's own mean plotted as a larger black-outlined
    marker, connected by a thin line to its group's centroid (drawn as a
    star), with the between-group centroid distance drawn as a dashed
    segment. Uses whichever first two columns `feats` has (PC1/PC2 when
    PCA-reduced, the module's default) -- this is real data, not a
    schematic (see _fig_metric_schematic for the conceptual version).
    """
    pos_label, neg_label = cohort_values
    cohorts_str = _as_str_series(pd.Series(cohorts)).values
    pos_mask = cohorts_str == pos_label
    neg_mask = cohorts_str == neg_label

    x = feats[:, 0]
    y = feats[:, 1] if feats.shape[1] > 1 else np.zeros_like(x)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(x[pos_mask], y[pos_mask], s=10, alpha=0.25, color='tab:blue',
               label=f'{pos_label} neurons', zorder=1)
    ax.scatter(x[neg_mask], y[neg_mask], s=10, alpha=0.25, color='tab:orange',
               label=f'{neg_label} neurons', zorder=1)

    mouse_means = {}
    for m in np.unique(mice):
        mask = mice == m
        mouse_means[m] = (x[mask].mean(), y[mask].mean(), cohorts_str[mask][0])

    pos_pts = np.array([[mx, my] for mx, my, c in mouse_means.values() if c == pos_label])
    neg_pts = np.array([[mx, my] for mx, my, c in mouse_means.values() if c == neg_label])
    pos_centroid = pos_pts.mean(axis=0) if len(pos_pts) else None
    neg_centroid = neg_pts.mean(axis=0) if len(neg_pts) else None

    for m, (mx, my, c) in mouse_means.items():
        color = 'tab:blue' if c == pos_label else 'tab:orange'
        centroid = pos_centroid if c == pos_label else neg_centroid
        if centroid is not None:
            ax.plot([mx, centroid[0]], [my, centroid[1]], color=color, lw=0.7,
                     alpha=0.5, zorder=3)
        ax.scatter(mx, my, s=90, color=color, edgecolors='k', linewidths=1.2, zorder=5)

    if pos_centroid is not None:
        ax.scatter(*pos_centroid, marker='*', s=400, color='tab:blue',
                   edgecolors='k', linewidths=1.3, zorder=6, label=f'{pos_label} centroid')
    if neg_centroid is not None:
        ax.scatter(*neg_centroid, marker='*', s=400, color='tab:orange',
                   edgecolors='k', linewidths=1.3, zorder=6, label=f'{neg_label} centroid')
    if pos_centroid is not None and neg_centroid is not None:
        ax.plot([pos_centroid[0], neg_centroid[0]], [pos_centroid[1], neg_centroid[1]],
                'k--', lw=1.5, zorder=6, label='between-group distance')

    dim_label = 'PC' if feats.shape[1] > 1 and hasattr(feats, 'shape') else 'dim'
    ax.set_xlabel(f'{dim_label}1')
    ax.set_ylabel(f'{dim_label}2')
    ax.set_title(f"{title}\nsmall dots: neurons -- outlined: mouse means -- "
                 f"stars: group centroids", fontsize=9)
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    return fig


def _fig_metric_schematic(sigma2_example=1.0):
    """
    SCHEMATIC (simulated illustration, NOT observed data) explaining the
    two core quantities this module computes: the RBF kernel k(u,v) and
    MMD^2 as a squared distance between group kernel-mean embeddings. For
    actual-data illustrations of the same quantities on real neurons, see
    _fig_mmd_method_overview (panels B-D) and _fig_neuron_mouse_point_cloud
    instead -- this figure is purely conceptual.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    d = np.linspace(0, 4, 200)
    k = np.exp(-d ** 2 / (2 * sigma2_example))
    axes[0].plot(d, k, color='tab:purple', lw=2)
    axes[0].axhline(1, color='0.8', lw=0.7)
    axes[0].set_xlabel(r'$\|u - v\|$')
    axes[0].set_ylabel(r'$k(u,v) = \exp(-\|u-v\|^2 / 2\sigma^2)$')
    axes[0].set_title('RBF kernel: similarity decays\nwith distance', fontsize=9)

    rng = np.random.default_rng(0)
    cloud_pos = rng.normal(loc=[1.5, 1.0], scale=0.3, size=(30, 2))
    cloud_neg = rng.normal(loc=[-1.0, -0.5], scale=0.35, size=(30, 2))
    axes[1].scatter(*cloud_pos.T, s=12, color='tab:blue', alpha=0.5, label='R+ (schematic)')
    axes[1].scatter(*cloud_neg.T, s=12, color='tab:orange', alpha=0.5, label='R- (schematic)')
    mean_pos, mean_neg = cloud_pos.mean(0), cloud_neg.mean(0)
    axes[1].scatter(*mean_pos, marker='*', s=300, color='tab:blue', edgecolors='k')
    axes[1].scatter(*mean_neg, marker='*', s=300, color='tab:orange', edgecolors='k')
    axes[1].annotate('', xy=mean_pos, xytext=mean_neg,
                      arrowprops=dict(arrowstyle='<->', color='k', lw=1.5))
    mid = (mean_pos + mean_neg) / 2
    axes[1].text(mid[0], mid[1] + 0.3,
                 r'MMD$^2$ = squared distance' + '\nbetween kernel-mean embeddings',
                 ha='center', fontsize=8)
    axes[1].set_title('MMD$^2$: distance between\ngroup embeddings', fontsize=9)
    axes[1].legend(fontsize=7)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.suptitle('SCHEMATIC -- conceptual illustration, not observed data',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    return fig


def _build_effects_table(per_area_df, cv_summary_df, cfg, alpha=None):
    """
    Table of effects: one row per (area, condition) significant in the
    main test (q_value < alpha), analogous to the reference report's
    discovery tables. Adds 'cv_consistent' (the CV significant_both flag,
    i.e. the odd/even-consistency indicator) alongside, when CV was run.
    """
    alpha = alpha if alpha is not None else cfg['fdr_alpha']
    if not len(per_area_df):
        return pd.DataFrame()
    sig = per_area_df[per_area_df['q_value'] < alpha].copy()
    if len(sig) == 0:
        return sig
    if len(cv_summary_df) and 'significant_both' in cv_summary_df.columns:
        cv_lookup = cv_summary_df.set_index(['area', 'feature_version'])['significant_both']
        sig['cv_consistent'] = sig.apply(
            lambda r: cv_lookup.get((r['area'], r['feature_version']), np.nan), axis=1)
    else:
        sig['cv_consistent'] = np.nan
    cols = ['area', 'feature_version', 'mmd2_obs', 'mmd2_z', 'p_value', 'q_value', 'cv_consistent']
    return sig[cols].sort_values('q_value').reset_index(drop=True)


def _fig_table(df, title='', col_fmt=None, max_rows=40):
    """
    Render a DataFrame as a simple table page (matplotlib table) for the
    multi-page report. col_fmt: optional {col_name: '{:.4g}'-style format
    string} for per-column number formatting; booleans render as yes/no.
    """
    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, f"{title}\n(no rows to show)", ha='center', va='center')
        ax.axis('off')
        return fig

    df = df.head(max_rows).copy()
    col_fmt = col_fmt or {}
    for c, fmt in col_fmt.items():
        if c in df.columns:
            df[c] = df[c].apply(lambda v: fmt.format(v) if pd.notna(v) else '')
    for c in df.columns:
        col = df[c]
        if col.map(lambda v: isinstance(v, (bool, np.bool_)) or pd.isna(v)).all():
            df[c] = col.map(lambda v: 'yes' if v is True else ('no' if v is False else '-'))

    fig, ax = plt.subplots(figsize=(min(14, 1.5 * len(df.columns) + 2), 0.35 * len(df) + 1.2))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)
    ax.set_title(title, fontsize=11, pad=14)
    fig.tight_layout()
    return fig


def _fig_cv_null_grid(null_store, cv_summary_df, cfg, title=''):
    """
    Grid of null-distribution histograms -- rows = areas (AGGREGATED
    first), columns = conditions -- consolidating both CV directions into
    one overlaid cell per (area, condition), RED-FRAMED where that
    combination is significant_both (odd/even-consistent) per
    cv_summary_df's BH-FDR correction. Mirrors the reference report's
    per-panel null-distribution grids, but with both directions shown
    together rather than as separate odd->even / even->odd grids.

    null_store : dict {(area, feature_version): {'null_A','obs_A',
        'null_B','obs_B','is_agg'}} -- see run_mmd_cv / _run_cv_combos.
    """
    if not null_store:
        return None
    areas = sorted({k[0] for k in null_store}, key=lambda a: (a != 'ALL_AREAS_AGGREGATED', a))
    conditions = sorted({k[1] for k in null_store})

    sig_lookup = {}
    if len(cv_summary_df) and 'significant_both' in cv_summary_df.columns:
        for _, row in cv_summary_df.iterrows():
            sig_lookup[(row['area'], row['feature_version'])] = bool(row['significant_both'])

    nrow, ncol = len(areas), len(conditions)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 1.8 * nrow), squeeze=False)
    for i, area in enumerate(areas):
        for j, cond in enumerate(conditions):
            ax = axes[i, j]
            entry = null_store.get((area, cond))
            if entry is None:
                ax.axis('off')
                continue
            ax.hist(entry['null_A'], bins=25, color='tab:blue', alpha=0.4,
                     density=True, label='fit odd/test even')
            ax.hist(entry['null_B'], bins=25, color='tab:orange', alpha=0.4,
                     density=True, label='fit even/test odd')
            ax.axvline(entry['obs_A'], color='tab:blue', lw=1.2)
            ax.axvline(entry['obs_B'], color='tab:orange', lw=1.2)
            ax.set_xticks([])
            ax.set_yticks([])
            if sig_lookup.get((area, cond), False):
                for spine in ax.spines.values():
                    spine.set_edgecolor('crimson')
                    spine.set_linewidth(2.5)
            if i == 0:
                ax.set_title(cond, fontsize=7, rotation=30, ha='left')
            if j == 0:
                ax.set_ylabel(area, fontsize=7, rotation=0, ha='right', va='center')
    handles = [mpatches.Patch(color='tab:blue', alpha=0.4, label='fit odd, test even'),
               mpatches.Patch(color='tab:orange', alpha=0.4, label='fit even, test odd')]
    fig.legend(handles=handles, loc='upper right', fontsize=7)
    fig.suptitle(f"{title}\nred frame = odd/even-consistent "
                 f"(significant BOTH directions, BH-FDR)", fontsize=10)
    fig.tight_layout()
    return fig


def generate_peth_report(feature_matrix_path, config_path=None, output_dir=None,
                          make_figures=True, run_cv=True, results=None):
    """
    Build a single multi-page PDF report (peth_report.pdf), modeled on a
    hand-built reference report, covering:
        Page 1: metric schematics (RBF kernel, MMD^2) -- conceptual, not
                real data (see _fig_metric_schematic).
        Page 2: neuron/mouse point-cloud visualization -- REAL data, for
                the single most-significant (area, condition) combination
                (see _fig_neuron_mouse_point_cloud).
        Page 3: table of effects -- every (area, condition) surviving
                BH-FDR in the main test, with the CV significant_both flag
                alongside as an odd/even-consistency indicator (see
                _build_effects_table).
        Page 4: grid of CV null histograms, one cell per (area,
                condition), red-framed where odd/even-consistent (see
                _fig_cv_null_grid). Omitted if CV wasn't run.

    This is a REPORT wrapper around run_all() -- every number in it comes
    from that same analysis, not a separate computation. Pass an already-
    computed `results` dict (as returned by run_all) to skip re-running
    the whole pipeline; otherwise run_all() is called internally with the
    given feature_matrix_path/config_path/output_dir/make_figures/run_cv.

    SCOPE NOTE: this reports the MMD method only (this module's own
    test). It does NOT implement the PCA-pseudo-F or energy-distance
    methods, or their 3-method cross-comparison, from the reference
    report that inspired this layout -- those are separate statistical
    methods requiring substantial new implementation, not a reporting
    change, and weren't part of what was asked for here.

    Returns
    -------
    str : path to the written peth_report.pdf
    """
    if results is None:
        results = run_all(feature_matrix_path, config_path=config_path,
                           output_dir=output_dir, make_figures=make_figures, run_cv=run_cv)
    per_area_df = results['per_area']
    cv_summary_df = results.get('cv_summary', pd.DataFrame())
    null_store = results.get('cv_null_store', {})

    cfg = dict(DEFAULT_MMD_CFG)
    if config_path is not None:
        with open(config_path, 'r') as f:
            file_cfg = yaml.load(f, Loader=yaml.FullLoader) or {}
        cfg.update(file_cfg.get('mmd', {}))

    out_dir = str(output_dir) if output_dir is not None else str(
        pathlib.Path(feature_matrix_path).parent / 'mmd')
    os.makedirs(out_dir, exist_ok=True)

    rep_area, rep_cond = None, None
    if len(per_area_df):
        rep_row = per_area_df.sort_values('q_value').iloc[0]
        rep_area, rep_cond = rep_row['area'], rep_row['feature_version']

    report_path = os.path.join(out_dir, 'peth_report.pdf')
    with PdfPages(report_path) as pdf:
        fig = _fig_metric_schematic()
        pdf.savefig(fig)
        plt.close(fig)

        if rep_area is not None:
            feature_matrix, neuron_meta, X_odd, X_even = _load_npz_data(feature_matrix_path)
            if cfg['trial_type_slices'] is None and 'n_bins_list' in neuron_meta:
                cfg['trial_type_slices'] = _build_default_trial_type_slices(
                    neuron_meta['n_bins_list'], cond_labels=cfg.get('cond_labels'))
            feature_versions = _slice_feature_matrix(
                feature_matrix, cfg['trial_type_slices'],
                exclude_bins=cfg.get('exclude_bins'), exclude_from_full=cfg.get('exclude_from_full'))
            feats_raw, mice, cohorts, _ = _get_neuron_data(
                neuron_meta, feature_versions[rep_cond], cfg, area=rep_area)
            feats_pca, _ = _maybe_pca_reduce(feats_raw, cfg)
            fig = _fig_neuron_mouse_point_cloud(
                feats_pca, mice, cohorts, cohort_values=_cohort_values(cfg),
                title=f"Representative example: area={rep_area}, condition={rep_cond}")
            pdf.savefig(fig)
            plt.close(fig)

        effects_df = _build_effects_table(per_area_df, cv_summary_df, cfg)
        fig = _fig_table(
            effects_df, title="Table of effects (BH-FDR significant, main test)",
            col_fmt={'mmd2_obs': '{:.4g}', 'mmd2_z': '{:.2f}',
                     'p_value': '{:.4g}', 'q_value': '{:.4g}'})
        pdf.savefig(fig)
        plt.close(fig)

        if null_store:
            fig = _fig_cv_null_grid(
                null_store, cv_summary_df, cfg,
                title="Odd/even CV null distributions, every area x condition")
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)

    print(f"[MMD] report written to {report_path}")
    return report_path


def run_all(feature_matrix_path, config_path=None,
            output_dir=None, make_figures=True,
            test_n_neurons=None, test_random_subset=False,
            run_cv=True):
    """
    Single entry point. Loads EVERYTHING (feature matrix + per-neuron
    metadata, plus X_odd/X_even if present) from `feature_matrix_path` --
    no unit_table, no ROC join -- optionally merges an `mmd:` section from
    `config_path`'s YAML over DEFAULT_MMD_CFG, then runs, IN THIS ORDER:
    (1) the aggregated MMD, (2) the odd/even cross-validation pass (see
    run_mmd_cv; only if X_odd/X_even are present and run_cv=True), and
    (3) the per-area MMD sweep -- deliberately last, since it's by far the
    slowest step (many small tests across every valid area), so the
    aggregated and CV results/figures are already fully on disk well
    before it finishes. Within each of these, if PCA testing is enabled
    (see pca_test_variance_retained), the PCA-space test for each
    condition computes -- and its figures (pca_variance/, pc_loading/)
    save -- before that condition's full-dimensional test.

    Parameters
    ----------
    feature_matrix_path : str or Path
        Path to feature_matrix.npz (must contain X plus the per-neuron
        metadata arrays listed in _load_npz_data's docstring; X_odd/X_even
        are optional and only needed for run_cv).
    config_path : str or Path or None
        Optional path to config.yaml. If it has an `mmd:` section, that
        sub-dict is merged over DEFAULT_MMD_CFG.
    output_dir : str or Path or None
        Where results/figures are written. Defaults to
        `feature_matrix_path.parent / 'mmd'` if not given (or, in test
        mode, that path with a '_TEST' suffix -- see test_n_neurons).
    make_figures : bool
    test_n_neurons : int or None
        TEST MODE. If given, restricts the analysis to only the first
        `test_n_neurons` neurons (or a random subset, see
        test_random_subset) BEFORE any area filtering or testing --
        exercises the entire pipeline end-to-end (loading, valid-area
        filtering, both MMD tests, all figures) in seconds/minutes instead
        of hours, to catch config/data problems before committing to a
        full run. Output is written to a separate '_TEST'-suffixed
        directory so it can never be mistaken for or overwrite a real run.
        With few neurons, cfg['min_mice_per_area']/['min_neurons_per_mouse']
        may leave 0 valid areas -- that's expected and still a useful
        smoke test of the "0 valid areas" code path itself; lower those
        thresholds via config_path/an explicit cfg override, or increase
        test_n_neurons, if you want to see an actual test run through.
        NOTE: test_n_neurons does NOT currently subsample X_odd/X_even, so
        run_cv is skipped automatically in test mode.
    test_random_subset : bool
        Only used when test_n_neurons is set. False (default): take the
        first test_n_neurons neurons, in their existing npz order (fastest,
        fully deterministic). True: draw a random test_n_neurons subset
        instead (seeded by cfg['mmd_random_seed']) -- more likely to span
        multiple mice/areas if the npz happens to be ordered by
        mouse/session, at the cost of no longer being the literal "first N".
    run_cv : bool
        If True (default) and X_odd/X_even are present in the npz, also
        runs the odd/even cross-validation pass after the main tests.
        Set False to skip it even when the matrices are available (e.g.
        for a quick iteration where CV's ~2x extra test count isn't
        needed yet).

    Returns
    -------
    dict with keys 'per_area', 'aggregated', 'valid_areas', 'cv',
    'cv_summary'. 'cv'/'cv_summary' are empty DataFrames if CV didn't run
    (no X_odd/X_even, run_cv=False, or test mode).
    """
    feature_matrix_path = pathlib.Path(feature_matrix_path)
    t_run_start = time.time()
    feature_matrix, neuron_meta, X_odd, X_even = _load_npz_data(feature_matrix_path)

    cfg = dict(DEFAULT_MMD_CFG)
    file_cfg = {}
    if config_path is not None:
        with open(config_path, 'r') as f:
            file_cfg = yaml.load(f, Loader=yaml.FullLoader) or {}
        cfg.update(file_cfg.get('mmd', {}))
        print(f"[MMD] merged mmd: config from {config_path}")

    default_out = (str(output_dir) if output_dir is not None
                   else str(feature_matrix_path.parent / 'mmd'))
    if test_n_neurons is not None:
        n_before = feature_matrix.shape[0]
        feature_matrix, neuron_meta = _subsample_npz_data(
            feature_matrix, neuron_meta, test_n_neurons,
            random_subset=test_random_subset, seed=cfg['mmd_random_seed'])
        default_out = default_out.rstrip('/\\') + '_TEST'
        print(f"[MMD] *** TEST MODE *** subsampled {n_before} -> "
              f"{feature_matrix.shape[0]} neurons "
              f"({'random' if test_random_subset else 'first'} subset). "
              f"Output -> {default_out} (NOT a real result). "
              f"run_mmd_cv is skipped in test mode (X_odd/X_even not subsampled).")
        X_odd, X_even = None, None  # not subsampled -> would misalign with feature_matrix

    cfg['output_dir'] = default_out
    print(f"[MMD] output_dir = {cfg['output_dir']}")

    if cfg.get('cond_labels') is None and file_cfg:
        # Pull REAL condition names (e.g. "Whisker hit") from the shared
        # pipeline's own get_conditions(cfg), using the FULL pipeline
        # config (file_cfg, not just its `mmd:` sub-dict) -- this is the
        # same cfg build_feature_matrix.py itself used, so period/
        # modality/merge_passive/include_lick_conditions are already
        # whatever produced this exact feature_matrix.npz. Falls back to
        # cond0/cond1/... (silently, with a printed note) if
        # rastermap_utils isn't importable here.
        derived = _get_cond_labels_from_pipeline_cfg(file_cfg)
        if derived is not None:
            cfg['cond_labels'] = derived
            print(f"[MMD] derived cond_labels from get_conditions(cfg): {derived}")

    if cfg['trial_type_slices'] is None and 'n_bins_list' in neuron_meta:
        cfg['trial_type_slices'] = _build_default_trial_type_slices(
            neuron_meta['n_bins_list'], cond_labels=cfg.get('cond_labels'))
        print(f"[MMD] auto-built trial_type_slices from n_bins_list: "
              f"{list(cfg['trial_type_slices'])}")

    print("[MMD] === computing valid areas ===")
    valid_areas = get_valid_areas(neuron_meta, cfg)
    print(f"[MMD] {len(valid_areas)} valid areas: {valid_areas}")
    if test_n_neurons is not None and not valid_areas:
        print("[MMD] *** TEST MODE *** 0 valid areas with this subsample -- "
              "the pipeline itself still runs fine (see run_mmd_per_area's "
              "empty-result handling), but try a larger test_n_neurons or a "
              "lower min_mice_per_area/min_neurons_per_mouse to exercise an "
              "actual test rather than only the empty-result path.")

    print("[MMD] === running aggregated MMD (largest test, runs first) ===")
    agg_df = run_mmd_aggregated(neuron_meta, feature_matrix, cfg,
                                 valid_areas=valid_areas, make_figures=make_figures)

    cv_df = pd.DataFrame()
    cv_summary_df = pd.DataFrame()
    cv_null_store = {}
    if run_cv and X_odd is not None and X_even is not None:
        print("[MMD] === running odd/even cross-validation "
              "(before the per-area sweep) ===")
        cv_df, cv_summary_df, cv_null_store = run_mmd_cv(
            neuron_meta, X_odd, X_even, cfg,
            valid_areas=valid_areas, make_figures=make_figures)
    elif run_cv:
        print("[MMD] run_cv=True but X_odd/X_even unavailable -- skipping CV.")

    print("[MMD] === running per-area MMD (slowest step, runs last) ===")
    per_area_df = run_mmd_per_area(neuron_meta, feature_matrix, cfg,
                                    valid_areas=valid_areas, make_figures=make_figures)

    # #6: evidence-concordance summary (main test vs CV; PERMANOVA can be
    # cross-referenced by re-calling _build_evidence_concordance yourself
    # with permanova_significant={area: bool, ...} -- this module has no
    # access to that result on its own).
    concordance_df, method_cols = _build_evidence_concordance(per_area_df, cv_summary_df, cfg)
    if len(concordance_df):
        concordance_df.to_csv(os.path.join(cfg['output_dir'], 'evidence_concordance.csv'), index=False)
        if make_figures and method_cols:
            fig = _fig_evidence_concordance(
                concordance_df, method_cols,
                title="Evidence concordance across areas ('full' condition)")
            _save(fig, os.path.join(cfg['output_dir'], 'evidence_concordance.pdf'))

    # Table of effects and the metric schematic, as standalone outputs --
    # both are also assembled into peth_report.pdf by generate_peth_report(),
    # but are produced here too so they exist without needing that separate
    # call.
    effects_df = _build_effects_table(per_area_df, cv_summary_df, cfg)
    effects_df.to_csv(os.path.join(cfg['output_dir'], 'effects_table.csv'), index=False)
    if make_figures:
        fig = _fig_table(
            effects_df, title="Table of effects (BH-FDR significant, main test)",
            col_fmt={'mmd2_obs': '{:.4g}', 'mmd2_z': '{:.2f}',
                     'p_value': '{:.4g}', 'q_value': '{:.4g}'})
        _save(fig, os.path.join(cfg['output_dir'], 'effects_table.pdf'))

        fig = _fig_metric_schematic()
        _save(fig, os.path.join(cfg['output_dir'], 'metric_schematic.pdf'))

    n_sig_area = int((per_area_df['q_value'] < cfg['fdr_alpha']).sum()) if len(per_area_df) else 0
    n_sig_agg = int((agg_df['q_value'] < cfg['fdr_alpha']).sum()) if len(agg_df) else 0
    n_sig_cv = int(cv_summary_df['significant_both'].sum()) if 'significant_both' in cv_summary_df else 0
    print(f"[MMD] === done in {time.time() - t_run_start:.1f}s total === "
          f"{n_sig_area}/{len(per_area_df)} per-area tests significant, "
          f"{n_sig_agg}/{len(agg_df)} aggregated tests significant, "
          f"{n_sig_cv}/{len(cv_summary_df)} CV combinations significant BOTH "
          f"directions (BH-FDR alpha={cfg['fdr_alpha']})"
          + (" *** TEST MODE -- not a real result ***" if test_n_neurons is not None else ""))

    return {'per_area': per_area_df, 'aggregated': agg_df, 'valid_areas': valid_areas,
            'cv': cv_df, 'cv_summary': cv_summary_df, 'cv_null_store': cv_null_store,
            'evidence_concordance': concordance_df}


if __name__ == '__main__':
    print(
        "This module is meant to be imported and called with a path to "
        "feature_matrix.npz -- no unit_table needed:\n\n"
        "    import cluster_mmd_analysis as mmd\n"
        "    results = mmd.run_all(feature_matrix_path, config_path=path_to_config)\n"
    )