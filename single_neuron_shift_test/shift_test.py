"""
shift_test.py
--------------
Vectorized implementation of the "shift test" for independence of two
autocorrelated time series (Harris, 2021, "A Shift Test for Independence
in Generic Time Series", arXiv:2012.06862; reference code:
github.com/kdharris101/nonsense-correlations).

Idea
----
Two autocorrelated series can show strong "instantaneous" correlation
purely because both drift slowly (Yule, 1926). Standard significance
tests (Pearson p-value, regression F-test, etc.) assume i.i.d. samples
and so wildly overstate significance.

The shift test instead asks: is the association between X and Y at
zero lag bigger than it is at many non-zero lags? Concretely:

  1. Take a fixed central window of X, X[N : T-N]  (length D = T - 2N).
  2. For each shift s in -N..N, compute V_s = score(X[N:T-N], Y[s+N : s+T-N]).
  3. V_0 is the real, unshifted association.
  4. m = #{s : V_s is at least as extreme as V_0}.
  5. Conservative p = m / (N+1); this is a valid p-value (never anticonservative)
     PROVIDED the shifted series (Y here) is itself (locally) stationary
     and independent of X under the null. An approximate p = m / (2N+1)
     has ~2x the power but only holds in the N -> infinity limit and can
     leak false positives for small N.

Which series should be the one that gets shifted?
The shifted series must be the (locally) stationary one. If you are
testing "neuron vs. behaviour, controlling for drift", you generally
want to shift the *behavioural / task* signal (e.g. trial-by-trial
performance, or the drift/probe-motion regressor) and keep the neural
trace fixed, on the logic that slow probe drift is closer to stationary
noise than neural spiking is. But the test is symmetric in validity
terms -- it only requires that whichever one you shift be stationary.
Pick based on your scientific question and check both directions if
unsure.

This module provides:
  - sliding_shifts:        build all shifted copies of a 1D array, vectorized
  - pearson_corr / spearman_corr: vectorized correlation of an (T,) series
                            against many (T, C) series at once (no Python loop
                            over neurons)
  - abs_pearson_corr / abs_spearman_corr: absolute-value versions, used by
                            default for a two-sided test with no assumed
                            direction (see shift_test docstring for why this
                            keeps the conservative guarantee exact)
  - shift_test:             the core test for ONE pair of series
  - shift_test_many:        the same test for MANY series (e.g. neurons) vs.
                            ONE reference series (e.g. behaviour), fully
                            vectorized over neurons AND shifts
  - shift_test_pairwise:    many-vs-many (e.g. all neurons vs all neurons,
                            or all neurons vs several behavioural regressors)

All "association" scores are computed with simple vectorized numpy/scipy
operations -- no Python-level loop over shifts or over channels. For T
timepoints, N shifts and C channels this costs O((2N+1) * T * C) flops,
which is what the reference notebook's loop also costs, but here it is
done as dense matrix algebra so it is fast in practice for realistic
N (tens) and C (hundreds-thousands of neurons).
"""

from __future__ import annotations
import numpy as np
from scipy import stats


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _check_1d(y, name="y"):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {y.shape}")
    return y


def sliding_shifts(y: np.ndarray, N: int) -> np.ndarray:
    """
    Build the (2N+1, T-2N) matrix of shifted, centrally-cropped copies of y.

    Row s (s = 0..2N, corresponding to shift = s - N) is
        y[s : s + (T - 2N)]
    i.e. row N (the middle row) is the unshifted central segment
    y[N : T-N], matching V_0 in the paper.

    Parameters
    ----------
    y : (T,) array
    N : int, max shift magnitude. Requires T > 2N.

    Returns
    -------
    (2N+1, D) array, D = T - 2N
    """
    y = _check_1d(y)
    T = y.shape[0]
    D = T - 2 * N
    if D <= 0:
        raise ValueError(f"T={T} must be > 2N={2*N}")
    # stride trick: build an overlapping-window view, no copy
    stride = y.strides[0]
    shifted = np.lib.stride_tricks.as_strided(
        y, shape=(2 * N + 1, D), strides=(stride, stride)
    )
    return shifted.copy()  # copy so callers can't accidentally corrupt y


# --------------------------------------------------------------------------
# vectorized association scores: ONE reference series vs MANY channels
# --------------------------------------------------------------------------

def pearson_corr(ref: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """
    Pearson correlation of `ref` (D,) against every column of `mat` (D, C).
    Vectorized: one pass, no loop over C. Returns (C,) array.
    """
    ref = _check_1d(ref, "ref")
    mat = np.asarray(mat, dtype=float)
    D = ref.shape[0]
    if mat.shape[0] != D:
        raise ValueError(f"ref has length {D}, mat has {mat.shape[0]} rows")

    r = ref - ref.mean()
    m = mat - mat.mean(axis=0, keepdims=True)
    num = r @ m  # (C,)
    denom = np.sqrt((r @ r) * np.sum(m * m, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / denom
    return out


def spearman_corr(ref: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """
    Spearman correlation of `ref` (D,) against every column of `mat` (D, C).
    Vectorized via rank transform + pearson_corr (no scipy loop over columns).
    """
    ref_rank = stats.rankdata(ref)
    mat_rank = np.apply_along_axis(stats.rankdata, 0, np.asarray(mat))
    return pearson_corr(ref_rank, mat_rank)


def abs_pearson_corr(ref: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """|Pearson correlation|, vectorized. Use this (the default in
    shift_test_many) when you have no directional hypothesis: folding the
    absolute value into the score itself keeps the test exactly two-sided
    while preserving the paper's one-sided proof (m = #{s: |V_s| >= |V_0|})."""
    return np.abs(pearson_corr(ref, mat))


def abs_spearman_corr(ref: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """|Spearman correlation|, vectorized. See abs_pearson_corr."""
    return np.abs(spearman_corr(ref, mat))


def _pearson_corr_all_shifts(ref_shifted: np.ndarray, mat_center: np.ndarray) -> np.ndarray:
    """
    Fully vectorized Pearson correlation between every row of ref_shifted
    (S, D) and every column of mat_center (D, C), for all S shifts and C
    channels at once -- no Python loop over shifts. Returns (S, C).
    """
    S, D = ref_shifted.shape
    r = ref_shifted - ref_shifted.mean(axis=1, keepdims=True)  # (S, D)
    m = mat_center - mat_center.mean(axis=0, keepdims=True)  # (D, C)
    num = r @ m  # (S, C)
    r_ss = np.sum(r * r, axis=1)  # (S,)
    m_ss = np.sum(m * m, axis=0)  # (C,)
    denom = np.sqrt(r_ss[:, None] * m_ss[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / denom
    return out


def _spearman_corr_all_shifts(ref_shifted: np.ndarray, mat_center: np.ndarray) -> np.ndarray:
    """Spearman version of _pearson_corr_all_shifts: rank once, then a single
    matrix multiply across all shifts and channels."""
    ref_ranks = np.apply_along_axis(stats.rankdata, 1, ref_shifted)  # (S, D), ranked once
    mat_ranks = np.apply_along_axis(stats.rankdata, 0, mat_center)  # (D, C), ranked once
    return _pearson_corr_all_shifts(ref_ranks, mat_ranks)


# --------------------------------------------------------------------------
# core test, ONE pair of series
# --------------------------------------------------------------------------

def shift_test(
    x: np.ndarray,
    y: np.ndarray,
    N: int,
    score_fn=None,
    bigger_is_more_associated: bool = True,
):
    """
    Shift test for independence between two 1D series x and y (Harris 2021).
    `y` is the one that gets shifted; it must be the (locally) stationary one.

    Parameters
    ----------
    x, y : (T,) arrays
    N : max shift
    score_fn : callable(x_seg, y_seg) -> float, default |Pearson correlation|.
        Must accept two (D,) arrays. Use a custom function for e.g. mutual
        information, regression error, log-odds ratio of a contingency
        table (as in the paper's categorical example), etc. The default
        uses the absolute value of the correlation, which makes the test
        two-sided "for free": m = #{s : |V_s| >= |V_0|} is exactly the
        paper's one-sided statistic applied to the score |V|, so the
        conservative bound m/(N+1) still holds rigorously. If you have a
        directional hypothesis (e.g. "this neuron's firing rate should
        increase with performance"), pass score_fn returning the signed
        correlation instead -- that gives a more powerful, still-valid
        one-sided test.
    bigger_is_more_associated : if True (default), a *larger* score_fn value
        means more associated (e.g. |correlation], or correlation under a
        directional hypothesis). Set False for score functions where
        smaller = more associated (e.g. prediction error, as in the
        reference notebook's ridge_error).

    Returns
    -------
    dict with:
        m                : count of shifts at least as associated as the unshifted one
        p_conservative   : m / (N+1), valid finite-N p-value (Theorem 2 / Corollary)
        p_approx         : m / (2N+1), valid only as N -> inf, ~2x power
        scores           : (2N+1,) array of V_s (using score_fn), ordered by shift -N..N
        shift0_index     : index into `scores` corresponding to shift=0 (== N)
        score_at_shift0  : score_fn value at lag 0 -- this is what m/p are based on.
                            With the default abs_pearson_corr score_fn this is
                            ALWAYS POSITIVE and carries no direction information.
        sign_at_shift0   : the raw SIGNED Pearson correlation at lag 0, computed
                            independently of whatever score_fn you used. THIS is
                            what tells you the direction of the effect (positive
                            vs negative association). The p-value on its own says
                            only "this is unlikely under independence" and carries
                            no sign information, especially with a two-sided score
                            like the default |correlation|.
    """
    if score_fn is None:
        score_fn = lambda a, b: abs(pearson_corr(a, b[:, None])[0])

    x = _check_1d(x, "x")
    y = _check_1d(y, "y")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")

    T = x.shape[0]
    D = T - 2 * N
    if D <= 0:
        raise ValueError(f"T={T} must be > 2N={2*N}")

    x_center = x[N : T - N]
    y_shifted = sliding_shifts(y, N)  # (2N+1, D)
    sign_at_shift0 = float(pearson_corr(x_center, y_shifted[N][:, None])[0])

    scores = np.array([score_fn(x_center, y_shifted[s]) for s in range(2 * N + 1)])
    return _summarize(scores, N, bigger_is_more_associated, sign_at_shift0)


def _summarize(scores: np.ndarray, N: int, bigger_is_more_associated: bool, sign_at_shift0: float = None):
    s0 = scores[N]
    if bigger_is_more_associated:
        m = int((scores >= s0).sum())
    else:
        m = int((scores <= s0).sum())

    p_cons = min(m / (N + 1), 1.0)
    p_appx = min(m / (2 * N + 1), 1.0)
    out = {
        "m": m,
        "p_conservative": p_cons,
        "p_approx": p_appx,
        "scores": scores,
        "shift0_index": N,
        "score_at_shift0": s0,
    }
    if sign_at_shift0 is not None:
        out["sign_at_shift0"] = sign_at_shift0
    return out


# --------------------------------------------------------------------------
# vectorized test, MANY channels (e.g. neurons) vs ONE reference series
# --------------------------------------------------------------------------

def shift_test_many(
    ref: np.ndarray,
    mat: np.ndarray,
    N: int,
    corr_fn=abs_pearson_corr,
    bigger_is_more_associated: bool = True,
    shift_ref: bool = True,
):
    """
    Run the shift test between a single reference series `ref` (e.g.
    behavioural performance, or a drift/probe-motion regressor) and many
    channels `mat` (e.g. one row per neuron) at once. Fully vectorized:
    no Python loop over neurons, and no Python loop over shifts beyond
    building the shift matrix.

    Parameters
    ----------
    ref : (T,) array -- the single series, e.g. behaviour or drift
    mat : (T, C) array -- e.g. C neurons x T timepoints, transposed to T x C
          (pass mat.T if your neural data is C x T)
    N : max shift
    corr_fn : vectorized association function with signature
        corr_fn(ref_1d, mat_2d) -> (C,) array. Default abs_pearson_corr
        (two-sided by construction, see its docstring). Pass pearson_corr
        directly for a one-sided, signed test if you have a directional
        hypothesis, or abs_spearman_corr / spearman_corr for rank-based
        versions robust to outliers and nonlinearity.
    bigger_is_more_associated : see shift_test docstring.
    shift_ref : if True (default), `ref` is the one that gets shifted across
        lags and `mat` columns stay fixed in the center window. This is the
        natural choice when ref is the more plausibly-stationary signal
        (e.g. drift, or a behavioural performance curve). Set False to shift
        the columns of `mat` instead and keep `ref` fixed -- use this if mat
        is actually the more stationary signal in your application.

    Returns
    -------
    dict with:
        m                : (C,) int array
        p_conservative   : (C,) float array
        p_approx         : (C,) float array
        scores           : (2N+1, C) array, V_s per channel (using corr_fn), ordered shift -N..N
        shift0_index     : int, index of shift=0 in `scores`
        score_at_shift0  : (C,) corr_fn value at lag 0 per channel -- this is what
                            m/p are based on. With the default abs_pearson_corr
                            this is ALWAYS POSITIVE and carries no direction info.
        sign_at_shift0   : (C,) the raw SIGNED Pearson correlation at lag 0 for
                            every channel, computed independently of corr_fn.
                            THIS is what tells you direction (positive vs negative
                            association) for each channel -- the p-value alone
                            does not, especially with the default two-sided score.
    """
    ref = _check_1d(ref, "ref")
    mat = np.asarray(mat, dtype=float)
    T = ref.shape[0]
    if mat.shape[0] != T:
        raise ValueError(f"ref has length {T}, mat has {mat.shape[0]} rows (expected T x C)")
    C = mat.shape[1]
    D = T - 2 * N
    if D <= 0:
        raise ValueError(f"T={T} must be > 2N={2*N}")

    scores = np.empty((2 * N + 1, C))

    if shift_ref:
        mat_center = mat[N : T - N]  # (D, C), fixed
        ref_shifted = sliding_shifts(ref, N)  # (2N+1, D)
        sign_at_shift0 = pearson_corr(ref_shifted[N], mat_center)  # (C,), raw signed r at lag 0
        if corr_fn in (pearson_corr, abs_pearson_corr):
            scores = _pearson_corr_all_shifts(ref_shifted, mat_center)
            if corr_fn is abs_pearson_corr:
                scores = np.abs(scores)
        elif corr_fn in (spearman_corr, abs_spearman_corr):
            scores = _spearman_corr_all_shifts(ref_shifted, mat_center)
            if corr_fn is abs_spearman_corr:
                scores = np.abs(scores)
        else:
            # generic fallback: any user-supplied corr_fn(ref_1d, mat_2d) -> (C,)
            for s in range(2 * N + 1):
                scores[s] = corr_fn(ref_shifted[s], mat_center)
    else:
        ref_center = ref[N : T - N]  # (D,), fixed
        mat_at_shift0 = mat[N : T - N]  # same window as lag 0
        sign_at_shift0 = pearson_corr(ref_center, mat_at_shift0)  # (C,), raw signed r at lag 0
        for s in range(2 * N + 1):
            # build shifted window of mat for this lag, one column-block at a time
            lo, hi = s, s + D
            scores[s] = corr_fn(ref_center, mat[lo:hi])

    s0 = scores[N]  # (C,)
    if bigger_is_more_associated:
        m = (scores >= s0).sum(axis=0)
    else:
        m = (scores <= s0).sum(axis=0)

    p_cons = np.minimum(m / (N + 1), 1.0)
    p_appx = np.minimum(m / (2 * N + 1), 1.0)

    return {
        "m": m,
        "p_conservative": p_cons,
        "p_approx": p_appx,
        "scores": scores,
        "shift0_index": N,
        "score_at_shift0": s0,
        "sign_at_shift0": sign_at_shift0,
    }


# --------------------------------------------------------------------------
# convenience: many-vs-many (e.g. all neurons vs all neurons / regressors)
# --------------------------------------------------------------------------

def shift_test_pairwise(
    mat_x: np.ndarray,
    mat_y: np.ndarray,
    N: int,
    corr_fn=abs_pearson_corr,
    bigger_is_more_associated: bool = True,
):
    """
    Run shift_test_many for every column of mat_x against the full set
    mat_y, looping (in Python) only over the columns of mat_x -- the
    inner comparison against all of mat_y's columns is vectorized.
    Useful for e.g. testing every neuron against several candidate
    nuisance regressors (drift, pupil, motion...) at once.

    Parameters
    ----------
    mat_x : (T, Cx) array, e.g. Cx neurons
    mat_y : (T, Cy) array, e.g. Cy behavioural/nuisance regressors
    N : max shift
    corr_fn, bigger_is_more_associated : see shift_test_many

    Returns
    -------
    dict of (Cx, Cy) arrays: m, p_conservative, p_approx
    """
    mat_x = np.asarray(mat_x, dtype=float)
    mat_y = np.asarray(mat_y, dtype=float)
    Cx = mat_x.shape[1]
    Cy = mat_y.shape[1]

    m = np.empty((Cx, Cy))
    p_cons = np.empty((Cx, Cy))
    p_appx = np.empty((Cx, Cy))

    for i in range(Cx):
        res = shift_test_many(
            mat_x[:, i], mat_y, N, corr_fn=corr_fn,
            bigger_is_more_associated=bigger_is_more_associated,
        )
        m[i] = res["m"]
        p_cons[i] = res["p_conservative"]
        p_appx[i] = res["p_approx"]

    return {"m": m, "p_conservative": p_cons, "p_approx": p_appx}


# --------------------------------------------------------------------------
# quick self-test / demo (mirrors the paper's simulated example, real-valued)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 500
    N = 19

    def make_drifting_series(T, tau=80.0, rng=rng):
        """slow random walk-ish drift, like probe drift / slow behavioural state"""
        steps = rng.standard_normal(T) / np.sqrt(tau)
        return np.cumsum(steps)

    # Case 1: independent drift, no real correlation
    x_indep = make_drifting_series(T)
    y_indep = make_drifting_series(T)

    # Case 2: genuinely correlated (shared signal) + independent drift noise
    shared = make_drifting_series(T)
    x_corr = shared + 0.3 * make_drifting_series(T)
    y_corr = shared + 0.3 * make_drifting_series(T)

    naive_p_indep = stats.pearsonr(x_indep, y_indep).pvalue
    naive_p_corr = stats.pearsonr(x_corr, y_corr).pvalue

    res_indep = shift_test(x_indep, y_indep, N)
    res_corr = shift_test(x_corr, y_corr, N)

    print("=== Independent drifting series (single example) ===")
    print(f"naive Pearson p          = {naive_p_indep:.4g}  (often falsely significant)")
    print(f"shift test conservative p= {res_indep['p_conservative']:.4g}")
    print(f"shift test approximate p = {res_indep['p_approx']:.4g}")
    print(f"sign at lag 0 (direction)= {res_indep['sign_at_shift0']:+.3f}")

    print("\n=== Genuinely correlated series (single example) ===")
    print(f"naive Pearson p          = {naive_p_corr:.4g}")
    print(f"shift test conservative p= {res_corr['p_conservative']:.4g}")
    print(f"shift test approximate p = {res_corr['p_approx']:.4g}")
    print(f"sign at lag 0 (direction)= {res_corr['sign_at_shift0']:+.3f}")

    # --- calibration check: repeat the null many times, confirm false
    # positive rate stays at/below nominal alpha despite autocorrelation ---
    n_sims = 500
    alpha = 0.05
    false_pos_naive = 0
    false_pos_cons = 0
    false_pos_appx = 0
    for _ in range(n_sims):
        xa = make_drifting_series(T)
        ya = make_drifting_series(T)
        if stats.pearsonr(xa, ya).pvalue <= alpha:
            false_pos_naive += 1
        r = shift_test(xa, ya, N)
        false_pos_cons += r["p_conservative"] <= alpha
        false_pos_appx += r["p_approx"] <= alpha

    print(f"\n=== Calibration over {n_sims} null simulations (true independence) ===")
    print(f"naive Pearson false-positive rate         = {false_pos_naive/n_sims:.3f}  (should be ~{alpha}, but autocorrelation inflates it)")
    print(f"shift test conservative false-positive rate = {false_pos_cons/n_sims:.3f}  (should be <= {alpha})")
    print(f"shift test approximate false-positive rate  = {false_pos_appx/n_sims:.3f}  (should be ~<= {alpha} for large N)")

    # vectorized many-channel demo: 200 "neurons", a subset truly tuned to behaviour
    C = 200
    n_true = 10
    behaviour = make_drifting_series(T)
    neurons = np.stack([make_drifting_series(T) for _ in range(C)], axis=1)
    neurons[:, :n_true] += 1.2 * behaviour[:, None]  # strong, clearly real coupling

    res_many = shift_test_many(behaviour, neurons, N, corr_fn=abs_pearson_corr)
    sig = res_many["p_conservative"] <= 0.05
    print(f"\n=== {C} simulated neurons, {n_true} truly tuned to behaviour ===")
    print(f"flagged significant (p_conservative<=0.05): {sig.sum()} / {C}")
    print(f"of the {n_true} truly tuned neurons, flagged: {sig[:n_true].sum()} / {n_true}")
    print(f"sign_at_shift0 for flagged neurons: {np.round(res_many['sign_at_shift0'][sig], 2)}")
    print("(all positive here, since we added +1.2*behaviour to those neurons)")
