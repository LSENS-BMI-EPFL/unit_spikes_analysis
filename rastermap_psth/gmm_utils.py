"""
gmm_utils.py — Spectral Embedding + GMM clustering
    Python translation of the MATLAB pipeline A3 → A4 → A5.

Pipeline
--------
1.  (Optional) Determine N_PCA via permutation test (A3)
2.  Row-center X; project neurons into PCA space → PCA_Data (N × N_PCA)
3.  Build Gaussian similarity matrix S; tune σ so mean(S_off_diag) ≈ 0.5 (A4)
4.  Normalized Laplacian  L = I − D^{−1/2} S D^{−1/2}
5.  Sparse Lanczos eigensolver → keep eigenvectors 1 .. n_spectral  (skip trivial ev 0)
6.  Fit GMM in spectral space; optionally select K via BIC sweep (A5)

Consumed by run_clustering.py via the `do_gmm` config flag.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from scipy.optimize import brentq
from scipy.stats import ttest_1samp
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed

from rastermap_psth.rastermap_utils import _save


# ── Default config additions (merged into cfg in run_clustering) ──────────────
GMM_CFG_DEFAULTS: dict = dict(
    do_gmm               = False,
    # PCA step
    gmm_n_pca            = None,   # int → use fixed; None → permutation test
    gmm_pca_permutations = 100,    # number of shuffle iterations for PC selection
    # Spectral embedding step
    gmm_sigma            = None,   # float → use fixed; None → auto-tune to mean(S)=0.5
    gmm_n_spectral       = 13,     # eigenvectors to keep after skipping trivial ev 0
    # GMM step
    gmm_k                = None,   # int → use fixed K; None → BIC sweep
    gmm_k_range          = list(range(5, 45)),
    gmm_n_init           = 50,     # replicates for final GMM fit (MATLAB used 5000)
    gmm_covariance_type  = "diag", # matches MATLAB CovarianceType='diagonal'
    gmm_max_iter         = 1000,
)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — PCA neuron coordinates
# ─────────────────────────────────────────────────────────────────────────────

def _row_center(X: np.ndarray) -> np.ndarray:
    """Subtract per-neuron mean (matches MATLAB row-centering in A3/A4)."""
    return X - X.mean(axis=1, keepdims=True)


def _pca_coords(X_c: np.ndarray, n_pca: int) -> np.ndarray:
    """
    Return neuron coordinates in PCA space, shape (N_neurons, n_pca).

    Matches MATLAB A4:
        [Coeff, ~, ~] = pca(X', 'Centered', false)   % X' is (N_time × N_neurons)
        PCA_Data = Coeff(:, 1:N_PCA)                  % (N_neurons × N_PCA)

    sklearn equivalent:
        fit PCA on X_c.T  (N_time × N_neurons);
        pca.components_.T  gives (N_neurons × N_PCA) = Coeff.
    These are the left singular vectors of X_c (eigenvectors of X_c @ X_c.T),
    representing each neuron's loading on each principal temporal mode.
    """
    pca = PCA(n_components=n_pca, svd_solver="full")
    pca.fit(X_c.T)                  # (N_time, N_neurons) — neurons as variables
    return pca.components_.T        # (N_neurons, n_pca)


def determine_n_pca(X: np.ndarray, cfg: dict) -> int:
    """
    Determine number of significant PCs via permutation test (MATLAB A3).

    Each neuron's time series is independently shuffled n_perm times.
    PC i is retained when:
      (a) latent_real[i] > mean(null[:, 0])             — exceeds top null mode
      (b) ttest_1samp(null[:, 0], popmean=latent_real[i]) p < 0.05/n_PCs   — Bonferroni

    This is a faithful translation of the MATLAB one-sample t-test formulation
    (`ttest(Null_Sigma(:,1), Latent(i))`), which uses the rank-0 null distribution
    as the common reference, making the criterion more conservative for higher PCs.

    If cfg['gmm_n_pca'] is set, skip the test and return that value directly.
    """
    if cfg.get("gmm_n_pca") is not None:
        n_pca = int(cfg["gmm_n_pca"])
        print(f"  [GMM] N_PCA fixed by config: {n_pca}")
        return n_pca

    n_perm = cfg.get("gmm_pca_permutations", 100)
    X_c    = _row_center(X)
    n_max  = min(X_c.shape[0] - 1, X_c.shape[1] - 1)

    print(f"  [GMM] Permutation test for N_PCA ({n_perm} shuffles) ...")
    pca_full = PCA(n_components=n_max, svd_solver="full")
    pca_full.fit(X_c.T)
    latent_real = pca_full.explained_variance_      # (n_max,)

    rng = np.random.default_rng(0)
    null_latent = np.zeros((n_perm, n_max))
    for j in range(n_perm):
        Xs = X_c.copy()
        for i in range(Xs.shape[0]):
            Xs[i] = Xs[i, rng.permutation(Xs.shape[1])]
        pca_null = PCA(n_components=n_max, svd_solver="full")
        pca_null.fit(Xs.T)
        null_latent[j] = pca_null.explained_variance_

    null_mean0 = null_latent[:, 0].mean()
    alpha_bonf = 0.05 / n_max
    p_vals     = np.array([
        ttest_1samp(null_latent[:, 0], popmean=latent_real[i]).pvalue
        for i in range(n_max)
    ])
    mask  = (latent_real > null_mean0) & (p_vals < alpha_bonf)
    idx   = np.where(mask)[0]
    n_pca = int(idx.max() + 1) if len(idx) > 0 else 10
    print(f"  [GMM] N_PCA selected: {n_pca}  "
          f"(null_mean0={null_mean0:.4f}, alpha_bonf={alpha_bonf:.2e})")
    return n_pca


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Similarity matrix + σ tuning
# ─────────────────────────────────────────────────────────────────────────────

def _build_similarity(PCA_Data: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian kernel similarity: S_ij = exp(−d_ij / σ)."""
    D = squareform(pdist(PCA_Data, "euclidean"))
    return np.exp(-D / sigma)


def tune_sigma(PCA_Data: np.ndarray, target_mean: float = 0.5,
               bounds: tuple = (1e-5, 50.0)) -> float:
    """
    Binary search for σ such that mean(off-diagonal S) ≈ target_mean.
    Matches the MATLAB paper description (tuned value σ=0.07975 for their data).

    Falls back to the 10th-percentile of inter-neuron distances if the
    root-finding fails (e.g. all neurons are identical).
    """
    n = len(PCA_Data)

    def objective(sig: float) -> float:
        S = _build_similarity(PCA_Data, sig)
        np.fill_diagonal(S, 0.0)
        return S.sum() / (n * (n - 1)) - target_mean

    try:
        sigma = brentq(objective, *bounds, xtol=1e-6, maxiter=300)
    except ValueError:
        D_flat = squareform(pdist(PCA_Data, "euclidean"))
        d_vals = D_flat[D_flat > 0]
        sigma  = float(np.percentile(d_vals, 10)) if len(d_vals) > 0 else 1.0
        print(f"  [GMM] σ auto-tune failed; fallback σ={sigma:.5f}")

    print(f"  [GMM] σ tuned: {sigma:.6f}  (target mean(S)={target_mean})")
    return sigma


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Normalized Laplacian eigenvectors
# ─────────────────────────────────────────────────────────────────────────────

def spectral_embedding(
        PCA_Data:    np.ndarray,
        sigma:       float,
        n_spectral:  int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the normalized Laplacian and return its n_spectral smallest
    non-trivial eigenvectors.

    Matches MATLAB spect_clust.m:
        DinvS = diag(sum(S,2).^(-0.5))
        L     = I − DinvS * S * DinvS
        [V, Λ] = eig(L)               % full decomposition in MATLAB
        Spectral_Data = V(:, 2:K)     % skip trivial ev 0

    For N > ~5 000 a full O(N³) eigendecomposition is impractical;
    we use scipy's sparse Lanczos solver (eigsh, which='SM') to extract
    only the n_spectral+1 smallest eigenvalues in O(N·k) time.

    Returns
    -------
    spectral_data : (N_neurons, n_spectral)
    eigenvalues   : (n_spectral + 1,)   — includes trivial λ₀ ≈ 0
    """
    S = _build_similarity(PCA_Data, sigma)

    # Degree matrix D^{-1/2}
    deg      = S.sum(axis=1)                            # (N,)
    d_inv_sq = np.where(deg > 0, deg ** -0.5, 0.0)

    # Normalized Laplacian (symmetric): L = I − D^{-1/2} S D^{-1/2}
    L_norm = d_inv_sq[:, None] * S * d_inv_sq[None, :]
    L      = np.eye(len(S), dtype=np.float64) - L_norm
    del S, L_norm                                       # free ~N² memory

    # Sparse Krylov eigensolver — only k_req smallest eigenpairs
    k_req      = n_spectral + 1
    vals, vecs = eigsh(L, k=k_req, which="SM", tol=1e-6)
    order      = np.argsort(vals)                       # ensure ascending
    vals, vecs = vals[order], vecs[:, order]            # (N, k_req)

    # Skip the trivial eigenvector (λ₀ ≈ 0, constant vector)
    spectral_data = vecs[:, 1:]                         # (N, n_spectral)

    print(f"  [GMM] Eigenvalues (first {k_req}): "
          f"{np.round(vals, 5).tolist()}")
    return spectral_data, vals


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — GMM with optional BIC sweep
# ─────────────────────────────────────────────────────────────────────────────

def _fit_one_gmm(spectral_data: np.ndarray, k: int,
                 cov_type: str, max_iter: int, n_init: int, seed: int
                 ) -> tuple[GaussianMixture, float]:
    gmm = GaussianMixture(
        n_components    = k,
        covariance_type = cov_type,
        max_iter        = max_iter,
        n_init          = n_init,
        random_state    = seed,
    )
    gmm.fit(spectral_data)
    return gmm, float(gmm.bic(spectral_data))


def select_k_bic(
        spectral_data: np.ndarray,
        cfg:           dict,
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    BIC sweep: fit GMM with n_init=3 per K (fast pass); return best K and curve.
    The final GMM at best K is then refitted with full n_init in fit_gmm().
    """
    k_range  = np.asarray(cfg.get("gmm_k_range", list(range(5, 45))), dtype=int)
    cov_type = cfg.get("gmm_covariance_type", "diag")
    max_iter = cfg.get("gmm_max_iter", 1000)
    print(f"  [GMM] BIC sweep K={k_range[0]}..{k_range[-1]} (n_init=3 per K) ...")

    results = Parallel(n_jobs=-1)(
        delayed(_fit_one_gmm)(spectral_data, int(k), cov_type, max_iter, 3, int(k))
        for k in k_range
    )
    bic_vals = np.array([b for _, b in results])
    best_k   = int(k_range[np.argmin(bic_vals)])
    print(f"  [GMM] BIC-optimal K = {best_k}")
    return best_k, k_range, bic_vals


def fit_gmm(spectral_data: np.ndarray, k: int, cfg: dict) -> GaussianMixture:
    """Fit GMM at the chosen K with full n_init replicates."""
    cov_type = cfg.get("gmm_covariance_type", "diag")
    max_iter = cfg.get("gmm_max_iter", 1000)
    n_init   = cfg.get("gmm_n_init", 50)
    print(f"  [GMM] Fitting GMM: K={k}, covariance={cov_type}, n_init={n_init} ...")
    gmm = GaussianMixture(
        n_components    = k,
        covariance_type = cov_type,
        max_iter        = max_iter,
        n_init          = n_init,
        random_state    = 42,
    )
    gmm.fit(spectral_data)
    print(f"  [GMM] Converged={gmm.converged_}  "
          f"BIC={gmm.bic(spectral_data):.1f}  "
          f"log-likelihood={gmm.lower_bound_:.4f}")
    return gmm


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def fit_spectral_gmm(
        X:       np.ndarray,
        cfg:     dict,
        out_dir: Optional[Path] = None,
) -> dict:
    """
    Run the full SE-GMM pipeline on the pre-normalized PSTH matrix X.

    Parameters
    ----------
    X       : (N_neurons, N_features)  pre-normalized concatenated PSTHs
    cfg     : config dict (GMM keys merged with GMM_CFG_DEFAULTS by caller)
    out_dir : if given, save gmm_results.npz here

    Returns
    -------
    dict with keys
        gmm_labels    (N,) int   — hard cluster assignments (argmax of posterior)
        gmm_probs     (N, K)     — soft posterior probabilities
        gmm_model     GaussianMixture
        k             int        — number of clusters used
        n_pca         int
        sigma         float
        spectral_data (N, n_spectral)
        eigenvalues   (n_spectral+1,)
        bic_curve     (k_range, bic_vals) or None
    """
    cfg = {**GMM_CFG_DEFAULTS, **cfg}

    # 1 — PCA
    n_pca    = determine_n_pca(X, cfg)
    X_c      = _row_center(X)
    PCA_Data = _pca_coords(X_c, n_pca)
    print(f"  [GMM] PCA_Data: {PCA_Data.shape}")

    # 2 — Sigma
    sigma = cfg.get("gmm_sigma") or tune_sigma(PCA_Data)

    # 3 — Spectral embedding
    n_spectral            = int(cfg.get("gmm_n_spectral", 13))
    spectral_data, eigvals = spectral_embedding(PCA_Data, sigma, n_spectral)
    print(f"  [GMM] Spectral data: {spectral_data.shape}")

    # 4 — GMM
    bic_curve = None
    if cfg.get("gmm_k") is None:
        best_k, k_range, bic_vals = select_k_bic(spectral_data, cfg)
        bic_curve = (k_range, bic_vals)
    else:
        best_k = int(cfg["gmm_k"])

    gmm        = fit_gmm(spectral_data, best_k, cfg)
    gmm_labels = gmm.predict(spectral_data)
    gmm_probs  = gmm.predict_proba(spectral_data)

    result = dict(
        gmm_labels    = gmm_labels,
        gmm_probs     = gmm_probs,
        gmm_model     = gmm,
        k             = best_k,
        n_pca         = n_pca,
        sigma         = sigma,
        spectral_data = spectral_data,
        eigenvalues   = eigvals,
        bic_curve     = bic_curve,
    )

    if out_dir is not None:
        save_gmm_npz(Path(out_dir) / "gmm_results.npz", result)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — label conversion, save
# ─────────────────────────────────────────────────────────────────────────────

def gmm_labels_to_isort_boundaries(
        gmm_labels: np.ndarray, k: int
) -> tuple[np.ndarray, list[int]]:
    """
    Convert integer GMM cluster labels to the (isort, boundaries) format
    expected by fig6_cluster_profiles and fig5b_kmeans_matrix.

    Neurons are sorted by label (cluster 0 first, then 1, …), matching
    the order used in fig5b_kmeans_matrix's internal logic.
    """
    isort      = np.argsort(gmm_labels, kind="stable")
    boundaries = np.cumsum(
        [(gmm_labels == ki).sum() for ki in range(k)]
    )[:-1].tolist()
    return isort, boundaries


def save_gmm_npz(path: Path, result: dict) -> None:
    payload = dict(
        gmm_labels    = result["gmm_labels"],
        gmm_probs     = result["gmm_probs"],
        spectral_data = result["spectral_data"],
        eigenvalues   = result["eigenvalues"],
        n_pca         = np.array(result["n_pca"]),
        sigma         = np.array(result["sigma"]),
        k             = np.array(result["k"]),
    )
    if result["bic_curve"] is not None:
        payload["bic_k_range"] = result["bic_curve"][0]
        payload["bic_values"]  = result["bic_curve"][1]
    np.savez_compressed(path, **payload)
    print(f"  Saved -> {Path(path).name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figures — GMM-specific diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def figGMM_pca_variance(X: np.ndarray, n_pca: int, out_dir: Path) -> None:
    """Scree + cumulative variance for the GMM-internal PCA (mirrors fig7)."""
    X_c   = _row_center(X)
    n_max = min(n_pca * 4, X_c.shape[0] - 1, X_c.shape[1] - 1)
    pca   = PCA(n_components=n_max, svd_solver="full").fit(X_c.T)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.plot(pca.explained_variance_ratio_ * 100, "o-", ms=3, lw=1, color="steelblue")
    ax1.axvline(n_pca - 1, color="crimson", ls="--", lw=1.2, label=f"N_PCA={n_pca}")
    ax1.set_xlabel("PC index"); ax1.set_ylabel("Variance explained (%)")
    ax1.set_title("Scree plot (GMM PCA)"); ax1.legend(fontsize=8)

    ax2.plot(cumvar, "o-", ms=3, lw=1, color="steelblue")
    ax2.axvline(n_pca - 1, color="crimson", ls="--", lw=1.2, label=f"N_PCA={n_pca}")
    ax2.axhline(80, color="grey", ls=":", lw=1); ax2.axhline(95, color="grey", ls=":", lw=1)
    ax2.set_xlabel("PC index"); ax2.set_ylabel("Cumulative variance (%)")
    ax2.set_title("Cumulative variance (GMM PCA)"); ax2.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir / "figGMM_pca_variance", dpi=300)


def figGMM_bic_curve(
        k_range: np.ndarray, bic_vals: np.ndarray, best_k: int, out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_range, bic_vals, "o-", ms=4, lw=1.5, color="steelblue")
    ax.axvline(best_k, color="crimson", ls="--", lw=1.5, label=f"Best K = {best_k}")
    ax.set_xlabel("Number of clusters K")
    ax.set_ylabel("BIC")
    ax.set_title("GMM BIC curve")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir / "figGMM_bic_curve", dpi=300)


def figGMM_eigenvalues(
        eigenvalues: np.ndarray, n_spectral: int, out_dir: Path
) -> None:
    """Normalized Laplacian eigenspectrum with cut-off marker."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eigenvalues, "o-", ms=4, lw=1.5, color="steelblue")
    ax.axvline(n_spectral, color="crimson", ls="--", lw=1.2,
               label=f"{n_spectral} eigenvectors kept")
    ax.set_xlabel("Eigenvector index (0 = trivial)")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Normalized Laplacian — eigenspectrum")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir / "figGMM_eigenvalues", dpi=300)


def figGMM_spectral_scatter(
        spectral_data: np.ndarray, gmm_labels: np.ndarray, k: int, out_dir: Path
) -> None:
    """3-D scatter of the first 3 spectral dimensions, colored by GMM cluster."""
    if spectral_data.shape[1] < 3:
        print("  [GMM] figGMM_spectral_scatter: need ≥3 spectral dims, skipping")
        return
    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")
    sc  = ax.scatter(
        spectral_data[:, 0], spectral_data[:, 1], spectral_data[:, 2],
        c=gmm_labels, cmap="turbo", s=3, alpha=0.5, linewidths=0,
    )
    fig.colorbar(sc, ax=ax, shrink=0.4, label=f"GMM cluster (K={k})")
    ax.set_xlabel("Spectral 1"); ax.set_ylabel("Spectral 2"); ax.set_zlabel("Spectral 3")
    ax.set_title("Spectral embedding — GMM clusters")
    fig.tight_layout()
    _save(fig, out_dir / "figGMM_spectral_scatter", dpi=300)


def figGMM_assignment_entropy(gmm_probs: np.ndarray, out_dir: Path) -> None:
    """
    Histogram of per-neuron posterior entropy.
    Exploits GMM's soft assignments — high entropy = uncertain/mixed neuron.
    """
    from scipy.stats import entropy
    H   = np.array([entropy(p) for p in gmm_probs])
    med = float(np.median(H))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(H, bins=60, color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(med, color="crimson", ls="--", lw=1.5, label=f"Median = {med:.3f} nats")
    ax.set_xlabel("Posterior entropy (nats)")
    ax.set_ylabel("Neuron count")
    ax.set_title("GMM assignment uncertainty\n"
                 "(low entropy = confident assignment, high = mixed)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir / "figGMM_assignment_entropy", dpi=300)