"""
gmm_utils.py — Spectral Embedding + GMM clustering (Standard and Bayesian)

Pipeline
--------
1.  (Optional) Determine N_PCA via permutation test
2.  Row-centre X; project neurons into PCA space  ->  PCA_data  (N x N_PCA)
3.  Build Gaussian similarity matrix S; tune sigma so mean(S_off_diag) ~ 0.5
4.  Normalised Laplacian  L = I - D^{-1/2} S D^{-1/2}
5.  Sparse Lanczos -> keep eigenvectors 1..n_spectral  (skip trivial ev 0)
6a. Standard GMM:  BIC sweep -> pick K -> fit GaussianMixture
6b. Bayesian GMM:  fit BayesianGaussianMixture (DP-GMM) with K_max=100

Cross-validation (Nystrom out-of-sample extension)
---------------------------------------------------
The spectral embedding is non-parametric, so we cannot directly project
held-out data into the training manifold.  The Nystrom approximation
reconstructs the projection analytically:

    v~_i(x_new) = (1/lambda~_i) sum_j K~(x_new, x_j^train) * v_i[j]

where K~ is the degree-normalised Gaussian kernel and
lambda~_i = 1 - lambda_i(L) is the i-th eigenvalue of D^{-1/2}SD^{-1/2}.

PCA convention
--------------
Neurons are treated as *samples* and time bins as *features*
(pca.fit(X_c) with X_c: N_neurons x N_time).  This enables trivial
out-of-sample projection: pca.transform(X_c_new).
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse.linalg import eigsh
from scipy.optimize import brentq
from scipy.stats import ttest_1samp
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from joblib import Parallel, delayed

from rastermap_psth.rastermap_utils import _save


# ── Config defaults ───────────────────────────────────────────────────────────
GMM_CFG_DEFAULTS: dict = dict(
    do_gmm                       = False,
    # PCA
    gmm_n_pca                    = None,
    gmm_pca_permutations         = 100,
    # Spectral embedding
    gmm_sigma                    = None,
    gmm_n_spectral               = 13,
    # Standard GMM
    gmm_do_standard              = True,
    gmm_k                        = 100,
    gmm_k_range                  = list(range(30, 200)),
    gmm_n_init                   = 50,
    gmm_covariance_type          = "diag",
    gmm_max_iter                 = 1000,
    # Bayesian (DP-GMM)
    gmm_do_bayesian              = True,
    gmm_bayesian_n_components    = 200,
    gmm_bayesian_concentration   = 100.0,
    gmm_bayesian_covariance_type = "diag",
    gmm_bayesian_max_iter        = 2000,
    gmm_bayesian_n_init          = 3,
    gmm_bayesian_weight_threshold= None,
)


# ── Step 1: PCA ───────────────────────────────────────────────────────────────

def _row_center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=1, keepdims=True)


def _fit_pca(X_c: np.ndarray, n_pca: int) -> tuple:
    """Fit PCA with neurons as samples (N_neurons x N_time).
    Returns (PCA_data, pca_obj).  pca.transform(X_c_new) gives out-of-sample coords.
    """
    pca = PCA(n_components=n_pca, svd_solver="full")
    pca.fit(X_c)
    return pca.transform(X_c), pca   # (N_neurons, n_pca)


def determine_n_pca(X: np.ndarray, cfg: dict) -> int:
    if cfg.get("gmm_n_pca") is not None:
        n = int(cfg["gmm_n_pca"])
        print(f"  [GMM] N_PCA fixed: {n}")
        return n
    n_perm = cfg.get("gmm_pca_permutations", 100)
    X_c    = _row_center(X)
    n_max  = min(X_c.shape[0] - 1, X_c.shape[1] - 1)
    print(f"  [GMM] Permutation test for N_PCA ({n_perm} shuffles) ...")
    latent_real = PCA(n_components=n_max, svd_solver="full").fit(X_c).explained_variance_
    rng = np.random.default_rng(0)
    null_latent = np.zeros((n_perm, n_max))
    for j in range(n_perm):
        Xs = X_c.copy()
        for i in range(Xs.shape[0]):
            Xs[i] = Xs[i, rng.permutation(Xs.shape[1])]
        null_latent[j] = PCA(n_components=n_max, svd_solver="full").fit(Xs).explained_variance_
    null_mean0 = null_latent[:, 0].mean()
    alpha_bonf = 0.05 / n_max
    p_vals = np.array([ttest_1samp(null_latent[:, 0], popmean=latent_real[i]).pvalue
                       for i in range(n_max)])
    mask  = (latent_real > null_mean0) & (p_vals < alpha_bonf)
    idx   = np.where(mask)[0]
    n_pca = int(idx.max() + 1) if len(idx) > 0 else 10
    print(f"  [GMM] N_PCA selected: {n_pca}")
    return n_pca


# ── Step 2: Similarity + sigma ────────────────────────────────────────────────

def _build_similarity(PCA_data: np.ndarray, sigma: float) -> np.ndarray:
    D = squareform(pdist(PCA_data, "euclidean"))
    return np.exp(-D / sigma)

# TODO: sensitiviity analysis of sigma for SE

def tune_sigma(PCA_data: np.ndarray, target_mean: float = 0.5,
               bounds: tuple = (1e-5, 1e4)) -> float:
    n = len(PCA_data)
    def obj(sig):
        S = _build_similarity(PCA_data, sig)
        np.fill_diagonal(S, 0.0)
        return S.sum() / (n * (n - 1)) - target_mean
    try:
        sigma = brentq(obj, *bounds, xtol=1e-6, maxiter=300)
    except ValueError:
        d_vals = squareform(pdist(PCA_data, "euclidean"))
        sigma  = float(np.percentile(d_vals[d_vals > 0], 10)) if d_vals.any() else 1.0
        print(f"  [GMM] sigma auto-tune failed; fallback sigma={sigma:.5f}")
    print(f"  [GMM] sigma={sigma:.6f}  (target mean(S)={target_mean})")
    return sigma


# ── Step 3: Normalised Laplacian eigenvectors ─────────────────────────────────

def spectral_embedding(PCA_data: np.ndarray, sigma: float,
                       n_spectral: int) -> tuple:
    """Returns (spectral_data, eigenvalues, eigvecs_full, d).

    spectral_data : (N, n_spectral)     embedding (trivial ev skipped)
    eigenvalues   : (n_spectral+1,)     eigenvalues of L (includes trivial)
    eigvecs_full  : (N, n_spectral+1)   all eigvecs incl. trivial (for Nystrom)
    d             : (N,)                degree vector (for Nystrom normalisation)
    """
    S   = _build_similarity(PCA_data, sigma)
    deg = S.sum(axis=1)
    d_inv_sq = np.where(deg > 0, deg ** -0.5, 0.0)
    L_norm = d_inv_sq[:, None] * S * d_inv_sq[None, :]
    L      = np.eye(len(S), dtype=np.float64) - L_norm
    del S, L_norm
    k_req      = n_spectral + 1
    vals, vecs = eigsh(L, k=k_req, which="SM", tol=1e-6)
    order      = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    print(f"  [GMM] Eigenvalues (L): {np.round(vals, 5).tolist()}")
    return vecs[:, 1:], vals, vecs, deg   # spectral, eigenvalues, eigvecs_full, d


# ── Step 3b: Nystrom out-of-sample extension ──────────────────────────────────

def nystrom_extension(PCA_data_new: np.ndarray, PCA_data_train: np.ndarray,
                      sigma: float, d_train: np.ndarray,
                      eigvecs_full: np.ndarray, eigenvalues_L: np.ndarray
                      ) -> np.ndarray:
    """Project new neurons into training spectral space via Nystrom approximation.

    v~_i(x_new) = (1/lambda~_i) sum_j K~(x_new, x_j) * v_i[j]

    where K~ is degree-normalised and lambda~_i = 1 - lambda_i(L).

    Returns (N_new, n_spectral) -- trivial eigenvector (index 0) is skipped.
    """
    K_cross = np.exp(-cdist(PCA_data_new, PCA_data_train, "euclidean") / sigma)
    d_new   = K_cross.sum(axis=1)
    d_new_inv_sq   = np.where(d_new   > 0, d_new   ** -0.5, 0.0)
    d_train_inv_sq = np.where(d_train > 0, d_train ** -0.5, 0.0)
    K_norm  = d_new_inv_sq[:, None] * K_cross * d_train_inv_sq[None, :]
    lambda_norm = np.clip(1.0 - eigenvalues_L, 1e-10, None)
    spectral_all = (K_norm @ eigvecs_full) / lambda_norm[None, :]
    return spectral_all[:, 1:]    # skip trivial eigenvector


# ── Step 4a: Standard GMM + BIC ───────────────────────────────────────────────

def _fit_one_gmm(spectral_data, k, cov_type, max_iter, n_init, seed):
    gmm = GaussianMixture(n_components=k, covariance_type=cov_type,
                          max_iter=max_iter, n_init=n_init, random_state=seed)
    gmm.fit(spectral_data)
    return gmm, float(gmm.bic(spectral_data))


def select_k_bic(spectral_data: np.ndarray, cfg: dict) -> tuple:
    k_range  = np.asarray(cfg.get("gmm_k_range", list(range(5, 45))), int)
    cov_type = cfg.get("gmm_covariance_type", "diag")
    max_iter = cfg.get("gmm_max_iter", 1000)
    print(f"  [GMM] BIC sweep K={k_range[0]}..{k_range[-1]} ...")
    results  = Parallel(n_jobs=-1)(
        delayed(_fit_one_gmm)(spectral_data, int(k), cov_type, max_iter, 3, int(k))
        for k in k_range)
    bic_vals = np.array([b for _, b in results])
    best_k   = int(k_range[np.argmin(bic_vals)])
    print(f"  [GMM] BIC-optimal K={best_k}")
    return best_k, k_range, bic_vals


def fit_gmm(spectral_data: np.ndarray, k: int, cfg: dict) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components    = k,
        covariance_type = cfg.get("gmm_covariance_type", "diag"),
        max_iter        = cfg.get("gmm_max_iter", 1000),
        n_init          = cfg.get("gmm_n_init", 50),
        random_state    = 42,
    )
    gmm.fit(spectral_data)
    print(f"  [GMM] Converged={gmm.converged_}  BIC={gmm.bic(spectral_data):.1f}")
    return gmm


# ── Step 4b: Bayesian GMM (DP-GMM) ───────────────────────────────────────────

def fit_bayesian_gmm(spectral_data: np.ndarray, cfg: dict) -> BayesianGaussianMixture:
    """Fit DP-GMM favouring many active clusters (~100).

    concentration=10, N=10000 -> E[K_active] ~ 10*log(10000) ~ 92.
    """
    K_max = cfg.get("gmm_bayesian_n_components", 100)
    conc  = cfg.get("gmm_bayesian_concentration", 100.0)
    bgmm  = BayesianGaussianMixture(
        n_components                    = K_max,
        covariance_type                 = cfg.get("gmm_bayesian_covariance_type", "diag"),
        weight_concentration_prior_type = "dirichlet_process",
        weight_concentration_prior      = conc,
        max_iter                        = cfg.get("gmm_bayesian_max_iter", 2000),
        n_init                          = cfg.get("gmm_bayesian_n_init", 3),
        random_state                    = 42,
    )
    bgmm.fit(spectral_data)
    thr      = _bayesian_threshold(K_max, cfg)
    n_active = (bgmm.weights_ > thr).sum()
    print(f"  [GMM-Bayes] n_active={n_active}/{K_max}  alpha={conc}  (thr={thr:.5f})")
    return bgmm


def _bayesian_threshold(K_max: int, cfg: dict) -> float:
    t = cfg.get("gmm_bayesian_weight_threshold")
    return t if t is not None else 1.0 / (10 * K_max)


def bayesian_active_labels(bgmm: BayesianGaussianMixture, spectral_data: np.ndarray,
                            cfg: dict) -> tuple:
    """Return (remapped_labels, k_active).

    Neurons assigned to pruned components are reassigned to the nearest
    active component centroid.
    """
    K_max    = bgmm.n_components
    thr      = _bayesian_threshold(K_max, cfg)
    active   = np.where(bgmm.weights_ > thr)[0]
    raw      = bgmm.predict(spectral_data)
    means    = bgmm.means_
    label_map = {}
    for k in range(K_max):
        if k in active:
            label_map[k] = int(np.searchsorted(active, k))
        else:
            dists        = np.linalg.norm(means[k] - means[active], axis=1)
            label_map[k] = int(np.argmin(dists))
    remapped = np.array([label_map[l] for l in raw], dtype=int)
    return remapped, len(active)


# ── Convenience wrapper (global only, backward compat) ───────────────────────

def fit_spectral_gmm(X: np.ndarray, cfg: dict,
                     out_dir: Optional[Path] = None) -> dict:
    cfg = {**GMM_CFG_DEFAULTS, **cfg}
    n_pca              = determine_n_pca(X, cfg)
    if X.shape[0]<2000:
        n_pca=n_pca
    else:
        print('Info: setting number of PCs (GMM) too much larger than what determine_n_pca does...')
        n_pca = 2000

    X_c                = _row_center(X)
    PCA_data, _pca_obj = _fit_pca(X_c, n_pca)
    sigma              = cfg.get("gmm_sigma") or tune_sigma(PCA_data)
    n_spectral         = int(cfg.get("gmm_n_spectral", 13))
    spectral, eigvals, eigvecs_full, d = spectral_embedding(PCA_data, sigma, n_spectral)
    bic_curve = None
    if cfg.get("gmm_k") is None:
        best_k, k_range, bic_vals = select_k_bic(spectral, cfg)
        bic_curve = (k_range, bic_vals)
    else:
        best_k = int(cfg["gmm_k"])
    best_k = int(cfg["gmm_k"])
    gmm        = fit_gmm(spectral, best_k, cfg)
    gmm_labels = gmm.predict(spectral)
    gmm_probs  = gmm.predict_proba(spectral)
    return dict(gmm_labels=gmm_labels, gmm_probs=gmm_probs, gmm_model=gmm,
                k=best_k, n_pca=n_pca, sigma=sigma,
                spectral_data=spectral, eigenvalues=eigvals, bic_curve=bic_curve)


# ── Helpers ───────────────────────────────────────────────────────────────────

def gmm_labels_to_isort_boundaries(labels: np.ndarray, k: int) -> tuple:
    isort      = np.argsort(labels, kind="stable")
    boundaries = np.cumsum([(labels == ki).sum() for ki in range(k)])[:-1].tolist()
    return isort, boundaries


def save_gmm_npz(path: Path, result: dict) -> None:
    payload = dict(
        gmm_labels    = result["gmm_labels"],
        gmm_probs     = result.get("gmm_probs", np.array([])),
        spectral_data = result["spectral_data"],
        eigenvalues   = result["eigenvalues"],
        n_pca         = np.array(result["n_pca"]),
        sigma         = np.array(result["sigma"]),
        k             = np.array(result["k"]),
    )
    if result.get("bic_curve") is not None:
        payload["bic_k_range"] = result["bic_curve"][0]
        payload["bic_values"]  = result["bic_curve"][1]
    np.savez_compressed(path, **payload)
    print(f"  Saved -> {Path(path).name}")


# ── Figures ───────────────────────────────────────────────────────────────────

def _pfx(prefix: str) -> str:
    return f"{prefix}_" if prefix else ""


def figGMM_pca_variance(X, n_pca, out_dir, prefix=""):
    X_c   = _row_center(X)
    n_show = min(n_pca * 4, X_c.shape[0] - 1, X_c.shape[1] - 1)
    pca   = PCA(n_components=n_show, svd_solver="full").fit(X_c)
    cum   = np.cumsum(pca.explained_variance_ratio_) * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.plot(pca.explained_variance_ratio_ * 100, "o-", ms=3, lw=1, color="steelblue")
    ax1.axvline(n_pca - 1, color="crimson", ls="--", lw=1.2, label=f"N_PCA={n_pca}")
    ax1.set_xlabel("PC"); ax1.set_ylabel("Variance (%)"); ax1.set_title("Scree"); ax1.legend(fontsize=8)
    ax2.plot(cum, "o-", ms=3, lw=1, color="steelblue")
    ax2.axvline(n_pca - 1, color="crimson", ls="--", lw=1.2)
    for t in (80, 95): ax2.axhline(t, color="grey", ls=":", lw=0.8)
    ax2.set_xlabel("PC"); ax2.set_ylabel("Cumulative variance (%)"); ax2.set_title("Cumulative")
    fig.tight_layout()
    _save(fig, out_dir / f"{_pfx(prefix)}figGMM_pca_variance", dpi=300)


def figGMM_bic_curve(k_range, bic_vals, best_k, out_dir, prefix=""):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_range, bic_vals, "o-", ms=4, lw=1.5, color="steelblue")
    ax.axvline(best_k, color="crimson", ls="--", lw=1.5, label=f"Best K={best_k}")
    ax.set_xlabel("K"); ax.set_ylabel("BIC"); ax.set_title("GMM BIC curve")
    ax.legend(fontsize=9); fig.tight_layout()
    _save(fig, out_dir / f"{_pfx(prefix)}figGMM_bic_curve", dpi=300)


def figGMM_eigenvalues(eigenvalues, n_spectral, out_dir, prefix=""):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eigenvalues, "o-", ms=4, lw=1.5, color="steelblue")
    ax.axvline(n_spectral, color="crimson", ls="--", lw=1.2, label=f"{n_spectral} kept")
    ax.set_xlabel("Eigenvector index (0=trivial)"); ax.set_ylabel("Eigenvalue (L)")
    ax.set_title("Normalised Laplacian eigenspectrum"); ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir / f"{_pfx(prefix)}figGMM_eigenvalues", dpi=300)


def figGMM_spectral_scatter(spectral_data, labels, k, out_dir, prefix=""):
    if spectral_data.shape[1] < 3: return
    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")
    sc  = ax.scatter(spectral_data[:, 0], spectral_data[:, 1], spectral_data[:, 2],
                     c=labels, cmap="turbo", s=3, alpha=0.5, linewidths=0)
    fig.colorbar(sc, ax=ax, shrink=0.4, label=f"Cluster (K={k})")
    ax.set_xlabel("Spectral 1"); ax.set_ylabel("Spectral 2"); ax.set_zlabel("Spectral 3")
    ax.set_title("Spectral embedding — cluster labels"); fig.tight_layout()
    _save(fig, out_dir / f"{_pfx(prefix)}figGMM_spectral_scatter", dpi=300)


def figGMM_assignment_entropy(gmm_probs, out_dir, prefix=""):
    from scipy.stats import entropy
    H   = np.array([entropy(p) for p in gmm_probs])
    med = float(np.median(H))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(H, bins=60, color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(med, color="crimson", ls="--", lw=1.5, label=f"Median={med:.3f} nats")
    ax.set_xlabel("Posterior entropy (nats)"); ax.set_ylabel("Neuron count")
    ax.set_title("GMM assignment uncertainty  (low=confident, high=mixed)")
    ax.legend(fontsize=9); fig.tight_layout()
    _save(fig, out_dir / f"{_pfx(prefix)}figGMM_assignment_entropy", dpi=300)


def figGMM_active_components(bgmm, K_max, cfg, out_dir, prefix=""):
    weights  = bgmm.weights_
    thr      = _bayesian_threshold(K_max, cfg)
    active   = weights > thr
    fig, ax  = plt.subplots(figsize=(min(20, K_max * 0.18 + 2), 4))
    ax.bar(np.where(~active)[0], weights[~active], color="lightgrey", label="Pruned")
    ax.bar(np.where(active)[0],  weights[active],  color="steelblue",
           label=f"Active (n={active.sum()})")
    ax.axhline(thr, color="crimson", ls="--", lw=1.2, label=f"Threshold={thr:.4f}")
    ax.set_xlabel("Component index"); ax.set_ylabel("Mixture weight")
    ax.set_title(f"Bayesian GMM — weights  (K_max={K_max})")
    ax.legend(fontsize=8); fig.tight_layout()
    _save(fig, out_dir / f"{_pfx(prefix)}figGMM_bayesian_active_components", dpi=200)


def figGMM_nystrom_sanity(spectral_train, spectral_new, out_dir, prefix=""):
    """Overlay distributions of training vs Nystrom-projected coordinates."""
    n_dims = min(4, spectral_train.shape[1])
    fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 3.5))
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.hist(spectral_train[:, i], bins=40, alpha=0.6, color="steelblue",
                density=True, label="Train (odd)")
        ax.hist(spectral_new[:, i],   bins=40, alpha=0.6, color="tomato",
                density=True, label="Nystrom (even)")
        ax.set_title(f"Dim {i+1}"); ax.set_xlabel("Coordinate")
        if i == 0: ax.legend(fontsize=7)
    fig.suptitle("Nystrom sanity: train vs projected", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / f"{_pfx(prefix)}figGMM_nystrom_sanity", dpi=200)