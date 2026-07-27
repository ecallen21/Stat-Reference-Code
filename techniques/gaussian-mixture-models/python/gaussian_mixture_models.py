"""Gaussian Mixture Models via the EM algorithm (Reference §9.12).

Model: each observation is drawn from one of K multivariate Gaussians with
unknown mixing proportions pi_k, means mu_k, and covariances Sigma_k:

    p(x) = sum_k pi_k * N(x | mu_k, Sigma_k)

EM iterates:
    E-step:  gamma_ik  =  pi_k N(x_i | mu_k, Sigma_k) / sum_j pi_j N(x_i | mu_j, Sigma_j)
             (posterior responsibility of component k for point i)
    M-step:  N_k       =  sum_i gamma_ik
             pi_k      =  N_k / N
             mu_k      =  sum_i gamma_ik x_i / N_k
             Sigma_k   =  sum_i gamma_ik (x_i - mu_k)(x_i - mu_k)' / N_k

Guaranteed monotonic increase in log-likelihood. Multiple restarts help avoid
local optima.

Model selection: BIC = -2 log L + p log N  (p = free parameters); pick K that
minimizes BIC. Also common: AIC. Since GMM is a density model, cluster
assignments come from the argmax of the responsibilities.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _log_gaussian(X, mu, Sigma):
    """Log N(x_i | mu, Sigma) for each row of X. Uses Cholesky for stability."""
    p = X.shape[1]
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        # regularize a bit
        L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(p))
    diff = X - mu
    z = np.linalg.solve(L, diff.T).T
    quad = (z ** 2).sum(axis=1)
    log_det = 2.0 * np.log(np.diag(L)).sum()
    return -0.5 * (p * math.log(2 * math.pi) + log_det + quad)


def gmm_em(X, K: int, max_iter: int = 200, tol: float = 1e-6,
           n_restarts: int = 5, seed: int = 0) -> dict:
    """EM for K-component GMM with multiple random restarts."""
    X = np.asarray(X, dtype=float)
    N, p = X.shape
    rng = np.random.default_rng(seed)
    best = None
    all_ll = []
    for restart in range(n_restarts):
        # Initialize means from random data points; covariances from data covariance
        idx = rng.choice(N, size=K, replace=False)
        mus = X[idx].copy()
        cov0 = np.cov(X, rowvar=False, ddof=1)
        Sigmas = np.array([cov0 for _ in range(K)])
        pis = np.full(K, 1.0 / K)
        ll_prev = -np.inf
        for _ in range(max_iter):
            # E-step
            log_probs = np.column_stack([_log_gaussian(X, mus[k], Sigmas[k]) + math.log(pis[k])
                                          for k in range(K)])
            # log-sum-exp for stability
            mx = log_probs.max(axis=1, keepdims=True)
            log_sum = mx.squeeze() + np.log(np.exp(log_probs - mx).sum(axis=1))
            gamma = np.exp(log_probs - log_sum[:, None])
            ll = float(log_sum.sum())
            if abs(ll - ll_prev) < tol:
                break
            ll_prev = ll
            # M-step
            Nk = gamma.sum(axis=0)
            pis = Nk / N
            for k in range(K):
                mus[k] = (gamma[:, k:k + 1] * X).sum(axis=0) / max(Nk[k], 1e-12)
                diff = X - mus[k]
                Sigmas[k] = (gamma[:, k:k + 1] * diff).T @ diff / max(Nk[k], 1e-12)
                Sigmas[k] += 1e-6 * np.eye(p)          # ridge for numerical stability
        all_ll.append(ll)
        if best is None or ll > best["log_lik"]:
            labels = np.argmax(gamma, axis=1)
            best = {"pi": pis.copy(), "mu": mus.copy(), "Sigma": Sigmas.copy(),
                     "log_lik": ll, "labels": labels.tolist(),
                     "responsibilities": gamma.copy(),
                     "restart": restart}
    # BIC: -2 log L + p_free log N,  with p_free = K * (p + p(p+1)/2) + (K-1)
    p_free = K * (p + p * (p + 1) / 2) + (K - 1)
    bic = -2 * best["log_lik"] + p_free * math.log(N)
    aic = -2 * best["log_lik"] + 2 * p_free
    return {"pi": best["pi"].tolist(),
            "mu": best["mu"].tolist(),
            "Sigma": best["Sigma"].tolist(),
            "log_lik": best["log_lik"],
            "labels": best["labels"],
            "responsibilities_head": best["responsibilities"][:5].tolist(),
            "AIC": aic, "BIC": bic,
            "K": K, "N": N, "p": p,
            "best_restart_index": best["restart"],
            "all_restart_log_liks": all_ll,
            "method": "GMM via EM (multiple restarts)"}


def bic_select_k(X, k_grid=range(1, 7), seed: int = 0) -> dict:
    """Fit GMM for a range of K and report BIC/AIC per K."""
    results = []
    best_k = None
    for K in k_grid:
        f = gmm_em(X, K, seed=seed)
        results.append({"K": K, "log_lik": f["log_lik"], "AIC": f["AIC"], "BIC": f["BIC"]})
        if best_k is None or f["BIC"] < results[best_k[0]]["BIC"]:
            best_k = (len(results) - 1, K)
    return {"per_k": results,
            "best_K_by_BIC": best_k[1] if best_k else None}


def library_versions(X, K):
    from sklearn.mixture import GaussianMixture
    m = GaussianMixture(n_components=K, n_init=5, random_state=0).fit(X)
    return {"sklearn log_lik (mean)": float(m.score(X)),
            "sklearn total log_lik": float(m.score(X) * X.shape[0]),
            "sklearn BIC": float(m.bic(X)),
            "sklearn AIC": float(m.aic(X)),
            "sklearn weights": m.weights_.tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(71)
    # Two well-separated clusters + one overlapping
    n_per = 150
    Sigma1 = np.array([[1.0, 0.2], [0.2, 0.6]])
    Sigma2 = np.array([[0.7, -0.3], [-0.3, 1.2]])
    Sigma3 = np.array([[0.5, 0.0], [0.0, 0.5]])
    X1 = rng.multivariate_normal([0, 0], Sigma1, n_per)
    X2 = rng.multivariate_normal([5, 5], Sigma2, n_per)
    X3 = rng.multivariate_normal([1, 3], Sigma3, n_per)
    X = np.vstack([X1, X2, X3])

    print("=== GMM (K=3) ===")
    fit = gmm_em(X, K=3, n_restarts=5)
    print(f"  log-lik: {fit['log_lik']:.4f}")
    print(f"  BIC:     {fit['BIC']:.4f}")
    print(f"  pi:      {[f'{p:.3f}' for p in fit['pi']]}")
    for k, mu in enumerate(fit["mu"]):
        print(f"  mu_{k}:    {[f'{v:+.3f}' for v in mu]}")

    print("\n=== BIC selection over K = 1..5 ===")
    sel = bic_select_k(X, range(1, 6))
    for r in sel["per_k"]:
        print(f"  K={r['K']}: log-lik={r['log_lik']:.2f}, BIC={r['BIC']:.2f}, AIC={r['AIC']:.2f}")
    print(f"  best K by BIC: {sel['best_K_by_BIC']}")

    print("\n--- library (sklearn) ---")
    for k, v in library_versions(X, K=3).items():
        print(f"  {k}: {v}")
