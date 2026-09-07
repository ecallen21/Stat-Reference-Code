"""Finite mixture of regressions (Reference Sec 36.3).

DeSarbo-Cron 1988.  Model heterogeneous slopes / intercepts as a
K-component mixture of linear regressions:

  y_i | z = k ~ N(x_i^T beta_k, sigma_k^2)
  z_i        ~ Categorical(pi_1, ..., pi_K)

Estimated by EM: E-step gives posterior class probs; M-step
weighted-OLS + variance updates within each class.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.stats import norm


def em_mixture_regression(X, y, K=2, n_iter=200, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    n, p = X.shape
    # Random initialisation of betas via bootstrap OLS on random subsets
    betas = []
    sigmas = []
    for k in range(K):
        idx = rng.choice(n, size=n // K, replace=False)
        b, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        betas.append(b)
        sigmas.append(float(np.std(y[idx] - X[idx] @ b) + 0.1))
    betas = np.array(betas); sigmas = np.array(sigmas)
    pi = np.full(K, 1 / K)

    ll_prev = -np.inf
    for _ in range(n_iter):
        # E-step
        lik = np.stack([pi[k] * norm.pdf(y, loc=X @ betas[k], scale=sigmas[k])
                         for k in range(K)], axis=1)
        P = lik / (lik.sum(axis=1, keepdims=True) + 1e-300)
        # M-step
        pi = P.mean(axis=0)
        for k in range(K):
            W = np.diag(P[:, k])
            # Weighted LS: (X'WX) beta = X'W y
            A = X.T @ W @ X; b = X.T @ W @ y
            betas[k] = np.linalg.solve(A + 1e-8 * np.eye(p), b)
            resid = y - X @ betas[k]
            sigmas[k] = np.sqrt(max((P[:, k] * resid ** 2).sum() / P[:, k].sum(), 1e-6))
        ll = float(np.log(lik.sum(axis=1) + 1e-300).sum())
        if abs(ll - ll_prev) < tol: break
        ll_prev = ll
    return {"pi": pi, "betas": betas, "sigmas": sigmas, "loglik": ll_prev,
            "posteriors": P}


if __name__ == "__main__":
    print("=== Finite mixture of regressions (EM) ===\n")
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(-1, 1, n)
    # Two-component mixture: opposite slopes
    z = rng.integers(0, 2, n)
    y = np.where(z == 0, 1.0 * x + 0.5, -1.0 * x - 0.5) + rng.normal(0, 0.3, n)
    X = np.column_stack([np.ones(n), x])

    r = em_mixture_regression(X, y, K=2)
    print(f"  Estimated pi   : {np.round(r['pi'], 3)}   (true 0.5 / 0.5)")
    print(f"  Estimated betas per class (intercept, slope):")
    for k, b in enumerate(r["betas"]):
        print(f"    class {k}: intercept={b[0]:+.3f}   slope={b[1]:+.3f}   sigma={r['sigmas'][k]:.3f}")

    # Recover class labels via MAP posterior
    labels = np.argmax(r["posteriors"], axis=1)
    acc = max(float(np.mean(labels == z)), float(np.mean(labels == 1 - z)))
    print(f"  Label recovery accuracy (best alignment): {acc:.3f}\n")

    print("--- library cross-check (R flexmix::flexmix; Python custom + statsmodels) ---")
