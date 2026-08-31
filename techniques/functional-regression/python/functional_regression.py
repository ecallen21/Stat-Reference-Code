"""Functional regression (Reference Sec 31.3).

Scalar-on-function regression:  y_i = alpha + int X_i(t) beta(t) dt + eps.

Two practical fitting routes:
  1. FPC regression: expand X in top-K functional principal components
     and regress y on the FPC scores.
  2. Basis expansion + roughness penalty: beta(t) = sum_j c_j phi_j(t),
     ridge-penalise second-derivative of beta.

Here we implement the FPC-regression route on synthetic curves + linear
scalar target and recover the coefficient function beta(t).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _fpca(X, k):
    mean = X.mean(axis=0)
    Xc = X - mean
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mean, Vt[:k], Xc @ Vt[:k].T


def fit_fpc_regression(X, y, k=3):
    mean, phi, scores = _fpca(X, k=k)
    # Regress y on the FPC scores
    A = np.hstack([np.ones((len(y), 1)), scores])
    beta_hat = np.linalg.lstsq(A, y, rcond=None)[0]
    alpha = float(beta_hat[0])
    b_scores = beta_hat[1:]
    # Coefficient function beta(t) = sum_k b_scores[k] * phi[k](t) (approx up to dt scale)
    beta_t = phi.T @ b_scores
    return {"alpha": alpha, "beta_scores": b_scores, "beta_t": beta_t,
             "mean": mean, "phi": phi, "scores": scores}


if __name__ == "__main__":
    print("=== Functional regression via FPC scores ===\n")
    rng = np.random.default_rng(0)
    n, T = 200, 60
    t = np.linspace(0, 1, T)
    dt = t[1] - t[0]
    # Simulate curves as random linear combinations of 3 known basis functions.
    b1 = np.sin(2 * np.pi * t); b2 = np.cos(2 * np.pi * t); b3 = t
    c = rng.normal(0, 1, (n, 3))
    X = c @ np.stack([b1, b2, b3]) + 0.05 * rng.normal(0, 1, (n, T))
    # True coefficient function; scalar response = int X * beta(t) dt
    beta_true = 2 * np.sin(2 * np.pi * t) - t
    y = X @ beta_true * dt + rng.normal(0, 0.3, n)

    fit = fit_fpc_regression(X, y, k=3)
    beta_hat = fit["beta_t"] / dt          # convert back to per-unit-t scale

    # Correlation of estimated vs true coefficient function
    corr = float(np.corrcoef(beta_hat, beta_true)[0, 1])
    print(f"  scalar prediction R^2 = "
          f"{1 - np.var(y - fit['alpha'] - fit['scores'] @ fit['beta_scores']) / np.var(y):.3f}")
    print(f"  corr(estimated beta(t), true beta(t)) = {corr:.3f}")
    print(f"  mean |diff|: {float(np.mean(np.abs(beta_hat - beta_true))):.3f}\n")

    print("--- library cross-check (R refund::pfr; R fda::fRegress; Python scikit-fda) ---")
