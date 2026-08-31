"""Probabilistic PCA (Reference Sec 25.10).

Tipping & Bishop (1999) 'Probabilistic principal component analysis.'

Generative model:
  z_i ~ N(0, I_q)             latent factor (q-dim)
  x_i | z_i ~ N(W z_i + mu, sigma^2 I_d)

MLE has a CLOSED-FORM solution (Tipping-Bishop Thm):
  W_ML = U_q (Lambda_q - sigma^2 I)^{1/2} R          (any rotation R)
  sigma^2_ML = 1/(d-q) * sum_{k=q+1}^d lambda_k       (residual variance)

where (U_q, Lambda_q) are the top-q eigenpairs of the sample covariance.

PPCA gives:
  * Full likelihood (missing-data handling, model comparison).
  * Bayesian extension (Bayesian PCA, Bishop 1999).
  * Sensible for the presence of measurement noise.

Here we implement closed-form PPCA MLE, plus an EM variant for cases with
MISSING VALUES (Tipping-Bishop Sec 4).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def ppca_ml(X, q):
    """Closed-form MLE for PPCA."""
    n, d = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu
    S = Xc.T @ Xc / n
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(-vals)
    vals = vals[order]; vecs = vecs[:, order]
    sigma2 = float(np.mean(vals[q:])) if d > q else 1e-6
    L = np.diag(np.sqrt(np.maximum(vals[:q] - sigma2, 0.0)))
    W = vecs[:, :q] @ L
    return {"mu": mu, "W": W, "sigma2": sigma2,
             "eigenvalues": vals, "top_eigvecs": vecs[:, :q]}


def ppca_em(X_missing, q, max_iter=100, tol=1e-6):
    """EM for PPCA with missing entries (NaN = missing)."""
    n, d = X_missing.shape
    mask = ~np.isnan(X_missing)
    X = np.where(mask, X_missing, 0.0)
    # Init with observed-column means then closed form on filled matrix
    mu = np.nanmean(X_missing, axis=0)
    for _ in range(20):
        X_filled = np.where(mask, X_missing, mu)
        model = ppca_ml(X_filled, q)
        # E-step: E[z | x] = M^-1 W^T (x - mu),   M = W^T W + sigma^2 I
        W = model["W"]; sigma2 = model["sigma2"]; mu = model["mu"]
        M = W.T @ W + sigma2 * np.eye(q)
        M_inv = np.linalg.inv(M)
        z_exp = (X_filled - mu) @ W @ M_inv.T
        # Reconstruct missing entries
        X_recon = z_exp @ W.T + mu
        X_missing = np.where(mask, X_missing, X_recon)
    return model, X_missing


if __name__ == "__main__":
    print("=== Probabilistic PCA (Tipping-Bishop 1999) ===\n")
    rng = np.random.default_rng(0)
    d, q, n = 8, 3, 300
    W_true = rng.normal(0, 1, (d, q))
    sigma2_true = 0.25
    z = rng.normal(0, 1, (n, q))
    X = z @ W_true.T + rng.normal(0, np.sqrt(sigma2_true), (n, d))

    model = ppca_ml(X, q=3)
    print(f"  true sigma^2 = {sigma2_true:.3f}   MLE sigma^2 = {model['sigma2']:.3f}")
    # Compare W subspace via principal angles
    U, _, _ = np.linalg.svd(model["W"], full_matrices=False)
    U_true, _, _ = np.linalg.svd(W_true, full_matrices=False)
    sing = np.linalg.svd(U.T @ U_true, compute_uv=False)
    angles = np.arccos(np.clip(sing, 0, 1))
    print(f"  principal angles between subspaces (rad): {np.round(angles, 3).tolist()}"
          "   (all ~ 0 = perfect match)")

    # Missing-data demo
    X_miss = X.copy()
    miss_frac = 0.15
    mask_miss = rng.random(X.shape) < miss_frac
    X_miss[mask_miss] = np.nan
    model_em, X_filled = ppca_em(X_miss, q=3, max_iter=50)
    rmse_impute = float(np.sqrt(np.mean((X_filled[mask_miss] - X[mask_miss]) ** 2)))
    col_means = np.nanmean(X_miss, axis=0)
    mean_impute = np.tile(col_means, (n, 1))
    rmse_mean = float(np.sqrt(np.mean((mean_impute[mask_miss] - X[mask_miss]) ** 2)))
    print(f"\n  Missing-data imputation:")
    print(f"    baseline (column mean) RMSE = {rmse_mean:.3f}")
    print(f"    PPCA-EM                RMSE = {rmse_impute:.3f}\n")
    print("--- library cross-check (sklearn.decomposition.PCA / PPCA; R pcaMethods) ---")
