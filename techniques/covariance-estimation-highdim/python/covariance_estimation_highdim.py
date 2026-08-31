"""High-dim covariance estimation (Reference Sec 32.11).

When p is comparable to n, the SAMPLE COVARIANCE has huge estimation
error and is often SINGULAR (p >= n).  Standard remedies:

  1. LEDOIT-WOLF SHRINKAGE (2004): S_shrink = alpha * Target + (1-alpha) * S
     with Target = mu_hat * I (mu_hat = tr(S)/p) and alpha chosen
     analytically to minimise Frobenius risk.
  2. BANDED / TAPERED covariance (Bickel-Levina 2008) for ordered
     predictors.
  3. GRAPHICAL LASSO (see gaussian-graphical-model) — sparse precision.

Here we implement Ledoit-Wolf + a simple banded estimator, and compare
Frobenius risk against the sample covariance.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def ledoit_wolf(X):
    """Ledoit-Wolf 2004 diagonal-target shrinkage covariance."""
    n, p = X.shape
    S = np.cov(X, rowvar=False, ddof=0)
    mu = float(np.trace(S) / p)
    F = mu * np.eye(p)                                # target
    d = np.sum((S - F) ** 2)
    # b_bar: estimator of variance of the sample covariance
    Xc = X - X.mean(axis=0)
    b = 0.0
    for i in range(n):
        outer = np.outer(Xc[i], Xc[i]) - S
        b += float(np.sum(outer ** 2)) / n ** 2
    alpha = min(1.0, b / max(d, 1e-12))
    return alpha * F + (1 - alpha) * S, alpha


def banded_cov(X, bandwidth=5):
    """Zero out entries |i - j| > bandwidth."""
    S = np.cov(X, rowvar=False, ddof=0)
    p = S.shape[0]
    ii, jj = np.indices(S.shape)
    mask = np.abs(ii - jj) <= bandwidth
    return S * mask


def frob(A, B):
    return float(np.linalg.norm(A - B, ord="fro"))


if __name__ == "__main__":
    print("=== High-dim covariance estimation ===\n")
    rng = np.random.default_rng(0)
    p = 40
    n = 30                                      # p > n so sample cov is singular
    # Truth: tridiagonal covariance
    ii, jj = np.indices((p, p))
    Sigma_true = 0.7 ** np.abs(ii - jj)          # AR(1)-like, PSD by construction
    X = rng.multivariate_normal(np.zeros(p), Sigma_true, n)

    S_hat = np.cov(X, rowvar=False, ddof=0)
    S_lw, alpha = ledoit_wolf(X)
    S_band = banded_cov(X, bandwidth=2)

    print(f"  n = {n}, p = {p}   p > n so sample cov rank <= {n - 1}, singular")
    print(f"  Ledoit-Wolf shrinkage intensity alpha = {alpha:.3f}\n")
    print(f"  Frobenius error to truth:")
    print(f"    sample covariance           {frob(S_hat, Sigma_true):.3f}")
    print(f"    Ledoit-Wolf shrinkage       {frob(S_lw, Sigma_true):.3f}")
    print(f"    banded (bandwidth = 2)      {frob(S_band, Sigma_true):.3f}\n")
    print("--- library cross-check (sklearn.covariance.LedoitWolf / OAS; R corpcor / covShrink / CovTools) ---")
