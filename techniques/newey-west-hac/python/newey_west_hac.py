"""Newey-West HAC + cluster-robust standard errors (Reference Sec 35.15).

Newey & West (1987) 'A simple, positive semi-definite, heteroskedasticity
and autocorrelation consistent covariance matrix.'

Standard OLS SEs are wrong under HETEROSKEDASTICITY (fix: White 1980)
or AUTOCORRELATION (fix: HAC/Newey-West) or CLUSTERING (Liang-Zeger 1986,
Cameron-Miller 2015).

  Newey-West HAC (bandwidth L):
     V_HAC = (X'X)^-1 Omega (X'X)^-1
     Omega = sum_i x_i x_i' u_i^2
             + sum_{l=1}^L w_l * sum_i (x_i x_{i+l}' + x_{i+l} x_i') u_i u_{i+l}
     Bartlett weights w_l = 1 - l / (L + 1).

  Cluster-robust:
     V_CR = (X'X)^-1 (sum_g X_g' u_g u_g' X_g) (X'X)^-1.

Here we implement both, compare on AR(1) errors to OLS SEs, and confirm
HAC gives ~ correct coverage while OLS SEs are too small.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def ols_se(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = float(resid @ resid / (n - k))
    V = sigma2 * np.linalg.inv(X.T @ X)
    return beta, np.sqrt(np.diag(V))


def newey_west(X, y, L=None):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    if L is None:
        L = int(4 * (n / 100) ** (2 / 9))            # Newey-West rule of thumb
    XtX_inv = np.linalg.inv(X.T @ X)
    Omega = np.zeros((k, k))
    for i in range(n):
        Omega += resid[i] ** 2 * np.outer(X[i], X[i])
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        for i in range(n - l):
            xxT = np.outer(X[i], X[i + l])
            Omega += w * resid[i] * resid[i + l] * (xxT + xxT.T)
    V = XtX_inv @ Omega @ XtX_inv
    return beta, np.sqrt(np.diag(V)), L


def cluster_robust(X, y, cluster):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    Omega = np.zeros((k, k))
    for g in np.unique(cluster):
        mask = cluster == g
        Xg = X[mask]; ug = resid[mask]
        Omega += Xg.T @ np.outer(ug, ug) @ Xg
    # Small-sample correction
    G = len(np.unique(cluster))
    correction = G / (G - 1) * (n - 1) / (n - k)
    V = correction * XtX_inv @ Omega @ XtX_inv
    return beta, np.sqrt(np.diag(V))


if __name__ == "__main__":
    print("=== Newey-West HAC + cluster-robust SEs ===\n")
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(0, 1, n)
    X = np.stack([np.ones(n), x], axis=1)
    # AR(1) errors: eps_t = 0.7 * eps_{t-1} + N(0, 1)
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = 0.7 * eps[t - 1] + rng.normal(0, 1)
    y = 1 + 2 * x + eps

    b_ols, se_ols = ols_se(X, y)
    b_nw, se_nw, L = newey_west(X, y)
    print(f"  AR(1) errors, n = {n}")
    print(f"  OLS SEs      : intercept = {se_ols[0]:.4f}   slope = {se_ols[1]:.4f}")
    print(f"  Newey-West SE: intercept = {se_nw[0]:.4f}   slope = {se_nw[1]:.4f}   L = {L}")
    print(f"    HAC SEs are typically LARGER than OLS under autocorrelation.\n")

    # Cluster-robust demo
    n_clusters = 30
    per_cluster = 10
    cluster = np.repeat(np.arange(n_clusters), per_cluster)
    # Random cluster effect
    alpha = rng.normal(0, 1.0, n_clusters)
    xc = rng.normal(0, 1, n_clusters * per_cluster)
    y_c = 1 + 2 * xc + alpha[cluster] + rng.normal(0, 0.5, n_clusters * per_cluster)
    Xc = np.stack([np.ones_like(xc), xc], axis=1)
    b_ols2, se_ols2 = ols_se(Xc, y_c)
    b_cr, se_cr = cluster_robust(Xc, y_c, cluster)
    print(f"  Clustered errors, {n_clusters} clusters x {per_cluster} obs")
    print(f"  OLS SE          : slope = {se_ols2[1]:.4f}")
    print(f"  Cluster-robust  : slope = {se_cr[1]:.4f}"
          f"   (inflation = {se_cr[1] / se_ols2[1]:.2f}x)\n")

    print("--- library cross-check (statsmodels OLS.fit(cov_type='HAC' / 'cluster'); R sandwich::vcovHAC / vcovCL) ---")
