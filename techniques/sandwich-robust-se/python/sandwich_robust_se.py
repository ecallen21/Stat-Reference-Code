"""Heteroscedasticity-robust and cluster-robust standard errors for OLS
(Reference §5.7, §5.8; White 1980, Liang-Zeger 1986).

OLS coefficient variance is:
    Cov(beta_hat) = (X' X)^-1 X' Omega X (X' X)^-1
Under homoscedasticity Omega = sigma^2 I -> (X' X)^-1 sigma^2 (the usual SE).
Under heteroscedasticity or clustering, Omega is not scalar; the classical
SE is wrong.

Sandwich estimators plug in a data-driven Omega_hat:

HC0 (White 1980)          Omega = diag(r_i^2)
HC1 (Stata default)       HC0 scaled by n / (n - k)
HC2                       Omega = diag(r_i^2 / (1 - h_ii))         h_ii = leverage
HC3 (MacKinnon-White '85) Omega = diag((r_i / (1 - h_ii))^2)  -- less biased in small n

Cluster-robust (Liang-Zeger 1986)
    Omega_hat = sum_g X_g' r_g r_g' X_g
    Standard adjustment: sqrt((G / (G - 1)) * (n - 1) / (n - k)) * ...

Use cluster-robust when errors are correlated within group (repeated
measures, spatial or hierarchical structure).  Rule of thumb: need >= 40-50
clusters for reliable inference.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def ols_with_robust_se(X, y, cluster=None) -> dict:
    """OLS fit with classical + HC0/HC1/HC3 + optional cluster-robust SEs."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    r = y - X @ beta
    # Classical (homoscedastic)
    sigma2 = float(r @ r / (n - k))
    se_cls = np.sqrt(np.diag(sigma2 * XtX_inv))
    # HC0 (White)
    bread = XtX_inv
    meat_hc0 = X.T @ (X * (r ** 2)[:, None])
    cov_hc0 = bread @ meat_hc0 @ bread
    se_hc0 = np.sqrt(np.diag(cov_hc0))
    # HC1
    cov_hc1 = cov_hc0 * n / (n - k)
    se_hc1 = np.sqrt(np.diag(cov_hc1))
    # HC3 (leverage-adjusted)
    H = X @ XtX_inv @ X.T
    h = np.diag(H)
    r_hc3 = r / (1 - h)
    meat_hc3 = X.T @ (X * (r_hc3 ** 2)[:, None])
    cov_hc3 = bread @ meat_hc3 @ bread
    se_hc3 = np.sqrt(np.diag(cov_hc3))
    out = {"beta": beta,
           "se_classical": se_cls,
           "se_hc0": se_hc0, "se_hc1": se_hc1, "se_hc3": se_hc3,
           "n": int(n), "k": int(k)}
    if cluster is not None:
        cluster = np.asarray(cluster)
        clusters = np.unique(cluster); G = len(clusters)
        meat = np.zeros_like(bread)
        for g in clusters:
            idx = cluster == g
            Xg = X[idx]; rg = r[idx]
            u = Xg.T @ rg
            meat = meat + np.outer(u, u)
        cov_cl = bread @ meat @ bread
        # Stata-style small-sample correction
        c = (G / (G - 1)) * ((n - 1) / (n - k))
        se_cl = np.sqrt(np.diag(c * cov_cl))
        out["se_cluster"] = se_cl; out["n_clusters"] = int(G)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Heteroscedastic errors: SE(x) grows with x ===")
    n = 400
    x = rng.uniform(-2, 2, n)
    X = np.column_stack([np.ones(n), x])
    y = 1 + 2 * x + (0.5 + np.abs(x)) * rng.normal(size=n)
    r = ols_with_robust_se(X, y)
    for j, name in enumerate(("intercept", "x")):
        print(f"  {name}: beta = {r['beta'][j]:6.3f}   "
              f"classical SE = {r['se_classical'][j]:.4f}   "
              f"HC3 SE = {r['se_hc3'][j]:.4f}")

    print("\n=== Cluster-correlated errors: 40 clusters of size 10 ===")
    n_clusters = 40; n_per = 10; n = n_clusters * n_per
    cluster = np.repeat(np.arange(n_clusters), n_per)
    u_c = rng.normal(0, 2, n_clusters)     # cluster-level shock
    x = rng.normal(size=n)
    y = 1 + 2 * x + u_c[cluster] + rng.normal(0, 0.5, n)
    X = np.column_stack([np.ones(n), x])
    r = ols_with_robust_se(X, y, cluster=cluster)
    print(f"  intercept: SE classical = {r['se_classical'][0]:.4f}, "
          f"HC3 = {r['se_hc3'][0]:.4f}, cluster = {r['se_cluster'][0]:.4f}")
    print(f"  x:         SE classical = {r['se_classical'][1]:.4f}, "
          f"HC3 = {r['se_hc3'][1]:.4f}, cluster = {r['se_cluster'][1]:.4f}")

    print("\n--- library cross-check (statsmodels HC3 / cluster) ---")
    try:
        import statsmodels.api as sm
        m1 = sm.OLS(y, X).fit(cov_type="HC3")
        m2 = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": cluster})
        print(f"  statsmodels HC3     : SE = {m1.bse.round(4)}")
        print(f"  statsmodels cluster : SE = {m2.bse.round(4)}")
    except Exception as ex:
        print(f"  (statsmodels unavailable: {ex})")
