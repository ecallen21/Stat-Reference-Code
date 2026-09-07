"""Wild cluster bootstrap (Reference Sec 12.15).

Cameron, Gelbach & Miller 2008 'Bootstrap-based improvements for
inference with clustered errors', REStat; MacKinnon & Webb 2018. The
wild cluster bootstrap gives more accurate CIs / p-values than
cluster-robust sandwich SEs when the number of clusters is small
(G < 40, or with unbalanced cluster sizes).

Bootstrap scheme (for OLS  y = X * beta + u):

    For b = 1..B:
        For each cluster g, draw w_g in {+1, -1} (Rademacher).
        y_star_g = X_g * beta_hat + w_g * u_hat_g       # keeps cluster structure
        Fit OLS on (X, y_star) -> beta_hat_star^b, t_star^b
    p-value = 2 * min(F(t_obs), 1 - F(t_obs))            # t-percentile CI

We compare cluster-robust SE (CR1), naive OLS SE and wild-cluster
bootstrap on a simulated cluster-heteroskedastic setting with small G.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def cluster_robust_se(X, resid, g):
    """CR1 sandwich: (X'X)^-1 sum_g X_g' u_g u_g' X_g (X'X)^-1 * (G/(G-1)) * (n-1)/(n-k)."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    G = len(np.unique(g))
    S = np.zeros((k, k))
    for cluster in np.unique(g):
        idx = g == cluster
        u = resid[idx]
        X_g = X[idx]
        s = X_g.T @ u
        S += np.outer(s, s)
    adj = (G / (G - 1)) * ((n - 1) / (n - k))
    V = adj * XtX_inv @ S @ XtX_inv
    return np.sqrt(np.diag(V))


def wild_cluster_bootstrap(X, y, g, B=1000, seed=0, coef_index=1):
    n, k = X.shape
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta_hat
    rng = np.random.default_rng(seed)
    G = np.unique(g)
    beta_star = np.zeros(B)
    for b in range(B):
        w = {c: rng.choice([-1, 1]) for c in G}
        u_star = np.array([w[gi] * resid[i] for i, gi in enumerate(g)])
        y_star = X @ beta_hat + u_star
        b_star = np.linalg.lstsq(X, y_star, rcond=None)[0]
        beta_star[b] = b_star[coef_index]
    return {"beta_hat": beta_hat[coef_index],
            "boot_dist": beta_star,
            "CI_95_percentile": (float(np.quantile(beta_star, 0.025)),
                                  float(np.quantile(beta_star, 0.975)))}


if __name__ == "__main__":
    print("=== Wild cluster bootstrap (Cameron-Gelbach-Miller) ===\n")
    rng = np.random.default_rng(0)
    G = 12        # only 12 clusters -- 'few-clusters' regime
    n_per = 40
    n = G * n_per
    g = np.repeat(np.arange(G), n_per)
    x = rng.normal(size=n)
    #  Cluster-level shock: u_g ~ Normal(0, sigma_g^2), heterogeneous
    cluster_shock = np.random.default_rng(1).normal(scale=1.0, size=G)
    within_shock = rng.normal(scale=0.5, size=n)
    y = 0.5 * x + cluster_shock[g] + within_shock

    X = np.c_[np.ones(n), x]
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta_hat

    ols_se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * np.var(resid, ddof=X.shape[1])))
    cr_se = cluster_robust_se(X, resid, g)
    wc = wild_cluster_bootstrap(X, y, g, B=1000, seed=0, coef_index=1)

    print(f"  n = {n} ({G} clusters of {n_per})")
    print(f"  beta_hat (x coef) = {beta_hat[1]:.3f}  (truth 0.5)\n")
    print(f"  OLS  SE(x)                 = {ols_se[1]:.3f}  <- WRONG (assumes iid)")
    print(f"  Cluster-robust CR1 SE(x)   = {cr_se[1]:.3f}   asymptotic; may be off with G<40")
    print(f"  Wild-cluster bootstrap 95% CI (pctile) = "
          f"[{wc['CI_95_percentile'][0]:.3f}, {wc['CI_95_percentile'][1]:.3f}]")
    ci_width = wc['CI_95_percentile'][1] - wc['CI_95_percentile'][0]
    approx_se = ci_width / (2 * 1.96)
    print(f"  Implied SE(x) from wild-boot CI (approx)     = {approx_se:.3f}")

    print("\n--- library cross-check (fwildclusterboot / lmtest R; wildboottest Python) ---")
