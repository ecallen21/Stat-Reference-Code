"""Meta-regression (Reference Sec 22.5).

Extension of random-effects meta-analysis to a per-study
covariate (moderator).  For k studies with effect y_i, SE sigma_i,
covariate x_i:

  y_i = beta_0 + beta_1 x_i + u_i + eps_i
  u_i ~ N(0, tau^2)          (between-study heterogeneity)
  eps_i ~ N(0, sigma_i^2)    (within-study sampling error)

Estimate (beta, tau^2) by REML.  Report R^2 = 1 - tau_res^2/tau_null^2
for proportion of heterogeneity explained.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize_scalar


def _reml_ll(tau2, y, se, X):
    var = se ** 2 + tau2
    W = np.diag(1 / var)
    B = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    resid = y - X @ B
    # REML log-likelihood (negative)
    ll = -0.5 * (np.log(var).sum() + (resid ** 2 / var).sum()
                 + np.log(np.linalg.det(X.T @ W @ X)))
    return -ll


def meta_regression(y, se, x):
    """Simple univariate meta-regression with REML tau^2."""
    y = np.asarray(y, dtype=float); se = np.asarray(se, dtype=float); x = np.asarray(x, dtype=float)
    X = np.column_stack([np.ones_like(y), x])
    r = minimize_scalar(_reml_ll, bounds=(0, 100), args=(y, se, X), method="bounded")
    tau2 = float(r.x)
    W = np.diag(1 / (se ** 2 + tau2))
    B_hat = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    cov_B = np.linalg.inv(X.T @ W @ X)
    se_B = np.sqrt(np.diag(cov_B))
    # Null model tau^2 (intercept only)
    r_null = minimize_scalar(_reml_ll, bounds=(0, 100), args=(y, se, np.ones_like(y)[:, None]),
                             method="bounded")
    tau2_null = float(r_null.x)
    R2 = max(0.0, 1 - tau2 / max(tau2_null, 1e-12))
    return {"intercept": float(B_hat[0]), "slope": float(B_hat[1]),
            "SE_intercept": float(se_B[0]), "SE_slope": float(se_B[1]),
            "tau2": tau2, "tau2_null": tau2_null, "R2_heterog": float(R2)}


if __name__ == "__main__":
    print("=== Meta-regression: moderator explains between-study heterogeneity ===\n")
    rng = np.random.default_rng(0)
    K = 20
    x = rng.uniform(0, 1, K)              # moderator (e.g., study year normalised)
    true_beta = [0.5, -0.8]
    tau_true = 0.10
    se = rng.uniform(0.05, 0.20, K)
    y = true_beta[0] + true_beta[1] * x + rng.normal(0, tau_true, K) + rng.normal(0, se, K)

    r = meta_regression(y, se, x)
    print(f"  True beta = {true_beta}, true tau = {tau_true}")
    print(f"  Estimated intercept = {r['intercept']:+.3f}   SE = {r['SE_intercept']:.3f}")
    print(f"  Estimated slope     = {r['slope']:+.3f}   SE = {r['SE_slope']:.3f}")
    print(f"  Estimated tau^2     = {r['tau2']:.3f}   (null tau^2 = {r['tau2_null']:.3f})")
    print(f"  Proportion of heterogeneity explained by moderator: {r['R2_heterog']:.1%}\n")

    print("--- library cross-check (R metafor::rma; Python custom + statsmodels) ---")
