"""Group LASSO (Reference Sec 32.9).

Yuan & Lin (2006) 'Model selection and estimation in regression with
grouped variables.'

Groups of features (e.g. dummy-coded factor levels, spline basis
functions per predictor) are selected TOGETHER by an L2-then-L1
penalty:

  min_beta   0.5 || y - X beta ||^2  +  lambda * sum_g sqrt(|g|) * ||beta_g||_2.

Solved by group-wise soft thresholding (block coordinate descent):

  beta_g <- max(0, 1 - lambda * sqrt(|g|) / ||beta_g_ols||) * beta_g_ols.

Here we implement group LASSO with block-coordinate descent, apply to
synthetic data with 5 groups of size 4 (only 2 groups truly active).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def group_lasso_bcd(X, y, groups, lam, max_iter=200, tol=1e-6):
    """Block-coordinate descent for group LASSO."""
    n, d = X.shape
    beta = np.zeros(d)
    unique_g = list(range(int(max(groups)) + 1))
    XtX_diag = {g: float(np.linalg.norm(X[:, groups == g], ord=2) ** 2 / n) for g in unique_g}
    for _ in range(max_iter):
        beta_old = beta.copy()
        r = y - X @ beta
        for g in unique_g:
            idx = np.where(groups == g)[0]
            Xg = X[:, idx]
            rho = Xg.T @ r / n + XtX_diag[g] * beta[idx]
            group_norm = float(np.linalg.norm(rho))
            thr = lam * np.sqrt(len(idx))
            if group_norm <= thr:
                new = np.zeros(len(idx))
            else:
                new = (1 - thr / group_norm) * rho / max(XtX_diag[g], 1e-12)
            r = r + Xg @ (beta[idx] - new)
            beta[idx] = new
        if np.max(np.abs(beta - beta_old)) < tol: break
    return beta


if __name__ == "__main__":
    print("=== Group LASSO (Yuan-Lin 2006) ===\n")
    rng = np.random.default_rng(0)
    n, K, sz = 200, 5, 4
    d = K * sz
    X = rng.normal(0, 1, (n, d))
    groups = np.repeat(np.arange(K), sz)
    beta_true = np.zeros(d)
    beta_true[0:sz] = [2.0, -1.0, 1.5, 0.8]
    beta_true[3 * sz:4 * sz] = [1.2, 1.8, -0.9, 0.6]
    y = X @ beta_true + rng.normal(0, 0.5, n)

    beta = group_lasso_bcd(X, y, groups, lam=0.15)
    active_groups = [g for g in range(K) if np.linalg.norm(beta[groups == g]) > 1e-4]
    print(f"  active groups selected: {active_groups}   (truth = [0, 3])")
    for g in range(K):
        norm = float(np.linalg.norm(beta[groups == g]))
        print(f"    group {g}   ||beta_g||_2 = {norm:.3f}   entries = {beta[groups == g].round(3).tolist()}")
    print("\n  Group LASSO zeros out entire uninformative groups.\n")
    print("--- library cross-check (R grplasso / gglasso; Python group-lasso pip pkg / celer) ---")
