"""Seemingly Unrelated Regression (Reference Sec 35.7).

Zellner (1962) 'An efficient method of estimating seemingly unrelated
regressions and tests for aggregation bias.'

M equations sharing NO common regressors but CORRELATED ERRORS:

  y_1 = X_1 beta_1 + eps_1
  y_2 = X_2 beta_2 + eps_2
  ...
  y_M = X_M beta_M + eps_M
  Cov(eps_i, eps_j) = sigma_ij * I.

FEASIBLE GLS (Zellner 1962):
  1. Fit each equation by OLS -> residuals.
  2. Estimate Sigma_hat = (1/n) e' e   (M x M cross-eq covariance).
  3. Stack the system and apply GLS with kron(Sigma_hat^-1, I).

If regressors are identical across equations OR Sigma is diagonal,
SUR reduces to OLS on each equation. Efficiency gain over OLS
increases with the cross-equation error correlation.

Here we implement two-equation FGLS SUR and compare to equation-by-
equation OLS when cross-equation errors are strongly correlated.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def ols_by_equation(Xs, ys):
    return [np.linalg.lstsq(X, y, rcond=None)[0] for X, y in zip(Xs, ys)]


def sur_fgls(Xs, ys):
    """Zellner FGLS SUR."""
    M = len(Xs)
    n = Xs[0].shape[0]
    beta_ols = ols_by_equation(Xs, ys)
    resid = [y - X @ b for X, y, b in zip(Xs, ys, beta_ols)]
    R = np.stack(resid, axis=1)              # (n, M)
    Sigma = R.T @ R / n                       # (M, M)
    S_inv = np.linalg.inv(Sigma + 1e-8 * np.eye(M))

    # Block-diagonal design; stacked GLS
    k_sizes = [X.shape[1] for X in Xs]
    K = sum(k_sizes)
    XtX = np.zeros((K, K))
    Xty = np.zeros(K)
    row_start = 0
    for i in range(M):
        col_start_i = sum(k_sizes[:i])
        for j in range(M):
            col_start_j = sum(k_sizes[:j])
            XtX[col_start_i:col_start_i + k_sizes[i],
                 col_start_j:col_start_j + k_sizes[j]] += (S_inv[i, j]
                                                             * Xs[i].T @ Xs[j])
            if j == i:
                Xty[col_start_i:col_start_i + k_sizes[i]] += (S_inv[i, j]
                                                                 * Xs[i].T @ ys[j])
            else:
                Xty[col_start_i:col_start_i + k_sizes[i]] += (S_inv[i, j]
                                                                 * Xs[i].T @ ys[j])
    beta_all = np.linalg.solve(XtX, Xty)
    out = []
    for i in range(M):
        cs = sum(k_sizes[:i])
        out.append(beta_all[cs:cs + k_sizes[i]])
    return out, Sigma


if __name__ == "__main__":
    print("=== Seemingly Unrelated Regression (Zellner 1962) ===\n")
    rng = np.random.default_rng(0)
    n = 100
    X1 = np.stack([np.ones(n), rng.normal(0, 1, n)], axis=1)
    X2 = np.stack([np.ones(n), rng.normal(0, 1, n)], axis=1)
    beta1_true = np.array([1.0, 0.7])
    beta2_true = np.array([-0.5, 1.5])
    # Correlated errors: (eps_1, eps_2) ~ N(0, Sigma_true)
    Sigma_true = np.array([[1.0, 0.8], [0.8, 1.0]])
    L = np.linalg.cholesky(Sigma_true)
    E = rng.normal(0, 1, (n, 2)) @ L.T
    y1 = X1 @ beta1_true + E[:, 0]
    y2 = X2 @ beta2_true + E[:, 1]

    beta_ols = ols_by_equation([X1, X2], [y1, y2])
    beta_sur, Sigma_hat = sur_fgls([X1, X2], [y1, y2])

    print(f"  true Sigma:\n{Sigma_true}")
    print(f"  estimated Sigma:\n{Sigma_hat.round(3)}\n")
    print(f"  eq 1 truth = {beta1_true}")
    print(f"    OLS = {beta_ols[0].round(3).tolist()}")
    print(f"    SUR = {beta_sur[0].round(3).tolist()}")
    print(f"  eq 2 truth = {beta2_true}")
    print(f"    OLS = {beta_ols[1].round(3).tolist()}")
    print(f"    SUR = {beta_sur[1].round(3).tolist()}")

    print("\n  SUR = OLS if regressors identical or Sigma diagonal;\n"
          "  gains efficiency when the errors are strongly correlated (as here).\n")
    print("--- library cross-check (R systemfit::systemfit; Python linearmodels.system.SUR) ---")
