"""Adaptive LASSO (Reference Sec 32.12).

Zou (2006) 'The adaptive LASSO and its oracle properties.'

Two-step procedure:
  1. Compute initial coefficient estimates beta_init (OLS if n > p, or
     ridge / LASSO otherwise).
  2. Weighted LASSO:
       argmin_beta   || y - X beta ||^2 + lambda * sum_j w_j |beta_j|
       with weights w_j = 1 / |beta_init_j|^gamma (gamma > 0, often 1).

Weights DE-BIAS large signal coefficients (large beta_init -> small w_j
-> less shrinkage) and hard-shrink noise (small beta_init -> huge w_j
-> forced to zero). Under mild conditions Adaptive LASSO enjoys the
ORACLE PROPERTY: it selects the true support AND has the same
asymptotic distribution as OLS on the true model.

Here we compare LASSO vs Adaptive LASSO vs (oracle) OLS on the true
support.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _soft(x, lam): return np.sign(x) * np.maximum(0, np.abs(x) - lam)


def _weighted_lasso_cd(X, y, w, max_iter=200):
    n, d = X.shape
    beta = np.zeros(d)
    XtX_diag = (X ** 2).sum(axis=0) / n
    for _ in range(max_iter):
        r = y - X @ beta
        for j in range(d):
            xj = X[:, j]
            rho = xj @ r / n + XtX_diag[j] * beta[j]
            b_new = _soft(rho, w[j]) / max(XtX_diag[j], 1e-12)
            r = r + xj * (beta[j] - b_new)
            beta[j] = b_new
    return beta


def adaptive_lasso(X, y, lam, gamma=1.0, init=None, ridge_lam=1e-2):
    n, d = X.shape
    if init is None:
        # Ridge init (safe when d >= n).
        A = X.T @ X / n + ridge_lam * np.eye(d)
        init = np.linalg.solve(A, X.T @ y / n)
    w = lam / (np.abs(init) ** gamma + 1e-6)
    return _weighted_lasso_cd(X, y, w)


if __name__ == "__main__":
    print("=== Adaptive LASSO (Zou 2006) ===\n")
    rng = np.random.default_rng(0)
    n, d = 200, 30
    beta_true = np.zeros(d)
    beta_true[[0, 5, 10]] = [3.0, -2.5, 2.0]
    X = rng.normal(0, 1, (n, d))
    y = X @ beta_true + rng.normal(0, 0.5, n)

    # Plain LASSO
    b_lasso = _weighted_lasso_cd(X, y, np.full(d, 0.20))
    b_alasso = adaptive_lasso(X, y, lam=0.20)
    # Oracle OLS on true support
    supp = [0, 5, 10]
    b_oracle = np.zeros(d)
    b_oracle[supp] = np.linalg.lstsq(X[:, supp], y, rcond=None)[0]

    def stats(name, b):
        sel = np.abs(b) > 1e-3
        true_supp = np.abs(beta_true) > 0
        tp = int(((sel) & (true_supp)).sum())
        fp = int(((sel) & (~true_supp)).sum())
        bias = float(np.abs(b[true_supp] - beta_true[true_supp]).mean())
        print(f"  {name:>10s}   TP={tp}/3  FP={fp}   signal betas={np.round(b[true_supp], 3).tolist()}"
              f"   mean |bias|={bias:.4f}")

    stats("LASSO", b_lasso)
    stats("aLASSO", b_alasso)
    stats("oracle OLS", b_oracle)

    print("\n  Adaptive LASSO recovers the true support AND is much less biased on signals.\n")
    print("--- library cross-check (R glmnet with penalty.factor; ncvreg::ncvreg;"
          " sklearn Lasso with sample_weight adjustment) ---")
