"""Multicollinearity diagnostics: VIF & condition number (Reference Sec 41.8).

  VIF_j = 1 / (1 - R^2_j)   where R^2_j is from regressing predictor j
                             on all OTHER predictors.
    * VIF > 5  -> concern.
    * VIF > 10 -> serious multicollinearity.

  CONDITION NUMBER = sqrt(max eigenvalue / min eigenvalue) of X^T X.
    * < 15  ok.
    * 15-30 moderate.
    * > 30  severe.

Fixes: centre interactions, ridge, PCA reduction, domain-guided
variable consolidation.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def vif(X, names=None):
    n, k = X.shape
    if names is None:
        names = [f"x{i}" for i in range(k)]
    vifs = {}
    for j in range(k):
        y = X[:, j]
        Xo = np.delete(X, j, axis=1)
        # OLS R^2
        A = np.column_stack([np.ones(n), Xo])
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        y_hat = A @ beta
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        R2 = 1 - ss_res / max(ss_tot, 1e-12)
        vifs[names[j]] = float(1 / max(1 - R2, 1e-12))
    return vifs


def condition_number(X):
    # Scale first so units don't dominate
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    _, s, _ = np.linalg.svd(Xs, full_matrices=False)
    return float(s.max() / s.min())


if __name__ == "__main__":
    print("=== Multicollinearity: VIF + condition number ===\n")
    rng = np.random.default_rng(0)
    n = 300
    # x1, x2 nearly identical (multicollinear); x3 independent
    x1 = rng.normal(0, 1, n)
    x2 = x1 + rng.normal(0, 0.05, n)
    x3 = rng.normal(0, 1, n)
    X = np.column_stack([x1, x2, x3])
    vifs = vif(X, names=["x1", "x2", "x3"])
    for k, v in vifs.items():
        print(f"  VIF({k}) = {v:.2f}")
    print(f"  Condition number = {condition_number(X):.2f}\n")

    # Fix: drop x2
    X_fixed = np.column_stack([x1, x3])
    vifs_fixed = vif(X_fixed, names=["x1", "x3"])
    for k, v in vifs_fixed.items():
        print(f"  after dropping x2: VIF({k}) = {v:.2f}")
    print(f"  Condition number = {condition_number(X_fixed):.2f}\n")

    print("--- library cross-check (R car::vif, performance::check_collinearity; Python statsmodels VIF) ---")
