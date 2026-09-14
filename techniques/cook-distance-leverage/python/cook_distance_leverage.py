"""Cook's Distance / Leverage (Reference Sec 47.333).

Cook 1977 Technometrics. Regression influence diagnostics:

    h_ii = X (X^T X)^-1 X^T[i, i]                          leverage
    D_i = r_i^2 / (p * (1 - h_ii)) * h_ii / (1 - h_ii)     Cook's D
        (r_i = internally studentised residual)

Rules of thumb:
    h_ii > 2 p / n         high leverage
    D_i > 4 / n            influential
    D_i > 1                 highly influential
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def leverage_and_cooks_d(X, y):
    """Return arrays of leverage h_ii, studentized residuals, and Cook's D."""
    n, p = X.shape
    XTX_inv = np.linalg.inv(X.T @ X)
    H = X @ XTX_inv @ X.T
    h = np.diag(H)
    beta = XTX_inv @ X.T @ y
    y_pred = X @ beta
    resid = y - y_pred
    sigma2 = float((resid ** 2).sum() / (n - p))
    stud_resid = resid / (np.sqrt(sigma2 * (1 - h)) + 1e-12)
    cooks_d = stud_resid ** 2 * h / (p * (1 - h) + 1e-12)
    return h, stud_resid, cooks_d


if __name__ == "__main__":
    print("=== Cook's Distance / Leverage (Cook 1977) ===\n")
    rng = np.random.default_rng(0)

    # Regression with a couple of high-leverage and high-influence points
    n = 40; p = 3
    X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    beta_true = np.array([1.0, 2.0, -1.0])
    y = X @ beta_true + rng.normal(0, 0.5, n)

    # Inject high-leverage outlier and high-influence outlier
    X[0, 1] = 8.0; X[0, 2] = 6.0                                    # far in x-space
    y[5] = 20.0                                                     # response outlier
    y[10] = -15.0

    h, r, D = leverage_and_cooks_d(X, y)
    thresh_h = 2 * p / n
    thresh_D = 4 / n

    print(f"  n = {n}, p = {p}, high-leverage threshold h = {thresh_h:.3f}, "
          f"Cook threshold D = {thresh_D:.3f}\n")
    print(f"  Top-5 points by Cook's D:")
    top = np.argsort(D)[::-1][:5]
    for i in top:
        flags = []
        if h[i] > thresh_h: flags.append("HIGH_LEV")
        if D[i] > thresh_D: flags.append("INFL")
        if D[i] > 1: flags.append("VERY_INFL")
        print(f"    idx {i:>3}   h = {h[i]:.3f}   stud_resid = {r[i]:>+.2f}   "
              f"D = {D[i]:.3f}   {' '.join(flags)}")

    print(f"\n  Injected: idx 0 (high leverage in X), idx 5 and 10 (response outliers)")
    print(f"  D > 4/n flags points to investigate; D > 1 typically requires action.")

    print("\n--- library cross-check (statsmodels OLSInfluence.cooks_distance; car::influence.measures R) ---")
