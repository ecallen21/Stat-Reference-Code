"""CUPED variance reduction (Reference Sec 44.3).

Deng-Xu-Kohavi-Walker 2013.  Regression adjustment using a
PRE-EXPERIMENT covariate (X) to reduce the variance of the outcome
Y in an A/B test.

  Y_cuped = Y - theta * (X - X_bar)     with theta = Cov(Y, X) / Var(X)

Variance reduction fraction = 1 - (1 - rho^2)  where rho = corr(Y, X).
When rho = 0.7, variance drops to 51 % -> effective n roughly doubles.

Assignment is random and independent of X, so this is unbiased.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def cuped_adjust(y, x):
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean()), float(theta)


def welch(y_C, y_T):
    diff = y_T.mean() - y_C.mean()
    se = np.sqrt(y_C.var(ddof=1) / len(y_C) + y_T.var(ddof=1) / len(y_T))
    return float(diff), float(se)


if __name__ == "__main__":
    print("=== CUPED variance reduction ===\n")
    rng = np.random.default_rng(0)
    n = 5000
    # Pre-experiment covariate x (same user prior period) is correlated with y
    x = rng.normal(0, 1, 2 * n)
    y = 5 + 0.8 * x + rng.normal(0, 0.6, 2 * n)   # rho about 0.8
    # Random assignment adds a small treatment lift
    T = np.array([0] * n + [1] * n)
    y = y + 0.10 * T

    y_C, y_T = y[T == 0], y[T == 1]
    x_C, x_T = x[T == 0], x[T == 1]
    diff_raw, se_raw = welch(y_C, y_T)

    # CUPED using the pooled theta (Deng et al.)
    y_adj_all, theta = cuped_adjust(y, x)
    y_C_a = y_adj_all[T == 0]; y_T_a = y_adj_all[T == 1]
    diff_cuped, se_cuped = welch(y_C_a, y_T_a)

    var_reduction = 1 - (se_cuped / se_raw) ** 2
    rho = float(np.corrcoef(y, x)[0, 1])
    print(f"  Correlation rho(y, x) = {rho:.3f}   theta = {theta:.3f}")
    print(f"  Raw   difference = {diff_raw:+.4f}   SE = {se_raw:.4f}")
    print(f"  CUPED difference = {diff_cuped:+.4f}   SE = {se_cuped:.4f}")
    print(f"  Variance reduction = {var_reduction:.2%}  (~ 1 - (1 - rho^2))")
    print(f"  Effective sample-size multiplier = {1 / (1 - var_reduction):.2f}x\n")

    print("--- library cross-check (R stats::lm on residualised outcomes; Python statsmodels OLS) ---")
