"""Functional linear model (Reference Sec 31.7).

Ramsay-Silverman FDA (2005), Ch 12-16.

Three main flavours:
  (A) SCALAR-ON-FUNCTION:  y_i = alpha + int X_i(t) beta(t) dt + eps.
       (see functional-regression)
  (B) FUNCTION-ON-SCALAR:  Y_i(t) = alpha(t) + sum_p beta_p(t) x_ip + eps(t).
       Coefficient FUNCTIONS regressed on scalar covariates.
  (C) FUNCTION-ON-FUNCTION: Y_i(t) = alpha(t) + int X_i(s) beta(s, t) ds + eps(t).

Here we implement FUNCTION-ON-SCALAR: for each t we run pointwise OLS on
scalar predictors, then optionally smooth the coefficient trajectory
beta_p(t) with a P-spline penalty. Demo: two-group differences in a
growth-curve setup.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def bspline_basis(t, n_knots=10, degree=3):
    knots = np.linspace(t.min(), t.max(), n_knots)
    cols = [np.ones_like(t), t, t ** 2, t ** 3]
    for k in knots[1:-1]:
        cols.append(np.maximum(0.0, t - k) ** degree)
    return np.stack(cols, axis=1)


def _diff_pen(d, order=2):
    D = np.diff(np.eye(d), n=order, axis=0)
    return D.T @ D


def function_on_scalar(Y, X, t, n_knots=10, lam=0.5):
    """
    Y: (n, T) response curves
    X: (n, p) scalar covariates (with intercept in column 0)
    Return: beta_hat as (p, T) smooth coefficient functions.
    """
    n, T = Y.shape
    # Pointwise OLS: beta_t = (X'X)^-1 X'Y[:, t]
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX + 1e-6 * np.eye(X.shape[1]))
    beta_raw = XtX_inv @ X.T @ Y                     # (p, T)
    # Smooth each row via P-splines
    Phi = bspline_basis(t, n_knots)
    R = _diff_pen(Phi.shape[1], 2)
    A = Phi.T @ Phi + lam * R
    beta_smooth = np.zeros_like(beta_raw)
    for j in range(beta_raw.shape[0]):
        c = np.linalg.solve(A, Phi.T @ beta_raw[j])
        beta_smooth[j] = Phi @ c
    return beta_raw, beta_smooth


if __name__ == "__main__":
    print("=== Functional linear model (function-on-scalar) ===\n")
    rng = np.random.default_rng(0)
    n_per = 40
    T = 50
    t = np.linspace(0, 1, T)
    # Group means: baseline sinusoid + a group-2 shift function.
    alpha_true = 1.5 * np.sin(2 * np.pi * t)
    beta_true = 0.6 * (t - 0.5) * (t > 0.3)          # only turns on later in t
    n = 2 * n_per
    group = np.concatenate([np.zeros(n_per), np.ones(n_per)])
    Y = np.zeros((n, T))
    for i in range(n):
        Y[i] = alpha_true + beta_true * group[i] + 0.4 * rng.normal(0, 1, T)

    X = np.column_stack([np.ones(n), group])
    beta_raw, beta_smooth = function_on_scalar(Y, X, t, n_knots=10, lam=1.0)

    err_raw = float(np.mean((beta_raw[1] - beta_true) ** 2))
    err_smooth = float(np.mean((beta_smooth[1] - beta_true) ** 2))
    print(f"  intercept beta_0(t) recovered mean = {beta_smooth[0].mean():.3f}   (truth ~ 0.0 avg)")
    print(f"  group coefficient beta_1(t) MSE:")
    print(f"    raw pointwise:   {err_raw:.4f}")
    print(f"    P-spline smooth: {err_smooth:.4f}"
          f"   ({100 * (err_raw - err_smooth) / err_raw:.0f}% reduction)")
    print(f"\n  Coefficient value at t=0.8 (should be positive):   beta_smooth = {beta_smooth[1, int(0.8*T)]:.3f}"
          f"   truth = {beta_true[int(0.8*T)]:.3f}")
    print(f"  Coefficient value at t=0.1 (should be near 0):     beta_smooth = {beta_smooth[1, int(0.1*T)]:.3f}"
          f"   truth = {beta_true[int(0.1*T)]:.3f}\n")

    print("--- library cross-check (R fda::fRegress; refund::fosr; Python scikit-fda) ---")
