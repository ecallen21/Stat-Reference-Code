"""Functional basis smoothing (Reference Sec 31.1).

Ramsay-Silverman FDA (2005), Ch 4-5.

Represent a curve x(t) in a BASIS  x(t) = sum_j c_j phi_j(t). Common
choices:
  * FOURIER (periodic data).
  * B-SPLINES (smooth, local support).
  * WAVELETS (non-stationary features).

Smoothing balances DATA FIT vs SMOOTHNESS via a roughness penalty:

  min_c  sum_i (y_i - phi(t_i)' c)^2  +  lambda * int (D^2 x(t))^2 dt.

Solved as penalised least squares: c_hat = (Phi' Phi + lambda R)^-1 Phi' y,
with R the second-difference matrix on basis coefficients (P-splines,
Eilers-Marx 1996).

Here we implement B-spline P-spline smoothing on a noisy synthetic curve
and show the lambda-CV trade-off.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def bspline_basis(t, n_knots=15, degree=3):
    knots = np.linspace(t.min(), t.max(), n_knots)
    cols = [np.ones_like(t), t, t ** 2, t ** 3]
    for k in knots[1:-1]:
        cols.append(np.maximum(0.0, t - k) ** degree)
    return np.stack(cols, axis=1)


def _difference_penalty(d, order=2):
    D = np.diff(np.eye(d), n=order, axis=0)
    return D.T @ D


def fit_psplines(t, y, n_knots=15, lam=1.0, order=2):
    Phi = bspline_basis(t, n_knots)
    p = Phi.shape[1]
    R = _difference_penalty(p, order)
    A = Phi.T @ Phi + lam * R
    return np.linalg.solve(A, Phi.T @ y), Phi


def loo_cv_mse(t, y, lam_grid, n_knots=15):
    """Leave-one-out CV mean squared error over lambda grid."""
    Phi = bspline_basis(t, n_knots)
    p = Phi.shape[1]
    R = _difference_penalty(p, 2)
    scores = []
    for lam in lam_grid:
        H = Phi @ np.linalg.solve(Phi.T @ Phi + lam * R, Phi.T)
        h_ii = np.diag(H)
        resid = y - H @ y
        cv = float(np.mean((resid / (1 - h_ii)) ** 2))
        scores.append(cv)
    return np.array(scores)


if __name__ == "__main__":
    print("=== Functional basis smoothing (P-splines) ===\n")
    rng = np.random.default_rng(0)
    n = 120
    t = np.linspace(0, 1, n)
    truth = np.sin(2 * np.pi * t) + 0.5 * t
    y = truth + 0.3 * rng.normal(0, 1, n)

    lam_grid = np.logspace(-4, 2, 12)
    cv = loo_cv_mse(t, y, lam_grid)
    best_lam = float(lam_grid[cv.argmin()])
    print(f"  CV-optimal lambda: {best_lam:.4f}   CV score = {cv.min():.4f}")

    c, Phi = fit_psplines(t, y, lam=best_lam)
    y_smooth = Phi @ c
    mse = float(np.mean((y_smooth - truth) ** 2))
    mse_raw = float(np.mean((y - truth) ** 2))
    print(f"  raw       MSE vs truth: {mse_raw:.4f}")
    print(f"  smoothed  MSE vs truth: {mse:.4f}"
          f"   ({100 * (mse_raw - mse) / mse_raw:.0f}% reduction)")

    print("\n--- library cross-check (R fda::smooth.basis; mgcv::gam s(t,bs='ps'); Python scikit-fda) ---")
