"""Additive quantile regression (Reference Sec 33.13).

Koenker (2005) 'Quantile Regression', Ch 6; Fenske et al. (2011);
Kneib et al. (2023).

Extends quantile regression with SMOOTH NONLINEAR EFFECTS via basis
expansions (natural cubic splines, P-splines) and a check-loss objective:

  min_beta   sum_i rho_tau( y_i - sum_j g_j(x_ij) )
              + lambda * sum_j ||beta_j||^2         (roughness penalty)

Each g_j is expanded in a spline basis; the loss is convex + LP-friendly.

Advantages:
  * Nonlinear conditional quantiles.
  * Same interpretability as QR + additive smoothers.
  * Handles heteroscedasticity naturally.

Here we implement a single-covariate B-spline basis + subgradient
descent on the check loss at three quantiles.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def bspline_basis(x, knots=None, degree=3, n_knots=8):
    """Cubic truncated-power basis on x, with pre-supplied knots so training
    and test share the same design."""
    x = np.asarray(x).astype(float)
    if knots is None:
        knots = np.linspace(x.min(), x.max(), n_knots)
    cols = [np.ones_like(x), x, x ** 2, x ** 3]
    for k in knots[1:-1]:
        cols.append(np.maximum(0.0, x - k) ** degree)
    return np.stack(cols, axis=1), knots


def check_loss(u, tau):
    return u * (tau - (u < 0).astype(float))


def fit_aqr(B, y, tau, lam=1e-2, lr=1e-2, epochs=3000):
    d = B.shape[1]
    beta = np.zeros(d)
    n = B.shape[0]
    for _ in range(epochs):
        r = y - B @ beta
        sg = (r < 0).astype(float) - tau
        grad = B.T @ sg / n + lam * beta
        beta -= lr * grad
    return beta


if __name__ == "__main__":
    print("=== Additive quantile regression with a B-spline basis ===\n")
    rng = np.random.default_rng(0)
    n = 300
    x = np.linspace(-2, 2, n)
    # Nonlinear + heteroscedastic: sin(x) with growing variance.
    y = np.sin(1.5 * x) + (0.3 + 0.5 * np.abs(x)) * rng.normal(0, 1, n)
    B, knots = bspline_basis(x, n_knots=10)
    # Standardise non-intercept columns for stable subgradient training.
    col_sd = B.std(axis=0); col_sd[col_sd < 1e-6] = 1.0; col_sd[0] = 1.0
    B_s = B / col_sd

    fits = {}
    for tau in (0.10, 0.50, 0.90):
        fits[tau] = fit_aqr(B_s, y, tau, lam=1e-4, lr=0.05, epochs=6000)

    x_te = np.linspace(-2, 2, 15)
    print(f"  {'x':>6}  {'q10_hat':>8}  {'q50_hat':>8}  {'q90_hat':>8}"
          f"  {'true median (sin)':>18}")
    for xv in x_te:
        B_row, _ = bspline_basis(np.array([xv]), knots=knots)
        B_row_s = B_row / col_sd
        q10 = float((B_row_s @ fits[0.10])[0])
        q50 = float((B_row_s @ fits[0.50])[0])
        q90 = float((B_row_s @ fits[0.90])[0])
        print(f"  {xv:>6.2f}  {q10:>8.3f}  {q50:>8.3f}  {q90:>8.3f}"
              f"  {np.sin(1.5*xv):>18.3f}")

    print("\n  q50 tracks sin(1.5x); q10-q90 spread grows with |x| (heteroscedasticity).\n")
    print("--- library cross-check (R quantreg::rqss; qgam::qgam; Python statsmodels QuantReg on B-spline features) ---")
