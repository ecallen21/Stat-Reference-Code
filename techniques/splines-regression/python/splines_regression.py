"""Spline regression: cubic + natural + B-spline (Reference §5.12).

Approximate f(x) with piecewise cubic polynomials joined at KNOTS so that
the function and its first two derivatives are continuous.  Three common
basis choices:

1) Truncated-power cubic
    B(x) = (1, x, x^2, x^3, (x - k_1)_+^3, ..., (x - k_K)_+^3)
    Simple, numerically ill-conditioned for large K.

2) Natural cubic spline
    Constrained to be LINEAR beyond the boundary knots.  Fewer degrees of
    freedom and much better extrapolation than a plain cubic.

3) B-spline basis (de Boor)
    Local support, numerically stable, standard for GAMs.  scipy.interpolate
    exposes BSpline / splrep.

Fit each by OLS (unpenalized) or ridge (penalized -> smoothing splines).
See gam/ for the penalized version with automatic smoothing.

Trade-offs vs GAM
    Splines with fixed K knots are simple and interpretable but require
    picking K.  GAMs pick K large and use a penalty to control smoothness.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def cubic_spline_basis(x, knots):
    """Truncated-power cubic basis: [1, x, x^2, x^3, (x - k_1)_+^3, ...]."""
    x = np.asarray(x, dtype=float); k = np.asarray(knots, dtype=float)
    cols = [np.ones_like(x), x, x ** 2, x ** 3]
    for kj in k:
        cols.append(np.maximum(x - kj, 0) ** 3)
    return np.column_stack(cols)


def natural_cubic_spline_basis(x, knots):
    """Natural cubic spline basis (Hastie-Tibshirani-Friedman 2009, eq. 5.5).

    Uses K-2 basis functions in addition to the intercept and linear term:
        N_1(x) = 1, N_2(x) = x
        N_{k+2}(x) = d_k(x) - d_{K-1}(x)
        d_k(x) = ((x - k_k)_+^3 - (x - k_K)_+^3) / (k_K - k_k)
    """
    x = np.asarray(x, dtype=float); k = np.sort(np.asarray(knots, dtype=float))
    K = len(k)
    if K < 2: raise ValueError("need at least 2 knots")
    cols = [np.ones_like(x), x]
    def d(j):
        return (np.maximum(x - k[j], 0) ** 3 - np.maximum(x - k[K - 1], 0) ** 3) / (k[K - 1] - k[j])
    for j in range(K - 2):
        cols.append(d(j) - d(K - 2))
    return np.column_stack(cols)


def fit_spline_regression(x, y, knots, basis: str = "cubic") -> dict:
    """OLS on the chosen spline basis."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if basis == "cubic":
        B = cubic_spline_basis(x, knots)
    elif basis == "natural":
        B = natural_cubic_spline_basis(x, knots)
    elif basis == "bspline":
        from scipy.interpolate import BSpline
        # Clamped cubic B-spline: augment interior knots with 4 boundary knots on each side
        k_int = np.asarray(knots, dtype=float)
        lo, hi = float(x.min()), float(x.max())
        t = np.concatenate([[lo] * 4, k_int, [hi] * 4])
        n_basis = len(t) - 4          # degree + 1 = 4 boundary augmentation on each side
        B = np.zeros((len(x), n_basis))
        for i in range(n_basis):
            c = np.zeros(n_basis); c[i] = 1
            B[:, i] = BSpline(t, c, 3, extrapolate=False)(x)
        B = np.nan_to_num(B, nan=0.0)
    else:
        raise ValueError("basis must be 'cubic', 'natural', or 'bspline'")
    beta, *_ = np.linalg.lstsq(B, y, rcond=None)
    y_hat = B @ beta
    return {"beta": beta, "fitted": y_hat,
            "basis": basis, "knots": np.asarray(knots),
            "df": int(B.shape[1]),
            "rss": float(np.sum((y - y_hat) ** 2)),
            "method": f"Spline regression ({basis}, unpenalized OLS)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    x = np.sort(rng.uniform(-3, 3, n))
    f_true = np.sin(1.5 * x) + 0.3 * x
    y = f_true + rng.normal(0, 0.3, n)
    knots = np.quantile(x, np.linspace(0.1, 0.9, 6))

    for basis in ("cubic", "natural", "bspline"):
        r = fit_spline_regression(x, y, knots=knots, basis=basis)
        rmse = math.sqrt(np.mean((r["fitted"] - f_true) ** 2))
        print(f"=== {basis:8s}: df = {r['df']:2d}, RMSE vs truth = {rmse:.4f}")

    print("\n=== Comparison to polynomial regression ===")
    Xpoly = np.column_stack([x ** k for k in range(1, 7)])
    Xpoly = np.column_stack([np.ones_like(x), Xpoly])
    b, *_ = np.linalg.lstsq(Xpoly, y, rcond=None)
    y_hat = Xpoly @ b
    rmse = math.sqrt(np.mean((y_hat - f_true) ** 2))
    print(f"  degree-6 polynomial: RMSE vs truth = {rmse:.4f}")

    print("\n--- library cross-check (scipy.interpolate.UnivariateSpline) ---")
    try:
        from scipy.interpolate import UnivariateSpline
        spl = UnivariateSpline(x, y, k=3, s=0.5)
        rmse = math.sqrt(np.mean((spl(x) - f_true) ** 2))
        print(f"  scipy UnivariateSpline: RMSE vs truth = {rmse:.4f}")
    except Exception as ex:
        print(f"  (scipy unavailable: {ex})")
