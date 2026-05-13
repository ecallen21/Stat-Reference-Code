"""Splines and segmented (piecewise linear) regression (Reference §5.4, §5.22, §5.26, §5.34).

Flexible alternatives to polynomial regression. The key idea: choose a few
KNOTS, fit local polynomial pieces between them, and constrain the pieces to
join smoothly. The columns of the corresponding design matrix are the spline
BASIS FUNCTIONS, and the rest is just OLS.

What this file implements
-------------------------
1. PIECEWISE LINEAR (segmented) basis with K known breakpoints:
       [1, x, max(0, x - k_1), max(0, x - k_2), ..., max(0, x - k_K)]
   Coefficients are the slope changes at each knot.

2. NATURAL CUBIC SPLINE basis (Hastie & Tibshirani, ESL Ch. 5):
       N_0(x) = 1
       N_1(x) = x
       N_{j+1}(x) = d_j(x) - d_{K-1}(x)    for j = 1, ..., K-2
       d_j(x) = ( (x - k_j)_+^3 - (x - k_K)_+^3 ) / (k_K - k_j)
   These force LINEAR behavior in the tails (beyond the first and last knot)
   -- the main reason to prefer them over plain cubic splines.

3. A SMALL BREAKPOINT-SEARCH helper for "segmented" regression with an
   UNKNOWN breakpoint (Muggeo's idea, simplified): grid-search the knot
   location to minimize RSS.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


# ---------- piecewise-linear (segmented) basis -----------------------------
def piecewise_linear_basis(x, knots) -> np.ndarray:
    """Design matrix [1, x, (x - k1)_+, ..., (x - kK)_+]."""
    x = np.asarray(x, dtype=float)
    cols = [np.ones_like(x), x]
    for k in knots:
        cols.append(np.maximum(0.0, x - k))
    return np.column_stack(cols)


def fit_piecewise_linear(x, y, knots):
    X = piecewise_linear_basis(x, knots)
    beta, *_ = np.linalg.lstsq(X, np.asarray(y), rcond=None)
    yhat = X @ beta
    return {"beta": beta.tolist(), "knots": list(knots),
            "fitted": yhat.tolist(), "rss": float(((y - yhat) ** 2).sum())}


def search_breakpoint(x, y, n_grid: int = 50):
    """Find the single breakpoint minimizing RSS for a two-segment piecewise-linear fit."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    candidates = np.linspace(np.quantile(x, 0.1), np.quantile(x, 0.9), n_grid)
    best = (math.inf, None, None)
    for k in candidates:
        res = fit_piecewise_linear(x, y, knots=[k])
        if res["rss"] < best[0]:
            best = (res["rss"], k, res)
    return {"breakpoint": best[1], "rss": best[0], "fit": best[2]}


# ---------- natural cubic spline basis -------------------------------------
def natural_cubic_basis(x, knots) -> np.ndarray:
    """Natural cubic spline basis with K knots; K columns total (including intercept and linear).

    Follows Hastie, Tibshirani & Friedman, ESL (5.4).
    """
    x = np.asarray(x, dtype=float)
    knots = np.asarray(sorted(knots), dtype=float)
    K = len(knots)
    if K < 3:
        raise ValueError("need at least 3 knots for a natural cubic spline")
    def d(j):
        kj = knots[j]; kK = knots[K - 1]
        return (np.maximum(0.0, x - kj) ** 3 - np.maximum(0.0, x - kK) ** 3) / (kK - kj)
    cols = [np.ones_like(x), x]
    dK1 = d(K - 2)
    for j in range(K - 2):
        cols.append(d(j) - dK1)
    return np.column_stack(cols)


def fit_natural_cubic_spline(x, y, knots) -> dict:
    """Natural cubic spline regression via OLS on the spline basis."""
    X = natural_cubic_basis(x, knots)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    n, p = X.shape
    rss = float(((y - yhat) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    return {"beta": beta.tolist(), "knots": list(knots),
            "fitted": yhat.tolist(), "rss": rss,
            "df": p, "r_squared": 1 - rss / tss}


def library_versions(x, y, n_knots: int = 5):
    out = {}
    try:
        from patsy import dmatrix
        import statsmodels.api as sm
        df_design = dmatrix(f"cr(x, df={n_knots})", {"x": x}, return_type="dataframe")
        res = sm.OLS(y, df_design).fit()
        out["patsy/statsmodels cr() basis params"] = res.params.tolist()
        out["patsy/statsmodels cr() rsquared"] = float(res.rsquared)
    except Exception as exc:
        out["patsy"] = f"not used ({exc})"
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(8)
    n = 200
    x = np.linspace(0, 10, n)
    # Two regimes: slope 1 below x=4, slope -0.5 above
    y_true = np.where(x < 4, 0.5 + 1.0 * x, 4.5 - 0.5 * (x - 4))
    y = y_true + rng.normal(0, 0.3, n)

    print("=== Piecewise linear with KNOWN knot at x = 4 ===")
    res = fit_piecewise_linear(x, y, knots=[4])
    print(f"  coefficients: {[round(b, 3) for b in res['beta']]}")
    print(f"  slope segment 1 (x < 4)        = beta[1]              = {res['beta'][1]:.4f}")
    print(f"  slope segment 2 (x > 4)        = beta[1] + beta[2]    = "
          f"{res['beta'][1] + res['beta'][2]:.4f}")
    print(f"  RSS = {res['rss']:.4f}")

    print("\n=== Breakpoint search (unknown knot) ===")
    bk = search_breakpoint(x, y)
    print(f"  estimated breakpoint = {bk['breakpoint']:.3f}   RSS = {bk['rss']:.4f}")

    print("\n=== Natural cubic spline (5 knots at quantiles) ===")
    knots = np.quantile(x, [0.1, 0.3, 0.5, 0.7, 0.9])
    ncs = fit_natural_cubic_spline(x, y, knots=knots)
    print(f"  knots = {knots.tolist()}")
    print(f"  R^2 = {ncs['r_squared']:.4f}   df = {ncs['df']}")
    print(f"  coefficient vector length = {len(ncs['beta'])}")

    print("\n--- library ---")
    for k, v in library_versions(x, y, n_knots=5).items():
        print(f"  {k}: {v}")
