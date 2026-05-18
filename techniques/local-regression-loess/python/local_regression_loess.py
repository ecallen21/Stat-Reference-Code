"""Local regression / LOESS (Reference §6.22).

At each target point x_0, fit a low-degree polynomial (typically degree 1 or 2)
to the data inside a neighborhood, weighted by distance to x_0. Predict
y_hat(x_0) from that local fit.

  weight w_i = K( (x_i - x_0) / h(x_0) )      with K typically the tricube kernel:
             tricube(u) = (1 - |u|^3)^3 for |u| <= 1, else 0
  h(x_0) chosen so the neighborhood holds a SPAN fraction of the data
         (e.g. span = 0.5 -> half the data go into each local fit).

Output is a smooth curve. Robust LOWESS adds an iteratively reweighted step
that downweights outliers; we expose the non-robust version (the more common
default for smoothing scatterplots).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _tricube(u):
    return np.where(np.abs(u) < 1, (1 - np.abs(u) ** 3) ** 3, 0.0)


def loess_fit(x: Sequence[float], y: Sequence[float], x_grid: Sequence[float],
              span: float = 0.5, degree: int = 1) -> dict:
    """LOESS fit at each point of ``x_grid``.

    Parameters
    ----------
    x, y : data points.
    x_grid : where to evaluate the smooth curve.
    span : fraction of x-data in each local neighborhood, in (0, 1].
    degree : 1 (locally linear) or 2 (locally quadratic).
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = x.size
    q = max(2, int(math.ceil(span * n)))
    out = np.zeros(len(x_grid))
    for i, x0 in enumerate(x_grid):
        d = np.abs(x - x0)
        # Bandwidth so q closest points are inside [0, 1]
        h = max(np.partition(d, min(q - 1, n - 1))[q - 1], 1e-15)
        w = _tricube(d / h)
        # Local polynomial fit
        cols = [np.ones_like(x)]
        for p in range(1, degree + 1):
            cols.append((x - x0) ** p)
        X_loc = np.column_stack(cols)
        # Weighted normal equations
        WX = X_loc * w[:, None]
        beta, *_ = np.linalg.lstsq(WX.T @ X_loc, WX.T @ y, rcond=None)
        out[i] = beta[0]      # intercept at x_0
    return {"x_grid": list(x_grid), "y_smoothed": out.tolist(),
            "span": span, "degree": degree}


def library_versions(x, y, x_grid):
    out = {}
    try:
        import statsmodels.api as sm
        # statsmodels.nonparametric.lowess returns sorted (x, smoothed_y)
        sm_out = sm.nonparametric.lowess(y, x, frac=0.5, return_sorted=True)
        out["statsmodels lowess (first 5 (x, y))"] = sm_out[:5].tolist()
    except Exception as exc:
        out["statsmodels"] = f"error: {exc}"
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(9)
    x = np.linspace(0, 10, 200)
    y = np.sin(x) + 0.3 * x + rng.normal(0, 0.4, 200)
    grid = np.linspace(0, 10, 30)

    print("=== LOESS, span = 0.3 (more local), degree = 1 ===")
    res = loess_fit(x, y, grid, span=0.3, degree=1)
    for xg, ys in zip(res["x_grid"][:10], res["y_smoothed"][:10]):
        print(f"  x = {xg:5.2f}  smoothed y = {ys:+.4f}")

    print("\n=== LOESS, span = 0.7 (smoother), degree = 2 ===")
    res2 = loess_fit(x, y, grid, span=0.7, degree=2)
    for xg, ys in zip(res2["x_grid"][:10], res2["y_smoothed"][:10]):
        print(f"  x = {xg:5.2f}  smoothed y = {ys:+.4f}")

    print("\n--- library (statsmodels lowess) ---")
    for k, v in library_versions(x, y, grid).items():
        print(f"  {k}: {v}")
