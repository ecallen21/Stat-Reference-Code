"""Theil-Sen Slope Estimator (Reference §6.32).

A robust nonparametric estimator of the slope in y = beta_0 + beta_1 x + eps.

  beta_1_hat = median of all pairwise slopes  (y_i - y_j) / (x_i - x_j)  for i < j

  beta_0_hat = median of  y_i - beta_1_hat * x_i

Properties
  - 29.3% breakdown (vs. 0% for OLS).
  - Distribution-free: no normality assumption on eps.
  - Asymptotically normal; standard error from the median's bootstrap, or
    from inverting Kendall's tau test (Sen 1968).
  - Less efficient than OLS at the normal but vastly more robust to outliers.

We expose:
  - ``theil_sen_slope`` : point estimate
  - ``theil_sen_ci``    : CI via the Kendall-tau-based formula (Sen 1968)
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy import stats


def theil_sen_slope(x: Sequence[float], y: Sequence[float]) -> dict:
    """Theil-Sen estimate of (intercept, slope) and the pairwise slopes."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = len(x)
    slopes = []
    for i, j in combinations(range(n), 2):
        dx = x[j] - x[i]
        if dx == 0: continue
        slopes.append((y[j] - y[i]) / dx)
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))
    return {"intercept": intercept, "slope": slope,
            "n_pairs": len(slopes), "method": "Theil-Sen"}


def theil_sen_ci(x: Sequence[float], y: Sequence[float], conf: float = 0.95) -> dict:
    """Confidence interval for the slope via the Kendall-tau Sen (1968) formula."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = len(x)
    slopes = sorted((y[j] - y[i]) / (x[j] - x[i])
                    for i, j in combinations(range(n), 2)
                    if x[j] != x[i])
    N = len(slopes)
    # Var of Kendall S under H0 (no ties form):
    var_S = n * (n - 1) * (2 * n + 5) / 18
    z = stats.norm.ppf(0.5 + conf / 2)
    half = z * math.sqrt(var_S)
    M1 = max(0, int((N - half) / 2))
    M2 = min(N - 1, int((N + half) / 2))
    return {"slope_estimate": theil_sen_slope(x, y)["slope"],
            "ci_lower": slopes[M1], "ci_upper": slopes[M2],
            "conf": conf, "method": "Theil-Sen (Sen 1968 CI)"}


def library_versions(x, y):
    from scipy.stats import theilslopes
    slope, intercept, lo, hi = theilslopes(y, x, 0.95)
    return {"scipy.stats.theilslopes (slope, int, lo95, hi95)":
            (float(slope), float(intercept), float(lo), float(hi))}


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    n = 60
    x = rng.uniform(0, 10, n)
    y = 1.0 + 2.0 * x + rng.normal(0, 1.0, n)
    # Inject outliers
    y[0] = 50; y[5] = -30

    print("=== Theil-Sen on contaminated data (true slope = 2) ===")
    ts = theil_sen_slope(x.tolist(), y.tolist())
    print(f"  intercept = {ts['intercept']:+.4f}   slope = {ts['slope']:+.4f}")
    ci = theil_sen_ci(x.tolist(), y.tolist())
    print(f"  95% slope CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    # Compare to OLS
    ols_slope = float(np.cov(x, y, ddof=0)[0, 1] / x.var())
    print(f"\n  OLS slope (pulled by outliers): {ols_slope:.4f}")

    print("\n--- library (scipy.stats.theilslopes) ---")
    for k, v in library_versions(x, y).items():
        print(f"  {k}: {v}")
