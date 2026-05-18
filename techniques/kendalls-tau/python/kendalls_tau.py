"""Kendall's tau (Reference §4.3).

A rank correlation measured directly from pairs of observations:

  - Concordant pair (i, j): (x_i - x_j) and (y_i - y_j) have the SAME sign.
  - Discordant pair       : OPPOSITE signs.
  - Tied pair             : either delta is 0.

  C = #concordant,  D = #discordant
  T_x = #pairs tied in x only,  T_y = #pairs tied in y only,
  T_xy = #pairs tied in BOTH x and y

  tau-a = (C - D) / (n(n-1)/2)                          # ignores ties
  tau-b = (C - D) / sqrt((n0 - T_x)(n0 - T_y))           # corrects for ties
            with n0 = n(n-1)/2

Significance test for tau (no ties): z = tau / SE,
    SE_tau = sqrt(2(2n + 5) / (9n(n - 1)))
and we use the same z form for tau-b as an approximation (matches scipy).

Why use tau over Spearman?
  - Direct interpretation as P(concordant) - P(discordant).
  - More robust to extreme outliers and to small samples.
  - Slower for very large n (O(n^2) by definition, though O(n log n) algorithms exist).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _kendall_counts(x: Sequence[float], y: Sequence[float]):
    """Return (C, D, T_x_only, T_y_only, T_both, n)."""
    n = len(x)
    C = D = Tx = Ty = Txy = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                Txy += 1
            elif dx == 0:
                Tx += 1
            elif dy == 0:
                Ty += 1
            elif (dx > 0) == (dy > 0):
                C += 1
            else:
                D += 1
    return C, D, Tx, Ty, Txy, n


def kendall_tau_a(x: Sequence[float], y: Sequence[float]) -> float:
    """tau-a = (C - D) / (n(n-1)/2). Ignores ties (use only when there are none)."""
    C, D, *_ , n = _kendall_counts(x, y)
    return (C - D) / (n * (n - 1) / 2)


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """tau-b = (C - D) / sqrt((n0 - T_x)(n0 - T_y)). Handles ties correctly."""
    C, D, Tx, Ty, _, n = _kendall_counts(x, y)
    n0 = n * (n - 1) / 2
    denom = math.sqrt((n0 - Tx) * (n0 - Ty))
    return (C - D) / denom if denom > 0 else float("nan")


def kendall_test(x: Sequence[float], y: Sequence[float],
                 alternative: str = "two-sided") -> dict:
    """Kendall's tau-b with the normal-approximation z-test (matches scipy)."""
    C, D, Tx, Ty, Txy, n = _kendall_counts(x, y)
    tau = kendall_tau_b(x, y)
    # Variance of (C - D) under H0 -- exact form with tie correction would be
    # heavier; the simple no-ties form usually suffices.
    var = n * (n - 1) * (2 * n + 5) / 18.0
    if var <= 0:
        z = float("nan"); p = float("nan")
    else:
        z = (C - D) / math.sqrt(var)
        if alternative == "two-sided":
            p = 2 * stats.norm.sf(abs(z))
        elif alternative == "greater":
            p = float(stats.norm.sf(z))
        else:
            p = float(stats.norm.cdf(z))
    return {"tau_b": tau, "concordant": C, "discordant": D,
            "tied_x": Tx, "tied_y": Ty, "tied_both": Txy,
            "z": z, "p_value": float(p),
            "method": "Kendall tau-b", "alternative": alternative}


def library_versions(x, y):
    res_b = stats.kendalltau(x, y, variant="b")
    return {"scipy.stats.kendalltau (tau-b)": (float(res_b.statistic), float(res_b.pvalue))}


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(7)
    n = 50
    x = rng.normal(0, 1, n)
    y = 0.7 * x + math.sqrt(1 - 0.49) * rng.normal(0, 1, n)
    print(f"=== Kendall on n = {n} (no ties) ===")
    print(f"  tau-a = {kendall_tau_a(x.tolist(), y.tolist()):.4f}")
    for k, v in kendall_test(x.tolist(), y.tolist()).items():
        print(f"  {k:14s}: {v}")

    print("\n=== With ties (ordinal categories 1..5) ===")
    xt = rng.integers(1, 6, 60).tolist(); yt = rng.integers(1, 6, 60).tolist()
    for k, v in kendall_test(xt, yt).items():
        print(f"  {k:14s}: {v}")

    print("\n--- library (scipy.stats.kendalltau) ---")
    for k, v in library_versions(x, y).items():
        print(f"  {k}: {v}")
