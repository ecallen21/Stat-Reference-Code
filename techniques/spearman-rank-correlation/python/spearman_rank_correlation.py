"""Spearman rank correlation (Reference §4.2).

Spearman's rho is Pearson's r applied to the *ranks* of x and y. It measures
any MONOTONIC association (not just linear) and is robust to outliers and to
non-linear-but-monotonic transformations.

When all ranks are distinct (no ties):
    rho = 1 - 6 * sum(d_i^2) / (n (n^2 - 1)),    d_i = rank(x_i) - rank(y_i)

With ties, use the general formula r_pearson(rank(x), rank(y)) -- which is
what this implementation does (it works in both cases).

Significance: same t-test as for Pearson, applied to the ranks.
CI: Fisher z on the rank correlation, with a slight variance correction
(SE(z) approx 1.03 / sqrt(n-3); we use the plain 1/sqrt(n-3) as a simple default
matching most software).
"""
from __future__ import annotations

import math
from typing import Sequence

from scipy import stats


def _rank_average(x: Sequence[float]):
    """Average ranks (ties get the mean of the ranks they span)."""
    n = len(x)
    pairs = sorted(range(n), key=lambda i: x[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[pairs[j + 1]] == x[pairs[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # ranks are 1-based; average of i+1 .. j+1
        for k in range(i, j + 1):
            ranks[pairs[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rho via Pearson r on average ranks."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    rx = _rank_average(x); ry = _rank_average(y)
    mx = sum(rx) / len(rx); my = sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy)


def spearman_test(x: Sequence[float], y: Sequence[float],
                  alternative: str = "two-sided", conf: float = 0.95) -> dict:
    """Spearman's rho with t-test and Fisher-z CI."""
    n = len(x)
    r = spearman_rho(x, y)
    if abs(r) == 1.0:
        return {"rho": r, "p_value": 0.0, "ci_lower": r, "ci_upper": r,
                "method": "Spearman"}
    t = r * math.sqrt((n - 2) / (1 - r * r))
    df = n - 2
    if alternative == "two-sided":
        p = 2 * stats.t.sf(abs(t), df)
    elif alternative == "greater":
        p = float(stats.t.sf(t, df))
    else:
        p = float(stats.t.cdf(t, df))
    z = math.atanh(r); se = 1 / math.sqrt(n - 3)
    zc = stats.norm.ppf(0.5 + conf / 2)
    return {"rho": r, "t": t, "df": df, "p_value": float(p),
            "ci_lower": math.tanh(z - zc * se),
            "ci_upper": math.tanh(z + zc * se),
            "method": "Spearman", "alternative": alternative}


def library_versions(x, y):
    res = stats.spearmanr(x, y)
    return {"scipy.stats.spearmanr": (float(res.statistic), float(res.pvalue))}


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(6)
    n = 60
    # Strong but nonlinear monotonic relationship: y = exp(x) + noise
    x = rng.normal(0, 1, n)
    y = np.exp(x) + rng.normal(0, 0.3, n)

    print(f"=== Spearman on n = {n} (monotonic, nonlinear) ===")
    for k, v in spearman_test(x.tolist(), y.tolist()).items():
        print(f"  {k:10s}: {v}")

    # Pearson for contrast (using a fresh inline implementation, no path tricks)
    mx = float(np.mean(x)); my = float(np.mean(y))
    sxy = float(np.sum((x - mx) * (y - my)))
    sxx = float(np.sum((x - mx) ** 2)); syy = float(np.sum((y - my) ** 2))
    pearson_r = sxy / math.sqrt(sxx * syy)
    print(f"\nFor contrast, Pearson r on the same (monotonic, nonlinear) data: {pearson_r:.4f}")
    print("(Spearman captures the monotone relationship; Pearson misses the curvature.)")

    print("\n--- library (scipy.stats.spearmanr) ---")
    for k, v in library_versions(x, y).items():
        print(f"  {k}: {v}")
