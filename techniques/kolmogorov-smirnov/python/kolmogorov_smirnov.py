"""Kolmogorov-Smirnov Test (Reference §6.7).

Distribution-equality tests built on the SUP NORM of the empirical CDF
difference.

  ONE-SAMPLE   H0: F = F0 (a fully specified target distribution).
       D_n = sup_x | F_n(x) - F_0(x) |
       The exact distribution of D_n under H0 is tabulated; here we use the
       Kolmogorov asymptotic limit and a finite-n approximation.

  TWO-SAMPLE   H0: F_X = F_Y.
       D_{m,n} = sup_x | F_X^m(x) - F_Y^n(x) |
       Under H0, sqrt(m n / (m + n)) D ~ Kolmogorov distribution.

Both versions are CONSISTENT against any difference in distribution, but
they have lower power against specific alternatives than tests tailored to
those (e.g. K-S has less power against pure shift than Mann-Whitney).

Caveats with fitted parameters
------------------------------
If F_0 has parameters estimated from the data (e.g. normal with sample mean
and SD), the standard K-S p-value is INVALID -- it's too conservative.
Use Lilliefors (`techniques/normality-tests`) for that case.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import bisect    # stdlib: binary search on a sorted list (O(log n) insertion-point)
import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Callable, Sequence    # stdlib: type hints (Callable = function; Sequence = indexable iterable)

from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def one_sample_ks(x: Sequence[float], cdf: Callable[[float], float]) -> dict:
    """One-sample K-S against a fully specified continuous CDF ``cdf``."""
    x = sorted(x); n = len(x)
    # F_n jumps by 1/n at each x_i; check both sides of each jump
    d_plus = max((i + 1) / n - cdf(v) for i, v in enumerate(x))
    d_minus = max(cdf(v) - i / n for i, v in enumerate(x))
    D = max(d_plus, d_minus)
    # scipy uses the exact small-n distribution; we use the asymptotic limit
    p = float(stats.kstwobign.sf(math.sqrt(n) * D))
    return {"D": D, "D_plus": d_plus, "D_minus": d_minus, "n": n,
            "p_value_asymptotic": p, "method": "1-sample KS"}


def two_sample_ks(x: Sequence[float], y: Sequence[float]) -> dict:
    """Two-sample K-S: H0: F_X = F_Y."""
    sx = sorted(x); sy = sorted(y); m = len(sx); n = len(sy)
    # Walk through all jumps from either ECDF
    all_pts = sorted(set(sx + sy))
    d = 0.0
    for v in all_pts:
        fx = bisect.bisect_right(sx, v) / m
        fy = bisect.bisect_right(sy, v) / n
        d = max(d, abs(fx - fy))
    en = math.sqrt(m * n / (m + n))
    p = float(stats.kstwobign.sf(en * d))
    return {"D": d, "m": m, "n": n,
            "p_value_asymptotic": p, "method": "2-sample KS"}


def library_versions(x, cdf_name, params, x2):
    res1 = stats.kstest(x, cdf_name, args=params)
    res2 = stats.ks_2samp(x, x2)
    return {"scipy.stats.kstest (1-sample)": (float(res1.statistic), float(res1.pvalue)),
            "scipy.stats.ks_2samp": (float(res2.statistic), float(res2.pvalue))}


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(5)

    print("=== One-sample K-S: is x ~ N(0,1)? ===")
    x = rng.normal(0, 1, 200).tolist()
    res = one_sample_ks(x, lambda v: stats.norm.cdf(v))
    for k, v in res.items():
        print(f"  {k:18s}: {v}")

    print("\n=== Two-sample K-S: do x and y come from the same distribution? ===")
    y = rng.normal(0.4, 1.0, 180).tolist()
    res2 = two_sample_ks(x, y)
    for k, v in res2.items():
        print(f"  {k:18s}: {v}")

    print("\n--- library (scipy) ---")
    for k, v in library_versions(x, "norm", (0, 1), y).items():
        print(f"  {k}: {v}")
