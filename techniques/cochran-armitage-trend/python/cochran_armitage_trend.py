"""Cochran-Armitage trend test (Reference Sec 4.16).

Cochran 1954; Armitage 1955. Tests for a LINEAR trend in binary
outcome across K ordered categories (e.g., dose levels or exposure
tertiles). More powerful than chi-square when the alternative is
truly monotone.

Data: K groups; group i has n_i subjects, r_i events. Assign scores
x_i (typically 1, 2, ..., K or the dose value).

Test statistic:

    T = sum_i x_i * (r_i - n_i * p_bar)   /
        sqrt( p_bar * (1 - p_bar) * (sum_i n_i * x_i^2 - N * x_bar^2) )

with p_bar = sum(r_i) / N, N = sum(n_i), x_bar = sum(n_i x_i) / N.
Under H0 (no trend) T ~ Normal(0, 1). Squared T ~ Chi^2_1.

Contrast: overall chi-square tests independence but not monotonicity;
CAT is 'a chi-square directional partition'.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def cochran_armitage(r, n, x=None):
    r = np.asarray(r, dtype=float); n = np.asarray(n, dtype=float)
    K = len(r)
    if x is None:
        x = np.arange(1, K + 1, dtype=float)
    x = np.asarray(x, dtype=float)
    N = n.sum()
    p_bar = r.sum() / N
    x_bar = (n * x).sum() / N
    num = ((r - n * p_bar) * x).sum()
    denom = np.sqrt(p_bar * (1 - p_bar) * ((n * x ** 2).sum() - N * x_bar ** 2))
    z = num / denom
    p_two = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"T": float(z), "p_two_sided": float(p_two), "chi2": float(z ** 2)}


def chi_square(r, n):
    """Standard 2xK chi-square for comparison."""
    contingency = np.array([r, n - r])
    return stats.chi2_contingency(contingency, correction=False)


if __name__ == "__main__":
    print("=== Cochran-Armitage trend test ===\n")
    #  Example 1: clear monotone increase in event rate across 4 dose levels
    n = np.array([100, 100, 100, 100])
    r = np.array([10, 15, 25, 40])
    p_hat = r / n
    print(f"  Doses = 1, 2, 3, 4;  event rates = {p_hat}")
    ca = cochran_armitage(r, n)
    print(f"    Cochran-Armitage  Z = {ca['T']:+.3f}   two-sided p = {ca['p_two_sided']:.4g}")
    cs, pcs, *_ = chi_square(r, n)
    print(f"    3-df chi-square  X = {cs:.3f}   p = {pcs:.4g}")

    #  Example 2: same overall proportion but non-monotone (spike in middle)
    r2 = np.array([15, 40, 20, 15])
    p_hat = r2 / n
    print(f"\n  Non-monotone: event rates = {p_hat}")
    ca2 = cochran_armitage(r2, n)
    print(f"    Cochran-Armitage  Z = {ca2['T']:+.3f}   two-sided p = {ca2['p_two_sided']:.4g}")
    cs2, pcs2, *_ = chi_square(r2, n)
    print(f"    3-df chi-square  X = {cs2:.3f}   p = {pcs2:.4g}")

    print("\n  Cochran-Armitage picks the linear-trend signal in example 1;")
    print("  the overall chi-square is significant in both, missing the shape.")

    print("\n--- library cross-check (DescTools / prop.trend.test R; statsmodels Python) ---")
