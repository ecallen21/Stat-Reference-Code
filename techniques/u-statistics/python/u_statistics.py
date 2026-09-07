"""U-statistics (Reference Sec 46.9).

Hoeffding 1948 'A class of statistics with asymptotically normal
distribution', Ann Math Stat. A U-statistic is an unbiased estimator
of a population parameter theta = E[h(X_1, ..., X_m)] built by
averaging a symmetric kernel h of order m over all m-tuples from a
sample of size n:

    U_n = 1 / C(n, m) * sum_{i1 < i2 < ... < im} h(X_i1, ..., X_im)

Classical examples:
    m = 1, h(x)          = x                -> sample mean
    m = 2, h(x, y)       = (x - y)^2 / 2    -> sample variance
    m = 2, h(x, y)       = |x - y| / 2      -> Gini mean difference / 2
    m = 3, h(x, y, z)    = 1[y between x and z] -> tests of trend

Hoeffding's variance formula:

    Var(U_n) = m^2 / n * sigma_1^2 + O(n^-2)
    sigma_1^2 = Var(h_1(X_1)) with h_1(x) = E[h(x, X_2, ..., X_m)].

Under standard regularity conditions:

    sqrt(n) * (U_n - theta) --d--> Normal(0, m^2 * sigma_1^2).

We implement U-statistics for mean, variance, and Gini mean difference,
plus the Hoeffding projection variance and a bootstrap check.
"""
from __future__ import annotations    # stdlib

from itertools import combinations    # kernel enumeration

import numpy as np    # numerical arrays


def u_stat(x, kernel, order):
    """Exact U-statistic via combinations. O(n^m) -- use for small n."""
    n = len(x)
    vals = [kernel(*[x[i] for i in tup]) for tup in combinations(range(n), order)]
    return float(np.mean(vals))


def u_stat_fast_mean(x):
    return float(np.mean(x))


def u_stat_fast_variance(x):
    #  h(x, y) = (x - y)^2 / 2; U = sample variance with (n-1) denom
    n = len(x)
    return float(np.var(x, ddof=1))


def u_stat_gini(x):
    """Gini mean difference / 2: h(x, y) = |x - y| / 2.

    Vectorised via sort trick: sum_{i<j} |x_i - x_j| = sum_i (2 i - n + 1) x_(i).
    """
    n = len(x)
    x_sorted = np.sort(x)
    coef = 2 * np.arange(1, n + 1) - n - 1
    total_abs = float(np.sum(coef * x_sorted))
    return total_abs / (n * (n - 1))


def hoeffding_projection_var(x, kernel, order, n_mc=500, seed=0):
    """Estimate sigma_1^2 = Var(h_1(x)) by Monte Carlo over the sample."""
    rng = np.random.default_rng(seed)
    n = len(x)
    h1 = np.zeros(n)
    for i in range(n):
        vals = []
        for _ in range(n_mc):
            others = rng.choice(np.delete(np.arange(n), i), size=order - 1, replace=False)
            vals.append(kernel(x[i], *x[others]))
        h1[i] = np.mean(vals)
    sigma1_sq = np.var(h1, ddof=1)
    return sigma1_sq


if __name__ == "__main__":
    print("=== U-statistics ===\n")
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=400)

    print(f"  U-statistic (mean)     = {u_stat_fast_mean(x):+.4f}   (truth 0.00)")
    print(f"  U-statistic (variance) = {u_stat_fast_variance(x):+.4f}  (truth 1.00)")

    G_true = 2 / np.sqrt(np.pi)    # Gini mean difference for N(0, 1)
    print(f"  Gini mean difference   = {u_stat_gini(x) * 2:+.4f}    (truth {G_true:.4f})")

    #  Hoeffding projection variance for the variance U-statistic
    def h_var(xx, yy):
        return 0.5 * (xx - yy) ** 2

    sigma1_sq = hoeffding_projection_var(x, h_var, order=2, n_mc=200)
    var_U = 4 / len(x) * sigma1_sq        # m^2 / n * sigma_1^2, m=2
    print(f"\n  Variance-U Hoeffding projection var = {sigma1_sq:.4f}")
    print(f"  Approx SE(U_var)                     = {np.sqrt(var_U):.4f}")

    #  Bootstrap SE for the variance U-statistic
    B = 800
    b = np.zeros(B)
    for k in range(B):
        idx = rng.integers(0, len(x), size=len(x))
        b[k] = np.var(x[idx], ddof=1)
    print(f"  Bootstrap SE(U_var)                  = {np.std(b, ddof=1):.4f}")

    print("\n--- library cross-check (Ustat R; from-scratch Python) ---")
