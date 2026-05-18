"""Mann-Whitney U test / Wilcoxon rank-sum test (Reference §6.3).

H0: the two distributions are the same (equivalently, that P(X1 > X2) = 0.5).

Algorithm
---------
1. Combine the two samples; rank everything together (ties get average ranks).
2. R1 = sum of ranks for sample 1.
3. U1 = R1 - n1 (n1 + 1) / 2                       # rank sum minus baseline
   U2 = n1 n2 - U1
4. Normal approximation:
       E[U1] = n1 n2 / 2
       Var[U1] = n1 n2 (n1 + n2 + 1) / 12 - tie correction
       z = (U1 - E[U1]) / sqrt(Var[U1])  with continuity correction

Nonparametric alternative to the two-sample t-test. Assumes:
  - independent samples
  - same shape under H0 (or interpret as "stochastic dominance" otherwise)
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _average_ranks_combined(x1, x2):
    combined = list(x1) + list(x2)
    pairs = sorted(range(len(combined)), key=lambda i: combined[i])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and combined[pairs[j + 1]] == combined[pairs[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[pairs[k]] = avg
        i = j + 1
    return ranks


def mann_whitney(x1: Sequence[float], x2: Sequence[float],
                 alternative: str = "two-sided") -> dict:
    """Mann-Whitney U test with the normal approximation.

    Returns U1 (relative to sample 1), R1, R2, z, p, and the
    rank-biserial correlation (= 2 U1 / (n1 n2) - 1; see `techniques/effect-sizes`).
    """
    n1, n2 = len(x1), len(x2); n = n1 + n2
    ranks = _average_ranks_combined(x1, x2)
    R1 = sum(ranks[:n1]); R2 = sum(ranks[n1:])
    U1 = R1 - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1
    mu = n1 * n2 / 2
    # tie correction
    from collections import Counter
    tie_counts = Counter(list(x1) + list(x2)).values()
    tie_corr = sum(t ** 3 - t for t in tie_counts if t > 1) / (n * (n - 1))
    var = n1 * n2 / 12 * (n + 1 - tie_corr)
    if alternative == "two-sided":
        z = (U1 - mu - 0.5 * math.copysign(1.0, U1 - mu)) / math.sqrt(var)
        p = 2 * stats.norm.sf(abs(z))
    elif alternative == "greater":
        z = (U1 - mu - 0.5) / math.sqrt(var)
        p = stats.norm.sf(z)
    elif alternative == "less":
        z = (U1 - mu + 0.5) / math.sqrt(var)
        p = stats.norm.cdf(z)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    rank_biserial = 2 * U1 / (n1 * n2) - 1
    return {"U1": U1, "U2": U2, "R1": R1, "R2": R2,
            "n1": n1, "n2": n2, "z": z, "p_value": float(p),
            "rank_biserial": rank_biserial,
            "method": "Mann-Whitney U (normal approx)"}


def library_versions(x1, x2):
    res = stats.mannwhitneyu(x1, x2, alternative="two-sided")
    return {"scipy.stats.mannwhitneyu": (float(res.statistic), float(res.pvalue))}


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(2)
    a = rng.normal(50, 10, 30).tolist()
    b = rng.normal(55, 12, 28).tolist()

    print("=== Mann-Whitney U (a vs b) ===")
    for k, v in mann_whitney(a, b).items():
        print(f"  {k:14s}: {v}")

    print("\n--- library (scipy.stats.mannwhitneyu) ---")
    for k, v in library_versions(a, b).items():
        print(f"  {k}: {v}")
