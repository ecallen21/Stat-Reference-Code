"""Wilcoxon Signed-Rank Test (Reference §6.2).

Tests H0: the distribution of (x - m0) is symmetric around 0.

For paired data: apply to the differences x1 - x2.

Algorithm
---------
1. Compute d_i = x_i - m0.
2. Drop ties (d_i = 0).
3. Rank |d_i| with ties getting average ranks.
4. Sum the ranks of positive d's -> W+ (or signed sum: W = sum sign(d_i) * rank).
5. Under H0 (symmetric around 0), W+ has a known distribution; we use the
   normal approximation with continuity correction (matches scipy's default).

Stronger than the sign test (`techniques/sign-test`) because it uses the
magnitudes, but requires symmetry of the differences under H0. If you doubt
symmetry, prefer the sign test or the bootstrap.
"""
from __future__ import annotations

import math
from typing import Sequence

from scipy import stats


def _average_ranks(absvals):
    pairs = sorted(range(len(absvals)), key=lambda i: absvals[i])
    ranks = [0.0] * len(absvals)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and absvals[pairs[j + 1]] == absvals[pairs[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[pairs[k]] = avg
        i = j + 1
    return ranks


def wilcoxon_signed_rank(x: Sequence[float], m0: float = 0.0,
                         alternative: str = "two-sided") -> dict:
    """One-sample Wilcoxon signed-rank test of H0: median(x) = m0 (under symmetry)."""
    d = [v - m0 for v in x if v != m0]
    n_eff = len(d)
    if n_eff == 0:
        return {"W_plus": 0, "n_effective": 0, "p_value": 1.0,
                "method": "Wilcoxon signed-rank"}
    abs_d = [abs(v) for v in d]
    ranks = _average_ranks(abs_d)
    W_pos = sum(r for r, v in zip(ranks, d) if v > 0)
    W_neg = sum(r for r, v in zip(ranks, d) if v < 0)
    # Normal-approximation p with continuity correction
    mu = n_eff * (n_eff + 1) / 4
    # Tie correction in the variance
    from collections import Counter
    tie_counts = Counter(abs_d).values()
    tie_correction = sum(t ** 3 - t for t in tie_counts if t > 1) / 48
    var = n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24 - tie_correction
    # use W_pos as the statistic; sign-based z
    if alternative == "two-sided":
        z = (W_pos - mu - 0.5 * math.copysign(1.0, W_pos - mu)) / math.sqrt(var)
        p = 2 * stats.norm.sf(abs(z))
    elif alternative == "greater":
        z = (W_pos - mu - 0.5) / math.sqrt(var)
        p = stats.norm.sf(z)
    elif alternative == "less":
        z = (W_pos - mu + 0.5) / math.sqrt(var)
        p = stats.norm.cdf(z)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    return {"W_plus": W_pos, "W_neg": W_neg, "n_effective": n_eff,
            "z": z, "p_value": float(p),
            "method": "Wilcoxon signed-rank (normal approx)"}


def paired_wilcoxon(x1: Sequence[float], x2: Sequence[float],
                    alternative: str = "two-sided") -> dict:
    """Paired Wilcoxon = one-sample signed-rank test on x1 - x2."""
    if len(x1) != len(x2):
        raise ValueError("paired test requires equal-length samples")
    d = [a - b for a, b in zip(x1, x2)]
    return wilcoxon_signed_rank(d, m0=0.0, alternative=alternative)


def library_versions(x, m0=0.0):
    return {"scipy.stats.wilcoxon": stats.wilcoxon([v - m0 for v in x],
                                                   alternative="two-sided")}


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(1)

    print("=== One-sample, H0: median = 50 ===")
    x = rng.normal(52, 8, 30).tolist()
    for k, v in wilcoxon_signed_rank(x, m0=50).items():
        print(f"  {k:15s}: {v}")

    print("\n=== Paired (post vs pre) ===")
    pre = rng.normal(100, 10, 25).tolist()
    post = [p + rng.normal(3, 4) for p in pre]
    for k, v in paired_wilcoxon(post, pre).items():
        print(f"  {k:15s}: {v}")

    print("\n--- library (scipy.stats.wilcoxon) ---")
    for k, v in library_versions(x, m0=50).items():
        print(f"  {k}: {v}")
