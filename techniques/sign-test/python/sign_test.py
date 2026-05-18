"""The Sign Test (Reference §6.1).

The minimal-assumption test about a median (or paired differences):

  One-sample : H0: median(x) = m0
               Let S = #{x_i > m0}; ignore ties.
               Under H0, S ~ Binomial(n_no_tie, 0.5).

  Paired     : Test that the median of (x1 - x2) is 0 -- a one-sample sign test
               on the differences.

Only assumes continuous, independent observations. Compared to Wilcoxon
signed-rank (techniques/wilcoxon-signed-rank), the sign test uses LESS
information about the differences (only their signs, not their magnitudes),
so it has lower power but also fewer assumptions (no symmetry needed).
"""
from __future__ import annotations

import math
from typing import Sequence

from scipy import stats


def sign_test(x: Sequence[float], m0: float = 0.0,
              alternative: str = "two-sided") -> dict:
    """One-sample sign test of H0: median(x) = m0.

    ``alternative``: "two-sided" / "greater" / "less".
    """
    n_pos = sum(1 for v in x if v > m0)
    n_neg = sum(1 for v in x if v < m0)
    n_tie = sum(1 for v in x if v == m0)
    n_eff = n_pos + n_neg
    if n_eff == 0:
        return {"S": 0, "n_effective": 0, "p_value": 1.0, "method": "sign test"}

    if alternative == "two-sided":
        # scipy's binomtest two-sided 'method of small p-values'
        p_value = float(stats.binomtest(n_pos, n_eff, 0.5).pvalue)
    elif alternative == "greater":
        p_value = float(stats.binomtest(n_pos, n_eff, 0.5, alternative="greater").pvalue)
    elif alternative == "less":
        p_value = float(stats.binomtest(n_pos, n_eff, 0.5, alternative="less").pvalue)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    return {"S_pos": n_pos, "n_neg": n_neg, "n_tie": n_tie,
            "n_effective": n_eff, "p_value": p_value, "method": "sign test"}


def paired_sign_test(x1: Sequence[float], x2: Sequence[float],
                     alternative: str = "two-sided") -> dict:
    """Paired sign test: H0: median(x1 - x2) = 0."""
    if len(x1) != len(x2):
        raise ValueError("paired test requires equal-length samples")
    d = [a - b for a, b in zip(x1, x2)]
    return sign_test(d, m0=0.0, alternative=alternative)


def library_versions(x, m0=0.0):
    try:
        from statsmodels.stats.descriptivestats import sign_test as sm_sign
        stat, p = sm_sign(x, mu0=m0)
        return {"statsmodels sign_test": (float(stat), float(p))}
    except Exception as exc:
        return {"statsmodels": f"error: {exc}"}


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)

    print("=== One-sample (median > 50?) ===")
    x = rng.normal(52, 10, 25).tolist()
    print(f"  observed median = {sorted(x)[len(x)//2]:.2f}")
    for k, v in sign_test(x, m0=50, alternative="greater").items():
        print(f"  {k:14s}: {v}")

    print("\n=== Paired (post > pre?) ===")
    pre = rng.normal(100, 10, 20).tolist()
    post = [p + rng.normal(2, 4) for p in pre]
    for k, v in paired_sign_test(post, pre, alternative="greater").items():
        print(f"  {k:14s}: {v}")

    print("\n--- library (statsmodels sign_test) ---")
    for k, v in library_versions(x, m0=50).items():
        print(f"  {k}: {v}")
