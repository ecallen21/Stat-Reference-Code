"""Point-biserial correlation (Reference §4.4).

Pearson's r between a continuous variable y and a binary variable x (coded
0/1). Algebraically:

    r_pb = (mean(y | x=1) - mean(y | x=0)) / sd(y) * sqrt(p (1 - p))

where p = fraction with x = 1. So r_pb is just Pearson with a 0/1 variable --
the formula above is the same calculation in a more interpretable form.

Connection to the two-sample t-test:
    t = r_pb * sqrt((n - 2) / (1 - r_pb^2))   <-  identical to Student's t
on the two groups. So a point-biserial r is "the t-test expressed as a
correlation." Same caveats: assumes the two groups have similar variance.

Distinguish from
  - biserial r: y is continuous, x is a CONTINUOUS variable that has been
    dichotomized (an underlying normal is assumed). Not implemented here.
  - rank-biserial: nonparametric analogue derived from Mann-Whitney U; see
    techniques/effect-sizes.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _mean(x): return sum(x) / len(x)


def point_biserial(x: Sequence[int], y: Sequence[float]) -> dict:
    """Point-biserial correlation between binary ``x`` (0/1) and continuous ``y``.

    Also returns the equivalent two-sample t-statistic and p-value.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if any(v not in (0, 1) for v in x):
        raise ValueError("x must be 0/1")
    n = len(x)
    y1 = [yi for xi, yi in zip(x, y) if xi == 1]
    y0 = [yi for xi, yi in zip(x, y) if xi == 0]
    n1, n0 = len(y1), len(y0)
    if n1 == 0 or n0 == 0:
        raise ValueError("both groups must be non-empty")
    m1, m0 = _mean(y1), _mean(y0)
    p = n1 / n
    # sd(y) using the full sample (population form, matching scipy)
    my = _mean(y)
    sy = math.sqrt(sum((v - my) ** 2 for v in y) / n)
    r_pb = (m1 - m0) / sy * math.sqrt(p * (1 - p))
    t = r_pb * math.sqrt((n - 2) / (1 - r_pb ** 2))
    df = n - 2
    p_val = float(2 * stats.t.sf(abs(t), df))
    return {"r_pb": r_pb, "mean_group_1": m1, "mean_group_0": m0,
            "n1": n1, "n0": n0, "t": t, "df": df, "p_value": p_val}


def library_versions(x, y):
    res = stats.pointbiserialr(x, y)
    return {"scipy.stats.pointbiserialr": (float(res.statistic), float(res.pvalue))}


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(8)
    n = 100
    x = rng.integers(0, 2, n).tolist()
    y = [rng.normal(50 if xi == 0 else 56, 10) for xi in x]
    print("=== Point-biserial: continuous y vs. binary x ===")
    for k, v in point_biserial(x, y).items():
        print(f"  {k:14s}: {v}")
    print("\n--- library (scipy.stats.pointbiserialr) ---")
    for k, v in library_versions(x, y).items():
        print(f"  {k}: {v}")

    # Equivalent two-sample t-test
    g1 = [yi for xi, yi in zip(x, y) if xi == 1]
    g0 = [yi for xi, yi in zip(x, y) if xi == 0]
    t, p = stats.ttest_ind(g1, g0, equal_var=True)
    print(f"\nEquivalent Student t-test: t = {float(t):.4f}  p = {float(p):.4g}")
