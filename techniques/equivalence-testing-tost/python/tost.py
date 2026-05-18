"""Equivalence testing via Two One-Sided Tests (TOST) (Reference §3.21).

A classic t-test asks "are the means *different*?" -- you can never *prove*
equality, only fail to reject difference. TOST flips this: you specify an
equivalence margin [lower, upper] (the smallest difference you care about,
often called -delta to +delta), and you reject "the means differ by more than
the margin" if BOTH one-sided tests are significant.

Reject H0_equivalence if:
    t_lower = (diff - lower) / SE   exceeds  t_{1-alpha, df}  AND
    t_upper = (diff - upper) / SE   is below -t_{1-alpha, df}

Equivalently, the overall TOST p-value is the max of the two one-sided p-values.
At alpha = 0.05, this is the "90% CI rule": the means are declared equivalent
iff the 100*(1 - 2*alpha)% (so 90%) CI for the difference lies entirely inside
[lower, upper].

Variants implemented
--------------------
- ``tost_two_sample`` : Welch (default) or Student two-sample t (signs handled
                        like ``techniques/t-tests``)
- ``tost_paired``     : TOST on paired differences (one-sample TOST against 0)
- ``tost_one_sample`` : TOST that ``mean(x)`` lies in [lower, upper]

Choosing the equivalence margin is a domain decision. Common defaults:
  - bioequivalence (drug exposure ratios): [80%, 125%] on the log-transformed
    ratio of geometric means
  - educational / psychological: Cohen's d in [-0.2, 0.2]
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _mean(x): return sum(x) / len(x)


def _var(x, ddof=1):
    m = _mean(x); return sum((v - m) ** 2 for v in x) / (len(x) - ddof)


def tost_one_sample(x: Sequence[float], lower: float, upper: float,
                    alpha: float = 0.05) -> dict:
    """One-sample TOST: is mean(x) within [lower, upper]?

    Parameters
    ----------
    x : numeric sample.
    lower, upper : equivalence bounds in the same units as ``x``.
    alpha : per-test significance level (commonly 0.05 -> 90% CI rule).
    """
    n = len(x); m = _mean(x); s = math.sqrt(_var(x))
    se = s / math.sqrt(n); df = n - 1
    t_lo = (m - lower) / se          # H0: mean <= lower  vs.  H1: mean > lower
    t_up = (m - upper) / se          # H0: mean >= upper  vs.  H1: mean < upper
    p_lo = float(stats.t.sf(t_lo, df))
    p_up = float(stats.t.cdf(t_up, df))
    p_tost = max(p_lo, p_up)
    tcrit = stats.t.ppf(1 - alpha, df)
    ci90_low = m - tcrit * se
    ci90_high = m + tcrit * se
    return {"mean": m, "se": se, "df": df,
            "t_lower": t_lo, "p_lower": p_lo,
            "t_upper": t_up, "p_upper": p_up,
            "p_tost": p_tost, "equivalent": p_tost < alpha,
            "ci_inner_low": ci90_low, "ci_inner_high": ci90_high,
            "margin": (lower, upper)}


def tost_two_sample(x1: Sequence[float], x2: Sequence[float],
                    lower: float, upper: float, alpha: float = 0.05,
                    equal_var: bool = False) -> dict:
    """Two-sample TOST on mean(x1) - mean(x2).

    Sign convention: ``mean(x1) - mean(x2)``. ``equal_var=False`` -> Welch (default).
    """
    n1, n2 = len(x1), len(x2)
    m1, m2 = _mean(x1), _mean(x2)
    v1, v2 = _var(x1), _var(x2)
    diff = m1 - m2
    if equal_var:
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = math.sqrt(sp2 * (1 / n1 + 1 / n2))
        df = n1 + n2 - 2
    else:
        se = math.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        )
    t_lo = (diff - lower) / se
    t_up = (diff - upper) / se
    p_lo = float(stats.t.sf(t_lo, df))
    p_up = float(stats.t.cdf(t_up, df))
    p_tost = max(p_lo, p_up)
    tcrit = stats.t.ppf(1 - alpha, df)
    return {"mean_diff": diff, "se": se, "df": df,
            "t_lower": t_lo, "p_lower": p_lo,
            "t_upper": t_up, "p_upper": p_up,
            "p_tost": p_tost, "equivalent": p_tost < alpha,
            "ci_inner_low": diff - tcrit * se,
            "ci_inner_high": diff + tcrit * se,
            "margin": (lower, upper),
            "method": "Welch" if not equal_var else "Student"}


def tost_paired(x1: Sequence[float], x2: Sequence[float],
                lower: float, upper: float, alpha: float = 0.05) -> dict:
    """Paired TOST: TOST on the differences ``x1 - x2`` against ``[lower, upper]``."""
    if len(x1) != len(x2):
        raise ValueError("paired TOST requires equal-length samples")
    return tost_one_sample([a - b for a, b in zip(x1, x2)], lower, upper, alpha)


def library_versions(x1, x2, lower, upper):
    out = {}
    try:
        from statsmodels.stats.weightstats import ttost_ind
        p, (t1, p1, df1), (t2, p2, df2) = ttost_ind(x1, x2, lower, upper, usevar="unequal")
        out["statsmodels ttost_ind (Welch)"] = {
            "p_tost": float(p), "p_lower": float(p1), "p_upper": float(p2)}
    except Exception as exc:
        out["statsmodels"] = f"error: {exc}"
    return out


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(13)
    # Two "equivalent" groups -- means differ by 0.4, equivalence margin +/- 2.0
    a = rng.normal(50, 8, 40).tolist()
    b = rng.normal(50.4, 8, 38).tolist()
    print("=== Two-sample TOST  margin = (-2.0, +2.0)  alpha = 0.05 ===")
    for k, v in tost_two_sample(a, b, lower=-2.0, upper=2.0).items():
        print(f"  {k:14s}: {v}")
    print("\n--- library (statsmodels) ---")
    for k, v in library_versions(a, b, -2.0, 2.0).items():
        print(f"  {k}: {v}")

    # Pre/post pairs where the change is "small" -- test equivalence to no change
    pre = rng.normal(100, 12, 25).tolist()
    post = [p + rng.normal(0.5, 4) for p in pre]
    print("\n=== Paired TOST: change-from-baseline within +/- 3 points ===")
    for k, v in tost_paired(post, pre, lower=-3.0, upper=3.0).items():
        print(f"  {k:14s}: {v}")
