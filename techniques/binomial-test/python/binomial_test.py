"""Binomial test (Reference §3.22).

H0:  the probability of "success" in a single Bernoulli trial is  p == p0
Observed: x successes in n independent trials.

Variants implemented
--------------------
- Exact binomial test    : sums the binomial PMF in the tail(s); the gold standard
                           for small n.
- Mid-p binomial test    : exact tail minus 0.5 * P(observed); less conservative
                           than the exact test (Lancaster 1961). Useful when the
                           exact test feels over-conservative.
- Normal-approx z-test   : (p_hat - p0) / sqrt(p0 (1-p0) / n)  -- the large-n
                           Wald-under-null form (same as techniques/z-tests).
                           Cross-check against the exact result; differs by < a
                           percent or two once n*p0(1-p0) > ~10.

Two-sided p-value (the "method of small p-values") matches scipy.stats.binomtest:
sum the PMF over all k with P(K = k) <= P(K = x), for K ~ Binomial(n, p0).
"""
from __future__ import annotations

import math

from scipy import stats


def _pmf(n: int, k: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def exact_binomial_test(x: int, n: int, p0: float = 0.5,
                        alternative: str = "two-sided") -> dict:
    """Exact binomial test.

    Parameters
    ----------
    x : observed number of successes.
    n : number of trials.
    p0 : null-hypothesis success probability.
    alternative : "two-sided" / "greater" / "less".
    """
    if not (0 <= x <= n):
        raise ValueError("require 0 <= x <= n")
    p_hat = x / n
    if alternative == "greater":
        p = sum(_pmf(n, k, p0) for k in range(x, n + 1))
    elif alternative == "less":
        p = sum(_pmf(n, k, p0) for k in range(0, x + 1))
    elif alternative == "two-sided":
        p_x = _pmf(n, x, p0)
        p = sum(_pmf(n, k, p0) for k in range(n + 1) if _pmf(n, k, p0) <= p_x + 1e-15)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    return {"x": x, "n": n, "p_hat": p_hat, "p0": p0,
            "p_value": min(1.0, p), "method": "exact"}


def mid_p_binomial_test(x: int, n: int, p0: float = 0.5,
                        alternative: str = "two-sided") -> dict:
    """Mid-p version: exact tail probability minus half of the observed-table probability."""
    p_x = _pmf(n, x, p0)
    if alternative == "greater":
        p = sum(_pmf(n, k, p0) for k in range(x + 1, n + 1)) + 0.5 * p_x
    elif alternative == "less":
        p = sum(_pmf(n, k, p0) for k in range(0, x)) + 0.5 * p_x
    elif alternative == "two-sided":
        ex = exact_binomial_test(x, n, p0, "two-sided")["p_value"]
        p = ex - 0.5 * p_x
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    return {"x": x, "n": n, "p_hat": x / n, "p0": p0,
            "p_value": max(0.0, p), "method": "mid-p"}


def normal_approx_binomial_test(x: int, n: int, p0: float = 0.5,
                                alternative: str = "two-sided",
                                continuity: bool = False) -> dict:
    """Normal-approximation z-test for a binomial proportion. ``continuity`` toggles Yates."""
    p_hat = x / n
    se0 = math.sqrt(p0 * (1 - p0) / n)
    diff = abs(p_hat - p0)
    if continuity:
        diff = max(0.0, diff - 1.0 / (2 * n))
    z = math.copysign(diff, p_hat - p0) / se0 if se0 > 0 else 0.0
    if alternative == "two-sided":
        p = 2 * stats.norm.sf(abs(z))
    elif alternative == "greater":
        p = float(stats.norm.sf(z))
    else:
        p = float(stats.norm.cdf(z))
    return {"x": x, "n": n, "p_hat": p_hat, "p0": p0,
            "z": z, "p_value": p, "method": "normal-approx" + (" (Yates)" if continuity else "")}


def library_versions(x, n, p0):
    out = {
        "scipy.stats.binomtest (exact two-sided)": stats.binomtest(x, n, p0).pvalue,
        "scipy.stats.binomtest greater": stats.binomtest(x, n, p0, alternative="greater").pvalue,
    }
    return out


if __name__ == "__main__":
    # Example: a coin gave 60 heads in 100 flips. Is it fair?
    x, n, p0 = 60, 100, 0.5
    print(f"=== x = {x}, n = {n}, p0 = {p0} ===")
    for fn, label in [(exact_binomial_test, "exact"),
                      (mid_p_binomial_test, "mid-p"),
                      (normal_approx_binomial_test, "normal")]:
        res = fn(x, n, p0)
        print(f"  {label:8s}: p = {res['p_value']:.6f}")

    # Tiny-n example where exact and normal disagree noticeably
    print("\n=== x = 9, n = 10, p0 = 0.5  (tiny n) ===")
    for fn, label in [(exact_binomial_test, "exact"),
                      (mid_p_binomial_test, "mid-p"),
                      (normal_approx_binomial_test, "normal"),
                      (lambda x, n, p0: normal_approx_binomial_test(x, n, p0, continuity=True),
                       "normal+Yates")]:
        res = fn(9, 10, 0.5)
        print(f"  {label:14s}: p = {res['p_value']:.6f}")

    print("\n--- library (scipy.stats.binomtest) ---")
    for k, v in library_versions(x, n, p0).items():
        print(f"  {k}: {v}")
