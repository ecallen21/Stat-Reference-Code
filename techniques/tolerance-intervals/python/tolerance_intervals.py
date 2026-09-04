"""Tolerance intervals (Reference Sec 38.16).

A (P, 1 - alpha) TOLERANCE INTERVAL is a random interval that
contains AT LEAST proportion P of the population with confidence
1 - alpha.

  * CONFIDENCE INTERVAL  -> about a PARAMETER.
  * PREDICTION INTERVAL  -> about a SINGLE FUTURE observation.
  * TOLERANCE INTERVAL   -> about a PROPORTION of the population.

Normal two-sided (Howe 1969 approximation):

  xbar +/- k2(n, P, alpha) * s     with

  k2 ~ z_{(1+P)/2} * sqrt( (n - 1) * (1 + 1/n) / chi2_{alpha, n-1} )

Nonparametric (order-statistic based, Wilks 1941):

  For P = coverage and 1 - alpha = confidence, the smallest r such
  that P{X_(n - r + 1) >= X_(r) covers >= P} >= 1 - alpha uses the
  Beta-Binomial identity.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats    # normal + chi2 + beta


def normal_ti(x, P=0.95, alpha=0.05):
    """Two-sided normal tolerance interval (Howe 1969 approximation)."""
    n = len(x)
    xbar = np.mean(x); s = np.std(x, ddof=1)
    z = stats.norm.ppf((1 + P) / 2)
    chi = stats.chi2.ppf(alpha, df=n - 1)
    k = z * np.sqrt((n - 1) * (1 + 1 / n) / chi)
    return {"lower": float(xbar - k * s), "upper": float(xbar + k * s),
            "k2": float(k), "n": n, "P": P, "conf": 1 - alpha}


def nonparametric_ti(x, P=0.95, alpha=0.05):
    """Nonparametric two-sided TI using order statistics + Beta-Binomial identity."""
    x = np.sort(np.asarray(x))
    n = len(x)
    # Find largest r such that P(coverage >= P) = 1 - Beta.cdf(P, n - 2r + 1, 2r) >= 1 - alpha
    for r in range(1, n // 2 + 1):
        cov_prob = 1 - stats.beta.cdf(P, n - 2 * r + 1, 2 * r)
        if cov_prob < 1 - alpha:
            r -= 1
            break
    if r < 1:
        return {"lower": float("-inf"), "upper": float("inf"), "r": 0,
                "n": n, "P": P, "conf": 1 - alpha,
                "note": "sample too small for requested (P, conf)"}
    return {"lower": float(x[r - 1]), "upper": float(x[n - r]),
            "r": r, "n": n, "P": P, "conf": 1 - alpha}


if __name__ == "__main__":
    print("=== Tolerance intervals: normal + nonparametric ===\n")
    rng = np.random.default_rng(0)
    n = 60
    x = rng.normal(loc=100, scale=15, size=n)   # blood-pressure-like
    print(f"  n = {n}, sample mean = {x.mean():.2f}, sample sd = {x.std(ddof=1):.2f}")

    for (P, alpha) in [(0.90, 0.05), (0.95, 0.05), (0.99, 0.05)]:
        n_ti = normal_ti(x, P=P, alpha=alpha)
        np_ti = nonparametric_ti(x, P=P, alpha=alpha)
        print(f"\n  (P = {P:.2f}, conf = {1 - alpha:.2f})")
        print(f"    Normal TI       : [{n_ti['lower']:.2f}, {n_ti['upper']:.2f}]"
              f"   (k2 = {n_ti['k2']:.3f})")
        if np_ti['r'] > 0:
            print(f"    Nonparametric TI: [{np_ti['lower']:.2f}, {np_ti['upper']:.2f}]"
                  f"   (order stats r = {np_ti['r']}, n - r + 1 = {n - np_ti['r'] + 1})")
        else:
            print(f"    Nonparametric TI: {np_ti['note']}")

    # Coverage sanity check (should exceed P in expectation)
    B = 2000
    covers = 0
    P_check = 0.95
    for _ in range(B):
        x_new = rng.normal(loc=100, scale=15, size=n)
        n_ti = normal_ti(x_new, P=P_check, alpha=0.05)
        # Proportion of population covered under true N(100, 15):
        cover = stats.norm.cdf(n_ti['upper'], loc=100, scale=15) - stats.norm.cdf(n_ti['lower'], loc=100, scale=15)
        covers += cover >= P_check
    print(f"\n  Simulated confidence that TI covers >= {P_check}: {covers / B:.3f} (target = 0.95)\n")

    print("--- library cross-check (R tolerance::normtol.int/nptol.int; Python custom) ---")
