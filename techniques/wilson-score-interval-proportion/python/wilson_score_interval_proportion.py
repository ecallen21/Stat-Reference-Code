"""Wilson score + Agresti-Coull + Jeffreys CIs for a proportion (Reference Sec 4.18).

Wilson 1927; Agresti & Coull 1998; Brown, Cai & DasGupta 2001. Better
confidence intervals for a binomial proportion than the classical
Wald interval, especially near p = 0 or p = 1.

For x successes in n trials:

    WALD:            p_hat +/- z * sqrt(p_hat (1 - p_hat) / n)
    WILSON SCORE:    (2n p_hat + z^2 +/- z sqrt(z^2 + 4n p_hat (1 - p_hat))) / (2(n + z^2))
    AGRESTI-COULL:   Wald with p_tilde = (x + z^2/2) / (n + z^2)
    JEFFREYS:        Beta(x + 1/2, n - x + 1/2) quantiles
    CLOPPER-PEARSON: exact Beta CI

Wald under-covers dramatically for small n or p near 0/1; Wilson and
Agresti-Coull have coverage close to nominal in nearly all cases.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def wald(x, n, alpha=0.05):
    p = x / n; z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p * (1 - p) / n)
    return (max(0.0, p - z * se), min(1.0, p + z * se))


def wilson(x, n, alpha=0.05):
    p = x / n; z = stats.norm.ppf(1 - alpha / 2)
    denom = 2 * (n + z ** 2)
    centre = (2 * n * p + z ** 2) / denom
    hw = z * np.sqrt(z ** 2 + 4 * n * p * (1 - p)) / denom
    return (max(0.0, centre - hw), min(1.0, centre + hw))


def agresti_coull(x, n, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    x_t = x + z ** 2 / 2; n_t = n + z ** 2
    p_t = x_t / n_t
    hw = z * np.sqrt(p_t * (1 - p_t) / n_t)
    return (max(0.0, p_t - hw), min(1.0, p_t + hw))


def jeffreys(x, n, alpha=0.05):
    lo = 0.0 if x == 0 else stats.beta.ppf(alpha / 2, x + 0.5, n - x + 0.5)
    hi = 1.0 if x == n else stats.beta.ppf(1 - alpha / 2, x + 0.5, n - x + 0.5)
    return (float(lo), float(hi))


def clopper_pearson(x, n, alpha=0.05):
    lo = 0.0 if x == 0 else stats.beta.ppf(alpha / 2, x, n - x + 1)
    hi = 1.0 if x == n else stats.beta.ppf(1 - alpha / 2, x + 1, n - x)
    return (float(lo), float(hi))


if __name__ == "__main__":
    print("=== Wilson / Agresti-Coull / Jeffreys / Clopper-Pearson CIs for proportion ===\n")
    for x, n in [(3, 10), (0, 10), (30, 100), (98, 100)]:
        print(f"  x = {x}, n = {n}   (p_hat = {x / n:.3f})")
        for name, fn in [("Wald", wald), ("Wilson", wilson),
                          ("Agresti-Coull", agresti_coull), ("Jeffreys", jeffreys),
                          ("Clopper-Pearson", clopper_pearson)]:
            lo, hi = fn(x, n)
            print(f"    {name:>15s}: ({lo:.3f}, {hi:.3f})")
        print()

    print("  Wald misbehaves at x=0 (upper bound 0) and near boundary; use Wilson or Jeffreys.")

    print("\n--- library cross-check (binom / PropCIs R; statsmodels.stats.proportion Python) ---")
