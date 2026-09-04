"""Sample size + Minimum Detectable Effect (Reference Sec 44.2).

Standard planning-time calculations for an A/B test.  Reverse
formulations:
  * Given (baseline, MDE, alpha, power) -> n per arm.
  * Given (baseline, n per arm, alpha, power) -> MDE.

Two-proportion test (equal-split):
  n_per_arm = ((z_alpha/2 * sqrt(2 p_bar q_bar) + z_beta * sqrt(p_C q_C + p_T q_T))^2) / delta^2

Two-sample continuous:
  n_per_arm = 2 (z_alpha/2 + z_beta)^2 * sigma^2 / delta^2
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def n_prop(p_C, mde_abs, alpha=0.05, power=0.80):
    p_T = p_C + mde_abs
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p_C + p_T) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar))
           + z_b * np.sqrt(p_C * (1 - p_C) + p_T * (1 - p_T))) ** 2
    return int(np.ceil(num / mde_abs ** 2))


def n_ttest(sigma, mde_abs, alpha=0.05, power=0.80):
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return int(np.ceil(2 * (z_a + z_b) ** 2 * sigma ** 2 / mde_abs ** 2))


def mde_prop(p_C, n_per_arm, alpha=0.05, power=0.80):
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    # Approximation using p_bar = p_C
    return float((z_a + z_b) * np.sqrt(2 * p_C * (1 - p_C) / n_per_arm))


def mde_ttest(sigma, n_per_arm, alpha=0.05, power=0.80):
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return float((z_a + z_b) * sigma * np.sqrt(2 / n_per_arm))


if __name__ == "__main__":
    print("=== A/B sample size + MDE ===\n")
    for p_C, mde in [(0.05, 0.005), (0.05, 0.001), (0.20, 0.01)]:
        n = n_prop(p_C, mde)
        print(f"  Baseline p_C={p_C}, MDE (abs)={mde}   -> n per arm = {n:,}")

    print()
    for sigma, mde in [(1.0, 0.05), (10.0, 0.5)]:
        n = n_ttest(sigma, mde)
        print(f"  sigma={sigma}, MDE={mde}   -> n per arm = {n:,}")

    print("\n  Reverse: given n_per_arm, what's the MDE?")
    for p_C, n in [(0.05, 10000), (0.05, 100000)]:
        print(f"    p_C={p_C}, n_per_arm={n}   MDE_abs = {mde_prop(p_C, n):.4f}"
              f"   MDE_rel = {mde_prop(p_C, n) / p_C:.2%}")

    print("\n--- library cross-check (R pwr; Python statsmodels.stats.power) ---")
