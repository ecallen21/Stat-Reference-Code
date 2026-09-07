"""Bland-Altman agreement analysis (Reference Sec 13.16).

Bland & Altman 1986 'Statistical methods for assessing agreement
between two methods of clinical measurement', Lancet. Compares two
continuous measurement methods A and B on the same units:

    bias           = mean(A - B)
    limits of      = bias +/- 1.96 * sd(A - B)     (95 % of new differences
      agreement                                     expected within)
    proportional   = regression slope of (A - B) on (A + B) / 2
      bias

Bland-Altman PLOT: y-axis (A - B), x-axis (A + B) / 2, with reference
lines at bias and LoA. Reveals mean-dependent bias / heteroskedasticity.

Distinct from correlation / R^2 -- two methods can correlate 0.99 yet
disagree substantially (constant offset).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def bland_altman(a, b):
    diff = a - b
    mean = (a + b) / 2
    bias = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    loa_low = bias - 1.96 * sd_diff
    loa_high = bias + 1.96 * sd_diff
    #  Proportional-bias regression
    slope, intercept = np.polyfit(mean, diff, 1)
    #  CIs on bias and LoA (Bland-Altman 1999 formulas)
    n = len(a)
    se_bias = sd_diff / np.sqrt(n)
    se_loa = sd_diff * np.sqrt(3 / n)
    return {"n": n, "bias": bias, "sd_diff": sd_diff,
            "loa_low": loa_low, "loa_high": loa_high,
            "bias_95_CI": (bias - 1.96 * se_bias, bias + 1.96 * se_bias),
            "loa_low_95_CI": (loa_low - 1.96 * se_loa, loa_low + 1.96 * se_loa),
            "loa_high_95_CI": (loa_high - 1.96 * se_loa, loa_high + 1.96 * se_loa),
            "prop_bias_slope": float(slope),
            "prop_bias_intercept": float(intercept)}


if __name__ == "__main__":
    print("=== Bland-Altman agreement analysis ===\n")
    rng = np.random.default_rng(0)
    n = 200

    #  Method A: reference, N(100, 15)
    a = rng.normal(100, 15, n)
    #  Method B: has +1.5 bias plus multiplicative error
    b = 0.98 * a + 1.5 + rng.normal(scale=3, size=n)

    r = bland_altman(a, b)
    print(f"  n = {r['n']}   bias = {r['bias']:+.2f}   sd(diff) = {r['sd_diff']:.2f}")
    print(f"  95 % LoA  = [{r['loa_low']:+.2f}, {r['loa_high']:+.2f}]")
    print(f"  Bias 95% CI            = ({r['bias_95_CI'][0]:+.2f}, {r['bias_95_CI'][1]:+.2f})")
    print(f"  Lower LoA 95% CI       = ({r['loa_low_95_CI'][0]:+.2f}, {r['loa_low_95_CI'][1]:+.2f})")
    print(f"  Upper LoA 95% CI       = ({r['loa_high_95_CI'][0]:+.2f}, {r['loa_high_95_CI'][1]:+.2f})")
    print(f"  Proportional bias      = slope {r['prop_bias_slope']:+.4f}  "
          f"intercept {r['prop_bias_intercept']:+.2f}")

    print(f"\n  Correlation vs agreement:")
    print(f"    corr(A, B) = {np.corrcoef(a, b)[0, 1]:.4f}    <- large but MISLEADING for agreement")
    print(f"    Bland-Altman shows the ~+/-6 unit LoA gap that correlation hides.")

    print("\n--- library cross-check (blandr / BlandAltmanLeh R; pyCompare Python) ---")
