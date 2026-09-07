"""Normality tests (Reference Sec 3.24, 3.25).

Four classical omnibus tests of H0: y ~ Normal:

  SHAPIRO-WILK       -- most powerful for moderate n; scipy.stats.shapiro.
  ANDERSON-DARLING   -- weighted tail-sensitive Cramer-von Mises variant.
  JARQUE-BERA        -- moment-based (skew + excess kurtosis).
  KOLMOGOROV-SMIRNOV -- distributional distance (conservative with estimated
                        parameters; use Lilliefors correction).

Choose one:
  * Small n (< 50)   -> Shapiro-Wilk.
  * Moderate n       -> Shapiro or Anderson-Darling.
  * Very large n (> 5000) -> tests almost always reject; use QQ-plot instead.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def shapiro(y):
    W, p = stats.shapiro(y)
    return {"W": float(W), "p": float(p)}


def anderson_darling(y):
    r = stats.anderson(y, dist="norm")
    return {"A2": float(r.statistic),
            "critical_5pct": float(r.critical_values[2]),
            "reject_at_5pct": bool(r.statistic > r.critical_values[2])}


def jarque_bera(y):
    JB, p = stats.jarque_bera(y)
    return {"JB": float(JB), "p": float(p)}


def kolmogorov_smirnov(y):
    # Standardise then compare to N(0, 1)
    y_z = (y - y.mean()) / y.std(ddof=1)
    D, p = stats.kstest(y_z, "norm")
    return {"D": float(D), "p_naive": float(p),
            "note": "p is anti-conservative when mean/sd estimated; use Lilliefors."}


if __name__ == "__main__":
    print("=== Normality tests: Shapiro / Anderson-Darling / Jarque-Bera / KS ===\n")
    rng = np.random.default_rng(0)
    n = 200

    for name, y in [("Normal(0, 1)",     rng.normal(0, 1, n)),
                    ("Uniform(0, 1)",    rng.uniform(0, 1, n)),
                    ("Exponential(1)",   rng.exponential(1, n)),
                    ("Student-t(df=3)",  rng.standard_t(3, n))]:
        print(f"  Sample: {name}")
        s = shapiro(y); ad = anderson_darling(y); jb = jarque_bera(y); ks = kolmogorov_smirnov(y)
        print(f"    Shapiro-Wilk  W = {s['W']:.4f}   p = {s['p']:.4f}")
        print(f"    A-D           A2 = {ad['A2']:.3f}  crit_5% = {ad['critical_5pct']:.3f}"
              f"   reject5% = {ad['reject_at_5pct']}")
        print(f"    Jarque-Bera   JB = {jb['JB']:.3f}  p = {jb['p']:.4f}")
        print(f"    K-S           D = {ks['D']:.4f}   p = {ks['p_naive']:.4f}")
        print()

    print("--- library cross-check (R nortest, stats::shapiro.test; Python scipy.stats) ---")
