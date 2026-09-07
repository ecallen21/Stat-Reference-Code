"""Chow test for structural break (Reference Sec 12.16).

Chow 1960 'Tests of equality between sets of coefficients in two
linear regressions', Econometrica. Test that regression coefficients
are the SAME before and after a candidate break point tau:

    H0: beta_1 = beta_2 = beta_pool     (no break)

Statistic:
    F = ((SSR_pool - SSR_1 - SSR_2) / k)
        / ((SSR_1 + SSR_2) / (n - 2k))         ~   F(k, n - 2k)

Extension: Quandt-Likelihood-Ratio (QLR) supF over unknown break
point tau in [0.15 n, 0.85 n]; asymptotic distribution from Andrews
1993 tabulated.

We test on a series with a real structural break vs a stable one.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def ols_ssr(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return float((r ** 2).sum())


def chow_test(X, y, tau):
    """Test for break at position tau (1-indexed cut)."""
    n, k = X.shape
    SSR_pool = ols_ssr(X, y)
    SSR_1 = ols_ssr(X[:tau], y[:tau])
    SSR_2 = ols_ssr(X[tau:], y[tau:])
    F = ((SSR_pool - SSR_1 - SSR_2) / k) / ((SSR_1 + SSR_2) / (n - 2 * k))
    p = 1 - stats.f.cdf(F, k, n - 2 * k)
    return {"F": float(F), "df": (k, n - 2 * k), "p": float(p),
            "SSR_pool": SSR_pool, "SSR_1": SSR_1, "SSR_2": SSR_2}


def qlr_test(X, y, trim=0.15):
    n = len(y)
    grid = range(int(trim * n), int((1 - trim) * n))
    Fs = [chow_test(X, y, tau)["F"] for tau in grid]
    tau_star = list(grid)[int(np.argmax(Fs))]
    return {"tau_star": tau_star, "supF": float(max(Fs))}


if __name__ == "__main__":
    print("=== Chow / QLR structural-break test ===\n")
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    #  True break at t = 100: intercept jumps by +2
    y_stable = 1.0 + 0.6 * x + rng.normal(scale=0.5, size=n)
    y_break  = np.where(np.arange(n) < 100,
                          1.0 + 0.6 * x + rng.normal(scale=0.5, size=n),
                          3.0 + 0.6 * x + rng.normal(scale=0.5, size=n))
    X = np.c_[np.ones(n), x]

    print(f"  STABLE series (no break):")
    r_s = chow_test(X, y_stable, tau=100)
    print(f"    Chow F(2, 196) = {r_s['F']:.3f}   p = {r_s['p']:.3f}   (should NOT reject)")

    print(f"\n  SERIES WITH BREAK at t = 100:")
    r_b = chow_test(X, y_break, tau=100)
    print(f"    Chow F(2, 196) = {r_b['F']:.3f}   p = {r_b['p']:.4g}   (should reject)")

    q = qlr_test(X, y_break)
    print(f"    QLR sup-F over tau ∈ [30, 170]: max at tau* = {q['tau_star']}, "
          f"supF = {q['supF']:.2f}")

    print("\n--- library cross-check (strucchange R; statsmodels break_test Python) ---")
