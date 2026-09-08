"""Jarque-Bera test (Reference Sec 47.103).

Jarque & Bera 1980 'Efficient tests for normality, homoscedasticity
and serial independence of regression residuals'. Tests normality
via SAMPLE SKEWNESS and EXCESS KURTOSIS:

    JB = n/6 [ S^2 + (K - 3)^2 / 4 ]

with S = sample skewness, K = sample kurtosis. Under H_0 (normal),
JB ~ chi^2_2. Fast alternative to Anderson-Darling; less powerful
in the tails but preferred by econometricians for regression
residuals.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats    # chi^2 CDF


def jarque_bera(x):
    x = np.asarray(x); n = len(x)
    mu = x.mean(); s2 = x.var()
    if s2 <= 0:
        return {"JB": 0.0, "p": 1.0, "skew": 0.0, "kurt": 3.0}
    z = (x - mu) / np.sqrt(s2)
    S = float((z ** 3).mean())
    K = float((z ** 4).mean())          # non-excess kurtosis
    JB = n / 6.0 * (S ** 2 + (K - 3) ** 2 / 4.0)
    p = float(1 - stats.chi2.cdf(JB, df=2))
    return {"JB": float(JB), "p": p, "skew": S, "kurt": K}


if __name__ == "__main__":
    print("=== Jarque-Bera normality test (Jarque-Bera 1980) ===\n")
    rng = np.random.default_rng(0)
    scenarios = [
        ("N(0, 1)                   ", rng.normal(size=500)),
        ("Uniform(-1, 1)            ", rng.uniform(-1, 1, size=500)),
        ("t3 (heavy tails)          ", rng.standard_t(3, size=500)),
        ("Log-normal (right-skew)  ", rng.lognormal(size=500)),
        ("Exponential(1) (skew=2)  ", rng.exponential(size=500)),
        ("Mixture of two Gaussians ",
         np.concatenate([rng.normal(-3, 1, size=250), rng.normal(3, 1, size=250)])),
    ]
    print(f"  {'scenario':30s} {'JB':>9s}   {'p':>7s}   {'skew':>7s}  {'kurt':>6s}")
    for name, x in scenarios:
        r = jarque_bera(x)
        print(f"  {name:30s} {r['JB']:9.2f}   {r['p']:7.4f}   "
              f"{r['skew']:+7.3f}  {r['kurt']:6.2f}")

    print("\n  Chi^2(2) 5% cvalue = 5.99; 1% = 9.21. JB > 6 -> normality rejected at 5%.")
    print("  Detects both skew and kurtosis departures; less power in tails than A-D.")

    print("\n--- library cross-check (tseries::jarque.bera.test R; statsmodels Python) ---")
