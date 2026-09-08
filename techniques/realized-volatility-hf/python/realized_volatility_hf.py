"""Realized volatility from high-frequency data (Sec 47.75).

Andersen & Bollerslev 1998 'Answering the skeptics: yes, standard
volatility models do provide accurate forecasts', Int Econ Rev 39.
Given intraday log-prices p_0, p_1, ..., p_M sampled M+1 times
across the day, realized variance:

    RV = sum_{i=1}^{M}  (p_i - p_{i-1})^2

is a consistent (as M -> inf) estimator of integrated variance
under a continuous semimartingale. Bipower variation (Barndorff-
Nielsen-Shephard 2004) is jump-robust:

    BV = pi/2 * sum_{i=2}^{M}  |r_i| * |r_{i-1}|

so RV - BV -> jump variation. Two-scales estimator (Zhang-Mykland-
Ait-Sahalia 2005) corrects microstructure-noise bias.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def realized_variance(prices):
    r = np.diff(np.log(prices))
    return float((r ** 2).sum())


def bipower_variation(prices):
    r = np.diff(np.log(prices))
    return float(np.pi / 2 * (np.abs(r[1:]) * np.abs(r[:-1])).sum())


def two_scales_rv(prices, K=5):
    """Zhang-Mykland-Ait-Sahalia 2005 two-scales RV, K subsample offset."""
    logp = np.log(prices)
    n = len(logp) - 1
    RV_all = float(((np.diff(logp)) ** 2).sum())
    # slow-scale: subsample every K steps
    RV_slow = 0.0
    for k in range(K):
        sub = logp[k::K]
        RV_slow += ((np.diff(sub)) ** 2).sum()
    RV_slow /= K
    nbar = n / K
    return float(RV_slow - (nbar / n) * RV_all)


if __name__ == "__main__":
    print("=== Realized volatility (Andersen-Bollerslev 1998) ===\n")
    rng = np.random.default_rng(0)

    T = 6.5           # trading hours in a day
    sigma_true = 0.02 # 2% daily vol
    for M in [12, 78, 390, 1560]:                       # every 30 min, 5 min, 1 min, 15 s
        dt = T / M
        r = sigma_true * np.sqrt(1 / T) * np.sqrt(dt) * rng.normal(size=M)
        # Convert to prices via log-price random walk
        p = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
        RV = realized_variance(p)
        print(f"  M = {M:5d} obs   RV = {RV:.6e}   sqrt(RV) = {np.sqrt(RV):.4f}   "
              f"(truth = {sigma_true:.4f})")

    print("\n  Jump-robust comparison (add one 10-sigma jump at midday):")
    M = 390
    dt = T / M
    r = sigma_true * np.sqrt(1 / T) * np.sqrt(dt) * rng.normal(size=M)
    r[M // 2] += 10 * sigma_true * np.sqrt(1 / T) * np.sqrt(dt)
    p = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    RV = realized_variance(p); BV = bipower_variation(p)
    print(f"  RV = {RV:.6e}   BV = {BV:.6e}   jump variation = {RV - BV:.6e}")

    print("\n  Microstructure-noise example (add IID N(0, 0.01) noise to each price):")
    p_noisy = p * (1 + 0.001 * rng.normal(size=len(p)))
    print(f"  RV(noisy)          = {realized_variance(p_noisy):.6e}")
    print(f"  Two-scales RV(K=5) = {two_scales_rv(p_noisy, K=5):.6e}")
    print(f"  RV(clean)          = {RV:.6e}")

    print("\n--- library cross-check (highfrequency R; MFE / arch Python) ---")
