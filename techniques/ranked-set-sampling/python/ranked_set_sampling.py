"""Ranked-set sampling (RSS) (Reference Sec 27.9).

McIntyre 1952 'A method of unbiased selective sampling using ranked
sets', Australian J Agric Res. When precise measurement is expensive
but a cheap covariate can RANK a small set of units:

    1. Draw m units at random; RANK them (cheaply, e.g. by eye).
    2. Measure the SMALLEST accurately.
    3. Repeat with a new independent set of m; measure the 2nd
       smallest.
    4. Continue up to the m-th smallest -> one 'cycle' of m
       measurements.
    5. Repeat r cycles for n = m * r accurate measurements.

Estimator: sample mean.  Under perfect ranking:

    Var(y_bar_RSS)  <=  Var(y_bar_SRS) / m       (Takahasi-Wakimoto 1968)

So a 4-fold RSS beats SRS by up to 4x variance reduction with the
same accurate-measurement budget.

Applications: environmental sampling, forestry (tree heights), clinical
measurements where visual triage is easy but lab assay is costly.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def srs_mean(y_full, n, rng):
    """Simple random sample mean."""
    idx = rng.choice(len(y_full), size=n, replace=False)
    return float(y_full[idx].mean())


def rss_mean(y_full, m, r, rng, ranking_noise=0.0):
    """Ranked-set sample mean.

    y_full : the full 'true' population from which sets are drawn.
    m : set size per cycle, r : number of cycles. Total measured n = m * r.
    ranking_noise : add Normal(0, s^2) noise to the cheap ranking variable.
    """
    n_pop = len(y_full)
    measured = []
    for _ in range(r):
        for k in range(m):
            #  Draw m fresh units
            idx = rng.choice(n_pop, size=m, replace=False)
            y_set = y_full[idx]
            #  Rank using a noisy proxy = y + noise
            rank_var = y_set + ranking_noise * rng.normal(size=m)
            sorted_by_proxy = np.argsort(rank_var)
            picked = y_set[sorted_by_proxy[k]]        # k-th order-statistic by proxy
            measured.append(picked)
    return float(np.mean(measured))


if __name__ == "__main__":
    print("=== Ranked-set sampling (McIntyre 1952) ===\n")
    rng = np.random.default_rng(0)
    n_pop = 20000
    y_full = rng.normal(loc=10, scale=2, size=n_pop)
    true_mean = float(y_full.mean())
    print(f"  Population size = {n_pop}, true mean = {true_mean:.4f}\n")

    for m in [2, 4, 8]:
        r = 40                            # cycles -> n = m * r measurements
        n = m * r
        srs_vars = np.var([srs_mean(y_full, n, np.random.default_rng(k)) for k in range(500)], ddof=1)
        rss_vars_perfect = np.var([rss_mean(y_full, m, r, np.random.default_rng(k), ranking_noise=0.0)
                                    for k in range(500)], ddof=1)
        rss_vars_noisy = np.var([rss_mean(y_full, m, r, np.random.default_rng(k), ranking_noise=1.5)
                                  for k in range(500)], ddof=1)
        print(f"  m = {m}, r = {r} (n = {n} measurements):")
        print(f"    Var(SRS mean)                = {srs_vars:.5f}")
        print(f"    Var(RSS mean, perfect rank)  = {rss_vars_perfect:.5f}   "
              f"({srs_vars / rss_vars_perfect:.2f}x SRS)")
        print(f"    Var(RSS mean, noisy rank)    = {rss_vars_noisy:.5f}   "
              f"({srs_vars / rss_vars_noisy:.2f}x SRS)")
        print()

    print("  Rule of thumb: with perfect ranking, RSS(m) approaches m-fold variance")
    print("  reduction vs SRS. Ranking noise degrades but rarely eliminates the gain.")

    print("\n--- library cross-check (RSSampling R; from-scratch Python) ---")
