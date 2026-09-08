"""Bai-Perron multiple structural breaks (Reference Sec 12.17).

Bai & Perron 1998, 2003. Extends Chow / QLR to detect MULTIPLE unknown
break points in a linear regression:

    y_t = X_t * beta_j + eps_t    for t in (tau_{j-1}, tau_j]  (j = 1..m+1)

with m unknown breaks tau_1 < ... < tau_m. Bai-Perron efficiently
finds the global least-squares break locations via dynamic programming
(O(T^2)), and provides sup-F, WDMax and UDMax tests for the NUMBER
of breaks + BIC-based selection.

We implement the DP recursion for m fixed breaks and use BIC to
choose m in {0, 1, 2, 3} on a series with known break structure.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def segment_sse(X, y, a, b):
    """OLS SSE on y[a:b+1] regressed on X[a:b+1]."""
    Xs = X[a:b + 1]; ys = y[a:b + 1]
    if Xs.shape[0] < Xs.shape[1] + 1:
        return np.inf
    beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    return float(np.sum((ys - Xs @ beta) ** 2))


def bai_perron_dp(X, y, m, min_size=15):
    """Find m + 1 segments minimising total SSE. Returns break indices + SSE."""
    T = len(y)
    #  Pre-compute segment SSEs
    #  ssq[i, j] = SSE on [i..j]
    ssq = np.full((T, T), np.inf)
    for i in range(T):
        for j in range(i + min_size - 1, T):
            ssq[i, j] = segment_sse(X, y, i, j)
    #  DP: OPT[k][j] = min total SSE using k breaks on [0..j]
    OPT = np.full((m + 1, T), np.inf)
    PREV = np.full((m + 1, T), -1, dtype=int)
    for j in range(min_size - 1, T):
        OPT[0, j] = ssq[0, j]
    for k in range(1, m + 1):
        for j in range((k + 1) * min_size - 1, T):
            for tau in range(k * min_size - 1, j - min_size + 1):
                cost = OPT[k - 1, tau] + ssq[tau + 1, j]
                if cost < OPT[k, j]:
                    OPT[k, j] = cost; PREV[k, j] = tau
    #  Recover breaks
    breaks = []; j = T - 1; k = m
    while k > 0:
        tau = PREV[k, j]; breaks.append(tau); j = tau; k -= 1
    return {"breaks": sorted(breaks), "SSE": float(OPT[m, T - 1])}


def bic_select_m(X, y, m_max=3, min_size=15):
    T = len(y); p = X.shape[1]
    results = {}
    for m in range(0, m_max + 1):
        r = bai_perron_dp(X, y, m=m, min_size=min_size)
        k_params = (m + 1) * p + m           # m break positions + segment coefs
        sigma2 = r["SSE"] / T
        bic = T * np.log(sigma2) + k_params * np.log(T)
        results[m] = {"breaks": r["breaks"], "SSE": r["SSE"], "BIC": float(bic)}
    best_m = min(results, key=lambda m: results[m]["BIC"])
    return {"selected_m": best_m, "by_m": results}


if __name__ == "__main__":
    print("=== Bai-Perron multiple structural breaks ===\n")
    rng = np.random.default_rng(0)
    T = 200
    x = np.ones((T, 1))
    #  True: 2 breaks at t = 70, 140 -> means (0, 2, -1)
    y = np.concatenate([np.zeros(70), 2 * np.ones(70), -1 * np.ones(60)]) + rng.normal(scale=0.5, size=T)

    print(f"  T = {T}, intercept-only model, true breaks at t = 70, 140\n")
    r = bic_select_m(x, y, m_max=3, min_size=20)
    print(f"  BIC-selected m = {r['selected_m']}")
    for m, info in r["by_m"].items():
        print(f"    m = {m}   breaks = {info['breaks']}   "
              f"SSE = {info['SSE']:.1f}   BIC = {info['BIC']:.1f}")

    print("\n--- library cross-check (strucchange::breakpoints R; ruptures Python) ---")
