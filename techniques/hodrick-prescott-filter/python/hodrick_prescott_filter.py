"""Hodrick-Prescott Filter (Reference Sec 47.143).

Hodrick & Prescott 1997 'Postwar U.S. business cycles: an empirical
investigation', JMCB 29. Decomposes a time series y_t into a smooth
trend tau_t and cyclical component c_t = y_t - tau_t by minimising:

    sum_t (y_t - tau_t)^2 + lambda * sum_t ((tau_{t+1} - tau_t) - (tau_t - tau_{t-1}))^2

lambda controls the smoothness / fidelity trade-off. Typical values:
lambda = 100 (annual), 1600 (quarterly), 129600 (monthly).

Closed-form solution: tau = (I + lambda K'K)^{-1} y, where K is the
2nd-difference operator matrix.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.sparse import diags, eye as speye    # sparse tridiagonal ops
from scipy.sparse.linalg import spsolve    # sparse linear solve


def hp_filter(y, lam=1600):
    """Extract HP-trend of time series y with smoothness parameter lam."""
    y = np.asarray(y, dtype=float)
    T = len(y)
    # 2nd-difference operator K: (T-2) x T
    K = diags([1, -2, 1], [0, 1, 2], shape=(T - 2, T)).tocsc()
    A = speye(T) + lam * (K.T @ K)
    tau = spsolve(A.tocsc(), y)
    cycle = y - tau
    return {"trend": tau, "cycle": cycle}


if __name__ == "__main__":
    print("=== Hodrick-Prescott filter (Hodrick-Prescott 1997) ===\n")

    rng = np.random.default_rng(0)
    T = 200
    t = np.arange(T)
    # Smooth quadratic trend + AR-like cycle + noise
    trend_true = 0.05 * t + 0.0003 * t ** 2
    cycle_true = 2.0 * np.sin(2 * np.pi * t / 20)
    y = trend_true + cycle_true + rng.normal(scale=0.5, size=T)

    for lam in [10, 1600, 129600]:
        r = hp_filter(y, lam=lam)
        rmse_trend = float(np.sqrt(np.mean((r["trend"] - trend_true) ** 2)))
        rmse_cycle = float(np.sqrt(np.mean((r["cycle"] - cycle_true) ** 2)))
        print(f"  lambda = {lam:6d}   trend RMSE = {rmse_trend:.3f}   "
              f"cycle RMSE = {rmse_cycle:.3f}")

    print("\n  Small lambda: trend hugs data (cycle ~ noise).")
    print("  Large lambda: trend becomes linear (cycle carries curvature + oscillation).")
    print("  lambda=1600 is the standard quarterly-macro default.")

    print("\n--- library cross-check (statsmodels.tsa.filters.hp_filter; mFilter R) ---")
