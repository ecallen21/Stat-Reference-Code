"""Seasonal-trend decomposition (Reference §13.24, §13.47, §13.54).

Split a time series into three parts:

    y_t  =  T_t  +  S_t  +  R_t    (additive)
    y_t  =  T_t  *  S_t  *  R_t    (multiplicative)

    T_t = trend-cycle (long-run + smooth movement)
    S_t = seasonal (repeating pattern of period m)
    R_t = residual / remainder

Three flavors:

§13.47 Classical decomposition
    - Trend via a centered moving average of length m.
    - Seasonal via averaging detrended values within each seasonal position.
    - Residual = whatever's left.
    Fast but rigid: assumes constant seasonal pattern.

§13.24 STL (Seasonal-Trend decomposition using Loess; Cleveland 1990)
    - Trend via LOESS smooth.
    - Seasonal via LOESS on each seasonal cycle position.
    - Iterative: converges after a few outer loops.
    - Handles CHANGING seasonal pattern and is ROBUST to outliers (with the
      robust option).

§13.54 X-13ARIMA-SEATS (US Census Bureau)
    - Reg-ARIMA pre-adjustment (trading days, holidays, outliers).
    - SEATS: signal-extraction via SARIMA state-space.
    - The reference for official statistics; requires the external X-13 binary.
    Not implemented from scratch here; the R file points at seasonal::seas().
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def classical_decomposition(y, m: int, model: str = "additive") -> dict:
    """Classical trend + seasonal + remainder decomposition.

    Parameters
    ----------
    y : univariate series.
    m : seasonal period (12 monthly, 4 quarterly, ...).
    model : 'additive' or 'multiplicative'.
    """
    y = np.asarray(y, dtype=float); n = len(y)
    # Centered moving average for trend
    trend = np.full(n, np.nan)
    half = m // 2
    for t in range(half, n - half):
        window = y[t - half: t + half + 1]
        if m % 2 == 0:
            # 2-sided average of the m-window then centered
            trend[t] = 0.5 * (window[:-1].mean() + window[1:].mean())
        else:
            trend[t] = window.mean()
    # Detrended series
    if model == "additive":
        detrended = y - trend
    else:
        detrended = y / trend
    # Seasonal averages
    seasonal_avg = np.zeros(m)
    counts = np.zeros(m)
    for t in range(n):
        if not np.isnan(detrended[t]):
            seasonal_avg[t % m] += detrended[t]; counts[t % m] += 1
    seasonal_avg = seasonal_avg / np.clip(counts, 1, None)
    # Center seasonal to sum to 0 (additive) or product to 1 (multiplicative)
    if model == "additive":
        seasonal_avg = seasonal_avg - seasonal_avg.mean()
    else:
        seasonal_avg = seasonal_avg / seasonal_avg.mean()
    seasonal = np.array([seasonal_avg[t % m] for t in range(n)])
    if model == "additive":
        remainder = y - trend - seasonal
    else:
        remainder = y / (trend * seasonal)
    return {"trend": trend.tolist(), "seasonal": seasonal.tolist(),
            "remainder": remainder.tolist(),
            "seasonal_pattern": seasonal_avg.tolist(),
            "model": model, "m": m,
            "method": f"classical decomposition ({model})"}


def stl_decomposition(y, period: int, robust: bool = False) -> dict:
    """STL via statsmodels (canonical implementation)."""
    from statsmodels.tsa.seasonal import STL
    stl = STL(np.asarray(y, dtype=float), period=period, robust=robust)
    res = stl.fit()
    return {"trend": res.trend.tolist(),
            "seasonal": res.seasonal.tolist(),
            "resid": res.resid.tolist(),
            "period": period, "robust": robust,
            "method": "STL (statsmodels)"}


if __name__ == "__main__":
    rng = np.random.default_rng(23)
    n = 120; m = 12
    t = np.arange(n)
    trend = 10 + 0.05 * t
    seasonal = 2 * np.sin(2 * np.pi * t / m)
    remainder = rng.normal(0, 0.4, n)
    y = trend + seasonal + remainder

    print("=== Classical additive decomposition (m = 12) ===")
    d = classical_decomposition(y, m=m, model="additive")
    print(f"  seasonal pattern (first 12): {[f'{s:+.3f}' for s in d['seasonal_pattern']]}")
    center = n // 2
    print(f"  trend at t = {center}: {d['trend'][center]:.3f}  (true = {trend[center]:.3f})")
    print(f"  remainder SD: {np.std([r for r in d['remainder'] if not np.isnan(r)]):.4f}   (true = 0.4)")

    print("\n=== STL ===")
    s = stl_decomposition(y, period=m)
    print(f"  trend at t = {center}: {s['trend'][center]:.3f}")
    print(f"  seasonal at t = {center}: {s['seasonal'][center]:+.3f}")
    print(f"  resid SD: {np.std(s['resid']):.4f}")
