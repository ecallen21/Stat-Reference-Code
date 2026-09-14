"""Holt-Winters Triple Exponential Smoothing (Reference Sec 47.309).

Holt 1957; Winters 1960 Management Science. Decompose a series
into LEVEL, TREND, and SEASONAL components with three smoothing
parameters (alpha, beta, gamma):

    Additive:      l_t = alpha (y_t - s_{t-m}) + (1 - alpha)(l_{t-1} + b_{t-1})
                    b_t = beta (l_t - l_{t-1}) + (1 - beta) b_{t-1}
                    s_t = gamma (y_t - l_{t-1} - b_{t-1}) + (1 - gamma) s_{t-m}
                    forecast: l_t + h b_t + s_{t-m+h_m}

Multiplicative version divides / multiplies by s. Classical
benchmark for seasonal series before Prophet / ARIMA.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def holt_winters_additive(y, m, alpha=0.3, beta=0.1, gamma=0.2, h=12):
    """Additive Holt-Winters with seasonal period m and forecast horizon h."""
    n = len(y)
    l = np.zeros(n); b = np.zeros(n); s = np.zeros(n + h)
    # Initialise from first two seasons
    l[m - 1] = y[:m].mean()
    b[m - 1] = (y[m:2 * m].mean() - y[:m].mean()) / m
    for i in range(m): s[i] = y[i] - l[m - 1]
    for t in range(m, n):
        l[t] = alpha * (y[t] - s[t - m]) + (1 - alpha) * (l[t - 1] + b[t - 1])
        b[t] = beta * (l[t] - l[t - 1]) + (1 - beta) * b[t - 1]
        s[t] = gamma * (y[t] - l[t - 1] - b[t - 1]) + (1 - gamma) * s[t - m]
    # Forecasts
    forecasts = np.array([l[-1] + (k + 1) * b[-1] + s[n - m + (k % m)]
                          for k in range(h)])
    return l, b, s[:n], forecasts


if __name__ == "__main__":
    print("=== Holt-Winters Additive (Holt 1957; Winters 1960) ===\n")
    rng = np.random.default_rng(0)

    # Simulate: trend 0.5/step + monthly seasonality + noise
    m = 12; T = 5 * m
    t = np.arange(T)
    seasonal = 5 * np.sin(2 * np.pi * t / m)
    y = 10 + 0.5 * t + seasonal + rng.normal(0, 1, T)

    l, b, s, fc = holt_winters_additive(y, m=m, alpha=0.4, beta=0.1, gamma=0.3, h=12)
    print(f"  Series length T = {T}, seasonal period m = {m}")
    print(f"  Estimated final level = {l[-1]:.2f}   trend b = {b[-1]:.3f}")
    print(f"  Seasonal amplitude range: [{s[-m:].min():.2f}, {s[-m:].max():.2f}]")

    print(f"\n  12-step-ahead forecasts:")
    print(f"    {'h':>3}  {'forecast':>10}")
    for h_, f in enumerate(fc, 1):
        print(f"    {h_:>3}  {f:>10.2f}")

    # RMSE on last season (in-sample proxy)
    y_true_last_season = y[-m:]
    l_last, _, _, _ = holt_winters_additive(y[:-m], m=m, alpha=0.4, beta=0.1, gamma=0.3, h=m)
    y_pred = l_last[3]
    rmse = float(np.sqrt(np.mean((y_true_last_season - fc) ** 2)))
    print(f"\n  Illustrative in-sample forecast RMSE:  ~ {rmse:.2f}")

    print("\n--- library cross-check (statsmodels.tsa.holtwinters; forecast::hw R) ---")
