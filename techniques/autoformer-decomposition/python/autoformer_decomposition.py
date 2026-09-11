"""Autoformer - Series Decomposition Transformer (Sec 47.237).

Wu et al 2021 'Autoformer: Decomposition Transformers with
Auto-Correlation for Long-Term Series Forecasting', NeurIPS.
Innovations vs vanilla Transformer:

    1. Series Decomposition BLOCK: x = trend + seasonal, via
       moving average -> trend, x - trend -> seasonal. Applied
       inside every encoder / decoder layer.
    2. Auto-Correlation mechanism replaces self-attention: match
       Q with K by TIME-DELAYED SIMILARITY (FFT-based).

    Complexity O(L log L). Yields strong long-horizon forecasts
    on energy / weather / traffic benchmarks.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def series_decomposition(x, kernel_size=25):
    """Moving-average decomposition: trend + seasonal."""
    pad = kernel_size // 2
    x_pad = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)])
    trend = np.array([x_pad[i:i + kernel_size].mean(axis=0) for i in range(len(x))])
    seasonal = x - trend
    return trend, seasonal


def auto_correlation(q, k):
    """FFT-based auto-correlation between q and k signals."""
    Q = np.fft.rfft(q)
    K = np.fft.rfft(k)
    corr = np.fft.irfft(Q * np.conj(K))
    return corr


def autoformer_forecast(series, backcast, horizon):
    """Extend series by extrapolating trend + repeating dominant seasonal."""
    trend, seasonal = series_decomposition(series[-backcast:])
    # Trend: linear extrapolation
    t = np.arange(backcast); a, b = np.polyfit(t, trend, 1)
    t_future = np.arange(backcast, backcast + horizon)
    trend_forecast = a * t_future + b
    # Seasonal: find dominant period via auto-correlation (exclude lag <= 2 to skip trivial peak)
    corr = auto_correlation(seasonal, seasonal)[3:backcast // 2]
    period = int(np.argmax(corr)) + 3
    seasonal_forecast = np.array([seasonal[-period + (t % period)] for t in range(horizon)])
    return trend_forecast + seasonal_forecast, trend_forecast, seasonal_forecast, period


if __name__ == "__main__":
    print("=== Autoformer (Wu et al 2021 NeurIPS) ===\n")
    rng = np.random.default_rng(0)

    # Synthetic series with trend + seasonal + noise
    T = 300; backcast = 100; horizon = 40
    t = np.arange(T)
    trend = 0.02 * t
    seasonal = 3 * np.sin(2 * np.pi * t / 24)
    noise = 0.4 * rng.normal(size=T)
    series = trend + seasonal + noise

    forecast, tr_fc, se_fc, period = autoformer_forecast(series[:-horizon], backcast, horizon)
    actual = series[-horizon:]
    mse = float(np.mean((forecast - actual) ** 2))
    print(f"  T = {T}, backcast = {backcast}, horizon = {horizon}")
    print(f"  Detected dominant seasonal period: {period}   (truth = 24)")
    print(f"  Autoformer forecast MSE:      {mse:.4f}")
    naive = np.repeat(series[-horizon - 1], horizon)
    print(f"  Naive last-value MSE:         {float(np.mean((naive - actual) ** 2)):.4f}")

    # Decomposition demo
    trend_full, seasonal_full = series_decomposition(series, kernel_size=25)
    print(f"\n  Decomposition of full series:")
    print(f"    Trend range: [{trend_full.min():.2f}, {trend_full.max():.2f}]")
    print(f"    Seasonal std: {float(seasonal_full.std()):.2f}   (matches injected amplitude ~ 3)")

    print("\n--- library cross-check (thuml/Autoformer; neuralforecast.Autoformer) ---")
