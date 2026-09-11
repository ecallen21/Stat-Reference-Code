"""N-BEATS - Neural Basis Expansion (Reference Sec 47.234).

Oreshkin, Carpov, Chapados & Bengio 2020 'N-BEATS: Neural Basis
Expansion Analysis for Interpretable Time Series Forecasting',
ICLR. Pure deep-learning forecaster; NO recurrence, NO attention.

    Stack of BLOCKS, each fits a partial signal:
    - Backcast: reconstruct the past.
    - Forecast: predict the future.
    Residuals passed to the next block ('doubly residual').

Blocks use polynomial / seasonal basis functions -> interpretable.
Won the M4 competition ahead of every classical baseline.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def polynomial_basis(t, degree, horizon):
    """Basis: powers of normalised t up to `degree`."""
    tn = t / horizon
    return np.stack([tn ** d for d in range(degree + 1)], axis=-1)


def fourier_basis(t, n_harmonics, period):
    """Basis: sin / cos harmonics."""
    return np.stack([f(2 * np.pi * k * t / period) for k in range(1, n_harmonics + 1)
                        for f in (np.sin, np.cos)], axis=-1)


def n_beats_block(input_hist, backcast_len, forecast_len, basis, rng):
    """One block: fit basis coefficients to input_hist, return backcast + forecast."""
    t_hist = np.arange(backcast_len)
    t_future = np.arange(backcast_len, backcast_len + forecast_len)
    B_hist = basis(t_hist)                                       # (backcast_len, K)
    B_future = basis(t_future)                                   # (forecast_len, K)
    # LS coefficients
    coefs = np.linalg.lstsq(B_hist, input_hist, rcond=None)[0]
    backcast = B_hist @ coefs
    forecast = B_future @ coefs
    return backcast, forecast, coefs


def n_beats_forecast(series, backcast_len, forecast_len, blocks):
    """Doubly-residual stacking of blocks."""
    residual = series[-backcast_len:].copy()
    total_forecast = np.zeros(forecast_len)
    per_block = []
    for basis in blocks:
        backcast, forecast, coefs = n_beats_block(residual, backcast_len,
                                                       forecast_len, basis, None)
        residual = residual - backcast
        total_forecast = total_forecast + forecast
        per_block.append({"forecast": forecast, "coefs": coefs})
    return total_forecast, per_block


if __name__ == "__main__":
    print("=== N-BEATS (Oreshkin et al 2020 ICLR) ===\n")
    rng = np.random.default_rng(0)

    T = 200; backcast_len = 60; forecast_len = 20
    t = np.arange(T)
    # Trend + seasonal + noise
    trend = 0.05 * t
    seasonal = 3 * np.sin(2 * np.pi * t / 24)
    noise = 0.4 * rng.normal(size=T)
    series = trend + seasonal + noise

    # Two blocks: trend (polynomial deg 3) + seasonal (Fourier 3 harmonics, period 24)
    blocks = [
        lambda t: polynomial_basis(t, degree=1, horizon=backcast_len + forecast_len),
        lambda t: fourier_basis(t, n_harmonics=3, period=24),
    ]

    forecast, per_block = n_beats_forecast(series[:-forecast_len], backcast_len,
                                              forecast_len, blocks)
    actual = series[-forecast_len:]
    mse = float(np.mean((forecast - actual) ** 2))
    print(f"  Series length {T}, backcast {backcast_len}, forecast horizon {forecast_len}")
    print(f"  N-BEATS forecast MSE:      {mse:.4f}")
    print(f"    Trend block contributes  : mean = {per_block[0]['forecast'].mean():.2f}")
    print(f"    Seasonal block contributes: max amplitude = "
          f"{float(np.max(np.abs(per_block[1]['forecast']))):.2f}")

    # Naive baseline: last value
    naive = np.repeat(series[-forecast_len - 1], forecast_len)
    print(f"  Naive last-value baseline MSE: {float(np.mean((naive - actual) ** 2)):.4f}")

    print("\n--- library cross-check (darts.models.NBEATSModel; pytorch-forecasting; neuralforecast) ---")
