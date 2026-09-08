"""Prophet-style forecasting (Reference Sec 47.116).

Taylor & Letham 2018 'Forecasting at scale', American Statistician
72(1). Additive decomposable model:

    y(t) = g(t) + s(t) + h(t) + eps

    * g(t)  piecewise-linear trend with automatic change-points
    * s(t)  Fourier-series seasonality
    * h(t)  user-supplied holiday effects
    * eps   IID noise

Bayesian model in Stan by default. Handles missing days, outliers,
change points; friendly for analyst-first forecasting. Simplified
here as changepoint-piecewise trend + Fourier seasonality fit by
least squares.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def build_features(t, changepoints, period, K_seasonal):
    """Stack: intercept + linear trend + hinge basis at changepoints
    + K seasonal sines/cosines of period P.
    """
    X = [np.ones_like(t), t]
    for cp in changepoints:
        X.append(np.maximum(t - cp, 0.0))
    for k in range(1, K_seasonal + 1):
        X.append(np.sin(2 * np.pi * k * t / period))
        X.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(X)


def prophet_fit(t, y, n_changepoints=8, period=None, K_seasonal=6):
    if period is None:
        period = (t.max() - t.min()) / 4
    changepoints = np.linspace(t.min() + 0.1 * (t.max() - t.min()),
                                t.max() - 0.1 * (t.max() - t.min()),
                                n_changepoints)
    X = build_features(t, changepoints, period, K_seasonal)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"beta": beta, "changepoints": changepoints,
            "period": period, "K_seasonal": K_seasonal}


def prophet_predict(t, fit):
    X = build_features(t, fit["changepoints"], fit["period"], fit["K_seasonal"])
    return X @ fit["beta"]


if __name__ == "__main__":
    print("=== Prophet forecasting (Taylor-Letham 2018) ===\n")
    rng = np.random.default_rng(0)
    n = 240
    t = np.arange(n, dtype=float)
    # Piecewise trend + weekly (7) seasonality + noise
    trend = 0.02 * t + 0.06 * np.maximum(t - 80, 0) - 0.05 * np.maximum(t - 180, 0)
    seasonal = 2.0 * np.sin(2 * np.pi * t / 30) + 1.0 * np.cos(2 * np.pi * t / 30)
    y = trend + seasonal + 0.3 * rng.normal(size=n)

    fit = prophet_fit(t[:200], y[:200], n_changepoints=8, period=30, K_seasonal=6)
    yhat_tr = prophet_predict(t[:200], fit)
    yhat_te = prophet_predict(t[200:], fit)

    train_rmse = float(np.sqrt(((y[:200] - yhat_tr) ** 2).mean()))
    test_rmse = float(np.sqrt(((y[200:] - yhat_te) ** 2).mean()))
    # Persistence baseline
    persist = y[199] * np.ones(n - 200)
    persist_rmse = float(np.sqrt(((y[200:] - persist) ** 2).mean()))
    # Seasonal-naive (repeat 30-lag)
    snaive = y[200 - 30:200 - 30 + (n - 200)]
    snaive_rmse = float(np.sqrt(((y[200:] - snaive) ** 2).mean()))

    print(f"  Training set (t=0..200) RMSE = {train_rmse:.3f}")
    print(f"  Held-out h=40   Prophet    RMSE = {test_rmse:.3f}")
    print(f"  Held-out h=40   persistence RMSE = {persist_rmse:.3f}")
    print(f"  Held-out h=40   seasonal-naive RMSE = {snaive_rmse:.3f}")

    print("\n--- library cross-check (prophet R + Python; neuralprophet Python) ---")
