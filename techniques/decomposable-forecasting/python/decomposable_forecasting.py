"""Decomposable forecasting (Reference §13.21) - Prophet-style additive model.

    y_t = g(t) + s(t) + h(t) + eps_t
        g(t) : trend      (piecewise linear with automatic changepoints)
        s(t) : seasonality (Fourier series at fixed period)
        h(t) : holiday / event effects (indicator regressors)
        eps  : residual noise

Prophet (Taylor & Letham 2018) is the canonical implementation.  The
strength is that non-statistician users can add domain knowledge as
regressor columns and the model absorbs it.

Piecewise linear trend
    Pick S candidate changepoints uniformly on training period.  Fit
    g(t) = k * t + m + sum_s delta_s (t - s)_+
    where delta_s is a change in slope at s.  Regularize delta with an
    L1 penalty (Prophet uses a Laplace prior); most delta_s ~ 0, so only
    a few real changepoints emerge.

Fourier seasonality
    s(t) = sum_{k=1}^K [ a_k cos(2 pi k t / P) + b_k sin(2 pi k t / P) ]
    Fit by OLS.

Holidays
    Binary regressors, optional lag structure.

This module fits the additive components jointly by ridge-regularized OLS.
Not a re-implementation of Prophet -- it is a lighter cousin that shows
the mechanics and produces essentially the same shape on smooth series.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _trend_basis(t, changepoints):
    """Piecewise-linear basis: intercept, t, and hinge functions at changepoints."""
    t = np.asarray(t, dtype=float)
    basis = [np.ones_like(t), t]
    for s in changepoints:
        basis.append(np.maximum(t - s, 0))
    return np.column_stack(basis)


def _fourier_basis(t, period: float, K: int):
    """Fourier series basis: cos and sin at k * (2 pi / period), k = 1..K."""
    t = np.asarray(t, dtype=float)
    cols = []
    for k in range(1, K + 1):
        cols.append(np.cos(2 * math.pi * k * t / period))
        cols.append(np.sin(2 * math.pi * k * t / period))
    return np.column_stack(cols)


def decomposable_fit(t, y, period: float = 7.0, K_fourier: int = 3,
                     n_changepoints: int = 25, holidays: dict = None,
                     l2: float = 0.01) -> dict:
    """Fit an additive trend + Fourier + holiday model by ridge regression.

    t         : time index (e.g. days since start).
    y         : outcome.
    period    : seasonal period.
    K_fourier : number of Fourier harmonics.
    holidays  : dict {name: list_of_t_indices}; each becomes a 0/1 regressor.
    l2        : ridge penalty on all coefficients except the intercept.
    """
    t = np.asarray(t, dtype=float); y = np.asarray(y, dtype=float); n = len(t)
    changepoints = np.quantile(t, np.linspace(0.02, 0.9, n_changepoints))
    X_trend = _trend_basis(t, changepoints)
    X_seas = _fourier_basis(t, period, K_fourier)
    parts = [X_trend, X_seas]
    holiday_names = []
    if holidays:
        for name, idxs in holidays.items():
            col = np.zeros(n); col[list(idxs)] = 1.0
            parts.append(col.reshape(-1, 1))
            holiday_names.append(name)
    X = np.column_stack(parts)
    # Ridge regression (leave intercept unpenalized)
    p = X.shape[1]
    R = l2 * np.eye(p); R[0, 0] = 0
    beta = np.linalg.solve(X.T @ X + R, X.T @ y)
    y_fit = X @ beta
    n_tr = X_trend.shape[1]; n_se = X_seas.shape[1]
    return {"beta": beta, "trend": X_trend @ beta[:n_tr],
            "seasonal": X_seas @ beta[n_tr:n_tr + n_se],
            "holidays": X[:, n_tr + n_se:] @ beta[n_tr + n_se:] if holidays else np.zeros(n),
            "fitted": y_fit,
            "residual": y - y_fit,
            "changepoints": changepoints,
            "period": float(period), "K_fourier": int(K_fourier),
            "holiday_names": holiday_names,
            "method": "Decomposable additive forecast (piecewise-linear trend + Fourier + holidays)"}


def decomposable_forecast(fit, t_new) -> np.ndarray:
    """Extend a fit to new time points."""
    t_new = np.asarray(t_new, dtype=float)
    X_tr = _trend_basis(t_new, fit["changepoints"])
    X_se = _fourier_basis(t_new, fit["period"], fit["K_fourier"])
    n_tr = X_tr.shape[1]; n_se = X_se.shape[1]
    beta = fit["beta"]
    # Holidays default to zero on new dates (users would supply new indicator cols in practice)
    return X_tr @ beta[:n_tr] + X_se @ beta[n_tr:n_tr + n_se]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 365
    t = np.arange(n).astype(float)
    trend_true = 0.02 * t + np.maximum(t - 200, 0) * (-0.03)
    seasonal_true = 3 * np.sin(2 * math.pi * t / 7) + 1 * np.cos(2 * math.pi * t / 7)
    holiday_effect = np.zeros(n); holiday_effect[[50, 200, 300]] = 4
    y = trend_true + seasonal_true + holiday_effect + rng.normal(0, 1, n)

    fit = decomposable_fit(t, y, period=7.0, K_fourier=3,
                            n_changepoints=15,
                            holidays={"event": [50, 200, 300]}, l2=0.05)
    print("=== Decomposable additive fit ===")
    print(f"  in-sample RMSE = {np.sqrt(np.mean(fit['residual'] ** 2)):.3f}  (noise SD = 1.0)")

    # Extrapolate 30 days
    t_new = np.arange(n, n + 30).astype(float)
    y_fc = decomposable_forecast(fit, t_new)
    print(f"  30-day forecast preview: {y_fc[:5].round(2)} ...")

    print("\n=== Decomposition preview (first 30 days) ===")
    print(f"  trend[:5]:    {fit['trend'][:5].round(2)}")
    print(f"  seasonal[:5]: {fit['seasonal'][:5].round(2)}")

    print("\n--- library cross-check (prophet) ---")
    try:
        from prophet import Prophet
        import pandas as pd
        df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=n),
                            "y": y})
        m = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=True)
        m.fit(df)
        f = m.make_future_dataframe(periods=30)
        p = m.predict(f)
        print(f"  prophet in-sample MAE: {np.mean(np.abs(y - p['yhat'].values[:n])):.3f}")
    except Exception as ex:
        print(f"  (prophet not available: {ex})")
