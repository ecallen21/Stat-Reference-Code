"""SARIMA + ARIMAX / transfer function (Reference §13.6, §13.25).

SARIMA(p, d, q)(P, D, Q, s): seasonal ARIMA
    - (p, d, q) are the standard non-seasonal orders (on lag-1 dynamics)
    - (P, D, Q) are the SEASONAL orders (on lag-s dynamics)
    - s is the seasonal period (12 for monthly-annual, 4 for quarterly-annual,
      7 for daily-weekly, etc.)

    Applied to a series that also has a within-period cycle. Fitting typically
    handled by state-space MLE (statsmodels SARIMAX / R's arima).

ARIMAX (§13.25): ARIMA + eXogenous regressors:
    x_t = beta' Z_t  +  ARIMA(p, d, q) process on the residual
Interpretation: the regression term absorbs known drivers; ARIMA absorbs
leftover autocorrelated noise. Same as "regression with ARIMA errors."

Transfer function (§13.25): more general -- allows the exogenous input to
propagate to the output through its own ARMA-style dynamics. Rarely built
from scratch outside specialized software; statsmodels supports it via SARIMAX.

This file uses statsmodels SARIMAX (the authoritative implementation) as the
primary interface for both.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def fit_sarima(x, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0)) -> dict:
    """SARIMA fit via statsmodels SARIMAX.

    Parameters
    ----------
    order : (p, d, q) non-seasonal orders.
    seasonal_order : (P, D, Q, s) seasonal orders and period.
    """
    import warnings
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(x, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
    return {"order": order, "seasonal_order": seasonal_order,
            "params": dict(zip(fit.param_names, fit.params.tolist())),
            "AIC": float(fit.aic), "BIC": float(fit.bic),
            "log_lik": float(fit.llf),
            "forecast_next_12": fit.forecast(steps=12).tolist(),
            "method": "SARIMA via statsmodels state-space MLE"}


def fit_arimax(x, exog, order=(1, 0, 1)) -> dict:
    """ARIMAX (regression with ARIMA errors) via SARIMAX with exog."""
    import warnings
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    x = np.asarray(x, dtype=float); exog = np.asarray(exog, dtype=float)
    if exog.ndim == 1: exog = exog[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(x, exog=exog, order=order,
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
    return {"order": order,
            "params": dict(zip(fit.param_names, fit.params.tolist())),
            "AIC": float(fit.aic), "BIC": float(fit.bic),
            "log_lik": float(fit.llf),
            "exog_p_dim": int(exog.shape[1]),
            "method": "ARIMAX (regression with ARIMA errors) via SARIMAX"}


if __name__ == "__main__":
    rng = np.random.default_rng(17)
    n = 240; s = 12
    # SARIMA(1, 0, 0)(1, 1, 0, 12): AR(1) + seasonal AR(1) on 12-month cycle
    t = np.arange(n)
    seasonal = 3 * np.sin(2 * np.pi * t / s)
    trend = 0.02 * t
    ar_noise = np.zeros(n); ar_noise[0] = rng.normal()
    for i in range(1, n):
        ar_noise[i] = 0.6 * ar_noise[i - 1] + rng.normal()
    y = trend + seasonal + ar_noise

    print("=== SARIMA(1, 1, 1)(1, 1, 1, 12) on synthetic monthly data ===")
    fit = fit_sarima(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, s))
    print(f"  params: {fit['params']}")
    print(f"  AIC = {fit['AIC']:.2f}")
    print(f"  next 12 forecasts: {[f'{v:.2f}' for v in fit['forecast_next_12']]}")

    print("\n=== ARIMAX with exogenous seasonal Fourier terms ===")
    # exog: annual sine/cosine to model seasonality
    exog = np.column_stack([np.sin(2 * np.pi * t / s),
                             np.cos(2 * np.pi * t / s)])
    fit_ax = fit_arimax(y, exog, order=(1, 1, 1))
    print(f"  params: {fit_ax['params']}")
    print(f"  AIC = {fit_ax['AIC']:.2f}")
