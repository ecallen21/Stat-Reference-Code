"""Exponential Smoothing family (Reference §13.3, §13.43, §13.56).

Family of forecasting methods that weight past observations with exponentially
decaying importance:

    Simple Exponential Smoothing (SES) - level only:
        l_t = alpha * y_t + (1 - alpha) * l_{t-1}
        forecast: y_hat_{t+h} = l_t

    Holt (level + trend):
        l_t = alpha * y_t + (1 - alpha) * (l_{t-1} + b_{t-1})
        b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
        forecast: y_hat_{t+h} = l_t + h * b_t

    Holt-Winters ADDITIVE (level + trend + seasonal):
        l_t = alpha * (y_t - s_{t - m}) + (1 - alpha) * (l_{t-1} + b_{t-1})
        b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
        s_t = gamma * (y_t - l_{t-1} - b_{t-1}) + (1 - gamma) * s_{t - m}
        forecast: y_hat_{t+h} = l_t + h * b_t + s_{t + h - m*ceil(h/m)}

    Holt-Winters MULTIPLICATIVE: same but seasonal component is a ratio.

    ETS (State space, §13.43): unified error/trend/seasonal specification with
      maximum-likelihood estimation of smoothing parameters. Handled by
      statsmodels ETSModel; the from-scratch code here optimizes alpha, beta,
      gamma by grid search / BFGS on SSE.

    TBATS (§13.43): trigonometric BATS extension for multiple seasonalities;
      supported by R's `forecast::tbats`. Noted here rather than implemented.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import optimize    # BFGS on the SSE surface


def ses_forecast(y, alpha: float, h: int = 1) -> dict:
    """Simple Exponential Smoothing with fixed alpha."""
    y = np.asarray(y, dtype=float); n = len(y)
    l = np.empty(n); l[0] = y[0]
    for t in range(1, n):
        l[t] = alpha * y[t] + (1 - alpha) * l[t - 1]
    fitted = np.concatenate([[y[0]], l[:-1]])           # one-step-ahead fitted
    forecast = [float(l[-1])] * h
    sse = float(((y - fitted) ** 2).sum())
    return {"alpha": alpha, "level": l.tolist(),
            "fitted_1step": fitted.tolist(),
            "forecast": forecast, "SSE": sse}


def fit_ses(y, h: int = 12) -> dict:
    """SES with alpha chosen by minimizing SSE."""
    y = np.asarray(y, dtype=float)
    res = optimize.minimize_scalar(lambda a: ses_forecast(y, a)["SSE"],
                                    bounds=(0.001, 0.999), method="bounded")
    alpha = float(res.x)
    return {**ses_forecast(y, alpha, h),
            "method": f"SES with alpha_hat = {alpha:.4f}"}


def holt_forecast(y, alpha: float, beta: float, h: int = 1) -> dict:
    """Holt's method: level + trend, both with fixed smoothing constants."""
    y = np.asarray(y, dtype=float); n = len(y)
    l = np.empty(n); b = np.empty(n)
    l[0] = y[0]; b[0] = y[1] - y[0] if n > 1 else 0.0
    for t in range(1, n):
        l[t] = alpha * y[t] + (1 - alpha) * (l[t - 1] + b[t - 1])
        b[t] = beta * (l[t] - l[t - 1]) + (1 - beta) * b[t - 1]
    fitted = np.concatenate([[y[0]], l[:-1] + b[:-1]])
    forecast = [float(l[-1] + (k + 1) * b[-1]) for k in range(h)]
    sse = float(((y - fitted) ** 2).sum())
    return {"alpha": alpha, "beta": beta,
            "level": l.tolist(), "trend": b.tolist(),
            "fitted_1step": fitted.tolist(),
            "forecast": forecast, "SSE": sse}


def fit_holt(y, h: int = 12) -> dict:
    """Holt's method with (alpha, beta) chosen by SSE minimization."""
    y = np.asarray(y, dtype=float)
    res = optimize.minimize(lambda p: holt_forecast(y, p[0], p[1])["SSE"],
                             x0=[0.3, 0.1], bounds=[(0.001, 0.999)] * 2)
    a, b = float(res.x[0]), float(res.x[1])
    return {**holt_forecast(y, a, b, h),
            "method": f"Holt with (alpha, beta) = ({a:.4f}, {b:.4f})"}


def holt_winters_additive(y, m: int, alpha: float = None, beta: float = None,
                            gamma: float = None, h: int = 12) -> dict:
    """Holt-Winters additive seasonal, m = seasonal period."""
    y = np.asarray(y, dtype=float); n = len(y)
    if n < 2 * m:
        raise ValueError("need at least 2 full seasonal cycles")
    # Initial seasonal via first-cycle deviations from first-cycle mean
    l0 = float(np.mean(y[:m]))
    s0 = y[:m] - l0
    b0 = (np.mean(y[m: 2 * m]) - np.mean(y[:m])) / m

    def sse_hw(params):
        a, b, g = params
        l = np.empty(n); b_arr = np.empty(n); s = np.empty(n + m); fitted = np.empty(n)
        l[0] = l0; b_arr[0] = b0; s[:m] = s0
        for t in range(n):
            l_prev = l[t - 1] if t > 0 else l0
            b_prev = b_arr[t - 1] if t > 0 else b0
            s_t_m = s[t]                      # index into s corresponds to t (offset m)
            if t > 0:
                l[t] = a * (y[t] - s_t_m) + (1 - a) * (l_prev + b_prev)
                b_arr[t] = b * (l[t] - l_prev) + (1 - b) * b_prev
                s[t + m] = g * (y[t] - l_prev - b_prev) + (1 - g) * s_t_m
            fitted[t] = l_prev + b_prev + s_t_m if t > 0 else y[0]
        return float(((y - fitted) ** 2).sum()), l, b_arr, s

    if alpha is None or beta is None or gamma is None:
        res = optimize.minimize(lambda p: sse_hw(p)[0],
                                 x0=[0.3, 0.1, 0.1], bounds=[(0.001, 0.999)] * 3)
        alpha, beta, gamma = res.x
    _, l, b_arr, s = sse_hw([alpha, beta, gamma])
    # Forecast h steps
    forecast = []
    for k in range(1, h + 1):
        seas_idx = ((n - m) + ((k - 1) % m)) + m       # index into s array
        forecast.append(float(l[-1] + k * b_arr[-1] + s[seas_idx if seas_idx < len(s) else -m]))
    return {"alpha": float(alpha), "beta": float(beta), "gamma": float(gamma),
            "m": m, "level_last": float(l[-1]), "trend_last": float(b_arr[-1]),
            "seasonal_last_m": s[-m:].tolist(),
            "forecast": forecast,
            "method": f"Holt-Winters additive (m={m})"}


def library_versions(y, m):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    fit = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=m).fit()
    return {"statsmodels HW params": {
                "alpha": float(fit.model.params["smoothing_level"]),
                "beta": float(fit.model.params["smoothing_trend"]),
                "gamma": float(fit.model.params["smoothing_seasonal"])},
            "statsmodels forecast head 5": fit.forecast(steps=5).tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(19)
    n = 120; m = 12
    t = np.arange(n)
    y = 10 + 0.05 * t + 2 * np.sin(2 * np.pi * t / m) + rng.normal(0, 0.5, n)

    print("=== SES ===")
    ses = fit_ses(y, h=6)
    print(f"  {ses['method']}")
    print(f"  forecast next 6 (flat): {[f'{v:.2f}' for v in ses['forecast']]}")

    print("\n=== Holt (level + trend) ===")
    ht = fit_holt(y, h=6)
    print(f"  {ht['method']}")
    print(f"  forecast next 6: {[f'{v:.2f}' for v in ht['forecast']]}")

    print("\n=== Holt-Winters additive (level + trend + seasonal) ===")
    hw = holt_winters_additive(y, m=m, h=12)
    print(f"  alpha, beta, gamma = ({hw['alpha']:.3f}, {hw['beta']:.3f}, {hw['gamma']:.3f})")
    print(f"  forecast next 12: {[f'{v:.2f}' for v in hw['forecast']]}")

    print("\n--- library (statsmodels HW) ---")
    for k, v in library_versions(y, m).items():
        print(f"  {k}: {v}")
