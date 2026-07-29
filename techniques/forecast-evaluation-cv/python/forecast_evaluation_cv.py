"""Forecast evaluation: time-series CV + accuracy metrics + reconciliation
(Reference §13.23, §13.31, §13.35, §13.36, §13.45, §13.51).

TIME-SERIES CROSS-VALIDATION (§13.23):
    You CANNOT random-split time series (would leak future into past).
    Instead:
        - Expanding window: fit on [1..t], test on t+1..t+h. Grow t.
        - Rolling origin (fixed window): fit on [t-w+1..t], test on t+1..t+h. Slide t.

ACCURACY METRICS (§13.31, §13.51):
    MAE   = mean(|y - y_hat|)                        (units of y)
    RMSE  = sqrt(mean((y - y_hat)^2))                (units of y; penalizes big misses)
    MAPE  = mean(|y - y_hat| / |y|) * 100%           (%; blows up for y ~ 0)
    sMAPE = mean(2 * |y - y_hat| / (|y| + |y_hat|)) * 100%
    MASE  = MAE / naive-MAE                          (scale-free; 1 = as good as naive)
    CRPS  = probabilistic scoring rule; density- or ensemble-based

MULTI-STEP STRATEGIES (§13.35):
    - RECURSIVE: fit one-step model; iterate.
    - DIRECT: fit one model per horizon h.
    - HYBRID: recursive up to some horizon, direct beyond.

HIERARCHICAL RECONCILIATION (§13.45):
    Aggregate constraints when forecasts must sum (regions -> country;
    products -> category). Methods:
    - Bottom-up: sum base-level forecasts.
    - Top-down: allocate top forecast by historical shares.
    - MinT (Wickramasuriya et al. 2019): minimum-trace optimal combination.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Callable    # stdlib: type hint for functions

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mape(y, yhat):
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs((y - yhat) / np.where(np.abs(y) > 1e-12, y, 1))) * 100)


def smape(y, yhat):
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    denom = np.abs(y) + np.abs(yhat)
    return float(np.mean(2 * np.abs(y - yhat) / np.where(denom > 1e-12, denom, 1)) * 100)


def mase(y_train, y_test, yhat_test, m: int = 1):
    """MASE using in-sample naive forecast (seasonal m=1 or m > 1) as baseline."""
    y_train = np.asarray(y_train, dtype=float)
    naive_mae_train = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(mae(y_test, yhat_test) / max(naive_mae_train, 1e-12))


def all_metrics(y_train, y_test, yhat_test, seasonal_m: int = 1) -> dict:
    return {"MAE": mae(y_test, yhat_test),
            "RMSE": rmse(y_test, yhat_test),
            "MAPE_pct": mape(y_test, yhat_test),
            "sMAPE_pct": smape(y_test, yhat_test),
            "MASE": mase(y_train, y_test, yhat_test, seasonal_m)}


def expanding_window_cv(y, fit_forecast_fn: Callable, initial: int, h: int = 1,
                         step: int = 1) -> dict:
    """Expanding-window CV. ``fit_forecast_fn(train, h) -> h forecasts``."""
    y = np.asarray(y, dtype=float); n = len(y)
    origins = list(range(initial, n - h + 1, step))
    all_forecasts = []; all_actuals = []
    per_origin = []
    for t in origins:
        train = y[:t]; test = y[t: t + h]
        fc = np.asarray(fit_forecast_fn(train, h), dtype=float)
        per_origin.append({"origin": t, "forecast": fc.tolist(),
                            "actual": test.tolist()})
        all_forecasts.append(fc); all_actuals.append(test)
    all_forecasts = np.concatenate(all_forecasts)
    all_actuals = np.concatenate(all_actuals)
    metrics = {"MAE": mae(all_actuals, all_forecasts),
                "RMSE": rmse(all_actuals, all_forecasts),
                "MAPE_pct": mape(all_actuals, all_forecasts)}
    return {"per_origin_head": per_origin[:5], "overall_metrics": metrics,
            "n_origins": len(origins), "h": h,
            "method": f"expanding-window CV, h={h}"}


def hierarchical_bottom_up(base_forecasts):
    """Bottom-up: totals are sum of base-level forecasts.

    ``base_forecasts`` : dict {series_name: array of forecasts}. The 'total'
    is the sum of all base-level values.
    """
    total = np.zeros_like(next(iter(base_forecasts.values())))
    for arr in base_forecasts.values():
        total = total + np.asarray(arr, dtype=float)
    return {"base_forecasts": {k: list(v) for k, v in base_forecasts.items()},
            "reconciled_total": total.tolist(),
            "method": "hierarchical bottom-up"}


if __name__ == "__main__":
    rng = np.random.default_rng(47)
    n = 200
    # Linear trend + noise
    t = np.arange(n); y = 10 + 0.3 * t + rng.normal(0, 2, n)

    # A simple naive-drift forecaster
    def drift_forecast(train, h):
        slope = (train[-1] - train[0]) / (len(train) - 1)
        return np.array([train[-1] + slope * k for k in range(1, h + 1)])

    print("=== Expanding-window CV with drift forecaster, h = 1 ===")
    cv = expanding_window_cv(y, drift_forecast, initial=50, h=1, step=5)
    print(f"  origins: {cv['n_origins']}, MAE: {cv['overall_metrics']['MAE']:.3f},"
          f" RMSE: {cv['overall_metrics']['RMSE']:.3f}")

    print("\n=== All accuracy metrics on a single hold-out ===")
    y_train = y[:150]; y_test = y[150:]
    yhat = drift_forecast(y_train, len(y_test))
    for k, v in all_metrics(y_train, y_test, yhat).items():
        print(f"  {k:10s}: {v:.4f}")

    # Bottom-up reconciliation demo
    base = {"region_A": [10, 11, 12],
             "region_B": [5, 6, 7],
             "region_C": [8, 9, 10]}
    print("\n=== Hierarchical bottom-up reconciliation ===")
    r = hierarchical_bottom_up(base)
    print(f"  reconciled TOTAL forecast: {r['reconciled_total']}")
    print(f"  base_forecasts: {r['base_forecasts']}")
