"""Time series anomaly detection (Reference §13.29).

Three complementary approaches:

1) Hampel filter (rolling median + MAD)
    For each t, look at a window of half-width k.  Estimate a robust
    center (median) and robust scale (MAD * 1.4826).  Flag as anomalous
    if |y_t - median| > threshold * MAD.  Robust to isolated outliers
    even when they cluster; no time-structure assumption.

2) STL-residual anomalies
    Decompose y_t = trend + seasonal + remainder via STL.  Flag remainder
    points whose magnitude exceeds 3 * IQR of the remainders.  Handles
    signals with strong trend and seasonality.

3) ARIMA-residual anomalies
    Fit an ARIMA (or any predictive model), flag standardized residuals
    with |z| > threshold.  Extends to prediction-interval-based
    online anomaly detection.

Contrast with 'multivariate-outlier-detection' (Batch 13) which handles
IID multivariate points without time structure.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def hampel_filter(y, window: int = 7, n_sigmas: float = 3.0) -> dict:
    """Rolling-window median + MAD anomaly detection."""
    y = np.asarray(y, dtype=float); n = len(y); k = window
    flags = np.zeros(n, dtype=bool); scores = np.zeros(n)
    for t in range(n):
        lo = max(0, t - k); hi = min(n, t + k + 1)
        w = y[lo:hi]
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        sigma = 1.4826 * mad if mad > 0 else 1e-8
        z = abs(y[t] - med) / sigma
        scores[t] = z
        flags[t] = z > n_sigmas
    return {"z_scores": scores, "outlier_flags": flags,
            "n_flagged": int(flags.sum()),
            "window": int(window), "threshold": float(n_sigmas),
            "method": "Hampel filter (rolling median + MAD)"}


def stl_residual_anomalies(y, period: int, n_iqr: float = 3.0) -> dict:
    """STL decomposition -> flag remainder points beyond n_iqr * IQR."""
    try:
        from statsmodels.tsa.seasonal import STL
    except Exception as ex:
        return {"error": f"statsmodels STL unavailable: {ex}"}
    y = np.asarray(y, dtype=float)
    res = STL(y, period=period, robust=True).fit()
    r = res.resid
    q75, q25 = np.percentile(r, [75, 25])
    iqr = q75 - q25
    lo = q25 - n_iqr * iqr; hi = q75 + n_iqr * iqr
    flags = (r < lo) | (r > hi)
    return {"remainder": r, "outlier_flags": flags,
            "n_flagged": int(flags.sum()),
            "iqr_bounds": (float(lo), float(hi)),
            "method": "STL-remainder IQR anomaly detection"}


def residual_z_anomalies(y_pred, y_obs, sigma_est=None, threshold: float = 3.0) -> dict:
    """Predictive-residual z-score anomaly flags."""
    y_pred = np.asarray(y_pred, dtype=float); y_obs = np.asarray(y_obs, dtype=float)
    resid = y_obs - y_pred
    if sigma_est is None:
        sigma_est = float(np.std(resid))
    z = resid / sigma_est
    flags = np.abs(z) > threshold
    return {"residuals": resid, "z_scores": z, "outlier_flags": flags,
            "n_flagged": int(flags.sum()),
            "sigma_est": float(sigma_est),
            "method": "Standardized-residual anomaly detection"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 300
    t = np.arange(T)
    y = 5 + 0.05 * t + 2 * np.sin(2 * math.pi * t / 30) + rng.normal(0, 0.5, T)
    # Inject 5 anomalies
    anom_idx = [50, 100, 175, 220, 280]
    y_dirty = y.copy()
    for i in anom_idx: y_dirty[i] += rng.choice([-1, 1]) * 5.0

    print("=== Hampel filter ===")
    r = hampel_filter(y_dirty, window=7, n_sigmas=3.0)
    detected = set(np.where(r["outlier_flags"])[0].tolist())
    truth = set(anom_idx)
    print(f"  detected: {sorted(detected)}")
    print(f"  precision = {len(detected & truth) / max(1, len(detected)):.3f}")
    print(f"  recall    = {len(detected & truth) / len(truth):.3f}")

    print("\n=== STL-residual anomalies (period = 30) ===")
    r = stl_residual_anomalies(y_dirty, period=30, n_iqr=3.0)
    if "error" not in r:
        detected = set(np.where(r["outlier_flags"])[0].tolist())
        print(f"  detected: {sorted(detected)}")
        print(f"  precision = {len(detected & truth) / max(1, len(detected)):.3f}")
        print(f"  recall    = {len(detected & truth) / len(truth):.3f}")
    else:
        print(f"  {r['error']}")

    print("\n=== ARIMA-residual anomalies (simple: 1-step lag prediction) ===")
    # Naive 1-step predictor
    y_pred = np.concatenate([[y_dirty[0]], y_dirty[:-1]])
    r = residual_z_anomalies(y_pred, y_dirty, threshold=3.0)
    detected = set(np.where(r["outlier_flags"])[0].tolist())
    print(f"  detected: {sorted(detected)}")
    print(f"  precision = {len(detected & truth) / max(1, len(detected)):.3f}")
    print(f"  recall    = {len(detected & truth) / len(truth):.3f}")
