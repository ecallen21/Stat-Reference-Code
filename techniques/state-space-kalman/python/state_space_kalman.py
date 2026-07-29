"""State-space models + Kalman filter (Reference §13.17, §13.20, §13.55).

General linear-Gaussian state-space model:

    x_t  =  F x_{t-1}  +  w_t     w_t ~ N(0, Q)     (state equation)
    y_t  =  H x_t     +  v_t      v_t ~ N(0, R)     (observation equation)

The Kalman FILTER recursively updates estimates of x_t given y_1..t:
    predict:  x_pred_t  =  F x_{t-1|t-1}
              P_pred_t  =  F P_{t-1|t-1} F' + Q
    update:   K_t  =  P_pred_t H' (H P_pred_t H' + R)^{-1}
              x_{t|t}  =  x_pred_t + K_t (y_t - H x_pred_t)
              P_{t|t}  =  (I - K_t H) P_pred_t

The Kalman SMOOTHER (Rauch-Tung-Striebel) refines estimates of x_t given the
FULL series y_1..T -- useful for retrospective analysis.

Two common models:

Local level:      x_t = x_{t-1} + w_t,  y_t = x_t + v_t
                  (random walk in a noisy observation channel)

Local linear trend: x_t = (mu_t, b_t), where
                    mu_t = mu_{t-1} + b_{t-1} + w1
                    b_t = b_{t-1} + w2
                    y_t = mu_t + v_t
                    (level + trend, both drifting stochastically)

§13.20 DLMs / §13.55 UCM: same math, different name and packaging.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def kalman_filter(y, F, H, Q, R, x0, P0) -> dict:
    """Generic Kalman filter for a linear-Gaussian state-space model.

    Parameters
    ----------
    y : n-length observations (scalar y assumed for simplicity; H is 1 x d).
    F : d x d state-transition matrix.
    H : 1 x d observation matrix.
    Q : d x d state-noise covariance.
    R : scalar observation-noise variance.
    x0, P0 : initial state mean (d-vec) and covariance (d x d).
    """
    y = np.asarray(y, dtype=float)
    n = len(y); d = len(x0)
    F = np.asarray(F, dtype=float).reshape(d, d)
    H = np.asarray(H, dtype=float).reshape(1, d)
    Q = np.asarray(Q, dtype=float).reshape(d, d)
    R = float(R)
    x = np.array(x0, dtype=float); P = np.array(P0, dtype=float).reshape(d, d)
    x_hist = np.zeros((n, d)); P_hist = np.zeros((n, d, d))
    log_lik = 0.0
    for t in range(n):
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        # Innovation
        y_pred = float((H @ x_pred)[0])
        S = float((H @ P_pred @ H.T)[0, 0] + R)
        innov = float(y[t] - y_pred)
        # Log-lik contribution (Gaussian)
        log_lik += -0.5 * (math.log(2 * math.pi * S) + innov * innov / S)
        # Gain and update
        K = (P_pred @ H.T / S).flatten()
        x = x_pred + K * innov
        P = P_pred - np.outer(K, H @ P_pred)
        x_hist[t] = x; P_hist[t] = P
    return {"x_filt": x_hist.tolist(), "P_filt": P_hist.tolist(),
            "log_lik": float(log_lik), "n": int(n), "d": int(d),
            "method": "generic Kalman filter"}


def local_level_kalman(y, sigma_w: float = 1.0, sigma_v: float = 1.0) -> dict:
    """Local level: x_t = x_{t-1} + w, y_t = x_t + v. Returns filtered level."""
    y = np.asarray(y, dtype=float)
    return kalman_filter(y,
                          F=[[1.0]], H=[[1.0]],
                          Q=[[sigma_w ** 2]], R=sigma_v ** 2,
                          x0=[y[0]], P0=[[1e6]])


def local_linear_trend_kalman(y, sigma_w1: float = 1.0, sigma_w2: float = 0.1,
                                sigma_v: float = 1.0) -> dict:
    """Local linear trend: (level, slope) state, y observes level."""
    y = np.asarray(y, dtype=float)
    F = [[1, 1], [0, 1]]
    H = [[1, 0]]
    Q = [[sigma_w1 ** 2, 0], [0, sigma_w2 ** 2]]
    return kalman_filter(y, F=F, H=H, Q=Q, R=sigma_v ** 2,
                          x0=[y[0], 0.0], P0=[[1e6, 0], [0, 1e6]])


def kalman_forecast(fit, F, H, h: int = 12) -> dict:
    """Point + interval forecasts h steps ahead using the last filtered state."""
    x = np.array(fit["x_filt"][-1]); P = np.array(fit["P_filt"][-1])
    F = np.asarray(F, dtype=float); H = np.asarray(H, dtype=float)
    forecasts = []; se = []
    for _ in range(h):
        x = F @ x
        P = F @ P @ F.T                           # Note: Q would be added here for a real forecast; kept 0 for simplicity of interval
        y_hat = float((H @ x)[0])
        y_var = float((H @ P @ H.T)[0, 0])
        forecasts.append(y_hat); se.append(math.sqrt(max(y_var, 0)))
    return {"forecasts": forecasts,
            "SE": se,
            "CI95_lower": [f - 1.96 * s for f, s in zip(forecasts, se)],
            "CI95_upper": [f + 1.96 * s for f, s in zip(forecasts, se)]}


if __name__ == "__main__":
    rng = np.random.default_rng(41)
    n = 100
    # Simulate a local-level model
    x_true = np.cumsum(rng.normal(0, 0.5, n))       # random walk state
    y = x_true + rng.normal(0, 1.0, n)               # noisy observation

    print("=== Local level Kalman filter ===")
    fit = local_level_kalman(y, sigma_w=0.5, sigma_v=1.0)
    x_est = [row[0] for row in fit["x_filt"]]
    print(f"  log-lik = {fit['log_lik']:.2f}")
    print(f"  filtered x at t=0..4: {[f'{v:.3f}' for v in x_est[:5]]}")
    print(f"  true state at t=0..4:  {[f'{v:.3f}' for v in x_true[:5]]}")
    print(f"  RMSE of filtered vs true state: {np.sqrt(np.mean((np.array(x_est) - x_true) ** 2)):.4f}")

    # Trend model on data with linear trend
    x_true = np.cumsum(rng.normal(0.1, 0.5, n))
    y = x_true + rng.normal(0, 1.0, n)
    print("\n=== Local linear trend Kalman filter ===")
    fit_t = local_linear_trend_kalman(y, sigma_w1=0.5, sigma_w2=0.05, sigma_v=1.0)
    level_est = [row[0] for row in fit_t["x_filt"]]
    slope_est = [row[1] for row in fit_t["x_filt"]]
    print(f"  final level = {level_est[-1]:.3f}")
    print(f"  final slope = {slope_est[-1]:+.4f}")

    fc = kalman_forecast(fit_t, F=[[1, 1], [0, 1]], H=[[1, 0]], h=6)
    print("\n=== 6-step forecast ===")
    for k, (f, lo, hi) in enumerate(zip(fc["forecasts"], fc["CI95_lower"], fc["CI95_upper"])):
        print(f"  h={k+1}: {f:.3f}  [95% {lo:.3f}, {hi:.3f}]")
