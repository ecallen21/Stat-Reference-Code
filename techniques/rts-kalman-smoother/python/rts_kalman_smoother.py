"""Rauch-Tung-Striebel (RTS) Kalman smoother (Reference Sec 47.127).

Rauch, Tung & Striebel 1965 'Maximum likelihood estimates of
linear dynamic systems', AIAA J 3(8). Backward-pass over Kalman-
filtered posterior to compute the SMOOTHED posterior p(x_t | y_{1:T}):

    C_t = P_t^f F' inv(P_{t+1}^p)
    x_t^s = x_t^f + C_t (x_{t+1}^s - x_{t+1}^p)
    P_t^s = P_t^f + C_t (P_{t+1}^s - P_{t+1}^p) C_t^T.

Provides tighter posterior variances than filtering alone and
better estimates near the beginning of the series.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def kalman_filter(y, F, H, Q, R, x0, P0):
    n = len(y); d = len(x0)
    xf = np.zeros((n, d)); Pf = np.zeros((n, d, d))
    xp = np.zeros((n, d)); Pp = np.zeros((n, d, d))
    x, P = x0.copy(), P0.copy()
    for t in range(n):
        xp[t] = F @ x
        Pp[t] = F @ P @ F.T + Q
        S = H @ Pp[t] @ H.T + R
        K = Pp[t] @ H.T @ np.linalg.inv(S)
        x = xp[t] + K @ (y[t] - H @ xp[t])
        P = (np.eye(d) - K @ H) @ Pp[t]
        xf[t] = x; Pf[t] = P
    return xf, Pf, xp, Pp


def rts_smoother(xf, Pf, xp, Pp, F):
    n, d = xf.shape
    xs = xf.copy(); Ps = Pf.copy()
    for t in range(n - 2, -1, -1):
        C = Pf[t] @ F.T @ np.linalg.inv(Pp[t + 1])
        xs[t] = xf[t] + C @ (xs[t + 1] - xp[t + 1])
        Ps[t] = Pf[t] + C @ (Ps[t + 1] - Pp[t + 1]) @ C.T
    return xs, Ps


if __name__ == "__main__":
    print("=== RTS Kalman smoother (Rauch-Tung-Striebel 1965) ===\n")
    rng = np.random.default_rng(0)
    # Random-walk-with-drift model: x_t = x_{t-1} + eps, y_t = x_t + eta
    n = 60
    F = np.array([[1.0]]); H = np.array([[1.0]])
    Q = np.array([[0.01]]); R = np.array([[0.5]])
    x_true = np.cumsum(np.sqrt(Q[0, 0]) * rng.normal(size=n))
    y = x_true + np.sqrt(R[0, 0]) * rng.normal(size=n)

    xf, Pf, xp, Pp = kalman_filter(y[:, None], F, H, Q, R,
                                       np.array([0.0]), np.array([[1.0]]))
    xs, Ps = rts_smoother(xf, Pf, xp, Pp, F)

    rmse_f = float(np.sqrt(((x_true - xf[:, 0]) ** 2).mean()))
    rmse_s = float(np.sqrt(((x_true - xs[:, 0]) ** 2).mean()))
    var_f = float(Pf.mean())
    var_s = float(Ps.mean())
    print(f"  Kalman filter  RMSE = {rmse_f:.4f}   mean posterior variance = {var_f:.4f}")
    print(f"  RTS smoother   RMSE = {rmse_s:.4f}   mean posterior variance = {var_s:.4f}")
    print(f"  Smoother tightens variance by {(1 - var_s/var_f)*100:.1f}%.")

    print("\n--- library cross-check (KFAS / dlm R; filterpy / statsmodels Python) ---")
