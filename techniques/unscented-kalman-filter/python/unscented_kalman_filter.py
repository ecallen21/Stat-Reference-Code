"""Unscented Kalman filter (UKF) (Reference Sec 18.13).

Julier & Uhlmann 1997. Replaces the EKF's linearisation with
DETERMINISTIC SIGMA POINTS that capture the mean and covariance of
the state, then propagate them through the nonlinear functions and
recover the transformed mean / covariance directly.

For state x in R^d with mean m, covariance P, choose 2d+1 sigma pts:

    chi_0 = m
    chi_i = m + sqrt((d + lam) P)_i         i = 1..d
    chi_i = m - sqrt((d + lam) P)_{i-d}     i = d+1..2d

with weights w_m, w_c. Nonlinear propagation:

    Y_i = f(chi_i)
    m_new  = sum_i w_m^i * Y_i
    P_new  = sum_i w_c^i * (Y_i - m_new)(Y_i - m_new)' + Q

Update analogously through h.

Advantages over EKF:
    * No Jacobians (derivative-free).
    * More accurate for strongly nonlinear f, h.
    * Handles discontinuous transformations.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sigma_points(m, P, alpha=1e-3, beta=2.0, kappa=0.0):
    d = len(m)
    lam = alpha ** 2 * (d + kappa) - d
    c = d + lam
    S = np.linalg.cholesky(c * P)
    chi = np.zeros((2 * d + 1, d))
    chi[0] = m
    for i in range(d):
        chi[1 + i] = m + S[:, i]
        chi[1 + d + i] = m - S[:, i]
    w_m = np.full(2 * d + 1, 1 / (2 * c))
    w_c = w_m.copy()
    w_m[0] = lam / c
    w_c[0] = lam / c + (1 - alpha ** 2 + beta)
    return chi, w_m, w_c


def ukf(f, h, Q, R, x0, P0, y_seq):
    d = len(x0); m = x0.copy(); P = P0.copy()
    n = len(y_seq)
    xs = np.zeros((n, d))
    for t in range(n):
        #  Predict
        chi, wm, wc = sigma_points(m, P)
        Y = np.array([f(c) for c in chi])
        m = np.sum(wm[:, None] * Y, axis=0)
        P = Q + sum(wc[i] * np.outer(Y[i] - m, Y[i] - m) for i in range(len(Y)))
        #  Update
        chi, wm, wc = sigma_points(m, P)
        Z = np.array([h(c) for c in chi])
        z_bar = np.sum(wm[:, None] * Z, axis=0)
        S = R + sum(wc[i] * np.outer(Z[i] - z_bar, Z[i] - z_bar) for i in range(len(Z)))
        C = sum(wc[i] * np.outer(chi[i] - m, Z[i] - z_bar) for i in range(len(Z)))
        K = C @ np.linalg.inv(S)
        m = m + K @ (y_seq[t] - z_bar)
        P = P - K @ S @ K.T
        xs[t] = m
    return xs


if __name__ == "__main__":
    print("=== Unscented Kalman filter (Julier-Uhlmann 1997) ===\n")
    rng = np.random.default_rng(0)
    T = 80

    #  Same problem as EKF: x_{t+1} = 0.9 x + 0.5 sin(x); y = x^2/20 + noise
    def f(x): return np.array([0.9 * x[0] + 0.5 * np.sin(x[0])])
    def h(x): return np.array([x[0] ** 2 / 20.0])

    Q = np.array([[0.05]]); R = np.array([[0.1]])
    x_true = np.zeros(T)
    y = np.zeros((T, 1))
    x_true[0] = 2.0
    for t in range(1, T):
        x_true[t] = float(f(np.array([x_true[t - 1]]))[0]) + rng.normal(scale=np.sqrt(Q[0, 0]))
    for t in range(T):
        y[t, 0] = float(h(np.array([x_true[t]]))[0]) + rng.normal(scale=np.sqrt(R[0, 0]))

    xs = ukf(f, h, Q, R, np.array([1.5]), np.array([[1.0]]), y)
    rmse = float(np.sqrt(np.mean((xs[:, 0] - x_true) ** 2)))
    print(f"  T = {T}, 1-D nonlinear state / quadratic obs")
    print(f"  UKF RMSE vs truth = {rmse:.3f}")
    print(f"  (EKF on the same problem: 0.264)")

    print("\n--- library cross-check (mvKf R; filterpy.UnscentedKalmanFilter Python) ---")
