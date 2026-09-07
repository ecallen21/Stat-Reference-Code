"""Extended Kalman filter (EKF) (Reference Sec 18.12).

Extension of the Kalman filter to nonlinear dynamics / observations
by LINEARISING around the current mean (first-order Taylor):

    x_{t+1} = f(x_t) + eta_t         eta ~ N(0, Q)
    y_t     = h(x_t) + eps_t          eps ~ N(0, R)

At each cycle, replace f, h with their Jacobians F = df/dx, H = dh/dx
evaluated at the current estimate:

    Predict:  x = f(x_prev),           P = F P F' + Q
    Update:   K = P H' (H P H' + R)^{-1}
              x = x + K (y - h(x))
              P = (I - K H) P

Widely used in navigation (GPS/INS), robotics, target tracking. Fails
when nonlinearity is strong -> use UKF or particle filter.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def ekf(f, F_jac, h, H_jac, Q, R, x0, P0, y_seq):
    n = len(y_seq); d = len(x0)
    x = x0.copy(); P = P0.copy()
    xs = np.zeros((n, d))
    for t in range(n):
        #  Predict
        x = f(x)
        F = F_jac(x)
        P = F @ P @ F.T + Q
        #  Update
        H = H_jac(x)
        y_pred = h(x)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (y_seq[t] - y_pred)
        P = (np.eye(d) - K @ H) @ P
        xs[t] = x
    return xs


if __name__ == "__main__":
    print("=== Extended Kalman filter (EKF) ===\n")
    rng = np.random.default_rng(0)
    T = 80

    #  Nonlinear state: x_{t+1} = 0.9 x + 0.5 sin(x) + eta
    #  Nonlinear obs:   y_t     = x_t^2 / 20 + eps
    def f(x): return 0.9 * x + 0.5 * np.sin(x)
    def F_jac(x): return 0.9 + 0.5 * np.cos(x)     # 1-D Jacobian
    def h(x): return x ** 2 / 20.0
    def H_jac(x): return x / 10.0

    Q = np.array([[0.05]]); R = np.array([[0.1]])
    x_true = np.zeros(T)
    y = np.zeros((T, 1))
    x_true[0] = 2.0
    for t in range(1, T):
        x_true[t] = float(f(x_true[t - 1])) + rng.normal(scale=np.sqrt(Q[0, 0]))
    for t in range(T):
        y[t, 0] = float(h(x_true[t])) + rng.normal(scale=np.sqrt(R[0, 0]))

    xs = ekf(lambda x: np.array([float(f(x[0]))]),
             lambda x: np.array([[float(F_jac(x[0]))]]),
             lambda x: np.array([float(h(x[0]))]),
             lambda x: np.array([[float(H_jac(x[0]))]]),
             Q, R, np.array([1.5]), np.array([[1.0]]), y)

    rmse = float(np.sqrt(np.mean((xs[:, 0] - x_true) ** 2)))
    print(f"  T = {T}, 1-D nonlinear state, quadratic observation")
    print(f"  EKF RMSE vs truth = {rmse:.3f}")

    print("\n--- library cross-check (KFAS + custom Jacobian R; filterpy.ExtendedKalmanFilter Python) ---")
