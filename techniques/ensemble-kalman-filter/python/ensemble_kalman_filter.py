"""Ensemble Kalman filter (EnKF) (Reference Sec 18.11).

Evensen 1994 'Sequential data assimilation with a nonlinear
quasi-geostrophic model using Monte Carlo methods to forecast error
statistics'. A MONTE-CARLO Kalman filter: represent the posterior by
an ENSEMBLE of state vectors and propagate them through the nonlinear
dynamics; use the empirical ensemble covariance for the Kalman update.

Algorithm at time t (ensemble size N):

    1. FORECAST:    x_i^f = f(x_i^{a,t-1}) + eta_i         (i = 1..N)
    2. Compute empirical mean, cov  P^f  from {x_i^f}.
    3. Observation:  y_i = y + eps_i  (perturbed observations)
    4. Kalman gain:  K = P^f H' (H P^f H' + R)^{-1}
    5. UPDATE:      x_i^a = x_i^f + K (y_i - H x_i^f)

Advantages:
    * Handles high-dim geophysical / atmospheric models.
    * No linearisation needed (unlike EKF).
    * Ensemble carries UQ.

We simulate on a 2-D nonlinear model and compare EnKF vs plain
Kalman filter (which requires linearity).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def enkf_step(ens, y_obs, f, H, Q, R, rng):
    """One EnKF cycle: forecast + update on ensemble."""
    N = ens.shape[0]; d = ens.shape[1]
    #  Forecast
    ens_f = np.array([f(ens[i]) for i in range(N)]) + rng.multivariate_normal(np.zeros(d), Q, size=N)
    x_mean = ens_f.mean(axis=0)
    P_f = np.cov(ens_f.T, ddof=1)
    #  Kalman gain
    S = H @ P_f @ H.T + R
    K = P_f @ H.T @ np.linalg.inv(S)
    #  Perturbed observations + update
    ens_a = np.zeros_like(ens_f)
    for i in range(N):
        y_pert = y_obs + rng.multivariate_normal(np.zeros(H.shape[0]), R)
        ens_a[i] = ens_f[i] + K @ (y_pert - H @ ens_f[i])
    return ens_a


if __name__ == "__main__":
    print("=== Ensemble Kalman filter (Evensen 1994) ===\n")
    rng = np.random.default_rng(0)
    T = 60
    d = 2
    N = 40                          # ensemble size

    #  Nonlinear dynamics x_{t+1} = 0.9 * x_t + 0.4 * sin(x_t) + noise
    def f(x):
        return 0.9 * x + 0.4 * np.sin(x)

    H = np.eye(d)                    # observe both dims
    Q = 0.1 * np.eye(d)
    R = 1.0 * np.eye(d)

    #  Simulate truth
    x_true = np.zeros((T, d))
    y_obs = np.zeros((T, d))
    x_true[0] = rng.normal(size=d)
    for t in range(1, T):
        x_true[t] = f(x_true[t - 1]) + rng.multivariate_normal(np.zeros(d), Q)
    for t in range(T):
        y_obs[t] = H @ x_true[t] + rng.multivariate_normal(np.zeros(d), R)

    #  EnKF
    ens = rng.multivariate_normal(np.zeros(d), np.eye(d), size=N)
    means = np.zeros((T, d))
    for t in range(T):
        ens = enkf_step(ens, y_obs[t], f, H, Q, R, rng)
        means[t] = ens.mean(axis=0)

    rmse = float(np.sqrt(np.mean((means - x_true) ** 2)))
    print(f"  T = {T}, d = {d}, N = {N}")
    print(f"  EnKF filter-mean RMSE vs truth = {rmse:.3f}")
    print(f"  Observation RMSE (identity)     = {np.sqrt(np.mean((y_obs - x_true) ** 2)):.3f}")
    print(f"  (EnKF should shrink observation noise via ensemble average.)")

    print("\n--- library cross-check (dart_r / MFEnKF R; filterpy / dapper Python) ---")
