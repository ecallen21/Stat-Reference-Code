"""BSTS — Bayesian Structural Time Series (Scott & Varian 2014).

State-space model with local-level + seasonal components:

    y_t = mu_t + tau_t + epsilon_t
    mu_t = mu_{t-1} + eta_t                    (local level)
    tau_t = -sum_{k=1..S-1} tau_{t-k} + zeta_t (seasonal S)

Simplified from-scratch: run Kalman filter with known variances
and produce forecasts + credible intervals. Full BSTS also has
regression + spike-slab feature selection.
"""

import numpy as np    # arrays


def kalman_filter_local_level_seasonal(y, S, sigma_eps, sigma_eta, sigma_zeta,
                                        sigma_slope=0.02):
    """State: (mu, slope, tau_1, ..., tau_{S-1}) — local-linear trend + seasonal."""
    n = len(y)
    dim = 2 + (S - 1)
    F = np.zeros((dim, dim))
    F[0, 0] = 1.0
    F[0, 1] = 1.0    # mu <- mu + slope
    F[1, 1] = 1.0    # slope random walk
    F[2, 2:] = -1.0    # tau roll with dummy seasonal
    for i in range(3, dim):
        F[i, i - 1] = 1.0
    Q = np.diag([sigma_eta ** 2, sigma_slope ** 2, sigma_zeta ** 2] + [0.0] * (S - 2))
    H = np.zeros(dim)
    H[0] = 1.0    # mu
    H[2] = 1.0    # tau_1
    R = sigma_eps ** 2
    a = np.zeros(dim)
    P = np.eye(dim) * 1e3
    xs = np.empty((n, dim))
    Ps = np.empty((n, dim, dim))
    for t in range(n):
        # predict
        a = F @ a
        P = F @ P @ F.T + Q
        # update
        v = y[t] - H @ a
        Ft = H @ P @ H + R
        K = P @ H / Ft
        a = a + K * v
        P = P - np.outer(K, H) @ P
        xs[t] = a
        Ps[t] = P
    return xs, Ps, F, Q, H, R


def forecast(a, P, F, Q, H, R, horizon):
    """Forward-simulate the Kalman filter for point forecasts + CI."""
    n = horizon
    means = np.empty(n)
    vars_ = np.empty(n)
    for t in range(n):
        a = F @ a
        P = F @ P @ F.T + Q
        means[t] = H @ a
        vars_[t] = H @ P @ H + R
    return means, np.sqrt(vars_)


def demo():
    print("=== BSTS — Bayesian Structural Time Series (Scott-Varian 2014) ===")
    rng = np.random.default_rng(2026)
    S = 12
    n, h = 60, 12
    t = np.arange(n + h)
    trend = 0.4 * t
    seasonal = 3.0 * np.sin(2 * np.pi * t / S) + 1.5 * np.cos(2 * np.pi * t / (S / 2))
    y_all = 20 + trend + seasonal + rng.normal(0, 0.5, size=n + h)
    y_train, y_test = y_all[:n], y_all[n:]

    xs, Ps, F, Q, H, R = kalman_filter_local_level_seasonal(
        y_train, S=S, sigma_eps=0.5, sigma_eta=0.4, sigma_zeta=0.5
    )
    a, P = xs[-1], Ps[-1]
    fcast_mean, fcast_sd = forecast(a, P, F, Q, H, R, h)
    mae = np.mean(np.abs(fcast_mean - y_test))
    coverage = np.mean(
        (fcast_mean - 1.96 * fcast_sd <= y_test) &
        (y_test <= fcast_mean + 1.96 * fcast_sd)
    )
    print(f"  Test MAE          = {mae:.3f}")
    print(f"  95% CI coverage    = {coverage:.2f}  (target 0.95)")
    print(f"  First-6 forecast   : {np.round(fcast_mean[:6], 2)}")
    print(f"  First-6 truth      : {np.round(y_test[:6], 2)}")


if __name__ == "__main__":
    demo()
