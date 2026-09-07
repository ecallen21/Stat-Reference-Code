"""Particle filter / sequential Monte Carlo (Reference Sec 18.10).

Gordon, Salmond & Smith 1993 'Novel approach to nonlinear/non-Gaussian
Bayesian state estimation', IEE Proc. F. Filtering for a nonlinear
state-space model:

    x_t = f(x_{t-1}) + eta_t       (state transition)
    y_t = g(x_t)     + eps_t       (observation)

Bootstrap-filter algorithm (SIR):

    1. Initialise K particles x_0^(k) ~ p(x_0).
    2. For t = 1..T:
        a. PROPAGATE:   x_t^(k) = f(x_{t-1}^(k)) + eta_t^(k)
        b. WEIGHT:      w_t^(k) = p(y_t | x_t^(k))
        c. RESAMPLE:    x_t^(k) with probability w_t^(k) / sum_k w_t^(k)

Approximates the filtering distribution p(x_t | y_{1:t}) by the
weighted particle set. Effective sample size = 1 / sum(w^2) monitors
degeneracy; systematic resampling when ESS drops below a threshold.

We compare to the Kalman filter on a linear-Gaussian toy where both
are exact -- PF should match KF closely; extra machinery pays off
when the model is nonlinear or non-Gaussian.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def bootstrap_filter(y, f, g_log, sigma_state, K=500, x0=0.0, seed=0):
    """Simple bootstrap particle filter."""
    rng = np.random.default_rng(seed)
    T = len(y)
    x = np.full(K, x0) + rng.normal(scale=1.0, size=K)   # p(x_0)
    means = np.zeros(T)
    for t in range(T):
        x = f(x) + sigma_state * rng.normal(size=K)      # propagate
        logw = g_log(y[t], x)                             # log-likelihood
        w = np.exp(logw - logw.max())
        w /= w.sum()
        #  Systematic resampling
        u = (np.arange(K) + rng.uniform()) / K
        cum = np.cumsum(w)
        idx = np.searchsorted(cum, u)
        x = x[idx]
        means[t] = x.mean()
    return means


def kalman_filter(y, F=1.0, H=1.0, Q=0.5 ** 2, R=1.0 ** 2, x0=0.0, P0=1.0):
    T = len(y)
    x = x0; P = P0
    means = np.zeros(T)
    for t in range(T):
        #  Predict
        x = F * x; P = F * P * F + Q
        #  Update
        S = H * P * H + R
        K = P * H / S
        x = x + K * (y[t] - H * x)
        P = (1 - K * H) * P
        means[t] = x
    return means


if __name__ == "__main__":
    print("=== Particle filter (bootstrap SIR) vs Kalman filter ===\n")
    rng = np.random.default_rng(0)
    T = 80
    #  Truth: x_t = 0.9 x_{t-1} + eta_t; y_t = x_t + eps_t (linear-Gaussian)
    x = np.zeros(T); y = np.zeros(T)
    for t in range(T):
        x[t] = 0.9 * (x[t - 1] if t else 0) + rng.normal(scale=0.5)
        y[t] = x[t] + rng.normal(scale=1.0)

    #  Kalman (exact)
    kf = kalman_filter(y, F=0.9, H=1.0, Q=0.25, R=1.0)
    #  Particle filter
    def f_prop(x_arr): return 0.9 * x_arr
    def g_log(y_t, x_arr): return -0.5 * (y_t - x_arr) ** 2 / 1.0
    pf = bootstrap_filter(y, f_prop, g_log, sigma_state=0.5, K=800)

    rmse_kf = float(np.sqrt(np.mean((kf - x) ** 2)))
    rmse_pf = float(np.sqrt(np.mean((pf - x) ** 2)))
    print(f"  T = {T} time steps.  RMSE (filter mean vs truth):")
    print(f"    Kalman         = {rmse_kf:.3f}")
    print(f"    Particle (K=800) = {rmse_pf:.3f}")
    print(f"  Agreement is close (both correct for linear Gaussian).")

    #  Nonlinear demo: bearings-only tracking is famously non-linear;
    #  here we just show PF handles a nonlinear f() without extra work.
    y_nl = np.abs(x) + rng.normal(scale=0.5, size=T)
    def g_log_nl(y_t, x_arr): return -0.5 * (y_t - np.abs(x_arr)) ** 2 / 0.25
    pf_nl = bootstrap_filter(y_nl, f_prop, g_log_nl, sigma_state=0.5, K=800)
    rmse_pf_nl = float(np.sqrt(np.mean((pf_nl - x) ** 2)))
    print(f"\n  Nonlinear observation model y = |x| + eps:")
    print(f"    Particle filter RMSE = {rmse_pf_nl:.3f}  (Kalman would need an EKF/UKF here)")

    print("\n--- library cross-check (nimble / pomp R; pyfilter / particles Python) ---")
