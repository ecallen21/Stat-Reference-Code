"""Theta method (Assimakopoulos & Nikolopoulos 2000).

Decompose the series into two "theta lines":
    theta_1 = short-term filtered (SES)
    theta_2 = long-term trend (linear regression, coefficient 2)
Reconstruct forecast as their average. Winner of the M3
competition among simple statistical models.

Simple decomposition:
    theta = 0: linear trend (2 * mean removed via regression)
    theta = 2: doubled curvature emphasises short-term features
Standard variant averages theta=0 (linear trend) and theta=2 (SES).
"""

import numpy as np    # arrays


def ses(y, alpha):
    """Simple Exponential Smoothing (returns level series and one-step-ahead level)."""
    L = np.empty_like(y, dtype=float)
    L[0] = y[0]
    for t in range(1, len(y)):
        L[t] = alpha * y[t] + (1 - alpha) * L[t - 1]
    return L


def theta_forecast(y, horizon, alpha=0.3):
    n = len(y)
    t = np.arange(n)
    # linear regression for theta_0 line (trend)
    x_mean, y_mean = t.mean(), y.mean()
    slope = np.sum((t - x_mean) * (y - y_mean)) / np.sum((t - x_mean) ** 2)
    intercept = y_mean - slope * x_mean
    theta_0 = intercept + slope * t
    theta_0_fc = intercept + slope * np.arange(n, n + horizon)
    # theta_2 line: y with doubled curvature (Assimakopoulos-Nikolopoulos definition)
    theta_2 = 2 * y - theta_0
    # forecast theta_2 with SES
    L = ses(theta_2, alpha)
    theta_2_fc = np.full(horizon, L[-1])
    fc = 0.5 * (theta_0_fc + theta_2_fc)
    return fc


def demo():
    print("=== Theta method (Assimakopoulos-Nikolopoulos 2000) ===")
    rng = np.random.default_rng(2026)
    n, h = 60, 12
    t = np.arange(n + h)
    trend = 0.5 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    noise = rng.normal(0, 1, size=n + h)
    y_all = 20 + trend + seasonal + noise
    y_train, y_test = y_all[:n], y_all[n:]

    fc = theta_forecast(y_train, horizon=h, alpha=0.3)
    naive = np.full(h, y_train[-1])
    linear = np.polyval(np.polyfit(np.arange(n), y_train, 1), np.arange(n, n + h))

    mae_theta = np.mean(np.abs(fc - y_test))
    mae_naive = np.mean(np.abs(naive - y_test))
    mae_linear = np.mean(np.abs(linear - y_test))
    print(f"  Test MAE  theta      = {mae_theta:.3f}")
    print(f"  Test MAE  naive last = {mae_naive:.3f}")
    print(f"  Test MAE  linear     = {mae_linear:.3f}")

    print(f"\n  First-6 theta forecast : {np.round(fc[:6], 2)}")
    print(f"  First-6 truth          : {np.round(y_test[:6], 2)}")


if __name__ == "__main__":
    demo()
