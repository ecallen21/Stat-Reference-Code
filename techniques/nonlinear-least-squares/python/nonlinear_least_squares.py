"""Nonlinear least squares (Reference Sec 35.6).

Fit y_i = f(x_i; theta) + eps_i by minimising sum (y_i - f(x_i; theta))^2.

Gauss-Newton iteration:
  J_ij = df(x_i; theta) / dtheta_j
  theta <- theta + (J' J)^-1 J' r,  r = y - f(theta).

Levenberg-Marquardt adds a damping term:
  theta <- theta + (J' J + lambda I)^-1 J' r.

Here we fit a Michaelis-Menten equation  y = V x / (K + x)  and a 4-PL
logistic  y = a + (d - a) / (1 + (x / c)^b) via Levenberg-Marquardt.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def lm_fit(f, jac, y, x, theta0, lam=1e-2, max_iter=100, tol=1e-8):
    theta = np.array(theta0, dtype=float).copy()
    for it in range(max_iter):
        r = y - f(x, theta)
        J = jac(x, theta)
        gT = J.T @ r
        H = J.T @ J
        d_theta = np.linalg.solve(H + lam * np.diag(np.diag(H)), gT)
        theta_new = theta + d_theta
        r_new = y - f(x, theta_new)
        if float(r_new @ r_new) < float(r @ r):
            theta = theta_new
            lam *= 0.7
        else:
            lam *= 2.0
        if np.max(np.abs(d_theta)) < tol:
            break
    return theta, it + 1


def michaelis_menten(x, theta):
    V, K = theta
    return V * x / (K + x)


def mm_jac(x, theta):
    V, K = theta
    denom = K + x
    return np.stack([x / denom, -V * x / denom ** 2], axis=1)


def four_pl(x, theta):
    a, b, c, d = theta
    return a + (d - a) / (1 + (x / c) ** b)


def four_pl_jac(x, theta):
    a, b, c, d = theta
    xoc = x / c
    ratio = xoc ** b
    denom = 1 + ratio
    log_xoc = np.log(np.maximum(x, 1e-9) / c)
    dda = 1 - 1 / denom
    ddd = 1 / denom
    ddb = -(d - a) * ratio * log_xoc / denom ** 2
    ddc = (d - a) * b * ratio / (c * denom ** 2)
    return np.stack([dda, ddb, ddc, ddd], axis=1)


if __name__ == "__main__":
    print("=== Nonlinear least squares (Levenberg-Marquardt) ===\n")
    rng = np.random.default_rng(0)

    # Michaelis-Menten
    x = np.linspace(0.5, 20, 40)
    V_true, K_true = 5.0, 3.0
    y = V_true * x / (K_true + x) + rng.normal(0, 0.15, len(x))
    theta_hat, n_iter = lm_fit(michaelis_menten, mm_jac, y, x, [1.0, 1.0])
    print(f"  Michaelis-Menten fit:  V_hat = {theta_hat[0]:.3f} (true {V_true}), "
          f"K_hat = {theta_hat[1]:.3f} (true {K_true})   iterations = {n_iter}")

    # 4PL logistic dose-response
    x = np.logspace(-1, 3, 40)
    theta_true = [0.05, 1.5, 30.0, 1.0]           # a, b, c, d
    y = four_pl(x, theta_true) + rng.normal(0, 0.02, len(x))
    theta_hat, n_iter = lm_fit(four_pl, four_pl_jac, y, x,
                                 [0.1, 1.0, 10.0, 0.8])
    print(f"  4-PL logistic fit:")
    print(f"    a_hat = {theta_hat[0]:.4f} (true {theta_true[0]})   b_hat = {theta_hat[1]:.4f} (true {theta_true[1]})")
    print(f"    c_hat = {theta_hat[2]:.4f} (true {theta_true[2]})   d_hat = {theta_hat[3]:.4f} (true {theta_true[3]})")
    print(f"    iterations = {n_iter}\n")
    print("--- library cross-check (scipy.optimize.curve_fit / least_squares; R nls / minpack.lm) ---")
