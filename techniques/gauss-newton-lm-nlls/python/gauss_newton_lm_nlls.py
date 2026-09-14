"""Gauss-Newton / Levenberg-Marquardt (Reference Sec 47.321).

Levenberg 1944 QAM; Marquardt 1963 SIAM. Nonlinear least
squares solves min sum r_i(x)^2 by iterating:

    Gauss-Newton:      (J^T J) dx = -J^T r     (J = Jacobian of r)
    Levenberg-Marquardt: (J^T J + lambda I) dx = -J^T r

LM interpolates between Gauss-Newton (lambda -> 0) and gradient
descent (lambda -> infty); lambda is adapted based on step
quality.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def levenberg_marquardt(residual_fn, jacobian_fn, x0, max_iter=50, tol=1e-8):
    x = x0.copy()
    lam = 1e-3
    r = residual_fn(x); loss = float(r @ r)
    history = [loss]
    for k in range(max_iter):
        J = jacobian_fn(x)
        A = J.T @ J + lam * np.eye(len(x))
        g = J.T @ r
        dx = np.linalg.solve(A, -g)
        x_new = x + dx
        r_new = residual_fn(x_new)
        loss_new = float(r_new @ r_new)
        if loss_new < loss:                                       # accept
            x = x_new; r = r_new; loss = loss_new
            lam = max(lam / 3, 1e-10)                              # trust GN
        else:                                                     # reject, damp
            lam *= 3
        history.append(loss)
        if np.linalg.norm(g) < tol: break
    return x, history


if __name__ == "__main__":
    print("=== Levenberg-Marquardt / Gauss-Newton (LM 1944; Marquardt 1963) ===\n")
    rng = np.random.default_rng(0)

    # Nonlinear regression: y = a * exp(-b * x) + c + noise
    N = 100
    x = np.linspace(0, 5, N)
    y = 3.0 * np.exp(-0.7 * x) + 0.5 + 0.05 * rng.standard_normal(N)

    def residual_fn(theta):
        a, b, c = theta
        return a * np.exp(-b * x) + c - y

    def jacobian_fn(theta):
        a, b, c = theta
        J = np.zeros((N, 3))
        J[:, 0] = np.exp(-b * x)
        J[:, 1] = -a * x * np.exp(-b * x)
        J[:, 2] = 1.0
        return J

    x0 = np.array([1.0, 1.0, 0.0])
    theta_hat, hist = levenberg_marquardt(residual_fn, jacobian_fn, x0, max_iter=50)
    print(f"  True parameters:      a = 3.00,  b = 0.70,  c = 0.50")
    print(f"  LM estimate:          a = {theta_hat[0]:.3f}, "
          f"b = {theta_hat[1]:.3f}, c = {theta_hat[2]:.3f}")
    print(f"  Iterations:           {len(hist) - 1}   (loss {hist[0]:.2f} -> {hist[-1]:.4f})\n")

    print(f"  Loss trajectory:")
    for k in [0, 1, 3, 5, 10, len(hist) - 1]:
        print(f"    iter {k:>3}   RSS = {hist[k]:.4f}")

    print(f"\n  LM behaviour:")
    print(f"    - lam small: near Gauss-Newton, quadratic conv near optimum")
    print(f"    - lam large: steepest-descent style, robust far from optimum")

    print("\n--- library cross-check (scipy.optimize.least_squares(method='lm'); nls in R) ---")
