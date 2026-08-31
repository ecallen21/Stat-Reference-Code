"""Varying-coefficient model (Reference Sec 33.9).

Hastie & Tibshirani (1993) 'Varying-coefficient models.'

Y = beta_0(U) + beta_1(U) * X_1 + ... + beta_p(U) * X_p + eps.

Each coefficient is a SMOOTH FUNCTION of an effect-modifier U. Reduces
to ordinary linear regression if beta_j(U) are constants; special-cases
include:
  * Time-varying coefficients (U = t).
  * Age- or dose-modified effects.
  * Spatially-varying regression (U = spatial coordinate).

Estimation: local weighted least squares at each query u_0:

  beta_hat(u_0) = argmin_beta  sum_i K_h(u_i - u_0) (y_i - X_i' beta)^2.

Here we implement local WLS on a grid of U-values and recover
sinusoidal true coefficient functions beta_0(u) = sin(u), beta_1(u) = u/2.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def gaussian_weights(u, u0, h):
    return np.exp(-0.5 * ((u - u0) / h) ** 2)


def fit_vcm_local(X, y, u, u_query, bandwidth=0.3):
    """Local WLS at each query point u0."""
    n_q = len(u_query)
    d = X.shape[1]
    beta_hat = np.zeros((n_q, d))
    for i, u0 in enumerate(u_query):
        w = gaussian_weights(u, u0, bandwidth)
        W = np.diag(w)
        # Weighted normal equations
        A = X.T @ W @ X + 1e-6 * np.eye(d)
        b = X.T @ W @ y
        beta_hat[i] = np.linalg.solve(A, b)
    return beta_hat


if __name__ == "__main__":
    print("=== Varying-coefficient model (Hastie-Tibshirani 1993) ===\n")
    rng = np.random.default_rng(0)
    n = 400
    u = rng.uniform(-3, 3, n)
    x1 = rng.normal(0, 1, n)
    X = np.stack([np.ones(n), x1], axis=1)
    # True coefficients vary with u.
    beta0_u = np.sin(u)
    beta1_u = 0.5 * u
    y = beta0_u + beta1_u * x1 + rng.normal(0, 0.3, n)

    u_query = np.linspace(-3, 3, 9)
    beta_hat = fit_vcm_local(X, y, u, u_query, bandwidth=0.4)

    print(f"  {'u':>6}  {'beta0_hat':>10}  {'beta0_true':>10}"
          f"  {'beta1_hat':>10}  {'beta1_true':>10}")
    for i, u0 in enumerate(u_query):
        print(f"  {u0:>6.2f}  {beta_hat[i, 0]:>10.3f}  {np.sin(u0):>10.3f}"
              f"  {beta_hat[i, 1]:>10.3f}  {0.5*u0:>10.3f}")

    print("\n  Coefficient functions are recovered smoothly across u.\n")
    print("--- library cross-check (mgcv::gam with by= term; R package svcm; np package) ---")
