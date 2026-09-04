"""Generalized Method of Moments (Reference Sec 35.5).

Hansen (1982) 'Large sample properties of generalized method of moments
estimators.'

Choose parameters theta to make sample moment conditions g_hat(theta)
as close to zero as possible in a weighted quadratic norm:

  theta_hat = argmin_theta  g_hat(theta)' W g_hat(theta),
  where g_hat = (1/n) sum_i g(z_i; theta).

  * Just-identified: dim(g) = dim(theta). W doesn't matter.
  * Over-identified: dim(g) > dim(theta). Optimal W = S^-1 (Hansen).

Two-step GMM:
  1. Solve with W = I, get theta_hat_1.
  2. Estimate S_hat = (1/n) sum g(z_i; theta_hat_1) g(z_i; theta_hat_1)'.
  3. Resolve with W = S_hat^-1 -> theta_hat_2 is efficient.

Overidentifying test (Hansen J):  n * g_hat(theta_hat)' W g_hat(theta_hat)  ~  chi^2(q - p).

Here we implement GMM for a simple mean-estimation problem with
over-identifying moment conditions and verify the efficient two-step
estimator + J test.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays

from scipy.optimize import minimize as _min
from scipy.stats import chi2 as _chi2


def gmm(g_fn, theta_init, W=None, options=None):
    """g_fn(theta) returns an (n, q) matrix of moment contributions."""
    def obj(theta):
        g = g_fn(theta).mean(axis=0)
        return float(g @ W @ g)
    res = _min(obj, theta_init, method="Nelder-Mead", options=options or {"xatol": 1e-8})
    return res.x, float(res.fun)


def two_step_gmm(g_fn, theta_init, options=None):
    q = g_fn(theta_init).shape[1]
    W = np.eye(q)
    theta1, _ = gmm(g_fn, theta_init, W=W, options=options)
    g_i = g_fn(theta1)
    S = g_i.T @ g_i / len(g_i)
    S_reg = S + 1e-6 * np.eye(q)
    W_opt = np.linalg.inv(S_reg)
    theta2, obj = gmm(g_fn, theta1, W=W_opt, options=options)
    return {"theta_1": theta1, "theta_2": theta2, "W_opt": W_opt,
             "J": len(g_i) * obj, "df": q - len(theta_init)}


if __name__ == "__main__":
    print("=== Generalized Method of Moments (Hansen 1982) ===\n")
    rng = np.random.default_rng(0)
    n = 300
    # Truth: X_i ~ N(mu, 1), estimate mu using two moment conditions:
    #  (1) mean = mu
    #  (2) mean of x^2 = mu^2 + 1  (uses variance)
    mu_true = 2.5
    X = rng.normal(mu_true, 1, n)

    def g_fn(theta):
        mu = theta[0]
        return np.stack([X - mu, X ** 2 - (mu ** 2 + 1)], axis=1)

    res = two_step_gmm(g_fn, np.array([0.0]))
    print(f"  n = {n}, true mu = {mu_true}")
    print(f"  theta_hat (step 1, W=I)          = {res['theta_1'][0]:.4f}")
    print(f"  theta_hat (step 2, W = S^-1)     = {res['theta_2'][0]:.4f}")
    print(f"  Hansen J statistic               = {res['J']:.3f}   df = {res['df']}")
    print(f"  p-value                          = {1 - _chi2.cdf(res['J'], res['df']):.3f}")
    print("\n  For a correctly specified model J should be small; p > 0.05 = don't reject.\n")
    print("--- library cross-check (statsmodels.sandbox.regression.gmm; R gmm) ---")
