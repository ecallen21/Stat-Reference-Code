"""GAMLSS — Generalized Additive Models for Location, Scale, and Shape
(Reference Sec 33.6).

Rigby & Stasinopoulos (2005) 'Generalized additive models for location,
scale and shape.'

Model the FULL conditional distribution of Y by letting each
distribution parameter be its own regression:

  Y | X ~ D(mu = f1(X),  sigma = f2(X),  nu = f3(X),  tau = f4(X))

For a Gaussian:  D = N(mu(X), sigma(X)^2)  -> location + scale.
For a Beta:      mu, sigma  or  mu, phi (concentration).
For a t:         mu, sigma, nu (df)  -> location + scale + tails.

Advantages:
  * Heteroscedasticity, skew, kurtosis are ALL regressed.
  * Prediction intervals correctly track scale.
  * Distribution-free family choice (100+ available in R package gamlss).

Here we implement Gaussian GAMLSS (mean and log-sd both linear in x)
via alternating IRLS/coordinate descent, fit synthetic heteroscedastic
data, and compare parameter recovery + prediction interval width to
OLS + fixed-sd assumption.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays

from scipy.stats import norm as _norm


def fit_gamlss_normal(X_mu, X_sigma, y, lr=0.01, epochs=2000, seed=0):
    """Fit Y ~ N(X_mu @ beta, exp(X_sigma @ gamma)^2) via full-data gradient."""
    rng = np.random.default_rng(seed)
    beta = np.zeros(X_mu.shape[1])
    gamma = np.zeros(X_sigma.shape[1])
    n = len(y)
    for _ in range(epochs):
        mu = X_mu @ beta
        log_sig = X_sigma @ gamma
        sig2 = np.exp(2 * log_sig)
        r = y - mu
        # Neg-log-lik = 0.5 (2 log_sig + r^2 / sig2)
        d_beta = -X_mu.T @ (r / sig2) / n
        d_gamma = X_sigma.T @ (1 - r ** 2 / sig2) / n
        beta -= lr * d_beta
        gamma -= lr * d_gamma
    return beta, gamma


if __name__ == "__main__":
    print("=== GAMLSS (Gaussian: mean + log-sd both linear in X) ===\n")
    rng = np.random.default_rng(0)
    n = 400
    x = np.linspace(-2, 2, n)
    mu_true = 1.0 + 0.5 * x
    sig_true = np.exp(-0.5 + 0.6 * x)                 # sd RISES with x
    y = mu_true + sig_true * rng.normal(0, 1, n)

    X_mu = np.stack([np.ones(n), x], axis=1)
    X_sigma = np.stack([np.ones(n), x], axis=1)

    beta, gamma = fit_gamlss_normal(X_mu, X_sigma, y)
    print(f"  mean betas       [intercept, slope]: {np.round(beta, 3).tolist()}   truth [1.000, 0.500]")
    print(f"  log-sd gammas    [intercept, slope]: {np.round(gamma, 3).tolist()}   truth [-0.500, 0.600]")

    # OLS + fixed-sd baseline
    B = np.linalg.lstsq(X_mu, y, rcond=None)[0]
    resid_sd = float((y - X_mu @ B).std())
    print(f"\n  Baseline OLS betas               : {np.round(B, 3).tolist()}   fixed sd = {resid_sd:.3f}")

    # Prediction interval widths (95%) at chosen x-values
    print(f"\n  95% predictive band width at three x-values:")
    print(f"  {'x':>5}  {'GAMLSS':>10}  {'OLS-fixed':>10}   (True sd)")
    z = 1.96
    for xv in (-1.5, 0.0, 1.5):
        row = np.array([1.0, xv])
        w_gam = 2 * z * float(np.exp(row @ gamma))
        w_ols = 2 * z * resid_sd
        true_sd = float(np.exp(-0.5 + 0.6 * xv))
        print(f"  {xv:>5.1f}  {w_gam:>10.3f}  {w_ols:>10.3f}   ({true_sd:.3f})")

    print("\n--- library cross-check (R package gamlss; Python distfit; pyGAM.LinearGAM) ---")
