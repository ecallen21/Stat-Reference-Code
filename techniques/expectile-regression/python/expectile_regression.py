"""Expectile regression (Reference Ch 33 semiparametric).

Newey & Powell (1987) 'Asymmetric least squares estimation and testing.'

Expectiles are the ASYMMETRIC-SQUARED-LOSS analogue of quantiles:

  min_theta   sum_i  w_tau(y_i - theta)  * (y_i - theta)^2
    where  w_tau(u) = tau   if u > 0
                    = 1-tau if u <= 0.

Solved by iteratively-reweighted least squares (IRLS): given current
theta_hat, compute weights w_i and refit weighted regression.

Advantages over QR:
  * SMOOTH DIFFERENTIABLE loss => faster + exact CI via sandwich.
  * Sensitive to distribution SPREAD (unlike median-based QR).
  * Coherent risk measure (Bellini-Klar-Muller-Rosazza-Gianin 2014).

Disadvantages:
  * NOT invariant to monotone transformations of y.
  * Less interpretable than quantiles (not a probability).

Here we implement IRLS-based expectile regression at tau = 0.10, 0.50,
0.90 on synthetic heteroscedastic data.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def expectile_weights(r, tau):
    return np.where(r > 0, tau, 1 - tau)


def fit_expectile(X, y, tau=0.5, tol=1e-7, max_iter=100):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(max_iter):
        r = y - X @ beta
        w = expectile_weights(r, tau)
        WX = X * w[:, None]
        beta_new = np.linalg.solve(WX.T @ X + 1e-8 * np.eye(X.shape[1]),
                                     WX.T @ y)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new; break
        beta = beta_new
    return beta


if __name__ == "__main__":
    print("=== Expectile regression (Newey-Powell 1987) ===\n")
    rng = np.random.default_rng(0)
    n = 400
    x = np.linspace(-2, 2, n)
    y = 1.0 + 0.5 * x + (0.4 + 0.6 * np.abs(x)) * rng.normal(0, 1, n)
    X = np.stack([np.ones(n), x], axis=1)

    print(f"  {'tau':>5}  {'intercept':>10}  {'slope':>7}   (mean at tau=0.5 should ~ OLS)")
    for tau in (0.10, 0.25, 0.50, 0.75, 0.90):
        b = fit_expectile(X, y, tau=tau)
        print(f"  {tau:>5.2f}  {b[0]:>10.3f}  {b[1]:>7.3f}")

    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"\n  OLS:  intercept={beta_ols[0]:.3f}   slope={beta_ols[1]:.3f}"
          "   (matches expectile at tau=0.5)\n")
    print("--- library cross-check (R expectreg; statsmodels ExpectileRegressor; pyExpectiles) ---")
