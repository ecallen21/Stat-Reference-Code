"""Stochastic Frontier Analysis (Reference Sec 35.22).

Aigner, Lovell & Schmidt (1977); Meeusen & van den Broeck (1977).

Production frontier:  y_i = x_i' beta + v_i - u_i
  v_i ~ N(0, sigma_v^2)     (symmetric noise)
  u_i ~ N^+(0, sigma_u^2)   (half-normal INEFFICIENCY, u_i >= 0)

The composite error e_i = v_i - u_i has a NORMAL-HALF-NORMAL density:

  f(e) = 2 / sigma * phi(e / sigma) * Phi(-e * lambda / sigma)
  with lambda = sigma_u / sigma_v,  sigma = sqrt(sigma_v^2 + sigma_u^2).

Firm-specific inefficiency estimated by Jondrow et al. (1982):
  E[u_i | e_i] = sigma* * (phi(mu*_i / sigma*) / (1 - Phi(-mu*_i / sigma*)) - mu*_i / sigma*).

Here we fit SFA by MLE + report the technical-efficiency scores
TE_i = exp(-u_i).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays

from scipy.optimize import minimize as _min
from scipy.stats import norm as _norm


def sfa_neg_log_lik(params, X, y):
    n, k = X.shape
    beta = params[:k]
    log_sigma = params[k]; log_lam = params[k + 1]
    sigma = np.exp(log_sigma); lam = np.exp(log_lam)
    e = y - X @ beta
    z = e * lam / sigma
    return -np.sum(np.log(2 / sigma * _norm.pdf(e / sigma) * _norm.cdf(-z) + 1e-30))


def fit_sfa(X, y):
    k = X.shape[1]
    beta0 = np.linalg.lstsq(X, y, rcond=None)[0]
    x0 = np.concatenate([beta0, [0.0, 0.0]])
    res = _min(sfa_neg_log_lik, x0, args=(X, y), method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
    beta = res.x[:k]
    sigma = float(np.exp(res.x[k])); lam = float(np.exp(res.x[k + 1]))
    sigma_u = float(sigma * lam / np.sqrt(1 + lam ** 2))
    sigma_v = float(sigma / np.sqrt(1 + lam ** 2))
    e = y - X @ beta
    # Jondrow et al. E[u_i | e_i]
    sigma_star = sigma_u * sigma_v / sigma
    mu_star = -e * sigma_u ** 2 / sigma ** 2
    u_hat = sigma_star * (_norm.pdf(mu_star / sigma_star)
                            / (1 - _norm.cdf(-mu_star / sigma_star) + 1e-30)
                            + mu_star / sigma_star)
    te = np.exp(-u_hat)
    return {"beta": beta, "sigma": sigma, "lambda": lam,
             "sigma_u": sigma_u, "sigma_v": sigma_v, "TE": te}


if __name__ == "__main__":
    print("=== Stochastic frontier analysis (Aigner-Lovell-Schmidt 1977) ===\n")
    rng = np.random.default_rng(0)
    n = 300
    x = np.stack([np.ones(n), rng.normal(0, 1, n), rng.normal(0, 1, n)], axis=1)
    beta_true = np.array([2.0, 0.5, 0.3])
    v = rng.normal(0, 0.3, n)
    u = np.abs(rng.normal(0, 0.7, n))                 # half-normal inefficiency
    y = x @ beta_true + v - u

    fit = fit_sfa(x, y)
    print(f"  true beta = {beta_true}")
    print(f"  estimated beta = {fit['beta'].round(3).tolist()}")
    print(f"  sigma_v (true 0.30, est {fit['sigma_v']:.3f})   "
          f"sigma_u (true ~0.70, est {fit['sigma_u']:.3f})")
    print(f"  lambda (sigma_u / sigma_v, true ~2.33, est {fit['lambda']:.3f})")
    print(f"\n  Technical-efficiency scores (TE = exp(-u_i)) summary:")
    print(f"    mean = {fit['TE'].mean():.3f}   median = {np.median(fit['TE']):.3f}"
          f"   min = {fit['TE'].min():.3f}   max = {fit['TE'].max():.3f}\n")
    print("--- library cross-check (R frontier; Python pysfa; frontier via reticulate) ---")
