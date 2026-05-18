"""Gamma regression (Reference §7.25).

GLM for continuous POSITIVE outcomes (costs, lengths of stay, durations):

    Y_i ~ Gamma(shape = 1/phi, scale = phi * mu_i)    => E[Y] = mu, Var[Y] = phi * mu^2

The log link is the conventional choice:

    log mu_i = x_i' beta

so exp(beta_j) is a MULTIPLICATIVE effect on the mean. phi is the dispersion.

Compared to alternatives:
  - OLS on log(Y) models the median (geometric mean) and is fragile to back-
    transformation.
  - Gamma GLM with log link models the mean directly on the original scale.
  - Inverse-Gaussian regression has an even heavier right tail.
  - Tweedie regression generalizes Gamma + Poisson (semi-continuous data).

IRLS:
  w_i = mu_i^2 / Var(Y_i) = 1                    (for log link, dispersion cancels)
  Actually for canonical inverse link the working weights differ; for log link
  with Gamma: w_i = 1 (constant) and z_i = eta_i + (y_i - mu_i) / mu_i.

We use log link (most common in epidemiology and cost modeling).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def fit(X, y, max_iter: int = 50, tol: float = 1e-8) -> dict:
    """Gamma GLM with log link via IRLS."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    if (y <= 0).any():
        raise ValueError("Gamma regression requires strictly positive y")
    n, p = X.shape
    # Init via log(y)
    beta, *_ = np.linalg.lstsq(X, np.log(y), rcond=None)
    ll_prev = -np.inf; converged = False; it = 0
    for it in range(1, max_iter + 1):
        eta = X @ beta; mu = np.exp(np.clip(eta, -500, 500))
        # For log link the IRLS weight is 1 (mu cancels in the weight expression)
        z = eta + (y - mu) / mu
        # Solve normal equations
        beta_new, *_ = np.linalg.lstsq(X, z, rcond=None)
        # Pearson residual squared sum drives dispersion estimate
        if np.max(np.abs(beta_new - beta)) < tol: converged = True; break
        beta = beta_new
    eta = X @ beta; mu = np.exp(eta)
    # Pearson chi^2 -> dispersion phi
    pearson_chi2 = float(np.sum(((y - mu) / mu) ** 2))
    phi = pearson_chi2 / (n - p)
    # Var(beta) = phi * (X' X)^{-1}   (for log link with unit weights)
    var_beta = phi * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(var_beta))
    z_stats = beta / se; p_vals = 2 * stats.norm.sf(np.abs(z_stats))
    # Deviance (gamma form): 2 * sum( -log(y/mu) + (y - mu)/mu )
    dev = 2.0 * float(np.sum(-np.log(y / mu) + (y - mu) / mu))
    return {"beta": beta.tolist(), "se": se.tolist(),
            "z": z_stats.tolist(), "p_values": p_vals.tolist(),
            "exp_beta": np.exp(beta).tolist(),
            "dispersion_phi": phi, "pearson_chi_square": pearson_chi2,
            "deviance": dev, "iterations": it, "converged": converged}


def library_versions(X_no_const, y):
    import statsmodels.api as sm
    X = sm.add_constant(X_no_const)
    res = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
    return {"statsmodels Gamma params": res.params.tolist(),
            "statsmodels Gamma bse": res.bse.tolist(),
            "statsmodels Gamma scale (= phi)": float(res.scale),
            "statsmodels Gamma deviance": float(res.deviance)}


if __name__ == "__main__":
    rng = np.random.default_rng(8)
    n = 300
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    mu_true = np.exp(2.0 + 0.5 * x1 - 0.3 * x2)
    shape = 4.0                                          # 1/phi
    y = rng.gamma(shape, mu_true / shape)                 # mean = mu_true
    X = np.column_stack([np.ones(n), x1, x2])

    res = fit(X, y)
    print(f"=== Gamma regression ({res['iterations']} iters) ===")
    for j, nm in enumerate(["intercept", "x1", "x2"]):
        print(f"  {nm:10s} beta = {res['beta'][j]:+.4f}  SE = {res['se'][j]:.4f}  "
              f"exp(beta) = {res['exp_beta'][j]:.4f}  p = {res['p_values'][j]:.4g}")
    print(f"  dispersion phi = {res['dispersion_phi']:.4f}  (true 1/shape = {1/shape:.4f})")
    print(f"  deviance       = {res['deviance']:.4f}")
    print("\n--- library (statsmodels Gamma + log link) ---")
    for k, v in library_versions(np.column_stack([x1, x2]), y).items():
        print(f"  {k}: {v}")
