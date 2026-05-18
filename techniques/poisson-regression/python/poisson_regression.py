"""Poisson regression (Reference §7.12, §7.43).

Model count outcomes:
    Y_i ~ Poisson(mu_i),   log(mu_i) = x_i' beta + offset_i

The log link is canonical; exp(beta_j) is the INCIDENCE RATE RATIO (IRR) for
x_j. The OFFSET term log(exposure_i) -- person-time, area, etc. -- lets the
fitted mu represent a RATE (events per unit exposure) when both sides of the
equation are needed.

IRLS:
  mu_i = exp(eta_i + offset_i)
  w_i  = mu_i                     (Poisson variance = mean)
  z_i  = eta_i + (y_i - mu_i) / mu_i
  beta <- (X' W X)^-1 X' W z      (weighted least squares step)

Assumes Var(Y_i) = mu_i. If actual variance > mean (OVERDISPERSION), the
standard errors are too small. Remedies (next techniques): negative binomial,
quasi-Poisson, modified Poisson (sandwich SEs), or zero-inflated models.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def fit(X, y, offset=None, max_iter: int = 50, tol: float = 1e-8) -> dict:
    """Poisson GLM by IRLS.

    Parameters
    ----------
    X : design matrix (intercept included).
    y : nonneg integer count response.
    offset : optional log(exposure) vector for rate modeling.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    if offset is None:
        offset = np.zeros(X.shape[0])
    offset = np.asarray(offset, dtype=float)
    n, p = X.shape
    # Initial: log(y + 0.5) so we don't take log(0)
    eta = np.log(np.maximum(y, 0.5)) - offset
    beta, *_ = np.linalg.lstsq(X, eta, rcond=None)
    ll_prev = -np.inf; converged = False; it = 0
    for it in range(1, max_iter + 1):
        eta = X @ beta
        mu = np.exp(np.clip(eta + offset, -500, 500))
        w = np.clip(mu, 1e-12, None)
        z = eta + (y - mu) / w
        sw = np.sqrt(w)
        Xw = X * sw[:, None]; zw = z * sw
        beta_new, *_ = np.linalg.lstsq(Xw, zw, rcond=None)
        # log-lik (drop constant -log(y!))
        ll = float(np.sum(y * np.log(np.clip(mu, 1e-15, None)) - mu))
        if abs(ll - ll_prev) < tol:
            converged = True
            break
        ll_prev = ll; beta = beta_new
    eta = X @ beta; mu = np.exp(eta + offset); w = np.clip(mu, 1e-12, None)
    var_beta = np.linalg.pinv(X.T @ (X * w[:, None]))
    se = np.sqrt(np.diag(var_beta))
    z_stats = beta / se; p_vals = 2 * stats.norm.sf(np.abs(z_stats))
    irr = np.exp(beta); zc = stats.norm.ppf(0.975)
    irr_lo = np.exp(beta - zc * se); irr_hi = np.exp(beta + zc * se)
    pearson_chi2 = float(np.sum((y - mu) ** 2 / np.clip(mu, 1e-12, None)))
    pearson_dispersion = pearson_chi2 / (n - p)
    deviance = 2 * float(np.sum(
        y * np.log(np.where(y > 0, y, 1) / np.clip(mu, 1e-12, None)) - (y - mu)
    ))
    aic = 2 * p - 2 * float(np.sum(y * np.log(np.clip(mu, 1e-15, None)) - mu
                                    - np.array([math.lgamma(yi + 1) for yi in y])))
    return {"beta": beta.tolist(), "se": se.tolist(),
            "z": z_stats.tolist(), "p_values": p_vals.tolist(),
            "irr": irr.tolist(),
            "irr_ci_lower": irr_lo.tolist(), "irr_ci_upper": irr_hi.tolist(),
            "deviance": deviance, "pearson_chi_square": pearson_chi2,
            "pearson_dispersion": pearson_dispersion,
            "aic": aic, "iterations": it, "converged": converged}


def library_versions(X_no_const, y, offset=None):
    import statsmodels.api as sm
    X = sm.add_constant(X_no_const)
    res = sm.GLM(y, X, family=sm.families.Poisson(),
                 offset=offset).fit()
    return {"statsmodels Poisson params": res.params.tolist(),
            "statsmodels Poisson bse": res.bse.tolist(),
            "statsmodels Poisson deviance": float(res.deviance),
            "statsmodels Poisson aic": float(res.aic)}


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    n = 300
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    exposure = rng.uniform(0.5, 3.0, n)
    eta = 0.5 + 0.6 * x1 - 0.3 * x2
    mu = exposure * np.exp(eta)
    y = rng.poisson(mu)
    X = np.column_stack([np.ones(n), x1, x2])

    print("=== Poisson with offset = log(exposure) ===")
    res = fit(X, y, offset=np.log(exposure))
    for j, nm in enumerate(["intercept", "x1", "x2"]):
        print(f"  {nm:10s} beta = {res['beta'][j]:+.4f}  SE = {res['se'][j]:.4f}  "
              f"IRR = {res['irr'][j]:.4f} [{res['irr_ci_lower'][j]:.3f}, "
              f"{res['irr_ci_upper'][j]:.3f}]  p = {res['p_values'][j]:.4g}")
    print(f"  deviance = {res['deviance']:.4f}   "
          f"Pearson dispersion = {res['pearson_dispersion']:.4f}   "
          f"AIC = {res['aic']:.4f}")
    print("  (Pearson dispersion >> 1 -> overdispersion; consider NB or modified Poisson.)")

    print("\n--- library (statsmodels GLM Poisson) ---")
    for k, v in library_versions(np.column_stack([x1, x2]), y,
                                 offset=np.log(exposure)).items():
        print(f"  {k}: {v}")
