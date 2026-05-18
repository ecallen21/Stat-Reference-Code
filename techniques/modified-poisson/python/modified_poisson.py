"""Modified (robust) Poisson regression for risk ratios (Reference §7.9, §7.53).

When the outcome is BINARY (0/1) but you want the RISK RATIO instead of the
odds ratio, Zou (2004) showed that fitting a Poisson GLM with log link gives
unbiased log-RR estimates -- but with the wrong SEs (Poisson assumes
Var = mean, while Bernoulli has Var = p (1-p)).

Fix: refit the Poisson and replace the SEs with HUBER-WHITE SANDWICH SEs:

    Var_robust(beta) = (X' W X)^{-1} ( X' diag(e_i^2) X ) (X' W X)^{-1}

with e_i = y_i - mu_i. The point estimates are the Poisson MLEs; only the SEs
change.

Why use this:
  - With common outcomes (prevalence > 10%), odds ratios overstate the RR.
  - Log-binomial (Y ~ Bernoulli, log link) is the "right" model but often
    fails to converge.
  - Modified Poisson is the standard workaround in epidemiology.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def modified_poisson(X, y) -> dict:
    """Poisson fit + sandwich SEs for risk-ratio estimation from binary y.

    Parameters
    ----------
    X : design matrix (intercept included).
    y : 0/1 outcomes.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    # Poisson IRLS for beta (same as techniques/poisson-regression)
    beta = np.zeros(p); ll_prev = -np.inf
    for _ in range(50):
        eta = X @ beta
        mu = np.exp(np.clip(eta, -500, 500))
        w = np.clip(mu, 1e-12, None)
        z = eta + (y - mu) / w
        sw = np.sqrt(w); Xw = X * sw[:, None]; zw = z * sw
        beta_new, *_ = np.linalg.lstsq(Xw, zw, rcond=None)
        ll = float(np.sum(y * np.log(np.clip(mu, 1e-15, None)) - mu))
        if abs(ll - ll_prev) < 1e-8: break
        ll_prev = ll; beta = beta_new
    eta = X @ beta; mu = np.exp(eta)
    e = y - mu
    # Naive Poisson variance (would be wrong)
    XtWX = X.T @ (X * mu[:, None])
    A_inv = np.linalg.pinv(XtWX)
    naive_se = np.sqrt(np.diag(A_inv))
    # Sandwich (Huber-White) variance
    B = X.T @ (X * (e ** 2)[:, None])
    var_robust = A_inv @ B @ A_inv
    # HC0 -> HC1 small-sample correction
    var_robust *= n / (n - p)
    se_robust = np.sqrt(np.diag(var_robust))
    z_stats = beta / se_robust
    p_vals = 2 * stats.norm.sf(np.abs(z_stats))
    rr = np.exp(beta); zc = stats.norm.ppf(0.975)
    return {"beta": beta.tolist(),
            "naive_poisson_se": naive_se.tolist(),
            "robust_se": se_robust.tolist(),
            "z": z_stats.tolist(), "p_values": p_vals.tolist(),
            "risk_ratio": rr.tolist(),
            "rr_ci_lower": np.exp(beta - zc * se_robust).tolist(),
            "rr_ci_upper": np.exp(beta + zc * se_robust).tolist()}


def library_versions(X_no_const, y):
    import statsmodels.api as sm
    X = sm.add_constant(X_no_const)
    res = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type="HC1")
    return {"statsmodels Poisson HC1 params": res.params.tolist(),
            "statsmodels Poisson HC1 bse": res.bse.tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(6)
    n = 500
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    # Common-outcome scenario (~40% prevalence)
    eta = -0.4 + 0.4 * x1 - 0.3 * x2
    p_true = np.clip(np.exp(eta), 0, 1)        # log link on Bernoulli
    y = (rng.uniform(0, 1, n) < p_true).astype(int)
    X = np.column_stack([np.ones(n), x1, x2])

    print("=== Modified Poisson (binary y) ===")
    res = modified_poisson(X, y)
    for j, nm in enumerate(["intercept", "x1", "x2"]):
        print(f"  {nm:10s} beta = {res['beta'][j]:+.4f}")
        print(f"             naive SE = {res['naive_poisson_se'][j]:.4f}  "
              f"robust SE = {res['robust_se'][j]:.4f}")
        print(f"             RR = {res['risk_ratio'][j]:.4f} "
              f"[{res['rr_ci_lower'][j]:.3f}, {res['rr_ci_upper'][j]:.3f}]  "
              f"p = {res['p_values'][j]:.4g}")

    print("\n--- library (statsmodels.GLM Poisson, cov_type='HC1') ---")
    for k, v in library_versions(np.column_stack([x1, x2]), y).items():
        print(f"  {k}: {v}")
