"""Ordinal logistic regression (Proportional Odds Model) (Reference §7.2).

For an ordinal outcome Y in {1, 2, ..., K}, model the K-1 cumulative logits
with a SHARED slope vector:

    logit P(Y <= k | x) = alpha_k - x' beta,    k = 1, ..., K-1

  alpha_1 < alpha_2 < ... < alpha_{K-1} are category-specific intercepts.
  beta does NOT depend on k -- this is the "proportional odds" assumption.
  Interpretation: exp(beta_j) is the OR for moving up one category for a
  unit increase in x_j (the SAME effect at every cut).

We fit by direct MLE (BFGS through scipy.optimize) -- IRLS works too but the
multi-cutpoint constraints (alpha_k strictly increasing) make BFGS on a
reparameterized log-difference simpler.

Output: alpha (K-1 cutpoints), beta (p slopes), SEs, p-values, ORs.

The "shared slope" is restrictive. Test it via the Score Test for Proportional
Odds (Brant test). The partial-proportional-odds model (§7.3, not implemented)
relaxes it per predictor.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import optimize, stats    # optimize: BFGS / Brent solvers;  stats: distributions / tests


def _build_params(theta, K, p):
    """Reparameterize: theta = (alpha_1, delta_1, ..., delta_{K-2}, beta)
       where alpha_k = alpha_1 + sum_{j < k} exp(delta_j)  enforces ordering."""
    alpha = np.empty(K - 1)
    alpha[0] = theta[0]
    for k in range(1, K - 1):
        alpha[k] = alpha[k - 1] + math.exp(theta[k])
    beta = theta[K - 1:K - 1 + p]
    return alpha, beta


def _neg_log_lik(theta, X, y, K):
    p = X.shape[1]
    alpha, beta = _build_params(theta, K, p)
    eta_extra = -(X @ beta)
    # Cumulative probabilities P(Y <= k); add +inf at top and -inf below.
    n = X.shape[0]; ll = 0.0
    for i in range(n):
        cum_prev = 0.0
        for k in range(K):
            if k < K - 1:
                z = alpha[k] + eta_extra[i]
                cum = 1.0 / (1.0 + math.exp(-z))
            else:
                cum = 1.0
            pk = cum - cum_prev
            if y[i] == k + 1:
                ll += math.log(max(pk, 1e-15))
            cum_prev = cum
    return -ll


def fit(X, y) -> dict:
    """Proportional-odds ordinal logistic regression.

    Parameters
    ----------
    X : n x p design matrix WITHOUT intercept (the model has K-1 intercepts).
    y : ordinal response, integers in {1, 2, ..., K}.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
    K = int(np.max(y)); p = X.shape[1]
    # Initial: alpha_k as logit of cumulative proportions, beta = 0
    cum_props = np.array([np.mean(y <= k) for k in range(1, K)])
    cum_props = np.clip(cum_props, 1e-6, 1 - 1e-6)
    alpha0 = np.log(cum_props / (1 - cum_props))
    delta0 = np.log(np.diff(alpha0))                       # log spacings
    theta0 = np.concatenate([[alpha0[0]], delta0, np.zeros(p)])
    res = optimize.minimize(_neg_log_lik, theta0, args=(X, y, K),
                            method="BFGS", options={"gtol": 1e-7, "disp": False})
    theta = res.x
    alpha, beta = _build_params(theta, K, p)
    # Approximate SE from inverse Hessian (BFGS returns hess_inv)
    H_inv = res.hess_inv
    # Map the alpha-delta reparameterization back to direct alpha + beta SEs:
    # the part of H_inv corresponding to beta is in the last p positions.
    se_beta = np.sqrt(np.diag(H_inv)[-p:])
    z = beta / se_beta; p_vals = 2 * stats.norm.sf(np.abs(z))
    or_ = np.exp(beta); zc = stats.norm.ppf(0.975)
    or_lo = np.exp(beta - zc * se_beta); or_hi = np.exp(beta + zc * se_beta)
    return {"alpha": alpha.tolist(), "beta": beta.tolist(),
            "se_beta": se_beta.tolist(), "z_beta": z.tolist(),
            "p_values": p_vals.tolist(),
            "odds_ratio": or_.tolist(),
            "or_ci_lower": or_lo.tolist(), "or_ci_upper": or_hi.tolist(),
            "K": K, "log_likelihood": float(-res.fun),
            "converged": bool(res.success), "iterations": int(res.nit)}


def library_versions(X, y):
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        mod = OrderedModel(y, X, distr="logit"); res = mod.fit(method="bfgs", disp=False)
        return {"statsmodels OrderedModel params": res.params.tolist()}
    except Exception as exc:
        return {"statsmodels OrderedModel": f"error: {exc}"}


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n = 400
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    eta = 1.0 * x1 - 0.7 * x2
    # Latent: y* = eta + logistic noise; categorize by thresholds
    noise = -np.log(1 / rng.uniform(1e-9, 1 - 1e-9, n) - 1)   # logistic(0,1)
    y_star = eta + noise
    thresholds = [-1.0, 0.5, 2.0]
    y = np.digitize(y_star, thresholds) + 1                    # in {1, 2, 3, 4}
    X = np.column_stack([x1, x2])

    res = fit(X, y)
    print(f"K = {res['K']} categories,  log-likelihood = {res['log_likelihood']:.4f}")
    print(f"  cutpoints (alpha): {[round(a, 4) for a in res['alpha']]}")
    print("  slopes (beta):")
    for j, nm in enumerate(["x1", "x2"]):
        print(f"    {nm}: beta = {res['beta'][j]:+.4f}  SE = {res['se_beta'][j]:.4f}  "
              f"z = {res['z_beta'][j]:+.2f}  p = {res['p_values'][j]:.4g}  "
              f"OR = {res['odds_ratio'][j]:.3f} [{res['or_ci_lower'][j]:.3f}, "
              f"{res['or_ci_upper'][j]:.3f}]")
    print("\n--- library (statsmodels OrderedModel) ---")
    for k, v in library_versions(X, y).items():
        print(f"  {k}: {v}")
