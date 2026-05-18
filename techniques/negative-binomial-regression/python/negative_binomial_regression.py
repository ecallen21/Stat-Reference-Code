"""Negative binomial regression (Reference §7.13).

For count outcomes with OVERDISPERSION (Var(Y) > E[Y]):

    Y_i ~ NB(mu_i, theta),   log(mu_i) = x' beta + offset
    Var(Y_i) = mu_i + mu_i^2 / theta

theta > 0 is the dispersion; as theta -> infinity, NB collapses to Poisson.

We do joint MLE on (beta, log_theta) via BFGS for numerical stability -- the
alternating IRLS scheme is sensitive to the initial theta near the boundary.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import optimize, special, stats    # optimize: solvers; special: gammaln/beta; stats: distributions/tests


def _neg_log_lik(params, X, y, offset):
    """params = (beta_1, ..., beta_p, log_theta)"""
    p = X.shape[1]
    beta = params[:p]; theta = math.exp(params[p])
    eta = X @ beta + offset
    mu = np.exp(np.clip(eta, -500, 500))
    return -float(np.sum(
        special.gammaln(y + theta) - special.gammaln(theta) - special.gammaln(y + 1)
        + theta * np.log(theta / (theta + mu))
        + y * np.log(mu / (theta + mu))
    ))


def fit(X, y, offset=None) -> dict:
    """NB MLE via BFGS on (beta, log_theta)."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    if offset is None:
        offset = np.zeros(X.shape[0])
    offset = np.asarray(offset, dtype=float)
    n, p = X.shape
    # Init beta from Poisson fit, theta = 1 (log_theta = 0)
    eta0 = np.log(np.maximum(y, 0.5)) - offset
    beta0, *_ = np.linalg.lstsq(X, eta0, rcond=None)
    init = np.concatenate([beta0, [0.0]])
    res = optimize.minimize(_neg_log_lik, init, args=(X, y, offset),
                            method="BFGS", options={"gtol": 1e-7, "disp": False})
    beta = res.x[:p]; theta = math.exp(res.x[p])
    se_all = np.sqrt(np.diag(res.hess_inv))
    se_beta = se_all[:p]
    z_stats = beta / se_beta
    p_vals = 2 * stats.norm.sf(np.abs(z_stats))
    irr = np.exp(beta); zc = stats.norm.ppf(0.975)
    return {"beta": beta.tolist(), "se": se_beta.tolist(),
            "z": z_stats.tolist(), "p_values": p_vals.tolist(),
            "irr": irr.tolist(),
            "irr_ci_lower": np.exp(beta - zc * se_beta).tolist(),
            "irr_ci_upper": np.exp(beta + zc * se_beta).tolist(),
            "theta": theta, "alpha": 1.0 / theta,
            "log_likelihood": float(-res.fun),
            "iterations": int(res.nit), "converged": bool(res.success)}


def library_versions(X_no_const, y):
    try:
        import statsmodels.api as sm
        X = sm.add_constant(X_no_const)
        res = sm.NegativeBinomial(y, X).fit(disp=False)
        return {"statsmodels NB params (beta + alpha at end)": res.params.tolist(),
                "statsmodels NB bse": res.bse.tolist(),
                "statsmodels NB llf": float(res.llf)}
    except Exception as exc:
        return {"statsmodels": f"error: {exc}"}


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    n = 400
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    mu_true = np.exp(0.5 + 0.6 * x1 - 0.4 * x2)
    theta_true = 2.0
    gamma_draws = rng.gamma(theta_true, mu_true / theta_true)
    y = rng.poisson(gamma_draws)
    X = np.column_stack([np.ones(n), x1, x2])

    res = fit(X, y)
    print(f"=== Negative Binomial ({res['iterations']} iters, converged={res['converged']}) ===")
    for j, nm in enumerate(["intercept", "x1", "x2"]):
        print(f"  {nm:10s} beta = {res['beta'][j]:+.4f}  SE = {res['se'][j]:.4f}  "
              f"IRR = {res['irr'][j]:.4f} [{res['irr_ci_lower'][j]:.3f}, "
              f"{res['irr_ci_upper'][j]:.3f}]  p = {res['p_values'][j]:.4g}")
    print(f"  theta (dispersion) = {res['theta']:.4f}   alpha = 1/theta = {res['alpha']:.4f}")
    print(f"  log-likelihood     = {res['log_likelihood']:.4f}")
    print(f"\n  (true theta = {theta_true})")
    print("\n--- library (statsmodels.NegativeBinomial) ---")
    for k, v in library_versions(np.column_stack([x1, x2]), y).items():
        print(f"  {k}: {v}")
