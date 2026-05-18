"""Probit regression (Reference §7.10).

Binary GLM with the PROBIT link:

    P(y = 1 | x) = Phi(x' beta),   Phi = standard normal CDF

Fit by MLE / IRLS. Compared to logistic regression:

  - The latent-variable story is "y = 1 iff y_star = x' beta + eps > 0" with
    eps ~ N(0, 1). Logistic uses eps ~ Logistic(0, 1) instead.
  - Coefficient magnitudes are smaller (logit-betas are roughly 1.6x probit-betas
    because the logistic has heavier tails than the standard normal).
  - Marginal effects are easier to compare to OLS because the latent y_star is on a
    normal scale.
  - There's no clean "odds ratio" interpretation -- coefficients are in
    standard-deviation units of the latent variable. Report average marginal
    effects (see ``techniques/marginal-effects``).

We implement IRLS with the same working-response structure as logistic, but
weights and the working response use the normal PDF instead of pi(1-pi).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def fit(X, y, max_iter: int = 50, tol: float = 1e-8) -> dict:
    """Probit GLM by Fisher scoring / IRLS."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta = np.zeros(p); ll_prev = -np.inf; converged = False; it = 0
    for it in range(1, max_iter + 1):
        eta = X @ beta
        pi = stats.norm.cdf(eta)
        phi = stats.norm.pdf(eta)
        # IRLS working response and weights for canonical link form:
        pi_safe = np.clip(pi, 1e-12, 1 - 1e-12)
        # derivative of mu wrt eta = phi; weight = phi^2 / (pi (1-pi))
        w = (phi ** 2) / (pi_safe * (1 - pi_safe))
        w = np.clip(w, 1e-12, None)
        z = eta + (y - pi) / phi
        sw = np.sqrt(w)
        Xw = X * sw[:, None]; zw = z * sw
        beta_new, *_ = np.linalg.lstsq(Xw, zw, rcond=None)
        ll = float(np.sum(y * np.log(pi_safe) + (1 - y) * np.log(1 - pi_safe)))
        if abs(ll - ll_prev) < tol:
            converged = True
            break
        ll_prev = ll; beta = beta_new
    eta = X @ beta; pi = np.clip(stats.norm.cdf(eta), 1e-12, 1 - 1e-12)
    phi = stats.norm.pdf(eta)
    w = (phi ** 2) / (pi * (1 - pi))
    var_beta = np.linalg.pinv(X.T @ (X * w[:, None]))
    se = np.sqrt(np.diag(var_beta))
    z_stats = beta / se; p_vals = 2 * stats.norm.sf(np.abs(z_stats))
    ll_full = float(np.sum(y * np.log(pi) + (1 - y) * np.log(1 - pi)))
    p_bar = float(np.mean(y))
    ll_null = float(n * (p_bar * math.log(max(p_bar, 1e-15))
                          + (1 - p_bar) * math.log(max(1 - p_bar, 1e-15))))
    # Average marginal effect of x_j: mean over i of phi(x_i' beta) * beta_j
    ame = phi.mean() * beta
    return {"beta": beta.tolist(), "se": se.tolist(),
            "z": z_stats.tolist(), "p_values": p_vals.tolist(),
            "average_marginal_effect": ame.tolist(),
            "log_likelihood": ll_full, "ll_null": ll_null,
            "lr_chi_square": 2 * (ll_full - ll_null),
            "iterations": it, "converged": converged}


def library_versions(X_no_const, y):
    import statsmodels.api as sm
    res = sm.Probit(y, sm.add_constant(X_no_const)).fit(disp=False)
    return {"statsmodels Probit params": res.params.tolist(),
            "statsmodels Probit bse": res.bse.tolist(),
            "statsmodels Probit llf": float(res.llf)}


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n = 300
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    eta = -0.3 + 0.8 * x1 - 0.5 * x2
    y = (rng.uniform(0, 1, n) < stats.norm.cdf(eta)).astype(int)
    X = np.column_stack([np.ones(n), x1, x2])

    res = fit(X, y)
    print(f"=== Probit fit ({res['iterations']} iters, converged={res['converged']}) ===")
    for j, nm in enumerate(["intercept", "x1", "x2"]):
        print(f"  {nm:10s} beta = {res['beta'][j]:+.4f}  SE = {res['se'][j]:.4f}  "
              f"z = {res['z'][j]:+.2f}  p = {res['p_values'][j]:.4g}  "
              f"AME = {res['average_marginal_effect'][j]:+.4f}")
    print(f"  log-likelihood = {res['log_likelihood']:.4f}")
    print("\n--- library (statsmodels.Probit) ---")
    for k, v in library_versions(np.column_stack([x1, x2]), y).items():
        print(f"  {k}: {v}")
