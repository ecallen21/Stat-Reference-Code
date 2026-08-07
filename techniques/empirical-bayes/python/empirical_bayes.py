"""Empirical Bayes (Reference §14.17, §14.18).

'EB' plugs a POINT ESTIMATE of the hyperprior into an otherwise Bayesian
analysis, instead of putting a full hyperprior on the hyperparameters
(which is fully-Bayes / hierarchical).  Cheap, fast, and often gives
essentially the same shrinkage as a full hierarchical fit.

Two canonical examples:

1) Beta-Binomial EB
    Each group j has y_j / n_j observed success rate.
    Model: theta_j ~ Beta(alpha, beta),  y_j | theta_j ~ Binomial(n_j, theta_j).
    Estimate (alpha, beta) by method-of-moments on the observed rates:
        mean p_bar   = alpha / (alpha + beta)
        variance v   = alpha beta / ((alpha + beta)^2 (alpha + beta + 1))
    Plug (alpha, beta) into the posterior of each theta_j:
        theta_j | y ~ Beta(alpha + y_j, beta + n_j - y_j)

2) James-Stein estimator
    For J independent estimates y_j ~ Normal(theta_j, sigma^2/n_j) with common
    (unknown) mean, the JS shrinkage estimator dominates the raw y_j when J >= 3:
        theta_hat_j = ybar + (1 - (J - 3) sigma^2 / sum((y_j - ybar)^2)) (y_j - ybar)
    Equivalent to plug-in EB with a Normal(mu, tau^2) hyperprior on the theta_j.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def eb_beta_binomial(successes, trials) -> dict:
    """Empirical Bayes for a set of binomial rates.

    Estimate (alpha, beta) by method-of-moments, then return EB posterior
    mean and 95% credible interval for each group.
    """
    y = np.asarray(successes, dtype=float); n = np.asarray(trials, dtype=float)
    p = y / n; J = len(y)
    p_bar = float(p.mean())
    v = float(p.var(ddof=1))
    # Method of moments on Beta hyperprior
    denom = max(v - p_bar * (1 - p_bar) / n.mean(), 1e-8)
    ab_sum = p_bar * (1 - p_bar) / denom - 1
    alpha_hat = max(ab_sum * p_bar, 0.5)
    beta_hat = max(ab_sum * (1 - p_bar), 0.5)
    # EB posterior for each group
    a_post = alpha_hat + y; b_post = beta_hat + n - y
    theta_eb = a_post / (a_post + b_post)
    from scipy.stats import beta as beta_dist
    ci = np.column_stack([beta_dist.ppf(0.025, a_post, b_post),
                          beta_dist.ppf(0.975, a_post, b_post)])
    return {"alpha_hat": float(alpha_hat), "beta_hat": float(beta_hat),
            "hyperprior_mean": float(alpha_hat / (alpha_hat + beta_hat)),
            "hyperprior_pseudo_n": float(alpha_hat + beta_hat),
            "raw_rate": p,
            "eb_posterior_mean": theta_eb,
            "eb_credible_95": ci,
            "n_groups": int(J),
            "method": "Empirical Bayes Beta-Binomial (method of moments)"}


def james_stein(y, sigma) -> dict:
    """James-Stein shrinkage estimator for J >= 3 independent Normal estimates.

    y      : length-J vector of independent estimates
    sigma  : known SE of each y_j (scalar or length-J vector)
    """
    y = np.asarray(y, dtype=float); J = len(y)
    if J < 3: raise ValueError("James-Stein requires J >= 3 groups")
    ybar = float(y.mean())
    if np.ndim(sigma) == 0:
        var = float(sigma) ** 2
        shrinkage = 1 - (J - 3) * var / np.sum((y - ybar) ** 2)
        js = ybar + shrinkage * (y - ybar)
    else:
        # Weighted / heteroscedastic variant (Stein 1962; Efron-Morris)
        w = 1 / np.asarray(sigma, dtype=float) ** 2
        shrinkage = 1 - (J - 3) / np.sum(w * (y - ybar) ** 2)
        js = ybar + max(shrinkage, 0) * (y - ybar)
        return {"grand_mean": ybar, "shrinkage_factor": float(shrinkage),
                "raw": y, "js_estimate": js,
                "method": "James-Stein (heteroscedastic Efron-Morris variant)"}
    return {"grand_mean": ybar, "shrinkage_factor": float(shrinkage),
            "raw": y, "js_estimate": js,
            "method": "James-Stein estimator"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== EB Beta-Binomial on 20 batting-average-style rates ===")
    true_theta = rng.beta(30, 70, 20)  # true rate ~0.3 with spread
    n_trials = rng.integers(20, 200, 20)
    successes = rng.binomial(n_trials, true_theta)
    r = eb_beta_binomial(successes, n_trials)
    print(f"  hyperprior alpha = {r['alpha_hat']:.2f}, beta = {r['beta_hat']:.2f}")
    print(f"  hyperprior mean rate = {r['hyperprior_mean']:.3f}")
    print("  group  n  y   raw     EB      truth")
    for j in range(20):
        print(f"  {j:3d}   {n_trials[j]:3d} {successes[j]:3d} {r['raw_rate'][j]:.3f}   {r['eb_posterior_mean'][j]:.3f}   {true_theta[j]:.3f}")
    # Compare accuracy
    print(f"\n  RMSE(raw   vs truth) = {np.sqrt(((r['raw_rate'] - true_theta) ** 2).mean()):.4f}")
    print(f"  RMSE(EB    vs truth) = {np.sqrt(((r['eb_posterior_mean'] - true_theta) ** 2).mean()):.4f}")

    print("\n=== James-Stein on J=10 independent Normal estimates ===")
    true_theta = np.array([-2, -1, -0.5, 0, 0, 0.5, 1, 1.5, 2, 3])
    y = true_theta + rng.normal(0, 1, len(true_theta))
    r = james_stein(y, sigma=1.0)
    print(f"  raw:  {np.round(y, 2)}")
    print(f"  JS:   {np.round(r['js_estimate'], 2)}")
    print(f"  true: {true_theta}")
    print(f"  shrinkage factor = {r['shrinkage_factor']:.4f}")
    print(f"  MSE(raw vs truth) = {np.mean((y - true_theta) ** 2):.4f}")
    print(f"  MSE(JS  vs truth) = {np.mean((r['js_estimate'] - true_theta) ** 2):.4f}")
