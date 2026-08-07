"""Bayesian hierarchical models (Reference §14.15, §14.16).

Groups share statistical strength through a common prior over group-level
parameters -- classic PARTIAL POOLING.  The canonical example is Rubin's
'8 schools' (Rubin 1981; Gelman BDA ch 5):

    y_j     ~ Normal(theta_j, sigma_j^2)     j = 1, ..., J   (sigma_j known)
    theta_j ~ Normal(mu, tau^2)
    mu      ~ Normal(0, 100^2)               (diffuse hyperprior on mean)
    tau     ~ HalfCauchy(5)  or InvGamma(0.5, 0.5)  (weakly informative)

Compared to fully-pooled (theta_j = mu) or unpooled (theta_j = y_j) fits,
the hierarchical estimator shrinks each theta_j toward mu by an amount that
depends on the relative sizes of sigma_j and tau -- shrinkage is stronger
for imprecise groups and weaker for precise ones.

Extends immediately to random-slopes / random-intercept regression, meta-
analysis, and multilevel logistic / Poisson models (see 'bayesian-glms').
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def hierarchical_normal_gibbs(y, sigma, mu_prior_var: float = 1e4,
                              tau2_prior_alpha: float = 0.5,
                              tau2_prior_beta: float = 0.5,
                              n_iter: int = 8000, seed: int = 0) -> dict:
    """Full Gibbs sampler for the hierarchical Normal model."""
    y = np.asarray(y, dtype=float); sigma = np.asarray(sigma, dtype=float)
    J = len(y); rng = np.random.default_rng(seed)
    mu = float(y.mean()); tau2 = float(y.var(ddof=1))
    theta = y.copy()
    thetas = np.empty((n_iter, J)); mus = np.empty(n_iter); tau2s = np.empty(n_iter)
    for t in range(n_iter):
        # theta_j | mu, tau, y
        v_j = 1 / (1 / sigma ** 2 + 1 / tau2)
        m_j = v_j * (y / sigma ** 2 + mu / tau2)
        theta = rng.normal(m_j, np.sqrt(v_j))
        # mu | theta, tau
        prec = J / tau2 + 1 / mu_prior_var
        mu = rng.normal((theta.sum() / tau2) / prec, math.sqrt(1 / prec))
        # tau^2 | theta, mu
        a = tau2_prior_alpha + J / 2
        b = tau2_prior_beta + 0.5 * np.sum((theta - mu) ** 2)
        tau2 = 1 / rng.gamma(a, 1 / b)
        thetas[t] = theta; mus[t] = mu; tau2s[t] = tau2
    burn = n_iter // 5
    thetas = thetas[burn:]; mus = mus[burn:]; tau2s = tau2s[burn:]
    return {"theta_post_mean": thetas.mean(0),
            "theta_post_sd": thetas.std(0),
            "theta_95": np.quantile(thetas, [0.025, 0.975], axis=0).T,
            "mu_mean": float(mus.mean()),
            "mu_95": tuple(np.quantile(mus, [0.025, 0.975])),
            "tau_mean": float(np.sqrt(tau2s).mean()),
            "tau_95": tuple(np.quantile(np.sqrt(tau2s), [0.025, 0.975])),
            "shrinkage_by_group": 1 - v_j / sigma ** 2 if False else None,
            "n_groups": int(J), "n_iter_kept": int(len(mus)),
            "method": "Hierarchical Normal (partial pooling) via Gibbs"}


def three_way_comparison(y, sigma, mu_prior_var: float = 1e4,
                         n_iter: int = 8000, seed: int = 0) -> dict:
    """Compare unpooled, fully-pooled, and hierarchical (partial-pooled) fits."""
    y = np.asarray(y, dtype=float); sigma = np.asarray(sigma, dtype=float)
    # Fully pooled: single mean weighted by precision
    prec = 1 / sigma ** 2
    theta_pool = (prec * y).sum() / prec.sum()
    theta_pool = np.full_like(y, theta_pool)
    # Unpooled
    theta_unpool = y.copy()
    # Hierarchical
    r = hierarchical_normal_gibbs(y, sigma, mu_prior_var=mu_prior_var,
                                   n_iter=n_iter, seed=seed)
    return {"y_observed": y, "sigma_j": sigma,
            "theta_unpooled": theta_unpool,
            "theta_fully_pooled": theta_pool,
            "theta_hierarchical": r["theta_post_mean"],
            "mu_posterior_mean": r["mu_mean"],
            "tau_posterior_mean": r["tau_mean"]}


if __name__ == "__main__":
    # Rubin 8-schools data
    y = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=float)
    sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=float)

    print("=== 8-schools hierarchical fit ===")
    r = hierarchical_normal_gibbs(y, sigma, n_iter=10000, seed=0)
    print(f"  overall mu:  {r['mu_mean']:.3f}, 95% CrI = ({r['mu_95'][0]:.3f}, {r['mu_95'][1]:.3f})")
    print(f"  between-school tau: {r['tau_mean']:.3f}, 95% CrI = ({r['tau_95'][0]:.3f}, {r['tau_95'][1]:.3f})")
    print(f"  posterior theta:")
    for j in range(8):
        print(f"    school {j+1}: y = {y[j]:5.1f} +/- {sigma[j]:.0f},  theta = {r['theta_post_mean'][j]:5.2f} +/- {r['theta_post_sd'][j]:.2f}")

    print("\n=== Three-way comparison (unpooled vs pooled vs partial) ===")
    cmp = three_way_comparison(y, sigma)
    print(f"  y:                {cmp['y_observed']}")
    print(f"  unpooled:         {cmp['theta_unpooled']}")
    print(f"  fully pooled:     {np.round(cmp['theta_fully_pooled'], 2)}")
    print(f"  hierarchical:     {np.round(cmp['theta_hierarchical'], 2)}")

    print("\n--- library cross-check (pymc, if available) ---")
    try:
        import pymc as pm
        with pm.Model():
            mu = pm.Normal("mu", 0, 100)
            tau = pm.HalfCauchy("tau", 5)
            theta = pm.Normal("theta", mu=mu, sigma=tau, shape=len(y))
            pm.Normal("obs", mu=theta, sigma=sigma, observed=y)
            trace = pm.sample(2000, tune=1000, cores=1, progressbar=False, random_seed=0)
        print(f"  pymc theta posterior mean: {trace.posterior['theta'].mean(dim=('chain','draw')).values.round(2)}")
    except Exception as ex:
        print(f"  (pymc not available: {ex})")
