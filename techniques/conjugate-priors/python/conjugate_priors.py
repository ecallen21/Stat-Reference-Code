"""Conjugate priors (Reference §14.1, §14.2, §14.3).

A prior is CONJUGATE to a likelihood if the posterior is in the same family
as the prior.  Closed-form posteriors are analytically convenient and give
the correct Bayesian answer without MCMC.

Three canonical pairs:

    Beta - Binomial
        theta   ~ Beta(alpha, beta)
        y | theta ~ Binomial(n, theta)
        theta | y ~ Beta(alpha + y, beta + n - y)
        posterior predictive: Beta-Binomial

    Gamma - Poisson
        lambda  ~ Gamma(alpha, rate = beta)
        y | lambda ~ Poisson(lambda)
        lambda | y ~ Gamma(alpha + sum y, beta + n)
        posterior predictive: Negative-Binomial

    Normal (known sigma^2) - Normal
        mu      ~ Normal(mu_0, tau_0^2)
        y | mu  ~ Normal(mu, sigma^2)
        mu | y  ~ Normal(mu_n, tau_n^2)
            precision update: 1/tau_n^2 = 1/tau_0^2 + n / sigma^2
            mean:            mu_n = tau_n^2 (mu_0 / tau_0^2 + n y_bar / sigma^2)
        posterior predictive for a new observation: Normal(mu_n, tau_n^2 + sigma^2)

Weakly-informative default priors (Gelman): Beta(1, 1) uniform;
Gamma(alpha ~ 0.5, beta ~ 0.5) weakly informative; Normal(0, 100^2) diffuse.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def beta_binomial_update(alpha_prior: float, beta_prior: float, successes: int, trials: int) -> dict:
    """Beta prior x Binomial likelihood -> Beta posterior."""
    a_post = alpha_prior + successes
    b_post = beta_prior + trials - successes
    mean = a_post / (a_post + b_post)
    var = (a_post * b_post) / ((a_post + b_post) ** 2 * (a_post + b_post + 1))
    lo, hi = stats.beta.ppf([0.025, 0.975], a_post, b_post)
    return {"prior": (alpha_prior, beta_prior),
            "posterior_alpha": float(a_post), "posterior_beta": float(b_post),
            "posterior_mean": float(mean), "posterior_var": float(var),
            "credible_95": (float(lo), float(hi)),
            "method": "Beta-Binomial conjugate update"}


def gamma_poisson_update(alpha_prior: float, rate_prior: float, counts) -> dict:
    """Gamma(shape=alpha, rate=beta) prior x Poisson likelihood -> Gamma posterior."""
    y = np.asarray(counts, dtype=float)
    n = len(y); s = float(y.sum())
    a_post = alpha_prior + s
    b_post = rate_prior + n
    mean = a_post / b_post
    var = a_post / (b_post ** 2)
    lo, hi = stats.gamma.ppf([0.025, 0.975], a_post, scale=1 / b_post)
    return {"prior": (alpha_prior, rate_prior),
            "posterior_shape": float(a_post), "posterior_rate": float(b_post),
            "posterior_mean": float(mean), "posterior_var": float(var),
            "credible_95": (float(lo), float(hi)),
            "method": "Gamma-Poisson conjugate update"}


def normal_normal_update(mu_prior: float, tau2_prior: float, sigma2_known: float, y) -> dict:
    """Normal prior on mu x Normal likelihood (known sigma^2) -> Normal posterior."""
    y = np.asarray(y, dtype=float); n = len(y); ybar = float(y.mean())
    precision_post = 1 / tau2_prior + n / sigma2_known
    tau2_post = 1 / precision_post
    mu_post = tau2_post * (mu_prior / tau2_prior + n * ybar / sigma2_known)
    lo, hi = stats.norm.ppf([0.025, 0.975], loc=mu_post, scale=math.sqrt(tau2_post))
    # Posterior predictive for a NEW obs: mean mu_post, var tau2_post + sigma^2
    pred_var = tau2_post + sigma2_known
    lo_p, hi_p = stats.norm.ppf([0.025, 0.975], loc=mu_post, scale=math.sqrt(pred_var))
    return {"prior_mean": mu_prior, "prior_var": tau2_prior, "known_sigma2": sigma2_known,
            "posterior_mean": float(mu_post), "posterior_var": float(tau2_post),
            "credible_95_mu": (float(lo), float(hi)),
            "pred_mean_new_obs": float(mu_post), "pred_var_new_obs": float(pred_var),
            "predictive_95_new_obs": (float(lo_p), float(hi_p)),
            "method": "Normal-Normal (known variance) conjugate update"}


def _print(d):
    for k, v in d.items():
        if isinstance(v, tuple): print(f"  {k}: ({v[0]:.4f}, {v[1]:.4f})")
        elif isinstance(v, float): print(f"  {k}: {v:.4f}")
        else: print(f"  {k}: {v}")


if __name__ == "__main__":
    print("=== Beta-Binomial: 8/12 successes with Uniform Beta(1,1) prior ===")
    _print(beta_binomial_update(1, 1, successes=8, trials=12))

    print("\n=== Gamma-Poisson: 5 draws with rate-1 exponential-ish prior ===")
    rng = np.random.default_rng(0)
    y_pois = rng.poisson(3.5, 5)
    print(f"  data: {y_pois.tolist()}")
    _print(gamma_poisson_update(alpha_prior=1.0, rate_prior=1.0, counts=y_pois))

    print("\n=== Normal-Normal (known sigma^2 = 4), n = 20, ybar ~ 5 ===")
    y = rng.normal(5, 2, 20)
    _print(normal_normal_update(mu_prior=0.0, tau2_prior=100.0, sigma2_known=4.0, y=y))

    print("\n--- library cross-check (scipy stats) ---")
    # Beta-Binomial posterior mean matches scipy Beta.mean of (a+y, b+n-y)
    ab = (1 + 8, 1 + 12 - 8)
    print(f"  scipy Beta({ab[0]}, {ab[1]}).mean() = {stats.beta(*ab).mean():.4f}")
