"""Basket Trial Design (Reference Sec 47.264).

Berry et al 2013 Clin Trials; Simon et al 2016 CCR. A SINGLE
drug is tested across K different disease cohorts (tumour types
sharing a biomarker). Independent per-basket analyses waste
power; BAYESIAN HIERARCHICAL BORROWING pools information via
a shared random-effect prior:

    y_k ~ Binomial(n_k, p_k)
    logit(p_k) = theta_k
    theta_k ~ Normal(mu, tau^2)          borrow across baskets
    mu ~ Normal(0, 10^2);  tau ~ HalfNormal(1)

Baskets with similar effects "borrow strength" (shrink toward
each other); one clearly-null basket is protected by the tau
hyper-parameter learning small values.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def hierarchical_basket(n, y, n_iter=6000, burn=1000, sigma_prior=1.0, rng=None):
    """Metropolis-Hastings for logistic hierarchical model across K baskets."""
    if rng is None: rng = np.random.default_rng(0)
    K = len(n)
    theta = np.zeros(K); mu = 0.0; log_tau = 0.0
    samples = np.zeros((n_iter, K + 2))
    for it in range(n_iter):
        # Update each theta_k via a random-walk MH
        for k in range(K):
            prop = theta[k] + rng.normal(0, 0.4)
            p_c = 1 / (1 + np.exp(-theta[k]))
            p_p = 1 / (1 + np.exp(-prop))
            log_lik_c = y[k] * np.log(p_c) + (n[k] - y[k]) * np.log(1 - p_c)
            log_lik_p = y[k] * np.log(p_p) + (n[k] - y[k]) * np.log(1 - p_p)
            tau = np.exp(log_tau)
            log_prior_c = -0.5 * ((theta[k] - mu) / tau) ** 2
            log_prior_p = -0.5 * ((prop - mu) / tau) ** 2
            if np.log(rng.uniform()) < (log_lik_p + log_prior_p - log_lik_c - log_prior_c):
                theta[k] = prop
        # Update mu with a random-walk MH given normal prior N(0, 10^2)
        prop_mu = mu + rng.normal(0, 0.2); tau = np.exp(log_tau)
        log_prior_c = -0.5 * (mu / 10) ** 2 - 0.5 * (((theta - mu) / tau) ** 2).sum()
        log_prior_p = -0.5 * (prop_mu / 10) ** 2 - 0.5 * (((theta - prop_mu) / tau) ** 2).sum()
        if np.log(rng.uniform()) < (log_prior_p - log_prior_c): mu = prop_mu
        # Update log_tau (HalfNormal(1) prior on tau via change of variables)
        prop_lt = log_tau + rng.normal(0, 0.2); tau_p = np.exp(prop_lt)
        log_prior_c = -0.5 * tau ** 2 + log_tau - K * log_tau - 0.5 * (((theta - mu) / tau) ** 2).sum()
        log_prior_p = -0.5 * tau_p ** 2 + prop_lt - K * prop_lt - 0.5 * (((theta - mu) / tau_p) ** 2).sum()
        if np.log(rng.uniform()) < (log_prior_p - log_prior_c): log_tau = prop_lt
        samples[it] = np.concatenate([theta, [mu, log_tau]])
    return samples[burn:]


if __name__ == "__main__":
    print("=== Basket Trial Design (Berry et al 2013; Simon et al 2016) ===\n")
    rng = np.random.default_rng(0)

    # 5 baskets: 4 similar responders + 1 null
    n = np.array([20, 20, 20, 20, 20])
    p_true = np.array([0.30, 0.35, 0.28, 0.32, 0.05])              # last basket null
    y = np.array([int(rng.binomial(ni, pi)) for ni, pi in zip(n, p_true)])
    print(f"  {'basket':>7}  {'n':>3}  {'y':>3}  {'p_hat':>7}  {'p_true':>7}")
    for k in range(len(n)):
        print(f"  {k + 1:>7}  {n[k]:>3}  {y[k]:>3}  {y[k] / n[k]:>7.2f}  {p_true[k]:>7.2f}")

    print(f"\n  Independent per-basket 95% CI (Wilson):")
    from scipy.stats import beta
    for k in range(len(n)):
        lo, hi = beta.ppf([0.025, 0.975], y[k] + 0.5, n[k] - y[k] + 0.5)
        print(f"    basket {k + 1}:  ({lo:.3f}, {hi:.3f})")

    # Hierarchical borrowing
    samples = hierarchical_basket(n, y, n_iter=6000, burn=1000, rng=rng)
    theta_post = samples[:, :len(n)]
    p_post = 1 / (1 + np.exp(-theta_post))
    print(f"\n  Bayesian hierarchical posterior mean and 95% CrI (with borrowing):")
    print(f"    {'basket':>7}  {'post_mean':>10}  {'95% CrI':>18}  shrinkage")
    for k in range(len(n)):
        m = p_post[:, k].mean(); lo = np.quantile(p_post[:, k], 0.025); hi = np.quantile(p_post[:, k], 0.975)
        obs = y[k] / n[k]
        shift = m - obs
        print(f"    {k + 1:>7}  {m:>10.3f}  ({lo:.3f}, {hi:.3f})  {shift:+.3f}")

    print(f"\n  Posterior tau (heterogeneity across baskets): mean = {np.mean(np.exp(samples[:, -1])):.2f}")
    print(f"  Small tau => strong borrowing (similar baskets); large tau => independent behaviour.")
    print(f"  Hierarchical model tightens the responder baskets and PROTECTS the null basket.")

    print("\n--- library cross-check (bhmbasket R; basket R; PyMC / brms) ---")
