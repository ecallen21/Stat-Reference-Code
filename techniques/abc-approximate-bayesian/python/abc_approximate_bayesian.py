"""Approximate Bayesian Computation (Reference §14.27).

'Likelihood-free' Bayesian inference.  Applicable when the likelihood
p(y | theta) is impossible or expensive to evaluate, but SIMULATION from
the model p(y | theta) is easy (population genetics, epidemiology,
stochastic simulators).

Rejection ABC (Pritchard et al. 1999):
    for i = 1 .. N:
        theta_i ~ prior
        y_sim   = simulator(theta_i)
        if dist(summary(y_sim), summary(y_obs)) <= epsilon:
            keep theta_i as an approximate posterior draw

Small epsilon -> tight approximation but many rejections.  Standard practice:
generate many candidates and KEEP the closest fraction q (e.g. q = 0.001).

Regression adjustment (Beaumont-Zhang-Balding 2002)
    Locally regress theta on the summary distance and back-project to
    the observed summary.  Removes finite-epsilon bias.

Extensions: ABC-MCMC, ABC-SMC (sequential Monte Carlo), automatic
summary-statistic selection via machine learning.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def abc_rejection(prior_sampler, simulator, summary_fn, y_obs,
                   N: int = 20000, quantile: float = 0.01, seed: int = 0) -> dict:
    """Rejection ABC: keep the `quantile` fraction of theta with the closest simulated summary."""
    rng = np.random.default_rng(seed)
    s_obs = np.atleast_1d(summary_fn(y_obs))
    thetas = []; dists = []; sums = []
    for _ in range(N):
        theta = prior_sampler(rng)
        y_sim = simulator(theta, rng)
        s_sim = np.atleast_1d(summary_fn(y_sim))
        thetas.append(np.atleast_1d(theta))
        sums.append(s_sim)
        dists.append(float(np.sqrt(np.sum((s_sim - s_obs) ** 2))))
    thetas = np.array(thetas); dists = np.array(dists); sums = np.array(sums)
    n_keep = max(int(N * quantile), 1)
    idx = np.argsort(dists)[:n_keep]
    kept = thetas[idx]
    return {"posterior_draws": kept,
            "distances_kept": dists[idx],
            "sums_kept": sums[idx],
            "s_obs": s_obs,
            "epsilon_effective": float(dists[idx].max()),
            "N_total": int(N), "n_kept": int(n_keep),
            "method": "ABC rejection"}


def abc_regression_adjust(res) -> np.ndarray:
    """Beaumont-Zhang-Balding local-linear regression adjustment."""
    kept = res["posterior_draws"]
    sums = res["sums_kept"]
    s_obs = res["s_obs"]
    # theta ~ intercept + (s - s_obs) beta  weighted by Epanechnikov kernel
    d = np.linalg.norm(sums - s_obs, axis=1)
    eps = res["epsilon_effective"] or 1.0
    w = np.maximum(1 - (d / eps) ** 2, 0)
    X = np.column_stack([np.ones(len(kept)), sums - s_obs])
    adjusted = np.empty_like(kept)
    for k in range(kept.shape[1]):
        beta = np.linalg.pinv(X.T @ (X * w[:, None])) @ (X.T @ (w * kept[:, k]))
        # adjust to s = s_obs
        adjusted[:, k] = kept[:, k] - (sums - s_obs) @ beta[1:]
    return adjusted


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Simulator: n=50 obs from Normal(mu, sigma=1); true mu = 3
    n_obs = 50; true_mu = 3.0
    y_obs = rng.normal(true_mu, 1.0, n_obs)

    def prior_sampler(rng_): return rng_.uniform(-5, 10)
    def simulator(theta, rng_): return rng_.normal(theta, 1.0, n_obs)
    def summary_fn(y): return np.array([np.mean(y), np.std(y, ddof=1)])

    print("=== ABC rejection (uniform prior on mu, 20000 candidates, keep 1%) ===")
    r = abc_rejection(prior_sampler, simulator, summary_fn, y_obs,
                       N=20000, quantile=0.01, seed=0)
    print(f"  posterior mean of mu:  {r['posterior_draws'].mean():.3f}   (true {true_mu})")
    print(f"  posterior 95% CI:      ({np.quantile(r['posterior_draws'], 0.025):.3f},"
          f" {np.quantile(r['posterior_draws'], 0.975):.3f})")
    print(f"  epsilon (max kept dist): {r['epsilon_effective']:.4f}")

    print("\n=== Regression-adjusted posterior ===")
    adj = abc_regression_adjust(r)
    print(f"  adjusted mean mu:      {adj.mean():.3f}")
    print(f"  adjusted 95% CI:       ({np.quantile(adj, 0.025):.3f},"
          f" {np.quantile(adj, 0.975):.3f})")

    print("\n=== Analytic Normal-Normal posterior (uniform prior) ===")
    print(f"  posterior mean = ybar = {y_obs.mean():.3f}")
    print(f"  posterior 95% CI = ybar +/- 1.96 / sqrt(n) = "
          f"({y_obs.mean() - 1.96 / math.sqrt(n_obs):.3f},"
          f" {y_obs.mean() + 1.96 / math.sqrt(n_obs):.3f})")
