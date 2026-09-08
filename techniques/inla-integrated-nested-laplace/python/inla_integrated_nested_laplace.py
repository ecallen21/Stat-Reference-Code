"""INLA -- integrated nested Laplace approximation (Reference Sec 25.13).

Rue, Martino & Chopin 2009 'Approximate Bayesian inference for latent
Gaussian models by using integrated nested Laplace approximations',
JRSS-B. Fast deterministic alternative to MCMC for models with
GAUSSIAN LATENT FIELD:

    y | x, theta ~ likelihood(x, theta)   observations
    x | theta    ~ N(0, Q(theta)^{-1})    Gaussian latent field
    theta         ~ pi(theta)               hyperparameters

INLA:
    1. Laplace-approximate p(theta | y) via mode + Hessian.
    2. For each theta node on a grid, Laplace-approximate p(x_i | theta, y).
    3. Integrate over theta with weights from step 1:
       p(x_i | y) ~= sum_k w_k * p(x_i | theta_k, y).

Widely used in spatial epidemiology (BYM / BYM2), disease mapping,
smoothing splines, and geostatistics. Full-featured software is
r-inla; here we implement the smallest version -- Laplace
approximation for a simple Gaussian latent + fixed hyperparameter.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def laplace_gaussian_latent(y, prior_prec, likelihood_prec):
    """Simple version: y_i | x_i ~ N(x_i, 1/lik_prec), x ~ N(0, 1/prior_prec).

    Posterior x | y is Gaussian: mean = (Q + P)^-1 * P * y, var = (Q + P)^-1.
    """
    n = len(y)
    Q = prior_prec * np.eye(n)
    P = likelihood_prec * np.eye(n)
    post_prec = Q + P
    post_mean = np.linalg.solve(post_prec, P @ y)
    return post_mean, np.linalg.inv(post_prec)


def inla_1d(y, log_lik, prior_log_prec_range=(-3, 3), n_grid=15):
    """Grid-based INLA-style integration over a scalar precision hyperparameter."""
    log_precs = np.linspace(*prior_log_prec_range, n_grid)
    log_marginals = np.zeros(n_grid)
    means = []
    for k, log_p in enumerate(log_precs):
        prec = np.exp(log_p)
        post_mean, post_cov = laplace_gaussian_latent(y, prec, likelihood_prec=1.0)
        #  Marginal log likelihood ~ log p(y | prec)
        log_marginals[k] = log_lik(y, prec)
        means.append(post_mean)
    #  Normalise marginal
    w = np.exp(log_marginals - log_marginals.max())
    w /= w.sum()
    #  INLA posterior of x_i = weighted mixture
    inla_mean = np.sum([w[k] * means[k] for k in range(n_grid)], axis=0)
    return {"weights": w, "prec_grid": np.exp(log_precs),
            "posterior_mean": inla_mean}


def log_marg_gaussian(y, prec, likelihood_prec=1.0):
    """log p(y) for y_i | x_i ~ N(x_i, 1), x ~ N(0, 1/prec) => y ~ N(0, 1 + 1/prec)."""
    from scipy.stats import norm
    sd = np.sqrt(1 / prec + 1 / likelihood_prec)
    return float(np.sum(norm.logpdf(y, loc=0, scale=sd)))


if __name__ == "__main__":
    print("=== INLA-style Laplace + hyperparameter grid ===\n")
    rng = np.random.default_rng(0)
    n = 50
    #  x true ~ N(0, 1) then y = x + N(0, 1)
    x_true = rng.normal(size=n)
    y = x_true + rng.normal(size=n)

    r = inla_1d(y, log_marg_gaussian)
    top_idx = np.argsort(r["weights"])[::-1][:3]
    print(f"  Top-3 posterior weights on precision grid:")
    for i in top_idx:
        print(f"    prec = {r['prec_grid'][i]:.3f}   weight = {r['weights'][i]:.3f}")
    #  Compare posterior mean to Bayes-optimal (with true prec = 1)
    post_mean_true = np.linalg.solve(np.eye(n) + np.eye(n), np.eye(n) @ y)
    err = float(np.mean((r["posterior_mean"] - x_true) ** 2))
    err_true = float(np.mean((post_mean_true - x_true) ** 2))
    print(f"\n  MSE(INLA posterior mean vs x_true)     = {err:.3f}")
    print(f"  MSE(Bayes posterior mean w/ true prec)  = {err_true:.3f}")

    print("\n--- library cross-check (INLA R (r-inla.org); pymc / numpyro Python) ---")
