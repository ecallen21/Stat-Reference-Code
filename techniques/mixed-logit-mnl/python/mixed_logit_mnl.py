"""Mixed logit (random-coefficient MNL) (Reference Sec 47.54).

McFadden & Train 2000 'Mixed MNL models for discrete response',
J Appl Econ 15(5). Extends multinomial logit by letting each
individual n have RANDOM taste coefficients beta_n ~ f(beta | theta),
integrated out:

    P_n(i) = integral  exp(x_ni' beta) / sum_j exp(x_nj' beta)   f(beta | theta) d beta

Approximated by simulated log-likelihood (Halton draws or QMC):

    P_n^S(i) = (1/R) sum_r  exp(x_ni' beta_n^r) / sum_j exp(x_nj' beta_n^r)

Removes IIA (independence of irrelevant alternatives), allows
taste heterogeneity, correlated errors across alternatives.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize    # simulated MLE
from scipy.stats import qmc    # Halton draws


def _sim_choice_probs(X, mu, log_sigma, halton):
    """Return simulated P(i chosen) for a rank-3 array X of shape (n, J, K)."""
    n, J, K = X.shape
    sigma = np.exp(log_sigma)
    # Halton -> normal draws (n, R, K)
    R = halton.shape[0]
    Z = halton
    betas = mu[None, None, :] + sigma[None, None, :] * Z[None, :, :]    # (1, R, K)
    # utilities (n, J, R)
    util = np.einsum("njk,rk->njr", X, betas[0])
    util -= util.max(axis=1, keepdims=True)
    p = np.exp(util)
    p = p / p.sum(axis=1, keepdims=True)
    return p.mean(axis=2)    # (n, J)


def fit_mixed_logit(X, y, R=200, seed=0):
    """Fit mixed logit with independent normal taste distribution."""
    n, J, K = X.shape
    halton = qmc.Halton(d=K, seed=seed).random(R)
    # convert to standard normal via inverse CDF
    from scipy.stats import norm    # inverse CDF
    Z = norm.ppf(np.clip(halton, 1e-6, 1 - 1e-6))

    def negloglik(theta):
        mu = theta[:K]; log_sigma = theta[K:]
        p = _sim_choice_probs(X, mu, log_sigma, Z)
        p_chosen = p[np.arange(n), y]
        return -np.sum(np.log(np.clip(p_chosen, 1e-12, None)))

    x0 = np.concatenate([np.zeros(K), np.log(np.ones(K) * 0.5)])
    res = minimize(negloglik, x0, method="BFGS", options={"maxiter": 200})
    mu = res.x[:K]; sigma = np.exp(res.x[K:])
    return {"mu": mu, "sigma": sigma, "loglik": -float(res.fun)}


if __name__ == "__main__":
    print("=== Mixed logit MNL (McFadden-Train 2000) ===\n")
    rng = np.random.default_rng(0)
    n, J, K = 800, 3, 2
    X = rng.normal(size=(n, J, K))
    true_mu = np.array([1.0, -0.5])
    true_sigma = np.array([0.4, 0.3])
    beta_n = true_mu + true_sigma * rng.normal(size=(n, K))
    util = np.einsum("njk,nk->nj", X, beta_n) + rng.gumbel(size=(n, J))
    y = util.argmax(axis=1)
    print(f"  n={n}, J={J} alternatives, K={K} attributes")
    print(f"  Truth: mu = {true_mu}, sigma = {true_sigma}")

    fit = fit_mixed_logit(X, y, R=200)
    print(f"\n  MLE: mu    = {np.round(fit['mu'], 3)}")
    print(f"  MLE: sigma = {np.round(fit['sigma'], 3)}")
    print(f"  Log-lik    = {fit['loglik']:.2f}")

    print("\n--- library cross-check (mlogit / gmnl R; xlogit / pylogit Python) ---")
