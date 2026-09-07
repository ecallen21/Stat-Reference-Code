"""Reversible-jump MCMC (Reference Sec 25.6).

Green 1995 'Reversible jump Markov chain Monte Carlo computation and
Bayesian model determination', Biometrika. Extends Metropolis-Hastings
to a space where the DIMENSION of the parameter vector can change,
allowing joint sampling over models AND their parameters.

    x = (k, theta_k)     where k indexes the model and theta_k in R^{n_k}.

RJ-MCMC proposes:
    * a move within model k (standard MH), OR
    * a jump to model k' with dimension-matching bijection u ~ q(u)
      and Jacobian J.

Acceptance ratio (jump k -> k'):

    A = [pi(k', theta_k') / pi(k, theta_k)]  x  [q(u') / q(u)]  x  |J|

Applications:
    * Bayesian model choice: variable selection, mixture components
      number, change-point number, spline knot number.

Toy demo: sample number k of Gaussian mixture components in {1, 2, 3}
plus per-component means for a small dataset.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.stats import norm


def log_prior(k, mu):
    """Prior: k Uniform{1,2,3}; mu iid Normal(0, 2)."""
    if not (1 <= k <= 3) or len(mu) != k:
        return -np.inf
    return np.sum(norm.logpdf(mu, 0, 2))


def log_lik(y, k, mu, sigma=1.0):
    """Equal-weight mixture: log sum_j (1/k) * phi((y - mu_j)/sigma)."""
    logp = np.zeros(len(y))
    stack = np.array([norm.logpdf(y, m, sigma) for m in mu]).T
    m_max = stack.max(axis=1)
    logp = m_max + np.log(np.mean(np.exp(stack - m_max[:, None]), axis=1))
    return logp.sum()


def within_move(mu, step):
    return mu + step * np.random.default_rng().normal(size=len(mu))


def rjmcmc(y, n_iter=8000, seed=0):
    rng = np.random.default_rng(seed)
    k = 1
    mu = np.array([y.mean()])
    trace_k = np.zeros(n_iter, dtype=int)
    for it in range(n_iter):
        u = rng.uniform()
        if u < 0.5:
            #  Within-model MH
            mu_new = mu + 0.3 * rng.normal(size=len(mu))
            log_a = (log_prior(k, mu_new) + log_lik(y, k, mu_new)
                     - log_prior(k, mu) - log_lik(y, k, mu))
            if np.log(rng.uniform()) < log_a:
                mu = mu_new
        else:
            #  Trans-dimensional move (Green split-merge lite)
            if k < 3 and rng.uniform() < 0.5:
                #  BIRTH: propose new mu_new from Normal(0, 2)
                new_mu = rng.normal(0, 2)
                mu_new = np.append(mu, new_mu)
                log_q_ratio = -norm.logpdf(new_mu, 0, 2)     # proposal density
                log_a = (log_prior(k + 1, mu_new) + log_lik(y, k + 1, mu_new)
                         - log_prior(k, mu) - log_lik(y, k, mu) + log_q_ratio)
                if np.log(rng.uniform()) < log_a:
                    k += 1; mu = mu_new
            elif k > 1:
                #  DEATH: drop a random component
                idx = rng.integers(0, k)
                mu_new = np.delete(mu, idx)
                dropped = mu[idx]
                log_q_ratio = norm.logpdf(dropped, 0, 2)
                log_a = (log_prior(k - 1, mu_new) + log_lik(y, k - 1, mu_new)
                         - log_prior(k, mu) - log_lik(y, k, mu) + log_q_ratio)
                if np.log(rng.uniform()) < log_a:
                    k -= 1; mu = mu_new
        trace_k[it] = k
    return trace_k


if __name__ == "__main__":
    print("=== Reversible-jump MCMC: mixture-model order selection ===\n")
    rng = np.random.default_rng(0)
    #  True generative: 2-component mixture at mu = -2, 2
    n = 200
    z = rng.integers(0, 2, size=n)
    y = np.where(z == 0, rng.normal(-2, 1, n), rng.normal(2, 1, n))

    trace = rjmcmc(y, n_iter=8000, seed=0)
    #  Discard burn-in
    trace_use = trace[2000:]
    freq = np.bincount(trace_use, minlength=4)[1:4] / len(trace_use)
    print(f"  Posterior P(k = 1) = {freq[0]:.3f}")
    print(f"  Posterior P(k = 2) = {freq[1]:.3f}   <- truth")
    print(f"  Posterior P(k = 3) = {freq[2]:.3f}")

    print("\n--- library cross-check (rjmcmc R, coda R; pymc / arviz Python) ---")
