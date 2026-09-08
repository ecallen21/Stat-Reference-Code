"""Elliptical slice sampling (Reference Sec 25.12).

Murray, Adams & MacKay 2010 'Elliptical slice sampling', AISTATS.
An MCMC sampler tailored to GAUSSIAN PRIOR + arbitrary likelihood:

    prior:      f ~ N(0, K)
    posterior:  p(f | data) proportional-to  L(f) * N(f; 0, K)

Update:
    1. Sample nu ~ N(0, K).
    2. Draw log u = log L(f) + log(uniform).
    3. Angle theta ~ Uniform(0, 2 pi); bracket = [theta - 2 pi, theta].
    4. Propose f' = f * cos(theta) + nu * sin(theta).
    5. Accept if log L(f') > log u; else shrink bracket to exclude
       theta and retry.

Advantages:
    * No tuning parameters (no step size).
    * Preserves Gaussian prior exactly.
    * Great for GP posteriors and factor models.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def elliptical_slice(f, log_L, K_chol, rng):
    """One ESS update. f: current, log_L: callable, K_chol: Cholesky of prior cov."""
    n = len(f)
    nu = K_chol @ rng.normal(size=n)
    log_u = log_L(f) + np.log(rng.uniform())
    theta = rng.uniform(0, 2 * np.pi)
    lo, hi = theta - 2 * np.pi, theta
    while True:
        f_new = f * np.cos(theta) + nu * np.sin(theta)
        if log_L(f_new) > log_u:
            return f_new
        if theta < 0: lo = theta
        else:         hi = theta
        theta = rng.uniform(lo, hi)


def ess_run(log_L, K, n_iter=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = K.shape[0]
    L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
    f = L @ rng.normal(size=n)
    chain = np.zeros((n_iter, n))
    for i in range(n_iter):
        f = elliptical_slice(f, log_L, L, rng)
        chain[i] = f
    return chain


if __name__ == "__main__":
    print("=== Elliptical slice sampling (Murray-Adams-MacKay 2010) ===\n")
    #  Toy: prior f ~ N(0, K) with K = squared-exp kernel; likelihood
    #  f ~ Normal(y, sigma^2) so posterior = GP regression posterior.
    rng = np.random.default_rng(0)
    n = 15
    x = np.linspace(0, 1, n)
    D2 = (x[:, None] - x[None, :]) ** 2
    K = np.exp(-D2 / (2 * 0.15 ** 2))
    y = np.sin(2 * np.pi * x) + 0.2 * rng.normal(size=n)
    sigma2 = 0.04

    def log_L(f):
        return -0.5 * np.sum((y - f) ** 2) / sigma2

    chain = ess_run(log_L, K, n_iter=2000, seed=0)
    burn = 500
    f_post = chain[burn:].mean(axis=0)
    print(f"  Sample size = {n}, kernel lengthscale = 0.15")
    print(f"  Posterior mean f_hat:  {f_post.round(2).tolist()}")
    print(f"  Observed y            : {y.round(2).tolist()}")
    print(f"  MSE(f_hat, sin(2 pi x)): {float(np.mean((f_post - np.sin(2 * np.pi * x)) ** 2)):.4f}")

    #  Compare with analytic GP posterior mean K (K + sigma^2 I)^-1 y
    inv = np.linalg.solve(K + sigma2 * np.eye(n), y)
    f_analytic = K @ inv
    print(f"  Analytic GP mean:      {f_analytic.round(2).tolist()}")
    print(f"  Corr(ESS, analytic)   = {float(np.corrcoef(f_post, f_analytic)[0, 1]):.4f}")

    print("\n--- library cross-check (spBayes / gpflow.gpmc R/Python; gpytorch ESS Python) ---")
