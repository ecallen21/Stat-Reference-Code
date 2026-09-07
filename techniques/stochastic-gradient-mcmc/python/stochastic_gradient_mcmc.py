"""Stochastic-gradient MCMC (Reference Sec 25.9).

Welling & Teh 2011 'Bayesian learning via stochastic gradient
Langevin dynamics' (SGLD); Chen, Fox & Guestrin 2014 SGHMC. Scales
MCMC to large datasets by using MINI-BATCH gradient estimates instead
of the full-data likelihood gradient.

SGLD update at step t (learning rate eps_t):

    theta_{t+1} = theta_t
                  + (eps_t / 2) * (grad log prior + (n / b) * sum_{i in batch_t} grad log lik_i)
                  + Normal(0, eps_t)

The injected noise makes the trajectory approximate posterior Langevin
dynamics; step size annealed to zero gives asymptotic exactness
(Teh, Thiery & Vollmer 2016).

We fit a Bayesian linear regression on synthetic data with SGLD,
comparing to the analytic posterior mean.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sgld_bayes_linreg(X, y, sigma2=1.0, prior_var=100.0,
                       n_iter=5000, batch_size=32, eps0=1e-3,
                       ann_pow=0.55, seed=0):
    """SGLD for beta ~ N(0, prior_var * I),  y = X @ beta + N(0, sigma2)."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    beta = np.zeros(p)
    chain = np.zeros((n_iter, p))
    for t in range(n_iter):
        eps_t = eps0 * (t + 1) ** (-ann_pow / 2)     # step-size schedule (loose)
        idx = rng.choice(n, size=batch_size, replace=False)
        Xb = X[idx]; yb = y[idx]
        #  grad log p(y | beta) = X'(y - X beta) / sigma2
        grad_lik = Xb.T @ (yb - Xb @ beta) / sigma2 * (n / batch_size)
        grad_pri = -beta / prior_var
        noise = rng.normal(scale=np.sqrt(eps_t), size=p)
        beta = beta + (eps_t / 2) * (grad_lik + grad_pri) + noise
        chain[t] = beta
    return chain


if __name__ == "__main__":
    print("=== Stochastic-gradient Langevin dynamics (SGLD) ===\n")
    rng = np.random.default_rng(0)
    n = 5000; p = 4
    X = rng.normal(size=(n, p))
    beta_true = np.array([0.8, -0.3, 0.5, 0.0])
    y = X @ beta_true + rng.normal(scale=1.0, size=n)

    chain = sgld_bayes_linreg(X, y, sigma2=1.0, prior_var=100.0,
                               n_iter=6000, batch_size=64, eps0=5e-4)
    burn = 2000
    post_mean = chain[burn:].mean(axis=0)
    post_sd = chain[burn:].std(axis=0, ddof=1)

    #  Analytic posterior mean & sd
    A = X.T @ X / 1.0 + np.eye(p) / 100.0
    mu_analytic = np.linalg.solve(A, X.T @ y / 1.0)
    sd_analytic = np.sqrt(np.diag(np.linalg.inv(A)))

    print(f"  True beta      = {beta_true}")
    print(f"  SGLD post mean = {post_mean.round(3).tolist()}")
    print(f"  Analytic mean  = {mu_analytic.round(3).tolist()}")
    print(f"  SGLD post sd   = {post_sd.round(3).tolist()}")
    print(f"  Analytic sd    = {sd_analytic.round(3).tolist()}")
    print(f"\n  SGLD scales to n large via mini-batch grads; step-size annealing")
    print(f"  makes it asymptotically exact (Teh-Thiery-Vollmer 2016).")

    print("\n--- library cross-check (SGmcmc R, tfp SGMCMC Python; numpyro Python) ---")
