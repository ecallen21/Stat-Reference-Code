"""Bayesian quantile regression (Reference Sec 33.2).

Yu & Moyeed (2001) 'Bayesian quantile regression.'

Regular quantile regression at level tau in (0, 1) minimises the
CHECK LOSS  rho_tau(u) = u * (tau - I{u < 0}).

The ASYMMETRIC LAPLACE likelihood

  p(y | mu, sigma, tau) = tau (1 - tau) / sigma  *  exp( - rho_tau((y - mu)/sigma) )

produces MLE == quantile regression at level tau. Combining this
likelihood with a Gaussian prior on the coefficient vector gives a
proper Bayesian posterior that we can sample by Metropolis-Hastings.

Advantages over frequentist QR:
  * FULL posterior + credible intervals without asymptotic sandwich SEs.
  * Easy to add priors / regularisation / hierarchical structure.
  * Handles small n well.

Here we sample the Bayesian QR posterior on synthetic heteroscedastic
data at three levels (tau = 0.10, 0.50, 0.90) and report posterior
medians + 95% credible intervals.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def check_loss(u, tau):
    return u * (tau - (u < 0).astype(float))


def _log_asymmetric_laplace(y, mu, sigma, tau):
    r = (y - mu) / sigma
    return np.log(tau * (1 - tau)) - np.log(sigma) - check_loss(r, tau)


def log_posterior(beta, log_sigma, X, y, tau, prior_sd=10.0):
    mu = X @ beta
    sigma = np.exp(log_sigma)
    ll = _log_asymmetric_laplace(y, mu, sigma, tau).sum()
    prior = -0.5 * np.sum(beta ** 2) / (prior_sd ** 2) - 0.5 * log_sigma ** 2
    return ll + prior


def metropolis(X, y, tau=0.5, n_iter=5000, burn=500, step=0.05, seed=0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    beta = np.zeros(d)
    log_sigma = 0.0
    chain = np.zeros((n_iter, d + 1))
    lp = log_posterior(beta, log_sigma, X, y, tau)
    accept = 0
    for t in range(n_iter):
        beta_p = beta + rng.normal(0, step, d)
        ls_p = log_sigma + rng.normal(0, step)
        lp_p = log_posterior(beta_p, ls_p, X, y, tau)
        if np.log(rng.random()) < lp_p - lp:
            beta, log_sigma, lp = beta_p, ls_p, lp_p
            accept += 1
        chain[t, :d] = beta; chain[t, d] = log_sigma
    return chain[burn:], accept / n_iter


if __name__ == "__main__":
    print("=== Bayesian quantile regression (Yu-Moyeed 2001) ===\n")
    rng = np.random.default_rng(0)
    n = 300
    x = np.linspace(-2, 2, n)
    # Heteroscedastic: sd grows with |x|; quantile slopes differ.
    y = 1.0 + 0.5 * x + (0.4 + 0.6 * np.abs(x)) * rng.normal(0, 1, n)
    X = np.stack([np.ones(n), x], axis=1)

    print(f"  {'tau':>5}  {'intercept (med [95% CrI])':>30}"
          f"  {'slope (med [95% CrI])':>28}  {'acc rate':>10}")
    for tau in (0.10, 0.50, 0.90):
        chain, acc = metropolis(X, y, tau=tau, n_iter=6000, burn=1000, step=0.03)
        ints = chain[:, 0]; sls = chain[:, 1]
        i_lo, i_med, i_hi = np.quantile(ints, [0.025, 0.5, 0.975])
        s_lo, s_med, s_hi = np.quantile(sls, [0.025, 0.5, 0.975])
        print(f"  {tau:>5.2f}  {i_med:>7.3f} [{i_lo:>5.3f}, {i_hi:>5.3f}]"
              f"  {s_med:>7.3f} [{s_lo:>5.3f}, {s_hi:>5.3f}]"
              f"  {acc:>10.2f}")

    print("\n  Slopes should be similar across tau since the mean function is linear;\n"
          "  intercepts and CrI widths reveal the heteroscedastic scale.\n")
    print("--- library cross-check (statsmodels QuantReg + laplace-priors; brms brmsfamily='asym_laplace') ---")
