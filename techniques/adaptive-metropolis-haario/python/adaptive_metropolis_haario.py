"""Adaptive Metropolis (Haario, Saksman & Tamminen 2001) (Reference Sec 25.8).

Random-walk Metropolis-Hastings where the proposal covariance is
UPDATED during sampling to match the empirical posterior covariance:

    Sigma_t = (2.38^2 / d) * Cov(X_1, ..., X_{t-1}) + eps * I

Diminishing-adaptation and containment conditions (Roberts & Rosenthal
2007) ensure ergodicity despite the non-Markov proposals.

Advantages:
    * No manual step-size tuning.
    * Handles correlated / poorly-scaled targets automatically.
    * Provably ergodic with Roberts-Rosenthal-style safeguards.

We implement AM on a 2-D anisotropic Gaussian target and compare to a
naive RWMH with identity proposal.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def rwmh(logpi, x0, prop_cov, n_iter=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = len(x0)
    x = x0.copy()
    lp = logpi(x)
    chain = np.zeros((n_iter, d))
    accept = 0
    for t in range(n_iter):
        prop = x + rng.multivariate_normal(np.zeros(d), prop_cov)
        lp_prop = logpi(prop)
        if np.log(rng.uniform()) < lp_prop - lp:
            x = prop; lp = lp_prop; accept += 1
        chain[t] = x
    return chain, accept / n_iter


def adaptive_metropolis(logpi, x0, n_iter=5000, adapt_start=200, seed=0):
    rng = np.random.default_rng(seed)
    d = len(x0)
    x = x0.copy()
    lp = logpi(x)
    chain = np.zeros((n_iter, d))
    accept = 0
    mean = np.zeros(d)
    cov = np.eye(d)
    s_d = 2.38 ** 2 / d
    for t in range(n_iter):
        if t < adapt_start:
            prop_cov = 0.1 * np.eye(d)
        else:
            prop_cov = s_d * (cov + 1e-4 * np.eye(d))
        prop = x + rng.multivariate_normal(np.zeros(d), prop_cov)
        lp_prop = logpi(prop)
        if np.log(rng.uniform()) < lp_prop - lp:
            x = prop; lp = lp_prop; accept += 1
        chain[t] = x
        #  Update running mean & covariance (Welford)
        if t == 0:
            mean = x.copy()
        else:
            new_mean = mean + (x - mean) / (t + 1)
            cov = (t * cov + np.outer(x - mean, x - new_mean)) / (t + 1)
            mean = new_mean
    return chain, accept / n_iter


if __name__ == "__main__":
    print("=== Adaptive Metropolis (Haario 2001) ===\n")
    #  Target: N([1, -1], Sigma) with Sigma highly correlated / anisotropic
    Sigma = np.array([[4.0, 3.6], [3.6, 4.0]])
    Sigma_inv = np.linalg.inv(Sigma)
    mu = np.array([1.0, -1.0])

    def logpi(x):
        d = x - mu
        return -0.5 * d @ Sigma_inv @ d

    x0 = np.array([0.0, 0.0])
    chain_rw, acc_rw = rwmh(logpi, x0, prop_cov=0.1 * np.eye(2), n_iter=4000, seed=0)
    chain_am, acc_am = adaptive_metropolis(logpi, x0, n_iter=4000, seed=0)

    #  Discard burn-in, compare covariance recovery
    burn = 500
    cov_rw = np.cov(chain_rw[burn:].T)
    cov_am = np.cov(chain_am[burn:].T)
    print(f"  True Sigma =\n    {Sigma[0]}\n    {Sigma[1]}\n")
    print(f"  Naive RWMH (cov=0.1 I):  accept = {acc_rw:.3f}")
    print(f"    Estimated cov =\n      {cov_rw[0].round(2)}\n      {cov_rw[1].round(2)}")
    print(f"\n  Adaptive Metropolis:      accept = {acc_am:.3f}")
    print(f"    Estimated cov =\n      {cov_am[0].round(2)}\n      {cov_am[1].round(2)}")

    #  ESS proxy (autocorr time)
    from scipy.stats import pearsonr
    for name, ch in [("RWMH", chain_rw), ("AM", chain_am)]:
        rho = pearsonr(ch[burn:-1, 0], ch[burn + 1:, 0])[0]
        print(f"  {name}: lag-1 autocorrelation on dim 0 = {rho:.3f}")

    print("\n--- library cross-check (MCMCpack / adaptMCMC R; pymc AdaptiveMetropolis Python) ---")
