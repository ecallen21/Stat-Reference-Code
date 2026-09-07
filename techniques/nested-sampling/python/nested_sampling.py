"""Nested sampling (Reference Sec 25.10).

Skilling 2006 'Nested sampling for general Bayesian computation',
Bayesian Anal. Estimates the Bayesian EVIDENCE Z = int L(theta) *
pi(theta) dtheta by transforming the multi-dim integral into a 1-D
integral of L wrt PRIOR MASS X in (0, 1]:

    Z = int_0^1 L(X) dX
    X_i approx exp(-i / N)                (N 'live points')

Algorithm:
    1. Sample N points from the prior; find the WORST (lowest L).
    2. Record its likelihood L_i and prior-mass estimate X_i.
    3. Replace with a NEW sample from prior CONSTRAINED to L > L_i.
    4. Repeat until L stops growing appreciably.
    5. Z_hat = sum_i (X_{i-1} - X_i) * L_i.

Advantages:
    * Handles MULTIMODAL posteriors better than MCMC.
    * Simultaneous evidence + posterior sample.
    * Model-selection framework (compare Bayes factors).

We implement a minimal rejection-based nested sampler on a 1-D
Gaussian likelihood with uniform prior, checking against the
analytic evidence.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def rejection_prior_sample_above(logL, L_thresh, prior_sampler, rng, max_try=5000):
    for _ in range(max_try):
        theta = prior_sampler(rng)
        if logL(theta) > L_thresh:
            return theta, logL(theta)
    return None, None


def nested_sampling(logL, prior_sampler, N=100, n_iter=1500, seed=0):
    rng = np.random.default_rng(seed)
    live_theta = [prior_sampler(rng) for _ in range(N)]
    live_logL = np.array([logL(t) for t in live_theta])
    log_X = 0.0                              # log of prior mass
    log_Z = -np.inf
    for i in range(n_iter):
        idx = int(np.argmin(live_logL))
        L_min = live_logL[idx]
        theta_dead = live_theta[idx]
        #  Prior-mass shell
        log_X_new = -(i + 1) / N
        log_w = np.log(np.exp(log_X) - np.exp(log_X_new) + 1e-300)
        log_Z = np.logaddexp(log_Z, L_min + log_w)
        log_X = log_X_new
        #  Replace worst with new sample above L_min
        new_theta, new_L = rejection_prior_sample_above(logL, L_min, prior_sampler, rng)
        if new_theta is None:
            break
        live_theta[idx] = new_theta
        live_logL[idx] = new_L
    return log_Z, log_X


if __name__ == "__main__":
    print("=== Nested sampling (Skilling 2006) ===\n")
    #  Prior: theta ~ Uniform(-5, 5).  Likelihood: N(0, 1).
    #  Analytic evidence Z = integral over prior of Normal density
    #                       = P(-5 < Z < 5) / 10  approx 1 / 10 = 0.1
    #  But wait: L is a *likelihood* not a density on theta; we treat
    #  L(theta) = (2 pi)^-0.5 * exp(-theta^2 / 2), so Z = 0.1 as above.
    def prior_sampler(rng):
        return rng.uniform(-5, 5)

    def logL(theta):
        return -0.5 * theta ** 2 - 0.5 * np.log(2 * np.pi)

    log_Z, log_X = nested_sampling(logL, prior_sampler, N=80, n_iter=1500)
    Z_true = 0.1                         # 1 / 10 * integral of Normal ~ 0.1
    print(f"  True Z          = {Z_true:.4f}")
    print(f"  NS log Z_hat    = {log_Z:.3f}   -> Z_hat = {np.exp(log_Z):.4f}")

    print("\n--- library cross-check (dynesty / nestle / ultranest Python; nestedlyeq R) ---")
