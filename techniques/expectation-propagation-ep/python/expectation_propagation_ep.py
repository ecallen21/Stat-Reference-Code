"""Expectation Propagation (Reference Sec 47.120).

Minka 2001 'Expectation propagation for approximate Bayesian
inference', UAI. Approximate a posterior p(theta) proportional to
prior * prod_i f_i(theta) by a member q(theta) of an exponential
family. Iterate over sites i:

    q_{-i}(theta) proportional to q(theta) / f_i_tilde(theta)          (cavity)
    q_new(theta) = KL-project ( q_{-i}(theta) f_i(theta) )   (moment match)
    f_i_tilde(theta) proportional to q_new(theta) / q_{-i}(theta)      (site update)

Powerful for Gaussian-process classification (Rasmussen-Williams 2006
ch 3.6) and Bayesian NN Laplace-EP hybrids.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.stats import norm    # standard normal


def ep_binary_gp_1d(X, y, sigma_prior=2.0, n_iter=30, tol=1e-6):
    """EP for a 1-D Bayesian logistic regression: prior N(0, sigma_prior^2),
    likelihood Phi(y_i x_i beta) (probit). Site params (nu_i, tau_i).
    """
    n = len(X)
    nu = np.zeros(n); tau = np.zeros(n)
    for it in range(n_iter):
        max_change = 0.0
        for i in range(n):
            # Compute cavity marginal q_{-i}(beta) = N(mu_c, s2_c)
            s2 = 1.0 / (1.0 / sigma_prior ** 2 + tau.sum() - tau[i])
            mu = s2 * (nu.sum() - nu[i])
            # Moment-match tilted N(mu_c, s2_c) * Phi(y_i x_i beta)
            z = (y[i] * X[i] * mu) / np.sqrt(1 + X[i] ** 2 * s2)
            phi = norm.pdf(z); Phi = norm.cdf(z) + 1e-12
            # New marginal moments
            alpha = y[i] * X[i] / np.sqrt(1 + X[i] ** 2 * s2) * (phi / Phi)
            beta_up = alpha * (alpha + mu * X[i] * y[i] / np.sqrt(1 + X[i] ** 2 * s2))
            mu_new = mu + s2 * alpha
            s2_new = s2 * (1 - s2 * beta_up)
            # Convert to site params
            tau_new = max(1.0 / s2_new - 1.0 / s2, 0.0)
            nu_new = mu_new / s2_new - mu / s2
            max_change = max(max_change, abs(tau[i] - tau_new), abs(nu[i] - nu_new))
            tau[i] = tau_new; nu[i] = nu_new
        if max_change < tol:
            break
    # Posterior
    s2_post = 1.0 / (1.0 / sigma_prior ** 2 + tau.sum())
    mu_post = s2_post * nu.sum()
    return {"mu": float(mu_post), "s2": float(s2_post), "iters": it + 1}


if __name__ == "__main__":
    print("=== EP (Minka 2001) demo: probit regression ===\n")
    rng = np.random.default_rng(0)
    n = 300
    beta_true = 1.5
    X = rng.normal(size=n)
    y = np.where(rng.random(n) < norm.cdf(beta_true * X), 1, -1)

    ep = ep_binary_gp_1d(X, y, sigma_prior=2.0)
    print(f"  EP posterior mean = {ep['mu']:.3f}   sd = {np.sqrt(ep['s2']):.3f}")
    print(f"  Truth beta = {beta_true}")

    # Grid-based ground-truth via prior * likelihood
    grid = np.linspace(-3, 5, 401)
    log_prior = -0.5 * grid ** 2 / 4
    log_lik = np.sum(np.log(norm.cdf(y[:, None] * X[:, None] * grid[None, :]) + 1e-12), axis=0)
    log_post = log_prior + log_lik
    log_post -= log_post.max()
    p = np.exp(log_post); p /= p.sum()
    mean_grid = float((grid * p).sum())
    sd_grid = float(np.sqrt(((grid - mean_grid) ** 2 * p).sum()))
    print(f"\n  Grid posterior mean = {mean_grid:.3f}   sd = {sd_grid:.3f}   (target)")

    print("\n--- library cross-check (limited R; EPGL / GPy Python) ---")
