"""Inverse-Gaussian GLM (Reference Sec 47.325).

Tweedie 1957; McCullagh & Nelder 1989 (GLM textbook). GLM with
INVERSE GAUSSIAN response distribution:

    Y ~ IG(mu, sigma^2)         E[Y] = mu, Var(Y) = sigma^2 * mu^3

The canonical link is 1/mu^2, but log-link is more common in
practice. Used for skewed positive-only outcomes with variance
that grows CUBICALLY with the mean (rare events, extreme skew).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def fit_ig_glm(X, y, link="log", max_iter=50):
    """IRLS for Inverse Gaussian GLM."""
    n, d = X.shape
    beta = np.zeros(d)
    if link == "log":
        beta[0] = np.log(y.mean() + 1e-9)
    for _ in range(max_iter):
        eta = X @ beta
        if link == "log":
            mu = np.exp(eta)
            g_prime = 1.0 / (mu + 1e-9)                            # d eta / d mu
        else:
            mu = np.maximum(eta, 1e-9)
            g_prime = np.ones_like(mu)
        V = mu ** 3
        w = 1.0 / (g_prime ** 2 * V + 1e-12)
        z = eta + (y - mu) * g_prime
        WX = X * w[:, None]
        beta_new = np.linalg.solve(X.T @ WX, X.T @ (w * z))
        if np.linalg.norm(beta_new - beta) < 1e-8: break
        beta = beta_new
    mu = np.exp(X @ beta) if link == "log" else X @ beta
    resid = (y - mu) / np.sqrt(mu ** 3)
    sigma2 = float((resid ** 2).sum() / (n - d))
    return beta, sigma2


if __name__ == "__main__":
    print("=== Inverse-Gaussian GLM (Tweedie 1957) ===\n")
    rng = np.random.default_rng(0)

    n = 500
    X = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
    true_beta = np.array([1.0, 2.0])
    mu_true = np.exp(X @ true_beta)

    # Draw IG samples via the standard Michael-Schucany-Haas transform (1976).
    # For IG(mu, lambda) where Var = mu^3 / lambda, and sigma^2 = 1 / lambda.
    def sample_ig(mu, lam, rng):
        v = rng.standard_normal()
        yv = v * v
        x = mu + (mu ** 2 * yv) / (2 * lam) - \
            (mu / (2 * lam)) * np.sqrt(4 * mu * lam * yv + mu ** 2 * yv ** 2)
        u = rng.uniform()
        return x if u <= mu / (mu + x) else mu ** 2 / x

    sigma2 = 0.5
    lam = 1.0 / sigma2
    y = np.array([sample_ig(m, lam, rng) for m in mu_true])
    phi = sigma2                                                  # notation used in report
    print(f"  n = {n}, log-link, dispersion phi = {phi}")
    print(f"  y: mean = {y.mean():.2f}, std = {y.std():.2f}, min = {y.min():.3f}")

    beta_hat, sigma2_hat = fit_ig_glm(X, y, link="log", max_iter=50)
    print(f"\n  True beta:      {true_beta.tolist()}")
    print(f"  Estimated beta: {beta_hat.round(3).tolist()}")
    print(f"  Estimated dispersion: {sigma2_hat:.4f}   (true {phi})")

    print(f"\n  IG is preferred over Gamma when data have a HEAVIER right tail")
    print(f"  and MORE peaked mode near 0 - reliability, waiting times.")

    print("\n--- library cross-check (statsmodels.genmod.families.InverseGaussian; MASS::glm.nb R) ---")
