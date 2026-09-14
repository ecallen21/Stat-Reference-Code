"""Tweedie GLM (Reference Sec 47.323).

Tweedie 1984; Jorgensen 1987 JRSS-B. Exponential-dispersion
family with power variance function:

    Var(Y) = phi * mu^p

Special cases:
    p = 0  Normal (identity)
    p = 1  Poisson (unit variance function mu)
    p = 2  Gamma
    p = 3  Inverse Gaussian
    p in (1, 2)  COMPOUND POISSON-GAMMA — mass at 0, continuous on (0, inf)

The compound-Poisson-gamma is standard for INSURANCE CLAIMS
(some policies have zero claims; positive claims are continuous).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def tweedie_log_lik(y, mu, phi, p):
    """log-density (up to a normalising constant) for Tweedie with power p in (1, 2)."""
    # For continuous positive part, use the saddlepoint / series expansion (see Dunn-Smyth 2005)
    # Here we use the DEVIANCE-based approximation good enough for MLE
    d = 2 * ((y * (y ** (1 - p) - mu ** (1 - p)) / (1 - p))
             - (y ** (2 - p) - mu ** (2 - p)) / (2 - p))
    return -0.5 * d / phi


def fit_tweedie_irls(X, y, p=1.5, max_iter=50):
    """IRLS fit for Tweedie GLM with log-link mean."""
    n, d = X.shape
    beta = np.zeros(d); beta[0] = np.log(y.mean() + 1e-6)
    for _ in range(max_iter):
        eta = X @ beta
        mu = np.exp(eta)
        # Variance: mu^p; weights = mu^(2-p) for log link, working z = eta + (y - mu) / mu
        w = mu ** (2 - p)
        z = eta + (y - mu) / (mu + 1e-9)
        WX = X * w[:, None]
        beta_new = np.linalg.solve(X.T @ WX, X.T @ (w * z))
        if np.linalg.norm(beta_new - beta) < 1e-8: break
        beta = beta_new
    # Dispersion phi estimate: mean of Pearson residuals squared / n - d
    mu = np.exp(X @ beta)
    pearson = (y - mu) / np.sqrt(mu ** p)
    phi = float((pearson ** 2).sum() / (n - d))
    return beta, phi


if __name__ == "__main__":
    print("=== Tweedie GLM (Tweedie 1984; Jorgensen 1987) ===\n")
    rng = np.random.default_rng(0)

    # Simulate compound Poisson-gamma insurance data (p = 1.5)
    n = 1000
    X = np.column_stack([np.ones(n), rng.uniform(0, 1, n), rng.uniform(0, 1, n)])
    true_beta = np.array([1.0, 2.0, -0.5])
    mu_true = np.exp(X @ true_beta)
    # For each i, draw N ~ Poisson(mu^(2-p) / (phi(2-p))) claim events, each Gamma
    phi = 1.0; p = 1.5
    lam = mu_true ** (2 - p) / (phi * (2 - p))
    N_i = rng.poisson(lam)
    y = np.zeros(n)
    for i in range(n):
        if N_i[i] == 0: continue
        shape = (2 - p) / (p - 1); scale = phi * (p - 1) * mu_true[i] ** (p - 1)
        y[i] = rng.gamma(N_i[i] * shape, scale)

    print(f"  Simulated compound Poisson-gamma with p = 1.5:")
    print(f"    Fraction of zeros: {(y == 0).mean() * 100:.1f}%")
    print(f"    Mean of nonzero:   {y[y > 0].mean():.2f}")
    print(f"    True beta:         {true_beta.tolist()}")

    beta_hat, phi_hat = fit_tweedie_irls(X, y, p=1.5, max_iter=100)
    print(f"    Estimated beta:    {beta_hat.round(2).tolist()}")
    print(f"    Estimated phi:     {phi_hat:.3f}")

    print(f"\n  Tweedie GLM is standard for:")
    print(f"    - insurance claim amounts (excess zeros + positive continuous)")
    print(f"    - rainfall data (dry spells + storm totals)")
    print(f"    - fisheries catch (many zeros + skewed positive)")

    print("\n--- library cross-check (statsmodels.genmod.families.Tweedie; tweedie R package) ---")
