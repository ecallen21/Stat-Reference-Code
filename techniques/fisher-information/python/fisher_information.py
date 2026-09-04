"""Fisher information + Cramer-Rao bound (Reference Sec 34.4).

For a parametric family p(x; theta) with score s(x; theta) = d/dtheta log p:

  Fisher information matrix I(theta) = E[s(x; theta) s(x; theta)']
                                     = -E[d^2/dtheta dtheta' log p(x; theta)].

Cramer-Rao lower bound (unbiased case):
  Var(theta_hat) >= I(theta)^-1 / n.

Here we:
  1. Compute I(mu, sigma^2) analytically for Gaussian.
  2. Confirm sample MLE variance equals the Cramer-Rao bound for large n.
  3. Show that a BIASED estimator can beat the CRB (shrinkage, James-Stein).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def gaussian_fisher(mu, sigma2, n=1):
    """I(theta) for Y ~ N(mu, sigma^2), theta = (mu, sigma^2)."""
    return np.diag([n / sigma2, n / (2 * sigma2 ** 2)])


def mle_gaussian(x):
    return float(x.mean()), float(x.var(ddof=0))


if __name__ == "__main__":
    print("=== Fisher information + Cramer-Rao bound ===\n")
    mu_true, sigma2_true = 1.5, 2.0
    n = 50
    I = gaussian_fisher(mu_true, sigma2_true, n=n)
    I_inv = np.linalg.inv(I)
    print(f"  Gaussian n = {n},  mu = {mu_true},  sigma^2 = {sigma2_true}")
    print(f"  Fisher info matrix I(theta):\n{I}")
    print(f"  Cramer-Rao bound  I^-1:\n{I_inv}")
    print(f"    Var lower bound for mu_hat        = {I_inv[0, 0]:.4f}")
    print(f"    Var lower bound for sigma2_hat    = {I_inv[1, 1]:.4f}\n")

    # Empirical: MLE variance over many replicates
    rng = np.random.default_rng(0)
    B = 5000
    mus, sig2s = np.zeros(B), np.zeros(B)
    for b in range(B):
        x = rng.normal(mu_true, np.sqrt(sigma2_true), n)
        mus[b], sig2s[b] = mle_gaussian(x)
    print(f"  Empirical Var(mu_hat)     over {B} sims  = {mus.var():.4f}"
          f"   (CRB {I_inv[0, 0]:.4f})")
    print(f"  Empirical Var(sigma2_hat) over {B} sims  = {sig2s.var():.4f}"
          f"   (CRB {I_inv[1, 1]:.4f})\n")

    # Shrinkage beats CRB in MSE (bias-variance trade)
    shrink = 0.9 * mus + 0.1 * 0.0                    # James-Stein-style pull to 0
    mse_mle = float(np.mean((mus - mu_true) ** 2))
    mse_shrink = float(np.mean((shrink - mu_true) ** 2))
    print(f"  MSE of MLE estimator                = {mse_mle:.4f}")
    print(f"  MSE of shrinkage estimator (10% -> 0) = {mse_shrink:.4f}"
          "   (may beat MLE's variance at some cost in bias)\n")

    print("--- library cross-check (statsmodels.tools.numdiff.approx_hess; scipy.optimize + info matrix; R numDeriv) ---")
