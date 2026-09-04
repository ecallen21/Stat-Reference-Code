"""Measurement error / errors-in-variables (Reference Sec 38.7).

CLASSICAL error:  W = X + U,    U | X = 0,    Var(U) = sigma_u^2.
BERKSON error:    X = W + U.

Effect on regression:
  If Y = beta_0 + beta_1 X + eps but we regress on W (with classical
  error), the naive slope is ATTENUATED:

        beta_1^naive = beta_1 * sigma_x^2 / (sigma_x^2 + sigma_u^2)
                     = beta_1 * lambda           (lambda < 1)

Fixes here:

  1. REGRESSION CALIBRATION -- replace W by E[X | W] (linear in W
     under joint normality) and refit.
  2. SIMEX -- Simulation-Extrapolation (Cook-Stefanski 1994):
       add extra noise at levels zeta, refit, extrapolate zeta -> -1.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def regression_calibration(w, y, sigma_u2):
    """Naive slope + regression-calibration corrected slope."""
    n = len(w)
    xbar = w.mean(); ybar = y.mean()
    Sww = ((w - xbar) ** 2).sum() / (n - 1)
    Swy = ((w - xbar) * (y - ybar)).sum() / (n - 1)
    beta_naive = Swy / Sww
    # Estimate lambda = Var(X) / (Var(X) + sigma_u2) via Var(X) = Sww - sigma_u2
    var_x_hat = max(Sww - sigma_u2, 1e-9)
    lam = var_x_hat / (var_x_hat + sigma_u2)
    beta_rc = beta_naive / lam
    intercept_rc = ybar - beta_rc * xbar
    return {"beta_naive": float(beta_naive), "beta_rc": float(beta_rc),
            "intercept_rc": float(intercept_rc), "lambda": float(lam)}


def simex(w, y, sigma_u2, zetas=(0.5, 1.0, 1.5, 2.0), B=100, seed=0):
    """SIMEX (Cook-Stefanski 1994): add noise, refit, extrapolate to zeta = -1."""
    rng = np.random.default_rng(seed)
    est = []
    for z in zetas:
        betas = []
        for _ in range(B):
            e = rng.normal(0, np.sqrt(z * sigma_u2), size=len(w))
            w_perturbed = w + e
            xbar = w_perturbed.mean(); ybar = y.mean()
            Sww = ((w_perturbed - xbar) ** 2).sum()
            Swy = ((w_perturbed - xbar) * (y - ybar)).sum()
            betas.append(Swy / Sww)
        est.append(np.mean(betas))
    # Fit quadratic in zeta, then evaluate at zeta = -1
    zs = np.asarray(zetas, dtype=float)
    A = np.vstack([np.ones_like(zs), zs, zs ** 2]).T
    coef, *_ = np.linalg.lstsq(A, np.asarray(est), rcond=None)
    beta_simex = coef @ np.array([1, -1, 1])
    return {"beta_simex": float(beta_simex), "zeta_curve": list(zip(zetas, est))}


if __name__ == "__main__":
    print("=== Measurement error models: regression calibration + SIMEX ===\n")
    rng = np.random.default_rng(0)
    n = 500
    beta0, beta1 = 2.0, 1.5
    sigma_x = 1.0
    sigma_u = 0.7        # measurement-error SD
    sigma_e = 0.5
    X = rng.normal(0, sigma_x, n)
    U = rng.normal(0, sigma_u, n)
    W = X + U
    Y = beta0 + beta1 * X + rng.normal(0, sigma_e, n)

    rc = regression_calibration(W, Y, sigma_u ** 2)
    print(f"  True beta_1 = {beta1}")
    print(f"  Naive OLS on W:   beta_1_hat = {rc['beta_naive']:.3f}    (attenuation lambda hat = {rc['lambda']:.3f})")
    print(f"  Regression calibration: beta_1_hat = {rc['beta_rc']:.3f}")

    sim = simex(W, Y, sigma_u ** 2)
    print(f"  SIMEX (extrapolated to zeta = -1): beta_1_hat = {sim['beta_simex']:.3f}\n")

    print("--- library cross-check (R simex/mecor; Python custom + scipy.odr) ---")
