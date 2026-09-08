"""Quantile IV regression (Reference Sec 15.44).

Chernozhukov & Hansen 2005, 2008 'An IV model of quantile treatment
effects', Econometrica. Combines IV identification with quantile-
regression estimation:

    Q_tau(Y | D, X) = D * alpha(tau) + X * beta(tau)

when D is ENDOGENOUS, use instrument Z. The Chernozhukov-Hansen (CH)
estimator solves

    alpha_hat(tau) = argmin_a ||gamma_hat(a, tau)||
                                        with
    gamma_hat = argmin_g sum_i rho_tau(Y_i - D_i * a - X_i * b - Z_i * g)

The intuition: at the true alpha, Z has no residual effect on
conditional quantile (analogous to IV moment condition).

We implement a compact 1-D CH-QIV via a grid over alpha for a
scalar D.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize    # inner quantile fit


def check_loss(u, tau):
    return np.where(u >= 0, tau * u, (tau - 1) * u).sum()


def qr_fit(y, X, tau):
    """Simple quantile regression via check-loss minimisation."""
    p = X.shape[1]

    def obj(b):
        return check_loss(y - X @ b, tau)

    r = minimize(obj, np.zeros(p), method="Nelder-Mead",
                  options={"xatol": 1e-4, "fatol": 1e-4})
    return r.x


def ch_quantile_iv(y, d, X, Z, tau, grid=None):
    """Chernozhukov-Hansen 1-D IV quantile regression."""
    if grid is None:
        grid = np.linspace(-2, 3, 51)
    #  For each candidate alpha, fit QR of (y - d * alpha) on [X, Z]
    #  and find alpha making the Z coefficient (gamma) closest to 0.
    gammas = np.zeros(len(grid))
    for k, a in enumerate(grid):
        beta = qr_fit(y - d * a, np.c_[X, Z], tau)
        gammas[k] = beta[-1]                     # last coefficient = gamma
    idx = int(np.argmin(np.abs(gammas)))
    return {"alpha_hat": float(grid[idx]),
            "gamma_at_alpha_hat": float(gammas[idx]),
            "grid": grid, "gamma_grid": gammas}


if __name__ == "__main__":
    print("=== Chernozhukov-Hansen quantile IV regression ===\n")
    rng = np.random.default_rng(0)
    n = 500
    Z = rng.normal(size=n)
    u = rng.normal(size=n)                        # endogenous shock
    d = 0.6 * Z + 0.7 * u + rng.normal(size=n)   # endogenous D
    #  True model: Q_0.5(y | d) = 1.0 * d + N(0)  but endogeneity biases OLS
    y = 1.0 * d - 0.5 * u + rng.normal(scale=0.5, size=n)

    #  Naive median regression (no IV)
    beta_naive = qr_fit(y, np.c_[np.ones(n), d], 0.5)
    print(f"  Naive median regression coefficient on d = {beta_naive[1]:.3f}  "
          f"(biased due to endogeneity, truth 1.00)")

    #  Chernozhukov-Hansen IV quantile regression
    r = ch_quantile_iv(y, d, np.ones((n, 1)), Z.reshape(-1, 1), tau=0.5)
    print(f"  CH-QIV median coefficient on d           = {r['alpha_hat']:.3f}   (truth 1.00)")

    print("\n--- library cross-check (quantreg + ivqr / QuantReg-IV R;\n"
          "                          econml.iv / linearmodels quantile-IV custom) ---")
