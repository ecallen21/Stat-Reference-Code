"""Fay-Herriot small-area estimation model (Reference Sec 27.6).

Fay & Herriot 1979 'Estimates of income for small places: an
application of James-Stein procedures to census data', JASA. The
canonical area-level linear mixed model for small-area estimation.

Model:
    theta_i        = x_i' * beta + v_i           (true small-area mean)
    y_i (direct)  = theta_i + e_i                 (survey estimate with known SE)
    v_i ~ N(0, sigma_v^2)                         (area random effect)
    e_i ~ N(0, D_i)                                (design-based sampling var, known)

Estimator: shrunken EBLUP of theta_i:

    theta_hat_i = gamma_i * y_i + (1 - gamma_i) * x_i' * beta_hat
    gamma_i     = sigma_v^2 / (sigma_v^2 + D_i)

Highly-precise areas (small D_i) get gamma near 1 -> direct estimate
kept; noisy areas (large D_i) get shrunk toward the covariate model.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize_scalar    # variance MLE


def fay_herriot_fit(y, X, D):
    """MLE of (beta, sigma_v^2). Returns EBLUPs and shrinkage weights."""
    n, p = X.shape

    def gls_beta(sv2):
        V_inv = np.diag(1 / (sv2 + D))
        A = X.T @ V_inv @ X
        b = X.T @ V_inv @ y
        return np.linalg.solve(A, b)

    def neg_ll(sv2):
        if sv2 <= 0:
            return 1e12
        b = gls_beta(sv2)
        r = y - X @ b
        V = sv2 + D
        return 0.5 * (np.log(V).sum() + (r ** 2 / V).sum())

    res = minimize_scalar(neg_ll, bounds=(1e-6, 100), method="bounded")
    sv2 = res.x
    beta = gls_beta(sv2)
    #  EBLUP for each area
    gamma = sv2 / (sv2 + D)
    theta_hat = gamma * y + (1 - gamma) * (X @ beta)
    return {"beta": beta, "sigma_v_sq": sv2, "gamma": gamma,
            "theta_hat": theta_hat, "loglik": -res.fun}


if __name__ == "__main__":
    print("=== Fay-Herriot small-area model ===\n")
    rng = np.random.default_rng(0)
    n = 100          # small areas
    p = 2

    #  Covariate model + true random-effect variance
    X = np.c_[np.ones(n), rng.normal(size=n)]
    beta_true = np.array([2.0, 0.7])
    sigma_v = 0.5
    theta_true = X @ beta_true + rng.normal(scale=sigma_v, size=n)

    #  Direct survey estimates with heterogeneous, KNOWN sampling variances D_i
    D = rng.gamma(shape=2.0, scale=0.15, size=n)
    y = theta_true + rng.normal(scale=np.sqrt(D), size=n)

    r = fay_herriot_fit(y, X, D)

    print(f"  n areas = {n}")
    print(f"  True (beta_0, beta_1)      = ({beta_true[0]:.2f}, {beta_true[1]:.2f})")
    print(f"  Estimated (beta_0, beta_1) = ({r['beta'][0]:.3f}, {r['beta'][1]:.3f})")
    print(f"  True sigma_v^2  = {sigma_v ** 2:.3f}")
    print(f"  Est. sigma_v^2  = {r['sigma_v_sq']:.3f}")
    print(f"  Shrinkage weights (min, mean, max) = "
          f"({r['gamma'].min():.2f}, {r['gamma'].mean():.2f}, {r['gamma'].max():.2f})")

    #  MSE against theta_true
    mse_direct = float(np.mean((y - theta_true) ** 2))
    mse_synth = float(np.mean((X @ r['beta'] - theta_true) ** 2))
    mse_eblup = float(np.mean((r["theta_hat"] - theta_true) ** 2))
    print(f"\n  MSE vs true theta:")
    print(f"    direct-survey estimator = {mse_direct:.4f}")
    print(f"    synthetic (regression only) = {mse_synth:.4f}")
    print(f"    Fay-Herriot EBLUP         = {mse_eblup:.4f}")
    print(f"    EBLUP saves {(1 - mse_eblup / mse_direct) * 100:.0f} % of direct MSE.")

    print("\n--- library cross-check (sae R, JoSAE R; samplics Python) ---")
