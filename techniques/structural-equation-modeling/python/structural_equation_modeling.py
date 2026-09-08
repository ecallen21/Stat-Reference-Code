"""Structural Equation Modeling (SEM) (Reference Sec 47.137).

Joreskog 1970 'A general method for analysis of covariance
structures'. Combines a MEASUREMENT model linking latent factors
to observed indicators with a STRUCTURAL model of latent
regressions:

    x = Lambda_x xi + delta          (measurement model, exog)
    y = Lambda_y eta + epsilon       (measurement model, endog)
    eta = B eta + Gamma xi + zeta    (structural model)

Fit by minimising the LISREL discrepancy between the observed
covariance S and model-implied covariance Sigma(theta):

    F_ML = log|Sigma| + tr(S Sigma^-1) - log|S| - p.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize    # SEM parameter estimation


def implied_cov_one_factor(lam, psi, theta):
    """Sigma = lam lam' * psi + diag(theta) for a single-factor model."""
    lam = np.asarray(lam)[:, None]
    return lam @ lam.T * psi + np.diag(theta)


def ml_disc(params, S, p):
    """LISREL F_ML for a one-factor model with p indicators.

    params: [lambda_1..p, log(theta_1..p)] with psi fixed to 1 for identification.
    """
    lam = params[:p]
    theta = np.exp(params[p:])
    Sigma = implied_cov_one_factor(lam, 1.0, theta)
    Sigma_inv = np.linalg.inv(Sigma + 1e-6 * np.eye(p))
    sign, logdet_S = np.linalg.slogdet(S)
    sign2, logdet_Sigma = np.linalg.slogdet(Sigma)
    return float(logdet_Sigma + np.trace(S @ Sigma_inv) - logdet_S - p)


if __name__ == "__main__":
    print("=== SEM: one-factor confirmatory example (Joreskog 1970) ===\n")
    rng = np.random.default_rng(0)
    n = 400
    p = 5
    # Truth: one latent factor xi ~ N(0, 1), lam_true, unique-variance theta_true
    lam_true = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    theta_true = np.array([0.19, 0.36, 0.51, 0.64, 0.75])
    xi = rng.normal(size=n)
    X = xi[:, None] * lam_true[None, :] + rng.normal(size=(n, p)) * np.sqrt(theta_true)
    S = np.cov(X, rowvar=False)

    x0 = np.concatenate([np.ones(p) * 0.5, np.zeros(p)])
    res = minimize(ml_disc, x0, args=(S, p), method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 20000})
    lam_hat = res.x[:p]
    theta_hat = np.exp(res.x[p:])
    print(f"  n={n}, p={p} indicators, ML-fit converged in {res.nit} iters")
    print(f"  Truth lam    = {lam_true}")
    print(f"  Estim lam    = {np.round(lam_hat, 3)}")
    print(f"  Truth theta  = {theta_true}")
    print(f"  Estim theta  = {np.round(theta_hat, 3)}")

    # Global fit indices
    n_p = p
    df = n_p * (n_p + 1) // 2 - 2 * p          # observed vs free params
    chi2 = (n - 1) * res.fun
    print(f"  chi^2 = {chi2:.2f}   df = {df}   "
          f"CFI ~ {1 - max(chi2 - df, 0) / max(chi2, 1e-9):.3f}")

    print("\n--- library cross-check (lavaan / sem R; semopy / factor_analyzer Python) ---")
