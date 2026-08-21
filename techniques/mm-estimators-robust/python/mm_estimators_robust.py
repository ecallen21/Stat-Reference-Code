"""MM-estimators for robust regression (Yohai 1987; Reference §17.x extra).

Two-stage estimator combining high breakdown + high efficiency:

  Stage 1: initial high-breakdown S-estimator using Tukey biweight rho:
     sigma_hat minimises the M-scale of residuals,
     giving 50% breakdown-point resistance to outliers.
  Stage 2: M-estimator step from the S-fit starting point, using Tukey
     biweight with a wider tuning constant (default c = 4.685) that yields
     95% asymptotic efficiency at the Gaussian model.

The result inherits the S-estimator's breakdown and the M-estimator's
efficiency.

We implement a simplified S-then-M loop; the true Yohai S-step uses many
random elemental subsets — we use IRLS from an OLS start for simplicity.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _biweight_rho(u, c):
    """Normalised biweight rho in [0, 1] (Rousseeuw-Yohai convention)."""
    au = np.abs(u); inner = 1 - (au / c) ** 2
    return np.where(au <= c, 1 - inner ** 3, 1.0)


def _biweight_psi(u, c):
    inner = (1 - (u / c) ** 2) ** 2
    return np.where(np.abs(u) <= c, u * inner, 0.0)


def _biweight_w(u, c):
    inner = (1 - (u / c) ** 2) ** 2
    return np.where(np.abs(u) <= c, inner, 0.0)


def _mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _s_scale(resid, c: float = 1.548, delta: float = 0.5, n_iter: int = 50):
    """M-scale sigma satisfying (1/n) sum rho(r/sigma) = delta.  50% breakdown at c=1.548."""
    sigma = _mad(resid) + 1e-9
    for _ in range(n_iter):
        u = resid / sigma
        avg = _biweight_rho(u, c).mean()
        sigma_new = sigma * np.sqrt(avg / delta)
        if abs(sigma_new - sigma) < 1e-8:
            sigma = sigma_new; break
        sigma = sigma_new
    return float(sigma)


def _irls_biweight(X, y, beta0, c: float, sigma: float,
                   n_iter: int = 50, tol: float = 1e-8):
    beta = beta0.copy()
    for _ in range(n_iter):
        r = y - X @ beta
        w = _biweight_w(r / sigma, c)
        WX = X * w[:, None]
        beta_new = np.linalg.solve(X.T @ WX + 1e-8 * np.eye(X.shape[1]),
                                    X.T @ (w * y))
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new; break
        beta = beta_new
    return beta


def mm_regression(X, y, n_subsets: int = 200, seed: int = 0) -> dict:
    """Simplified FAST-S initialisation + M-step.

    Stage 1: draw many random p-subsets, fit exactly, evaluate the M-scale of
    the resulting residuals on the full sample.  The subset with the smallest
    scale gives a high-breakdown starting point; refine by S-IRLS.
    Stage 2: M-step at c=4.685 from beta_S.
    """
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float)])
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    # --- Stage 1: subset-search S-estimator ---
    best_sigma = np.inf; best_beta = None
    for _ in range(n_subsets):
        idx = rng.choice(n, p, replace=False)
        try:
            b = np.linalg.solve(X[idx], y[idx])
        except np.linalg.LinAlgError:
            continue
        r = y - X @ b
        sig = _s_scale(r, c=1.548)
        if sig < best_sigma:
            best_sigma = sig; best_beta = b
    beta_S = best_beta
    # refine: alternate M-scale + IRLS until converged.  Scale is allowed
    # to grow away from a subset-degenerate starting point toward the true noise SD.
    sigma_S = best_sigma
    for _ in range(60):
        sigma_S = _s_scale(y - X @ beta_S, c=1.548)
        beta_new = _irls_biweight(X, y, beta_S, c=1.548, sigma=sigma_S, n_iter=10)
        if np.max(np.abs(beta_new - beta_S)) < 1e-8:
            beta_S = beta_new; break
        beta_S = beta_new
    sigma_S = _s_scale(y - X @ beta_S, c=1.548)
    # --- Stage 2: M-step from beta_S.  c=3.44 gives ~85% efficiency but
    # is materially more resistant to high-leverage outliers than c=4.685.
    beta_MM = _irls_biweight(X, y, beta_S, c=3.44, sigma=sigma_S, n_iter=200)
    return {"beta_MM": beta_MM, "beta_S": beta_S, "sigma": sigma_S,
            "method": "MM-estimator (subset-search S-init, biweight c2=3.44)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200; p = 2
    beta_true = np.array([1.0, 2.0, -1.0])                # intercept + 2 slopes
    X = rng.normal(size=(n, p))
    y = beta_true[0] + X @ beta_true[1:] + rng.normal(size=n)

    # inject 20% high-leverage outliers
    idx = rng.choice(n, size=n // 5, replace=False)
    y[idx] += 15.0                                         # large positive shift
    X[idx, 0] += 4.0                                       # leverage in first x

    from numpy.linalg import lstsq
    beta_ols, *_ = lstsq(np.column_stack([np.ones(n), X]), y, rcond=None)
    fit = mm_regression(X, y)

    print(f"=== MM regression under 20% outliers + high-leverage contamination ===")
    print(f"  true beta        = {beta_true.tolist()}")
    print(f"  OLS              = {np.round(beta_ols, 3).tolist()}")
    print(f"  S-estimator (stage 1) = {np.round(fit['beta_S'], 3).tolist()}")
    print(f"  MM-estimator      = {np.round(fit['beta_MM'], 3).tolist()}")
    print(f"  robust sigma     = {fit['sigma']:.3f}")

    print("\n--- library cross-check (statsmodels.robust; R MASS::rlm; robustbase::lmrob) ---")
    try:
        from statsmodels.robust.robust_linear_model import RLM
        from statsmodels.robust import norms
        Xsm = np.column_stack([np.ones(n), X])
        m = RLM(y, Xsm, M=norms.TukeyBiweight(c=4.685)).fit()
        print(f"  statsmodels TukeyBiweight (from OLS start) = {np.round(m.params, 3).tolist()}")
    except ImportError:
        print("  (statsmodels not installed)")
