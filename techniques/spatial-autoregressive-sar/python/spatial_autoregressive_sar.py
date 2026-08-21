"""Spatial autoregressive models: SAR lag + SAR error (Reference §23.9).

Spatial Lag Model (SLM / SAR-lag)
    y = rho W y + X beta + eps,     eps ~ N(0, sigma^2 I)
    rho: spatial dependence in the OUTCOME.

Spatial Error Model (SEM / SAR-error)
    y = X beta + u,   u = lambda W u + eps
    lambda: spatial dependence in the ERROR term.

Both estimated by concentrated maximum likelihood: profile out (beta, sigma)
and search over rho (or lambda) in the interval of stability
(-1/eig_max, +1/eig_max).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy.optimize import minimize_scalar    # SciPy 1-D optimizer


def sar_lag_mle(y, X, W) -> dict:
    """Spatial lag MLE via concentrated log-likelihood over rho."""
    y = np.asarray(y, dtype=float); X = np.asarray(X, dtype=float); W = np.asarray(W, dtype=float)
    n, k = X.shape
    def neg_ll(rho):
        A = np.eye(n) - rho * W
        Ay = A @ y
        beta = np.linalg.solve(X.T @ X, X.T @ Ay)
        resid = Ay - X @ beta
        sig2 = float(resid @ resid / n)
        sign, logdet = np.linalg.slogdet(A)
        if sign <= 0: return 1e10
        return -(logdet - 0.5 * n * math.log(2 * math.pi * sig2) - 0.5 * n)
    # Coarse grid then refine to avoid local flats
    grid = np.linspace(-0.9, 0.9, 41)
    grid_vals = [neg_ll(r) for r in grid]
    best_r = grid[int(np.argmin(grid_vals))]
    res = minimize_scalar(neg_ll, bounds=(max(-0.99, best_r - 0.1), min(0.99, best_r + 0.1)),
                          method="bounded")
    rho = float(res.x)
    A = np.eye(n) - rho * W
    Ay = A @ y
    beta = np.linalg.solve(X.T @ X, X.T @ Ay)
    resid = Ay - X @ beta
    sig2 = float(resid @ resid / n)
    return {"rho": rho, "beta": beta, "sigma2": sig2, "log_lik": float(-res.fun),
            "method": "SAR lag MLE (concentrated likelihood)"}


def sar_error_mle(y, X, W) -> dict:
    """Spatial error MLE via concentrated log-likelihood over lambda."""
    y = np.asarray(y, dtype=float); X = np.asarray(X, dtype=float); W = np.asarray(W, dtype=float)
    n, k = X.shape
    def neg_ll(lam):
        B = np.eye(n) - lam * W
        BX = B @ X; By = B @ y
        beta = np.linalg.solve(BX.T @ BX, BX.T @ By)
        resid = By - BX @ beta
        sig2 = float(resid @ resid / n)
        sign, logdet = np.linalg.slogdet(B)
        if sign <= 0: return 1e10
        return -(logdet - 0.5 * n * math.log(2 * math.pi * sig2) - 0.5 * n)
    res = minimize_scalar(neg_ll, bounds=(-0.99, 0.99), method="bounded")
    lam = float(res.x)
    B = np.eye(n) - lam * W
    BX = B @ X; By = B @ y
    beta = np.linalg.solve(BX.T @ BX, BX.T @ By)
    resid = By - BX @ beta
    sig2 = float(resid @ resid / n)
    return {"lambda": lam, "beta": beta, "sigma2": sig2, "log_lik": float(-res.fun),
            "method": "SAR error MLE"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Grid with row-normalized rook weights
    m = 10; n = m * m
    coords = np.array([(i, j) for i in range(m) for j in range(m)], dtype=float)
    D = np.abs(coords[:, None] - coords[None, :]).sum(-1)
    W = (D == 1).astype(float); W = W / W.sum(1, keepdims=True)
    # Simulate y = (I - rho W)^-1 (X beta + eps) with a spatially correlated x
    # so the spatial dependence is identifiable.
    x = coords[:, 0] + rng.normal(0, 0.5, n)          # gradient covariate
    X = np.column_stack([np.ones(n), x])
    beta_true = np.array([1.0, 0.5]); rho_true = 0.5
    eps = rng.normal(0, 0.5, n)                        # small noise
    y = np.linalg.solve(np.eye(n) - rho_true * W, X @ beta_true + eps)

    r = sar_lag_mle(y, X, W)
    print(f"=== SAR lag MLE (true rho = 0.5, beta = 1.0, 0.5) ===")
    print(f"  rho = {r['rho']:.3f}, beta = {r['beta'].round(3)}, sigma^2 = {r['sigma2']:.3f}")
    print(f"  log-lik = {r['log_lik']:.2f}")

    # OLS ignoring spatial dependence (biased)
    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f"\n=== OLS (biased) ===")
    print(f"  beta = {beta_ols.round(3)}")

    r = sar_error_mle(y, X, W)
    print(f"\n=== SAR error MLE ===")
    print(f"  lambda = {r['lambda']:.3f}, beta = {r['beta'].round(3)}, sigma^2 = {r['sigma2']:.3f}")

    print("\n--- library cross-check (R spatialreg::lagsarlm / errorsarlm) ---")
