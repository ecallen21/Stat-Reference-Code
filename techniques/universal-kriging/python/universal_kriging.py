"""Universal (drift) kriging (Reference §23.x extra).

Ordinary kriging assumes a constant mean.  Universal kriging models

    Z(s) = X(s) * beta + delta(s)

where X(s) is a p-column drift design at s and delta(s) is a zero-mean
stationary residual with known variogram.

Prediction at s0 solves the augmented kriging system:

    [ Sigma  X   ] [ lambda ]   [ sigma0 ]
    [ X^T    0   ] [ mu     ] = [ x0     ]

    Z_hat(s0) = lambda^T Z,   variance = sigma^2 - lambda^T sigma0 - mu^T x0

where Sigma_ij = gamma-cov of observations, sigma0_i = gamma-cov to s0,
and x0 is the drift vector at s0.

External-drift kriging = universal kriging with a single external covariate.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math

import numpy as np    # numerical arrays + linear algebra


def _exp_cov(dist, sill: float = 1.0, range_: float = 1.0, nugget: float = 0.0):
    """Exponential COVARIANCE: sill * exp(-h / range) + nugget on diagonal."""
    C = sill * np.exp(-dist / range_)
    if C.ndim == 2:
        np.fill_diagonal(C, C.diagonal() + nugget)
    return C


def universal_kriging(coords_obs, Z_obs, drift_obs,
                      coords_pred, drift_pred,
                      sill: float = 1.0, range_: float = 1.0,
                      nugget: float = 0.01) -> dict:
    coords_obs = np.asarray(coords_obs, dtype=float)
    Z = np.asarray(Z_obs, dtype=float)
    X = np.asarray(drift_obs, dtype=float)
    coords_pred = np.asarray(coords_pred, dtype=float)
    Xp = np.asarray(drift_pred, dtype=float)
    n, p = X.shape; m = len(coords_pred)

    # observation covariance
    D_oo = np.sqrt(((coords_obs[:, None] - coords_obs[None, :]) ** 2).sum(-1))
    Sigma = _exp_cov(D_oo, sill, range_, nugget)

    # augmented LHS
    K = np.zeros((n + p, n + p))
    K[:n, :n] = Sigma
    K[:n, n:] = X; K[n:, :n] = X.T
    K_inv = np.linalg.inv(K + 1e-8 * np.eye(n + p))

    # per-prediction solve
    preds = np.zeros(m); vars_ = np.zeros(m)
    for k in range(m):
        d0 = np.sqrt(((coords_obs - coords_pred[k]) ** 2).sum(-1))
        sigma0 = _exp_cov(d0, sill, range_, 0.0)
        rhs = np.concatenate([sigma0, Xp[k]])
        sol = K_inv @ rhs
        lam = sol[:n]; mu = sol[n:]
        preds[k] = float(lam @ Z)
        vars_[k] = float(sill - lam @ sigma0 - mu @ Xp[k])
    return {"pred": preds, "var": vars_,
            "method": "universal kriging (exponential covariance)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # true field with linear north-south trend + Gaussian residual
    n = 60
    coords = rng.uniform(0, 10, (n, 2))
    beta_true = np.array([1.0, 0.5, -0.3])                # intercept + x-trend + y-trend
    Xd = np.column_stack([np.ones(n), coords[:, 0], coords[:, 1]])
    trend = Xd @ beta_true
    # residual: sample from N(0, Sigma) via Cholesky
    D = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(-1))
    Sigma = _exp_cov(D, sill=1.0, range_=2.0, nugget=0.01)
    L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(n))
    delta = L @ rng.normal(size=n)
    Z = trend + delta + rng.normal(scale=0.2, size=n)

    # predict on a grid
    gx = np.linspace(0.5, 9.5, 8); gy = np.linspace(0.5, 9.5, 8)
    coords_pred = np.array([(x, y) for x in gx for y in gy])
    Xp = np.column_stack([np.ones(len(coords_pred)),
                          coords_pred[:, 0], coords_pred[:, 1]])
    fit = universal_kriging(coords, Z, Xd, coords_pred, Xp,
                             sill=1.0, range_=2.0, nugget=0.01)

    # accuracy: LOO RMSE
    loo_err = []
    for i in range(n):
        idx = np.arange(n) != i
        f = universal_kriging(coords[idx], Z[idx], Xd[idx],
                              coords[i:i+1], Xd[i:i+1],
                              sill=1.0, range_=2.0, nugget=0.01)
        loo_err.append(float(Z[i] - f["pred"][0]))
    loo_err = np.array(loo_err)

    print(f"=== Universal kriging (drift = 1, x, y; exp covariance) ===")
    print(f"  n obs = {n},  n predictions = {len(coords_pred)}")
    print(f"  LOO RMSE = {math.sqrt((loo_err**2).mean()):.4f}   "
          f"(noise sd ~ 0.2, residual sill 1)")
    print(f"  mean prediction variance = {fit['var'].mean():.4f}")
    print(f"  min / max predicted Z = {fit['pred'].min():.3f} / {fit['pred'].max():.3f}")
    print(f"  min / max observed  Z = {Z.min():.3f} / {Z.max():.3f}")

    print("\n--- library cross-check (R gstat::krige with formula ~ x + y) ---")
