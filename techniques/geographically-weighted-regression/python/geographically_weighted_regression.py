"""Geographically Weighted Regression (Reference §23.11; Brunsdon et al. 1996).

Local weighted linear regression, run at each observation location:

    beta_i = (X^T W_i X)^-1 X^T W_i y

W_i is a diagonal matrix of kernel weights depending on distance from
location i (Gaussian or bisquare).  Bandwidth chosen by CV (leave-one-out).

Output: n x p coefficient surface (one beta_hat per location).

Contrast with global OLS: GWR relaxes the assumption of homogeneous
relationship across space.  Useful for exploring spatial NON-STATIONARITY.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _gaussian_weights(d, h):
    return np.exp(-0.5 * (d / h) ** 2)


def gwr_fit(coords, X, y, bandwidth: float, kernel: str = "gaussian") -> dict:
    coords = np.asarray(coords, dtype=float); X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta_local = np.zeros((n, p))
    y_hat = np.zeros(n)
    for i in range(n):
        d = np.sqrt(((coords - coords[i]) ** 2).sum(-1))
        w = _gaussian_weights(d, bandwidth)
        WX = X * w[:, None]
        beta = np.linalg.solve(X.T @ WX + 1e-8 * np.eye(p), X.T @ (w * y))
        beta_local[i] = beta
        y_hat[i] = float(X[i] @ beta)
    resid = y - y_hat
    return {"beta_local": beta_local, "y_hat": y_hat,
            "rss": float(resid @ resid),
            "bandwidth": bandwidth,
            "method": "Geographically Weighted Regression (Gaussian kernel)"}


def gwr_cv_bandwidth(coords, X, y, grid=None) -> dict:
    """Leave-one-out CV over a grid of bandwidths."""
    coords = np.asarray(coords, dtype=float); n = len(y)
    if grid is None:
        d = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(-1))
        grid = np.linspace(d[d > 0].min() * 2, d.max() / 2, 8)
    scores = []
    for h in grid:
        loo = 0.0
        for i in range(n):
            d = np.sqrt(((coords - coords[i]) ** 2).sum(-1))
            w = _gaussian_weights(d, h); w[i] = 0                # LOO
            WX = X * w[:, None]
            try:
                beta = np.linalg.solve(X.T @ WX + 1e-8 * np.eye(X.shape[1]), X.T @ (w * y))
                loo += (y[i] - X[i] @ beta) ** 2
            except np.linalg.LinAlgError:
                loo += 1e10
        scores.append(loo)
    best = int(np.argmin(scores))
    return {"grid": grid, "scores": scores, "best_h": float(grid[best])}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    coords = rng.uniform(0, 10, (n, 2))
    # True beta varies with location: beta_x = 1 + coords[:, 0] / 5
    x = rng.normal(size=n); X = np.column_stack([np.ones(n), x])
    beta_true_x = 1 + coords[:, 0] / 5
    y = 2 + beta_true_x * x + rng.normal(0, 0.5, n)

    cv = gwr_cv_bandwidth(coords, X, y)
    print(f"=== GWR CV bandwidth: {cv['best_h']:.3f} ===")
    fit = gwr_fit(coords, X, y, bandwidth=cv["best_h"])
    print(f"  in-sample RMSE = {math.sqrt(fit['rss'] / n):.3f}   (noise sd 0.5)")
    print(f"\n  Fitted beta_x by coords[:, 0] quartile:")
    for q in (0.25, 0.5, 0.75):
        cutoff = np.quantile(coords[:, 0], q)
        mask = np.abs(coords[:, 0] - cutoff) < 0.5
        print(f"    coords[0] ~ {cutoff:.2f}: mean beta_x = {fit['beta_local'][mask, 1].mean():.3f}   true = {1 + cutoff / 5:.3f}")

    print("\n--- library cross-check (R spgwr::gwr / GWmodel) ---")
