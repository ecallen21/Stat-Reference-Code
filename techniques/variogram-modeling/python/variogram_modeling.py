"""Empirical and theoretical variograms (Reference §23.6).

Semivariogram
    gamma(h) = 0.5 * E[(Z(x + h) - Z(x))^2]

Estimated by binning pairs by distance:
    gamma_hat(h_bin) = 1 / (2 |N(h_bin)|) sum_{(i, j) in N(h_bin)} (Z_i - Z_j)^2

Fit a theoretical model with parameters (nugget, sill, range):

    Spherical:  gamma(h) = n + (s - n) [1.5 (h/r) - 0.5 (h/r)^3]     for h <= r,
                           = s                                          for h > r
    Exponential: gamma(h) = n + (s - n) (1 - exp(-3 h / r))
    Gaussian:    gamma(h) = n + (s - n) (1 - exp(-3 (h/r)^2))

Used as input to KRIGING.  The fitted variogram governs spatial covariance
structure.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def empirical_variogram(coords, values, n_bins: int = 10, max_dist: float = None) -> dict:
    coords = np.asarray(coords, dtype=float); values = np.asarray(values, dtype=float)
    n = len(values)
    d = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(-1))
    dv = 0.5 * (values[:, None] - values[None, :]) ** 2
    iu = np.triu_indices(n, k=1)
    d_all = d[iu]; dv_all = dv[iu]
    if max_dist is None:
        max_dist = float(np.percentile(d_all, 60))     # avoid noisy tail
    edges = np.linspace(0, max_dist, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    gamma = []
    for k in range(n_bins):
        m = (d_all >= edges[k]) & (d_all < edges[k + 1])
        gamma.append(float(dv_all[m].mean()) if m.any() else float("nan"))
    return {"h": centers, "gamma": np.array(gamma), "n_bins": int(n_bins)}


def _sph(h, n, s, r):
    g = np.where(h < r, n + (s - n) * (1.5 * h / r - 0.5 * (h / r) ** 3), s)
    return g


def _exp(h, n, s, r):
    return n + (s - n) * (1 - np.exp(-3 * h / r))


def _gau(h, n, s, r):
    return n + (s - n) * (1 - np.exp(-3 * (h / r) ** 2))


def fit_variogram(h, gamma_hat, model: str = "spherical") -> dict:
    """WLS fit of nugget, sill, range."""
    h = np.asarray(h); g = np.asarray(gamma_hat)
    valid = ~np.isnan(g); h = h[valid]; g = g[valid]
    model_fn = {"spherical": _sph, "exponential": _exp, "gaussian": _gau}[model]
    def loss(p):
        n, s, r = p[0], p[1], p[2]
        if n < 0 or s <= n or r <= 0: return 1e10
        return float(np.sum((model_fn(h, n, s, r) - g) ** 2))
    p0 = [0.0, float(g.max()), float(h.max() / 2)]
    res = minimize(loss, p0, method="Nelder-Mead")
    n, s, r = res.x
    return {"nugget": float(max(n, 0)),
            "sill": float(s), "range": float(r),
            "model": model,
            "fitted": model_fn(h, n, s, r),
            "residual_ss": float(res.fun),
            "method": f"Variogram fit ({model})"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 400
    coords = rng.uniform(0, 10, (n, 2))
    # Simulate Gaussian random field with exponential covariance
    d = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(-1))
    cov = 2.0 * np.exp(-d / 2.0) + 0.5 * np.eye(n)
    L = np.linalg.cholesky(cov + 1e-6 * np.eye(n))
    values = L @ rng.normal(size=n)

    ev = empirical_variogram(coords, values, n_bins=12, max_dist=5)
    print("=== Empirical variogram ===")
    for h_v, g_v in zip(ev["h"], ev["gamma"]):
        print(f"  h = {h_v:.3f}   gamma_hat = {g_v:.3f}")

    for model in ("exponential", "spherical", "gaussian"):
        fit = fit_variogram(ev["h"], ev["gamma"], model=model)
        print(f"\n=== {model} fit ===")
        print(f"  nugget = {fit['nugget']:.3f}, sill = {fit['sill']:.3f}, range = {fit['range']:.3f}")
        print(f"  residual SS = {fit['residual_ss']:.4f}")

    print("\n--- library cross-check (R gstat::variogram / fit.variogram) ---")
