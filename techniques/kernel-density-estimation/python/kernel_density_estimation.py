"""Kernel Density Estimation (Reference §6.21).

Smooth, nonparametric estimate of an unknown probability density:

    f_hat(x) = (1 / (n h)) * sum_i K( (x - x_i) / h )

where K is a kernel (symmetric, integrates to 1) and h > 0 is the bandwidth.

This file
---------
- Standard kernels: gaussian, epanechnikov, uniform, triangular, cosine.
- Bandwidth rules:
    Silverman (rule of thumb):
        h = 0.9 * min(sd, IQR/1.34) * n^(-1/5)
    Scott:
        h = sd * n^(-1/5)
- Evaluate on a user-supplied grid; integrate the density (should be ~ 1).
- ISE bias-variance tradeoff: small h -> bumpy / high variance; large h -> over-smooth.

For multivariate KDE see ``scipy.stats.gaussian_kde``; here we focus on the
univariate case.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def _gauss(u): return np.exp(-0.5 * u ** 2) / math.sqrt(2 * math.pi)
def _epan(u): return np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0.0)
def _unif(u): return np.where(np.abs(u) <= 1, 0.5, 0.0)
def _tri(u): return np.where(np.abs(u) <= 1, 1 - np.abs(u), 0.0)
def _cos(u): return np.where(np.abs(u) <= 1, (math.pi / 4) * np.cos(math.pi * u / 2), 0.0)

KERNELS = {"gaussian": _gauss, "epanechnikov": _epan, "uniform": _unif,
           "triangular": _tri, "cosine": _cos}


def silverman_h(x):
    x = np.asarray(x, dtype=float); n = x.size
    sd = x.std(ddof=1); q1, q3 = np.quantile(x, [0.25, 0.75])
    return 0.9 * min(sd, (q3 - q1) / 1.34) * n ** (-0.2)


def scott_h(x):
    x = np.asarray(x, dtype=float); return x.std(ddof=1) * x.size ** (-0.2)


def kde_evaluate(x, grid, h=None, kernel: str = "gaussian"):
    """Evaluate f_hat at each point in ``grid``."""
    x = np.asarray(x, dtype=float); grid = np.asarray(grid, dtype=float)
    if h is None:
        h = silverman_h(x)
    K = KERNELS[kernel]
    out = np.zeros_like(grid, dtype=float)
    for xi in x:
        out += K((grid - xi) / h)
    out /= (len(x) * h)
    return out, h


def library_versions(x, grid):
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(x)
    return {"scipy gaussian_kde at grid[:5]": kde(grid[:5]).tolist(),
            "scipy bandwidth_method default factor": float(kde.factor)}


if __name__ == "__main__":
    rng = np.random.default_rng(8)
    # Bimodal mixture
    x = np.concatenate([rng.normal(-2, 0.8, 200), rng.normal(2, 1.0, 300)])
    grid = np.linspace(-6, 6, 200)

    print("=== Silverman / Scott bandwidths ===")
    print(f"  Silverman h = {silverman_h(x):.4f}")
    print(f"  Scott     h = {scott_h(x):.4f}")

    for kernel in ("gaussian", "epanechnikov", "triangular"):
        f, h = kde_evaluate(x, grid, kernel=kernel)
        area = float(np.trapezoid(f, grid))
        peak = float(grid[int(np.argmax(f))])
        print(f"  {kernel:14s}: peak at x = {peak:.2f}, integral = {area:.4f}, h = {h:.4f}")

    print("\n--- library (scipy gaussian_kde) ---")
    for k, v in library_versions(x, grid).items():
        print(f"  {k}: {v}")
