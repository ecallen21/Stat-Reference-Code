"""Inverse Distance Weighting (Shepard 1968) - Reference §23.8.

Weighted average of neighbouring values, weight inversely proportional to
distance:

    Z_hat(x_0) = sum_i (Z_i / d_i^p) / sum_i (1 / d_i^p)

    p : power parameter (typically 2)
    Optional: restrict to k-nearest neighbours only.

If a query point coincides with a sample, return the sample value exactly
(avoid divide-by-zero).

Simpler and faster than kriging but has no principled uncertainty
estimate and is sensitive to sampling density.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def idw(coords, values, x_new, power: float = 2.0, k: int = None) -> np.ndarray:
    coords = np.asarray(coords, dtype=float); values = np.asarray(values, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    if x_new.ndim == 1: x_new = x_new.reshape(1, -1)
    preds = np.zeros(len(x_new))
    for m in range(len(x_new)):
        d = np.sqrt(((coords - x_new[m]) ** 2).sum(-1))
        if (d < 1e-10).any():
            preds[m] = values[np.argmin(d)]; continue
        if k is not None:
            idx = np.argsort(d)[:k]
            d = d[idx]; vals = values[idx]
        else:
            vals = values
        w = 1 / d ** power
        preds[m] = float((w * vals).sum() / w.sum())
    return preds


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 80
    coords = rng.uniform(0, 10, size=(n, 2))
    z_true = np.sin(coords[:, 0]) + np.cos(coords[:, 1])
    values = z_true + rng.normal(0, 0.3, n)

    xg, yg = np.meshgrid(np.linspace(0, 10, 5), np.linspace(0, 10, 5))
    x_new = np.column_stack([xg.ravel(), yg.ravel()])
    truth_grid = np.sin(x_new[:, 0]) + np.cos(x_new[:, 1])

    for p in (1, 2, 4):
        pred = idw(coords, values, x_new, power=p)
        rmse = math.sqrt(np.mean((pred - truth_grid) ** 2))
        print(f"=== IDW power = {p} ===  RMSE = {rmse:.3f}")

    print("\n=== IDW with k = 5 nearest ===")
    pred = idw(coords, values, x_new, power=2, k=5)
    rmse = math.sqrt(np.mean((pred - truth_grid) ** 2))
    print(f"  RMSE = {rmse:.3f}")

    print("\n--- library cross-check (R gstat::krige with model = NULL for IDW) ---")
