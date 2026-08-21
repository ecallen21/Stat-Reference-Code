"""Ordinary kriging spatial interpolation (Reference §23.7).

Best Linear Unbiased Predictor (BLUP) of Z(x_0) as a weighted sum:
    Z_hat(x_0) = sum_i lambda_i Z(x_i)
    subject to  sum_i lambda_i = 1  (unbiasedness)

Weights minimize prediction variance -> solve linear system:
    | Gamma  1 | | lambda |     | gamma_0 |
    | 1'     0 | |  mu    |  =  |   1     |

where Gamma[i, j] = gamma(||x_i - x_j||), the variogram model.

Kriging variance at x_0:
    sigma_K^2(x_0) = lambda' gamma_0 + mu

Requires a fitted variogram (see 'variogram-modeling').
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _exp_variogram(h, nugget, sill, range_):
    return nugget + (sill - nugget) * (1 - np.exp(-3 * h / range_))


def ordinary_kriging(coords, values, x_new, variogram_params: dict) -> dict:
    """Ordinary kriging predictor + variance at grid of new points x_new.

    variogram_params = {"nugget": ..., "sill": ..., "range": ..., "model": "exponential"}
    """
    coords = np.asarray(coords, dtype=float); values = np.asarray(values, dtype=float)
    x_new = np.asarray(x_new, dtype=float); n = len(values); n_new = len(x_new)
    nugget = variogram_params["nugget"]; sill = variogram_params["sill"]; rng = variogram_params["range"]
    if variogram_params.get("model", "exponential") != "exponential":
        raise NotImplementedError("only exponential in this demo")
    d = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(-1))
    Gamma = _exp_variogram(d, nugget, sill, rng)
    # Augment with the unbiasedness constraint
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = Gamma; K[n, :n] = 1; K[:n, n] = 1; K[n, n] = 0
    preds = np.zeros(n_new); vars_ = np.zeros(n_new)
    for m in range(n_new):
        d0 = np.sqrt(((coords - x_new[m]) ** 2).sum(-1))
        gamma0 = _exp_variogram(d0, nugget, sill, rng)
        b = np.concatenate([gamma0, [1.0]])
        try:
            sol = np.linalg.solve(K + 1e-10 * np.eye(n + 1), b)
        except np.linalg.LinAlgError:
            preds[m] = np.nan; vars_[m] = np.nan; continue
        lambdas = sol[:n]; mu = sol[n]
        preds[m] = float(lambdas @ values)
        vars_[m] = float(lambdas @ gamma0 + mu)
    return {"prediction": preds, "variance": vars_,
            "std_error": np.sqrt(np.maximum(vars_, 0)),
            "method": "Ordinary kriging (exponential variogram)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 80
    coords = rng.uniform(0, 10, size=(n, 2))
    # True field: smooth trend + noise
    z_true = np.sin(coords[:, 0]) + np.cos(coords[:, 1])
    values = z_true + rng.normal(0, 0.3, n)

    # Simple variogram: fit an exponential (using known-ish parameters here)
    fit_params = {"nugget": 0.1, "sill": 1.0, "range": 3.0, "model": "exponential"}

    # Interpolate on a small grid
    xg, yg = np.meshgrid(np.linspace(0, 10, 5), np.linspace(0, 10, 5))
    x_new = np.column_stack([xg.ravel(), yg.ravel()])
    r = ordinary_kriging(coords, values, x_new, fit_params)
    z_true_grid = np.sin(x_new[:, 0]) + np.cos(x_new[:, 1])

    print("=== Ordinary kriging on 5x5 grid ===")
    print("  x   y     pred    truth   +/- SE")
    for k in range(0, 25, 5):
        print(f"  {x_new[k, 0]:.1f} {x_new[k, 1]:.1f}   {r['prediction'][k]:6.3f}   {z_true_grid[k]:6.3f}   {r['std_error'][k]:.3f}")

    rmse = math.sqrt(np.mean((r["prediction"] - z_true_grid) ** 2))
    print(f"\n  RMSE(pred vs truth on grid) = {rmse:.3f}")
    print("\n--- library cross-check (R gstat::krige) ---")
