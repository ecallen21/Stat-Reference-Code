"""Explainable boosting machine (EBM) (Reference Sec 47.31).

Nori, Jenkins, Koch & Caruana 2019 'InterpretML: A unified framework
for machine learning interpretability'; Lou et al. 2013 'Accurate
intelligible models with pairwise interactions'. An INHERENTLY
INTERPRETABLE gradient-boosted GAM:

    f(x) = beta_0 + sum_j f_j(x_j) + sum_{j<k} f_{jk}(x_j, x_k)

with each f_j fit by a tiny bag of decision trees, ROUND-ROBIN over
features. The per-feature shape function f_j(x_j) can be plotted
directly.

We implement a compact EBM baseline: cycle through features, fit a
small tree stump to residuals, add to that feature's shape function.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def stump_fit(x, resid, min_leaf=5, n_bins=32):
    """Fit best 1-D split on x -> mean-residual per side."""
    order = np.argsort(x)
    xs = x[order]; rs = resid[order]
    n = len(x)
    best = (None, 0.0, 0.0, np.inf)
    for k in range(min_leaf, n - min_leaf):
        left = rs[:k]; right = rs[k:]
        loss = float(np.sum((left - left.mean()) ** 2) +
                     np.sum((right - right.mean()) ** 2))
        if loss < best[3]:
            best = (float(xs[k]), float(left.mean()), float(right.mean()), loss)
    cut, mu_l, mu_r, _ = best
    def shape(z):
        return np.where(z <= cut, mu_l, mu_r)
    return shape, {"cut": cut, "mu_l": mu_l, "mu_r": mu_r}


def ebm_fit(X, y, rounds=200, lr=0.1, seed=0):
    n, p = X.shape
    residual = y - y.mean()
    shape_fns = [[] for _ in range(p)]         # list of stump shapes per feature
    for it in range(rounds):
        j = it % p                              # round-robin over features
        shape, _ = stump_fit(X[:, j], residual)
        contrib = lr * shape(X[:, j])
        residual = residual - contrib
        shape_fns[j].append((shape, lr))
    return {"intercept": float(y.mean()), "shape_fns": shape_fns}


def ebm_predict(model, X):
    p = X.shape[1]
    yhat = np.full(len(X), model["intercept"])
    for j in range(p):
        for shape, lr in model["shape_fns"][j]:
            yhat = yhat + lr * shape(X[:, j])
    return yhat


def ebm_shape(model, j, grid):
    val = np.zeros_like(grid, dtype=float)
    for shape, lr in model["shape_fns"][j]:
        val = val + lr * shape(grid)
    return val


if __name__ == "__main__":
    print("=== Explainable boosting machine (Nori 2019 / Lou 2013) ===\n")
    rng = np.random.default_rng(0)
    n = 800
    X = rng.normal(size=(n, 4))
    #  True additive f: 1.5 * tanh(x0) + 0.5 * x1^2 + 0.3 * x2 (x3 irrelevant)
    y = 1.5 * np.tanh(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.3 * X[:, 2] + 0.3 * rng.normal(size=n)

    model = ebm_fit(X, y, rounds=200, lr=0.1)
    yhat = ebm_predict(model, X)
    mse = float(np.mean((yhat - y) ** 2))
    r2 = 1 - mse / float(y.var())
    print(f"  n = {n}, p = 4   In-sample MSE = {mse:.4f}   R^2 = {r2:.3f}")

    #  Inspect shape function for x0 (should look like tanh)
    grid = np.linspace(-2, 2, 9)
    s0 = ebm_shape(model, 0, grid)
    print(f"\n  Shape f_0(x0) at grid points {grid.round(2).tolist()}:")
    print(f"    EBM  f_0    = {s0.round(3).tolist()}")
    print(f"    Truth tanh  = {(1.5 * np.tanh(grid)).round(3).tolist()}")

    s3 = ebm_shape(model, 3, grid)
    print(f"\n  Shape f_3(x3) (irrelevant): max |val| = {float(np.max(np.abs(s3))):.3f}   "
          f"(should be small)")

    print("\n--- library cross-check (interpret / interpret-community R/Python) ---")
