"""BART -- Bayesian additive regression trees (Reference Sec 5.16).

Chipman, George & McCulloch 2010. Sum-of-trees regression model with
prior regularisation:

    y = sum_{k=1}^m g(x; T_k, M_k) + eps      eps ~ N(0, sigma^2)

with:
    * m trees (typically 50-200).
    * Regularisation prior shrinks each tree toward a WEAK LEARNER.
    * Backfitting MCMC updates each (T_k, M_k) in turn: propose
      grow / prune / change / swap on the tree structure, then draw
      leaf means from Normal.

Posterior gives PPI / credible intervals for f(x) and can identify
important predictors via variable-selection frequencies.

We implement a MINIMAL 'backfitting stumps' surrogate: m stumps
sequentially fit residuals, then re-fit round-robin (like an EBM
without the deep MCMC + prior).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def best_stump(X, resid, min_leaf=5):
    n, p = X.shape
    best = (0, 0.0, 0.0, 0.0, np.inf)   # (feature, cut, mu_l, mu_r, sse)
    for j in range(p):
        order = np.argsort(X[:, j])
        xs = X[order, j]; rs = resid[order]
        for k in range(min_leaf, n - min_leaf):
            mu_l = rs[:k].mean(); mu_r = rs[k:].mean()
            sse = float(np.sum((rs[:k] - mu_l) ** 2) + np.sum((rs[k:] - mu_r) ** 2))
            if sse < best[4]:
                best = (j, float(xs[k]), mu_l, mu_r, sse)
    return best


def stump_predict(stump, X):
    j, c, mu_l, mu_r, _ = stump
    return np.where(X[:, j] <= c, mu_l, mu_r)


def bart_stumps(X, y, m=30, sweeps=15, sigma_shrink=1.0, seed=0):
    """Minimal backfitting: m stumps each fit on residuals, round-robin."""
    rng = np.random.default_rng(seed)
    n = len(y)
    y_mean = y.mean()
    stumps = [(0, 0.0, 0.0, 0.0, np.inf) for _ in range(m)]  # placeholder
    trees_pred = np.zeros((m, n))
    for sw in range(sweeps):
        for k in range(m):
            #  Residuals excluding this stump
            resid = y - y_mean - trees_pred.sum(axis=0) + trees_pred[k]
            #  Shrink residuals toward 0 by 1/(sigma_shrink+1) as a prior proxy
            resid = resid / (1 + sigma_shrink / m)
            new_stump = best_stump(X, resid)
            stumps[k] = new_stump
            trees_pred[k] = stump_predict(new_stump, X)
    return {"y_mean": y_mean, "stumps": stumps}


def bart_predict(model, X):
    y = np.full(len(X), model["y_mean"])
    for stump in model["stumps"]:
        y = y + stump_predict(stump, X)
    return y


if __name__ == "__main__":
    print("=== BART (Bayesian additive regression trees) -- stumps surrogate ===\n")
    rng = np.random.default_rng(0)
    n = 500; p = 3
    X = rng.normal(size=(n, p))
    #  y = sin(pi x0) + 0.5 x1^2 + noise
    y = np.sin(np.pi * X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.4 * rng.normal(size=n)

    model = bart_stumps(X, y, m=40, sweeps=10, sigma_shrink=1.0)
    y_hat = bart_predict(model, X)
    mse = float(np.mean((y_hat - y) ** 2))
    r2 = 1 - mse / float(y.var())
    print(f"  n={n}, p={p}   m=40 stumps, 10 sweeps")
    print(f"  In-sample MSE = {mse:.3f}   R^2 = {r2:.3f}")
    print(f"\n  BART's Bayesian layer (this demo lacks) gives credible intervals AND")
    print(f"  variable-inclusion frequencies; use BART / dbarts in R for real work.")

    print("\n--- library cross-check (BART / dbarts / bartMachine R; pymc-bart Python) ---")
