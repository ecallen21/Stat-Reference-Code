"""Feature importance: permutation + partial dependence + ICE
(Reference §26.16).

Permutation importance (Breiman 2001; Fisher-Rudin-Dominici 2019)
    Baseline score = model score on a validation set.
    For each feature j:
        shuffle column j (breaking its association with y),
        recompute the score.
    Drop in score = importance of j.  Repeat several shuffles for stability.

Advantages over MDI (mean decrease in impurity)
    - Model-agnostic.
    - Correctly handles correlated features (though still not perfect --
      permuting a correlated feature can create "impossible" data points).

Partial Dependence Plot (PDP)
    E_X_{-j}[ f_hat(x_j, X_{-j}) ] as a function of x_j.
    Global average effect of x_j on the prediction.

Individual Conditional Expectation (ICE) plot
    For each individual i, plot f_hat(x_j, x_{i, -j}) as a function of x_j.
    Reveals heterogeneous effects that PDP averages away.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def permutation_importance(model_predict, X, y, score_fn, n_repeats: int = 5,
                            seed: int = 0) -> dict:
    """Permutation importance for a fitted model.

    model_predict : callable X -> predictions.
    score_fn : callable (y_true, y_pred) -> score (higher = better).
    """
    X = np.asarray(X, dtype=float).copy(); y = np.asarray(y)
    rng = np.random.default_rng(seed)
    base_score = score_fn(y, model_predict(X))
    n, p = X.shape
    importances = np.zeros(p)
    stds = np.zeros(p)
    for j in range(p):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            drops.append(base_score - score_fn(y, model_predict(X_perm)))
        importances[j] = np.mean(drops); stds[j] = np.std(drops, ddof=1)
    return {"importances": importances, "stds": stds, "baseline_score": float(base_score),
            "method": "Permutation feature importance"}


def partial_dependence(model_predict, X, feature: int, grid: np.ndarray = None) -> dict:
    """Compute PDP for a single feature over a grid."""
    X = np.asarray(X, dtype=float)
    if grid is None:
        grid = np.linspace(np.min(X[:, feature]), np.max(X[:, feature]), 30)
    pdp = np.zeros(len(grid))
    for i, v in enumerate(grid):
        X_mod = X.copy(); X_mod[:, feature] = v
        pdp[i] = float(np.mean(model_predict(X_mod)))
    return {"grid": grid, "pdp": pdp, "feature": int(feature)}


def ice_lines(model_predict, X, feature: int, grid: np.ndarray = None,
              n_show: int = 30, seed: int = 0) -> dict:
    """Individual conditional expectation for a subset of observations."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_show, len(X)), replace=False)
    if grid is None:
        grid = np.linspace(np.min(X[:, feature]), np.max(X[:, feature]), 30)
    lines = np.zeros((len(idx), len(grid)))
    for i, ii in enumerate(idx):
        for k, v in enumerate(grid):
            x = X[ii].copy(); x[feature] = v
            lines[i, k] = float(model_predict(x.reshape(1, -1)))
    return {"grid": grid, "lines": lines, "indices": idx}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    X = rng.normal(size=(n, 5))
    # True model: only x_0 and x_1 matter; x_2 - x_4 are noise
    y = 2 * X[:, 0] + X[:, 1] * X[:, 0] + rng.normal(0, 0.3, n)

    # Fit a simple regression tree
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "decision-tree", "python"))
    from decision_tree import build_tree, predict as tree_predict, _mse_impurity
    root = build_tree(X, y, _mse_impurity, max_depth=8, min_samples_leaf=5)
    def model_predict(X_new): return tree_predict(root, np.asarray(X_new, dtype=float))

    print("=== Permutation importance (drop in R^2) ===")
    def r2(y_true, y_pred): return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    imp = permutation_importance(model_predict, X, y, r2, n_repeats=10)
    print("  feature  importance  std")
    for j in range(5):
        print(f"    x{j}      {imp['importances'][j]:.4f}     {imp['stds'][j]:.4f}")

    print("\n=== Partial-dependence of x0 (5-value preview) ===")
    pdp = partial_dependence(model_predict, X, feature=0)
    for i in (0, 6, 15, 22, 29):
        print(f"  x0 = {pdp['grid'][i]:6.3f}   PDP = {pdp['pdp'][i]:6.3f}")

    print("\n--- library cross-check (sklearn permutation_importance / partial_dependence) ---")
    print("  sklearn.inspection.permutation_importance / partial_dependence")
