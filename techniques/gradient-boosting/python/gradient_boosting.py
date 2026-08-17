"""Gradient Boosting Machine (Reference §26.8; Friedman 2001).

Sequential additive model:
    F_0(x) = mean(y)                        initial constant
    for m = 1, ..., M:
        r_i = -grad Loss(y_i, F_{m-1}(x_i))     (pseudo-residuals; for L2 loss = y - F)
        fit shallow tree h_m to (X, r)
        F_m = F_{m-1} + nu * h_m(x)              (nu = learning rate)

Trees are typically SHALLOW (depth 3-6) to keep bias moderate; the ensemble
reduces bias through many additive updates while shrinkage nu (0.01-0.1)
regularizes.

Loss functions
    Regression: L2 (squared error), L1 (Huber for robustness)
    Classification: log-loss / binary deviance

Modern flavors: XGBoost, LightGBM, CatBoost - add second-order info (Newton),
histogram binning, sparsity-aware splits.

The demo below implements simple L2 gradient boosting on regression.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "decision-tree", "python"))
from decision_tree import build_tree, predict as tree_predict, _mse_impurity  # noqa: E402


def gbm_regression(X, y, n_trees: int = 100, learning_rate: float = 0.1,
                    max_depth: int = 3, min_samples_leaf: int = 5) -> dict:
    """L2 gradient boosting for regression."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    F = np.full_like(y, float(y.mean()))
    trees = []
    for m in range(n_trees):
        r = y - F                                          # negative gradient for L2 loss
        tree = build_tree(X, r, _mse_impurity, max_depth, min_samples_leaf)
        update = tree_predict(tree, X)
        F = F + learning_rate * update
        trees.append(tree)
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=float)
        F0 = np.full(len(X_new), float(y.mean()))
        for t in trees:
            F0 = F0 + learning_rate * tree_predict(t, X_new)
        return F0
    return {"predict": predict, "trees": trees,
            "learning_rate": learning_rate, "n_trees": int(n_trees),
            "max_depth": int(max_depth),
            "method": "L2 gradient boosting (Friedman 2001)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 400
    X = rng.uniform(-3, 3, size=(n, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0, 0.3, n)

    print("=== L2 gradient boosting (100 trees, depth 3, lr 0.1) ===")
    fit = gbm_regression(X, y, n_trees=100, learning_rate=0.1, max_depth=3)
    y_hat = fit["predict"](X)
    print(f"  in-sample RMSE = {math.sqrt(np.mean((y - y_hat) ** 2)):.3f}   (noise sd 0.3)")

    # Test on held-out
    Xte = rng.uniform(-3, 3, size=(200, 2))
    yte = np.sin(Xte[:, 0]) + 0.5 * Xte[:, 1] + rng.normal(0, 0.3, 200)
    print(f"  test RMSE     = {math.sqrt(np.mean((fit['predict'](Xte) - yte) ** 2)):.3f}")

    print("\n--- library cross-check (sklearn GradientBoostingRegressor) ---")
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        m = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
        m.fit(X, y)
        print(f"  sklearn test RMSE = {math.sqrt(np.mean((m.predict(Xte) - yte) ** 2)):.3f}")
    except Exception as ex:
        print(f"  (sklearn unavailable: {ex})")
