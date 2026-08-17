"""CART decision tree (Reference §26.6; Breiman-Friedman-Olshen-Stone 1984).

Recursive binary partitioning: at each node choose (feature, threshold)
that best splits the data by an impurity criterion:

    Regression: variance reduction  DeltaI = Var(parent) - (nL/n) Var(L) - (nR/n) Var(R)
    Classification: Gini            Gini = 1 - sum_c p_c^2
                     Entropy        H = - sum_c p_c log p_c

Recurse until stopping condition:
    - max_depth reached
    - min_samples in node
    - improvement below threshold

Prediction: leaf mean (regression) or majority class (classification).

Overfitting: unpruned trees overfit; use MAX_DEPTH or MIN_SAMPLES_LEAF
to control complexity, or grow-then-prune with cost-complexity pruning.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")
    def __init__(self, value=None):
        self.feature = None; self.threshold = None
        self.left = None; self.right = None; self.value = value


def _gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum(); return 1 - np.sum(p ** 2)


def _mse_impurity(y):
    return float(np.var(y)) if len(y) > 0 else 0.0


def _best_split(X, y, impurity, min_samples_leaf: int):
    n, p = X.shape
    best_gain = 0.0; best = None
    parent_i = impurity(y)
    for j in range(p):
        vals = np.unique(X[:, j])
        for t in vals[:-1]:                              # threshold between distinct values
            L = X[:, j] <= t; R = ~L
            if L.sum() < min_samples_leaf or R.sum() < min_samples_leaf: continue
            gain = parent_i - (L.sum() * impurity(y[L]) + R.sum() * impurity(y[R])) / n
            if gain > best_gain:
                best_gain = gain; best = (j, float(t))
    return best, best_gain


def build_tree(X, y, impurity, max_depth: int, min_samples_leaf: int, depth: int = 0):
    node = Node()
    if depth >= max_depth or len(y) < 2 * min_samples_leaf:
        node.value = _leaf_value(y, impurity); return node
    split, gain = _best_split(X, y, impurity, min_samples_leaf)
    if split is None or gain < 1e-7:
        node.value = _leaf_value(y, impurity); return node
    j, t = split
    node.feature = j; node.threshold = t
    L = X[:, j] <= t
    node.left = build_tree(X[L], y[L], impurity, max_depth, min_samples_leaf, depth + 1)
    node.right = build_tree(X[~L], y[~L], impurity, max_depth, min_samples_leaf, depth + 1)
    return node


def _leaf_value(y, impurity):
    if impurity is _gini:
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]
    return float(np.mean(y))


def predict_one(node, x):
    while node.feature is not None:
        node = node.left if x[node.feature] <= node.threshold else node.right
    return node.value


def predict(node, X):
    return np.array([predict_one(node, xi) for xi in X])


def fit_tree(X, y, task: str = "regression", max_depth: int = 5, min_samples_leaf: int = 5) -> dict:
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    impurity = _mse_impurity if task == "regression" else _gini
    root = build_tree(X, y, impurity, max_depth, min_samples_leaf)
    return {"root": root, "task": task,
            "predict": lambda X_new: predict(root, np.asarray(X_new, dtype=float)),
            "method": f"CART decision tree ({task}, max_depth={max_depth})"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Regression tree on nonlinear target ===")
    X = rng.uniform(-3, 3, size=(400, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0, 0.3, 400)
    fit = fit_tree(X, y, task="regression", max_depth=6, min_samples_leaf=5)
    y_hat = fit["predict"](X)
    print(f"  in-sample RMSE = {math.sqrt(np.mean((y - y_hat) ** 2)):.3f}")

    print("\n=== Classification tree on synthetic 3-class ===")
    X = np.vstack([rng.normal([0, 0], 1, (100, 2)),
                    rng.normal([4, 0], 1, (100, 2)),
                    rng.normal([2, 4], 1, (100, 2))])
    y = np.repeat([0, 1, 2], 100)
    fit = fit_tree(X, y, task="classification", max_depth=5)
    yhat = fit["predict"](X)
    print(f"  training accuracy = {(yhat == y).mean():.3f}")

    print("\n--- library cross-check (sklearn DecisionTreeClassifier) ---")
    try:
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X, y)
        print(f"  sklearn training accuracy = {clf.score(X, y):.3f}")
    except Exception as ex:
        print(f"  (sklearn unavailable: {ex})")
