"""Random Forest (Reference §26.7; Breiman 2001).

Ensemble of decision trees:
    1. Bootstrap sample the training data for each tree.
    2. At each split consider a RANDOM SUBSET of features (typically sqrt(p)
       for classification, p/3 for regression).
    3. Grow trees deep (no pruning).
    4. Aggregate: mean (regression) or majority vote / probability average
       (classification).

Bias-variance
    Individual deep trees are LOW BIAS but HIGH VARIANCE.  Averaging cancels
    variance -> forest has lower total error.  Feature subsetting further
    decorrelates trees, boosting the variance reduction.

Out-of-bag (OOB) error
    Each tree's bootstrap sample leaves ~1/3 of observations out.  Predict
    each observation with only the trees that didn't see it -> OOB error
    (a cheap CV substitute).

Feature importance
    Mean decrease in impurity averaged across trees, OR permutation
    importance (see feature-importance).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)

# Reuse the decision-tree building blocks
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "decision-tree", "python"))
from decision_tree import build_tree, predict as tree_predict, _gini, _mse_impurity   # noqa: E402


def _grow_tree_random_features(X, y, impurity, max_depth: int, min_samples_leaf: int,
                                mtry: int, rng):
    """Grow a CART tree where each split considers only a random subset of mtry features."""
    n, p = X.shape
    # Simple wrapper: mask features to a random subset at each split (approximation via passthrough)
    class RFNode:
        __slots__ = ("feature", "threshold", "left", "right", "value")
    def _best_split_random(X_sub, y_sub, feat_ids):
        best_gain = 0.0; best = None
        parent_i = impurity(y_sub); n_sub = len(y_sub)
        for j in feat_ids:
            vals = np.unique(X_sub[:, j])
            for t in vals[:-1]:
                L = X_sub[:, j] <= t; R = ~L
                if L.sum() < min_samples_leaf or R.sum() < min_samples_leaf: continue
                gain = parent_i - (L.sum() * impurity(y_sub[L]) + R.sum() * impurity(y_sub[R])) / n_sub
                if gain > best_gain:
                    best_gain = gain; best = (j, float(t))
        return best, best_gain
    def _build(X_sub, y_sub, depth):
        node = RFNode(); node.feature = None; node.left = None; node.right = None
        if depth >= max_depth or len(y_sub) < 2 * min_samples_leaf:
            node.value = _leaf(y_sub); return node
        feat_ids = rng.choice(p, size=min(mtry, p), replace=False)
        split, gain = _best_split_random(X_sub, y_sub, feat_ids)
        if split is None or gain < 1e-9:
            node.value = _leaf(y_sub); return node
        j, t = split
        node.feature = j; node.threshold = t
        L = X_sub[:, j] <= t
        node.left = _build(X_sub[L], y_sub[L], depth + 1)
        node.right = _build(X_sub[~L], y_sub[~L], depth + 1)
        return node
    def _leaf(y_sub):
        if impurity is _gini:
            vals, cnts = np.unique(y_sub, return_counts=True); return vals[np.argmax(cnts)]
        return float(np.mean(y_sub))
    return _build(X, y, 0)


def _rf_predict(root, X):
    def _pred_one(x):
        node = root
        while node.feature is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value
    return np.array([_pred_one(x) for x in X])


def random_forest(X, y, task: str = "regression", n_trees: int = 100,
                  max_depth: int = 12, min_samples_leaf: int = 3,
                  mtry: int = None, seed: int = 0) -> dict:
    X = np.asarray(X, dtype=float); y = np.asarray(y); n, p = X.shape
    rng = np.random.default_rng(seed)
    impurity = _gini if task == "classification" else _mse_impurity
    if mtry is None:
        mtry = int(round(math.sqrt(p))) if task == "classification" else max(1, p // 3)
    trees = []; oob_preds = [[] for _ in range(n)]
    for b in range(n_trees):
        boot = rng.integers(0, n, n)
        oob = np.setdiff1d(np.arange(n), np.unique(boot))
        root = _grow_tree_random_features(X[boot], y[boot], impurity, max_depth,
                                           min_samples_leaf, mtry, rng)
        trees.append(root)
        if len(oob) > 0:
            oob_pred = _rf_predict(root, X[oob])
            for i, idx in enumerate(oob): oob_preds[idx].append(oob_pred[i])
    def predict(X_new):
        preds = np.array([_rf_predict(t, np.asarray(X_new, dtype=float)) for t in trees])
        if task == "classification":
            out = []
            for j in range(preds.shape[1]):
                vals, cnts = np.unique(preds[:, j], return_counts=True)
                out.append(vals[np.argmax(cnts)])
            return np.array(out)
        return preds.mean(axis=0)
    # OOB score
    oob_valid = [i for i in range(n) if len(oob_preds[i]) > 0]
    if task == "classification":
        y_oob = []
        for i in oob_valid:
            v, c = np.unique(oob_preds[i], return_counts=True); y_oob.append(v[np.argmax(c)])
        oob_score = float(np.mean([y_oob[k] == y[i] for k, i in enumerate(oob_valid)]))
    else:
        y_oob = [np.mean(oob_preds[i]) for i in oob_valid]
        oob_score = float(math.sqrt(np.mean([(y_oob[k] - y[i]) ** 2 for k, i in enumerate(oob_valid)])))
    return {"predict": predict, "oob_score": oob_score,
            "n_trees": n_trees, "mtry": mtry,
            "method": f"Random Forest ({task}, {n_trees} trees, mtry={mtry})"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("=== Random forest classification (3 blobs) ===")
    X = np.vstack([rng.normal([0, 0], 1, (100, 2)),
                    rng.normal([4, 0], 1, (100, 2)),
                    rng.normal([2, 4], 1, (100, 2))])
    y = np.repeat([0, 1, 2], 100)
    fit = random_forest(X, y, task="classification", n_trees=30, max_depth=8, seed=0)
    print(f"  training accuracy = {(fit['predict'](X) == y).mean():.3f}")
    print(f"  OOB accuracy      = {fit['oob_score']:.3f}")

    print("\n=== Random forest regression ===")
    X = rng.uniform(-3, 3, size=(400, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0, 0.3, 400)
    fit = random_forest(X, y, task="regression", n_trees=30, max_depth=8, seed=0)
    y_hat = fit["predict"](X)
    print(f"  in-sample RMSE = {math.sqrt(np.mean((y - y_hat) ** 2)):.3f}")
    print(f"  OOB RMSE       = {fit['oob_score']:.3f}")

    print("\n--- library cross-check (sklearn RandomForest) ---")
    try:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=30, max_depth=8, oob_score=True, random_state=0)
        X = np.vstack([rng.normal([0, 0], 1, (100, 2)), rng.normal([4, 0], 1, (100, 2)),
                        rng.normal([2, 4], 1, (100, 2))])
        y = np.repeat([0, 1, 2], 100)
        clf.fit(X, y)
        print(f"  sklearn OOB accuracy = {clf.oob_score_:.3f}")
    except Exception as ex:
        print(f"  (sklearn unavailable: {ex})")
