"""Model stacking / super-learner (Reference §26.14; Wolpert 1992).

Combine predictions of several BASE learners via a META learner.  To
avoid overfitting, meta features must come from OUT-OF-FOLD predictions:

    1. Split training data into K folds.
    2. For each base model b and fold k:
        train on (folds != k), predict on fold k.
        Collect OOF prediction for each obs.
    3. Train META model on OOF predictions -> y.
    4. For test time: fit each base on ALL training data, feed test
        predictions into the meta model.

This is IDENTICAL to van der Laan et al.'s SUPER LEARNER when base + meta
are the same generic 'stack'.

Typical setup: diverse base learners (linear + tree + kNN + boosted);
simple meta (ridge / logistic).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def stacking_regression(X, y, base_learners: list, meta_fn,
                         n_folds: int = 5, seed: int = 0) -> dict:
    """Stacking for regression.

    base_learners: list of callables (X_train, y_train, X_test) -> predictions.
    meta_fn: callable (X_meta_train, y_train, X_meta_test) -> final predictions.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n = len(y); B = len(base_learners)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n); folds = np.array_split(perm, n_folds)
    OOF = np.zeros((n, B))
    for b, learn in enumerate(base_learners):
        for k in range(n_folds):
            te = folds[k]; tr = np.concatenate([folds[i] for i in range(n_folds) if i != k])
            OOF[te, b] = learn(X[tr], y[tr], X[te])
    # Refit each base on all data (for test time)
    def predict(X_new):
        preds = np.column_stack([learn(X, y, X_new) for learn in base_learners])
        return meta_fn(OOF, y, preds)
    # Also compute OOF meta-prediction for evaluation
    oof_meta = meta_fn(OOF, y, OOF)
    return {"OOF": OOF, "predict": predict,
            "oof_predictions": oof_meta,
            "n_base": int(B), "n_folds": int(n_folds),
            "method": "Stacked regression (K-fold OOF + meta learner)"}


def _linear_learner(Xtr, ytr, Xte):
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xtr)), Xtr]), ytr, rcond=None)
    return np.column_stack([np.ones(len(Xte)), Xte]) @ beta


def _tree_learner(Xtr, ytr, Xte):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "decision-tree", "python"))
    from decision_tree import build_tree, predict, _mse_impurity
    root = build_tree(Xtr, ytr, _mse_impurity, max_depth=6, min_samples_leaf=5)
    return predict(root, Xte)


def _knn_learner(Xtr, ytr, Xte, k: int = 5):
    return np.array([np.mean(ytr[np.argsort(np.linalg.norm(Xtr - xi, axis=1))[:k]]) for xi in Xte])


def _ridge_meta(OOF_tr, y_tr, OOF_te, lam: float = 0.1):
    X = np.column_stack([np.ones(len(OOF_tr)), OOF_tr])
    p = X.shape[1]
    R = lam * np.eye(p); R[0, 0] = 0
    beta = np.linalg.solve(X.T @ X + R, X.T @ y_tr)
    return np.column_stack([np.ones(len(OOF_te)), OOF_te]) @ beta


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 400
    X = rng.uniform(-3, 3, size=(n, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0, 0.3, n)
    X_te = rng.uniform(-3, 3, size=(200, 2))
    y_te = np.sin(X_te[:, 0]) + 0.5 * X_te[:, 1] + rng.normal(0, 0.3, 200)

    fit = stacking_regression(X, y, [_linear_learner, _tree_learner, _knn_learner],
                                _ridge_meta, n_folds=5, seed=0)
    print("=== Stacking (linear + tree + kNN) with ridge meta ===")
    print(f"  OOF RMSE = {math.sqrt(np.mean((fit['oof_predictions'] - y) ** 2)):.3f}")
    print(f"  test RMSE = {math.sqrt(np.mean((fit['predict'](X_te) - y_te) ** 2)):.3f}")
    print("\n  Individual OOF RMSE per base:")
    for b in range(fit["n_base"]):
        rmse_b = math.sqrt(np.mean((fit["OOF"][:, b] - y) ** 2))
        print(f"    base {b}: {rmse_b:.3f}")

    print("\n--- Note: sklearn.ensemble.StackingRegressor implements the same idea. ---")
