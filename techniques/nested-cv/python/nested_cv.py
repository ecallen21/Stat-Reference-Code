"""Nested cross-validation + stratified repeated CV (Reference §10.13).

Plain K-fold CV that also picks hyperparameters gives an OPTIMISTICALLY BIASED
performance estimate: the model is being scored on data that was used (via the
inner tuning) to pick its own hyperparameters.

Nested CV separates the two:
    OUTER loop (K_outer folds): held-out test set for HONEST performance.
    INNER loop (K_inner folds within each outer training set): pick the best
        hyperparameter for that outer fold.
    Refit on outer-training with the winning hyperparameter; score on outer-test.
    Aggregate the K_outer test scores.

The outer-loop scores are a valid estimate of the tuning-and-training procedure's
generalization performance -- NOT of any specific fitted model. The single best
hyperparameter across the whole dataset is a separate refit (usually done after
nested CV to get the production model).

Stratified repeated CV: repeat K-fold R times with different random seeds and
average, reducing variance from the particular fold assignment.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Callable, Sequence    # stdlib: type hints

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _kfold_indices(n, k, rng):
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, test


def _stratified_kfold_indices(y, k, rng):
    y = np.asarray(y); n = y.size
    classes = np.unique(y)
    class_folds = {}
    for c in classes:
        idx_c = np.where(y == c)[0]; rng.shuffle(idx_c)
        class_folds[c] = np.array_split(idx_c, k)
    for i in range(k):
        test = np.concatenate([class_folds[c][i] for c in classes])
        train = np.setdiff1d(np.arange(n), test)
        yield train, test


def nested_cv(X, y, fit_fn: Callable, predict_fn: Callable, score_fn: Callable,
              hyperparameter_grid, k_outer: int = 5, k_inner: int = 3,
              stratified: bool = False, seed: int = 0) -> dict:
    """Nested K-fold CV.

    Parameters
    ----------
    fit_fn(X_train, y_train, hp) -> model.
    predict_fn(model, X) -> predictions.
    score_fn(y_true, y_pred) -> scalar (HIGHER is better; flip sign for loss).
    hyperparameter_grid : iterable of hyperparameters to search.
    """
    X = np.asarray(X); y = np.asarray(y); n = X.shape[0]
    rng = np.random.default_rng(seed)
    outer_scores = []
    picked_hps = []
    splitter = _stratified_kfold_indices if stratified else _kfold_indices
    outer_split_arg = y if stratified else n
    for i, (train_outer, test_outer) in enumerate(splitter(outer_split_arg, k_outer, rng)):
        X_tr, y_tr = X[train_outer], y[train_outer]
        # inner grid search
        inner_rng = np.random.default_rng(seed + 1000 + i)
        best_hp = None; best_score = -np.inf
        for hp in hyperparameter_grid:
            inner_scores = []
            for train_in, test_in in splitter(y_tr if stratified else len(train_outer), k_inner, inner_rng):
                m = fit_fn(X_tr[train_in], y_tr[train_in], hp)
                p = predict_fn(m, X_tr[test_in])
                inner_scores.append(float(score_fn(y_tr[test_in], p)))
            mean_inner = float(np.mean(inner_scores))
            if mean_inner > best_score:
                best_score = mean_inner; best_hp = hp
        picked_hps.append(best_hp)
        # refit on all outer-training, score on outer-test
        model = fit_fn(X_tr, y_tr, best_hp)
        pred = predict_fn(model, X[test_outer])
        outer_scores.append(float(score_fn(y[test_outer], pred)))
    scores = np.array(outer_scores)
    return {"outer_scores": scores.tolist(),
            "mean_score": float(scores.mean()),
            "SE_score": float(scores.std(ddof=1) / math.sqrt(len(scores))),
            "hyperparameters_picked_per_outer_fold": picked_hps,
            "k_outer": k_outer, "k_inner": k_inner,
            "method": "nested K-fold CV"}


def stratified_repeated_cv(X, y, fit_fn, predict_fn, score_fn,
                            k: int = 5, n_repeats: int = 10, seed: int = 0) -> dict:
    """Stratified K-fold repeated ``n_repeats`` times with different seeds."""
    X = np.asarray(X); y = np.asarray(y); n = X.shape[0]
    all_scores = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        rep_scores = []
        for train, test in _stratified_kfold_indices(y, k, rng):
            m = fit_fn(X[train], y[train])
            p = predict_fn(m, X[test])
            rep_scores.append(float(score_fn(y[test], p)))
        all_scores.append(float(np.mean(rep_scores)))
    all_scores = np.array(all_scores)
    return {"per_repeat_mean": all_scores.tolist(),
            "overall_mean": float(all_scores.mean()),
            "SE_across_repeats": float(all_scores.std(ddof=1) / math.sqrt(n_repeats)),
            "k": k, "n_repeats": n_repeats,
            "method": "stratified repeated K-fold CV"}


if __name__ == "__main__":
    # Ridge-regression example: hyperparameter = regularization strength
    rng = np.random.default_rng(59)
    n = 200
    X = rng.normal(0, 1, size=(n, 5))
    beta_true = np.array([1.0, -0.5, 0.3, 0.0, 0.0])
    y = X @ beta_true + rng.normal(0, 0.5, n)

    def ridge_fit(X_tr, y_tr, lam):
        p = X_tr.shape[1]
        beta = np.linalg.solve(X_tr.T @ X_tr + lam * np.eye(p), X_tr.T @ y_tr)
        return beta
    def ridge_predict(beta, X):
        return X @ beta
    def neg_mse(y_true, y_pred):
        return -float(np.mean((y_true - y_pred) ** 2))     # higher = better

    hp_grid = [0.01, 0.1, 1.0, 10.0]
    print("=== Nested 5x3 CV for ridge regression ===")
    out = nested_cv(X, y, ridge_fit, ridge_predict, neg_mse, hp_grid,
                     k_outer=5, k_inner=3)
    print(f"  outer neg-MSE per fold: {[f'{s:.4f}' for s in out['outer_scores']]}")
    print(f"  mean neg-MSE = {out['mean_score']:.4f}  (SE = {out['SE_score']:.4f})")
    print(f"  hyperparameters chosen per outer fold: {out['hyperparameters_picked_per_outer_fold']}")

    # Stratified repeated CV example: classification-style label
    y_class = (X[:, 0] > 0).astype(int)
    def logreg_fit(X_tr, y_tr):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=200).fit(X_tr, y_tr)
    def logreg_predict(m, X): return m.predict(X)
    def accuracy(y_true, y_pred): return float(np.mean(y_true == y_pred))

    print("\n=== Stratified 5-fold CV repeated 10 times for logistic regression ===")
    out2 = stratified_repeated_cv(X, y_class, logreg_fit, logreg_predict, accuracy,
                                    k=5, n_repeats=10)
    print(f"  per-repeat mean accuracy: {[f'{s:.4f}' for s in out2['per_repeat_mean']]}")
    print(f"  overall = {out2['overall_mean']:.4f}  (SE across repeats = {out2['SE_across_repeats']:.4f})")
