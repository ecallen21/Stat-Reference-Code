"""K-fold, stratified K-fold, and LOOCV (Reference §10.8, §10.12).

Cross-validation estimates the OUT-OF-SAMPLE performance of a model by
partitioning the data, training on one part, testing on the held-out part,
and averaging the test-set metric.

    K-fold          : partition into K folds; train on K-1, test on 1; K rounds.
    LOOCV           : special case K = n; each observation is a fold.
    Stratified K-fold: preserves class proportions in each fold (for classification).

Trade-offs:
    - Small K (2, 5) : less variance, more BIAS (training set much smaller than n).
    - Large K (10, LOOCV) : less bias, more VARIANCE (fold predictions correlated).
    K = 5 or 10 is the usual sweet spot; LOOCV mostly for smallish n.

This file implements all three, plus a simple prediction-error convenience for
OLS regression and any sklearn-compatible model.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Callable, Sequence    # stdlib: type hints (Callable = function; Sequence = indexable iterable)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def kfold_indices(n: int, k: int, shuffle: bool = True, seed: int = 0):
    """Yield (train_idx, test_idx) pairs for K-fold CV."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, test


def stratified_kfold_indices(y, k: int, shuffle: bool = True, seed: int = 0):
    """Yield (train_idx, test_idx) pairs preserving class proportions per fold."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    n = y.size
    classes = np.unique(y)
    # For each class, split its indices into k folds
    class_folds = {}
    for c in classes:
        idx_c = np.where(y == c)[0]
        if shuffle: rng.shuffle(idx_c)
        class_folds[c] = np.array_split(idx_c, k)
    for i in range(k):
        test_parts = [class_folds[c][i] for c in classes]
        test = np.concatenate(test_parts)
        train = np.setdiff1d(np.arange(n), test)
        yield train, test


def loocv_indices(n: int):
    """Yield leave-one-out (train_idx, test_idx) pairs."""
    for i in range(n):
        yield np.setdiff1d(np.arange(n), [i]), np.array([i])


def cv_score(X, y, fit_fn: Callable, predict_fn: Callable,
             score_fn: Callable, splitter, **splitter_kwargs) -> dict:
    """Generic cross-validation.

    Parameters
    ----------
    fit_fn(X_train, y_train) -> model.
    predict_fn(model, X_test) -> predictions.
    score_fn(y_true, y_pred) -> scalar (higher = better OR lower = better -- your call).
    splitter: one of kfold_indices / stratified_kfold_indices / loocv_indices.
    """
    X = np.asarray(X); y = np.asarray(y)
    scores = []
    for train, test in splitter(**splitter_kwargs):
        model = fit_fn(X[train], y[train])
        pred = predict_fn(model, X[test])
        scores.append(float(score_fn(y[test], pred)))
    scores = np.array(scores)
    return {"fold_scores": scores.tolist(),
            "mean_score": float(scores.mean()),
            "SE_score": float(scores.std(ddof=1) / math.sqrt(len(scores))),
            "n_folds": len(scores)}


# ---- Convenience: OLS regression MSE via K-fold -------------------------

def ols_fit(X_train, y_train):
    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    return beta

def ols_predict(beta, X_test):
    return X_test @ beta

def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def library_versions(X, y, k=5):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
    scores_kf = -cross_val_score(LinearRegression(), X, y, cv=KFold(k, shuffle=True, random_state=0),
                                  scoring="neg_mean_squared_error")
    scores_loo = -cross_val_score(LinearRegression(), X, y, cv=LeaveOneOut(),
                                   scoring="neg_mean_squared_error")
    return {"sklearn K-fold MSE mean": float(scores_kf.mean()),
            "sklearn LOOCV MSE mean":  float(scores_loo.mean())}


if __name__ == "__main__":
    rng = np.random.default_rng(41)
    n = 200
    X = np.column_stack([np.ones(n), rng.normal(0, 1, n), rng.normal(0, 1, n)])
    y = 1.5 + 0.8 * X[:, 1] - 0.3 * X[:, 2] + rng.normal(0, 0.5, n)

    print("=== 5-fold CV MSE for OLS ===")
    out = cv_score(X, y, ols_fit, ols_predict, mse,
                   splitter=kfold_indices, n=n, k=5, seed=0)
    print(f"  fold scores: {[f'{s:.4f}' for s in out['fold_scores']]}")
    print(f"  mean MSE = {out['mean_score']:.4f}  (SE = {out['SE_score']:.4f})")

    print("\n=== 10-fold CV MSE for OLS ===")
    out = cv_score(X, y, ols_fit, ols_predict, mse,
                   splitter=kfold_indices, n=n, k=10, seed=0)
    print(f"  mean MSE = {out['mean_score']:.4f}  (SE = {out['SE_score']:.4f})")

    print("\n=== LOOCV MSE for OLS ===")
    out = cv_score(X, y, ols_fit, ols_predict, mse,
                   splitter=loocv_indices, n=n)
    print(f"  mean MSE = {out['mean_score']:.4f}  (SE = {out['SE_score']:.4f})")

    # Stratified CV example: classification-style label with 3 classes
    y_class = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
    print("\n=== Stratified 5-fold: class counts per test fold ===")
    from collections import Counter
    for i, (train, test) in enumerate(stratified_kfold_indices(y_class, k=5, seed=0)):
        print(f"  fold {i}: test counts = {dict(Counter(y_class[test]))}")

    print("\n--- library (sklearn) ---")
    for k, v in library_versions(X, y).items():
        print(f"  {k}: {v}")
