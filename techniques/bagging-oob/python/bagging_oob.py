"""Bagging + Out-Of-Bag error (Reference Sec 47.58).

Breiman 1996 'Bagging predictors', Machine Learning 24(2). Train B
weak learners on IID bootstrap resamples; average their
predictions:

    f_bag(x) = 1/B * sum_{b=1}^B f_b(x).

Reduces variance without changing bias. Out-Of-Bag (OOB) estimate:
for each training point i, average predictions from bags that did
NOT include i -- gives a free CV-like estimate at cost 0.

Prob(sample i missed by a given bag) = (1 - 1/n)^n -> 1/e ~ 0.368.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from sklearn.tree import DecisionTreeRegressor    # base learner


def bagging_regressor(X, y, B=200, max_depth=None, seed=0):
    """Bagging + OOB. Returns learners, OOB preds, OOB R^2."""
    rng = np.random.default_rng(seed)
    n = len(X)
    preds_sum = np.zeros(n)
    counts = np.zeros(n)
    learners = []
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[idx] = False
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=b).fit(X[idx], y[idx])
        preds = tree.predict(X[oob_mask])
        preds_sum[oob_mask] += preds
        counts[oob_mask] += 1
        learners.append(tree)
    valid = counts > 0
    yhat_oob = np.zeros(n); yhat_oob[valid] = preds_sum[valid] / counts[valid]
    ss_res = np.sum((y[valid] - yhat_oob[valid]) ** 2)
    ss_tot = np.sum((y[valid] - y[valid].mean()) ** 2)
    r2_oob = 1 - ss_res / ss_tot
    return {"learners": learners, "y_oob": yhat_oob, "r2_oob": float(r2_oob),
            "coverage": float(valid.mean())}


if __name__ == "__main__":
    print("=== Bagging + Out-Of-Bag (Breiman 1996) ===\n")
    rng = np.random.default_rng(0)
    n, p = 500, 10
    X = rng.normal(size=(n, p))
    y = 3 * X[:, 0] - 2 * X[:, 1] * X[:, 2] + np.sin(2 * X[:, 3]) + 0.3 * rng.normal(size=n)

    # Baseline single tree R^2 via 5-fold CV
    from sklearn.model_selection import cross_val_score    # CV wrapper
    single = DecisionTreeRegressor(max_depth=None, random_state=0)
    r2_single = cross_val_score(single, X, y, cv=5, scoring="r2").mean()
    print(f"  Single deep tree  R^2 (5-fold CV) = {r2_single:.3f}")

    for B in [10, 50, 200]:
        r = bagging_regressor(X, y, B=B, max_depth=None, seed=0)
        print(f"  Bagging B={B:3d}  OOB R^2 = {r['r2_oob']:.3f}   "
              f"OOB coverage = {r['coverage']:.3f}")

    print(f"\n  E[fraction never sampled] = (1 - 1/n)^n -> 1/e = {1/np.e:.3f}")
    print("  As B grows, OOB coverage -> 1.0 (every training point eventually excluded).")

    print("\n--- library cross-check (ipred R; sklearn.BaggingRegressor Python) ---")
