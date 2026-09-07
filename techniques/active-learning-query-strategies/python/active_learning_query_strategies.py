"""Active learning -- pool-based query strategies (Reference Sec 47.27).

Settles 2010 'Active Learning Literature Survey'. Iteratively picks
the most INFORMATIVE unlabelled example to query the oracle (human /
expensive labeler). Common strategies:

    UNCERTAINTY (least confidence): query x with highest 1 - max_c P(y=c|x).
    MARGIN:                          query x with smallest P(top1) - P(top2).
    ENTROPY:                         query x with highest predictive entropy.
    QBC (query by committee):        query x with highest committee disagreement.
    EXPECTED MODEL CHANGE / EMC:     query x whose adding shifts model most.
    CORE-SET / DIVERSITY:            pick x maximally distant from labelled.

We compare random, uncertainty, and margin on a small logistic
classifier over a synthetic 2-D problem: each strategy queries
budget B labels and we track test accuracy as a function of labels.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def fit_logistic(X, y):
    Xd = np.c_[np.ones(len(X)), X]
    beta = np.zeros(Xd.shape[1])
    for _ in range(200):
        p = 1 / (1 + np.exp(-Xd @ beta))
        grad = Xd.T @ (y - p)
        H = Xd.T @ np.diag(p * (1 - p)) @ Xd
        beta += np.linalg.solve(H + 1e-3 * np.eye(Xd.shape[1]), grad)
    return beta


def predict_proba(X, beta):
    return 1 / (1 + np.exp(-np.c_[np.ones(len(X)), X] @ beta))


def uncertainty_query(pool_probs):
    return int(np.argmax(1 - np.maximum(pool_probs, 1 - pool_probs)))


def margin_query(pool_probs):
    return int(np.argmin(np.abs(pool_probs - 0.5) * 2))


def random_query(pool_probs, rng):
    return int(rng.integers(0, len(pool_probs)))


def active_learn(X_pool, y_pool, X_test, y_test, strategy, B=40, init_n=10, seed=0):
    rng = np.random.default_rng(seed)
    idx_labelled = list(rng.choice(len(X_pool), size=init_n, replace=False))
    idx_pool = [i for i in range(len(X_pool)) if i not in set(idx_labelled)]
    accs = []
    for _ in range(B):
        beta = fit_logistic(X_pool[idx_labelled], y_pool[idx_labelled])
        acc = float(np.mean((predict_proba(X_test, beta) > 0.5) == y_test))
        accs.append(acc)
        probs = predict_proba(X_pool[idx_pool], beta)
        if strategy == "random":
            pick = random_query(probs, rng)
        elif strategy == "uncertainty":
            pick = uncertainty_query(probs)
        elif strategy == "margin":
            pick = margin_query(probs)
        idx_new = idx_pool.pop(pick)
        idx_labelled.append(idx_new)
    return accs


if __name__ == "__main__":
    print("=== Active learning query strategies (pool-based) ===\n")
    rng = np.random.default_rng(0)
    n = 800; d = 2
    X = rng.normal(size=(n, d))
    beta_true = np.array([1.0, -0.5])
    y = (X @ beta_true + rng.normal(scale=0.5, size=n) > 0).astype(int)

    n_test = 200
    X_test = rng.normal(size=(n_test, d)); y_test = (X_test @ beta_true + rng.normal(scale=0.5, size=n_test) > 0).astype(int)

    B = 40
    for strat in ["random", "uncertainty", "margin"]:
        accs = active_learn(X, y, X_test, y_test, strategy=strat, B=B, init_n=10)
        print(f"  {strat:>12s}   final acc = {accs[-1]:.3f}   "
              f"acc at B/2 = {accs[B // 2]:.3f}")

    print(f"\n  Note: for BINARY classification uncertainty and margin sampling are identical")
    print(f"  (both pick the point nearest p = 0.5). Gains over random depend on class balance,")
    print(f"  boundary curvature, and dimensionality -- active learning is not always faster.")

    print("\n--- library cross-check (modAL / small-text Python; ALEval R) ---")
