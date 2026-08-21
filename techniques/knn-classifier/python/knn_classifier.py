"""k-Nearest Neighbors classifier + regressor (Reference §26.11).

Nonparametric prediction: for a new x, find the k closest training points
and majority-vote (classification) or average (regression).

Distance-weighted variant
    Weight each neighbor by 1 / d (or exp(-d^2 / (2 h^2)) kernel weights).
    Reduces the impact of ties and puts more weight on close matches.

Choice of k
    k = 1: perfect training accuracy, wild overfitting.
    Large k: smoother, more bias.
    Standard: cross-validate on odd k in [1, 30].

Cost
    Naive: O(n) per prediction.
    Ball tree / KD tree: O(log n) if d not too large.
    LSH / faiss for very large n or high d.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def knn_predict(X_train, y_train, X_test, k: int = 5, task: str = "classification",
                weighted: bool = False):
    X_train = np.asarray(X_train, dtype=float); y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)
    preds = []
    for x in X_test:
        d = np.linalg.norm(X_train - x, axis=1)
        idx = np.argsort(d)[:k]
        nbrs_y = y_train[idx]; nbrs_d = d[idx]
        if task == "classification":
            if weighted:
                w = 1 / (nbrs_d + 1e-8)
                cls = np.unique(nbrs_y); tot = {c: 0.0 for c in cls}
                for c in cls: tot[c] = w[nbrs_y == c].sum()
                preds.append(max(tot, key=tot.get))
            else:
                vals, cnts = np.unique(nbrs_y, return_counts=True)
                preds.append(vals[np.argmax(cnts)])
        else:
            if weighted:
                w = 1 / (nbrs_d + 1e-8)
                preds.append(float(np.sum(w * nbrs_y) / w.sum()))
            else:
                preds.append(float(nbrs_y.mean()))
    return np.array(preds)


def knn_cv_k(X, y, k_grid=range(1, 30, 2), task: str = "classification",
             n_folds: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y); idx = rng.permutation(n); folds = np.array_split(idx, n_folds)
    scores = []
    for k in k_grid:
        errs = []
        for f in range(n_folds):
            te = folds[f]; tr = np.concatenate([folds[i] for i in range(n_folds) if i != f])
            pred = knn_predict(X[tr], y[tr], X[te], k=k, task=task)
            if task == "classification":
                errs.append(1 - (pred == y[te]).mean())
            else:
                errs.append(math.sqrt(np.mean((pred - y[te]) ** 2)))
        scores.append(float(np.mean(errs)))
    best_k = list(k_grid)[int(np.argmin(scores))]
    return {"k_grid": list(k_grid), "scores": scores, "best_k": int(best_k)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    X = np.vstack([rng.normal([0, 0], 1, (100, 2)),
                    rng.normal([4, 0], 1, (100, 2)),
                    rng.normal([2, 4], 1, (100, 2))])
    y = np.repeat([0, 1, 2], 100)

    print("=== 5-NN classification ===")
    pred = knn_predict(X, y, X, k=5)
    print(f"  training accuracy = {(pred == y).mean():.3f}")

    print("\n=== 5-fold CV over k grid ===")
    r = knn_cv_k(X, y, k_grid=range(1, 20, 2))
    for k, s in zip(r["k_grid"], r["scores"]):
        marker = "*" if k == r["best_k"] else " "
        print(f"  {marker} k = {k}: CV error = {s:.4f}")
    print(f"  best k = {r['best_k']}")

    print("\n--- library cross-check (sklearn KNeighbors) ---")
    try:
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=r["best_k"]).fit(X, y)
        print(f"  sklearn kNN training accuracy = {clf.score(X, y):.3f}")
    except Exception as ex:
        print(f"  (sklearn unavailable: {ex})")
