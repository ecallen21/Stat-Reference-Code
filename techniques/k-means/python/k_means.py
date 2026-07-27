"""k-means clustering (Reference §9.9).

Partitions n observations into k clusters by iteratively:
    1. Assigning each point to its nearest current centroid.
    2. Recomputing each centroid as the mean of its assigned points.
Repeat until assignments stop changing.

Objective: minimize total within-cluster sum of squares
    W = sum_c sum_{x in cluster c} ||x - mu_c||^2   ("inertia")

Initialization matters (Lloyd's algorithm converges to a local minimum):
    - random    : pick k random points as initial centroids (fast, noisy)
    - k-means++ : first centroid random; each subsequent centroid picked with
                  probability proportional to D(x)^2 -- the squared distance to
                  the nearest already-picked centroid. Provably ~O(log k) worse
                  than optimal on average; the default in scikit-learn.

Typical practice: run several restarts with k-means++ and keep the lowest inertia.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _kmeans_plus_plus_init(X, k, rng):
    """k-means++ centroid initialization."""
    n = X.shape[0]
    centroids = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centroids)[None, :, :]) ** 2).sum(-1), axis=1)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        centroids.append(X[rng.choice(n, p=probs)])
    return np.array(centroids)


def kmeans(X, k: int, init: str = "kmeans++", n_restarts: int = 10,
           max_iter: int = 300, tol: float = 1e-6, seed: int = 0) -> dict:
    """Lloyd's algorithm with optional k-means++ init and multiple restarts.

    Returns
    -------
    dict with best labels (n,), centroids (k, p), final inertia, and
    per-restart inertias.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    rng = np.random.default_rng(seed)
    best = None
    all_inertias = []
    for restart in range(n_restarts):
        if init == "kmeans++":
            C = _kmeans_plus_plus_init(X, k, rng)
        elif init == "random":
            C = X[rng.choice(n, size=k, replace=False)]
        else:
            raise ValueError("init must be 'kmeans++' or 'random'")
        for _ in range(max_iter):
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            labels = np.argmin(d2, axis=1)
            new_C = np.array([X[labels == c].mean(axis=0) if np.any(labels == c) else C[c]
                              for c in range(k)])
            if np.max(np.abs(new_C - C)) < tol:
                C = new_C; break
            C = new_C
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        labels = np.argmin(d2, axis=1)
        inertia = float(d2[np.arange(n), labels].sum())
        all_inertias.append(inertia)
        if best is None or inertia < best["inertia"]:
            best = {"labels": labels.tolist(), "centroids": C.tolist(),
                    "inertia": inertia, "restart": restart}
    return {"labels": best["labels"],
            "centroids": best["centroids"],
            "inertia": best["inertia"],
            "best_restart_index": best["restart"],
            "all_restart_inertias": all_inertias,
            "init": init, "k": k, "n": n, "p": p,
            "method": "k-means (Lloyd) with k-means++ init"}


def library_versions(X, k):
    from sklearn.cluster import KMeans
    m = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0).fit(X)
    return {"sklearn inertia": float(m.inertia_),
            "sklearn labels head": m.labels_[:15].tolist(),
            "sklearn centroids": m.cluster_centers_.tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(51)
    centers = np.array([[0, 0], [4, 4], [8, 0]])
    X = np.vstack([rng.normal(c, 0.7, size=(40, 2)) for c in centers])

    print("=== k-means (k=3, k-means++, 10 restarts) ===")
    out = kmeans(X, k=3, init="kmeans++", n_restarts=10)
    from collections import Counter
    print(f"  inertia = {out['inertia']:.4f}")
    print(f"  cluster sizes: {dict(Counter(out['labels']))}")
    print(f"  centroids: {out['centroids']}")
    print(f"  best restart index: {out['best_restart_index']}")

    print("\n--- library (sklearn) ---")
    for k, v in library_versions(X, k=3).items():
        print(f"  {k}: {v}")
