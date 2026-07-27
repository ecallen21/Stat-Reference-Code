"""DBSCAN: density-based spatial clustering (Reference §9.11).

Ester, Kriegel, Sander & Xu (1996). Two parameters:
    eps    : radius of a point's neighborhood
    minPts : minimum number of points (including itself) to be a 'core' point

Definitions:
    core point   : has >= minPts neighbors within eps
    border point : has < minPts neighbors but lies within eps of some core
    noise point  : neither

Algorithm:
    For each unvisited point:
        find its eps-neighborhood
        if not core: label as noise (may be reclassified as border later)
        else: start a new cluster, add the point + all density-reachable points
              (breadth-first expansion of connected core points, plus their
              border neighbors)

Advantages over k-means:
    - No need to pre-specify k
    - Handles non-convex / arbitrarily shaped clusters
    - Robust to outliers (labels them explicitly as noise)

Common heuristic for eps: plot the k-distance graph (distance to the k-th
nearest neighbor for each point, sorted) and look for a knee -- that y-value
is a reasonable eps for minPts = k + 1.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _neighborhood(X, i, eps):
    """Return indices within eps (Euclidean) of point i."""
    d = np.sqrt(((X - X[i]) ** 2).sum(axis=1))
    return np.where(d <= eps)[0]


def dbscan(X, eps: float, min_pts: int) -> dict:
    """DBSCAN from scratch.

    Returns
    -------
    dict with ``labels`` (n,) where -1 = noise, otherwise 0..K-1 cluster IDs;
    also core/border/noise indicators.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    labels = -np.ones(n, dtype=int)                # -1 = noise / unvisited
    visited = np.zeros(n, dtype=bool)
    is_core = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]: continue
        visited[i] = True
        nbrs = _neighborhood(X, i, eps)
        if len(nbrs) < min_pts:
            labels[i] = -1                        # stays noise (may be reclassified as border)
            continue
        # start a new cluster and expand
        labels[i] = cluster_id
        is_core[i] = True
        seeds = list(nbrs)
        k = 0
        while k < len(seeds):
            j = seeds[k]
            if not visited[j]:
                visited[j] = True
                jnbrs = _neighborhood(X, j, eps)
                if len(jnbrs) >= min_pts:
                    is_core[j] = True
                    seeds.extend([q for q in jnbrs if q not in seeds])
            if labels[j] == -1:
                labels[j] = cluster_id
            k += 1
        cluster_id += 1

    n_noise = int((labels == -1).sum())
    K = cluster_id
    return {"labels": labels.tolist(),
            "n_clusters": K,
            "n_noise": n_noise,
            "n_core": int(is_core.sum()),
            "eps": eps, "min_pts": min_pts,
            "cluster_sizes": [int((labels == c).sum()) for c in range(K)],
            "method": "DBSCAN"}


def k_distance_graph(X, k: int) -> list:
    """Distance to the k-th nearest neighbor for each point, sorted ascending.

    Useful for choosing eps: plot this and look for a knee (elbow). The
    y-value at the knee is a good eps for min_pts = k + 1.
    """
    X = np.asarray(X, dtype=float)
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    kth = np.sort(d, axis=1)[:, k - 1]
    return sorted(kth.tolist())


def library_versions(X, eps, min_pts):
    from sklearn.cluster import DBSCAN
    m = DBSCAN(eps=eps, min_samples=min_pts).fit(X)
    return {"sklearn labels head": m.labels_[:20].tolist(),
            "sklearn n_clusters": int(m.labels_.max() + 1),
            "sklearn n_noise": int((m.labels_ == -1).sum())}


if __name__ == "__main__":
    rng = np.random.default_rng(61)
    # Two "moon" shapes -- classic non-convex DBSCAN example
    n_per = 100
    t1 = rng.uniform(0, np.pi, n_per)
    X1 = np.column_stack([np.cos(t1), np.sin(t1)]) + rng.normal(0, 0.08, size=(n_per, 2))
    t2 = rng.uniform(0, np.pi, n_per)
    X2 = np.column_stack([1 + np.cos(t2), -np.sin(t2) + 0.5]) + rng.normal(0, 0.08, size=(n_per, 2))
    noise = rng.uniform(-1, 2, size=(20, 2))
    X = np.vstack([X1, X2, noise])

    print("=== k-distance graph (k=4) for eps selection ===")
    kd = k_distance_graph(X, k=4)
    print(f"  distances (sample): [{kd[0]:.3f}, {kd[50]:.3f}, {kd[-30]:.3f}, {kd[-1]:.3f}]")
    print(f"  pick eps ~ knee -- try 0.15 for min_pts=4")

    print("\n=== DBSCAN (eps=0.15, min_pts=4) ===")
    out = dbscan(X, eps=0.15, min_pts=4)
    print(f"  n_clusters: {out['n_clusters']}, n_noise: {out['n_noise']}, n_core: {out['n_core']}")
    print(f"  cluster sizes: {out['cluster_sizes']}")

    print("\n--- library (sklearn) ---")
    for k, v in library_versions(X, eps=0.15, min_pts=4).items():
        print(f"  {k}: {v}")
