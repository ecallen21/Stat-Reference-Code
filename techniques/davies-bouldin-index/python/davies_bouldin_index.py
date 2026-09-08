"""Davies-Bouldin cluster-validation index (Reference Sec 47.82).

Davies & Bouldin 1979 'A cluster separation measure', IEEE TPAMI.
Ratio of intra-cluster scatter to inter-cluster distance:

    S_i = mean_{x in C_i}  || x - centroid_i ||
    M_ij = || centroid_i - centroid_j ||
    R_ij = (S_i + S_j) / M_ij
    DB = (1/K) sum_i  max_{j != i}  R_ij

LOWER = better (well-separated compact clusters give DB near 0).
Rule of thumb: pick K minimising DB.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def davies_bouldin(X, labels):
    labs = np.unique(labels)
    K = len(labs)
    centroids = np.array([X[labels == L].mean(axis=0) for L in labs])
    S = np.array([np.linalg.norm(X[labels == L] - centroids[i], axis=1).mean()
                   for i, L in enumerate(labs)])
    D = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (S[:, None] + S[None, :]) / (D + 1e-12)
    np.fill_diagonal(R, -np.inf)
    return float(R.max(axis=1).mean())


if __name__ == "__main__":
    print("=== Davies-Bouldin index (Davies-Bouldin 1979) ===\n")
    rng = np.random.default_rng(0)

    centers = np.array([[0, 0], [5, 0], [2.5, 4]])
    X = np.vstack([c + 0.3 * rng.normal(size=(80, 2)) for c in centers])

    from sklearn.cluster import KMeans    # base clusterer
    print("  Well-separated 3-cluster data:")
    for k in range(2, 8):
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(X)
        db = davies_bouldin(X, km.labels_)
        marker = "  <-- min" if k == 3 else ""
        print(f"    k = {k}:  DB = {db:.3f}{marker}")

    # Overlapping clusters
    Xo = np.vstack([c + 1.2 * rng.normal(size=(80, 2)) for c in centers])
    print("\n  Overlapping 3-cluster data:")
    dbs = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Xo)
        db = davies_bouldin(Xo, km.labels_)
        dbs.append(db)
    for k, db in zip(range(2, 8), dbs):
        marker = "  <-- min" if db == min(dbs) else ""
        print(f"    k = {k}:  DB = {db:.3f}{marker}")

    print("\n  DB rule: choose the K with the LOWEST index.")
    print("\n--- library cross-check (fpc R; sklearn.metrics.davies_bouldin_score Python) ---")
