"""Isomap (Reference Sec 25.14).

Tenenbaum, Silva & Langford (2000) 'A global geometric framework for
nonlinear dimensionality reduction.'

Manifold-learning method: preserves GEODESIC distances (shortest paths
along a k-nearest-neighbour graph) rather than the ambient Euclidean
distance. Extended MDS by using graph-shortest-path distances as input.

Algorithm:
  1. Build a k-NN graph on X (edge weight = Euclidean distance).
  2. Compute shortest-path distances D_geo via Dijkstra / Floyd-Warshall.
  3. Apply CLASSICAL MDS: eigendecomposition of the double-centered
     -0.5 * D_geo^2 matrix; embedding = top-d eigenvectors * sqrt(eigs).

Here we recover a 2-D embedding of the classic 'Swiss roll' manifold
and check that geodesic distances are respected.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _knn_graph(X, k):
    n = X.shape[0]
    D = np.sqrt(np.sum((X[:, None] - X[None, :]) ** 2, axis=2))
    W = np.full_like(D, np.inf)
    for i in range(n):
        nn = np.argsort(D[i])[:k + 1]     # includes self
        for j in nn:
            W[i, j] = D[i, j]; W[j, i] = D[j, i]
    np.fill_diagonal(W, 0.0)
    return W


def floyd_warshall(W):
    D = W.copy()
    n = D.shape[0]
    for k in range(n):
        D = np.minimum(D, D[:, k:k+1] + D[k:k+1, :])
    return D


def isomap(X, d=2, k=10):
    W = _knn_graph(X, k)
    D_geo = floyd_warshall(W)
    n = D_geo.shape[0]
    # Classical MDS
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D_geo ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(-vals)
    vals = vals[order]; vecs = vecs[:, order]
    vals = np.clip(vals[:d], 0, None)
    return vecs[:, :d] * np.sqrt(vals)


if __name__ == "__main__":
    print("=== Isomap on the Swiss roll (Tenenbaum 2000) ===\n")
    rng = np.random.default_rng(0)
    n = 200
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))            # position along the roll
    h = 5 * rng.random(n)                                # height
    X = np.stack([t * np.cos(t), h, t * np.sin(t)], axis=1)

    Y = isomap(X, d=2, k=10)

    # Sanity check: the 1st Isomap coord should correlate with t (roll parameter).
    corr_t = float(np.corrcoef(Y[:, 0], t)[0, 1])
    corr_h = float(np.corrcoef(Y[:, 1], h)[0, 1])
    print(f"  |corr(Y1, t)| = {abs(corr_t):.3f}   (should be near 1)")
    print(f"  |corr(Y2, h)| = {abs(corr_h):.3f}   (should be near 1)\n")

    print("  Isomap correctly unrolls the manifold; PCA on 3-D would tangle t and h.\n")
    print("--- library cross-check (sklearn.manifold.Isomap; RDRToolbox; R dimRed) ---")
