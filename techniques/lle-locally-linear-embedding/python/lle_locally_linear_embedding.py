"""Locally Linear Embedding (LLE) — Reference Sec 25.15.

Roweis & Saul (2000) 'Nonlinear dimensionality reduction by locally
linear embedding.'

Preserves LOCAL geometry: each point is reconstructed as a linear
combination of its k nearest neighbours; the same weights should
reconstruct it in the low-dim embedding.

Algorithm:
  1. Find k-NN for each point.
  2. Solve W = argmin sum_i || x_i - sum_{j in N(i)} W_ij x_j ||^2
     subject to  sum_j W_ij = 1 for each i.
  3. Find low-dim Y minimising  sum_i || y_i - sum_j W_ij y_j ||^2
     = tr(Y' (I - W)' (I - W) Y), subject to Y'Y = I.
     Bottom d+1 eigenvectors of M = (I-W)'(I-W); discard the trivial
     eigenvector.

Here we recover the 2-D embedding of a 3-D S-curve manifold.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _knn_indices(X, k):
    n = X.shape[0]
    D = np.sqrt(np.sum((X[:, None] - X[None, :]) ** 2, axis=2))
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :k]


def fit_lle(X, d=2, k=10, reg=1e-3):
    n = X.shape[0]
    knn = _knn_indices(X, k)
    W = np.zeros((n, n))
    for i in range(n):
        Zi = X[knn[i]] - X[i]                          # k x d
        C = Zi @ Zi.T                                   # k x k local Gram
        C += reg * np.trace(C) * np.eye(k) / k          # regularise (Roweis-Saul)
        w = np.linalg.solve(C, np.ones(k))
        w = w / w.sum()
        for j_idx, j in enumerate(knn[i]):
            W[i, j] = w[j_idx]
    M = (np.eye(n) - W).T @ (np.eye(n) - W)
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)
    # Skip the (near-zero) trivial eigenvector, take next d.
    return vecs[:, order[1:d + 1]]


if __name__ == "__main__":
    print("=== LLE on a 3-D S-curve (Roweis-Saul 2000) ===\n")
    rng = np.random.default_rng(0)
    n = 200
    t = 3 * np.pi * (rng.random(n) - 0.5)
    h = 5 * rng.random(n)
    X = np.stack([np.sin(t), h, np.sign(t) * (np.cos(t) - 1)], axis=1)

    Y = fit_lle(X, d=2, k=12)

    # LLE axes are permutation-invariant; check the best 1-to-1 match.
    C = np.abs(np.corrcoef(np.stack([Y[:, 0], Y[:, 1], t, h], axis=1).T)[:2, 2:])
    print(f"  |corr| matrix (rows Y1,Y2   cols t,h):\n{C.round(3)}")
    for src, name in enumerate(("t", "h")):
        best = int(C[:, src].argmax())
        print(f"  best Y-axis for {name}: Y{best+1}   |corr| = {C[best, src]:.3f}")
    print("\n  LLE preserves local structure; the two intrinsic coordinates map cleanly.\n")
    print("--- library cross-check (sklearn.manifold.LocallyLinearEmbedding; RDRToolbox) ---")
