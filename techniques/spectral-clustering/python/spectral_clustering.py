"""Spectral clustering (Reference Sec 47.78).

Ng, Jordan & Weiss 2001 'On spectral clustering: analysis and an
algorithm', NeurIPS. Steps:

    1. Build affinity matrix   W_ij = exp( -||x_i - x_j||^2 / (2 sigma^2) ).
    2. Degree matrix D = diag(W 1); normalised Laplacian
                            L_sym = I - D^{-1/2} W D^{-1/2}.
    3. Take the K smallest eigenvectors of L_sym as embedding.
    4. Normalise rows to unit norm; k-means on rows -> cluster labels.

Captures non-convex cluster shapes (moons, rings) that fail
k-means directly.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from sklearn.cluster import KMeans    # final k-means step


def spectral_cluster(X, K, sigma=None):
    n = len(X)
    sq = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    if sigma is None:
        # local-scale heuristic: use 7th-nearest-neighbour dist per point
        d_sorted = np.sort(np.sqrt(sq), axis=1)
        sigma = float(d_sorted[:, 7].mean())
    W = np.exp(-sq / (2 * sigma ** 2))
    # k-NN sparsification: keep top-10 neighbours per row (symmetrise)
    k_nn = 10
    for i in range(n):
        order = np.argsort(-W[i])
        keep = order[:k_nn + 1]
        mask = np.zeros(n, dtype=bool); mask[keep] = True
        W[i, ~mask] = 0
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0)
    d = W.sum(axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(d + 1e-12)
    L_sym = np.eye(n) - (D_inv_sqrt[:, None] * W * D_inv_sqrt[None, :])
    vals, vecs = np.linalg.eigh(L_sym)
    U = vecs[:, :K]
    # Row-normalise
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    labels = KMeans(n_clusters=K, n_init=20, random_state=0).fit_predict(U)
    return {"labels": labels, "sigma": sigma, "eigvals": vals[:K + 3]}


if __name__ == "__main__":
    print("=== Spectral clustering (Ng-Jordan-Weiss 2001) ===\n")
    rng = np.random.default_rng(0)
    n = 300
    theta = rng.uniform(0, np.pi, size=n)
    # Two moons
    X1 = np.column_stack([np.cos(theta), np.sin(theta)]) + 0.05 * rng.normal(size=(n, 2))
    X2 = np.column_stack([1 - np.cos(theta), -np.sin(theta) + 0.5]) + 0.05 * rng.normal(size=(n, 2))
    X = np.vstack([X1, X2])
    y_true = np.array([0] * n + [1] * n)

    # k-means baseline
    km = KMeans(n_clusters=2, n_init=20, random_state=0).fit_predict(X)
    def ari(a, b):
        from sklearn.metrics import adjusted_rand_score
        return adjusted_rand_score(a, b)
    print(f"  Two moons (n={2*n}):")
    print(f"    K-means ARI            = {ari(y_true, km):.3f}")

    sc = spectral_cluster(X, K=2)
    print(f"    Spectral cluster ARI   = {ari(y_true, sc['labels']):.3f}   "
          f"(sigma = {sc['sigma']:.3f})")

    # Rings
    ang = rng.uniform(0, 2 * np.pi, size=n)
    R1 = np.column_stack([np.cos(ang), np.sin(ang)]) + 0.05 * rng.normal(size=(n, 2))
    R2 = 2.5 * np.column_stack([np.cos(ang), np.sin(ang)]) + 0.05 * rng.normal(size=(n, 2))
    Xr = np.vstack([R1, R2])
    y_ring = np.array([0] * n + [1] * n)
    km_r = KMeans(n_clusters=2, n_init=20, random_state=0).fit_predict(Xr)
    sc_r = spectral_cluster(Xr, K=2)
    print(f"\n  Concentric rings (n={2*n}):")
    print(f"    K-means ARI            = {ari(y_ring, km_r):.3f}")
    print(f"    Spectral cluster ARI   = {ari(y_ring, sc_r['labels']):.3f}")

    print("\n--- library cross-check (kernlab / RSpectra R; sklearn.cluster.SpectralClustering Python) ---")
