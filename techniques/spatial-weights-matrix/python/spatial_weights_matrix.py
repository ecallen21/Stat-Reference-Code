"""Spatial weights matrix W (Reference §23.2).

The (n x n) matrix W encodes which locations are 'neighbours':
    W_ij = weight of location j when computing statistics at i.
    W_ii = 0 by convention.

Common constructions
    Rook / Queen contiguity: W_ij = 1 if regions share an edge (rook) or
        edge/vertex (queen), else 0.  For grids / polygons.
    Distance band: W_ij = 1 if d_ij <= threshold, else 0.
    k-Nearest Neighbours: W_ij = 1 if j is among i's k closest points.
    Kernel-weighted: W_ij = exp(-d_ij^2 / (2 h^2)) or similar.

Row standardization
    W_ij <- W_ij / sum_k W_ik  so each row sums to 1.  Standard for
    Moran's I, spatial regression models.

The demo below uses point coordinates (no polygon topology).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _pairwise_dist(coords):
    coords = np.asarray(coords, dtype=float)
    d = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def W_distance_band(coords, threshold: float, row_standardize: bool = True) -> np.ndarray:
    D = _pairwise_dist(coords); n = D.shape[0]
    W = (D <= threshold).astype(float); np.fill_diagonal(W, 0.0)
    if row_standardize:
        rs = W.sum(1); rs[rs == 0] = 1.0
        W = W / rs[:, None]
    return W


def W_knn(coords, k: int, row_standardize: bool = True) -> np.ndarray:
    D = _pairwise_dist(coords); n = D.shape[0]
    W = np.zeros_like(D)
    for i in range(n):
        d = D[i].copy(); d[i] = np.inf
        idx = np.argsort(d)[:k]
        W[i, idx] = 1.0
    if row_standardize:
        rs = W.sum(1); rs[rs == 0] = 1.0
        W = W / rs[:, None]
    return W


def W_kernel(coords, bandwidth: float, kernel: str = "gaussian",
             row_standardize: bool = True) -> np.ndarray:
    D = _pairwise_dist(coords); n = D.shape[0]
    if kernel == "gaussian":
        W = np.exp(-0.5 * (D / bandwidth) ** 2)
    elif kernel == "bisquare":
        u = D / bandwidth
        W = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
    else: raise ValueError("kernel must be 'gaussian' or 'bisquare'")
    np.fill_diagonal(W, 0.0)
    if row_standardize:
        rs = W.sum(1); rs[rs == 0] = 1.0
        W = W / rs[:, None]
    return W


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10, size=(20, 2))

    print("=== W distance-band (threshold = 2, row-standardized) ===")
    W = W_distance_band(coords, threshold=2.0)
    print(f"  shape: {W.shape}   avg #neighbours: {(W > 0).sum(1).mean():.2f}")

    print("\n=== W kNN (k = 4, row-standardized) ===")
    W = W_knn(coords, k=4)
    print(f"  shape: {W.shape}   avg #neighbours: {(W > 0).sum(1).mean():.2f}")

    print("\n=== W Gaussian kernel (bandwidth = 2) ===")
    W = W_kernel(coords, bandwidth=2.0, kernel="gaussian")
    print(f"  row sums = 1? {np.allclose(W.sum(1), 1.0)}")
    print(f"  W[0, :5] = {W[0, :5].round(3)}")

    print("\n--- library cross-check (R spdep::poly2nb / knn2nb / nb2listw) ---")
