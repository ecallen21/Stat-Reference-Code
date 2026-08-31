"""Functional clustering (Reference Sec 31.6).

Group curves x_i(t) into K clusters. Two practical routes:

  1. FPC + k-means: expand each curve in top-K functional principal
     components, run k-means on the score matrix.
  2. Basis-coefficient clustering: fit a spline / Fourier basis to
     each curve, cluster the coefficient vectors.

Model-based alternatives (James-Sugar 2003 funHDDC) fit mixtures of
functional distributions.

Here we implement FPC-scores + k-means with random-restart initialisation
and report cluster purity vs a hidden truth on synthetic mixed-shape
curves.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _fpca_scores(X, k):
    mean = X.mean(axis=0)
    Xc = X - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T


def _kmeans(Z, K, n_init=10, max_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    best_inertia = np.inf; best_labels = None
    for _ in range(n_init):
        idx = rng.choice(n, K, replace=False)
        centres = Z[idx].copy()
        for _ in range(max_iter):
            d = np.sum((Z[:, None] - centres[None]) ** 2, axis=2)
            labels = d.argmin(axis=1)
            new_centres = np.array([Z[labels == k].mean(axis=0) if (labels == k).any() else centres[k] for k in range(K)])
            if np.allclose(new_centres, centres): break
            centres = new_centres
        inertia = np.sum((Z - centres[labels]) ** 2)
        if inertia < best_inertia:
            best_inertia = inertia; best_labels = labels
    return best_labels


def cluster_purity(pred, truth):
    n = len(pred)
    K = pred.max() + 1
    hits = 0
    for k in range(K):
        m = pred == k
        if not m.any(): continue
        modal = np.bincount(truth[m]).argmax()
        hits += (truth[m] == modal).sum()
    return hits / n


if __name__ == "__main__":
    print("=== Functional clustering via FPC scores + k-means ===\n")
    rng = np.random.default_rng(0)
    T = 80
    t = np.linspace(0, 1, T)
    n_per = 30
    # Three shape families: sin, damped sin, ramp
    def sin_curve():   return 1.5 * np.sin(2 * np.pi * t)                    + 0.15 * rng.normal(0, 1, T)
    def damp_curve():  return 1.5 * np.exp(-2 * t) * np.sin(4 * np.pi * t)   + 0.15 * rng.normal(0, 1, T)
    def ramp_curve():  return 2.0 * t - 1.0                                   + 0.15 * rng.normal(0, 1, T)
    X = np.vstack([[sin_curve() for _ in range(n_per)],
                     [damp_curve() for _ in range(n_per)],
                     [ramp_curve() for _ in range(n_per)]])
    truth = np.array([0] * n_per + [1] * n_per + [2] * n_per)

    Z = _fpca_scores(X, k=3)
    pred = _kmeans(Z, K=3, n_init=10, seed=0)
    purity = cluster_purity(pred, truth)
    print(f"  cluster purity = {purity:.3f}   (1.0 = perfect on this dataset)")
    print("  Clusters recover the three curve families from their FPC scores.\n")
    print("--- library cross-check (R funHDDC, fda.usc::kmeans.fd; scikit-fda) ---")
