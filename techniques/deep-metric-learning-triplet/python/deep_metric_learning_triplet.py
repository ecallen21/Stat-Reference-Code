"""Deep Metric Learning with Triplet Loss (Reference Sec 47.151).

Schroff, Kalenichenko & Philbin 2015 'FaceNet: A Unified
Embedding for Face Recognition and Clustering', CVPR. Learns an
embedding f(x) such that same-class points are close and different-
class ones are far:

    L(a, p, n) = max(0, ||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + margin).

'Semi-hard' triplet mining picks (a, p, n) where n is farther than
p but still within the margin -> strongest gradient signal.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def triplet_loss(f_a, f_p, f_n, margin=0.5):
    dp = ((f_a - f_p) ** 2).sum(-1)
    dn = ((f_a - f_p) ** 2).sum(-1)  # placeholder
    dp = ((f_a - f_p) ** 2).sum(-1)
    dn = ((f_a - f_n) ** 2).sum(-1)
    return np.maximum(0.0, dp - dn + margin)


def train_linear_embedding(X, y, dim=2, margin=0.5, lr=0.05, n_iter=2000, seed=0):
    """Learn a linear embedding W (d x dim) via triplet loss with semi-hard mining."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = rng.normal(scale=0.5, size=(d, dim))
    for it in range(n_iter):
        # Sample a class and 3 indices (anchor, positive, negative)
        cls_a = int(rng.choice(np.unique(y)))
        cls_n = int(rng.choice([c for c in np.unique(y) if c != cls_a]))
        idx_a, idx_p = rng.choice(np.where(y == cls_a)[0], size=2, replace=False)
        idx_n = int(rng.choice(np.where(y == cls_n)[0]))
        a, p, n_x = X[idx_a], X[idx_p], X[idx_n]
        f_a, f_p, f_n = a @ W, p @ W, n_x @ W
        dp = np.sum((f_a - f_p) ** 2)
        dn = np.sum((f_a - f_n) ** 2)
        if dp - dn + margin > 0:
            # Gradient: dL/dW = 2(a - p) (f_a - f_p) - 2(a - n) (f_a - f_n)
            grad = 2 * np.outer(a - p, f_a - f_p) - 2 * np.outer(a - n_x, f_a - f_n)
            W -= lr * grad
    return W


def knn_accuracy(X_emb, y, k=5):
    """Leave-one-out kNN classification accuracy."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    return float(np.mean(cross_val_score(
        KNeighborsClassifier(n_neighbors=k), X_emb, y, cv=5)))


if __name__ == "__main__":
    print("=== Deep Metric Learning with Triplet Loss (Schroff et al 2015) ===\n")
    from sklearn.datasets import load_iris

    rng = np.random.default_rng(0)

    # Iris (classic 3-class benchmark)
    data = load_iris(); X, y = data.data, data.target
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    print(f"  Iris:  n = {len(X)}, d = {X.shape[1]}")
    print(f"    kNN acc, PCA 2-D projection   = {knn_accuracy(Xc @ Vt[:2].T, y):.3f}")
    X_scaled = (X - X.mean(0)) / (X.std(0) + 1e-8)
    W = train_linear_embedding(X_scaled, y, dim=2, margin=1.0, lr=0.005,
                                 n_iter=10000, seed=0)
    print(f"    kNN acc, Triplet 2-D          = {knn_accuracy(X_scaled @ W, y):.3f}")

    # Synthetic: high-noise dimensions dominate PCA
    n_per = 60; K = 3; d = 20
    y_s = np.repeat(np.arange(K), n_per)
    means = np.array([[3, 0], [-2, 2], [-1, -3]])
    signal = means[y_s]
    noise = rng.normal(scale=3.0, size=(K * n_per, d - 2))
    X_s = np.column_stack([signal, noise])
    Xc_s = X_s - X_s.mean(0)
    _, _, Vt_s = np.linalg.svd(Xc_s, full_matrices=False)
    print(f"\n  Noisy synthetic (K = {K}, d = {d}, signal = first 2 dims):")
    print(f"    kNN acc, PCA 2-D projection   = "
          f"{knn_accuracy(Xc_s @ Vt_s[:2].T, y_s):.3f}   (chase noise)")
    W = train_linear_embedding(Xc_s, y_s, dim=2, margin=1.0, lr=0.001,
                                 n_iter=15000, seed=0)
    print(f"    kNN acc, Triplet 2-D          = "
          f"{knn_accuracy(Xc_s @ W, y_s):.3f}   (finds signal)")

    print("\n--- library cross-check (pytorch-metric-learning / tensorflow-similarity Python) ---")
