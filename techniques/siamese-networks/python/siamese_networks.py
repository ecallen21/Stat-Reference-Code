"""Siamese Networks (Reference Sec 47.152).

Bromley et al 1994 'Signature Verification Using a Siamese Time
Delay Neural Network'; Koch, Zemel & Salakhutdinov 2015 'Siamese
Neural Networks for One-shot Image Recognition'. Two identical
sub-networks share weights and process a PAIR of inputs; the head
compares embeddings:

    L_contrastive(x1, x2, y) = y * D^2 + (1 - y) * max(0, m - D)^2

with D = ||f(x1) - f(x2)||_2, y=1 for same-class pairs. Enables
one-shot / few-shot learning by scoring similarity against
reference exemplars.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def contrastive_loss(D, y, margin=1.0):
    """Hadsell-Chopra-LeCun 2006 contrastive loss."""
    return y * D ** 2 + (1 - y) * np.maximum(0, margin - D) ** 2


def train_siamese_linear(X, y, dim=2, margin=1.0, lr=0.005, n_iter=5000, seed=0):
    """Learn a linear embedding W via contrastive loss on random same / diff pairs."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = rng.normal(scale=0.5, size=(d, dim))
    for it in range(n_iter):
        # Balanced positive / negative sampling
        if rng.uniform() < 0.5:                                # positive pair
            c = int(rng.choice(np.unique(y)))
            i, j = rng.choice(np.where(y == c)[0], size=2, replace=False)
            same = 1.0
        else:                                                   # negative pair
            c1, c2 = rng.choice(np.unique(y), size=2, replace=False)
            i = int(rng.choice(np.where(y == c1)[0]))
            j = int(rng.choice(np.where(y == c2)[0]))
            same = 0.0
        x1, x2 = X[i], X[j]
        f1, f2 = x1 @ W, x2 @ W
        diff = f1 - f2
        D = float(np.linalg.norm(diff) + 1e-8)
        if same:
            grad = 2 * np.outer(x1 - x2, diff)                  # push closer
        else:
            if D < margin:
                grad = -2 * (margin - D) / D * np.outer(x1 - x2, diff)
            else:
                grad = 0.0
        W -= lr * grad
    return W


def one_shot_accuracy(X_emb, y, k_way=3, seed=0):
    """K-way one-shot: for each query, score similarity vs 1 exemplar per class."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    correct = 0; total = 0
    for _ in range(500):
        chosen = rng.choice(classes, size=k_way, replace=False)
        support = [int(rng.choice(np.where(y == c)[0])) for c in chosen]
        target_c = int(rng.choice(chosen))
        query = int(rng.choice([q for q in np.where(y == target_c)[0] if q not in support]))
        dists = [float(np.linalg.norm(X_emb[query] - X_emb[s])) for s in support]
        pred_c = chosen[int(np.argmin(dists))]
        correct += (pred_c == target_c)
        total += 1
    return correct / total


if __name__ == "__main__":
    print("=== Siamese Networks (Bromley 1994; Koch et al 2015) ===\n")
    from sklearn.datasets import load_iris

    data = load_iris(); X, y = data.data, data.target
    X_scaled = (X - X.mean(0)) / (X.std(0) + 1e-8)
    print(f"  Iris:  n = {len(X)}, d = {X.shape[1]}, k_way = 3 one-shot task")

    # Baseline: 3-way one-shot in raw feature space
    acc_raw = one_shot_accuracy(X_scaled, y, k_way=3, seed=0)
    print(f"    3-way one-shot acc, raw features  = {acc_raw:.3f}")

    # Siamese embedding
    W = train_siamese_linear(X_scaled, y, dim=2, margin=1.5, lr=0.01,
                                n_iter=8000, seed=0)
    acc_sia = one_shot_accuracy(X_scaled @ W, y, k_way=3, seed=0)
    print(f"    3-way one-shot acc, Siamese 2-D   = {acc_sia:.3f}")

    # Noisy synthetic: harder case
    rng = np.random.default_rng(0)
    n_per = 60; K = 4; d = 20
    y_s = np.repeat(np.arange(K), n_per)
    means = rng.normal(scale=3.0, size=(K, 2))
    signal = means[y_s]
    noise = rng.normal(scale=2.5, size=(K * n_per, d - 2))
    X_s = np.column_stack([signal, noise])
    acc_raw = one_shot_accuracy(X_s, y_s, k_way=4, seed=0)
    W = train_siamese_linear(X_s - X_s.mean(0), y_s, dim=2, margin=1.5, lr=0.001,
                                n_iter=15000, seed=0)
    acc_sia = one_shot_accuracy((X_s - X_s.mean(0)) @ W, y_s, k_way=4, seed=0)
    print(f"\n  Noisy synthetic (K = 4, d = 20, signal in first 2 dims):")
    print(f"    4-way one-shot acc, raw            = {acc_raw:.3f}")
    print(f"    4-way one-shot acc, Siamese 2-D   = {acc_sia:.3f}")

    print("\n--- library cross-check (pytorch-metric-learning / keras Siamese Python) ---")
