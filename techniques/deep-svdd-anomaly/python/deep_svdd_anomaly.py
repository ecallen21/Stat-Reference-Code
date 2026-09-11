"""Deep SVDD - Deep Support Vector Data Description (Sec 47.240).

Ruff et al 2018 'Deep One-Class Classification', ICML. Learn a
neural network phi_theta that maps normal training data into a
compact hypersphere of centre c in the embedding space:

    L = (1 / n) * sum_i ||phi_theta(x_i) - c||^2  +  lambda * ||theta||^2

Anomaly score at test time = distance from c:

    s(x) = ||phi_theta(x) - c||

Unsupervised; only NORMAL data at training time. Related to OC-SVM
but with learned features.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def phi_toy(x, W):
    """Toy encoder: single linear layer with tanh."""
    return np.tanh(x @ W)


def train_deep_svdd(X_normal, n_iter=500, lr=0.05, seed=0):
    """Minimize distance to centre c (set to mean of initial embeddings)."""
    rng = np.random.default_rng(seed)
    d_in = X_normal.shape[1]; d_out = 8
    W = rng.normal(scale=0.3, size=(d_in, d_out))
    # Set c = mean of initial embeddings (must be nonzero to avoid trivial solution)
    c = phi_toy(X_normal, W).mean(axis=0)
    c = np.where(np.abs(c) < 0.01, 0.01, c)                      # avoid zeros
    for it in range(n_iter):
        embed = phi_toy(X_normal, W)
        # Grad: ||embed - c||^2 -> gradient in W direction shrinks embedding toward c
        diff = embed - c                                          # (n, d_out)
        # Approx grad wrt W: X.T @ (diff * (1 - embed^2))
        grad = X_normal.T @ (diff * (1 - embed ** 2)) / len(X_normal)
        W -= lr * grad
    return W, c


def svdd_anomaly_score(X, W, c):
    embed = phi_toy(X, W)
    return np.linalg.norm(embed - c, axis=1)


if __name__ == "__main__":
    print("=== Deep SVDD (Ruff et al 2018 ICML) ===\n")
    rng = np.random.default_rng(0)

    # Normal data: cluster around (0, 0); anomalies: scattered
    n_normal = 200; d = 4
    X_normal = rng.normal(size=(n_normal, d)) * 0.5
    X_anomaly = rng.normal(size=(30, d)) * 2.0 + 3.0             # shifted

    W, c = train_deep_svdd(X_normal, n_iter=500, lr=0.05, seed=0)

    normal_scores = svdd_anomaly_score(X_normal, W, c)
    anomaly_scores = svdd_anomaly_score(X_anomaly, W, c)

    threshold = float(np.percentile(normal_scores, 95))
    tp = int((anomaly_scores > threshold).sum())
    fp = int((normal_scores > threshold).sum())
    print(f"  {n_normal} normal + 30 anomaly samples in R^{d}, encoder d_out=8")
    print(f"  Normal-score  mean = {float(normal_scores.mean()):.3f}   "
            f"std = {float(normal_scores.std()):.3f}")
    print(f"  Anomaly-score mean = {float(anomaly_scores.mean()):.3f}   "
            f"std = {float(anomaly_scores.std()):.3f}")
    print(f"  Threshold at normal-95 %ile = {threshold:.3f}:")
    print(f"    True positives (anomalies flagged):  {tp} / 30")
    print(f"    False positives (normals flagged):   {fp} / {n_normal}")

    print("\n--- library cross-check (deep-svdd repo; pyod.models.DeepSVDD) ---")
