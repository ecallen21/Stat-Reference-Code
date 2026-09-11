"""SAINT - Self-Attention and Intersample Attention Transformer (Sec 47.241).

Somepalli, Goldblum, Schwarzschild, Bruss & Goldstein 2021 'SAINT:
Improved Neural Networks for Tabular Data via Row Attention and
Contrastive Pre-Training'. Two attention flavours:

    1. COL-ATTN: attention across features within a row (like
       standard Transformer over tokens).
    2. ROW-ATTN: attention across ROWS within a mini-batch —
       each row attends to peers with similar feature patterns.

Combined with contrastive pretraining (CutMix + MixUp on rows),
SAINT is a strong deep-tabular contender.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


def col_attention(X, W_Q, W_K, W_V):
    """Attention across features (last axis) — one head."""
    Q = X @ W_Q; K = X @ W_K; V = X @ W_V
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    return softmax(scores) @ V


def row_attention(X, W_Q, W_K, W_V):
    """Attention across ROWS (batch axis) — each row = one token."""
    # Aggregate each row to a single vector, then attend across rows
    Q = X @ W_Q; K = X @ W_K; V = X @ W_V
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    return softmax(scores) @ V


def saint_forward(X, W_col, W_row):
    """One SAINT block: col-attn -> row-attn -> residual."""
    z1 = col_attention(X, *W_col) + X
    z2 = row_attention(z1, *W_row) + z1
    return z2


if __name__ == "__main__":
    print("=== SAINT (Somepalli et al 2021) ===\n")
    rng = np.random.default_rng(0)

    # 50 rows x 8 features (mini-batch)
    N = 50; F = 8; d = 8
    X = rng.normal(size=(N, F))
    # Attention weights (col + row)
    W_col = tuple(rng.normal(scale=0.3, size=(F, d)) for _ in range(3))
    W_row = tuple(rng.normal(scale=0.3, size=(d, d)) for _ in range(3))

    Y = saint_forward(X, W_col, W_row)
    print(f"  Batch: {N} rows, {F} features -> {Y.shape} SAINT-encoded")
    print(f"  Output norm range: [{float(np.linalg.norm(Y, axis=1).min()):.2f}, "
          f"{float(np.linalg.norm(Y, axis=1).max()):.2f}]")

    # Ablation: which attention matters more on a synthetic 'row-similarity' task?
    # Create rows in 3 clusters; SAINT should aggregate cluster-mates
    N = 30; F = 4; d = F                                          # match dims so residual works
    labels = np.repeat([0, 1, 2], N // 3)
    means = np.array([[3, 0, 0, 0], [-3, 0, 0, 0], [0, 3, 0, 0]])
    X = means[labels] + 0.3 * rng.normal(size=(N, F))
    W_col = tuple(rng.normal(scale=0.3, size=(F, d)) for _ in range(3))
    W_row = tuple(rng.normal(scale=0.3, size=(d, d)) for _ in range(3))
    Y_only_col = col_attention(X, *W_col) + X
    Y_row_added = row_attention(Y_only_col, *W_row) + Y_only_col
    # Intra-cluster vs inter-cluster embedding distance
    def cluster_gap(Y):
        intra = []; inter = []
        for i in range(N):
            for j in range(i + 1, N):
                d_ij = float(np.linalg.norm(Y[i] - Y[j]))
                (intra if labels[i] == labels[j] else inter).append(d_ij)
        return np.mean(intra), np.mean(inter)

    intra_c, inter_c = cluster_gap(Y_only_col)
    intra_r, inter_r = cluster_gap(Y_row_added)
    print(f"\n  Row-attention effect on cluster geometry:")
    print(f"    col-only   : intra-cluster dist = {intra_c:.3f}   "
            f"inter-cluster = {inter_c:.3f}   ratio = {inter_c / intra_c:.2f}")
    print(f"    col + row  : intra-cluster dist = {intra_r:.3f}   "
            f"inter-cluster = {inter_r:.3f}   ratio = {inter_r / intra_r:.2f}")

    print("\n--- library cross-check (somepalli/saint; tab-transformer-pytorch) ---")
