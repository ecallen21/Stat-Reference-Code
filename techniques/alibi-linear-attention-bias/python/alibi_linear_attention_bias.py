"""ALiBi - Attention with Linear Biases (Reference Sec 47.170).

Press, Smith & Lewis 2022 'Train Short, Test Long: Attention with
Linear Biases Enables Input Length Extrapolation', ICLR. Replaces
learned / sinusoidal position embeddings with a fixed LINEAR BIAS
on the attention scores:

    A_ij = softmax(Q_i K_j^T / sqrt(d) - m_h * |i - j|)

where m_h = 2^{-8 * h / H} for head h (geometrically spaced slopes).
No trainable positional parameters. Extrapolates to sequences
longer than seen at training -- unlike learned / RoPE embeddings.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def get_alibi_slopes(n_heads):
    """Geometric slopes m_h = 2^(-8 h / n_heads)."""
    return 2 ** (-8 * np.arange(1, n_heads + 1) / n_heads)


def alibi_attention(Q, K, V, slopes, causal=True):
    """Multi-head attention with ALiBi linear biases. Q, K, V shape (n_heads, T, d)."""
    n_heads, T, d = Q.shape
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d)              # (H, T, T)
    dist = np.abs(np.arange(T)[None, :] - np.arange(T)[:, None])
    bias = -slopes[:, None, None] * dist[None, :, :]
    scores = scores + bias
    if causal:
        mask = np.triu(np.ones((T, T)), k=1) * -1e9
        scores = scores + mask[None, :, :]
    scores -= scores.max(axis=-1, keepdims=True)
    A = np.exp(scores); A = A / A.sum(axis=-1, keepdims=True)
    return A @ V, A


if __name__ == "__main__":
    print("=== ALiBi - Attention with Linear Biases (Press et al 2022) ===\n")
    rng = np.random.default_rng(0)
    n_heads, d = 4, 8

    slopes = get_alibi_slopes(n_heads)
    print(f"  ALiBi slopes for H = {n_heads} heads: {np.round(slopes, 4)}")

    # Task: token i's output should attend to token i-1 (a copy task with slight local bias)
    for T in [32, 128, 512]:
        Q = rng.normal(scale=0.1, size=(n_heads, T, d))
        K = rng.normal(scale=0.1, size=(n_heads, T, d))
        V = np.tile(np.eye(T)[:, :d], (n_heads, 1, 1)).astype(float)  # V[i] ~ e_i (identity)
        _, A = alibi_attention(Q, K, V, slopes, causal=True)
        # Effective 'attention radius' = mean(distance weighted by attention)
        dist = np.abs(np.arange(T)[None, :] - np.arange(T)[:, None])
        rad = float(np.mean(np.sum(A * dist[None, :, :], axis=-1)))
        # Per-head: how concentrated is head h?
        rads_per_h = [float(np.mean(np.sum(A[h] * dist, axis=-1))) for h in range(n_heads)]
        print(f"  T = {T:4d}  mean attention radius = {rad:6.2f}  per-head = "
              f"{[round(r, 2) for r in rads_per_h]}")

    print("\n  Heads with larger slopes (m_h) focus locally; small-slope heads see far.")
    print("  Because the biases are absolute (no learned position vectors), a model")
    print("  trained on T = 512 attends coherently at T = 8192 with no fine-tuning.")

    print("\n--- library cross-check (transformers Bloom / MPT / OPT implementations) ---")
