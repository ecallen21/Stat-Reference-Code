"""Sliding-Window Attention (Reference Sec 47.187).

Beltagy, Peters & Cohan 2020 'Longformer'; Jiang et al 2023
'Mistral 7B'. Restrict each token's attention to a fixed-size
LOCAL WINDOW of the past W tokens instead of the full history:

    attention_mask[i, j] = 1 if 0 <= i - j <= W else 0.

Cost: O(T * W) instead of O(T^2); memory / compute scale linearly
in sequence length. Combined with global 'sink' tokens (Attention
Sinks, Xiao 2024), enables streaming inference over very long
contexts.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


def full_causal_attention(Q, K, V):
    T, d = Q.shape
    scores = Q @ K.T / np.sqrt(d)
    mask = np.triu(np.ones((T, T)), k=1) * -1e9
    return softmax(scores + mask) @ V


def sliding_window_attention(Q, K, V, W, sink_tokens=0):
    """Causal + sliding window: token i attends to [max(0, i - W + 1) : i + 1].

    Optional 'sink': first `sink_tokens` positions always attended (Xiao 2024).
    """
    T, d = Q.shape
    scores = Q @ K.T / np.sqrt(d)
    idx = np.arange(T)
    # In-window OR in-sink
    in_window = (idx[None, :] <= idx[:, None]) & (idx[None, :] >= idx[:, None] - W + 1)
    in_sink = (idx[None, :] < sink_tokens) & (idx[None, :] <= idx[:, None])
    mask = np.where(in_window | in_sink, 0.0, -1e9)
    return softmax(scores + mask) @ V


def flops_per_layer(T, W, d, mode="full"):
    """FLOPs for softmax(QK^T / sqrt(d)) V."""
    if mode == "full":
        return 4 * T * T * d
    return 4 * T * W * d


if __name__ == "__main__":
    print("=== Sliding-Window Attention (Beltagy 2020; Jiang 2023 Mistral) ===\n")
    rng = np.random.default_rng(0)

    T, d = 128, 32
    Q = rng.normal(scale=0.5, size=(T, d))
    K = rng.normal(scale=0.5, size=(T, d))
    V = rng.normal(scale=0.5, size=(T, d))

    out_full = full_causal_attention(Q, K, V)

    for W in [8, 32, 64, 128]:
        out_sw = sliding_window_attention(Q, K, V, W=W, sink_tokens=0)
        rel_diff = float(np.linalg.norm(out_sw - out_full) / np.linalg.norm(out_full))
        print(f"  W = {W:3d}   rel-diff vs full attention = {rel_diff:.4f}   "
              f"kept-fraction = {min(W / T, 1.0):.2f}")

    # Add attention sink of 4 tokens (Xiao 2024 StreamingLLM finding)
    out_sw_sink = sliding_window_attention(Q, K, V, W=16, sink_tokens=4)
    out_sw_nosink = sliding_window_attention(Q, K, V, W=16, sink_tokens=0)
    print(f"\n  W = 16, without sink: rel-diff = "
          f"{np.linalg.norm(out_sw_nosink - out_full) / np.linalg.norm(out_full):.4f}")
    print(f"  W = 16, +sink (4 tok): rel-diff = "
          f"{np.linalg.norm(out_sw_sink - out_full) / np.linalg.norm(out_full):.4f}")

    print(f"\n  Compute scaling for T = 8192, d = 128:")
    print(f"    Full attention: {flops_per_layer(8192, None, 128, mode='full') / 1e9:.2f} GFLOPS / layer")
    print(f"    Sliding W = 512: {flops_per_layer(8192, 512, 128, mode='sw') / 1e9:.2f} GFLOPS / layer  (16x cheaper)")
    print(f"    Sliding W = 128: {flops_per_layer(8192, 128, 128, mode='sw') / 1e9:.2f} GFLOPS / layer  (64x cheaper)")

    print("\n--- library cross-check (transformers Mistral / Longformer / FlashAttention) ---")
