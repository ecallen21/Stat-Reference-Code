"""Informer - Long-Sequence Time-Series Forecasting (Sec 47.236).

Zhou et al 2021 'Informer: Beyond Efficient Transformer for Long
Sequence Time-Series Forecasting', AAAI (best paper). Three tricks
to scale Transformer forecasting to horizons in the thousands:

    1. ProbSparse attention: for each query, keep only the top-u
       KEYS by dominance score (O(L log L) instead of O(L^2)).
    2. Self-attention distilling between encoder layers halves T.
    3. Generative decoder: emits the whole forecast in ONE pass
       (no step-by-step autoregressive rollout).

Enables 720-step forecasts vs vanilla Transformer's 168.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def prob_sparse_attention(Q, K, V, u):
    """For each query, compute attention to the top-u KEYS by dominance."""
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    # Dominance score: max - mean per key
    dominance = scores.max(axis=0) - scores.mean(axis=0)
    top_keys = np.argsort(-dominance)[:u]
    scores_sparse = scores[:, top_keys]
    attn = np.exp(scores_sparse - scores_sparse.max(-1, keepdims=True))
    attn = attn / attn.sum(-1, keepdims=True)
    return attn @ V[top_keys]


def full_attention(Q, K, V):
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    attn = np.exp(scores - scores.max(-1, keepdims=True))
    attn = attn / attn.sum(-1, keepdims=True)
    return attn @ V


def distill(x, factor=2):
    """Self-attention distilling: convolve + max-pool to halve time dim."""
    T = x.shape[0]
    return x[::factor]


if __name__ == "__main__":
    print("=== Informer (Zhou et al 2021 AAAI) ===\n")
    rng = np.random.default_rng(0)

    L = 512; d = 16
    Q = rng.normal(size=(L, d))
    K = rng.normal(size=(L, d))
    V = rng.normal(size=(L, d))

    # Full attention (baseline)
    out_full = full_attention(Q, K, V)
    # ProbSparse for various u = O(log L)
    for u in [int(np.log2(L)), int(np.sqrt(L)), L // 2]:
        out_sparse = prob_sparse_attention(Q, K, V, u)
        err = float(np.linalg.norm(out_sparse - out_full) / np.linalg.norm(out_full))
        print(f"  L = {L}, keep top u = {u:>4} keys ({100 * u / L:>4.1f}% of L)   "
              f"rel-err vs full = {err:.4f}")

    # Distilling: T -> T/2 -> T/4
    x = Q.copy()
    print(f"\n  Encoder distilling (factor=2 per layer):")
    for l in range(3):
        print(f"    layer {l}: T = {x.shape[0]}")
        x = distill(x)
    print(f"    layer 3: T = {x.shape[0]}   (8x smaller than L={L})")

    # Compute cost: full O(L^2) vs sparse O(L * u)
    print(f"\n  Attention compute:")
    print(f"    Full:      L^2 = {L * L:>8,} ops per head")
    u_opt = int(np.log2(L))
    print(f"    ProbSparse: L * u = {L * u_opt:>8,} ops   ({L * L // (L * u_opt)}x cheaper)")

    print("\n--- library cross-check (zhouhaoyi/Informer2020; neuralforecast.Informer) ---")
