"""Multi-Query Attention - MQA (Reference Sec 47.186).

Shazeer 2019 'Fast Transformer Decoding: One Write-Head is All
You Need'. Multi-Head Attention (MHA) has H independent (Q, K, V)
projections. MQA keeps H query heads but SHARES a single K, V
across all heads:

    MHA:   K, V shape (H, T, d_head)
    MQA:   K, V shape (   T, d_head)      -- 1/H the memory

Grouped-Query Attention (Ainslie 2023) generalises to G groups.
MQA has minor quality loss but dramatically reduces the KV-cache
memory and bandwidth in autoregressive inference.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


def mha_attention(Q, K, V):
    """Q, K, V shape (H, T, d). Standard MHA: each head has its own K, V."""
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(Q.shape[-1])
    return softmax(scores) @ V


def mqa_attention(Q, K, V):
    """Q shape (H, T, d), K, V shape (T, d) shared across all H heads."""
    H, T, d = Q.shape
    scores = Q @ K.T / np.sqrt(d)                               # (H, T, T)
    return softmax(scores) @ V


def gqa_attention(Q, K, V, n_groups):
    """Grouped-query: H heads split into `n_groups` groups; each group shares K, V."""
    H, T, d = Q.shape
    # Broadcast K, V from (G, T, d) to (H, T, d)
    heads_per_group = H // n_groups
    K_full = np.repeat(K, heads_per_group, axis=0)
    V_full = np.repeat(V, heads_per_group, axis=0)
    return mha_attention(Q, K_full, V_full)


def kv_cache_bytes(H, T, d, bytes_per_elem=2, mode="mha", n_groups=None):
    """Per-layer KV cache size."""
    if mode == "mha":
        return 2 * H * T * d * bytes_per_elem                   # K + V, per head
    if mode == "mqa":
        return 2 * T * d * bytes_per_elem                       # shared K, V
    if mode == "gqa":
        return 2 * n_groups * T * d * bytes_per_elem
    raise ValueError(mode)


if __name__ == "__main__":
    print("=== Multi-Query Attention / GQA (Shazeer 2019, Ainslie 2023) ===\n")

    rng = np.random.default_rng(0)
    H, T, d = 32, 128, 64
    Q = rng.normal(scale=0.5, size=(H, T, d))
    K_mha = rng.normal(scale=0.5, size=(H, T, d))
    V_mha = rng.normal(scale=0.5, size=(H, T, d))
    K_shared = K_mha.mean(axis=0)                                # simulate MQA / GQA
    V_shared = V_mha.mean(axis=0)

    out_mha = mha_attention(Q, K_mha, V_mha)
    out_mqa = mqa_attention(Q, K_shared, V_shared)
    # GQA with 8 groups: 4 heads per group
    K_gqa = K_mha.reshape(8, 4, T, d).mean(axis=1)              # (G, T, d)
    V_gqa = V_mha.reshape(8, 4, T, d).mean(axis=1)
    out_gqa = gqa_attention(Q, K_gqa, V_gqa, n_groups=8)

    print(f"  H = {H} heads, T = {T} tokens, d_head = {d}")
    print(f"  Output shape (all three): {out_mha.shape}")
    print(f"  MHA vs MQA output diff = {float(np.linalg.norm(out_mha - out_mqa)):.3f}")
    print(f"  MHA vs GQA output diff = {float(np.linalg.norm(out_mha - out_gqa)):.3f}   (closer to MHA)")

    print(f"\n  KV-cache memory per layer at T = 2048, d = 128, H = 32 (FP16):")
    print(f"    MHA:  {kv_cache_bytes(32, 2048, 128, mode='mha'):>10,} bytes")
    print(f"    GQA-8: {kv_cache_bytes(32, 2048, 128, mode='gqa', n_groups=8):>9,} bytes  (4x smaller)")
    print(f"    GQA-4: {kv_cache_bytes(32, 2048, 128, mode='gqa', n_groups=4):>9,} bytes  (8x smaller)")
    print(f"    MQA:   {kv_cache_bytes(32, 2048, 128, mode='mqa'):>9,} bytes  (32x smaller)")

    print("\n  MQA-only quality drop is small (~0.5 pp perplexity); GQA (LLaMA-2, Mistral)")
    print("  gets ~all the benefit at 8-1 grouping.")

    print("\n--- library cross-check (transformers LlamaAttention.num_key_value_heads Python) ---")
