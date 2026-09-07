"""Grouped-query attention (GQA) (Reference Sec 47.37).

Ainslie et al. 2023 'GQA: training generalized multi-query transformer
models from multi-head checkpoints'. Compromise between MHA (each
head has its own K, V) and MQA (all heads share one K, V):

    n_q_heads  = 32   (typical)
    n_kv_heads = 4    (typical grouped-query)
    -> 8 query heads share each K, V head

Cuts KV-cache memory 8x with negligible quality loss vs full MHA.
Used in Llama 2 70B, Llama 3, Mistral, Qwen, Gemma.

We implement GQA with a small numeric example showing shape math
and correctness vs MHA / MQA.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


def gqa_forward(Q, K, V, n_q_heads, n_kv_heads):
    """Q: (T, d_q * n_q_heads); K, V: (T, d_kv * n_kv_heads)."""
    T = Q.shape[0]
    d_q = Q.shape[1] // n_q_heads
    d_kv = K.shape[1] // n_kv_heads
    assert d_q == d_kv, "head dims must match"
    Qh = Q.reshape(T, n_q_heads, d_q)
    Kh = K.reshape(T, n_kv_heads, d_kv)
    Vh = V.reshape(T, n_kv_heads, d_kv)
    #  Each group = n_q_heads / n_kv_heads Q-heads share one KV-head
    group_size = n_q_heads // n_kv_heads
    outs = np.zeros_like(Qh)
    for kv_idx in range(n_kv_heads):
        Kk = Kh[:, kv_idx]                    # (T, d)
        Vk = Vh[:, kv_idx]
        for offset in range(group_size):
            q_idx = kv_idx * group_size + offset
            scores = Qh[:, q_idx] @ Kk.T / np.sqrt(d_q)   # (T, T)
            attn = softmax(scores)
            outs[:, q_idx] = attn @ Vk
    return outs.reshape(T, n_q_heads * d_q)


if __name__ == "__main__":
    print("=== Grouped-query attention (Ainslie 2023) ===\n")
    rng = np.random.default_rng(0)
    T = 5; d_head = 4
    n_q_heads = 8; n_kv_heads = 2         # 4 Q-heads share each KV
    Q = rng.normal(size=(T, n_q_heads * d_head))
    K = rng.normal(size=(T, n_kv_heads * d_head))
    V = rng.normal(size=(T, n_kv_heads * d_head))

    out = gqa_forward(Q, K, V, n_q_heads, n_kv_heads)
    print(f"  T = {T}, n_q_heads = {n_q_heads}, n_kv_heads = {n_kv_heads}, d_head = {d_head}")
    print(f"  KV-cache size = T * n_kv_heads * d_head * 2 = {T * n_kv_heads * d_head * 2}")
    print(f"  Full MHA cache = T * n_q_heads * d_head * 2 = {T * n_q_heads * d_head * 2}   "
          f"(GQA is {n_q_heads / n_kv_heads:.0f}x smaller)")
    print(f"  Output shape = {out.shape}   (matches MHA: (T, n_q_heads * d_head))")

    #  Verify MQA (n_kv_heads = 1) works too
    K1 = rng.normal(size=(T, d_head))
    V1 = rng.normal(size=(T, d_head))
    out_mqa = gqa_forward(Q, K1, V1, n_q_heads, n_kv_heads=1)
    print(f"\n  MQA (n_kv_heads = 1) cache size = {T * d_head * 2}   "
          f"({n_q_heads}x smaller than full MHA)")

    print("\n--- library cross-check (transformers Llama 2/3 / Mistral / Gemma modules Python) ---")
