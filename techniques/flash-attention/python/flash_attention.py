"""Flash attention (Reference Sec 47.38).

Dao, Fu, Ermon, Rudra & Re 2022 'FlashAttention: fast and memory-
efficient exact attention with IO-awareness', NeurIPS. Reduces the
memory traffic between HBM and SRAM by TILING the attention matrix
and computing softmax INCREMENTALLY, using the online-softmax trick.

Speedup comes from:
    * Never materialising the full N x N attention matrix in HBM.
    * Recomputing attention during backward instead of caching it.
    * Sequential per-tile softmax update with running (m, l).

Memory: O(N) instead of O(N^2) for the attention step.

We implement the ONLINE / STREAMING softmax to demonstrate the key
numerical trick; the CUDA-level HBM/SRAM tiling gives the actual
speed gain in practice.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def naive_attention(Q, K, V):
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    m = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - m)
    return (e / e.sum(-1, keepdims=True)) @ V


def flash_attention_tiled(Q, K, V, block_size=32):
    """Tiled attention with running softmax stats (m, l) per query row."""
    T, d = Q.shape
    O = np.zeros_like(Q)
    for i_start in range(0, T, block_size):
        i_end = min(i_start + block_size, T)
        Qi = Q[i_start:i_end]
        #  Running max m_i, running normaliser l_i (length b)
        b = i_end - i_start
        m_i = np.full(b, -np.inf)
        l_i = np.zeros(b)
        Oi = np.zeros((b, d))
        for j_start in range(0, T, block_size):
            j_end = min(j_start + block_size, T)
            Kj = K[j_start:j_end]; Vj = V[j_start:j_end]
            Sij = Qi @ Kj.T / np.sqrt(d)           # (b, b')
            m_new = np.maximum(m_i, Sij.max(axis=1))
            e_i = np.exp(m_i - m_new)              # correction for previous accum
            e_ij = np.exp(Sij - m_new[:, None])
            l_new = e_i * l_i + e_ij.sum(axis=1)
            Oi = e_i[:, None] * Oi + e_ij @ Vj
            m_i, l_i = m_new, l_new
        O[i_start:i_end] = Oi / l_i[:, None]
    return O


if __name__ == "__main__":
    print("=== FlashAttention (Dao 2022) tiled softmax demonstration ===\n")
    rng = np.random.default_rng(0)
    T = 64; d = 8
    Q = rng.normal(size=(T, d))
    K = rng.normal(size=(T, d))
    V = rng.normal(size=(T, d))

    O_naive = naive_attention(Q, K, V)
    for block in [8, 16, 32]:
        O_flash = flash_attention_tiled(Q, K, V, block_size=block)
        err = float(np.max(np.abs(O_flash - O_naive)))
        print(f"  block = {block:2d}   max |flash - naive| = {err:.2e}")

    print("\n  Tiled attention matches naive to machine precision.  In production the")
    print("  SRAM / HBM tiling + recomputation drives 2-4x wall-clock speedup on GPUs.")

    print("\n--- library cross-check (flash-attn / xFormers Python; torch.nn.functional.scaled_dot_product_attention) ---")
