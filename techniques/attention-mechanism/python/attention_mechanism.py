"""Scaled dot-product attention + multi-head attention (Reference §27.5).

Given queries Q (n_q, d_k), keys K (n_k, d_k), values V (n_k, d_v):

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Multi-head attention: run h heads in parallel with different linear projections
of Q, K, V and concatenate outputs.

Uses:
  * Self-attention: Q = K = V from the same source (transformer encoder).
  * Cross-attention: Q from decoder, K/V from encoder.
  * Causal mask: upper-triangular -inf mask for autoregressive generation.

Demo: content-addressable memory - a query retrieves the closest key's value.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None) -> dict:
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = _softmax(scores, axis=-1)
    return {"output": weights @ V, "attention": weights}


def multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads: int) -> dict:
    """X: (n, d_model). W_q/k/v: (d_model, d_model); W_o: (d_model, d_model).
    Split each projection into n_heads with d_head = d_model / n_heads."""
    n, d_model = X.shape
    d_head = d_model // n_heads
    Q = (X @ W_q).reshape(n, n_heads, d_head).transpose(1, 0, 2)   # (h, n, d_head)
    K = (X @ W_k).reshape(n, n_heads, d_head).transpose(1, 0, 2)
    V = (X @ W_v).reshape(n, n_heads, d_head).transpose(1, 0, 2)
    outs = []
    all_weights = []
    for h in range(n_heads):
        out = scaled_dot_product_attention(Q[h], K[h], V[h])
        outs.append(out["output"]); all_weights.append(out["attention"])
    concat = np.stack(outs).transpose(1, 0, 2).reshape(n, d_model)
    return {"output": concat @ W_o, "attention_per_head": all_weights}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Demo: 4 keys with distinct values; query is a corrupted copy of key 2
    d_k = 8; d_v = 4
    K = rng.normal(size=(4, d_k))
    V = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    Q = K[2:3] + 0.1 * rng.normal(size=(1, d_k))          # noisy copy of key 2

    r = scaled_dot_product_attention(Q, K, V)
    print("=== Scaled dot-product attention: query ~ key 2 ===")
    print(f"  attention weights = {np.round(r['attention'][0], 3).tolist()}   "
          f"(expected ~ 1 on position 2)")
    print(f"  output            = {np.round(r['output'][0], 3).tolist()}   "
          f"(expected ~ V[2] = [0, 0, 1, 0])")

    # multi-head on a sequence
    n = 6; d_model = 16; n_heads = 4
    X = rng.normal(size=(n, d_model))
    W_q = rng.normal(scale=0.2, size=(d_model, d_model))
    W_k = rng.normal(scale=0.2, size=(d_model, d_model))
    W_v = rng.normal(scale=0.2, size=(d_model, d_model))
    W_o = rng.normal(scale=0.2, size=(d_model, d_model))
    mh = multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads=n_heads)
    print(f"\n=== Multi-head self-attention (n={n}, d_model={d_model}, heads={n_heads}) ===")
    print(f"  output shape           = {mh['output'].shape}")
    print(f"  attention per-head[0] row sums = "
          f"{np.round(mh['attention_per_head'][0].sum(axis=-1), 3).tolist()}   "
          f"(should each be 1.000)")

    # causal mask demo
    mask = np.tril(np.ones((n, n), dtype=bool))
    causal = scaled_dot_product_attention(X, X, X, mask=mask)
    print(f"\n=== Causal mask (autoregressive) ===")
    print(f"  attention[3, :]: {np.round(causal['attention'][3], 3).tolist()}   "
          f"(positions > 3 should be 0)")

    print("\n--- library cross-check (torch.nn.functional.scaled_dot_product_attention) ---")
    try:
        import torch
        import torch.nn.functional as F
        Qt = torch.tensor(Q); Kt = torch.tensor(K); Vt = torch.tensor(V)
        out = F.scaled_dot_product_attention(Qt, Kt, Vt).numpy()
        print(f"  torch output       = {np.round(out[0], 3).tolist()}")
    except ImportError:
        print("  (pytorch not installed)")
