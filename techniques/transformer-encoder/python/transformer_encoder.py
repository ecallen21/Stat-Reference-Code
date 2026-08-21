"""Transformer encoder block (Reference §27.6).

Composition:
    x' = LayerNorm(x + MultiHeadAttention(x, x, x))            (pre-norm)
    y  = LayerNorm(x' + FeedForward(x'))
where FeedForward = Linear -> GELU -> Linear.

The pre-norm variant (LayerNorm before each sublayer) trains more stably than
the original post-norm formulation.

Positional encoding: sinusoidal (Vaswani 2017) added to input token embeddings.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math

import numpy as np    # numerical arrays + linear algebra


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-6):
    mu = x.mean(axis=-1, keepdims=True); sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sd + eps)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def sinusoidal_encoding(n_pos: int, d_model: int) -> np.ndarray:
    pe = np.zeros((n_pos, d_model))
    pos = np.arange(n_pos)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


def _multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads):
    n, d_model = X.shape
    d_head = d_model // n_heads
    Q = (X @ W_q).reshape(n, n_heads, d_head).transpose(1, 0, 2)
    K = (X @ W_k).reshape(n, n_heads, d_head).transpose(1, 0, 2)
    V = (X @ W_v).reshape(n, n_heads, d_head).transpose(1, 0, 2)
    outs = []
    for h in range(n_heads):
        s = Q[h] @ K[h].T / math.sqrt(d_head)
        w = _softmax(s, axis=-1)
        outs.append(w @ V[h])
    return (np.stack(outs).transpose(1, 0, 2).reshape(n, d_model)) @ W_o


def encoder_block(X, W_q, W_k, W_v, W_o, W_ff1, b_ff1, W_ff2, b_ff2, n_heads):
    """Pre-norm transformer encoder block."""
    # sub-layer 1: multi-head self-attention
    xn = layer_norm(X)
    attn = _multi_head_attention(xn, W_q, W_k, W_v, W_o, n_heads)
    X = X + attn
    # sub-layer 2: position-wise feed-forward
    xn = layer_norm(X)
    ff = gelu(xn @ W_ff1 + b_ff1) @ W_ff2 + b_ff2
    X = X + ff
    return layer_norm(X)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 8; d_model = 32; n_heads = 4; d_ff = 64

    W_q = rng.normal(scale=0.1, size=(d_model, d_model))
    W_k = rng.normal(scale=0.1, size=(d_model, d_model))
    W_v = rng.normal(scale=0.1, size=(d_model, d_model))
    W_o = rng.normal(scale=0.1, size=(d_model, d_model))
    W_ff1 = rng.normal(scale=0.1, size=(d_model, d_ff))
    b_ff1 = np.zeros(d_ff)
    W_ff2 = rng.normal(scale=0.1, size=(d_ff, d_model))
    b_ff2 = np.zeros(d_model)

    # random token embeddings + sinusoidal positional encoding
    tok_emb = rng.normal(scale=0.5, size=(n, d_model))
    pos_enc = sinusoidal_encoding(n, d_model)
    X = tok_emb + pos_enc

    Y = encoder_block(X, W_q, W_k, W_v, W_o, W_ff1, b_ff1, W_ff2, b_ff2, n_heads)

    print(f"=== Transformer encoder block (pre-norm, MHA + FFN + residuals) ===")
    print(f"  input shape        = {X.shape}")
    print(f"  output shape       = {Y.shape}")
    print(f"  input  mean row-var = {X.var(axis=-1).mean():.4f}")
    print(f"  output mean row-var = {Y.var(axis=-1).mean():.4f}")
    print(f"  LayerNorm keeps per-row var ~ 1: {np.round(Y.var(axis=-1), 3).tolist()}")
    print(f"\n  positional encoding sanity: PE[0, :4] = "
          f"{np.round(pos_enc[0, :4], 3).tolist()}  (sin/cos alternation)")
    print(f"  PE[5, :4] = {np.round(pos_enc[5, :4], 3).tolist()}")

    # stack multiple blocks
    Y2 = Y
    for _ in range(3):
        Y2 = encoder_block(Y2, W_q, W_k, W_v, W_o, W_ff1, b_ff1, W_ff2, b_ff2, n_heads)
    print(f"\n  after 4 stacked encoder blocks: shape {Y2.shape}, row-var = "
          f"{Y2.var(axis=-1).mean():.4f}")

    print("\n--- library cross-check (torch.nn.TransformerEncoderLayer) ---")
    try:
        import torch
        import torch.nn as nn
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                          dim_feedforward=d_ff, activation="gelu",
                                          norm_first=True, batch_first=True)
        Xt = torch.tensor(X[None], dtype=torch.float32)
        out = enc(Xt)
        print(f"  torch encoder-layer output shape = {tuple(out.shape)}")
    except ImportError:
        print("  (pytorch not installed)")
