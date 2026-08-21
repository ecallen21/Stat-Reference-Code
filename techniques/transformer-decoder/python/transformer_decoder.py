"""Transformer decoder block (Reference §27.x extra).

Standard Vaswani-style decoder block, pre-norm variant:

    x1 = LayerNorm(x)
    x  = x + MaskedMultiHeadSelfAttention(x1)              (causal mask)
    x1 = LayerNorm(x)
    x  = x + MultiHeadCrossAttention(x1, memory)           (encoder-decoder attention)
    x1 = LayerNorm(x)
    x  = x + FeedForward(x1)

For DECODER-ONLY (GPT / LLaMA) drop the cross-attention.
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


def _mha(Q_src, K_src, V_src, W_q, W_k, W_v, W_o, n_heads, causal: bool = False):
    n_q, d_model = Q_src.shape
    d_head = d_model // n_heads
    Q = (Q_src @ W_q).reshape(n_q, n_heads, d_head).transpose(1, 0, 2)
    K = (K_src @ W_k).reshape(K_src.shape[0], n_heads, d_head).transpose(1, 0, 2)
    V = (V_src @ W_v).reshape(V_src.shape[0], n_heads, d_head).transpose(1, 0, 2)
    outs = []
    for h in range(n_heads):
        s = Q[h] @ K[h].T / math.sqrt(d_head)
        if causal:
            mask = np.triu(np.ones((n_q, K.shape[1]), dtype=bool), k=1)
            s = np.where(mask, -1e9, s)
        w = _softmax(s, axis=-1)
        outs.append(w @ V[h])
    return (np.stack(outs).transpose(1, 0, 2).reshape(n_q, d_model)) @ W_o


def decoder_block(x, memory, params, n_heads):
    p = params
    # sub-layer 1: masked self-attention
    xn = layer_norm(x)
    attn = _mha(xn, xn, xn, p["W_q1"], p["W_k1"], p["W_v1"], p["W_o1"],
                 n_heads, causal=True)
    x = x + attn
    # sub-layer 2: cross-attention with encoder memory (skip if memory is None)
    if memory is not None:
        xn = layer_norm(x)
        cross = _mha(xn, memory, memory, p["W_q2"], p["W_k2"], p["W_v2"], p["W_o2"],
                      n_heads, causal=False)
        x = x + cross
    # sub-layer 3: FFN
    xn = layer_norm(x)
    ff = gelu(xn @ p["W_ff1"] + p["b_ff1"]) @ p["W_ff2"] + p["b_ff2"]
    return layer_norm(x + ff)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 6; m = 10; d_model = 32; n_heads = 4; d_ff = 64

    def _init(shape): return rng.normal(scale=0.1, size=shape)
    params = {
        "W_q1": _init((d_model, d_model)), "W_k1": _init((d_model, d_model)),
        "W_v1": _init((d_model, d_model)), "W_o1": _init((d_model, d_model)),
        "W_q2": _init((d_model, d_model)), "W_k2": _init((d_model, d_model)),
        "W_v2": _init((d_model, d_model)), "W_o2": _init((d_model, d_model)),
        "W_ff1": _init((d_model, d_ff)), "b_ff1": np.zeros(d_ff),
        "W_ff2": _init((d_ff, d_model)), "b_ff2": np.zeros(d_model),
    }
    x = rng.normal(scale=0.5, size=(n, d_model))
    memory = rng.normal(scale=0.5, size=(m, d_model))

    # encoder-decoder variant
    y_enc_dec = decoder_block(x, memory, params, n_heads)
    # decoder-only (causal LM) variant
    y_dec_only = decoder_block(x, None, params, n_heads)

    print(f"=== Transformer decoder block ===")
    print(f"  input shape                    = {x.shape}")
    print(f"  memory (encoder output) shape = {memory.shape}")
    print(f"  encoder-decoder output shape  = {y_enc_dec.shape}")
    print(f"  decoder-only  output shape    = {y_dec_only.shape}")
    print(f"  LayerNorm keeps row-var = 1:   {np.round(y_enc_dec.var(axis=-1), 3).tolist()}")

    # verify causality: token 3 should not see tokens 4/5.  Check by perturbing
    # x[4] and confirming y[3] is unchanged.
    x2 = x.copy()
    # Perturb only some entries of row 4 (uniform shift is absorbed by LayerNorm).
    x2[4] += 3.0 * np.arange(d_model)
    y2 = decoder_block(x2, None, params, n_heads)
    diff_by_pos = np.linalg.norm(y2 - y_dec_only, axis=1)
    print(f"\n  causality check: perturb x[4] with a non-uniform offset;")
    print(f"    ||y_new - y_old|| per position: {np.round(diff_by_pos, 4).tolist()}")
    print(f"  positions 0..3 must be exactly 0 (causal mask hides token 4 from earlier tokens);")
    print(f"  positions 4..5 should show change.")

    print("\n--- library cross-check (torch.nn.TransformerDecoderLayer / GPTBlock) ---")
