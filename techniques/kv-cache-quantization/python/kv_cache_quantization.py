"""KV-Cache Quantization (Reference Sec 47.174).

Sheng et al 2023 'FlexGen'; Liu et al 2023 'KIVI'. LLM inference is
memory-bound; the K, V cache grows as O(L * T * H * d). Quantising
the cache to INT8 / INT4 halves (or quarters) memory + bandwidth
while preserving generation quality:

    K, V (FP16) -> per-token / per-channel affine INT8:
    q = round((x - z) / s),  x_hat = q * s + z

Recovery quality depends on granularity: per-token for K
(preserves attention-score scale) and per-channel for V.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def quantize_int8_per_channel(K):
    """Per-channel symmetric INT8 quant of K (T, d). Best for K (KIVI 2023)."""
    scale = np.abs(K).max(axis=0) / 127
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.round(K / scale[None, :]), -127, 127).astype(np.int8)
    return q, scale, np.zeros_like(scale)


def dequantize_int8_per_channel(q, scale, zero):
    return q.astype(np.float32) * scale[None, :]


def quantize_int4_per_group(V, group=16):
    """Per-group INT4 quant of V (T, d). Groups columns."""
    T, d = V.shape
    assert d % group == 0
    q = np.zeros_like(V, dtype=np.int8)
    scales = np.zeros((T, d // group))
    zeros = np.zeros((T, d // group))
    for g in range(d // group):
        Vg = V[:, g * group:(g + 1) * group]
        s = (Vg.max(1) - Vg.min(1)) / 15
        s = np.where(s == 0, 1.0, s)
        z = Vg.min(1)
        q[:, g * group:(g + 1) * group] = np.round((Vg - z[:, None]) / s[:, None])
        scales[:, g] = s; zeros[:, g] = z
    return q, scales, zeros


def dequantize_int4_per_group(q, scales, zeros, group=16):
    T, d = q.shape
    V = np.zeros_like(q, dtype=np.float32)
    for g in range(d // group):
        V[:, g * group:(g + 1) * group] = (q[:, g * group:(g + 1) * group].astype(np.float32)
                                              * scales[:, g:g + 1] + zeros[:, g:g + 1])
    return V


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


if __name__ == "__main__":
    print("=== KV-Cache Quantization (Sheng et al 2023; Liu et al 2023) ===\n")
    rng = np.random.default_rng(0)

    T, d = 256, 64
    Q_query = rng.normal(scale=0.5, size=(1, d))
    K_fp16 = rng.normal(scale=0.5, size=(T, d))
    V_fp16 = rng.normal(scale=0.5, size=(T, d))

    # Reference: full-precision attention
    scores_ref = softmax(Q_query @ K_fp16.T / np.sqrt(d))
    out_ref = scores_ref @ V_fp16

    # INT8 K cache, FP16 V
    K_q, K_s, K_z = quantize_int8_per_channel(K_fp16)
    K_deq = dequantize_int8_per_channel(K_q, K_s, K_z)
    scores_int8k = softmax(Q_query @ K_deq.T / np.sqrt(d))
    out_int8k = scores_int8k @ V_fp16
    err_int8k = float(np.linalg.norm(out_int8k - out_ref) / np.linalg.norm(out_ref))

    # INT4 V cache (per-group), FP16 K
    V_q, V_s, V_z = quantize_int4_per_group(V_fp16, group=16)
    V_deq = dequantize_int4_per_group(V_q, V_s, V_z, group=16)
    out_int4v = scores_ref @ V_deq
    err_int4v = float(np.linalg.norm(out_int4v - out_ref) / np.linalg.norm(out_ref))

    # Both
    scores_int8_int4 = softmax(Q_query @ K_deq.T / np.sqrt(d))
    out_int8k_int4v = scores_int8_int4 @ V_deq
    err_full = float(np.linalg.norm(out_int8k_int4v - out_ref) / np.linalg.norm(out_ref))

    print(f"  Sequence T = {T}, d_head = {d}, one attention query")
    print(f"  Attention output relative error vs FP32 reference:")
    print(f"    INT8 K only        : {err_int8k:.4f}")
    print(f"    INT4 V only        : {err_int4v:.4f}")
    print(f"    INT8 K + INT4 V    : {err_full:.4f}")

    print(f"\n  Memory (bytes) for {T} tokens x d_head {d}:")
    print(f"    FP16 K+V:      {T * d * 4:>7,}")
    print(f"    INT8 K, FP16 V: {T * d + T * d * 2:>7,}  ({100 * (1 - (T * d + T * d * 2) / (T * d * 4)):.1f}% saved)")
    print(f"    INT8 K, INT4 V: {T * d + T * d // 2 + T * (d // 16) * 2 * 2:>7,}  "
          f"({100 * (1 - (T * d + T * d // 2 + T * (d // 16) * 2 * 2) / (T * d * 4)):.1f}% saved)")

    print("\n--- library cross-check (bitsandbytes / gptq / awq / kv-cache-int Python) ---")
