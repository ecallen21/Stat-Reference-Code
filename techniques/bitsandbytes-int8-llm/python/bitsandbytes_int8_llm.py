"""LLM.int8() / bitsandbytes (Reference Sec 47.179).

Dettmers, Lewis, Belkada & Zettlemoyer 2022 'LLM.int8(): 8-bit
Matrix Multiplication for Transformers at Scale', NeurIPS. Two
key insights:

    1. Mixed-precision decomposition: identify OUTLIER FEATURES
       (a small set of columns with magnitudes >> 6) and keep
       them in FP16; quantise the rest to INT8.
    2. Vector-wise quantisation: per-row of A, per-column of B.

Enables lossless (< 0.1 pt zero-shot degradation) INT8 inference
for models up to 175 B parameters -> half the memory of FP16.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def vector_wise_int8_matmul(A, B, outlier_thresh=6.0):
    """LLM.int8() forward pass: separate outlier columns of A (features)."""
    # 1. Detect outlier feature dims (columns of A whose max |val| > threshold)
    col_abs_max = np.abs(A).max(axis=0)
    outlier_cols = col_abs_max > outlier_thresh
    A_out, A_int = A[:, outlier_cols], A[:, ~outlier_cols]
    B_out, B_int = B[outlier_cols], B[~outlier_cols]

    # 2. FP16 path for outliers
    Y_fp = A_out @ B_out                                        # small

    # 3. INT8 path for rest, vector-wise quant
    row_scales_A = np.abs(A_int).max(axis=1) / 127
    col_scales_B = np.abs(B_int).max(axis=0) / 127
    row_scales_A = np.where(row_scales_A == 0, 1.0, row_scales_A)
    col_scales_B = np.where(col_scales_B == 0, 1.0, col_scales_B)
    A_q = np.clip(np.round(A_int / row_scales_A[:, None]), -127, 127).astype(np.int32)
    B_q = np.clip(np.round(B_int / col_scales_B[None, :]), -127, 127).astype(np.int32)
    Y_int_raw = A_q @ B_q                                       # INT32 accumulate
    Y_int = Y_int_raw * (row_scales_A[:, None] * col_scales_B[None, :])
    return Y_fp + Y_int, int(outlier_cols.sum())


if __name__ == "__main__":
    print("=== LLM.int8() / bitsandbytes (Dettmers et al 2022) ===\n")

    rng = np.random.default_rng(0)
    # Simulate a linear layer with a few outlier features (typical of Transformer FFN)
    m, k, n = 128, 512, 512
    A = rng.normal(size=(m, k))
    outlier_feat = rng.choice(k, size=10, replace=False)         # 10 outlier feature dims
    A[:, outlier_feat] *= 20.0                                    # blow them up
    B = rng.normal(scale=0.1, size=(k, n))

    Y_ref = A @ B

    # Naive INT8 without outlier decomposition (per-tensor)
    scale_A = np.abs(A).max() / 127
    scale_B = np.abs(B).max() / 127
    Aq = np.clip(np.round(A / scale_A), -127, 127).astype(np.int32)
    Bq = np.clip(np.round(B / scale_B), -127, 127).astype(np.int32)
    Y_naive = (Aq @ Bq).astype(np.float32) * scale_A * scale_B
    err_naive = float(np.linalg.norm(Y_naive - Y_ref) / np.linalg.norm(Y_ref))

    # LLM.int8() with outlier decomposition
    Y_llm, n_outliers = vector_wise_int8_matmul(A, B, outlier_thresh=6.0)
    err_llm = float(np.linalg.norm(Y_llm - Y_ref) / np.linalg.norm(Y_ref))

    print(f"  Matrix multiply (m, k, n) = ({m}, {k}, {n}) with {len(outlier_feat)} injected outliers")
    print(f"  Detected outlier feature dims: {n_outliers}")
    print(f"  Naive per-tensor INT8       rel-err = {err_naive:.4f}")
    print(f"  LLM.int8() decomposed       rel-err = {err_llm:.4f}")

    print(f"\n  Memory saved vs FP16: {100 * (1 - (k - n_outliers) / (2 * k)):.1f}%")
    print(f"  (outlier cols still FP16, integer cols in INT8 = 1 byte vs FP16's 2 bytes)")

    print("\n--- library cross-check (bitsandbytes.nn.Linear8bitLt Python) ---")
