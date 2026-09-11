"""Mixed-Precision Training (Reference Sec 47.168).

Micikevicius et al 2018 'Mixed Precision Training', ICLR. Uses
FP16 (or BF16) for matrix multiplications and activations to
2-4x speed on tensor-core GPUs, while keeping:

    1. A FP32 MASTER COPY of weights (loss / gradient scaling
       happens in FP32).
    2. LOSS SCALING: multiply loss by S before backward pass, then
       unscale gradients by 1/S -- shifts small gradients into
       the FP16-representable range and back.

Below: a demonstration of FP16 vs FP32 numerics and gradient
scaling, using numpy's float16.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def matmul_precision(A, B, dtype=np.float16):
    return (A.astype(dtype) @ B.astype(dtype)).astype(np.float64)


def matmul_error(A, B, dtype):
    ref = A @ B                                                # FP64 reference
    approx = matmul_precision(A, B, dtype)
    return float(np.mean(np.abs(ref - approx) / (np.abs(ref) + 1e-12)))


def loss_scaled_grad(W, X, y, scale=1.0):
    """Return grad computed via scaled loss, then unscaled."""
    p = 1 / (1 + np.exp(-(X.astype(np.float16) @ W.astype(np.float16))))
    scaled_grad = (X.T @ (p - y)) * scale                       # scaled backward
    return (scaled_grad / scale).astype(np.float32)             # unscale after


if __name__ == "__main__":
    print("=== Mixed-Precision Training (Micikevicius et al 2018) ===\n")

    rng = np.random.default_rng(0)
    # 1. Matrix-multiplication precision comparison
    A = rng.normal(size=(200, 300))
    B = rng.normal(size=(300, 200))
    for dtype in [np.float16, np.float32, np.float64]:
        err = matmul_error(A, B, dtype)
        print(f"  matmul relative error, dtype = {np.dtype(dtype).name:>7} : {err:.3e}")

    # 2. FP16 representable range
    fp16_max = np.finfo(np.float16).max
    fp16_min = np.finfo(np.float16).tiny                        # smallest normal
    print(f"\n  FP16 max = {fp16_max:.2e}, min-normal = {fp16_min:.2e}")
    print(f"  FP32 max = {np.finfo(np.float32).max:.2e}, min-normal = {np.finfo(np.float32).tiny:.2e}")

    # 3. Small gradient underflow demo
    small_grad = np.array([1e-4, 1e-6, 1e-8], dtype=np.float32)
    fp16_cast = small_grad.astype(np.float16).astype(np.float32)
    print(f"\n  Small gradients in FP32:   {small_grad}")
    print(f"  Cast to FP16 and back:    {fp16_cast}")
    print(f"  ==> 1e-4 preserved, 1e-8 underflows to 0 in FP16 (below fp16 min-normal 6.1e-5).")

    print(f"\n  With loss scaling S = 2^15 = 32768 before backward:")
    scaled = (small_grad * 32768).astype(np.float16)
    unscaled = (scaled.astype(np.float32) / 32768)
    print(f"    scaled  (FP16) = {scaled.astype(np.float32)}")
    print(f"    unscaled (FP32) = {unscaled}   (small values recovered)")

    print("\n  Master FP32 weights + FP16 forward / backward + dynamic loss scaling")
    print("  gives 2-4x speed on tensor-core GPUs with no accuracy loss on most tasks.")

    print("\n--- library cross-check (torch.amp / tensorflow.mixed_precision Python) ---")
