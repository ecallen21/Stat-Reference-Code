"""GPTQ - Generative Pre-trained Transformer Quantization (Sec 47.180).

Frantar, Ashkboos, Hoefler & Alistarh 2023 'GPTQ: Accurate
Post-Training Quantization for Generative Pre-trained Transformers',
ICLR. Post-training quantization guided by a small CALIBRATION
dataset. For each linear layer minimises reconstruction error:

    W_hat = argmin_{W' in Q}  ||W X - W' X||_F^2

Uses Optimal Brain Surgeon (Hassibi 1993) style second-order
updates: quantize columns one at a time, propagating the
compensation to the remaining columns via H^-1.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def quantize_int4(w, scale=None):
    """Symmetric per-tensor INT4 (-8..7) quantise -> dequant."""
    if scale is None:
        scale = np.abs(w).max() / 7 + 1e-8
    q = np.clip(np.round(w / scale), -8, 7)
    return q * scale


def gptq_quantize_layer(W, X_cal, bits=4, group_size=None):
    """GPTQ: quantise weight matrix W (d_out, d_in) using calibration X_cal (n, d_in).

    Simplified version: quantise columns of W left-to-right; after each column,
    update remaining columns with a Cholesky-based OBS compensation.
    """
    d_out, d_in = W.shape
    H = 2 * (X_cal.T @ X_cal) / len(X_cal) + 1e-5 * np.eye(d_in)  # Hessian
    # Cholesky-based inverse for stability
    L = np.linalg.cholesky(H)
    Hinv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(d_in)))

    W_q = W.copy()
    for j in range(d_in):
        d = Hinv[j, j]
        # Quantise column j
        w_col = W_q[:, j]
        w_q_col = quantize_int4(w_col)
        err = (w_col - w_q_col) / d                              # OBS scaling
        W_q[:, j] = w_q_col
        # Compensate future columns
        if j + 1 < d_in:
            W_q[:, j + 1:] -= np.outer(err, Hinv[j, j + 1:])
    return W_q


if __name__ == "__main__":
    print("=== GPTQ (Frantar et al 2023) ===\n")
    rng = np.random.default_rng(0)

    d_in, d_out, n_cal = 128, 64, 512
    W_true = rng.normal(scale=0.5, size=(d_out, d_in))
    X_cal = rng.normal(size=(n_cal, d_in))
    X_test = rng.normal(size=(200, d_in))

    Y_ref = X_test @ W_true.T

    # Baseline: naive round-to-nearest INT4 (no calibration)
    scale = np.abs(W_true).max() / 7
    W_rtn = np.clip(np.round(W_true / scale), -8, 7) * scale
    err_rtn = float(np.linalg.norm(X_test @ W_rtn.T - Y_ref) / np.linalg.norm(Y_ref))

    # GPTQ INT4 with calibration
    W_gptq = gptq_quantize_layer(W_true, X_cal, bits=4)
    err_gptq = float(np.linalg.norm(X_test @ W_gptq.T - Y_ref) / np.linalg.norm(Y_ref))

    print(f"  Linear layer (d_in, d_out) = ({d_in}, {d_out}), n_cal = {n_cal}")
    print(f"  Naive RTN INT4  rel-err = {err_rtn:.4f}")
    print(f"  GPTQ INT4       rel-err = {err_gptq:.4f}")
    print(f"  Reduction: {100 * (1 - err_gptq / err_rtn):.1f}% of the naive error is removed")

    print(f"\n  INT4 storage: {d_out * d_in * 0.5 / 1024:.1f} KB "
          f"(vs FP16: {d_out * d_in * 2 / 1024:.1f} KB, 4x saving)")

    print("\n--- library cross-check (auto-gptq / autogptq / optimum Python) ---")
