"""AWQ - Activation-aware Weight Quantization (Reference Sec 47.181).

Lin et al 2024 'AWQ: Activation-aware Weight Quantization for
LLM Compression and Acceleration', MLSys. Not all weights matter
equally: those multiplied by LARGE-MAGNITUDE ACTIVATIONS ('salient
weights', typically ~1% of columns) drive most of the error.

    1. Compute per-channel activation statistics from calibration.
    2. SCALE up salient weight columns W[:, j] by s_j (chosen to
       balance mean(|W|) ~ mean(|s * W|)).
    3. Scale down the corresponding input activations x_j / s_j
       (folded into upstream layer norm / linear).
    4. Quantize the rescaled weights.

Since post-rescale weights have smaller max / min, INT4 wastes
fewer levels on outliers -> less quant error.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def quantize_int4_group(W, group_size=64):
    """Symmetric per-group INT4 quant of W (d_out, d_in). Groups columns."""
    d_out, d_in = W.shape
    W_q = np.zeros_like(W)
    for g in range(0, d_in, group_size):
        Wg = W[:, g:g + group_size]
        scale = np.abs(Wg).max(axis=1, keepdims=True) / 7
        scale = np.where(scale == 0, 1.0, scale)
        q = np.clip(np.round(Wg / scale), -8, 7)
        W_q[:, g:g + group_size] = q * scale
    return W_q


def awq_quantize(W, X_cal, group_size=64, alpha_grid=None):
    """Search over alpha in [0, 1] for the best per-channel scale s_j = mean(|x_j|)^alpha."""
    if alpha_grid is None:
        alpha_grid = np.linspace(0.0, 1.0, 11)
    s_x = np.abs(X_cal).mean(axis=0) + 1e-6                    # per-channel activation stats
    best_err = np.inf; best_W_q = None; best_alpha = None
    Y_ref = X_cal @ W.T
    for alpha in alpha_grid:
        s = s_x ** alpha
        s = s / s.mean()                                          # normalise
        W_scaled = W * s[None, :]                                # scale up salient cols
        W_q = quantize_int4_group(W_scaled, group_size=group_size)
        # Undo the scaling in the effective forward:  y ~ (X / s) @ W_q^T
        X_scaled = X_cal / s[None, :]
        err = float(np.linalg.norm(X_scaled @ W_q.T - Y_ref) / np.linalg.norm(Y_ref))
        if err < best_err:
            best_err = err; best_W_q = W_q; best_alpha = alpha
    return best_W_q, s_x, best_alpha, best_err


if __name__ == "__main__":
    print("=== AWQ - Activation-aware Weight Quantization (Lin et al 2024) ===\n")
    rng = np.random.default_rng(0)

    d_in, d_out, n_cal = 128, 64, 512
    W = rng.normal(scale=0.3, size=(d_out, d_in))

    # Simulate skewed activation distribution: 5 salient channels
    X_cal = rng.normal(size=(n_cal, d_in))
    salient = rng.choice(d_in, size=5, replace=False)
    X_cal[:, salient] *= 8.0

    X_test = rng.normal(size=(200, d_in))
    X_test[:, salient] *= 8.0                                    # same distribution at test
    Y_ref = X_test @ W.T

    # Baseline: RTN INT4 group quant
    W_rtn = quantize_int4_group(W, group_size=64)
    err_rtn = float(np.linalg.norm(X_test @ W_rtn.T - Y_ref) / np.linalg.norm(Y_ref))

    # AWQ (search alpha in [0, 1])
    W_awq, s_x, alpha_star, calib_err = awq_quantize(W, X_cal, group_size=64)
    # Apply the same s_x^alpha_star scaling to test-time X
    s = (s_x ** alpha_star); s = s / s.mean()
    Y_awq = (X_test / s[None, :]) @ W_awq.T
    err_awq = float(np.linalg.norm(Y_awq - Y_ref) / np.linalg.norm(Y_ref))

    print(f"  Linear layer (d_in, d_out) = ({d_in}, {d_out}), {len(salient)} salient input channels")
    print(f"  RTN INT4 (per-group=64)     rel-err = {err_rtn:.4f}")
    print(f"  AWQ INT4 (alpha* = {alpha_star:.1f})       rel-err = {err_awq:.4f}")
    print(f"  Reduction: {100 * (1 - err_awq / err_rtn):.1f}% of RTN error is removed")

    print("\n  AWQ rescales weights so salient channels' effective scale shrinks;")
    print("  in real deployments the inverse activation scaling folds into upstream")
    print("  LayerNorm / previous linear, adding zero runtime cost.")

    print("\n--- library cross-check (llm-awq / autoawq / vllm awq_marlin Python) ---")
