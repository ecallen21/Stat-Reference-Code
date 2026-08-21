"""Post-training quantisation + magnitude pruning (Reference §27.x extra).

QUANTIZATION (int8):
  * Symmetric affine:  x_q = round(x / scale),  scale = max(|x|) / 127
  * Asymmetric affine: x_q = round((x - zp) / scale),  scale = (max - min) / 255
    zp (zero-point) is an int shift so that 0.0 maps to an int.

PRUNING (magnitude-based):
  * Global threshold: zero the |x| < tau tail of a weight tensor.
  * Structured pruning: whole rows / channels / attention heads (better for
    hardware speed-ups) rather than individual weights.

Trade-off: quantisation gives 2-4x memory + 2-4x throughput; pruning gives
sparsity that translates into throughput only with sparse kernels
(NVIDIA 2:4 structured sparsity, etc.).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def quantize_symmetric_int8(x) -> dict:
    scale = float(np.max(np.abs(x))) / 127.0
    if scale == 0:
        scale = 1.0
    q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return {"q": q, "scale": scale}


def dequantize_symmetric(qdict) -> np.ndarray:
    return qdict["q"].astype(float) * qdict["scale"]


def quantize_asymmetric_uint8(x) -> dict:
    xmax = float(x.max()); xmin = float(x.min())
    if xmax == xmin:
        return {"q": np.zeros_like(x, dtype=np.uint8), "scale": 1.0, "zp": 0}
    scale = (xmax - xmin) / 255.0
    zp = int(round(-xmin / scale))
    q = np.clip(np.round(x / scale + zp), 0, 255).astype(np.uint8)
    return {"q": q, "scale": scale, "zp": zp}


def dequantize_asymmetric(qdict) -> np.ndarray:
    return (qdict["q"].astype(float) - qdict["zp"]) * qdict["scale"]


def magnitude_prune(x, sparsity: float = 0.5) -> np.ndarray:
    """Zero out the |x| below the sparsity-quantile threshold."""
    flat = np.abs(x).ravel()
    if sparsity <= 0.0:
        return x.copy()
    tau = float(np.quantile(flat, sparsity))
    return np.where(np.abs(x) >= tau, x, 0.0)


def structured_prune_rows(x, keep_frac: float = 0.5) -> np.ndarray:
    """Keep only the k rows with highest L2 norm."""
    norms = np.linalg.norm(x, axis=1)
    k = max(1, int(keep_frac * x.shape[0]))
    keep = np.argsort(-norms)[:k]
    out = np.zeros_like(x); out[keep] = x[keep]
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    W = rng.normal(size=(64, 128))                        # a fully-connected layer

    print(f"=== Quantisation (int8) ===")
    sym = quantize_symmetric_int8(W)
    W_sym = dequantize_symmetric(sym)
    rmse_sym = float(np.sqrt(((W - W_sym) ** 2).mean()))
    print(f"  symmetric   int8: RMSE = {rmse_sym:.5f}   scale = {sym['scale']:.5f}")
    print(f"  bytes saved: {W.nbytes} -> {sym['q'].nbytes}  ({W.nbytes / sym['q'].nbytes:.1f}x)")

    asym = quantize_asymmetric_uint8(W)
    W_asym = dequantize_asymmetric(asym)
    rmse_asym = float(np.sqrt(((W - W_asym) ** 2).mean()))
    print(f"  asymmetric uint8: RMSE = {rmse_asym:.5f}   scale = {asym['scale']:.5f}   zp = {asym['zp']}")
    print(f"  (asymmetric uses full 256-value range regardless of sign asymmetry)")

    print(f"\n=== Magnitude pruning (unstructured) ===")
    for s in (0.3, 0.5, 0.7, 0.9):
        Wp = magnitude_prune(W, sparsity=s)
        zeros = (Wp == 0).mean()
        rmse = float(np.sqrt(((W - Wp) ** 2).mean()))
        print(f"  sparsity {s:>4}: actual zeros = {zeros:.3f}, RMSE vs dense = {rmse:.5f}")

    print(f"\n=== Structured pruning (row L2 norm) ===")
    for k in (0.75, 0.5, 0.25):
        Wr = structured_prune_rows(W, keep_frac=k)
        kept_rows = np.sum(np.any(Wr != 0, axis=1))
        rmse = float(np.sqrt(((W - Wr) ** 2).mean()))
        print(f"  keep {k:>4}: rows kept = {kept_rows} / {W.shape[0]}, "
              f"RMSE = {rmse:.5f}")

    # Combined: prune then quantise
    Wp = magnitude_prune(W, sparsity=0.5)
    q = quantize_symmetric_int8(Wp)
    Wr = dequantize_symmetric(q)
    rmse_both = float(np.sqrt(((W - Wr) ** 2).mean()))
    print(f"\n=== 50% prune + int8 quantise combined ===")
    print(f"  RMSE vs dense = {rmse_both:.5f}   memory ~ 1/8 (0.5 sparse * 1/4 bit ratio)")

    print("\n--- library cross-check (torch.quantization; torch.nn.utils.prune; bitsandbytes; llm-int8) ---")
