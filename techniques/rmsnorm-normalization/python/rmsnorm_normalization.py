"""RMSNorm - Root Mean Square Layer Normalisation (Reference Sec 47.171).

Zhang & Sennrich 2019 'Root Mean Square Layer Normalization',
NeurIPS. Simplifies LayerNorm by dropping the mean-centring:

    LayerNorm:  y = (x - mu) / sigma  * gamma + beta
    RMSNorm:    y = x / RMS(x)       * gamma        where RMS(x) = sqrt(mean(x^2))

Same computational form but 7-64% faster and equivalent quality on
Transformers. Used in LLaMA, GPT-NeoX, T5-v1.1, Mistral, and Gemma.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def layer_norm(x, gamma, beta, eps=1e-6):
    """LayerNorm: mean-center + scale."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


def rms_norm(x, gamma, eps=1e-6):
    """RMSNorm: RMS-scale only, no mean centring, no bias."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms * gamma


if __name__ == "__main__":
    print("=== RMSNorm (Zhang & Sennrich 2019) ===\n")
    rng = np.random.default_rng(0)

    d = 512
    x = rng.normal(loc=0.5, scale=2.0, size=(128, d))
    gamma = np.ones(d); beta = np.zeros(d)

    y_ln = layer_norm(x, gamma, beta)
    y_rms = rms_norm(x, gamma)

    print(f"  Input:   d = {d}, batch = 128, mean = {x.mean():.3f}, std = {x.std():.3f}")
    print(f"  LayerNorm output: mean = {y_ln.mean():.3f}, std = {y_ln.std():.3f}")
    print(f"  RMSNorm  output: mean = {y_rms.mean():.3f}, std = {y_rms.std():.3f}")

    # Timing comparison (1000 forward passes)
    import time
    n_reps = 1000
    t0 = time.perf_counter()
    for _ in range(n_reps): _ = layer_norm(x, gamma, beta)
    t_ln = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(n_reps): _ = rms_norm(x, gamma)
    t_rms = time.perf_counter() - t0
    print(f"\n  Timing over {n_reps} passes (128 x {d}):")
    print(f"    LayerNorm:  {t_ln * 1000:.1f} ms")
    print(f"    RMSNorm:    {t_rms * 1000:.1f} ms   ({100 * (1 - t_rms / t_ln):.1f}% faster)")

    # Both centre features to same scale; RMS just skips the mean centring
    for scale in [0.1, 1.0, 10.0]:
        x = rng.normal(scale=scale, size=(64, d))
        y_ln = layer_norm(x, gamma, beta)
        y_rms = rms_norm(x, gamma)
        print(f"  input scale = {scale:5.1f}  LayerNorm ||y|| = {np.linalg.norm(y_ln[0]):.2f}  "
              f"RMSNorm ||y|| = {np.linalg.norm(y_rms[0]):.2f}")

    print("\n  RMSNorm has no bias term and no mean subtraction -> fewer ops,")
    print("  no synchronisation across shard for the mean in TP; same quality.")

    print("\n--- library cross-check (torch.nn.RMSNorm; transformers LlamaRMSNorm) ---")
