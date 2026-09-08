"""YaRN context extension (Reference Sec 47.49).

Peng, Quesnelle, Sharkey & Chan 2023 'YaRN: Efficient context window
extension of large language models'. Extends RoPE to longer contexts
than seen in training via:

  1. NTK-aware interpolation: preserve high-freq bands, scale low-freq.
  2. NTK-by-parts:            keep untouched frequencies for tokens
                              within original length, interpolate for tokens
                              beyond it.
  3. Attention-temperature scaling: t = 0.1 * log(scale) + 1 to preserve
                                     attention entropy.

Effect: original context L trained -> effective context L * s at
much lower perplexity than naive extrapolation.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def rope_freqs(dim, base=10_000.0):
    return 1.0 / (base ** (np.arange(0, dim, 2) / dim))


def yarn_rescale_freqs(freqs, scale, orig_ctx, dim, alpha=1.0, beta=32.0):
    """Apply NTK-by-parts rescaling: interp lowest-freq bands, preserve highest.

    `preserve` = 1 -> keep freq unchanged; `preserve` = 0 -> divide by scale.
    Ramp is on r = orig_ctx / wavelen (# original contexts per wavelength):
    r > beta  -> preserve (many rotations fit, band already reliable);
    r < alpha -> interpolate (wavelength longer than orig ctx).
    """
    wavelen = 2 * np.pi / freqs
    r = orig_ctx / wavelen
    preserve = np.clip((r - alpha) / (beta - alpha), 0.0, 1.0)
    return freqs * preserve + (freqs / scale) * (1.0 - preserve)


def yarn_attention_temperature(scale):
    return 0.1 * np.log(scale) + 1.0


def rope_angle(pos, freqs):
    return np.outer(pos, freqs)


if __name__ == "__main__":
    print("=== YaRN context extension (Peng-Quesnelle-Sharkey-Chan 2023) ===\n")
    dim = 64
    orig_ctx = 2048
    scales = [1.0, 2.0, 4.0, 8.0, 16.0]

    freqs = rope_freqs(dim)
    print(f"  dim={dim}, orig context={orig_ctx}")
    print(f"  Base RoPE: {len(freqs)} freq bands; "
          f"lowest={freqs.min():.2e}  highest={freqs.max():.2e}\n")

    print("  scale | attn temp t | frac freqs preserved (high) | max wavelength (tokens)")
    for s in scales:
        f = yarn_rescale_freqs(freqs, s, orig_ctx, dim)
        t = yarn_attention_temperature(s)
        preserved = float((np.isclose(f, freqs, rtol=1e-6)).mean())
        max_wave = float(2 * np.pi / f.min())
        print(f"    {s:5.1f} | {t:5.3f}       | {preserved:.2f}                        | {max_wave:10.0f}")

    print("\n  NTK-by-parts leaves the highest-frequency bands untouched")
    print("  (recovering exact rotation for nearby tokens) and interpolates")
    print("  only the LOW freqs that would alias at extended context.")
    print("\n--- library cross-check (transformers / vllm Python; not standard R) ---")
