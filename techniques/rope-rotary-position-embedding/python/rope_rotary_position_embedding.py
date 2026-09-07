"""RoPE -- rotary position embedding (Reference Sec 47.36).

Su et al. 2021 'RoFormer: enhanced transformer with rotary position
embedding'. Position information is injected by ROTATING pairs of
embedding dimensions by an angle proportional to position:

    q_pos = R(pos * theta) * q
    k_pos = R(pos * theta) * k

with theta_i = 10000^{-2i/d}. Attention inner products then depend
only on the RELATIVE offset (pos_q - pos_k), giving a natural inductive
bias for translation invariance in sequence models.

Advantages: no learned positional tokens, extrapolates to longer
sequences (with rescaling), used in Llama / Mistral / Qwen / Gemma.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def rope_freqs(d, base=10000.0):
    """Frequencies theta_i for the 2i-th and (2i+1)-th dims (d even)."""
    return base ** (-np.arange(0, d, 2) / d)


def apply_rope(x, positions):
    """x: (seq, d). Rotate pairs (x[2i], x[2i+1]) by pos * theta_i."""
    d = x.shape[1]; assert d % 2 == 0
    freqs = rope_freqs(d)                      # length d/2
    angles = np.outer(positions, freqs)        # (seq, d/2)
    cos = np.cos(angles); sin = np.sin(angles)
    x1 = x[:, 0::2]; x2 = x[:, 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    y = np.empty_like(x)
    y[:, 0::2] = y1; y[:, 1::2] = y2
    return y


if __name__ == "__main__":
    print("=== RoPE -- rotary position embedding (Su 2021) ===\n")
    rng = np.random.default_rng(0)
    d = 8; seq = 6
    Q = rng.normal(size=(seq, d))
    K = rng.normal(size=(seq, d))

    positions = np.arange(seq)
    Q_rope = apply_rope(Q, positions)
    K_rope = apply_rope(K, positions)

    #  Attention scores with RoPE: q_i^T k_j should depend only on (i - j)
    scores = Q_rope @ K_rope.T
    print(f"  Attention Q_rope K_rope^T (rows = q positions, cols = k positions):")
    print(np.array2string(scores, precision=2))

    #  Test relative-shift invariance: shift both Q and K positions by +2
    Q_shift = apply_rope(Q, positions + 2)
    K_shift = apply_rope(K, positions + 2)
    scores_shift = Q_shift @ K_shift.T
    max_err = float(np.max(np.abs(scores - scores_shift)))
    print(f"\n  max |scores(pos) - scores(pos + 2)| = {max_err:.2e}")
    print(f"  ~ 0  -> Attention only depends on the RELATIVE offset (i - j).")

    print("\n--- library cross-check (rotary-embedding-torch Python; xFormers) ---")
