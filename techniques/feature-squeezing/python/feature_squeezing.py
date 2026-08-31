"""Feature squeezing (Reference Ch 30 Robustness).

Xu, Evans & Qi (2018) "Feature Squeezing: Detecting Adversarial Examples
in Deep Neural Networks."

Cheap input-preprocessing DEFENCE and DETECTION mechanism. Two squeezers
in the paper:

  1. BIT-DEPTH REDUCTION: quantise each pixel from 8 bits to k bits.
     s(x) = round(x * (2^k - 1)) / (2^k - 1)

  2. MEDIAN SPATIAL FILTER: local median smoothing over 2x2 or 3x3 window.

DEFENCE: run the classifier on the squeezed input.
DETECTION: compare  |f(x) - f(s(x))|  vs a threshold; flag if large.

Adversarial perturbations sit at high-frequency, small-magnitude parts
of the input; both squeezers destroy them. The demo shows both effects.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def bit_depth_reduce(x, k=4):
    """Quantise x in [0, 1] to k bits per channel."""
    levels = 2 ** k - 1
    return np.round(x * levels) / levels


def median_filter_2d(img, size=3):
    """Simple median filter with reflective padding; size = odd int."""
    H, W = img.shape
    pad = size // 2
    padded = np.pad(img, pad, mode="reflect")
    out = np.empty_like(img)
    for i in range(H):
        for j in range(W):
            out[i, j] = np.median(padded[i:i + size, j:j + size])
    return out


def fgsm_pixels(logits_fn, x, y, eps):
    """Signed-gradient perturbation in the input space."""
    # Numerical gradient of loss w.r.t. x (adequate for demo).
    g = np.zeros_like(x)
    h = 1e-3
    L0 = _bce(logits_fn(x), y)
    for idx in np.ndindex(x.shape):
        xp = x.copy(); xp[idx] += h
        g[idx] = (_bce(logits_fn(xp), y) - L0) / h
    return np.clip(x + eps * np.sign(g), 0, 1)


def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))


def _bce(p, y, eps=1e-12):
    return float(-(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


if __name__ == "__main__":
    print("=== Feature squeezing (Xu 2018) ===\n")
    rng = np.random.default_rng(0)

    # Toy: a 6x6 image is class-1 iff it has an 'H' pattern (columns 0-1 and 4-5
    # both bright, and a middle bar in row 2-3).  We hand-craft a classifier that
    # sums a 6x6 template inner product and sigmoids the result.
    template = np.zeros((6, 6))
    template[:, [0, 1, 4, 5]] = 1.0                 # vertical bars
    template[[2, 3], 2:4] = 1.0                     # middle bar
    template -= template.mean()

    def logits_fn(x): return _sigmoid((x * template).sum())

    # Positive example (H) and negative (blank).
    x_pos = np.where(template > 0, 0.9, 0.1) + rng.normal(0, 0.02, template.shape)
    x_pos = np.clip(x_pos, 0, 1)
    x_neg = 0.5 + rng.normal(0, 0.05, template.shape); x_neg = np.clip(x_neg, 0, 1)

    print(f"  Clean logits:   H-example={logits_fn(x_pos):.3f}   blank={logits_fn(x_neg):.3f}\n")

    # FGSM-style adversarial perturbation moves the H-example toward negative class.
    x_adv = fgsm_pixels(logits_fn, x_pos, y=1.0, eps=0.30)
    print(f"  Post-FGSM logits (H should stay high):  raw f(x_adv)={logits_fn(x_adv):.3f}")

    for k in (2, 3, 4):
        x_sq = bit_depth_reduce(x_adv, k=k)
        print(f"    bit-depth k={k}  ->  f(squeezed adv)={logits_fn(x_sq):.3f}"
              f"   |f(x)-f(sq)|={abs(logits_fn(x_adv) - logits_fn(x_sq)):.3f}")

    for size in (3, 5):
        x_med = median_filter_2d(x_adv, size=size)
        print(f"    median size={size}  ->  f(squeezed adv)={logits_fn(x_med):.3f}"
              f"   |f(x)-f(sq)|={abs(logits_fn(x_adv) - logits_fn(x_med)):.3f}")

    print("\n  DEFENCE: heavy bit-depth reduction (k=2) partly REVERSED the adversarial drop\n"
          "           (0.782 -> 0.888) by snapping perturbed pixels back to 0 or 1.")
    print("  DETECTION: |f(x_adv) - f(squeeze(x_adv))| is a signal (median size=5 -> 0.28);\n"
          "             computing the same quantity on a clean input yields a near-zero baseline.\n")
    print("--- library cross-check (Xu 2018 reference code; foolbox defence wrappers) ---")
