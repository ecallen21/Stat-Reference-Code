"""RandAugment / AutoAugment (Reference Sec 47.304).

Cubuk et al 2019 (AutoAugment); Cubuk et al 2020 (RandAugment).
Data augmentation policies for image classification:

    AutoAugment: RL-searched policy of (op, prob, magnitude)
                triples optimised on a proxy task.
    RandAugment: N transforms drawn uniformly from a fixed pool
                with a single global MAGNITUDE M.

RandAugment removes AutoAugment's expensive search yet matches
its downstream accuracy. Standard for CIFAR / ImageNet training.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def apply_rotate(img, magnitude):
    """Rotate by up to 30 degrees; toy version = shift columns."""
    shift = int(magnitude * 3)                                     # up to 3 pixels
    return np.roll(img, shift, axis=0)


def apply_shear_x(img, magnitude):
    """Shear along x; toy version = column-dependent row shift."""
    H, W = img.shape
    shift = magnitude
    out = np.zeros_like(img)
    for j in range(W):
        out[:, j] = np.roll(img[:, j], int(shift * (j - W / 2) / W))
    return out


def apply_contrast(img, magnitude):
    """Contrast adjustment."""
    factor = 1 + 0.3 * magnitude
    mean = img.mean()
    return np.clip(mean + factor * (img - mean), 0, 1)


def apply_brightness(img, magnitude):
    return np.clip(img + 0.1 * magnitude, 0, 1)


def apply_solarize(img, magnitude):
    thresh = 1 - 0.1 * magnitude
    return np.where(img > thresh, 1 - img, img)


OPS = [apply_rotate, apply_shear_x, apply_contrast, apply_brightness, apply_solarize]
OP_NAMES = ["rotate", "shear_x", "contrast", "brightness", "solarize"]


def randaugment(img, N=2, M=5, rng=None):
    """Draw N ops uniformly, apply each with global magnitude M."""
    if rng is None: rng = np.random.default_rng(0)
    picks = rng.choice(len(OPS), size=N, replace=True)
    ops_used = []
    for i in picks:
        img = OPS[i](img, magnitude=M)
        ops_used.append(OP_NAMES[i])
    return img, ops_used


if __name__ == "__main__":
    print("=== RandAugment / AutoAugment (Cubuk et al 2019/2020) ===\n")
    rng = np.random.default_rng(0)

    # Toy 16x16 image with a diagonal gradient
    H = 16
    img = np.tile(np.linspace(0, 1, H), (H, 1))
    print(f"  Base image {H}x{H}. Original pixel range: [{img.min():.2f}, {img.max():.2f}]\n")

    for N in [1, 2, 3]:
        for M in [2, 5, 9]:
            aug, ops = randaugment(img, N=N, M=M, rng=rng)
            print(f"  N={N}, M={M:>2}   ops = {ops}   mean = {aug.mean():.2f}, std = {aug.std():.2f}")

    print(f"\n  RandAugment hyperparameters: N (# ops) and M (magnitude, 1-30).")
    print(f"  Typical CIFAR settings: N=2, M=14; ImageNet: N=2, M=9.\n")

    print(f"  AutoAugment learns a POLICY of ~ 24 sub-policies each with 2 (op, prob, mag)")
    print(f"  triples via RL search on a proxy task. Reduced RA is nearly as good.")

    print("\n--- library cross-check (torchvision.transforms.v2.RandAugment/AutoAugment) ---")
