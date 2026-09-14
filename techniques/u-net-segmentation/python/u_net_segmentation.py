"""U-Net Segmentation (Reference Sec 47.297).

Ronneberger, Fischer & Brox 2015 MICCAI. Encoder-decoder with
SKIP CONNECTIONS for pixel-level classification:

    encoder: successive 3x3 conv + downsample (contracts)
    decoder: transpose-conv upsample + concat with skip
    output:   1x1 conv to n_classes channels + softmax / sigmoid

The skip connections preserve high-resolution spatial detail
lost by downsampling. Standard for biomedical / satellite /
industrial-inspection segmentation.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def downsample(x):
    """2x2 average-pool downsampling (batch, H, W) -> (batch, H/2, W/2)."""
    B, H, W = x.shape
    return x.reshape(B, H // 2, 2, W // 2, 2).mean(axis=(2, 4))


def upsample(x):
    """Nearest-neighbour 2x upsampling."""
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def dice_score(pred, target, eps=1e-6):
    inter = float((pred * target).sum())
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


if __name__ == "__main__":
    print("=== U-Net Segmentation (Ronneberger et al 2015 MICCAI) ===\n")
    rng = np.random.default_rng(0)

    # Toy 32x32 image with a circular target region
    H = 32
    img = np.zeros((1, H, H))
    ci, cj = 16, 16; r = 7
    yy, xx = np.ogrid[:H, :H]
    mask = ((yy - ci) ** 2 + (xx - cj) ** 2) <= r ** 2
    img[0][mask] = 1.0
    # Add noise
    img_noisy = img + 0.3 * rng.standard_normal(img.shape)

    print(f"  Toy image {H}x{H} with circular target region (radius {r})\n")

    # Simulate a mini U-Net: down -> down -> up -> up
    x = img_noisy
    print(f"  Encoder:")
    d1 = downsample(x); print(f"    downsample 1: shape {d1.shape}")
    d2 = downsample(d1); print(f"    downsample 2: shape {d2.shape}")

    print(f"\n  Decoder (with skip connections):")
    u1 = upsample(d2); print(f"    upsample 1:  shape {u1.shape}   (+ skip from d1: shape {d1.shape})")
    concat1 = u1 + d1                                             # simplified concat
    u2 = upsample(concat1); print(f"    upsample 2:  shape {u2.shape}   (+ skip from x: shape {x.shape})")
    concat2 = u2 + x

    # "1x1 conv" simulated as thresholding
    prob = (concat2 > 0.3).astype(float)
    print(f"\n  Threshold predictions (proxy for final 1x1 conv):")
    print(f"    Dice score vs truth:  {dice_score(prob[0], img[0]):.3f}")
    print(f"    (Real U-Net trained with soft Dice / BCE achieves >0.95 on clean cases)")

    print(f"\n  Skip connections preserve fine-grained boundary information that")
    print(f"  pooling would otherwise lose - key to sharp segmentation masks.")

    print("\n--- library cross-check (segmentation-models-pytorch.Unet; monai.networks.nets.UNet) ---")
