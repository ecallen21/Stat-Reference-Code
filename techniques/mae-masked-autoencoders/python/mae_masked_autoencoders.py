"""MAE - Masked Autoencoders (Reference Sec 47.228).

He, Chen, Xie, Li, Dollar & Girshick 2022 'Masked Autoencoders Are
Scalable Vision Learners', CVPR. Asymmetric encoder-decoder ViT:

    1. Divide image into patches; randomly MASK 75% of patches.
    2. ENCODER: process only VISIBLE 25% of patches.
    3. DECODER: shallow, reconstruct pixel values of masked patches
       from encoder outputs + [MASK] token positional embeddings.
    4. Loss = MSE on MASKED patches only.

75% mask ratio is high vs BERT's 15% because images are more
locally-redundant. Enables scaling to ViT-Huge on ImageNet with
strong linear-probe / fine-tune performance.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def patchify(image, patch_size):
    """Split (H, W) image into (n_patches, patch_size^2) flat patches."""
    H, W = image.shape
    p = patch_size
    patches = image.reshape(H // p, p, W // p, p).transpose(0, 2, 1, 3).reshape(-1, p * p)
    return patches


def unpatchify(patches, H, W, patch_size):
    p = patch_size
    return patches.reshape(H // p, W // p, p, p).transpose(0, 2, 1, 3).reshape(H, W)


def mae_forward(image, patch_size, mask_ratio, W_enc, W_dec, rng):
    """MAE forward: encode visible patches, decode all positions."""
    patches = patchify(image, patch_size)                        # (N, P^2)
    N = len(patches)
    n_masked = int(mask_ratio * N)
    perm = rng.permutation(N)
    masked_idx, visible_idx = perm[:n_masked], perm[n_masked:]
    # ENCODER: only visible patches -> hidden
    hidden = patches[visible_idx] @ W_enc                        # (n_visible, d_hidden)
    # DECODER: place hidden + mask tokens back in order, decode all
    d_hidden = hidden.shape[-1]
    full_hidden = np.zeros((N, d_hidden))
    full_hidden[visible_idx] = hidden
    # mask token = learned position-free zero embedding (we use zeros)
    # positional signal: add index-dependent bias
    pos_bias = 0.01 * np.arange(N)[:, None]
    reconstruction = (full_hidden + pos_bias) @ W_dec            # (N, P^2)
    # Loss only on masked patches
    loss = float(np.mean((reconstruction[masked_idx] - patches[masked_idx]) ** 2))
    return reconstruction, masked_idx, loss


if __name__ == "__main__":
    print("=== MAE - Masked Autoencoders (He et al 2022 CVPR) ===\n")
    rng = np.random.default_rng(0)

    # Build a 32x32 image with a simple pattern (checkerboard + gradient)
    H = W = 32
    img = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            img[i, j] = ((i // 4 + j // 4) % 2) * (0.5 + 0.5 * i / H)

    patch_size = 4
    n_patches = (H // patch_size) * (W // patch_size)
    d_hidden = 32
    W_enc = rng.normal(scale=0.3, size=(patch_size * patch_size, d_hidden))
    W_dec = rng.normal(scale=0.3, size=(d_hidden, patch_size * patch_size))

    # Try various mask ratios
    print(f"  Image = {H}x{W}, patch = {patch_size}x{patch_size} -> {n_patches} patches")
    print(f"  {'mask_ratio':>11}  {'n_visible':>10}  {'recon MSE (masked)':>19}")
    for r in [0.15, 0.50, 0.75, 0.90]:
        _, _, loss = mae_forward(img, patch_size, r, W_enc, W_dec, rng)
        n_vis = int(n_patches * (1 - r))
        print(f"  {r:>11.2f}  {n_vis:>10d}  {loss:>19.4f}")

    print(f"\n  MAE uses 75 % mask so the encoder handles only ~25 % of patches — 3-4x")
    print(f"  faster than fully-connected training. The decoder is shallow / small.")

    print("\n--- library cross-check (facebookresearch/mae; timm masked-autoencoder) ---")
