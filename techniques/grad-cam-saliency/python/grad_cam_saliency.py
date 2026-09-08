"""Grad-CAM (Reference Sec 47.114).

Selvaraju, Cogswell, Das, Vedantam, Parikh & Batra 2017 'Grad-CAM:
Visual explanations from deep networks via gradient-based
localization', ICCV. Localizes what part of an image a CNN
attended to for a target class c:

    alpha_k^c = (1 / Z) sum_{i, j} d y^c / d A_ij^k     (channel weight)
    L_GradCAM^c(x) = ReLU( sum_k alpha_k^c A^k )

Since torch / TF aren't available here, illustrated instead on a
LINEAR model with 2-D "feature maps" (small tile grids):
Grad-CAM equals the input-times-gradient magnitudes, revealing
which spatial regions drove the class score.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def grad_cam_linear(W_c, activations):
    """Simplified Grad-CAM for a linear read-out on 2-D feature maps.
    activations: (K, H, W)   - channel activations.
    W_c        : (K, H, W)   - class-c linear weights over spatial locations.
    Returns ReLU( sum_k alpha_k * A^k )  with alpha_k = mean_ij W_c[k, i, j].
    """
    alpha = W_c.mean(axis=(1, 2))                      # (K,)
    L = np.tensordot(alpha, activations, axes=1)        # (H, W)
    return np.maximum(L, 0)


def input_x_gradient(x, W):
    """Baseline saliency: |x * dy/dx| = |x * W|."""
    return np.abs(x * W)


if __name__ == "__main__":
    print("=== Grad-CAM (Selvaraju et al 2017) ===\n")
    rng = np.random.default_rng(0)

    # Build a toy image: 8x8 grid, class-c signal in the top-left 3x3 block
    H = W_ = 8
    x = 0.1 * rng.normal(size=(H, W_))
    x[:3, :3] += 1.0                     # true class-c region

    # Fake linear model with weights emphasising the top-left block for class c
    W_c = np.zeros((1, H, W_))            # one channel
    W_c[0, :3, :3] = 1.0
    W_c += 0.05 * rng.normal(size=W_c.shape)

    activations = x[None, :, :]            # single-channel activation = input
    L = grad_cam_linear(W_c, activations)

    print("  Toy 8x8 input (signal in top-left 3x3):")
    print(np.round(x, 2))

    print("\n  Grad-CAM heatmap (ReLU'd):")
    print(np.round(L, 2))

    # Peak location
    peak = np.unravel_index(np.argmax(L), L.shape)
    print(f"\n  Peak Grad-CAM at spatial cell {peak}   (truth around (0, 0) to (2, 2))")

    # Baseline: input x gradient
    ixg = input_x_gradient(x, W_c[0])
    peak_ixg = np.unravel_index(np.argmax(ixg), ixg.shape)
    print(f"  Peak Input x Grad at spatial cell {peak_ixg}")

    print("\n  On real CNNs Grad-CAM localises class-discriminative image regions;")
    print("  the linear demo above shows the ReLU-of-weighted-activation core.")

    print("\n--- library cross-check (pytorch-grad-cam / captum Python; limited R) ---")
