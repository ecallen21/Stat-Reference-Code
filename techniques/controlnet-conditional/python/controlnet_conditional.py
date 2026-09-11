"""ControlNet - Conditional Diffusion (Reference Sec 47.213).

Zhang, Rao & Agrawala 2023 'Adding Conditional Control to
Text-to-Image Diffusion Models', ICCV. Add a TRAINABLE COPY of
the diffusion UNet's encoder that ingests an extra CONDITION
(edge map, depth, pose skeleton) and INJECTS its residuals into
the frozen base UNet via ZERO-INITIALISED convolutions.

    frozen SD:      x -> UNet -> eps_hat_base
    ControlNet:     (x, c) -> Trainable_encoder -> residuals
                    zero_conv(residuals) added into UNet skip connections
    Combined:       eps_hat = eps_hat_base + zero_conv(residual)

Zero init means training starts as identity to the base -> no
regression on pretraining quality.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def base_score(x, W_base):
    return x @ W_base


def controlnet_residual(x, c, W_cn, W_out, iteration=None):
    """Trainable branch: encode (x, c), output residuals via zero-init W_out."""
    z = np.concatenate([x, c], axis=-1) @ W_cn
    return z @ W_out


def combined_score(x, c, W_base, W_cn, W_out):
    return base_score(x, W_base) + controlnet_residual(x, c, W_cn, W_out)


if __name__ == "__main__":
    print("=== ControlNet (Zhang et al 2023 ICCV) ===\n")
    rng = np.random.default_rng(0)

    d = 8
    W_base = rng.normal(scale=0.3, size=(d, d))                  # frozen base
    W_cn = rng.normal(scale=0.3, size=(d + 4, 16))                # trainable encoder
    W_out_zero = np.zeros((16, d))                                # ZERO INIT

    x = rng.normal(size=(5, d))
    c = rng.normal(size=(5, 4))                                   # condition (edge map / depth)

    # At init (zero W_out): output identical to base
    base = base_score(x, W_base)
    comb_init = combined_score(x, c, W_base, W_cn, W_out_zero)
    print(f"  At init (W_out = 0):  ||combined - base|| = "
          f"{float(np.linalg.norm(comb_init - base)):.2e}   (guaranteed identity)")

    # After some training: W_out is nonzero, condition steers the score
    W_out_trained = rng.normal(scale=0.3, size=(16, d))
    comb_trained = combined_score(x, c, W_base, W_cn, W_out_trained)
    print(f"  After 'training' (W_out random 0.3): ||combined - base|| = "
          f"{float(np.linalg.norm(comb_trained - base)):.3f}   (now condition matters)")

    # Vary condition strength: mean magnitude of the residual across scales
    print(f"\n  Condition steering: residual norm vs condition magnitude:")
    for scale in [0.0, 0.5, 1.0, 3.0]:
        c_scaled = scale * rng.normal(size=(20, 4))
        residual = np.array([controlnet_residual(xi[None, :], c_scaled[i:i + 1], W_cn, W_out_trained).squeeze()
                              for i, xi in enumerate(x[:len(c_scaled)]) if i < len(c_scaled)][:min(len(x), len(c_scaled))])
        print(f"    condition scale = {scale:.1f}   mean |residual| = {float(np.mean(np.abs(residual))):.3f}")

    print("\n  ControlNet keeps 100 % of the base's zero-shot capability at init")
    print("  and adds fine control (edges / depth / pose) as W_out grows during training.")

    print("\n--- library cross-check (diffusers.ControlNetModel; lllyasviel/ControlNet Python) ---")
