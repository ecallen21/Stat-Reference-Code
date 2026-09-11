"""Latent Diffusion Models - LDM / Stable Diffusion (Sec 47.211).

Rombach, Blattmann, Lorenz, Esser & Ommer 2022 'High-Resolution
Image Synthesis with Latent Diffusion Models', CVPR (aka Stable
Diffusion). Two stages:

    1. TRAIN a VAE that compresses 512x512 images to 64x64 latents.
    2. TRAIN a diffusion model in this LATENT SPACE (not pixel space).

Diffusion in a 64x compressed latent is ~64x cheaper per step
while a good VAE preserves visual quality. Enabled consumer-GPU
text-to-image generation.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def toy_encoder(x, W_enc):
    """Toy VAE encoder: linear compress d -> d_lat."""
    return x @ W_enc


def toy_decoder(z, W_dec):
    """Toy VAE decoder: linear expand d_lat -> d."""
    return z @ W_dec


def diffuse_and_denoise_latent(z0, T, sigma_max, rng):
    """Add Gaussian noise then remove (oracle 'score net' = z0)."""
    sigma = sigma_max * np.linspace(1, 0, T + 1)
    z = z0 + sigma[0] * rng.normal(size=z0.shape)
    for t in range(T):
        # Oracle denoiser: pretend to know z0. Real DM = trained UNet.
        z = 0.5 * z + 0.5 * z0 + 0.1 * rng.normal(size=z0.shape)
    return z


if __name__ == "__main__":
    print("=== Latent Diffusion / Stable Diffusion (Rombach et al 2022 CVPR) ===\n")
    rng = np.random.default_rng(0)

    # Toy data: 100 samples of 256-dim vectors that lie on a 16-dim linear manifold
    d_pixel = 256; d_latent = 16; n = 100
    U = rng.normal(size=(d_pixel, d_latent))                     # true manifold basis
    z_true = rng.normal(size=(n, d_latent))
    x_pixel = z_true @ U.T + 0.05 * rng.normal(size=(n, d_pixel))

    # Learn VAE-style encoder / decoder via SVD (best linear reconstruction)
    U_svd, s, Vt = np.linalg.svd(x_pixel, full_matrices=False)
    W_enc = Vt[:d_latent].T                                       # (d_pixel, d_latent)
    W_dec = Vt[:d_latent]                                         # (d_latent, d_pixel)

    # Encode -> latent
    z = toy_encoder(x_pixel, W_enc)
    # Reconstruction quality
    x_hat = toy_decoder(z, W_dec)
    rec_err = float(np.linalg.norm(x_pixel - x_hat) / np.linalg.norm(x_pixel))
    print(f"  Pixel dim = {d_pixel}, latent dim = {d_latent}, compression = {d_pixel / d_latent:.0f}x")
    print(f"  VAE reconstruction relative error: {rec_err:.4f}")

    # Now diffuse + denoise in LATENT space (much cheaper than pixel space)
    z_denoised = diffuse_and_denoise_latent(z, T=20, sigma_max=1.0, rng=rng)
    x_recovered = toy_decoder(z_denoised, W_dec)
    gen_err = float(np.linalg.norm(x_pixel - x_recovered) / np.linalg.norm(x_pixel))
    print(f"\n  After latent-space diffusion + decode: rel error = {gen_err:.4f}")

    # Cost comparison
    pixel_diff_flops = 20 * d_pixel * d_pixel * 4                # naive matmul-scaled
    latent_diff_flops = 20 * d_latent * d_latent * 4
    print(f"\n  Diffusion FLOPs at 20 steps:")
    print(f"    pixel-space:   {pixel_diff_flops:>10,}")
    print(f"    latent-space:  {latent_diff_flops:>10,}   ({pixel_diff_flops / latent_diff_flops:.0f}x cheaper)")

    print("\n--- library cross-check (diffusers.StableDiffusionPipeline / AutoencoderKL Python) ---")
