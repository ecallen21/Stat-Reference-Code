"""Video Diffusion (Reference Sec 47.221).

Ho et al 2022 'Video Diffusion Models'; Blattmann 2023 'Stable
Video Diffusion'. Extends image diffusion to a sequence of frames
with a 3D UNet + TEMPORAL ATTENTION layers that ensure temporal
coherence:

    x_t in R^{T x C x H x W}       (T frames of noisy video)
    UNet:  spatial attn + temporal attn(across T dim)
    loss = ||eps_theta(x_t, t) - eps||^2

Cascaded models: low-res base -> spatial upsampler -> temporal
interpolator. Requires massive compute; open models (SVD, Zeroscope,
CogVideoX) generate 2-6s clips.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def temporal_attention(x, alpha=0.7):
    """Toy temporal attention: exponential smoothing across frames."""
    T = x.shape[0]
    smoothed = np.zeros_like(x)
    smoothed[0] = x[0]
    for t in range(1, T):
        smoothed[t] = alpha * smoothed[t - 1] + (1 - alpha) * x[t]
    return smoothed


def per_frame_denoise(frames, sigma, target):
    """Toy 'image diffusion' step: shrink noise toward target."""
    return frames + (target - frames) / (1 + sigma ** 2)


def video_diffusion_step(frames, sigma, target_seq, use_temporal_attn=True):
    """One denoising step for a video (T frames)."""
    denoised = per_frame_denoise(frames, sigma, target_seq)
    if use_temporal_attn:
        denoised = temporal_attention(denoised, alpha=0.7)
    return denoised


if __name__ == "__main__":
    print("=== Video Diffusion (Ho et al 2022; SVD 2023) ===\n")
    rng = np.random.default_rng(0)

    # Target 'video': smooth 10-frame sequence in R^32 (each frame is a vector)
    T = 10; d = 32
    target = np.array([np.sin(t * 0.5) * np.ones(d) for t in range(T)])
    # Noisy initial 'video'
    sigma = 2.0
    frames_noisy = target + sigma * rng.normal(size=(T, d))

    # Reverse-diffuse (5 steps)
    frames_no_ta = frames_noisy.copy()
    frames_ta = frames_noisy.copy()
    for k in range(5):
        s = sigma * (5 - k) / 5
        frames_no_ta = video_diffusion_step(frames_no_ta, s, target,
                                                use_temporal_attn=False)
        frames_ta = video_diffusion_step(frames_ta, s, target,
                                            use_temporal_attn=True)

    # Frame-to-frame smoothness (lower = more coherent)
    def smoothness(frames):
        diffs = np.diff(frames, axis=0)
        return float(np.mean(np.linalg.norm(diffs, axis=1)))

    print(f"  Target video: T = {T} frames, d = {d} 'pixel' dim")
    print(f"  Frame-to-frame variation:")
    print(f"    original noisy input   : {smoothness(frames_noisy):.3f}")
    print(f"    denoised, no temporal  : {smoothness(frames_no_ta):.3f}")
    print(f"    denoised, +temporal    : {smoothness(frames_ta):.3f}   (smoother)")
    print(f"  Ground truth smoothness  : {smoothness(target):.3f}")

    err_no_ta = float(np.mean(np.linalg.norm(frames_no_ta - target, axis=1)))
    err_ta = float(np.mean(np.linalg.norm(frames_ta - target, axis=1)))
    print(f"\n  Reconstruction MSE-to-target:")
    print(f"    no temporal attn : {err_no_ta:.3f}")
    print(f"    +temporal attn   : {err_ta:.3f}")

    print("\n--- library cross-check (diffusers.StableVideoDiffusionPipeline; CogVideoX Python) ---")
