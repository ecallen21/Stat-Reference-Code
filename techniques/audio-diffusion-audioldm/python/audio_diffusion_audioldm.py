"""Audio Diffusion / AudioLDM (Reference Sec 47.230).

Liu et al 2023 'AudioLDM: Text-to-Audio Generation with Latent
Diffusion Models', ICML. Diffusion in a mel-spectrogram-based
latent space, decoded by a HiFi-GAN vocoder:

    1. Waveform -> mel-spectrogram (2D 'image').
    2. VAE encodes mel-spec to latent z.
    3. Diffusion model in z-space, text-conditioned via CLAP encoder.
    4. Decode z -> mel-spec -> waveform (vocoder).

Enables text-to-audio: 'sound of thunder', 'wooden creaking'.
Similar tech: StableAudio, MusicGen.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def mel_spectrogram(waveform, n_mels=64, n_frames=100):
    """Toy mel-spectrogram: STFT-magnitude then log-mel-bin."""
    L = len(waveform)
    frame_len = max(1, L // n_frames)
    spec = np.zeros((n_mels, n_frames))
    for t in range(n_frames):
        start = t * frame_len
        segment = waveform[start:start + frame_len * 2]
        if len(segment) < 2: continue
        F = np.abs(np.fft.rfft(segment))[:n_mels]
        spec[:len(F), t] = np.log(F + 1e-6)
    return spec


def diffusion_denoise_toy(spec_noisy, spec_target, n_steps=20):
    """Toy denoiser: linear shrinkage toward target."""
    x = spec_noisy.copy()
    for k in range(n_steps):
        x = 0.9 * x + 0.1 * spec_target
    return x


def griffin_lim_toy(spec, waveform_len):
    """Very-toy 'vocoder': reconstruct waveform by inverse-FFT averaging."""
    frames = spec.shape[1]
    frame_len = max(1, waveform_len // frames)
    wave = np.zeros(waveform_len)
    for t in range(frames):
        # Recover approximate amplitudes
        F_amp = np.exp(spec[:, t]) - 1e-6
        F = F_amp * np.exp(1j * np.random.default_rng(t).uniform(0, 2 * np.pi, size=len(F_amp)))
        segment = np.fft.irfft(F, n=2 * frame_len)
        start = t * frame_len
        end = min(start + len(segment), waveform_len)
        wave[start:end] += segment[:end - start]
    return wave


if __name__ == "__main__":
    print("=== Audio Diffusion / AudioLDM (Liu et al 2023 ICML) ===\n")
    rng = np.random.default_rng(0)

    # Target audio: 1-second sine wave at 440 Hz + a burst of noise
    sr = 8000; T = 1.0
    t = np.arange(0, T, 1 / sr)
    target_wave = 0.6 * np.sin(2 * np.pi * 440 * t) + 0.2 * (rng.uniform(size=len(t)) - 0.5)
    target_spec = mel_spectrogram(target_wave, n_mels=64, n_frames=100)

    # Noisy input in spectrogram space
    noisy_spec = target_spec + 2.0 * rng.normal(size=target_spec.shape)

    # Diffusion denoise
    denoised_spec = diffusion_denoise_toy(noisy_spec, target_spec, n_steps=20)

    err_before = float(np.mean((noisy_spec - target_spec) ** 2))
    err_after = float(np.mean((denoised_spec - target_spec) ** 2))
    print(f"  1-second audio at 8 kHz -> 64-mel x 100-frame spectrogram")
    print(f"  MSE(noisy_spec, target)     : {err_before:.4f}")
    print(f"  MSE(denoised_spec, target)  : {err_after:.4f}   ({100 * (1 - err_after / err_before):.1f}% reduction)")

    # 'Vocoder' back to waveform
    wave_recon = griffin_lim_toy(denoised_spec, len(target_wave))
    signal_pow = float(np.mean(target_wave ** 2))
    noise_pow = float(np.mean((wave_recon - target_wave) ** 2))
    print(f"  Reconstruction SNR (dB): {10 * np.log10(signal_pow / max(noise_pow, 1e-9)):.2f}")

    print("\n  Real AudioLDM uses a proper CLAP text-encoder + HiFi-GAN vocoder;")
    print("  generates 10-second clips from text like 'sound of thunder' in ~1 s on GPU.")

    print("\n--- library cross-check (audiocraft / audioldm / stable-audio-tools; diffusers audio pipelines) ---")
