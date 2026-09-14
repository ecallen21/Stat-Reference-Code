"""Hilbert Transform / Analytic Signal (Reference Sec 47.331).

Hilbert 1912; Gabor 1946. Construct the analytic signal

    z(t) = x(t) + i * H(x)(t)

where H(x) is the Hilbert transform of x. From z(t) one reads:

    A(t) = |z(t)|       instantaneous amplitude / envelope
    phi(t) = arg z(t)   instantaneous phase
    f(t) = d phi / dt / (2 pi)   instantaneous frequency

Widely used in EEG / speech / vibration analysis and in the
Hilbert-Huang transform (with EMD).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def analytic_signal(x):
    """Compute z(t) = x(t) + i * H(x)(t) via FFT."""
    N = len(x)
    X = np.fft.fft(x)
    H = np.zeros(N)
    if N % 2 == 0:
        H[0] = 1; H[N // 2] = 1
        H[1:N // 2] = 2
    else:
        H[0] = 1
        H[1:(N + 1) // 2] = 2
    return np.fft.ifft(X * H)


def instantaneous(x, fs=1.0):
    z = analytic_signal(x)
    amp = np.abs(z)
    phi = np.unwrap(np.angle(z))
    freq = np.gradient(phi) / (2 * np.pi) * fs
    return amp, phi, freq


if __name__ == "__main__":
    print("=== Hilbert Transform / Analytic Signal (Hilbert 1912) ===\n")
    rng = np.random.default_rng(0)

    fs = 1000.0                                                     # Hz
    t = np.arange(0, 2, 1 / fs)
    # AM signal: envelope A(t) grows linearly, carrier at 50 Hz
    A_true = 1.0 + 0.5 * t
    x = A_true * np.sin(2 * np.pi * 50 * t)

    amp, phi, freq = instantaneous(x, fs=fs)
    print(f"  AM signal: 50 Hz carrier with growing envelope A(t) = 1 + 0.5 t")
    print(f"  Length {len(x)} samples at {fs:.0f} Hz\n")

    print(f"  {'t':>6}  {'true A':>8}  {'est A':>8}  {'est f':>8}")
    for ti in [0.1, 0.5, 1.0, 1.5, 1.9]:
        idx = int(ti * fs)
        print(f"  {ti:>6.2f}  {A_true[idx]:>8.3f}  {amp[idx]:>8.3f}  {freq[idx]:>8.3f}")

    # Chirp: frequency ramps from 20 Hz to 80 Hz
    chirp_freq = 20 + 30 * t
    x2 = np.cos(2 * np.pi * (20 * t + 15 * t ** 2))
    _, _, f_est = instantaneous(x2, fs=fs)
    print(f"\n  Chirp signal (freq 20 -> 80 Hz):")
    for ti in [0.1, 0.5, 1.0, 1.5, 1.9]:
        idx = int(ti * fs)
        print(f"    t = {ti:>4.2f}   true freq = {chirp_freq[idx]:>5.1f} Hz   est = {f_est[idx]:>5.1f} Hz")

    print("\n--- library cross-check (scipy.signal.hilbert; matlab hilbert) ---")
