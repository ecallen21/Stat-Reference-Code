"""Welch power spectral density (Reference Sec 47.94).

Welch 1967 'The use of fast Fourier transform for the estimation
of power spectra: a method based on time averaging over short,
modified periodograms', IEEE Trans Audio 15(2). Estimate the PSD
S(f) of a wide-sense stationary signal by:

    1. Split x into K overlapping (50%) segments of length L.
    2. Multiply each by a window w (Hann default) to reduce leakage.
    3. Compute |FFT|^2 of each windowed segment.
    4. Average.

Bias-variance trade-off: longer L -> finer freq resolution but
noisier estimate; more segments (overlap) -> less variance.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def welch_psd(x, fs, nperseg=256, noverlap=None, window="hann"):
    n = len(x)
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap
    if window == "hann":
        w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nperseg) / (nperseg - 1))
    else:
        w = np.ones(nperseg)
    U = (w * w).sum()      # normalisation for the window
    Sxx = np.zeros(nperseg // 2 + 1)
    K = 0
    for start in range(0, n - nperseg + 1, step):
        seg = x[start:start + nperseg] * w
        X = np.fft.rfft(seg)
        Sxx += np.abs(X) ** 2
        K += 1
    Sxx /= K * fs * U
    Sxx[1:-1] *= 2      # one-sided
    f = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return f, Sxx


if __name__ == "__main__":
    print("=== Welch PSD (Welch 1967) ===\n")
    rng = np.random.default_rng(0)
    fs = 1000.0
    T = 4.0
    N = int(T * fs)
    t = np.arange(N) / fs

    # Tones at 60 Hz (amp 1) and 220 Hz (amp 0.5) + white noise
    x = 1.0 * np.sin(2 * np.pi * 60 * t) + 0.5 * np.sin(2 * np.pi * 220 * t) \
        + 0.3 * rng.normal(size=N)

    for L in [256, 512, 1024, 2048]:
        f, S = welch_psd(x, fs, nperseg=L)
        # Peaks
        peaks = np.argsort(S)[::-1][:3]
        peak_freqs = np.sort(f[peaks])
        # Total power in +-2 Hz around 60 and 220
        p60 = float(S[(f > 58) & (f < 62)].sum() * (f[1] - f[0]))
        p220 = float(S[(f > 218) & (f < 222)].sum() * (f[1] - f[0]))
        print(f"  nperseg = {L:4d}   df = {f[1]-f[0]:6.3f} Hz   "
              f"top-3 peak freqs = {np.round(peak_freqs, 1)}   "
              f"P(60) = {p60:.3f}, P(220) = {p220:.3f}")

    # Parseval-consistent scaling check
    var_x = x.var()
    total_pwr = float(S.sum() * (f[1] - f[0]))
    print(f"\n  Total PSD area = {total_pwr:.3f}   sample variance = {var_x:.3f}  (should match)")

    print("\n--- library cross-check (spectral::spectrum R; scipy.signal.welch Python) ---")
