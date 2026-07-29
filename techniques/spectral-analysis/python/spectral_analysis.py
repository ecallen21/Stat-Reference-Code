"""Spectral analysis (Reference §13.18).

Frequency-domain view of a stationary time series.  A stationary series can
be decomposed into a superposition of sinusoids at different frequencies;
the SPECTRAL DENSITY f(omega) describes how variance is distributed across
those frequencies.  Peaks in f(omega) reveal periodicities.

Raw periodogram
    I(omega_k) = (1 / T) |sum_t y_t e^{-i omega_k t}|^2      omega_k = 2 pi k / T

The periodogram is asymptotically unbiased for f but INCONSISTENT --
its variance does not shrink as T grows.  Two standard fixes:

1) Smoothing (Daniell / Bartlett-Priestley kernel over frequencies)
    Trades bias for variance; wider smoother = smoother spectrum but
    peaks get blurred.

2) Welch's method (segment + window + average)
    Split y into overlapping segments, apply a taper (Hann / Hamming),
    compute periodogram on each, average across segments.  scipy.signal.welch.

Applications
    - Detecting periodicities (daily / weekly / annual cycles).
    - AR / MA process diagnostics: AR(1) with positive phi has monotone
      declining f; white noise has flat f.
    - EEG / seismic / audio analysis.

Deferred to a separate technique
    Wavelet analysis (§13.19), EMD (§13.58), locally stationary spectra
    (§13.59) - for signals whose spectral content changes over time.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def periodogram(y, fs: float = 1.0) -> dict:
    """Raw periodogram of a real time series."""
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    T = len(y)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    I = (np.abs(Y) ** 2) / T
    return {"frequency": freqs, "spectrum": I, "T": int(T), "fs": float(fs),
            "method": "Raw periodogram"}


def daniell_periodogram(y, m: int = 3, fs: float = 1.0) -> dict:
    """Daniell-smoothed periodogram: running mean of width 2m+1 over frequencies."""
    r = periodogram(y, fs)
    I = r["spectrum"]
    w = 2 * m + 1
    kernel = np.ones(w) / w
    # Reflect edges to avoid edge bias
    padded = np.concatenate([I[m:0:-1], I, I[-2:-m - 2:-1]])
    smoothed = np.convolve(padded, kernel, mode="valid")
    return {"frequency": r["frequency"], "spectrum": smoothed,
            "smoothing_width": w,
            "T": r["T"], "fs": r["fs"],
            "method": f"Daniell-smoothed periodogram (width {w})"}


def welch_spectrum(y, nperseg: int = 128, overlap: float = 0.5, fs: float = 1.0) -> dict:
    """Welch's averaged modified periodogram with a Hann window."""
    y = np.asarray(y, dtype=float); y = y - y.mean(); T = len(y)
    step = max(1, int(nperseg * (1 - overlap)))
    window = 0.5 - 0.5 * np.cos(2 * math.pi * np.arange(nperseg) / (nperseg - 1))
    U = (window ** 2).sum()  # window normalization
    segs = []
    start = 0
    while start + nperseg <= T:
        seg = y[start:start + nperseg] * window
        Y = np.fft.rfft(seg)
        segs.append((np.abs(Y) ** 2) / (U * fs))
        start += step
    S = np.mean(segs, axis=0)
    # One-sided normalization: double all interior bins
    S = S.copy(); S[1:-1] *= 2
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return {"frequency": freqs, "spectrum": S,
            "n_segments": len(segs), "nperseg": nperseg,
            "T": int(T), "fs": float(fs),
            "method": "Welch averaged periodogram with Hann window"}


def dominant_period(y, fs: float = 1.0, method: str = "welch", **kw) -> dict:
    """Report the frequency with maximum spectral power and its corresponding period."""
    if method == "welch":
        r = welch_spectrum(y, fs=fs, **kw)
    elif method == "daniell":
        r = daniell_periodogram(y, fs=fs, **kw)
    else:
        r = periodogram(y, fs=fs)
    # Skip 0 frequency
    idx = np.argmax(r["spectrum"][1:]) + 1
    f_peak = float(r["frequency"][idx])
    return {"dominant_frequency": f_peak,
            "dominant_period": float(1 / f_peak) if f_peak > 0 else float("inf"),
            "peak_power": float(r["spectrum"][idx]),
            "method": r["method"]}


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    T = 512; fs = 1.0
    t = np.arange(T)
    # Signal: two sinusoids at periods 20 and 8 plus noise
    y = 1.5 * np.sin(2 * math.pi * t / 20) + 0.7 * np.sin(2 * math.pi * t / 8) + rng.normal(0, 0.5, T)

    print("=== Raw periodogram peak ===")
    r = dominant_period(y, method="raw")
    print(f"  peak freq = {r['dominant_frequency']:.4f}  ->  period = {r['dominant_period']:.2f}")

    print("\n=== Daniell-smoothed peak ===")
    r = dominant_period(y, method="daniell", m=3)
    print(f"  peak freq = {r['dominant_frequency']:.4f}  ->  period = {r['dominant_period']:.2f}")

    print("\n=== Welch peak ===")
    r = dominant_period(y, method="welch", nperseg=128)
    print(f"  peak freq = {r['dominant_frequency']:.4f}  ->  period = {r['dominant_period']:.2f}")

    print("\n--- library cross-check (scipy.signal.welch) ---")
    try:
        from scipy import signal
        f_sp, S_sp = signal.welch(y - y.mean(), fs=fs, nperseg=128)
        idx = np.argmax(S_sp[1:]) + 1
        print(f"  scipy welch peak freq = {f_sp[idx]:.4f}  ->  period = {1 / f_sp[idx]:.2f}")
    except Exception as ex:
        print(f"  (scipy signal not available: {ex})")
