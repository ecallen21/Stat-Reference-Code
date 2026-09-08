"""Butterworth Bandpass Filter (Reference Sec 47.144).

Butterworth 1930 'On the theory of filter amplifiers', Exp
Wireless & the Wireless Engr 7. Maximally flat magnitude response
in the passband:

    |H(jw)|^2 = 1 / (1 + (w/w_c)^{2n}).

Order-n filter with cutoff w_c. Bandpass = cascade of highpass
(remove low freqs) + lowpass (remove high freqs). Signal digital
form: bilinear-transform a continuous prototype to an IIR filter.

Practical for biosignal (ECG, EEG), audio, vibration filtering.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.signal import butter, filtfilt    # digital IIR design + zero-phase filtering


def butter_bandpass(low, high, fs, order=4):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return b, a


def filter_bandpass(x, low, high, fs, order=4):
    """Zero-phase forward-backward Butterworth bandpass."""
    b, a = butter_bandpass(low, high, fs, order)
    return filtfilt(b, a, x)


if __name__ == "__main__":
    print("=== Butterworth bandpass (Butterworth 1930) ===\n")

    rng = np.random.default_rng(0)

    fs = 1000.0                            # 1 kHz sampling
    T = 2.0                                # 2 seconds
    t = np.arange(0, T, 1 / fs)
    # 3 components: 2 Hz (low), 50 Hz (target), 300 Hz (high)
    x = (np.sin(2 * np.pi * 2 * t) + 1.0 * np.sin(2 * np.pi * 50 * t)
         + 0.7 * np.sin(2 * np.pi * 300 * t) + 0.1 * rng.normal(size=t.shape))

    for lo, hi in [(30, 70), (0.5, 10), (200, 400)]:
        y = filter_bandpass(x, lo, hi, fs, order=4)
        # Estimate energy in target band using DFT
        X = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(t), 1 / fs)
        band = (freqs >= lo) & (freqs <= hi)
        pct = np.abs(X[band]).sum() / np.abs(X).sum()
        print(f"  passband [{lo:5.1f}, {hi:5.1f}] Hz    "
              f"energy fraction in band = {pct:.3f}")

    # Sanity: the 50 Hz component should dominate after 30-70 filter
    y = filter_bandpass(x, 30, 70, fs, order=6)
    peak = float(freqs[np.argmax(np.abs(np.fft.rfft(y)))])
    print(f"\n  Bandpass 30-70 Hz peak frequency = {peak:.2f} Hz   (truth: 50)")

    print("\n--- library cross-check (scipy.signal.butter+filtfilt; signal R) ---")
