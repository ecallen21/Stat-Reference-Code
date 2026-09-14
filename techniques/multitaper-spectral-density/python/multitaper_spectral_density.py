"""Multitaper Spectral Estimation (Reference Sec 47.332).

Thomson 1982 Proc IEEE. Average K periodograms of the same
series computed under K discrete prolate spheroidal sequences
(DPSS, aka Slepian tapers):

    S_hat(f) = 1/K sum_k |sum_t x_t * w_k(t) * exp(-i 2 pi f t)|^2

Trades a small bias for a HUGE reduction in variance vs a
single-taper periodogram; standard for stationary geophysics,
neuroscience, and astrophysics spectra.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def multitaper_psd(x, fs=1.0, NW=4, K=None):
    from scipy.signal.windows import dpss
    from scipy.signal import periodogram
    N = len(x)
    if K is None: K = 2 * NW - 1
    tapers = dpss(N, NW, Kmax=K)                                    # K x N
    psd_list = []
    for w in tapers:
        _, P = periodogram(x * w, fs=fs, nfft=N)
        psd_list.append(P)
    return _, np.mean(np.array(psd_list), axis=0), K


if __name__ == "__main__":
    print("=== Multitaper Spectral Estimation (Thomson 1982) ===\n")
    rng = np.random.default_rng(0)

    from scipy.signal import periodogram, welch
    fs = 1000
    t = np.arange(0, 4, 1 / fs)
    x = np.sin(2 * np.pi * 60 * t) + 0.5 * np.sin(2 * np.pi * 120 * t) + rng.normal(0, 0.5, len(t))

    f_p, P_period = periodogram(x, fs=fs)
    f_w, P_welch = welch(x, fs=fs, nperseg=1024)
    f_mt, P_mt, K = multitaper_psd(x, fs=fs, NW=4)

    print(f"  Signal: 60 Hz + 0.5*120 Hz + Gaussian noise, {len(x)} samples at {fs} Hz")
    print(f"  Methods: periodogram (single window), Welch (segmented), multitaper (K={K})\n")

    # Peaks
    def peak_freq(f, P, target):
        idx = np.argmin(np.abs(f - target))
        return float(P[idx])
    print(f"  Peak height at 60 Hz vs baseline (~30 Hz):")
    print(f"    Periodogram:  {peak_freq(f_p, P_period, 60):.2f} / {peak_freq(f_p, P_period, 30):.4f}")
    print(f"    Welch:        {peak_freq(f_w, P_welch, 60):.2f} / {peak_freq(f_w, P_welch, 30):.4f}")
    print(f"    Multitaper:   {peak_freq(f_mt, P_mt, 60):.2f} / {peak_freq(f_mt, P_mt, 30):.4f}")

    # Variance around baseline noise floor
    band_idx = (f_p > 200) & (f_p < 400)                            # noise region
    print(f"\n  PSD std in noise band 200-400 Hz (lower = better variance):")
    print(f"    Periodogram: {np.std(P_period[band_idx]):.4f}")
    band_idx_w = (f_w > 200) & (f_w < 400)
    print(f"    Welch:       {np.std(P_welch[band_idx_w]):.4f}")
    band_idx_mt = (f_mt > 200) & (f_mt < 400)
    print(f"    Multitaper:  {np.std(P_mt[band_idx_mt]):.4f}")

    print("\n--- library cross-check (mne.time_frequency.psd_multitaper; nitime; spectrum) ---")
