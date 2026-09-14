"""Short-Time Fourier Transform (Allen & Rabiner 1977;
Portnoff 1976; Griffin-Lim 1984).

Slice the signal into overlapping windowed segments and take
the FFT of each:

    X(k, m) = sum_n x[n] * w[n - mH] * exp(-j 2*pi*k*n/N)

where w is a taper (Hann), N is the FFT length, H is the hop.

Produces a time-frequency (spectrogram) representation; the
Griffin-Lim algorithm inverts the magnitude STFT back to time
domain.
"""

import numpy as np    # arrays + fft


def stft(x, n_fft, hop, window):
    """Return complex STFT matrix of shape (n_frames, n_fft // 2 + 1)."""
    n = len(x)
    n_frames = 1 + (n - n_fft) // hop
    stft_out = np.empty((n_frames, n_fft // 2 + 1), dtype=complex)
    for i in range(n_frames):
        seg = x[i * hop: i * hop + n_fft] * window
        stft_out[i] = np.fft.rfft(seg)
    return stft_out


def istft(X, hop, window):
    """Overlap-add reconstruction (Allen 1977 constant-overlap-add)."""
    n_frames, n_bins = X.shape
    n_fft = 2 * (n_bins - 1)
    n = (n_frames - 1) * hop + n_fft
    y = np.zeros(n)
    wsum = np.zeros(n)
    for i in range(n_frames):
        seg = np.fft.irfft(X[i], n=n_fft)
        y[i * hop: i * hop + n_fft] = y[i * hop: i * hop + n_fft] + seg * window
        wsum[i * hop: i * hop + n_fft] = wsum[i * hop: i * hop + n_fft] + window ** 2
    y[wsum > 1e-10] = y[wsum > 1e-10] / wsum[wsum > 1e-10]
    return y


def demo():
    print("=== Short-Time Fourier Transform (Allen-Rabiner 1977) ===")
    fs = 1000
    T = 2.0
    t = np.linspace(0, T, int(fs * T), endpoint=False)
    # linear chirp: 50 Hz -> 250 Hz over 2 seconds
    freq_of_t = 50 + (250 - 50) * (t / T)
    phase = 2 * np.pi * np.cumsum(freq_of_t) / fs
    x = np.sin(phase)

    n_fft = 256
    hop = 64
    window = np.hanning(n_fft)
    X = stft(x, n_fft, hop, window)
    print(f"  Signal length {len(x)}, STFT shape = {X.shape} (frames x bins)")

    # locate peak bin per frame
    peaks = np.argmax(np.abs(X), axis=1)
    peak_hz = peaks * fs / n_fft
    frame_times = np.arange(X.shape[0]) * hop / fs + n_fft / (2 * fs)
    print("  Peak frequency vs time (samples):")
    for i in [0, X.shape[0] // 4, X.shape[0] // 2, 3 * X.shape[0] // 4, X.shape[0] - 1]:
        true_freq = 50 + (250 - 50) * frame_times[i] / T
        print(f"    t={frame_times[i]:.2f}s : detected {peak_hz[i]:>5.1f} Hz, "
              f"true {true_freq:>5.1f} Hz")

    y = istft(X, hop, window)
    # trim to compare (edges have COLA transient)
    n_valid = min(len(x), len(y))
    err = np.sqrt(np.mean((x[n_fft:n_valid - n_fft] - y[n_fft:n_valid - n_fft]) ** 2))
    print(f"\n  RMS reconstruction error (interior) = {err:.2e}")


if __name__ == "__main__":
    demo()
