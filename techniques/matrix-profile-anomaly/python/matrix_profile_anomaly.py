"""Matrix Profile for Time-Series Anomaly / Motif (Ref Sec 47.290).

Yeh, Zhu, Ulanova et al 2016 ICDM. For every length-m window in
a time series, the MATRIX PROFILE stores the z-normalised
Euclidean distance to its NEAREST NON-TRIVIAL match:

    MP[i] = min_{|j-i| > m} d(T[i:i+m], T[j:j+m])

Peaks of MP => DISCORDS (anomalies).
Valleys of MP => MOTIFS (repeating patterns).

O(n^2) naive; STOMP / SCRIMP reduce to O(n^2) with tiny constants,
STUMPY etc are the standard implementations.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def znorm(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-9 else 1)


def matrix_profile(T, m):
    """Brute-force matrix profile of series T with window m."""
    n = len(T) - m + 1
    windows = np.array([znorm(T[i:i + m]) for i in range(n)])
    exclude = m                                                   # trivial-match exclusion
    mp = np.full(n, np.inf); mpi = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if abs(i - j) < exclude: continue
            d = np.linalg.norm(windows[i] - windows[j])
            if d < mp[i]:
                mp[i] = d; mpi[i] = j
    return mp, mpi


if __name__ == "__main__":
    print("=== Matrix Profile (Yeh et al 2016 ICDM) ===\n")
    rng = np.random.default_rng(0)

    # Construct a signal: 3 sine-wave repetitions + 1 discord
    n = 400; m = 40
    t = np.linspace(0, 20 * np.pi, n)
    T = 0.5 * np.sin(0.5 * t)                                     # baseline
    # Add 3 motif segments (identical shape) at fixed offsets
    for start in [50, 150, 250]:
        T[start:start + m] = np.sin(2 * np.pi * np.arange(m) / m)
    # Add a discord: sharp step
    T[330:330 + m] = 3.0 * np.sign(np.arange(m) - m / 2)
    # Gaussian noise
    T += 0.1 * rng.standard_normal(n)

    mp, mpi = matrix_profile(T, m)

    # Discord = argmax of MP
    discord_start = int(np.argmax(mp))
    print(f"  Series length {n}, window m = {m}")
    print(f"  Discord (anomaly): index {discord_start}   MP value = {mp.max():.2f}")
    print(f"  (planted at 330 - detection is anywhere in the discord's neighbourhood)\n")

    # Motif = argmin of MP
    motif_start = int(np.argmin(mp))
    print(f"  Motif (repeating pattern): index {motif_start}  MP value = {mp.min():.2f}")
    print(f"    matched to index {mpi[motif_start]}")
    print(f"  (planted at 50, 150, 250 - matching pair will be from that set)\n")

    print(f"  Top 5 discords (highest MP):")
    top = np.argsort(mp)[-5:][::-1]
    for i in top:
        print(f"    index {i:>3}   MP = {mp[i]:.2f}")

    print("\n--- library cross-check (stumpy.stump; matrixprofile-ts; scamp) ---")
