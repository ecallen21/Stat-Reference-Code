"""Sample Entropy (Reference Sec 47.329).

Richman & Moorman 2000 Am J Physiol. Signal-complexity measure
robust to short-length bias in ApEn. Given tolerance r and
template length m:

    A = # pairs of length-(m+1) sequences within r
    B = # pairs of length-m     sequences within r
    SampEn(m, r, N) = -log(A / B)

Lower values = more regular / self-similar signal; higher =
more random. Common in HRV, EEG, and fault-detection analytics.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sample_entropy(x, m=2, r=None):
    x = np.asarray(x, float); N = len(x)
    if r is None: r = 0.2 * x.std()
    def count(seqs):
        n = len(seqs)
        c = 0
        for i in range(n):
            d = np.max(np.abs(seqs - seqs[i]), axis=1)
            c += int(np.sum(d <= r) - 1)                          # exclude self
        return c
    seqs_m = np.array([x[i:i + m] for i in range(N - m)])
    seqs_m1 = np.array([x[i:i + m + 1] for i in range(N - m)])
    B = count(seqs_m); A = count(seqs_m1)
    if A == 0 or B == 0: return float("inf")
    return -np.log(A / B)


if __name__ == "__main__":
    print("=== Sample Entropy (Richman & Moorman 2000) ===\n")
    rng = np.random.default_rng(0)

    N = 500
    # Signals with known complexity: sine, chaotic (logistic map), gaussian white noise
    t = np.arange(N)
    x_sine = np.sin(0.1 * t)
    # Logistic map r=4 (fully chaotic)
    x_chaos = np.zeros(N); x_chaos[0] = 0.7
    for i in range(1, N): x_chaos[i] = 4 * x_chaos[i - 1] * (1 - x_chaos[i - 1])
    x_gauss = rng.standard_normal(N)

    for name, x in [("pure sine", x_sine), ("chaotic map", x_chaos), ("gaussian noise", x_gauss)]:
        se = sample_entropy(x, m=2, r=0.2 * x.std())
        print(f"  {name:>15}   SampEn(2, 0.2*std) = {se:.3f}")

    print(f"\n  Pure sine has near-zero SampEn (highly regular).")
    print(f"  Chaotic map is INTERMEDIATE (deterministic but irregular).")
    print(f"  Gaussian noise has highest SampEn.\n")

    # Tolerance sensitivity
    print(f"  Tolerance sensitivity on gaussian noise:")
    for r_frac in [0.1, 0.2, 0.5]:
        se = sample_entropy(x_gauss, m=2, r=r_frac * x_gauss.std())
        print(f"    r = {r_frac} * std   SampEn = {se:.3f}")

    print("\n--- library cross-check (antropy.sample_entropy; nolds.sampen; pyeeg) ---")
