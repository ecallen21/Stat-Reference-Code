"""Taguchi methods (Reference Sec 17.15).

Genichi Taguchi 1986.  Robust-design methodology for engineering
quality:

  * INNER ARRAY (control factors) x OUTER ARRAY (noise factors).
  * ORTHOGONAL ARRAYS (L4, L8, L9, L16, L18, L27) reduce runs
    while preserving main-effect orthogonality.
  * SIGNAL-TO-NOISE RATIOS quantify robustness:
      Smaller-the-better  : SNR = -10 log(mean(y^2))
      Larger-the-better   : SNR = -10 log(mean(1/y^2))
      Nominal-the-best    : SNR = 10 log(mean(y)^2 / var(y))

Choose control-factor levels that MAXIMISE SNR while separately
tuning the mean.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def L4_array():
    """Standard L4(2^3): 4 runs, 3 factors at 2 levels each."""
    return np.array([[1, 1, 1],
                      [1, 2, 2],
                      [2, 1, 2],
                      [2, 2, 1]])


def L9_array():
    """L9(3^4): 9 runs, 4 factors at 3 levels each."""
    return np.array([[1, 1, 1, 1],
                      [1, 2, 2, 2],
                      [1, 3, 3, 3],
                      [2, 1, 2, 3],
                      [2, 2, 3, 1],
                      [2, 3, 1, 2],
                      [3, 1, 3, 2],
                      [3, 2, 1, 3],
                      [3, 3, 2, 1]])


def snr(y, kind="nominal-the-best"):
    y = np.asarray(y, dtype=float)
    if kind == "smaller-the-better":
        return float(-10 * np.log10((y ** 2).mean()))
    if kind == "larger-the-better":
        return float(-10 * np.log10((1 / (y ** 2)).mean()))
    return float(10 * np.log10((y.mean() ** 2) / max(y.var(ddof=1), 1e-12)))


if __name__ == "__main__":
    print("=== Taguchi methods: L9 orthogonal array + SNR ===\n")
    L9 = L9_array()
    print(f"  L9 array:\n{L9}\n")

    # Simulate: 4 control factors, response depends on 2 of them + noise
    rng = np.random.default_rng(0)
    n_noise = 5           # 5 outer-array replicates per row
    Y = np.zeros((9, n_noise))
    for i, row in enumerate(L9):
        for j in range(n_noise):
            Y[i, j] = 8 - 0.3 * row[0] + 0.7 * (row[1] == 3) + rng.normal(0, 0.4)

    # SNR per run (nominal-the-best target = 8)
    snr_vals = np.array([snr(Y[i], "nominal-the-best") for i in range(9)])
    means = Y.mean(axis=1)
    print(f"  {'run':>3s}  factors A/B/C/D  {'mean':>8s}  {'SNR':>7s}")
    for i in range(9):
        print(f"  {i + 1:>3d}  {L9[i].tolist()}      {means[i]:>8.3f}  {snr_vals[i]:>7.2f}")

    # Per-factor-level SNR average -> pick level maximising SNR
    print("\n  Per-level mean SNR:")
    for f in range(4):
        for lv in (1, 2, 3):
            mask = L9[:, f] == lv
            print(f"    factor {chr(65 + f)} level {lv}: mean SNR = {snr_vals[mask].mean():.2f}")
        print()

    print("--- library cross-check (R DoE.base::oa.design, qualityTools; Python pyDOE2) ---")
