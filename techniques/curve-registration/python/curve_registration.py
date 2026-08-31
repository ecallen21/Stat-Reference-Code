"""Curve registration / phase-amplitude alignment (Reference Sec 31.11).

Ramsay-Silverman (2005) 'Functional Data Analysis', Ch 7.

Curves x_i(t) often differ by TIMING (phase) as well as MAGNITUDE
(amplitude). Un-aligned curves confuse FPCA / clustering / regression.

Two approaches:
  1. LANDMARK REGISTRATION: identify a small number of features
     (peak times, zero-crossings), warp t so those landmarks coincide.
  2. CONTINUOUS REGISTRATION: minimise a criterion that jointly
     estimates warp function h_i(t) and template mu(t).

Here we implement LANDMARK REGISTRATION with a piecewise-linear warp
+ show how it reduces cross-curve variance and aligns peaks.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def find_peak(x, t):
    """Return the t-coordinate of the largest value."""
    return float(t[np.argmax(x)])


def piecewise_linear_warp(t, landmarks_from, landmarks_to):
    """Interpolate t_new = h(t) so landmarks_from -> landmarks_to;
       linear between anchors, extrapolate at edges."""
    lm_f = np.concatenate([[t[0]], landmarks_from, [t[-1]]])
    lm_t = np.concatenate([[t[0]], landmarks_to, [t[-1]]])
    return np.interp(t, lm_f, lm_t)


def register_by_peak(X, t, target_peak=None):
    """Landmark register by aligning each curve's peak to target_peak."""
    n = X.shape[0]
    peaks = np.array([find_peak(X[i], t) for i in range(n)])
    if target_peak is None:
        target_peak = float(np.median(peaks))
    X_aligned = np.zeros_like(X)
    for i in range(n):
        t_new = piecewise_linear_warp(t, np.array([peaks[i]]),
                                        np.array([target_peak]))
        X_aligned[i] = np.interp(t, t_new, X[i])
    return X_aligned, peaks, target_peak


if __name__ == "__main__":
    print("=== Landmark curve registration (peak alignment) ===\n")
    rng = np.random.default_rng(0)
    T = 200
    t = np.linspace(0, 1, T)
    n = 25
    # Bell-shaped curves with random peak positions (phase misalignment).
    peaks_true = rng.uniform(0.35, 0.65, n)
    X = np.array([np.exp(-((t - p) ** 2) / (2 * 0.05 ** 2)) for p in peaks_true])
    X = X + 0.02 * rng.normal(0, 1, X.shape)

    var_before = float(np.mean(X.var(axis=0)))
    X_reg, peaks, target = register_by_peak(X, t)
    var_after = float(np.mean(X_reg.var(axis=0)))

    print(f"  peak positions (before registration): mean {peaks.mean():.3f}   sd {peaks.std():.3f}")
    print(f"  cross-curve mean var  before: {var_before:.4f}")
    print(f"  cross-curve mean var  after : {var_after:.4f}"
          f"   ({100 * (var_before - var_after) / var_before:.1f}% reduction)")

    # Recheck: peaks of aligned curves should all be ~ target.
    peaks_after = np.array([find_peak(X_reg[i], t) for i in range(n)])
    print(f"  aligned peak positions: mean {peaks_after.mean():.3f}   sd {peaks_after.std():.4f}"
          f"   (target = {target:.3f})\n")

    print("--- library cross-check (R fda::landmarkreg; fda.usc; Python fdasrsf) ---")
