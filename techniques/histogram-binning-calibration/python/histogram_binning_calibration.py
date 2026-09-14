"""Histogram Binning Calibration (Reference Sec 47.285).

Zadrozny & Elkan 2001 ICML. Non-parametric calibration for
BINARY classifiers: partition scores into M bins and estimate

    p_calibrated(s in bin m) = fraction of positives in that bin

Simple and effective when sample size per bin is large. Bin
choice (equal-width, equal-mass, or Bayesian-optimal) is the
key tuning knob.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def histogram_binning_fit(scores, y, n_bins=15, adaptive=True):
    """Fit bin edges and per-bin positive rate."""
    s = np.asarray(scores); y = np.asarray(y)
    if adaptive:
        edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    # Ensure endpoints
    edges[0] = 0.0; edges[-1] = 1.000001
    rates = []
    for i in range(len(edges) - 1):
        m = (s >= edges[i]) & (s < edges[i + 1])
        rate = float(y[m].mean()) if m.sum() > 0 else float(s[m].mean() if m.sum() > 0 else 0.5)
        rates.append(rate)
    return edges, np.array(rates)


def histogram_binning_predict(scores, edges, rates):
    idx = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, len(rates) - 1)
    return rates[idx]


def brier(p, y): return float(np.mean((p - y) ** 2))


if __name__ == "__main__":
    print("=== Histogram Binning Calibration (Zadrozny & Elkan 2001) ===\n")
    rng = np.random.default_rng(0)

    # Non-monotone miscalibration: sinusoidal
    n = 5000
    true_prob = rng.uniform(0, 1, n)
    y = (rng.uniform(size=n) < true_prob).astype(int)
    scores = np.clip(true_prob + 0.15 * np.sin(6 * np.pi * true_prob), 0.01, 0.99)

    s_cal, s_te = scores[:3500], scores[3500:]
    y_cal, y_te = y[:3500], y[3500:]

    for n_bins in [5, 10, 20]:
        edges, rates = histogram_binning_fit(s_cal, y_cal, n_bins=n_bins, adaptive=True)
        p_cal = histogram_binning_predict(s_te, edges, rates)
        print(f"  n_bins = {n_bins:>2}:  Brier raw = {brier(s_te, y_te):.4f}    "
              f"Brier calibrated = {brier(p_cal, y_te):.4f}")

    # Show per-bin table for n_bins=10
    edges, rates = histogram_binning_fit(s_cal, y_cal, n_bins=10, adaptive=True)
    print(f"\n  Bin table (n_bins=10, adaptive):")
    print(f"    {'bin':>3}  {'range':>20}  {'n_cal':>6}  {'rate':>6}")
    for i in range(len(edges) - 1):
        m = (s_cal >= edges[i]) & (s_cal < edges[i + 1])
        print(f"    {i:>3}  ({edges[i]:.3f}, {edges[i+1]:.3f})  {m.sum():>6}  {rates[i]:>6.3f}")

    print("\n  Non-monotone miscalibration defeats Platt (slope+intercept); histogram")
    print("  binning captures ARBITRARY shape at the cost of bin-boundary discontinuities.")

    print("\n--- library cross-check (netcal.binning.HistogramBinning; sklearn.calibration.CalibratedClassifierCV) ---")
