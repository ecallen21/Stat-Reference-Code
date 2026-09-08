"""Wasserstein GAN (Reference Sec 47.109).

Arjovsky, Chintala & Bottou 2017 'Wasserstein GAN', ICML. Replaces
JS divergence in the original GAN with the EARTH-MOVER'S (Wasserstein-1)
distance:

    W_1(P, Q) = sup_{|f|_L <= 1}  E_P[f] - E_Q[f].

Trains a Lipschitz-1 CRITIC f via weight clipping (WGAN) or a
gradient penalty (WGAN-GP; Gulrajani et al 2017). Empirical benefits:
    * meaningful loss curve correlating with sample quality
    * far less mode collapse than JS-GAN
    * no need for careful G / D balancing.

Illustrated here via the closed-form 1-D Wasserstein-1 distance
between empirical measures (via sorted quantile matching), and by
comparing to a KDE / JS surrogate.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def wasserstein_1d(x, y):
    """Exact W_1 between two 1-D empirical measures via sorted quantiles."""
    x_sorted = np.sort(x); y_sorted = np.sort(y)
    if len(x_sorted) != len(y_sorted):
        # Interpolate to common quantile grid
        q = np.linspace(0, 1, max(len(x_sorted), len(y_sorted)))
        Fx = np.quantile(x_sorted, q); Fy = np.quantile(y_sorted, q)
    else:
        Fx, Fy = x_sorted, y_sorted
    return float(np.mean(np.abs(Fx - Fy)))


def js_divergence_hist(x, y, bins=40):
    """Jensen-Shannon on histograms -- rough JS-GAN surrogate."""
    lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
    edges = np.linspace(lo, hi, bins + 1)
    P, _ = np.histogram(x, bins=edges, density=True)
    Q, _ = np.histogram(y, bins=edges, density=True)
    P = P / max(P.sum(), 1e-12); Q = Q / max(Q.sum(), 1e-12)
    M = 0.5 * (P + Q)
    def _kl(a, b):
        return float(np.sum(np.where(a > 0, a * np.log((a + 1e-12) / (b + 1e-12)), 0.0)))
    return 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)


if __name__ == "__main__":
    print("=== Wasserstein GAN distance vs JS (Arjovsky-Chintala-Bottou 2017) ===\n")
    rng = np.random.default_rng(0)
    n = 1000
    print("  Two 'thin' distributions with sliding support overlap:\n")
    print(f"  {'shift':>7s}   {'W_1 (Wasserstein)':>18s}   {'JS (density)':>15s}")
    for shift in [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]:
        P = rng.normal(loc=0.0, scale=0.1, size=n)
        Q = rng.normal(loc=shift, scale=0.1, size=n)
        W1 = wasserstein_1d(P, Q)
        JS = js_divergence_hist(P, Q)
        print(f"  {shift:7.2f}   {W1:18.4f}   {JS:15.4f}")

    print("\n  Wasserstein grows SMOOTHLY with the shift (informative gradient).")
    print("  JS saturates near log(2) once supports disjoint (vanishing gradient),")
    print("  causing the original GAN's training instabilities WGAN was designed to fix.")

    print("\n--- library cross-check (limited R; torch + torch-geometric WGAN Python) ---")
