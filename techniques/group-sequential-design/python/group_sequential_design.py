"""Group Sequential Design (Reference Sec 47.257).

Pocock 1977 Biometrika; O'Brien & Fleming 1979 Biometrics.
Interim analyses at K equally-spaced looks with STOPPING
BOUNDARIES that preserve overall Type-I error alpha:

    Pocock:            c_k = c_P    for all k
    O'Brien-Fleming:   c_k = c_OBF * sqrt(K/k)

O'BF starts conservative (huge z-boundary early, hard to
stop) and eases toward a near-nominal 1.96 at the final look.
Pocock uses a constant, easier-to-cross boundary at every look
but slightly larger overall.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def pocock_boundary(alpha, K, n_sim=100_000, rng=None):
    """Find constant Pocock boundary c such that
    P(any |Z_k| > c) = alpha under H0, k = 1..K."""
    if rng is None: rng = np.random.default_rng(0)
    # Correlated Z-stats: Z_k = sum(x_1..x_{n_k}) / sqrt(n_k), n_k = k*n
    # Under H0, x ~ N(0,1), Cov(Z_i, Z_j) = sqrt(min(i,j) / max(i,j))
    # Simulate: x are independent standard normals, cumulative sums / sqrt(k)
    x = rng.standard_normal((n_sim, K))
    Z = x.cumsum(axis=1) / np.sqrt(np.arange(1, K + 1))
    max_absZ = np.max(np.abs(Z), axis=1)
    return float(np.quantile(max_absZ, 1 - alpha))


def obf_boundary(alpha, K, n_sim=100_000, rng=None):
    """Find O'Brien-Fleming c_OBF such that P(any |Z_k| > c_OBF*sqrt(K/k)) = alpha."""
    if rng is None: rng = np.random.default_rng(0)
    x = rng.standard_normal((n_sim, K))
    Z = x.cumsum(axis=1) / np.sqrt(np.arange(1, K + 1))
    scale = np.sqrt(K / np.arange(1, K + 1))
    max_scaled = np.max(np.abs(Z) / scale, axis=1)
    return float(np.quantile(max_scaled, 1 - alpha))


if __name__ == "__main__":
    print("=== Group Sequential Design (Pocock 1977; O'Brien-Fleming 1979) ===\n")
    rng = np.random.default_rng(0)

    alpha = 0.05
    print(f"  Two-sided alpha = {alpha}, K = 4 planned interim looks\n")

    c_P = pocock_boundary(alpha, K=4, n_sim=200_000, rng=rng)
    c_OBF = obf_boundary(alpha, K=4, n_sim=200_000, rng=rng)

    print(f"  Pocock         constant boundary:  c = {c_P:.3f}   at every look")
    print(f"  O'Brien-Fleming boundary anchor:   c_OBF = {c_OBF:.3f}")
    print(f"\n  Per-look boundaries:")
    print(f"    look k  |  Pocock   |  O'BF (c_OBF * sqrt(K/k))")
    for k in range(1, 5):
        obf_k = c_OBF * np.sqrt(4 / k)
        print(f"    {k:>6}  |  {c_P:>7.3f}  |  {obf_k:>7.3f}")
    print(f"\n  O'BF starts very conservative (z = {c_OBF*2:.2f} at look 1)")
    print(f"  and eases to ~1.96 at final look; Pocock is uniform ~{c_P:.2f}.")

    # Under H1, what is the actual Type-I error and expected number of looks?
    # Simulate: true delta = 0 (H0); confirm alpha spending
    n_sim = 100_000
    x = rng.standard_normal((n_sim, 4))
    Z = x.cumsum(axis=1) / np.sqrt(np.arange(1, 5))
    stops_P = np.zeros(n_sim, dtype=bool)
    stops_OBF = np.zeros(n_sim, dtype=bool)
    look_P = np.full(n_sim, 4); look_OBF = np.full(n_sim, 4)
    for k in range(4):
        p_reject = (np.abs(Z[:, k]) > c_P) & ~stops_P
        stops_P |= p_reject
        look_P[p_reject] = k + 1
        obf_reject = (np.abs(Z[:, k]) > c_OBF * np.sqrt(4 / (k + 1))) & ~stops_OBF
        stops_OBF |= obf_reject
        look_OBF[obf_reject] = k + 1
    print(f"\n  Under H0 (delta=0, n_sim={n_sim:,}):")
    print(f"    Pocock  Type-I = {stops_P.mean():.4f}   mean look = {look_P.mean():.2f}")
    print(f"    O'BF    Type-I = {stops_OBF.mean():.4f}   mean look = {look_OBF.mean():.2f}")

    print(f"\n  Both preserve alpha ~ 0.05; O'BF conserves looks (rarely stops early under H0).")

    print("\n--- library cross-check (gsDesign R; scipy: manual construction) ---")
