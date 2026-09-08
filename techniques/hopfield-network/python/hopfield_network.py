"""Hopfield network (Reference Sec 47.90).

Hopfield 1982 'Neural networks and physical systems with emergent
collective computational abilities', PNAS 79. Recurrent binary
network for content-addressable memory. Store P patterns
xi^k in {-1, +1}^N via Hebbian outer-product rule:

    W = (1/N) sum_k xi^k (xi^k)^T ,   W_ii = 0.

Asynchronous update:
    s_i <- sign( sum_j W_ij s_j ).

Converges to a fixed point (local energy minimum). Capacity
~0.14 N patterns before spurious minima dominate (Amit-Gutfreund-
Sompolinsky 1985). Modern continuous Hopfield (Ramsauer et al 2020)
recovers with exponential capacity.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def hopfield_train(patterns):
    """Hebbian storage of P patterns in {-1, +1}^N."""
    P, N = patterns.shape
    W = patterns.T @ patterns / N
    np.fill_diagonal(W, 0)
    return W


def hopfield_recall(W, s, max_iter=100, rng=None):
    rng = rng or np.random.default_rng(0)
    N = len(s)
    s = s.copy().astype(int)
    for _ in range(max_iter):
        old = s.copy()
        for i in rng.permutation(N):
            s[i] = 1 if W[i] @ s >= 0 else -1
        if np.array_equal(s, old):
            break
    return s


if __name__ == "__main__":
    print("=== Hopfield network (Hopfield 1982) ===\n")
    rng = np.random.default_rng(0)
    N = 100
    for P in [5, 10, 15, 20]:
        patterns = rng.choice([-1, 1], size=(P, N))
        W = hopfield_train(patterns)
        # Test: flip a fraction of bits and see if the network recovers
        for flip_frac in [0.05, 0.15, 0.30]:
            hits = 0
            for k in range(P):
                s = patterns[k].copy()
                flips = rng.choice(N, size=int(flip_frac * N), replace=False)
                s[flips] *= -1
                s_recall = hopfield_recall(W, s, max_iter=20, rng=rng)
                hits += int(np.array_equal(s_recall, patterns[k]))
            print(f"  P = {P:2d} patterns, {int(flip_frac*100):2d}% flipped: "
                  f"exact recovery {hits}/{P}")
        print()

    print(f"  Capacity limit ~0.14 N = {0.14 * N:.0f} patterns.")
    print("\n--- library cross-check (limited R; hflayers Python via torch) ---")
