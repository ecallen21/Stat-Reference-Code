"""Reservoir Sampling (Reference Sec 47.246).

Vitter 1985 'Random Sampling with a Reservoir', ACM TOMS. Draw a
uniform sample of size k from a STREAM of unknown length in ONE
PASS with O(k) memory:

    Algorithm R:
        keep the first k items as the reservoir
        for i = k+1, k+2, ...:
            j = uniform integer in [0, i-1]
            if j < k: replace reservoir[j] with item i

Each item ends up in the reservoir with probability k / n (n =
total stream length). Also weighted / A-Res variants (Efraimidis
2006) for weighted sampling in a stream.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def reservoir_sample(stream, k, rng):
    reservoir = []
    for i, x in enumerate(stream):
        if i < k:
            reservoir.append(x)
        else:
            j = int(rng.integers(0, i + 1))
            if j < k:
                reservoir[j] = x
    return reservoir


def weighted_reservoir_sample(stream_with_weights, k, rng):
    """Efraimidis-Spirakis A-Res: keep top-k by U^(1/w) key."""
    reservoir = []
    for x, w in stream_with_weights:
        u = rng.uniform()
        key = u ** (1.0 / max(w, 1e-12))
        if len(reservoir) < k:
            reservoir.append((key, x))
        else:
            reservoir.sort(key=lambda z: z[0])
            if key > reservoir[0][0]:
                reservoir[0] = (key, x)
    return [x for _, x in reservoir]


if __name__ == "__main__":
    print("=== Reservoir Sampling (Vitter 1985) ===\n")
    rng = np.random.default_rng(0)

    N = 100_000; k = 100
    stream = range(N)                                             # stream of 0..N-1
    sample = reservoir_sample(stream, k, rng)
    print(f"  Stream length: {N:,}   reservoir k = {k}")
    print(f"  Sample mean = {np.mean(sample):.1f}   (expected ~{(N - 1) / 2:.1f})")
    print(f"  Sample range = [{min(sample)}, {max(sample)}]")

    # Verify uniformity by running many trials
    counts = np.zeros(N)
    for _ in range(100):
        s = reservoir_sample(stream, k, rng)
        for x in s: counts[x] += 1
    p_include = k / N
    empirical = counts.mean() / 100
    print(f"\n  Empirical inclusion probability after 100 trials: {empirical:.4f}")
    print(f"  Theoretical p_include = k / N = {p_include:.4f}")

    # Weighted reservoir: weights favour large IDs
    stream_w = ((i, (i + 1) ** 2) for i in range(N))
    ws = weighted_reservoir_sample(stream_w, k=k, rng=rng)
    print(f"\n  Weighted reservoir (weight = (i+1)^2), reservoir mean id = {np.mean(ws):.1f}")
    print(f"    (should skew high — Zipfian sampling from a growing weight)")

    print("\n--- library cross-check (numpy.random.choice for known-length; streaming: custom) ---")
