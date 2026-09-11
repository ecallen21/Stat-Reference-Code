"""Count-Min Sketch (Reference Sec 47.245).

Cormode & Muthukrishnan 2005 'An Improved Data Stream Summary: The
Count-Min Sketch and Its Applications', JAlg. Probabilistic
frequency estimator for streams: use k hash functions to store
counts in a k x w table, and query by taking the MIN of the k
counters:

    increment(x): for each hash h_j, table[j, h_j(x) % w] += 1
    estimate(x): return min_j table[j, h_j(x) % w]

Space O(k * w) regardless of stream size; error bounded by
epsilon * (total count) with probability 1 - delta, where
w = ceil(e / eps), k = ceil(ln 1/delta).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


class CountMinSketch:
    def __init__(self, w=256, k=5, seed=0):
        self.w, self.k = w, k
        self.table = np.zeros((k, w), dtype=int)
        rng = np.random.default_rng(seed)
        self.hash_seeds = rng.integers(1, 2 ** 31, size=k)

    def _hashes(self, x):
        return [(hash((int(s), x))) % self.w for s in self.hash_seeds]

    def add(self, x, c=1):
        for j, h in enumerate(self._hashes(x)):
            self.table[j, h] += c

    def estimate(self, x):
        return int(min(self.table[j, h] for j, h in enumerate(self._hashes(x))))


if __name__ == "__main__":
    print("=== Count-Min Sketch (Cormode-Muthukrishnan 2005) ===\n")
    rng = np.random.default_rng(0)

    # Stream: 100k tokens with skewed frequency
    n_unique = 500
    weights = np.arange(1, n_unique + 1) ** -1.5                 # Zipf-like
    weights = weights / weights.sum()
    stream = rng.choice(n_unique, size=100_000, p=weights)

    true_counts = np.bincount(stream, minlength=n_unique)
    for (w, k) in [(64, 3), (256, 5), (1024, 5), (4096, 7)]:
        cms = CountMinSketch(w=w, k=k, seed=0)
        for x in stream:
            cms.add(int(x))
        estimates = np.array([cms.estimate(int(i)) for i in range(n_unique)])
        # CMS is upper-biased: estimate >= truth
        errors = estimates - true_counts
        rel_err = float(np.mean(errors / (true_counts + 1)))
        print(f"  w = {w:>4}, k = {k}   memory = {w * k * 8:>7,} bytes   "
              f"mean over-estimate = {rel_err * 100:>5.1f}%   max = {int(errors.max()):>4}")

    print(f"\n  Full exact count: {len(true_counts) * 8:,} bytes for {n_unique} unique tokens")

    print("\n  Count-Min Sketch trades a small over-estimation for O(1) memory")
    print("  regardless of unique-token count; standard heavy-hitters / top-K primitive.")

    print("\n--- library cross-check (datasketch.CountMinSketch; probables Python) ---")
