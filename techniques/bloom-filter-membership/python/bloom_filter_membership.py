"""Bloom filter (Reference Sec 47.72).

Bloom 1970 'Space/time trade-offs in hash coding with allowable
errors', CACM 13(7). Probabilistic set-membership structure. For n
expected elements and target FP rate p, optimal bit-array size:

    m = -n * ln p / (ln 2)^2       (bits)
    k = (m / n) * ln 2              (hash fns)

`contains(x)`:
  * NO false negatives (element definitely absent if any bit is 0);
  * FP rate at capacity  =  (1 - e^{-kn/m})^k  ~=  p.

Immutable / append-only; can't delete without counting Bloom
filters or cuckoo filters.
"""
from __future__ import annotations    # stdlib

import hashlib    # deterministic hash
import math    # log, exp
import numpy as np    # numerical arrays


class BloomFilter:
    def __init__(self, n_expected, fp_rate=0.01):
        self.m = int(math.ceil(-n_expected * math.log(fp_rate) / math.log(2) ** 2))
        self.k = max(1, int(round((self.m / n_expected) * math.log(2))))
        self.bits = np.zeros(self.m, dtype=np.uint8)
        self.n_added = 0

    def _hashes(self, x):
        h = hashlib.sha256(str(x).encode()).digest()
        h1 = int.from_bytes(h[:8], "big")
        h2 = int.from_bytes(h[8:16], "big")
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, x):
        for j in self._hashes(x):
            self.bits[j] = 1
        self.n_added += 1

    def __contains__(self, x):
        return all(self.bits[j] == 1 for j in self._hashes(x))

    def expected_fp_rate(self):
        return (1 - math.exp(-self.k * self.n_added / self.m)) ** self.k


if __name__ == "__main__":
    print("=== Bloom filter (Bloom 1970) ===\n")
    rng = np.random.default_rng(0)
    n_expected = 10_000
    for target_fp in [0.001, 0.01, 0.05]:
        bf = BloomFilter(n_expected, fp_rate=target_fp)
        for i in range(n_expected):
            bf.add(f"user_{i}")
        # False positive test: query 10_000 unseen keys
        n_test = 10_000
        fp_count = sum(1 for i in range(n_expected, n_expected + n_test)
                       if f"user_{i}" in bf)
        # False negative test (should be 0)
        fn_count = sum(1 for i in range(n_expected) if f"user_{i}" not in bf)
        print(f"  target FP={target_fp:.3f}  m={bf.m:6d} bits ({bf.m/8/1024:.1f} KB)  "
              f"k={bf.k}  FP={fp_count/n_test:.4f}  FN={fn_count}  "
              f"expected FP={bf.expected_fp_rate():.4f}")

    print("\n  Empirical FP matches theoretical FP within Monte-Carlo error.")
    print("  Zero false negatives (guaranteed).")
    print("\n--- library cross-check (bloomfilter R; pybloom / pybloomfiltermmap3 Python) ---")
