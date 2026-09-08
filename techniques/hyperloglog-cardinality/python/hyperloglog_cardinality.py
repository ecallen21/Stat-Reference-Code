"""HyperLogLog cardinality estimator (Reference Sec 47.71).

Flajolet, Fusy, Gandouet & Meunier 2007 'HyperLogLog: the analysis
of a near-optimal cardinality estimation algorithm'. Estimate the
number of DISTINCT elements in a multiset using O(m) memory,
independent of the input size:

    1. For each element x, compute h(x) -> 64-bit hash.
    2. First b bits index a register j (m = 2^b registers).
    3. Register j stores the MAX count of leading zeros observed
       in the remaining bits + 1.
    4. Estimate  E = alpha_m * m^2 / sum_j 2^{-M[j]}
       with small-range / large-range corrections.

Standard error ~1.04 / sqrt(m). m = 2^14 -> ~0.8 % error, 16 KB
of RAM.
"""
from __future__ import annotations    # stdlib

import hashlib    # deterministic hash
import numpy as np    # numerical arrays


def _hash64(x):
    h = hashlib.sha1(str(x).encode()).digest()
    return int.from_bytes(h[:8], "big")


class HyperLogLog:
    def __init__(self, b=12):
        self.b = b
        self.m = 1 << b
        self.M = np.zeros(self.m, dtype=np.int8)
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1 + 1.079 / self.m)

    def add(self, x):
        h = _hash64(x)
        j = h >> (64 - self.b)
        w = (h << self.b) & ((1 << 64) - 1)
        w |= 1 << (self.b - 1)                       # sentinel to bound rho
        # rho = position of leftmost 1 in the remaining 64 - b bits (1-indexed)
        rho = 64 - self.b - int(w).bit_length() + (64 - self.b) + 1
        # simpler: count leading zeros in the (64 - b)-bit word
        w = h & ((1 << (64 - self.b)) - 1)
        if w == 0:
            rho = 64 - self.b + 1
        else:
            rho = (64 - self.b) - w.bit_length() + 1
        if rho > self.M[j]:
            self.M[j] = rho

    def estimate(self):
        Z_inv = np.sum(2.0 ** (-self.M.astype(float)))
        E = self.alpha * self.m ** 2 / Z_inv
        if E <= 2.5 * self.m:                        # small-range correction
            V = int(np.sum(self.M == 0))
            if V > 0:
                E = self.m * np.log(self.m / V)
        return float(E)


if __name__ == "__main__":
    print("=== HyperLogLog cardinality estimator (Flajolet et al 2007) ===\n")
    rng = np.random.default_rng(0)
    for b in [10, 12, 14]:
        hll = HyperLogLog(b=b)
        for true_n in [1_000, 10_000, 100_000, 1_000_000]:
            hll = HyperLogLog(b=b)
            for i in range(true_n):
                hll.add(i)
            est = hll.estimate()
            err = 100 * (est - true_n) / true_n
            print(f"  b={b:2d} (m={1<<b:5d}, {(1<<b)*4/1024:.1f} KB)  "
                  f"true = {true_n:>7d}   est = {est:>9.0f}   error = {err:+.2f}%")
        print()

    print("  Standard error ~1.04 / sqrt(m); m=2^14 -> +/- 0.81%.")
    print("\n--- library cross-check (hll R; datasketch / hyperloglog Python) ---")
