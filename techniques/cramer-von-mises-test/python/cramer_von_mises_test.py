"""Cramer-von Mises test (Reference Sec 47.102).

Cramer 1928; von Mises 1931. One-sample EDF-based GoF:

    W^2 = n * integral (F_n(x) - F_0(x))^2 dF_0(x)
        = 1/(12 n) + sum_i [ U_(i) - (2i - 1) / (2 n) ]^2

Weights all deviations equally (unlike A-D which upweights tails).
Two-sample variant (Anderson 1962) exists as well.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats    # cdf functions + tabulated cvm cvals


def cvm_one_sample(x, dist="norm", args=(), estimate=True):
    x = np.sort(np.asarray(x))
    n = len(x)
    d = getattr(stats, dist)
    if estimate:
        args = d.fit(x)
    U = d.cdf(x, *args)
    U = np.clip(U, 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    W2 = 1.0 / (12 * n) + float(np.sum((U - (2 * i - 1) / (2 * n)) ** 2))
    # p-value via scipy's built-in one-sample CVM test (asymptotic)
    res = stats.cramervonmises(x, lambda z: d.cdf(z, *args))
    return {"W2": float(W2), "p": float(res.pvalue)}


def cvm_two_sample(x, y):
    n, m = len(x), len(y)
    Z = np.concatenate([x, y])
    R = np.argsort(np.argsort(Z)) + 1
    Rx = R[:n]; Ry = R[n:]
    # Anderson 1962 / Baumgartner-Weiss-Schindler analogue
    T = (n * np.sum((Rx - np.arange(1, n + 1)) ** 2)
          + m * np.sum((Ry - np.arange(1, m + 1)) ** 2)) / (n * m * (n + m)) - (4 * n * m - 1) / (6 * (n + m))
    return {"T": float(T)}


if __name__ == "__main__":
    print("=== Cramer-von Mises tests (Cramer 1928; von Mises 1931) ===\n")
    rng = np.random.default_rng(0)

    print("  One-sample vs Normal (params estimated):")
    scenarios = [
        ("N(0, 1)         ", rng.normal(size=500)),
        ("t3 (heavy tail) ", rng.standard_t(3, size=500)),
        ("Uniform         ", rng.uniform(-1, 1, size=500)),
        ("Laplace         ", rng.laplace(size=500)),
    ]
    for name, x in scenarios:
        r = cvm_one_sample(x, dist="norm")
        print(f"    {name}  W^2 = {r['W2']:.4f}   p = {r['p']}")

    print("\n  Two-sample W^2:")
    A = rng.normal(size=200); B = rng.normal(loc=0.3, size=200)
    r = cvm_two_sample(A, B)
    print(f"    N(0, 1) vs N(0.3, 1):   T = {r['T']:.3f} (larger = more different)")
    A2 = rng.normal(size=200); B2 = rng.normal(size=200)
    r = cvm_two_sample(A2, B2)
    print(f"    N(0, 1) vs N(0, 1):     T = {r['T']:.3f}")

    print("\n--- library cross-check (goftest R; scipy.stats.cramervonmises Python) ---")
