"""Split-half reliability + Spearman-Brown correction (Reference §22.4).

Split-half method
    Split a test into two halves; correlate the two half-scores -> r_hh.
    Reliability of a HALF is r_hh; but you want reliability of the FULL
    test, which is longer.

Spearman-Brown prophecy formula
    Reliability of a test of length k times the original:
        rho_k = k * rho_1 / (1 + (k - 1) * rho_1)
    Split-half specific case (k = 2):
        rho_full = 2 r_hh / (1 + r_hh)

Split methods
    - Odd-even (interleave items).
    - First-vs-second-half.
    - Random.
    - Guttman lower bound (min correlation over splits).

Alternatives without splitting: Cronbach's alpha (average over all splits).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def spearman_brown(r: float, k: float = 2) -> float:
    """Reliability if you lengthen the test by factor k, given original r."""
    return k * r / (1 + (k - 1) * r)


def split_half(X, method: str = "odd_even", seed: int = 0) -> dict:
    """Split-half reliability with Spearman-Brown correction."""
    X = np.asarray(X, dtype=float); n, K = X.shape
    rng = np.random.default_rng(seed)
    if method == "odd_even":
        A_idx = np.arange(0, K, 2); B_idx = np.arange(1, K, 2)
    elif method == "first_second":
        A_idx = np.arange(0, K // 2); B_idx = np.arange(K // 2, K)
    elif method == "random":
        perm = rng.permutation(K); A_idx = perm[:K // 2]; B_idx = perm[K // 2:]
    else:
        raise ValueError("method must be 'odd_even', 'first_second', 'random'")
    sA = X[:, A_idx].sum(axis=1); sB = X[:, B_idx].sum(axis=1)
    r_hh = float(np.corrcoef(sA, sB)[0, 1])
    r_full = spearman_brown(r_hh, k=2)
    return {"split_method": method,
            "r_halves": r_hh,
            "spearman_brown_reliability": r_full,
            "n_items_A": int(len(A_idx)), "n_items_B": int(len(B_idx)),
            "method": "Split-half reliability + Spearman-Brown correction"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, K = 300, 10
    theta = rng.normal(0, 1, n)
    lam = np.full(K, 0.7)
    X = theta[:, None] * lam[None, :] + rng.normal(0, 0.5, (n, K))

    for method in ("odd_even", "first_second", "random"):
        r = split_half(X, method=method, seed=0)
        print(f"=== {method} split ===")
        print(f"  r_halves            = {r['r_halves']:.4f}")
        print(f"  Spearman-Brown full = {r['spearman_brown_reliability']:.4f}")

    print("\n=== Spearman-Brown prophecy (r = 0.7, various k) ===")
    for k in (0.5, 1.0, 2.0, 3.0, 5.0):
        print(f"  test length {k}x -> reliability = {spearman_brown(0.7, k):.4f}")

    print("\n--- library cross-check (R psych::splitHalf) ---")
