"""Conditional mutual information + CMI-based independence test
(Reference Sec 34.13).

  I(X; Y | Z)  =  H(X | Z) + H(Y | Z) - H(X, Y | Z)
              =  H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z).

Conditional independence X _|_ Y | Z  <=>  I(X; Y | Z) = 0.

CMI-based CONDITIONAL INDEPENDENCE TEST via permutation:
  * Permute Y within strata of Z (or use LOCAL permutations).
  * Empirical p-value = fraction of permuted CMI >= observed.

Here we compute discrete CMI + permutation p-values on:
  (a) X and Y independent given Z (X drives Y only through Z).
  (b) X and Y directly related, conditioning on Z insufficient.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _entropy(*cols):
    n = len(cols[0])
    stacked = np.stack(cols, axis=1)
    _, counts = np.unique(stacked, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def conditional_mi(x, y, z):
    """I(X; Y | Z) via entropy decomposition."""
    return (_entropy(x, z) + _entropy(y, z)
            - _entropy(x, y, z) - _entropy(z))


def cmi_permutation_test(x, y, z, B=300, seed=0):
    """Permute Y within Z-strata and compare to observed CMI."""
    rng = np.random.default_rng(seed)
    obs = conditional_mi(x, y, z)
    n = len(x)
    hits = 0
    y_perm = y.copy()
    for _ in range(B):
        for zv in np.unique(z):
            mask = np.where(z == zv)[0]
            y_perm[mask] = rng.permutation(y[mask])
        val = conditional_mi(x, y_perm, z)
        if val >= obs: hits += 1
    return obs, hits / B


if __name__ == "__main__":
    print("=== Conditional mutual information + CI test ===\n")
    rng = np.random.default_rng(0)
    n = 3000

    # (a) X ↔ Y only through Z: Z generates X and Y independently
    z = rng.integers(0, 3, n)
    xa = (z + rng.integers(0, 2, n)) % 4
    ya = (z + rng.integers(0, 2, n)) % 4
    cmi_a, p_a = cmi_permutation_test(xa, ya, z, B=200)
    print(f"  (a) X ⊥⊥ Y | Z:")
    print(f"    I(X; Y | Z) = {cmi_a:.4f}   permutation p = {p_a:.3f}")
    print(f"    unconditional I(X; Y) = {conditional_mi(xa, ya, np.zeros(n, dtype=int)):.4f}"
          f"   (may be > 0 due to Z-mediation)\n")

    # (b) X directly influences Y, Z is a red herring
    xb = rng.integers(0, 3, n)
    yb = (xb + rng.integers(0, 2, n)) % 3
    zb = rng.integers(0, 3, n)
    cmi_b, p_b = cmi_permutation_test(xb, yb, zb, B=200)
    print(f"  (b) X → Y direct, Z independent:")
    print(f"    I(X; Y | Z) = {cmi_b:.4f}   permutation p = {p_b:.3f}   (should be small p)\n")

    print("--- library cross-check (Python NPEET; R condMI; causaldiscovery::ci_test) ---")
