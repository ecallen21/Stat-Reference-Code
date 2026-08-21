"""Fractional factorial designs (Reference §16.4).

Full 2^k factorial explodes with k: 2^7 = 128 runs.  Fractional factorials
2^(k-p) run only 1/2^p of the full grid by ALIASING higher-order interactions
with main effects and lower-order interactions.

Design generation
    1. Start with a full 2^(k-p) base design in the first (k - p) factors.
    2. Assign additional factors by column products (generators):
        e.g. k = 5, p = 1:  E = ABCD (a 2^(5-1) = 16-run resolution V design)

Resolution
    - Resolution III: main effects aliased with two-factor interactions.  Screening only.
    - Resolution IV:  main effects clear of 2fi's; 2fi's aliased with 2fi's.  Common.
    - Resolution V+:  main effects and 2fi's clear.  Preferred when 2fi's matter.

Alias structure: multiply each factor / interaction by the defining relation
I = ABCDE (Res V generator for the above).  Any effect equal to I is confounded
with the grand mean.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from itertools import product    # stdlib: cartesian product for factorial designs

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def full_factorial_2k(k: int) -> np.ndarray:
    """2^k full factorial in +/-1 coding."""
    return np.array(list(product([-1, 1], repeat=k)), dtype=int)


def fractional_factorial(k: int, generators: dict) -> tuple:
    """Fractional 2^(k-p) design.

    generators: mapping of new-factor-name -> product of existing base factors.
                e.g. {'E': ('A', 'B', 'C', 'D')} for a 2^(5-1) with E = ABCD.
    Returns (design_matrix (n_runs x k), factor_names)
    """
    base_k = k - len(generators)
    base_names = [chr(ord('A') + i) for i in range(base_k)]
    D = full_factorial_2k(base_k)
    all_cols = {base_names[i]: D[:, i] for i in range(base_k)}
    for new, factors in generators.items():
        col = np.ones(D.shape[0], dtype=int)
        for f in factors:
            col *= all_cols[f]
        all_cols[new] = col
    names = base_names + list(generators.keys())
    return np.column_stack([all_cols[n] for n in names]), names


def alias_structure(k: int, generators: dict, max_order: int = 2) -> list:
    """Enumerate the alias structure by multiplying each effect by the defining relation."""
    D, names = fractional_factorial(k, generators)
    # Defining relation: for each generator like E = ABCD -> word = "ABCDE" is identity.
    words = []
    for new, factors in generators.items():
        words.append("".join(factors) + new)
    # For each single effect (main + 2fi), compute product with each defining word
    from itertools import combinations
    effects = names + ["".join(sorted(pair)) for pair in combinations(names, 2)]
    aliases = []
    for e in effects:
        alias_list = [e]
        for w in words:
            merged = "".join(sorted(set(e) ^ set(w)))
            if merged and merged != e: alias_list.append(merged if merged else "I")
        aliases.append({"effect": e, "aliased_with": alias_list[1:]})
    return aliases


if __name__ == "__main__":
    print("=== 2^(5-1) design with E = ABCD (Resolution V, 16 runs) ===")
    D, names = fractional_factorial(5, {"E": ("A", "B", "C", "D")})
    print(f"  {D.shape[0]} runs (vs 2^5 = 32 for the full factorial)")
    print(f"  factors: {names}")
    print("  first 4 rows:")
    print(D[:4])

    print("\n=== Alias structure (main effects) ===")
    from itertools import combinations
    for a in alias_structure(5, {"E": ("A", "B", "C", "D")}):
        if len(a["effect"]) <= 2:
            print(f"  {a['effect']} = " + " = ".join(a["aliased_with"]))

    print("\n=== 2^(7-2) design (Resolution IV, 32 runs) ===")
    D, names = fractional_factorial(7, {"F": ("A", "B", "C"), "G": ("B", "C", "D")})
    print(f"  {D.shape[0]} runs (vs 2^7 = 128 full); factors = {names}")

    print("\n--- library cross-check (pyDOE2 fracfact / R FrF2) ---")
    print("  R: FrF2::FrF2(nruns = 16, nfactors = 5, generators = 'ABCD')")
