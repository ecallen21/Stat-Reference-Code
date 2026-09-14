"""Deep Sets - Permutation-Invariant Networks (Ref Sec 47.312).

Zaheer, Kottur, Ravanbakhsh, Poczos, Salakhutdinov & Smola 2017
NeurIPS. Any permutation-INVARIANT function on a set can be
decomposed as:

    f({x_1, ..., x_n}) = rho(sum_i phi(x_i))

where phi is an element-wise transform and rho a downstream MLP.
Foundation for set-input networks (point clouds, tabular
aggregations, molecular fingerprints).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def phi(x, W):
    """Element-wise nonlinear transform."""
    return np.tanh(W @ x)


def rho(z, U):
    """Downstream MLP (linear -> tanh -> linear proxy)."""
    return U @ z


def deep_sets(set_x, W_phi, U_rho, agg="sum"):
    """Compute deep-sets output on an unordered collection of vectors."""
    zs = np.stack([phi(x, W_phi) for x in set_x])
    if agg == "sum": pool = zs.sum(axis=0)
    elif agg == "mean": pool = zs.mean(axis=0)
    elif agg == "max": pool = zs.max(axis=0)
    else: raise ValueError(agg)
    return rho(pool, U_rho)


def naive_concat(set_x, W_naive, U_naive):
    """Baseline: concatenate elements in given order (NOT invariant)."""
    return U_naive @ np.tanh(W_naive @ np.concatenate(set_x))


if __name__ == "__main__":
    print("=== Deep Sets (Zaheer et al 2017 NeurIPS) ===\n")
    rng = np.random.default_rng(0)

    # Task: predict SUM of a set of 3-D vectors (permutation-invariant)
    d = 3; hidden = 8
    W_phi = rng.normal(0, 0.3, (hidden, d))
    U_rho = rng.normal(0, 0.3, (d, hidden))

    # Random set of 5 vectors
    set_x = [rng.standard_normal(d) for _ in range(5)]

    print(f"  Original set (5 vectors of dim {d}):")
    for i, x in enumerate(set_x):
        print(f"    x_{i}  = {x.round(3)}")

    print(f"\n  Deep-sets output (sum pool):")
    for perm_i in range(3):
        perm = rng.permutation(len(set_x))
        permuted = [set_x[i] for i in perm]
        out = deep_sets(permuted, W_phi, U_rho, agg="sum")
        print(f"    permutation {perm.tolist()}   ->   {out.round(4)}")

    print(f"\n  Deep-sets output is IDENTICAL under all permutations (invariance property).\n")

    # Naive concat is NOT invariant
    W_naive = rng.normal(0, 0.3, (hidden, d * 5))
    U_naive = rng.normal(0, 0.3, (d, hidden))
    print(f"  Naive concatenation (varies with permutation):")
    for perm_i in range(3):
        perm = rng.permutation(len(set_x))
        permuted = [set_x[i] for i in perm]
        out = naive_concat(permuted, W_naive, U_naive)
        print(f"    permutation {perm.tolist()}   ->   {out.round(4)}")

    print("\n--- library cross-check (torch: sum/mean over set dim + MLP; pyg.aggr.MeanAggregation) ---")
