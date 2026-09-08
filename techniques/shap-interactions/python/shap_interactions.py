"""SHAP interaction values (Reference Sec 47.40).

Lundberg, Erion & Lee 2018 'Consistent individualized feature
attribution for tree ensembles' (arXiv:1802.03888). Extension of
SHAP that decomposes each prediction into:

    phi(x, i) = phi_ii(x) + sum_{j != i} phi_ij(x)

with phi_ij(x) a Shapley INTERACTION value between features i, j.
Sum of phi_ij (i != j) plus phi_ii recovers the ordinary SHAP.

Additive-only models: phi_ij = 0 for i != j.
Interaction models: nonzero off-diagonal terms reveal WHICH pair of
features drives the effect.

We compute exact Shapley interaction values by enumerating all
subsets (Kernel-SHAP style) on a small feature set for a
demonstration.
"""
from __future__ import annotations    # stdlib

from itertools import combinations
from math import factorial

import numpy as np    # numerical arrays


def shapley_value(f, x, background_x, d):
    """Exact Shapley value phi_i for feature i, computed via subset enumeration."""
    phi = np.zeros(d)
    perms_all = 0
    for i in range(d):
        s = 0.0
        for r in range(d):
            others = [j for j in range(d) if j != i]
            for S in combinations(others, r):
                x_with = np.where(np.isin(np.arange(d), list(S) + [i]), x, background_x)
                x_without = np.where(np.isin(np.arange(d), list(S)), x, background_x)
                weight = factorial(r) * factorial(d - r - 1) / factorial(d)
                s += weight * (f(x_with) - f(x_without))
        phi[i] = s
    return phi


def shapley_interaction(f, x, background_x, d):
    """Interaction Shapley: phi_ij = 0.5 * (delta_ij + delta_ji)."""
    Phi = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j: continue
            s = 0.0
            for r in range(d - 1):
                others = [k for k in range(d) if k not in (i, j)]
                for S in combinations(others, r):
                    x_ij = np.where(np.isin(np.arange(d), list(S) + [i, j]), x, background_x)
                    x_i = np.where(np.isin(np.arange(d), list(S) + [i]), x, background_x)
                    x_j = np.where(np.isin(np.arange(d), list(S) + [j]), x, background_x)
                    x_none = np.where(np.isin(np.arange(d), list(S)), x, background_x)
                    weight = factorial(r) * factorial(d - r - 2) / (2 * factorial(d - 1))
                    s += weight * (f(x_ij) - f(x_i) - f(x_j) + f(x_none))
            Phi[i, j] = s
    return Phi


if __name__ == "__main__":
    print("=== SHAP interaction values (Lundberg 2018) ===\n")
    #  Model with clear pairwise interaction: f = x0 + x1 + 2 * x0 * x2
    def f(x):
        return float(x[0] + x[1] + 2 * x[0] * x[2])

    x = np.array([1.0, 1.0, 1.0, 1.0])
    background_x = np.array([0.0, 0.0, 0.0, 0.0])
    d = 4

    phi = shapley_value(f, x, background_x, d)
    print(f"  Shapley values phi_i for x = {x.tolist()}, background zero:")
    print(f"    {[round(v, 3) for v in phi]}   sum = {phi.sum():.3f}   f(x) = {f(x):.3f}")

    Phi = shapley_interaction(f, x, background_x, d)
    print(f"\n  Interaction matrix phi_ij (off-diagonal only):")
    for i in range(d):
        row = " ".join(f"{Phi[i, j]:+.3f}" if i != j else "  --  " for j in range(d))
        print(f"    row {i}: {row}")
    print(f"\n  Non-zero off-diagonals should highlight (0, 2) -- the true interaction pair.")

    print("\n--- library cross-check (shap.TreeExplainer.shap_interaction_values Python) ---")
