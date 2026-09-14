"""Sobol Variance-Based Sensitivity (Ref Sec 47.334).

Sobol 1993; Saltelli et al 2008. Decompose the variance of a
scalar output f(X_1, ..., X_d) into contributions from each
input and their interactions:

    Var(Y) = sum_i V_i + sum_{i<j} V_{ij} + ... + V_{1..d}
    S_i     = V_i / Var(Y)                       (first order)
    S_Ti    = 1 - V_{~i} / Var(Y)                 (total effect)

S_i measures the main effect; S_Ti − S_i captures interaction
involvement. Standard sensitivity index in engineering /
climate / epi models.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sobol_indices(f, d, N=4096, rng=None):
    """Saltelli-style Sobol estimation. Returns (S_i, S_Ti) arrays."""
    if rng is None: rng = np.random.default_rng(0)
    # Two independent sample matrices A, B of size (N, d)
    A = rng.uniform(size=(N, d))
    B = rng.uniform(size=(N, d))
    fA = np.array([f(x) for x in A]); fB = np.array([f(x) for x in B])
    var_total = np.var(np.concatenate([fA, fB]), ddof=1)
    S = np.zeros(d); ST = np.zeros(d)
    for i in range(d):
        AB_i = A.copy(); AB_i[:, i] = B[:, i]                       # replace col i
        fABi = np.array([f(x) for x in AB_i])
        # Saltelli 2010 estimators
        S[i] = np.mean(fB * (fABi - fA)) / var_total
        ST[i] = 0.5 * np.mean((fA - fABi) ** 2) / var_total
    return S, ST


if __name__ == "__main__":
    print("=== Sobol Variance-Based Sensitivity (Sobol 1993; Saltelli 2008) ===\n")
    rng = np.random.default_rng(0)

    # Ishigami function (classic test): 3 inputs, uniform on [0, 1] here
    def ishigami(x):
        u = 2 * np.pi * x - np.pi                                   # rescale to [-pi, pi]
        return float(np.sin(u[0]) + 7 * np.sin(u[1]) ** 2 + 0.1 * u[2] ** 4 * np.sin(u[0]))

    S, ST = sobol_indices(ishigami, d=3, N=4096, rng=rng)
    print(f"  Ishigami function on [-pi, pi]^3 (via [0,1]^3 rescale):")
    print(f"    First-order S_i:   {S.round(3).tolist()}")
    print(f"    Total-effect S_Ti: {ST.round(3).tolist()}")
    print(f"\n  Known values (analytic):")
    print(f"    S_i    ~ (0.31, 0.44, 0.00)")
    print(f"    S_Ti   ~ (0.56, 0.44, 0.24)")

    print(f"\n  Interpretation:")
    print(f"    X_2 has largest first-order effect (sin^2 term)")
    print(f"    X_1 gains substantially from interactions (S_T1 >> S_1)")
    print(f"    X_3 has zero main effect but interacts through X_1 * X_3^4")

    print("\n--- library cross-check (SALib.analyze.sobol; sensitivity R; UQpy) ---")
