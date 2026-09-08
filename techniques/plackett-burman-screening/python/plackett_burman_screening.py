"""Plackett-Burman screening design (Reference Sec 47.63).

Plackett & Burman 1946 'The design of optimum multifactorial
experiments', Biometrika 33(4). Two-level fractional factorial for
SCREENING main effects of many factors with few runs. Design
matrix D of size N x k with N a multiple of 4, entries in {-1, +1},
orthogonal columns:  D^T D = N * I.

Main-effect estimates:  beta_hat_j = (1/N) * sum_i D_{ij} * y_i.

Trades resolution for run count: main effects are aliased with
2-factor interactions -- best used as a FIRST-PASS to prune large
factor lists before running a higher-resolution design.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def hadamard(n):
    """Sylvester-Hadamard for n a power of 2."""
    if n == 1:
        return np.array([[1]])
    H = hadamard(n // 2)
    return np.block([[H, H], [H, -H]])


def plackett_burman_design(n_factors):
    """Return an N x n_factors PB design; N is next multiple-of-4 >= n_factors + 1."""
    N = 4 * int(np.ceil((n_factors + 1) / 4))
    # Use n up to 12 factors via N=12 known first-row (Plackett-Burman 1946):
    if N == 12:
        first_row = np.array([1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1])
        D = np.zeros((12, 11), dtype=int)
        D[0] = first_row
        for i in range(1, 11):
            D[i] = np.roll(D[i - 1], -1)
        D[11] = -1
        return D[:, :n_factors].astype(float)
    # Fallback: Hadamard with column 0 dropped (works for N = 2^m)
    if (N & (N - 1)) != 0:
        raise ValueError(f"No canonical PB design coded for N={N}; use N=12 or power of 2")
    H = hadamard(N)
    return H[:, 1:1 + n_factors].astype(float)


if __name__ == "__main__":
    print("=== Plackett-Burman screening design (1946) ===\n")
    k = 8
    D = plackett_burman_design(k)
    N = D.shape[0]
    print(f"  {k} factors -> N = {N} runs (design matrix size {D.shape})")
    print(f"  Orthogonality (D^T D):")
    print(np.round(D.T @ D, 2))

    # Fake experiment: only factors 0, 2, 5 are active
    rng = np.random.default_rng(0)
    true_beta = np.array([0.0, 3.0, 0.0, -2.0, 0.0, 0.0, 1.5, 0.0])
    y = 10 + D @ true_beta + 0.5 * rng.normal(size=N)

    # Main-effect estimates via ordinary least squares
    Xf = np.column_stack([np.ones(N), D])
    beta_hat, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    print(f"\n  True effects:   {true_beta}")
    print(f"  Estim effects: {np.round(beta_hat[1:], 3)}")
    ranked = np.argsort(np.abs(beta_hat[1:]))[::-1]
    print(f"  |effect| ranking: {list(ranked)}   (truly active: [1, 3, 6])")

    print("\n--- library cross-check (FrF2 / DoE.base R; pyDOE Python) ---")
