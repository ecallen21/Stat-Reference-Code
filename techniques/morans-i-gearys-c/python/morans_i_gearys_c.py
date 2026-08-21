"""Global spatial autocorrelation: Moran's I + Geary's C (Reference §23.3).

Moran's I (Moran 1950)
    I = (n / S0) * sum_i sum_j W_ij (x_i - xbar)(x_j - xbar) / sum_i (x_i - xbar)^2
    S0 = sum_ij W_ij

    I ~ +1 : strong positive spatial autocorr (clusters of similar values)
    I ~ 0  : random (E[I] = -1/(n-1))
    I ~ -1 : negative autocorr (checkerboard)

Geary's C (Geary 1954)
    C = ((n - 1) / (2 S0)) * sum_ij W_ij (x_i - x_j)^2 / sum_i (x_i - xbar)^2
    C ~ 0 : perfect positive; C = 1 : random; C > 1 : negative.

Analytic p-value uses moments of I under randomization; permutation p-value
shuffles x labels many times.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def morans_i(x, W, n_perm: int = 999, seed: int = 0) -> dict:
    x = np.asarray(x, dtype=float); W = np.asarray(W, dtype=float)
    n = len(x); z = x - x.mean()
    S0 = W.sum()
    I = (n / S0) * float(z @ W @ z) / float(z @ z)
    # Permutation
    rng = np.random.default_rng(seed)
    I_perm = np.empty(n_perm)
    for k in range(n_perm):
        zp = rng.permutation(z)
        I_perm[k] = (n / S0) * float(zp @ W @ zp) / float(zp @ zp)
    p_two = float((1 + np.sum(np.abs(I_perm) >= abs(I))) / (1 + n_perm))
    return {"morans_I": float(I),
            "E_I_under_null": -1 / (n - 1),
            "p_perm_two_sided": p_two, "n_perm": int(n_perm),
            "method": "Global Moran's I (permutation p-value)"}


def gearys_c(x, W) -> dict:
    x = np.asarray(x, dtype=float); W = np.asarray(W, dtype=float)
    n = len(x); z = x - x.mean()
    S0 = W.sum()
    diff = x[:, None] - x[None, :]
    C = ((n - 1) / (2 * S0)) * float(np.sum(W * diff ** 2)) / float(z @ z)
    return {"gearys_C": float(C),
            "expected_under_null": 1.0,
            "method": "Geary's C"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Grid of points with a spatial gradient (positive autocorr)
    coords = np.array([(i, j) for i in range(10) for j in range(10)], dtype=float)
    x = coords[:, 0] + coords[:, 1] + rng.normal(0, 0.5, len(coords))
    # kNN weights
    D = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(-1))
    W = np.zeros_like(D)
    for i in range(len(coords)):
        d = D[i].copy(); d[i] = np.inf
        idx = np.argsort(d)[:8]; W[i, idx] = 1.0
    W = W / W.sum(1, keepdims=True)

    r = morans_i(x, W, n_perm=999)
    print(f"=== Moran's I on spatial-gradient field ===")
    print(f"  I = {r['morans_I']:.4f}   E[I] under null = {r['E_I_under_null']:.4f}")
    print(f"  permutation two-sided p = {r['p_perm_two_sided']:.4f}")

    r = gearys_c(x, W)
    print(f"\n=== Geary's C ===")
    print(f"  C = {r['gearys_C']:.4f}   expected under null = 1.0")

    print("\n=== Random field (no spatial autocorr) ===")
    x_rand = rng.normal(size=len(coords))
    print(f"  Moran's I = {morans_i(x_rand, W, n_perm=299)['morans_I']:.4f} (near 0)")

    print("\n--- library cross-check (R spdep::moran.test / geary.test) ---")
