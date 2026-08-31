"""Functional PCA (Reference Sec 31.2).

Ramsay & Silverman (2005) 'Functional Data Analysis.' Ch 8.

Each observation is a CURVE x_i(t) over a common domain. FPCA finds
orthonormal basis functions phi_k(t) and per-curve SCORES xi_ik such that

  x_i(t) ~= mu(t) + sum_k xi_ik * phi_k(t).

Two flavours:
  * Discrete FPCA: evaluate curves on a common grid, run PCA on the
    matrix of function values (per-column normalisation optional).
  * Basis-expanded FPCA: represent each curve in a spline basis, run
    PCA on the coefficient vectors.

Here we implement the discrete variant and recover principal functions
on a synthetic dataset of shifted sinusoids.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def fpca(X, k=3):
    """X: (n, T) curve values on a common grid. Return (mean, phi, scores)."""
    mean = X.mean(axis=0)
    Xc = X - mean
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    # phi = right singular vectors (functions); scores = U * s.
    phi = Vt[:k]
    scores = Xc @ phi.T
    var_expl = (s ** 2 / (s ** 2).sum())[:k]
    return mean, phi, scores, var_expl


if __name__ == "__main__":
    print("=== Functional PCA on shifted sinusoids ===\n")
    rng = np.random.default_rng(0)
    n = 50; T = 100
    t = np.linspace(0, 2 * np.pi, T)
    # Two hidden factors: amplitude and phase shift.
    amp = rng.normal(1.0, 0.3, n)
    shift = rng.normal(0.0, 0.4, n)
    X = np.array([a * np.sin(t + s) for a, s in zip(amp, shift)]) + 0.05 * rng.normal(0, 1, (n, T))

    mean_hat, phi, scores, ve = fpca(X, k=3)

    print(f"  variance explained per PC: {np.round(ve, 3).tolist()}   sum first 2: {ve[:2].sum():.3f}")
    print(f"\n  Correlation between scores and hidden factors:")
    print(f"    corr(PC1 scores, amplitude)      = {abs(np.corrcoef(scores[:, 0], amp)[0,1]):.3f}")
    print(f"    corr(PC1 scores, shift)          = {abs(np.corrcoef(scores[:, 0], shift)[0,1]):.3f}")
    print(f"    corr(PC2 scores, amplitude)      = {abs(np.corrcoef(scores[:, 1], amp)[0,1]):.3f}")
    print(f"    corr(PC2 scores, shift)          = {abs(np.corrcoef(scores[:, 1], shift)[0,1]):.3f}")
    print("\n  First two functional PCs align with amplitude + phase-shift variability.\n")
    print("--- library cross-check (R fda::pca.fd; scikit-fda FPCA; Python fdasrsf) ---")
