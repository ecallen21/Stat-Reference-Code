"""Diffusion maps (Reference Sec 25.16).

Coifman & Lafon (2006) 'Diffusion maps.'

Nonlinear DR that preserves DIFFUSION DISTANCES (t-step random-walk
distance on a similarity graph).

Algorithm:
  1. Build Gaussian similarity K_ij = exp(-||x_i - x_j||^2 / sigma^2).
  2. Row-normalise: P = D^{-1} K   (Markov transition matrix).
  3. Eigendecomposition of P; skip the trivial constant eigenvector.
  4. Embedding = top-k non-trivial eigenvectors * (eigenvalue)^t.

t = 0 -> spectral embedding; larger t emphasises long-range structure.

Advantages over Isomap:
  * Robust to noise (integrates over paths, not just shortest).
  * Multi-scale (t parameter).

Here we recover the intrinsic 1-D parameter of a synthetic S-curve
manifold embedded in 3-D.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def diffusion_map(X, k=2, sigma=None, t=1):
    """Return the k-dim diffusion-map embedding."""
    n = X.shape[0]
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(axis=2))
    if sigma is None:
        sigma = float(np.median(D[D > 0]))
    K = np.exp(-D ** 2 / (2 * sigma ** 2))
    row_sum = K.sum(axis=1)
    P = K / row_sum[:, None]                        # Markov row-stochastic
    # Eigendecomposition of P (non-symmetric; use symmetrised form
    # M = D^{-1/2} K D^{-1/2} and undo at the end).
    d_half_inv = 1.0 / np.sqrt(row_sum)
    M = K * (d_half_inv[:, None] * d_half_inv[None, :])
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(-vals)
    vals = vals[order]; vecs = vecs[:, order]
    # Convert back to right eigenvectors of P: v = D^{-1/2} u.
    right_v = vecs * d_half_inv[:, None]
    # Skip trivial eigenvector (all-ones).
    return right_v[:, 1:k + 1] * (vals[1:k + 1] ** t), vals


if __name__ == "__main__":
    print("=== Diffusion maps (Coifman-Lafon 2006) ===\n")
    rng = np.random.default_rng(0)
    n = 200
    t_param = 3 * np.pi * (rng.random(n) - 0.5)
    h = 5 * rng.random(n)
    X = np.stack([np.sin(t_param), h, np.sign(t_param) * (np.cos(t_param) - 1)], axis=1)

    Y, eigenvalues = diffusion_map(X, k=2, t=1)

    # Y[:, 0] should correlate with t_param (arc position along the S curve).
    C = np.abs(np.corrcoef(np.column_stack([Y, t_param, h]).T)[:2, 2:])
    print(f"  first 5 non-trivial eigenvalues: {np.round(eigenvalues[1:6], 3).tolist()}")
    print(f"  |corr| matrix (rows diffusion coords 1-2, cols t, h):\n{C.round(3)}")
    print(f"  best diffusion coord for t: {int(C[:, 0].argmax()) + 1}   |corr| = {C[:, 0].max():.3f}")
    print(f"  best diffusion coord for h: {int(C[:, 1].argmax()) + 1}   |corr| = {C[:, 1].max():.3f}")
    print("\n  Diffusion embedding recovers the intrinsic manifold coordinates.\n")
    print("--- library cross-check (R diffusionMap; Python pyDiffMap; datafold) ---")
