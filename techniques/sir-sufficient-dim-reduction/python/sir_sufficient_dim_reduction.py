"""Sliced Inverse Regression (SIR) + SAVE — Reference Sec 25.3.

Li, K.-C. (1991) 'Sliced inverse regression for dimension reduction.'
Cook & Weisberg (1991) 'SAVE.'

SUFFICIENT DIMENSION REDUCTION: find a d x k matrix B such that Y is
INDEPENDENT of X given B'X. The columns of B span the CENTRAL SUBSPACE.

SIR algorithm:
  1. Standardise X so cov(X) = I.
  2. Slice Y into H bins; compute the WITHIN-SLICE MEAN of X per slice.
  3. Form M = sum_h p_h * (mean_h X)(mean_h X)'.
  4. Central-subspace estimate = leading eigenvectors of M, back-
     transformed to original X-scale by cov(X)^{-1/2}.

Advantages:
  * SUPERVISED dim reduction (uses Y).
  * Recovers the direction of dependence, not just variance.
  * Rate: sqrt(n) for the estimated directions.

Limitation: SIR fails when the dependence on Y is symmetric around 0
(e.g. Y = X'B squared); SAVE fixes this by using slice covariances.

Here we implement basic SIR + report the direction cosine against the
truth on synthetic single-index data.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _slice_indices(y, H):
    """Assign each y_i to one of H equal-count slices."""
    order = np.argsort(y)
    n = len(y)
    slice_ids = np.zeros(n, dtype=int)
    edges = np.linspace(0, n, H + 1).astype(int)
    for h in range(H):
        slice_ids[order[edges[h]:edges[h + 1]]] = h
    return slice_ids


def sir(X, y, H=10, k=1):
    """Sliced Inverse Regression: return the top-k central-subspace directions."""
    n, d = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu
    Sigma = Xc.T @ Xc / n + 1e-6 * np.eye(d)
    # Whiten
    L = np.linalg.cholesky(Sigma)
    Z = np.linalg.solve(L, Xc.T).T
    slice_id = _slice_indices(y, H)
    # M = sum_h p_h * mu_h mu_h'
    M = np.zeros((d, d))
    for h in range(H):
        m = slice_id == h
        if not m.any(): continue
        z_bar = Z[m].mean(axis=0)
        M += (m.mean()) * np.outer(z_bar, z_bar)
    # Eigendecomposition
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(-vals)
    B_z = vecs[:, order[:k]]
    # Back-transform to X scale
    B_x = np.linalg.solve(L.T, B_z)
    # Unit-normalise each column
    B_x = B_x / np.linalg.norm(B_x, axis=0, keepdims=True)
    return B_x, vals[order[:k]]


if __name__ == "__main__":
    print("=== Sliced Inverse Regression (Li 1991) ===\n")
    rng = np.random.default_rng(0)
    n = 800
    d = 5
    X = rng.normal(0, 1, (n, d))
    beta_true = np.array([1.0, 2.0, 0.0, -1.0, 0.0])
    beta_true /= np.linalg.norm(beta_true)
    z = X @ beta_true
    y = np.sin(3 * z) + rng.normal(0, 0.2, n)

    B_hat, eigs = sir(X, y, H=10, k=1)
    b_hat = B_hat[:, 0]

    cos_align = abs(float(b_hat @ beta_true))
    print(f"  true direction:      {np.round(beta_true, 3).tolist()}")
    print(f"  SIR-estimated dir:   {np.round(b_hat, 3).tolist()}")
    print(f"  |cos(angle)|:        {cos_align:.4f}   (1.0 = perfect)")
    print(f"  top SIR eigenvalues: {np.round(eigs, 4).tolist()}\n")

    print("--- library cross-check (dr R package; sliced Python; edrGraphicalTools) ---")
