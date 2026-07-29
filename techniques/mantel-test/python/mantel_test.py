"""Mantel test (Reference §9.18).

Correlation between TWO distance matrices X and Y (same n subjects, different
measures). Tests: does 'closer in X-space' imply 'closer in Y-space'?

    Mantel r = Pearson correlation of the upper-triangle elements of X and Y.

Since the entries within a distance matrix are NOT independent, a direct
p-value from correlation is invalid. Instead, PERMUTE rows/cols of one matrix
(equivalent to relabeling subjects) many times and count how often the
permuted r >= observed r.

Applications:
    - Ecology: is genetic distance correlated with geographic distance?
    - Epidemiology: are two dissimilarity measures on subjects (say, symptom
      profiles and biomarker profiles) related?
    - Consistency: do two different clustering distances agree?

Partial Mantel test (Smouse-Long-Sokal 1986): correlation between X and Y
after PARTIALLING OUT a third matrix Z. Useful when Z is a known confounder.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def mantel_test(X_dist, Y_dist, n_perm: int = 999, seed: int = 0) -> dict:
    """Mantel test on two precomputed distance matrices."""
    X_dist = np.asarray(X_dist, dtype=float)
    Y_dist = np.asarray(Y_dist, dtype=float)
    n = X_dist.shape[0]
    iu = np.triu_indices(n, k=1)
    x_flat = X_dist[iu]; y_flat = Y_dist[iu]
    r_obs = float(np.corrcoef(x_flat, y_flat)[0, 1])
    rng = np.random.default_rng(seed)
    r_perm = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(n)
        Y_p = Y_dist[np.ix_(perm, perm)]
        r_perm[b] = float(np.corrcoef(x_flat, Y_p[iu])[0, 1])
    p_two = float((1 + np.sum(np.abs(r_perm) >= abs(r_obs))) / (1 + n_perm))
    return {"mantel_r": r_obs,
            "p_value_two_sided": p_two,
            "n_perm": n_perm, "n": int(n),
            "method": "Mantel test (permutation on distance matrices)"}


def partial_mantel(X_dist, Y_dist, Z_dist, n_perm: int = 999, seed: int = 0) -> dict:
    """Partial Mantel: correlation between X and Y partialling out Z."""
    n = X_dist.shape[0]; iu = np.triu_indices(n, k=1)
    x = X_dist[iu]; y = Y_dist[iu]; z = Z_dist[iu]
    # Regress x on z and y on z; correlate residuals
    def residuals(a, b):
        A = np.column_stack([np.ones_like(b), b]); coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef
    r_obs = float(np.corrcoef(residuals(x, z), residuals(y, z))[0, 1])
    rng = np.random.default_rng(seed)
    r_perm = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(n)
        y_p = Y_dist[np.ix_(perm, perm)][iu]
        r_perm[b] = float(np.corrcoef(residuals(x, z), residuals(y_p, z))[0, 1])
    p_two = float((1 + np.sum(np.abs(r_perm) >= abs(r_obs))) / (1 + n_perm))
    return {"partial_mantel_r": r_obs,
            "p_value_two_sided": p_two,
            "n_perm": n_perm,
            "method": "Partial Mantel (Smouse-Long-Sokal 1986)"}


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    n = 40
    # Simulate: locations in R^2 -> geographic dist; a related feature -> feature dist
    loc = rng.normal(0, 1, size=(n, 2))
    feat = 0.7 * loc + rng.normal(0, 0.5, size=(n, 2))
    def dist_mat(X):
        d = X[:, None, :] - X[None, :, :]
        return np.sqrt((d ** 2).sum(-1))
    Dg = dist_mat(loc); Df = dist_mat(feat)

    print("=== Mantel test (should reject; distances are correlated) ===")
    r = mantel_test(Dg, Df)
    print(f"  Mantel r = {r['mantel_r']:.4f}, p = {r['p_value_two_sided']:.4f}")

    # Add an unrelated third matrix Z
    z_val = rng.normal(0, 1, size=(n, 2))
    Dz = dist_mat(z_val)
    print("\n=== Partial Mantel (X ~ Y | Z) ===")
    r_p = partial_mantel(Dg, Df, Dz)
    print(f"  partial Mantel r = {r_p['partial_mantel_r']:.4f}, p = {r_p['p_value_two_sided']:.4f}")
