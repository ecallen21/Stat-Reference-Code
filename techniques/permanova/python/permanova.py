"""PERMANOVA - Permutational MANOVA (Reference §9.17).

Non-parametric MANOVA on a DISTANCE MATRIX (Anderson 2001). Does the
between-group distance exceed what would be expected under a random
label assignment?

Statistic (pseudo-F):
    SS_total    = sum_{i<j} d_ij^2 / n
    SS_within   = sum over groups g of  sum_{i<j in g} d_ij^2 / n_g
    SS_between  = SS_total - SS_within
    F = (SS_between / (K - 1)) / (SS_within / (n - K))

p-value from PERMUTING group labels many times and computing F on each shuffle.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _pseudo_F(D2, groups):
    n = len(groups)
    K = len(np.unique(groups))
    SS_total = D2.sum() / (2 * n)
    SS_within = 0.0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        n_g = len(idx)
        if n_g > 1:
            SS_within += D2[np.ix_(idx, idx)].sum() / (2 * n_g)
    SS_between = SS_total - SS_within
    if SS_within <= 0 or n - K <= 0: return float("inf")
    return (SS_between / (K - 1)) / (SS_within / (n - K))


def permanova(D, groups, n_perm: int = 999, seed: int = 0) -> dict:
    """Run PERMANOVA on precomputed distance matrix D and grouping vector.

    D : n x n symmetric distance matrix.
    groups : length-n group labels.
    """
    D = np.asarray(D, dtype=float)
    D2 = D * D
    groups = np.asarray(groups)
    n = len(groups); K = len(np.unique(groups))
    F_obs = _pseudo_F(D2, groups)
    rng = np.random.default_rng(seed)
    F_perm = np.empty(n_perm)
    for b in range(n_perm):
        F_perm[b] = _pseudo_F(D2, rng.permutation(groups))
    p = float((1 + np.sum(F_perm >= F_obs)) / (1 + n_perm))
    return {"pseudo_F": float(F_obs),
            "df_between": K - 1, "df_within": n - K,
            "p_value_perm": p,
            "n_perm": n_perm, "K_groups": K, "n": int(n),
            "method": "PERMANOVA (Anderson 2001)"}


def library_versions(D, groups):
    try:
        from skbio.stats.distance import permanova as sk_permanova, DistanceMatrix
        dm = DistanceMatrix(D)
        r = sk_permanova(dm, list(groups))
        return {"scikit-bio permanova": r.to_dict()}
    except Exception as ex:
        return {"scikit-bio (optional)": f"not available: {ex}"}


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    # Two groups with different multivariate centers
    n_per = 30
    X_a = rng.normal(0, 1, size=(n_per, 4))
    X_b = rng.normal(0.7, 1, size=(n_per, 4))
    X = np.vstack([X_a, X_b])
    groups = np.array(["A"] * n_per + ["B"] * n_per)
    # Euclidean distance matrix
    diffs = X[:, None, :] - X[None, :, :]
    D = np.sqrt((diffs ** 2).sum(-1))

    print("=== PERMANOVA on Euclidean distance (2 groups, shift = 0.7) ===")
    r = permanova(D, groups)
    print(f"  pseudo-F = {r['pseudo_F']:.4f}   df = ({r['df_between']}, {r['df_within']})")
    print(f"  p (999 perms) = {r['p_value_perm']:.4f}")

    print("\n--- library ---")
    for k, v in library_versions(D, groups).items():
        print(f"  {k}: {v}")
