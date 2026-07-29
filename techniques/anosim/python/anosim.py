"""ANOSIM - Analysis of Similarities (Reference §9.19).

Rank-based non-parametric analog of one-way ANOVA on a DISTANCE MATRIX.
Clarke (1993).  Widely used in ecology alongside (or instead of) PERMANOVA.

    Convert the upper-triangle distances to RANKS (1 = closest).
    Let r_B = mean rank of pairs in DIFFERENT groups,
        r_W = mean rank of pairs in the SAME group.
    R = (r_B - r_W) / (N (N - 1) / 4)      with N = n(n - 1) / 2 pairs

R in [-1, 1]:
    R > 0   -> within-group distances smaller than between-group  (groups separate)
    R ~ 0   -> within and between distances are similar           (no group effect)
    R < 0   -> within-group distances larger                       (unusual)

p-value: permute group labels and recompute R many times.

Relationship to PERMANOVA
    ANOSIM uses ranks and is more robust to outliers but ignores magnitude
    of distances; PERMANOVA uses squared distances directly and is more
    powerful when distances are meaningful.  Both share the label-permutation
    p-value machinery and the same "location vs dispersion" caveat.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def _anosim_R(rank_mat, groups, iu):
    same = groups[iu[0]] == groups[iu[1]]
    r_flat = rank_mat[iu]
    r_W = r_flat[same].mean()
    r_B = r_flat[~same].mean()
    N = len(r_flat)
    return (r_B - r_W) / (N / 2)


def anosim(D, groups, n_perm: int = 999, seed: int = 0) -> dict:
    """ANOSIM on precomputed distance matrix D and grouping vector.

    D : n x n symmetric distance matrix.
    groups : length-n group labels.
    """
    D = np.asarray(D, dtype=float)
    groups = np.asarray(groups)
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    ranks = stats.rankdata(D[iu])
    rank_mat = np.zeros_like(D)
    rank_mat[iu] = ranks; rank_mat.T[iu] = ranks
    R_obs = _anosim_R(rank_mat, groups, iu)
    rng = np.random.default_rng(seed)
    R_perm = np.empty(n_perm)
    for b in range(n_perm):
        R_perm[b] = _anosim_R(rank_mat, rng.permutation(groups), iu)
    p = float((1 + np.sum(R_perm >= R_obs)) / (1 + n_perm))
    return {"R": float(R_obs), "p_value_perm": p,
            "n_perm": n_perm, "n": int(n),
            "K_groups": int(len(np.unique(groups))),
            "method": "ANOSIM (Clarke 1993)"}


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    n_per = 20
    X_a = rng.normal(0, 1, size=(n_per, 3))
    X_b = rng.normal(1.2, 1, size=(n_per, 3))
    X_c = rng.normal(-1.2, 1, size=(n_per, 3))
    X = np.vstack([X_a, X_b, X_c])
    groups = np.array(["A"] * n_per + ["B"] * n_per + ["C"] * n_per)
    diffs = X[:, None, :] - X[None, :, :]
    D = np.sqrt((diffs ** 2).sum(-1))

    print("=== ANOSIM (3 groups, shifted centers) ===")
    r = anosim(D, groups, n_perm=999)
    print(f"  R = {r['R']:.4f}, p = {r['p_value_perm']:.4f} ({r['n_perm']} perms)")

    print("\n=== ANOSIM (null: single group) ===")
    r_null = anosim(D, np.array(["X"] * (3 * n_per) if False else np.random.default_rng(0).choice(["A", "B", "C"], 3 * n_per)))
    print(f"  R = {r_null['R']:.4f}, p = {r_null['p_value_perm']:.4f}")

    print("\n--- library cross-check (scikit-bio) ---")
    try:
        from skbio.stats.distance import anosim as sk_anosim, DistanceMatrix
        dm = DistanceMatrix(D)
        r_sk = sk_anosim(dm, list(groups), permutations=999)
        print(f"  scikit-bio anosim: R = {r_sk['test statistic']:.4f}, p = {r_sk['p-value']:.4f}")
    except Exception as ex:
        print(f"  (scikit-bio not available: {ex})")
