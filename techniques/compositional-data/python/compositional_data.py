"""Compositional data analysis (Reference Sec 38.2).

Compositional data live in the SIMPLEX -- vectors of positive parts
that sum to a constant (proportions, percentages).  Aitchison (1986)
showed classical statistics on the simplex are misleading:

  * Correlations are spurious -- constraint sum(x) = 1 induces
    negative dependence.
  * Distances are non-Euclidean.

Aitchison's fix: transform to real coordinates via log-ratios.

  * ALR (additive log ratio): alr(x)_i = log(x_i / x_D).  Non-unique
    but simple.
  * CLR (centred log ratio): clr(x)_i = log(x_i / gmean(x)).  Sums to
    zero (rank-deficient) but symmetric.
  * ILR (isometric log ratio): orthonormal basis on the simplex; full
    rank and preserves Euclidean geometry (Egozcue et al. 2003).

Analyses (regression, PCA, clustering) proceed in log-ratio
coordinates.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _gmean(x, axis=-1):
    return np.exp(np.log(x).mean(axis=axis))


def closure(x):
    """Renormalise rows to sum to 1."""
    x = np.asarray(x, dtype=float)
    return x / x.sum(axis=-1, keepdims=True)


def clr(x):
    """Centred log-ratio transform (rows sum to 0)."""
    x = closure(x)
    g = _gmean(x, axis=-1)[..., None]
    return np.log(x / g)


def alr(x, denom=-1):
    """Additive log-ratio transform (drops one column)."""
    x = closure(x)
    xd = x[..., denom][..., None]
    return np.log(np.delete(x, denom, axis=-1) / xd)


def ilr(x):
    """Isometric log-ratio via Gram-Schmidt on a sequential binary partition."""
    x = closure(x)
    D = x.shape[-1]
    # Sequential-binary-partition basis (Egozcue-Pawlowsky-Glahn-Mateu-Figueras 2003).
    z = np.zeros(x.shape[:-1] + (D - 1,))
    for i in range(D - 1):
        r = D - i - 1                     # size of denominator group
        num = np.log(x[..., i])
        den = np.log(x[..., i + 1:]).mean(axis=-1)
        z[..., i] = np.sqrt(r / (r + 1)) * (num - den)
    return z


def aitchison_distance(x, y):
    """||clr(x) - clr(y)||_2 -- Aitchison distance on the simplex."""
    return float(np.linalg.norm(clr(x) - clr(y)))


if __name__ == "__main__":
    print("=== Compositional data analysis: ALR / CLR / ILR + Aitchison distance ===\n")
    rng = np.random.default_rng(0)
    parts = ["A", "B", "C", "D"]
    # 6 synthetic compositions summing to 1
    raw = rng.dirichlet(alpha=[2, 3, 4, 1], size=6)
    print("  Raw compositions (rows sum = 1):")
    for i, r in enumerate(raw):
        print(f"    obs {i}:  " + "  ".join(f"{p}={v:.3f}" for p, v in zip(parts, r)))

    x = raw
    print("\n  CLR (rows sum = 0):")
    for r in clr(x):
        print("    " + "  ".join(f"{v:+.3f}" for v in r))

    print("\n  Spurious correlation warning -- Pearson corr on raw parts:")
    corr = np.corrcoef(x, rowvar=False)
    print(f"    corr(A, B) raw     = {corr[0, 1]:+.3f}")
    corr_clr = np.corrcoef(clr(x), rowvar=False)
    print(f"    corr(A, B) in CLR  = {corr_clr[0, 1]:+.3f}")

    print("\n  Aitchison distance between obs 0 and obs 1:")
    print(f"    d_A = {aitchison_distance(x[0], x[1]):.3f}   (vs Euclidean {np.linalg.norm(x[0] - x[1]):.3f})")

    print("\n--- library cross-check (R compositions/robCompositions; Python scikit-bio) ---")
