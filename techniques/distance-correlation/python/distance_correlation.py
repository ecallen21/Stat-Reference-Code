"""Distance correlation (Reference §4.8).

Szekely, Rizzo & Bakirov (2007). Unlike Pearson/Spearman/Kendall, distance
correlation dCor = 0 if and only if x and y are INDEPENDENT -- it detects
nonlinear and non-monotonic dependence too. Defined for vectors of arbitrary
dimension (we use scalars / 1-D arrays here).

Algorithm (O(n^2))
------------------
1. Pairwise distance matrices  a_{ij} = |x_i - x_j|,  b_{ij} = |y_i - y_j|.
2. Double-center them (subtract row mean, col mean, add grand mean):
     A = a - row_mean(a) - col_mean(a) + grand_mean(a)
     B = b - row_mean(b) - col_mean(b) + grand_mean(b)
3. Distance covariance squared:  dCov^2(x, y) = mean( A .* B )
   Distance variances:           dVar^2(x)    = mean( A .* A )
   Distance correlation:         dCor(x, y)   = sqrt( dCov^2 / sqrt(dVar^2(x) * dVar^2(y)) )

Properties
  - dCor in [0, 1]; dCor = 0 iff x and y are independent.
  - dCor = 1 only when y is a linear function of x (for scalar variables).
  - No distribution assumptions. Useful for testing independence directly.

Significance: permutation test (shuffle y and recompute dCor). The asymptotic
distribution of n * dCov^2 under independence has a known form but is awkward
to invert; permutation is the standard.
"""
from __future__ import annotations

import numpy as np


def _double_centered(M: np.ndarray) -> np.ndarray:
    row = M.mean(axis=1, keepdims=True)
    col = M.mean(axis=0, keepdims=True)
    grand = M.mean()
    return M - row - col + grand


def distance_correlation(x, y) -> dict:
    """Distance correlation between (n,) vectors ``x`` and ``y``.

    Returns dCor, dCov^2, dVar^2(x), dVar^2(y), and the energy statistic n * dCov^2.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")
    n = x.shape[0]
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = _double_centered(a)
    B = _double_centered(b)
    dcov2 = float((A * B).mean())
    dvarx2 = float((A * A).mean())
    dvary2 = float((B * B).mean())
    dcor = (dcov2 / np.sqrt(dvarx2 * dvary2)) ** 0.5 if dvarx2 > 0 and dvary2 > 0 else 0.0
    return {"dCor": float(dcor), "dCov_sq": dcov2,
            "dVar_sq_x": dvarx2, "dVar_sq_y": dvary2,
            "energy_statistic_nT": n * dcov2, "n": n}


def distance_correlation_test(x, y, n_perm: int = 500, rng=None) -> dict:
    """Permutation-test p-value for H0: x and y are independent."""
    rng = rng or np.random.default_rng(0)
    obs = distance_correlation(x, y)["dCor"]
    y_arr = np.asarray(y, dtype=float)
    sim = np.empty(n_perm)
    for k in range(n_perm):
        sim[k] = distance_correlation(x, rng.permutation(y_arr))["dCor"]
    p = float((sim >= obs).mean())
    return {"dCor": obs, "p_value": p, "n_permutations": n_perm}


def library_versions(x, y):
    out = {}
    try:
        import dcor  # type: ignore
        out["dcor.distance_correlation"] = float(dcor.distance_correlation(x, y))
    except ImportError:
        out["dcor"] = "install 'dcor' for the reference implementation (pip install dcor)"
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("=== Independent x, y ~ N(0,1)  (population dCor = 0) ===")
    x = rng.normal(0, 1, 200); y = rng.normal(0, 1, 200)
    print(distance_correlation_test(x, y, n_perm=300, rng=rng))
    print("(sample dCor > 0 due to finite-sample noise; the permutation p-value should be large)")

    print("\n=== Strong nonlinear relationship: y = x^2 + noise ===")
    print("(Pearson r ~ 0; dCor should be large)")
    x = np.linspace(-3, 3, 200); y = x ** 2 + rng.normal(0, 0.4, 200)
    print(distance_correlation(x, y))
    print("permutation p-value:", distance_correlation_test(x, y, n_perm=300, rng=rng))

    print("\n--- library ---")
    for k, v in library_versions(x, y).items():
        print(f"  {k}: {v}")
