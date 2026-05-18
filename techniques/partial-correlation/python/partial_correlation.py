"""Partial and semi-partial (part) correlation (Reference §4.5).

Both ask: what's the association between x and y once we account for the
linear effect of one or more covariates Z = (z_1, ..., z_p)?

Definitions
-----------
Partial correlation r_{xy . Z}
  - Residualize x and y on Z (via OLS), then correlate the residuals.
  - For a single covariate z, closed form:
        r_{xy.z} = (r_{xy} - r_{xz} r_{yz}) / sqrt((1 - r_{xz}^2)(1 - r_{yz}^2))

Semi-partial (part) correlation r_{x(y.Z)}
  - Residualize ONLY y on Z (or vice versa), then correlate the y-residuals with x.
  - For one covariate:
        r_{x(y.z)} = (r_{xy} - r_{xz} r_{yz}) / sqrt(1 - r_{yz}^2)
  - Interpretation: incremental "unique" association of y with x after removing
    z's linear effect from y; r^2 = the unique R^2 contribution of x.

Both extend to multiple covariates Z via the residual definition. We
implement the matrix-inverse form for partial correlation among many variables
(the precision matrix gives ALL pairwise partials in one shot).

Test for H0: rho_partial = 0 is a t-test with df = n - 2 - |Z|:
    t = r * sqrt((n - 2 - p) / (1 - r^2)),  p = # of conditioning variables
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def pearson_r(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mx, my = x.mean(), y.mean()
    sxy = float(np.sum((x - mx) * (y - my)))
    sxx = float(np.sum((x - mx) ** 2)); syy = float(np.sum((y - my) ** 2))
    return sxy / math.sqrt(sxx * syy)


def _ols_residuals(y, Z):
    """OLS residuals of y on columns of Z (with an intercept)."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    X = np.column_stack([np.ones(len(y)), Z])
    beta, *_ = np.linalg.lstsq(X, np.asarray(y, dtype=float), rcond=None)
    return np.asarray(y, dtype=float) - X @ beta


def partial_correlation(x, y, Z) -> dict:
    """Partial correlation r_{xy . Z}, by residualizing both x and y on Z."""
    rx = _ols_residuals(x, Z)
    ry = _ols_residuals(y, Z)
    r = pearson_r(rx, ry)
    n = len(x)
    p = 1 if np.ndim(Z) == 1 else np.shape(Z)[1]
    df = n - 2 - p
    if abs(r) == 1.0:
        return {"partial_r": r, "df": df, "p_value": 0.0,
                "n_covariates": p, "method": "partial"}
    t = r * math.sqrt(df / (1 - r * r))
    p_val = float(2 * stats.t.sf(abs(t), df))
    return {"partial_r": r, "df": df, "p_value": p_val,
            "n_covariates": p, "method": "partial"}


def semi_partial_correlation(x, y, Z, residualize: str = "y") -> dict:
    """Semi-partial r_{x(y.Z)} (default: residualize y on Z, leave x alone)."""
    if residualize == "y":
        ry = _ols_residuals(y, Z); rx = np.asarray(x, dtype=float)
    elif residualize == "x":
        rx = _ols_residuals(x, Z); ry = np.asarray(y, dtype=float)
    else:
        raise ValueError("residualize must be 'x' or 'y'")
    r = pearson_r(rx, ry)
    n = len(x)
    p = 1 if np.ndim(Z) == 1 else np.shape(Z)[1]
    df = n - 2 - p
    t = r * math.sqrt(df / (1 - r * r)) if abs(r) < 1 else float("inf")
    p_val = float(2 * stats.t.sf(abs(t), df))
    return {"semi_partial_r": r, "df": df, "p_value": p_val,
            "residualized": residualize, "method": "semi-partial"}


def partial_correlation_matrix(X) -> np.ndarray:
    """All pairwise partial correlations among the columns of ``X`` (each pair
    conditional on the rest). Computed from the inverse of the covariance matrix:
        r_{ij | rest} = -P_ij / sqrt(P_ii P_jj),   P = inv(cov(X))
    """
    X = np.asarray(X, dtype=float)
    P = np.linalg.inv(np.cov(X, rowvar=False))
    d = np.sqrt(np.diag(P))
    R = -P / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


def library_versions(x, y, z):
    try:
        import pingouin as pg
        import pandas as pd
        df = pd.DataFrame({"x": x, "y": y, "z": z})
        return {"pingouin.partial_corr (x, y | z)": pg.partial_corr(df, x="x", y="y", covar="z")}
    except ImportError:
        return {"note": "install 'pingouin' for pg.partial_corr"}


if __name__ == "__main__":
    rng = np.random.default_rng(9)
    n = 100
    z = rng.normal(0, 1, n)
    # x and y both depend on z; their direct dependence is small
    x = 0.7 * z + 0.3 * rng.normal(0, 1, n)
    y = 0.8 * z + 0.2 * rng.normal(0, 1, n)

    print(f"r_xy raw            = {pearson_r(x, y):+.4f}")
    print(f"r_xy | z (partial)  = {partial_correlation(x, y, z)['partial_r']:+.4f}")
    print(f"r_x(y.z) (semi)     = {semi_partial_correlation(x, y, z)['semi_partial_r']:+.4f}")
    print()
    print("Partial test:");      [print(f"  {k:14s}: {v}") for k, v in partial_correlation(x, y, z).items()]
    print("Semi-partial test:"); [print(f"  {k:14s}: {v}") for k, v in semi_partial_correlation(x, y, z).items()]

    print("\n=== Partial-correlation matrix among (x, y, z) ===")
    print(partial_correlation_matrix(np.column_stack([x, y, z])))

    print("\n--- library ---")
    for k, v in library_versions(x, y, z).items():
        print(f"  {k}:\n{v}")
