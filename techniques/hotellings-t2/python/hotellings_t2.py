"""Hotelling's T-squared test (Reference §9.1, §9.28).

The multivariate generalization of the t-test. Given p-dimensional observations
X_i, tests whether the mean vector equals a hypothesized value (one-sample) or
whether two multivariate samples share a common mean vector (two-sample).

One-sample
----------
    T^2 = n * (xbar - mu_0)' S^{-1} (xbar - mu_0)
    F = ((n - p) / (p (n - 1))) * T^2   ~  F(p, n - p) under H0

Two-sample (equal covariances, pooled S)
----------------------------------------
    T^2 = (n1 * n2 / (n1 + n2)) * (xbar1 - xbar2)' S_pool^{-1} (xbar1 - xbar2)
    F = ((n1 + n2 - p - 1) / (p (n1 + n2 - 2))) * T^2   ~  F(p, n1 + n2 - p - 1)

If p == 1 both reduce to the ordinary t-test.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def one_sample_t2(X, mu0) -> dict:
    """One-sample Hotelling's T^2 vs mean vector ``mu0``.

    Parameters
    ----------
    X : n x p matrix of observations.
    mu0 : length-p null-hypothesis mean vector.
    """
    X = np.asarray(X, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    n, p = X.shape
    if len(mu0) != p:
        raise ValueError("mu0 length must equal number of columns of X")
    xbar = X.mean(axis=0)
    S = np.cov(X, rowvar=False, ddof=1)         # p x p sample cov
    diff = xbar - mu0
    T2 = n * float(diff @ np.linalg.solve(S, diff))
    F = ((n - p) / (p * (n - 1))) * T2
    df1 = p; df2 = n - p
    p_val = float(stats.f.sf(F, df1, df2))
    return {"T_squared": T2, "F": F, "df1": df1, "df2": df2,
            "p_value": p_val, "n": n, "p_dim": p,
            "mean_vector": xbar.tolist(),
            "method": "Hotelling's T^2, one-sample"}


def two_sample_t2(X1, X2) -> dict:
    """Two-sample Hotelling's T^2 (equal covariance assumption; pooled S)."""
    X1 = np.asarray(X1, dtype=float); X2 = np.asarray(X2, dtype=float)
    n1, p = X1.shape
    n2, p2 = X2.shape
    if p != p2:
        raise ValueError("X1 and X2 must have the same number of columns")
    m1 = X1.mean(axis=0); m2 = X2.mean(axis=0)
    S1 = np.cov(X1, rowvar=False, ddof=1)
    S2 = np.cov(X2, rowvar=False, ddof=1)
    S_pool = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
    diff = m1 - m2
    T2 = (n1 * n2 / (n1 + n2)) * float(diff @ np.linalg.solve(S_pool, diff))
    df1 = p; df2 = n1 + n2 - p - 1
    F = (df2 / (p * (n1 + n2 - 2))) * T2
    p_val = float(stats.f.sf(F, df1, df2))
    return {"T_squared": T2, "F": F, "df1": df1, "df2": df2,
            "p_value": p_val, "n1": n1, "n2": n2, "p_dim": p,
            "mean_diff": diff.tolist(),
            "method": "Hotelling's T^2, two-sample (equal covariances)"}


def library_versions(X1, X2):
    """statsmodels doesn't ship a direct T^2 but MANOVA gives the same F."""
    from statsmodels.multivariate.manova import MANOVA
    import pandas as pd
    Xc = np.vstack([X1, X2])
    grp = np.array([1] * len(X1) + [2] * len(X2))
    df = pd.DataFrame(Xc, columns=[f"v{i}" for i in range(Xc.shape[1])])
    df["g"] = grp
    m = MANOVA.from_formula(f"{' + '.join(df.columns[:-1])} ~ g", data=df)
    res = m.mv_test().results["g"]["stat"]
    # Hotelling-Lawley Trace for 2 groups equals the 2-sample T^2 / (n1+n2-2) something,
    # but we just report the F test that matches:
    return {"statsmodels MANOVA Hotelling-Lawley":
            {"F": float(res.loc["Hotelling-Lawley trace", "F Value"]),
             "num_df": float(res.loc["Hotelling-Lawley trace", "Num DF"]),
             "den_df": float(res.loc["Hotelling-Lawley trace", "Den DF"]),
             "p": float(res.loc["Hotelling-Lawley trace", "Pr > F"])}}


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    # p = 3 dimensions; mu_true = [1.0, 2.0, 3.0]
    n = 60
    mu_true = np.array([1.0, 2.0, 3.0])
    Sigma = np.array([[1.0, 0.3, 0.2],
                      [0.3, 1.0, 0.4],
                      [0.2, 0.4, 1.0]])
    X = rng.multivariate_normal(mu_true, Sigma, size=n)

    print("=== One-sample T^2 vs mu0 = [1, 2, 3] (true mean; expect large p) ===")
    out = one_sample_t2(X, [1.0, 2.0, 3.0])
    for k, v in out.items(): print(f"  {k:14s}: {v}")

    print("\n=== One-sample T^2 vs mu0 = [0, 0, 0] (expect tiny p) ===")
    out = one_sample_t2(X, [0.0, 0.0, 0.0])
    for k, v in out.items(): print(f"  {k:14s}: {v}")

    # Two-sample
    mu_shift = mu_true + np.array([0.5, 0.0, -0.3])
    Y = rng.multivariate_normal(mu_shift, Sigma, size=55)
    print("\n=== Two-sample T^2 (moderate mean shift) ===")
    out = two_sample_t2(X, Y)
    for k, v in out.items(): print(f"  {k:14s}: {v}")

    print("\n--- library (statsmodels MANOVA) ---")
    for k, v in library_versions(X, Y).items():
        print(f"  {k}: {v}")
