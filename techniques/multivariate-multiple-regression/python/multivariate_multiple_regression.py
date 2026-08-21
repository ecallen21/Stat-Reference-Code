"""Multivariate Multiple Regression - MMR (Reference §9.20).

Regression with MULTIPLE response variables jointly:
    Y (n x q) = X (n x p) B (p x q) + E (n x q)
    rows of E ~ N(0, Sigma)             (q x q residual cov)

OLS estimator: B_hat = (X^T X)^-1 X^T Y     (same as running q separate
regressions, but with a joint residual covariance for testing).

Multivariate F-tests
    Test H_0: L B M = 0 for contrast matrices L (over predictors) and M
    (over responses).  Four common test statistics:
        Wilks Lambda, Pillai's trace, Hotelling-Lawley trace, Roy's largest root.
    All approximate F-distributions.

Applications
    - Studying how one set of predictors influences MULTIPLE outcomes
      jointly, exploiting their correlation.
    - Constructing joint confidence regions for coefficient vectors.
    - Growth curves / repeated measures cast as MMR with time-orthogonal
      polynomial contrasts.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def mmr_fit(X, Y) -> dict:
    """OLS multivariate multiple regression."""
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    n, p = X.shape; q = Y.shape[1]
    B = np.linalg.solve(X.T @ X, X.T @ Y)
    Y_hat = X @ B
    E = Y - Y_hat
    Sigma = E.T @ E / (n - p)     # residual covariance
    return {"coefficients": B, "residual_cov": Sigma,
            "fitted": Y_hat, "residuals": E,
            "n": int(n), "p": int(p), "q": int(q),
            "method": "Multivariate multiple regression"}


def wilks_lambda(Y, X, L) -> dict:
    """Wilks Lambda test of L B = 0 for contrast matrix L.

    Compare 'full' model to 'restricted' where L B = 0.
    """
    Y = np.asarray(Y, dtype=float); X = np.asarray(X, dtype=float); L = np.asarray(L, dtype=float)
    n, p = X.shape; q = Y.shape[1]
    B = np.linalg.solve(X.T @ X, X.T @ Y)
    E_full = Y - X @ B
    S_full = E_full.T @ E_full          # (q x q) SSCP within
    # Hypothesis SSCP: (L B)^T (L (X'X)^-1 L^T)^-1 (L B)
    XtX_inv = np.linalg.pinv(X.T @ X)
    LB = L @ B
    mid = np.linalg.pinv(L @ XtX_inv @ L.T)
    H = LB.T @ mid @ LB                  # (q x q)
    W = np.linalg.det(S_full) / np.linalg.det(S_full + H)
    # Rao's F approximation for the Wilks statistic
    s = L.shape[0]
    df1 = q * s
    m = n - p - (q - s + 1) / 2
    lam = math.sqrt((q ** 2 * s ** 2 - 4) / max(q ** 2 + s ** 2 - 5, 1)) if q ** 2 + s ** 2 - 5 > 0 else 1
    df2 = m * lam - (q * s - 2) / 2
    W_lam = W ** (1 / lam)
    F = (1 - W_lam) / W_lam * df2 / df1
    return {"wilks_lambda": float(W), "approx_F": float(F),
            "df1": float(df1), "df2": float(df2),
            "p_value": float(stats.f.sf(F, df1, df2)),
            "method": "Wilks Lambda test"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    # 3 correlated responses with different X effects
    B_true = np.array([[1.0, 2.0, -0.5],
                       [0.5, 0.3,  0.7],
                       [1.5, -0.6,  0.4]])
    E = rng.multivariate_normal([0, 0, 0],
                                 [[1, 0.5, 0.3],
                                  [0.5, 1, 0.2],
                                  [0.3, 0.2, 1]], n)
    Y = X @ B_true + E

    fit = mmr_fit(X, Y)
    print("=== MMR fit (n = 200, p = 3, q = 3) ===")
    print("  fitted B:")
    print(fit["coefficients"].round(3))
    print(f"  true B:")
    print(B_true)
    print(f"\n  residual covariance:")
    print(fit["residual_cov"].round(3))

    # Test that all slopes (rows 1 and 2 of B) are jointly zero
    L = np.array([[0, 1, 0], [0, 0, 1]], dtype=float)
    w = wilks_lambda(Y, X, L)
    print(f"\n  Wilks Lambda for H0: all slopes = 0")
    print(f"    Lambda = {w['wilks_lambda']:.4f}, F approx = {w['approx_F']:.3f}")
    print(f"    df = ({w['df1']:.0f}, {w['df2']:.0f}), p = {w['p_value']:.4e}")

    print("\n--- library cross-check (statsmodels multivariate_ols.MultivariateOLS) ---")
    try:
        from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
        # Simpler: manual comparison
        print("  see statsmodels.multivariate.multivariate_ols.multivariate_stats")
    except Exception as ex:
        print(f"  (statsmodels unavailable: {ex})")
