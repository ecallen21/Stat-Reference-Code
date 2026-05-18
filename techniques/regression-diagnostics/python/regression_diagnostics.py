"""Regression diagnostics after OLS (Reference §5.6, §5.30, §5.39).

After fitting y = X beta + eps via OLS, check the assumptions and identify
problematic observations. The standard quantities, all derivable from
H = X (X'X)^{-1} X' (the "hat matrix") and the residuals:

    h_i        leverage          = H_ii   (in [0, 1], sum_i h_i = p)
    e_i        raw residual      = y_i - y_hat_i
    s_i        std residual      = e_i / (sigma_hat * sqrt(1 - h_i))
    t_i        studentized resid = e_i / (sigma_(i)_hat * sqrt(1 - h_i))
                                   (sigma estimated WITHOUT observation i; "deleted t")
    D_i        Cook's distance   = (s_i^2 / p) * (h_i / (1 - h_i))
    DFFITS_i   change in prediction at i, scaled
                                 = t_i * sqrt(h_i / (1 - h_i))
    DFBETAS_ij scaled change in beta_j when obs i is dropped
                                 = (beta_j - beta_(i)_j) / SE(beta_(i)_j)

Rules of thumb (Belsley/Kuh/Welsch 1980):
    high leverage      :  h_i > 2 p / n       (some authors use 3 p / n)
    influential point  :  |D_i| > 4 / n        OR  |DFFITS_i| > 2 sqrt(p / n)
    influential on j-th coeff :  |DFBETAS_ij| > 2 / sqrt(n)

A point can be high-leverage (extreme x) without being influential (if its
residual is small), and vice-versa. Cook's D combines the two.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def fit_ols(X, y):
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def hat_matrix(X):
    X = np.asarray(X, dtype=float)
    # H = X (X'X)^{-1} X'
    return X @ np.linalg.pinv(X.T @ X) @ X.T


def diagnostics(X, y) -> dict:
    """Per-observation diagnostics for an OLS fit.

    Returns a dict with arrays of length n:
      leverage, residuals, std_residuals, studentized (deleted) residuals,
      cooks_d, dffits, dfbetas (n x p), plus the rule-of-thumb thresholds.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta = fit_ols(X, y)
    yhat = X @ beta
    e = y - yhat
    H = hat_matrix(X)
    h = np.diag(H)
    rss = float(e @ e); sigma2 = rss / (n - p); sigma = math.sqrt(sigma2)
    s = e / (sigma * np.sqrt(np.clip(1 - h, 1e-15, None)))   # standardized
    # Deleted-sigma form for studentized residuals
    sigma2_i = ((n - p) * sigma2 - e ** 2 / np.clip(1 - h, 1e-15, None)) / (n - p - 1)
    sigma_i = np.sqrt(np.clip(sigma2_i, 0, None))
    t_i = e / (sigma_i * np.sqrt(np.clip(1 - h, 1e-15, None)))
    cooks = (s ** 2 / p) * (h / np.clip(1 - h, 1e-15, None))
    dffits = t_i * np.sqrt(h / np.clip(1 - h, 1e-15, None))
    # DFBETAS_ij = (beta_j - beta_(i)_j) / SE(beta_(i)_j)
    XtX_inv = np.linalg.pinv(X.T @ X)
    R = XtX_inv @ X.T                                       # p x n
    # Change in beta when dropping obs i:  R[:, i] * e_i / (1 - h_i)
    delta_beta = (R * (e / np.clip(1 - h, 1e-15, None)))    # p x n
    # SE_(i) of beta_j approximated using sigma_(i): SE_j_(i) = sigma_(i) * sqrt(diag(XtX_inv)_j)
    diagXtX = np.sqrt(np.diag(XtX_inv))
    dfbetas = (delta_beta.T / (np.outer(sigma_i, diagXtX))).T  # p x n
    return {
        "n": n, "p": p,
        "leverage": h.tolist(),
        "residuals": e.tolist(),
        "std_residuals": s.tolist(),
        "studentized": t_i.tolist(),
        "cooks_d": cooks.tolist(),
        "dffits": dffits.tolist(),
        "dfbetas": dfbetas.T.tolist(),   # n x p
        "sigma_hat": sigma, "df_residual": n - p,
        "thresholds": {
            "leverage_high":  2 * p / n,
            "cooks_d_large":  4 / n,
            "dffits_large":   2 * math.sqrt(p / n),
            "dfbetas_large":  2 / math.sqrt(n),
        },
    }


def library_versions(X, y):
    import statsmodels.api as sm
    import pandas as pd
    res = sm.OLS(y, X).fit()
    inf = res.get_influence()
    return {"statsmodels leverage (first 5)": inf.hat_matrix_diag[:5].tolist(),
            "statsmodels cooks_d (first 5)": inf.cooks_distance[0][:5].tolist(),
            "statsmodels dffits (first 5)": inf.dffits[0][:5].tolist(),
            "statsmodels studentized_resid (first 5)":
                inf.resid_studentized_external[:5].tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 30
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    y = 1 + 2 * x1 - 1.5 * x2 + rng.normal(0, 1, n)
    # Inject an influential point: extreme x1 AND extreme y
    x1[0] = 5.0; y[0] = -10.0
    X = np.column_stack([np.ones(n), x1, x2])

    d = diagnostics(X, y)
    print("=== Per-observation diagnostics (first 6 rows) ===")
    print(f"  {'i':>3s} {'lev':>8s} {'std_r':>8s} {'stud':>8s} {'cookD':>8s} {'dffits':>8s}")
    for i in range(6):
        print(f"  {i:3d} {d['leverage'][i]:8.4f} {d['std_residuals'][i]:8.3f} "
              f"{d['studentized'][i]:8.3f} {d['cooks_d'][i]:8.4f} {d['dffits'][i]:8.3f}")
    print("\nThresholds:", d["thresholds"])

    # Flag suspicious points
    h_thr = d["thresholds"]["leverage_high"]
    cd_thr = d["thresholds"]["cooks_d_large"]
    flagged = [i for i in range(n) if d["leverage"][i] > h_thr or d["cooks_d"][i] > cd_thr]
    print(f"\nFlagged indices (high leverage or large Cook's D): {flagged}")

    print("\n--- library (statsmodels.get_influence) ---")
    for k, v in library_versions(X, y).items():
        print(f"  {k}: {v}")
