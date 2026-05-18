"""Weighted Least Squares (Reference §5.10).

When the errors are heteroscedastic, OLS is still unbiased but no longer
efficient and the standard errors are wrong. WLS gives each observation a
weight w_i inversely proportional to its error variance:

    beta_hat_WLS = (X' W X)^{-1} X' W y,    W = diag(w_1, ..., w_n)

Equivalent to OLS on the transformed system
    (W^{1/2} X)  (beta)  =  (W^{1/2} y)
which is why the algorithm is essentially "scale rows by sqrt(w_i), then OLS."

Choosing the weights
  - If Var(eps_i) is known proportional to some function of x:  w_i = 1 / Var
    (common: w_i = 1 / x_i^2 for multiplicative error, w_i = 1 / x_i for Poisson-like)
  - From replicates: w_i = n_i (if y_i is an average of n_i observations)
  - Two-stage / iteratively re-weighted: fit OLS, regress |e_i| or e_i^2 on the
    predictors, derive weights from that fit, re-fit WLS. (We show this loop.)
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def wls_fit(X, y, weights) -> dict:
    """WLS fit via the transformation X_w = W^{1/2} X, y_w = W^{1/2} y, then OLS."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    if (w <= 0).any():
        raise ValueError("all weights must be > 0")
    sw = np.sqrt(w)
    Xw = X * sw[:, None]; yw = y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    yhat = X @ beta; e = y - yhat
    n, p = X.shape
    # Weighted RSS / sigma^2
    wrss = float((w * e ** 2).sum())
    sigma2 = wrss / (n - p)
    # Var(beta) = sigma^2 (X' W X)^{-1}
    var_beta = sigma2 * np.linalg.pinv(X.T @ (X * w[:, None]))
    se = np.sqrt(np.diag(var_beta))
    t = beta / se; p_vals = 2 * stats.t.sf(np.abs(t), n - p)
    return {"beta": beta.tolist(), "se": se.tolist(),
            "t_stats": t.tolist(), "p_values": p_vals.tolist(),
            "sigma_hat": math.sqrt(sigma2), "df_residual": n - p,
            "weighted_rss": wrss, "residuals": e.tolist(),
            "fitted": yhat.tolist()}


def iteratively_reweighted_LS(X, y, max_iter: int = 10, tol: float = 1e-6) -> dict:
    """Two-stage / iterative scheme for unknown weights.

    Start with OLS, derive weights from the squared residuals via an auxiliary
    regression, refit WLS, repeat until coefficients stabilize.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    w = np.ones(n)
    beta_prev = np.zeros(p)
    for it in range(max_iter):
        res = wls_fit(X, y, w)
        beta = np.asarray(res["beta"])
        if it > 0 and np.max(np.abs(beta - beta_prev)) < tol:
            break
        beta_prev = beta
        e = np.asarray(res["residuals"])
        # Fit log(e^2) on X (with intercept) to get a variance model
        log_e2 = np.log(np.maximum(e ** 2, 1e-12))
        gamma, *_ = np.linalg.lstsq(X, log_e2, rcond=None)
        var_hat = np.exp(X @ gamma)
        w = 1 / np.maximum(var_hat, 1e-12)
    res["iterations"] = it + 1
    return res


def library_versions(X, y, weights):
    import statsmodels.api as sm
    res = sm.WLS(y, X, weights=weights).fit()
    return {"statsmodels params": res.params.tolist(),
            "statsmodels bse": res.bse.tolist(),
            "statsmodels rsquared": float(res.rsquared)}


if __name__ == "__main__":
    rng = np.random.default_rng(6)
    n = 200
    x = rng.uniform(1, 10, n)
    # Variance grows with x:  sd_i = 0.5 * x_i
    y = 1.0 + 2.0 * x + rng.normal(0, 0.5 * x, n)
    X = np.column_stack([np.ones(n), x])

    print("=== OLS (wrong: ignores heteroscedasticity) ===")
    res_ols = wls_fit(X, y, weights=np.ones(n))
    print(f"  beta = {res_ols['beta']}   SE = {res_ols['se']}")

    print("\n=== WLS with KNOWN weights w_i = 1 / x_i^2 ===")
    w_true = 1 / x ** 2
    res_wls = wls_fit(X, y, weights=w_true)
    print(f"  beta = {res_wls['beta']}   SE = {res_wls['se']}")
    print("  (Note the SEs on the slope are smaller / more honest.)")

    print("\n=== Iteratively reweighted (unknown weights) ===")
    res_iter = iteratively_reweighted_LS(X, y)
    print(f"  beta = {res_iter['beta']}   SE = {res_iter['se']}")
    print(f"  iterations = {res_iter['iterations']}")

    print("\n--- library (statsmodels.WLS) ---")
    for k, v in library_versions(X, y, w_true).items():
        print(f"  {k}: {v}")
