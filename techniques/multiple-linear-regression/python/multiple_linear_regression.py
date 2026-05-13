"""Multiple linear regression (Reference §5.2).

Model: y = X beta + eps,   eps ~ N(0, sigma^2 I_n)

with X an n x p design matrix (the first column is usually a column of 1s for
the intercept). From-scratch OLS via the normal equations:

    beta_hat = (X' X)^{-1} X' y
    Var(beta_hat) = sigma^2 (X' X)^{-1}
    sigma_hat^2  = RSS / (n - p)

Outputs everything ``lm()`` / ``OLS()`` print: coefficients, SEs, t-stats,
p-values, 95% CIs, residual SE, R^2, adjusted R^2, the F-test for "all slopes
= 0", and ANOVA decomposition.

R^2 and friends
    R^2          = 1 - RSS / TSS
    adj R^2      = 1 - (1 - R^2) * (n - 1) / (n - p)         penalizes extra predictors
    overall F    = ((TSS - RSS) / (p - 1)) / (RSS / (n - p))  ~ F(p - 1, n - p)

For numerical stability we solve via ``numpy.linalg.lstsq`` rather than inverting
X'X directly.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


def fit(X, y, names: Sequence[str] | None = None) -> dict:
    """OLS fit of y on X (X already includes any intercept column).

    Parameters
    ----------
    X : array-like, shape (n, p).
    y : array-like, length n.
    names : optional column names for nicer output.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    if names is None:
        names = [f"x{i}" for i in range(p)]
    # Solve via QR / lstsq for stability
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    rss = float(resid @ resid)
    df_r = n - p
    sigma2 = rss / df_r
    sigma = math.sqrt(sigma2)
    # (X'X)^-1 for the variance of beta_hat
    XtX_inv = np.linalg.pinv(X.T @ X)
    var_beta = sigma2 * XtX_inv
    se_beta = np.sqrt(np.diag(var_beta))
    t_stats = beta / se_beta
    p_vals = 2 * stats.t.sf(np.abs(t_stats), df_r)
    # CI on each coefficient
    tc = stats.t.ppf(0.975, df_r)
    ci_lower = beta - tc * se_beta
    ci_upper = beta + tc * se_beta
    # R^2 / adjusted R^2 / overall F
    y_mean = y.mean()
    tss = float(((y - y_mean) ** 2).sum())
    r2 = 1 - rss / tss if tss > 0 else float("nan")
    # Adjusted R^2 only meaningful with an intercept
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p)
    # Overall F (test that all NON-INTERCEPT slopes are 0)
    has_intercept = bool(np.allclose(X[:, 0], 1.0))
    if has_intercept and p > 1:
        F = (r2 / (p - 1)) / ((1 - r2) / (n - p))
        p_F = float(stats.f.sf(F, p - 1, n - p))
    else:
        F = p_F = float("nan")
    return {"n": n, "p": p, "df_residual": df_r,
            "names": list(names),
            "beta": beta.tolist(), "se_beta": se_beta.tolist(),
            "t_stats": t_stats.tolist(), "p_values": p_vals.tolist(),
            "ci_lower": ci_lower.tolist(), "ci_upper": ci_upper.tolist(),
            "rss": rss, "sigma_hat": sigma,
            "r_squared": r2, "adj_r_squared": adj_r2,
            "F_overall": F, "p_F_overall": p_F,
            "residuals": resid.tolist(), "fitted": y_hat.tolist(),
            "vcov": var_beta.tolist()}


def predict(fit_obj: dict, X_new, conf: float = 0.95,
            kind: str = "confidence") -> dict:
    """Predict at new design rows.

    ``kind`` = "confidence" -> CI for mean response; "prediction" -> PI for new y.
    """
    X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
    beta = np.asarray(fit_obj["beta"]); V = np.asarray(fit_obj["vcov"])
    sigma = fit_obj["sigma_hat"]; df_r = fit_obj["df_residual"]
    yhat = X_new @ beta
    # var(yhat_i) = x_i^T V x_i  (for mean response)
    var_mean = np.einsum("ij,jk,ik->i", X_new, V, X_new)
    if kind == "prediction":
        var = var_mean + sigma * sigma
    elif kind == "confidence":
        var = var_mean
    else:
        raise ValueError("kind must be 'confidence' or 'prediction'")
    se = np.sqrt(var)
    tc = stats.t.ppf(0.5 + conf / 2, df_r)
    return {"yhat": yhat.tolist(), "se": se.tolist(),
            "lower": (yhat - tc * se).tolist(),
            "upper": (yhat + tc * se).tolist(), "kind": kind}


def library_versions(X_no_const, y):
    import statsmodels.api as sm
    X = sm.add_constant(X_no_const)
    res = sm.OLS(y, X).fit()
    return {"statsmodels params": res.params.tolist(),
            "statsmodels bse": res.bse.tolist(),
            "statsmodels rsquared": float(res.rsquared),
            "statsmodels rsquared_adj": float(res.rsquared_adj),
            "statsmodels F": (float(res.fvalue), float(res.f_pvalue))}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 80
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + 0.0 * x3 + rng.normal(0, 1.0, n)
    X = np.column_stack([np.ones(n), x1, x2, x3])

    res = fit(X, y, names=["intercept", "x1", "x2", "x3"])
    print("=== Coefficient table ===")
    print(f"  {'name':10s} {'beta':>10s} {'se':>10s} {'t':>8s} {'p':>10s} "
          f"{'95% CI':>20s}")
    for i, nm in enumerate(res["names"]):
        print(f"  {nm:10s} {res['beta'][i]:10.4f} {res['se_beta'][i]:10.4f} "
              f"{res['t_stats'][i]:8.3f} {res['p_values'][i]:10.4g}  "
              f"[{res['ci_lower'][i]:.3f}, {res['ci_upper'][i]:.3f}]")
    print(f"\n  sigma_hat = {res['sigma_hat']:.4f}   df_resid = {res['df_residual']}")
    print(f"  R^2 = {res['r_squared']:.4f}   adj R^2 = {res['adj_r_squared']:.4f}")
    print(f"  Overall F({res['p']-1}, {res['df_residual']}) = "
          f"{res['F_overall']:.3f}   p = {res['p_F_overall']:.4g}")

    # CI for the mean at a new row, plus PI
    new_row = [[1.0, 0.5, -1.0, 0.0]]
    print(f"\n=== Prediction at x = (intercept=1, x1=0.5, x2=-1, x3=0) ===")
    print("CI :", predict(res, new_row, kind="confidence"))
    print("PI :", predict(res, new_row, kind="prediction"))

    print("\n--- library (statsmodels) ---")
    for k, v in library_versions(np.column_stack([x1, x2, x3]), y).items():
        print(f"  {k}: {v}")
