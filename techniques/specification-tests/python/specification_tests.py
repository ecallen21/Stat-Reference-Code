"""Specification and diagnostic tests for OLS regression (Reference §5.7, §5.21).

Formal tests of the linear-model assumptions:

  - Breusch-Pagan          H0: var(eps_i) = sigma^2 (homoscedasticity)
                           Regress e_i^2 on the predictors; n * R^2 ~ chi^2_{p-1}.
  - White's test           Like BP but adds squares and cross-products of the
                           regressors -- tests homoscedasticity AND any
                           omitted nonlinearity in X.
  - Durbin-Watson          H0: no first-order autocorrelation in residuals.
                           DW = sum((e_t - e_{t-1})^2) / sum(e_t^2)  in [0, 4]
                           DW ~ 2 -> no autocorrelation; DW < 2 positive; > 2 negative.
  - Ramsey RESET           H0: model is correctly specified (no omitted nonlinear
                           terms).
                           Re-fit with y_hat^2 (and optionally y_hat^3) as
                           extra regressors; test the new terms' joint
                           significance via the F statistic.
  - Shapiro-Wilk           on the residuals -- normality (see techniques/normality-tests).
  - Jarque-Bera (5.21)     a quick skew+kurt normality test on the residuals.

All build directly on a fitted OLS, so the inputs are X, y. We also expose a
helper to evaluate everything in one call.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import NamedTuple, Sequence    # stdlib: type hints (Sequence = indexable iterable; NamedTuple = tuple with named fields)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


class OLSFit(NamedTuple):
    """Plain OLS fit. Unpacks like a tuple: ``beta, yhat, e = _ols(X, y)``."""
    beta: np.ndarray       # coefficient vector
    fitted: np.ndarray     # fitted values y_hat = X @ beta
    residuals: np.ndarray  # residuals e = y - y_hat


def _ols(X, y):
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    e = y - yhat
    return OLSFit(beta=beta, fitted=yhat, residuals=e)


def breusch_pagan(X, y) -> dict:
    """Breusch-Pagan test for heteroscedasticity.

    Regress squared residuals on the original predictors; the LM statistic
    n * R^2 from that auxiliary regression is chi^2_{p-1} under H0.
    """
    X = np.asarray(X, dtype=float); _, _, e = _ols(X, y)
    n, p = X.shape
    z = e ** 2
    # Aux regression: z on X (X already has its intercept if needed)
    beta_aux, *_ = np.linalg.lstsq(X, z, rcond=None)
    z_hat = X @ beta_aux
    ss_res = float(((z - z_hat) ** 2).sum())
    ss_tot = float(((z - z.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    LM = n * r2
    df = p - 1   # subtract the intercept column
    return {"LM": LM, "df": df, "p_value": float(stats.chi2.sf(LM, df)),
            "method": "Breusch-Pagan (LM)"}


def whites_test(X, y) -> dict:
    """White's test = Breusch-Pagan on an expanded regressor set with squares + cross-products."""
    X = np.asarray(X, dtype=float); n, p = X.shape
    # Drop the intercept (assumed first col) for the squaring/cross step
    has_int = bool(np.allclose(X[:, 0], 1.0))
    Xc = X[:, 1:] if has_int else X
    cols = [np.ones(n)] if has_int else []
    cols.extend(Xc[:, j] for j in range(Xc.shape[1]))
    cols.extend(Xc[:, j] ** 2 for j in range(Xc.shape[1]))
    for i in range(Xc.shape[1]):
        for j in range(i + 1, Xc.shape[1]):
            cols.append(Xc[:, i] * Xc[:, j])
    Z = np.column_stack(cols)
    _, _, e = _ols(X, y)
    z = e ** 2
    beta_aux, *_ = np.linalg.lstsq(Z, z, rcond=None)
    z_hat = Z @ beta_aux
    ss_res = float(((z - z_hat) ** 2).sum())
    ss_tot = float(((z - z.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    LM = n * r2
    df = Z.shape[1] - 1
    return {"LM": LM, "df": df, "p_value": float(stats.chi2.sf(LM, df)),
            "method": "White"}


def durbin_watson(X, y) -> dict:
    """DW statistic for first-order autocorrelation in residuals.

    DW = sum_{t=2..n} (e_t - e_{t-1})^2  /  sum_t e_t^2,  in [0, 4].
    A formal p-value depends on X; ``lmtest::dwtest`` computes one. Here we
    return the statistic and the rule-of-thumb interpretation.
    """
    _, _, e = _ols(X, y)
    num = float(((np.diff(e)) ** 2).sum())
    den = float((e ** 2).sum())
    dw = num / den
    interp = ("positive autocorrelation" if dw < 1.5
              else "negative autocorrelation" if dw > 2.5
              else "no strong evidence of autocorrelation")
    return {"DW": dw, "interpretation": interp,
            "method": "Durbin-Watson (no p-value here; use lmtest::dwtest)"}


def ramsey_reset(X, y, powers=(2,)) -> dict:
    """Ramsey RESET: omitted-nonlinearity test.

    Re-fit with extra columns y_hat^2, y_hat^3, ... and run an F-test on their
    joint significance.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    beta0, yhat, e0 = _ols(X, y)
    n, p = X.shape
    extras = np.column_stack([yhat ** k for k in powers])
    X_aug = np.column_stack([X, extras])
    _, _, e1 = _ols(X_aug, y)
    rss0 = float((e0 ** 2).sum()); rss1 = float((e1 ** 2).sum())
    q = extras.shape[1]                                      # # of extra cols
    F = ((rss0 - rss1) / q) / (rss1 / (n - p - q))
    p_val = float(stats.f.sf(F, q, n - p - q))
    return {"F": F, "df1": q, "df2": n - p - q, "p_value": p_val,
            "powers": tuple(powers), "method": "Ramsey RESET"}


def shapiro_residuals(X, y) -> dict:
    _, _, e = _ols(X, y)
    sw = stats.shapiro(e)
    return {"W": float(sw.statistic), "p_value": float(sw.pvalue),
            "method": "Shapiro-Wilk on residuals"}


def jarque_bera_residuals(X, y) -> dict:
    _, _, e = _ols(X, y)
    jb = stats.jarque_bera(e)
    return {"JB": float(jb.statistic), "p_value": float(jb.pvalue),
            "method": "Jarque-Bera on residuals"}


def run_all(X, y) -> dict:
    return {"breusch_pagan": breusch_pagan(X, y),
            "white": whites_test(X, y),
            "durbin_watson": durbin_watson(X, y),
            "ramsey_reset": ramsey_reset(X, y),
            "shapiro_residuals": shapiro_residuals(X, y),
            "jarque_bera_residuals": jarque_bera_residuals(X, y)}


def library_versions(X, y):
    from statsmodels.stats.diagnostic import (het_breuschpagan, het_white,
                                              linear_reset)
    from statsmodels.stats.stattools import durbin_watson as sm_dw
    import statsmodels.api as sm
    res = sm.OLS(y, X).fit()
    bp = het_breuschpagan(res.resid, X)
    wh = het_white(res.resid, X)
    dw = sm_dw(res.resid)
    rt = linear_reset(res, power=2, use_f=True)
    return {"statsmodels BP (LM, p, F, p_F)": list(map(float, bp)),
            "statsmodels White (LM, p, F, p_F)": list(map(float, wh)),
            "statsmodels DW": float(dw),
            "statsmodels RESET (F)": (float(rt.statistic), float(rt.pvalue))}


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    X = np.column_stack([np.ones(n), x1, x2])

    print("=== Clean (homoscedastic, no omitted nonlinearity) ===")
    y_clean = 1 + 2 * x1 - x2 + rng.normal(0, 1, n)
    for k, v in run_all(X, y_clean).items():
        print(f"  {k:24s}: {v}")

    print("\n=== Heteroscedastic (Var(eps) grows with x1) ===")
    y_het = 1 + 2 * x1 - x2 + rng.normal(0, 1 + 0.5 * np.abs(x1), n)
    for k, v in run_all(X, y_het).items():
        print(f"  {k:24s}: {v}")

    print("\n=== Omitted nonlinearity (true model has x1^2) ===")
    y_nl = 1 + 2 * x1 + 0.7 * x1 ** 2 - x2 + rng.normal(0, 1, n)
    for k, v in run_all(X, y_nl).items():
        print(f"  {k:24s}: {v}")

    print("\n--- library (statsmodels) ---")
    for k, v in library_versions(X, y_het).items():
        print(f"  {k}: {v}")
