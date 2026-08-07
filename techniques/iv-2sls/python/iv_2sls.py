"""Instrumental variables / two-stage least squares (Reference §5.22).

Endogeneity: OLS is biased when a covariate X_endog is correlated with the
error term (omitted variables, reverse causation, measurement error).
Instrumental variables (IV) uses one or more INSTRUMENTS Z that satisfy:
    (i)  RELEVANCE:  Z is correlated with X_endog (given exogenous covariates)
    (ii) EXCLUSION:  Z affects Y only through X_endog (given the model)

Two-stage least squares (2SLS)
    Stage 1: OLS regress X_endog on Z + exogenous X -> X_hat
    Stage 2: OLS regress y on X_hat + exogenous X

Consistent for the causal effect of X_endog under (i) + (ii).

Weak-instrument diagnostic
    First-stage F on the excluded instruments; rule of thumb F >= 10 for
    a single endogenous variable (Staiger-Stock 1997).

Standard errors
    Naive 2SLS SEs are wrong because they use residuals from y - X_hat beta
    instead of y - X beta.  Correct SE:
        Cov(beta_2SLS) = sigma^2 (X_hat' X_hat)^-1
    with sigma^2 from residuals y - X beta_2SLS.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def two_sls(y, X_endog, X_exog, Z_instruments) -> dict:
    """Two-stage least squares.

    y             : outcome.
    X_endog       : endogenous regressor(s), n x k1.
    X_exog        : exogenous regressors (with intercept column), n x k2.
    Z_instruments : instruments for X_endog (excluded from outcome eq), n x m.
    """
    y = np.asarray(y, dtype=float)
    X_endog = np.atleast_2d(np.asarray(X_endog, dtype=float)).reshape(len(y), -1)
    X_exog = np.atleast_2d(np.asarray(X_exog, dtype=float)).reshape(len(y), -1)
    Z_instruments = np.atleast_2d(np.asarray(Z_instruments, dtype=float)).reshape(len(y), -1)
    n = len(y); k1 = X_endog.shape[1]; k2 = X_exog.shape[1]; m = Z_instruments.shape[1]
    if m < k1: raise ValueError("must have at least as many instruments as endogenous regressors")

    # Stage 1: regress each column of X_endog on [Z, X_exog]
    Z_full = np.column_stack([Z_instruments, X_exog])
    X_hat = np.zeros_like(X_endog)
    F_first = np.zeros(k1)
    for j in range(k1):
        beta_s1, *_ = np.linalg.lstsq(Z_full, X_endog[:, j], rcond=None)
        X_hat[:, j] = Z_full @ beta_s1
        # F-test on the EXCLUDED instruments (first m columns)
        resid_full = X_endog[:, j] - X_hat[:, j]
        rss_full = float(resid_full @ resid_full)
        beta_s1r, *_ = np.linalg.lstsq(X_exog, X_endog[:, j], rcond=None)
        resid_res = X_endog[:, j] - X_exog @ beta_s1r
        rss_res = float(resid_res @ resid_res)
        F_first[j] = ((rss_res - rss_full) / m) / (rss_full / (n - m - k2))

    # Stage 2: regress y on [X_hat, X_exog]
    W = np.column_stack([X_hat, X_exog])
    beta_2sls, *_ = np.linalg.lstsq(W, y, rcond=None)
    # Correct residuals use ORIGINAL X (not X_hat) with beta_2sls
    W_orig = np.column_stack([X_endog, X_exog])
    resid = y - W_orig @ beta_2sls
    sigma2 = float(resid @ resid / (n - (k1 + k2)))
    cov = sigma2 * np.linalg.pinv(W.T @ W)
    se = np.sqrt(np.diag(cov))
    names = [f"endog{j+1}" for j in range(k1)] + [f"exog{j+1}" for j in range(k2)]
    return {"coefficients": {names[j]: float(beta_2sls[j]) for j in range(len(beta_2sls))},
            "se": {names[j]: float(se[j]) for j in range(len(se))},
            "first_stage_F": F_first,
            "sigma2": sigma2,
            "n": int(n),
            "weak_instrument_flag": bool(np.min(F_first) < 10),
            "method": "Two-stage least squares (2SLS)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    # DGP with endogeneity: u affects both x and y
    z = rng.normal(size=n)                     # instrument
    u = rng.normal(size=n)                     # unobserved confounder
    x = 0.5 + 1.5 * z + 0.8 * u + rng.normal(0, 0.5, n)  # endogenous X
    y = 1.0 + 2.0 * x + 0.6 * u + rng.normal(0, 0.5, n)  # true effect of x = 2.0

    X_endog = x.reshape(-1, 1)
    X_exog = np.ones((n, 1))
    Z = z.reshape(-1, 1)

    print("=== Data with endogeneity (true causal effect of x = 2.0) ===")
    print("\n  Naive OLS (biased):")
    W_ols = np.column_stack([np.ones(n), x])
    beta_ols, *_ = np.linalg.lstsq(W_ols, y, rcond=None)
    print(f"    intercept = {beta_ols[0]:.3f}, slope = {beta_ols[1]:.3f}")

    print("\n  2SLS with z as instrument:")
    r = two_sls(y, X_endog, X_exog, Z)
    print(f"    coefficients: {r['coefficients']}")
    print(f"    SE:           {r['se']}")
    print(f"    first-stage F = {r['first_stage_F'][0]:.2f}  (weak-instrument threshold 10)")
    print(f"    weak instrument? {r['weak_instrument_flag']}")

    print("\n--- library cross-check (linearmodels IV2SLS) ---")
    try:
        from linearmodels.iv import IV2SLS
        iv = IV2SLS(y, np.ones(n), x, z).fit()
        print(f"  linearmodels IV coefficient: {iv.params['endog.0']:.3f}")
    except Exception as ex:
        print(f"  (linearmodels unavailable: {ex})")
