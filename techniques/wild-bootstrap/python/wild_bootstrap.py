"""Wild bootstrap for regression with heteroscedastic errors (Reference §10.5).

Setting: y_i = X_i' beta + e_i,  Var(e_i) may vary with i (heteroscedasticity).
Case resampling (row bootstrap) works but loses efficiency; residual bootstrap
(sample e_i's with replacement, add to fitted values) assumes homoscedasticity.

Wild bootstrap keeps the ORIGINAL residual for each row but multiplies by a
mean-zero, unit-variance weight:

    e*_i  =  e_hat_i * w_i          where  E[w_i] = 0, Var[w_i] = 1
    y*_i  =  X_i' beta_hat + e*_i

Then refit the regression on (X, y*) and record the coefficient of interest.
The weights w_i preserve the per-observation error scale (via e_hat_i) while
inducing randomness -- so heteroscedasticity is preserved automatically.

Common weight distributions:
    Rademacher : w_i in {-1, +1} with p = 0.5
    Mammen 2-point : w_i in {(1 - sqrt(5))/2, (1 + sqrt(5))/2}  with probs matched
        so that E[w] = 0, E[w^2] = 1, E[w^3] = 1.  Gives third-moment accuracy
        (recommended over Rademacher in small samples).
    Standard normal : w_i ~ N(0, 1)   (less common)
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _wild_weights(scheme: str, size: int, rng):
    if scheme == "rademacher":
        return rng.choice([-1.0, 1.0], size=size)
    if scheme == "mammen":
        # Mammen (1993) 2-point distribution
        phi = (1 + math.sqrt(5)) / 2      # golden ratio
        p = phi / math.sqrt(5)             # prob of the smaller value
        a = -(math.sqrt(5) - 1) / 2        # (1 - sqrt(5))/2
        b = (math.sqrt(5) + 1) / 2         # (1 + sqrt(5))/2
        return np.where(rng.random(size=size) < p, a, b)
    if scheme == "normal":
        return rng.normal(0.0, 1.0, size=size)
    raise ValueError("scheme must be 'rademacher', 'mammen', or 'normal'")


def wild_bootstrap_regression(X, y, coef_index: int = None,
                                weights: str = "mammen", n_boot: int = 2000,
                                conf: float = 0.95, seed: int = 0) -> dict:
    """Wild bootstrap CI for OLS regression coefficient(s).

    Parameters
    ----------
    X : n x p design matrix (include a column of 1s for intercept if wanted).
    y : n-length response.
    coef_index : index of the coefficient to report. If None, returns SE and
        CIs for ALL coefficients.
    weights : 'rademacher' / 'mammen' / 'normal'.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    rng = np.random.default_rng(seed)
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta_hat; resid = y - yhat
    beta_star = np.empty((n_boot, p))
    for b in range(n_boot):
        w = _wild_weights(weights, n, rng)
        y_star = yhat + resid * w
        beta_b, *_ = np.linalg.lstsq(X, y_star, rcond=None)
        beta_star[b] = beta_b
    alpha = 1 - conf
    out = {"beta_hat": beta_hat.tolist(),
            "bootstrap_SE": beta_star.std(axis=0, ddof=1).tolist(),
            "weights": weights,
            "n_boot": n_boot, "n": n, "p": p,
            "method": f"wild bootstrap ({weights} weights)"}
    if coef_index is not None:
        lo, hi = np.quantile(beta_star[:, coef_index], [alpha / 2, 1 - alpha / 2])
        out["coef_index"] = coef_index
        out["CI_percentile"] = {"lower": float(lo), "upper": float(hi)}
    else:
        cis = [np.quantile(beta_star[:, k], [alpha / 2, 1 - alpha / 2]) for k in range(p)]
        out["CI_percentile_per_coef"] = [
            {"lower": float(lo), "upper": float(hi)} for lo, hi in cis]
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(29)
    n = 200
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    X = np.column_stack([np.ones(n), x1, x2])
    # Heteroscedastic errors: variance grows with |x1|
    err = rng.normal(0, 0.5 + np.abs(x1), size=n)
    y = 1.0 + 0.7 * x1 - 0.4 * x2 + err

    print("=== Wild bootstrap (Mammen weights) for slope of x1 (true = 0.7) ===")
    out = wild_bootstrap_regression(X, y, coef_index=1, weights="mammen", n_boot=2000)
    print(f"  beta_hat[x1] = {out['beta_hat'][1]:.4f}")
    print(f"  SE_boot      = {out['bootstrap_SE'][1]:.4f}")
    print(f"  CI (percentile): [{out['CI_percentile']['lower']:.4f}, {out['CI_percentile']['upper']:.4f}]")

    print("\n=== Wild bootstrap (Rademacher weights) for slope of x1 ===")
    out2 = wild_bootstrap_regression(X, y, coef_index=1, weights="rademacher", n_boot=2000)
    print(f"  SE_boot = {out2['bootstrap_SE'][1]:.4f}")
    print(f"  CI: [{out2['CI_percentile']['lower']:.4f}, {out2['CI_percentile']['upper']:.4f}]")

    print("\n=== Compare to naive OLS SE (would assume homoscedasticity) ===")
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta; resid = y - yhat
    sigma2 = float((resid ** 2).sum() / (n - X.shape[1]))
    XTX_inv = np.linalg.inv(X.T @ X)
    se_ols = np.sqrt(np.diag(sigma2 * XTX_inv))
    print(f"  OLS SE[x1] (homoscedastic) = {se_ols[1]:.4f}")
    print(f"  (Wild bootstrap SE is often LARGER for heteroscedastic errors)")

    print("\n=== Cross-check: HC1 robust SE (also handles heteroscedasticity) ===")
    import statsmodels.api as sm
    res_ols = sm.OLS(y, X).fit(cov_type="HC1")
    print(f"  HC1 SE[x1] = {float(res_ols.bse[1]):.4f}")
