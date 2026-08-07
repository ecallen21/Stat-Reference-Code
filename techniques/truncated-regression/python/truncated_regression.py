"""Truncated regression (Reference §5.18).

TRUNCATED sample: subjects with y outside a threshold are NEVER OBSERVED
(not just recorded at the boundary as in Tobit).  A study that only
enrolls households earning below the poverty line, a school that only
tests students above a score cutoff: OLS on these truncated samples is
biased even more severely than in the censored case.

Truncated Normal MLE
    y_i^* = X_i beta + eps_i,   eps_i ~ N(0, sigma^2)
    Observed only if L < y_i^* < U.
    Density given truncation:
        f(y | X) = phi((y - X beta) / sigma) / sigma
                   / [Phi((U - X beta) / sigma) - Phi((L - X beta) / sigma)]

MLE by BFGS on (beta, log sigma).

Contrast with censoring (Tobit)
    Censoring: boundary observations still contribute Pr(y* <= L) or
    Pr(y* >= U) mass.
    Truncation: they don't exist in the dataset at all -- we correct
    for the SELECTED sample by dividing by the truncation probability.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def truncated_normal_regression(X, y, lower: float = None, upper: float = None) -> dict:
    """MLE for a truncated-normal linear model.

    lower / upper : truncation points; None means one-sided.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    # Sanity: y must all lie in (lower, upper)
    if lower is not None and (y <= lower).any():
        raise ValueError("some y <= lower truncation")
    if upper is not None and (y >= upper).any():
        raise ValueError("some y >= upper truncation")

    def neg_ll(theta):
        beta = theta[:p]; sigma = math.exp(theta[p])
        mu = X @ beta
        z = (y - mu) / sigma
        num = stats.norm.logpdf(z) - math.log(sigma)
        if lower is None:
            denom = stats.norm.logcdf((upper - mu) / sigma)
        elif upper is None:
            denom = stats.norm.logsf((lower - mu) / sigma)
        else:
            denom = np.log(stats.norm.cdf((upper - mu) / sigma) - stats.norm.cdf((lower - mu) / sigma) + 1e-300)
        return -np.sum(num - denom)
    beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
    theta0 = np.concatenate([beta0, [math.log(y.std())]])
    res = minimize(neg_ll, theta0, method="BFGS")
    beta_hat = res.x[:p]; sigma_hat = math.exp(res.x[p])
    se = np.sqrt(np.diag(res.hess_inv))
    return {"beta": beta_hat, "sigma": sigma_hat,
            "se_beta": se[:p], "se_log_sigma": float(se[p]),
            "log_lik": float(-res.fun),
            "n": int(n),
            "truncation": (lower, upper),
            "method": "Truncated-normal linear regression (MLE)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    beta_true = np.array([1.0, 2.0]); sigma_true = 1.0
    # Rejection-sample truncated at y in (-inf, 3)  (upper truncation at 3)
    Xall = []; yall = []
    while len(yall) < 500:
        x_new = rng.normal(size=200)
        y_star = 1 + 2 * x_new + rng.normal(0, sigma_true, 200)
        mask = y_star < 3.0
        Xall.extend(x_new[mask]); yall.extend(y_star[mask])
    Xall = np.array(Xall[:500]); yall = np.array(yall[:500])
    X = np.column_stack([np.ones(500), Xall])

    print("=== Upper-truncated normal regression (y < 3) ===")
    print("\n=== Naive OLS on truncated sample (biased) ===")
    beta_ols, *_ = np.linalg.lstsq(X, yall, rcond=None)
    print(f"  OLS beta = {beta_ols.round(3)}  (true {beta_true})")

    print("\n=== Truncated-normal MLE ===")
    r = truncated_normal_regression(X, yall, upper=3.0)
    print(f"  beta = {r['beta'].round(3)}  (true {beta_true})")
    print(f"  sigma = {r['sigma']:.3f}  (true {sigma_true})")
    print(f"  log-lik = {r['log_lik']:.2f}")

    print("\n--- library cross-check (statsmodels: no direct; use custom / R truncreg) ---")
    print("  R: truncreg::truncreg(y ~ x, point = 3, direction = 'right')")
