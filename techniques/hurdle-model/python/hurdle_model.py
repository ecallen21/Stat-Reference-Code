"""Hurdle model for zero-heavy count / continuous data (Reference Sec 7.20).

Mullahy 1986 'Specification and testing of some modified count data
models'; Cragg 1971 (double-hurdle for continuous). A TWO-PART model:

    1. Binary hurdle: P(Y > 0 | X) -- logistic / probit.
    2. POSITIVE part: p(Y | Y > 0, X) -- truncated Poisson,
       zero-truncated NegBin, or log-normal for continuous.

Contrast with zero-inflated:
    * Zero-inflated: TWO sources of zeros (structural + sampling).
    * Hurdle:        ONE source; the two parts are independent stages.

We fit a Poisson hurdle to a synthetic dataset with strong excess
zeros, compare to plain Poisson.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize
from scipy.special import gammaln


def logistic_nll(beta, X, y_pos):
    p = 1 / (1 + np.exp(-(X @ beta)))
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return -np.sum(y_pos * np.log(p) + (1 - y_pos) * np.log(1 - p))


def truncated_poisson_nll(gamma, X, y):
    """Positive-Poisson: P(Y | Y > 0) = lam^y * exp(-lam) / (y! * (1 - exp(-lam)))."""
    lam = np.exp(X @ gamma)
    mask = y > 0
    yp = y[mask]; lp = lam[mask]
    ll = (yp * np.log(lp) - lp - gammaln(yp + 1)
          - np.log(1 - np.exp(-lp) + 1e-300))
    return -ll.sum()


def poisson_nll(gamma, X, y):
    lam = np.exp(X @ gamma)
    return -np.sum(y * np.log(lam + 1e-300) - lam - gammaln(y + 1))


def fit_hurdle(X, y):
    y_pos = (y > 0).astype(float)
    p = X.shape[1]
    beta = minimize(logistic_nll, np.zeros(p), args=(X, y_pos), method="L-BFGS-B").x
    gamma = minimize(truncated_poisson_nll, np.zeros(p), args=(X, y), method="L-BFGS-B").x
    return {"beta_hurdle": beta, "gamma_count": gamma}


if __name__ == "__main__":
    print("=== Hurdle model (Mullahy 1986) ===\n")
    rng = np.random.default_rng(0)
    n = 3000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.c_[np.ones(n), x1, x2]

    #  DGP: 60% zero via logistic, positive part Poisson(lambda = exp(0.4 + 0.5 x1))
    p_pos = 1 / (1 + np.exp(-(-0.4 + 1.0 * x1 - 0.2 * x2)))
    is_pos = rng.uniform(size=n) < p_pos
    lam = np.exp(0.4 + 0.5 * x1 - 0.1 * x2)
    y = np.where(is_pos, rng.poisson(lam), 0)

    print(f"  n = {n}, zero fraction = {(y == 0).mean() * 100:.1f}%")
    print(f"  True hurdle beta   ~ (-0.40, +1.00, -0.20)")
    print(f"  True positive gamma ~ (+0.40, +0.50, -0.10)\n")

    r = fit_hurdle(X, y)
    print(f"  Est hurdle beta     = {r['beta_hurdle'].round(3).tolist()}")
    print(f"  Est positive gamma  = {r['gamma_count'].round(3).tolist()}")

    #  Compare with plain Poisson (ignores excess zeros)
    beta_p = minimize(poisson_nll, np.zeros(3), args=(X, y), method="L-BFGS-B").x
    print(f"\n  Plain Poisson beta  = {beta_p.round(3).tolist()}  <- biased toward smaller lambda")
    print(f"  Reason: Poisson mixes structural zeros with true count zeros.")

    print("\n--- library cross-check (pscl / hurdlr R; statsmodels HurdleModel Python) ---")
