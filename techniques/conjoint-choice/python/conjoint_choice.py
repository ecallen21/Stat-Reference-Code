"""Conjoint / discrete choice experiment (Reference Sec 36.7).

McFadden 1974.  Respondents choose one alternative from a menu of
J options characterised by attributes X.  MULTINOMIAL LOGIT (MNL):

    U_ij = X_ij^T beta + eps_ij,       eps ~ Type-I extreme value
    Pr(y_i = j) = exp(X_ij^T beta) / sum_k exp(X_ik^T beta)

MLE on the log-likelihood.  MIXED LOGIT allows random coefficients
per respondent (heterogeneity).
"""
from __future__ import annotations    # stdlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np    # numerical arrays
from scipy.optimize import minimize


def _softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def mnl_loglik(beta, X, y):
    """X: (n, J, p); y: chosen alt (n,)."""
    U = X @ beta                              # (n, J)
    P = _softmax(U, axis=1)
    n = len(y)
    return float(np.log(P[np.arange(n), y] + 1e-300).sum())


def fit_mnl(X, y):
    p = X.shape[2]
    res = minimize(lambda b: -mnl_loglik(b, X, y), x0=np.zeros(p),
                   method="BFGS")
    return {"beta": res.x, "loglik": float(-res.fun), "converged": res.success}


if __name__ == "__main__":
    print("=== Conjoint / discrete choice: multinomial logit ===\n")
    rng = np.random.default_rng(0)
    n, J, p = 500, 3, 3
    # 3 attributes: price (negative preference), quality, brand indicator
    X = rng.normal(0, 1, (n, J, p))
    beta_true = np.array([-1.5, 1.0, 0.5])
    U = X @ beta_true + rng.gumbel(0, 1, (n, J))
    y = U.argmax(axis=1)

    r = fit_mnl(X, y)
    print(f"  True beta:      {beta_true}")
    print(f"  Estimated beta: {np.round(r['beta'], 3)}")
    print(f"  Log-likelihood: {r['loglik']:.2f}")
    # Baseline: random guessing gives ll = -n log(J)
    print(f"  Null (random) log-lik: {-n * np.log(J):.2f}")
    print(f"  McFadden pseudo-R^2: {1 - r['loglik'] / (-n * np.log(J)):.3f}\n")

    print("--- library cross-check (R mlogit, apollo, ChoiceModelR; Python xlogit, statsmodels) ---")
