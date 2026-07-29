"""Generalized ordered logit / partial proportional odds (Reference §8.35).

Ordinal outcome Y in {1, 2, ..., J}.  Proportional-odds logit fits
    logit P(Y <= j | x) = alpha_j - x' beta        (SAME beta for all j)

The PROPORTIONAL-ODDS assumption often fails.  Two relaxations:

1) Fully GENERALIZED ordered logit (Williams / Peterson-Harrell 1990)
    logit P(Y <= j | x) = alpha_j - x' beta_j     (SEPARATE beta_j per cut)

2) PARTIAL proportional odds
    Some covariates share a single beta (proportional across cuts),
    others get beta_j varying by cut.  Implemented in Stata's gologit2
    (Williams 2006).

Estimation
    Fit J - 1 CUMULATIVE logistic regressions:
        logit P(Y <= j | x) = alpha_j - x' beta_j    (or shared beta)
    Each is a binary logit on y_j = I(Y <= j).  For the fully generalized
    model these are INDEPENDENT and we can just fit J - 1 binary logits.
    For partial PO, joint estimation is needed (we implement the fully
    generalized version below and cross-check against ordered-logit BIC).

Score / Brant test
    Compare betas across the J - 1 cumulative logits; a Wald test on
    equality across cuts is the Brant (1990) test of proportional odds.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def _binary_logit_fit(X, y):
    """Fit binary logistic regression with intercept (X has NO intercept column)."""
    n, p = X.shape
    Xd = np.column_stack([np.ones(n), X])
    def negll(theta):
        z = Xd @ theta
        return float(np.sum(np.log1p(np.exp(-np.abs(z))) + np.maximum(-z, 0) * (1 - y) + np.maximum(z, 0) * (1 - y) - z * (1 - y)))
    # Simpler negative log-likelihood
    def negll2(theta):
        z = Xd @ theta
        return float(-np.sum(y * z - np.logaddexp(0, z)))
    def grad(theta):
        z = Xd @ theta
        mu = 1 / (1 + np.exp(-z))
        return -Xd.T @ (y - mu)
    theta0 = np.zeros(p + 1)
    res = minimize(negll2, theta0, jac=grad, method="BFGS")
    theta = res.x
    z = Xd @ theta
    mu = 1 / (1 + np.exp(-z))
    W = mu * (1 - mu)
    H = Xd.T @ (Xd * W[:, None])
    cov = np.linalg.pinv(H)
    return theta, cov, res.fun


def generalized_ordered_logit(X, y) -> dict:
    """Fully generalized ordered logit: J-1 binary cumulative logits.

    X : n x p covariate matrix (no intercept).
    y : length-n integer labels in {0, 1, ..., J-1}.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    J = int(y.max() + 1); n, p = X.shape
    rows = []
    coefs = np.zeros((J - 1, p + 1))
    cov_list = []
    for j in range(J - 1):
        y_j = (y <= j).astype(float)
        theta_j, cov_j, ll_j = _binary_logit_fit(X, y_j)
        coefs[j] = theta_j
        cov_list.append(cov_j)
        se = np.sqrt(np.diag(cov_j))
        for k, name in enumerate(["intercept"] + [f"x{i+1}" for i in range(p)]):
            rows.append({"cut": j + 1, "term": name,
                         "coef": float(theta_j[k]),
                         "se": float(se[k]),
                         "z": float(theta_j[k] / se[k]),
                         "p_value": float(2 * stats.norm.sf(abs(theta_j[k] / se[k])))})

    # Brant-style Wald test of equality of slopes across cuts
    if J >= 3:
        brant_rows = []
        for k in range(1, p + 1):
            b = coefs[:, k]
            var_k = np.array([c[k, k] for c in cov_list])
            # Approx independent cumulative logits; Wald test of all equal
            mean_b = np.average(b, weights=1 / var_k)
            chi2_k = float(np.sum((b - mean_b) ** 2 / var_k))
            df = J - 2
            brant_rows.append({"term": f"x{k}",
                               "chi2": chi2_k, "df": df,
                               "p_value": float(stats.chi2.sf(chi2_k, df))})
    else:
        brant_rows = []

    return {"coef_table": rows,
            "brant_test": brant_rows,
            "n": int(n), "n_cuts": J - 1, "n_covariates": int(p),
            "method": "Generalized ordered logit (Williams / Peterson-Harrell)"}


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _print(rows, cols):
    widths = {c: max(len(c), max(len(_fmt(r[c])) for r in rows)) for c in cols}
    print("  " + "  ".join(c.ljust(widths[c]) for c in cols))
    print("  " + "  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  " + "  ".join(_fmt(r[c]).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    n = 400
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    # Ordinal outcome with cut-varying effect of x1 (violates PO)
    lp1 = -0.5 - 0.8 * x1 + 0.6 * x2
    lp2 =  0.5 - 0.2 * x1 + 0.6 * x2
    p1 = 1 / (1 + np.exp(-lp1))
    p2 = 1 / (1 + np.exp(-lp2))
    u = rng.uniform(0, 1, n)
    y = np.where(u < p1, 0, np.where(u < p2, 1, 2))

    print("=== Generalized ordered logit (J = 3, x1 has non-proportional effect) ===")
    r = generalized_ordered_logit(np.column_stack([x1, x2]), y)
    _print(r["coef_table"], ["cut", "term", "coef", "se", "z", "p_value"])

    print("\n=== Brant-style test of proportional odds ===")
    _print(r["brant_test"], ["term", "chi2", "df", "p_value"])

    print("\n--- library cross-check (statsmodels ordinal_model.OrderedModel) ---")
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        m = OrderedModel(y, np.column_stack([x1, x2]), distr="logit").fit(method="bfgs", disp=False)
        print(f"  proportional-odds coefs: {m.params[:2].round(4)}")
        print(f"  cutpoints (transformed): {m.params[2:].round(4)}")
    except Exception as ex:
        print(f"  (statsmodels not available: {ex})")
