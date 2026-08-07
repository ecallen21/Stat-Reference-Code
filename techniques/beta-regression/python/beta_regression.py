"""Beta regression for proportions in (0, 1) (Reference §5.20).

Continuous outcomes strictly in (0, 1) -- proportions, rates, percentages
divided by 100 -- have variance that shrinks near 0 and 1.  OLS or logistic
regression on the logit is a workaround; Beta regression handles it cleanly
via a Beta distribution.

Reparameterization (Ferrari & Cribari-Neto 2004)
    Beta(mu * phi, (1 - mu) * phi)
    with mean mu in (0, 1) and precision phi > 0.
    Var(y) = mu (1 - mu) / (1 + phi)

Model:
    logit(mu_i) = X_i beta         mean submodel (link = logit)
    log(phi_i)  = Z_i gamma        (optional precision submodel; constant if absent)

MLE by BFGS on (beta, log phi) or (beta, gamma) if variable dispersion.

Boundary cases y = 0 or y = 1: transform y = (y (n - 1) + 0.5) / n
(Smithson & Verkuilen 2006), or use zero/one-inflated beta.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import special    # SciPy scalar-special functions (gamma, digamma, beta, lgamma)
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def _logit(p): return np.log(p / (1 - p))
def _sigmoid(x): return 1 / (1 + np.exp(-x))


def beta_regression(X, y, precision_link_x=None) -> dict:
    """Beta regression with logit mean-link and (optional) log precision-link.

    X : mean-submodel design matrix.
    y : outcome in (0, 1).
    precision_link_x : precision-submodel design (defaults to intercept only).
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    if not ((y > 0) & (y < 1)).all():
        raise ValueError("all y must be in the open interval (0, 1)")
    if precision_link_x is None:
        Z = np.ones((n, 1))
    else:
        Z = np.asarray(precision_link_x, dtype=float)
    q = Z.shape[1]
    def neg_ll(theta):
        beta = theta[:p]; gamma = theta[p:p + q]
        mu = _sigmoid(X @ beta)
        phi = np.exp(Z @ gamma)
        a = mu * phi; b = (1 - mu) * phi
        return -np.sum(
            special.gammaln(a + b) - special.gammaln(a) - special.gammaln(b)
            + (a - 1) * np.log(y) + (b - 1) * np.log1p(-y)
        )
    beta0 = np.zeros(p); beta0[0] = _logit(min(max(y.mean(), 0.01), 0.99))
    gamma0 = np.zeros(q); gamma0[0] = math.log(10.0)
    res = minimize(neg_ll, np.concatenate([beta0, gamma0]), method="BFGS")
    beta = res.x[:p]; gamma = res.x[p:p + q]
    se = np.sqrt(np.diag(res.hess_inv))
    return {"beta": beta, "gamma_precision": gamma,
            "se_beta": se[:p], "se_gamma": se[p:p + q],
            "log_lik": float(-res.fun),
            "n": int(n),
            "mean_precision": float(math.exp(gamma[0]) if q == 1 else np.exp(Z @ gamma).mean()),
            "method": "Beta regression (Ferrari-Cribari-Neto)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n); X = np.column_stack([np.ones(n), x])
    beta_true = np.array([0.0, 1.2]); phi_true = 20.0
    mu_true = _sigmoid(X @ beta_true)
    y = rng.beta(mu_true * phi_true, (1 - mu_true) * phi_true)

    print("=== Beta regression (constant precision) ===")
    r = beta_regression(X, y)
    print(f"  beta          = {r['beta'].round(3)}     (true {beta_true})")
    print(f"  precision phi = {r['mean_precision']:.3f}   (true {phi_true})")
    print(f"  log-lik = {r['log_lik']:.2f}")

    print("\n=== Beta regression with variable precision (log phi = z coefs) ===")
    z = rng.normal(size=n); Z = np.column_stack([np.ones(n), z])
    phi_true_var = np.exp(3.0 + 0.5 * z)
    y2 = rng.beta(mu_true * phi_true_var, (1 - mu_true) * phi_true_var)
    r = beta_regression(X, y2, precision_link_x=Z)
    print(f"  beta               = {r['beta'].round(3)}   (true [0, 1.2])")
    print(f"  gamma (precision)  = {r['gamma_precision'].round(3)}   (true [3.0, 0.5])")

    print("\n--- library cross-check (statsmodels BetaModel) ---")
    try:
        from statsmodels.othermod.betareg import BetaModel
        m = BetaModel(y, X).fit(disp=False)
        print(f"  statsmodels beta params: {m.params.round(3)}")
    except Exception as ex:
        print(f"  (statsmodels BetaModel unavailable: {ex})")
