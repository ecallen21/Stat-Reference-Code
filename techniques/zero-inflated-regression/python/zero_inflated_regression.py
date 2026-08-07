"""Zero-inflated and hurdle count-regression models (Reference §5.24).

Count outcomes often have MORE zeros than Poisson / NB predicts.  Two
model families:

1) Zero-inflated (Lambert 1992)
    y_i = 0                with probability pi_i          (structural zero)
    y_i ~ Poisson(mu_i)    with probability (1 - pi_i)    (potentially zero)

    logit(pi_i) = Z_i gamma        (zero-inflation submodel)
    log(mu_i)   = X_i beta          (count submodel)

    Likelihood mixes a point-mass at 0 with a Poisson.  Fit both submodels
    jointly by MLE (BFGS on the log-likelihood).

2) Hurdle (two-part / Cragg 1971)
    y_i = 0             with probability pi_i
    y_i | y_i > 0 ~ TruncPoisson(mu_i)     (zero-truncated)

    Separates the "did an event occur?" decision from the "how many, given
    at least one?" count.  Two independent likelihoods -> fit as two GLMs.

ZIP vs hurdle
    ZIP: two paths to zero (structural + sampling).  Interpretable as a
    latent binary "susceptibility" indicator.
    Hurdle: exactly one path to zero.  Easier to fit; better when zero
    process and count process are conceptually distinct.

Substitute Poisson with NegBin for over-dispersion -> ZINB / hurdle-NB.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def zip_fit(X, Z, y) -> dict:
    """Zero-inflated Poisson MLE.  X = count design, Z = zero-inflation design."""
    X = np.asarray(X, dtype=float); Z = np.asarray(Z, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape; q = Z.shape[1]
    def neg_ll(theta):
        beta = theta[:p]; gamma = theta[p:p + q]
        mu = np.exp(X @ beta); pi = 1 / (1 + np.exp(-(Z @ gamma)))
        ll_zero = np.log(pi + (1 - pi) * np.exp(-mu) + 1e-300)
        ll_pos = np.log(1 - pi + 1e-300) + y * np.log(mu + 1e-300) - mu - np.array([math.lgamma(v + 1) for v in y])
        return -np.sum(np.where(y == 0, ll_zero, ll_pos))
    theta0 = np.zeros(p + q); theta0[0] = math.log(max(y.mean(), 0.1))
    res = minimize(neg_ll, theta0, method="BFGS")
    beta = res.x[:p]; gamma = res.x[p:p + q]
    se = np.sqrt(np.diag(res.hess_inv))
    return {"beta_count": beta, "gamma_zi": gamma,
            "se_beta": se[:p], "se_gamma": se[p:p + q],
            "log_lik": float(-res.fun),
            "aic": float(2 * (p + q) + 2 * res.fun),
            "n": int(n), "p_count": int(p), "q_zi": int(q),
            "method": "Zero-inflated Poisson MLE (BFGS)"}


def hurdle_fit(X, Z, y) -> dict:
    """Hurdle Poisson: logistic for zero vs positive, truncated Poisson for positive."""
    X = np.asarray(X, dtype=float); Z = np.asarray(Z, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape; q = Z.shape[1]
    is_zero = (y == 0).astype(float)
    # Logistic regression for P(y > 0)
    def neg_ll_zero(gamma):
        z = Z @ gamma
        return -np.sum((1 - is_zero) * z - np.logaddexp(0, z))
    res_z = minimize(neg_ll_zero, np.zeros(q), method="BFGS")
    gamma = res_z.x
    # Truncated Poisson for y > 0
    pos = y > 0; Xp = X[pos]; yp = y[pos]
    def neg_ll_pos(beta):
        mu = np.exp(Xp @ beta)
        ll = yp * np.log(mu + 1e-300) - mu - np.log1p(-np.exp(-mu) + 1e-300) - np.array([math.lgamma(v + 1) for v in yp])
        return -np.sum(ll)
    res_p = minimize(neg_ll_pos, np.zeros(p), method="BFGS")
    beta = res_p.x
    return {"beta_count": beta, "gamma_hurdle": gamma,
            "se_beta": np.sqrt(np.diag(res_p.hess_inv)),
            "se_gamma": np.sqrt(np.diag(res_z.hess_inv)),
            "log_lik": float(-res_z.fun - res_p.fun),
            "aic": float(2 * (p + q) + 2 * (res_z.fun + res_p.fun)),
            "n": int(n),
            "method": "Hurdle Poisson (two-part MLE)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n); X = np.column_stack([np.ones(n), x])
    z = rng.normal(size=n); Z = np.column_stack([np.ones(n), z])
    pi_true = 1 / (1 + np.exp(-(-1.0 + 0.5 * z)))
    mu_true = np.exp(0.5 + 0.6 * x)
    y = np.where(rng.uniform(size=n) < pi_true, 0, rng.poisson(mu_true))

    print("=== Zero-inflated Poisson MLE ===")
    r = zip_fit(X, Z, y)
    print(f"  beta_count (intercept, x)   = {r['beta_count'].round(3)}  (true 0.5, 0.6)")
    print(f"  gamma_zi   (intercept, z)   = {r['gamma_zi'].round(3)}  (true -1.0, 0.5)")
    print(f"  log-lik = {r['log_lik']:.2f}, AIC = {r['aic']:.2f}")

    print("\n=== Hurdle Poisson ===")
    r = hurdle_fit(X, Z, y)
    print(f"  beta_count            = {r['beta_count'].round(3)}")
    print(f"  gamma_hurdle          = {r['gamma_hurdle'].round(3)}")
    print(f"  log-lik = {r['log_lik']:.2f}, AIC = {r['aic']:.2f}")

    print("\n--- library cross-check (statsmodels ZeroInflatedPoisson) ---")
    try:
        from statsmodels.discrete.count_model import ZeroInflatedPoisson
        m = ZeroInflatedPoisson(y, X, exog_infl=Z, inflation="logit").fit(disp=False)
        print(f"  statsmodels ZIP params: {m.params.round(3)}")
    except Exception as ex:
        print(f"  (statsmodels ZIP unavailable: {ex})")
