"""Bayesian linear regression (Reference §14.10, §14.11).

    y | X, beta, sigma^2 ~ Normal(X beta, sigma^2 I)

Normal-Inverse-Gamma conjugate prior:
    beta | sigma^2  ~ Normal(m_0, sigma^2 V_0)
    sigma^2         ~ InvGamma(a_0, b_0)

Joint posterior is Normal-Inverse-Gamma (Zellner 1971):
    V_n^-1 = V_0^-1 + X^T X
    m_n    = V_n (V_0^-1 m_0 + X^T y)
    a_n    = a_0 + n / 2
    b_n    = b_0 + 0.5 (y^T y + m_0^T V_0^-1 m_0 - m_n^T V_n^-1 m_n)

Marginal posterior of beta is a multivariate t.

Two default priors used below:
    - g-prior (Zellner):  V_0 = g (X^T X)^-1     -> shrinks toward 0 by g/(g+1)
    - Ridge-like:         V_0 = tau^2 I           -> penalized MLE at (X^T X + I/tau^2)

The MAP estimate coincides with Ridge regression when V_0 = tau^2 I.  Bayes
adds full posterior uncertainty AND posterior predictive intervals.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def bayesian_lr(X, y, prior: str = "g", g: float = 100.0, tau2: float = 100.0,
                a_0: float = 0.001, b_0: float = 0.001) -> dict:
    """Normal-InvGamma conjugate posterior for Bayesian linear regression.

    X : n x p design matrix (INCLUDE an intercept column if desired).
    y : length-n outcome.
    prior='g' -> Zellner g-prior V_0 = g (X'X)^-1
    prior='ridge' -> V_0 = tau2 I
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    m_0 = np.zeros(p)
    if prior == "g":
        V_0 = g * np.linalg.pinv(X.T @ X)
    elif prior == "ridge":
        V_0 = tau2 * np.eye(p)
    else: raise ValueError("prior must be 'g' or 'ridge'")
    V_0_inv = np.linalg.pinv(V_0)
    V_n_inv = V_0_inv + X.T @ X
    V_n = np.linalg.pinv(V_n_inv)
    m_n = V_n @ (V_0_inv @ m_0 + X.T @ y)
    a_n = a_0 + n / 2
    b_n = b_0 + 0.5 * (y @ y + m_0 @ V_0_inv @ m_0 - m_n @ V_n_inv @ m_n)
    # Marginal beta posterior is multivariate t with 2 a_n df, scale (b_n / a_n) V_n
    sigma2_hat = b_n / a_n
    beta_cov = (b_n / (a_n - 1)) * V_n if a_n > 1 else float("nan") * V_n
    beta_se = np.sqrt(np.diag(beta_cov))
    df = 2 * a_n
    t_q = stats.t.ppf(0.975, df)
    ci = np.column_stack([m_n - t_q * beta_se, m_n + t_q * beta_se])
    return {"posterior_mean_beta": m_n,
            "posterior_cov_beta": beta_cov,
            "posterior_se_beta": beta_se,
            "credible_95_beta": ci,
            "posterior_shape_sigma2": float(a_n),
            "posterior_rate_sigma2": float(b_n),
            "posterior_mean_sigma2": float(b_n / (a_n - 1)) if a_n > 1 else float("nan"),
            "prior_type": prior, "n": int(n), "p": int(p),
            "method": "Bayesian linear regression (Normal-InverseGamma conjugate)"}


def posterior_predictive_lr(fit, X_new, n_draws: int = 2000, seed: int = 0) -> dict:
    """Draw from the posterior predictive p(y_new | y) at rows of X_new."""
    rng = np.random.default_rng(seed)
    X_new = np.asarray(X_new, dtype=float)
    m_n = fit["posterior_mean_beta"]; V_n = fit["posterior_cov_beta"] / (fit["posterior_rate_sigma2"] / (fit["posterior_shape_sigma2"] - 1)) if fit["posterior_shape_sigma2"] > 1 else None
    a_n = fit["posterior_shape_sigma2"]; b_n = fit["posterior_rate_sigma2"]
    y_pred = np.empty((n_draws, len(X_new)))
    for t in range(n_draws):
        sig2 = 1 / rng.gamma(a_n, 1 / b_n)
        # V_beta = sig2 * V_n  where V_n is prior/posterior normal-covariance (scale)
        V_beta = fit["posterior_cov_beta"] * sig2 / (b_n / (a_n - 1))
        beta = rng.multivariate_normal(m_n, V_beta)
        y_pred[t] = rng.normal(X_new @ beta, math.sqrt(sig2))
    return {"y_pred_draws": y_pred,
            "y_pred_mean": y_pred.mean(0),
            "y_pred_95": np.quantile(y_pred, [0.025, 0.975], axis=0).T}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, p = 100, 3
    X = rng.normal(size=(n, p)); X = np.column_stack([np.ones(n), X])
    beta_true = np.array([1.0, 2.0, -1.5, 0.5])
    y = X @ beta_true + rng.normal(0, 1.0, n)

    print("=== Bayesian LR with Zellner g-prior (g = 100) ===")
    r = bayesian_lr(X, y, prior="g", g=100.0)
    for i, (b, se, ci) in enumerate(zip(r["posterior_mean_beta"], r["posterior_se_beta"], r["credible_95_beta"])):
        print(f"  beta_{i}: mean={b:.3f}, SE={se:.3f}, 95% CrI=({ci[0]:.3f}, {ci[1]:.3f})  true={beta_true[i]}")
    print(f"  posterior mean sigma^2 = {r['posterior_mean_sigma2']:.3f}  (true 1.0)")

    print("\n=== Bayesian LR with ridge-like prior tau^2 = 100 ===")
    r = bayesian_lr(X, y, prior="ridge", tau2=100.0)
    print(f"  posterior mean beta:  {r['posterior_mean_beta'].round(3)}")

    print("\n=== Posterior predictive at 3 new points ===")
    X_new = np.array([[1, 0, 0, 0], [1, 1, 1, 1], [1, -1, -1, -1]], dtype=float)
    pp = posterior_predictive_lr(bayesian_lr(X, y, prior="g", g=100.0), X_new)
    for i, (m, lo, hi) in enumerate(zip(pp["y_pred_mean"], pp["y_pred_95"][:, 0], pp["y_pred_95"][:, 1])):
        print(f"  x = {X_new[i, 1:]}:  y_pred mean = {m:.3f}, 95% PI = ({lo:.3f}, {hi:.3f})")

    print("\n--- cross-check: MAP with g-prior vs ridge coefficient with sensible tuning ---")
    from numpy.linalg import lstsq
    beta_ols, *_ = lstsq(X, y, rcond=None)
    print(f"  OLS coefs: {beta_ols.round(3)}")
    print(f"  posterior mean converges to OLS as g -> infinity (g=100 shrinks slightly)")
