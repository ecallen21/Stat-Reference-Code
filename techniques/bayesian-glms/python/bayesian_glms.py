"""Bayesian generalized linear models (Reference §14.12, §14.13).

    y_i | X, beta ~ ExponentialFamily(link(x_i^T beta))
    beta          ~ Normal(0, s^2 I)          (weakly informative prior)

Logistic and Poisson GLMs have no conjugate prior in general, so we sample
from the posterior with random-walk Metropolis using multivariate proposals
scaled by the negative Hessian at the MLE (Laplace-approximation flavor).

Weakly informative default (Gelman 2008):
    beta_intercept ~ Normal(0, 10^2)
    beta_slope     ~ Normal(0, 2.5^2)   (on standardized covariates)

Posterior summaries:
    posterior mean, 95% credible intervals, and posterior predictive
    (probability that a new obs takes value 1 for logistic; predictive
    lambda for Poisson).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def _log_prior(beta, prior_sd):
    return -0.5 * np.sum((beta / prior_sd) ** 2)


def _mh_sample(log_post, beta0, prop_cov, n_iter, seed):
    rng = np.random.default_rng(seed)
    beta = np.array(beta0, dtype=float); d = len(beta)
    samples = np.empty((n_iter, d)); lp = log_post(beta); n_acc = 0
    L = np.linalg.cholesky(prop_cov)
    for t in range(n_iter):
        prop = beta + L @ rng.standard_normal(d)
        lp_prop = log_post(prop)
        if math.log(rng.uniform()) < lp_prop - lp:
            beta, lp = prop, lp_prop; n_acc += 1
        samples[t] = beta
    return samples, n_acc / n_iter


def bayesian_logistic(X, y, prior_sd=None, n_iter: int = 6000, seed: int = 0) -> dict:
    """Bayesian logistic regression via MH with Laplace-approx proposal."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    if prior_sd is None:
        prior_sd = np.array([10.0] + [2.5] * (p - 1))
    def neg_log_post(b):
        z = X @ b; ll = np.sum(y * z - np.logaddexp(0, z))
        return -(ll + _log_prior(b, prior_sd))
    res = minimize(neg_log_post, np.zeros(p), method="BFGS")
    beta_hat = res.x
    # Hessian from BFGS' inverse-Hessian estimate; use it as proposal cov
    prop_cov = res.hess_inv * (2.38 ** 2 / p) + 1e-6 * np.eye(p)
    samples, acc = _mh_sample(lambda b: -neg_log_post(b), beta_hat, prop_cov, n_iter, seed)
    burn = n_iter // 5; S = samples[burn:]
    return {"posterior_mean": S.mean(0), "posterior_sd": S.std(0),
            "credible_95": np.quantile(S, [0.025, 0.975], axis=0).T,
            "map_estimate": beta_hat, "acceptance_rate": float(acc),
            "n_iter_kept": int(len(S)), "prior_sd": prior_sd,
            "samples": S,
            "method": "Bayesian logistic regression (MH + Laplace proposal)"}


def bayesian_poisson(X, y, prior_sd=None, n_iter: int = 6000, seed: int = 0) -> dict:
    """Bayesian Poisson (log-link) GLM via MH."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    if prior_sd is None:
        prior_sd = np.array([10.0] + [2.5] * (p - 1))
    def neg_log_post(b):
        eta = X @ b; ll = np.sum(y * eta - np.exp(eta))
        return -(ll + _log_prior(b, prior_sd))
    res = minimize(neg_log_post, np.zeros(p), method="BFGS")
    beta_hat = res.x
    prop_cov = res.hess_inv * (2.38 ** 2 / p) + 1e-6 * np.eye(p)
    samples, acc = _mh_sample(lambda b: -neg_log_post(b), beta_hat, prop_cov, n_iter, seed)
    burn = n_iter // 5; S = samples[burn:]
    return {"posterior_mean": S.mean(0), "posterior_sd": S.std(0),
            "credible_95": np.quantile(S, [0.025, 0.975], axis=0).T,
            "map_estimate": beta_hat, "acceptance_rate": float(acc),
            "n_iter_kept": int(len(S)), "prior_sd": prior_sd,
            "samples": S,
            "method": "Bayesian Poisson regression (MH + Laplace proposal)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Bayesian logistic regression ===")
    n, p = 300, 3
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p))])
    beta_true = np.array([-0.5, 1.2, -0.8, 0.4])
    prob = 1 / (1 + np.exp(-X @ beta_true))
    y = (rng.uniform(0, 1, n) < prob).astype(float)
    r = bayesian_logistic(X, y, n_iter=6000, seed=0)
    for i, (m, sd, ci) in enumerate(zip(r["posterior_mean"], r["posterior_sd"], r["credible_95"])):
        print(f"  beta_{i}: mean = {m:6.3f}, SD = {sd:.3f}, 95% CrI = ({ci[0]:6.3f}, {ci[1]:6.3f}) true = {beta_true[i]}")
    print(f"  acceptance rate: {r['acceptance_rate']:.3f}")

    print("\n=== Bayesian Poisson regression ===")
    beta_true = np.array([0.5, 0.7, -0.4])
    X = np.column_stack([np.ones(300), rng.normal(size=(300, 2))])
    lam = np.exp(X @ beta_true)
    y = rng.poisson(lam)
    r = bayesian_poisson(X, y, n_iter=6000, seed=0)
    for i, (m, sd, ci) in enumerate(zip(r["posterior_mean"], r["posterior_sd"], r["credible_95"])):
        print(f"  beta_{i}: mean = {m:6.3f}, SD = {sd:.3f}, 95% CrI = ({ci[0]:6.3f}, {ci[1]:6.3f}) true = {beta_true[i]}")
    print(f"  acceptance rate: {r['acceptance_rate']:.3f}")

    print("\n--- cross-check: statsmodels frequentist MLE ---")
    try:
        import statsmodels.api as sm
        m = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        print(f"  frequentist Poisson MLE: {m.params.round(3)}")
    except Exception as ex:
        print(f"  (statsmodels not available: {ex})")
