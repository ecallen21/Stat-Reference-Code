"""Laplace approximation of a posterior (Reference §14.29; Tierney-Kadane 1986).

Approximate an intractable posterior p(theta | y) by a Gaussian centered at
the MAP with covariance equal to the inverse negative Hessian of the log
posterior at the MAP:

    log p(theta | y) ~ log p(theta_MAP | y) - 0.5 (theta - theta_MAP)^T H (theta - theta_MAP)
    p(theta | y) ~ N(theta_MAP, H^-1)

Marginal likelihood (evidence) approximation:
    log p(y) ~ log p(y | theta_MAP) + log p(theta_MAP) + (d / 2) log(2 pi) - 0.5 log |H|

Applications
    - Fast Bayesian inference when MCMC is too expensive.
    - Building block of INLA (Integrated Nested Laplace Approximation, Rue
      et al. 2009): nested Laplace approximations for latent Gaussian
      models -- fast Bayes for GAMs, spatial models, hierarchical GLMs.
    - Approximate marginal likelihoods for BMA.

Limitations
    - Poor for MULTIMODAL posteriors (fits the mode you land in).
    - Poor for SKEWED posteriors near boundaries.
    - Good for LARGE-n regular models (posterior is Gaussian asymptotically).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def _num_hessian(f, x, eps: float = 1e-4):
    d = len(x); H = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            e_i = np.zeros(d); e_i[i] = eps
            e_j = np.zeros(d); e_j[j] = eps
            H[i, j] = (f(x + e_i + e_j) - f(x + e_i - e_j) - f(x - e_i + e_j) + f(x - e_i - e_j)) / (4 * eps ** 2)
    return H


def laplace_posterior(log_posterior, theta0, eps: float = 1e-4) -> dict:
    """Laplace approximation.

    log_posterior : callable, theta -> log p(theta | y) (unnormalized).
    theta0        : starting value near the MAP.
    """
    theta0 = np.atleast_1d(np.asarray(theta0, dtype=float))
    res = minimize(lambda t: -log_posterior(t), theta0, method="BFGS")
    theta_MAP = res.x
    # Negative Hessian of log posterior = Hessian of -log posterior
    H = _num_hessian(lambda t: -log_posterior(t), theta_MAP, eps=eps)
    cov = np.linalg.pinv(H)
    d = len(theta_MAP)
    logdet = np.linalg.slogdet(H)[1]
    log_evidence = float(log_posterior(theta_MAP) + 0.5 * d * math.log(2 * math.pi) - 0.5 * logdet)
    return {"MAP": theta_MAP, "cov": cov, "sd": np.sqrt(np.diag(cov)),
            "log_evidence_approx": log_evidence,
            "method": "Laplace approximation (Gaussian at MAP)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Beta-Binomial: Laplace vs analytic posterior ===")
    a0, b0 = 2, 2
    y, n = 8, 12
    def log_post_beta(theta):
        p = float(theta[0])
        if p <= 0 or p >= 1: return -1e10
        return (a0 - 1 + y) * math.log(p) + (b0 - 1 + n - y) * math.log(1 - p)
    r = laplace_posterior(log_post_beta, [0.5])
    a_post, b_post = a0 + y, b0 + n - y
    print(f"  Laplace MAP p   = {r['MAP'][0]:.3f}   (analytic Beta mode = {(a_post - 1) / (a_post + b_post - 2):.3f})")
    print(f"  Laplace SD  p   = {r['sd'][0]:.3f}   (analytic Beta SD    = {math.sqrt(a_post * b_post / ((a_post + b_post) ** 2 * (a_post + b_post + 1))):.3f})")
    print(f"  analytic Beta({a_post}, {b_post}) mean = {a_post / (a_post + b_post):.3f}")

    print("\n=== Bayesian logistic: Laplace approximation of the posterior ===")
    n = 200; x = rng.normal(size=n)
    beta_true = np.array([0.5, 1.5])
    prob = 1 / (1 + np.exp(-(beta_true[0] + beta_true[1] * x)))
    y_bin = (rng.uniform(size=n) < prob).astype(float)
    def log_post_logistic(theta):
        b0, b1 = theta
        z = b0 + b1 * x
        return np.sum(y_bin * z - np.logaddexp(0, z)) - 0.5 * (b0 ** 2 + b1 ** 2) / 100  # weak Normal prior
    r = laplace_posterior(log_post_logistic, [0.0, 0.0])
    print(f"  MAP (b0, b1)  = ({r['MAP'][0]:.3f}, {r['MAP'][1]:.3f})   (true 0.5, 1.5)")
    print(f"  SD  (b0, b1)  = ({r['sd'][0]:.3f}, {r['sd'][1]:.3f})")
    print(f"  log evidence approx = {r['log_evidence_approx']:.3f}")

    print("\n--- INLA / R-INLA is the production tool for latent-Gaussian models ---")
