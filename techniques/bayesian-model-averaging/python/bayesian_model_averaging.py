"""Bayesian Model Averaging (Reference §14.26).

Instead of picking one 'best' model and ignoring uncertainty about which
model is right, BMA weights predictions and parameter estimates by each
model's posterior probability:

    E[Delta | y]     = sum_k Pr(M_k | y) * E[Delta | y, M_k]
    Var[Delta | y]   = sum_k Pr(M_k | y) * Var[Delta | y, M_k]  +
                       sum_k Pr(M_k | y) (E[Delta | y, M_k] - E[Delta | y])^2

Model posterior via Bayes:
    Pr(M_k | y) = p(y | M_k) Pr(M_k) / sum_j p(y | M_j) Pr(M_j)

BIC approximation for the marginal likelihood:
    log p(y | M_k) ~ log p(y | theta_hat_k, M_k) - (d_k / 2) log n
    which gives:
    Pr(M_k | y) proportional to exp(-BIC_k / 2) Pr(M_k)

Posterior inclusion probability for variable j:
    PIP_j = sum_{k: j in M_k} Pr(M_k | y)

The demo below enumerates all 2^p subsets in a small regression and reports
BMA point estimates, credible intervals, and PIPs.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from itertools import combinations, chain    # stdlib: iterate all subset models

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _all_subsets(items):
    return list(chain.from_iterable(combinations(items, r) for r in range(len(items) + 1)))


def bic_bma_linear(X, y, prior_prob=None) -> dict:
    """Enumerate all subsets of columns of X, fit OLS, and report BIC-BMA."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    subsets = _all_subsets(range(p))
    n_models = len(subsets)
    if prior_prob is None:
        prior = np.ones(n_models) / n_models
    else:
        prior = np.asarray(prior_prob) / np.sum(prior_prob)
    log_ml = np.full(n_models, -np.inf)
    beta_hat = [np.zeros(p) for _ in subsets]
    beta_var = [np.zeros(p) for _ in subsets]
    for m, sub in enumerate(subsets):
        if len(sub) == 0:
            # Intercept-only model would need y_mean; assume no intercept for simplicity
            r2 = np.sum(y ** 2)
            sigma2 = r2 / n
            log_lik = -0.5 * n * (math.log(2 * math.pi) + math.log(sigma2) + 1)
            d = 1
        else:
            Xs = X[:, list(sub)]
            b, res, *_ = np.linalg.lstsq(Xs, y, rcond=None)
            resid = y - Xs @ b
            rss = float(resid @ resid)
            sigma2 = rss / n
            log_lik = -0.5 * n * (math.log(2 * math.pi) + math.log(sigma2) + 1)
            d = len(sub) + 1  # +1 for sigma^2
            beta_hat[m][list(sub)] = b
            beta_var[m][list(sub)] = sigma2 * np.diag(np.linalg.pinv(Xs.T @ Xs))
        BIC = -2 * log_lik + d * math.log(n)
        log_ml[m] = -BIC / 2 + math.log(prior[m] + 1e-300)
    log_ml -= log_ml.max()
    w = np.exp(log_ml); w /= w.sum()
    bma_beta = np.sum([w[m] * beta_hat[m] for m in range(n_models)], axis=0)
    bma_within_var = np.sum([w[m] * beta_var[m] for m in range(n_models)], axis=0)
    bma_between_var = np.sum([w[m] * (beta_hat[m] - bma_beta) ** 2 for m in range(n_models)], axis=0)
    bma_var = bma_within_var + bma_between_var
    pip = np.zeros(p)
    for m, sub in enumerate(subsets):
        for j in sub:
            pip[j] += w[m]
    return {"bma_beta": bma_beta,
            "bma_se": np.sqrt(bma_var),
            "pip": pip,
            "model_weights": w,
            "top_models": [subsets[i] for i in np.argsort(-w)[:5]],
            "top_weights": w[np.argsort(-w)[:5]],
            "n_models": int(n_models),
            "method": "BIC-approximated Bayesian model averaging"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, p = 100, 5
    X = rng.normal(size=(n, p))
    beta_true = np.array([1.5, -1.0, 0.0, 0.6, 0.0])
    y = X @ beta_true + rng.normal(0, 1.0, n)

    print(f"=== BIC-BMA over all {2**p} subsets of {p} covariates ===")
    r = bic_bma_linear(X, y)
    print(f"\n  variable  BMA beta   BMA SE     PIP")
    for j in range(p):
        print(f"    x{j+1}       {r['bma_beta'][j]:6.3f}    {r['bma_se'][j]:.3f}    {r['pip'][j]:.3f}  (true {beta_true[j]})")

    print("\n=== Top 5 models by posterior probability ===")
    for sub, w in zip(r["top_models"], r["top_weights"]):
        cols = "+".join(f"x{j+1}" for j in sub) or "(null)"
        print(f"  {cols:20s} w = {w:.4f}")

    print("\n--- library cross-check (BMA in R) ---")
    print("  R: BMA::bicreg(X, y) or BAS::bas.lm(y ~ ., data = ...)")
