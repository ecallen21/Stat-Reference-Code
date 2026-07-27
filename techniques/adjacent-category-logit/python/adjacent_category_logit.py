"""Adjacent-category logit model for ordinal outcomes (Reference §8.10).

For an ordinal outcome Y in {1, ..., K}, the adjacent-category (AC) logit
model contrasts each level with the NEXT level up:

    log P(Y = k) / P(Y = k + 1)  =  alpha_k - beta * X       for k = 1, ..., K - 1

Compare:
  - cumulative logit (proportional-odds): compares "above k" vs "at or below k"
  - continuation ratio: compares "above k" vs "at k, given >= k"
  - adjacent-category:  compares "at k" vs "at k+1" directly

The AC model is mathematically equivalent to a MULTINOMIAL logit with a
constraint that the coefficient vector for category k relative to a baseline
scales linearly in (K - k). That constraint makes AC a genuine ordinal model
that respects the ordering while remaining a proper multinomial.

Common-beta form (fit here):
    log P(Y = k) / P(Y = K)  =  gamma_k - (K - k) * X * beta
so beta shows up multiplied by -(K - k). Sign convention: **positive beta
means positive X shifts probability TOWARD the higher categories** -- the
standard direction, matching a rightward shift of the ordinal outcome.

Category-specific-beta form (also fit): each transition gets its own beta_k,
identical to K - 1 independent binary logistics on the pair {k, k+1}.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import optimize, stats    # optimize: BFGS;  stats: distributions/tests


def _softmax_ll(theta, X, y, K):
    """Multinomial-logit log-likelihood with the AC (linear-scaling) constraint.

    theta layout: first (K - 1) entries are gamma_1..gamma_{K-1}; then p entries
    are the common beta. We define P(Y = k) via
        eta_k = gamma_k - (K - k) * X * beta           for k = 1..K-1
        eta_K = 0                                      (baseline)
    Sign: positive beta -> higher category more likely (standard convention).
    """
    n, p = X.shape
    gamma = theta[:K - 1]
    beta = theta[K - 1:K - 1 + p]
    scale = np.arange(K - 1, 0, -1)                                  # K-1, K-2, ..., 1
    eta = gamma[None, :] - scale[None, :] * (X @ beta)[:, None]
    eta = np.column_stack([eta, np.zeros(n)])                        # baseline K
    # logsumexp
    mx = eta.max(axis=1, keepdims=True)
    lse = mx + np.log(np.exp(eta - mx).sum(axis=1, keepdims=True))
    # y is in {1..K}; convert to zero-based index
    idx = (y - 1)
    ll = float((eta[np.arange(n), idx] - lse.squeeze()).sum())
    return -ll


def fit_adjacent_category_common(X, y, K: int | None = None) -> dict:
    """Common-beta AC model via joint MLE (BFGS on the multinomial log-likelihood)."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
    n, p = X.shape
    if K is None:
        K = int(y.max())
    # good starting values from category-specific fits
    theta0 = np.zeros((K - 1) + p)
    res = optimize.minimize(_softmax_ll, theta0, args=(X, y, K),
                             method="BFGS", options={"gtol": 1e-8})
    gamma = res.x[:K - 1]
    beta = res.x[K - 1:]
    # SEs via Hessian inverse (BFGS provides an approximate inverse Hessian)
    cov = res.hess_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ll = -res.fun
    return {"gamma": gamma.tolist(),
            "beta_common": beta.tolist(),
            "SE_gamma": se[:K - 1].tolist(),
            "SE_beta_common": se[K - 1:].tolist(),
            "log_lik": float(ll),
            "K": K, "n_params": len(res.x),
            "method": "adjacent-category logit (common beta, joint MLE)"}


def _logistic_irls(X, y, max_iter=100, tol=1e-10):
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-12, None)
        z = eta + (y - mu) / w
        sw = np.sqrt(w); Xw = X * sw[:, None]; zw = z * sw
        beta_new, *_ = np.linalg.lstsq(Xw, zw, rcond=None)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new; break
        beta = beta_new
    eta = np.clip(X @ beta, -30, 30)
    ll = float(np.sum(y * eta - np.log1p(np.exp(eta))))
    return beta, ll


def fit_adjacent_category_pairs(X, y, K: int | None = None) -> dict:
    """K-1 independent binary logistics on adjacent-category subsets."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
    if K is None:
        K = int(y.max())
    pairs = []
    total_ll = 0.0
    total_params = 0
    for k in range(1, K):
        mask = (y == k) | (y == k + 1)
        Xk = np.column_stack([np.ones(mask.sum()), X[mask]])
        yk = (y[mask] == k + 1).astype(float)          # 1 if in the "next" category
        beta, ll = _logistic_irls(Xk, yk)
        pairs.append({"pair": f"{k} vs {k + 1}",
                       "n": int(mask.sum()),
                       "intercept": float(beta[0]),
                       "coefficients_beta_k": beta[1:].tolist(),
                       "log_lik": ll})
        total_ll += ll; total_params += len(beta)
    return {"pairs": pairs,
            "total_log_lik": total_ll,
            "n_params": total_params,
            "method": "adjacent-category logit (K-1 pairwise binary logistics)"}


def library_versions(X, y):
    """statsmodels doesn't provide AC directly; VGAM's acat is standard in R."""
    return {"note": "no direct Python library; R's VGAM::acat is the standard fit"}


if __name__ == "__main__":
    rng = np.random.default_rng(12)
    n = 600
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    X = np.column_stack([x1, x2])
    # Simulate K=4 outcome with common beta = (0.5, -0.3) and gammas (-1, -1.5, -2)
    beta = np.array([0.5, -0.3])
    gamma = np.array([-1.0, -1.5, -2.0])
    K = 4
    scale = np.arange(K - 1, 0, -1)
    eta = gamma[None, :] - scale[None, :] * (X @ beta)[:, None]     # sign matches fitter
    eta = np.column_stack([eta, np.zeros(n)])
    exp_eta = np.exp(eta - eta.max(axis=1, keepdims=True))
    probs = exp_eta / exp_eta.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(K, p=p) + 1 for p in probs])

    print("=== Common-beta AC (target: beta = [0.5, -0.3]) ===")
    fit = fit_adjacent_category_common(X, y, K=4)
    print(f"  beta_common = {fit['beta_common']}")
    print(f"  SE_beta = {fit['SE_beta_common']}")
    print(f"  gamma = {fit['gamma']}")
    print(f"  log-lik = {fit['log_lik']:.3f}")

    print("\n=== Pairwise category-specific AC ===")
    pf = fit_adjacent_category_pairs(X, y, K=4)
    for pair in pf["pairs"]:
        print(f"  {pair['pair']}: intercept={pair['intercept']:+.3f}, "
              f"beta_k={pair['coefficients_beta_k']}, n={pair['n']}")
