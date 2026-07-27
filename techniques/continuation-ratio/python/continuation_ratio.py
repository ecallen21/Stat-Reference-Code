"""Continuation-ratio model for ordinal outcomes (Reference §8.9).

For an ordinal outcome Y in {1, 2, ..., K}, the continuation-ratio (CR) model
decomposes the ordered response into K - 1 SEQUENTIAL binary transitions:

    logit P(Y > k | Y >= k, X)  =  alpha_k + X * beta_k     for k = 1, ..., K-1

Each transition is a binary logistic regression fitted on the subset who reached
at least stage k. Interpretation: "given you're still in play at stage k, what's
the odds of moving to a higher level?" -- natural for cancer progression,
educational attainment, disease severity.

Two variants:
    - Category-specific beta_k (full flexibility; fit K-1 separate logistics)
    - Common beta (proportional-CR; fit one logistic on stacked data with alpha_k dummies)

We implement BOTH -- category-specific by default, plus a common-beta version
via joint MLE, and a likelihood-ratio test comparing them.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _logistic_irls(X, y, max_iter=100, tol=1e-10):
    """Vanilla logistic-regression IRLS. y in {0, 1}."""
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
    # ll
    eta = np.clip(X @ beta, -30, 30)
    ll = float(np.sum(y * eta - np.log1p(np.exp(eta))))
    # cov
    W_diag = np.clip((1/(1+np.exp(-eta))) * (1 - 1/(1+np.exp(-eta))), 1e-12, None)
    XWX = X.T @ (W_diag[:, None] * X)
    try:
        cov = np.linalg.pinv(XWX)
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
    return beta, cov, ll


def fit_continuation_ratio(X, y, K: int | None = None) -> dict:
    """Category-specific CR model: fit K - 1 logistic regressions.

    Parameters
    ----------
    X : n x p design matrix WITHOUT the intercept column (it is added per-transition).
    y : n-length ordinal response with values in {1, ..., K}.

    Returns per-transition coefficients (intercept alpha_k + slopes beta_k) and SEs.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
    n, p = X.shape
    if K is None:
        K = int(y.max())
    transitions = []
    total_ll = 0.0
    total_params = 0
    for k in range(1, K):
        mask = y >= k
        Xk = np.column_stack([np.ones(mask.sum()), X[mask]])
        yk = (y[mask] > k).astype(float)
        beta_k, cov_k, ll_k = _logistic_irls(Xk, yk)
        se_k = np.sqrt(np.clip(np.diag(cov_k), 0, None))
        transitions.append({
            "transition": f"P(Y > {k} | Y >= {k})",
            "n_at_risk": int(mask.sum()),
            "n_advanced": int(yk.sum()),
            "intercept": float(beta_k[0]),
            "coefficients": beta_k[1:].tolist(),
            "SE_intercept": float(se_k[0]),
            "SE_coefficients": se_k[1:].tolist(),
            "log_lik": ll_k,
        })
        total_ll += ll_k
        total_params += p + 1
    return {"K": K, "n_transitions": K - 1,
            "transitions": transitions,
            "total_log_lik_category_specific": total_ll,
            "n_params_category_specific": total_params,
            "method": "continuation ratio (category-specific betas)"}


def fit_continuation_ratio_common(X, y, K: int | None = None) -> dict:
    """Proportional CR model: single beta shared across transitions.

    Stacks the K - 1 binary datasets and fits ONE logistic with K - 1
    intercept dummies + shared slopes.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
    n, p = X.shape
    if K is None:
        K = int(y.max())
    # Build stacked design
    X_stack = []; y_stack = []; alpha_dummy = []
    for k in range(1, K):
        mask = y >= k
        X_stack.append(X[mask])
        y_stack.append((y[mask] > k).astype(float))
        # alpha_k dummy: 1 for this transition, 0 for others
        alpha_dummy.append(np.eye(K - 1)[k - 1][None, :].repeat(mask.sum(), axis=0))
    X_stacked = np.vstack(X_stack)
    y_stacked = np.concatenate(y_stack)
    alpha_mat = np.vstack(alpha_dummy)
    # Design = [alpha_dummies | X_stacked]
    design = np.column_stack([alpha_mat, X_stacked])
    beta, cov, ll = _logistic_irls(design, y_stacked)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return {"alpha": beta[:K - 1].tolist(),
            "SE_alpha": se[:K - 1].tolist(),
            "beta_common": beta[K - 1:].tolist(),
            "SE_beta_common": se[K - 1:].tolist(),
            "log_lik": ll,
            "n_params": len(beta),
            "method": "continuation ratio (proportional / common beta)"}


def lr_test_proportionality(X, y, K: int | None = None) -> dict:
    """LR test: proportional CR (common beta) vs. category-specific CR."""
    full = fit_continuation_ratio(X, y, K)
    red = fit_continuation_ratio_common(X, y, K)
    delta_ll = full["total_log_lik_category_specific"] - red["log_lik"]
    delta_df = full["n_params_category_specific"] - red["n_params"]
    return {"log_lik_common": red["log_lik"],
            "log_lik_category_specific": full["total_log_lik_category_specific"],
            "LR_statistic": 2 * delta_ll,
            "delta_df": delta_df,
            "p_value": float(stats.chi2.sf(2 * delta_ll, delta_df)),
            "interpretation": ("large p -> common-beta OK; small p -> "
                               "reject proportionality (need category-specific betas)")}


def library_versions(X, y):
    """VGAM (R) has vglm(sratio=...) but Python doesn't ship a canonical CR."""
    return {"note": "Python does not ship a canonical CR fitter; see the R file for vglm."}


if __name__ == "__main__":
    rng = np.random.default_rng(9)
    n = 500
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    X = np.column_stack([x1, x2])
    # Simulate a 4-level ordinal Y whose transition odds vary with X (common beta)
    linpred_pos = 0.5 + 0.7 * x1 - 0.4 * x2
    y = np.ones(n, dtype=int)
    for k in range(1, 4):
        alpha_k = -0.5 - 0.3 * (k - 1)
        p_advance = 1 / (1 + np.exp(-(alpha_k + linpred_pos)))
        advance = rng.random(n) < p_advance
        y = np.where((y == k) & advance, k + 1, y)

    print("=== Category-specific CR (K = 4, so 3 transitions) ===")
    fit = fit_continuation_ratio(X, y, K=4)
    for t in fit["transitions"]:
        print(f"  {t['transition']:22s}: alpha={t['intercept']:+.3f}, "
              f"beta={t['coefficients']}, n_at_risk={t['n_at_risk']}")
    print(f"  total log-lik = {fit['total_log_lik_category_specific']:.3f}, "
          f"params = {fit['n_params_category_specific']}")

    print("\n=== Proportional CR (common beta) ===")
    red = fit_continuation_ratio_common(X, y, K=4)
    print(f"  alpha_k = {[f'{a:+.3f}' for a in red['alpha']]}")
    print(f"  common beta = {red['beta_common']}")
    print(f"  log-lik = {red['log_lik']:.3f}, params = {red['n_params']}")

    print("\n=== LR test: proportional vs. category-specific ===")
    for k, v in lr_test_proportionality(X, y, K=4).items():
        print(f"  {k:30s}: {v}")
