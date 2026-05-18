"""Multinomial logistic regression (Reference §7.4).

Outcome Y in {1, ..., K} (unordered categories). Pick a reference (say K) and
model the K-1 log-odds against it:

    log P(Y = k | x) / P(Y = K | x) = x' beta_k     for k = 1, ..., K-1
    P(Y = k | x) = exp(x' beta_k) / (1 + sum_{j != K} exp(x' beta_j))

Equivalently, the softmax of (x' beta_1, ..., x' beta_{K-1}, 0).

We fit by maximum likelihood via BFGS (scipy.optimize). Beta is a (K-1) x p
matrix; we flatten for optimization. Output: per-class beta, SE, p, OR.

For ORDERED outcomes, prefer ``techniques/ordinal-logistic`` -- the proportional-
odds model uses K - 1 + p parameters; multinomial uses (K - 1) * p (no shared
slope structure).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import optimize, stats    # optimize: BFGS / Brent solvers;  stats: distributions / tests


def _softmax(logits):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def _neg_log_lik(theta, X, y, K):
    n, p = X.shape
    B = theta.reshape(K - 1, p)
    # Logits for K-1 non-reference classes; last column 0 (reference).
    eta = X @ B.T                                      # n x (K-1)
    eta_full = np.column_stack([eta, np.zeros(n)])     # n x K
    P = _softmax(eta_full)
    # log-lik:  sum_i log P[i, y_i - 1]
    return -float(np.sum(np.log(np.clip(P[np.arange(n), y - 1], 1e-15, None))))


def fit(X, y) -> dict:
    """Multinomial logistic regression via BFGS MLE; reference class = K (max label)."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
    n, p = X.shape; K = int(y.max())
    theta0 = np.zeros((K - 1) * p)
    res = optimize.minimize(_neg_log_lik, theta0, args=(X, y, K),
                            method="BFGS", options={"gtol": 1e-7, "disp": False})
    B = res.x.reshape(K - 1, p)
    se = np.sqrt(np.diag(res.hess_inv)).reshape(K - 1, p)
    z = B / se; p_vals = 2 * stats.norm.sf(np.abs(z))
    or_ = np.exp(B); zc = stats.norm.ppf(0.975)
    or_lo = np.exp(B - zc * se); or_hi = np.exp(B + zc * se)
    # Predicted classes for accuracy
    eta_full = np.column_stack([X @ B.T, np.zeros(n)])
    P = _softmax(eta_full); y_hat = np.argmax(P, axis=1) + 1
    return {"reference_class": K,
            "beta": B.tolist(), "se": se.tolist(),
            "z": z.tolist(), "p_values": p_vals.tolist(),
            "odds_ratio_vs_ref": or_.tolist(),
            "or_ci_lower": or_lo.tolist(), "or_ci_upper": or_hi.tolist(),
            "log_likelihood": float(-res.fun),
            "accuracy": float(np.mean(y_hat == y)),
            "converged": bool(res.success), "iterations": int(res.nit)}


def library_versions(X, y):
    try:
        import statsmodels.api as sm
        mod = sm.MNLogit(y, sm.add_constant(X, has_constant="add")); res = mod.fit(disp=False)
        params = res.params
        return {"statsmodels MNLogit params":
                params.values.tolist() if hasattr(params, "values") else params.tolist()}
    except Exception as exc:
        return {"statsmodels": f"error: {exc}"}


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    n = 400
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    X = np.column_stack([np.ones(n), x1, x2])
    # Generate 3-class with known logit structure (reference class 3)
    B_true = np.array([[+0.5, +1.0, -0.5],     # class 1 vs 3
                       [-0.3, -0.6, +0.8]])    # class 2 vs 3
    eta = X @ B_true.T
    eta_full = np.column_stack([eta, np.zeros(n)])
    p_full = _softmax(eta_full)
    y = np.array([rng.choice(3, p=p_full[i]) + 1 for i in range(n)])

    res = fit(X, y)
    names = ["intercept", "x1", "x2"]
    for cls in range(res["reference_class"] - 1):
        print(f"=== Class {cls + 1} vs reference (class {res['reference_class']}) ===")
        for j, nm in enumerate(names):
            print(f"  {nm:10s} beta = {res['beta'][cls][j]:+.4f}  "
                  f"SE = {res['se'][cls][j]:.4f}  "
                  f"z = {res['z'][cls][j]:+.2f}  p = {res['p_values'][cls][j]:.4g}  "
                  f"OR = {res['odds_ratio_vs_ref'][cls][j]:.3f}")
    print(f"\nlog-likelihood = {res['log_likelihood']:.4f}")
    print(f"accuracy       = {res['accuracy']:.4f}")
    print("\n--- library (statsmodels MNLogit) ---")
    for k, v in library_versions(np.column_stack([x1, x2]), y).items():
        print(f"  {k}: {v}")
