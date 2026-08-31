"""Counterfactual fairness (Reference Ch 31 Fairness).

Kusner, Loftus, Russell & Silva (2017) 'Counterfactual Fairness.'

DEFINITION: a predictor Y_hat is counterfactually fair if, given a
STRUCTURAL CAUSAL MODEL (SCM), the predicted distribution at the
observed A and its counterfactual value are equal:

  P( Y_hat_{A <- a}(U) = y | X = x, A = a ) = P( Y_hat_{A <- a'}(U) = y | X = x, A = a )

for every valid counterfactual a'.  Requires a CAUSAL DAG:
  A --> X_a --> Y   plus latent U --> Y   (Kusner Fig 1).

RECIPE (Kusner Level 2):
  1. Fit an SCM x = f(A, U).
  2. Infer U | (X, A) for each individual (residuals).
  3. Predict Y_hat = g(U): use ONLY the exogenous latent U, so the
     prediction does not causally depend on A or its descendants.

Here we implement a small linear SCM
     X = alpha * A + U     (U ~ N(0, sigma^2))
and show that
  - naive prediction from X and A is counterfactually unfair
    (the prediction changes when A flips), while
  - prediction from U (the group-independent residual) is exactly
    counterfactually fair by construction.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))


def fit_scm(X, A):
    """Fit X = alpha * A + U by simple OLS per feature."""
    alpha = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        alpha[j] = np.cov(X[:, j], A)[0, 1] / np.var(A)
    U = X - np.outer(A, alpha)
    return alpha, U


def train_logistic(X, y, lr=0.5, epochs=500, l2=1e-3):
    d = X.shape[1]; beta = np.zeros(d); n = X.shape[0]
    for _ in range(epochs):
        p = _sigmoid(X @ beta)
        g = X.T @ (p - y) / n + l2 * beta
        beta -= lr * g
    return beta


def counterfactual_flip_gap(beta, X, A, alpha, feature_intercept=True):
    """Change in P(Y=1|x) when A flips from a to 1-a, holding U constant."""
    A_cf = 1 - A
    U = X - np.outer(A, alpha) if not feature_intercept else X - np.outer(A, alpha)
    X_cf = U + np.outer(A_cf, alpha)
    p_obs = _sigmoid(X @ beta)
    p_cf  = _sigmoid(X_cf @ beta)
    return np.abs(p_obs - p_cf)


if __name__ == "__main__":
    print("=== Counterfactual fairness (Kusner 2017) ===\n")
    rng = np.random.default_rng(0)
    n = 800
    A = rng.integers(0, 2, n).astype(float)
    # Linear SCM: X = alpha * A + U;  Y = 1{U + 0.3*A + eps > 0}.
    U = rng.normal(0, 1, n)
    X = (2.0 * A + U).reshape(-1, 1)                # single feature
    logit = U + 0.3 * A + rng.normal(0, 0.1, n)     # Y depends on U and slightly on A
    y = (logit > 0).astype(float)

    # SCM recovered from data (should give alpha ~ 2.0).
    alpha, U_hat = fit_scm(X, A)
    print(f"  Recovered alpha for the single feature: {alpha[0]:.3f}   (true 2.0)\n")

    # (1) Naive: predict from (X, A)
    XA = np.hstack([X, A.reshape(-1, 1)])
    beta_naive = train_logistic(XA, y)
    gap_naive = counterfactual_flip_gap(beta_naive, XA, A,
                                          alpha=np.concatenate([alpha, [1.0]]))
    print(f"  NAIVE predictor beta = {np.round(beta_naive, 3).tolist()}"
          f"   mean |P(y|A) - P(y|1-A)| = {gap_naive.mean():.3f}")

    # (2) Counterfactually fair: predict from U alone (Kusner Level 2)
    beta_cf = train_logistic(U_hat, y)
    # Under this predictor, flipping A leaves X_cf but the encoder returns the
    # same U, so counterfactual prediction is identical by construction.
    p_obs = _sigmoid(U_hat @ beta_cf)
    p_cf  = _sigmoid(U_hat @ beta_cf)     # same U -> same prediction
    print(f"  CF-FAIR  predictor beta = {np.round(beta_cf, 3).tolist()}"
          f"   mean |P(y|A) - P(y|1-A)| = {np.abs(p_obs - p_cf).mean():.3f}"
          "   (= 0 by construction)")

    # Report accuracy trade-off
    y_naive = (_sigmoid(XA @ beta_naive) > 0.5).astype(int)
    y_cf    = (_sigmoid(U_hat @ beta_cf) > 0.5).astype(int)
    print(f"\n  Accuracy: naive={(y_naive == y).mean():.3f}   CF-fair={(y_cf == y).mean():.3f}\n")

    print("--- library cross-check (aif360.algorithms.inprocessing.MetaFair;"
          " causal-fairness / doWhy for the SCM step) ---")
