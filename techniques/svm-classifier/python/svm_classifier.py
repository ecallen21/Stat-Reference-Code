"""Support Vector Machine classifier (Reference §26.9; Cortes-Vapnik 1995).

Binary classification: find the hyperplane w^T x + b that maximizes the
MARGIN 2 / ||w|| between the two classes.  For non-separable data, use
soft margin with slack variables and penalty C:

    minimize  0.5 ||w||^2 + C sum_i xi_i
    s.t.       y_i (w^T x_i + b) >= 1 - xi_i,   xi_i >= 0

Kernel trick: replace x^T x' with k(x, x') to allow nonlinear decision
boundaries without explicit feature mapping.

Common kernels
    Linear:      x^T x'
    Polynomial:  (gamma x^T x' + r)^d
    RBF / Gaussian: exp(-gamma ||x - x'||^2)

Solved via quadratic programming or sequential minimal optimization (SMO).
We use sklearn (which wraps LIBSVM) for the demo; a full SMO takes ~100
lines.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def linear_svm_pegasos(X, y, C: float = 1.0, n_iter: int = 5000, seed: int = 0) -> dict:
    """Linear soft-margin SVM via Pegasos (stochastic subgradient; Shalev-Shwartz 2007).

    y in {-1, +1}.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    lam = 1 / (C * n)                             # equivalent lambda for Pegasos
    rng = np.random.default_rng(seed)
    w = np.zeros(p); b = 0.0
    for t in range(1, n_iter + 1):
        i = int(rng.integers(0, n))
        eta_t = 1 / (lam * t)
        margin = y[i] * (X[i] @ w + b)
        if margin < 1:
            w = (1 - eta_t * lam) * w + eta_t * y[i] * X[i]
            b = b + eta_t * y[i]
        else:
            w = (1 - eta_t * lam) * w
    def predict(X_new):
        return np.sign(np.asarray(X_new, dtype=float) @ w + b)
    return {"w": w, "b": float(b), "C": float(C), "predict": predict,
            "method": "Linear SVM via Pegasos SGD"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Linear SVM (Pegasos) on linearly-separable 2-class data ===")
    X = np.vstack([rng.normal([0, 0], 0.7, (100, 2)), rng.normal([3, 3], 0.7, (100, 2))])
    y = np.concatenate([-np.ones(100), np.ones(100)])
    fit = linear_svm_pegasos(X, y, C=1.0, n_iter=8000)
    acc = float((fit["predict"](X) == y).mean())
    print(f"  w = {fit['w'].round(3)}, b = {fit['b']:.3f}")
    print(f"  training accuracy = {acc:.3f}")

    print("\n=== RBF SVM (via sklearn) on nonlinear 2-class ===")
    theta = rng.uniform(0, 2 * math.pi, 100); r = 1 + rng.normal(0, 0.15, 100)
    X_neg = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    theta = rng.uniform(0, 2 * math.pi, 100); r = 3 + rng.normal(0, 0.15, 100)
    X_pos = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    X = np.vstack([X_neg, X_pos]); y = np.concatenate([-np.ones(100), np.ones(100)])
    try:
        from sklearn.svm import SVC
        clf = SVC(kernel="rbf", C=1.0, gamma=0.5).fit(X, y)
        print(f"  sklearn RBF SVM accuracy = {clf.score(X, y):.3f}")
    except Exception as ex:
        print(f"  (sklearn unavailable: {ex})")
