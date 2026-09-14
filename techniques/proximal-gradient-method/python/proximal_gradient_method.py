"""Proximal Gradient Method (Reference Sec 47.317).

Combettes & Wajs 2005; Beck & Teboulle 2009. Minimise
composite objective f(x) + g(x) where f is smooth and g is a
possibly non-smooth "simple" regulariser:

    x_{k+1} = prox_{eta g}(x_k - eta grad f(x_k))
    prox_{eta g}(v) = argmin_z (0.5 ||z - v||^2 + eta g(z))

For g(x) = lambda ||x||_1 the prox is SOFT-THRESHOLDING; the
whole scheme is then ISTA. Extends to a huge family of
regularisers with closed-form proxes.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def soft_threshold(v, threshold):
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0)


def proximal_gradient_lasso(X, y, lam=0.1, lr=None, n_iter=200):
    """LASSO via ISTA (proximal gradient with L1)."""
    n, d = X.shape
    if lr is None: lr = 1.0 / np.linalg.eigvalsh(X.T @ X / n).max()
    beta = np.zeros(d); losses = []
    for _ in range(n_iter):
        grad = X.T @ (X @ beta - y) / n
        beta = soft_threshold(beta - lr * grad, lr * lam)
        losses.append(0.5 * np.mean((X @ beta - y) ** 2) + lam * np.abs(beta).sum())
    return beta, losses


if __name__ == "__main__":
    print("=== Proximal Gradient Method (Beck & Teboulle 2009) ===\n")
    rng = np.random.default_rng(0)

    n, d = 200, 50
    X = rng.standard_normal((n, d))
    beta_true = np.zeros(d); beta_true[:5] = [3, -2, 1.5, -0.5, 2]
    y = X @ beta_true + rng.normal(0, 0.5, n)

    beta, losses = proximal_gradient_lasso(X, y, lam=0.3, n_iter=300)
    nonzeros = (np.abs(beta) > 1e-3).sum()
    print(f"  n = {n}, d = {d}, 5 true nonzero coefficients")
    print(f"  Recovered nonzeros: {nonzeros}")
    print(f"  Top 5 estimated coefs: {np.round(np.sort(np.abs(beta))[::-1][:5], 3).tolist()}")
    print(f"  True top 5:          {[3.0, 2.0, 2.0, 1.5, 0.5]}\n")

    print(f"  Loss trajectory:")
    for it in [0, 5, 20, 50, 150, 299]:
        print(f"    iter {it:>4}   objective = {losses[it]:.4f}")

    print(f"\n  Proximal gradient handles L1, L2, box constraints, group-lasso, TV")
    print(f"  regularisers with closed-form or near-closed prox operators.")

    print("\n--- library cross-check (proxop Python; celer::Lasso; sklearn.linear_model.Lasso) ---")
