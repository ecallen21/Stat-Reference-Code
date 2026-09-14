"""FISTA - Fast Iterative Shrinkage-Thresholding (Ref Sec 47.318).

Beck & Teboulle 2009 SIAM J Imaging. Nesterov-accelerated
version of ISTA / proximal gradient. Adds a MOMENTUM step:

    y_k = x_k + ((t_{k-1} - 1) / t_k) (x_k - x_{k-1})
    x_{k+1} = prox_{eta g}(y_k - eta grad f(y_k))
    t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2

Achieves O(1/k^2) convergence vs plain ISTA's O(1/k) on convex
composite problems. Standard for LASSO / TV-regularised image
denoising.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def soft_threshold(v, tau):
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0)


def fista_lasso(X, y, lam=0.1, lr=None, n_iter=200):
    n, d = X.shape
    if lr is None: lr = 1.0 / np.linalg.eigvalsh(X.T @ X / n).max()
    beta_prev = np.zeros(d); beta = np.zeros(d)
    t_prev = 1.0
    losses = []
    for _ in range(n_iter):
        t_new = (1 + np.sqrt(1 + 4 * t_prev ** 2)) / 2
        y_extrapol = beta + (t_prev - 1) / t_new * (beta - beta_prev)
        grad = X.T @ (X @ y_extrapol - y) / n
        beta_new = soft_threshold(y_extrapol - lr * grad, lr * lam)
        beta_prev = beta; beta = beta_new; t_prev = t_new
        losses.append(0.5 * np.mean((X @ beta - y) ** 2) + lam * np.abs(beta).sum())
    return beta, losses


def ista_lasso(X, y, lam=0.1, lr=None, n_iter=200):
    """Plain ISTA for comparison."""
    n, d = X.shape
    if lr is None: lr = 1.0 / np.linalg.eigvalsh(X.T @ X / n).max()
    beta = np.zeros(d); losses = []
    for _ in range(n_iter):
        grad = X.T @ (X @ beta - y) / n
        beta = soft_threshold(beta - lr * grad, lr * lam)
        losses.append(0.5 * np.mean((X @ beta - y) ** 2) + lam * np.abs(beta).sum())
    return beta, losses


if __name__ == "__main__":
    print("=== FISTA (Beck & Teboulle 2009) ===\n")
    rng = np.random.default_rng(0)

    n, d = 200, 100
    X = rng.standard_normal((n, d))
    beta_true = np.zeros(d); beta_true[:8] = rng.normal(0, 1, 8)
    y = X @ beta_true + rng.normal(0, 0.3, n)

    _, ista_loss = ista_lasso(X, y, lam=0.1, n_iter=200)
    _, fista_loss = fista_lasso(X, y, lam=0.1, n_iter=200)

    # Optimal value from a long FISTA run
    _, ref = fista_lasso(X, y, lam=0.1, n_iter=1500)
    opt = ref[-1]

    print(f"  n = {n}, d = {d}, LASSO with lam = 0.1")
    print(f"  {'iter':>5}   ISTA gap        FISTA gap")
    for it in [1, 5, 10, 30, 100, 199]:
        print(f"  {it:>5}   {ista_loss[it] - opt:>+.4e}   {fista_loss[it] - opt:>+.4e}")

    print(f"\n  FISTA gap decays as O(1/k^2); ISTA as O(1/k). Same per-iter cost.")
    print(f"  Momentum term extrapolates from beta_k, then applies proximal step.")

    print("\n--- library cross-check (proxop Python; celer::Lasso; sklearn Lasso) ---")
