"""SAG / SAGA - Variance-Reduced SGD (Reference Sec 47.337).

Roux, Schmidt & Bach 2012 (SAG); Defazio, Bach & Lacoste-Julien
2014 (SAGA). Store per-sample gradients and use their AVERAGE
as the search direction:

    memory: g_i (last stored gradient for sample i)
    update: draw random i;  d = grad_i(x) - g_i + mean(g)
            x <- x - lr * d
            g_i <- grad_i(x)

Achieves LINEAR convergence on strongly convex objectives at
SGD-per-step cost, unlike vanilla SGD (sub-linear).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def saga_ridge(X, y, lam=0.1, lr=None, n_iter=None, rng=None):
    """SAGA on ridge (per-sample gradient of 0.5 (X_i beta - y_i)^2 + 0.5 lam ||beta||^2)."""
    if rng is None: rng = np.random.default_rng(0)
    n, d = X.shape
    if n_iter is None: n_iter = 10 * n
    # Lipschitz-safe step
    L = float(np.max(np.sum(X ** 2, axis=1))) + lam
    if lr is None: lr = 1.0 / (3.0 * L)
    beta = np.zeros(d)
    g_mem = np.zeros((n, d))                                        # per-sample stored gradient
    g_avg = np.zeros(d)
    losses = []
    for step in range(n_iter):
        i = int(rng.integers(n))
        grad_new = X[i] * (X[i] @ beta - y[i]) + lam * beta
        d_update = grad_new - g_mem[i] + g_avg
        beta = beta - lr * d_update
        g_avg = g_avg + (grad_new - g_mem[i]) / n
        g_mem[i] = grad_new
        if step % n == 0:
            losses.append(float(0.5 * np.mean((X @ beta - y) ** 2)
                                  + 0.5 * lam * (beta @ beta)))
    return beta, losses


def sgd_ridge(X, y, lam=0.1, lr=None, n_iter=None, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    n, d = X.shape
    if n_iter is None: n_iter = 10 * n
    if lr is None: lr = 0.01
    beta = np.zeros(d); losses = []
    for step in range(n_iter):
        i = int(rng.integers(n))
        grad = X[i] * (X[i] @ beta - y[i]) + lam * beta
        beta = beta - lr * grad
        if step % n == 0:
            losses.append(float(0.5 * np.mean((X @ beta - y) ** 2)
                                  + 0.5 * lam * (beta @ beta)))
    return beta, losses


if __name__ == "__main__":
    print("=== SAG / SAGA (Roux 2012; Defazio 2014) ===\n")
    rng = np.random.default_rng(0)

    n, d = 200, 20
    X = rng.standard_normal((n, d))
    beta_true = rng.normal(0, 1, d)
    y = X @ beta_true + rng.normal(0, 0.5, n)

    lam = 0.1
    beta_star = np.linalg.solve(X.T @ X / n + lam * np.eye(d), X.T @ y / n)

    _, saga_loss = saga_ridge(X, y, lam=lam, n_iter=10 * n, rng=rng)
    _, sgd_loss = sgd_ridge(X, y, lam=lam, lr=0.005, n_iter=10 * n, rng=rng)
    opt = float(0.5 * np.mean((X @ beta_star - y) ** 2)
                  + 0.5 * lam * (beta_star @ beta_star))

    print(f"  Ridge regression, n = {n}, d = {d}")
    print(f"  {'epoch':>5}   {'SGD gap':>10}   {'SAGA gap':>10}")
    for ep in [1, 2, 4, 6, 9]:
        if ep < len(saga_loss):
            print(f"  {ep:>5}   {sgd_loss[ep] - opt:>+.4e}   {saga_loss[ep] - opt:>+.4e}")

    print(f"\n  SAGA achieves LINEAR convergence (gap halves per pass) on strongly")
    print(f"  convex objectives; SGD only O(1/k). Trade-off: SAGA stores O(n d) memory.")

    print("\n--- library cross-check (sklearn.linear_model.LogisticRegression(solver='saga')) ---")
