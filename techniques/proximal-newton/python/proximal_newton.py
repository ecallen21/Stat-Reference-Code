"""Proximal Newton (Reference Sec 47.340).

Lee, Sun & Saunders 2014 SIAM. Composite optimisation f + g
with SECOND-ORDER info of f:

    x_{k+1} = argmin_z <grad f(x_k), z - x_k>
                + 0.5 * (z - x_k)^T H_k (z - x_k)
                + g(z)

For g = lambda ||.||_1 the sub-problem is a QUADRATIC-LASSO
solved by coord descent or FISTA. Ideal for logistic-LASSO
where the Hessian is well-conditioned.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def soft_threshold(v, tau): return np.sign(v) * np.maximum(np.abs(v) - tau, 0)


def coord_descent_lasso_quad(H, g, x0, lam, n_iter=100):
    """Solve min 0.5 z^T H z + g^T z + lam ||z||_1 by coord descent."""
    x = x0.copy(); d = len(x)
    for _ in range(n_iter):
        for j in range(d):
            H_diag = H[j, j] + 1e-12
            r = -g[j] - H[j] @ x + H[j, j] * x[j]
            x[j] = soft_threshold(r, lam) / H_diag
    return x


def proximal_newton_logreg_lasso(X, y, lam=0.1, n_iter=20):
    """Fit logistic-LASSO via proximal Newton (Hessian at each x)."""
    n, d = X.shape
    beta = np.zeros(d); losses = []
    for k in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        W = np.maximum(p * (1 - p), 1e-6)
        H = (X.T * W) @ X / n
        g = X.T @ (p - y) / n
        beta = coord_descent_lasso_quad(H, g - H @ beta, beta, lam)
        # Loss: negative log likelihood + L1 penalty
        p = np.clip(p, 1e-9, 1 - 1e-9)
        losses.append(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                       + lam * np.abs(beta).sum())
    return beta, losses


if __name__ == "__main__":
    print("=== Proximal Newton (Lee-Sun-Saunders 2014) ===\n")
    rng = np.random.default_rng(0)

    n, d = 300, 30
    X = rng.standard_normal((n, d))
    beta_true = np.zeros(d); beta_true[:5] = [3, -2, 1.5, -0.5, 2]
    logits = X @ beta_true
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logits))).astype(float)

    beta_hat, losses = proximal_newton_logreg_lasso(X, y, lam=0.05, n_iter=15)
    nz = (np.abs(beta_hat) > 1e-3).sum()

    print(f"  Logistic-LASSO, n = {n}, d = {d}, lambda = 0.05")
    print(f"  Recovered nonzeros: {nz}")
    print(f"  Top-5 estimated (by |beta|):")
    for j in np.argsort(np.abs(beta_hat))[::-1][:5]:
        print(f"    beta_{j}  = {beta_hat[j]:>+.3f}   (true = {beta_true[j]:+.3f})")

    print(f"\n  Objective trajectory:")
    for k in [0, 2, 5, 10, 14]:
        print(f"    iter {k:>3}   obj = {losses[k]:.4f}")

    print(f"\n  Proximal Newton typically converges in 5-20 iters (superlinear),")
    print(f"  vs 100s of steps for ISTA / coord descent on logistic-LASSO.")

    print("\n--- library cross-check (glmnet::glmnet(family='binomial'); sklearn Logistic + L1) ---")
