"""NOTEARS: continuous DAG learning (Reference Sec 47.55).

Zheng, Aragam, Ravikumar & Xing 2018 'DAGs with NO TEARS:
Continuous optimization for structure learning', NeurIPS.
Replaces the combinatorial acyclicity constraint of DAG structure
search with a SMOOTH, DIFFERENTIABLE equivalent:

    h(W) = tr( expm(W * W) ) - d = 0     <=>   W represents a DAG.

Minimises  0.5 ||X - X W||^2 / n + rho/2 * h(W)^2 + alpha * h(W)
                        + lambda ||W||_1
by augmented Lagrangian + coordinate / L-BFGS-B.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.linalg import expm    # matrix exponential (small-d)
from scipy.optimize import minimize    # L-BFGS-B


def notears_linear(X, lam=0.01, max_outer=30, rho_max=1e12, tol=1e-8):
    """NOTEARS for linear-Gaussian SEM. Returns weighted adjacency W (d x d)."""
    n, d = X.shape
    def _loss(W_flat):
        W = W_flat.reshape(d, d)
        R = X - X @ W
        loss = 0.5 * (R * R).sum() / n
        G_loss = -X.T @ R / n
        return loss, G_loss

    def _h(W):
        E = expm(W * W)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    def _augLagr(W_flat, rho, alpha, lam):
        W = W_flat.reshape(d, d)
        loss, G_loss = _loss(W_flat)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lam * np.abs(W).sum()
        G = G_loss + (rho * h + alpha) * G_h + lam * np.sign(W)
        return obj, G.ravel()

    W = np.zeros(d * d)
    rho, alpha, h = 1.0, 0.0, np.inf
    for outer in range(max_outer):
        while rho < rho_max:
            res = minimize(_augLagr, W, args=(rho, alpha, lam), jac=True,
                            method="L-BFGS-B", bounds=[(-3, 3)] * (d * d),
                            options={"maxiter": 100})
            W_new = res.x
            h_new, _ = _h(W_new.reshape(d, d))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        W = W_new
        h = h_new
        alpha += rho * h
        if h <= tol or rho >= rho_max:
            break
    return W.reshape(d, d)


if __name__ == "__main__":
    print("=== NOTEARS DAG learning (Zheng-Aragam-Ravikumar-Xing 2018) ===\n")
    rng = np.random.default_rng(0)
    d = 4; n = 800
    # Truth: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    W_true = np.zeros((d, d))
    W_true[0, 1] = 1.5; W_true[0, 2] = -1.0; W_true[1, 3] = 0.7; W_true[2, 3] = 0.9
    # Sample from linear-Gaussian SEM: X = X W + eps
    eps = 0.3 * rng.normal(size=(n, d))
    X = np.linalg.solve((np.eye(d) - W_true).T, eps.T).T

    W_hat = notears_linear(X, lam=0.02, max_outer=30)
    W_thr = np.where(np.abs(W_hat) > 0.15, W_hat, 0)
    print("  True W:")
    print(np.round(W_true, 2))
    print("\n  Estimated W (thresholded at 0.15):")
    print(np.round(W_thr, 2))
    tp = int(((W_true != 0) & (W_thr != 0)).sum())
    fp = int(((W_true == 0) & (W_thr != 0)).sum())
    fn = int(((W_true != 0) & (W_thr == 0)).sum())
    print(f"\n  Structural Hamming: TP={tp}  FP={fp}  FN={fn}")

    print("\n--- library cross-check (notearsC R; causalnex / dagma Python) ---")
