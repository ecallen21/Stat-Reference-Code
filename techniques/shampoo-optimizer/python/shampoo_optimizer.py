"""Shampoo — Preconditioned Stochastic Tensor Optimization
(Gupta, Koren & Singer 2018).

For a 2D weight matrix W (r x c), maintain two preconditioners:

    L = sum_t g_t g_t.T   (r x r) — left/statistics
    R = sum_t g_t.T g_t   (c x c) — right/statistics

Update:
    W = W - lr * L^{-1/4} @ g @ R^{-1/4}

Kronecker-factored approximation to full-matrix Adagrad —
storage O(r^2 + c^2) instead of O((rc)^2). Increasingly used
in LLM training (e.g., Muon/PaLM optimiser families).
"""

import numpy as np    # arrays


def matrix_power(A, p, eps=1e-6):
    """Compute A^p via symmetric eigendecomposition."""
    A = 0.5 * (A + A.T) + eps * np.eye(A.shape[0])
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return V @ np.diag(w ** p) @ V.T


def shampoo_step(W, L, R, g, lr, beta):
    L = beta * L + (1 - beta) * g @ g.T
    R = beta * R + (1 - beta) * g.T @ g
    Lm = matrix_power(L, -0.25)
    Rm = matrix_power(R, -0.25)
    W = W - lr * Lm @ g @ Rm
    return W, L, R


def sgd_step(W, g, lr):
    return W - lr * g


def demo():
    print("=== Shampoo (Gupta-Koren-Singer 2018 ICML) ===")
    rng = np.random.default_rng(2026)
    n, d_in, d_out = 200, 20, 15
    X = rng.standard_normal((n, d_in))
    W_true = rng.standard_normal((d_in, d_out)) * 0.5
    Y = X @ W_true + rng.standard_normal((n, d_out)) * 0.2
    XtX_n = X.T @ X / n
    XtY_n = X.T @ Y / n

    def loss(W):
        return 0.5 * np.mean((X @ W - Y) ** 2)

    def grad(W):
        return XtX_n @ W - XtY_n

    # Shampoo
    W = np.zeros((d_in, d_out))
    L = np.zeros((d_in, d_in))
    R = np.zeros((d_out, d_out))
    for step in range(300):
        W, L, R = shampoo_step(W, L, R, grad(W), lr=0.05, beta=0.9)
    mse_shampoo = loss(W)

    # SGD baseline
    W2 = np.zeros((d_in, d_out))
    for step in range(300):
        W2 = sgd_step(W2, grad(W2), lr=0.05)
    mse_sgd = loss(W2)

    print(f"  Shampoo MSE = {mse_shampoo:.4f},  ‖W‖ = {np.linalg.norm(W):.3f}")
    print(f"  SGD     MSE = {mse_sgd:.4f},  ‖W‖ = {np.linalg.norm(W2):.3f}")
    print(f"  Truth   ‖W‖ = {np.linalg.norm(W_true):.3f}")

    print(f"\n  Preconditioner state Shampoo : {L.nbytes + R.nbytes} bytes  (L + R)")
    print(f"  Adagrad-full     equivalent   : {(d_in * d_out) ** 2 * 8} bytes  (full Hessian)")


if __name__ == "__main__":
    demo()
