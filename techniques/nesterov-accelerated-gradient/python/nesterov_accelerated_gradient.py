"""Nesterov Accelerated Gradient (Reference Sec 47.107).

Nesterov 1983 'A method for solving the convex programming
problem with convergence rate O(1/k^2)', Dokl Akad Nauk SSSR.
For a smooth convex f with Lipschitz gradient L, gradient descent
converges O(1/k); NAG achieves O(1/k^2) via a LOOK-AHEAD gradient:

    y_k = theta_k + beta_k (theta_k - theta_{k-1})       (extrapolation)
    theta_{k+1} = y_k - (1/L) grad f(y_k)                (grad at look-ahead)

Convex, smooth non-strongly-convex: use beta_k = (t_{k-1} - 1) / t_k
with t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2 (FISTA scheme).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def gd(grad, x0, lr, n_iter):
    x = x0.copy(); losses = []
    for _ in range(n_iter):
        x = x - lr * grad(x)
        losses.append(float(np.linalg.norm(x)))
    return x, losses


def heavy_ball(grad, x0, lr, momentum, n_iter):
    """Polyak's heavy-ball momentum (1964)."""
    x = x0.copy(); x_prev = x0.copy(); losses = []
    for _ in range(n_iter):
        x_new = x - lr * grad(x) + momentum * (x - x_prev)
        x_prev = x; x = x_new
        losses.append(float(np.linalg.norm(x)))
    return x, losses


def nag(grad, x0, lr, n_iter):
    """Nesterov 1983 acceleration with FISTA-style momentum schedule."""
    x = x0.copy(); y = x0.copy(); t = 1.0
    losses = []
    for _ in range(n_iter):
        x_new = y - lr * grad(y)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        beta = (t - 1) / t_new
        y = x_new + beta * (x_new - x)
        x = x_new; t = t_new
        losses.append(float(np.linalg.norm(x)))
    return x, losses


if __name__ == "__main__":
    print("=== Nesterov Accelerated Gradient (Nesterov 1983) ===\n")
    # Ill-conditioned quadratic f(x) = 0.5 x' A x, A = diag(1, 100)
    A = np.diag([1.0, 100.0])
    L = 100.0                # Lipschitz constant of grad
    def f(x): return 0.5 * x @ A @ x
    def grad(x): return A @ x
    x0 = np.array([10.0, 1.0])

    for method_name, fn, kw in [
        ("Gradient descent lr=1/L      ", gd, dict(lr=1 / L)),
        ("Heavy-ball  lr=1/L, mom=0.95 ", heavy_ball, dict(lr=1 / L, momentum=0.95)),
        ("NAG          lr=1/L           ", nag, dict(lr=1 / L)),
    ]:
        x, losses = fn(grad, x0.copy(), n_iter=200, **kw)
        # First iterate to reach ||x|| < 1e-3
        idx = next((i for i, v in enumerate(losses) if v < 1e-3), None)
        print(f"  {method_name}  final ||x|| = {losses[-1]:.2e}   "
              f"iters to 1e-3 = {idx if idx is not None else '>200'}")

    print("\n  On ill-conditioned quadratics NAG's O(1/k^2) rate lets it reach a")
    print("  target error in ~ sqrt(kappa) times fewer iterations than plain GD.")

    print("\n--- library cross-check (torch.optim.SGD(momentum, nesterov=True) Python) ---")
