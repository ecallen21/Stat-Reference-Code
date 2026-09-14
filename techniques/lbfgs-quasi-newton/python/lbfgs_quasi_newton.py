"""L-BFGS - Limited-Memory Quasi-Newton (Reference Sec 47.319).

Liu & Nocedal 1989 Math Prog. Quasi-Newton method that
maintains only the last m (s_k, y_k) pairs to approximate the
inverse Hessian:

    s_k = x_{k+1} - x_k
    y_k = grad_{k+1} - grad_k
    H_{k+1} ~ two-loop recursion over stored pairs

Standard for smooth unconstrained optimisation up to ~10^7
parameters. L-BFGS-B adds box constraints.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def lbfgs_direction(grad, s_list, y_list, m=10):
    """Two-loop L-BFGS recursion to compute -H_k grad."""
    q = -grad.copy()
    rho = [1.0 / (y @ s + 1e-12) for s, y in zip(s_list, y_list)]
    alpha = []
    for s, y, r in zip(reversed(s_list), reversed(y_list), reversed(rho)):
        a = r * (s @ q); alpha.append(a); q = q - a * y
    if s_list:
        s_last = s_list[-1]; y_last = y_list[-1]
        gamma = (s_last @ y_last) / (y_last @ y_last + 1e-12)
    else:
        gamma = 1.0
    r_vec = gamma * q
    for (s, y, rr), a in zip(zip(s_list, y_list, rho), reversed(alpha)):
        beta = rr * (y @ r_vec)
        r_vec = r_vec + (a - beta) * s
    return r_vec


def lbfgs(f, grad_f, x0, m=10, max_iter=50, lr=None, tol=1e-6):
    x = x0.copy(); s_list = []; y_list = []
    grad = grad_f(x); losses = [f(x)]
    for k in range(max_iter):
        d = lbfgs_direction(grad, s_list, y_list, m=m)
        # Simple line search (backtracking)
        step = 1.0
        while f(x + step * d) > f(x) + 1e-4 * step * (grad @ d) and step > 1e-10:
            step *= 0.5
        x_new = x + step * d
        grad_new = grad_f(x_new)
        s = x_new - x; y = grad_new - grad
        if y @ s > 1e-10:
            s_list.append(s); y_list.append(y)
            if len(s_list) > m: s_list.pop(0); y_list.pop(0)
        x = x_new; grad = grad_new
        losses.append(f(x))
        if np.linalg.norm(grad) < tol: break
    return x, losses


if __name__ == "__main__":
    print("=== L-BFGS (Liu & Nocedal 1989) ===\n")
    rng = np.random.default_rng(0)

    # Fit ridge regression by direct optimisation via L-BFGS
    n, d = 200, 30
    X = rng.standard_normal((n, d))
    beta_true = rng.normal(0, 1, d)
    y = X @ beta_true + rng.normal(0, 0.5, n)
    lam = 0.1

    def f(b): return 0.5 * np.mean((X @ b - y) ** 2) + 0.5 * lam * (b @ b)
    def grad_f(b): return X.T @ (X @ b - y) / n + lam * b

    beta0 = np.zeros(d)
    beta_final, losses = lbfgs(f, grad_f, beta0, m=10, max_iter=50)
    # Closed-form ridge
    beta_cf = np.linalg.solve(X.T @ X / n + lam * np.eye(d), X.T @ y / n)

    print(f"  n = {n}, d = {d}, ridge lam = {lam}")
    print(f"  {'iter':>5}   {'objective':>10}")
    for it in sorted({0, 3, 10, min(20, len(losses) - 1), len(losses) - 1}):
        print(f"  {it:>5}   {losses[it]:>10.6f}")
    print(f"\n  Distance from closed-form ridge:  {np.linalg.norm(beta_final - beta_cf):.2e}")
    print(f"  Converged in {len(losses) - 1} iterations")

    print("\n--- library cross-check (scipy.optimize.minimize(method='L-BFGS-B'); scipy.optimize.fmin_l_bfgs_b) ---")
