"""Mirror Descent (Reference Sec 47.339).

Nemirovsky & Yudin 1983; Beck & Teboulle 2003. Generalise
gradient descent by choosing a MIRROR MAP Phi (Bregman
generator). Update:

    x_{k+1} = argmin_x <grad f(x_k), x> + (1/eta) D_Phi(x, x_k)

For probability-simplex constraints, Phi(x) = sum x_i log x_i
recovers EXPONENTIATED GRADIENT (multiplicative weights):

    x_{k+1, j} proportional to x_{k, j} * exp(-eta * grad_j)

Adapts step geometry to the constraint set, often beating plain
projected gradient in high dimensions.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def exp_grad_simplex(f, grad_f, d, eta=0.1, n_iter=200):
    """Mirror descent on probability simplex with entropy generator."""
    x = np.ones(d) / d
    hist = [f(x)]
    for _ in range(n_iter):
        g = grad_f(x)
        x = x * np.exp(-eta * g)
        x = x / x.sum()
        hist.append(f(x))
    return x, hist


def projected_gradient_simplex(f, grad_f, d, lr=0.05, n_iter=200):
    """Projected gradient onto the simplex (Duchi et al 2008)."""
    def proj(v):
        u = np.sort(v)[::-1]
        rho = np.max(np.where(u + (1 - np.cumsum(u)) / (np.arange(1, len(u) + 1)) > 0)[0]) + 1
        lam = (1 - np.sum(u[:rho])) / rho
        return np.maximum(v + lam, 0)
    x = np.ones(d) / d
    hist = [f(x)]
    for _ in range(n_iter):
        x = proj(x - lr * grad_f(x))
        hist.append(f(x))
    return x, hist


if __name__ == "__main__":
    print("=== Mirror Descent (Nemirovsky-Yudin 1983; Beck-Teboulle 2003) ===\n")
    rng = np.random.default_rng(0)

    # Toy: minimise weighted linear + quadratic on probability simplex
    d = 20
    c = rng.uniform(0, 1, d)                                        # loss coefficient
    A = rng.standard_normal((d, d)); A = A @ A.T / d
    def f(x): return float(c @ x + 0.5 * x @ A @ x)
    def grad_f(x): return c + A @ x

    x_md, hist_md = exp_grad_simplex(f, grad_f, d, eta=0.5, n_iter=200)
    x_pg, hist_pg = projected_gradient_simplex(f, grad_f, d, lr=0.05, n_iter=200)

    print(f"  Simplex-constrained convex optimisation, d = {d}\n")
    print(f"  Final objective:")
    print(f"    Mirror descent (exp-grad):     {hist_md[-1]:.4f}")
    print(f"    Projected gradient:            {hist_pg[-1]:.4f}\n")

    print(f"  {'iter':>5}   {'exp-grad':>10}   {'proj-grad':>10}")
    for k in [1, 5, 20, 50, 100, 200]:
        print(f"  {k:>5}   {hist_md[k]:>10.4f}   {hist_pg[k]:>10.4f}")

    print(f"\n  Solution support (indices with x_j > 0.05):")
    print(f"    Mirror descent:      {np.where(x_md > 0.05)[0].tolist()}")
    print(f"    Projected gradient:  {np.where(x_pg > 0.05)[0].tolist()}")

    print(f"\n  Mirror descent adapts the update to the simplex geometry via KL-Bregman;")
    print(f"  can vastly outperform Euclidean projected gradient in high-dim simplex problems.")

    print("\n--- library cross-check (cvxpy, or custom autograd + exp normalisation) ---")
