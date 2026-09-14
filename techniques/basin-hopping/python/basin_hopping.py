"""Basin hopping (Wales & Doye 1997).

Global optimisation by iterating:
    (1) local minimisation from current x
    (2) random perturbation x' = x + N(0, step^2)
    (3) accept x' if f(x') < f(x) OR by Metropolis rule at T.

Very effective on rugged energy landscapes such as protein
folding, atomic cluster minimisation, and multi-modal
statistical objectives.
"""

import numpy as np    # arrays + random


def local_min(f, x0, max_iter=100, lr=0.01):
    """Simple gradient-free local minimizer via random search."""
    x = x0.copy()
    fx = f(x)
    for _ in range(max_iter):
        cand = x + 0.01 * np.random.default_rng().standard_normal(x.shape)
        fc = f(cand)
        if fc < fx:
            x, fx = cand, fc
    return x, fx


def basin_hopping(f, x0, step=0.5, T=1.0, n_iter=100, seed=0,
                  local_opt=None):
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    if local_opt is None:
        try:
            from scipy.optimize import minimize
            local_opt = lambda f_, x_: minimize(f_, x_, method="Nelder-Mead").x
        except ImportError:
            local_opt = lambda f_, x_: local_min(f_, x_)[0]
    x = local_opt(f, x)
    best_x = x.copy()
    best_f = f(x)
    for _ in range(n_iter):
        x_prop = x + step * rng.standard_normal(len(x))
        x_prop = local_opt(f, x_prop)
        f_prop = f(x_prop)
        if f_prop < f(x) or rng.uniform() < np.exp(-(f_prop - f(x)) / T):
            x = x_prop
        if f_prop < best_f:
            best_f = f_prop
            best_x = x_prop.copy()
    return best_x, best_f


def demo():
    print("=== Basin hopping (Wales-Doye 1997) ===")

    # Multi-modal 2D function: (x-1)^2*(x+1)^2 + y^2 + 5 sin(3 x) sin(3 y)
    def f(v):
        x, y = v
        return (x - 1) ** 2 * (x + 1) ** 2 + y ** 2 + 5 * np.sin(3 * x) * np.sin(3 * y)

    x0 = np.array([2.5, 2.5])
    x_star, f_star = basin_hopping(f, x0, step=1.5, T=5.0, n_iter=50, seed=1)
    print(f"  Multi-modal 2D f: best x = {np.round(x_star, 3)}, f = {f_star:.4f}")

    # Rastrigin d=5
    def rastrigin(x):
        A = 10
        return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

    x0 = np.array([3.5, -2.0, 1.5, 0.7, -3.2])
    x_star, f_star = basin_hopping(rastrigin, x0, step=1.0, T=5.0, n_iter=100, seed=2)
    print(f"  Rastrigin d=5: best x = {np.round(x_star, 3)}, f = {f_star:.4f}  (global 0)")


if __name__ == "__main__":
    demo()
