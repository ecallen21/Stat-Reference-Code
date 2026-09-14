"""Differential Evolution (Storn & Price 1997).

Population-based evolutionary optimiser:

    for each x_i in population:
        pick 3 distinct others a, b, c
        v = a + F * (b - c)         (mutation)
        u = crossover(x_i, v, CR)
        if f(u) < f(x_i): replace x_i with u

Simple, gradient-free, competitive with CMA-ES / GA on many
benchmark suites.
"""

import numpy as np    # arrays + random


def differential_evolution(f, bounds, pop_size=30, F=0.7, CR=0.9,
                            max_iter=200, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    d = len(bounds)
    pop = lo + rng.uniform(size=(pop_size, d)) * (hi - lo)
    fit = np.array([f(x) for x in pop])
    history = [fit.min()]
    for gen in range(max_iter):
        for i in range(pop_size):
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = pop[rng.choice(idxs, size=3, replace=False)]
            v = a + F * (b - c)
            v = np.clip(v, lo, hi)
            mask = rng.uniform(size=d) < CR
            if not mask.any():
                mask[rng.integers(d)] = True
            u = np.where(mask, v, pop[i])
            fu = f(u)
            if fu < fit[i]:
                pop[i] = u
                fit[i] = fu
        history.append(fit.min())
    best_idx = int(np.argmin(fit))
    return pop[best_idx], fit[best_idx], history


def demo():
    print("=== Differential Evolution (Storn-Price 1997) ===")

    def rastrigin(x):
        A = 10
        return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

    for d in [2, 5, 10]:
        bounds = [(-5.12, 5.12)] * d
        x, fx, hist = differential_evolution(rastrigin, bounds,
                                             pop_size=30, max_iter=300, seed=1)
        print(f"  Rastrigin d={d:2d}: f* = {fx:.6f}  (global 0.0), "
              f"|x*| max = {np.max(np.abs(x)):.3f}")

    print("\nCompare scipy.optimize.differential_evolution:")
    try:
        from scipy.optimize import differential_evolution as scipy_de
        for d in [2, 5, 10]:
            bounds = [(-5.12, 5.12)] * d
            res = scipy_de(rastrigin, bounds, seed=1, maxiter=300, popsize=30)
            print(f"  scipy d={d:2d}: f* = {res.fun:.6f}")
    except ImportError:
        pass


if __name__ == "__main__":
    demo()
