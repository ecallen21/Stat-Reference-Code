"""Particle Swarm Optimization (Reference Sec 47.139).

Kennedy & Eberhart 1995 'Particle swarm optimization', IEEE
ICNN. Population-based global optimizer inspired by flocking
birds. Each particle i keeps position x_i, velocity v_i,
personal-best p_i; the swarm keeps global-best g:

    v_i <- w v_i + c1 r1 (p_i - x_i) + c2 r2 (g - x_i)
    x_i <- x_i + v_i

Inertia weight w decays over generations (Shi & Eberhart 1998).
Simple, derivative-free, and works well on non-differentiable
landscapes.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def pso(f, bounds, n_particles=30, n_iter=200, w_max=0.9, w_min=0.4, c1=1.5, c2=1.5, seed=0):
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    d = len(bounds)
    lo, hi = bounds[:, 0], bounds[:, 1]
    x = lo + (hi - lo) * rng.uniform(size=(n_particles, d))
    v = 0.1 * (hi - lo) * rng.normal(size=(n_particles, d))
    p_best = x.copy(); p_fit = np.array([f(xi) for xi in x])
    g = p_best[np.argmin(p_fit)]; g_fit = p_fit.min()
    for it in range(n_iter):
        w = w_max - (w_max - w_min) * it / max(n_iter - 1, 1)
        r1, r2 = rng.uniform(size=(n_particles, d)), rng.uniform(size=(n_particles, d))
        v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g - x)
        x = np.clip(x + v, lo, hi)
        fx = np.array([f(xi) for xi in x])
        improved = fx < p_fit
        p_best[improved] = x[improved]; p_fit[improved] = fx[improved]
        if p_fit.min() < g_fit:
            g = p_best[np.argmin(p_fit)]; g_fit = p_fit.min()
    return {"x": g, "f": float(g_fit)}


if __name__ == "__main__":
    print("=== Particle Swarm Optimization (Kennedy-Eberhart 1995) ===\n")

    # Rastrigin 5D (highly multimodal)
    def rastrigin(x):
        A = 10.0
        return float(A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))

    bounds = [(-5.12, 5.12)] * 5
    for iters in [50, 200, 500]:
        r = pso(rastrigin, bounds, n_particles=30, n_iter=iters, seed=0)
        print(f"  iters = {iters:3d}   f_best = {r['f']:.4f}   ||x||_∞ = {np.max(np.abs(r['x'])):.4f}")

    print("\n  Rastrigin optimum at x = 0 (all zeros), f = 0.")
    print("  PSO's swarm can escape local minima that ordinary gradient descent gets stuck in.")

    print("\n--- library cross-check (pyswarm / pyswarms Python; psoptim / pso R) ---")
