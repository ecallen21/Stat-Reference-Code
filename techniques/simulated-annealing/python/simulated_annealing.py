"""Simulated Annealing (Reference Sec 47.140).

Kirkpatrick, Gelatt & Vecchi 1983 'Optimization by simulated
annealing', Science 220. Metropolis-based global optimizer:

    1. Propose neighbor  x'  from a local kernel around x.
    2. Accept if f(x') < f(x); else accept with prob exp(-(f(x') - f(x))/T).
    3. Cool T on a schedule T_k = T_0 * alpha^k.

Converges (in probability) to a global optimum for logarithmic
cooling; geometric cooling is practical and fast.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sim_anneal(f, x0, T0=1.0, alpha=0.999, n_iter=5000, step=0.5, seed=0):
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    fx = f(x)
    best_x, best_f = x.copy(), fx
    T = T0
    for k in range(n_iter):
        x_new = x + step * rng.normal(size=x.shape)
        fx_new = f(x_new)
        dE = fx_new - fx
        if dE < 0 or rng.uniform() < np.exp(-dE / max(T, 1e-12)):
            x, fx = x_new, fx_new
            if fx < best_f:
                best_x, best_f = x.copy(), fx
        T *= alpha
    return {"x": best_x, "f": float(best_f), "T_final": float(T)}


def tsp_sa(dist, n_iter=20000, T0=2.0, alpha=0.9995, seed=0):
    """SA on a TSP: 2-opt neighbor, tour length as cost."""
    rng = np.random.default_rng(seed)
    n = len(dist)
    tour = list(rng.permutation(n))
    def cost(t): return sum(dist[t[i], t[(i + 1) % n]] for i in range(n))
    c = cost(tour)
    best = tour[:]; best_c = c
    T = T0
    for k in range(n_iter):
        i, j = sorted(rng.choice(n, size=2, replace=False))
        if j - i < 2: continue
        new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
        c_new = cost(new_tour)
        if c_new < c or rng.uniform() < np.exp(-(c_new - c) / max(T, 1e-12)):
            tour = new_tour; c = c_new
            if c < best_c:
                best = tour[:]; best_c = c
        T *= alpha
    return {"tour": best, "cost": float(best_c)}


if __name__ == "__main__":
    print("=== Simulated Annealing (Kirkpatrick-Gelatt-Vecchi 1983) ===\n")

    # Continuous: multi-well 1-D
    def f1(x): return float(x[0] ** 2 + 20 * np.sin(x[0]) ** 2)
    r = sim_anneal(f1, np.array([-3.0]), n_iter=20000, T0=5.0, alpha=0.9995, step=1.0)
    print(f"  1-D multi-well:  x* = {r['x'][0]:.3f}   f* = {r['f']:.4f}  (truth x=0, f=0)")

    # TSP with 15 random cities
    rng = np.random.default_rng(0)
    cities = rng.uniform(size=(15, 2))
    D = np.sqrt(((cities[:, None, :] - cities[None, :, :]) ** 2).sum(-1))
    r_tsp = tsp_sa(D, n_iter=50000, T0=1.5, alpha=0.9995)
    # Random baseline
    baselines = [sum(D[tour[i], tour[(i + 1) % 15]] for i in range(15))
                  for tour in [rng.permutation(15) for _ in range(500)]]
    print(f"\n  15-city TSP tour length: SA = {r_tsp['cost']:.3f}   "
          f"random-avg baseline = {np.mean(baselines):.3f}")

    print("\n--- library cross-check (scipy.optimize.dual_annealing / basinhopping; GenSA R) ---")
