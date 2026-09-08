"""Genetic Algorithm (Reference Sec 47.141).

Holland 1975 'Adaptation in Natural and Artificial Systems';
Goldberg 1989 'Genetic Algorithms in Search, Optimization and
Machine Learning'. Population-based search using biology-inspired
operators:

    1. Fitness ranking + tournament / roulette selection.
    2. Crossover (single-point / uniform / SBX) of two parents.
    3. Mutation (Gaussian / bit-flip) of offspring.
    4. Repeat over generations; keep the best solution.

Handles discrete search spaces (subset selection, scheduling) that
gradient methods can't touch.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def ga_binary(f, n_bits, pop_size=50, generations=100, cx=0.7, mut=0.02, seed=0):
    """Binary-string GA with tournament selection + 1-point crossover + bit-flip mutation."""
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, 2, size=(pop_size, n_bits))
    best_x, best_f = pop[0], f(pop[0])
    for gen in range(generations):
        fits = np.array([f(x) for x in pop])
        # Track best
        j = int(np.argmin(fits))
        if fits[j] < best_f:
            best_x = pop[j].copy(); best_f = float(fits[j])
        # Selection (tournament k=3) + crossover + mutation
        new_pop = []
        while len(new_pop) < pop_size:
            i1, j1, k1 = rng.choice(pop_size, 3, replace=False)
            i2, j2, k2 = rng.choice(pop_size, 3, replace=False)
            p1 = pop[[i1, j1, k1][np.argmin([fits[i1], fits[j1], fits[k1]])]]
            p2 = pop[[i2, j2, k2][np.argmin([fits[i2], fits[j2], fits[k2]])]]
            if rng.uniform() < cx:
                pt = int(rng.integers(1, n_bits))
                c1 = np.concatenate([p1[:pt], p2[pt:]])
                c2 = np.concatenate([p2[:pt], p1[pt:]])
            else:
                c1, c2 = p1.copy(), p2.copy()
            for c in (c1, c2):
                flip = rng.uniform(size=n_bits) < mut
                c[flip] = 1 - c[flip]
            new_pop.extend([c1, c2])
        pop = np.array(new_pop[:pop_size])
    return {"x": best_x, "f": best_f}


if __name__ == "__main__":
    print("=== Genetic Algorithm (Holland 1975; Goldberg 1989) ===\n")
    rng = np.random.default_rng(0)

    # 0/1 knapsack: pick items with total weight <= capacity to maximise value
    n = 30
    weights = rng.integers(1, 20, size=n)
    values = rng.integers(1, 30, size=n)
    capacity = int(0.4 * weights.sum())

    def knapsack_fitness(x):
        w = weights @ x; v = values @ x
        if w > capacity:
            return float(-v + 1000 * (w - capacity))       # penalty
        return float(-v)

    for gens in [20, 100, 500]:
        r = ga_binary(knapsack_fitness, n_bits=n, pop_size=60,
                        generations=gens, cx=0.8, mut=0.03, seed=0)
        picked = int(r["x"].sum())
        w = int(weights @ r["x"])
        v = int(values @ r["x"])
        print(f"  gens = {gens:3d}   fitness = {-r['f']:.1f}   "
              f"value = {v}   weight = {w} / {capacity}   ({picked} items)")

    # Baseline: greedy (value/weight ratio)
    ratio = values / weights
    order = np.argsort(-ratio)
    w_g = 0; v_g = 0
    for i in order:
        if w_g + weights[i] <= capacity:
            w_g += weights[i]; v_g += values[i]
    print(f"\n  Greedy value/weight baseline: value = {v_g}   weight = {w_g}")

    print("\n--- library cross-check (GA R; DEAP / pymoo Python) ---")
