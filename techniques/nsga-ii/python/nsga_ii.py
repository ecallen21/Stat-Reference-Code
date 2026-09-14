"""NSGA-II — Non-dominated Sorting Genetic Algorithm II.

Deb, Pratap, Agarwal & Meyarivan (2002). Multi-objective
evolutionary optimisation using:

    1. Fast non-dominated sorting into fronts F_1, F_2, ...
    2. Crowding-distance density estimate on each front
    3. (mu + lambda) selection favouring lower rank / higher crowd
    4. SBX crossover + polynomial mutation

Recovers the Pareto front for competing objectives in one run.
"""

import numpy as np    # arrays + random


def zdt1(x):
    """Zitzler-Deb-Thiele ZDT1 (convex Pareto front)."""
    f1 = x[..., 0]
    g = 1 + 9 * np.mean(x[..., 1:], axis=-1)
    f2 = g * (1 - np.sqrt(f1 / g))
    return np.stack([f1, f2], axis=-1)


def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def non_dominated_sort(F):
    n = len(F)
    S = [[] for _ in range(n)]
    nd = np.zeros(n, dtype=int)
    rank = np.zeros(n, dtype=int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                nd[p] = nd[p] + 1
        if nd[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                nd[q] = nd[q] - 1
                if nd[q] == 0:
                    rank[q] = i + 1
                    nxt.append(q)
        i = i + 1
        fronts.append(nxt)
    fronts.pop()
    return fronts, rank


def crowding_distance(F, front):
    idx = np.array(front)
    dist = np.zeros(len(idx))
    for m in range(F.shape[1]):
        order = np.argsort(F[idx, m])
        dist[order[0]] = dist[order[-1]] = np.inf
        fmin, fmax = F[idx, m].min(), F[idx, m].max()
        if fmax == fmin:
            continue
        for k in range(1, len(idx) - 1):
            dist[order[k]] = dist[order[k]] + (F[idx[order[k + 1]], m] - F[idx[order[k - 1]], m]) / (fmax - fmin)
    return dist


def sbx_crossover(x1, x2, eta=15, rng=None):
    u = rng.uniform(size=x1.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)),
                    (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    y1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
    y2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
    return np.clip(y1, 0, 1), np.clip(y2, 0, 1)


def poly_mutation(x, eta=20, prob=0.1, rng=None):
    mask = rng.uniform(size=x.shape) < prob
    u = rng.uniform(size=x.shape)
    delta = np.where(u < 0.5,
                     (2 * u) ** (1 / (eta + 1)) - 1,
                     1 - (2 * (1 - u)) ** (1 / (eta + 1)))
    return np.clip(np.where(mask, x + delta, x), 0, 1)


def nsga2(fn, dim, pop_size, n_gen, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.uniform(size=(pop_size, dim))
    F = fn(P)
    for gen in range(n_gen):
        # offspring
        idx = rng.permutation(pop_size)
        p1, p2 = P[idx[:pop_size // 2]], P[idx[pop_size // 2:]]
        c1, c2 = sbx_crossover(p1, p2, rng=rng)
        Q = np.vstack([c1, c2])
        Q = poly_mutation(Q, rng=rng)
        R = np.vstack([P, Q])
        FR = fn(R)
        fronts, rank = non_dominated_sort(FR)
        new_P, new_F = [], []
        for front in fronts:
            if len(new_P) + len(front) <= pop_size:
                new_P.extend(R[front])
                new_F.extend(FR[front])
            else:
                d = crowding_distance(FR, front)
                order = np.argsort(-d)
                slots = pop_size - len(new_P)
                sel = np.array(front)[order[:slots]]
                new_P.extend(R[sel])
                new_F.extend(FR[sel])
                break
        P = np.array(new_P)
        F = np.array(new_F)
    return P, F


def demo():
    print("=== NSGA-II (Deb et al 2002) ===")
    print("Test: ZDT1 (d=6, convex Pareto front from f1=0..1)")
    P, F = nsga2(zdt1, dim=6, pop_size=80, n_gen=100, seed=1)
    # true front: f2 = 1 - sqrt(f1) at g=1
    tf1 = np.linspace(0, 1, 200)
    tf2 = 1 - np.sqrt(tf1)
    idx = np.argsort(F[:, 0])
    print(f"  Final population front (sorted by f1):")
    for k in range(0, len(idx), max(1, len(idx) // 6)):
        f1, f2 = F[idx[k]]
        print(f"    f1 = {f1:.3f}, f2 = {f2:.3f}  (true = {1 - np.sqrt(max(f1, 0)):.3f})")

    d_to_pareto = np.mean([
        np.min(np.abs(F[i, 0] - tf1) + np.abs(F[i, 1] - tf2))
        for i in range(len(F))
    ])
    print(f"\nMean L1 distance to true Pareto front: {d_to_pareto:.4f}  (0 = perfect)")


if __name__ == "__main__":
    demo()
