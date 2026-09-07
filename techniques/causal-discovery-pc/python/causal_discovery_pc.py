"""Causal discovery -- PC algorithm (Reference Sec 15.35).

Spirtes, Glymour & Scheines 2000 'Causation, Prediction and Search'.
The PC (Peter-Clark) algorithm learns the skeleton and colliders of a
DAG from observational data using conditional-independence tests.

Two-phase procedure:

  1. START from complete undirected graph on {X_1, ..., X_p}.

     For each edge (i, j) test H0: X_i _||_ X_j | S for subsets S of
     the current neighbours of i (and of j) with |S| = 0, 1, 2, ...
     If ANY subset makes them independent -> remove edge (i, j) and
     save the separator sep(i, j) = S.

  2. ORIENT triples i - k - j (i, j non-adjacent) as colliders
     i -> k <- j  iff  k NOT in sep(i, j).

     Remaining Meek propagation rules orient more edges.

Returns a Completed Partially Directed Acyclic Graph (CPDAG) --
Markov equivalence class of DAGs consistent with the CI structure.

For simplicity we use partial correlation (Fisher z-test) which
requires Gaussian data.
"""
from __future__ import annotations    # stdlib

from itertools import combinations    # subset generation

import numpy as np    # numerical arrays
from scipy import stats


def partial_corr_test(cov, i, j, S, n, alpha=0.05):
    """Fisher-z test for X_i _||_ X_j | X_S under Gaussian."""
    idx = [i, j] + list(S)
    C = cov[np.ix_(idx, idx)]
    Cinv = np.linalg.pinv(C)
    #  partial corr of i, j given rest: -Cinv[0, 1] / sqrt(Cinv[0, 0]*Cinv[1, 1])
    r = -Cinv[0, 1] / np.sqrt(Cinv[0, 0] * Cinv[1, 1])
    r = np.clip(r, -0.999, 0.999)
    z = 0.5 * np.log((1 + r) / (1 - r))
    stat = np.sqrt(n - len(S) - 3) * abs(z)
    p = 2 * (1 - stats.norm.cdf(stat))
    return p > alpha, p, r   # independent?


def pc_skeleton(X, alpha=0.05, max_order=3):
    """Phase 1: recover the skeleton and separator sets."""
    n, p = X.shape
    cov = np.corrcoef(X.T)
    adj = np.ones((p, p), dtype=bool) & ~np.eye(p, dtype=bool)
    sep = {}
    l = 0
    while l <= max_order:
        removed_any = False
        for i, j in combinations(range(p), 2):
            if not adj[i, j]:
                continue
            neighbors_i = [k for k in range(p) if adj[i, k] and k != j]
            if len(neighbors_i) < l:
                continue
            for S in combinations(neighbors_i, l):
                indep, pval, r = partial_corr_test(cov, i, j, list(S), n, alpha)
                if indep:
                    adj[i, j] = False
                    adj[j, i] = False
                    sep[(i, j)] = list(S)
                    sep[(j, i)] = list(S)
                    removed_any = True
                    break
        l += 1
        if not removed_any:
            break
    return adj, sep


def orient_colliders(adj, sep, p):
    """Phase 2: orient v-structures i -> k <- j."""
    directed = np.zeros((p, p), dtype=bool)   # directed[i, j] means i -> j
    for k in range(p):
        neigh_k = [x for x in range(p) if adj[x, k]]
        for i, j in combinations(neigh_k, 2):
            if adj[i, j]:                # only consider non-adjacent i, j
                continue
            S = sep.get((i, j), sep.get((j, i), None))
            if S is not None and k not in S:
                directed[i, k] = True
                directed[j, k] = True
    return directed


if __name__ == "__main__":
    print("=== PC algorithm -- causal discovery via CI tests ===\n")
    rng = np.random.default_rng(0)
    n = 3000
    #  Ground-truth DAG:
    #    X0 -> X2 <- X1        (v-structure at X2)
    #    X2 -> X3
    #    X4 (isolated)
    X0 = rng.normal(size=n)
    X1 = rng.normal(size=n)
    X2 = 0.8 * X0 + 0.8 * X1 + rng.normal(scale=0.5, size=n)
    X3 = 0.9 * X2 + rng.normal(scale=0.5, size=n)
    X4 = rng.normal(size=n)
    X = np.c_[X0, X1, X2, X3, X4]

    adj, sep = pc_skeleton(X, alpha=0.01, max_order=3)
    print("  Estimated skeleton (undirected adjacency):")
    for i, j in combinations(range(5), 2):
        if adj[i, j]:
            print(f"    X{i} -- X{j}")

    directed = orient_colliders(adj, sep, p=5)
    print("\n  Oriented v-structures:")
    for i in range(5):
        for j in range(5):
            if directed[i, j]:
                print(f"    X{i} -> X{j}")

    print("\n  Truth:")
    print("    X0 -> X2, X1 -> X2   (v-structure)")
    print("    X2 -- X3             (undirected -- Markov equivalent)")
    print("    X4 isolated")

    print("\n--- library cross-check (pcalg R; causal-learn Python) ---")
