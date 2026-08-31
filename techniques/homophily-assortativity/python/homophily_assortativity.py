"""Homophily + assortativity (Reference Sec 30.17).

Newman (2003) 'Mixing patterns in networks.'

DISCRETE ATTRIBUTE MIXING:
  * mixing matrix e_ij = fraction of edges between group i and group j.
  * modularity Q = sum_i (e_ii - a_i^2), with a_i = sum_j e_ij.
  * assortativity coefficient r = Q / (1 - sum a_i^2)  in [-1, 1].
    r > 0 = homophily (like-with-like), r < 0 = heterophily.

DEGREE ASSORTATIVITY (Newman 2002): correlation of the degree of the
two nodes at either end of a random edge.

Here we compute both on a synthetic 3-group graph with tunable within-
group edge probability, and confirm r > 0 under homophily.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def mixing_matrix(A, groups):
    n_groups = int(max(groups)) + 1
    e = np.zeros((n_groups, n_groups))
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                # Each undirected edge contributes 1 to e[gi,gj] + e[gj,gi] (symmetric)
                e[groups[i], groups[j]] += 0.5
                e[groups[j], groups[i]] += 0.5
    total = e.sum()
    return e / max(total, 1e-9)


def attribute_assortativity(A, groups):
    e = mixing_matrix(A, groups)
    a = e.sum(axis=1)
    Q = float(np.trace(e) - (a ** 2).sum())
    denom = 1 - (a ** 2).sum()
    return Q / max(denom, 1e-9)


def degree_assortativity(A):
    """Newman 2002: correlation of degrees at each endpoint of an edge."""
    deg = A.sum(axis=1)
    edges = np.transpose(np.nonzero(np.triu(A, 1)))
    if len(edges) < 2:
        return 0.0
    x = deg[edges[:, 0]]; y = deg[edges[:, 1]]
    xy = np.concatenate([x, y])
    yx = np.concatenate([y, x])
    return float(np.corrcoef(xy, yx)[0, 1])


def simulate_graph(n_per, K, within, cross, seed=0):
    rng = np.random.default_rng(seed)
    n = n_per * K
    groups = np.repeat(np.arange(K), n_per)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = within if groups[i] == groups[j] else cross
            if rng.random() < p:
                A[i, j] = A[j, i] = 1
    return A, groups


if __name__ == "__main__":
    print("=== Attribute assortativity + degree assortativity (Newman 2003) ===\n")
    for within, cross, label in [(0.5, 0.05, "HOMOPHILY"),
                                    (0.15, 0.15, "NEUTRAL"),
                                    (0.05, 0.5, "HETEROPHILY")]:
        A, g = simulate_graph(20, 3, within, cross, seed=1)
        r_attr = attribute_assortativity(A, g)
        r_deg = degree_assortativity(A)
        print(f"  {label:>12}  (p_within={within}, p_cross={cross})"
              f"   attribute r = {r_attr:>6.3f}   degree r = {r_deg:>6.3f}")

    print("\n  attribute r > 0 under homophily, < 0 under heterophily.\n")
    print("--- library cross-check (igraph::assortativity_nominal / assortativity_degree; networkx.algorithms.assortativity) ---")
