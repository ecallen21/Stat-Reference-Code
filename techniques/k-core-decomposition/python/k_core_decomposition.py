"""k-core decomposition and coreness (Reference §24.x extra).

The k-core of a graph G is the maximal subgraph in which every vertex has
degree >= k.  The coreness (or core number) of a vertex v is the largest k
for which v belongs to the k-core.

Batagelj-Zaversnik (2003) O(m + n) algorithm:
  * Sort vertices by current degree.
  * Repeatedly remove the min-degree vertex v (assign coreness = current
    degree), decrementing degrees of its neighbours.

k-truss (Cohen 2008): edge-based analogue.  The k-truss is the maximal
subgraph in which every edge belongs to at least k - 2 triangles.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def coreness(A) -> np.ndarray:
    """Batagelj-Zaversnik O(m + n) core-number computation.

    core(v) = max k such that v belongs to the k-core.  Property: once a
    vertex with degree d is removed, every subsequently-removed vertex has
    coreness >= d (nested-cores invariant).
    """
    A = np.asarray(A, dtype=int); n = A.shape[0]
    deg = A.sum(axis=1).astype(int)
    core = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    current_k = 0                                         # rolling coreness floor
    for _ in range(n):
        alive = np.where(remaining)[0]
        if len(alive) == 0:
            break
        idx_min = alive[np.argmin(deg[alive])]
        current_k = max(current_k, int(deg[idx_min]))
        core[idx_min] = current_k
        remaining[idx_min] = False
        for v in np.where(A[idx_min])[0]:
            if remaining[v]:
                deg[v] -= 1
    return core


def k_core_subgraph(A, k: int):
    A = np.asarray(A, dtype=int)
    core = coreness(A)
    mask = core >= k
    return A[np.ix_(mask, mask)], np.where(mask)[0]


def edge_triangle_counts(A):
    """Number of triangles each edge participates in."""
    A = np.asarray(A, dtype=int); n = A.shape[0]
    T = np.zeros((n, n), dtype=int)
    for i in range(n):
        nbrs_i = set(np.where(A[i])[0])
        for j in np.where(A[i])[0]:
            if j > i:
                nbrs_j = set(np.where(A[j])[0])
                T[i, j] = T[j, i] = len(nbrs_i & nbrs_j)
    return T


def k_truss(A, k: int):
    """k-truss: repeatedly drop edges belonging to < k-2 triangles."""
    A = np.asarray(A, dtype=int).copy()
    while True:
        T = edge_triangle_counts(A)
        bad = (A == 1) & (T < k - 2)
        if not bad.any():
            break
        A[bad] = 0
    # remove isolated vertices for the subgraph
    keep = A.sum(axis=1) > 0
    return A[np.ix_(keep, keep)], np.where(keep)[0]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # graph = two cliques (K5 and K7) joined by a single bridging edge + a pendant
    n = 5 + 7 + 1
    A = np.zeros((n, n), dtype=int)
    for i in range(5):                                    # K5
        for j in range(i + 1, 5):
            A[i, j] = A[j, i] = 1
    for i in range(5, 12):                                # K7
        for j in range(i + 1, 12):
            A[i, j] = A[j, i] = 1
    A[4, 5] = A[5, 4] = 1                                 # bridge
    A[0, 12] = A[12, 0] = 1                                # pendant

    core = coreness(A)
    print("=== k-core decomposition (K5 ~ K7 + bridge + pendant) ===")
    for v in range(n):
        role = ("pendant" if v == 12 else "K5" if v < 5 else "K7")
        print(f"  node {v:>2} ({role}): coreness = {core[v]}")
    print(f"\n  max coreness (degeneracy) = {int(core.max())}")

    # k-core subgraphs
    for k in (2, 4, 6):
        _, members = k_core_subgraph(A, k)
        print(f"  {k}-core: {len(members)} vertices — {members.tolist()}")

    # k-truss
    for k in (3, 4, 5):
        Ak, members = k_truss(A, k)
        print(f"  {k}-truss: {len(members)} vertices — {members.tolist()}")

    print("\n--- library cross-check (networkx.core_number / networkx.k_core) ---")
    try:
        import networkx as nx
        G = nx.from_numpy_array(A)
        cn = nx.core_number(G)
        max_diff = max(abs(core[v] - cn[v]) for v in range(n))
        print(f"  max diff coreness vs nx.core_number = {max_diff}")
    except ImportError:
        print("  (networkx not installed)")
