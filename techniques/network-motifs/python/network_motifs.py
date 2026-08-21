"""Network motifs: 3-node census + Z-scores vs random baseline (Reference §24.11).

For an undirected simple graph there are two connected 3-node subgraphs:

  * open 2-path (wedge)   -- 3 nodes, 2 edges
  * closed triangle        -- 3 nodes, 3 edges

For each pattern count its occurrences in the graph and compare against
random graphs preserving the degree sequence (configuration model /
double-edge-swap null).  A large positive Z-score identifies an
over-represented motif; a large negative Z identifies an anti-motif.

Directed 3-node motifs: 13 distinct patterns (Milo et al. 2002).  Here we
handle the undirected case; the same Z-score framework generalises.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def count_triangles(A) -> int:
    A = np.asarray(A, dtype=float)
    return int(np.trace(np.linalg.matrix_power(A, 3)) / 6)


def count_wedges(A) -> int:
    """Number of open 2-paths (wedges): sum_v C(deg(v), 2) - 3 * triangles.
    Each triangle contains 3 wedges centred at each of its vertices."""
    d = np.asarray(A).sum(axis=1).astype(int)
    triples = int((d * (d - 1) // 2).sum())
    return triples - 3 * count_triangles(A)


def _degree_swap_rewire(A, n_swaps: int, seed: int = 0):
    """Double-edge swap: pick two edges (a,b) and (c,d), rewire to (a,d) & (c,b)
    if that keeps a simple graph."""
    rng = np.random.default_rng(seed)
    A = A.copy(); n = A.shape[0]
    edges = np.array(np.where(np.triu(A, 1))).T.tolist()
    ok = 0
    for _ in range(n_swaps):
        i, j = rng.integers(0, len(edges), size=2)
        if i == j: continue
        a, b = edges[i]; c, d = edges[j]
        if len({a, b, c, d}) < 4: continue
        if A[a, d] or A[c, b]: continue
        A[a, b] = A[b, a] = 0; A[c, d] = A[d, c] = 0
        A[a, d] = A[d, a] = 1; A[c, b] = A[b, c] = 1
        edges[i] = (a, d); edges[j] = (c, b); ok += 1
    return A


def motif_z_scores(A, n_null: int = 100, swaps_per_edge: int = 5,
                    seed: int = 0) -> dict:
    A = np.asarray(A, dtype=int)
    obs_tri = count_triangles(A); obs_wed = count_wedges(A)
    rng = np.random.default_rng(seed)
    tri_null = []; wed_null = []
    m = int(A.sum() // 2)
    for s in range(n_null):
        A_r = _degree_swap_rewire(A, n_swaps=swaps_per_edge * m, seed=int(rng.integers(1e9)))
        tri_null.append(count_triangles(A_r))
        wed_null.append(count_wedges(A_r))
    tri_null = np.array(tri_null); wed_null = np.array(wed_null)
    return {"triangles": {"obs": obs_tri, "null_mean": float(tri_null.mean()),
                          "null_sd": float(tri_null.std(ddof=1)),
                          "z": float((obs_tri - tri_null.mean()) /
                                     (tri_null.std(ddof=1) + 1e-12))},
            "wedges":    {"obs": obs_wed, "null_mean": float(wed_null.mean()),
                          "null_sd": float(wed_null.std(ddof=1)),
                          "z": float((obs_wed - wed_null.mean()) /
                                     (wed_null.std(ddof=1) + 1e-12))},
            "n_null": n_null,
            "method": "3-node motif Z-scores vs degree-preserving null"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 40
    # graph #1: Erdős-Rényi
    p = 0.15
    A_er = (rng.uniform(size=(n, n)) < p).astype(int)
    A_er = np.triu(A_er, 1); A_er = A_er + A_er.T

    # graph #2: 2-clique-of-cliques, triangle-heavy
    A_cl = np.zeros((n, n), dtype=int)
    for i in range(20):
        for j in range(i + 1, 20):
            A_cl[i, j] = A_cl[j, i] = 1
    for i in range(20, 40):
        for j in range(i + 1, 40):
            A_cl[i, j] = A_cl[j, i] = 1

    print("=== 3-node motif Z-scores vs degree-preserving null (n=40, 100 nulls) ===")
    for name, A in [("Erdős-Rényi",   A_er),
                    ("two-clique",    A_cl)]:
        r = motif_z_scores(A, n_null=100, swaps_per_edge=5, seed=1)
        print(f"\n  {name}:")
        print(f"    triangles  obs={r['triangles']['obs']:5d}   "
              f"null={r['triangles']['null_mean']:6.2f} +/- {r['triangles']['null_sd']:5.2f}"
              f"   Z={r['triangles']['z']:+7.2f}")
        print(f"    wedges     obs={r['wedges']['obs']:5d}   "
              f"null={r['wedges']['null_mean']:6.2f} +/- {r['wedges']['null_sd']:5.2f}"
              f"   Z={r['wedges']['z']:+7.2f}")

    print("\n--- library cross-check (Python graph-tool.clustering; R igraph::triad_census) ---")
