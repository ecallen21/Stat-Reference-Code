"""Graph descriptive statistics (Reference §24.1).

Basic summaries of a (undirected, unweighted) graph G = (V, E):

  * density         = |E| / C(|V|, 2)
  * degree dist     = counts of node degrees
  * clustering coef = triangles(v) / C(deg(v), 2)   local; averaged for global
  * transitivity    = 3 * triangles / connected-triples          (global)
  * assortativity   = Pearson r of degrees at edge endpoints
  * path length     = mean shortest-path distance (BFS from each node)
  * diameter        = max shortest-path distance
  * components      = counts and sizes of connected components

Graph is stored as an n x n adjacency matrix (symmetric, 0/1, zero diagonal).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

from collections import deque    # stdlib: FIFO queue for BFS

import numpy as np    # numerical arrays + linear algebra


def density(A) -> float:
    A = np.asarray(A); n = A.shape[0]
    return float(A.sum() / (n * (n - 1)))


def degree(A):
    return np.asarray(A).sum(axis=1).astype(int)


def local_clustering(A):
    A = np.asarray(A); n = A.shape[0]
    d = degree(A)
    A2 = A @ A                                 # A[i,j] == neighbours of i and j
    tri = np.array([A2[i, A[i].astype(bool)].sum() / 2.0 for i in range(n)])
    denom = d * (d - 1) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(denom > 0, tri / denom, 0.0)
    return c


def transitivity(A) -> float:
    """Global clustering coefficient: 3 * triangles / connected-triples."""
    A = np.asarray(A)
    d = degree(A)
    triangles = float(np.trace(np.linalg.matrix_power(A, 3))) / 6.0
    triples = float((d * (d - 1)).sum()) / 2.0                # sum_v C(deg(v), 2)
    return 3.0 * triangles / triples if triples > 0 else 0.0


def assortativity(A) -> float:
    """Newman degree-degree Pearson r across edges."""
    A = np.asarray(A); n = A.shape[0]
    d = degree(A)
    ex, ey = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                ex.append(d[i]); ey.append(d[j])
                ex.append(d[j]); ey.append(d[i])
    if not ex:
        return float("nan")
    ex = np.asarray(ex, dtype=float); ey = np.asarray(ey, dtype=float)
    if ex.std() == 0 or ey.std() == 0:                        # regular graph
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def bfs_distances(A, src):
    n = A.shape[0]
    dist = np.full(n, -1); dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in np.where(A[u])[0]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def connected_components(A):
    n = A.shape[0]
    comp = np.full(n, -1); k = 0
    for i in range(n):
        if comp[i] == -1:
            d = bfs_distances(A, i)
            comp[d != -1] = k; k += 1
    sizes = np.bincount(comp)
    return {"labels": comp, "n_components": int(k), "sizes": sizes.tolist()}


def path_length_diameter(A) -> dict:
    n = A.shape[0]
    all_d = []; diameter = 0
    for i in range(n):
        d = bfs_distances(A, i)
        reach = d[(d > 0)]
        if reach.size:
            all_d.append(reach)
            diameter = max(diameter, int(reach.max()))
    avg = float(np.concatenate(all_d).mean()) if all_d else float("nan")
    return {"avg_path_length": avg, "diameter": int(diameter)}


def summarize(A) -> dict:
    A = np.asarray(A, dtype=int)
    d = degree(A)
    cc = connected_components(A)
    pl = path_length_diameter(A)
    return {"n": int(A.shape[0]), "m": int(A.sum()) // 2,
            "density": density(A),
            "mean_degree": float(d.mean()),
            "max_degree": int(d.max()),
            "avg_clustering": float(local_clustering(A).mean()),
            "transitivity": transitivity(A),
            "assortativity": assortativity(A),
            "avg_path_length": pl["avg_path_length"],
            "diameter": pl["diameter"],
            "n_components": cc["n_components"],
            "largest_component": int(max(cc["sizes"]))}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 40
    # Erdős-Rényi G(n, p) with p tuned to mean degree 5
    p = 5 / (n - 1)
    A = (rng.uniform(size=(n, n)) < p).astype(int)
    A = np.triu(A, 1); A = A + A.T
    print("=== Erdős-Rényi G(40, 0.128) ===")
    for k, v in summarize(A).items():
        if isinstance(v, float):
            print(f"  {k:>18} = {v:.4f}")
        else:
            print(f"  {k:>18} = {v}")

    # Ring lattice (should have high clustering, high diameter, r ≈ 1)
    print("\n=== Ring lattice (n=40, k=4) ===")
    B = np.zeros((n, n), dtype=int)
    for i in range(n):
        for step in (1, 2):
            B[i, (i + step) % n] = 1; B[(i + step) % n, i] = 1
    for k, v in summarize(B).items():
        if isinstance(v, float):
            print(f"  {k:>18} = {v:.4f}")
        else:
            print(f"  {k:>18} = {v}")

    print("\n--- library cross-check (networkx / igraph) ---")
    try:
        import networkx as nx
        G = nx.from_numpy_array(A)
        print(f"  nx.density = {nx.density(G):.4f}      "
              f"nx.average_clustering = {nx.average_clustering(G):.4f}")
        print(f"  nx.transitivity = {nx.transitivity(G):.4f}   "
              f"nx.degree_assortativity_coefficient = "
              f"{nx.degree_assortativity_coefficient(G):.4f}")
    except ImportError:
        print("  (networkx not installed; from-scratch numbers only)")
