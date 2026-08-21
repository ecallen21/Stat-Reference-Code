"""Bipartite networks: projection + bipartite modularity (Reference §24.10).

Bipartite graph B = (U, V, E): edges only between U and V.  Biadjacency
matrix B is |U| x |V|.

Two common one-mode projections:

  * Weighted projection on U: G_U = B B^T
      (entry [i, j] = # V-side neighbours shared by U-nodes i and j)
  * Hyperbolic (Newman) weighting: sum_{v ∈ N(i) ∩ N(j)} 1 / (deg(v) - 1)
      down-weights ubiquitous V-nodes.

Barber (2007) bipartite modularity Q_B for a partition assigning U-nodes to
communities c(u) and V-nodes to c(v):

    Q_B = (1 / m) * sum_{u, v} [ B_uv - k_u * d_v / m ] * 1{c(u) == c(v)}
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def project_weighted(B, side: str = "U"):
    """Weighted co-occurrence projection.  side='U' -> BB^T, side='V' -> B^T B."""
    B = np.asarray(B, dtype=float)
    G = B @ B.T if side == "U" else B.T @ B
    np.fill_diagonal(G, 0)
    return G


def project_newman(B, side: str = "U"):
    """Newman hyperbolic projection: shared V's down-weighted by 1/(deg(v)-1)."""
    B = np.asarray(B, dtype=float)
    if side != "U":
        B = B.T
    deg_v = B.sum(axis=0)
    w = np.where(deg_v > 1, 1.0 / (deg_v - 1), 0.0)
    # G[i,j] = sum_v B[i,v] * B[j,v] * w[v]
    G = (B * w) @ B.T
    np.fill_diagonal(G, 0)
    return G


def bipartite_modularity(B, labels_u, labels_v) -> float:
    B = np.asarray(B, dtype=float); m = B.sum()
    if m == 0:
        return 0.0
    ku = B.sum(axis=1); dv = B.sum(axis=0)
    P = np.outer(ku, dv) / m                              # null expectation
    same = (labels_u[:, None] == labels_v[None, :]).astype(float)
    return float(((B - P) * same).sum() / m)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 2-community bipartite: block-1 U's link to block-1 V's, etc.
    nU, nV = 8, 10
    U_block = np.array([0]*4 + [1]*4)
    V_block = np.array([0]*5 + [1]*5)
    B = np.zeros((nU, nV), dtype=int)
    for u in range(nU):
        for v in range(nV):
            p = 0.9 if U_block[u] == V_block[v] else 0.05
            B[u, v] = int(rng.uniform() < p)

    print("=== Bipartite network (|U|=8, |V|=10, 2 planted communities) ===")
    print(f"  biadjacency density = {B.mean():.3f}")

    Gu = project_weighted(B, "U")
    print(f"\n  weighted U-projection (BB^T):")
    print("   " + "\n   ".join("  ".join(f"{v:>3.0f}" for v in row) for row in Gu))

    Gu_n = project_newman(B, "U")
    print(f"\n  Newman-weighted U-projection (down-weight ubiquitous V):")
    print("   " + "\n   ".join("  ".join(f"{v:>5.2f}" for v in row) for row in Gu_n))

    Q = bipartite_modularity(B, U_block, V_block)
    print(f"\n  bipartite modularity of planted partition = {Q:.4f}")
    Q_rand = bipartite_modularity(B,
                                    rng.integers(0, 2, nU),
                                    rng.integers(0, 2, nV))
    print(f"  bipartite modularity of random partition  = {Q_rand:.4f}")

    print("\n--- library cross-check (networkx.bipartite; igraph::bipartite_projection) ---")
    try:
        import networkx as nx
        from networkx.algorithms import bipartite
        G = nx.Graph()
        G.add_nodes_from(range(nU), bipartite=0)
        G.add_nodes_from(range(nU, nU + nV), bipartite=1)
        for u in range(nU):
            for v in range(nV):
                if B[u, v]:
                    G.add_edge(u, nU + v)
        Gu_nx = bipartite.weighted_projected_graph(G, range(nU))
        w = sum(d["weight"] for _, _, d in Gu_nx.edges(data=True))
        print(f"  nx weighted projection total edge weight = {w}   "
              f"(scratch = {int(Gu[np.triu_indices(nU, 1)].sum())})")
    except ImportError:
        print("  (networkx not installed)")
