"""Kleinberg HITS: authorities and hubs (Reference §24.x extra).

Directed graph.  For each node compute two scores:

  * authority a_i = sum of hub scores of pages pointing TO i
  * hub       h_i = sum of authority scores of pages i points TO

Iterate:
  a <- A^T h,  normalise; h <- A a, normalise.
Converges to the leading eigenvector of A^T A (authority) and A A^T (hub).

Contrast:
  * PageRank: single stationary distribution of a random walk with damping.
  * HITS: two scores; a "good hub" points to many "good authorities".
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def hits(A, tol: float = 1e-9, max_iter: int = 1000, normalize: str = "l2") -> dict:
    A = np.asarray(A, dtype=float); n = A.shape[0]
    a = np.ones(n) / np.sqrt(n)
    h = np.ones(n) / np.sqrt(n)
    for it in range(max_iter):
        a_new = A.T @ h
        h_new = A @ a_new
        if normalize == "l2":
            a_new /= np.linalg.norm(a_new) or 1.0
            h_new /= np.linalg.norm(h_new) or 1.0
        else:                                              # sum-1
            a_new /= a_new.sum() or 1.0
            h_new /= h_new.sum() or 1.0
        if np.max(np.abs(a_new - a)) < tol and np.max(np.abs(h_new - h)) < tol:
            a, h = a_new, h_new; break
        a, h = a_new, h_new
    return {"authority": a, "hub": h, "n_iter": it + 1,
            "method": "HITS via power iteration"}


if __name__ == "__main__":
    # bipartite-ish directed graph:
    # nodes 0..4 are hubs (each points to authorities 5..9)
    # nodes 5..9 are authorities (each has many in-edges)
    n = 10
    A = np.zeros((n, n), dtype=int)
    # every hub points to every authority
    for h in range(5):
        for a in range(5, 10):
            A[h, a] = 1
    # add a few random hub-hub and authority-authority citations (weak)
    A[0, 1] = A[6, 7] = 1

    fit = hits(A)
    print(f"=== HITS on a 5-hub / 5-authority directed graph ===")
    print(f"  iterations to converge: {fit['n_iter']}")
    print(f"\n  {'node':>4} {'authority':>10} {'hub':>10}")
    for i in range(n):
        print(f"  {i:>4} {fit['authority'][i]:>10.4f} {fit['hub'][i]:>10.4f}")

    print(f"\n  top-3 by authority: {list(np.argsort(-fit['authority'])[:3])}")
    print(f"  top-3 by hub      : {list(np.argsort(-fit['hub'])[:3])}")

    print("\n--- library cross-check (networkx.hits) ---")
    try:
        import networkx as nx
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        h_nx, a_nx = nx.hits(G, normalized=True, tol=1e-9, max_iter=1000)
        # normalize scratch to sum=1 for comparison
        a_sum = fit["authority"] / fit["authority"].sum()
        h_sum = fit["hub"] / fit["hub"].sum()
        max_diff_a = max(abs(a_sum[i] - a_nx[i]) for i in range(n))
        max_diff_h = max(abs(h_sum[i] - h_nx[i]) for i in range(n))
        print(f"  max diff authority (scratch vs nx) = {max_diff_a:.2e}")
        print(f"  max diff hub       (scratch vs nx) = {max_diff_h:.2e}")
    except ImportError:
        print("  (networkx not installed)")
