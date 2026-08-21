"""Centrality measures for graphs (Reference §24.2).

Five classical centralities on an undirected adjacency matrix A:

  * degree_c(v)      = deg(v) / (n - 1)
  * closeness_c(v)   = (n - 1) / sum_u d(v, u)          (u reachable)
  * betweenness_c(v) = sum_{s != v != t} sigma_{st}(v) / sigma_{st}   (Brandes)
  * eigenvector_c    = leading eigenvector of A          (Perron)
  * katz_c           = (I - alpha * A^T)^{-1} * 1        with alpha < 1/rho(A)
  * pagerank_c       = power iteration of the stochastic matrix
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

from collections import deque    # stdlib: FIFO queue for BFS

import numpy as np    # numerical arrays + linear algebra


def degree_centrality(A):
    A = np.asarray(A); n = A.shape[0]
    return A.sum(axis=1) / (n - 1)


def closeness_centrality(A):
    A = np.asarray(A); n = A.shape[0]
    c = np.zeros(n)
    for s in range(n):
        dist = np.full(n, -1); dist[s] = 0; q = deque([s])
        while q:
            u = q.popleft()
            for v in np.where(A[u])[0]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1; q.append(v)
        reachable = dist[dist > 0]
        if reachable.size:
            c[s] = reachable.size / reachable.sum()
    return c


def betweenness_centrality(A):
    """Brandes (2001) algorithm for unweighted graphs."""
    A = np.asarray(A); n = A.shape[0]
    Cb = np.zeros(n)
    for s in range(n):
        S = []
        P = [[] for _ in range(n)]
        sigma = np.zeros(n); sigma[s] = 1
        dist = np.full(n, -1); dist[s] = 0
        q = deque([s])
        while q:
            v = q.popleft(); S.append(v)
            for w in np.where(A[v])[0]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1; q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]; P[w].append(v)
        delta = np.zeros(n)
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                Cb[w] += delta[w]
    Cb /= 2.0                                             # undirected: each pair counted twice
    # normalise to [0, 1]: max possible = (n-1)(n-2)/2
    if n > 2:
        Cb /= (n - 1) * (n - 2) / 2.0
    return Cb


def eigenvector_centrality(A, tol: float = 1e-9, max_iter: int = 1000):
    A = np.asarray(A, dtype=float); n = A.shape[0]
    x = np.ones(n) / np.sqrt(n)
    for _ in range(max_iter):
        y = A @ x
        norm = np.linalg.norm(y)
        if norm == 0:
            return x
        y = y / norm
        if np.linalg.norm(y - x) < tol:
            return y
        x = y
    return x


def katz_centrality(A, alpha: float = None, beta: float = 1.0):
    A = np.asarray(A, dtype=float); n = A.shape[0]
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    if alpha is None:
        alpha = 0.9 / rho
    v = np.linalg.solve(np.eye(n) - alpha * A.T, beta * np.ones(n))
    return v / np.linalg.norm(v)


def pagerank(A, damping: float = 0.85, tol: float = 1e-9, max_iter: int = 500):
    """Undirected PageRank via power iteration on the transition matrix."""
    A = np.asarray(A, dtype=float); n = A.shape[0]
    d = A.sum(axis=1)
    d[d == 0] = 1                                        # dangling → self-uniform
    M = (A.T / d)                                        # column-stochastic transition
    r = np.ones(n) / n
    for _ in range(max_iter):
        r_new = damping * (M @ r) + (1 - damping) / n
        if np.abs(r_new - r).sum() < tol:
            return r_new
        r = r_new
    return r


if __name__ == "__main__":
    # Karate-club-esque toy: two 5-cliques joined by a bridge (7-8)
    n = 10
    A = np.zeros((n, n), dtype=int)
    for i in range(5):
        for j in range(i + 1, 5):
            A[i, j] = A[j, i] = 1
    for i in range(5, 10):
        for j in range(i + 1, 10):
            A[i, j] = A[j, i] = 1
    A[4, 5] = A[5, 4] = 1                                # bridge

    print("=== Centrality of a two-clique bridge graph ===")
    print(f"  {'v':>2} {'deg':>6} {'clos':>6} {'betw':>6} {'eig':>6} "
          f"{'katz':>6} {'pr':>6}")
    dc = degree_centrality(A)
    cc = closeness_centrality(A)
    bc = betweenness_centrality(A)
    ec = eigenvector_centrality(A)
    kc = katz_centrality(A)
    pr = pagerank(A)
    for v in range(n):
        print(f"  {v:>2} {dc[v]:>6.3f} {cc[v]:>6.3f} {bc[v]:>6.3f} "
              f"{ec[v]:>6.3f} {kc[v]:>6.3f} {pr[v]:>6.3f}")

    top_bet = int(np.argmax(bc))
    print(f"\n  highest betweenness = node {top_bet} (expected bridge node 4 or 5): "
          f"C_B = {bc[top_bet]:.4f}")

    print("\n--- library cross-check (networkx) ---")
    try:
        import networkx as nx
        G = nx.from_numpy_array(A)
        bc_nx = nx.betweenness_centrality(G)
        ec_nx = nx.eigenvector_centrality_numpy(G)
        pr_nx = nx.pagerank(G)
        print(f"  scratch top betw = {bc[top_bet]:.4f}   nx = {bc_nx[top_bet]:.4f}")
        print(f"  scratch eig[0]   = {ec[0]:.4f}   nx = {ec_nx[0]:.4f}")
        print(f"  scratch pr[0]    = {pr[0]:.4f}   nx = {pr_nx[0]:.4f}")
    except ImportError:
        print("  (networkx not installed)")
