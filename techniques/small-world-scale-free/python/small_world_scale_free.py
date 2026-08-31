"""Small-world + scale-free network models (Reference Sec 30.12).

Watts & Strogatz (1998) 'Collective dynamics of small-world networks.'
Barabasi & Albert (1999) 'Emergence of scaling in random networks.'

  Small-world (Watts-Strogatz): start from a ring lattice; rewire each
    edge with prob p to a random node. High clustering + short path
    lengths -- like real social networks.

  Scale-free (Barabasi-Albert): grow the network by preferential
    attachment; a new node with m edges attaches to existing nodes
    with prob proportional to their degree. Degree distribution is a
    POWER LAW P(k) ~ k^{-3}.

Here we generate both and compute the signature statistics:
  * clustering coefficient
  * average shortest-path length
  * degree distribution slope on log-log plot
Contrast against Erdos-Renyi baseline.
"""
from __future__ import annotations    # stdlib

from collections import deque

import numpy as np    # numerical arrays


def ring_lattice(n, k):
    """k-neighbour ring lattice: each node connects to its k nearest
    neighbours on the ring."""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(1, k // 2 + 1):
            A[i, (i + j) % n] = A[(i + j) % n, i] = 1
    return A


def watts_strogatz(n, k, p, seed=0):
    """WS small-world: rewire each edge of a k-ring with prob p."""
    rng = np.random.default_rng(seed)
    A = ring_lattice(n, k)
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if rng.random() < p:
                # Choose a random new neighbour that isn't self / duplicate.
                old = (i + j) % n
                new = rng.integers(0, n)
                if new != i and A[i, new] == 0:
                    A[i, old] = A[old, i] = 0
                    A[i, new] = A[new, i] = 1
    return A


def barabasi_albert(n, m, seed=0):
    """Grow BA graph: n nodes, each new node attaches to m existing by pref attachment."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))
    # Start with m+1 complete graph
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            A[i, j] = A[j, i] = 1
    for t in range(m + 1, n):
        deg = A.sum(axis=1)[:t]
        prob = deg / deg.sum()
        targets = rng.choice(t, size=m, replace=False, p=prob)
        for u in targets:
            A[t, u] = A[u, t] = 1
    return A


def erdos_renyi(n, p, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.random((n, n)) < p
    A = np.triu(U, 1).astype(float)
    return A + A.T


def clustering_coefficient(A):
    n = A.shape[0]
    tri = 0; triples = 0
    for i in range(n):
        nbrs = np.where(A[i] > 0)[0]
        k = len(nbrs)
        if k < 2: continue
        triples += k * (k - 1) / 2
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                if A[nbrs[a], nbrs[b]] > 0:
                    tri += 1
    return tri / max(triples, 1)


def average_shortest_path(A):
    n = A.shape[0]
    total, count = 0.0, 0
    for src in range(n):
        d = -np.ones(n); d[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in np.where(A[u] > 0)[0]:
                if d[v] < 0:
                    d[v] = d[u] + 1
                    q.append(v)
        for v in range(n):
            if v != src and d[v] > 0:
                total += d[v]; count += 1
    return total / max(count, 1)


def degree_power_law_slope(A):
    """Fit log P(k >= x) ~ - alpha log x for the tail of the degree distribution."""
    deg = A.sum(axis=1).astype(int)
    vals = np.sort(deg)[::-1]
    if len(vals) < 3: return float("nan")
    x = np.log(np.arange(1, len(vals) + 1))
    y = np.log(vals + 1e-9)
    m = np.mean(x * y) - np.mean(x) * np.mean(y)
    v = np.mean(x ** 2) - np.mean(x) ** 2
    return -float(m / max(v, 1e-9))


if __name__ == "__main__":
    print("=== Small-world + scale-free network models ===\n")
    n = 60
    print(f"  Graph size n = {n}\n")
    print(f"  {'model':>20}  {'clustering':>11}  {'avg path':>9}  {'degree exponent':>16}")
    for label, A in [
        ("Erdos-Renyi p=0.10", erdos_renyi(n, 0.10, seed=0)),
        ("Ring lattice k=6", ring_lattice(n, 6)),
        ("Watts-Strogatz p=0.05", watts_strogatz(n, 6, 0.05, seed=0)),
        ("Watts-Strogatz p=0.30", watts_strogatz(n, 6, 0.30, seed=0)),
        ("Barabasi-Albert m=3", barabasi_albert(n, 3, seed=0)),
    ]:
        C = clustering_coefficient(A)
        L = average_shortest_path(A)
        alpha = degree_power_law_slope(A)
        print(f"  {label:>20}  {C:>11.3f}  {L:>9.2f}  {alpha:>16.2f}")

    print("\n  Watts-Strogatz (small p) has high clustering + short paths -- 'small world'.")
    print("  Barabasi-Albert has a power-law tail (steep negative slope on log-log).\n")
    print("--- library cross-check (networkx.generators.random_graphs; R igraph::sample_smallworld / sample_pa) ---")
