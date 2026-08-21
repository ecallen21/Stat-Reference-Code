"""Random-graph models: Erdős-Rényi, Watts-Strogatz, Barabási-Albert (Reference §24.4).

Three classical generators:

  * Erdős-Rényi G(n, p): each of C(n, 2) edges independent Bernoulli(p).
      -> Poisson degree with mean (n - 1) p; low clustering.
  * Watts-Strogatz(n, k, beta): start from ring lattice (2k neighbours);
      rewire each edge with probability beta.  -> high clustering + short paths
      for small beta ("small-world regime").
  * Barabási-Albert(n, m): grow by attaching each new node to m existing
      nodes with probability proportional to degree.
      -> power-law degree distribution (P(k) ~ k^-3 asymptotically).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def erdos_renyi(n: int, p: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = (rng.uniform(size=(n, n)) < p).astype(int)
    A = np.triu(A, 1); return A + A.T


def watts_strogatz(n: int, k: int, beta: float, seed: int = 0):
    """Ring lattice with 2k neighbours per node; rewire with prob beta."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for step in range(1, k + 1):
            A[i, (i + step) % n] = 1; A[(i + step) % n, i] = 1
    for i in range(n):
        for step in range(1, k + 1):
            j = (i + step) % n
            if rng.uniform() < beta:
                A[i, j] = A[j, i] = 0
                # rewire to a random new target
                candidates = np.where((A[i] == 0))[0]
                candidates = candidates[candidates != i]
                if candidates.size:
                    new_j = int(rng.choice(candidates))
                    A[i, new_j] = A[new_j, i] = 1
    return A


def barabasi_albert(n: int, m: int, seed: int = 0):
    """Preferential attachment.  Seed graph = m-clique."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=int)
    for i in range(m):
        for j in range(i + 1, m):
            A[i, j] = A[j, i] = 1
    degrees = A.sum(axis=1).astype(float)
    for new in range(m, n):
        probs = degrees[:new] / degrees[:new].sum()
        targets = rng.choice(new, size=m, replace=False, p=probs)
        for t in targets:
            A[new, t] = A[t, new] = 1
        degrees[targets] += 1; degrees[new] = m
    return A


def _clustering(A) -> float:
    A = np.asarray(A); n = A.shape[0]
    d = A.sum(1)
    tri_per = np.zeros(n)
    for i in range(n):
        nbrs = np.where(A[i])[0]
        if len(nbrs) < 2:
            continue
        sub = A[np.ix_(nbrs, nbrs)]
        tri_per[i] = sub.sum() / 2.0
    denom = d * (d - 1) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(denom > 0, tri_per / denom, 0.0)
    return float(c.mean())


if __name__ == "__main__":
    n = 200; k = 4; p = 2 * k / (n - 1)

    A_er = erdos_renyi(n, p, seed=1)
    A_ws = watts_strogatz(n, k, beta=0.05, seed=1)
    A_ba = barabasi_albert(n, m=k, seed=1)

    def _stats(A, name):
        d = A.sum(1)
        print(f"  {name:>18}: mean deg = {d.mean():.2f}  "
              f"max deg = {d.max()}  cluster = {_clustering(A):.4f}")

    print("=== Random graph models (n=200, target mean degree ~8) ===")
    _stats(A_er, "Erdős-Rényi")
    _stats(A_ws, "Watts-Strogatz")
    _stats(A_ba, "Barabási-Albert")

    # degree distribution tail for BA
    d_ba = A_ba.sum(1)
    counts = np.bincount(d_ba)
    print(f"\n  BA degree top-5: {sorted(d_ba, reverse=True)[:5]}")
    print(f"  BA fraction with deg > 20: "
          f"{(d_ba > 20).mean():.3f}  (ER: {(A_er.sum(1) > 20).mean():.3f})")

    print("\n--- library cross-check (networkx) ---")
    try:
        import networkx as nx
        Gnx = nx.watts_strogatz_graph(n, 2 * k, 0.05, seed=1)
        print(f"  nx Watts-Strogatz clustering = {nx.average_clustering(Gnx):.4f}")
        Gnx = nx.barabasi_albert_graph(n, k, seed=1)
        d_nx = np.array([d for _, d in Gnx.degree()])
        print(f"  nx BA max degree = {d_nx.max()}   fraction > 20 = "
              f"{(d_nx > 20).mean():.3f}")
    except ImportError:
        print("  (networkx not installed)")
