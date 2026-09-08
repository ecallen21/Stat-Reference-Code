"""HNSW ANN search (Reference Sec 47.85).

Malkov & Yashunin 2018 'Efficient and robust approximate nearest
neighbor search using Hierarchical Navigable Small World graphs',
IEEE TPAMI 42(4). Multi-layer proximity graph:

    * Layer 0: full data, dense k-NN edges.
    * Higher layers: sparser random subsets, long-range edges.

Search descends from top layer via greedy graph traversal, keeping
`ef` nearest candidates at each step -- O(log n) query on average.
Empirically the leading ANN algorithm in ANN-Benchmarks for
recall > 0.9.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
import time    # wall-clock timing


def _brute_topk(X, q, k):
    d = ((X - q) ** 2).sum(-1)
    top = np.argsort(d)[:k]
    return set(top.tolist())


if __name__ == "__main__":
    print("=== HNSW ANN (Malkov-Yashunin 2018) ===\n")
    rng = np.random.default_rng(0)
    n, d = 50_000, 32
    X = rng.normal(size=(n, d)).astype(np.float32)
    Q = rng.normal(size=(100, d)).astype(np.float32)

    # Baseline: brute-force NN
    t0 = time.time()
    truth = [_brute_topk(X, q, 10) for q in Q]
    brute_time = time.time() - t0
    print(f"  Brute-force NN over {n} vectors, k=10, |Q|={len(Q)}: "
          f"{brute_time:.2f} s")

    try:
        import hnswlib    # canonical library
        HAVE_HNSWLIB = True
    except ImportError:
        HAVE_HNSWLIB = False

    if HAVE_HNSWLIB:
        print("\n  hnswlib HNSW (M=16):")
        for ef in [10, 50, 200]:
            idx = hnswlib.Index(space="l2", dim=d)
            idx.init_index(max_elements=n, ef_construction=200, M=16)
            idx.add_items(X, np.arange(n))
            idx.set_ef(ef)
            t0 = time.time()
            hits = 0
            for q, true in zip(Q, truth):
                lbls, dists = idx.knn_query(q, k=10)
                hits += len(set(lbls[0].tolist()) & true)
            recall = hits / (10 * len(Q))
            print(f"    ef = {ef:3d}   query time = {time.time()-t0:.2f} s   "
                  f"Recall@10 = {recall:.3f}")
    else:
        print("\n  hnswlib not installed -- fallback single-layer greedy proximity graph.")
        # Build a k-NN graph on X, then greedy-descend from a random seed
        from scipy.spatial import cKDTree
        tree = cKDTree(X)
        k_nn = 20
        _, nbrs = tree.query(X, k=k_nn + 1)
        nbrs = nbrs[:, 1:]        # drop self

        def greedy_search(q, ef):
            """Greedy graph search: maintain top-ef candidate list."""
            import heapq
            seed = int(rng.integers(n))
            visited = {seed}
            d0 = float(((X[seed] - q) ** 2).sum())
            candidates = [(d0, seed)]                   # min-heap (d, id)
            best = [(-d0, seed)]                        # max-heap on -d
            while candidates:
                d_c, c = heapq.heappop(candidates)
                if best and d_c > -best[0][0] and len(best) >= ef:
                    break
                for nb in nbrs[c]:
                    if nb in visited: continue
                    visited.add(int(nb))
                    d_nb = float(((X[nb] - q) ** 2).sum())
                    if len(best) < ef or d_nb < -best[0][0]:
                        heapq.heappush(candidates, (d_nb, int(nb)))
                        heapq.heappush(best, (-d_nb, int(nb)))
                        if len(best) > ef: heapq.heappop(best)
            top = sorted(best, key=lambda t: -t[0])[:10]
            return {i for _, i in top}

        for ef in [20, 100, 400]:
            t0 = time.time()
            hits = 0
            for q, true in zip(Q, truth):
                pred = greedy_search(q, ef)
                hits += len(pred & true)
            recall = hits / (10 * len(Q))
            print(f"    ef = {ef:3d}   query time = {time.time()-t0:.2f} s   "
                  f"Recall@10 = {recall:.3f}")

    print("\n--- library cross-check (RcppHNSW R; hnswlib / faiss / scann Python) ---")
