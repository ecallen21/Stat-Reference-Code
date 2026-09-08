"""Reciprocal Rank Fusion (Reference Sec 47.131).

Cormack, Clarke & Buettcher 2009 'Reciprocal rank fusion
outperforms Condorcet and individual rank learning methods',
SIGIR. Simple unsupervised way to fuse multiple ranked lists:

    RRF(d) = sum_{r in R} 1 / (k + rank_r(d))     with k typically 60.

Robust in practice for hybrid dense + sparse retrieval (vector +
BM25). Insensitive to raw-score scales; treats each ranker as
producing an ordinal ranking only.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def rrf(rankings, k=60):
    """Given a list of ordered document-id lists, return dict of fused scores."""
    scores = {}
    for r in rankings:
        for rank, doc in enumerate(r, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def ndcg_at_k_binary(rels_in_order, k=None):
    r = np.asarray(rels_in_order, dtype=float)
    if k is None: k = len(r)
    disc = 1.0 / np.log2(np.arange(2, len(r[:k]) + 2))
    dcg = float(((2 ** r[:k] - 1) * disc).sum())
    ideal = np.sort(r)[::-1]
    idcg = float(((2 ** ideal[:k] - 1) * disc).sum())
    return dcg / max(idcg, 1e-12)


if __name__ == "__main__":
    print("=== Reciprocal Rank Fusion (Cormack-Clarke-Buettcher 2009) ===\n")
    rng = np.random.default_rng(0)

    N = 100
    # True gold relevance: 10 relevant items out of 100
    rel_set = set(rng.choice(N, size=10, replace=False).tolist())

    # Two rankers with different biases: BM25-like (weighted for content) vs vector-like
    def bm25_like():
        scores = rng.normal(size=N)
        # Boost half of the true positives
        boost_ids = rng.choice(list(rel_set), size=6, replace=False)
        for i in boost_ids: scores[i] += 3
        return list(np.argsort(-scores))
    def dense_like():
        scores = rng.normal(size=N)
        boost_ids = rng.choice(list(rel_set), size=6, replace=False)
        for i in boost_ids: scores[i] += 3
        return list(np.argsort(-scores))

    r_bm25 = bm25_like()
    r_dense = dense_like()
    fused = list(rrf([r_bm25, r_dense], k=60).keys())

    def eval_(order):
        top_rels = [1 if d in rel_set else 0 for d in order[:20]]
        return ndcg_at_k_binary(top_rels, k=10)

    print(f"  NDCG@10 -- BM25-like       = {eval_(r_bm25):.3f}")
    print(f"  NDCG@10 -- Dense-like      = {eval_(r_dense):.3f}")
    print(f"  NDCG@10 -- RRF fused       = {eval_(fused):.3f}")

    # Sensitivity to k
    print()
    for k in [1, 10, 60, 200]:
        fused = list(rrf([r_bm25, r_dense], k=k).keys())
        print(f"  RRF (k={k:3d})   NDCG@10 = {eval_(fused):.3f}")

    print("\n--- library cross-check (pytrec_eval / ranx Python; TSAR R) ---")
