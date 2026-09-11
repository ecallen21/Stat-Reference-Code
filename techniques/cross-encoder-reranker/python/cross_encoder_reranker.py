"""Cross-Encoder Reranker (Reference Sec 47.190).

Nogueira & Cho 2019 'Passage Re-ranking with BERT'; Wang et al 2020
'MiniLM'. Bi-encoder retrieval (DPR, ColBERT) is fast but shallow:
q and p are encoded INDEPENDENTLY. Cross-encoders re-rank the top-K
by feeding [q; SEP; p] through a single Transformer:

    score(q, p) = MLP(BERT([q; SEP; p]))

Much more accurate (attention across q and p) but O(K) forward
passes per query, so used only for reranking the top ~100 from a
cheap first-stage retriever.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def bi_encoder_score(q, p, vocab, W):
    """Independent mean-of-embeds dot product."""
    def enc(t):
        toks = [vocab[w] for w in t.lower().split() if w in vocab]
        if not toks: return np.zeros(W.shape[1])
        v = W[toks].mean(0); n = np.linalg.norm(v)
        return v / n if n > 0 else v
    return float(enc(q) @ enc(p))


def cross_encoder_score(q, p, vocab, W, W_pair):
    """Toy cross-encoder: bag-of-cross-terms feature -> linear score.

    For each (q_i, p_j) token pair, look up interaction weight; sum.
    """
    q_toks = [vocab[w] for w in q.lower().split() if w in vocab]
    p_toks = [vocab[w] for w in p.lower().split() if w in vocab]
    if not q_toks or not p_toks: return 0.0
    q_emb = W[q_toks]                                           # (|q|, d)
    p_emb = W[p_toks]                                           # (|p|, d)
    # Interaction: q_i . (W_pair @ p_j)  averaged
    return float(np.mean(q_emb @ W_pair @ p_emb.T))


def ndcg_at_k(ranked_labels, k):
    """NDCG using binary relevance."""
    from math import log2
    dcg = sum((r / log2(i + 2)) for i, r in enumerate(ranked_labels[:k]))
    ideal = sum(1 / log2(i + 2) for i in range(sum(ranked_labels[:k])))
    return dcg / max(ideal, 1e-12)


if __name__ == "__main__":
    print("=== Cross-Encoder Reranker (Nogueira-Cho 2019) ===\n")
    rng = np.random.default_rng(0)

    # 20-passage corpus: 5 golden passages with keyword; 15 distractors
    corpus = [f"gold passage about the important topic of kw{i}" for i in range(5)] + \
              [f"other passage about topic {i} with unrelated content" for i in range(15)]
    queries = [f"what is kw{i}" for i in range(5)]
    golds = [i for i in range(5)]

    all_words = set(w for t in corpus + queries for w in t.lower().split())
    vocab = {w: i for i, w in enumerate(sorted(all_words))}
    d = 8
    W = rng.normal(scale=0.1, size=(len(vocab), d))
    # Give each kw its own basis direction (simulate trained embeddings)
    for k in range(5):
        idx = vocab[f"kw{k}"]
        W[idx] = 0
        W[idx, k % d] = 3.0
    # Cross-encoder pair-interaction: identity boosts same-basis matches
    W_pair = np.eye(d) * 2.0 + rng.normal(scale=0.2, size=(d, d))

    print(f"  Corpus: {len(corpus)} passages, {len(queries)} queries, gold at i = query_id")

    # Bi-encoder scoring
    print(f"\n  {'query':<25}  {'bi-top1':<15}  {'cross-top1':<15}  {'gold'}")
    bi_r1 = 0; cx_r1 = 0
    for qi, q in enumerate(queries):
        bi_scores = [bi_encoder_score(q, p, vocab, W) for p in corpus]
        top_k = np.argsort(-np.array(bi_scores))[:5]              # first-stage top-5
        cx_rerank = sorted(top_k, key=lambda i: -cross_encoder_score(q, corpus[i], vocab, W, W_pair))
        bi_top = int(np.argmax(bi_scores)); cx_top = cx_rerank[0]
        bi_r1 += (bi_top == golds[qi]); cx_r1 += (cx_top == golds[qi])
        print(f"  {q:<25}  {'p' + str(bi_top):<15}  {'p' + str(cx_top):<15}  p{golds[qi]}")

    print(f"\n  Recall@1: bi-encoder {bi_r1}/{len(queries)}   cross-encoder-rerank {cx_r1}/{len(queries)}")

    print("\n--- library cross-check (sentence-transformers CrossEncoder / MiniLM / MonoT5) ---")
