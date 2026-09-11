"""Dense Passage Retrieval - DPR (Reference Sec 47.188).

Karpukhin et al 2020 'Dense Passage Retrieval for Open-Domain
Question Answering', EMNLP. Two BERT encoders (question encoder
E_q, passage encoder E_p) trained with in-batch contrastive loss:

    sim(q, p) = E_q(q) . E_p(p)

Positive: (question, gold passage); negatives: other passages in
batch + optional hard negatives from BM25. Retrieval = FAISS
inner-product search over dense passage embeddings.

Beats sparse BM25 by ~10 pt Recall@20 on Natural Questions.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def encode_bag(text, vocab, W, d):
    """Toy encoder: mean of embed rows for tokens in text; L2-normalise."""
    toks = [t for t in text.lower().split() if t in vocab]
    if not toks: return np.zeros(d)
    v = np.mean([W[vocab[t]] for t in toks], axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def train_dpr(pairs, vocab, d=32, lr=0.05, n_epochs=200, seed=0):
    """Contrastive in-batch training of two encoders sharing embedding rows."""
    rng = np.random.default_rng(seed)
    Wq = rng.normal(scale=0.3, size=(len(vocab), d))
    Wp = rng.normal(scale=0.3, size=(len(vocab), d))
    for ep in range(n_epochs):
        # Encode all questions and passages in one 'batch'
        qs = np.array([encode_bag(q, vocab, Wq, d) for q, _ in pairs])
        ps = np.array([encode_bag(p, vocab, Wp, d) for _, p in pairs])
        # In-batch loss: positives on diagonal
        S = qs @ ps.T                                            # (N, N)
        S_exp = np.exp(S - S.max(axis=1, keepdims=True))
        soft = S_exp / S_exp.sum(axis=1, keepdims=True)
        target = np.eye(len(pairs))
        grad_S = (soft - target) / len(pairs)
        # Approximate gradient: push q_i toward p_i, away from p_j
        for i, (q, p) in enumerate(pairs):
            q_toks = [vocab[t] for t in q.lower().split() if t in vocab]
            p_toks = [vocab[t] for t in p.lower().split() if t in vocab]
            for t in q_toks:
                Wq[t] -= lr * (grad_S[i] @ ps) / len(q_toks)
            for j in range(len(pairs)):
                for t in p_toks:                                 # positive
                    Wp[t] -= lr * grad_S[i, j] * qs[i] / len(p_toks)
    return Wq, Wp


def recall_at_k(query_emb, passage_embs, true_idx, k):
    scores = passage_embs @ query_emb
    top = np.argsort(-scores)[:k]
    return int(true_idx in top)


if __name__ == "__main__":
    print("=== Dense Passage Retrieval - DPR (Karpukhin et al 2020) ===\n")

    # Small QA corpus: 8 (question, passage) pairs
    pairs = [
        ("who wrote 1984", "george orwell wrote 1984 in 1949"),
        ("what is the capital of france", "paris is the capital of france"),
        ("who painted the mona lisa", "leonardo da vinci painted the mona lisa"),
        ("what year did the moon landing happen", "apollo 11 landed on the moon in 1969"),
        ("who is the founder of microsoft", "bill gates and paul allen founded microsoft"),
        ("what is the speed of light", "light travels at 299792458 meters per second"),
        ("who wrote hamlet", "william shakespeare wrote hamlet"),
        ("what is the largest planet", "jupiter is the largest planet in the solar system"),
    ]
    vocab_list = sorted(set(w for q, p in pairs for w in (q + " " + p).lower().split()))
    vocab = {w: i for i, w in enumerate(vocab_list)}

    Wq, Wp = train_dpr(pairs, vocab, d=16, lr=0.5, n_epochs=200, seed=0)

    # Evaluate: for each question, is the true passage in top-k?
    passage_embs = np.array([encode_bag(p, vocab, Wp, 16) for _, p in pairs])
    for k in [1, 3]:
        acc = np.mean([recall_at_k(encode_bag(q, vocab, Wq, 16), passage_embs, i, k)
                          for i, (q, _) in enumerate(pairs)])
        print(f"  DPR Recall@{k} on {len(pairs)} pairs: {acc:.2f}")

    # BM25 baseline
    from collections import Counter
    from math import log
    def bm25_score(query, passage, corpus, k1=1.5, b=0.75):
        avgdl = np.mean([len(p.split()) for p in corpus])
        idf = {}
        for w in query.lower().split():
            df = sum(1 for p in corpus if w in p.lower())
            idf[w] = log((len(corpus) - df + 0.5) / (df + 0.5) + 1)
        toks = passage.lower().split()
        counts = Counter(toks); dl = len(toks)
        return sum(idf.get(w, 0) * counts[w] * (k1 + 1)
                     / (counts[w] + k1 * (1 - b + b * dl / avgdl))
                     for w in query.lower().split())

    corpus = [p for _, p in pairs]
    for k in [1, 3]:
        acc = np.mean([int(i in np.argsort([-bm25_score(q, p, corpus) for p in corpus])[:k])
                          for i, (q, _) in enumerate(pairs)])
        print(f"  BM25 Recall@{k}: {acc:.2f}")

    print("\n--- library cross-check (haystack / faiss / sentence-transformers Python) ---")
