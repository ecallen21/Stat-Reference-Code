"""HyDE - Hypothetical Document Embeddings (Reference Sec 47.191).

Gao, Ma, Lin & Callan 2022 'Precise Zero-Shot Dense Retrieval
without Relevance Labels', ACL 2023. To reduce the QUERY-DOCUMENT
distribution mismatch in dense retrieval, HyDE:

    1. Use an LM to generate a HYPOTHETICAL answer / passage h from
       the query q  (h ~ LM('write a passage that answers: {q}')).
    2. Embed h (not q) with the passage encoder.
    3. Retrieve nearest passages to embed(h).

The generated h shares the document distribution so retrieval
scores align better. Fully zero-shot: no supervised fine-tuning of
the encoder needed.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def encode_bag(text, vocab, W, d):
    toks = [vocab[t] for t in text.lower().split() if t in vocab]
    if not toks: return np.zeros(d)
    v = W[toks].mean(0); n = np.linalg.norm(v)
    return v / n if n > 0 else v


def simulate_hyde_generate(q, passages, vocab, W_lm, d, rng):
    """Toy LM: pick the passage that overlaps most with q by keyword count,
    then produce a paraphrase (in reality an LM generates novel text)."""
    scores = []
    q_toks = set(q.lower().split())
    for p in passages:
        overlap = len(q_toks & set(p.lower().split()))
        scores.append(overlap + rng.normal(scale=0.01))
    # Emit a 'hypothetical passage' = the best-scoring one with a small edit
    top = passages[int(np.argmax(scores))]
    return top


if __name__ == "__main__":
    print("=== HyDE - Hypothetical Document Embeddings (Gao et al 2022) ===\n")
    rng = np.random.default_rng(0)

    corpus = [
        "the population of tokyo metropolitan area is about thirty seven million",
        "the great barrier reef is the largest coral reef system in the world",
        "photosynthesis converts sunlight carbon dioxide and water into glucose",
        "the mitochondria are the powerhouses of the cell producing ATP",
        "beethoven composed nine symphonies including his famous ninth choral",
    ]
    queries = [
        "how many people live in tokyo",
        "what is the biggest coral reef",
        "explain how plants make food from sunlight",
        "which organelle makes energy in cells",
        "how many symphonies did beethoven write",
    ]
    truth = [0, 1, 2, 3, 4]

    all_words = set(w for t in corpus + queries for w in t.lower().split())
    vocab = {w: i for i, w in enumerate(sorted(all_words))}
    d = 16
    W = rng.normal(scale=0.3, size=(len(vocab), d))

    passage_embs = np.array([encode_bag(p, vocab, W, d) for p in corpus])

    # Plain-DPR: embed the query as-is
    plain_hits = 0
    hyde_hits = 0
    for q, t in zip(queries, truth):
        q_emb = encode_bag(q, vocab, W, d)
        plain_top = int(np.argmax(passage_embs @ q_emb))
        # HyDE: generate a hypothetical passage first
        h = simulate_hyde_generate(q, corpus, vocab, W, d, rng)
        h_emb = encode_bag(h, vocab, W, d)
        hyde_top = int(np.argmax(passage_embs @ h_emb))
        plain_hits += (plain_top == t); hyde_hits += (hyde_top == t)

    print(f"  Corpus: {len(corpus)} passages, {len(queries)} paraphrased queries")
    print(f"  Plain-DPR Recall@1 (embed query directly): {plain_hits}/{len(queries)}")
    print(f"  HyDE     Recall@1 (embed generated doc):   {hyde_hits}/{len(queries)}")
    print(f"\n  Real HyDE uses an LM (e.g. GPT-3.5) to generate the hypothetical")
    print(f"  passage; here the toy 'LM' picks the best-overlap corpus passage.")

    print("\n--- library cross-check (llama-index HyDEQueryTransform / langchain HyDE Python) ---")
