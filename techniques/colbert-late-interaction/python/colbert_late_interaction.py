"""ColBERT - Contextualised Late Interaction over BERT (Sec 47.189).

Khattab & Zaharia 2020 'ColBERT: Efficient and Effective Passage
Search via Contextualized Late Interaction over BERT', SIGIR.
Unlike DPR (one vector per doc), ColBERT keeps a VECTOR PER TOKEN
and scores at query time with a LATE-INTERACTION MaxSim operator:

    score(q, p) = sum_{q_i in q}  max_{p_j in p} <q_i, p_j>.

Retrieval keeps token-level granularity (better recall on rare
terms) at the price of much larger index size. ColBERTv2
(Santhanam 2022) quantises tokens down to 32 bytes each.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def encode_tokens(text, vocab, W):
    """Return an array of L2-normalised token embeddings for the text."""
    toks = [vocab[t] for t in text.lower().split() if t in vocab]
    if not toks: return np.zeros((0, W.shape[1]))
    embs = W[toks]
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-8)


def colbert_score(q_embs, p_embs):
    """MaxSim: sum_i max_j <q_i, p_j>."""
    if len(q_embs) == 0 or len(p_embs) == 0: return 0.0
    sim = q_embs @ p_embs.T                                     # (|q|, |p|)
    return float(sim.max(axis=1).sum())


def dpr_score(q_embs, p_embs):
    """Single-vector cosine: mean-pool then dot."""
    if len(q_embs) == 0 or len(p_embs) == 0: return 0.0
    q_vec = q_embs.mean(axis=0); q_vec /= max(np.linalg.norm(q_vec), 1e-8)
    p_vec = p_embs.mean(axis=0); p_vec /= max(np.linalg.norm(p_vec), 1e-8)
    return float(q_vec @ p_vec)


if __name__ == "__main__":
    print("=== ColBERT Late Interaction (Khattab-Zaharia 2020) ===\n")
    rng = np.random.default_rng(0)

    # Corpus with a rare-term challenge: mostly filler, key term matters
    corpus = [
        "the movie is fun and enjoyable and full of surprises for everyone",
        "the film about quantum entanglement in physics is beautifully explained",
        "the sports match was intense with dramatic comebacks and last-minute goals",
        "the recipe requires fresh basil, tomato, mozzarella and olive oil",
        "the algorithm uses backpropagation to train a deep neural network",
    ]
    queries = [
        "quantum entanglement",                                  # matches doc 1
        "backpropagation",                                       # matches doc 4
        "recipe basil",                                          # matches doc 3
    ]
    truth = [1, 4, 3]

    # Random per-token embeddings (as if from a small LM)
    all_words = set(w for t in corpus + queries for w in t.lower().split())
    vocab = {w: i for i, w in enumerate(sorted(all_words))}
    W = rng.normal(scale=0.5, size=(len(vocab), 32))

    corpus_embs = [encode_tokens(t, vocab, W) for t in corpus]
    print(f"  Corpus: {len(corpus)} passages ({sum(len(e) for e in corpus_embs)} total tokens)")
    print(f"  Vocab: {len(vocab)} words, 32-D per-token embeddings")

    print(f"\n  {'query':<25}  {'DPR mean-pool top':<20}  {'ColBERT MaxSim top':<20}  {'truth'}")
    dpr_hits = 0; col_hits = 0
    for q, t in zip(queries, truth):
        q_embs = encode_tokens(q, vocab, W)
        dpr_scores = [dpr_score(q_embs, pe) for pe in corpus_embs]
        col_scores = [colbert_score(q_embs, pe) for pe in corpus_embs]
        d_top, c_top = int(np.argmax(dpr_scores)), int(np.argmax(col_scores))
        dpr_hits += (d_top == t); col_hits += (c_top == t)
        print(f"  {q:<25}  {corpus[d_top][:18]:<20}  {corpus[c_top][:18]:<20}  {corpus[t][:15]}")

    print(f"\n  Retrieval accuracy: DPR mean-pool {dpr_hits}/{len(queries)}, "
          f"ColBERT MaxSim {col_hits}/{len(queries)}")
    print(f"  Random-init embeddings favour ColBERT because MaxSim finds the ONE token")
    print(f"  match (e.g. 'quantum') even when the rest of the passage is noise.")

    print("\n--- library cross-check (colbert-ai / ragatouille / pyserini) ---")
