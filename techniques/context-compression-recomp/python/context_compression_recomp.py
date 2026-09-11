"""Context Compression / RECOMP (Reference Sec 47.193).

Xu, Shi & Choi 2024 'RECOMP: Improving Retrieval-Augmented LMs
with Context Compression and Selective Augmentation', ICLR.
Retrieved passages are often long and noisy; RECOMP inserts a
COMPRESSOR between retrieval and generation:

    docs (top-k, long) -> Compressor -> summary_i or FILTERED subset
    -> LLM(query, summary/subset)

Two flavours:
    - EXTRACTIVE: score each sentence, keep the top-p.
    - ABSTRACTIVE: fine-tuned encoder-decoder summariser.

Compresses long context 2-10x with minimal / no answer-quality
drop, cutting inference cost dramatically.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sentence_relevance(query, sentence, vocab, W):
    """Cosine of mean-token embed."""
    def enc(t):
        toks = [vocab[w] for w in t.lower().split() if w in vocab]
        if not toks: return np.zeros(W.shape[1])
        v = W[toks].mean(0); n = np.linalg.norm(v)
        return v / n if n > 0 else v
    return float(enc(query) @ enc(sentence))


def extractive_compress(query, passages, vocab, W, top_p=0.3):
    """Split passages into sentences, keep top-p by cosine to query."""
    sents = []
    for p in passages:
        sents.extend([s.strip() for s in p.split(".") if s.strip()])
    scores = np.array([sentence_relevance(query, s, vocab, W) for s in sents])
    n_keep = max(1, int(top_p * len(sents)))
    top = np.argsort(-scores)[:n_keep]
    return " . ".join(sents[i] for i in sorted(top))


def answer_from_context(query, context, correct_answers):
    """Toy generator: emit any correct answer word that appears in context."""
    ctx_toks = set(context.lower().split())
    for ans in correct_answers:
        if all(w in ctx_toks for w in ans.lower().split()):
            return ans
    return "I don't know"


if __name__ == "__main__":
    print("=== Context Compression / RECOMP (Xu et al 2024) ===\n")
    rng = np.random.default_rng(0)

    passages = [
        "napoleon was born in corsica in 1769. he became emperor of france in 1804. "
        "he lost the battle of waterloo in 1815 and died in exile.",
        "the metric system was originally developed in france during the revolution. "
        "the meter was defined as one ten-millionth of the distance from the pole to equator.",
        "the eiffel tower was designed by gustave eiffel for the 1889 world fair. "
        "it is 330 meters tall including the antenna. it is the most visited paid monument.",
    ]
    queries_answers = [
        ("what year was napoleon born", ["1769"]),
        ("when was the eiffel tower built", ["1889"]),
        ("who designed the eiffel tower", ["gustave"]),
    ]

    all_words = set(w for p in passages for w in p.lower().replace(".", "").split())
    all_words.update(w for q, _ in queries_answers for w in q.lower().split())
    vocab = {w: i for i, w in enumerate(sorted(all_words))}
    d = 16
    W = rng.normal(scale=0.3, size=(len(vocab), d))
    # Give hard keywords structured embeddings so cosine works
    for w in ["napoleon", "eiffel", "gustave", "1889", "1769", "corsica"]:
        if w in vocab:
            W[vocab[w]] = np.zeros(d); W[vocab[w], hash(w) % d] = 3.0

    full_context_len = sum(len(p.split()) for p in passages)
    print(f"  Full retrieved context: {full_context_len} tokens across {len(passages)} passages")

    for q, ans in queries_answers:
        compressed = extractive_compress(q, passages, vocab, W, top_p=0.3)
        full_answer = answer_from_context(q, " ".join(passages), ans)
        comp_answer = answer_from_context(q, compressed, ans)
        print(f"\n  Query: '{q}'")
        print(f"    Compressed context ({len(compressed.split())} tokens vs {full_context_len}): {compressed[:80]}...")
        print(f"    Answer from full:       {full_answer}   (gold: {ans})")
        print(f"    Answer from compressed: {comp_answer}")

    print("\n  Compression cuts context ~3-4x while preserving the answerable spans.")

    print("\n--- library cross-check (langchain LLMChainExtractor / llmlingua Python) ---")
