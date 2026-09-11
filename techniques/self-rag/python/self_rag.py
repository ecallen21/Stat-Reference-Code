"""Self-RAG (Reference Sec 47.192).

Asai, Wu, Wang, Sil & Hajishirzi 2024 'Self-RAG: Learning to
Retrieve, Generate, and Critique through Self-Reflection', ICLR.
The LM learns REFLECTION TOKENS that let it:

    Retrieve?           yes / no  (decide when to fetch context)
    IsRel(d, q)?        yes / partial / irrelevant per passage
    IsSup(gen, d)?      supported / partially / not-supported
    IsUse(gen)          overall utility 1-5

At inference the model generates candidate answers with different
retrieved subsets, scored by the reflection tokens, and picks the
best. Beats standard RAG on ambiguous or misleading contexts.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def retrieve(query, corpus, k=3):
    """Toy retrieval: BM25-lite ranking by keyword overlap."""
    q_toks = set(query.lower().split())
    scores = [len(q_toks & set(p.lower().split())) for p in corpus]
    return list(np.argsort(-np.array(scores))[:k])


def is_relevant(query, passage, threshold=1):
    """Reflection token IsRel."""
    q_toks = set(query.lower().split())
    p_toks = set(passage.lower().split())
    overlap = len(q_toks & p_toks)
    if overlap >= threshold + 1: return "yes"
    if overlap == threshold: return "partial"
    return "irrelevant"


def generate_from_passage(query, passage, correct_answers):
    """Toy generator: if passage contains any correct-answer keyword, emit it."""
    for ans in correct_answers:
        if any(w in passage.lower() for w in ans.lower().split()):
            return ans
    return "I don't know"


def is_supported(gen, passage):
    """Reflection token IsSup."""
    if gen == "I don't know": return "not-supported"
    if any(w in passage.lower() for w in gen.lower().split()):
        return "supported"
    return "partially"


def utility_score(gen):
    """Reflection token IsUse: 1-5."""
    if gen == "I don't know": return 1
    return 5


def self_rag_answer(query, corpus, correct_answers, k=3):
    """Retrieve top-k, generate + score for each, pick highest utility supported."""
    docs = retrieve(query, corpus, k=k)
    best = None; best_score = -np.inf
    critiques = []
    for d in docs:
        p = corpus[d]
        rel = is_relevant(query, p)
        if rel == "irrelevant": continue
        gen = generate_from_passage(query, p, correct_answers)
        sup = is_supported(gen, p)
        u = utility_score(gen)
        score = u + (2 if sup == "supported" else 0) + (1 if rel == "yes" else 0)
        critiques.append({"doc": d, "gen": gen, "rel": rel, "sup": sup, "u": u, "score": score})
        if score > best_score:
            best_score = score; best = gen
    return best or "I don't know", critiques


def plain_rag_answer(query, corpus, correct_answers, k=1):
    """Plain RAG: take top-1 retrieval, generate, return."""
    docs = retrieve(query, corpus, k=k)
    if not docs: return "I don't know"
    return generate_from_passage(query, corpus[docs[0]], correct_answers)


if __name__ == "__main__":
    print("=== Self-RAG (Asai et al 2024) ===\n")

    corpus = [
        "the eiffel tower is in paris france and was built in 1889",
        "the great wall of china stretches over thirteen thousand miles",
        "the eiffel year 2024 saw record visitors for the tower",   # misleading numeric
        "the leaning tower of pisa is located in italy",
        "france is a country in western europe with paris as its capital",
    ]
    queries_answers = [
        ("what year was the eiffel tower built", ["1889"]),
        ("where is the eiffel tower located", ["paris"]),
        ("what is the capital of france", ["paris"]),
    ]

    print(f"  {'query':<40}  {'plain-RAG':<20}  {'self-RAG':<20}  {'gold'}")
    plain_hits = 0; sr_hits = 0
    for q, ans in queries_answers:
        pl = plain_rag_answer(q, corpus, ans, k=1)
        sr, _ = self_rag_answer(q, corpus, ans, k=3)
        plain_hits += (pl.lower() in [a.lower() for a in ans])
        sr_hits += (sr.lower() in [a.lower() for a in ans])
        print(f"  {q:<40}  {pl:<20}  {sr:<20}  {'/'.join(ans)}")
    print(f"\n  Plain-RAG accuracy: {plain_hits}/{len(queries_answers)}")
    print(f"  Self-RAG accuracy:   {sr_hits}/{len(queries_answers)}")

    print("\n--- library cross-check (self-rag GitHub / langchain custom Self-RAG) ---")
