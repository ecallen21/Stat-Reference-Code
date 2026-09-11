"""RAGAS - Retrieval-Augmented Generation Assessment (Sec 47.218).

Es, James, Espinosa-Anke & Schockaert 2024 'RAGAS: Automated
Evaluation of Retrieval-Augmented Generation'. Reference-free
metrics for RAG pipelines:

    FAITHFULNESS   = fraction of answer claims supported by retrieved context.
    ANSWER RELEVANCE = cosine(embed(gen answer), embed(query)).
    CONTEXT PRECISION = signal-to-noise of retrieved chunks.
    CONTEXT RECALL   = fraction of gold-answer claims present in retrieved context.

Uses an LM to extract claims + judge support; no gold answers
required for the first two.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def extract_claims(text):
    """Toy LM claim extractor: split on periods, keep sentences with a verb / noun."""
    return [s.strip() for s in text.split(".") if s.strip()]


def support_claim(claim, context):
    """Toy 'is claim supported by context?' judge — check keyword overlap."""
    c_toks = set(claim.lower().split())
    ctx_toks = set(context.lower().split())
    common = c_toks & ctx_toks
    # 'stopword' filter
    common -= {"the", "a", "an", "is", "are", "was", "were", "of", "in", "and"}
    return len(common) >= 2


def cosine_bag(text_a, text_b, vocab):
    """Cosine of bag-of-words vectors."""
    def bow(t):
        v = np.zeros(len(vocab))
        for w in t.lower().split():
            if w in vocab: v[vocab[w]] += 1
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    a = bow(text_a); b = bow(text_b)
    return float(a @ b)


def faithfulness(gen_answer, retrieved_context):
    """Fraction of answer claims supported by retrieved context."""
    claims = extract_claims(gen_answer)
    if not claims: return 0.0
    return sum(support_claim(c, retrieved_context) for c in claims) / len(claims)


def answer_relevance(gen_answer, question, vocab):
    return cosine_bag(gen_answer, question, vocab)


def context_recall(gold_answer, retrieved_context):
    """Fraction of gold-answer claims present in retrieved context."""
    claims = extract_claims(gold_answer)
    if not claims: return 0.0
    return sum(support_claim(c, retrieved_context) for c in claims) / len(claims)


if __name__ == "__main__":
    print("=== RAGAS (Es et al 2024) ===\n")

    question = "When was the Eiffel Tower built and how tall is it?"
    retrieved_context = ("The Eiffel Tower was designed by Gustave Eiffel and completed in 1889 "
                          "for the World Fair. Its total height is 330 meters including the antenna.")
    gold_answer = "The Eiffel Tower was built in 1889 and is 330 meters tall."

    answers = [
        {"name": "Good", "text": "The Eiffel Tower was built in 1889 and stands 330 meters tall including the antenna."},
        {"name": "Hallucinated", "text": "The Eiffel Tower was built in 1900 and is 500 meters tall."},
        {"name": "Off-topic", "text": "I don't know when it was built but Paris has many museums."},
    ]

    # Build vocab across all texts
    all_text = " ".join([question, retrieved_context, gold_answer] + [a["text"] for a in answers])
    vocab = {w: i for i, w in enumerate(sorted(set(all_text.lower().split())))}

    print(f"  {'answer':<15}  {'faithfulness':>13}  {'ans_relevance':>14}  {'ctx_recall':>11}")
    for a in answers:
        f = faithfulness(a["text"], retrieved_context)
        r = answer_relevance(a["text"], question, vocab)
        c = context_recall(gold_answer, retrieved_context)
        print(f"  {a['name']:<15}  {f:>13.2f}  {r:>14.2f}  {c:>11.2f}")

    print("\n  Real RAGAS uses an LM to extract atomic claims and score support / relevance;")
    print("  correlates with human judgments 0.65-0.80 depending on metric and domain.")

    print("\n--- library cross-check (ragas Python; deepeval; trulens-eval) ---")
