"""Extractive question answering (Reference §25.x extra).

Extractive QA: given a passage P and a question Q, predict the (start, end)
span in P that answers Q.  Modeled as two token-classification heads on top
of a contextual encoder (BERT-style):

    logits_start[t] = w_s . h_t
    logits_end[t]   = w_e . h_t

Trained by cross-entropy on the true (start, end) positions.

Two other classical approaches:
  * IR-based: retrieve top-k passages (BM25), then extract a span.
  * Reader-Retriever (RAG): retriever + generator LLM.

Here we implement a WORD-OVERLAP-based baseline that finds the passage span
of size k with the most overlap with the question tokens.  Deep versions
plug in transformer encoders + start/end heads (see huggingface QA pipeline).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import re    # stdlib: sentence + token splitting

from collections import Counter    # stdlib: bag counts

import numpy as np    # numerical arrays + linear algebra


STOPWORDS = {"the", "a", "an", "to", "of", "in", "on", "at", "for", "is", "are",
              "was", "were", "and", "or", "with", "by", "from"}


def _tokenise(s: str):
    return [w.lower() for w in re.findall(r"\w+", s) if w.lower() not in STOPWORDS]


def extract_answer_sentence(passage: str, question: str) -> dict:
    """Sentence retrieval with IDF-weighted question overlap.  Common terms
    (words in most sentences) get low weight so rare keywords steer the match."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage) if s.strip()]
    q_tokens = set(_tokenise(question))
    # IDF from the passage sentences themselves
    N = len(sentences)
    df = Counter()
    for s in sentences:
        for w in set(_tokenise(s)):
            df[w] += 1
    def _idf(w): return np.log((N + 1) / (df.get(w, 0) + 1)) + 1
    best = ("", -1.0)
    for s in sentences:
        s_tokens = set(_tokenise(s))
        score = float(sum(_idf(w) for w in q_tokens if w in s_tokens))
        if score > best[1]:
            best = (s, score)
    return {"answer": best[0], "score": best[1],
            "method": "IDF-weighted sentence-retrieval baseline"}


if __name__ == "__main__":
    passage = (
        "Marie Curie was a Polish and naturalized-French physicist and chemist. "
        "She conducted pioneering research on radioactivity. "
        "She was the first woman to win a Nobel Prize, the first person to win the Nobel Prize twice, "
        "and the only person to win Nobel Prizes in two different scientific fields. "
        "Her husband Pierre Curie was a co-winner of her first Nobel Prize."
    )
    qas = [
        ("What did Marie Curie research?", "radioactivity"),
        ("Who was Marie Curie's husband?", "Pierre Curie"),
        ("How many Nobel Prizes did Marie Curie win?", "twice"),
        ("What nationalities did Marie Curie hold?", "Polish and naturalized-French"),
    ]
    print("=== QA sentence-retrieval baseline ===\n")
    hits = 0
    for q, gold in qas:
        r = extract_answer_sentence(passage, q)
        ok = gold.lower() in r["answer"].lower()
        hits += ok
        print(f"  [{'ok' if ok else '..'}]  Q: {q}")
        print(f"        gold: {gold}")
        print(f"        pred: '{r['answer']}'  (score = {r['score']:.3f})\n")
    print(f"  {hits}/{len(qas)} answers include the gold span.")

    print("--- library cross-check (huggingface pipeline('question-answering'); haystack) ---")
