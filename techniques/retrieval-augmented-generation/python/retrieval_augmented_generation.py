"""Retrieval-augmented generation (RAG) (Reference Sec 47.19).

Lewis et al. 2020 'Retrieval-augmented generation for knowledge-
intensive NLP tasks', NeurIPS. Combines a parametric generator (LM)
with a non-parametric memory (corpus + retriever). At inference:

    1. ENCODE the user query as a dense vector q.
    2. RETRIEVE top-k documents from the corpus by cosine similarity.
    3. GROUND the LM by concatenating retrieved passages with the
       query and generating the answer conditioned on both.

Advantages:
    * Injects fresh / private knowledge without model retraining.
    * Cites sources (retrieved passages).
    * Smaller LM sufficient because facts come from the corpus.

We implement a MINIMAL RAG: TF-IDF retriever + template-based
'generator' that assembles an answer from retrieved passages. This
demonstrates the pipeline shape without requiring a large LM.
"""
from __future__ import annotations    # stdlib

import re    # tokenisation
from collections import Counter

import numpy as np    # numerical arrays


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def build_tfidf(docs):
    """Return TF-IDF matrix (n_docs x n_terms) plus vocab."""
    tokens = [tokenize(d) for d in docs]
    vocab = sorted({w for doc in tokens for w in doc})
    idx = {w: i for i, w in enumerate(vocab)}
    n = len(docs); V = len(vocab)
    tf = np.zeros((n, V))
    for i, doc in enumerate(tokens):
        for w, c in Counter(doc).items():
            tf[i, idx[w]] = c / len(doc)
    df = (tf > 0).sum(axis=0)
    idf = np.log((n + 1) / (df + 1)) + 1.0
    return tf * idf, vocab, idx


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def retrieve(query, tfidf, vocab, idx, top_k=3):
    q_tokens = tokenize(query)
    q = np.zeros(len(vocab))
    for w in q_tokens:
        if w in idx:
            q[idx[w]] += 1
    if q.sum() == 0:
        return []
    scores = [cosine(q, tfidf[i]) for i in range(len(tfidf))]
    order = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in order]


def generate_answer(query, retrieved_texts):
    """Template-based 'LM': stitches together the retrieved facts.

    Real RAG substitutes a large decoder here (Llama, Mistral, GPT).
    """
    header = f"Answer to '{query}':"
    lines = [f"  - (source {i + 1}) {t}" for i, t in enumerate(retrieved_texts)]
    return "\n".join([header] + lines)


if __name__ == "__main__":
    print("=== Retrieval-augmented generation (RAG) -- minimal pipeline ===\n")
    #  A tiny corpus about biostatistics techniques.
    corpus = [
        "The Cox proportional hazards model estimates hazard ratios from right-censored survival data.",
        "IPTW uses propensity scores to reweight observations for average causal effects.",
        "Random survival forests extend random forests to survival outcomes using log-rank splits.",
        "K-means clustering partitions n observations into k clusters by minimising within-cluster variance.",
        "Bayesian hierarchical models pool information across groups via prior distributions on effects.",
        "The Kaplan-Meier estimator provides a nonparametric estimate of the survival function.",
    ]
    tfidf, vocab, idx = build_tfidf(corpus)
    print(f"  Corpus: {len(corpus)} docs, vocab size = {len(vocab)}\n")

    query = "survival model with censoring"
    print(f"  Query: {query}")
    hits = retrieve(query, tfidf, vocab, idx, top_k=3)
    print(f"  Top-3 retrieval:")
    for i, s in hits:
        print(f"    doc {i} (score {s:.3f})  ->  {corpus[i][:70]}...")

    answer = generate_answer(query, [corpus[i] for i, _ in hits])
    print("\n" + answer)

    print("\n--- library cross-check (LangChain / LlamaIndex Python; ellmer R) ---")
