"""TextRank extractive summarisation (Mihalcea-Tarau 2004; Reference §25.12).

Extractive summariser: pick the most "important" sentences from the input.

Build a sentence graph:
  * nodes = sentences
  * edge weight(s_i, s_j) = cosine similarity of their TF-IDF vectors
    (or an overlap ratio, or a BM25 kernel).
Run PageRank on that weighted undirected graph; the top-K PageRank scores
give the summary.

Simple, unsupervised, and surprisingly strong across languages.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import re    # stdlib: sentence splitting

from collections import Counter    # stdlib: bag counts

import numpy as np    # numerical arrays + linear algebra


def _split_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text.strip())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _tokenise(sent: str) -> list:
    sent = re.sub(r"[^\w\s']", " ", sent.lower())
    return sent.split()


def _tfidf_sentences(sents):
    tokenised = [_tokenise(s) for s in sents]
    vocab = {}
    for t in tokenised:
        for w in t:
            if w not in vocab:
                vocab[w] = len(vocab)
    X = np.zeros((len(tokenised), len(vocab)))
    for i, t in enumerate(tokenised):
        for w, c in Counter(t).items():
            X[i, vocab[w]] = c
    N = len(tokenised); df = (X > 0).sum(axis=0)
    idf = np.log((N + 1) / (df + 1)) + 1
    W = X * idf[None, :]
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    return np.where(norms > 0, W / norms, 0.0)


def _pagerank(M, damping: float = 0.85, tol: float = 1e-7, max_iter: int = 200):
    n = M.shape[0]
    r = np.ones(n) / n
    d = M.sum(axis=1); d[d == 0] = 1
    P = (M.T / d).T                                       # row-stochastic (M is symmetric)
    for _ in range(max_iter):
        r_new = damping * (P.T @ r) + (1 - damping) / n
        if np.abs(r_new - r).sum() < tol:
            return r_new
        r = r_new
    return r


def textrank(text: str, top_k: int = 3, damping: float = 0.85) -> dict:
    sents = _split_sentences(text)
    if len(sents) <= top_k:
        return {"sentences": sents, "summary": sents,
                "scores": [1.0] * len(sents),
                "method": "TextRank (no extraction needed; input too short)"}
    W = _tfidf_sentences(sents)
    sim = W @ W.T
    np.fill_diagonal(sim, 0.0)
    scores = _pagerank(sim, damping=damping)
    order = np.argsort(-scores)[:top_k]
    order.sort()                                          # preserve document order
    return {"sentences": sents, "summary": [sents[i] for i in order],
            "scores": scores.tolist(),
            "method": "TextRank (cosine similarity edges + PageRank)"}


if __name__ == "__main__":
    text = ("Renewable energy is derived from natural resources that are replenished naturally. "
            "Solar power converts sunlight directly into electricity using photovoltaic cells or concentrated solar thermal systems. "
            "Wind power uses turbines to generate electricity from moving air, both on land and at sea. "
            "Hydroelectric power extracts energy from flowing water in rivers and dams. "
            "The main challenges of renewable energy include intermittent generation and grid storage requirements. "
            "Improvements in battery technology and smart grid management are addressing these challenges. "
            "Government policy and market incentives play a critical role in accelerating adoption. "
            "Together these sources form the backbone of modern low-carbon electricity systems.")

    res = textrank(text, top_k=3)
    print("=== TextRank extractive summary ===\n")
    print(f"  input: {len(res['sentences'])} sentences\n")
    for i, s in enumerate(res["sentences"]):
        marker = "*" if s in res["summary"] else " "
        print(f"  {marker} [{res['scores'][i]:.4f}] {s}")
    print("\n  --- summary (top 3, in document order) ---")
    for s in res["summary"]:
        print(f"  * {s}")

    print("\n--- library cross-check (sumy TextRankSummarizer / summa) ---")
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.text_rank import TextRankSummarizer
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summ = TextRankSummarizer()
        print(f"  sumy summary sentences:")
        for s in summ(parser.document, 3):
            print(f"    * {s}")
    except ImportError:
        print("  (sumy not installed)")
