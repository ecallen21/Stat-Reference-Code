"""Abstractive summarisation (Reference §25.x extra).

Given a document, generate a short summary in NEW words (may include
rewordings) — vs `textrank-summarization` which extracts sentences verbatim.

Standard architecture: encoder-decoder transformer (T5, BART, Pegasus, mT5)
trained on (article, summary) pairs.  Decoding uses beam search or nucleus
sampling with length penalties.

We demonstrate a rule-based sentence-fusion baseline: extract the top
sentences by TextRank, then simplify with a set of hand rules (drop
parentheticals, drop after commas, replace pronouns).  Not competitive
with T5 / BART but shows the surface-form vs semantic-form contrast.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import re    # stdlib: sentence splitting + regex simplification

from collections import Counter    # stdlib

import numpy as np    # numerical arrays + linear algebra


def _tokenise(s: str):
    return [w.lower() for w in re.findall(r"\w+", s)]


def _sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _tfidf(sents):
    tokenised = [_tokenise(s) for s in sents]
    vocab = {}
    for t in tokenised:
        for w in t:
            if w not in vocab:
                vocab[w] = len(vocab)
    X = np.zeros((len(sents), len(vocab)))
    for i, t in enumerate(tokenised):
        for w, c in Counter(t).items():
            X[i, vocab[w]] = c
    df = (X > 0).sum(axis=0)
    idf = np.log((len(sents) + 1) / (df + 1)) + 1
    W = X * idf[None, :]
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    return np.where(norms > 0, W / norms, 0.0)


def _simplify(sentence: str) -> str:
    # drop parentheticals
    s = re.sub(r"\([^)]*\)", "", sentence)
    # drop " -- ..." dashed parentheticals
    s = re.sub(r"\s+--\s+[^.]+", "", s)
    # collapse whitespace
    return re.sub(r"\s+", " ", s).strip()


def abstractive_summary(text: str, top_k: int = 3, damping: float = 0.85) -> dict:
    sents = _sentences(text)
    if len(sents) <= top_k:
        return {"summary": " ".join(_simplify(s) for s in sents)}
    W = _tfidf(sents); sim = W @ W.T
    np.fill_diagonal(sim, 0.0)
    # PageRank
    r = np.ones(len(sents)) / len(sents)
    d = sim.sum(axis=1); d[d == 0] = 1
    P = sim / d[:, None]
    for _ in range(50):
        r = damping * (P.T @ r) + (1 - damping) / len(sents)
    top_idx = np.argsort(-r)[:top_k]
    top_idx.sort()
    picked = [sents[i] for i in top_idx]
    simplified = [_simplify(s) for s in picked]
    return {"summary": " ".join(simplified), "picked_indices": top_idx.tolist(),
            "simplified": simplified,
            "method": "extract-then-simplify pseudo-abstractive baseline"}


if __name__ == "__main__":
    article = (
        "Renewable energy sources, including solar, wind, hydro, and geothermal power, "
        "have grown rapidly in the past decade -- reaching about 30% of the global electricity mix in 2023. "
        "Solar and wind power (both intermittent) benefit greatly from advances in battery storage, "
        "which allow surplus electricity to be captured and released as needed. "
        "Governments (from the European Union to India and China) have introduced large subsidies to accelerate this transition. "
        "Critics argue that the pace of change is still insufficient to meet 2050 net-zero commitments, "
        "citing continued fossil-fuel investment and grid inertia. "
        "Nevertheless, the International Energy Agency projects that renewables will supply over half of new electricity generation by 2030."
    )
    r = abstractive_summary(article, top_k=3)
    print("=== Abstractive-flavour summary (extract + simplify) ===\n")
    print(f"  article length : {len(article)} chars, {len(_sentences(article))} sentences\n")
    print(f"  picked sentences (in doc order): {r['picked_indices']}")
    print(f"\n  summary:\n    \"{r['summary']}\"")

    print("\n--- library cross-check (huggingface pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')) ---")
