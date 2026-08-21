"""Semantic textual similarity (Reference §25.x extra).

Given a static word-embedding table:
    sent_vec(s) = mean_{w in s} vec(w)   (or IDF-weighted mean, or SIF)
    sim(a, b) = cosine(sent_vec(a), sent_vec(b))

Bag-of-embeddings + cosine is a strong baseline for semantic similarity
between short texts; often within 5-10 pts of learned sentence-transformer
embeddings (SBERT) on STS benchmarks.

Modern SOTA: sentence-transformers, INSTRUCTOR, GTE, E5, embedding APIs.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _mean_vec(sentence, vecs, idf=None):
    ws = [w for w in sentence if w in vecs]
    if not ws:
        return np.zeros(next(iter(vecs.values())).shape)
    if idf is not None:
        weights = np.array([idf.get(w, 1.0) for w in ws])
        weights = weights / weights.sum()
        return sum(weights[i] * vecs[w] for i, w in enumerate(ws))
    return np.mean([vecs[w] for w in ws], axis=0)


def cosine(u, v) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def sts_pairs(pairs, vecs, idf=None) -> list:
    out = []
    for a, b in pairs:
        va = _mean_vec(a, vecs, idf); vb = _mean_vec(b, vecs, idf)
        out.append(cosine(va, vb))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # toy hand-picked 8-D embeddings — semantically similar words placed near each other
    def _v(*x): return np.array(x, dtype=float)
    vecs = {
        # animals
        "dog":     _v(1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "puppy":   _v(0.9, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "cat":     _v(0.8, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "kitten":  _v(0.7, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        # tech
        "server":  _v(0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0),
        "database":_v(0.0, 0.0, 0.9, 0.3, 0.0, 0.0, 0.0, 0.0),
        "api":     _v(0.0, 0.0, 0.85, 0.3, 0.0, 0.0, 0.0, 0.0),
        # food
        "pizza":   _v(0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0),
        "pasta":   _v(0.0, 0.0, 0.0, 0.0, 0.9, 0.3, 0.0, 0.0),
        "burger":  _v(0.0, 0.0, 0.0, 0.0, 0.85, 0.3, 0.0, 0.0),
        # verbs (spread across)
        "loves":   _v(0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 1.0, 0.1),
        "adores":  _v(0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.95, 0.15),
        "the":     _v(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1.0),
        "a":       _v(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.9),
    }

    pairs = [
        ("the dog loves the cat".split(), "the puppy adores the kitten".split()),      # very similar
        ("the dog loves the cat".split(), "the server queries a database".split()),    # unrelated
        ("the api calls the server".split(), "a database queries the api".split()),    # similar (tech)
        ("the pizza is the burger".split(), "a puppy loves a kitten".split()),         # unrelated
        ("the dog loves the cat".split(), "a puppy adores a kitten".split()),          # similar
    ]

    sims = sts_pairs(pairs, vecs)
    print(f"=== Bag-of-embeddings cosine similarity ===")
    for (a, b), s in zip(pairs, sims):
        print(f"  {s:+.3f}   \"{' '.join(a)}\"  <->  \"{' '.join(b)}\"")

    print("\n--- library cross-check (sentence-transformers 'all-MiniLM-L6-v2' / spaCy) ---")
