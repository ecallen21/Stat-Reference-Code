"""Word embeddings: Skip-Gram with Negative Sampling (Reference §25.3).

Word2vec (Mikolov et al. 2013b): learn dense vectors v_w in R^d such that
words in similar contexts have similar vectors.

Skip-Gram objective (SGNS):
    max sum_{(w, c) in D+}  log sigma(v_w . u_c)
        + sum_{(w, c') in D-}  log sigma(-v_w . u_{c'})

with D+ = observed (word, context) pairs from a sliding window and D- = k
negatives per positive sampled from p(w)^0.75.

We implement a tiny SGD-trained SGNS from scratch and compute cosine
similarities to validate ("king - man + woman ~ queen"-style analogies need
much larger corpora).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math

from collections import Counter    # stdlib: bag counts

import numpy as np    # numerical arrays + linear algebra


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_sgns(sentences, dim: int = 30, window: int = 3,
               neg: int = 5, epochs: int = 30, lr: float = 0.05,
               min_count: int = 1, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    counts = Counter(w for s in sentences for w in s)
    vocab = {w: i for i, w in enumerate(sorted(counts) )
             if counts[w] >= min_count}
    V = len(vocab); inv_vocab = {i: w for w, i in vocab.items()}

    # negative-sampling distribution ~ freq^0.75
    freqs = np.array([counts[inv_vocab[i]] for i in range(V)], dtype=float) ** 0.75
    p_neg = freqs / freqs.sum()

    W = rng.normal(scale=0.1, size=(V, dim))               # center embeddings
    C = rng.normal(scale=0.1, size=(V, dim))               # context embeddings

    for ep in range(epochs):
        for s in sentences:
            ids = [vocab[w] for w in s if w in vocab]
            for i, wid in enumerate(ids):
                for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                    if i == j:
                        continue
                    cid = ids[j]
                    pos_score = _sigmoid(W[wid] @ C[cid])
                    W[wid] -= lr * (pos_score - 1) * C[cid]
                    C[cid] -= lr * (pos_score - 1) * W[wid]
                    # negatives
                    negs = rng.choice(V, neg, p=p_neg)
                    for nid in negs:
                        if nid == cid:
                            continue
                        neg_score = _sigmoid(W[wid] @ C[nid])
                        W[wid] -= lr * neg_score * C[nid]
                        C[nid] -= lr * neg_score * W[wid]
    return {"vocab": vocab, "W": W, "C": C, "dim": dim,
            "method": "SGNS via naive SGD (from scratch)"}


def cosine(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def most_similar(word, model, top_k: int = 5):
    W = model["W"]; vocab = model["vocab"]
    if word not in vocab:
        return []
    v = W[vocab[word]]
    sims = W @ v / (np.linalg.norm(W, axis=1) * np.linalg.norm(v) + 1e-12)
    inv = {i: w for w, i in vocab.items()}
    order = np.argsort(-sims)
    return [(inv[i], float(sims[i])) for i in order if inv[i] != word][:top_k]


if __name__ == "__main__":
    # tiny toy corpus with two topical clusters:
    #   "animal" sentences and "computer" sentences
    animal_words = "dog cat bird fish rabbit".split()
    animal_actions = "barks meows chirps swims hops".split()
    computer_words = "server database api script cloud".split()
    tech_actions = "boots queries returns crashes scales".split()
    rng = np.random.default_rng(0)
    sentences = []
    # animal-domain sentences (~300) with words drawn only from the animal domain
    for _ in range(300):
        w1 = animal_words[rng.integers(len(animal_words))]
        a  = animal_actions[rng.integers(len(animal_actions))]
        w2 = animal_words[rng.integers(len(animal_words))]
        a2 = animal_actions[rng.integers(len(animal_actions))]
        sentences.append([w1, a, w2, a2])
    # tech-domain sentences (~300) with words drawn only from the tech domain
    for _ in range(300):
        w1 = computer_words[rng.integers(len(computer_words))]
        a  = tech_actions[rng.integers(len(tech_actions))]
        w2 = computer_words[rng.integers(len(computer_words))]
        a2 = tech_actions[rng.integers(len(tech_actions))]
        sentences.append([w1, a, w2, a2])

    m = train_sgns(sentences, dim=15, window=2, neg=8, epochs=30, lr=0.03)
    print(f"=== SGNS trained: |V|={len(m['vocab'])}, dim={m['dim']} ===")
    for probe in ["cat", "server", "api"]:
        top = most_similar(probe, m, top_k=5)
        print(f"\n  most similar to {probe!r}:")
        for w, s in top:
            print(f"    {w:<12}  cos = {s:+.3f}")

    print("\n--- library cross-check (gensim.models.Word2Vec) ---")
    try:
        from gensim.models import Word2Vec
        g = Word2Vec(sentences=sentences, vector_size=20, window=2,
                     min_count=1, sg=1, negative=5, epochs=50, seed=0)
        for probe in ["cat", "server"]:
            print(f"  gensim top-3 similar to {probe!r}: "
                  f"{[w for w, _ in g.wv.most_similar(probe, topn=3)]}")
    except ImportError:
        print("  (gensim not installed)")
