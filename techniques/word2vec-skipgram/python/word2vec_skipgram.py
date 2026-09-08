"""word2vec skip-gram (Reference Sec 47.117).

Mikolov, Chen, Corrado & Dean 2013 'Efficient estimation of word
representations in vector space' + Mikolov, Sutskever, Chen,
Corrado & Dean 2013 'Distributed representations of words and
phrases and their compositionality'. Skip-gram trains word vectors
to predict CONTEXT words:

    max_theta  E_{(w, c) ~ corpus} log sigma(v_c . v_w)
              + k E_{c'~P_noise} log sigma(-v_c' . v_w)      (SGNS)

with `v_w` INPUT (target) and `v_c` OUTPUT (context) embeddings.
Negative-sampling (SGNS) avoids the O(V) softmax denominator.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_sgns(corpus, vocab, window=2, dim=32, k_neg=5, epochs=30, lr=0.05, seed=0):
    """SGNS on tokenised corpus (list of token IDs)."""
    rng = np.random.default_rng(seed)
    V = len(vocab)
    W_in = rng.normal(size=(V, dim)) * 0.1     # target vectors
    W_out = rng.normal(size=(V, dim)) * 0.1    # context vectors
    unigram = np.bincount(corpus, minlength=V) ** 0.75
    unigram /= unigram.sum()

    for ep in range(epochs):
        for i, w in enumerate(corpus):
            lo, hi = max(0, i - window), min(len(corpus), i + window + 1)
            for j in range(lo, hi):
                if j == i: continue
                c = corpus[j]
                v_w = W_in[w]; v_c = W_out[c]
                # Positive update
                s = sigmoid(v_c @ v_w)
                dW_in = (s - 1) * v_c
                dW_out = (s - 1) * v_w
                W_out[c] -= lr * dW_out
                W_in[w] -= lr * dW_in
                # Negative samples
                for _ in range(k_neg):
                    n = int(rng.choice(V, p=unigram))
                    if n == c: continue
                    v_n = W_out[n]
                    sn = sigmoid(-v_n @ v_w)
                    W_out[n] -= lr * ((1 - sn) * v_w)
                    W_in[w] -= lr * ((1 - sn) * v_n)
    return W_in, W_out


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


if __name__ == "__main__":
    print("=== word2vec skip-gram (SGNS) (Mikolov et al 2013) ===\n")
    rng = np.random.default_rng(0)

    # Toy corpus: three semantic clusters via repeated co-occurrence
    sentences = [
        ["king", "queen", "royal", "crown", "throne"] * 5,
        ["cat", "dog", "pet", "animal", "fur"] * 5,
        ["car", "truck", "vehicle", "engine", "wheel"] * 5,
    ]
    corpus = [w for s in sentences for w in s]
    vocab = list(dict.fromkeys(corpus))
    stoi = {w: i for i, w in enumerate(vocab)}
    corpus_ids = np.array([stoi[w] for w in corpus])

    W_in, W_out = train_sgns(corpus_ids, vocab, window=3, dim=16,
                                k_neg=3, epochs=40, lr=0.05, seed=0)

    print("  Cosine similarities of learned word vectors (W_in):\n")
    pairs = [("king", "queen"), ("king", "cat"), ("king", "car"),
              ("cat", "dog"), ("cat", "car"),
              ("car", "truck"), ("car", "dog")]
    for a, b in pairs:
        s = cosine(W_in[stoi[a]], W_in[stoi[b]])
        print(f"    {a:8s} vs {b:8s}: {s:+.3f}")

    print("\n  Within-cluster similarities should exceed cross-cluster ones.")
    print("\n--- library cross-check (text2vec R; gensim.models.Word2Vec Python) ---")
