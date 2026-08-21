"""Document clustering via k-means on TF-IDF vectors (Reference §25.5).

Pipeline:
  1. Tokenise + preprocess -> bag of words per document.
  2. TF-IDF vectorise (L2-normalised).
  3. Spherical k-means (equivalent to cosine k-means on L2-normalised vectors).
  4. Report clusters + evaluation vs known labels: purity, NMI, ARI.

Evaluation:
  * purity: fraction of documents in dominant true class per cluster.
  * NMI: normalised mutual information between clustering and labels.
  * ARI: adjusted Rand index (adjusted for chance).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math

from collections import Counter    # stdlib: bag counts

import numpy as np    # numerical arrays + linear algebra


def _tfidf(docs, l2_norm: bool = True):
    vocab = {}
    for d in docs:
        for w in d:
            if w not in vocab:
                vocab[w] = len(vocab)
    X = np.zeros((len(docs), len(vocab)))
    for i, d in enumerate(docs):
        for w, c in Counter(d).items():
            X[i, vocab[w]] = c
    N = len(docs); df = (X > 0).sum(axis=0)
    idf = np.log((N + 1) / (df + 1)) + 1
    W = X * idf[None, :]
    if l2_norm:
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        W = np.where(norms > 0, W / norms, 0.0)
    return W, vocab, idf


def _spherical_kmeans_single(X, K, seed, n_iter: int = 100):
    rng = np.random.default_rng(seed)
    n = len(X)
    centres = X[rng.choice(n, K, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        sims = X @ centres.T                              # (n, K)
        new_labels = sims.argmax(axis=1)
        if (new_labels == labels).all():
            break
        labels = new_labels
        for k in range(K):
            mask = labels == k
            if mask.any():
                v = X[mask].sum(axis=0)
                centres[k] = v / (np.linalg.norm(v) + 1e-12)
    # cohesion = sum of similarities to assigned centre (bigger is better)
    cohesion = float((X * centres[labels]).sum())
    return labels, centres, cohesion


def spherical_kmeans(X, K, n_starts: int = 15, n_iter: int = 100, seed: int = 0):
    """Multiple restarts; return the fit with the highest cosine cohesion."""
    rng = np.random.default_rng(seed)
    best = (None, None, -np.inf)
    for _ in range(n_starts):
        s = int(rng.integers(1e9))
        labels, centres, coh = _spherical_kmeans_single(X, K, s, n_iter)
        if coh > best[2]:
            best = (labels, centres, coh)
    return best[0], best[1]


def purity(labels, truth):
    labels = np.asarray(labels); truth = np.asarray(truth)
    total = 0
    for c in np.unique(labels):
        counts = Counter(truth[labels == c]); total += counts.most_common(1)[0][1]
    return total / len(truth)


def nmi(labels, truth):
    labels = np.asarray(labels); truth = np.asarray(truth); n = len(labels)
    # mutual information
    contingency = {}
    for c, t in zip(labels, truth):
        contingency[(c, t)] = contingency.get((c, t), 0) + 1
    Hc = -sum((v / n) * math.log(v / n) for v in Counter(labels).values())
    Ht = -sum((v / n) * math.log(v / n) for v in Counter(truth).values())
    MI = 0.0
    lc = Counter(labels); lt = Counter(truth)
    for (c, t), n_ct in contingency.items():
        MI += (n_ct / n) * math.log(n_ct * n / (lc[c] * lt[t]))
    return 2 * MI / (Hc + Ht + 1e-12)


def ari(labels, truth):
    """Adjusted Rand Index."""
    labels = np.asarray(labels); truth = np.asarray(truth); n = len(labels)
    from math import comb
    a = Counter(labels); b = Counter(truth); joint = Counter()
    for c, t in zip(labels, truth):
        joint[(c, t)] += 1
    sum_ct = sum(comb(v, 2) for v in joint.values())
    sum_c = sum(comb(v, 2) for v in a.values())
    sum_t = sum(comb(v, 2) for v in b.values())
    N2 = comb(n, 2)
    expected = sum_c * sum_t / N2
    max_ = (sum_c + sum_t) / 2
    if max_ == expected:
        return 1.0 if sum_ct == max_ else 0.0
    return (sum_ct - expected) / (max_ - expected)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # simulate documents from 3 topics with disjoint vocabularies
    topics = [
        "score match team goal player win coach league season points".split(),
        "server database api cloud code query script deploy bug cache".split(),
        "recipe cook flavor spice sauce bake fresh chef sweet meal".split(),
    ]
    D = 60; doc_len = 30
    docs = []; truth = []
    for d in range(D):
        t = d % 3; truth.append(t)
        docs.append([topics[t][rng.integers(len(topics[t]))]
                     for _ in range(doc_len)])

    X, vocab, idf = _tfidf(docs)
    print(f"=== Document clustering (D={D}, |V|={len(vocab)}, K=3) ===")
    labels, _ = spherical_kmeans(X, K=3, seed=1)
    print(f"  purity = {purity(labels, truth):.3f}")
    print(f"  NMI    = {nmi(labels, truth):.3f}")
    print(f"  ARI    = {ari(labels, truth):.3f}")

    print("\n--- library cross-check (sklearn.cluster.KMeans on the same TF-IDF) ---")
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        km = KMeans(n_clusters=3, random_state=1, n_init=10).fit(X)
        print(f"  sklearn NMI = {normalized_mutual_info_score(truth, km.labels_):.3f}")
        print(f"  sklearn ARI = {adjusted_rand_score(truth, km.labels_):.3f}")
    except ImportError:
        print("  (sklearn not installed)")
