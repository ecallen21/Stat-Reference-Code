"""Topic-coherence metrics for topic-model evaluation (Reference §25.11).

For a topic represented by its top-N words W = {w_1, ..., w_N}, coherence
scores summarise how often the top words co-occur.

  UMass (Mimno et al. 2011): uses document-level co-occurrence probabilities.
      c_UMass(W) = (2 / (N (N - 1))) sum_{i < j} log( (D(w_i, w_j) + 1) / D(w_j) )
  where D(w) = # docs containing w, D(w_i, w_j) = # docs containing both.

  UCI / PMI-based: p and pmi from a large reference corpus (external).
      c_UCI(W) = (2 / (N (N - 1))) sum_{i < j} PMI(w_i, w_j)
      PMI(a, b) = log( p(a, b) / (p(a) p(b)) )

  Perplexity: -log-lik per token; lower is better. Not always aligned with
  human coherence — Chang et al. 2009 famously showed perplexity and human
  interpretability can move in opposite directions.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math

import numpy as np    # numerical arrays + linear algebra


def umass_coherence(topic_words, docs) -> float:
    """topic_words: list of top-N terms; docs: list of token lists."""
    docs = [set(d) for d in docs]
    N = len(topic_words)
    D_wj = {w: sum(1 for d in docs if w in d) for w in topic_words}
    total = 0.0; pairs = 0
    for i in range(N):
        for j in range(i + 1, N):
            w_i, w_j = topic_words[i], topic_words[j]
            D_ij = sum(1 for d in docs if w_i in d and w_j in d)
            denom = D_wj[w_j] if D_wj[w_j] > 0 else 1
            total += math.log((D_ij + 1) / denom)
            pairs += 1
    return total / max(pairs, 1)


def uci_pmi_coherence(topic_words, docs, epsilon: float = 1e-12) -> float:
    docs = [set(d) for d in docs]
    N = len(topic_words); n_docs = len(docs)
    p_w = {w: sum(1 for d in docs if w in d) / n_docs for w in topic_words}
    total = 0.0; pairs = 0
    for i in range(N):
        for j in range(i + 1, N):
            w_i, w_j = topic_words[i], topic_words[j]
            p_ij = sum(1 for d in docs if w_i in d and w_j in d) / n_docs
            if p_ij > 0 and p_w[w_i] > 0 and p_w[w_j] > 0:
                total += math.log(p_ij / (p_w[w_i] * p_w[w_j]))
            else:
                total += math.log(epsilon)
            pairs += 1
    return total / max(pairs, 1)


def perplexity(log_lik: float, n_tokens: int) -> float:
    return math.exp(-log_lik / n_tokens)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Simulate a corpus of 200 docs across 3 topics with disjoint vocab
    topics_true = [
        "score match team goal player win coach league season points".split(),
        "server database api cloud code query script deploy bug cache".split(),
        "recipe cook flavor spice sauce bake fresh chef sweet meal".split(),
    ]
    docs = []
    for _ in range(200):
        t = rng.integers(3)
        docs.append([topics_true[t][rng.integers(10)] for _ in range(20)])

    print("=== Topic coherence for GOOD topics (drawn from true DGP) ===")
    for k, topic in enumerate(topics_true):
        c_umass = umass_coherence(topic[:6], docs)
        c_uci = uci_pmi_coherence(topic[:6], docs)
        print(f"  topic {k} (top-6 true words): UMass = {c_umass:+.3f}   "
              f"UCI-PMI = {c_uci:+.3f}")

    print("\n=== Topic coherence for a BAD topic (mixed vocab from all 3 topics) ===")
    bad_topic = ["score", "server", "recipe", "goal", "database", "chef"]
    c_umass = umass_coherence(bad_topic, docs)
    c_uci = uci_pmi_coherence(bad_topic, docs)
    print(f"  bad topic (mixed): UMass = {c_umass:+.3f}   UCI-PMI = {c_uci:+.3f}")

    # sanity: a bad (mixed) topic should have MUCH more negative UMass and PMI
    print(f"\n  (good topics have UMass ~ -1 to -2; bad mixed topics << that.)")

    print("\n--- library cross-check (gensim.models.CoherenceModel) ---")
    try:
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel
        d = Dictionary(docs)
        cm = CoherenceModel(topics=[t[:6] for t in topics_true], texts=docs,
                            dictionary=d, coherence="u_mass")
        print(f"  gensim u_mass for the 3 true topics: {cm.get_coherence():.3f}")
    except ImportError:
        print("  (gensim not installed)")
