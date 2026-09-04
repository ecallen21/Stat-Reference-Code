"""Collocation statistics: PMI + Dunning G^2 (Reference Sec 42.12).

Given a corpus, identify STATISTICALLY-SIGNIFICANT word pairs
(bigrams) using:

  POINTWISE MUTUAL INFORMATION:
    PMI(x, y) = log( P(x, y) / (P(x) * P(y)) )

  DUNNING LOG-LIKELIHOOD RATIO (Dunning 1993):
    2 * [ll(H1) - ll(H0)] on a 2x2 co-occurrence table.

Dunning is more robust than PMI at low counts (Manning-Schutze 1999
Ch 5).  Both work over any collocation window (bigram, skip-gram).
"""
from __future__ import annotations    # stdlib

import re
from math import log

import numpy as np    # numerical arrays


def bigrams(tokens):
    return list(zip(tokens[:-1], tokens[1:]))


def _counts(docs):
    tok = [re.findall(r"\w+", d.lower()) for d in docs]
    uni = {}; bi = {}
    for t in tok:
        for w in t:
            uni[w] = uni.get(w, 0) + 1
        for (a, b) in bigrams(t):
            bi[(a, b)] = bi.get((a, b), 0) + 1
    N = sum(uni.values())
    return uni, bi, N, tok


def pmi(uni, bi, N):
    scores = {}
    for (a, b), n_ab in bi.items():
        p_ab = n_ab / (N - len(bi))     # approx pair count
        p_a = uni[a] / N; p_b = uni[b] / N
        scores[(a, b)] = float(log(p_ab / (p_a * p_b + 1e-12) + 1e-12))
    return scores


def dunning_g2(uni, bi, N):
    """Dunning 1993 2x2 log-likelihood for each bigram."""
    scores = {}
    N_bi = sum(bi.values())
    for (a, b), n_ab in bi.items():
        k11 = n_ab
        k12 = uni[a] - n_ab             # a not followed by b
        k21 = uni[b] - n_ab             # b not preceded by a
        k22 = N - k11 - k12 - k21
        # Compute log-likelihood
        def _l(k, n, p):
            if k == 0 or (n - k) == 0 or p <= 0 or p >= 1:
                return 0.0
            return k * log(p) + (n - k) * log(1 - p)
        n1 = k11 + k12; n2 = k21 + k22
        p1 = k11 / max(n1, 1); p2 = k21 / max(n2, 1)
        p  = (k11 + k21) / max(n1 + n2, 1)
        ll_h0 = _l(k11, n1, p) + _l(k21, n2, p)
        ll_h1 = _l(k11, n1, p1) + _l(k21, n2, p2)
        scores[(a, b)] = 2 * (ll_h1 - ll_h0)
    return scores


if __name__ == "__main__":
    print("=== Collocation statistics: PMI + Dunning G^2 ===\n")
    docs = [
        "chest pain radiating to left arm.",
        "chest pain worsened by exertion.",
        "chest pain relieved by rest.",
        "New chest pain overnight.",
        "The chest pain returned after coffee.",
        "Left arm numbness noted separately.",
        "Coffee for breakfast.",
        "Rest after exertion.",
    ]
    uni, bi, N, _ = _counts(docs)
    p = pmi(uni, bi, N)
    g = dunning_g2(uni, bi, N)
    top_p = sorted(p.items(), key=lambda kv: -kv[1])[:6]
    top_g = sorted(g.items(), key=lambda kv: -kv[1])[:6]

    print("  Top bigrams by PMI:")
    for (a, b), s in top_p:
        print(f"    {a:>10s} {b:<12s}  PMI = {s:+.3f}   n = {bi[(a, b)]}")
    print("\n  Top bigrams by Dunning G^2:")
    for (a, b), s in top_g:
        print(f"    {a:>10s} {b:<12s}  G^2 = {s:.2f}   n = {bi[(a, b)]}")

    print("\n--- library cross-check (R quanteda::textstat_collocations; Python nltk collocations, gensim Phrases) ---")
