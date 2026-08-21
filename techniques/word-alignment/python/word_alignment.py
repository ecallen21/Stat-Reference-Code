"""IBM Model 1 word alignment (Brown et al. 1993; Reference §25.x extra).

Generative model of a source sentence e = (e_1, ..., e_l) from a target sentence
f = (f_1, ..., f_m).  Each source word e_i is aligned to some target position
a_i in {0, ..., m} (0 = NULL).  Model 1 treats all alignment positions as
equally likely:

    P(e, a | f) = (1 / (m + 1)^l) prod_i t(e_i | f_{a_i})

EM iterates:
  * E: expected translation-pair counts c(e | f) = sum_pairs sum_i sum_j
       [ t(e_i | f_j) / sum_k t(e_i | f_k) ] * 1{e_i = e, f_j = f}
  * M: t(e | f) = c(e | f) / sum_e' c(e' | f)

Extracted alignments: for each source word e_i, argmax_j t(e_i | f_j).

Model 1 is the classical starting point for statistical MT; later models
(2-5, HMM, fastalign) add positional / fertility structure.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

from collections import defaultdict    # stdlib: dict of counters

import numpy as np    # numerical arrays + linear algebra


def train_ibm_model_1(pairs, n_iter: int = 20) -> dict:
    """pairs: list of (source_tokens, target_tokens)."""
    source_vocab = sorted({w for s, _ in pairs for w in s})
    target_vocab = sorted({w for _, t in pairs for w in t} | {"<NULL>"})
    # uniform init
    t = defaultdict(lambda: defaultdict(lambda: 1.0 / len(source_vocab)))
    for _ in range(n_iter):
        count = defaultdict(lambda: defaultdict(float))
        total = defaultdict(float)
        for src, tgt in pairs:
            tgt_null = ["<NULL>"] + list(tgt)
            for e in src:
                denom = sum(t[e][f] for f in tgt_null)
                for f in tgt_null:
                    delta = t[e][f] / (denom + 1e-12)
                    count[e][f] += delta
                    total[f] += delta
        for e in list(count):
            for f, c in count[e].items():
                t[e][f] = c / (total[f] + 1e-12)
    return {"t": t, "src_vocab": source_vocab, "tgt_vocab": target_vocab,
            "method": f"IBM Model 1 EM ({n_iter} iters)"}


def align(source, target, model) -> list:
    tgt_null = ["<NULL>"] + list(target)
    alignments = []
    for i, e in enumerate(source):
        scores = [(j, model["t"][e][f]) for j, f in enumerate(tgt_null)]
        best_j, best_p = max(scores, key=lambda x: x[1])
        alignments.append((i, best_j - 1, best_p))          # -1 to shift back (0 = NULL becomes -1)
    return alignments


if __name__ == "__main__":
    # tiny English -> French toy corpus with clear alignments
    pairs = [
        ("the house".split(), "la maison".split()),
        ("the book".split(), "le livre".split()),
        ("a book".split(), "un livre".split()),
        ("a house".split(), "une maison".split()),
        ("the small book".split(), "le petit livre".split()),
        ("a small house".split(), "une petite maison".split()),
        ("the big book".split(), "le grand livre".split()),
        ("a big house".split(), "une grande maison".split()),
    ] * 10                                                # repeat for stability

    m = train_ibm_model_1(pairs, n_iter=30)

    # translation table for a few words
    print("=== IBM Model 1: top-3 French translations per English word ===")
    for e in ("the", "a", "book", "house", "small", "big"):
        top = sorted(m["t"][e].items(), key=lambda kv: -kv[1])[:3]
        print(f"  {e:>5} -> {[(f, round(p, 3)) for f, p in top]}")

    # alignment demo
    src = "the small house".split(); tgt = "la petite maison".split()
    print(f"\n  alignment for {src} <-> {tgt}:")
    for i, j, p in align(src, tgt, m):
        target_word = "<NULL>" if j < 0 else tgt[j]
        print(f"    {src[i]:>7} -> {target_word:>7}   p = {p:.3f}")

    print("\n--- library cross-check (fastalign / GIZA++ / eflomal / awesome-align) ---")
