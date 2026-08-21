"""n-gram language model + Laplace / Kneser-Ney smoothing (Reference §25.x extra).

    P(w_1, ..., w_T) = prod_t P(w_t | w_{t-n+1}, ..., w_{t-1})

Two smoothing schemes:
  * Laplace (add-alpha):
        P(w | h) = (count(h w) + alpha) / (count(h) + alpha * V)
  * Kneser-Ney (interpolated, absolute-discounted):
        P_KN(w | h) = max(count(h w) - D, 0) / count(h)
                      + lambda(h) * P_KN(w | h_shorter)

Perplexity on held-out data:
    PPL = exp( - (1 / N) * sum_i log P(w_i | h_i) )
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math

from collections import Counter, defaultdict    # stdlib: bag counts

import numpy as np    # numerical arrays + linear algebra


def _pad(seq, n):
    return ["<s>"] * (n - 1) + list(seq) + ["</s>"]


def train_laplace_ngram(corpus, n: int = 3, alpha: float = 1.0) -> dict:
    """corpus: list of token lists."""
    counts = defaultdict(Counter); ctx_counts = Counter()
    vocab = set()
    for s in corpus:
        s = _pad(s, n)
        for i in range(n - 1, len(s)):
            ctx = tuple(s[i - n + 1: i]); w = s[i]
            counts[ctx][w] += 1; ctx_counts[ctx] += 1; vocab.add(w)
    return {"n": n, "alpha": alpha, "counts": counts,
            "ctx_counts": ctx_counts, "vocab": vocab,
            "V": len(vocab), "method": f"{n}-gram + Laplace(a={alpha})"}


def _p_laplace(model, ctx, w):
    c = model["counts"].get(ctx, Counter())
    return (c[w] + model["alpha"]) / (model["ctx_counts"].get(ctx, 0) + model["alpha"] * model["V"])


def perplexity(model, corpus) -> float:
    n = model["n"]; total_log = 0.0; total_n = 0
    for s in corpus:
        s = _pad(s, n)
        for i in range(n - 1, len(s)):
            ctx = tuple(s[i - n + 1: i]); w = s[i]
            p = _p_laplace(model, ctx, w)
            total_log += math.log(p + 1e-12)
            total_n += 1
    return math.exp(-total_log / max(total_n, 1))


def train_kn_bigram(corpus, D: float = 0.75) -> dict:
    """Simplified interpolated Kneser-Ney (bigram only)."""
    bi = Counter(); uni = Counter()
    left_types = defaultdict(set)                          # unique lefts per word (for continuation prob)
    for s in corpus:
        s = _pad(s, 2)
        for i in range(1, len(s)):
            bi[(s[i - 1], s[i])] += 1
            uni[s[i]] += 1
            left_types[s[i]].add(s[i - 1])
    total_bi = sum(bi.values())
    ctx_totals = defaultdict(int)
    ctx_types = defaultdict(int)
    for (a, b), c in bi.items():
        ctx_totals[a] += c; ctx_types[a] += 1
    # continuation probability: P_cont(w) = |{a : count(a, w) > 0}| / |unique bigrams|
    p_cont = {w: len(left_types[w]) / max(len(bi), 1) for w in uni}
    return {"D": D, "bi": bi, "ctx_totals": ctx_totals,
            "ctx_types": ctx_types, "p_cont": p_cont, "V": len(uni),
            "method": "Kneser-Ney bigram (interpolated, D=0.75)"}


def _p_kn(model, prev, w):
    D = model["D"]
    ctx_total = model["ctx_totals"].get(prev, 0)
    if ctx_total > 0:
        first = max(model["bi"].get((prev, w), 0) - D, 0) / ctx_total
        lam = D * model["ctx_types"][prev] / ctx_total
        return first + lam * model["p_cont"].get(w, 1 / model["V"])
    return model["p_cont"].get(w, 1 / model["V"])


def perplexity_kn(model, corpus) -> float:
    total_log = 0.0; total_n = 0
    for s in corpus:
        s = _pad(s, 2)
        for i in range(1, len(s)):
            p = _p_kn(model, s[i - 1], s[i])
            total_log += math.log(p + 1e-12); total_n += 1
    return math.exp(-total_log / max(total_n, 1))


if __name__ == "__main__":
    # tiny corpus of short sentences
    train = [
        "the cat sat on the mat".split(),
        "the dog sat on the mat".split(),
        "the cat chased the mouse".split(),
        "a dog chased a cat".split(),
        "the mouse ate cheese".split(),
        "the cat ate fish".split(),
        "a small dog sat quietly".split(),
    ] * 3
    test = [
        "the cat ate cheese".split(),
        "a dog sat on the mat".split(),
    ]

    for n in (1, 2, 3):
        m = train_laplace_ngram(train, n=n, alpha=1.0)
        ppl = perplexity(m, test)
        print(f"  {n}-gram Laplace(alpha=1) test perplexity = {ppl:.2f}")

    kn = train_kn_bigram(train, D=0.75)
    ppl_kn = perplexity_kn(kn, test)
    print(f"  bigram Kneser-Ney (D=0.75) test perplexity = {ppl_kn:.2f}")

    # generation demo: sample from Laplace trigram
    m3 = train_laplace_ngram(train, n=3, alpha=1.0)
    rng = np.random.default_rng(0)
    def _sample():
        ctx = ("<s>", "<s>")
        out = []
        for _ in range(20):
            c = m3["counts"].get(ctx, Counter())
            words = list(c.keys()) + [w for w in m3["vocab"] if w not in c]
            probs = np.array([_p_laplace(m3, ctx, w) for w in words])
            probs = probs / probs.sum()
            w = str(rng.choice(words, p=probs))
            if w == "</s>":
                break
            out.append(w)
            ctx = (ctx[1], w)
        return " ".join(out)
    print(f"\n  sampled trigram sentence: \"{_sample()}\"")
    print(f"  sampled trigram sentence: \"{_sample()}\"")

    print("\n--- library cross-check (nltk.lm.MLE / KneserNeyInterpolated) ---")
