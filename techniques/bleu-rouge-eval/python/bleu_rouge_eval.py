"""BLEU and ROUGE for text-generation evaluation (Reference §25.x extra).

BLEU (Papineni et al. 2002): modified n-gram precision with a brevity penalty.
    BLEU-N = BP * exp( sum_n w_n * log p_n )
    p_n = sum_ngram min(count_cand(g), max_ref(g)) / sum_ngram count_cand(g)
    BP = 1 if |c| > |r| else exp(1 - |r| / |c|)

ROUGE-L (Lin 2004): longest-common-subsequence based F-measure between
candidate and reference; used for summarisation evaluation.

Both are surface-form metrics; they don't measure meaning.  Modern practice
adds BERTScore, COMET (translation), and human evaluation.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math

from collections import Counter    # stdlib: bag counts


def _ngrams(seq, n):
    return [tuple(seq[i: i + n]) for i in range(len(seq) - n + 1)]


def bleu_score(candidate, references, weights=(0.25, 0.25, 0.25, 0.25)) -> dict:
    """candidate: list of tokens; references: list of token lists."""
    ps = []
    for n in range(1, len(weights) + 1):
        cand_ngrams = Counter(_ngrams(candidate, n))
        if not cand_ngrams:
            ps.append(0.0); continue
        max_ref = Counter()
        for ref in references:
            for ng, c in Counter(_ngrams(ref, n)).items():
                max_ref[ng] = max(max_ref[ng], c)
        clipped = sum(min(c, max_ref[ng]) for ng, c in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        ps.append(clipped / total)
    # brevity penalty against the closest-length reference
    c = len(candidate)
    r = min((len(ref) for ref in references), key=lambda x: (abs(x - c), x))
    bp = 1.0 if c > r else math.exp(1 - r / c) if c > 0 else 0.0
    if min(ps) == 0.0:
        bleu = 0.0
    else:
        bleu = bp * math.exp(sum(w * math.log(p) for w, p in zip(weights, ps)))
    return {"BLEU": bleu, "brevity_penalty": bp,
            "precisions_by_n": ps,
            "method": "BLEU (Papineni 2002)"}


def rouge_l(candidate, reference) -> dict:
    """ROUGE-L via LCS."""
    m, n = len(candidate), len(reference)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if candidate[i - 1] == reference[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return {"ROUGE-L_P": 0.0, "ROUGE-L_R": 0.0, "ROUGE-L_F": 0.0}
    p = lcs / m; r = lcs / n
    beta = 1.2                                             # ROUGE-L uses beta = 1.2 (recall-weighted)
    f = (1 + beta ** 2) * p * r / (r + beta ** 2 * p)
    return {"ROUGE-L_P": p, "ROUGE-L_R": r, "ROUGE-L_F": f,
            "lcs": lcs, "method": "ROUGE-L (Lin 2004)"}


if __name__ == "__main__":
    ref = "the cat sat on the mat".split()
    cands = [
        "the cat sat on the mat".split(),
        "the cat sat on a mat".split(),
        "a cat is sitting on the mat".split(),
        "dog barks loudly".split(),
        "sat cat the mat the on".split(),                  # shuffled — same unigrams
    ]

    print("=== BLEU-4 + ROUGE-L (single reference) ===\n")
    print(f"  reference : {' '.join(ref)}\n")
    for c in cands:
        b = bleu_score(c, [ref])
        r = rouge_l(c, ref)
        print(f"  candidate : {' '.join(c)!r}")
        print(f"    BLEU-4        = {b['BLEU']:.3f}   "
              f"BP = {b['brevity_penalty']:.3f}   "
              f"p_{{1..4}} = {[round(x, 3) for x in b['precisions_by_n']]}")
        print(f"    ROUGE-L (F)   = {r['ROUGE-L_F']:.3f}   "
              f"(P={r['ROUGE-L_P']:.3f}, R={r['ROUGE-L_R']:.3f})\n")

    print("--- library cross-check (nltk.translate.bleu_score / rouge-score) ---")
    try:
        from nltk.translate.bleu_score import sentence_bleu
        for c in cands[:2]:
            nb = sentence_bleu([ref], c)
            print(f"  nltk BLEU-4 for {c!r} = {nb:.3f}")
    except ImportError:
        print("  (nltk not installed)")
