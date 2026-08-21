"""Modern generation-evaluation metrics: chrF and BERTScore surrogate (Reference §25.x extra).

  * chrF (Popovic 2015): F-measure over character n-grams.
        P = |chrN(cand) intersect chrN(ref)| / |chrN(cand)|
        R = |chrN(cand) intersect chrN(ref)| / |chrN(ref)|
        F_beta = (1 + beta^2) * P * R / (R + beta^2 P),   beta = 2 (chrF++ uses beta=2)
    Strong for morphologically rich languages; language-agnostic.

  * BERTScore (Zhang 2020): cosine-similarity F1 over CONTEXTUAL token
    embeddings.  Needs a pretrained encoder.  Here we mock with static
    embeddings to demonstrate the machinery — real BERTScore uses BERT-large
    contextual vectors and reports higher correlation with human judgement.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

from collections import Counter    # stdlib: bag counts

import numpy as np    # numerical arrays + linear algebra


def chrf(cand: str, ref: str, n_max: int = 6, beta: float = 2.0) -> dict:
    """Character n-gram F_beta (chrF++ with word n-grams uses beta = 2)."""
    cand = cand.strip(); ref = ref.strip()
    if not cand or not ref:
        return {"chrF": 0.0, "precisions": [], "recalls": []}
    ps = []; rs = []
    for n in range(1, n_max + 1):
        cg = Counter(cand[i: i + n] for i in range(len(cand) - n + 1))
        rg = Counter(ref[i: i + n] for i in range(len(ref) - n + 1))
        inter = sum((cg & rg).values())
        p = inter / max(sum(cg.values()), 1)
        r = inter / max(sum(rg.values()), 1)
        ps.append(p); rs.append(r)
    P = np.mean(ps); R = np.mean(rs)
    F = (1 + beta ** 2) * P * R / max(R + beta ** 2 * P, 1e-12)
    return {"chrF": float(F), "precisions": ps, "recalls": rs}


def bert_score_surrogate(cand_tokens, ref_tokens, vecs: dict) -> dict:
    """Cosine-similarity F1 over token embeddings (surrogate for real BERTScore)."""
    def _mat(toks):
        V = np.stack([vecs[t] for t in toks if t in vecs])
        return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    A = _mat(cand_tokens); B = _mat(ref_tokens)
    if A.size == 0 or B.size == 0:
        return {"F1": 0.0, "P": 0.0, "R": 0.0}
    sim = A @ B.T                                             # (nc, nr)
    P = sim.max(axis=1).mean()
    R = sim.max(axis=0).mean()
    F = 2 * P * R / max(P + R, 1e-12)
    return {"F1": float(F), "P": float(P), "R": float(R)}


if __name__ == "__main__":
    ref = "the cat sat on the mat"
    cands = [
        "the cat sat on the mat",
        "a cat sat on a mat",
        "the feline rested upon the rug",                    # paraphrase
        "dog barks loudly",                                   # unrelated
    ]
    print(f"=== chrF (character n-grams, beta = 2) ===")
    for c in cands:
        r = chrf(c, ref)
        print(f"  chrF = {r['chrF']:.3f}   \"{c}\"")

    # toy static embeddings that place synonyms nearby
    def _v(*x): return np.array(x, dtype=float)
    vecs = {
        "the": _v(0.1, 0.1),  "a": _v(0.1, 0.1),
        "cat": _v(1.0, 0.0),  "feline": _v(0.95, 0.1),
        "sat": _v(0.0, 1.0),  "rested": _v(0.1, 0.95),
        "on": _v(0.05, 0.05), "upon": _v(0.05, 0.1),
        "mat": _v(0.5, 0.5),  "rug": _v(0.55, 0.45),
        "dog": _v(0.9, -0.5), "barks": _v(-0.9, 0.5),
        "loudly": _v(-0.5, -0.5),
    }
    print(f"\n=== BERTScore surrogate (toy embeddings; real uses BERT contextual) ===")
    for c in cands:
        r = bert_score_surrogate(c.split(), ref.split(), vecs)
        print(f"  BERTScore-F1 = {r['F1']:.3f}   \"{c}\"")

    print("\n--- library cross-check (bert_score; sacrebleu chrF; unbabel-comet) ---")
