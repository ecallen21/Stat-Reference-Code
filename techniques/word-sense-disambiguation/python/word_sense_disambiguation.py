"""Word-sense disambiguation (Reference §25.x extra).

Choose the intended sense of an ambiguous word given its context.  Two
classical approaches:

  * LESK (1986): score each candidate sense by the count of overlapping words
    between its dictionary gloss and the target word's context.
        best_sense = argmax_s |context(w) ∩ gloss(s)|
    Extended Lesk allows glosses of related senses (hypernyms, hyponyms).

  * EMBEDDING-BASED WSD: compare cosine similarity between the average
    context-word embedding and the average gloss-word embedding of each
    sense.  A strong classical baseline before transformer-based WSD
    (BERT-context vs sense-key vectors from GlossBERT / AWE).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


STOPWORDS = {"the", "a", "an", "to", "of", "in", "on", "at", "for", "with",
              "and", "or", "is", "it", "he", "she", "they", "we", "you"}


def _tokens(s):
    return [w.lower() for w in s.split() if w.lower() not in STOPWORDS]


def lesk(target: str, context: str, sense_glosses: dict) -> dict:
    """sense_glosses: {sense_name: gloss_string}."""
    ctx_bag = set(_tokens(context)) - {target.lower()}
    scores = {}
    for sense, gloss in sense_glosses.items():
        overlap = ctx_bag & set(_tokens(gloss))
        scores[sense] = len(overlap)
    best = max(scores, key=scores.get)
    return {"predicted_sense": best, "overlaps": scores}


def embedding_wsd(target: str, context: str, sense_glosses: dict,
                   vecs: dict) -> dict:
    """Cosine similarity between averaged context and averaged gloss vectors."""
    def _mean(tokens):
        v = [vecs[t] for t in tokens if t in vecs]
        return np.mean(v, axis=0) if v else None
    ctx_vec = _mean([t for t in _tokens(context) if t != target.lower()])
    if ctx_vec is None:
        return {"predicted_sense": next(iter(sense_glosses))}
    def _cos(u, v): return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    scores = {}
    for sense, gloss in sense_glosses.items():
        gv = _mean(_tokens(gloss))
        if gv is None:
            scores[sense] = 0.0
        else:
            scores[sense] = _cos(ctx_vec, gv)
    best = max(scores, key=scores.get)
    return {"predicted_sense": best, "cosines": scores}


if __name__ == "__main__":
    # Target: "bank" — two senses
    senses = {
        "bank.financial": "financial institution money account loan credit cash deposit",
        "bank.river": "sloping land beside river water stream shore",
    }
    tests = [
        ("bank", "I deposited cash at the bank yesterday", "bank.financial"),
        ("bank", "we walked along the river bank at dawn", "bank.river"),
        ("bank", "the credit union offered a loan at the bank", "bank.financial"),
        ("bank", "wild flowers grew on the shore near the bank", "bank.river"),
    ]

    print("=== Lesk WSD (gloss-overlap) ===")
    correct = 0
    for w, ctx, truth in tests:
        r = lesk(w, ctx, senses)
        ok = r["predicted_sense"] == truth
        correct += ok
        print(f"  [{'ok' if ok else '!!'}] truth={truth}  pred={r['predicted_sense']}  "
              f"overlaps={r['overlaps']}   \"{ctx}\"")
    print(f"  Lesk accuracy: {correct}/{len(tests)}")

    # Toy embeddings: financial-sense words point in one direction, river-sense in another
    def _v(*x): return np.array(x, dtype=float)
    vecs = {
        "deposited": _v(1, 0), "cash": _v(1, 0), "credit": _v(1, 0),
        "loan": _v(1, 0), "union": _v(1, 0), "offered": _v(0.5, 0.1),
        "yesterday": _v(0.2, 0.1),
        "river": _v(0, 1), "walked": _v(0.1, 0.5), "shore": _v(0, 1),
        "dawn": _v(0, 0.7), "wild": _v(0, 0.6), "flowers": _v(0, 0.7),
        "grew": _v(0, 0.6), "near": _v(0, 0.4),
        "financial": _v(1, 0), "institution": _v(1, 0), "money": _v(1, 0),
        "account": _v(1, 0), "deposit": _v(1, 0),
        "sloping": _v(0, 1), "land": _v(0, 0.7), "beside": _v(0, 0.5),
        "water": _v(0, 1), "stream": _v(0, 1),
        "at": _v(0.1, 0.1), "the": _v(0.1, 0.1), "we": _v(0.1, 0.1),
        "along": _v(0.1, 0.4), "on": _v(0.1, 0.1),
    }
    print("\n=== Embedding-based WSD (bag-of-embeddings cosine) ===")
    correct = 0
    for w, ctx, truth in tests:
        r = embedding_wsd(w, ctx, senses, vecs)
        ok = r["predicted_sense"] == truth
        correct += ok
        cos_str = {k: round(v, 3) for k, v in r["cosines"].items()}
        print(f"  [{'ok' if ok else '!!'}] truth={truth}  pred={r['predicted_sense']}  "
              f"cos={cos_str}   \"{ctx}\"")
    print(f"  embedding-WSD accuracy: {correct}/{len(tests)}")

    print("\n--- library cross-check (nltk wordnet + lesk; pywsd; GlossBERT via HF) ---")
