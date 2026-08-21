"""Coreference resolution: mention-pair scoring (Reference §25.x extra).

Task: link mentions that refer to the same real-world entity.
    "Alice went to the store.  She bought milk."   -> {Alice, She}
    "Bob loves his dog.  The dog barks."           -> {Bob, his}, {dog, The dog}

Two classical approaches:

  * Rule-based (Hobbs 1978): syntactic + lexical rules for pronoun resolution.
    Backbone of many pre-neural systems; brittle but interpretable.

  * Mention-pair scoring (Soon-Ng-Lim 2001; Bengtson-Roth 2008): for each
    (candidate antecedent, anaphor) pair, score compatibility with a
    classifier over features (gender, number, string match, distance,
    animacy, ...).  Cluster mentions greedily by pair-score.

We implement a simple rule + feature-based mention-pair scorer on toy
sentences with hand-labelled mentions.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


# hand-crafted feature lexicon
GENDER = {"alice": "F", "she": "F", "her": "F", "hers": "F",
          "bob": "M", "he": "M", "his": "M", "him": "M",
          "carol": "F", "david": "M", "eve": "F",
          "dog": "N", "cat": "N", "book": "N", "car": "N", "house": "N"}
NUMBER = {"they": "PL", "them": "PL", "their": "PL"}
ANIMATE = {"alice": True, "bob": True, "carol": True, "david": True, "eve": True,
            "she": True, "he": True, "they": True, "him": True, "her": True,
            "dog": True, "cat": True,
            "book": False, "car": False, "house": False}


def _lower(m): return " ".join(w.lower() for w in m)


def _head(m): return m[-1].lower()


def _gender(m):
    return GENDER.get(_head(m), "?")


def _number(m):
    h = _head(m)
    if h in NUMBER: return NUMBER[h]
    if h.endswith("s") and h not in ("she", "his", "hers"): return "PL"
    return "SG"


def _animate(m): return ANIMATE.get(_head(m), False)


def _pair_features(anaphor, antecedent, distance):
    fa_g = _gender(anaphor); fb_g = _gender(antecedent)
    fa_n = _number(anaphor); fb_n = _number(antecedent)
    return {
        "gender_agree": int(fa_g in (fb_g, "?") or fb_g == "?"),
        "number_agree": int(fa_n == fb_n),
        "animacy_agree": int(_animate(anaphor) == _animate(antecedent)),
        "string_match": int(_lower(anaphor) == _lower(antecedent)),
        "head_match": int(_head(anaphor) == _head(antecedent)),
        "distance": distance,
        "anaphor_is_pronoun": int(_head(anaphor) in
                                  {"he", "she", "it", "they", "him", "her",
                                   "his", "hers", "them", "their"}),
    }


def score_pair(features) -> float:
    """Hand-crafted scoring: gender + number + animacy + head-match + distance."""
    s = 0.0
    s += 2.0 * features["gender_agree"]
    s += 2.0 * features["number_agree"]
    s += 1.5 * features["animacy_agree"]
    s += 3.0 * features["head_match"]
    s -= 0.4 * features["distance"]
    return s


def cluster_mentions(mentions) -> list:
    """Greedy: for each mention (left-to-right), link to the best antecedent above threshold."""
    parent = list(range(len(mentions)))
    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def _union(i, j):
        parent[_find(i)] = _find(j)
    for i in range(1, len(mentions)):
        best = None; best_score = 0.5
        for j in range(i):
            feats = _pair_features(mentions[i], mentions[j], i - j)
            if not feats["gender_agree"] or not feats["number_agree"]:
                continue
            if not feats["animacy_agree"]:
                continue
            s = score_pair(feats)
            if s > best_score:
                best = j; best_score = s
        if best is not None:
            _union(i, best)
    clusters = {}
    for i in range(len(mentions)):
        clusters.setdefault(_find(i), []).append(mentions[i])
    return list(clusters.values())


if __name__ == "__main__":
    docs = [
        [("Alice",), ("Alice",), ("she",), ("her",)],           # 4 mentions of Alice
        [("Bob",), ("his", "dog"), ("The", "dog"), ("Bob",)],    # 2 chains
        [("Carol",), ("David",), ("she",), ("him",)],            # she->Carol, him->David
    ]
    print("=== Mention-pair coreference (rule + features + greedy clustering) ===\n")
    for mentions in docs:
        clusters = cluster_mentions(mentions)
        print(f"  input mentions : {[' '.join(m) for m in mentions]}")
        for c in clusters:
            print(f"    cluster: {[' '.join(m) for m in c]}")
        print()

    print("--- library cross-check (spaCy coref extension; huggingface allennlp; NeuralCoref) ---")
