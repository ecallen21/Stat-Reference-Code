"""Dictionary-based text scoring (Reference Sec 42.14).

Count words belonging to predefined semantic categories -- the LIWC
paradigm.  Fast, transparent, and reproducible; performance depends
entirely on dictionary quality and domain match.

Deliverable per document:
  score(category) = (# tokens in category) / (# tokens in document)
"""
from __future__ import annotations    # stdlib

import re


DICTIONARY = {
    "positive_emotion": {"good", "great", "excellent", "happy", "love", "wonderful", "improved"},
    "negative_emotion": {"bad", "poor", "sad", "angry", "hate", "terrible", "worse"},
    "health":           {"pain", "cough", "fever", "hospital", "doctor", "clinic", "diagnosis"},
    "certainty":        {"sure", "certain", "definitely", "always", "never"},
    "hedging":          {"maybe", "perhaps", "possibly", "might", "seems", "appears"},
}


def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


def score(text, dictionary=DICTIONARY):
    toks = tokenize(text)
    n = max(len(toks), 1)
    out = {}
    for cat, words in dictionary.items():
        c = sum(1 for t in toks if t in words)
        out[cat] = {"count": c, "prop": c / n}
    return {"n_tokens": n, "scores": out}


if __name__ == "__main__":
    print("=== Dictionary-based text scoring (LIWC-style) ===\n")
    docs = [
        "The patient had a great improvement -- love the new medication.",
        "This is terrible; the pain is worse and hospital visits are frequent.",
        "Definitely certain the diagnosis is correct.",
        "Perhaps the cough might be viral; possibly requires further work-up.",
    ]
    for i, d in enumerate(docs):
        r = score(d)
        cats = "  ".join(f"{k}={v['prop']:.2f}" for k, v in r["scores"].items())
        print(f"  doc {i} (n={r['n_tokens']:>2d}): {d!r}")
        print(f"    {cats}\n")

    print("--- library cross-check (R quanteda::dictionary/tokens_lookup, LIWC (commercial); Python empath, custom) ---")
