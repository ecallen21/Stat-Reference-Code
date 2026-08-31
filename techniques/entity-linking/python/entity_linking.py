"""Entity linking (Reference §25.x extra).

Given a mention M in a context sentence, LINK it to a canonical entity in
a knowledge base (Wikidata / Wikipedia / a domain KB) or return NIL.

Two subtasks:
  1. CANDIDATE GENERATION: from the mention string, retrieve a shortlist of
     KB entities (alias index, TF-IDF, phonetic hashing).
  2. RANKING / DISAMBIGUATION: use context to pick the right entity.
     Baseline: bag-of-context-words vs each candidate's description; cosine.
     SOTA: BLINK (Wu 2020), GENRE (Cao 2021), ReFinED (Ayoola 2022).

Here we implement a small alias-index + context-cosine ranker over a mini KB.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import re    # stdlib

from collections import Counter    # stdlib

import numpy as np    # numerical arrays + linear algebra


# tiny knowledge base
KB = {
    "Q1": {"name": "Michael Jordan",       "aliases": ["Michael Jordan", "MJ", "M. Jordan"],
            "description": "American professional basketball player Chicago Bulls NBA six championships"},
    "Q2": {"name": "Michael I. Jordan",    "aliases": ["Michael Jordan", "M.I. Jordan", "Michael I. Jordan"],
            "description": "American computer scientist statistical machine learning Berkeley professor"},
    "Q3": {"name": "Apple Inc.",           "aliases": ["Apple", "Apple Inc.", "Apple Computer"],
            "description": "American multinational technology company iPhone iPad Mac Cupertino"},
    "Q4": {"name": "Apple (fruit)",        "aliases": ["Apple", "apples"],
            "description": "Edible fruit produced by an apple tree Malus domestica grown in orchards"},
    "Q5": {"name": "Paris, France",        "aliases": ["Paris"],
            "description": "Capital of France European city Eiffel Tower Seine Louvre"},
    "Q6": {"name": "Paris, Texas",         "aliases": ["Paris"],
            "description": "City in Lamar County Texas United States"},
}


def _tokens(s):
    return set(re.findall(r"\w+", s.lower()))


def candidate_generation(mention: str, kb: dict = KB) -> list:
    m = mention.lower()
    return [k for k, v in kb.items() if m in [a.lower() for a in v["aliases"]]]


def rank_by_context(mention: str, context: str, candidates: list,
                     kb: dict = KB) -> list:
    ctx = _tokens(context) - _tokens(mention)
    scored = []
    for cid in candidates:
        desc = _tokens(kb[cid]["description"])
        overlap = len(ctx & desc) / max(len(desc), 1) ** 0.5
        scored.append((cid, overlap))
    scored.sort(key=lambda x: -x[1])
    return scored


def link_mention(mention: str, context: str, kb: dict = KB) -> dict:
    cands = candidate_generation(mention, kb)
    if not cands:
        return {"linked": None, "reason": "no candidate"}
    scored = rank_by_context(mention, context, cands, kb)
    return {"linked": scored[0][0], "linked_name": kb[scored[0][0]]["name"],
            "scores": [(kb[c]["name"], round(s, 3)) for c, s in scored],
            "method": "alias-index + context-cosine EL"}


if __name__ == "__main__":
    tests = [
        ("Michael Jordan", "Michael Jordan won six NBA championships with the Chicago Bulls.", "Q1"),
        ("Michael Jordan", "Michael Jordan is a professor of computer science at Berkeley.", "Q2"),
        ("Apple", "Apple released a new iPhone last quarter.", "Q3"),
        ("Apple", "I ate an apple for lunch; it was crunchy and sweet.", "Q4"),
        ("Paris", "Paris is on the Seine river and hosts the Eiffel Tower.", "Q5"),
        ("Paris", "Paris is a small city in Lamar County Texas.", "Q6"),
    ]
    print("=== Entity linking (alias-index + context-cosine) ===\n")
    hits = 0
    for mention, context, gold in tests:
        r = link_mention(mention, context)
        ok = r["linked"] == gold
        hits += ok
        print(f"  [{'ok' if ok else '..'}] mention: '{mention}'   gold: {gold} ({KB[gold]['name']})")
        print(f"        pred: {r['linked']} ({r['linked_name']})")
        print(f"        scores: {r['scores']}\n")
    print(f"  {hits}/{len(tests)} correct.")

    print("--- library cross-check (BLINK, GENRE, ReFinED; spaCy EntityLinker; wikidata sparql) ---")
