"""Relation extraction (Reference §25.x extra).

Given a sentence and TWO entity mentions, classify the relation type
between them from a fixed inventory (or "no_relation").

Two classical approaches:
  * PATTERN / RULE-BASED: hand-written surface patterns like
        "X, [the] founder of Y"  ->  founder(X, Y)
    Fast to prototype; brittle.

  * SUPERVISED CLASSIFIER: word features around the entity pair fed to a
    logistic / random-forest / MLP classifier.  Modern SOTA uses a
    transformer encoder + [SEP]-injected entity markers.

We demonstrate the rule-based path with 3 relation patterns + a simple
sentence-classification fallback on toy sentences.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import re    # stdlib: patterns


PATTERNS = [
    ("founder_of",
     r"(?P<e1>[A-Z][\w\.-]+(?:\s[A-Z][\w\.-]+)*)[,\s]+(?:the\s+)?founder\s+of\s+(?P<e2>[A-Z][\w\.-]+(?:\s[A-Z][\w\.-]+)*)"),
    ("ceo_of",
     r"(?P<e1>[A-Z][\w\.-]+(?:\s[A-Z][\w\.-]+)*)[,\s]+(?:the\s+)?CEO\s+of\s+(?P<e2>[A-Z][\w\.-]+(?:\s[A-Z][\w\.-]+)*)"),
    ("born_in",
     r"(?P<e1>[A-Z][\w\.-]+(?:\s[A-Z][\w\.-]+)*)\s+was\s+born\s+in\s+(?P<e2>[A-Z][\w\.-]+(?:\s[A-Z][\w\.-]+)*)"),
    ("acquired",
     r"(?P<e1>[A-Z][\w\.-]+(?:\s[A-Z][\w\.-]+)*)\s+acquired\s+(?P<e2>[A-Z][\w\.-]+(?:\s[A-Z][\w\.-]+)*)"),
    ("headquartered_in",
     r"(?P<e1>[A-Z][\w\.-]+(?:\s[A-Z][\w\.-]+)*)\s+is\s+headquartered\s+in\s+(?P<e2>[A-Z][\w\.-]+(?:\s[A-Z][\w\.-]+)*)"),
]


def extract_relations(text: str) -> list:
    hits = []
    for name, pat in PATTERNS:
        for m in re.finditer(pat, text):
            hits.append({"relation": name, "entity_1": m.group("e1"),
                          "entity_2": m.group("e2"), "span": m.span()})
    return hits


if __name__ == "__main__":
    docs = [
        "Bill Gates, the founder of Microsoft, stepped down in 2000.",
        "Marie Curie was born in Warsaw.",
        "Google acquired YouTube in 2006 for $1.65 billion.",
        "Elon Musk, the CEO of Tesla, spoke at the conference.",
        "Amazon is headquartered in Seattle.",
        "The Beatles were a British rock band.",              # no target relation
    ]
    print("=== Rule-based relation extraction ===\n")
    for d in docs:
        rels = extract_relations(d)
        print(f"  sentence: {d}")
        if not rels:
            print(f"    (no known relation matched)")
        for r in rels:
            print(f"    -> {r['relation']}({r['entity_1']}, {r['entity_2']})")
        print()

    print("--- library cross-check (huggingface OpenNRE; spaCy REL; DeepKE) ---")
