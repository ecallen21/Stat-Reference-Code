"""Clinical NLP: concept extraction + negation (Reference Sec 42.5).

cTAKES / MedSpaCy / scispacy identify clinical concepts in free
text: DRUGS, DIAGNOSES, PROCEDURES, LABS, and their MODIFIERS
(negation, uncertainty, family history, historical).

Compact rule-based demo:
  * Match concepts from a small clinical lexicon.
  * NegEx (Chapman 2001): scan a window before/after the concept
    for a negation trigger ("no", "denies", "without", "ruled
    out").  Flag the concept as NEGATED, HYPOTHETICAL, or POSITIVE.
"""
from __future__ import annotations    # stdlib

import re


LEXICON = {
    "pneumonia":  "diagnosis",
    "asthma":     "diagnosis",
    "diabetes":   "diagnosis",
    "fever":      "symptom",
    "cough":      "symptom",
    "chest pain": "symptom",
    "aspirin":    "drug",
    "insulin":    "drug",
    "metformin":  "drug",
}
NEG_TRIGGERS = ["no", "denies", "without", "no evidence of", "ruled out", "negative for"]
UNCERTAIN    = ["possible", "probable", "cannot rule out", "suspected"]
WINDOW = 6                     # tokens before the concept


def extract(text):
    text_lc = text.lower()
    hits = []
    for concept in LEXICON:
        for m in re.finditer(rf"\b{re.escape(concept)}\b", text_lc):
            # Same-sentence context only: back up to last '.', '?', or '!'
            sent_start = max(text_lc.rfind(".", 0, m.start()),
                             text_lc.rfind("?", 0, m.start()),
                             text_lc.rfind("!", 0, m.start())) + 1
            pre = text_lc[sent_start:m.start()]
            pre_tokens = pre.split()[-WINDOW:]
            pre_txt = " ".join(pre_tokens)
            status = "POSITIVE"
            if any(t in pre_txt for t in UNCERTAIN):
                status = "UNCERTAIN"
            if any(t in pre_txt for t in NEG_TRIGGERS):
                status = "NEGATED"
            hits.append({"concept": concept, "type": LEXICON[concept],
                         "status": status, "char_start": m.start()})
    return hits


if __name__ == "__main__":
    print("=== Clinical NLP: concept + negation extraction (NegEx-style) ===\n")
    notes = [
        "Patient with a 3-day history of cough and fever.  Started on aspirin.",
        "No evidence of pneumonia on chest X-ray.  Continues insulin.",
        "Ruled out diabetes.  Possible asthma exacerbation.",
        "Denies chest pain.  History of metformin use.",
    ]
    for i, n in enumerate(notes):
        print(f"  Note {i + 1}: {n!r}")
        for h in extract(n):
            print(f"    {h['type']:>9s}  {h['concept']:<12s}  status = {h['status']}")
        print()
    print("--- library cross-check (R clinspacy; Python medspacy/scispacy + negspacy) ---")
