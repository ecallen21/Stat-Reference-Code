"""Readability indices (Reference Sec 42.16).

Formula-based estimates of grade level required to comfortably read
a passage.  All rely on syllable / word / sentence counts:

  FLESCH READING EASE:     206.835 - 1.015 * ASL - 84.6 * ASW
  FLESCH-KINCAID GRADE:     0.39 * ASL + 11.8 * ASW - 15.59
  GUNNING FOG INDEX:        0.4 * (ASL + %complex_words * 100)
  SMOG:                    1.0430 * sqrt(polysyllables * 30 / n_sentences) + 3.1291
  COLEMAN-LIAU INDEX:       0.0588 * L - 0.296 * S - 15.8
                            (L, S per-100-words letter/sentence counts)

Ubiquitous in health-literacy assessment of patient materials;
Crossley 2008 warns formulaic scores miss cohesion, coherence,
vocabulary depth.
"""
from __future__ import annotations    # stdlib

import re


VOWELS = "aeiouy"

def _sentences(text):
    return [s for s in re.split(r"[.!?]+", text) if s.strip()]

def _words(text):
    return re.findall(r"[a-zA-Z']+", text)

def _syllables(word):
    """Naive syllable count."""
    word = word.lower()
    count = 0; prev = False
    for c in word:
        v = c in VOWELS
        if v and not prev:
            count += 1
        prev = v
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def readability(text):
    sents = _sentences(text); words = _words(text)
    n_sent = max(len(sents), 1); n_word = max(len(words), 1)
    syllables = [_syllables(w) for w in words]
    ASL = n_word / n_sent                       # avg sentence length in words
    ASW = sum(syllables) / n_word               # avg syllables per word
    complex_ = sum(1 for s in syllables if s >= 3)
    pct_complex = complex_ / n_word
    n_letters = sum(len(w) for w in words)
    L = n_letters / n_word * 100
    S = n_sent / n_word * 100

    flesch = 206.835 - 1.015 * ASL - 84.6 * ASW
    fk = 0.39 * ASL + 11.8 * ASW - 15.59
    fog = 0.4 * (ASL + pct_complex * 100)
    smog = 1.0430 * ((complex_ * 30 / n_sent) ** 0.5) + 3.1291 if n_sent >= 3 else None
    cli = 0.0588 * L - 0.296 * S - 15.8
    return {"n_words": n_word, "n_sentences": n_sent, "ASL": ASL, "ASW": ASW,
            "Flesch": flesch, "Flesch_Kincaid": fk, "Gunning_Fog": fog,
            "SMOG": smog, "Coleman_Liau": cli}


if __name__ == "__main__":
    print("=== Readability indices (Flesch, F-K, Fog, SMOG, C-L) ===\n")
    simple = ("The cat sat on the mat. The dog ran to the tree. "
              "The bird flew away.  We watched them play.  It was fun.")
    complex_ = ("Notwithstanding the aforementioned considerations regarding the "
                "epidemiological ramifications of pharmacotherapeutic interventions, "
                "the clinician must nevertheless prioritise individualised assessment "
                "of comorbidities and heterogeneous therapeutic responsiveness.")
    for name, text in [("simple", simple), ("complex", complex_)]:
        r = readability(text)
        print(f"  {name}: n_words = {r['n_words']}, n_sents = {r['n_sentences']}")
        print(f"    Flesch Reading Ease = {r['Flesch']:.1f}   (higher = easier)")
        print(f"    Flesch-Kincaid Grade = {r['Flesch_Kincaid']:.1f}")
        print(f"    Gunning Fog          = {r['Gunning_Fog']:.1f}")
        print(f"    SMOG                 = {r['SMOG']:.1f}" if r["SMOG"] else "    SMOG                 = n/a")
        print(f"    Coleman-Liau         = {r['Coleman_Liau']:.1f}\n")

    print("--- library cross-check (R quanteda::textstat_readability, koRpus; Python textstat) ---")
