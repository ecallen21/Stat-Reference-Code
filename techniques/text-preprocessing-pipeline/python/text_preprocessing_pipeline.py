"""Text preprocessing pipeline (Reference Sec 42.11).

Standard NLP preprocessing chain:

  * TOKENISE     : split text into tokens.
  * NORMALISE    : lower-case, strip punctuation.
  * STOP WORDS   : remove function words.
  * STEM         : Porter/Snowball -> crude morphological reduction.
  * LEMMATISE   : dictionary-based reduction to lemma (requires
                  POS + vocabulary).

Manning et al. 2008: preprocessing is task-dependent; over-
aggressive normalisation can destroy the signal (e.g., dropping
digits harms clinical dose extraction).
"""
from __future__ import annotations    # stdlib

import re


STOPWORDS = {"the", "a", "an", "of", "for", "and", "or", "to", "in", "on",
             "with", "at", "is", "was", "were", "are", "be", "been", "by"}


def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[a-z]+)?|\d+", text.lower())


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]


def porter_stem(word):
    """Compact subset of the Porter stemmer suffix rules."""
    suffixes = [("sses", "ss"), ("ies", "i"), ("ss", "ss"), ("s", ""),
                ("ing", ""), ("ed", ""), ("er", ""), ("ly", ""), ("ational", "ate"),
                ("izer", "ize"), ("iveness", "ive"), ("ate", "at")]
    for suf, rep in suffixes:
        if word.endswith(suf) and len(word) - len(suf) > 2:
            return word[: -len(suf)] + rep
    return word


LEMMA = {
    "running": "run", "runs": "run", "ran": "run",
    "better": "good", "best": "good", "good": "good",
    "children": "child", "was": "be", "were": "be", "is": "be",
    "cars": "car", "buses": "bus",
}
def lemmatise(word):
    return LEMMA.get(word, word)


def preprocess(text, stops=True, stem=False, lemma=False):
    toks = tokenize(text)
    if stops:
        toks = remove_stopwords(toks)
    if lemma:
        toks = [lemmatise(t) for t in toks]
    if stem:
        toks = [porter_stem(t) for t in toks]
    return toks


if __name__ == "__main__":
    print("=== Text preprocessing pipeline: tokenise -> stop-words -> stem / lemmatise ===\n")
    text = "The children were running better than the buses -- studying various things carefully."
    print(f"  Raw:            {text!r}")
    print(f"  Tokenised:      {tokenize(text)}")
    print(f"  Minus stopwords:{preprocess(text)}")
    print(f"  Stemmed:        {preprocess(text, stem=True)}")
    print(f"  Lemmatised:     {preprocess(text, lemma=True)}")
    print(f"  Both:           {preprocess(text, lemma=True, stem=True)}\n")

    print("--- library cross-check (R quanteda tokens/tokens_stem, tidytext; Python nltk, spacy, sklearn) ---")
