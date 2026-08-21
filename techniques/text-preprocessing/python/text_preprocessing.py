"""Text preprocessing basics: tokenisation, stopwords, stemming, lemmatisation
(Reference §25.1).

  * TOKENISATION  — split text into tokens (words / subwords / characters).
  * NORMALISATION — lowercasing, punctuation stripping, unicode NFKC.
  * STOPWORD REMOVAL — drop function words that carry little topical signal.
  * STEMMING       — Porter stemmer (rule-based suffix stripping).
  * LEMMATISATION  — dictionary-based reduction to canonical form.

Modern NLP replaces most of this with subword tokenisers (BPE, WordPiece,
SentencePiece) that preserve morphology. For classical bag-of-words models
these steps are still standard.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import re    # stdlib: regex for tokenisation

import unicodedata    # stdlib: unicode normalisation


ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "her", "his", "in", "is", "it", "its", "of", "on",
    "she", "that", "the", "they", "this", "to", "was", "were", "with",
    "will", "you", "your", "we", "our", "or", "not", "but", "if", "so",
    "have", "had", "do", "does", "did", "would", "can", "could",
}


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s'-]", " ", text)                 # strip non-word chars
    return re.sub(r"\s+", " ", text).strip()


def tokenise(text: str) -> list:
    return normalise(text).split()


def remove_stopwords(tokens, stopwords=ENGLISH_STOPWORDS) -> list:
    return [t for t in tokens if t not in stopwords]


def _porter_step1(word):
    """Simplified Porter stemmer step 1 (plurals + past participles)."""
    if word.endswith("sses"): return word[:-2]
    if word.endswith("ies"):  return word[:-2]
    if word.endswith("ss"):    return word
    if word.endswith("s"):     return word[:-1]
    return word


def _porter_step2(word):
    if word.endswith("eed"):
        return word[:-1] if len(word) > 4 else word
    for suf in ("ed", "ing"):
        if word.endswith(suf) and any(c in "aeiou" for c in word[: -len(suf)]):
            return word[: -len(suf)]
    return word


def stem(word: str) -> str:
    return _porter_step2(_porter_step1(word))


_LEMMA_MAP = {
    "geese": "goose", "mice": "mouse", "men": "man", "women": "woman",
    "children": "child", "feet": "foot", "teeth": "tooth", "leaves": "leaf",
    "went": "go", "gone": "go", "goes": "go", "going": "go",
    "was": "be", "were": "be", "been": "be", "am": "be", "is": "be", "are": "be",
    "had": "have", "has": "have", "having": "have",
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
}


def lemmatise(word: str) -> str:
    if word in _LEMMA_MAP:
        return _LEMMA_MAP[word]
    return stem(word)                                     # fallback


def preprocess(text: str, *, remove_stop: bool = True, stem_or_lemma: str = "stem") -> list:
    toks = tokenise(text)
    if remove_stop:
        toks = remove_stopwords(toks)
    if stem_or_lemma == "stem":
        toks = [stem(t) for t in toks]
    elif stem_or_lemma == "lemma":
        toks = [lemmatise(t) for t in toks]
    return toks


if __name__ == "__main__":
    docs = [
        "The children were running quickly through the park.",
        "Mice ran across the leaves; geese honked in the distance.",
        "She goes to the market where she has bought apples.",
    ]

    print("=== Text preprocessing pipeline ===\n")
    for d in docs:
        print(f"  raw           : {d!r}")
        print(f"  tokenised     : {tokenise(d)}")
        print(f"  no stopwords  : {remove_stopwords(tokenise(d))}")
        print(f"  stemmed       : {preprocess(d, stem_or_lemma='stem')}")
        print(f"  lemmatised    : {preprocess(d, stem_or_lemma='lemma')}")
        print()

    print("--- library cross-check (nltk / spacy / stanza) ---")
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        from nltk.tokenize import word_tokenize
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        d = docs[0]
        nltk_toks = word_tokenize(d.lower())
        print(f"  nltk tokens for doc 0: {nltk_toks[:8]}...")
        print(f"  nltk Porter stems    : {[ps.stem(t) for t in nltk_toks][:8]}...")
    except Exception as e:
        print(f"  (nltk unavailable: {type(e).__name__})")
