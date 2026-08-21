"""Sentiment analysis: lexicon-based + supervised (Reference §25.7).

  * LEXICON-BASED (VADER-style):
        polarity(doc) = sum_w lexicon[w] * (negation_flip if any) * intensifier
    Uses a small dictionary of scored words plus valence shifters (negation,
    intensifier).

  * SUPERVISED (logistic on TF-IDF):
        train from labelled examples using the standard text-classification
        pipeline.

VADER (Hutto-Gilbert 2014) tuned for social media; TextBlob uses a smaller
adjective lexicon; production systems fine-tune a transformer (RoBERTa-STS,
DeBERTa-large etc.).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import re    # stdlib: regex

import numpy as np    # numerical arrays + linear algebra

LEXICON = {
    "great": +2.5, "excellent": +3.0, "good": +1.8, "love": +2.7, "loved": +2.5,
    "amazing": +2.8, "fantastic": +2.7, "wonderful": +2.5, "perfect": +2.9,
    "happy": +1.9, "enjoy": +1.6, "enjoyed": +1.6, "best": +2.0,
    "bad": -2.0, "terrible": -2.8, "awful": -2.7, "hate": -2.7, "hated": -2.6,
    "worst": -2.8, "poor": -1.5, "boring": -1.8, "disappoint": -2.2,
    "sad": -1.7, "angry": -1.8, "annoying": -1.7, "waste": -2.2, "broken": -1.8,
}
NEGATIONS = {"not", "no", "never", "nothing", "hardly", "barely", "n't", "cannot"}
INTENSIFIERS = {"very": 1.3, "really": 1.3, "extremely": 1.5, "totally": 1.4,
                 "absolutely": 1.4, "so": 1.2}


def _tokenise(text):
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return text.split()


def lexicon_score(text: str) -> dict:
    toks = _tokenise(text)
    score = 0.0
    contributions = []
    i = 0
    while i < len(toks):
        w = toks[i]
        if w in LEXICON:
            val = LEXICON[w]
            # look back for negation (within 3 words) or intensifier
            flip = 1.0; mult = 1.0
            for lookback in range(1, 4):
                if i - lookback < 0:
                    break
                prev = toks[i - lookback]
                if prev in NEGATIONS:
                    flip = -1.0
                if prev in INTENSIFIERS:
                    mult = INTENSIFIERS[prev]
            contribution = val * flip * mult
            score += contribution
            contributions.append((w, contribution))
        i += 1
    label = "pos" if score > 0.1 else ("neg" if score < -0.1 else "neu")
    return {"score": float(score), "label": label,
            "contributions": contributions}


if __name__ == "__main__":
    examples = [
        "This movie was absolutely amazing, I loved every minute!",
        "Terrible product, would not recommend it to anyone.",
        "The food was good but the service was really bad.",
        "Not bad at all, actually pretty enjoyable.",
        "I hated the ending; the plot was boring and predictable.",
        "The interface is very clean and easy to use.",
    ]

    print("=== Lexicon-based sentiment ===")
    for text in examples:
        r = lexicon_score(text)
        print(f"  [{r['label']:>3} {r['score']:+.2f}] \"{text}\"")

    # supervised baseline: reuse the earlier text_classification pipeline
    print("\n=== Supervised classifier (logistic on TF-IDF) — 40 training + 20 test ===")
    rng = np.random.default_rng(0)
    positive_words = "great love excellent amazing wonderful fantastic perfect happy enjoyed best".split()
    negative_words = "terrible hate worst awful boring disappoint bad annoying angry waste".split()
    docs = []; y = []
    for _ in range(30):
        docs.append([positive_words[rng.integers(len(positive_words))] for _ in range(6)])
        y.append("pos")
        docs.append([negative_words[rng.integers(len(negative_words))] for _ in range(6)])
        y.append("neg")
    y = np.array(y)
    # split 40 train / 20 test
    idx = rng.permutation(len(docs))
    docs = [docs[i] for i in idx]; y = y[idx]
    n_tr = 40

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        vec = TfidfVectorizer(tokenizer=lambda x: x.split(), token_pattern=None, lowercase=False)
        X = vec.fit_transform([" ".join(d) for d in docs])
        clf = LogisticRegression(max_iter=200).fit(X[:n_tr], y[:n_tr])
        acc = clf.score(X[n_tr:], y[n_tr:])
        print(f"  test accuracy = {acc:.3f}   (n_test = {len(docs) - n_tr})")
    except ImportError:
        print("  (sklearn not installed)")

    print("\n--- library cross-check (VADER: nltk.sentiment.vader; TextBlob; transformers pipeline) ---")
