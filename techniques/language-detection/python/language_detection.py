"""Character n-gram language identification (Cavnar-Trenkle 1994; Reference §25.10).

Build a per-language profile of the top-K most frequent character n-grams
(n = 1..5).  Classify a new text by matching its profile against each
language's profile via an out-of-place distance:

    D(text_profile, lang_profile) = sum over top-K n-grams of |rank_text - rank_lang|
    (if a text n-gram is absent from lang_profile, contribute K + 1)

Pick the language with the smallest D.  Simple, fast, and strong for short
strings (>~ 40 chars) across dozens of languages.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

from collections import Counter    # stdlib: bag counts


def _char_ngrams(text, n_range=(1, 2, 3, 4, 5)):
    text = "_" + text.lower() + "_"                       # boundary markers
    grams = []
    for n in n_range:
        for i in range(len(text) - n + 1):
            grams.append(text[i: i + n])
    return grams


def build_profile(texts, top_k: int = 200) -> dict:
    counts = Counter()
    for t in texts:
        counts.update(_char_ngrams(t))
    ordered = [g for g, _ in counts.most_common(top_k)]
    return {g: r for r, g in enumerate(ordered)}


def out_of_place(text_profile: dict, lang_profile: dict, top_k: int) -> int:
    d = 0
    for g, r in text_profile.items():
        d += abs(r - lang_profile.get(g, top_k))          # penalty K for OOV
    return d


def classify(text: str, profiles: dict, top_k: int = 200) -> dict:
    tp = build_profile([text], top_k)
    scores = {lang: out_of_place(tp, p, top_k) for lang, p in profiles.items()}
    best = min(scores, key=scores.get)
    return {"pred": best, "distance": scores[best], "all": scores,
            "method": "Cavnar-Trenkle out-of-place n-gram distance"}


if __name__ == "__main__":
    # Toy corpora: a few short passages per language.
    train = {
        "en": [
            "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore.",
            "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole.",
            "It was the best of times, it was the worst of times. The truth is out there.",
        ],
        "es": [
            "El rapido zorro marron salta sobre el perro perezoso. Vender conchas en la playa.",
            "En un agujero en la tierra vivia un hobbit. No un agujero sucio y humedo.",
            "Era el mejor de los tiempos, era el peor de los tiempos. La verdad esta ahi.",
        ],
        "de": [
            "Der schnelle braune Fuchs springt uber den faulen Hund. Sie verkauft Muscheln.",
            "In einem Loch im Boden lebte ein Hobbit. Kein schmutziges, feuchtes Loch.",
            "Es war die beste aller Zeiten, es war die schlechteste aller Zeiten.",
        ],
    }
    profiles = {lang: build_profile(txts, top_k=200) for lang, txts in train.items()}

    tests = [
        ("Hello there, how are you doing today?", "en"),
        ("Buenos dias, como estas hoy?", "es"),
        ("Guten Morgen, wie geht es dir heute?", "de"),
        ("Would you like some more tea?", "en"),
        ("Nos vemos manana en la playa.", "es"),
        ("Ich lese ein interessantes Buch.", "de"),
    ]
    print("=== Language ID via Cavnar-Trenkle n-gram profiles ===")
    correct = 0
    for text, truth in tests:
        r = classify(text, profiles, top_k=200)
        ok = r["pred"] == truth
        correct += ok
        print(f"  [{'ok' if ok else '!!'}]  true={truth} pred={r['pred']}  "
              f"D_top3={sorted(r['all'].items(), key=lambda kv: kv[1])[:3]}   "
              f"\"{text[:40]}...\"")
    print(f"\n  accuracy = {correct}/{len(tests)} = {correct / len(tests):.2f}")

    print("\n--- library cross-check (langid, fasttext lid.176.bin, langdetect) ---")
    try:
        import langid
        for text, truth in tests[:3]:
            lang, _ = langid.classify(text)
            print(f"  langid: {lang} for \"{text[:30]}...\"  (true {truth})")
    except ImportError:
        print("  (langid not installed)")
