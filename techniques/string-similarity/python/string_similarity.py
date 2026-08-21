"""String similarity metrics (Reference §25.9).

  * LEVENSHTEIN (edit distance): min inserts + deletes + substitutions.
  * DAMERAU-LEVENSHTEIN: adds adjacent transpositions (single op).
  * JARO: matching-window + transposition-count similarity in [0, 1].
  * JARO-WINKLER: Jaro + prefix bonus (up to 4 matching prefix chars).
  * LONGEST COMMON SUBSEQUENCE: matches LCS ratio.
  * JACCARD on char n-grams: |set(A) ∩ set(B)| / |set(A) ∪ set(B)|.
  * COSINE on char n-grams: usual dot / norms.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math

import numpy as np    # numerical arrays + linear algebra


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def damerau_levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,
                           d[i][j - 1] + 1,
                           d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[n][m]


def jaro(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    match_dist = max(len(a), len(b)) // 2 - 1
    a_matches = [False] * len(a)
    b_matches = [False] * len(b)
    matches = 0
    for i in range(len(a)):
        lo = max(0, i - match_dist); hi = min(len(b), i + match_dist + 1)
        for j in range(lo, hi):
            if not b_matches[j] and a[i] == b[j]:
                a_matches[i] = True; b_matches[j] = True
                matches += 1; break
    if matches == 0:
        return 0.0
    k = 0; transpositions = 0
    for i in range(len(a)):
        if a_matches[i]:
            while not b_matches[k]:
                k += 1
            if a[i] != b[k]:
                transpositions += 1
            k += 1
    transpositions //= 2
    m = matches
    return (m / len(a) + m / len(b) + (m - transpositions) / m) / 3


def jaro_winkler(a: str, b: str, p: float = 0.1) -> float:
    j = jaro(a, b)
    prefix = 0
    for x, y in zip(a[:4], b[:4]):
        if x == y:
            prefix += 1
        else:
            break
    return j + prefix * p * (1 - j)


def _char_ngrams(s: str, n: int = 3) -> set:
    return {s[i: i + n] for i in range(len(s) - n + 1)}


def jaccard_ngram(a: str, b: str, n: int = 3) -> float:
    A = _char_ngrams(a, n); B = _char_ngrams(b, n)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def cosine_ngram(a: str, b: str, n: int = 3) -> float:
    from collections import Counter
    A = Counter(a[i: i + n] for i in range(len(a) - n + 1))
    B = Counter(b[i: i + n] for i in range(len(b) - n + 1))
    keys = set(A) | set(B)
    va = np.array([A[k] for k in keys]); vb = np.array([B[k] for k in keys])
    denom = np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12
    return float(va @ vb / denom)


if __name__ == "__main__":
    pairs = [
        ("kitten", "sitting"),
        ("form", "from"),
        ("Martha", "Marhta"),
        ("Dixon", "Dickson"),
        ("Elisabeth Callen", "Elizabeth Callen"),
        ("machine learning", "machine-learning"),
        ("apple", "orange"),
    ]
    print(f"{'a':>18} {'b':>18} {'lev':>4} {'dl':>3} {'jaro':>6} "
          f"{'jw':>6} {'jacc':>6} {'cos':>6}")
    for a, b in pairs:
        print(f"{a!r:>18} {b!r:>18} "
              f"{levenshtein(a, b):>4} {damerau_levenshtein(a, b):>3} "
              f"{jaro(a, b):>6.3f} {jaro_winkler(a, b):>6.3f} "
              f"{jaccard_ngram(a, b):>6.3f} {cosine_ngram(a, b):>6.3f}")

    print("\n--- library cross-check (python-Levenshtein / rapidfuzz / jellyfish) ---")
    try:
        from rapidfuzz.distance import Levenshtein as RFLev, JaroWinkler as RFJW
        for a, b in pairs[:2]:
            print(f"  rapidfuzz  {a!r} vs {b!r}: lev={RFLev.distance(a, b)}, "
                  f"jw={RFJW.similarity(a, b):.3f}")
    except ImportError:
        print("  (rapidfuzz not installed)")
