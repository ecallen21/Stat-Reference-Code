"""Quantitative content analysis (Reference Sec 42.17).

Krippendorff (2019).  Systematic manual coding of text into
categories with:
  1. UNITIZING     : define the coding unit (word, sentence, doc).
  2. SAMPLING      : draw a representative sample of text.
  3. CODING        : two or more human coders apply the codebook.
  4. RELIABILITY   : Krippendorff alpha (or Cohen kappa) on a
                     double-coded subset -- target alpha >= 0.80.
  5. ADJUDICATION  : reconcile disagreements before analysis.

Compact demo: two coders, small codebook, compute reliability.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def cohens_kappa(a, b):
    a = np.asarray(a); b = np.asarray(b)
    cats = sorted(set(a) | set(b))
    m = len(cats); N = len(a)
    idx = {c: i for i, c in enumerate(cats)}
    C = np.zeros((m, m))
    for x, y in zip(a, b):
        C[idx[x], idx[y]] += 1
    P_a = np.trace(C) / N
    P_e = (C.sum(axis=1) @ C.sum(axis=0)) / N ** 2
    return float((P_a - P_e) / (1 - P_e))


def krippendorff_alpha_nominal(codes):
    """Codes shape (n_items, n_raters); missing = None."""
    codes = [list(row) for row in codes]
    cats = sorted({c for r in codes for c in r if c is not None})
    K = len(cats); idx = {c: i for i, c in enumerate(cats)}
    O = np.zeros((K, K))
    for row in codes:
        row_c = [c for c in row if c is not None]
        m = len(row_c)
        if m < 2: continue
        counts = np.zeros(K)
        for c in row_c: counts[idx[c]] += 1
        for v in range(K):
            for w in range(K):
                delta = 1 if v == w else 0
                O[v, w] += counts[v] * (counts[w] - delta) / (m - 1)
    n_v = O.sum(axis=1); n = n_v.sum()
    disag = 1 - np.eye(K)
    D_o = (O * disag).sum() / max(n, 1)
    D_e = ((n_v[:, None] * n_v[None, :]) * disag).sum() / max(n * (n - 1), 1)
    return float(1 - D_o / D_e) if D_e > 0 else float("nan")


if __name__ == "__main__":
    print("=== Content analysis: coding + reliability (Cohen k + Krippendorff alpha) ===\n")
    # 12 items double-coded by two raters into {pos, neg, neu}
    coder1 = ["pos", "pos", "neg", "neu", "neu", "pos", "neg", "neg", "pos", "neu", "neg", "pos"]
    coder2 = ["pos", "pos", "neg", "neu", "pos", "pos", "neg", "neg", "pos", "neg", "neg", "pos"]

    k = cohens_kappa(coder1, coder2)
    print(f"  Cohen kappa (2 raters, 12 items) = {k:.3f}")

    codes = [list(pair) for pair in zip(coder1, coder2)]
    a = krippendorff_alpha_nominal(codes)
    print(f"  Krippendorff alpha (nominal)      = {a:.3f}")

    print("\n  Krippendorff (2019): alpha >= 0.80 = good reliability;")
    print("                       0.667 - 0.80 tentative; < 0.667 unreliable.\n")

    # Add a 3rd coder + some missing values
    coder3 = ["pos", "neu", "neg", "neu", "pos", "pos", None, "neg", "pos", "neu", "neg", None]
    codes_3 = [[a, b, c] for a, b, c in zip(coder1, coder2, coder3)]
    a3 = krippendorff_alpha_nominal(codes_3)
    print(f"  3-coder alpha (with missing): {a3:.3f}\n")

    print("--- library cross-check (R irr::kappa2/kripp.alpha, irrCAC; Python krippendorff, sklearn.cohen_kappa) ---")
