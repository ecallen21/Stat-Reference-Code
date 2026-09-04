"""Keyness analysis (Reference Sec 42.15).

Compare word frequencies between a TARGET corpus and a REFERENCE
corpus.  Test whether each word appears more (or less) often in
the target than expected.  Dunning's log-likelihood ratio G^2 is
the workhorse; chi-square is an alternative.

  G^2 = 2 * sum_{c in {target, ref}} O_c * log(O_c / E_c)
    where O_c = observed count in corpus c
          E_c = expected count under equal proportion

Effect size: log-ratio log(freq_target / freq_ref).
"""
from __future__ import annotations    # stdlib

import re

import numpy as np    # numerical arrays
from scipy import stats


def _counts(docs):
    counts = {}
    for d in docs:
        for w in re.findall(r"\w+", d.lower()):
            counts[w] = counts.get(w, 0) + 1
    return counts


def keyness(target_docs, ref_docs, min_count=3, alpha=0.05):
    cT = _counts(target_docs); cR = _counts(ref_docs)
    N_T = sum(cT.values()); N_R = sum(cR.values())
    words = sorted(set(cT) | set(cR))
    rows = []
    for w in words:
        o_t = cT.get(w, 0); o_r = cR.get(w, 0)
        if o_t + o_r < min_count:
            continue
        e_t = (o_t + o_r) * N_T / (N_T + N_R)
        e_r = (o_t + o_r) * N_R / (N_T + N_R)
        g2 = 0
        for o, e in [(o_t, e_t), (o_r, e_r)]:
            if o > 0 and e > 0:
                g2 += o * np.log(o / e)
        g2 = 2 * g2
        p = stats.chi2.sf(g2, df=1)
        lr = np.log((o_t / N_T + 1e-9) / (o_r / N_R + 1e-9))
        rows.append({"word": w, "target": o_t, "ref": o_r,
                     "log_ratio": float(lr), "G2": float(g2), "p": float(p)})
    rows = sorted(rows, key=lambda r: -r["G2"])
    return rows


if __name__ == "__main__":
    print("=== Keyness analysis: Dunning's G^2 log-likelihood ratio ===\n")
    target = [
        "Patients presented with pneumonia and fever.",
        "Chest X-ray confirmed pneumonia.",
        "Started antibiotics for pneumonia.",
        "Cough persisted; pneumonia recurred.",
    ]
    reference = [
        "The soccer team won the match easily.",
        "Football training resumed at the stadium.",
        "The stadium hosted a rugby match tonight.",
        "The soccer team celebrated the victory.",
        "Sports fans filled the stadium for the game.",
    ]
    rows = keyness(target, reference, min_count=2)
    print(f"  Top 8 key words in TARGET vs REFERENCE:")
    print(f"    {'word':<14s} {'tgt':>4s} {'ref':>4s} {'logRatio':>10s} {'G^2':>8s} {'p':>10s}")
    for r in rows[:8]:
        print(f"    {r['word']:<14s} {r['target']:>4d} {r['ref']:>4d}"
              f" {r['log_ratio']:>+10.3f} {r['G2']:>8.2f} {r['p']:>10.2e}")

    print("\n--- library cross-check (R quanteda::textstat_keyness; Python textacy/scipy custom) ---")
