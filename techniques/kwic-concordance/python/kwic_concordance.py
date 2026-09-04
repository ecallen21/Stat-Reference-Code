"""KWIC / concordance (Reference Sec 42.20).

Keyword-In-Context (KWIC) shows target-word occurrences with a
configurable window of preceding and following context.  Classical
corpus-linguistics tool for qualitative pattern spotting.

Related:
  * Concordance lines aligned on the keyword.
  * Collocations = statistically significant word pairs (see
    collocation-pmi for PMI / G^2 scoring).
"""
from __future__ import annotations    # stdlib

import re


def kwic(text, keyword, window=5):
    """Return keyword-in-context matches with `window` tokens on each side."""
    tokens = re.findall(r"\S+", text)
    key_lc = keyword.lower()
    hits = []
    for i, tok in enumerate(tokens):
        # Strip punctuation for comparison
        core = re.sub(r"[^\w]", "", tok.lower())
        if core == key_lc:
            left = " ".join(tokens[max(0, i - window):i])
            right = " ".join(tokens[i + 1:i + 1 + window])
            hits.append({"idx": i, "left": left, "keyword": tok, "right": right})
    return hits


if __name__ == "__main__":
    print("=== KWIC concordance ===\n")
    text = ("Patient started on aspirin for chest pain.  "
            "Continues aspirin for prophylaxis.  "
            "No side effects on aspirin.  "
            "Discontinued aspirin after bleeding episode.  "
            "Restarted aspirin per cardiology recommendation.")
    for h in kwic(text, "aspirin", window=4):
        print(f"    {h['left']:>40s}  [{h['keyword']}]  {h['right']}")

    print("\n  Useful for qualitatively inspecting usage patterns before or")
    print("  after quantitative frequency / keyness analysis.\n")
    print("--- library cross-check (R quanteda::kwic; Python nltk.Text.concordance, textacy) ---")
