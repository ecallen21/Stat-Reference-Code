"""Toolformer - Self-Taught Tool Use (Reference Sec 47.203).

Schick, Dwivedi-Yu, Dessi, Raileanu, Lomeli, Zettlemoyer, Cancedda
& Scialom 2023 'Toolformer: Language Models Can Teach Themselves
to Use Tools', NeurIPS. Fine-tunes an LM to insert API-call
tokens INTO its own generation:

    "The population of Kansas is [Calculator(2 + 3) = 5] approximately..."

Training pipeline:
    1. Prompt the LM to sample candidate API insertion points.
    2. Execute the API; keep the insertion if it REDUCES perplexity
       on the subsequent context.
    3. Fine-tune the LM on the filtered augmented sequences.
"""
from __future__ import annotations    # stdlib

import re    # regex for API insertions

import numpy as np    # numerical arrays


def toy_lm_perplexity(text, target_next_token_probs):
    """Toy 'LM perplexity' proxy: score matches a simple keyword bag."""
    # Return log-perplexity for the target token given `text` as context
    # Higher = worse
    if "5" in text: return 0.5
    if "2 + 3" in text: return 1.5
    return 2.0


def calculator_api(expr):
    try: return str(eval(expr))
    except Exception: return "ERR"


def wiki_api(entity):
    db = {"kansas": "Kansas has population 2.9M.",
           "spacex": "SpaceX was founded in 2002.",
           "python": "Python is a programming language."}
    return db.get(entity.lower(), "unknown")


def evaluate_api_call(context_before, api_call, api_fn, arg):
    """Compute L(t | c) with and without the inserted API result."""
    result = api_fn(arg)
    with_api = f"{context_before} [{api_call}({arg}) = {result}]"
    without = context_before
    L_with = toy_lm_perplexity(with_api, None)
    L_without = toy_lm_perplexity(without, None)
    return L_with < L_without, result, L_without - L_with


def toolformer_augment(text, candidates):
    """For each candidate insertion, evaluate improvement and keep the good ones."""
    kept = []
    for pos, api_name, arg in candidates:
        api_fn = {"Calculator": calculator_api, "Wiki": wiki_api}[api_name]
        keep, result, improvement = evaluate_api_call(text[:pos], api_name, api_fn, arg)
        if keep:
            kept.append({"pos": pos, "api": api_name, "arg": arg,
                          "result": result, "improvement": improvement})
    return kept


if __name__ == "__main__":
    print("=== Toolformer (Schick et al 2023 NeurIPS) ===\n")

    text = "The result is what everyone expected"
    candidates = [
        (17, "Calculator", "2 + 3"),                             # 'The result is 2 + 3 = 5'
        (17, "Calculator", "10 * 10"),
        (17, "Wiki", "python"),                                  # unrelated
    ]

    kept = toolformer_augment(text, candidates)
    print(f"  Candidate API insertions:")
    for c in candidates:
        _, res, imp = evaluate_api_call(text[:c[0]], c[1], {"Calculator": calculator_api, "Wiki": wiki_api}[c[1]], c[2])
        kept_flag = imp > 0
        print(f"    pos={c[0]}, {c[1]}({c[2]}) -> '{res}'   perplexity delta = {imp:+.2f}  "
              f"{'KEEP' if kept_flag else 'DROP'}")

    print(f"\n  Toolformer keeps {len(kept)} of {len(candidates)} insertions")
    print(f"  (those that REDUCE perplexity on the subsequent context).")

    # Full augmented text
    augmented = text
    for k in sorted(kept, key=lambda k: -k["pos"]):
        augmented = augmented[:k["pos"]] + f" [{k['api']}({k['arg']})={k['result']}]" + augmented[k["pos"]:]
    print(f"\n  Augmented text: '{augmented}'")
    print(f"  This filtered corpus is what Toolformer fine-tunes on.")

    print("\n--- library cross-check (transformers custom + api sandbox; lm-agent-tools) ---")
