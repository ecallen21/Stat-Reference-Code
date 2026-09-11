"""Benchmark Contamination Detection (Reference Sec 47.219).

Sainz et al 2023 'NLP Evaluation in trouble'; Golchin & Surdeanu
2023 'Time Travel in LLMs'; Xu et al 2024 'Benchmarking
Contamination in LLMs'. Common tests:

    1. GUIDED-COMPLETION test: give the LM the FIRST HALF of a
       benchmark example and see if it exactly completes it.
    2. Membership-inference: compare loss(perplexity) on training-
       time vs held-out examples.
    3. Rephrase test: paraphrase the example — a memorising model
       drops accuracy, a reasoning model doesn't.
    4. N-gram overlap: search the training data for the exact
       benchmark strings.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def guided_completion_test(prefix, gold_suffix, lm_complete):
    """Prompt the LM with prefix; check if it exactly emits gold_suffix."""
    completion = lm_complete(prefix)
    return completion.strip() == gold_suffix.strip()


def membership_inference(train_examples, held_out_examples, perplexity_fn):
    """Compare mean perplexity on training-set vs held-out set."""
    train_ppl = np.mean([perplexity_fn(t) for t in train_examples])
    held_ppl = np.mean([perplexity_fn(t) for t in held_out_examples])
    return {"train_ppl": train_ppl, "held_out_ppl": held_ppl,
             "gap": held_ppl - train_ppl}


def rephrase_test(orig_examples, paraphrased_examples, gold, model_answer):
    """Accuracy drop from orig -> paraphrased indicates memorisation."""
    orig_acc = np.mean([model_answer(e) == g for e, g in zip(orig_examples, gold)])
    para_acc = np.mean([model_answer(e) == g for e, g in zip(paraphrased_examples, gold)])
    return {"orig_acc": orig_acc, "paraphrased_acc": para_acc,
             "drop": orig_acc - para_acc}


if __name__ == "__main__":
    print("=== Benchmark Contamination Detection (Sainz 2023; Golchin 2024) ===\n")

    # Toy setup: a 'model' that has MEMORISED some benchmark examples
    memorised = {
        "The capital of France is": "Paris (recorded 2020).",
        "The largest planet is": "Jupiter (5 moons Galilean).",
    }
    novel = {"The capital of Germany is": "Berlin.",
              "The tallest mountain is": "Everest."}

    def lm_complete_memorised(prefix):
        return memorised.get(prefix.rstrip(), "I don't know")

    # 1. Guided-completion test
    print("  1. Guided-completion test:")
    for prefix, gold in list(memorised.items()) + list(novel.items()):
        exact = guided_completion_test(prefix, gold, lm_complete_memorised)
        source = "memorised" if prefix in memorised else "novel"
        print(f"     {'✓ MATCH' if exact else '✗ miss':<10}  ({source:<10})  {prefix!r}")

    # 2. Perplexity gap: memorised should be lower
    def toy_ppl(t):
        return 1.5 if t in memorised or t in novel else 5.0
    # Trick: only 'memorised' text is in training
    train = list(memorised.keys())
    held_out = list(novel.keys())
    mi = membership_inference(train, held_out, lambda t: 1.5 if t in memorised else 5.0)
    print(f"\n  2. Membership inference (perplexity gap):")
    print(f"     train_ppl = {mi['train_ppl']:.2f}   held_out_ppl = {mi['held_out_ppl']:.2f}   "
          f"gap = {mi['gap']:.2f}   (large gap -> likely contaminated)")

    # 3. Rephrase test: model recognises rephrased?
    orig = ["What is 2 + 2?"] * 3
    para = ["Add 2 and 2. What do you get?", "Sum of 2 and 2?", "2 plus 2 equals what?"]
    gold = ["4", "4", "4"]
    def model_ans(q): return "4" if "2 + 2" in q else "unknown"
    r = rephrase_test(orig, para, gold, model_ans)
    print(f"\n  3. Rephrase test (over-fits exact wording):")
    print(f"     orig acc = {r['orig_acc']:.2f}   paraphrased acc = {r['paraphrased_acc']:.2f}   "
          f"drop = {r['drop']:.2f}   (large drop -> likely memorised)")

    print("\n--- library cross-check (lm-eval-harness contamination flags; llm-contamination) ---")
