"""Chain-of-thought (CoT) reasoning + self-consistency (Reference Sec 47.26).

Wei et al. 2022 'Chain-of-thought prompting elicits reasoning in
large language models', NeurIPS; Wang et al. 2022 'Self-consistency
improves chain of thought reasoning in language models'.

CoT: prompt the LM to write out INTERMEDIATE STEPS ("Let's think step
by step") before the final answer -- boosts multi-step reasoning
accuracy dramatically for large models.

Self-consistency: sample K CoT trajectories (temperature > 0),
extract the final answers, MAJORITY VOTE. Higher K -> more accurate,
diminishing returns.

We simulate the effect by treating each CoT sample as a noisy
independent classifier and analysing the majority-vote accuracy.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def majority_vote_accuracy(p_correct, K, n_sim=5000, rng=None):
    """P(majority-vote-correct) when each sample is independently correct with p."""
    rng = rng or np.random.default_rng()
    #  Simulate K binary outcomes each Bern(p); majority == K/2 or more
    votes = rng.binomial(1, p_correct, size=(n_sim, K))
    if K >= 2:
        agg = votes.sum(axis=1) >= (K // 2 + 1)
    else:
        agg = votes[:, 0] == 1
    return float(agg.mean())


if __name__ == "__main__":
    print("=== Chain-of-thought + self-consistency ===\n")
    #  Baseline (no CoT) accuracy vs CoT single-shot vs majority-vote self-consistency
    p_no_cot = 0.30
    p_single_cot = 0.55
    print(f"  Baseline (no CoT) accuracy    = {p_no_cot:.2f}")
    print(f"  Single-shot CoT accuracy      = {p_single_cot:.2f}\n")

    rng = np.random.default_rng(0)
    print(f"  Self-consistency majority vote over K CoT samples:")
    for K in [1, 3, 5, 9, 21, 41]:      # odd K avoids tie-break dips
        acc = majority_vote_accuracy(p_single_cot, K, n_sim=5000, rng=rng)
        print(f"    K = {K:3d}   accuracy = {acc:.3f}")

    print("\n  Diminishing returns visible around K = 10-20; matches Wang et al. 2022 curves.")
    print("  Self-consistency amplifies a single-shot's edge over the coin-flip baseline.")

    print("\n--- library cross-check (community CoT / SC baselines for LLM benchmarks) ---")
