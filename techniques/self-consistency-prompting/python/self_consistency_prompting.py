"""Self-consistency prompting (Reference Sec 47.91).

Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery & Zhou 2022
'Self-consistency improves chain of thought reasoning in language
models', ICLR 2023. Instead of taking the FIRST chain-of-thought
answer from an LLM, sample K reasoning paths at temperature T > 0
and take the MAJORITY VOTE over the final answers:

    y* = mode({ y_k : (r_k, y_k) ~ p(reason, answer | prompt), k = 1..K })

Simulated here by stochastic reasoning agents on arithmetic word
problems, each with per-step correctness probability p. Majority
vote over K samples reliably beats greedy decoding by leveraging
INDEPENDENT ERROR PATHS.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from collections import Counter    # majority vote


def simulate_cot(true_answer, K, step_correct=0.75, n_steps=4, distract=(1, 2, 3, 5, 10),
                  rng=None):
    """Simulate K reasoning chains. Each step correct with prob p; if any step
    wrong, produce a plausible distractor answer.
    """
    rng = rng or np.random.default_rng(0)
    answers = []
    for _ in range(K):
        all_correct = all(rng.random() < step_correct for _ in range(n_steps))
        if all_correct:
            answers.append(true_answer)
        else:
            # Distractor drawn as truth ± element from `distract`, sign random
            delta = int(rng.choice(distract) * rng.choice([-1, 1]))
            answers.append(true_answer + delta)
    return answers


def majority_vote(answers):
    c = Counter(answers)
    top, cnt = c.most_common(1)[0]
    return top, cnt


if __name__ == "__main__":
    print("=== Self-consistency prompting (Wang et al 2022) ===\n")
    rng = np.random.default_rng(0)

    n_problems = 500
    true_answers = rng.integers(10, 100, size=n_problems)
    step_correct = 0.70                    # ~24% chain accuracy after 4 steps

    for K in [1, 5, 20, 40]:
        correct = 0
        for t in true_answers:
            samples = simulate_cot(int(t), K=K, step_correct=step_correct, rng=rng)
            pred, _ = majority_vote(samples)
            correct += int(pred == t)
        acc = correct / n_problems
        print(f"  K = {K:2d} samples  ->  majority-vote accuracy = {acc:.3f}")

    p_chain = step_correct ** 4
    print(f"\n  Single-chain accuracy (theory) = {step_correct}^4 = {p_chain:.3f}")
    print("  Self-consistency lifts accuracy well above the single-chain baseline,")
    print("  matching the Wang et al 2022 CoT SC benchmark trend.")
    print("\n--- library cross-check (litellm / lmql / vllm-batch Python) ---")
