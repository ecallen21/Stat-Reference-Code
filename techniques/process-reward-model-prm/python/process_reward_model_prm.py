"""Process Reward Model - PRM (Reference Sec 47.195).

Lightman et al 2023 'Let's Verify Step by Step' (OpenAI PRM800K);
Wang et al 2024 'Math-Shepherd'. Reward models for reasoning come
in two flavours:

    OUTCOME-Reward-Model (ORM): score only the final answer.
    PROCESS-Reward-Model (PRM): score EACH INTERMEDIATE STEP.

PRMs identify where the reasoning goes wrong even if the final
answer is right by luck. Combined with:
    - PRM-weighted best-of-N sampling: pick the sample with
      highest MIN or MEAN step reward.
    - PRM-guided beam / MCTS search over reasoning.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def score_reasoning_orm(reasoning, correct_answer):
    """Outcome reward: 1 if final answer matches, else 0."""
    final = reasoning[-1].strip()
    return 1.0 if final == correct_answer else 0.0


def score_reasoning_prm(reasoning, correct_intermediate, correct_answer):
    """Process reward: fraction of intermediate steps that match ground truth."""
    if not reasoning: return 0.0
    matches = 0
    for step in reasoning[:-1]:
        if any(gt in step for gt in correct_intermediate):
            matches += 1
    outcome = 1.0 if reasoning[-1].strip() == correct_answer else 0.0
    return 0.7 * matches / max(len(reasoning) - 1, 1) + 0.3 * outcome


if __name__ == "__main__":
    print("=== Process Reward Model - PRM (Lightman et al 2023) ===\n")

    # Toy math problem: 8 + 5 - 3 = ? (correct: 10)
    # Simulate 6 sampled chains-of-thought with correct/incorrect intermediate steps
    correct_answer = "10"
    correct_intermediate = ["13", "8+5=13"]

    samples = [
        (["step 1: 8 + 5 = 13", "step 2: 13 - 3 = 10", "10"], "chain A: right steps, right answer"),
        (["step 1: 8 + 5 = 14",  "step 2: 14 - 4 = 10", "10"], "chain B: WRONG steps, right answer (lucky)"),
        (["step 1: 8 + 5 = 13",  "step 2: 13 - 5 = 8", "8"], "chain C: right first step, wrong answer"),
        (["step 1: 8 + 5 = 13",  "step 2: 13 - 3 = 10", "10"], "chain D: right steps, right answer"),
        (["step 1: 8 + 5 = 13",  "step 2: 13 - 3 = 11", "11"], "chain E: right first step, wrong answer"),
        (["step 1: 8 - 5 = 3", "step 2: 3 + 3 = 6", "6"], "chain F: wrong steps, wrong answer"),
    ]

    print(f"  {'ID':<7}  {'ORM':>4}  {'PRM':>5}  {'description'}")
    for i, (chain, desc) in enumerate(samples):
        orm = score_reasoning_orm(chain, correct_answer)
        prm = score_reasoning_prm(chain, correct_intermediate, correct_answer)
        letter = chr(ord('A') + i)
        print(f"  chain {letter}  {orm:>4.1f}  {prm:>5.2f}  {desc}")

    # Best-of-N selection
    orm_scores = [score_reasoning_orm(c, correct_answer) for c, _ in samples]
    prm_scores = [score_reasoning_prm(c, correct_intermediate, correct_answer) for c, _ in samples]
    orm_best = int(np.argmax(orm_scores)); prm_best = int(np.argmax(prm_scores))
    print(f"\n  ORM best-of-N picks chain {chr(ord('A') + orm_best)}   "
          f"(all 'right answer' chains tie, may pick lucky one)")
    print(f"  PRM best-of-N picks chain {chr(ord('A') + prm_best)}   "
          f"(reasoning quality tiebreaker)")

    print("\n  Real PRMs (Lightman 2023, PRM800K) achieve 78% on MATH vs ORM's 72%.")

    print("\n--- library cross-check (openai/prm800k, math-shepherd, trl reward-model APIs) ---")
