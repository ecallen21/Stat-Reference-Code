"""MMLU-Style Benchmark Evaluation (Reference Sec 47.217).

Hendrycks et al 2021 'Measuring Massive Multitask Language
Understanding', ICLR. Multiple-choice QA benchmark spanning 57
academic + professional subjects. Standard evaluation protocol:

    - 5-SHOT prompt (5 (Q, A) demonstrations from the SAME subject).
    - Score by NEXT-TOKEN log-prob over {A, B, C, D}, take argmax.
    - Report macro accuracy (mean over subjects).

Widely used but has known issues: contamination (MMLU appears in
web-scraped corpora), letter bias (models over-select A/B), and
susceptibility to prompt sensitivity.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


LETTERS = ["A", "B", "C", "D"]


def score_mcq_by_logprob(question, choices, correct_idx, logprob_fn):
    """MMLU protocol: score P(letter | prompt) for each of A/B/C/D."""
    prompt = f"Question: {question}\n" + "\n".join(
        f"{L}. {c}" for L, c in zip(LETTERS, choices)) + "\nAnswer:"
    logprobs = [logprob_fn(prompt, L) for L in LETTERS]
    return int(np.argmax(logprobs)), logprobs


def build_5shot_prompt(demos, question, choices):
    parts = []
    for d in demos:
        parts.append(f"Question: {d['q']}\n" + "\n".join(
            f"{L}. {c}" for L, c in zip(LETTERS, d['choices'])) +
                        f"\nAnswer: {LETTERS[d['idx']]}\n")
    parts.append(f"Question: {question}\n" + "\n".join(
        f"{L}. {c}" for L, c in zip(LETTERS, choices)) + "\nAnswer:")
    return "\n".join(parts)


if __name__ == "__main__":
    print("=== MMLU-style benchmark eval (Hendrycks et al 2021) ===\n")
    rng = np.random.default_rng(0)

    # Toy 3-question 'algebra' subject
    subject = [
        {"q": "Solve: 3x = 12", "choices": ["3", "4", "5", "6"], "idx": 1},
        {"q": "Solve: x + 5 = 8", "choices": ["1", "2", "3", "4"], "idx": 2},
        {"q": "Solve: 2x + 4 = 10", "choices": ["1", "2", "3", "4"], "idx": 2},
        {"q": "Solve: x - 7 = 3", "choices": ["6", "10", "12", "-4"], "idx": 1},
        {"q": "Solve: 5x = 25", "choices": ["3", "5", "6", "10"], "idx": 1},
        {"q": "Solve: 8 - x = 3", "choices": ["3", "4", "5", "6"], "idx": 2},
    ]

    def logprob_ideal(prompt, letter):
        """Ideal LM: gives high log-prob to correct letter (found by parsing prompt)."""
        # Parse the final question, evaluate, return matching letter score
        q_line = [ln for ln in prompt.split("\n") if ln.startswith("Question: Solve:")][-1]
        expr = q_line.replace("Question: Solve:", "").strip()
        # Extract choices from the final question
        choice_lines = prompt.strip().split("\n")[-5:-1]
        # Evaluate expr for each letter as x
        try:
            lhs, rhs = expr.split("=")
            for L, cl in zip(LETTERS, choice_lines):
                val = cl.split(". ", 1)[1].strip()
                x_try = float(val)
                lhs_eval = lhs.replace("x", str(x_try))
                if abs(eval(lhs_eval) - float(rhs)) < 1e-6:
                    if L == letter: return 5.0
        except Exception:
            pass
        return 0.0

    def logprob_letter_A_bias(prompt, letter):
        return {"A": 2.0, "B": 0.0, "C": 0.0, "D": 0.0}[letter]

    hits_ideal = 0; hits_A = 0
    for s in subject:
        pred, _ = score_mcq_by_logprob(s["q"], s["choices"], s["idx"], logprob_ideal)
        hits_ideal += (pred == s["idx"])
        pred_A, _ = score_mcq_by_logprob(s["q"], s["choices"], s["idx"], logprob_letter_A_bias)
        hits_A += (pred_A == s["idx"])
    print(f"  Ideal LM (parses and solves) accuracy:     {hits_ideal / len(subject):.3f}")
    print(f"  Letter-A biased LM accuracy:               {hits_A / len(subject):.3f}")
    print(f"  Random-guessing accuracy (chance):         0.250")
    print(f"  (subject has {sum(1 for s in subject if s['idx'] == 0)}/{len(subject)} A-answers, so A-bias hits chance-ish)")

    print("\n  Real MMLU uses 5-shot with a strong LM's per-letter log-prob; scores")
    print("  are reported per-subject and macro-averaged.")

    print("\n--- library cross-check (lm-eval-harness / EleutherAI, opencompass, HELM) ---")
