"""LLM-as-a-Judge (Reference Sec 47.216).

Zheng et al 2023 'Judging LLM-as-a-Judge with MT-Bench and Chatbot
Arena'. Use a strong LM (GPT-4, Claude 3.5) as an automated judge
for open-ended response quality:

    - SINGLE-answer grading: score on 1-10
    - PAIRWISE comparison: which of A / B is better?
    - REFERENCE-BASED: compare to a gold answer

Cheaper than human eval, correlates ~0.8 with crowd preferences.
Known biases: POSITION bias (favours A in A/B), VERBOSITY bias,
SELF-preference (judge favours its own family).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def toy_judge(response, question, criteria=None):
    """Toy 'judge' — score based on presence of key terms + response length."""
    if criteria is None:
        criteria = ["specific", "clear", "example"]
    r = response.lower()
    hits = sum(k in r for k in criteria)
    length_penalty = max(0, min(1, len(response.split()) / 30))
    return round(3 + 2 * hits + 2 * length_penalty, 1)           # 3-10 scale


def pairwise_judge(response_A, response_B, question):
    """Compare two responses; return 'A', 'B', or 'tie'."""
    s_A = toy_judge(response_A, question)
    s_B = toy_judge(response_B, question)
    if abs(s_A - s_B) < 0.5: return "tie"
    return "A" if s_A > s_B else "B"


def position_bias_check(response_1, response_2, question, n_swap=10):
    """Present the same pair in both orders; measure agreement rate."""
    agrees = 0
    for _ in range(n_swap):
        v1 = pairwise_judge(response_1, response_2, question)
        v2 = pairwise_judge(response_2, response_1, question)
        # v1 == "A" means response_1 wins; v2 == "B" also means response_1 wins
        winner_1 = "response_1" if v1 == "A" else ("response_2" if v1 == "B" else "tie")
        winner_2 = "response_1" if v2 == "B" else ("response_2" if v2 == "A" else "tie")
        agrees += (winner_1 == winner_2)
    return agrees / n_swap


if __name__ == "__main__":
    print("=== LLM-as-a-Judge (Zheng et al 2023) ===\n")

    question = "Explain what an eigenvalue is."
    responses = [
        "An eigenvalue is a scalar lambda for a matrix A such that Av = lambda v for some vector v. "
        "For example, [[2,0],[0,3]] has eigenvalues 2 and 3, with eigenvectors [1,0] and [0,1].",
        "It's a number that describes how a matrix scales certain vectors.",
        "Eigenvalues... hmm, they're kinda important in linear algebra I think?",
    ]

    print(f"  Question: {question}\n")
    print("  Single-answer grading:")
    for i, r in enumerate(responses):
        s = toy_judge(r, question)
        print(f"    Response {i + 1} (len={len(r.split())} words) -> score {s} / 10")

    # Pairwise + position-bias check
    print("\n  Pairwise (with position-bias check):")
    for i in range(3):
        for j in range(i + 1, 3):
            v = pairwise_judge(responses[i], responses[j], question)
            agr = position_bias_check(responses[i], responses[j], question)
            print(f"    {i + 1} vs {j + 1}: {v}   position-swap agreement = {agr:.2f}")

    print("\n  Real LLM judges (GPT-4, Claude) have Cohen's kappa ~ 0.8 vs crowd raters,")
    print("  but need swap-order and self-preference debiasing to be reliable.")

    print("\n--- library cross-check (lm-eval-harness, MT-Bench, alpaca-eval, ragas) ---")
