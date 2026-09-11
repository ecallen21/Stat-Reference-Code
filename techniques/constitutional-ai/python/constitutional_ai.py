"""Constitutional AI - CAI (Reference Sec 47.183).

Bai et al 2022 'Constitutional AI: Harmlessness from AI Feedback',
Anthropic. Two-stage RLHF alternative that reduces the need for
human labels:

    Stage 1 (SL): CRITIQUE-AND-REVISE loop.
        For each response r:
            critique = LM("critique r against principle P")
            revised = LM("revise r using the critique")
        Fine-tune on (prompt, revised) pairs.
    Stage 2 (RL): RLAIF (RL from AI Feedback).
        Use an LM to score pairs of responses for harmfulness;
        train a preference model, then PPO the base model against it.

'Constitution' = set of natural-language principles (be helpful,
avoid harm, don't produce dangerous content, etc). CAI-trained
models are more harmless with less human data than pure RLHF.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


PRINCIPLES = [
    "Please rewrite the response to be more helpful.",
    "Please rewrite the response to be more truthful.",
    "Please rewrite the response to avoid harm.",
]


def critique(response, principle, tokens_helpful, tokens_harmful):
    """Toy critic: score = presence of harmful vs helpful token bags."""
    toks = set(response.lower().split())
    harm_score = len(toks & tokens_harmful) / max(len(toks), 1)
    help_score = len(toks & tokens_helpful) / max(len(toks), 1)
    if "harm" in principle:
        return harm_score
    return -help_score                                          # more helpful = lower critique 'concern'


def revise(response, critique_score, tokens_harmful, tokens_helpful):
    """Toy reviser: if critique flags harm, replace harmful tokens with helpful ones."""
    words = response.split()
    new = []
    for w in words:
        lw = w.lower()
        if critique_score > 0.05 and lw in tokens_harmful:
            new.append("safely")
        else:
            new.append(w)
    if critique_score > 0.05 and not any(w in tokens_helpful for w in [x.lower() for x in words]):
        new.append("and here is some helpful context.")
    return " ".join(new)


def cai_stage1(prompt_responses, principles, tokens_helpful, tokens_harmful):
    """Run one critique-and-revise pass over a batch of (prompt, response) pairs."""
    revised = []
    for pr, r in prompt_responses:
        for principle in principles:
            c = critique(r, principle, tokens_helpful, tokens_harmful)
            r = revise(r, c, tokens_harmful, tokens_helpful)
        revised.append((pr, r))
    return revised


def score_response(r, tokens_helpful, tokens_harmful):
    """Preference-model-style score."""
    toks = set(r.lower().split())
    return len(toks & tokens_helpful) - 2 * len(toks & tokens_harmful)


if __name__ == "__main__":
    print("=== Constitutional AI (Bai et al 2022) ===\n")

    tokens_helpful = {"helpful", "safely", "context", "assist", "explain", "clarify",
                          "carefully", "consider"}
    tokens_harmful = {"attack", "harm", "dangerous", "kill", "exploit", "malicious"}

    initial = [
        ("How do I make a website?",
         "You could kill the competition by launching a malicious attack..."),
        ("What's the weather?",
         "Just check outside; dangerous storms can arrive without warning."),
        ("Recommend a book.",
         "Consider Dune - a thoughtful sci-fi read."),
    ]

    print("  Before CAI critique-and-revise:")
    for pr, r in initial:
        s = score_response(r, tokens_helpful, tokens_harmful)
        print(f"    prompt: {pr}\n    response ({s:+d}): {r}")

    revised = cai_stage1(initial, PRINCIPLES, tokens_helpful, tokens_harmful)
    print("\n  After CAI critique-and-revise:")
    for pr, r in revised:
        s = score_response(r, tokens_helpful, tokens_harmful)
        print(f"    prompt: {pr}\n    response ({s:+d}): {r}")

    avg_before = np.mean([score_response(r, tokens_helpful, tokens_harmful)
                             for _, r in initial])
    avg_after = np.mean([score_response(r, tokens_helpful, tokens_harmful)
                             for _, r in revised])
    print(f"\n  Avg preference-model score: before = {avg_before:+.2f}, after = {avg_after:+.2f}")
    print("  Real CAI uses an LM to critique+revise and a preference model to score.")

    print("\n--- library cross-check (trlx / trl / anthropic reference impl) ---")
