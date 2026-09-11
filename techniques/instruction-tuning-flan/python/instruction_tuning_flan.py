"""Instruction Tuning / FLAN (Reference Sec 47.182).

Wei, Bosma, Zhao, Guu, Yu, Lester, Du, Dai, Le 2022 'Finetuned
Language Models Are Zero-Shot Learners', ICLR. Take a pretrained
LM and fine-tune it on MANY NLP tasks reformulated as INSTRUCTION-
FOLLOWING:

    "Translate this sentence to French: {source}" -> {target}
    "Summarize the following passage: {passage}"  -> {summary}
    "Is the sentiment positive, negative, or neutral? {text}" -> {label}

At inference the model generalises to NEW instruction types
(zero-shot) it never saw at training. Foundation of ChatGPT-style
assistants; multi-task training + natural-language task
descriptions is the key.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def build_instruction_prompt(task, x, template):
    """Format one example as an instruction-following prompt."""
    return template.format(task=task, x=x)


def linear_lm_score(prompt, target, embed_matrix, vocab):
    """Score a target given a prompt with a bag-of-vocab-tokens model."""
    tokens_p = [t for t in prompt.lower().split() if t in vocab]
    tokens_t = [t for t in target.lower().split() if t in vocab]
    if not tokens_p or not tokens_t: return 0.0
    p_emb = np.mean([embed_matrix[vocab[t]] for t in tokens_p], axis=0)
    t_emb = np.mean([embed_matrix[vocab[t]] for t in tokens_t], axis=0)
    return float(p_emb @ t_emb / (np.linalg.norm(p_emb) * np.linalg.norm(t_emb) + 1e-8))


def simulate_instruction_tuning(train_tasks, held_out_task, n_epochs=100, seed=0):
    """Fit a linear embed model on train_tasks; test on held_out_task."""
    rng = np.random.default_rng(seed)
    # Build vocabulary from all tasks
    all_text = " ".join(t["prompt"] + " " + t["target"]
                             for tsk in train_tasks + [held_out_task] for t in tsk["examples"])
    vocab_list = sorted(set(all_text.lower().split()))
    vocab = {w: i for i, w in enumerate(vocab_list)}
    d = 16
    W = rng.normal(scale=0.3, size=(len(vocab), d))
    # Nudge W so target-token embed aligns with prompt-token embed for training examples
    lr = 0.03
    for ep in range(n_epochs):
        for task in train_tasks:
            for ex in task["examples"]:
                p_toks = [t for t in ex["prompt"].lower().split() if t in vocab]
                t_toks = [t for t in ex["target"].lower().split() if t in vocab]
                if not p_toks or not t_toks: continue
                p_emb = np.mean([W[vocab[t]] for t in p_toks], axis=0)
                t_emb = np.mean([W[vocab[t]] for t in t_toks], axis=0)
                # Push p_emb toward t_emb
                for t in t_toks:
                    W[vocab[t]] += lr * (p_emb - t_emb) / len(t_toks)
    return W, vocab


if __name__ == "__main__":
    print("=== Instruction Tuning / FLAN (Wei et al 2022) ===\n")

    train_tasks = [
        {"name": "sentiment", "examples": [
            {"prompt": "classify the sentiment. text: i loved the film. sentiment:",
             "target": "positive"},
            {"prompt": "classify the sentiment. text: this was awful and boring. sentiment:",
             "target": "negative"},
            {"prompt": "classify the sentiment. text: the movie was excellent and moving. sentiment:",
             "target": "positive"},
            {"prompt": "classify the sentiment. text: hated every moment. sentiment:",
             "target": "negative"},
        ]},
        {"name": "topic", "examples": [
            {"prompt": "classify the topic. text: the goalkeeper scored. topic:",
             "target": "sports"},
            {"prompt": "classify the topic. text: the president signed the bill. topic:",
             "target": "politics"},
            {"prompt": "classify the topic. text: nvidia stock rallied. topic:",
             "target": "finance"},
            {"prompt": "classify the topic. text: election polls close today. topic:",
             "target": "politics"},
        ]},
    ]

    # Held-out task: language identification (never seen at training)
    held_out = {"name": "lang", "examples": [
        {"prompt": "classify the language. text: bonjour le monde. language:", "target": "french"},
        {"prompt": "classify the language. text: hallo welt. language:", "target": "german"},
    ]}

    W, vocab = simulate_instruction_tuning(train_tasks, held_out, seed=0)

    # In-distribution eval: sentiment classification
    print("  In-distribution (sentiment) task, zero-shot after instruction tuning:")
    for prompt in ["classify the sentiment. text: this was truly amazing. sentiment:",
                     "classify the sentiment. text: i hated it. sentiment:"]:
        scores = {"positive": linear_lm_score(prompt, "positive", W, vocab),
                   "negative": linear_lm_score(prompt, "negative", W, vocab)}
        pred = max(scores, key=scores.get)
        print(f"    '{prompt[-40:]}' -> {pred}   (scores {scores})")

    print("\n  Real FLAN-T5 / T0 / instruction-tuned LLMs generalise to unseen tasks")
    print("  because instruction phrasing (e.g. 'classify the...') is itself a")
    print("  transferable skill that emerges from multi-task instruction tuning.")

    print("\n--- library cross-check (transformers.T5 / FlanT5 / open-source instruct models) ---")
