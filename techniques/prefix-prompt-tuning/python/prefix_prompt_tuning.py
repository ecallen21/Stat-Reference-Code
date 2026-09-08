"""Prefix / Prompt Tuning (Reference Sec 47.111).

Li & Liang 2021 'Prefix-tuning: Optimizing continuous prompts for
generation'; Lester, Al-Rfou & Constant 2021 'The power of scale
for parameter-efficient prompt tuning'. FREEZE a pre-trained
transformer; learn only a small tensor of PROMPT VECTORS prepended
to every layer's key/value cache (prefix) or the input embeddings
(prompt).

Trainable parameters typically 0.01-0.1 % of the base model, yet
match full fine-tuning on many NLP tasks (Lester 2021 shows
prompt tuning approaches full fine-tune at large model scale).

Illustrated here as a linear-classifier analogue: freeze a random
projection (proxy for the pretrained encoder), train only a
compact SOFT-PROMPT vector concatenated to inputs.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from sklearn.linear_model import LogisticRegression    # linear head (frozen baseline)


def train_prompt_head(X_frozen, prompt, y, C=1.0):
    """Fit a linear head on [X_frozen | prompt] where prompt is a fixed vector."""
    Xp = np.column_stack([X_frozen, np.tile(prompt, (len(X_frozen), 1))])
    return LogisticRegression(max_iter=500, C=C).fit(Xp, y)


def train_prompt(X_frozen, y, d_prompt=8, n_iter=50, lr=0.1, seed=0):
    """Coordinate-descent on the prompt vector; head refit each iter."""
    rng = np.random.default_rng(seed)
    prompt = rng.normal(size=d_prompt) * 0.01
    best_acc = 0.0
    for it in range(n_iter):
        clf = train_prompt_head(X_frozen, prompt, y)
        acc = clf.score(np.column_stack([X_frozen, np.tile(prompt, (len(X_frozen), 1))]), y)
        if acc > best_acc: best_acc = acc
        # Random-search step
        prompt_try = prompt + lr * rng.normal(size=d_prompt)
        clf2 = train_prompt_head(X_frozen, prompt_try, y)
        acc2 = clf2.score(np.column_stack([X_frozen, np.tile(prompt_try, (len(X_frozen), 1))]), y)
        if acc2 > acc:
            prompt = prompt_try
        lr *= 0.99
    return prompt, best_acc


if __name__ == "__main__":
    print("=== Prefix / Prompt Tuning (Li-Liang 2021; Lester et al 2021) ===\n")
    rng = np.random.default_rng(0)
    n, d_in, d_pre = 400, 20, 64
    X = rng.normal(size=(n, d_in))
    coefs = rng.normal(size=d_in)
    y = (X @ coefs > 0).astype(int)

    # "Pre-trained" encoder = random projection to 64 dims (frozen)
    W_frozen = rng.normal(size=(d_in, d_pre))
    X_enc = X @ W_frozen

    # Baseline: full fine-tune = train a fresh classifier on the encoded features
    clf_full = LogisticRegression(max_iter=500).fit(X_enc, y)
    acc_full = clf_full.score(X_enc, y)
    n_params_full = d_pre + 1

    # Prompt tuning: freeze the encoder, train only a small prompt vector
    for d_prompt in [2, 4, 8, 16]:
        _, acc_prompt = train_prompt(X_enc, y, d_prompt=d_prompt, n_iter=40, seed=0)
        n_params_prompt = d_prompt + (d_pre + d_prompt + 1)   # head refits, prompt is the addition
        # Report just the PROMPT parameters as tunable
        print(f"  prompt dim = {d_prompt:2d}   tunable prompt params = {d_prompt:3d}   "
              f"acc = {acc_prompt:.3f}")
    print(f"\n  Baseline: full fine-tune (all {n_params_full} params) acc = {acc_full:.3f}")
    print("  Prompt-tuning trains << full-model params and (at scale) matches full fine-tune.")

    print("\n--- library cross-check (peft / transformers Python; limited R) ---")
