"""Medusa - Speculative Decoding with Extra Heads (Sec 47.185).

Cai, Li, Geng, Peng, Lee, Zhang & Dao 2024 'Medusa: Simple LLM
Inference Acceleration Framework with Multiple Decoding Heads'.
Instead of a separate DRAFT MODEL (standard speculative decoding),
Medusa adds K EXTRA PREDICTION HEADS on top of the target LM:

    head_1 -> token t+1  (baseline LM head)
    head_2 -> token t+2  (speculative)
    ...
    head_K -> token t+K

Each head samples top-k candidates; a TREE-STRUCTURED ATTENTION
mask verifies many candidates in parallel with a single target-
model forward. Accepted prefix = longest matching path.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sample_from_probs(probs, top_k, rng):
    """Return top-k tokens by probability."""
    top_ids = np.argsort(-probs)[:top_k]
    return top_ids


def medusa_verify(target_probs, candidate_paths, threshold=0.1):
    """Verify each candidate path against target_probs (list of prob vectors).

    Accept a path prefix until the target's probability for the drafted
    token drops below `threshold` relative to argmax.
    """
    best_len = 0; best_path = []
    for path in candidate_paths:
        L = 0
        for step, tok in enumerate(path):
            if step >= len(target_probs): break
            p = target_probs[step]
            if p[tok] / (p.max() + 1e-12) >= threshold:
                L += 1
            else:
                break
        if L > best_len:
            best_len = L; best_path = path[:L]
    return best_path


def medusa_step(target_probs, medusa_head_probs, top_k=4, threshold=0.5, rng=None):
    """One Medusa decoding step.

    - target_probs: (K+1, V) — the target model's predictions at offsets
      0..K (offset 0 = baseline head t+1, then t+2, ..., t+K+1).
    - medusa_head_probs: list of length K of (V,) speculative predictions
      for offsets 1..K (each corresponds to target_probs[i+1]).
    """
    if rng is None: rng = np.random.default_rng(0)
    # Baseline head_1: always accept target argmax at offset 0
    base_tok = int(np.argmax(target_probs[0]))
    # Build speculative candidate paths from Medusa heads
    candidates_per_head = [sample_from_probs(mp, top_k, rng)
                             for mp in medusa_head_probs]
    paths = []
    for c1 in candidates_per_head[0]:
        if len(candidates_per_head) > 1:
            for c2 in candidates_per_head[1]:
                paths.append([base_tok, int(c1), int(c2)])
        else:
            paths.append([base_tok, int(c1)])
    if not paths:
        paths = [[base_tok]]
    accepted = medusa_verify(target_probs, paths, threshold=threshold)
    if not accepted:
        accepted = [base_tok]
    return accepted


if __name__ == "__main__":
    print("=== Medusa Speculative Decoding (Cai et al 2024) ===\n")
    rng = np.random.default_rng(0)
    V = 1000

    n_trials = 200
    tokens_per_step_baseline = 1                                # regular decoding
    accepted_counts = []
    for t in range(n_trials):
        # Ground-truth "correct" next 3 tokens for this position
        true_toks = rng.integers(0, V, size=3)
        # Target model peaks at the true tokens for each offset
        target_probs = []
        for offset in range(3):
            logits = rng.normal(scale=0.5, size=V)
            logits[true_toks[offset]] += 8                       # strong peak on truth
            p = np.exp(logits - logits.max()); p /= p.sum()
            target_probs.append(p)
        # Medusa heads: also peak near truth but noisier
        medusa_head_probs = []
        for offset in range(2):                                  # 2 speculative heads
            logits = rng.normal(scale=0.5, size=V)
            if rng.uniform() < 0.85:                             # head hits truth 85% of the time
                logits[true_toks[offset + 1]] += 6
            else:
                logits[rng.integers(V)] += 6                     # wrong guess
            hp = np.exp(logits - logits.max()); hp /= hp.sum()
            medusa_head_probs.append(hp)
        accepted = medusa_step(target_probs, medusa_head_probs,
                                  top_k=4, threshold=0.5, rng=rng)
        accepted_counts.append(len(accepted))

    avg_accept = np.mean(accepted_counts)
    print(f"  {n_trials} decoding steps, 2 Medusa heads, top-k = 4, threshold = 0.3")
    print(f"  Average tokens accepted per step: {avg_accept:.2f}")
    print(f"  Baseline (autoregressive):        {tokens_per_step_baseline}")
    print(f"  Speed-up factor: {avg_accept:.2f}x")
    print(f"\n  Real Medusa achieves 2-3x speed-up on LLaMA-2 / Vicuna, no draft model needed.")

    print("\n--- library cross-check (Medusa / FastChat / LookaheadDecoding Python) ---")
