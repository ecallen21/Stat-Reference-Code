"""Speculative decoding (Reference Sec 47.21).

Leviathan, Kalman & Matias 2023 'Fast inference from transformers
via speculative decoding', ICML. A lossless inference-acceleration
trick: a small DRAFT model proposes k tokens, then a large TARGET
model verifies them in a single forward pass. Accepted tokens are
kept; rejected ones fall back to a target-sampled correction.

Speedup comes from:
    * Batched verification of k tokens costs ~ 1 target forward pass.
    * If a large fraction alpha of drafts are accepted, wall-clock
      drops by ~ k * alpha / (1 + draft_cost).

Acceptance rule (rejection sampling to preserve target distribution):

    Accept if U < p_target(x) / p_draft(x)  where U ~ Uniform(0, 1)

If rejected at position j, resample from the RESIDUAL distribution
r(x) = max(0, p_target(x) - p_draft(x)) / (1 - alpha_j).

We simulate speculative decoding with two categorical distributions
serving as 'target' and 'draft'; report acceptance rate and empirical
speedup.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sample_from(p, rng):
    return int(rng.choice(len(p), p=p))


def speculative_step(p_draft, p_target, k, rng):
    """One block: draft k tokens, verify, accept prefix.

    Returns (accepted_tokens, n_accepted). Guarantees the joint distribution
    of accepted tokens is identical to sampling k times from p_target.
    """
    accepted = []
    for j in range(k):
        x = sample_from(p_draft, rng)
        u = rng.uniform()
        if u <= p_target[x] / (p_draft[x] + 1e-12):
            accepted.append(x)
        else:
            #  Rejection: sample from the residual distribution
            r = np.maximum(p_target - p_draft, 0.0)
            r = r / max(r.sum(), 1e-12)
            accepted.append(sample_from(r, rng))
            return accepted, len(accepted)     # stop; next block starts fresh
    #  All k accepted -- bonus target sample "for free"
    accepted.append(sample_from(p_target, rng))
    return accepted, len(accepted)


def simulate_run(p_draft, p_target, total_tokens=1000, k=4, seed=0):
    rng = np.random.default_rng(seed)
    tokens = []
    n_blocks = 0
    n_accepted_total = 0
    while len(tokens) < total_tokens:
        block, n_acc = speculative_step(p_draft, p_target, k, rng)
        tokens.extend(block)
        n_blocks += 1
        n_accepted_total += n_acc - 1     # last token is the correction / bonus
    acc_rate = n_accepted_total / (n_blocks * k)
    speedup = len(tokens) / n_blocks     # tokens per (one target forward pass equivalent)
    return {"n_blocks": n_blocks, "n_tokens": len(tokens),
            "acceptance_rate": acc_rate, "empirical_speedup": speedup}


if __name__ == "__main__":
    print("=== Speculative decoding -- lossless inference speedup ===\n")
    rng = np.random.default_rng(0)
    V = 32
    #  Target distribution (spiky)
    p_target = rng.dirichlet(np.ones(V) * 0.3)
    #  Aligned draft (close to target)
    p_draft_good = 0.8 * p_target + 0.2 * (np.ones(V) / V)
    #  Poor draft (uniform-ish)
    p_draft_bad = 0.9 * (np.ones(V) / V) + 0.1 * p_target

    for label, p_draft in [("aligned draft (KL small)", p_draft_good),
                            ("uniform-ish draft (KL big)", p_draft_bad)]:
        r = simulate_run(p_draft, p_target, total_tokens=2000, k=4, seed=0)
        print(f"  {label}")
        print(f"    blocks used   = {r['n_blocks']}   for {r['n_tokens']} tokens")
        print(f"    acceptance    = {r['acceptance_rate']:.3f}")
        print(f"    speedup       = {r['empirical_speedup']:.2f}x  vs 1 tok/block")
        print()

    print("  Interpretation: with an aligned draft, ~4-5x per-token speedup;")
    print("  with a poor draft, the overhead can EXCEED plain autoregressive decoding.")

    print("\n--- library cross-check (vLLM / transformers.SpeculativeDecoder Python) ---")
