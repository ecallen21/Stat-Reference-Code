"""Top-k / top-p (nucleus) decoding (Reference Sec 47.110).

Fan, Lewis & Dauphin 2018 'Hierarchical neural story generation'
(top-k); Holtzman, Buys, Du, Forbes & Choi 2020 'The curious case
of neural text degeneration' (top-p / nucleus). Both truncate the
next-token distribution before sampling:

    top-k:  restrict to the K most-probable tokens.
    top-p:  restrict to the SMALLEST set whose cumulative
            probability >= p (adaptive vocab size).

Temperature T rescales logits before softmax:
    p_i ∝ exp(logit_i / T).

Illustrated with entropy / repetition metrics on toy long-tail
distributions.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def softmax(logits, T=1.0):
    z = logits / T - (logits / T).max()
    p = np.exp(z); return p / p.sum()


def top_k_sample(logits, K, T=1.0, rng=None):
    rng = rng or np.random.default_rng(0)
    idx = np.argsort(logits)[::-1][:K]
    masked = np.full_like(logits, -np.inf); masked[idx] = logits[idx]
    return int(rng.choice(len(logits), p=softmax(masked, T)))


def top_p_sample(logits, p, T=1.0, rng=None):
    rng = rng or np.random.default_rng(0)
    probs = softmax(logits, T)
    order = np.argsort(probs)[::-1]
    cum = np.cumsum(probs[order])
    cutoff = int(np.searchsorted(cum, p) + 1)
    keep = order[:cutoff]
    masked = np.full_like(logits, -np.inf); masked[keep] = logits[keep]
    return int(rng.choice(len(logits), p=softmax(masked, T)))


def entropy(p):
    return float(-np.sum(np.where(p > 0, p * np.log(p), 0.0)))


if __name__ == "__main__":
    print("=== Top-k / top-p decoding (Fan 2018; Holtzman 2020) ===\n")
    rng = np.random.default_rng(0)
    V = 200
    # Long-tail Zipf-like logits
    logits = -np.log(np.arange(1, V + 1)) * 2.0

    p_full = softmax(logits, T=1.0)
    print(f"  Full softmax entropy = {entropy(p_full):.3f}   (log V = {np.log(V):.3f})")

    for T in [0.5, 1.0, 1.5]:
        p = softmax(logits, T)
        print(f"    T = {T:.1f}  entropy = {entropy(p):.3f}   "
              f"top-1 prob = {p.max():.3f}   effective size = {int(1/p.max()):3d}")

    # Sampling-based repetition / diversity
    n_samples = 5000
    for K in [1, 5, 20, 50]:
        samples = [top_k_sample(logits, K, T=1.0, rng=rng) for _ in range(n_samples)]
        n_unique = len(set(samples))
        print(f"  top-k K={K:2d}   unique tokens in {n_samples} samples: {n_unique}")
    print()
    for p in [0.3, 0.6, 0.9, 0.99]:
        samples = [top_p_sample(logits, p, T=1.0, rng=rng) for _ in range(n_samples)]
        n_unique = len(set(samples))
        print(f"  top-p p={p:.2f}   unique tokens in {n_samples} samples: {n_unique}")

    print("\n  Larger K / p / T  ->  more diverse / less repetitive text.")
    print("  Greedy (K=1) collapses; full softmax (K=V, p=1) hallucinates.")

    print("\n--- library cross-check (transformers.generation Python; text/torch R) ---")
