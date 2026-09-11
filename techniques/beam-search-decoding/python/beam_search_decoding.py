"""Beam Search Decoding (Reference Sec 47.158).

Reddy 1977 'Speech understanding systems' (Hearsay-II). Approximate
best-first search over a discrete sequence-generation model. Keeps
top B partial hypotheses ('beams') at each step:

    for t = 1..T:
        for each active beam h:
            expand by all V symbols; score h||v with log p(v | h).
        keep the top B highest-scoring beams (with length norm)

O(T V B) instead of exact O(V^T). Beam width B trades quality
for cost. Length normalisation avoids favouring shorter sequences.
"""
from __future__ import annotations    # stdlib

import heapq    # priority queue for beams
import numpy as np    # numerical arrays


def beam_search(next_logits_fn, start_token, end_token, max_len=20,
                  beam_width=5, length_penalty=1.0):
    """Generic beam search.

    next_logits_fn(prefix) -> log-probability vector over vocabulary.
    Returns the top beam (highest length-normalised score).
    """
    beams = [(0.0, [start_token], False)]                       # (log-score, prefix, done)
    for step in range(max_len):
        candidates = []
        for score, seq, done in beams:
            if done:
                candidates.append((score, seq, True))
                continue
            logp = next_logits_fn(seq)
            top_v = np.argsort(logp)[-beam_width:]
            for v in top_v:
                v = int(v)
                new_seq = seq + [v]
                new_score = score + float(logp[v])
                is_done = (v == end_token)
                candidates.append((new_score, new_seq, is_done))
        # Length-normalised score: score / len^alpha
        candidates.sort(
            key=lambda x: x[0] / max(len(x[1]), 1) ** length_penalty,
            reverse=True,
        )
        beams = candidates[:beam_width]
        if all(b[2] for b in beams):
            break
    beams.sort(key=lambda x: x[0] / max(len(x[1]), 1) ** length_penalty, reverse=True)
    return beams


def greedy_decode(next_logits_fn, start_token, end_token, max_len=20):
    seq = [start_token]; score = 0.0
    for _ in range(max_len):
        logp = next_logits_fn(seq)
        v = int(np.argmax(logp))
        seq.append(v); score += float(logp[v])
        if v == end_token: break
    return score, seq


if __name__ == "__main__":
    print("=== Beam-search decoding (Reddy 1977; Lowerre 1976 Harpy) ===\n")
    rng = np.random.default_rng(0)

    # Toy bigram LM over V=8 tokens. Start=0, End=7.
    V = 8; start, end = 0, 7
    T_trans = rng.dirichlet(np.ones(V) * 0.3, size=V)           # peaky bigrams
    # Handcraft: greedy always goes to symbol 1; but the truly best sequence
    # requires a low-prob first step to unlock a high-prob chain.
    T_trans[0] = np.array([0, 0.55, 0.10, 0.10, 0.15, 0.05, 0.03, 0.02])
    T_trans[4] = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.90, 0.03, 0.02])
    T_trans[5] = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.90, 0.04])
    T_trans[6] = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.93])
    T_trans[1] = np.array([0.05, 0.20, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10])
    T_trans = T_trans / T_trans.sum(axis=1, keepdims=True)

    def next_logits(prefix):
        last = prefix[-1]
        return np.log(T_trans[last] + 1e-12)

    # Greedy
    g_score, g_seq = greedy_decode(next_logits, start, end, max_len=15)
    print(f"  Greedy:    seq = {g_seq}   log_prob = {g_score:.3f}")

    # Beam search for different widths
    for B in [1, 2, 5, 10]:
        beams = beam_search(next_logits, start, end, max_len=15,
                              beam_width=B, length_penalty=0.7)
        top = beams[0]
        print(f"  Beam B={B:2d}: seq = {top[1]}   log_prob = {top[0]:.3f}   len = {len(top[1])}")

    print("\n--- library cross-check (transformers.generate / fairseq / OpenNMT Python) ---")
