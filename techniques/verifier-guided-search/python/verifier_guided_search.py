"""Verifier-Guided Search (Reference Sec 47.205).

Uesato et al 2022 'Solving math word problems with process- and
outcome-based feedback'; Lightman 2023; Wang 2024 Math-Shepherd.
Combine LLM generation with a learned VERIFIER (reward / value
model) to prune bad reasoning paths during beam search or MCTS:

    for step t:
        candidates = generate_next_step(context)
        scores = verifier(context, candidates)
        keep top-b beams by score
    Backtrack from best final beam.

Substantially better sample efficiency than plain best-of-N when
compute is limited.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def next_steps(current_value, target):
    """Enumerate possible next arithmetic operations."""
    steps = []
    for op, fn in [("+2", lambda x: x + 2), ("+3", lambda x: x + 3),
                    ("*2", lambda x: x * 2), ("-1", lambda x: x - 1)]:
        steps.append((op, fn(current_value)))
    return steps


def verifier(value, target):
    """Learned verifier ≈ negative distance to target."""
    return -abs(value - target) - 0.05 * max(0, value - target * 2)


def verifier_beam_search(start, target, depth, beam=3):
    """Beam-search with per-step verifier scoring."""
    beams = [(start, [])]
    for d in range(depth):
        cands = []
        for val, path in beams:
            for op, nv in next_steps(val, target):
                cands.append((nv, path + [op]))
        cands.sort(key=lambda x: verifier(x[0], target), reverse=True)
        beams = cands[:beam]
    return beams[0]


def greedy_best_of_n(start, target, depth, N=20, rng=None):
    """Sample N random paths; pick the one closest to target."""
    if rng is None: rng = np.random.default_rng(0)
    best = None; best_score = -np.inf
    for _ in range(N):
        val = start; path = []
        for _ in range(depth):
            op, nv = next_steps(val, target)[int(rng.integers(4))]
            val = nv; path.append(op)
        s = verifier(val, target)
        if s > best_score:
            best_score = s; best = (val, path)
    return best


if __name__ == "__main__":
    print("=== Verifier-Guided Search (Uesato 2022; Lightman 2023) ===\n")
    rng = np.random.default_rng(0)

    tasks = [(1, 7, 3), (1, 17, 4), (1, 24, 5), (5, 100, 6), (2, 33, 5)]

    print(f"  {'start→target':<15}  {'depth':>5}  {'greedy-N=20':<15}  {'verifier-beam':<15}")
    ver_hits = 0; g_hits = 0
    for start, target, depth in tasks:
        v = verifier_beam_search(start, target, depth, beam=3)
        g = greedy_best_of_n(start, target, depth, N=20, rng=rng)
        v_ok = abs(v[0] - target) <= 1
        g_ok = abs(g[0] - target) <= 1
        ver_hits += v_ok; g_hits += g_ok
        print(f"  {str(start)+' → '+str(target):<15}  {depth:>5}  "
              f"{str(g[0])+' '+('✓' if g_ok else '✗'):<15}  "
              f"{str(v[0])+' '+('✓' if v_ok else '✗'):<15}")

    print(f"\n  Reached target (|error| ≤ 1): greedy {g_hits}/{len(tasks)}   "
          f"verifier-beam {ver_hits}/{len(tasks)}")

    print("\n--- library cross-check (llm-reasoners, tot-llm, princeton-nlp/tree-of-thought) ---")
