"""Tree of Thoughts - ToT (Reference Sec 47.184).

Yao, Yu, Zhao, Shafran, Griffiths, Cao & Narasimhan 2023 'Tree of
Thoughts: Deliberate Problem Solving with Large Language Models',
NeurIPS. Extends chain-of-thought by exploring MULTIPLE reasoning
paths as a tree and using explicit SEARCH (BFS / DFS) with
value-based pruning:

    1. Decompose problem -> thought steps.
    2. At each node, generate k candidate next thoughts.
    3. Evaluate each candidate (via an LM 'value' call).
    4. Keep top-b beams; expand until solution or budget exhausted.

Beats single-chain CoT on planning / puzzle / creative tasks
(Game-of-24, Creative Writing, mini-crosswords).
"""
from __future__ import annotations    # stdlib

from itertools import combinations, permutations    # candidate generation

import numpy as np    # numerical arrays


def game_of_24_value(state):
    """Value estimator: does the multiset contain 24? or a promising subexpression?"""
    if len(state) == 1:
        return 100 if abs(state[0] - 24) < 1e-6 else -10
    # Heuristic: presence of 8 * 3, 6 * 4, 12 * 2 building blocks helps
    for a, b in permutations(state, 2):
        if a * b == 24 or a + b == 24 or a - b == 24 or (b != 0 and a / b == 24):
            return 50
    # Otherwise: prefer states whose sum / product is close to 24
    return -abs(sum(state) - 24) / 10


def expand_game_of_24(state):
    """Generate all (a op b) -> new state successors."""
    successors = []
    for i, j in combinations(range(len(state)), 2):
        a, b = state[i], state[j]
        rest = [state[k] for k in range(len(state)) if k not in (i, j)]
        for r, op in [(a + b, "+"), (a - b, "-"), (b - a, "-r"),
                       (a * b, "*"),
                       (a / b if b != 0 else None, "/"),
                       (b / a if a != 0 else None, "/r")]:
            if r is None: continue
            successors.append((tuple(sorted(rest + [r])), op))
    # Dedupe
    seen = set(); dedup = []
    for s, op in successors:
        if s not in seen:
            seen.add(s); dedup.append((s, op))
    return dedup


def tot_search(initial_state, expand_fn, value_fn, budget=200, beam=5):
    """BFS-style ToT: keep top-beam by value at each depth."""
    frontier = [initial_state]; expansions = 0; visited = 0
    while frontier and expansions < budget:
        cands = []
        for s in frontier:
            for succ, _ in expand_fn(s):
                cands.append(succ)
                expansions += 1
                visited += 1
                v = value_fn(succ)
                if v >= 100:
                    return {"solution": succ, "expansions": expansions, "visited": visited}
        cands = sorted(set(cands), key=value_fn, reverse=True)[:beam]
        frontier = cands
    return {"solution": None, "expansions": expansions, "visited": visited}


def cot_greedy(initial_state, expand_fn, value_fn, max_depth=6):
    """Chain-of-thought analogue: greedy single-path search."""
    s = initial_state; visited = 0
    for _ in range(max_depth):
        succs = expand_fn(s); visited += len(succs)
        if not succs: break
        s = max((x[0] for x in succs), key=value_fn)
        if value_fn(s) >= 100:
            return {"solution": s, "visited": visited}
    return {"solution": None, "visited": visited}


if __name__ == "__main__":
    print("=== Tree of Thoughts (Yao et al 2023) ===\n")

    puzzles = [(3, 8, 8, 8), (4, 6, 8, 8), (5, 5, 5, 1), (1, 3, 4, 6),
                 (2, 3, 5, 12), (4, 7, 8, 8), (3, 3, 8, 8)]
    print(f"  {'puzzle':>15}  {'CoT-greedy':>12}  {'ToT (beam=8)':>15}")
    cot_wins = 0; tot_wins = 0
    for puz in puzzles:
        cot = cot_greedy(puz, expand_game_of_24, game_of_24_value)
        tot = tot_search(puz, expand_game_of_24, game_of_24_value, budget=500, beam=8)
        cot_ok = cot["solution"] is not None
        tot_ok = tot["solution"] is not None
        cot_wins += cot_ok; tot_wins += tot_ok
        print(f"  {str(puz):>15}  {'yes' if cot_ok else 'no':>12}  "
              f"{'yes' if tot_ok else 'no':>15}")
    print(f"\n  Total solved: CoT-greedy = {cot_wins}/{len(puzzles)}   "
            f"ToT (beam=8) = {tot_wins}/{len(puzzles)}")

    print("\n  Game of 24: reach 24 by using each of 4 input numbers once with + - * /.")
    print("  Greedy CoT often locks onto a bad first step. ToT's beam search recovers.")

    print("\n--- library cross-check (llama-cpp/tree-of-thought, langgraph, lmql Python) ---")
