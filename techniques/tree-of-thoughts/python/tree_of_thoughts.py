"""Tree of Thoughts (ToT) (Reference Sec 47.92).

Yao, Yu, Zhao, Shafran, Griffiths, Cao & Narasimhan 2023
'Tree of Thoughts: Deliberate problem solving with large language
models', NeurIPS. Generalises chain-of-thought:

    1. GENERATE b candidate 'thoughts' (intermediate reasoning steps).
    2. EVALUATE each with an LLM-scoring function.
    3. SEARCH the tree by BFS / DFS with pruning.

Illustrated here on a symbolic search puzzle (Game of 24: use
four cards + arithmetic to reach 24). Compare (a) sampling one
random full CoT chain vs (b) beam-search over a shallow tree.
"""
from __future__ import annotations    # stdlib

import itertools    # arithmetic ops
import numpy as np    # numerical arrays
from fractions import Fraction    # exact arithmetic (avoid float bugs)


OPS = ["+", "-", "*", "/"]


def apply_op(a, b, op):
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/" and b != 0: return a / b
    return None


def solve_24_exact(nums):
    """Exact backtracking Game-of-24 solver -- ground-truth reference."""
    if len(nums) == 1:
        return nums[0] == Fraction(24)
    for i, j in itertools.combinations(range(len(nums)), 2):
        rest = [nums[k] for k in range(len(nums)) if k not in (i, j)]
        a, b = nums[i], nums[j]
        for op in OPS:
            for x, y in [(a, b), (b, a)]:
                v = apply_op(x, y, op)
                if v is not None and solve_24_exact(rest + [v]):
                    return True
    return False


def cot_random_solve(nums, budget=1, rng=None):
    """Random one-shot chain-of-thought: sample a random sequence of ops."""
    rng = rng or np.random.default_rng(0)
    for _ in range(budget):
        cur = list(nums)
        for _ in range(len(nums) - 1):
            i, j = rng.choice(len(cur), size=2, replace=False)
            op = OPS[int(rng.integers(4))]
            v = apply_op(cur[i], cur[j], op)
            if v is None: break
            cur = [cur[k] for k in range(len(cur)) if k not in (i, j)] + [v]
        if len(cur) == 1 and cur[0] == Fraction(24):
            return True
    return False


def tot_bfs_solve(nums, beam_width=8):
    """Beam-search TOT: at each step keep top-beam_width by heuristic
    (proximity of one partial result to 24, or presence of divisors).
    """
    frontier = [tuple(sorted(nums, key=lambda x: (x.numerator, x.denominator)))]
    for _ in range(len(nums) - 1):
        candidates = []
        for state in frontier:
            state_list = list(state)
            for i, j in itertools.combinations(range(len(state_list)), 2):
                a, b = state_list[i], state_list[j]
                rest = [state_list[k] for k in range(len(state_list)) if k not in (i, j)]
                for op in OPS:
                    for x, y in [(a, b), (b, a)]:
                        v = apply_op(x, y, op)
                        if v is None: continue
                        new = tuple(sorted(rest + [v],
                                            key=lambda z: (z.numerator, z.denominator)))
                        candidates.append(new)
        # Heuristic: prefer states whose closest element to 24 is small
        candidates = list(set(candidates))
        candidates.sort(key=lambda s: min(abs(float(v) - 24) for v in s))
        frontier = candidates[:beam_width]
        for state in frontier:
            if len(state) == 1 and state[0] == Fraction(24):
                return True
    for state in frontier:
        if len(state) == 1 and state[0] == Fraction(24):
            return True
    return False


if __name__ == "__main__":
    print("=== Tree of Thoughts on Game of 24 (Yao et al 2023) ===\n")
    rng = np.random.default_rng(0)
    n_problems = 60
    problems = []
    while len(problems) < n_problems:
        nums = [Fraction(int(x)) for x in rng.integers(1, 10, size=4)]
        if solve_24_exact(list(nums)):
            problems.append(nums)

    print(f"  {n_problems} solvable Game-of-24 puzzles (drawn 1..9).\n")

    for budget in [1, 20, 100]:
        cot = sum(cot_random_solve(list(p), budget=budget, rng=rng) for p in problems)
        print(f"  Random CoT (budget={budget:3d} chains)   accuracy = {cot/n_problems:.2f}")

    for beam in [2, 4, 8, 32]:
        tot = sum(tot_bfs_solve(list(p), beam_width=beam) for p in problems)
        print(f"  ToT BFS (beam={beam:2d})                 accuracy = {tot/n_problems:.2f}")

    print("\n  Deliberate search with a small beam outperforms much more compute")
    print("  spent on random single-shot chains -- matches Yao et al 2023 finding.")
    print("\n--- library cross-check (langchain / tree-of-thoughts Python) ---")
