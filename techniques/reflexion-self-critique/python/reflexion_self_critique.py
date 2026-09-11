"""Reflexion - Self-Critique with Verbal RL (Reference Sec 47.201).

Shinn, Cassano, Berman, Gopinath, Narasimhan & Yao 2023 'Reflexion:
Language Agents with Verbal Reinforcement Learning', NeurIPS.
Instead of gradient RL, use SELF-GENERATED VERBAL FEEDBACK stored
in an EPISODIC MEMORY:

    for trial t = 1..T:
        action = LM(prompt, memory)
        result = env(action)
        if success: break
        reflection = LM("what went wrong? how to fix?", action, result)
        memory.append(reflection)

Beats greedy CoT on HumanEval / HotpotQA / AlfWorld by 5-20 pts.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def solve_with_memory(problem, memory, rng, verbose=False):
    """Toy 'LM': try random arithmetic; use memory to avoid past mistakes."""
    tried = {m["action"] for m in memory}
    for _ in range(100):
        op = rng.choice(["+", "-", "*", "/"])
        action = f"{problem[0]} {op} {problem[1]}"
        if action in tried: continue
        try:
            result = eval(action)
            correct = abs(result - problem[2]) < 1e-6
            return action, result, correct
        except Exception:
            continue
    return "no-op", None, False


def reflexion_reflect(action, result, target):
    """Toy critic: identify what went wrong."""
    if result is None:
        return {"action": action, "note": "no-op"}
    if abs(result - target) < 1e-6:
        return {"action": action, "note": "correct"}
    return {"action": action, "note": f"got {result}, target {target}"}


def reflexion_loop(problem, max_trials=5, seed=0):
    """Run Reflexion trials until success or budget exhausted."""
    rng = np.random.default_rng(seed)
    memory = []
    for t in range(max_trials):
        action, result, correct = solve_with_memory(problem, memory, rng)
        if correct:
            return {"success": True, "trials": t + 1, "final_action": action, "memory": memory}
        memory.append(reflexion_reflect(action, result, problem[2]))
    return {"success": False, "trials": max_trials, "final_action": action, "memory": memory}


def greedy_baseline(problem, max_trials=5, seed=0):
    """No memory: same LM but no reflection."""
    rng = np.random.default_rng(seed)
    for t in range(max_trials):
        action, result, correct = solve_with_memory(problem, [], rng)
        if correct:
            return {"success": True, "trials": t + 1, "final_action": action}
    return {"success": False, "trials": max_trials, "final_action": action}


if __name__ == "__main__":
    print("=== Reflexion (Shinn et al 2023 NeurIPS) ===\n")

    # 6 arithmetic puzzles: solve target with two operands + one op
    problems = [(6, 3, 18), (7, 4, 28), (12, 3, 4), (5, 5, 10), (8, 2, 6), (9, 3, 3)]

    reflexion_wins = 0; baseline_wins = 0
    print(f"  {'problem':<12}  {'baseline':<12}  {'reflexion':<15}")
    for p in problems:
        b = greedy_baseline(p, max_trials=3, seed=0)
        r = reflexion_loop(p, max_trials=3, seed=0)
        reflexion_wins += r["success"]; baseline_wins += b["success"]
        print(f"  {str(p):<12}  {'✓ in ' + str(b['trials']) if b['success'] else '✗':<12}  "
              f"{'✓ in ' + str(r['trials']) if r['success'] else '✗':<15}")
    print(f"\n  Solved: baseline {baseline_wins}/{len(problems)}   "
          f"reflexion {reflexion_wins}/{len(problems)}")

    print("\n  Real Reflexion uses an LM to generate rich verbal reflections; here")
    print("  the memory just stores 'don't try this action again' notes.")

    print("\n--- library cross-check (noahshinn024/reflexion, langgraph agents Python) ---")
