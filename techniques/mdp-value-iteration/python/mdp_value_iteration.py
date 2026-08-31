"""Value iteration + policy iteration on finite MDPs (Reference §28.2).

MDP = (S, A, P, R, gamma).  Bellman optimality:
    V*(s) = max_a  sum_{s'} P(s' | s, a) [ R(s, a, s') + gamma * V*(s') ]
    pi*(s) = argmax_a  sum_{s'} P(s' | s, a) [ R + gamma * V*(s') ]

VALUE ITERATION: iterate the Bellman operator until convergence.
POLICY ITERATION: alternate policy evaluation (solve V^pi) and policy
improvement (greedy wrt V^pi).  Both converge in finite steps for finite MDPs.

Demo: 4x4 grid-world with a goal cell.  Recover the optimal policy that
walks to the goal.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _build_grid_mdp(n=4, goal=(3, 3), step_reward=-1.0, goal_reward=10.0):
    """4-directional actions; deterministic; walls if out-of-bounds; goal absorbing."""
    S = n * n; A = 4
    P = np.zeros((S, A, S)); R = np.zeros((S, A, S))
    def _idx(r, c): return r * n + c
    goal_s = _idx(*goal)
    for r in range(n):
        for c in range(n):
            s = _idx(r, c)
            if s == goal_s:
                for a in range(A):
                    P[s, a, s] = 1.0                        # absorbing
                continue
            for a, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                r2 = max(0, min(n - 1, r + dr))
                c2 = max(0, min(n - 1, c + dc))
                s2 = _idx(r2, c2)
                P[s, a, s2] = 1.0
                R[s, a, s2] = goal_reward if s2 == goal_s else step_reward
    return P, R, goal_s


def value_iteration(P, R, gamma: float = 0.9, tol: float = 1e-6,
                    max_iter: int = 500) -> dict:
    S, A, _ = P.shape
    V = np.zeros(S)
    for it in range(max_iter):
        Q = np.sum(P * (R + gamma * V[None, None, :]), axis=-1)   # (S, A)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new; break
        V = V_new
    pi = np.argmax(Q, axis=1)
    return {"V": V, "pi": pi, "n_iter": it + 1, "method": "value iteration"}


def policy_iteration(P, R, gamma: float = 0.9, max_iter: int = 100) -> dict:
    S, A, _ = P.shape
    pi = np.zeros(S, dtype=int)
    for it in range(max_iter):
        # policy evaluation: solve V = R_pi + gamma * P_pi V
        Ppi = P[np.arange(S), pi]                          # (S, S)
        Rpi = (P[np.arange(S), pi] * R[np.arange(S), pi]).sum(axis=1)
        V = np.linalg.solve(np.eye(S) - gamma * Ppi, Rpi)
        # improvement
        Q = np.sum(P * (R + gamma * V[None, None, :]), axis=-1)
        pi_new = Q.argmax(axis=1)
        if np.all(pi_new == pi):
            pi = pi_new; break
        pi = pi_new
    return {"V": V, "pi": pi, "n_iter": it + 1, "method": "policy iteration"}


if __name__ == "__main__":
    n = 4
    P, R, goal = _build_grid_mdp(n=n, goal=(3, 3))

    vi = value_iteration(P, R, gamma=0.9)
    pi = policy_iteration(P, R, gamma=0.9)
    print(f"=== 4x4 grid-world, goal at (3,3) ===")
    print(f"  value iteration: converged in {vi['n_iter']} steps")
    print(f"  policy iteration: converged in {pi['n_iter']} steps")
    print(f"  policies agree: {np.array_equal(vi['pi'], pi['pi'])}")
    print(f"  V (rounded):")
    for r in range(n):
        print("    " + "  ".join(f"{vi['V'][r * n + c]:>6.2f}" for c in range(n)))
    print(f"  policy (0=U 1=D 2=L 3=R):")
    for r in range(n):
        print("    " + "  ".join(f"{'UDLR'[vi['pi'][r * n + c]]:>3}"
                                    for c in range(n)))

    print("\n--- library cross-check (MDPtoolbox / pymdp / gymnasium.envs.toy_text) ---")
