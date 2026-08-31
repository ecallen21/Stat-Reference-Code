"""Soft Actor-Critic — SAC (Haarnoja 2018; Reference §28.x extra).

Maximum-entropy RL: augment the reward with a policy-entropy bonus:
    J(pi) = E_pi [ sum_t r_t + alpha * H(pi(-|s_t)) ]

Two Q networks (double-Q trick) prevent overestimation:
    Q_target(s, a) = r + gamma * E_{a'} [ min(Q1_target(s', a'), Q2_target(s', a'))
                                            - alpha * log pi(a' | s') ]

Policy trained to maximise:
    J_pi(theta) = E_s [ E_{a ~ pi} [ alpha * log pi(a | s) - min_j Q_j(s, a) ] ]
(re-parameterisation trick a = mu_theta(s) + sigma_theta(s) * eps)

Temperature alpha can be a learnable Lagrangian variable tuned to target entropy.

Demo: tabular soft policy iteration on a 5-state MDP.  The demo showcases the
soft-Bellman update Q <- r + gamma * V^soft(s'), with V^soft(s') = alpha *
log sum_a exp(Q(s', a) / alpha).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=axis, keepdims=True)


class LineWorld:
    def __init__(self, n=5): self.n = n
    def reset(self): return 0
    def step(self, s, a):
        if s == self.n - 1:
            return s, 0.0, True
        s2 = max(0, min(self.n - 1, s + (1 if a == 1 else -1)))
        r = 10.0 if s2 == self.n - 1 else -1.0
        return s2, r, s2 == self.n - 1


def sac_tabular(env, alpha: float = 0.5, gamma: float = 0.9,
                 n_iter: int = 200) -> dict:
    """Soft policy iteration: repeat soft-Q evaluation + soft-greedy improvement.
    V_soft(s) = alpha * logsumexp(Q(s, :) / alpha).
    pi(a | s) = softmax(Q(s, :) / alpha)."""
    S, A = env.n, 2
    Q = np.zeros((S, A))
    # build transition + reward model (fully known here for exact updates)
    P = np.zeros((S, A, S)); R = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            if s == S - 1:                                  # absorbing
                P[s, a, s] = 1.0; continue
            s2 = max(0, min(S - 1, s + (1 if a == 1 else -1)))
            P[s, a, s2] = 1.0
            R[s, a, s2] = 10.0 if s2 == S - 1 else -1.0

    for _ in range(n_iter):
        # soft-value
        V_soft = alpha * np.log(np.sum(np.exp(Q / alpha), axis=1))
        # soft Bellman backup
        Q_new = np.sum(P * (R + gamma * V_soft[None, None, :]), axis=-1)
        if np.max(np.abs(Q_new - Q)) < 1e-8:
            Q = Q_new; break
        Q = Q_new
    pi = _softmax(Q / alpha, axis=1)
    return {"Q": Q, "V_soft": V_soft, "pi": pi,
            "method": f"soft policy iteration (alpha={alpha})"}


if __name__ == "__main__":
    env = LineWorld(n=5)
    print("=== Soft policy iteration (SAC-style) on LineWorld ===\n")
    for alpha in (0.1, 0.5, 2.0):
        r = sac_tabular(env, alpha=alpha, gamma=0.9)
        print(f"  alpha = {alpha}:")
        print(f"    V_soft(s) = {[round(float(v), 2) for v in r['V_soft']]}")
        print(f"    P(right | s) = {[round(float(p[1]), 3) for p in r['pi']]}")
        print(f"    (larger alpha -> softer policy, closer to uniform)\n")

    print("--- library cross-check (stable-baselines3.SAC; cleanrl/sac_continuous_action.py) ---")
