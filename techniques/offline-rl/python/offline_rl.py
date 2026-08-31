"""Offline reinforcement learning: CQL-lite (Reference §28.9).

Offline RL learns a policy from a FIXED dataset D of transitions
(no environment interaction).  Naive off-policy Q-learning fails because
the max-a' Q update at the target extrapolates into unseen actions, giving
overestimation.

CQL (Kumar 2020) adds a penalty that keeps Q(s, a) for OOD actions low:

    L_CQL = 0.5 * (Q - target)^2  +  alpha * ( log sum_a exp Q(s, a)
                                                - E_{a ~ D} Q(s, a) )

Result: Q gives low value to actions not seen in D, so the greedy policy
stays close to the behaviour policy that generated D.

Demo: build a small tabular Q learner with the CQL penalty.  Compare its
policy against naive Q-learning that ignores dataset support.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


class LineWorld:
    def __init__(self, n=5): self.n = n
    def reset(self): return 0
    def step(self, s, a):
        if s == self.n - 1:
            return s, 0.0, True
        s2 = max(0, min(self.n - 1, s + (1 if a == 1 else -1)))
        r = 10.0 if s2 == self.n - 1 else -1.0
        return s2, r, s2 == self.n - 1


def collect_offline_dataset(env, n_ep: int = 30, right_bias: float = 0.9, seed: int = 0):
    """Behaviour policy: with 'right_bias' probability go right; else random.
    This dataset covers many 'right' actions and fewer 'left' actions."""
    rng = np.random.default_rng(seed)
    D = []
    for _ in range(n_ep):
        s = env.reset(); done = False; steps = 0
        while not done and steps < 20:
            a = 1 if rng.uniform() < right_bias else int(rng.integers(2))
            s2, r, done = env.step(s, a)
            D.append((s, a, r, s2, done)); s = s2; steps += 1
    return D


def fit_offline_q(D, S: int, A: int, gamma: float = 0.9,
                   alpha_cql: float = 0.0, lr: float = 0.1,
                   n_iter: int = 200) -> dict:
    """Iterate Q-learning-style updates on the fixed dataset.
    alpha_cql = 0.0 -> naive; alpha_cql > 0 -> CQL-like penalty."""
    Q = np.zeros((S, A))
    for _ in range(n_iter):
        for s, a, r, s2, done in D:
            target = r + gamma * Q[s2].max() * (0 if done else 1)
            Q[s, a] += lr * (target - Q[s, a])
        if alpha_cql > 0:
            # CQL penalty step: pull DOWN Q at OOD actions, keep dataset-actions unchanged
            for s in range(S):
                dataset_actions = [a for (ss, a, *_) in D if ss == s]
                if not dataset_actions:
                    continue
                mean_data_q = float(np.mean([Q[s, a] for a in dataset_actions]))
                logsumexp = float(np.log(np.exp(Q[s]).sum()))
                for a in range(A):
                    if a not in dataset_actions:
                        Q[s, a] -= alpha_cql * lr * (logsumexp - mean_data_q)
    return {"Q": Q, "policy": Q.argmax(axis=1)}


if __name__ == "__main__":
    env = LineWorld(n=5)
    D = collect_offline_dataset(env, n_ep=30, right_bias=0.95)
    naive = fit_offline_q(D, S=env.n, A=2, alpha_cql=0.0)
    cql = fit_offline_q(D, S=env.n, A=2, alpha_cql=1.0)

    print(f"=== Offline Q-learning on {len(D)} transitions (right-biased dataset) ===")
    print(f"  behaviour policy: right 95% of the time; dataset covers 'right' well, 'left' rarely.")
    print(f"  action counts by state:")
    for s in range(env.n):
        cnt = [sum(1 for (ss, a, *_) in D if ss == s and a == ai) for ai in range(2)]
        print(f"    state {s}: L={cnt[0]}, R={cnt[1]}")

    print(f"\n  naive Q-learning (no CQL):")
    for s in range(env.n):
        print(f"    Q({s}, L)={naive['Q'][s, 0]:+.2f}, Q({s}, R)={naive['Q'][s, 1]:+.2f}, "
              f"policy={'R' if naive['policy'][s] == 1 else 'L'}")

    print(f"\n  CQL (alpha=1.0, penalises OOD actions):")
    for s in range(env.n):
        print(f"    Q({s}, L)={cql['Q'][s, 0]:+.2f}, Q({s}, R)={cql['Q'][s, 1]:+.2f}, "
              f"policy={'R' if cql['policy'][s] == 1 else 'L'}")

    print("\n--- library cross-check (d3rlpy CQL / BCQ / IQL / TD3+BC; RLlib offline) ---")
