"""Dyna-Q: model-based augmentation of Q-learning (Sutton 1990; Reference §28.7).

Dyna-Q maintains an ESTIMATED MODEL P_hat, R_hat learned from real experience,
then interleaves:

    Real step: (s, a, r, s') -> Q-learning update + record in model
    Planning : k times, sample a random (s, a) previously seen, imagine
               (r_hat, s'_hat) from the model, apply a Q-learning update

Effect: each real transition is replayed via model-based updates, giving
Q-learning-like accuracy with a fraction of the environment interactions.

Deep model-based analogues: World Models (Ha 2018), MBPO (Janner 2019),
Dreamer (Hafner 2020-2023), MuZero (Schrittwieser 2020).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

from collections import defaultdict    # stdlib

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


def dyna_q(env, n_ep: int = 40, planning: int = 10, alpha: float = 0.5,
            gamma: float = 0.9, eps: float = 0.1, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n, 2))
    model = {}                                             # (s, a) -> (r, s', done)
    seen = set()
    returns = []
    for ep in range(n_ep):
        s = env.reset(); done = False; total = 0.0
        while not done:
            a = int(rng.integers(2)) if rng.uniform() < eps else int(Q[s].argmax())
            s2, r, done = env.step(s, a)
            # real Q-learning
            Q[s, a] += alpha * (r + gamma * Q[s2].max() * (0 if done else 1) - Q[s, a])
            model[(s, a)] = (r, s2, done); seen.add((s, a))
            # planning updates from the model
            keys = list(seen)
            for _ in range(planning):
                sp, ap = keys[rng.integers(len(keys))]
                rp, s2p, dp = model[(sp, ap)]
                Q[sp, ap] += alpha * (rp + gamma * Q[s2p].max() * (0 if dp else 1)
                                       - Q[sp, ap])
            s = s2; total += r
        returns.append(total)
    return {"Q": Q, "returns": returns,
            "method": f"Dyna-Q ({planning} planning steps per real step)"}


def plain_q_learning(env, n_ep: int = 40, alpha: float = 0.5,
                      gamma: float = 0.9, eps: float = 0.1, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n, 2))
    returns = []
    for ep in range(n_ep):
        s = env.reset(); done = False; total = 0.0
        while not done:
            a = int(rng.integers(2)) if rng.uniform() < eps else int(Q[s].argmax())
            s2, r, done = env.step(s, a)
            Q[s, a] += alpha * (r + gamma * Q[s2].max() * (0 if done else 1) - Q[s, a])
            s = s2; total += r
        returns.append(total)
    return {"Q": Q, "returns": returns, "method": "plain Q-learning"}


if __name__ == "__main__":
    env = LineWorld(n=5)
    q_only = plain_q_learning(env, n_ep=40)
    dyna = dyna_q(env, n_ep=40, planning=10)
    print(f"=== Plain Q-learning vs Dyna-Q on LineWorld (5 states, 40 episodes) ===")
    print(f"  Q-learning:  mean return over last 10 episodes = "
          f"{np.mean(q_only['returns'][-10:]):.2f}")
    print(f"  Dyna-Q(10): mean return over last 10 episodes = "
          f"{np.mean(dyna['returns'][-10:]):.2f}")
    print(f"\n  Q-learning learning curve (first 10 episodes):")
    print(f"    {[round(x, 2) for x in q_only['returns'][:10]]}")
    print(f"  Dyna-Q(10)  learning curve (first 10 episodes):")
    print(f"    {[round(x, 2) for x in dyna['returns'][:10]]}")
    print(f"\n  Dyna-Q typically converges in ~4x fewer real steps than plain Q.")

    print("\n--- library cross-check (Dopamine, cleanrl, MBPO / Dreamer for deep MBRL) ---")
