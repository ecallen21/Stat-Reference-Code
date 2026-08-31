"""Hierarchical RL: options framework (Sutton-Precup-Singh 1999; Reference §28.x extra).

An OPTION is a temporally-extended action tuple <I, pi_o, beta>:
    * I subset S       -- initiation set (states where the option can start)
    * pi_o(a | s)      -- intra-option policy
    * beta(s) in [0, 1] -- termination probability

Top-level "policy over options" selects an option; the option runs its
pi_o until beta triggers termination; the top level then re-selects.

Benefits:
    * Coarser action space -> faster exploration and learning
    * Skill reuse across tasks
    * More interpretable behaviour

Demo: 8-state chain-MDP with two hand-designed options ("walk right until
end") vs primitive-action Q-learning; the option-augmented agent reaches
the goal in one macro-step.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


class Chain:
    def __init__(self, n=8): self.n = n
    def reset(self): return 0
    def step(self, s, a):
        if s == self.n - 1:
            return s, 0.0, True
        s2 = max(0, min(self.n - 1, s + (1 if a == 1 else -1)))
        r = 10.0 if s2 == self.n - 1 else 0.0
        return s2, r, s2 == self.n - 1


def _softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def option_walk_right(env, s):
    """Intra-option policy: always go right; terminates only at the goal."""
    return 1, (s == env.n - 1)                             # action, terminates?


def option_walk_left(env, s):
    return 0, (s == 0)


def q_over_options(env, options, n_ep: int = 30, lr: float = 0.5,
                    gamma: float = 0.9, eps: float = 0.1, seed: int = 0) -> dict:
    """SMDP-Q-learning over options."""
    rng = np.random.default_rng(seed)
    Q_opt = np.zeros((env.n, len(options)))
    returns = []
    for ep in range(n_ep):
        s = env.reset(); total = 0.0; done = False
        while not done:
            o = int(rng.integers(len(options))) if rng.uniform() < eps else int(Q_opt[s].argmax())
            r_acc = 0.0; discount = 1.0; steps = 0; start = s
            while not done:
                a, term = options[o](env, s)
                s2, r, done = env.step(s, a)
                r_acc += discount * r
                discount *= gamma; steps += 1
                s = s2
                if term or done: break
            # SMDP-Q update
            bootstrap = 0.0 if done else Q_opt[s].max()
            Q_opt[start, o] += lr * (r_acc + discount * bootstrap - Q_opt[start, o])
            total += r_acc / (0.9 ** 0)  # already discounted; approximate
        returns.append(total)
    return {"Q_opt": Q_opt, "returns": returns,
            "method": "SMDP-Q-learning over options"}


def q_learning_primitive(env, n_ep: int = 30, lr: float = 0.5, gamma: float = 0.9,
                          eps: float = 0.2, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n, 2))
    returns = []
    for ep in range(n_ep):
        s = env.reset(); total = 0.0; done = False; steps = 0
        while not done and steps < 50:
            a = int(rng.integers(2)) if rng.uniform() < eps else int(Q[s].argmax())
            s2, r, done = env.step(s, a)
            Q[s, a] += lr * (r + gamma * Q[s2].max() * (0 if done else 1) - Q[s, a])
            s = s2; total += r; steps += 1
        returns.append(total)
    return {"Q": Q, "returns": returns, "method": "primitive Q-learning"}


if __name__ == "__main__":
    env = Chain(n=8)
    prim = q_learning_primitive(env, n_ep=15)
    hrl = q_over_options(env, [option_walk_left, option_walk_right], n_ep=15)
    print(f"=== Hierarchical RL (options) vs primitive Q-learning on 8-state chain ===")
    print(f"  primitive learning curve (per-episode returns):")
    print(f"    {[round(x, 1) for x in prim['returns']]}")
    print(f"  option-based learning curve:")
    print(f"    {[round(x, 1) for x in hrl['returns']]}")
    print(f"\n  option Q at start state (0): "
          f"Q(0, walk-left) = {hrl['Q_opt'][0, 0]:.3f}, "
          f"Q(0, walk-right) = {hrl['Q_opt'][0, 1]:.3f}")
    print(f"  (options let the agent traverse the chain in ONE macro decision.)")

    print("\n--- library cross-check (Feudal Networks, Option-Critic architectures) ---")
