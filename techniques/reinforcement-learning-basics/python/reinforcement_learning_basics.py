"""Reinforcement learning basics: tabular Q-learning + REINFORCE (Reference §27.x extra).

Tabular Q-learning (Watkins 1989):
    Q(s, a) <- Q(s, a) + alpha [ r + gamma max_a' Q(s', a') - Q(s, a) ]

REINFORCE (Williams 1992) policy gradient:
    theta <- theta + alpha * G_t * grad log pi_theta(a_t | s_t)

Demo: a simple 4-state grid where the agent walks 0 -> 1 -> 2 -> 3 (goal).
Reward = -1 per step; +10 at the goal.  Both methods learn the optimal
policy (always move right).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


class LineWorld:
    """States 0..3; actions: 0 = left, 1 = right; state 3 is terminal goal."""
    def __init__(self, n=4): self.n = n
    def reset(self): return 0
    def step(self, s, a):
        if s == self.n - 1:
            return s, 0.0, True
        s2 = max(0, min(self.n - 1, s + (1 if a == 1 else -1)))
        r = 10.0 if s2 == self.n - 1 else -1.0
        return s2, r, s2 == self.n - 1


def q_learning(env, n_ep: int = 200, alpha: float = 0.5, gamma: float = 0.9,
                eps: float = 0.2, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n, 2))
    returns = []
    for ep in range(n_ep):
        s = env.reset(); done = False; total = 0.0
        while not done:
            a = rng.integers(2) if rng.uniform() < eps else int(Q[s].argmax())
            s2, r, done = env.step(s, a)
            Q[s, a] += alpha * (r + gamma * Q[s2].max() * (not done) - Q[s, a])
            s = s2; total += r
        returns.append(total)
    return {"Q": Q, "returns": returns,
            "method": "tabular Q-learning (epsilon-greedy)"}


def reinforce(env, n_ep: int = 300, alpha: float = 0.05, gamma: float = 0.9,
              seed: int = 0) -> dict:
    """Tabular softmax policy on states -> actions."""
    rng = np.random.default_rng(seed)
    theta = np.zeros((env.n, 2))
    returns = []
    for ep in range(n_ep):
        s = env.reset(); done = False
        traj = []; rewards = []
        while not done:
            probs = _softmax(theta[s])
            a = int(rng.choice(2, p=probs))
            s2, r, done = env.step(s, a)
            traj.append((s, a, probs)); rewards.append(r)
            s = s2
        # discounted returns
        G = 0.0; Gs = [0.0] * len(rewards)
        for t in range(len(rewards) - 1, -1, -1):
            G = rewards[t] + gamma * G; Gs[t] = G
        # gradient update
        for (s, a, probs), G_t in zip(traj, Gs):
            grad = -probs.copy(); grad[a] += 1                # d log pi_a / d theta_s
            theta[s] += alpha * G_t * grad
        returns.append(sum(rewards))
    return {"theta": theta, "returns": returns,
            "method": "REINFORCE (tabular softmax policy)"}


if __name__ == "__main__":
    env = LineWorld(n=4)

    ql = q_learning(env, n_ep=200, alpha=0.5, eps=0.1)
    print(f"=== Q-learning on LineWorld ===")
    print(f"  final Q-table:")
    for s in range(env.n):
        print(f"    state {s}: Q(left, right) = {np.round(ql['Q'][s], 3).tolist()}")
    print(f"  final policy (argmax): "
          f"{[('R' if a == 1 else 'L') for a in ql['Q'].argmax(axis=1)]}")
    print(f"  mean return over last 20 episodes: {np.mean(ql['returns'][-20:]):.2f}")

    rf = reinforce(env, n_ep=300, alpha=0.1)
    print(f"\n=== REINFORCE on LineWorld ===")
    print(f"  final theta:")
    for s in range(env.n):
        p = _softmax(rf["theta"][s])
        print(f"    state {s}: P(right) = {p[1]:.3f}")
    print(f"  mean return over last 20 episodes: {np.mean(rf['returns'][-20:]):.2f}")

    print("\n--- library cross-check (stable-baselines3 DQN / PPO / A2C; gymnasium) ---")
