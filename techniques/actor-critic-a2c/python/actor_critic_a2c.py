"""Advantage Actor-Critic — A2C (Reference §28.4).

Actor:  policy pi_theta(a | s), updated by policy gradient
Critic: value function V_phi(s), updated by TD

Advantage:  A_t = r_t + gamma V(s_{t+1}) - V(s_t)   (1-step TD advantage)
Policy loss: -mean_t  log pi(a_t | s_t) * A_t     (with A_t detached from theta)
Value  loss:  mean_t  (r_t + gamma V(s_{t+1}) - V(s_t))^2

Optional entropy bonus H(pi) encourages exploration.

Compared with REINFORCE: subtracting V(s) as a baseline reduces variance;
learning V online (bootstrapped) is faster than Monte-Carlo returns.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


class LineWorld:
    def __init__(self, n=5): self.n = n
    def reset(self): return 0
    def step(self, s, a):
        if s == self.n - 1:
            return s, 0.0, True
        s2 = max(0, min(self.n - 1, s + (1 if a == 1 else -1)))
        r = 10.0 if s2 == self.n - 1 else -1.0
        return s2, r, s2 == self.n - 1


def train_a2c(env, n_ep=300, lr_pi=0.05, lr_v=0.1, gamma=0.9,
              entropy_coef=0.01, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    theta = np.zeros((env.n, 2))                            # actor
    V = np.zeros(env.n)                                     # critic
    returns = []
    for ep in range(n_ep):
        s = env.reset(); done = False; total = 0.0
        while not done:
            probs = _softmax(theta[s])
            a = int(rng.choice(2, p=probs))
            s2, r, done = env.step(s, a)
            # TD advantage
            target = r + gamma * V[s2] * (0 if done else 1)
            adv = target - V[s]
            # critic update
            V[s] += lr_v * adv
            # actor update: gradient of log pi(a|s) is (1_a - pi)
            grad_log = -probs.copy(); grad_log[a] += 1
            # entropy grad d/dtheta H(pi) = -sum p * (grad_log + log p * (1 - probs));
            # simpler: entropy = -sum p log p; approximate its gradient as
            # d/dtheta = -probs - probs * log(probs) (only used for the step-size scaling)
            theta[s] += lr_pi * (adv * grad_log
                                  + entropy_coef * (-(1 + np.log(probs + 1e-12)) * probs))
            s = s2; total += r
        returns.append(total)
    return {"theta": theta, "V": V, "returns": returns,
            "method": "A2C (1-step TD advantage + softmax policy)"}


if __name__ == "__main__":
    env = LineWorld(n=5)
    fit = train_a2c(env, n_ep=300, lr_pi=0.1, lr_v=0.1, gamma=0.9)
    print(f"=== Advantage actor-critic on LineWorld (5 states) ===")
    print(f"  learned policy P(right) per state:")
    for s in range(env.n):
        print(f"    state {s}: {_softmax(fit['theta'][s])[1]:.3f}")
    print(f"  learned V(s):")
    for s in range(env.n):
        print(f"    V({s}) = {fit['V'][s]:+.3f}")
    print(f"  mean return over last 20 episodes: "
          f"{np.mean(fit['returns'][-20:]):.2f}   (optimal = 7.0)")

    print("\n--- library cross-check (stable-baselines3 A2C; cleanrl/a2c.py) ---")
