"""Exploration strategies (Reference §28.11).

Four workhorse exploration methods, benchmarked on a hard-to-explore
chain MDP.  The agent starts at state 0; goal at state n-1 gives a large
reward; step reward = 0.  Naive eps-greedy needs O(2^n) episodes to find
the goal by chance; intrinsic-motivation methods find it much faster.

Methods:
  * epsilon-greedy: uniform random exploration
  * softmax / Boltzmann: pi(a) = exp(Q(a) / tau) / Z
  * UCB1: augment action selection with sqrt(2 ln t / n_a)
  * count-based intrinsic reward:
        r_intrinsic(s) = beta / sqrt(N(s))
    The agent gets a bonus for visiting rarely-seen states.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


class ChainMDP:
    """States 0..n-1.  Actions: 0=left, 1=right.  Goal at n-1 gives +10, then absorbing.
    Step reward = 0 (very sparse)."""
    def __init__(self, n=8):
        self.n = n
    def reset(self): return 0
    def step(self, s, a):
        if s == self.n - 1:
            return s, 0.0, True
        s2 = max(0, min(self.n - 1, s + (1 if a == 1 else -1)))
        r = 10.0 if s2 == self.n - 1 else 0.0
        return s2, r, s2 == self.n - 1


def _softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def q_learn(env, action_fn, extras: dict = None, n_ep: int = 500,
             alpha: float = 0.5, gamma: float = 0.9, max_steps: int = 50,
             seed: int = 0) -> dict:
    """Generic Q-learning loop.  action_fn(Q, s, t, extras) -> action."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n, 2))
    if extras is None: extras = {}
    extras.setdefault("N", np.zeros(env.n))
    extras.setdefault("N_sa", np.zeros((env.n, 2)))
    returns = []; first_success = None
    t_global = 0
    for ep in range(n_ep):
        s = env.reset(); done = False; total = 0.0; steps = 0
        while not done and steps < max_steps:
            t_global += 1
            a = action_fn(Q, s, t_global, extras, rng)
            s2, r, done = env.step(s, a)
            r_int = extras.get("beta_intrinsic", 0.0) / max(np.sqrt(extras["N"][s2]), 1e-6)
            extras["N"][s2] += 1; extras["N_sa"][s, a] += 1
            Q[s, a] += alpha * (r + r_int + gamma * Q[s2].max() * (0 if done else 1)
                                 - Q[s, a])
            s = s2; total += r; steps += 1
        returns.append(total)
        if total > 0 and first_success is None:
            first_success = ep
    return {"Q": Q, "returns": returns, "first_success": first_success,
            "extras": extras}


def _eps_greedy(Q, s, t, extras, rng, eps=0.2):
    return int(rng.integers(2)) if rng.uniform() < eps else int(Q[s].argmax())


def _boltzmann(Q, s, t, extras, rng, tau=0.5):
    return int(rng.choice(2, p=_softmax(Q[s] / tau)))


def _ucb(Q, s, t, extras, rng, c=2.0):
    N_sa = extras["N_sa"][s]
    bonus = c * np.sqrt(np.log(max(t, 2)) / np.maximum(N_sa, 1e-6))
    return int((Q[s] + bonus).argmax())


if __name__ == "__main__":
    env = ChainMDP(n=8)
    n_ep = 500
    print(f"=== Chain-MDP exploration test (n_states={env.n}, sparse reward at end) ===")
    for name, fn, extras in [
        ("eps-greedy (0.2)", _eps_greedy, {}),
        ("Boltzmann (tau=0.5)", _boltzmann, {}),
        ("UCB1 (c=2)", _ucb, {}),
        ("count-based intrinsic", _eps_greedy,
         {"beta_intrinsic": 0.5}),
    ]:
        r = q_learn(env, fn, extras, n_ep=n_ep, seed=0)
        mean_late = float(np.mean(r["returns"][-50:]))
        fs = r["first_success"]
        print(f"  {name:>28}: first success at ep {fs}, "
              f"mean return over last 50 = {mean_late:.2f}")

    print("\n--- library cross-check (RND, ICM, NGU, curiosity-driven exploration papers) ---")
