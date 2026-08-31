"""Imitation learning: behavioural cloning + DAgger (Reference §28.8).

Given expert demonstrations D = { (s_i, a_i^expert) } , learn a policy pi to
mimic the expert.

  * BEHAVIOURAL CLONING (BC): supervised classification / regression:
        min_theta  E_(s, a) ~ D [ - log pi_theta(a | s) ]
    Simple; suffers from compounding-error distribution shift.

  * DAgger (Ross-Gordon-Bagnell 2011): iterate BC while adding data from
    the learner's own state distribution, LABELLED BY THE EXPERT:
        for iter in 1..N:
            roll out pi_theta -> collect visited states S_new
            query expert for actions a_new = expert(S_new)
            D <- D + {(s, a_new) for s in S_new}
            retrain pi_theta on D
    Provably matches expert performance under mild assumptions.

Demo on the LineWorld env where the expert is "always right".
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=-1, keepdims=True)


class LineWorld:
    def __init__(self, n=5, drift: float = 0.15):
        self.n = n; self.drift = drift
    def reset(self): return 0
    def step(self, s, a, rng):
        if s == self.n - 1:
            return s, 0.0, True
        # slippery: with p=drift, act opposite of a
        if rng.uniform() < self.drift:
            a = 1 - a
        s2 = max(0, min(self.n - 1, s + (1 if a == 1 else -1)))
        r = 10.0 if s2 == self.n - 1 else -1.0
        return s2, r, s2 == self.n - 1


def expert_policy(s, env):
    """Always move right."""
    return 1


def train_bc_or_dagger(env, method: str, n_iter: int = 10, n_ep: int = 5,
                        seed: int = 0) -> dict:
    """method in {'BC', 'DAgger'}."""
    rng = np.random.default_rng(seed)
    # tabular softmax policy
    theta = np.zeros((env.n, 2))
    # initial dataset: full expert rollout(s)
    D = []
    for _ in range(n_ep):
        s = env.reset(); done = False; steps = 0
        while not done and steps < 20:
            a = expert_policy(s, env)
            D.append((s, a))
            s, _, done = env.step(s, a, rng)
            steps += 1
    def _refit():
        # count-based supervised learning: theta[s, a] = log frequency of a at s
        counts = np.ones((env.n, 2))                       # Laplace smoothing
        for s, a in D:
            counts[s, a] += 1
        return np.log(counts)

    for it in range(n_iter):
        theta = _refit()
        if method == "DAgger":
            # roll out the CURRENT policy; record visited states; query expert
            s = env.reset(); done = False; steps = 0
            while not done and steps < 30:
                probs = _softmax(theta[s])
                a = int(rng.choice(2, p=probs))
                D.append((s, expert_policy(s, env)))
                s, _, done = env.step(s, a, rng)
                steps += 1
    # evaluate over many episodes
    theta = _refit()
    returns = []
    for _ in range(100):
        s = env.reset(); done = False; total = 0.0; steps = 0
        while not done and steps < 30:
            a = int(_softmax(theta[s]).argmax())
            s, r, done = env.step(s, a, rng)
            total += r; steps += 1
        returns.append(total)
    return {"theta": theta, "mean_return": float(np.mean(returns)),
            "n_data": len(D), "method": method}


if __name__ == "__main__":
    env = LineWorld(n=5, drift=0.15)
    print(f"=== Imitation learning on slippery LineWorld (drift=0.15) ===")
    bc = train_bc_or_dagger(env, "BC", n_iter=1, n_ep=5, seed=0)
    dagger = train_bc_or_dagger(env, "DAgger", n_iter=8, n_ep=5, seed=0)
    print(f"  BC (5 expert rollouts, no DAgger iterations):")
    print(f"    mean return = {bc['mean_return']:.2f}   dataset size = {bc['n_data']}")
    print(f"    policy P(right) per state: "
          f"{[round(_softmax(bc['theta'][s])[1], 2) for s in range(env.n)]}")
    print(f"  DAgger (8 iterations of learner-driven queries):")
    print(f"    mean return = {dagger['mean_return']:.2f}   dataset size = {dagger['n_data']}")
    print(f"    policy P(right) per state: "
          f"{[round(_softmax(dagger['theta'][s])[1], 2) for s in range(env.n)]}")

    print("\n--- library cross-check (imitation, stable-baselines3-contrib GAIL/AIRL) ---")
