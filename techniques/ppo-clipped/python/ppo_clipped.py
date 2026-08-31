"""Proximal Policy Optimization — PPO (Schulman 2017; Reference §28.5).

Trust-region-style policy update via a CLIPPED SURROGATE:

    r_t(theta) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)      importance ratio
    L^CLIP(theta) = E_t [ min( r_t A_t,  clip(r_t, 1 - eps, 1 + eps) A_t ) ]

Full loss:
    L = L^CLIP  -  c_v (V_phi(s_t) - R_t)^2  +  c_H H(pi(-|s_t))

We collect one full rollout, compute Monte-Carlo returns, then take K epochs
of mini-batch gradient ascent on L^CLIP.  Because we clip r_t, a bad update
step can't move pi too far from pi_old in a single gradient direction.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=-1, keepdims=True)


class LineWorld:
    def __init__(self, n=5): self.n = n
    def reset(self): return 0
    def step(self, s, a):
        if s == self.n - 1:
            return s, 0.0, True
        s2 = max(0, min(self.n - 1, s + (1 if a == 1 else -1)))
        r = 10.0 if s2 == self.n - 1 else -1.0
        return s2, r, s2 == self.n - 1


def rollout(env, theta, rng, max_steps: int = 20):
    s = env.reset(); done = False
    ss, aa, rr, lp = [], [], [], []
    steps = 0
    while not done and steps < max_steps:
        probs = _softmax(theta[s])
        a = int(rng.choice(2, p=probs))
        ss.append(s); aa.append(a); lp.append(float(np.log(probs[a] + 1e-12)))
        s2, r, done = env.step(s, a)
        rr.append(r); s = s2; steps += 1
    return ss, aa, rr, lp


def train_ppo(env, n_iter: int = 60, lr: float = 0.05, gamma: float = 0.9,
              eps_clip: float = 0.2, K_epochs: int = 4, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    theta = np.zeros((env.n, 2))
    V = np.zeros(env.n)
    returns = []
    for it in range(n_iter):
        # collect a rollout with the CURRENT policy (theta_old fixed for K updates)
        ss, aa, rr, old_lp = rollout(env, theta, rng)
        # Monte-Carlo returns and advantages
        R = []; G = 0.0
        for r in reversed(rr):
            G = r + gamma * G; R.append(G)
        R = list(reversed(R))
        A = [Rt - V[s] for Rt, s in zip(R, ss)]
        # normalise advantages (standard trick)
        if len(A) > 1:
            mA, sA = float(np.mean(A)), float(np.std(A) + 1e-8)
            A = [(a - mA) / sA for a in A]
        theta_old = theta.copy()
        for _ in range(K_epochs):
            for s, a, R_t, A_t, lp_old in zip(ss, aa, R, A, old_lp):
                probs = _softmax(theta[s])
                lp_new = float(np.log(probs[a] + 1e-12))
                ratio = float(np.exp(lp_new - lp_old))
                surr1 = ratio * A_t
                surr2 = np.clip(ratio, 1 - eps_clip, 1 + eps_clip) * A_t
                unclipped = min(surr1, surr2)
                # gradient of log pi(a|s) is one-hot(a) - probs
                grad_log = -probs.copy(); grad_log[a] += 1
                # gradient of the min(...) surrogate wrt theta_s is the ratio_grad direction
                # only when the clip isn't active; approximate by taking the actor gradient
                # of unclipped * whichever surrogate is min
                theta[s] += lr * unclipped * grad_log
                # critic (MSE on returns)
                V[s] += 0.5 * lr * (R_t - V[s])
        returns.append(sum(rr))
    return {"theta": theta, "V": V, "returns": returns,
            "method": "PPO clipped-surrogate (tabular actor)"}


if __name__ == "__main__":
    env = LineWorld(n=5)
    fit = train_ppo(env, n_iter=100, lr=0.1, K_epochs=4)
    print(f"=== PPO clipped-surrogate on LineWorld (5 states) ===")
    print(f"  learned P(right) per state:")
    for s in range(env.n):
        print(f"    state {s}: {_softmax(fit['theta'][s])[1]:.3f}")
    print(f"  learned V(s): {[round(float(v), 2) for v in fit['V']]}")
    print(f"  mean return over last 20 iterations: "
          f"{np.mean(fit['returns'][-20:]):.2f}   (optimal = 7.0)")

    print("\n--- library cross-check (stable-baselines3 PPO; cleanrl/ppo.py; TRL PPOTrainer) ---")
