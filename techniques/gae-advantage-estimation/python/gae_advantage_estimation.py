"""Generalised Advantage Estimation (Schulman 2016; Reference §28.12).

Given rewards r_t and value estimates V(s_t), define the 1-step TD residual:
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)     (with V(s_{T+1}) = 0)

GAE(lambda) is an exponentially-weighted sum of future TD residuals:
    A_t^GAE = sum_{k=0}^{T-t-1}  (gamma * lambda)^k * delta_{t+k}

Special cases:
    * lambda = 0 -> A_t = delta_t (high bias, low variance)
    * lambda = 1 -> A_t = R_t - V(s_t) (unbiased Monte-Carlo, high variance)
    * lambda in between -> continuous bias-variance trade-off

Used inside PPO, A2C, TRPO etc.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def compute_gae(rewards, values, gamma: float = 0.99, lam: float = 0.95,
                dones=None):
    """rewards, values: length T (values include V(s_T)=0 as bootstrap; here
    we assume the caller passes T+1 values, last being 0 for a terminal state).
    dones: optional list of episode-boundary flags at each step (True stops
    the recursion)."""
    T = len(rewards)
    values = np.concatenate([np.asarray(values, dtype=float), [0.0]])
    if dones is None:
        dones = [False] * T
    A = np.zeros(T); gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        A[t] = gae
    returns = A + values[:T]                              # value target for the critic
    return {"advantages": A, "returns": returns}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 10
    rewards = rng.normal(loc=0.5, size=T)                  # noisy positive rewards
    values = np.linspace(2.0, 0.5, T)                      # decreasing value estimates
    print(f"=== GAE comparison across lambda ===\n")
    print(f"  rewards : {np.round(rewards, 2).tolist()}")
    print(f"  values  : {np.round(values, 2).tolist()}")
    for lam in (0.0, 0.5, 0.95, 1.0):
        r = compute_gae(rewards, values, gamma=0.99, lam=lam)
        print(f"\n  lambda = {lam}:")
        print(f"    advantages = {np.round(r['advantages'], 3).tolist()}")
        print(f"    var(A)     = {float(r['advantages'].var(ddof=1)):.4f}")

    # sanity: lambda=1 advantages equal MC returns minus values
    mc_returns = np.array([sum((0.99 ** k) * rewards[t + k]
                                 for k in range(T - t)) for t in range(T)])
    r1 = compute_gae(rewards, values, gamma=0.99, lam=1.0)
    diff = float(np.max(np.abs(r1["returns"] - mc_returns)))
    print(f"\n  sanity: |returns(lam=1) - MC returns| max = {diff:.2e}   (should be ~0)")

    print("\n--- library cross-check (stable-baselines3 compute_gae; cleanrl/ppo.py) ---")
