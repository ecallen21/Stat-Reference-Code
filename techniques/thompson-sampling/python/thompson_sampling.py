"""Thompson sampling (Reference Sec 47.60).

Thompson 1933 'On the likelihood that one unknown probability
exceeds another in view of the evidence of two samples',
Biometrika 25. For each arm k maintain a posterior over its mean
reward. At each round:

    1. Draw one sample  theta_k ~ posterior_k  for every arm.
    2. Play argmax_k theta_k.
    3. Observe reward, update the posterior.

Bernoulli bandits use Beta(alpha, beta) conjugate updates. Regret
scales as O(log T) matching the Lai-Robbins lower bound (Agrawal
& Goyal 2012).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def thompson_bernoulli(true_means, T, seed=0):
    """Beta-Bernoulli Thompson sampling."""
    rng = np.random.default_rng(seed)
    K = len(true_means)
    alpha = np.ones(K); beta = np.ones(K)
    plays = np.zeros(K, dtype=int)
    rewards = np.zeros(T)
    regret = np.zeros(T)
    best = max(true_means)
    for t in range(T):
        theta = rng.beta(alpha, beta)
        k = int(np.argmax(theta))
        r = int(rng.random() < true_means[k])
        alpha[k] += r
        beta[k] += 1 - r
        plays[k] += 1
        rewards[t] = r
        regret[t] = best - true_means[k]
    return {"plays": plays, "rewards": rewards, "cum_regret": np.cumsum(regret),
            "posterior_mean": alpha / (alpha + beta)}


def ucb1(true_means, T, seed=0):
    """UCB1 baseline (Auer-Cesa-Bianchi-Fischer 2002)."""
    rng = np.random.default_rng(seed)
    K = len(true_means)
    mu_hat = np.zeros(K); n = np.zeros(K)
    regret = np.zeros(T); best = max(true_means)
    for t in range(T):
        if t < K:
            k = t
        else:
            ucb = mu_hat + np.sqrt(2 * np.log(t + 1) / np.maximum(n, 1))
            k = int(np.argmax(ucb))
        r = int(rng.random() < true_means[k])
        n[k] += 1
        mu_hat[k] += (r - mu_hat[k]) / n[k]
        regret[t] = best - true_means[k]
    return {"cum_regret": np.cumsum(regret), "n": n}


if __name__ == "__main__":
    print("=== Thompson sampling (Thompson 1933) ===\n")
    true_means = np.array([0.30, 0.50, 0.55, 0.60, 0.70])
    T = 5000
    ts = thompson_bernoulli(true_means, T, seed=0)
    ucb = ucb1(true_means, T, seed=0)
    print(f"  Arms: means = {true_means}")
    print(f"  T = {T}")
    print(f"  Thompson plays / arm:  {ts['plays']}")
    print(f"  Thompson posterior mean: {np.round(ts['posterior_mean'], 3)}")
    print(f"  Thompson total regret: {ts['cum_regret'][-1]:.2f}")
    print(f"  UCB1     plays / arm:  {ucb['n'].astype(int)}")
    print(f"  UCB1     total regret: {ucb['cum_regret'][-1]:.2f}")
    print(f"\n  Both converge to the best arm; Thompson usually has lower")
    print(f"  finite-time regret (matches Lai-Robbins asymptotic optimum).")

    print("\n--- library cross-check (contextual R; scikit-multiflow / mabwiser Python) ---")
