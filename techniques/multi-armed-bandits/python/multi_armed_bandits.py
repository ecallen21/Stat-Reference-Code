"""Multi-armed bandits (Reference §28.1).

k arms, each with unknown mean reward mu_a.  At each step choose an arm and
receive stochastic reward.  Regret over T rounds:
    R_T = T * mu_star - sum_t r_t.

Classical algorithms:
  * epsilon-greedy: pick arg-max estimated mean w.p. 1-eps; explore uniformly w.p. eps.
  * UCB1 (Auer et al. 2002):
        pick argmax_a  mu_hat_a + sqrt(2 * ln(t) / n_a)
    Achieves O(log T) regret.
  * Thompson sampling (Thompson 1933):
        sample mu_a ~ Beta(alpha_a, beta_a); pick argmax; update posterior with
        observed reward.  Bayesian; optimal in many settings.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def eps_greedy(mus, T: int, eps: float = 0.1, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); k = len(mus)
    n = np.zeros(k); Q = np.zeros(k); rewards = []; regret = []
    mu_star = float(np.max(mus))
    for t in range(1, T + 1):
        a = rng.integers(k) if rng.uniform() < eps else int(Q.argmax())
        r = rng.binomial(1, mus[a])
        n[a] += 1; Q[a] += (r - Q[a]) / n[a]
        rewards.append(r); regret.append(mu_star - mus[a])
    return {"Q": Q, "n": n, "cum_reward": float(sum(rewards)),
            "cum_regret": float(sum(regret))}


def ucb1(mus, T: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); k = len(mus)
    n = np.zeros(k); Q = np.zeros(k); rewards = []; regret = []
    mu_star = float(np.max(mus))
    for t in range(1, T + 1):
        if t <= k:
            a = t - 1
        else:
            ucb = Q + np.sqrt(2 * np.log(t) / np.maximum(n, 1e-9))
            a = int(ucb.argmax())
        r = rng.binomial(1, mus[a])
        n[a] += 1; Q[a] += (r - Q[a]) / n[a]
        rewards.append(r); regret.append(mu_star - mus[a])
    return {"Q": Q, "n": n, "cum_reward": float(sum(rewards)),
            "cum_regret": float(sum(regret))}


def thompson_bernoulli(mus, T: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); k = len(mus)
    alpha = np.ones(k); beta = np.ones(k)                  # Beta(1, 1) prior
    rewards = []; regret = []
    mu_star = float(np.max(mus))
    for _ in range(T):
        samples = rng.beta(alpha, beta)
        a = int(samples.argmax())
        r = rng.binomial(1, mus[a])
        alpha[a] += r; beta[a] += 1 - r
        rewards.append(r); regret.append(mu_star - mus[a])
    return {"alpha": alpha, "beta": beta,
            "post_mean": alpha / (alpha + beta),
            "cum_reward": float(sum(rewards)),
            "cum_regret": float(sum(regret))}


if __name__ == "__main__":
    mus = np.array([0.20, 0.35, 0.55, 0.40, 0.30])         # 5-arm Bernoulli bandit
    T = 2000
    print(f"=== 5-arm Bernoulli bandit, T={T}, true mus = {mus.tolist()} ===")
    for name, fn in [("eps-greedy (eps=0.1)", lambda: eps_greedy(mus, T, eps=0.1)),
                     ("UCB1",                 lambda: ucb1(mus, T)),
                     ("Thompson (Beta)",       lambda: thompson_bernoulli(mus, T))]:
        r = fn()
        avg_reward = r["cum_reward"] / T
        arm_best = int(r["n"].argmax()) if "n" in r else int((r["alpha"] - 1).argmax())
        print(f"  {name:>22}: cum reward = {r['cum_reward']:.0f}, "
              f"cum regret = {r['cum_regret']:.1f}, "
              f"most-pulled arm = {arm_best} (true best = 2)")

    print("\n--- library cross-check (bandits, contextualbandits, or roll-your-own in gym) ---")
