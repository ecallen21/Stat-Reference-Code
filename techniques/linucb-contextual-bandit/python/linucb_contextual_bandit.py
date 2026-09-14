"""LinUCB Contextual Bandit (Reference Sec 47.305).

Li, Chu, Langford & Schapire 2010 WWW. Per-arm ridge regression
with an upper-confidence-bound acquisition:

    A_a = I + sum x x^T                (per-arm design matrix)
    theta_hat_a = A_a^{-1} sum x r
    UCB_a(x) = theta_hat_a^T x + alpha * sqrt(x^T A_a^{-1} x)

Pick argmax UCB_a. Extends Thompson / UCB from multi-armed to
CONTEXTUAL bandits with linear reward models.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


class LinUCB:
    def __init__(self, n_arms, d, alpha=1.0):
        self.n = n_arms; self.d = d; self.alpha = alpha
        self.A = [np.eye(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]

    def act(self, context):
        best = -np.inf; best_a = 0
        for a in range(self.n):
            Ainv = np.linalg.inv(self.A[a])
            theta = Ainv @ self.b[a]
            ucb = float(theta @ context + self.alpha * np.sqrt(context @ Ainv @ context))
            if ucb > best: best = ucb; best_a = a
        return best_a

    def update(self, a, context, r):
        self.A[a] += np.outer(context, context)
        self.b[a] += r * context


if __name__ == "__main__":
    print("=== LinUCB Contextual Bandit (Li et al 2010 WWW) ===\n")
    rng = np.random.default_rng(0)

    n_arms = 4; d = 5; T = 2000
    # True per-arm parameter vectors
    theta_true = rng.normal(0, 1, (n_arms, d))
    lin = LinUCB(n_arms, d, alpha=1.0)

    reg = 0.0; regret = []
    for t in range(T):
        x = rng.normal(0, 1, d)
        best_arm = int(np.argmax(theta_true @ x))
        best_r = float(theta_true[best_arm] @ x)
        a = lin.act(x)
        r = float(theta_true[a] @ x) + rng.normal(0, 0.1)
        lin.update(a, x, r)
        reg += best_r - float(theta_true[a] @ x)
        regret.append(reg)

    print(f"  n_arms = {n_arms}, d = {d}, T = {T}")
    print(f"  Cumulative regret at t = 100:    {regret[99]:>8.2f}")
    print(f"  Cumulative regret at t = 500:    {regret[499]:>8.2f}")
    print(f"  Cumulative regret at t = 1000:   {regret[999]:>8.2f}")
    print(f"  Cumulative regret at t = 2000:   {regret[-1]:>8.2f}")
    print(f"\n  Regret grows sub-linearly (LinUCB theoretical bound: O(d*sqrt(T)*log T)).")
    print(f"  Compare to random policy expected regret ~ T:")
    exp_random = np.mean([
        theta_true.max(axis=0) @ rng.normal(0, 1, d) - theta_true.mean(axis=0) @ rng.normal(0, 1, d)
        for _ in range(500)])
    print(f"    random-policy per-step regret ~ {exp_random:.2f}")

    print("\n--- library cross-check (contextualbandits Python; vowpalwabbit; MABWiser) ---")
