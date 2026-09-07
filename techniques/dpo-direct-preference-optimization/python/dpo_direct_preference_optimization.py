"""DPO -- Direct Preference Optimization (Reference Sec 47.25).

Rafailov et al. 2023 'Direct Preference Optimization: your language
model is secretly a reward model', NeurIPS. Trains a policy pi_theta
directly on PREFERENCE PAIRS without training a reward model or
running PPO.

Given prompt x with preferred completion y_w and dispreferred y_l,
DPO minimises:

    L = - E[ log sigma( beta * (log pi_theta(y_w|x) / pi_ref(y_w|x)
                              - log pi_theta(y_l|x) / pi_ref(y_l|x))) ]

Equivalent to RLHF's Bradley-Terry + KL-regularised optimum in closed
form. Advantages over PPO+RM:
    * No separate reward model.
    * No PPO / rollout / value estimation.
    * More stable, cheaper, simpler code.

We simulate DPO on a toy 'bandit' where actions are numeric and the
policy is a softmax over actions; preferred action = one with higher
true reward.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def softmax_logits(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def log_prob(action, logits):
    lp = logits - np.log(np.sum(np.exp(logits - logits.max()))) - logits.max()
    return lp[action]


def dpo_step(logits, ref_logits, y_w, y_l, beta=0.5, lr=0.05):
    """One DPO gradient step."""
    lp_w = log_prob(y_w, logits) - log_prob(y_w, ref_logits)
    lp_l = log_prob(y_l, logits) - log_prob(y_l, ref_logits)
    diff = beta * (lp_w - lp_l)
    #  Loss = -log sigmoid(diff)
    grad_diff = -1 / (1 + np.exp(diff))
    #  d lp / d logits[a] = delta(a=action) - softmax(a)
    n = len(logits)
    grad_lp_w = -softmax_logits(logits).copy()
    grad_lp_w[y_w] += 1
    grad_lp_l = -softmax_logits(logits).copy()
    grad_lp_l[y_l] += 1
    grad_logits = grad_diff * beta * (grad_lp_w - grad_lp_l)
    return logits - lr * grad_logits


def train_dpo(rewards, n_pairs=800, beta=0.5, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(rewards)
    logits = np.zeros(n)
    ref_logits = np.zeros(n)                 # uniform reference
    for _ in range(n_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        if rewards[i] > rewards[j]:
            y_w, y_l = i, j
        else:
            y_w, y_l = j, i
        logits = dpo_step(logits, ref_logits, y_w, y_l, beta=beta, lr=lr)
    return softmax_logits(logits)


if __name__ == "__main__":
    print("=== DPO -- direct preference optimisation (Rafailov 2023) ===\n")
    rewards = np.array([0.1, 0.3, 0.9, 0.5, 0.7])
    print(f"  True reward per action: {rewards.tolist()}")

    for beta in [0.1, 0.5, 2.0]:
        pi = train_dpo(rewards, n_pairs=800, beta=beta, seed=0)
        print(f"  beta = {beta}   final policy = {pi.round(3).tolist()}")

    print("\n  Higher beta pushes the policy harder toward argmax(rewards) (action 2);")
    print("  lower beta stays closer to the uniform reference policy.")

    print("\n--- library cross-check (HuggingFace trl.DPOTrainer / DPOConfig Python) ---")
