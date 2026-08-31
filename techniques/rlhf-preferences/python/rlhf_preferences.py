"""RLHF preference learning: Bradley-Terry reward model + DPO (Reference §28.10).

Pipeline:
  1. Collect PREFERENCE PAIRS (x, y_w > y_l) from a labeller.
  2. Fit a REWARD MODEL under Bradley-Terry (Christiano 2017):
        P(y_w > y_l | x) = sigma( r_theta(x, y_w) - r_theta(x, y_l) )
     -> maximum-likelihood on the preference dataset.
  3. Optimise a policy against r_theta:
     a) PPO with KL to a reference policy (InstructGPT recipe).
     b) DPO (Rafailov 2023): direct preference optimisation - skip the reward
        model and PPO; a closed-form loss on the log-ratio does the same job.

We demonstrate the reward-model MLE and the DPO loss in a toy 3-token setup.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def bt_reward_mle(pairs, dim: int = 3, lr: float = 0.1,
                   n_iter: int = 300, seed: int = 0) -> dict:
    """pairs: list of (feats_winner, feats_loser).  r_theta(x, y) = theta . feats(x, y)."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(scale=0.1, size=dim)
    for _ in range(n_iter):
        grad = np.zeros(dim); loss = 0.0
        for fw, fl in pairs:
            fw = np.asarray(fw, dtype=float); fl = np.asarray(fl, dtype=float)
            d = fw - fl
            p = _sigmoid(theta @ d)
            grad += (p - 1) * d
            loss += -np.log(p + 1e-12)
        theta -= lr * grad / len(pairs)
    return {"theta": theta, "loss": loss / len(pairs),
            "method": "Bradley-Terry reward-model MLE"}


def dpo_loss(theta, theta_ref, pairs, beta: float = 0.1) -> float:
    """DPO loss: -log sigma( beta * [(log pi(y_w) - log pi_ref(y_w))
                                   - (log pi(y_l) - log pi_ref(y_l))] ).
    In this linear surrogate, treat log pi(y) = theta . feats(y)."""
    loss = 0.0
    for fw, fl in pairs:
        lw = theta @ fw - theta_ref @ fw
        ll = theta @ fl - theta_ref @ fl
        loss += -np.log(_sigmoid(beta * (lw - ll)) + 1e-12)
    return loss / len(pairs)


def dpo_train(pairs, theta_ref, dim: int = 3, lr: float = 0.5, beta: float = 0.5,
              n_iter: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    theta = theta_ref.copy() + 0.01 * rng.normal(size=dim)
    for _ in range(n_iter):
        grad = np.zeros(dim)
        for fw, fl in pairs:
            lw = theta @ fw - theta_ref @ fw
            ll = theta @ fl - theta_ref @ fl
            p = _sigmoid(beta * (lw - ll))
            grad += -(1 - p) * beta * (fw - fl)
        theta -= lr * grad / len(pairs)
    return {"theta": theta, "final_loss": dpo_loss(theta, theta_ref, pairs, beta=beta),
            "method": "Direct Preference Optimisation (Rafailov 2023)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # toy "response features": 3 features per response.  True reward theta* = [1, -1, 0.5].
    theta_star = np.array([1.0, -1.0, 0.5])
    def _rand_feat(): return rng.normal(size=3)
    n_pairs = 200
    pairs = []
    for _ in range(n_pairs):
        f1 = _rand_feat(); f2 = _rand_feat()
        w, l = (f1, f2) if (theta_star @ f1) > (theta_star @ f2) else (f2, f1)
        # flip with 10% noise
        if rng.uniform() < 0.1:
            w, l = l, w
        pairs.append((w, l))

    rm = bt_reward_mle(pairs, dim=3, lr=0.3, n_iter=500)
    cos = float(rm["theta"] @ theta_star /
                 (np.linalg.norm(rm["theta"]) * np.linalg.norm(theta_star)))
    print(f"=== Bradley-Terry reward-model MLE (n_pairs={n_pairs}, 10% label noise) ===")
    print(f"  learned reward theta = {np.round(rm['theta'], 3).tolist()}")
    print(f"  true    reward theta = {theta_star.tolist()}")
    print(f"  cosine(learned, true) = {cos:+.3f}")

    # DPO from a random reference policy
    theta_ref = np.zeros(3)                                # uniform-ish log-policy
    dpo = dpo_train(pairs, theta_ref, dim=3, lr=0.3, beta=1.0, n_iter=500)
    cos_dpo = float(dpo["theta"] @ theta_star /
                     (np.linalg.norm(dpo["theta"]) * np.linalg.norm(theta_star)))
    print(f"\n=== DPO (skips explicit reward model + PPO) ===")
    print(f"  DPO theta       = {np.round(dpo['theta'], 3).tolist()}")
    print(f"  cosine(DPO, true) = {cos_dpo:+.3f}   (should match the true preferences too)")

    print("\n--- library cross-check (TRL PPOTrainer / DPOTrainer / GRPOTrainer; trlx) ---")
