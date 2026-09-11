"""ORPO - Odds Ratio Preference Optimization (Sec 47.198).

Hong, Lee & Thorne 2024 'ORPO: Monolithic Preference Optimization
without Reference Model', EMNLP. Combines SFT and preference
learning in ONE loss with NO reference model:

    L_ORPO = L_SFT(chosen) + lambda * L_OR
    L_OR   = -log(sigmoid(log odds(chosen) - log odds(rejected)))
    odds(y|x) = P(y|x) / (1 - P(y|x))

Since it doesn't need pi_ref, ORPO halves memory vs DPO and can
train from scratch (SFT + preference simultaneously).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def log_odds(p, eps=1e-8):
    return np.log((p + eps) / (1 - p + eps))


def orpo_loss(logp_chosen, logp_rejected, lam=0.1):
    """ORPO combined SFT + OR loss for one (chosen, rejected) pair."""
    # SFT: maximise logp_chosen (equivalent to minimising negative logprob)
    l_sft = -logp_chosen
    # OR: log(sigmoid(log-odds ratio))
    p_c = np.exp(logp_chosen); p_r = np.exp(logp_rejected)
    lo_diff = log_odds(p_c) - log_odds(p_r)
    l_or = -np.log(1 / (1 + np.exp(-lo_diff)) + 1e-12)
    return l_sft + lam * l_or, l_sft, l_or


def train_orpo(pairs, lr=0.1, lam=0.1, n_iter=500, seed=0):
    """Scalar toy: pi(y_id) = softmax(theta)[y_id]."""
    rng = np.random.default_rng(seed)
    Y = int(max(max(c, r) for c, r in pairs)) + 1
    theta = rng.normal(scale=0.1, size=Y)
    hist = []
    for it in range(n_iter):
        z = theta - theta.max()
        p = np.exp(z); p = p / p.sum()
        total_loss = 0
        grad = np.zeros(Y)
        for c, r in pairs:
            logp_c = np.log(p[c] + 1e-12)
            logp_r = np.log(p[r] + 1e-12)
            L, _, _ = orpo_loss(logp_c, logp_r, lam=lam)
            total_loss += L
            # Finite-diff gradient (small demo)
            for k in range(Y):
                theta_p = theta.copy(); theta_p[k] += 1e-4
                zp = theta_p - theta_p.max()
                pp = np.exp(zp); pp /= pp.sum()
                Lp, _, _ = orpo_loss(np.log(pp[c] + 1e-12), np.log(pp[r] + 1e-12), lam=lam)
                grad[k] += (Lp - L) / 1e-4
        theta -= lr * grad / len(pairs)
        hist.append(total_loss / len(pairs))
    return theta, hist


if __name__ == "__main__":
    print("=== ORPO - Odds Ratio Preference Optimization (Hong et al 2024) ===\n")
    rng = np.random.default_rng(0)

    Y = 5
    # Preference pairs: id of chosen, id of rejected
    pairs = [(2, 0), (2, 1), (2, 3), (2, 4), (2, 0), (2, 1)]     # response 2 always chosen
    theta, hist = train_orpo(pairs, lr=0.5, lam=1.0, n_iter=150, seed=0)
    z = theta - theta.max(); p = np.exp(z); p = p / p.sum()

    print(f"  {len(pairs)} preference pairs, Y = {Y} response candidates")
    print(f"  Learned policy: {np.round(p, 3)}")
    print(f"  Chosen (response 2) probability = {p[2]:.3f}")
    print(f"  Loss trace: init = {hist[0]:.3f}, final = {hist[-1]:.3f}")
    print("\n  ORPO combines SFT (push up chosen) + OR (push down rejected)")
    print("  in one loss with NO reference model. Halves memory vs DPO.")

    print("\n--- library cross-check (trl.ORPOTrainer / axolotl ORPO config Python) ---")
