"""KTO - Kahneman-Tversky Optimization (Reference Sec 47.197).

Ethayarajh, Xu, Muennighoff, Jurafsky & Kiela 2024 'KTO: Model
Alignment as Prospect Theoretic Optimization'. Unlike DPO which
needs PAIRED (chosen, rejected) preferences, KTO uses UNPAIRED
labels: {desirable} vs {undesirable} rollouts only.

    L_KTO(x, y, label) = 1 - v(beta * (log pi(y|x) - log pi_ref(y|x)))
    where v is a Kahneman-Tversky value function (concave for gains,
    convex for losses, with LOSS AVERSION).

Requires ~10x less data than DPO because unpaired labels are much
easier to collect (thumbs-up / thumbs-down vs "which is better?").
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def kto_value_function(x, lambda_D=1.0, lambda_U=1.0, alpha=0.5):
    """Kahneman-Tversky value: concave gains, convex losses (with lambda_U
    for loss aversion). Simplified: v(x) = 1 - sigmoid(x)."""
    return 1.0 / (1 + np.exp(-x))


def kto_loss(logp, ref_logp, label, beta=0.1, lambda_D=1.0, lambda_U=1.0):
    """KTO loss for a single (x, y, +/-) example."""
    logratio = logp - ref_logp
    kl = max(0, np.mean(logratio))                             # E[KL] over the batch (est)
    if label == 1:                                              # desirable
        return lambda_D * (1 - kto_value_function(beta * (logratio - kl)))
    else:
        return lambda_U * (1 - kto_value_function(beta * (kl - logratio)))


def train_kto(examples, ref_logp, lr=0.1, n_iter=500, beta=0.1, seed=0):
    """Toy scalar-logp model: pi(y|x) = sigmoid(theta[y]). Train KTO loss."""
    rng = np.random.default_rng(seed)
    Y = int(max(y for _, y, _ in examples)) + 1
    theta = rng.normal(scale=0.1, size=Y)
    for it in range(n_iter):
        losses = 0
        grads = np.zeros(Y)
        # Compute mean-log-ratio (proxy for KL)
        logps = [np.log(1 / (1 + np.exp(-theta[y])) + 1e-12) for _, y, _ in examples]
        logratios = [lp - ref_logp[y] for lp, (_, y, _) in zip(logps, examples)]
        kl = max(0, np.mean(logratios))
        for i, (x, y, label) in enumerate(examples):
            logp = logps[i]
            logratio = logratios[i]
            if label == 1:
                z = beta * (logratio - kl)
                v = 1 / (1 + np.exp(-z))
                dl_dlogp = -beta * v * (1 - v)                   # d(1 - sigmoid(z))/dlogp
            else:
                z = beta * (kl - logratio)
                v = 1 / (1 + np.exp(-z))
                dl_dlogp = beta * v * (1 - v)
            # dlogp / dtheta[y] = 1 - sigmoid(theta[y])
            grads[y] += dl_dlogp * (1 - 1 / (1 + np.exp(-theta[y])))
            losses += kto_loss(logp, ref_logp[y], label, beta=beta)
        theta -= lr * grads / len(examples)
    return theta, losses / len(examples)


if __name__ == "__main__":
    print("=== KTO - Kahneman-Tversky Optimization (Ethayarajh et al 2024) ===\n")
    rng = np.random.default_rng(0)

    # 5 candidate responses; some liked (label=1), some disliked (label=0)
    # Only UNPAIRED labels — no need to pair them.
    Y = 5
    ref_logp = np.log([0.15, 0.20, 0.15, 0.25, 0.25])           # reference prob distr
    examples = [
        ("q", 0, 0),  # disliked
        ("q", 1, 1),  # liked
        ("q", 2, 0),  # disliked
        ("q", 3, 1),  # liked
        ("q", 4, 1),  # liked
        ("q", 0, 0),  # disliked (weighted)
        ("q", 1, 1),  # liked
    ]

    theta, final_loss = train_kto(examples, ref_logp, lr=0.5, n_iter=300, beta=0.5, seed=0)
    pi = 1 / (1 + np.exp(-theta))
    pi = pi / pi.sum()
    ref = np.exp(ref_logp); ref = ref / ref.sum()
    print(f"  Reference policy pi_ref: {np.round(ref, 3)}")
    print(f"  KTO-trained policy pi:   {np.round(pi, 3)}")
    liked_ids = sorted({y for _, y, l in examples if l == 1})
    disliked_ids = sorted({y for _, y, l in examples if l == 0})
    print(f"  Liked responses    ({liked_ids}) prob mass: "
          f"ref = {ref[liked_ids].sum():.3f}  KTO = {pi[liked_ids].sum():.3f}")
    print(f"  Disliked responses ({disliked_ids}) prob mass: "
          f"ref = {ref[disliked_ids].sum():.3f}  KTO = {pi[disliked_ids].sum():.3f}")

    print(f"\n  Final KTO loss: {final_loss:.4f}")
    print("  KTO uses unpaired labels; DPO needs (chosen, rejected) pairs.")

    print("\n--- library cross-check (trl.KTOTrainer / trl KTOConfig Python) ---")
