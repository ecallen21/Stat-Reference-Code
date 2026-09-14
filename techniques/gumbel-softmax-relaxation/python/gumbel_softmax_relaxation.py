"""Gumbel-Softmax / Concrete (Ref Sec 47.315).

Jang, Gu & Poole 2017 ICLR; Maddison, Mnih & Teh 2017 ICLR.
Continuous relaxation of a CATEGORICAL sample:

    g_i ~ Gumbel(0, 1)  (i = 1..K)
    y_i = softmax((log alpha_i + g_i) / tau)

As tau -> 0, y approaches a one-hot argmax (still differentiable
via the relaxation). Enables end-to-end training through
discrete stochastic layers (VAE with categorical latents,
routing networks).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def gumbel_sample(shape, rng):
    U = rng.uniform(1e-9, 1 - 1e-9, shape)
    return -np.log(-np.log(U))


def gumbel_softmax(logits, tau=1.0, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    g = gumbel_sample(logits.shape, rng)
    y = np.exp((logits + g) / tau)
    return y / y.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    print("=== Gumbel-Softmax / Concrete (Jang 2017; Maddison 2017) ===\n")
    rng = np.random.default_rng(0)

    # Categorical with probs [0.1, 0.3, 0.6]
    p = np.array([0.1, 0.3, 0.6])
    logits = np.log(p)

    print(f"  True categorical probs: {p.tolist()}\n")
    n_samples = 20_000
    for tau in [5.0, 1.0, 0.5, 0.1]:
        samples = np.stack([gumbel_softmax(logits, tau=tau, rng=rng) for _ in range(n_samples)])
        empirical_arg = np.eye(3)[samples.argmax(-1)].mean(axis=0)
        avg_top = samples.max(axis=-1).mean()
        print(f"  tau = {tau:>4}   avg top-value = {avg_top:.3f}   "
              f"empirical categorical: {empirical_arg.round(3).tolist()}")

    print(f"\n  As tau -> 0, top-value -> 1 (one-hot behaviour).")
    print(f"  As tau -> INF, top-value -> 1/K (uniform mass) even though logits differ.")

    print(f"\n  Gradient path:")
    print(f"    dL/d logits = dL/dy * dy/d(logits + g) * 1/tau")
    print(f"  Gumbel noise g provides EXPLORATION; tau controls SHARPNESS + variance.")

    # Straight-through variant (STE): forward one-hot, backward soft
    y_soft = gumbel_softmax(logits, tau=0.5, rng=rng)
    y_hard = np.eye(3)[y_soft.argmax()]
    print(f"\n  Straight-through: forward = {y_hard}, backward gradient = d(y_soft)/d(logits).")
    print(f"  Bias but low variance; often the practical choice for discrete latents.")

    print("\n--- library cross-check (torch.nn.functional.gumbel_softmax; TFP Concrete) ---")
