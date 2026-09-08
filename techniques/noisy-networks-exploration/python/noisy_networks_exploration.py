"""Noisy Networks for Exploration (Reference Sec 47.146).

Fortunato et al 2018 'Noisy Networks for Exploration', ICLR.
Replaces ordinary linear layer  y = Wx + b  with a factorised
Gaussian noisy layer:

    W = mu_W + sigma_W * eps_W,     b = mu_b + sigma_b * eps_b

Weights (mu, sigma) are learned; eps are resampled every forward
pass. Provides state-dependent exploration replacing epsilon-greedy;
the noise gets amortised away automatically as sigmas shrink.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


class NoisyLinear:
    """Factorised-Gaussian noisy linear layer y = (mu_W + sig_W eps_out eps_in^T) x + (mu_b + sig_b eps_out)."""

    def __init__(self, in_dim, out_dim, sigma_init=0.5, seed=0):
        self.in_dim, self.out_dim = in_dim, out_dim
        self.rng = np.random.default_rng(seed)
        mu_range = 1 / np.sqrt(in_dim)
        self.mu_W = self.rng.uniform(-mu_range, mu_range, size=(out_dim, in_dim))
        self.sig_W = np.full((out_dim, in_dim), sigma_init / np.sqrt(in_dim))
        self.mu_b = self.rng.uniform(-mu_range, mu_range, size=out_dim)
        self.sig_b = np.full(out_dim, sigma_init / np.sqrt(out_dim))

    def _f(self, x):
        return np.sign(x) * np.sqrt(np.abs(x))

    def forward(self, x, noisy=True):
        if noisy:
            e_in = self._f(self.rng.normal(size=self.in_dim))
            e_out = self._f(self.rng.normal(size=self.out_dim))
            W = self.mu_W + self.sig_W * np.outer(e_out, e_in)
            b = self.mu_b + self.sig_b * e_out
        else:
            W, b = self.mu_W, self.mu_b
        return W @ x + b


if __name__ == "__main__":
    print("=== Noisy Networks for Exploration (Fortunato et al 2018) ===\n")

    # Contextual bandit: 4 arms; only arm 2 gives high reward for context sign>0
    rng = np.random.default_rng(0)
    n_arms = 4

    def reward(ctx, arm):
        if arm == 2 and ctx > 0: return rng.normal(1.0, 0.1)
        if arm == 0 and ctx <= 0: return rng.normal(1.0, 0.1)
        return rng.normal(0.0, 0.1)

    # Train a noisy Q-network: 1-D ctx -> 4 arms
    net = NoisyLinear(1, n_arms, sigma_init=1.0, seed=0)
    lr = 0.05
    hist = []
    n_steps = 20000
    for step in range(n_steps):
        ctx = float(rng.normal())
        x = np.array([ctx])
        q = net.forward(x, noisy=True)                         # noisy action selection
        a = int(np.argmax(q))
        r = float(reward(ctx, a))
        # Gradient step on Q-value MSE (only for chosen arm)
        e_in = net._f(net.rng.normal(size=1))                  # fresh noise for gradient
        e_out = net._f(net.rng.normal(size=n_arms))
        q_pred_a = (net.mu_W[a] + net.sig_W[a] * e_out[a] * e_in) @ x \
                     + net.mu_b[a] + net.sig_b[a] * e_out[a]
        td_a = float(q_pred_a - r)
        net.mu_W[a] -= lr * td_a * x
        net.mu_b[a] -= lr * td_a
        net.sig_W[a] -= lr * td_a * e_out[a] * e_in * x
        net.sig_b[a] -= lr * td_a * e_out[a]
        # Anneal sigma toward zero
        net.sig_W *= 0.9998
        net.sig_b *= 0.9998
        hist.append(r)

    print(f"  Mean reward (first 500 steps):  {np.mean(hist[:500]):.3f}")
    print(f"  Mean reward (last 500 steps):   {np.mean(hist[-500:]):.3f}")
    print(f"  Final sigma_W avg magnitude:    {np.mean(np.abs(net.sig_W)):.4f}")
    print(f"  Final sigma_b avg magnitude:    {np.mean(np.abs(net.sig_b)):.4f}")

    # Greedy evaluation
    correct = 0
    for _ in range(1000):
        c = rng.normal()
        a = int(np.argmax(net.forward(np.array([c]), noisy=False)))
        if (c > 0 and a == 2) or (c <= 0 and a == 0):
            correct += 1
    print(f"\n  Greedy accuracy after training: {correct / 1000:.3f}   (chance = 0.25)")

    print("\n--- library cross-check (cleanrl / stable-baselines3 Python) ---")
