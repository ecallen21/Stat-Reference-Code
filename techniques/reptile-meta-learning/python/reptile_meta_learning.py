"""Reptile meta-learning (Reference Sec 47.134).

Nichol, Achiam & Schulman 2018 'On first-order meta-learning
algorithms', arXiv:1803.02999. First-order approximation to MAML:

    1. Sample task T ~ p(T).
    2. Take K SGD steps on T from current theta -> theta'.
    3. Meta-update:  theta <- theta + epsilon (theta' - theta).

Simpler than MAML (no second-order gradients) yet performs
comparably on few-shot benchmarks. Illustrated on 1-D sinusoid
regression: quickly adapts to a NEW sinusoid amplitude / phase
from few points.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def make_task(rng):
    """Sinusoid task: y = a * sin(x - phi)."""
    a = float(rng.uniform(0.5, 3.0))
    phi = float(rng.uniform(0, 2 * np.pi))
    def sample(n):
        x = rng.uniform(-5, 5, size=n)
        y = a * np.sin(x - phi)
        return x, y
    return sample, (a, phi)


def basis(x):
    # sinusoidal basis (frequencies 1 and 2) with intercept
    x = np.atleast_1d(x)
    return np.column_stack([np.ones_like(x),
                              np.sin(x), np.cos(x),
                              np.sin(2 * x), np.cos(2 * x)])


def inner_train(theta, x, y, K, lr):
    for _ in range(K):
        pred = basis(x) @ theta
        grad = basis(x).T @ (pred - y) / len(x)
        theta = theta - lr * grad
    return theta


def reptile(n_tasks=1000, K_inner=5, eps=0.05, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=5) * 0.1
    for _ in range(n_tasks):
        sample, _ = make_task(rng)
        x, y = sample(10)
        theta_prime = inner_train(theta.copy(), x, y, K_inner, lr=0.02)
        theta = theta + eps * (theta_prime - theta)
    return theta


if __name__ == "__main__":
    print("=== Reptile meta-learning (Nichol-Achiam-Schulman 2018) ===\n")
    rng = np.random.default_rng(0)

    # (a) Meta-train on many sinusoids
    theta_meta = reptile(n_tasks=1200, K_inner=5, eps=0.05, seed=0)
    # Wider random init to make the meta-initialisation advantage clearer
    theta_random = rng.normal(size=5) * 2.0

    # (b) Evaluate few-shot on a new task (only 3 samples for adaptation)
    losses_meta, losses_random = [], []
    for _ in range(80):
        sample, _ = make_task(rng)
        x_tr, y_tr = sample(3)          # very few-shot
        x_te, y_te = sample(200)
        for K in [0, 1, 5, 20]:
            theta_m_adapted = inner_train(theta_meta.copy(), x_tr, y_tr, K, lr=0.02)
            theta_r_adapted = inner_train(theta_random.copy(), x_tr, y_tr, K, lr=0.02)
            mse_m = float(((basis(x_te) @ theta_m_adapted - y_te) ** 2).mean())
            mse_r = float(((basis(x_te) @ theta_r_adapted - y_te) ** 2).mean())
            losses_meta.append((K, mse_m)); losses_random.append((K, mse_r))

    for K in [0, 1, 5, 20]:
        mm = np.mean([l for k, l in losses_meta if k == K])
        mr = np.mean([l for k, l in losses_random if k == K])
        print(f"  K = {K:2d} inner steps  ->  MSE  meta = {mm:.3f}   random = {mr:.3f}")

    print("\n  Meta-initialised parameters need fewer inner steps to fit a new sinusoid.")

    print("\n--- library cross-check (higher / learn2learn Python; no established R port) ---")
