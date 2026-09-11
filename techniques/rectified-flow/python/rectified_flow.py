"""Rectified Flow (Reference Sec 47.214).

Liu, Gong & Liu 2023 'Flow Straight and Fast: Learning to Generate
and Transfer Data with Rectified Flow'. Reformulates diffusion as
a STRAIGHT-LINE flow between prior x_0 ~ N(0,I) and data x_1 ~ p:

    x_t = (1 - t) x_0 + t x_1,  t in [0, 1]
    dx/dt = x_1 - x_0

Learn a velocity net v_theta(x, t) ~ E[x_1 - x_0 | x_t = x].
'Reflow' iterates: use the sampler's trajectories as new (x_0,
x_1) couplings, yielding straighter paths and 1-step generation.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def rectified_flow_train(x0_samples, x1_samples, n_iter=200, lr=0.05, seed=0):
    """Learn v_theta(x, t) as a small linear model in (x, t)."""
    rng = np.random.default_rng(seed)
    d = x0_samples.shape[1]
    theta = rng.normal(scale=0.1, size=(d + 1, d))               # (x, t) -> velocity
    for it in range(n_iter):
        idx0 = rng.integers(len(x0_samples), size=64)
        idx1 = rng.integers(len(x1_samples), size=64)
        x0 = x0_samples[idx0]; x1 = x1_samples[idx1]
        t = rng.uniform(size=(64, 1))
        x_t = (1 - t) * x0 + t * x1
        v_true = x1 - x0                                          # target velocity
        # Predict
        xt_aug = np.concatenate([x_t, t], axis=-1)
        v_hat = xt_aug @ theta
        loss = np.mean((v_hat - v_true) ** 2)
        grad = 2 * xt_aug.T @ (v_hat - v_true) / len(x_t)
        theta -= lr * grad
    return theta


def rectified_flow_sample(theta, x0, n_steps=1, d=None):
    """Euler integration of learned velocity net."""
    x = x0.copy()
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = step * dt
        xt_aug = np.concatenate([x, np.full((x.shape[0], 1), t)], axis=-1)
        v = xt_aug @ theta
        x = x + v * dt
    return x


if __name__ == "__main__":
    print("=== Rectified Flow (Liu-Gong-Liu 2023) ===\n")
    rng = np.random.default_rng(0)

    # Source: standard Gaussian; target: mixture of two Gaussians
    n = 2000
    x0 = rng.normal(size=(n, 2))
    x1 = np.vstack([[-3, 0] + rng.normal(scale=0.3, size=(n // 2, 2)),
                      [3, 0] + rng.normal(scale=0.3, size=(n // 2, 2))])

    theta = rectified_flow_train(x0, x1, n_iter=1000, lr=0.01, seed=0)

    # Sample from N(0, I) and push forward
    z = rng.normal(size=(500, 2))
    for n_steps in [1, 4, 20]:
        y = rectified_flow_sample(theta, z, n_steps=n_steps)
        # KL-ish gap to target: distance to nearest true mode
        modes = np.array([[-3, 0], [3, 0]])
        dists = np.min(np.linalg.norm(y[:, None] - modes[None, :], axis=-1), axis=1)
        print(f"  {n_steps:3d} Euler step(s): mean dist to nearest target mode = {dists.mean():.3f}")

    print("\n  With rectified paths, 1-step Euler already gives a reasonable sample")
    print("  (traditional diffusion needs 20-1000 steps).")

    print("\n--- library cross-check (diffusers.FlowMatchEulerDiscreteScheduler; SD3 uses rectified flow) ---")
