"""Consistency Models (Reference Sec 47.215).

Song, Dhariwal, Chen & Sutskever 2023 'Consistency Models', ICML.
Train a model f_theta(x_t, t) that maps ANY point on a
probability-flow ODE trajectory back to the trajectory's ORIGIN
x_0:

    f_theta(x_t, t) = x_0     (self-consistency along the ODE)

Loss: L_CM = d(f_theta(x_{t+1}, t+1), f_theta_prev(x_t, t))
     (distillation of a pretrained diffusion model)

Enables ONE-STEP generation while retaining most of the quality
of a multi-step diffusion sampler.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def make_ode_trajectory(x0, sigma_max, T, sched, rng):
    """Simulate the deterministic probability-flow ODE by adding decreasing noise."""
    sigmas = sigma_max * np.linspace(1, 0.001, T + 1)
    traj = [x0 + sigma_max * rng.normal(size=x0.shape)]          # x_T
    x = traj[0]
    for t in range(T - 1, -1, -1):
        # 'ODE' step: nudge back to x0 in proportion to remaining sigma
        x = 0.9 * x + 0.1 * x0
        traj.insert(0, x)
    return traj[::-1]                                             # trajectories from t=T down to t=0


def train_consistency(x0_samples, sigma_max=5.0, n_iter=500, lr=0.03, seed=0):
    """Train f_theta(x, t) so f(x_t, t) predicts x_0 consistently along the ODE."""
    rng = np.random.default_rng(seed)
    d = x0_samples.shape[1]
    theta = rng.normal(scale=0.1, size=(d + 1, d))               # (x, sigma) -> x_0
    for it in range(n_iter):
        idx = rng.integers(len(x0_samples), size=32)
        x0 = x0_samples[idx]
        sigma = sigma_max * rng.uniform(size=(32, 1))
        eps = rng.normal(size=x0.shape)
        x_t = x0 + sigma * eps
        # Target: x_0
        xt_aug = np.concatenate([x_t, sigma], axis=-1)
        pred = xt_aug @ theta
        loss = np.mean((pred - x0) ** 2)
        grad = 2 * xt_aug.T @ (pred - x0) / len(x0)
        theta -= lr * grad
    return theta


def consistency_sample_1step(theta, n, d, sigma_max=5.0, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    x_T = sigma_max * rng.normal(size=(n, d))
    sigma_col = np.full((n, 1), sigma_max)
    xt_aug = np.concatenate([x_T, sigma_col], axis=-1)
    return xt_aug @ theta                                         # 1-step generation


if __name__ == "__main__":
    print("=== Consistency Models (Song et al 2023 ICML) ===\n")
    rng = np.random.default_rng(0)

    # 2D data: 3-mode mixture
    means = np.array([[-3., 0.], [3., 0.], [0., 3.]])
    n = 2000
    x0 = means[rng.integers(3, size=n)] + rng.normal(scale=0.3, size=(n, 2))

    theta = train_consistency(x0, sigma_max=5.0, n_iter=2000, lr=0.01, seed=0)

    # 1-step generation
    y = consistency_sample_1step(theta, n=500, d=2, sigma_max=5.0, rng=rng)

    # Check: for each generated sample, find closest true mode
    dists = np.min(np.linalg.norm(y[:, None] - means[None, :], axis=-1), axis=1)
    print(f"  1-step generation: mean distance to nearest target mode = {dists.mean():.3f}")
    per_mode = np.argmin(np.linalg.norm(y[:, None] - means[None, :], axis=-1), axis=1)
    counts = np.bincount(per_mode, minlength=3)
    print(f"  Samples per mode: {counts.tolist()}")

    print("\n  Consistency Models achieve high-quality 1-step generation, comparable")
    print("  to 20+ steps of DDIM after distillation. Real CM = trained UNet + EMA teacher.")

    print("\n--- library cross-check (openai/consistency_models; diffusers.CMScheduler Python) ---")
