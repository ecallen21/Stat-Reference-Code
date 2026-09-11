"""DDIM - Denoising Diffusion Implicit Models (Sec 47.209).

Song, Meng & Ermon 2021 'Denoising Diffusion Implicit Models',
ICLR. Reformulates the DDPM reverse process as a NON-MARKOVIAN,
DETERMINISTIC ODE that can be integrated with FEWER STEPS:

    x_{t-1} = sqrt(alpha_{t-1}) * predicted_x0(x_t, eps_hat)
                + sqrt(1 - alpha_{t-1}) * eps_hat

Same trained score model as DDPM, but 10-50 steps instead of
1000. Deterministic sampling enables meaningful latent-space
interpolation.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def make_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)
    return {"betas": betas, "alphas": alphas, "alpha_bar": alpha_bar}


def forward_diffuse(x0, alpha_bar_t, rng):
    """Add noise: x_t = sqrt(alpha_bar) x_0 + sqrt(1 - alpha_bar) eps."""
    eps = rng.normal(size=x0.shape)
    return np.sqrt(alpha_bar_t) * x0 + np.sqrt(1 - alpha_bar_t) * eps, eps


def ddpm_reverse(x_t, eps_hat, sched, t, rng):
    """Standard DDPM reverse step (adds stochastic noise)."""
    alpha_t = sched["alphas"][t]
    alpha_bar_t = sched["alpha_bar"][t]
    coef = (1 - alpha_t) / np.sqrt(1 - alpha_bar_t)
    mean = (x_t - coef * eps_hat) / np.sqrt(alpha_t)
    if t > 0:
        var = sched["betas"][t]
        return mean + np.sqrt(var) * rng.normal(size=x_t.shape)
    return mean


def ddim_reverse(x_t, eps_hat, sched, t, t_prev, eta=0.0):
    """DDIM step: t -> t_prev (deterministic when eta=0)."""
    alpha_bar_t = sched["alpha_bar"][t]
    alpha_bar_prev = sched["alpha_bar"][t_prev] if t_prev >= 0 else 1.0
    x0_hat = (x_t - np.sqrt(1 - alpha_bar_t) * eps_hat) / np.sqrt(alpha_bar_t)
    dir_xt = np.sqrt(1 - alpha_bar_prev) * eps_hat
    return np.sqrt(alpha_bar_prev) * x0_hat + dir_xt


if __name__ == "__main__":
    print("=== DDIM (Song-Meng-Ermon 2021) ===\n")
    rng = np.random.default_rng(0)

    # 2D Gaussian mixture toy target
    means = np.array([[-2, 0], [2, 0]])
    def sample_target(n): return means[rng.integers(2, size=n)] + rng.normal(scale=0.3, size=(n, 2))

    # 'Trained' score: perfect noise-prediction given knowledge of forward process
    T = 100
    sched = make_schedule(T)

    def eps_hat(x_t, t, x0):
        """Oracle: pretend we know x_0 (in real DDPM this is a neural net)."""
        alpha_bar_t = sched["alpha_bar"][t]
        return (x_t - np.sqrt(alpha_bar_t) * x0) / np.sqrt(1 - alpha_bar_t)

    n = 500
    x0_true = sample_target(n)
    # Forward diffuse to noise
    x_T, _ = forward_diffuse(x0_true, sched["alpha_bar"][T - 1], rng)

    # DDPM: reverse 100 steps
    x = x_T.copy()
    for t in range(T - 1, -1, -1):
        e = eps_hat(x, t, x0_true)
        x = ddpm_reverse(x, e, sched, t, rng)
    ddpm_out = x.copy()

    # DDIM: reverse only 10 steps (skip 10x)
    steps = list(range(T - 1, -1, -10))
    x = x_T.copy()
    for i, t in enumerate(steps):
        t_prev = steps[i + 1] if i + 1 < len(steps) else -1
        e = eps_hat(x, t, x0_true)
        x = ddim_reverse(x, e, sched, t, t_prev, eta=0.0)
    ddim_out = x.copy()

    # Compare
    err_ddpm = float(np.mean(np.linalg.norm(ddpm_out - x0_true, axis=1)))
    err_ddim = float(np.mean(np.linalg.norm(ddim_out - x0_true, axis=1)))
    print(f"  Target: 2-mode Gaussian mixture in R^2, n = {n}")
    print(f"  DDPM ({T} steps)   avg recon dist = {err_ddpm:.4f}")
    print(f"  DDIM ({len(steps)} steps) avg recon dist = {err_ddim:.4f}   (10x fewer steps)")

    print("\n  With a real trained score network, DDIM's 20-50 steps typically match")
    print("  DDPM's 1000 steps in FID; 10 steps is often noticeably worse.")

    print("\n--- library cross-check (diffusers.DDIMScheduler / diffusers.DDPMScheduler Python) ---")
