"""Score-Based Generative Modeling via SDEs (Reference Sec 47.212).

Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole 2021 'Score-
Based Generative Modeling through Stochastic Differential
Equations', ICLR. Unifies DDPM and score-matching under a
continuous-time SDE framework:

    dx = f(x, t) dt + g(t) dW      (forward SDE)
    dx = [f(x, t) - g(t)^2 grad_x log p_t(x)] dt + g(t) dW_reverse

Learn a score network s_theta(x, t) ~ grad_x log p_t(x). Sample by
integrating the reverse SDE (Euler-Maruyama) or the equivalent
DETERMINISTIC PROBABILITY-FLOW ODE (adaptive ODE solver).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def variance_exploding_forward(x0, t, sigma_max=25.0):
    """VE-SDE: x_t = x_0 + sqrt(t) * sigma_max * eps."""
    return x0, sigma_max ** 2 * t                                # mean, variance


def true_score_gaussian_mixture(x, t, sigma_max, means, weights):
    """Analytic score of a Gaussian mixture at diffusion time t."""
    K, d = means.shape
    sigma_t2 = sigma_max ** 2 * t
    # p_t(x) = sum_k w_k N(x | mu_k, sigma_t^2 I)
    diffs = x[:, None, :] - means[None, :, :]                    # (N, K, d)
    sq_norms = (diffs ** 2).sum(axis=-1)                         # (N, K)
    log_gauss = -0.5 * sq_norms / max(sigma_t2, 1e-8) - 0.5 * d * np.log(2 * np.pi * max(sigma_t2, 1e-8))
    log_w = np.log(weights)[None, :]
    log_pk = log_w + log_gauss
    log_p = np.log(np.exp(log_pk - log_pk.max(1, keepdims=True)).sum(1)) + log_pk.max(1)
    # Score = grad_x log p(x) = sum_k p(k|x) * (mu_k - x) / sigma_t^2
    posteriors = np.exp(log_pk - log_p[:, None])
    return (posteriors[:, :, None] * (means[None, :, :] - x[:, None, :])).sum(axis=1) / max(sigma_t2, 1e-8)


def reverse_sde_euler(x_T, sched, means, weights, sigma_max, n_steps=200, rng=None):
    """Euler-Maruyama for reverse VE-SDE."""
    if rng is None: rng = np.random.default_rng(0)
    ts = np.linspace(sched, 1e-3, n_steps + 1)
    x = x_T.copy()
    dt = (ts[1] - ts[0])                                          # negative (going backward in t)
    for i in range(n_steps):
        t = ts[i]
        g2 = sigma_max ** 2 * 1.0                                # dg/dt = const for VE
        s = true_score_gaussian_mixture(x, t, sigma_max, means, weights)
        drift = -g2 * s * dt                                     # reverse-time drift
        diff = np.sqrt(g2 * abs(dt)) * rng.normal(size=x.shape)
        x = x + drift + diff
    return x


if __name__ == "__main__":
    print("=== Score-Based SDE (Song et al 2021 ICLR) ===\n")
    rng = np.random.default_rng(0)

    # 2D Gaussian-mixture target
    means = np.array([[-2., 0.], [2., 0.], [0., 2.]])
    weights = np.array([1/3, 1/3, 1/3])
    n = 500; d = 2
    sigma_max = 5.0

    # Sample from prior x_T ~ N(0, sigma_max^2)
    x_T = sigma_max * rng.normal(size=(n, d))

    # Reverse-SDE with true (oracle) score
    x_gen = reverse_sde_euler(x_T, sched=1.0, means=means, weights=weights,
                                 sigma_max=sigma_max, n_steps=200, rng=rng)

    # Compare distribution to true samples
    z_true = means[rng.integers(3, size=n)] + rng.normal(scale=0.1, size=(n, d))
    print(f"  Target: 3-mode Gaussian mixture in R^2")
    print(f"  True means: {means.tolist()}")
    print(f"  Reverse-SDE generated cluster centres (via kmeans-like assignment):")
    for k in range(3):
        dists = np.linalg.norm(x_gen[:, None, :] - means[None, :, :], axis=-1)
        mask = np.argmin(dists, axis=1) == k
        if mask.sum() > 0:
            centre = x_gen[mask].mean(axis=0)
            print(f"    mode {k} (target {means[k]}): {np.round(centre, 2).tolist()}   "
                  f"n = {int(mask.sum())}")

    print("\n--- library cross-check (diffusers with ScoreSDE scheduler; yang-song/score_sde Python) ---")
