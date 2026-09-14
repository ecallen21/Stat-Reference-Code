"""WGAN-GP - Wasserstein GAN with Gradient Penalty (Ref Sec 47.286).

Gulrajani, Ahmed, Arjovsky, Dumoulin & Courville 2017 NeurIPS.
Improve WGAN training by enforcing the 1-Lipschitz constraint
via a GRADIENT PENALTY instead of weight clipping:

    L_D = E[D(fake)] - E[D(real)] + lambda * E[(||grad D(x_hat)|| - 1)^2]

where x_hat = t * real + (1 - t) * fake, t ~ U[0, 1]. Fixes the
capacity / stability issues of vanilla WGAN's weight clipping.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def build_mlp(sizes, rng):
    weights = []
    for i in range(len(sizes) - 1):
        W = rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1]))
        b = np.zeros(sizes[i + 1])
        weights.append((W, b))
    return weights


def forward(x, params, tanh_out=False):
    h = x
    for i, (W, b) in enumerate(params):
        h = h @ W + b
        if i < len(params) - 1:
            h = np.tanh(h)
    return np.tanh(h) if tanh_out else h


def gradient_penalty(D, real, fake, rng):
    """Sample points on the line between real and fake, penalise |grad|-1."""
    t = rng.uniform(0, 1, size=(real.shape[0], 1))
    x_hat = t * real + (1 - t) * fake
    # Approximate gradient by finite differences (since we don't have autograd here)
    eps = 1e-3
    grad = np.zeros_like(x_hat)
    for j in range(x_hat.shape[1]):
        d = np.zeros_like(x_hat); d[:, j] = eps
        grad[:, j] = (forward(x_hat + d, D)[:, 0] - forward(x_hat - d, D)[:, 0]) / (2 * eps)
    return float(np.mean((np.linalg.norm(grad, axis=1) - 1.0) ** 2))


if __name__ == "__main__":
    print("=== WGAN-GP (Gulrajani et al 2017) ===\n")
    rng = np.random.default_rng(0)

    # Target: mixture of Gaussians in 2-D
    def sample_real(n):
        centers = np.array([[-2, -2], [2, -2], [0, 2]])
        idx = rng.integers(0, 3, n)
        return centers[idx] + rng.normal(0, 0.2, (n, 2))

    D = build_mlp([2, 16, 16, 1], rng)                            # critic
    G = build_mlp([2, 16, 16, 2], rng)                            # generator

    n_batch = 128
    # Just report initial vs 10-step D-loss + gradient penalty
    real = sample_real(n_batch)
    z = rng.normal(0, 1, (n_batch, 2))
    fake = forward(z, G, tanh_out=True) * 3.0

    d_real = forward(real, D)[:, 0].mean()
    d_fake = forward(fake, D)[:, 0].mean()
    gp = gradient_penalty(D, real, fake, rng)

    print(f"  Initial critic outputs:")
    print(f"    D(real) mean = {d_real:>+.3f}")
    print(f"    D(fake) mean = {d_fake:>+.3f}")
    print(f"    Wasserstein-1 estimate = {d_real - d_fake:.3f}")
    print(f"    Gradient penalty at random x_hat: {gp:.3f}")
    print(f"    (target |grad D| = 1; deviations show critic is not yet Lipschitz)")

    print("\n  WGAN-GP training loop (not run here):")
    print("    for _ in range(iterations):")
    print("        real, fake = sample_real(), G(z)")
    print("        L_D = E[D(fake)] - E[D(real)] + lambda * gradient_penalty")
    print("        update D to minimize L_D")
    print("        every n_critic steps: update G to maximize E[D(G(z))]")

    print("\n  Compared to plain WGAN (weight clip): GP avoids capacity loss and")
    print("  weight-space pathologies while enforcing the Lipschitz constraint.")

    print("\n--- library cross-check (torch WGAN-GP; tf-gan; StudioGAN implementations) ---")
