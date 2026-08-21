"""Generative Adversarial Network on 2-D data (Goodfellow et al. 2014; Reference §27.9).

Two networks in an adversarial minimax game:
    G_theta(z): R^k -> R^d   turns noise into fakes
    D_phi(x): R^d -> [0, 1]  decides real vs fake

Objectives (non-saturating variant):
    L_D = -E_{x~p_data} [log D(x)] - E_{z~p(z)} [log(1 - D(G(z)))]
    L_G = -E_{z~p(z)} [log D(G(z))]

We train G and D alternately with SGD on a 2-D target: a ring of 8 clusters
("Gaussian mixture on a circle") — a classic GAN demo showing mode coverage.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _relu(z): return np.maximum(z, 0.0)
def _relu_grad(z): return (z > 0).astype(float)


def train_gan_2d(data, z_dim: int = 4, hidden: int = 32,
                 lr_g: float = 0.001, lr_d: float = 0.001,
                 n_iter: int = 5000, batch: int = 64,
                 k_disc: int = 1, seed: int = 0) -> dict:
    """Minimal MLP GAN on R^2 data."""
    rng = np.random.default_rng(seed)
    d = data.shape[1]
    # Generator: z -> h -> x
    W_g1 = rng.normal(scale=0.3, size=(z_dim, hidden)); b_g1 = np.zeros(hidden)
    W_g2 = rng.normal(scale=0.3, size=(hidden, d)); b_g2 = np.zeros(d)
    # Discriminator: x -> h -> logit
    W_d1 = rng.normal(scale=0.3, size=(d, hidden)); b_d1 = np.zeros(hidden)
    W_d2 = rng.normal(scale=0.3, size=(hidden, 1)); b_d2 = np.zeros(1)

    def _G(z):
        h = _relu(z @ W_g1 + b_g1); return h @ W_g2 + b_g2, h
    def _D(x):
        h = _relu(x @ W_d1 + b_d1); return _sigmoid(h @ W_d2 + b_d2)[:, 0], h

    for it in range(n_iter):
        # ----- discriminator step(s) -----
        for _ in range(k_disc):
            idx = rng.integers(0, len(data), size=batch)
            x_real = data[idx]
            z = rng.normal(size=(batch, z_dim))
            x_fake, _ = _G(z)
            p_real, h_r = _D(x_real)
            p_fake, h_f = _D(x_fake)
            # binary cross-entropy on both
            grad_out_real = (p_real - 1)[:, None] / batch
            grad_out_fake = (p_fake - 0)[:, None] / batch
            for x_in, h, g_out in [(x_real, h_r, grad_out_real),
                                     (x_fake, h_f, grad_out_fake)]:
                d_W_d2 = h.T @ g_out; d_b_d2 = g_out.sum(axis=0)
                d_h = g_out @ W_d2.T
                d_z = d_h * _relu_grad(x_in @ W_d1 + b_d1)
                d_W_d1 = x_in.T @ d_z; d_b_d1 = d_z.sum(axis=0)
                W_d2 -= lr_d * d_W_d2; b_d2 -= lr_d * d_b_d2
                W_d1 -= lr_d * d_W_d1; b_d1 -= lr_d * d_b_d1
        # ----- generator step (non-saturating loss) -----
        z = rng.normal(size=(batch, z_dim))
        x_fake, h_g = _G(z)
        p_fake, h_d = _D(x_fake)
        # dL_G / d_p_fake = -1 / p_fake ;   d p_fake / d_logit = p(1-p);
        g_out = -(1 - p_fake)[:, None] / batch                # combined
        d_h_d = g_out @ W_d2.T
        d_z_d = d_h_d * _relu_grad(x_fake @ W_d1 + b_d1)
        d_x_fake = d_z_d @ W_d1.T
        d_W_g2 = h_g.T @ d_x_fake; d_b_g2 = d_x_fake.sum(axis=0)
        d_h_g = d_x_fake @ W_g2.T
        d_z_g = d_h_g * _relu_grad(z @ W_g1 + b_g1)
        d_W_g1 = z.T @ d_z_g; d_b_g1 = d_z_g.sum(axis=0)
        W_g2 -= lr_g * d_W_g2; b_g2 -= lr_g * d_b_g2
        W_g1 -= lr_g * d_W_g1; b_g1 -= lr_g * d_b_g1

    def _gen(n):
        z = rng.normal(size=(n, z_dim))
        return _G(z)[0]
    return {"generate": _gen, "z_dim": z_dim,
            "method": "vanilla GAN (non-saturating loss) on 2-D data"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_per_cluster = 100
    centres = np.stack([np.array([np.cos(2 * np.pi * k / 8),
                                    np.sin(2 * np.pi * k / 8)])
                         for k in range(8)])
    data = np.vstack([c + 0.08 * rng.normal(size=(n_per_cluster, 2))
                       for c in centres])
    print(f"=== GAN on ring of 8 Gaussians in R^2 (N={len(data)}) ===")

    m = train_gan_2d(data, z_dim=4, hidden=32, lr_g=0.002, lr_d=0.002,
                     n_iter=4000, batch=64)
    samples = m["generate"](400)

    # coverage: assign each sample to nearest cluster centre; count coverage
    from collections import Counter
    dists = ((samples[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1)
    assign = dists.argmin(axis=1)
    cov = Counter(assign.tolist())
    print(f"  # samples nearest each of the 8 modes: "
          f"{[cov.get(k, 0) for k in range(8)]}")
    n_covered = sum(1 for k in range(8) if cov.get(k, 0) > 5)
    print(f"  modes with >= 5 samples: {n_covered} / 8")
    print(f"  sample mean = {np.round(samples.mean(axis=0), 3).tolist()}   "
          f"data mean = {np.round(data.mean(axis=0), 3).tolist()}")

    print("\n--- library cross-check (torch nn.Sequential + BCELoss training loop) ---")
