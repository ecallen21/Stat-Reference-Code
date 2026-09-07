"""JEPA -- Joint Embedding Predictive Architecture (Reference Sec 47.23).

LeCun 2022 'A path towards autonomous machine intelligence'; Assran
et al. 2023 'I-JEPA: image joint-embedding predictive architecture'.
Self-supervised learning that predicts LATENT REPRESENTATIONS of a
target region from a context region, avoiding pixel-level
reconstruction (as in autoencoders / MAE) and generative denoising
(as in diffusion).

Setup:
    x  : full input (image / audio segment).
    x_ctx, x_tgt : disjoint context / target views (crops / masks).
    f_theta, f_zeta : student & (EMA) target encoders.
    g_phi : latent-space predictor.

Loss (I-JEPA form):
    L = || g_phi(f_theta(x_ctx))  -  f_zeta(x_tgt) ||_2^2

Prevents representation collapse via:
    * EMA target encoder (BYOL-style)
    * Predictor bottleneck
    * Multi-target multi-context masking

We simulate JEPA on a toy 1-D signal: encoder = linear projection,
predictor = MLP-lite (learned linear map). We show that with EMA the
loss decreases and the student learns to predict the target's
representation without collapsing to a constant.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def make_batch(rng, B, D):
    """Sample B toy 1-D signals of length D; context = first half, target = second half."""
    x = rng.normal(size=(B, D))
    ctx = x[:, :D // 2]
    tgt = x[:, D // 2:]
    return ctx, tgt


def train_jepa(D=20, d=8, epochs=400, lr=0.02, ema=0.99, batch=64, seed=0):
    rng = np.random.default_rng(seed)
    #  Student encoder theta (shape (D//2, d))
    theta = rng.normal(size=(D // 2, d)) * 0.3
    #  Target encoder zeta (EMA of theta), initialised as a copy of theta
    zeta = theta.copy()
    #  Predictor phi (d -> d)
    phi = np.eye(d) + 0.01 * rng.normal(size=(d, d))

    losses = []
    for ep in range(epochs):
        ctx, tgt = make_batch(rng, batch, D)
        z_ctx = ctx @ theta
        z_tgt = tgt @ zeta
        pred = z_ctx @ phi
        err = pred - z_tgt
        loss = float(np.mean(err ** 2))
        losses.append(loss)

        #  Gradients (student + predictor only; zeta is a stop-grad EMA)
        grad_phi = 2 * z_ctx.T @ err / batch
        grad_theta = 2 * ctx.T @ (err @ phi.T) / batch
        phi -= lr * grad_phi
        theta -= lr * grad_theta

        #  EMA update of target encoder
        zeta = ema * zeta + (1 - ema) * theta

    return theta, zeta, phi, losses


def rep_diversity(z):
    """Measure whether representation collapsed: rank / off-diagonal covariance mass."""
    z = z - z.mean(axis=0, keepdims=True)
    C = z.T @ z / len(z)
    eigs = np.linalg.eigvalsh(C)
    return {"tr": float(np.trace(C)), "max_eig": float(eigs[-1]), "min_eig": float(eigs[0])}


if __name__ == "__main__":
    print("=== JEPA (Joint-Embedding Predictive Architecture) -- toy demo ===\n")
    theta, zeta, phi, losses = train_jepa(D=20, d=8, epochs=400, ema=0.99)
    print(f"  Loss  epoch 1 = {losses[0]:.4f}")
    print(f"  Loss  epoch 100 = {losses[99]:.4f}")
    print(f"  Loss  epoch 400 = {losses[-1]:.4f}")

    rng = np.random.default_rng(1)
    ctx, tgt = make_batch(rng, 500, 20)
    z_ctx = ctx @ theta
    z_tgt = tgt @ zeta
    d1 = rep_diversity(z_ctx)
    d2 = rep_diversity(z_tgt)
    print(f"\n  Context repr eigenvalue spread: max/min = {d1['max_eig']:.3f} / {d1['min_eig']:.3f}")
    print(f"  Target  repr eigenvalue spread: max/min = {d2['max_eig']:.3f} / {d2['min_eig']:.3f}")
    print(f"  (Non-collapsed representations have min_eig > 0.)")

    print("\n--- library cross-check (I-JEPA / V-JEPA Meta AI Python; no R) ---")
