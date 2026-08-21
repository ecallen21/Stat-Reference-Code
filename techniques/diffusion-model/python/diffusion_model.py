"""Denoising Diffusion Probabilistic Model — DDPM (Ho et al. 2020; Reference §27.x extra).

Forward process (fixed): gradually add Gaussian noise across T steps to data x_0:
    x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps,   eps ~ N(0, I)
where alpha_bar_t = prod_{s <= t} (1 - beta_s).

Reverse process (learned): a neural denoiser predicts eps from (x_t, t):
    eps_theta(x_t, t) ~ eps
Loss (simple):  L = E_{t, x_0, eps} || eps - eps_theta(x_t, t) ||^2.

Sampling:
    x_{t-1} = (1 / sqrt(alpha_t)) [ x_t - (1 - alpha_t) / sqrt(1 - alpha_bar_t) eps_theta(x_t, t) ]
             + sigma_t z,   z ~ N(0, I)

We use a tiny MLP eps_theta on 2-D data (two moons) — the whole trick fits
in numpy and demonstrates the DDPM machinery.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math

import numpy as np    # numerical arrays + linear algebra


def _relu(z): return np.maximum(z, 0.0)
def _relu_grad(z): return (z > 0).astype(float)


def make_moons(n=400, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n1 = n // 2; n2 = n - n1
    th = rng.uniform(0, np.pi, n1)
    outer = np.column_stack([np.cos(th), np.sin(th)])
    th2 = rng.uniform(0, np.pi, n2)
    inner = np.column_stack([1 - np.cos(th2), 0.5 - np.sin(th2)])
    X = np.vstack([outer, inner]) + noise * rng.normal(size=(n, 2))
    return X


def train_ddpm(X, T: int = 40, hidden: int = 64,
               lr: float = 0.02, epochs: int = 500, seed: int = 0) -> dict:
    """Tiny denoiser: eps_theta(x_t, t) is a 2-hidden-layer MLP with a scalar
    time input concatenated."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float); N, d = X.shape
    betas = np.linspace(1e-4, 0.05, T)
    alphas = 1 - betas; alpha_bar = np.cumprod(alphas)

    d_in = d + 1                                          # append t
    W1 = rng.normal(scale=np.sqrt(2.0 / d_in), size=(d_in, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(scale=np.sqrt(2.0 / hidden), size=(hidden, hidden))
    b2 = np.zeros(hidden)
    W3 = rng.normal(scale=np.sqrt(2.0 / hidden), size=(hidden, d))
    b3 = np.zeros(d)

    losses = []
    for ep in range(epochs):
        # sample t uniformly for each example
        t = rng.integers(0, T, size=N)
        a_bar = alpha_bar[t][:, None]
        eps = rng.normal(size=X.shape)
        x_t = np.sqrt(a_bar) * X + np.sqrt(1 - a_bar) * eps
        inp = np.column_stack([x_t, t / T])
        # forward
        h1 = _relu(inp @ W1 + b1)
        h2 = _relu(h1 @ W2 + b2)
        eps_hat = h2 @ W3 + b3
        # loss
        loss = float(((eps - eps_hat) ** 2).mean())
        losses.append(loss)
        # backward
        d_eps = 2 * (eps_hat - eps) / (N * d)
        dW3 = h2.T @ d_eps; db3 = d_eps.sum(axis=0)
        dh2 = d_eps @ W3.T
        dh2_pre = dh2 * _relu_grad(h1 @ W2 + b2)
        dW2 = h1.T @ dh2_pre; db2 = dh2_pre.sum(axis=0)
        dh1 = dh2_pre @ W2.T
        dh1_pre = dh1 * _relu_grad(inp @ W1 + b1)
        dW1 = inp.T @ dh1_pre; db1 = dh1_pre.sum(axis=0)
        for W, dW, b, db in [(W3, dW3, b3, db3), (W2, dW2, b2, db2), (W1, dW1, b1, db1)]:
            W -= lr * dW; b -= lr * db

    def _denoise_sample(n: int):
        rr = np.random.default_rng(1)
        x = rr.normal(size=(n, d))
        for t in reversed(range(T)):
            inp = np.column_stack([x, np.full(n, t / T)])
            h1 = _relu(inp @ W1 + b1)
            h2 = _relu(h1 @ W2 + b2)
            eps_hat = h2 @ W3 + b3
            coef = (1 - alphas[t]) / np.sqrt(1 - alpha_bar[t])
            x = (x - coef * eps_hat) / np.sqrt(alphas[t])
            if t > 0:
                x += np.sqrt(betas[t]) * rr.normal(size=x.shape)
        return x

    return {"losses": losses, "sample": _denoise_sample, "T": T,
            "method": "DDPM (Ho et al. 2020, numpy MLP denoiser)"}


if __name__ == "__main__":
    X = make_moons(n=400, noise=0.05, seed=0)
    print(f"=== DDPM on two-moons (n={len(X)}, T=40 timesteps) ===")
    m = train_ddpm(X, T=40, hidden=64, lr=0.02, epochs=1500)
    samples = m["sample"](300)
    print(f"  final training MSE = {m['losses'][-1]:.4f}   "
          f"(initial {m['losses'][0]:.4f})")
    print(f"  sample stats: mean = {np.round(samples.mean(axis=0), 3).tolist()}, "
          f"sd = {np.round(samples.std(axis=0), 3).tolist()}")
    print(f"  data   stats: mean = {np.round(X.mean(axis=0), 3).tolist()}, "
          f"sd = {np.round(X.std(axis=0), 3).tolist()}")

    # coverage: samples that fall near a training point
    dists = np.min(((samples[:, None, :] - X[None, :, :]) ** 2).sum(-1), axis=1)
    print(f"  fraction of samples within radius 0.15 of a training point: "
          f"{(np.sqrt(dists) < 0.15).mean():.3f}")

    print("\n--- library cross-check (diffusers.DDPMScheduler, denoising-diffusion-pytorch) ---")
