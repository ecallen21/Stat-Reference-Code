"""Variational autoencoder (Kingma-Welling 2013; Reference §27.8).

Generative model with latent z ~ N(0, I) and observation likelihood
    p(x | z) = N(mu_theta(z), sigma^2 I).

Encoder q_phi(z | x) = N(mu_phi(x), diag(sig2_phi(x))) approximates p(z | x).

ELBO (evidence lower bound):
    E_q[log p(x | z)] - KL(q_phi(z | x) || p(z))
where the KL of two Gaussians has a closed form:
    KL = 1/2 sum_k (mu_k^2 + sig2_k - log sig2_k - 1)

Reparameterisation trick: z = mu + sig * eps, eps ~ N(0, I) — pushes the
stochasticity outside the encoder, allowing gradient flow.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _relu(z): return np.maximum(z, 0.0)
def _relu_grad(z): return (z > 0).astype(float)


def fit_vae(X, k: int = 2, hidden: int = 32, lr: float = 0.01,
            epochs: int = 800, sigma_x: float = 0.5, seed: int = 0) -> dict:
    """Minimal VAE with one-hidden-layer encoder + decoder, diagonal Gaussian q."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float); N, d = X.shape
    W1e = rng.normal(scale=0.1, size=(d, hidden)); b1e = np.zeros(hidden)
    W_mu = rng.normal(scale=0.1, size=(hidden, k)); b_mu = np.zeros(k)
    W_ls = rng.normal(scale=0.1, size=(hidden, k)); b_ls = np.zeros(k)
    W1d = rng.normal(scale=0.1, size=(k, hidden)); b1d = np.zeros(hidden)
    W2d = rng.normal(scale=0.1, size=(hidden, d)); b2d = np.zeros(d)

    losses = []
    for ep in range(epochs):
        h1 = _relu(X @ W1e + b1e)
        mu = h1 @ W_mu + b_mu
        log_sig2 = h1 @ W_ls + b_ls
        sig = np.exp(0.5 * log_sig2)
        eps = rng.normal(size=(N, k))
        z = mu + sig * eps                                # reparameterisation
        h2 = _relu(z @ W1d + b1d)
        x_hat = h2 @ W2d + b2d
        # ELBO components (per-example averages)
        rec = float(((X - x_hat) ** 2).sum() / (N * 2 * sigma_x ** 2))
        kl = float(0.5 * (mu ** 2 + np.exp(log_sig2) - log_sig2 - 1).sum() / N)
        neg_elbo = rec + kl
        losses.append(neg_elbo)
        # gradient descent step (manual, minimal)
        d_xhat = (x_hat - X) / (N * sigma_x ** 2)
        d_W2d = h2.T @ d_xhat; d_b2d = d_xhat.sum(axis=0)
        d_h2 = d_xhat @ W2d.T
        d_z_h = d_h2 * _relu_grad(z @ W1d + b1d)
        d_W1d = z.T @ d_z_h; d_b1d = d_z_h.sum(axis=0)
        d_z = d_z_h @ W1d.T
        d_mu_rec = d_z
        d_sig_rec = d_z * eps
        d_log_sig2_rec = 0.5 * sig * d_sig_rec
        # KL grads
        d_mu = d_mu_rec + mu / N
        d_log_sig2 = d_log_sig2_rec + 0.5 * (np.exp(log_sig2) - 1) / N
        d_W_mu = h1.T @ d_mu; d_b_mu = d_mu.sum(axis=0)
        d_W_ls = h1.T @ d_log_sig2; d_b_ls = d_log_sig2.sum(axis=0)
        d_h1 = d_mu @ W_mu.T + d_log_sig2 @ W_ls.T
        d_z_e = d_h1 * _relu_grad(X @ W1e + b1e)
        d_W1e = X.T @ d_z_e; d_b1e = d_z_e.sum(axis=0)
        for W, dW, b, db in [(W2d, d_W2d, b2d, d_b2d),
                              (W1d, d_W1d, b1d, d_b1d),
                              (W_mu, d_W_mu, b_mu, d_b_mu),
                              (W_ls, d_W_ls, b_ls, d_b_ls),
                              (W1e, d_W1e, b1e, d_b1e)]:
            W -= lr * dW; b -= lr * db
    return {"losses": losses, "W1e": W1e, "b1e": b1e,
            "W_mu": W_mu, "b_mu": b_mu, "W_ls": W_ls, "b_ls": b_ls,
            "W1d": W1d, "b1d": b1d, "W2d": W2d, "b2d": b2d,
            "method": "Kingma-Welling VAE (numpy)"}


def encode(X, m):
    h1 = _relu(X @ m["W1e"] + m["b1e"])
    return h1 @ m["W_mu"] + m["b_mu"]


def sample(m, n: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, k))
    h2 = _relu(z @ m["W1d"] + m["b1d"])
    return h2 @ m["W2d"] + m["b2d"]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Toy: mixture of two 2D Gaussians embedded in 5D
    d = 5; N = 500
    centres = np.array([[2.0, -2.0], [-2.0, 2.0]])
    U = rng.normal(size=(2, d))                             # embedding
    z_true = np.vstack([rng.normal(loc=c, size=(N // 2, 2)) for c in centres])
    X = z_true @ U + 0.1 * rng.normal(size=(N, d))

    m = fit_vae(X, k=2, hidden=16, lr=0.01, epochs=1500, sigma_x=0.5)
    z_hat = encode(X, m)
    x_samples = sample(m, 200, 2, seed=1)

    print(f"=== VAE (2-D latent, N={N}, mixture of 2 Gaussians in R^{d}) ===")
    print(f"  final -ELBO       = {m['losses'][-1]:.4f}")
    print(f"  encoded latent stats: mean = {np.round(z_hat.mean(axis=0), 3).tolist()}, "
          f"sd = {np.round(z_hat.std(axis=0), 3).tolist()}")
    print(f"  sample stats: mean = {np.round(x_samples.mean(axis=0), 3).tolist()}, "
          f"sd = {np.round(x_samples.std(axis=0), 3).tolist()}")
    print(f"  data   stats: mean = {np.round(X.mean(axis=0), 3).tolist()}, "
          f"sd = {np.round(X.std(axis=0), 3).tolist()}")

    print("\n--- library cross-check (torch nn.Module + reparameterization + KL loss) ---")
