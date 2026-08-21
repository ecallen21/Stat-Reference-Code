"""Autoencoder + denoising variant (Reference §27.7).

Encoder e_phi: R^d -> R^k (bottleneck)
Decoder d_theta: R^k -> R^d
Objective: minimise ||x - d(e(x))||^2   (reconstruction)

Denoising autoencoder (Vincent 2008): corrupt x with noise, then reconstruct
the CLEAN x from the noisy input:
    L = || x - d(e(x_tilde)) ||^2,   x_tilde = x + eps

We implement a linear autoencoder (essentially PCA when k < d and losses are
squared) plus a non-linear one-hidden-layer AE trained by SGD, and compare
their reconstruction MSE to PCA at the same rank.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _relu(z): return np.maximum(z, 0.0)
def _relu_grad(z): return (z > 0).astype(float)


def fit_autoencoder(X, k: int, hidden: int = 32, lr: float = 0.02,
                    epochs: int = 200, noise: float = 0.0,
                    seed: int = 0) -> dict:
    """Non-linear AE:  x -> ReLU(W1 x + b1) -> W2 (=bottleneck z) -> ReLU(W3 z + b3) -> W4 (=x_hat).
    If noise > 0, feed corrupted x to encoder but reconstruct clean x (denoising)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    W1 = rng.normal(scale=0.1, size=(d, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(scale=0.1, size=(hidden, k)); b2 = np.zeros(k)
    W3 = rng.normal(scale=0.1, size=(k, hidden)); b3 = np.zeros(hidden)
    W4 = rng.normal(scale=0.1, size=(hidden, d)); b4 = np.zeros(d)

    losses = []
    for ep in range(epochs):
        X_in = X + noise * rng.normal(size=X.shape) if noise > 0 else X
        a1 = _relu(X_in @ W1 + b1)
        z = a1 @ W2 + b2                                  # bottleneck (linear)
        a2 = _relu(z @ W3 + b3)
        x_hat = a2 @ W4 + b4
        loss = float(((x_hat - X) ** 2).mean())
        losses.append(loss)
        # backward
        d_xhat = 2 * (x_hat - X) / (N * d)
        d_W4 = a2.T @ d_xhat; d_b4 = d_xhat.sum(axis=0)
        d_a2 = d_xhat @ W4.T
        d_z_h = d_a2 * _relu_grad(z @ W3 + b3)
        d_W3 = z.T @ d_z_h; d_b3 = d_z_h.sum(axis=0)
        d_z = d_z_h @ W3.T
        d_W2 = a1.T @ d_z; d_b2 = d_z.sum(axis=0)
        d_a1 = d_z @ W2.T
        d_z_h1 = d_a1 * _relu_grad(X_in @ W1 + b1)
        d_W1 = X_in.T @ d_z_h1; d_b1 = d_z_h1.sum(axis=0)
        for W, dW, b, db in [(W4, d_W4, b4, d_b4), (W3, d_W3, b3, d_b3),
                              (W2, d_W2, b2, d_b2), (W1, d_W1, b1, d_b1)]:
            W -= lr * dW; b -= lr * db
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2,
            "W3": W3, "b3": b3, "W4": W4, "b4": b4,
            "losses": losses, "bottleneck_dim": k,
            "method": "non-linear autoencoder"
                       + (" (denoising)" if noise > 0 else "")}


def reconstruct(X, m):
    a1 = _relu(X @ m["W1"] + m["b1"])
    z = a1 @ m["W2"] + m["b2"]
    a2 = _relu(z @ m["W3"] + m["b3"])
    return a2 @ m["W4"] + m["b4"]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # data: mixture of 3 low-rank Gaussians in R^10
    d = 10; k_true = 3
    U = rng.normal(size=(d, k_true))
    Z = rng.normal(size=(400, k_true))
    X = Z @ U.T + 0.1 * rng.normal(size=(400, d))

    ae = fit_autoencoder(X, k=3, hidden=16, lr=0.1, epochs=2000)
    dae = fit_autoencoder(X, k=3, hidden=16, lr=0.1, epochs=2000, noise=0.3)
    x_hat_ae = reconstruct(X, ae)
    x_hat_dae = reconstruct(X, dae)
    mse_ae = float(((X - x_hat_ae) ** 2).mean())
    mse_dae = float(((X - x_hat_dae) ** 2).mean())

    # PCA baseline
    Xc = X - X.mean(axis=0)
    U_pca, S_pca, Vt_pca = np.linalg.svd(Xc, full_matrices=False)
    X_pca_rec = U_pca[:, :3] @ np.diag(S_pca[:3]) @ Vt_pca[:3] + X.mean(axis=0)
    mse_pca = float(((X - X_pca_rec) ** 2).mean())

    print(f"=== Autoencoder + denoising AE + PCA on rank-3 data in R^{d} (N=400) ===")
    print(f"  MSE(non-linear AE, k=3)      = {mse_ae:.5f}")
    print(f"  MSE(denoising AE, k=3, sig=0.3) = {mse_dae:.5f}")
    print(f"  MSE(PCA rank-3)              = {mse_pca:.5f}")
    print(f"  input variance               = {X.var():.5f}")
    print(f"\n  final AE training loss  = {ae['losses'][-1]:.5f}")
    print(f"  final DAE training loss = {dae['losses'][-1]:.5f}")

    print("\n--- library cross-check (torch nn.Sequential encoder + decoder) ---")
