"""beta-VAE for Disentanglement (Reference Sec 47.288).

Higgins et al 2017 ICLR. Modify the VAE ELBO to encourage
disentangled latent factors:

    ELBO_beta = E_q[log p(x|z)] - beta * KL(q(z|x) || p(z))

with beta > 1. Trades reconstruction fidelity for
independent latent axes. Standard for representation-learning
benchmarks (dSprites, 3D shapes).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def kl_gaussian(mu, log_var):
    return 0.5 * np.sum(-1 - log_var + mu ** 2 + np.exp(log_var), axis=1)


def elbo_beta(X_recon, X, mu, log_var, beta):
    """Toy Gaussian likelihood ELBO with adjustable beta."""
    recon = -0.5 * np.sum((X_recon - X) ** 2, axis=1)              # log p(x|z), sigma=1
    kl = kl_gaussian(mu, log_var)
    return float((recon - beta * kl).mean())


if __name__ == "__main__":
    print("=== beta-VAE (Higgins et al 2017 ICLR) ===\n")
    rng = np.random.default_rng(0)

    # Toy 2-D data with two axes: independent (0 to 3) x (0 to 3) grid
    N = 300
    z_true = rng.uniform(0, 3, (N, 2))                            # true latent factors
    X = np.column_stack([np.sin(z_true[:, 0]) + 0.1 * rng.normal(size=N),
                          np.cos(z_true[:, 1]) + 0.1 * rng.normal(size=N)])
    print(f"  Toy 2-D data with 2 true independent factors z1, z2 in [0, 3]\n")

    # Simulate simple linear encoder: mu = W X, log_var = fixed
    W = np.array([[1.0, 0.5], [-0.5, 1.0]])
    mu = X @ W.T
    log_var = np.log(0.1) * np.ones_like(mu)
    X_recon = X + rng.normal(0, 0.05, X.shape)                     # near-perfect reconstruction

    for beta in [1.0, 4.0, 16.0]:
        e = elbo_beta(X_recon, X, mu, log_var, beta)
        print(f"  beta = {beta:>4}   ELBO_beta = {e:>+.2f}")

    # Illustrate disentanglement metric: linear correlation between latent dims
    # After training, low correlation between mu[:, 0] and mu[:, 1] => disentangled
    corr = np.corrcoef(mu[:, 0], mu[:, 1])[0, 1]
    print(f"\n  Toy latent correlation |Corr(z_1, z_2)|: {abs(corr):.3f}")
    print(f"  (In real beta-VAE training, higher beta -> lower latent-axis correlation)\n")

    print("  beta-VAE demo pipeline in a real training run:")
    print("    1. Vary beta on a log grid (1, 4, 16, 64).")
    print("    2. Compute MIG / DCI / FactorVAE disentanglement scores.")
    print("    3. Report reconstruction quality (BCE / MSE) at each beta.")
    print("    Higher beta improves disentanglement but hurts fidelity.")

    print("\n--- library cross-check (disentanglement-lib; VAE zoo in PyTorch; NVIDIA-Kaolin) ---")
