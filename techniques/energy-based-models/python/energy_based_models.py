"""Energy-based models with contrastive divergence (Hinton 2002; Reference §27.x extra).

Model unnormalised density via an ENERGY function E_theta(x):
    p_theta(x) = exp(-E_theta(x)) / Z(theta),   Z intractable

Max-likelihood gradient:
    d/dtheta log p(x) = -d/dtheta E(x) + E_{x' ~ p_theta} [ d/dtheta E(x') ]
                        (positive)                          (negative)

Positive phase: gradient at data.  Negative phase: MCMC samples from p_theta.

Contrastive divergence (CD-k, Hinton 2002): run k Langevin / Gibbs steps
from EACH data point instead of from a random start.  Cheap approximation.

We fit a Gaussian energy E(x) = |x - mu|^2 / 2 sigma^2 (whose true density is
Gaussian) via CD-1 Langevin dynamics.  The learned energy landscape places
its minimum at the data mean.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _energy(x, mu, log_sig2):
    return 0.5 * np.sum((x - mu) ** 2, axis=-1) / np.exp(log_sig2)


def _grad_x_energy(x, mu, log_sig2):
    """dE/dx for Langevin sampling."""
    return (x - mu) / np.exp(log_sig2)


def langevin_step(x, mu, log_sig2, eps: float = 0.01, rng=None):
    if rng is None: rng = np.random.default_rng()
    return x - eps * _grad_x_energy(x, mu, log_sig2) + np.sqrt(2 * eps) * rng.normal(size=x.shape)


def train_ebm_cd1(X, dim: int = 2, n_iter: int = 400, lr: float = 0.02,
                   langevin_steps: int = 10, eps: float = 0.05,
                   seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    mu = rng.normal(scale=0.5, size=dim)
    log_sig2 = 0.0
    losses = []
    for it in range(n_iter):
        # positive-phase gradient: dE/dtheta at data (average)
        pos_grad_mu = -np.mean(X - mu, axis=0) / np.exp(log_sig2)
        # negative phase: k Langevin steps from each data point
        x_neg = X.copy()
        for _ in range(langevin_steps):
            x_neg = langevin_step(x_neg, mu, log_sig2, eps=eps, rng=rng)
        neg_grad_mu = -np.mean(x_neg - mu, axis=0) / np.exp(log_sig2)
        # CD-1 update:  theta -= lr * (positive - negative).  We only learn mu here
        # (fixed sigma^2 = 1) to keep the demo well-conditioned; production EBMs learn
        # a whole neural network of parameters via the same rule.
        d_mu = pos_grad_mu - neg_grad_mu
        mu -= lr * d_mu
        losses.append(float(np.mean(_energy(X, mu, log_sig2))))
    return {"mu": mu, "log_sig2": log_sig2, "sigma2": float(np.exp(log_sig2)),
            "losses": losses,
            "method": "EBM with contrastive divergence (Langevin, CD-k)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # data ~ N([2, -1], sigma^2 = 0.5)
    X = np.random.default_rng(0).normal(loc=[2.0, -1.0], scale=np.sqrt(0.5), size=(500, 2))

    m = train_ebm_cd1(X, dim=2, n_iter=300, lr=0.05, langevin_steps=15, eps=0.05)
    print(f"=== Energy-based model (Gaussian energy) with CD-15 Langevin ===")
    print(f"  data mean          = {X.mean(axis=0).round(3).tolist()}")
    print(f"  learned mu         = {np.round(m['mu'], 3).tolist()}")
    print(f"  data variance      = {X.var(axis=0).round(3).tolist()}")
    print(f"  learned sigma^2    = {m['sigma2']:.3f}   (data ~ 0.5)")

    # sample from the learned EBM via a long Langevin chain
    sampler_rng = np.random.default_rng(42)
    x = sampler_rng.normal(size=(300, 2))
    for _ in range(500):
        x = langevin_step(x, m["mu"], m["log_sig2"], eps=0.02, rng=sampler_rng)
    print(f"  sample stats:  mean = {np.round(x.mean(axis=0), 3).tolist()}, "
          f"sd = {np.round(x.std(axis=0), 3).tolist()}")
    print(f"  (learned energy is Gaussian centred at mu; stationary variance ~ sigma^2 = 1)")

    print("\n--- library cross-check (JEM, Grathwohl 2020; deep-EBM PyTorch examples) ---")
