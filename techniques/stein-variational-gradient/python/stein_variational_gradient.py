"""Stein Variational Gradient Descent (Reference Sec 47.87).

Liu & Wang 2016 'Stein Variational Gradient Descent: A general
purpose Bayesian inference algorithm', NeurIPS. Deterministic
particle-based Bayesian inference. Given target log-density
log p(x), evolve n particles by

    phi(x_i) = (1/n) sum_j  k(x_j, x_i) grad log p(x_j)  +  grad_{x_j} k(x_j, x_i)

    x_i <- x_i + epsilon * phi(x_i)

The first term is a weighted score push; the second a repulsion
that prevents mode collapse. Guarantees KL(q || p) decreases when
epsilon is small enough.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def svgd_step(X, grad_logp, eps=0.01):
    """One SVGD iteration on particles X (n, d)."""
    n, d = X.shape
    sq = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    med = np.median(sq)
    h = med / (2 * np.log(n + 1) + 1e-9)               # bandwidth heuristic
    K = np.exp(-sq / h)
    grads_logp = np.array([grad_logp(x) for x in X])
    # gradient of RBF kernel wrt x_j: (2 (x_i - x_j) / h) * K
    dK = np.zeros_like(X)
    for i in range(n):
        diff = X - X[i]                                 # (n, d)
        dK[i] = -2 * (K[:, i:i + 1] * diff).sum(axis=0) / h
    phi = (K @ grads_logp + dK) / n
    return X + eps * phi


def svgd(X0, grad_logp, n_iter=500, eps=0.05):
    X = X0.copy()
    for _ in range(n_iter):
        X = svgd_step(X, grad_logp, eps)
    return X


if __name__ == "__main__":
    print("=== SVGD (Liu-Wang 2016) ===\n")
    rng = np.random.default_rng(0)

    # Target: 1D mixture of N(-2, 1) and N(2, 1)
    def logp(x):
        p1 = np.exp(-0.5 * (x + 2) ** 2) / np.sqrt(2 * np.pi)
        p2 = np.exp(-0.5 * (x - 2) ** 2) / np.sqrt(2 * np.pi)
        return float(np.log(0.5 * p1 + 0.5 * p2 + 1e-300))

    def grad_logp(x):
        p1 = np.exp(-0.5 * (x + 2) ** 2)
        p2 = np.exp(-0.5 * (x - 2) ** 2)
        w1 = p1 / (p1 + p2)
        # d/dx log(0.5 p1 + 0.5 p2) = w1 * -(x+2) + w2 * -(x-2)
        return -w1 * (x + 2) - (1 - w1) * (x - 2)

    def grad_np(x_vec):
        return np.array([grad_logp(xi) for xi in x_vec])

    n = 30
    X0 = rng.normal(size=(n, 1)) * 0.5     # initialise near origin
    print(f"  Initial particle mean = {X0.mean():+.3f},   std = {X0.std():.3f}")

    X = svgd(X0, lambda x: np.array([grad_logp(x[0])]), n_iter=500, eps=0.05)
    print(f"  Final particle mean   = {X.mean():+.3f},   std = {X.std():.3f}")
    print(f"  Target (mixture of N(+-2, 1)) mean = 0, std = sqrt(5) = {np.sqrt(5):.3f}")

    # Fraction of particles in each mode
    left = float((X < 0).mean())
    print(f"  Fraction of particles in left mode  = {left:.2f} (target 0.50)")
    print(f"  Fraction of particles in right mode = {1 - left:.2f} (target 0.50)")

    print("\n--- library cross-check (no established R port; SVGD implementations in torch / jax) ---")
