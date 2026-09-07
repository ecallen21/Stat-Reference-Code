"""Hamiltonian Monte Carlo (Reference Sec 14.4).

Duane et al. 1987; Neal 2011 tutorial.  Simulate Hamiltonian
dynamics to propose distant states that traverse the target
posterior efficiently.

  q -> position (parameter)
  p -> momentum, refreshed each iteration from N(0, M)
  H(q, p) = U(q) + K(p)   with U(q) = -log posterior(q)

Leapfrog integrator + Metropolis-Hastings accept step preserve
detailed balance.

NUTS (Hoffman-Gelman 2014) auto-tunes trajectory length by
recursively doubling until the path U-turns.  Here we implement
plain HMC with fixed L and step size for demonstration.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def hmc(U, grad_U, q0, epsilon=0.05, L=20, n_iter=2000, seed=0):
    rng = np.random.default_rng(seed)
    q = q0.copy()
    d = len(q)
    samples = []
    accepts = 0
    for _ in range(n_iter):
        q_new = q.copy()
        p = rng.normal(size=d)
        current_H = U(q) + 0.5 * (p ** 2).sum()
        # Leapfrog
        p_new = p - 0.5 * epsilon * grad_U(q_new)
        for _ in range(L):
            q_new = q_new + epsilon * p_new
            p_new = p_new - epsilon * grad_U(q_new)
        p_new = p_new + 0.5 * epsilon * grad_U(q_new)     # correct final half-step
        proposed_H = U(q_new) + 0.5 * (p_new ** 2).sum()
        if np.log(rng.random()) < current_H - proposed_H:
            q = q_new; accepts += 1
        samples.append(q.copy())
    return np.array(samples), accepts / n_iter


if __name__ == "__main__":
    print("=== Hamiltonian Monte Carlo (HMC) ===\n")
    # Target: 2-D correlated Gaussian
    mu_true = np.array([1.0, -0.5])
    Sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
    Sigma_inv = np.linalg.inv(Sigma)
    U = lambda q: 0.5 * (q - mu_true) @ Sigma_inv @ (q - mu_true)
    grad_U = lambda q: Sigma_inv @ (q - mu_true)

    samples, acc = hmc(U, grad_U, q0=np.array([0.0, 0.0]), epsilon=0.15, L=20, n_iter=2000)
    burn = 500
    print(f"  Acceptance rate: {acc:.3f}   (target ~0.65-0.90)")
    print(f"  True mean:      {mu_true}")
    print(f"  HMC mean:       {samples[burn:].mean(axis=0).round(3)}")
    print(f"  True Sigma:\n{Sigma}")
    print(f"  HMC Sigma:\n{np.cov(samples[burn:].T).round(3)}\n")

    print("  NUTS extension: auto-tune L via no-U-turn criterion (Hoffman-Gelman 2014).\n")
    print("--- library cross-check (R rstan, brms; Python pymc, numpyro, blackjax) ---")
