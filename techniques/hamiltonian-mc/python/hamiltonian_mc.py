"""Hamiltonian Monte Carlo (Reference §14.8).

MCMC that uses GRADIENT information to make big, informed proposals across
the posterior.  Vastly outperforms random-walk MH in moderate/high dimension.

Idea
    Augment theta with a momentum p ~ N(0, M).  Define Hamiltonian
        H(theta, p) = -log pi(theta) + (1/2) p^T M^-1 p
    Move (theta, p) along Hamilton's equations for L leapfrog steps of size
    epsilon.  Accept the endpoint with probability min(1, exp(-Delta H)).

Leapfrog integrator (reversible + volume-preserving):
    p_{1/2} = p - (eps/2) grad U(theta)
    theta_1 = theta + eps M^-1 p_{1/2}
    p_1     = p_{1/2} - (eps/2) grad U(theta_1)

Tuning
    - Step size eps:  target acceptance ~0.65-0.9.
    - Trajectory length L:  long enough to explore, short enough not to
      make a full U-turn.  NUTS (Hoffman & Gelman 2014) adaptively selects L.

NUTS
    The No-U-Turn Sampler builds a binary tree of leapfrog steps in both
    directions, stopping once any subtrajectory makes a U-turn.  Removes
    both L and eps as user tuning parameters (via dual-averaging on eps).
    Standard implementation: Stan / PyMC / NumPyro.

Numerical gradient (default in demo) is fine for small problems; production
HMC uses autodiff (JAX / PyTorch / Stan / Zygote).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _num_grad(f, x, eps=1e-5):
    g = np.zeros_like(x)
    for i in range(len(x)):
        ei = np.zeros_like(x); ei[i] = eps
        g[i] = (f(x + ei) - f(x - ei)) / (2 * eps)
    return g


def hmc(log_target, grad_log_target, theta0, n_iter: int = 2000,
        eps: float = 0.1, L: int = 20, seed: int = 0) -> dict:
    """Vanilla HMC with unit mass matrix and fixed step count."""
    rng = np.random.default_rng(seed)
    theta = np.atleast_1d(np.asarray(theta0, dtype=float))
    d = theta.shape[0]
    samples = np.empty((n_iter, d))
    n_accept = 0
    U = lambda q: -log_target(q)
    grad_U = lambda q: -grad_log_target(q)
    for t in range(n_iter):
        p = rng.normal(size=d)
        q = theta.copy()
        p_cur = p.copy(); q_cur = q.copy()
        # Half step for momentum
        p = p - 0.5 * eps * grad_U(q)
        for _ in range(L - 1):
            q = q + eps * p
            p = p - eps * grad_U(q)
        q = q + eps * p
        p = p - 0.5 * eps * grad_U(q)
        # Negate p for reversibility (irrelevant for symmetric kinetic energy)
        p = -p
        H_new = U(q) + 0.5 * (p @ p)
        H_cur = U(q_cur) + 0.5 * (p_cur @ p_cur)
        if math.log(rng.uniform()) < H_cur - H_new:
            theta = q; n_accept += 1
        samples[t] = theta
    return {"samples": samples,
            "acceptance_rate": float(n_accept / n_iter),
            "step_size": float(eps), "n_leapfrog": int(L),
            "n_iter": int(n_iter), "dim": int(d),
            "method": "Hamiltonian Monte Carlo (leapfrog)"}


if __name__ == "__main__":
    # Target: N(mu, Sigma) with strong correlation to show HMC advantage.
    mu_t = np.array([1.0, -0.5, 2.0])
    Sigma_t = np.array([[1.0, 0.9, 0.4],
                        [0.9, 1.5, 0.7],
                        [0.4, 0.7, 2.0]])
    Sigma_inv = np.linalg.inv(Sigma_t)

    def log_target(q):
        d = q - mu_t
        return -0.5 * d @ Sigma_inv @ d

    def grad_log_target(q):
        return -Sigma_inv @ (q - mu_t)

    print("=== HMC on 3-D correlated Gaussian, 2000 iter, L=20, eps=0.2 ===")
    r = hmc(log_target, grad_log_target, theta0=[0.0, 0.0, 0.0],
            n_iter=2000, eps=0.2, L=20, seed=0)
    burn = 500
    S = r["samples"][burn:]
    print(f"  acceptance rate: {r['acceptance_rate']:.3f}")
    print(f"  empirical mean:  {S.mean(0).round(3)}  (true {mu_t})")
    print(f"  empirical cov:")
    print(np.cov(S.T).round(3))
    print(f"  true cov:")
    print(Sigma_t)

    print("\n=== Same target via numerical gradient (sanity) ===")
    r_ng = hmc(log_target, lambda q: _num_grad(log_target, q),
               theta0=[0.0, 0.0, 0.0], n_iter=1000, eps=0.2, L=20, seed=1)
    print(f"  acceptance rate: {r_ng['acceptance_rate']:.3f}, mean = {r_ng['samples'][200:].mean(0).round(3)}")
