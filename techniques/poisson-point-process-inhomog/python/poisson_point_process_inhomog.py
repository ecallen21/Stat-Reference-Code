"""Inhomogeneous Poisson process (Reference Sec 47.96).

Cox 1955 (foundational); Ogata 1981 thinning simulation. Point
process on [0, T] whose intensity depends on time:

    lambda(t) : R+ -> R+,   E[N(a, b)] = integral_a^b lambda(t) dt.

Sample via Ogata thinning: bound lambda(t) <= M, propose t ~
Poisson(M), accept with prob lambda(t) / M. MLE for a parametric
intensity by maximising

    log L(theta) = sum_i log lambda(t_i; theta) - integral_0^T lambda(t; theta) dt.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize    # MLE
from scipy.integrate import quad    # log-lik integral


def simulate_thinning(lam, M, T, rng=None):
    rng = rng or np.random.default_rng(0)
    events = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / M)
        if t >= T: break
        if rng.uniform() <= lam(t) / M:
            events.append(t)
    return np.array(events)


def loglik_ipp(theta, events, T, lam_fn):
    """Log-likelihood of an IPP with parametric intensity."""
    lams = np.array([lam_fn(t, theta) for t in events])
    if (lams <= 0).any():
        return 1e12
    integral, _ = quad(lambda t: lam_fn(t, theta), 0, T, limit=200)
    return -(np.sum(np.log(lams)) - integral)


if __name__ == "__main__":
    print("=== Inhomogeneous Poisson process (Cox 1955; Ogata 1981) ===\n")
    rng = np.random.default_rng(0)
    T = 100.0

    # Truth: lambda(t) = alpha + beta * sin(2 pi t / period) + gamma * exp(-t/tau)
    true = (0.4, 0.3, 2.0, 20.0)
    def lam_true(t):
        a, b, g, tau = true
        return a + b * np.sin(2 * np.pi * t / 25) + g * np.exp(-t / tau)

    M = 1.5 * max(lam_true(t) for t in np.linspace(0, T, 200))
    events = simulate_thinning(lam_true, M, T, rng)
    n_exp = quad(lam_true, 0, T)[0]
    print(f"  N = {len(events)} events on [0, {T}]  (expected {n_exp:.1f})")

    # MLE: fit lambda_hat(t; alpha, beta, gamma, tau)
    def lam_par(t, theta):
        a, b, g, tau = theta
        return a + b * np.sin(2 * np.pi * t / 25) + g * np.exp(-t / max(tau, 1e-3))

    res = minimize(loglik_ipp, x0=(0.5, 0.1, 1.0, 10.0),
                    args=(events, T, lam_par), method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 20000})
    print(f"\n  MLE: alpha={res.x[0]:.3f}, beta={res.x[1]:+.3f}, "
          f"gamma={res.x[2]:.3f}, tau={res.x[3]:.2f}")
    print(f"  Truth: alpha={true[0]}, beta={true[1]:+.3f}, "
          f"gamma={true[2]}, tau={true[3]}")

    print("\n--- library cross-check (spatstat / PtProcess R; tick / lifelines Python) ---")
