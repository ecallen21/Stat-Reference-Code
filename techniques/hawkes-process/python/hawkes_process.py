"""Hawkes process (Reference Sec 47.41).

Hawkes 1971 'Spectra of some self-exciting and mutually exciting
point processes', Biometrika 58(1). A self-exciting temporal point
process with conditional intensity

    lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))

each past event boosts intensity by `alpha`, decaying at rate `beta`.
Stationary iff `alpha / beta < 1` (branching ratio).

Simulation: Ogata thinning (Ogata 1981). MLE: exact log-likelihood
with a recursion for the summation kernel (Ozaki 1979).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize    # log-lik maximisation


def simulate_hawkes_thinning(mu, alpha, beta, T, rng=None):
    """Ogata thinning simulation of exp-kernel univariate Hawkes."""
    rng = rng or np.random.default_rng(0)
    events = []
    t = 0.0
    while t < T:
        lam_bar = mu + alpha * sum(np.exp(-beta * (t - ti)) for ti in events)
        w = rng.exponential(1.0 / lam_bar)
        t = t + w
        if t >= T:
            break
        lam_t = mu + alpha * sum(np.exp(-beta * (t - ti)) for ti in events)
        if rng.uniform() <= lam_t / lam_bar:
            events.append(t)
    return np.array(events)


def hawkes_loglik(params, events, T):
    """Exp-kernel Hawkes log-likelihood using Ozaki 1979 recursion."""
    mu, alpha, beta = params
    if mu <= 0 or alpha <= 0 or beta <= 0:
        return 1e12
    n = len(events)
    A = np.zeros(n)
    for i in range(1, n):
        A[i] = np.exp(-beta * (events[i] - events[i - 1])) * (1 + A[i - 1])
    lam_i = mu + alpha * A
    integral = mu * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - events)))
    return -(np.sum(np.log(lam_i)) - integral)


def fit_hawkes(events, T, x0=(0.5, 0.3, 1.0)):
    res = minimize(hawkes_loglik, x0, args=(events, T),
                   method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6})
    return {"mu": float(res.x[0]), "alpha": float(res.x[1]),
            "beta": float(res.x[2]), "loglik": -float(res.fun)}


if __name__ == "__main__":
    print("=== Hawkes process (Hawkes 1971) ===\n")
    rng = np.random.default_rng(0)
    mu, alpha, beta, T = 0.5, 0.6, 1.5, 500.0
    events = simulate_hawkes_thinning(mu, alpha, beta, T, rng)
    print(f"  Simulated N = {len(events)} events on [0, {T}]")
    print(f"  Branching ratio n = alpha/beta = {alpha/beta:.2f}")
    print(f"  Expected events E[N] = mu*T/(1-n) = {mu*T/(1-alpha/beta):.1f}")

    fit = fit_hawkes(events, T)
    print(f"\n  MLE: mu={fit['mu']:.3f} (truth {mu}), "
          f"alpha={fit['alpha']:.3f} (truth {alpha}), "
          f"beta={fit['beta']:.3f} (truth {beta})")
    print(f"  log-lik = {fit['loglik']:.2f}")

    print("\n--- library cross-check (hawkes / tick R / Python) ---")
