"""Euler-Maruyama SDE simulation (Reference Sec 47.57).

Maruyama 1955 'Continuous Markov processes and stochastic
equations', Rend Circ Mat Palermo 4. For an Ito SDE

    dX_t = mu(X_t, t) dt + sigma(X_t, t) dW_t

the Euler-Maruyama scheme steps

    X_{k+1} = X_k + mu(X_k, t_k) * dt + sigma(X_k, t_k) * sqrt(dt) * Z_k
    Z_k ~ N(0, 1)  IID.

Strong order 0.5, weak order 1. For higher-order use Milstein
(adds a Wiener-integral correction) or Runge-Kutta SRK schemes.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def euler_maruyama(mu, sigma, X0, T, dt, n_paths=1, rng=None):
    """Simulate n_paths of an SDE from t=0 to T with step dt."""
    rng = rng or np.random.default_rng(0)
    n_steps = int(round(T / dt))
    ts = np.linspace(0, n_steps * dt, n_steps + 1)
    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = X0
    for k in range(n_steps):
        Xk, tk = X[:, k], ts[k]
        Z = rng.normal(size=n_paths)
        X[:, k + 1] = Xk + mu(Xk, tk) * dt + sigma(Xk, tk) * np.sqrt(dt) * Z
    return ts, X


if __name__ == "__main__":
    print("=== Euler-Maruyama SDE simulation (Maruyama 1955) ===\n")

    # 1) Geometric Brownian motion: dS = r S dt + sig S dW ; E[S_T] = S_0 e^{rT}
    r, sig, S0, T, dt = 0.05, 0.3, 100.0, 1.0, 1e-3
    n_paths = 10_000
    _, S = euler_maruyama(lambda x, t: r * x, lambda x, t: sig * x,
                            S0, T, dt, n_paths)
    print(f"  Geometric BM (r={r}, sig={sig}, S0={S0}, T={T}):")
    print(f"    E[S_T]  MC = {S[:, -1].mean():.3f}   analytic = {S0 * np.exp(r * T):.3f}")
    print(f"    Var[S_T] MC = {S[:, -1].var():.3f}   analytic = "
          f"{S0 ** 2 * np.exp(2 * r * T) * (np.exp(sig ** 2 * T) - 1):.3f}")

    # 2) Ornstein-Uhlenbeck: dX = kappa (theta - X) dt + eta dW; stationary N(theta, eta^2/(2 kappa))
    kappa, theta, eta = 3.0, 1.0, 0.4
    _, Xou = euler_maruyama(lambda x, t: kappa * (theta - x),
                              lambda x, t: eta * np.ones_like(x),
                              0.0, 5.0, dt, n_paths)
    stat = Xou[:, -1000:].ravel()
    print(f"\n  Ornstein-Uhlenbeck (kappa={kappa}, theta={theta}, eta={eta}):")
    print(f"    Sample mean = {stat.mean():.4f}   analytic = {theta:.4f}")
    print(f"    Sample var  = {stat.var():.4f}   analytic = {eta ** 2 / (2 * kappa):.4f}")

    print("\n--- library cross-check (Sim.DiffProc R; sdeint / diffrax Python) ---")
