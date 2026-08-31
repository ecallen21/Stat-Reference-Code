"""Neural Ordinary Differential Equations (Chen et al. 2018; Reference §27.x extra).

Replace a stack of L residual blocks:
    z_{l+1} = z_l + f_theta(z_l)
with a CONTINUOUS-DEPTH ODE:
    dz/dt = f_theta(z(t), t)
    z(t1) = z(t0) + integral_{t0}^{t1} f_theta(z(t), t) dt

We compute the forward pass by a numerical ODE solver (Euler, RK4, or an
adaptive method).  Backprop can go through the solver (autodiff) or use the
ADJOINT METHOD (Pontryagin's principle) for O(1) memory.

Applications:
  * Continuous normalising flows (CNF).
  * Time series with irregular sampling.
  * Physics-informed neural networks.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _relu(z): return np.maximum(z, 0.0)


def f_theta(z, t, params):
    """A tiny MLP that plays the role of dz/dt (input is z, time t is appended)."""
    x = np.concatenate([z, [t]])
    h = np.tanh(x @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]


def euler_solver(z0, t0, t1, params, n_steps=20):
    dt = (t1 - t0) / n_steps
    z = z0.copy(); t = t0
    for _ in range(n_steps):
        z = z + dt * f_theta(z, t, params); t += dt
    return z


def rk4_solver(z0, t0, t1, params, n_steps=20):
    dt = (t1 - t0) / n_steps
    z = z0.copy(); t = t0
    for _ in range(n_steps):
        k1 = f_theta(z, t, params)
        k2 = f_theta(z + 0.5 * dt * k1, t + 0.5 * dt, params)
        k3 = f_theta(z + 0.5 * dt * k2, t + 0.5 * dt, params)
        k4 = f_theta(z + dt * k3, t + dt, params)
        z = z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += dt
    return z


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = 2
    params = {
        "W1": rng.normal(scale=0.3, size=(d + 1, 16)),
        "b1": np.zeros(16),
        "W2": rng.normal(scale=0.3, size=(16, d)),
        "b2": np.zeros(d),
    }
    z0 = np.array([1.0, 0.0])

    # Same trajectory computed by Euler at various step counts and by RK4
    for n in (5, 20, 200):
        z_end = euler_solver(z0, 0.0, 1.0, params, n_steps=n)
        print(f"  Euler   n_steps={n:>3}: z(1) = {np.round(z_end, 4).tolist()}")
    z_ref = rk4_solver(z0, 0.0, 1.0, params, n_steps=200)
    z_rk4 = rk4_solver(z0, 0.0, 1.0, params, n_steps=20)
    print(f"  RK4     n_steps= 20: z(1) = {np.round(z_rk4, 4).tolist()}")
    print(f"  RK4    n_steps=200: z(1) = {np.round(z_ref, 4).tolist()}   (reference)")
    print(f"\n  Euler(200) - RK4(200) error norm = "
          f"{np.linalg.norm(euler_solver(z0, 0.0, 1.0, params, n_steps=200) - z_ref):.2e}")
    print(f"  RK4(20)   - RK4(200) error norm = "
          f"{np.linalg.norm(z_rk4 - z_ref):.2e}   (RK4 much more accurate per step)")

    # Layer analogy: a ResNet with L blocks is Euler on a ODE with L steps
    print(f"\n  A residual net with L blocks == Euler solver with L steps of an ODE.")
    print(f"  Neural ODEs generalise this: solver is a hyperparameter, depth is continuous.")

    print("\n--- library cross-check (torchdiffeq odeint; diffrax; JAX diffeqpy) ---")
