"""Velocity Verlet / Leapfrog symplectic integrator
(Verlet 1967; Hairer-Lubich-Wanner 2006).

For a separable Hamiltonian
    H(q, p) = 0.5 p.T M^{-1} p + V(q)

Verlet updates half-kick / full-drift / half-kick:

    p_{k+1/2} = p_k - 0.5 h grad V(q_k)
    q_{k+1}   = q_k + h M^{-1} p_{k+1/2}
    p_{k+1}   = p_{k+1/2} - 0.5 h grad V(q_{k+1})

Symplectic — preserves phase-space volume; energy oscillates
around exact but does NOT drift. Backbone of MD, HMC/NUTS.
"""

import numpy as np    # arrays


def verlet_step(q, p, grad_V, h, M_inv):
    p_half = p - 0.5 * h * grad_V(q)
    q_next = q + h * (M_inv @ p_half)
    p_next = p_half - 0.5 * h * grad_V(q_next)
    return q_next, p_next


def rk4_step(q, p, grad_V, h, M_inv):
    """Comparison non-symplectic RK4 for Hamiltonian flow."""
    def f(qp):
        q_, p_ = qp[:len(q)], qp[len(q):]
        return np.concatenate([M_inv @ p_, -grad_V(q_)])
    y = np.concatenate([q, p])
    k1 = f(y)
    k2 = f(y + 0.5 * h * k1)
    k3 = f(y + 0.5 * h * k2)
    k4 = f(y + h * k3)
    y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y[:len(q)], y[len(q):]


def demo():
    print("=== Velocity Verlet / Leapfrog symplectic (Verlet 1967) ===")
    print("Test: harmonic oscillator H = 0.5 (p^2 + q^2)")

    def grad_V(q):
        return q    # V = 0.5 q^2

    M_inv = np.array([[1.0]])
    q, p = np.array([1.0]), np.array([0.0])
    E0 = 0.5 * (p @ p + q @ q)
    print(f"  Initial energy = {E0:.6f}")

    T, h = 5000.0, 0.5
    n = int(T / h)
    qs_v, ps_v = np.empty(n + 1), np.empty(n + 1)
    qs_v[0], ps_v[0] = q[0], p[0]
    q_v, p_v = q.copy(), p.copy()
    for k in range(n):
        q_v, p_v = verlet_step(q_v, p_v, grad_V, h, M_inv)
        qs_v[k + 1], ps_v[k + 1] = q_v[0], p_v[0]

    qs_r, ps_r = np.empty(n + 1), np.empty(n + 1)
    qs_r[0], ps_r[0] = q[0], p[0]
    q_r, p_r = q.copy(), p.copy()
    for k in range(n):
        q_r, p_r = rk4_step(q_r, p_r, grad_V, h, M_inv)
        qs_r[k + 1], ps_r[k + 1] = q_r[0], p_r[0]

    E_v = 0.5 * (ps_v ** 2 + qs_v ** 2)
    E_r = 0.5 * (ps_r ** 2 + qs_r ** 2)
    print(f"  After t={T}: Verlet ΔE range = "
          f"[{E_v.min() - E0:+.2e}, {E_v.max() - E0:+.2e}]  (bounded)")
    print(f"                RK4    ΔE range = "
          f"[{E_r.min() - E0:+.2e}, {E_r.max() - E0:+.2e}]  (drift)")
    print(f"  Final:        Verlet E = {E_v[-1]:.6f}, RK4 E = {E_r[-1]:.6f}, "
          f"exact = 0.500000")


if __name__ == "__main__":
    demo()
