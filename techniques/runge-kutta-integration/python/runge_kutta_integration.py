"""Runge-Kutta 4 integrator (Runge 1895; Kutta 1901).

Classical explicit 4-stage 4th-order integrator for
    dy/dt = f(t, y)

    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h,   y + h * k3)
    y_new = y + h/6 * (k1 + 2 k2 + 2 k3 + k4)

Local truncation error O(h^5), global O(h^4).
"""

import numpy as np    # arrays


def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def euler_step(f, t, y, h):
    return y + h * f(t, y)


def integrate(step_fn, f, t0, y0, tf, h):
    n = int(np.round((tf - t0) / h))
    ts = np.linspace(t0, tf, n + 1)
    ys = np.empty((n + 1, len(np.atleast_1d(y0))))
    ys[0] = y0
    for i in range(n):
        ys[i + 1] = step_fn(f, ts[i], ys[i], h)
    return ts, ys


def demo():
    print("=== Runge-Kutta 4 (Runge 1895; Kutta 1901) ===")
    print("\n1. Harmonic oscillator y'' + y = 0 => y = cos(t)")

    def harm(t, y):
        return np.array([y[1], -y[0]])

    y0 = np.array([1.0, 0.0])
    for h in [0.1, 0.05, 0.01]:
        _, ys_rk4 = integrate(rk4_step, harm, 0.0, y0, 10.0, h)
        _, ys_e = integrate(euler_step, harm, 0.0, y0, 10.0, h)
        err_rk4 = np.max(np.abs(ys_rk4[-1] - np.array([np.cos(10.0), -np.sin(10.0)])))
        err_e = np.max(np.abs(ys_e[-1] - np.array([np.cos(10.0), -np.sin(10.0)])))
        print(f"  h = {h:.3f}: RK4 err = {err_rk4:.2e}, Euler err = {err_e:.2e}")

    print("\n2. Lorenz-63 (chaotic)")

    def lorenz(t, y, s=10, r=28, b=8 / 3):
        return np.array([s * (y[1] - y[0]),
                         y[0] * (r - y[2]) - y[1],
                         y[0] * y[1] - b * y[2]])

    _, ys = integrate(rk4_step, lorenz, 0.0, np.array([1.0, 1.0, 1.0]), 20.0, 0.005)
    print(f"  RK4 trajectory length = {len(ys)}, x range = "
          f"[{ys[:, 0].min():.1f}, {ys[:, 0].max():.1f}] (Lorenz attractor)")


if __name__ == "__main__":
    demo()
