"""Nadam — Nesterov-accelerated Adam (Dozat 2016).

Blend of Adam and Nesterov momentum. Uses a look-ahead
version of the momentum in the update:

    m = beta1 m + (1 - beta1) g
    v = beta2 v + (1 - beta2) g**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    m_nesterov = beta1 * m_hat + (1 - beta1) * g / (1 - beta1**t)
    theta = theta - lr * m_nesterov / (sqrt(v_hat) + eps)

Slightly faster convergence than Adam on standard benchmarks.
"""

import numpy as np    # arrays


def nadam_step(theta, m, v, g, lr, beta1, beta2, eps, t):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    m_nest = beta1 * m_hat + (1 - beta1) * g / (1 - beta1 ** t)
    theta = theta - lr * m_nest / (np.sqrt(v_hat) + eps)
    return theta, m, v


def adam_step(theta, m, v, g, lr, beta1, beta2, eps, t):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta, m, v


def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
               for i in range(len(x) - 1))


def rosen_grad(x):
    g = np.zeros_like(x)
    for i in range(len(x) - 1):
        g[i] = g[i] - 400 * x[i] * (x[i + 1] - x[i] ** 2) - 2 * (1 - x[i])
        g[i + 1] = g[i + 1] + 200 * (x[i + 1] - x[i] ** 2)
    return g


def demo():
    print("=== Nadam vs Adam (Dozat 2016) ===")
    print("Rosenbrock d=10 (Global min f=0 at [1,...,1])")

    for name, step_fn in [("Nadam", nadam_step), ("Adam", adam_step)]:
        theta = np.full(10, -1.0)
        m, v = np.zeros_like(theta), np.zeros_like(theta)
        for t in range(1, 4001):
            g = rosen_grad(theta)
            theta, m, v = step_fn(theta, m, v, g, lr=0.02, beta1=0.9, beta2=0.999,
                                  eps=1e-8, t=t)
        f_final = rosenbrock(theta)
        print(f"  {name:6s} : f = {f_final:.3e},  x_mean = {theta.mean():.4f}  (target 1.0)")


if __name__ == "__main__":
    demo()
