"""RMSprop optimizer (Reference Sec 47.106).

Tieleman & Hinton 2012 (Coursera lecture 6e). Divide gradient by
a running RMS of squared gradients:

    v_t = beta * v_{t-1} + (1 - beta) * g_t^2
    theta_t = theta_{t-1} - eta * g_t / (sqrt(v_t) + eps)

Adaptive per-coordinate learning rate; predecessor of Adam
(which adds momentum on g_t itself). Popular for RNN training and
noisy-gradient regimes.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sgd(grad_fn, theta, lr, n_iter, rng=None):
    hist = [float(np.linalg.norm(theta))]
    for _ in range(n_iter):
        theta = theta - lr * grad_fn(theta)
        hist.append(float(np.linalg.norm(theta)))
    return theta, hist


def rmsprop(grad_fn, theta, lr, n_iter, beta=0.9, eps=1e-8):
    v = np.zeros_like(theta)
    hist = [float(np.linalg.norm(theta))]
    for _ in range(n_iter):
        g = grad_fn(theta)
        v = beta * v + (1 - beta) * g ** 2
        theta = theta - lr * g / (np.sqrt(v) + eps)
        hist.append(float(np.linalg.norm(theta)))
    return theta, hist


def adam(grad_fn, theta, lr, n_iter, b1=0.9, b2=0.999, eps=1e-8):
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    hist = [float(np.linalg.norm(theta))]
    for t in range(1, n_iter + 1):
        g = grad_fn(theta)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        m_hat = m / (1 - b1 ** t); v_hat = v / (1 - b2 ** t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        hist.append(float(np.linalg.norm(theta)))
    return theta, hist


if __name__ == "__main__":
    print("=== RMSprop (Tieleman-Hinton 2012) ===\n")
    rng = np.random.default_rng(0)

    # Ill-conditioned quadratic: f(x) = 0.5 x' A x with A = diag(1, 100).
    A = np.diag([1.0, 100.0])
    x_star = np.zeros(2)
    def f(x): return 0.5 * x @ A @ x
    def grad_f(x): return A @ x
    theta0 = np.array([10.0, 1.0])

    for method_name, fn, kw in [
        ("SGD lr=0.01     ", sgd, dict(lr=0.01)),
        ("SGD lr=0.02     ", sgd, dict(lr=0.02)),
        ("RMSprop lr=0.5  ", rmsprop, dict(lr=0.5)),
        ("Adam lr=0.5     ", adam, dict(lr=0.5)),
    ]:
        theta, hist = fn(grad_f, theta0.copy(), n_iter=200, **kw)
        print(f"  {method_name}  final ||theta|| = {hist[-1]:.4e}")

    # Noisy gradient example
    print("\n  Noisy quadratic (add N(0, 5) to each grad):")
    def grad_noisy(x): return A @ x + rng.normal(size=2) * 5

    for method_name, fn, kw in [
        ("SGD lr=0.005    ", sgd, dict(lr=0.005)),
        ("RMSprop lr=0.5  ", rmsprop, dict(lr=0.5)),
        ("Adam lr=0.5     ", adam, dict(lr=0.5)),
    ]:
        theta, hist = fn(grad_noisy, theta0.copy(), n_iter=500, **kw)
        print(f"  {method_name}  final ||theta|| = {hist[-1]:.4e}")

    print("\n--- library cross-check (keras / torch.optim.RMSprop Python; torch R) ---")
