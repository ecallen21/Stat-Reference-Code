"""Adagrad — Adaptive Sub-gradient Method (Duchi, Hazan & Singer 2011).

Per-parameter learning rate that shrinks proportionally to
the accumulated squared gradient:

    G_t = G_{t-1} + g_t**2
    theta = theta - lr / (sqrt(G_t) + eps) * g_t

Sparse features get large updates (small G), dense ones
get small updates. Regret bound O(sqrt(T)) for online convex
optimisation, tighter for sparse gradients.
"""

import numpy as np    # arrays


def adagrad_step(theta, G, g, lr, eps):
    G = G + g ** 2
    theta = theta - lr * g / (np.sqrt(G) + eps)
    return theta, G


def demo():
    print("=== Adagrad (Duchi-Hazan-Singer 2011 JMLR) ===")
    rng = np.random.default_rng(2026)
    n, d = 400, 50
    # sparse feature matrix (only 10% non-zero)
    X = rng.standard_normal((n, d)) * (rng.uniform(size=(n, d)) < 0.1)
    beta_true = np.zeros(d)
    beta_true[::5] = rng.standard_normal(d // 5) * 2
    y = X @ beta_true + rng.standard_normal(n) * 0.3

    def grad(theta, batch_idx):
        r = X[batch_idx] @ theta - y[batch_idx]
        return X[batch_idx].T @ r / len(batch_idx)

    def loss(theta):
        return 0.5 * np.mean((X @ theta - y) ** 2)

    # Adagrad vs plain SGD
    for name, use_adagrad in [("Adagrad", True), ("SGD", False)]:
        theta = np.zeros(d)
        G = np.zeros(d)
        rng2 = np.random.default_rng(0)
        for step in range(3000):
            idx = rng2.integers(0, n, size=16)
            g = grad(theta, idx)
            if use_adagrad:
                theta, G = adagrad_step(theta, G, g, lr=0.5, eps=1e-8)
            else:
                theta = theta - 0.01 * g
        print(f"  {name:8s}: MSE = {loss(theta):.4f}, "
              f"nonzero-coef err = {np.linalg.norm(theta[::5] - beta_true[::5]):.3f}, "
              f"zero-coef norm  = {np.linalg.norm(theta[~np.isin(np.arange(d), np.arange(0, d, 5))]):.3f}")


if __name__ == "__main__":
    demo()
