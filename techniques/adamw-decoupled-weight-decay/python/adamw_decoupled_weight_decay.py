"""AdamW — Decoupled Weight Decay (Loshchilov & Hutter 2019).

Adam mixes L2 penalty into the gradient BEFORE the moment
estimates, which distorts the effective decay. AdamW decouples
the weight-decay step from the gradient-based update:

    m = beta1 m + (1 - beta1) g
    v = beta2 v + (1 - beta2) g**2
    m_hat, v_hat = bias-correct(m, v)
    theta = theta - lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta)

Result: decoupled decay behaves like true L2 regularisation
regardless of the adaptive step, giving better generalisation
in transformers, ResNets, and modern LLM training.
"""

import numpy as np    # arrays


def adamw_step(theta, m, v, g, lr, wd, beta1, beta2, eps, t):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    theta = theta - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * theta)
    return theta, m, v


def adam_l2_step(theta, m, v, g, lr, wd, beta1, beta2, eps, t):
    """Original Adam with L2 (baked into gradient)."""
    g = g + wd * theta    # coupled L2
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta, m, v


def demo():
    print("=== AdamW vs Adam+L2 (Loshchilov-Hutter 2019) ===")
    rng = np.random.default_rng(2026)
    n, d = 200, 30
    X = rng.standard_normal((n, d))
    beta_true = rng.standard_normal(d) * 0.5
    y = X @ beta_true + rng.standard_normal(n) * 0.5
    XtX_n = X.T @ X / n
    Xty_n = X.T @ y / n

    def loss(theta):
        return 0.5 * np.mean((X @ theta - y) ** 2)

    def grad(theta):
        return XtX_n @ theta - Xty_n

    lr, wd = 0.05, 0.1
    b1, b2, eps = 0.9, 0.999, 1e-8
    theta_w = np.zeros(d)
    theta_l2 = np.zeros(d)
    m_w, v_w = np.zeros(d), np.zeros(d)
    m_l2, v_l2 = np.zeros(d), np.zeros(d)

    for t in range(1, 501):
        g = grad(theta_w)
        theta_w, m_w, v_w = adamw_step(theta_w, m_w, v_w, g, lr, wd, b1, b2, eps, t)
        g2 = grad(theta_l2)
        theta_l2, m_l2, v_l2 = adam_l2_step(theta_l2, m_l2, v_l2, g2, lr, wd, b1, b2, eps, t)

    print(f"  Final ‖theta‖  AdamW      = {np.linalg.norm(theta_w):.3f}")
    print(f"  Final ‖theta‖  Adam+L2    = {np.linalg.norm(theta_l2):.3f}")
    print(f"  Test MSE       AdamW      = {loss(theta_w):.4f}")
    print(f"  Test MSE       Adam+L2    = {loss(theta_l2):.4f}")
    print(f"  ‖theta_true‖             = {np.linalg.norm(beta_true):.3f}")


if __name__ == "__main__":
    demo()
