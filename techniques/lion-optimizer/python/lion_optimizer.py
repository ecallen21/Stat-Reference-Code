"""Lion — EvoLved Sign Momentum (Chen et al. 2023).

Discovered by Google via symbolic regression / program search.
Uses only the SIGN of the momentum-blended gradient direction,
which acts like a bounded-norm step:

    c = beta1 * m + (1 - beta1) * g
    theta = theta - lr * (sign(c) + wd * theta)
    m = beta2 * m + (1 - beta2) * g

Memory: half of AdamW (no second moment). Reported to match
or exceed AdamW on ViT, LLM training with lower FLOPs.
"""

import numpy as np    # arrays


def lion_step(theta, m, g, lr, wd, beta1, beta2):
    c = beta1 * m + (1 - beta1) * g
    theta = theta - lr * (np.sign(c) + wd * theta)
    m = beta2 * m + (1 - beta2) * g
    return theta, m


def adamw_step(theta, m, v, g, lr, wd, b1, b2, eps, t):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    theta = theta - lr * (mh / (np.sqrt(vh) + eps) + wd * theta)
    return theta, m, v


def demo():
    print("=== Lion vs AdamW (Chen et al 2023) ===")
    rng = np.random.default_rng(2026)
    n, d = 300, 40
    X = rng.standard_normal((n, d))
    beta_true = rng.standard_normal(d) * 0.3
    y = X @ beta_true + rng.standard_normal(n) * 0.5
    XtX_n = X.T @ X / n
    Xty_n = X.T @ y / n

    def grad(theta):
        return XtX_n @ theta - Xty_n

    def loss(theta):
        return 0.5 * np.mean((X @ theta - y) ** 2)

    # Lion: paper suggests LR ~ 1/10 of AdamW, WD ~ 3-10x higher
    theta_L = np.zeros(d)
    m_L = np.zeros(d)
    for t in range(1, 601):
        theta_L, m_L = lion_step(theta_L, m_L, grad(theta_L),
                                  lr=0.01, wd=0.03, beta1=0.9, beta2=0.99)

    theta_A = np.zeros(d)
    m_A, v_A = np.zeros(d), np.zeros(d)
    for t in range(1, 601):
        theta_A, m_A, v_A = adamw_step(theta_A, m_A, v_A, grad(theta_A),
                                        lr=0.1, wd=0.01, b1=0.9, b2=0.999,
                                        eps=1e-8, t=t)

    print(f"  Lion   : ‖theta‖ = {np.linalg.norm(theta_L):.3f}, MSE = {loss(theta_L):.4f}")
    print(f"  AdamW  : ‖theta‖ = {np.linalg.norm(theta_A):.3f}, MSE = {loss(theta_A):.4f}")
    print(f"  Truth  : ‖theta‖ = {np.linalg.norm(beta_true):.3f}")

    mem_lion = theta_L.nbytes + m_L.nbytes
    mem_adamw = theta_A.nbytes + m_A.nbytes + v_A.nbytes
    print(f"\n  Memory Lion  = {mem_lion} bytes  (theta + m)")
    print(f"  Memory AdamW = {mem_adamw} bytes  (theta + m + v)")


if __name__ == "__main__":
    demo()
