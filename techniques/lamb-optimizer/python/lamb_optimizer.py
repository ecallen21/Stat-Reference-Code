"""LAMB — Layer-wise Adaptive Moments for Batch training
(You et al. 2020 ICLR).

AdamW + per-LAYER learning-rate normalisation:

    r = m_hat / (sqrt(v_hat) + eps) + wd * theta
    lr_layer = lr * ||theta|| / ||r||     (clipped to [lo, hi])
    theta = theta - lr_layer * r

The per-layer trust ratio ||theta|| / ||r|| makes the effective
step invariant to the initial weight scale — critical for
training with EXTREMELY LARGE mini-batches (32k+ in the BERT
paper, up to 65k in ImageNet).
"""

import numpy as np    # arrays


def lamb_step_layer(theta, m, v, g, lr, wd, b1, b2, eps, t,
                    trust_lo=None, trust_hi=None):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    r = mh / (np.sqrt(vh) + eps) + wd * theta
    w_norm = np.linalg.norm(theta)
    r_norm = np.linalg.norm(r)
    trust = (w_norm / r_norm) if (w_norm > 0 and r_norm > 0) else 1.0
    if trust_lo is not None:
        trust = max(trust, trust_lo)
    if trust_hi is not None:
        trust = min(trust, trust_hi)
    theta = theta - lr * trust * r
    return theta, m, v


def adamw_step_layer(theta, m, v, g, lr, wd, b1, b2, eps, t):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    theta = theta - lr * (mh / (np.sqrt(vh) + eps) + wd * theta)
    return theta, m, v


def demo():
    print("=== LAMB — Layer-wise Adaptive Moments (You et al 2020) ===")
    rng = np.random.default_rng(2026)
    # two "layers" of different scale
    d1, d2 = 20, 40
    A = rng.standard_normal((200, d1))
    B = rng.standard_normal((200, d2)) * 5    # different scale
    layer1_true = rng.standard_normal(d1) * 0.3
    layer2_true = rng.standard_normal(d2) * 0.05    # smaller scale
    y = A @ layer1_true + B @ layer2_true + rng.standard_normal(200) * 0.3

    def grad_layer1(l1, l2):
        r = A @ l1 + B @ l2 - y
        return A.T @ r / len(y)

    def grad_layer2(l1, l2):
        r = A @ l1 + B @ l2 - y
        return B.T @ r / len(y)

    for name, use_lamb in [("LAMB ", True), ("AdamW", False)]:
        l1 = np.zeros(d1)
        l2 = np.zeros(d2)
        m1, v1 = np.zeros(d1), np.zeros(d1)
        m2, v2 = np.zeros(d2), np.zeros(d2)
        # need to init theta near a scale so LAMB trust ratio is meaningful
        l1 = rng.standard_normal(d1) * 0.1
        l2 = rng.standard_normal(d2) * 0.1
        for t in range(1, 601):
            g1 = grad_layer1(l1, l2)
            g2 = grad_layer2(l1, l2)
            if use_lamb:
                l1, m1, v1 = lamb_step_layer(l1, m1, v1, g1, lr=0.05, wd=0.001,
                                              b1=0.9, b2=0.999, eps=1e-8, t=t)
                l2, m2, v2 = lamb_step_layer(l2, m2, v2, g2, lr=0.05, wd=0.001,
                                              b1=0.9, b2=0.999, eps=1e-8, t=t)
            else:
                l1, m1, v1 = adamw_step_layer(l1, m1, v1, g1, lr=0.05, wd=0.001,
                                               b1=0.9, b2=0.999, eps=1e-8, t=t)
                l2, m2, v2 = adamw_step_layer(l2, m2, v2, g2, lr=0.05, wd=0.001,
                                               b1=0.9, b2=0.999, eps=1e-8, t=t)

        r = A @ l1 + B @ l2 - y
        mse = 0.5 * np.mean(r ** 2)
        e1 = np.linalg.norm(l1 - layer1_true)
        e2 = np.linalg.norm(l2 - layer2_true)
        print(f"  {name}: MSE = {mse:.4f}, layer-1 err = {e1:.4f}, layer-2 err = {e2:.4f}")


if __name__ == "__main__":
    demo()
