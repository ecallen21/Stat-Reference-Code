"""Adafactor — sublinear-memory adaptive optimiser
(Shazeer & Stern 2018).

For a WEIGHT MATRIX of shape (rows, cols), instead of storing
the full second moment v (which is rows*cols), factorise it
as an outer product of two vectors R (rows,) and C (cols,).
Memory becomes rows + cols instead of rows*cols.

Update:
    v_est[i, j] ≈ R[i] * C[j] / sum(R)
    theta -= lr * g / sqrt(v_est + eps)

Used to train very large transformer models (T5, PaLM) where
the AdamW state alone would be too large for GPU memory.
"""

import numpy as np    # arrays


def adafactor_matrix(theta, R, C, g, lr, beta2, eps1=1e-30, eps2=1e-3):
    """Adafactor update for a 2D weight matrix (rows, cols)."""
    U = g * g + eps1
    R = beta2 * R + (1 - beta2) * U.sum(axis=1)
    C = beta2 * C + (1 - beta2) * U.sum(axis=0)
    v_est = np.outer(R, C) / (R.sum() + eps1)
    update = g / (np.sqrt(v_est) + eps2)
    # RMS normalisation (Shazeer-Stern gradient-scale trick)
    rms = np.sqrt(np.mean(update ** 2))
    theta = theta - lr * update / max(1.0, rms / 1.0)
    return theta, R, C


def adamw_matrix(theta, m, v, g, lr, wd, b1, b2, eps, t):
    """AdamW baseline for comparison (stores full v of same shape)."""
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    theta = theta - lr * (mh / (np.sqrt(vh) + eps) + wd * theta)
    return theta, m, v


def demo():
    print("=== Adafactor vs AdamW (Shazeer-Stern 2018) ===")
    rng = np.random.default_rng(2026)
    n, d_in, d_out = 200, 40, 30
    X = rng.standard_normal((n, d_in))
    W_true = rng.standard_normal((d_in, d_out)) * 0.3
    Y = X @ W_true + rng.standard_normal((n, d_out)) * 0.3
    XtX_n = X.T @ X / n
    XtY_n = X.T @ Y / n

    def loss(W):
        return 0.5 * np.mean((X @ W - Y) ** 2)

    def grad(W):
        return XtX_n @ W - XtY_n

    W_af = np.zeros((d_in, d_out))
    R = np.zeros(d_in)
    C = np.zeros(d_out)
    for step in range(400):
        W_af, R, C = adafactor_matrix(W_af, R, C, grad(W_af),
                                       lr=0.05, beta2=0.999)

    W_aw = np.zeros((d_in, d_out))
    m_aw = np.zeros_like(W_aw)
    v_aw = np.zeros_like(W_aw)
    for t in range(1, 401):
        W_aw, m_aw, v_aw = adamw_matrix(W_aw, m_aw, v_aw, grad(W_aw),
                                         lr=0.05, wd=0.0, b1=0.9, b2=0.999,
                                         eps=1e-8, t=t)

    print(f"  Adafactor MSE = {loss(W_af):.4f},  ‖W‖ = {np.linalg.norm(W_af):.3f}")
    print(f"  AdamW     MSE = {loss(W_aw):.4f},  ‖W‖ = {np.linalg.norm(W_aw):.3f}")
    print(f"  Truth     ‖W‖ = {np.linalg.norm(W_true):.3f}")

    mem_af = (W_af.nbytes + R.nbytes + C.nbytes)
    mem_aw = (W_aw.nbytes + m_aw.nbytes + v_aw.nbytes)
    print(f"\n  Memory Adafactor : {mem_af:5d} bytes  (W + R + C)")
    print(f"  Memory AdamW     : {mem_aw:5d} bytes  (W + m + v)")
    print(f"  Savings          : {100 * (1 - mem_af / mem_aw):.1f}%")


if __name__ == "__main__":
    demo()
