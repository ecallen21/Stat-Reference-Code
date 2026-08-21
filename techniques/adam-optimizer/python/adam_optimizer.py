"""Adam optimiser + siblings (Reference §27.11).

SGD:            w  <- w - lr * g
Momentum SGD:   m  <- beta1 * m + g;                 w <- w - lr * m
Nesterov:       m  <- beta1 * m + g at lookahead;    w <- w - lr * m
RMSProp:        v  <- beta2 * v + (1 - beta2) g^2;   w <- w - lr * g / sqrt(v + eps)
Adam:           m  <- beta1 * m + (1 - beta1) * g
                v  <- beta2 * v + (1 - beta2) * g^2
                m_hat = m / (1 - beta1^t);   v_hat = v / (1 - beta2^t)
                w  <- w - lr * m_hat / (sqrt(v_hat) + eps)
AdamW:          Adam + DECOUPLED weight decay:  w  <- w - lr * wd * w  (before / after the Adam step)
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _sphere_grad(w):
    return 2 * w                                          # f(w) = ||w||^2 -> grad = 2 w


def _rosenbrock_grad(w):
    """Grad of the 2-D Rosenbrock function."""
    x, y = w
    return np.array([-2 * (1 - x) - 400 * x * (y - x ** 2),
                     200 * (y - x ** 2)])


def optimise(grad_fn, w0, method: str, lr: float = 0.05,
             beta1: float = 0.9, beta2: float = 0.999,
             wd: float = 0.0, n_iter: int = 500) -> dict:
    w = np.array(w0, dtype=float)
    m = np.zeros_like(w); v = np.zeros_like(w)
    traj = [w.copy()]
    for t in range(1, n_iter + 1):
        g = grad_fn(w)
        if method == "sgd":
            w = w - lr * g
        elif method == "momentum":
            m = beta1 * m + g
            w = w - lr * m
        elif method == "rmsprop":
            v = beta2 * v + (1 - beta2) * g ** 2
            w = w - lr * g / (np.sqrt(v) + 1e-8)
        elif method == "adam":
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** t); v_hat = v / (1 - beta2 ** t)
            w = w - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        elif method == "adamw":
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** t); v_hat = v / (1 - beta2 ** t)
            w = w - lr * (m_hat / (np.sqrt(v_hat) + 1e-8) + wd * w)
        else:
            raise ValueError(method)
        traj.append(w.copy())
    return {"w": w, "trajectory": np.array(traj)}


if __name__ == "__main__":
    print("=== Convex sphere f(w) = ||w||^2, w0 = [3, 4], 200 iterations ===")
    for meth in ("sgd", "momentum", "rmsprop", "adam", "adamw"):
        r = optimise(_sphere_grad, [3.0, 4.0], meth, lr=0.05, wd=0.01, n_iter=200)
        w_end = r["w"]; f_end = float((w_end ** 2).sum())
        print(f"  {meth:>8}: final ||w|| = {np.linalg.norm(w_end):.6f}   f = {f_end:.6f}")

    print("\n=== Rosenbrock (banana), w0 = [-1.2, 1.0], 5000 iterations ===")
    for meth, lr in [("sgd", 0.001), ("momentum", 0.001),
                       ("rmsprop", 0.01), ("adam", 0.01), ("adamw", 0.01)]:
        r = optimise(_rosenbrock_grad, [-1.2, 1.0], meth, lr=lr, wd=0.0, n_iter=5000)
        w_end = r["w"]
        # Rosenbrock minimum at (1, 1)
        err = float(np.linalg.norm(w_end - np.array([1.0, 1.0])))
        print(f"  {meth:>8} (lr={lr}): w_end = {np.round(w_end, 3).tolist()}   "
              f"||w - (1, 1)|| = {err:.4f}")

    print("\n--- library cross-check (torch.optim.Adam / AdamW / RMSprop / SGD) ---")
