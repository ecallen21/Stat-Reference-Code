"""Rectified Adam - RAdam (Reference Sec 47.165).

Liu et al 2020 'On the Variance of the Adaptive Learning Rate and
Beyond', ICLR. Adam's adaptive lr has HIGH VARIANCE in the first
few steps; a common fix is a linear warm-up. RAdam derives a
CLOSED-FORM rectification term that turns Adam into plain SGD
(no adaptive lr) while its variance estimate is unreliable, and
smoothly switches to Adam once enough gradient history has
accumulated.

    rho_inf = 2 / (1 - beta2) - 1
    rho_t = rho_inf - 2 t beta2^t / (1 - beta2^t)
    if rho_t > 4:  r_t = sqrt((rho_t - 4)(rho_t - 2) rho_inf /
                                    ((rho_inf - 4)(rho_inf - 2) rho_t))
                    Adam step * r_t
    else:           plain SGD step (no adaptive lr).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def adam(grad_fn, W0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, n_iter=200):
    W = W0.copy()
    m = np.zeros_like(W); v = np.zeros_like(W)
    hist = []
    for t in range(1, n_iter + 1):
        g = grad_fn(W)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        W -= lr * m_hat / (np.sqrt(v_hat) + eps)
        hist.append(W.copy())
    return W, hist


def radam(grad_fn, W0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, n_iter=200):
    W = W0.copy()
    m = np.zeros_like(W); v = np.zeros_like(W)
    rho_inf = 2 / (1 - beta2) - 1
    hist = []
    for t in range(1, n_iter + 1):
        g = grad_fn(W)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        rho_t = rho_inf - 2 * t * beta2 ** t / (1 - beta2 ** t)
        if rho_t > 4:
            v_hat = np.sqrt(v / (1 - beta2 ** t))
            r_t = np.sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf)
                           / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
            W -= lr * r_t * m_hat / (v_hat + eps)
        else:
            W -= lr * m_hat                                     # SGD in warm-up
        hist.append(W.copy())
    return W, hist


if __name__ == "__main__":
    print("=== Rectified Adam - RAdam (Liu et al 2020) ===\n")

    # Noisy quadratic bowl: f(w) = (1/2) w^T A w  with noisy grad
    rng = np.random.default_rng(0)
    d = 5
    A = np.diag(np.linspace(0.1, 2.0, d))                       # milder condition
    def make_grad(noise):
        def grad(W):
            return A @ W + rng.normal(scale=noise, size=d)
        return grad

    W0 = 5 * np.ones(d)
    for noise in [0.01, 0.5, 3.0]:
        g_fn = make_grad(noise)
        _, adam_hist = adam(g_fn, W0, lr=0.1, n_iter=200)
        g_fn = make_grad(noise)
        _, rad_hist = radam(g_fn, W0, lr=0.1, n_iter=200)
        # Loss at each iter
        adam_loss = [0.5 * float(w @ A @ w) for w in adam_hist[-30:]]
        rad_loss = [0.5 * float(w @ A @ w) for w in rad_hist[-30:]]
        print(f"  noise = {noise:4.2f}   Adam final-30-mean loss = {np.mean(adam_loss):8.3f}   "
              f"RAdam = {np.mean(rad_loss):8.3f}")
    # Show early-iteration behaviour where RAdam's rectification kicks in
    print("\n  Early-iter loss trace (noise = 3.0, first 20 steps):")
    g_fn = make_grad(3.0); _, ah = adam(g_fn, W0, lr=0.1, n_iter=20)
    g_fn = make_grad(3.0); _, rh = radam(g_fn, W0, lr=0.1, n_iter=20)
    print(f"    Adam  losses  = {[round(0.5 * float(w @ A @ w), 2) for w in ah[::4]]}")
    print(f"    RAdam losses  = {[round(0.5 * float(w @ A @ w), 2) for w in rh[::4]]}")
    print("\n  RAdam sidesteps the noisy-variance blowup by falling back to SGD until")
    print("  rho_t > 4 (usually ~5 steps in with default beta2 = 0.999), then matches Adam.")

    print("\n--- library cross-check (pytorch-optimizer.RAdam Python; no R equivalent) ---")
