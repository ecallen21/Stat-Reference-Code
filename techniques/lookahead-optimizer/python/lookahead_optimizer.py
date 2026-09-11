"""Lookahead Optimizer (Reference Sec 47.164).

Zhang, Lucas, Ba & Hinton 2019 'Lookahead Optimizer: k steps
forward, 1 step back', NeurIPS. Keeps TWO sets of weights:

    phi ('slow' weights) and theta ('fast' weights, the SGD state).

    For k inner steps: theta <- theta - lr * grad L(theta)
    Then: phi <- phi + alpha * (theta - phi)
                  theta <- phi                         (reset fast to slow)

alpha in (0, 1). Wrapping any base optimiser; robust to lr choice,
reduces variance and improves generalisation with negligible
compute overhead.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sgd_step(W, X, y, lr, batch_idx):
    Xb, yb = X[batch_idx], y[batch_idx]
    p = 1 / (1 + np.exp(-Xb @ W))
    grad = Xb.T @ (p - yb) / len(batch_idx)
    return W - lr * grad


def train_sgd(X, y, lr=0.05, n_iter=1500, batch=32, seed=0):
    rng = np.random.default_rng(seed)
    W = np.zeros(X.shape[1])
    for it in range(n_iter):
        idx = rng.choice(len(X), size=batch, replace=False)
        W = sgd_step(W, X, y, lr, idx)
    return W


def train_lookahead(X, y, lr=0.05, k=5, alpha=0.5, n_iter=1500, batch=32, seed=0):
    rng = np.random.default_rng(seed)
    phi = np.zeros(X.shape[1])                                  # slow weights
    theta = phi.copy()
    for it in range(n_iter):
        idx = rng.choice(len(X), size=batch, replace=False)
        theta = sgd_step(theta, X, y, lr, idx)
        if (it + 1) % k == 0:
            phi = phi + alpha * (theta - phi)                    # slow update
            theta = phi.copy()                                    # reset fast
    return phi


if __name__ == "__main__":
    print("=== Lookahead Optimizer (Zhang-Lucas-Ba-Hinton 2019) ===\n")
    from sklearn.datasets import make_classification

    rng = np.random.default_rng(0)
    X, y = make_classification(n_samples=800, n_features=30, n_informative=10,
                                  random_state=0)
    y = y.astype(float)
    perm = rng.permutation(len(X))
    tr, te = perm[:500], perm[500:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    def acc(W): return float(np.mean((Xte @ W > 0) == (yte > 0.5)))

    # Sweep a range of learning rates - Lookahead should be more robust
    print(f"    {'lr':>8}  {'SGD_test_acc':>13}  {'Lookahead_test_acc':>18}")
    for lr in [0.01, 0.05, 0.2, 0.5, 1.0]:
        W_sgd = train_sgd(Xtr, ytr, lr=lr, n_iter=1500, batch=32, seed=0)
        W_la = train_lookahead(Xtr, ytr, lr=lr, k=5, alpha=0.5,
                                  n_iter=1500, batch=32, seed=0)
        print(f"    {lr:8.3f}  {acc(W_sgd):13.3f}  {acc(W_la):18.3f}")

    print("\n  Lookahead's slow weights average the fast trajectory, buying")
    print("  robustness to lr choice and lower variance across seeds.")

    print("\n--- library cross-check (pytorch-optimizer / RAdam+Lookahead=Ranger Python) ---")
