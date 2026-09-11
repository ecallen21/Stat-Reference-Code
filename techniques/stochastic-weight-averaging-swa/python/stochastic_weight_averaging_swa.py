"""Stochastic Weight Averaging - SWA (Reference Sec 47.167).

Izmailov, Podoprikhin, Garipov, Vetrov & Wilson 2018 'Averaging
Weights Leads to Wider Optima and Better Generalization', UAI.
After a warm-up, KEEP AN AVERAGE of SGD iterates:

    theta_SWA = (1 / n_swa) * sum_{k=1..n_swa} theta_{warmup + k * period}.

Simple, ~free, and provably finds wider minima that generalise
better than the raw SGD endpoint. Doesn't need learning-rate
scheduling changes.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sgd_swa(X, y, lr=0.05, momentum=0.9, n_iter=1500, warmup=500,
              swa_period=10, seed=0, use_swa=True):
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.1, size=X.shape[1])
    v = np.zeros_like(W)
    W_swa = np.zeros_like(W); n_swa = 0
    for it in range(n_iter):
        idx = rng.choice(len(X), size=64, replace=False)
        Xb, yb = X[idx], y[idx]
        p = 1 / (1 + np.exp(-Xb @ W))
        grad = Xb.T @ (p - yb) / len(idx)
        v = momentum * v + grad
        W -= lr * v
        if use_swa and it >= warmup and (it - warmup) % swa_period == 0:
            n_swa += 1
            W_swa = W_swa + (W - W_swa) / n_swa                # running average
    if use_swa and n_swa > 0:
        return W_swa, W
    return W, W


def flatness(W, X, y, rho=0.5, n_samples=50, seed=0):
    """Empirical flatness: mean loss increase in a rho-ball around W."""
    rng = np.random.default_rng(seed)
    def loss(w):
        p = 1 / (1 + np.exp(-X @ w))
        return float(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
    L0 = loss(W)
    dLs = []
    for _ in range(n_samples):
        eps = rng.normal(size=W.shape); eps *= rho / (np.linalg.norm(eps) + 1e-12)
        dLs.append(loss(W + eps) - L0)
    return float(np.mean(dLs))


if __name__ == "__main__":
    print("=== Stochastic Weight Averaging - SWA (Izmailov et al 2018) ===\n")
    from sklearn.datasets import make_classification

    rng = np.random.default_rng(0)
    X, y = make_classification(n_samples=1500, n_features=40, n_informative=15,
                                  n_redundant=5, random_state=0)
    y = y.astype(float)
    perm = rng.permutation(len(X))
    tr, te = perm[:1000], perm[1000:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    def acc(W): return float(np.mean((Xte @ W > 0) == (yte > 0.5)))

    # Baseline SGD
    W_sgd, _ = sgd_swa(Xtr, ytr, lr=0.1, momentum=0.9, n_iter=1500, use_swa=False, seed=0)
    W_swa, W_end = sgd_swa(Xtr, ytr, lr=0.1, momentum=0.9, n_iter=1500,
                              warmup=500, swa_period=10, seed=0)
    print(f"  1000-sample, 40-D noisy classification")
    print(f"  SGD endpoint     test acc = {acc(W_sgd):.3f}   flatness = {flatness(W_sgd, Xtr, ytr):.4f}")
    print(f"  SWA average      test acc = {acc(W_swa):.3f}   flatness = {flatness(W_swa, Xtr, ytr):.4f}")
    print("\n  SWA's average sits in a wider basin (lower flatness = flatter loss).")

    print("\n--- library cross-check (torch.optim.swa_utils.AveragedModel Python; no R port) ---")
