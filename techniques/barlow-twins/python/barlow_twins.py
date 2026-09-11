"""Barlow Twins (Reference Sec 47.154).

Zbontar, Jing, Misra, LeCun & Deny 2021 'Barlow Twins:
Self-Supervised Learning via Redundancy Reduction', ICML. Two views
of the same input are encoded and passed to a projector; the
CROSS-CORRELATION matrix C between the two projections is pushed
toward the identity:

    L = sum_i (1 - C_ii)^2   +   lambda * sum_{i != j} C_ij^2
            (invariance term)       (redundancy-reduction term)

No negative pairs, no memory bank -> simpler than SimCLR / MoCo but
competitive on ImageNet linear-probe benchmarks.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def barlow_loss(ZA, ZB, lam=5e-3):
    """Barlow Twins loss on two batches of embeddings (N, D)."""
    N, D = ZA.shape
    # Normalise each dim (batch-stats)
    ZA = (ZA - ZA.mean(0)) / (ZA.std(0) + 1e-6)
    ZB = (ZB - ZB.mean(0)) / (ZB.std(0) + 1e-6)
    C = (ZA.T @ ZB) / N                                        # (D, D)
    on_diag = ((np.diag(C) - 1) ** 2).sum()
    off_diag = (C ** 2).sum() - (np.diag(C) ** 2).sum()
    return float(on_diag + lam * off_diag), C


def augment(x, rng, sigma=0.3):
    return x + rng.normal(scale=sigma, size=x.shape)


def train_barlow(X, dim=16, lam=5e-3, lr=0.02, batch=64, n_iter=800, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = rng.normal(scale=0.3, size=(d, dim))
    for it in range(n_iter):
        idx = rng.choice(n, size=batch, replace=False)
        A = np.array([augment(X[i], rng) for i in idx])
        B = np.array([augment(X[i], rng) for i in idx])
        ZA, ZB = A @ W, B @ W
        # SPSA-style finite-difference gradient
        delta = rng.choice([-1, 1], size=W.shape) * 1e-3
        L_plus, _ = barlow_loss(A @ (W + delta), B @ (W + delta), lam=lam)
        L_minus, _ = barlow_loss(A @ (W - delta), B @ (W - delta), lam=lam)
        grad = (L_plus - L_minus) / (2 * 1e-3) * delta / (delta ** 2 + 1e-12)
        W -= lr * grad
    return W


if __name__ == "__main__":
    print("=== Barlow Twins - Redundancy Reduction (Zbontar et al 2021) ===\n")
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    data = load_digits(); X, y = data.data / 16.0, data.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    print(f"  digits:  n = {len(X)}, d = {X.shape[1]}, K = 10")

    # Baseline: LR on raw
    print(f"    LR on raw pixels (all labels):           "
          f"{LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte):.3f}")

    # Baseline: random 16-D projection
    rng = np.random.default_rng(0)
    Wr = rng.normal(scale=0.3, size=(X.shape[1], 16))
    print(f"    LR on random-projection 16-D features:  "
          f"{LogisticRegression(max_iter=2000).fit(Xtr @ Wr, ytr).score(Xte @ Wr, yte):.3f}")

    # Barlow Twins-trained embedding
    W_bt = train_barlow(X, dim=16, lam=5e-3, lr=0.05, batch=64, n_iter=500, seed=0)
    # Diagonal / off-diagonal check
    idx = rng.choice(len(X), size=128, replace=False)
    A = np.array([augment(X[i], rng) for i in idx])
    B = np.array([augment(X[i], rng) for i in idx])
    _, C = barlow_loss(A @ W_bt, B @ W_bt)
    print(f"    Cross-corr diag mean = {float(np.diag(C).mean()):.3f}   "
          f"off-diag |mean| = {float(np.mean(np.abs(C - np.diag(np.diag(C))))):.3f}")
    print(f"    LR on Barlow-Twins 16-D features (SSL):  "
          f"{LogisticRegression(max_iter=2000).fit(Xtr @ W_bt, ytr).score(Xte @ W_bt, yte):.3f}")

    print("\n--- library cross-check (lightly / solo-learn Python) ---")
