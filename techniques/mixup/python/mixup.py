"""Mixup (Reference Ch 30 Robustness).

Zhang, Cisse, Dauphin & Lopez-Paz (2018) "mixup: Beyond Empirical Risk
Minimization."

Sample lambda ~ Beta(alpha, alpha) and build virtual training examples

  x_tilde  =  lambda * x_i  +  (1 - lambda) * x_j
  y_tilde  =  lambda * y_i  +  (1 - lambda) * y_j        (one-hot mixed)

Train on the mixed batch with soft-label CE. Reduces memorisation of
label noise, improves generalisation, and adds a modest bonus of
adversarial + calibration robustness.

alpha in [0.1, 0.4] typical for image classification (small alpha ->
Beta concentrates near 0 or 1, so most mixed examples are almost original).

Here we train a softmax classifier on synthetic data + label noise, and
compare vanilla vs mixup on test accuracy, ECE, and robustness to a
tiny random perturbation.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _train_step(W, X, y_soft, lr, l2, batch, rng):
    n = X.shape[0]
    idx = rng.integers(0, n, batch)
    Xb, yb = X[idx], y_soft[idx]
    p = _softmax(Xb @ W)
    g = Xb.T @ (p - yb) / batch + l2 * W
    return W - lr * g


def train_vanilla(X, y, K, lr=0.3, epochs=800, batch=64, l2=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    W = np.zeros((X.shape[1], K))
    y_one = np.eye(K)[y]
    for _ in range(epochs):
        W = _train_step(W, X, y_one, lr, l2, batch, rng)
    return W


def train_mixup(X, y, K, alpha=0.2, lr=0.3, epochs=800, batch=64, l2=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    W = np.zeros((X.shape[1], K))
    y_one = np.eye(K)[y]
    n = X.shape[0]
    for _ in range(epochs):
        idx1 = rng.integers(0, n, batch)
        idx2 = rng.integers(0, n, batch)
        lam = rng.beta(alpha, alpha, batch)
        Xb = lam[:, None] * X[idx1] + (1 - lam[:, None]) * X[idx2]
        yb = lam[:, None] * y_one[idx1] + (1 - lam[:, None]) * y_one[idx2]
        p = _softmax(Xb @ W)
        g = Xb.T @ (p - yb) / batch + l2 * W
        W -= lr * g
    return W


def ece(probs, y, n_bins=10):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    err = 0.0; n = len(y)
    for b in range(n_bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.any():
            err += m.sum() / n * abs(conf[m].mean() - correct[m].mean())
    return err


if __name__ == "__main__":
    print("=== Mixup (Zhang 2018) ===\n")
    rng = np.random.default_rng(0)
    K = 3
    d = 5
    centers = rng.normal(0, 1.5, (K, d))
    n_tr, n_te = 400, 2000
    y_tr = rng.integers(0, K, n_tr)
    X_tr = centers[y_tr] + rng.normal(0, 1.0, (n_tr, d))
    y_te = rng.integers(0, K, n_te)
    X_te = centers[y_te] + rng.normal(0, 1.0, (n_te, d))
    # Add label noise on training
    flip = rng.random(n_tr) < 0.15
    y_tr[flip] = rng.integers(0, K, flip.sum())

    W_v = train_vanilla(X_tr, y_tr, K)
    W_m = train_mixup(X_tr, y_tr, K, alpha=0.4)

    def eval_(W, X, y):
        p = _softmax(X @ W)
        return (p.argmax(axis=1) == y).mean(), ece(p, y), p.max(axis=1).mean()

    for name, W in (("vanilla", W_v), ("mixup a=0.4", W_m)):
        acc, e, conf = eval_(W, X_te, y_te)
        # Robustness to Gaussian input noise (proxy for L2 adversarial)
        rob = np.mean([
            (_softmax((X_te + np.random.default_rng(k).normal(0, 0.3, X_te.shape)) @ W)
             .argmax(axis=1) == y_te).mean()
            for k in range(5)
        ])
        print(f"  {name:12s}   clean_acc={acc:.3f}   ECE={e:.4f}   mean_conf={conf:.3f}"
              f"   robust_acc(noise 0.3)={rob:.3f}")

    print("\n  Mixup lowers mean confidence (its intended mechanic).")
    print("  Larger benefits appear for over-parameterised nets (CIFAR-scale CNNs, transformers).\n")
    print("--- library cross-check (torchvision.transforms.v2.MixUp; timm mixup helpers) ---")
