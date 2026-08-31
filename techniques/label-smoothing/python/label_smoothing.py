"""Label smoothing (Reference Ch 30 Robustness).

Szegedy, Vanhoucke, Ioffe, Shlens & Wojna (2016) "Rethinking the
Inception Architecture for Computer Vision."

Replace one-hot targets with a mixture:

  y_soft = (1 - eps) * one_hot(y) + eps / K.

Cross-entropy loss with soft targets = standard CE + eps * uniform-CE
term, which discourages the model from becoming arbitrarily confident.

Benefits:
  - Improves generalisation / test accuracy on CIFAR / ImageNet.
  - Better CALIBRATION -- reduces overconfidence.
  - Small robustness bonus against adversarial and distribution-shifted
    inputs.

Caveats:
  - Hurts model DISTILLATION (Muller 2019) because it collapses
    intra-class feature variance.
  - Optimal eps depends on # classes and data noise level.

Here we train a softmax classifier on a synthetic 3-class problem with
and without label smoothing, then compare ECE and NLL.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def train(X, y, K, eps=0.0, lr=0.5, epochs=400, l2=1e-3):
    d = X.shape[1]
    W = np.zeros((d, K))
    n = X.shape[0]
    y_one = np.eye(K)[y]
    y_soft = (1 - eps) * y_one + eps / K
    for _ in range(epochs):
        p = _softmax(X @ W)
        g = X.T @ (p - y_soft) / n + l2 * W
        W -= lr * g
    return W


def ece(probs, y_true, n_bins=10):
    """Expected Calibration Error over max-softmax confidence bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    err = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = (conf > edges[b]) & (conf <= edges[b + 1])
        if not mask.any(): continue
        gap = abs(conf[mask].mean() - correct[mask].mean())
        err += mask.sum() / n * gap
    return err


def nll(probs, y_true, eps=1e-12):
    return -np.log(probs[np.arange(len(y_true)), y_true] + eps).mean()


if __name__ == "__main__":
    print("=== Label smoothing (Szegedy 2016) ===\n")
    rng = np.random.default_rng(0)
    K = 3
    d = 4
    centers = rng.normal(0, 1.2, (K, d))    # closer centres -> harder problem
    n_tr, n_te = 200, 2000
    y_tr = rng.integers(0, K, n_tr)
    X_tr = centers[y_tr] + rng.normal(0, 1.0, (n_tr, d))
    y_te = rng.integers(0, K, n_te)
    X_te = centers[y_te] + rng.normal(0, 1.0, (n_te, d))
    # Add heavier label noise on train so the clean model overfits + becomes overconfident.
    flip = rng.random(n_tr) < 0.20
    y_tr[flip] = rng.integers(0, K, flip.sum())

    for eps in (0.0, 0.05, 0.10, 0.20):
        W = train(X_tr, y_tr, K, eps=eps, l2=0.0, epochs=2000)
        p = _softmax(X_te @ W)
        acc = (p.argmax(axis=1) == y_te).mean()
        e_ece = ece(p, y_te)
        e_nll = nll(p, y_te)
        max_conf = p.max(axis=1).mean()
        print(f"  eps={eps:.2f}  test_acc={acc:.3f}"
              f"  ECE={e_ece:.4f}   NLL={e_nll:.4f}   mean_confidence={max_conf:.3f}")

    print("\n  Label smoothing lowers mean confidence (its intended mechanic).")
    print("  Whether ECE improves depends on whether the base model was over- or under-confident;")
    print("  for over-parameterised nets that memorise training labels, LS usually reduces ECE.\n")
    print("--- library cross-check (torch.nn.CrossEntropyLoss(label_smoothing=eps); tf keras) ---")
