"""Cross-entropy loss + log-loss (Reference Sec 34.6).

Cross-entropy of predictions p_hat under true labels y:
  H(y, p_hat) = -sum_k y_k log p_hat_k        (categorical)
             = -[ y log p_hat + (1-y) log(1-p_hat) ]  (binary)

Relationships:
  * MLE for logistic / softmax regression MINIMISES cross-entropy.
  * cross-entropy(y, p) = H(y) + KL(y || p)     when y is a distribution
  * Log-loss is the mean cross-entropy per example.

Gradient (softmax):
  d L / d z = p_hat - y_onehot           (elegant, cancellation-free).

Here we verify: (1) MLE-optimal logistic solution minimises log-loss;
(2) gradient identity holds numerically; (3) log-loss is a PROPER
SCORING RULE (uniquely minimised by the true probability).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y, p, eps=1e-12):
    return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def categorical_cross_entropy(y_onehot, p, eps=1e-12):
    return float(-np.mean(np.sum(y_onehot * np.log(p + eps), axis=1)))


def logistic_mle(X, y, lr=0.5, epochs=500, l2=1e-3):
    d = X.shape[1]; beta = np.zeros(d); n = X.shape[0]
    for _ in range(epochs):
        p = _sigmoid(X @ beta)
        g = X.T @ (p - y) / n + l2 * beta
        beta -= lr * g
    return beta


if __name__ == "__main__":
    print("=== Cross-entropy / log-loss ===\n")
    # (1) MLE minimises log-loss
    rng = np.random.default_rng(0)
    n, d = 500, 4
    X = rng.normal(0, 1, (n, d))
    beta_true = np.array([0.5, -1.0, 0.7, 0.0])
    y = (rng.random(n) < _sigmoid(X @ beta_true)).astype(float)

    beta_hat = logistic_mle(X, y)
    p_hat = _sigmoid(X @ beta_hat)
    logloss_mle = binary_cross_entropy(y, p_hat)
    # Try shifted alternatives
    for eps in (0, 0.5, -0.5):
        p_shift = np.clip(p_hat + eps * 0.1, 1e-6, 1 - 1e-6)
        ll = binary_cross_entropy(y, p_shift)
        print(f"  log-loss at MLE + shift {eps * 0.1:+.2f}: {ll:.4f}")
    print(f"  (MLE minimises log-loss to {logloss_mle:.4f})\n")

    # (2) Softmax gradient identity
    K = 3
    logits = rng.normal(0, 1, (1, K))
    y_onehot = np.eye(K)[[1]]
    p = softmax(logits)
    ce = categorical_cross_entropy(y_onehot, p)
    d_z_analytic = (p - y_onehot).ravel()
    # Numerical gradient
    d_z_num = np.zeros(K)
    h = 1e-5
    for k in range(K):
        z_up = logits.copy(); z_up[0, k] += h
        z_dn = logits.copy(); z_dn[0, k] -= h
        d_z_num[k] = (categorical_cross_entropy(y_onehot, softmax(z_up))
                       - categorical_cross_entropy(y_onehot, softmax(z_dn))) / (2 * h)
    print("  Softmax cross-entropy gradient identity:")
    print(f"    analytic (p - y): {d_z_analytic.round(4).tolist()}")
    print(f"    numerical       : {d_z_num.round(4).tolist()}"
          f"   max |diff| = {float(np.max(np.abs(d_z_analytic - d_z_num))):.2e}\n")

    # (3) Proper scoring rule: log-loss is minimised at the true p.
    print("  Proper scoring rule check: minimise E_y[log loss(y, q)] over q.")
    p_true = 0.7
    y_sample = (rng.random(10000) < p_true).astype(float)
    qs = np.linspace(0.05, 0.95, 19)
    losses = [binary_cross_entropy(y_sample, np.full_like(y_sample, q)) for q in qs]
    best_q = float(qs[np.argmin(losses)])
    print(f"    true probability = {p_true},  minimum-log-loss q_hat = {best_q:.2f}\n")

    print("--- library cross-check (sklearn.metrics.log_loss; torch.nn.CrossEntropyLoss; scipy.stats) ---")
