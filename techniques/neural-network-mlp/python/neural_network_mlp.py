"""Multi-Layer Perceptron with backprop (Reference §27.1).

Feedforward neural network:
    h_l = phi(W_l h_{l-1} + b_l),   l = 1, ..., L
    y_hat = softmax(W_L h_{L-1} + b_L)   (classification)
                or W_L h_{L-1} + b_L    (regression)

Training via mini-batch SGD with backpropagation:
    loss = cross-entropy (classification) or MSE (regression)
    grad computed by chain rule; parameters updated by
        W <- W - lr * grad_W

Common activations: ReLU (default), tanh, sigmoid.  Regularization: L2
weight decay, dropout, early stopping.

The demo below implements a one-hidden-layer MLP with ReLU + softmax for
multi-class classification, trained by full-batch SGD.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=1, keepdims=True)


def _relu(z): return np.maximum(z, 0)


def mlp_classifier(X, y, hidden: int = 32, lr: float = 0.05,
                    n_iter: int = 2000, seed: int = 0) -> dict:
    """One-hidden-layer MLP with ReLU + softmax; full-batch gradient descent."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
    n, p = X.shape; K = int(y.max() + 1)
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, math.sqrt(2 / p), (p, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, math.sqrt(2 / hidden), (hidden, K))
    b2 = np.zeros(K)
    Y = np.eye(K)[y]                              # one-hot
    losses = []
    for t in range(n_iter):
        # Forward
        Z1 = X @ W1 + b1; H = _relu(Z1)
        Z2 = H @ W2 + b2; P = _softmax(Z2)
        loss = -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))
        losses.append(loss)
        # Backward
        dZ2 = (P - Y) / n
        dW2 = H.T @ dZ2; db2 = dZ2.sum(axis=0)
        dH = dZ2 @ W2.T
        dZ1 = dH * (Z1 > 0)
        dW1 = X.T @ dZ1; db1 = dZ1.sum(axis=0)
        # SGD update
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=float)
        H = _relu(X_new @ W1 + b1)
        return np.argmax(_softmax(H @ W2 + b2), axis=1)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2,
            "loss_history": losses, "predict": predict,
            "n_iter": int(n_iter),
            "method": "1-hidden-layer MLP (ReLU + softmax) via SGD"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal([0, 0], 0.6, (150, 2)),
                    rng.normal([3, 0], 0.6, (150, 2)),
                    rng.normal([1.5, 3], 0.6, (150, 2))])
    y = np.repeat([0, 1, 2], 150)

    fit = mlp_classifier(X, y, hidden=32, lr=0.05, n_iter=2000, seed=0)
    acc = (fit["predict"](X) == y).mean()
    print(f"=== MLP (1 x 32 hidden, ReLU + softmax, 2000 iters) ===")
    print(f"  final loss = {fit['loss_history'][-1]:.4f}")
    print(f"  training accuracy = {acc:.3f}")

    print("\n--- library cross-check (sklearn MLPClassifier) ---")
    try:
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=0).fit(X, y)
        print(f"  sklearn MLP training accuracy = {m.score(X, y):.3f}")
    except Exception as ex:
        print(f"  (sklearn unavailable: {ex})")
