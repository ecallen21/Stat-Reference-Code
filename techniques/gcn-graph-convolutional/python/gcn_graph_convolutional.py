"""Graph Convolutional Network (Reference Sec 47.148).

Kipf & Welling 2017 'Semi-Supervised Classification with Graph
Convolutional Networks', ICLR. Two-layer GCN:

    H^(1) = relu(  A_hat X W^(0) )
    H^(2) = softmax(A_hat H^(1) W^(1))

with A_hat = D^{-1/2} (A + I) D^{-1/2}   (symmetric normalisation
+ self-loops). Message passing / spectral graph convolution.

Semi-supervised: train cross-entropy on the labelled nodes only,
use graph structure to propagate to unlabelled ones.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def normalize_adj(A):
    """Symmetric-normalised adjacency with self-loops: D^-1/2 (A+I) D^-1/2."""
    A_tilde = A + np.eye(A.shape[0])
    d = A_tilde.sum(axis=1)
    d_inv_sqrt = 1 / np.sqrt(np.maximum(d, 1e-12))
    return A_tilde * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def gcn_train(X, A, y, train_mask, n_hidden=16, lr=0.1, l2=5e-4, n_epochs=200, seed=0):
    """2-layer GCN with cross-entropy loss on labelled nodes only."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    K = int(y.max()) + 1
    A_hat = normalize_adj(A)
    # Xavier init
    W0 = rng.normal(scale=np.sqrt(2 / (d + n_hidden)), size=(d, n_hidden))
    W1 = rng.normal(scale=np.sqrt(2 / (n_hidden + K)), size=(n_hidden, K))
    Y = np.eye(K)[y]
    for ep in range(n_epochs):
        # Forward
        H0 = A_hat @ X @ W0
        H0_relu = np.maximum(H0, 0)
        H1 = A_hat @ H0_relu @ W1
        P = softmax(H1)
        # Loss: cross-entropy on train_mask
        mask = train_mask.astype(float)[:, None]
        dL_dH1 = (P - Y) * mask / max(mask.sum(), 1)
        # Backprop
        dW1 = (A_hat @ H0_relu).T @ dL_dH1 + l2 * W1
        dH0_relu = A_hat.T @ dL_dH1 @ W1.T
        dH0 = dH0_relu * (H0 > 0)
        dW0 = (A_hat @ X).T @ dH0 + l2 * W0
        W0 -= lr * dW0
        W1 -= lr * dW1
    # Final predictions
    H0 = np.maximum(A_hat @ X @ W0, 0)
    P = softmax(A_hat @ H0 @ W1)
    return {"W0": W0, "W1": W1, "P": P, "pred": np.argmax(P, axis=1)}


if __name__ == "__main__":
    print("=== GCN (Kipf-Welling 2017) ===\n")
    rng = np.random.default_rng(0)

    # Synthetic 3-community stochastic block model
    n_per = 40; K = 3; n = n_per * K
    y = np.repeat(np.arange(K), n_per)
    p_in, p_out = 0.15, 0.01
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if y[i] == y[j] else p_out
            if rng.uniform() < p:
                A[i, j] = A[j, i] = 1
    # Node features: (weak, noisy) one-hot-ish per community + noise
    X = np.eye(K)[y] * 0.5 + rng.normal(scale=1.0, size=(n, K))
    # Semi-supervised: only 5 labels per class
    train_mask = np.zeros(n, dtype=bool)
    for k in range(K):
        idx = np.where(y == k)[0][:5]
        train_mask[idx] = True
    test_mask = ~train_mask

    r = gcn_train(X, A, y, train_mask, n_hidden=16, lr=0.5, n_epochs=200, seed=0)
    acc_test = float(np.mean(r["pred"][test_mask] == y[test_mask]))
    acc_train = float(np.mean(r["pred"][train_mask] == y[train_mask]))
    print(f"  3-community SBM: n = {n}, train nodes = {train_mask.sum()}")
    print(f"  GCN train acc = {acc_train:.3f}   test acc = {acc_test:.3f}")

    # Baseline: logistic on X alone (no graph structure)
    from sklearn.linear_model import LogisticRegression
    lr_bl = LogisticRegression(max_iter=500).fit(X[train_mask], y[train_mask])
    acc_baseline = lr_bl.score(X[test_mask], y[test_mask])
    print(f"  Baseline LR on X (no graph) test acc = {acc_baseline:.3f}")

    print("\n--- library cross-check (pytorch-geometric / dgl Python) ---")
