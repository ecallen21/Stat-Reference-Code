"""Graph Convolutional Network — Kipf-Welling 2017 (Reference §27.x extra).

For a graph with adjacency A (with self-loops), node features X in R^{n x d}:

    A_hat = A + I
    D_hat_ii = sum_j A_hat_ij
    H^{l+1} = sigma( D_hat^{-1/2} A_hat D_hat^{-1/2} H^l W^l )

Each layer aggregates features from 1-hop neighbours (symmetric-normalised
mean); L stacked layers give an L-hop receptive field.

Semi-supervised node classification:
    * Small labelled subset of nodes; loss = cross-entropy on labelled nodes.
    * The unlabelled nodes still contribute via message passing to their
      labelled neighbours.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=1, keepdims=True)


def _relu(z): return np.maximum(z, 0.0)
def _relu_grad(z): return (z > 0).astype(float)


def _sym_norm(A):
    A_hat = A + np.eye(A.shape[0])
    d = A_hat.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-9))
    return (A_hat * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]


def train_gcn(A, X, y, train_mask, hidden: int = 16, n_classes: int = None,
              lr: float = 0.05, epochs: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    A_norm = _sym_norm(np.asarray(A, dtype=float))
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
    train_mask = np.asarray(train_mask, dtype=bool)
    n, d = X.shape
    if n_classes is None: n_classes = int(y.max()) + 1
    W1 = rng.normal(scale=np.sqrt(2.0 / d), size=(d, hidden))
    W2 = rng.normal(scale=np.sqrt(2.0 / hidden), size=(hidden, n_classes))
    Y = np.zeros((n, n_classes)); Y[np.arange(n), y] = 1
    for ep in range(epochs):
        H1_pre = A_norm @ X @ W1
        H1 = _relu(H1_pre)
        logits = A_norm @ H1 @ W2
        probs = _softmax(logits)
        # loss only on labelled nodes
        n_lab = int(train_mask.sum())
        d_logits = probs.copy(); d_logits[np.arange(n), y] -= 1
        d_logits[~train_mask] = 0.0
        d_logits /= max(n_lab, 1)
        # backward
        dW2 = (A_norm @ H1).T @ d_logits
        dH1 = A_norm.T @ (d_logits @ W2.T)
        dH1_pre = dH1 * _relu_grad(H1_pre)
        dW1 = (A_norm @ X).T @ dH1_pre
        W1 -= lr * dW1; W2 -= lr * dW2
    logits = A_norm @ _relu(A_norm @ X @ W1) @ W2
    return {"W1": W1, "W2": W2, "logits": logits,
            "method": "2-layer GCN (Kipf-Welling 2017)"}


def predict_gcn(X, A, model):
    A_norm = _sym_norm(np.asarray(A, dtype=float))
    return (A_norm @ _relu(A_norm @ X @ model["W1"]) @ model["W2"]).argmax(axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 3-community stochastic block model with node features aligned with community
    sizes = [10, 10, 10]; K = 3; n = sum(sizes)
    z = np.repeat(range(K), sizes)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            p = 0.5 if z[i] == z[j] else 0.05
            A[i, j] = A[j, i] = int(rng.uniform() < p)
    # node features = random noise (no per-node signal about class); the GCN
    # must use graph structure to classify.
    X = rng.normal(size=(n, 5))
    # labelling: 2 known labels per class
    train_mask = np.zeros(n, dtype=bool)
    for c in range(K):
        idxs = np.where(z == c)[0][:2]; train_mask[idxs] = True

    m = train_gcn(A, X, z, train_mask, hidden=8, epochs=300, lr=0.05)
    pred = predict_gcn(X, A, m)
    labelled_acc = float((pred[train_mask] == z[train_mask]).mean())
    unlabelled_acc = float((pred[~train_mask] == z[~train_mask]).mean())

    print(f"=== 2-layer GCN on 3-community SBM (n={n}, |train|={train_mask.sum()}) ===")
    print(f"  train (labelled)   accuracy = {labelled_acc:.3f}")
    print(f"  test (unlabelled)  accuracy = {unlabelled_acc:.3f}")

    # contrast: MLP on node features alone (no graph)
    def _mlp():
        rr = np.random.default_rng(0)
        W1 = rr.normal(scale=np.sqrt(2.0 / X.shape[1]), size=(X.shape[1], 8))
        W2 = rr.normal(scale=np.sqrt(2.0 / 8), size=(8, K))
        for _ in range(400):
            H = _relu(X @ W1); logits = H @ W2
            probs = _softmax(logits)
            d_logits = probs.copy(); d_logits[np.arange(n), z] -= 1
            d_logits[~train_mask] = 0.0; d_logits /= max(train_mask.sum(), 1)
            dW2 = H.T @ d_logits
            dH = d_logits @ W2.T
            dH_pre = dH * _relu_grad(X @ W1)
            dW1 = X.T @ dH_pre
            W1 -= 0.05 * dW1; W2 -= 0.05 * dW2
        return (_relu(X @ W1) @ W2).argmax(axis=1)
    pred_mlp = _mlp()
    print(f"\n  MLP on features only (no graph): unlabelled acc = "
          f"{(pred_mlp[~train_mask] == z[~train_mask]).mean():.3f}")

    print("\n--- library cross-check (torch_geometric.nn.GCNConv; dgl.nn.GraphConv) ---")
