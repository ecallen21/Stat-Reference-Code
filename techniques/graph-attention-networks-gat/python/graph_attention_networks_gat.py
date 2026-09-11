"""Graph Attention Networks - GAT (Reference Sec 47.150).

Velickovic et al 2018 'Graph Attention Networks', ICLR. Replaces
GCN's fixed symmetric-normalised adjacency with LEARNED attention
weights on edges:

    e_ij = LeakyReLU(a^T [W h_i || W h_j])
    alpha_ij = softmax_j(e_ij)                (over neighbours of i)
    h_i^(new) = sigma(sum_j alpha_ij W h_j).

Multi-head attention averaged / concatenated. Handles heterophilic
graphs better than GCN because different neighbours can be weighted
differently.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def leaky_relu(x, alpha=0.2):
    return np.where(x > 0, x, alpha * x)


def gat_layer(X, A, W, a, softmax_axis=1):
    """Single-head GAT layer forward pass."""
    n = X.shape[0]
    H = X @ W                                                  # (n, F')
    # Compute attention scores over edges
    scores = np.zeros((n, n))
    a1, a2 = a[:W.shape[1]], a[W.shape[1]:]                    # split a into two halves
    e_i = H @ a1                                                # (n,)
    e_j = H @ a2                                                # (n,)
    scores = leaky_relu(e_i[:, None] + e_j[None, :])
    scores = np.where(A + np.eye(n) > 0, scores, -1e9)          # mask non-edges
    alpha = np.exp(scores - scores.max(axis=1, keepdims=True))
    alpha = alpha / alpha.sum(axis=1, keepdims=True)
    return alpha @ H, alpha


def gat_train(X, A, y, train_mask, n_hidden=8, n_heads=4, lr=0.1, l2=5e-4,
                n_epochs=200, seed=0):
    """2-layer multi-head GAT via numerical gradient shortcut on W2 only."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    K = int(y.max()) + 1
    Ws1 = [rng.normal(scale=0.3, size=(d, n_hidden)) for _ in range(n_heads)]
    a1s = [rng.normal(scale=0.3, size=2 * n_hidden) for _ in range(n_heads)]
    W2 = rng.normal(scale=0.3, size=(n_hidden * n_heads, K))
    a2 = rng.normal(scale=0.3, size=2 * K)

    def forward():
        heads = [gat_layer(X, A, Ws1[h], a1s[h])[0] for h in range(n_heads)]
        H1 = np.maximum(np.concatenate(heads, axis=1), 0)      # ELU replaced by ReLU
        H2, _ = gat_layer(H1, A, W2, a2)
        z = H2 - H2.max(axis=1, keepdims=True)
        e = np.exp(z); P = e / e.sum(axis=1, keepdims=True)
        return H1, P

    Y = np.eye(K)[y]
    for ep in range(n_epochs):
        H1, P = forward()
        mask = train_mask.astype(float)[:, None]
        # Simplified update via W2 pseudo-gradient
        dP = (P - Y) * mask / max(mask.sum(), 1)
        W2 -= lr * (H1.T @ dP + l2 * W2)
        # Perturb head weights by a fraction of the same signal
        for h in range(n_heads):
            slice_ = slice(h * n_hidden, (h + 1) * n_hidden)
            grad = X.T @ ((dP @ W2[slice_].T) * (H1[:, slice_] > 0))
            Ws1[h] -= 0.3 * lr * grad + lr * l2 * Ws1[h]
    _, P = forward()
    return {"pred": np.argmax(P, axis=1), "P": P}


if __name__ == "__main__":
    print("=== Graph Attention Network - GAT (Velickovic et al 2018) ===\n")
    rng = np.random.default_rng(0)

    n_per = 40; K = 3; n = n_per * K
    y = np.repeat(np.arange(K), n_per)
    p_in, p_out = 0.15, 0.01
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if y[i] == y[j] else p_out
            if rng.uniform() < p:
                A[i, j] = A[j, i] = 1
    X = np.eye(K)[y] * 0.5 + rng.normal(scale=1.0, size=(n, K))
    train_mask = np.zeros(n, dtype=bool)
    for k in range(K):
        idx = np.where(y == k)[0][:5]
        train_mask[idx] = True

    for n_heads in [1, 4, 8]:
        r = gat_train(X, A, y, train_mask, n_hidden=8, n_heads=n_heads,
                        lr=0.2, n_epochs=200, seed=0)
        acc = float(np.mean(r["pred"][~train_mask] == y[~train_mask]))
        print(f"  n_heads = {n_heads:2d}   test acc = {acc:.3f}")

    from sklearn.linear_model import LogisticRegression
    print(f"\n  Baseline LR (no graph) test acc = "
          f"{LogisticRegression(max_iter=500).fit(X[train_mask], y[train_mask]).score(X[~train_mask], y[~train_mask]):.3f}")

    print("\n--- library cross-check (pytorch-geometric / dgl Python) ---")
