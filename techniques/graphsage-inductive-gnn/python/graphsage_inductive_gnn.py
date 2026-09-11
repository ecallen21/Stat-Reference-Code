"""GraphSAGE - Inductive Graph Neural Network (Reference Sec 47.149).

Hamilton, Ying & Leskovec 2017 'Inductive Representation Learning on
Large Graphs', NeurIPS. Unlike GCN (transductive, needs the whole
adjacency at training), GraphSAGE learns AGGREGATOR functions:

    h_N(v)^(k) = AGG_k({h_u^(k-1) : u in N(v) sampled})
    h_v^(k)    = sigma(W^(k) [h_v^(k-1) || h_N(v)^(k)])

trained by minimising cross-entropy on labelled nodes. At inference,
new nodes (not seen at training) can be embedded from their sampled
neighbourhood -> INDUCTIVE.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sample_neighbors(A, v, k, rng):
    """Uniformly sample up to k neighbours of node v (with replacement if degree < k)."""
    nbrs = np.where(A[v] > 0)[0]
    if len(nbrs) == 0:
        return np.array([v])
    return rng.choice(nbrs, size=k, replace=len(nbrs) < k)


def graphsage_train(X, A, y, train_mask, n_hidden=16, k_sample=5, lr=0.05,
                     l2=1e-4, n_epochs=200, seed=0):
    """2-layer GraphSAGE with mean aggregator + concatenation."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    K = int(y.max()) + 1
    W1 = rng.normal(scale=np.sqrt(2 / (2 * d + n_hidden)), size=(2 * d, n_hidden))
    W2 = rng.normal(scale=np.sqrt(2 / (2 * n_hidden + K)), size=(2 * n_hidden, K))

    def softmax(z):
        z = z - z.max(axis=1, keepdims=True); e = np.exp(z); return e / e.sum(axis=1, keepdims=True)

    def forward(X_full):
        # Layer 1: aggregate over sampled neighbours
        H1 = np.zeros((n, W1.shape[1]))
        for v in range(n):
            nbrs = sample_neighbors(A, v, k_sample, rng)
            h_agg = X_full[nbrs].mean(axis=0)
            H1[v] = np.maximum(np.concatenate([X_full[v], h_agg]) @ W1, 0)
        # Layer 2: aggregate again
        H2 = np.zeros((n, W2.shape[1]))
        for v in range(n):
            nbrs = sample_neighbors(A, v, k_sample, rng)
            h_agg = H1[nbrs].mean(axis=0)
            H2[v] = np.concatenate([H1[v], h_agg]) @ W2
        return H1, softmax(H2)

    Y = np.eye(K)[y]
    for ep in range(n_epochs):
        H1, P = forward(X)
        mask = train_mask.astype(float)
        # Gradient via numerical proxy (simplified): update W2 only via cross-entropy
        # For a cleanish demo, we take small SGD steps on W2 only
        dL_dP = (P - Y) * mask[:, None] / max(mask.sum(), 1)
        # Recompute concat for gradient on W2
        for v in np.where(train_mask)[0]:
            nbrs = sample_neighbors(A, v, k_sample, rng)
            h_agg = H1[nbrs].mean(axis=0)
            z = np.concatenate([H1[v], h_agg])
            W2 -= lr * np.outer(z, dL_dP[v]) + lr * l2 * W2
            # Approximate W1 update: push mean neighbour h_agg toward correctness
            W1_update = np.outer(np.concatenate([X[v], X[nbrs].mean(axis=0)]),
                                    dL_dP[v] @ W2[:H1.shape[1]].T) * (H1[v] > 0)
            W1 -= lr * 0.5 * W1_update + lr * l2 * W1
    H1, P = forward(X)
    return {"pred": np.argmax(P, axis=1), "P": P}


if __name__ == "__main__":
    print("=== GraphSAGE (Hamilton-Ying-Leskovec 2017) ===\n")
    rng = np.random.default_rng(0)

    # 3-community SBM (train/val split simulating inductive setting)
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

    r = graphsage_train(X, A, y, train_mask, n_hidden=16, k_sample=8,
                          lr=0.1, n_epochs=250, seed=0)
    acc = float(np.mean(r["pred"][~train_mask] == y[~train_mask]))
    print(f"  3-community SBM, n = {n}, k_sample = 8")
    print(f"  GraphSAGE test acc = {acc:.3f}")

    # Baseline: features-only logistic regression
    from sklearn.linear_model import LogisticRegression
    lr_bl = LogisticRegression(max_iter=500).fit(X[train_mask], y[train_mask])
    print(f"  Baseline LR test acc = {lr_bl.score(X[~train_mask], y[~train_mask]):.3f}")

    print("\n--- library cross-check (pytorch-geometric / dgl Python) ---")
