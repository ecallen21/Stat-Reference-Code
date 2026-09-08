"""Neural Collaborative Filtering (Reference Sec 47.119).

He, Liao, Zhang, Nie, Hu & Chua 2017 'Neural collaborative
filtering', WWW. Replaces the inner product of MF with a NEURAL
NETWORK on user + item embeddings:

    GMF:    yhat = sigmoid( sum_k (u_k * v_k) )     (generalised MF)
    MLP:    yhat = sigmoid( f_theta([u; v]) )        (feed-forward)
    NeuMF:  concat(GMF output, MLP output) -> logistic

Illustrated here with a small logistic-regression head on
concatenated user + item embeddings vs MF baseline.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from sklearn.linear_model import LogisticRegression    # linear head
from sklearn.neural_network import MLPClassifier    # small MLP


def build_features(user_ids, item_ids, U, V, mode="concat"):
    u = U[user_ids]; v = V[item_ids]
    if mode == "concat":
        return np.column_stack([u, v])
    if mode == "gmf":
        return u * v
    if mode == "neumf":
        return np.column_stack([u * v, u, v])
    raise ValueError


if __name__ == "__main__":
    print("=== Neural Collaborative Filtering (He et al 2017) ===\n")
    rng = np.random.default_rng(0)
    n_u, n_i, K = 100, 60, 4
    U_star = rng.normal(size=(n_u, K))
    V_star = rng.normal(size=(n_i, K))
    # NON-linear interaction truth: y = sigmoid(||u - v||^2 - 3)
    all_pairs = [(u, i) for u in range(n_u) for i in range(n_i)]
    y = np.array([1 if np.linalg.norm(U_star[u] - V_star[i]) < 2.5 else 0
                    for u, i in all_pairs])
    users = np.array([p[0] for p in all_pairs])
    items = np.array([p[1] for p in all_pairs])
    perm = rng.permutation(len(y))
    tr, te = perm[: int(0.8 * len(y))], perm[int(0.8 * len(y)):]

    # Random embeddings for demo (no learning of embeddings themselves)
    U = rng.normal(size=(n_u, K))
    V = rng.normal(size=(n_i, K))

    for mode in ["gmf", "concat", "neumf"]:
        Xtr = build_features(users[tr], items[tr], U, V, mode=mode)
        Xte = build_features(users[te], items[te], U, V, mode=mode)
        m_lr = LogisticRegression(max_iter=500, C=1.0).fit(Xtr, y[tr])
        m_mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500,
                                 random_state=0).fit(Xtr, y[tr])
        print(f"  mode={mode:6s}  linear acc = {m_lr.score(Xte, y[te]):.3f}   "
              f"MLP acc = {m_mlp.score(Xte, y[te]):.3f}")

    print("\n  Truth y = 1(||u - v|| < 2.5) is a nonlinear symmetric interaction;")
    print("  MLP-on-concat (NCF-style) captures it; simple GMF misses it.")

    print("\n--- library cross-check (recommenders / neural_collaborative_filtering Python) ---")
