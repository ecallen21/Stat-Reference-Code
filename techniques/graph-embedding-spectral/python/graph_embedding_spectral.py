"""Spectral graph embedding (Reference §24.9).

Embed each node of a graph as a d-dimensional vector.  Three classical
spectral routes:

  * Laplacian eigenmaps (Belkin-Niyogi 2003): embed via the d bottom
    non-trivial eigenvectors of the graph Laplacian L = D - A.
  * Normalised-Laplacian embedding: eigenvectors of L_sym = I - D^{-1/2} A D^{-1/2}.
  * Adjacency spectral embedding: top-d eigenvectors of A scaled by sqrt(|eig|).

Downstream uses: node clustering (k-means on embedding), link prediction
(inner product), classification (feature vectors for a supervised head).

Non-spectral alternatives (DeepWalk, node2vec, graph neural networks) trade
closed-form solutions for scalability and richer capacity.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def laplacian_embedding(A, d: int = 2, normalized: bool = True) -> dict:
    A = np.asarray(A, dtype=float); n = A.shape[0]
    deg = A.sum(axis=1)
    if normalized:
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
        L = np.eye(n) - (A * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    else:
        L = np.diag(deg) - A
    w, V = np.linalg.eigh(L)
    # skip the trivial zero eigenvalue (constant vector)
    U = V[:, 1: 1 + d]
    return {"embedding": U, "eigenvalues": w[: 1 + d].tolist(),
            "method": "Laplacian eigenmaps"
                       + (" (normalized)" if normalized else "")}


def adjacency_spectral_embedding(A, d: int = 2) -> dict:
    A = np.asarray(A, dtype=float)
    w, V = np.linalg.eigh(A)
    # take d largest by absolute value
    order = np.argsort(-np.abs(w))
    top = order[:d]
    U = V[:, top] * np.sqrt(np.abs(w[top]))
    return {"embedding": U, "eigenvalues": w[top].tolist(),
            "method": "adjacency spectral embedding"}


def _kmeans(X, k: int, n_iter: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), k, replace=False); centres = X[idx].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        new_labels = d2.argmin(axis=1)
        if (new_labels == labels).all():
            break
        labels = new_labels
        for c in range(k):
            if (labels == c).any():
                centres[c] = X[labels == c].mean(axis=0)
    return labels


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 3-block SBM (assortative, mid-density)
    sizes = [15, 15, 15]; K = len(sizes); n = sum(sizes)
    z = np.repeat(range(K), sizes)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            p = 0.5 if z[i] == z[j] else 0.05
            A[i, j] = A[j, i] = int(rng.uniform() < p)

    print("=== Spectral embedding + k-means on 3-block SBM (n=45) ===")
    for name, fn in [("Laplacian (norm)", lambda: laplacian_embedding(A, d=2, normalized=True)),
                     ("Laplacian (unnorm)", lambda: laplacian_embedding(A, d=2, normalized=False)),
                     ("Adjacency spectral", lambda: adjacency_spectral_embedding(A, d=2))]:
        emb = fn()["embedding"]
        lbl = _kmeans(emb, K, seed=1)
        # majority-vote alignment
        from collections import Counter
        remap = {c: Counter(z[lbl == c]).most_common(1)[0][0] for c in np.unique(lbl)}
        acc = float(np.mean(np.array([remap[c] for c in lbl]) == z))
        print(f"  {name:>22}: cluster accuracy = {acc:.3f}   "
              f"embedding shape = {emb.shape}")

    print("\n--- library cross-check (scikit-learn SpectralEmbedding) ---")
    try:
        from sklearn.manifold import SpectralEmbedding
        se = SpectralEmbedding(n_components=2, affinity="precomputed", random_state=0)
        emb = se.fit_transform(A)
        lbl = _kmeans(emb, K, seed=1)
        remap = {c: Counter(z[lbl == c]).most_common(1)[0][0] for c in np.unique(lbl)}
        acc = float(np.mean(np.array([remap[c] for c in lbl]) == z))
        print(f"  sklearn SpectralEmbedding cluster accuracy = {acc:.3f}")
    except ImportError:
        print("  (scikit-learn not installed)")
