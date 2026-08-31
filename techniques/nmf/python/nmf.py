"""Non-negative matrix factorisation (NMF) — Reference Sec 25.2.

Lee & Seung (1999) 'Learning the parts of objects by non-negative
matrix factorization.'

Factorise a non-negative matrix V (n x d) as V ~= W @ H with W (n x k)
and H (k x d) BOTH NON-NEGATIVE. Non-negativity forces additive
combinations -> interpretable 'parts-based' representations (topic
models, spectra, gene programmes, images).

Multiplicative-update rules (Lee-Seung 2001):

  H <- H * (W' V) / (W' W H + eps)
  W <- W * (V H') / (W H H' + eps)

Objective: || V - W H ||_F^2.

Here we implement multiplicative updates and demonstrate topic-like
decomposition on a small synthetic term-document matrix.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def fit_nmf(V, k, max_iter=300, seed=0, eps=1e-9):
    rng = np.random.default_rng(seed)
    n, d = V.shape
    W = rng.uniform(0, 1, (n, k))
    H = rng.uniform(0, 1, (k, d))
    losses = []
    for _ in range(max_iter):
        H = H * (W.T @ V) / (W.T @ W @ H + eps)
        W = W * (V @ H.T) / (W @ H @ H.T + eps)
        losses.append(float(np.linalg.norm(V - W @ H, ord="fro") ** 2))
    return W, H, losses


if __name__ == "__main__":
    print("=== NMF: Lee-Seung multiplicative updates ===\n")
    rng = np.random.default_rng(0)
    # Synthetic 'topic-word' matrix: 6 documents, 5 words, 2 topics.
    words = ["dog", "cat", "law", "trial", "lawyer"]
    # topic 1 = animals; topic 2 = law
    W_true = np.array([
        [4, 3, 0, 0, 0],   # animals doc
        [3, 4, 0, 0, 0],
        [5, 2, 0, 0, 0],
        [0, 0, 4, 5, 3],   # law doc
        [0, 0, 3, 4, 5],
        [0, 0, 5, 3, 4],
    ], dtype=float)
    V = W_true + 0.2 * rng.uniform(0, 1, W_true.shape)     # add small non-neg noise
    print("  input matrix V (docs x words):")
    print(f"  words: {words}")
    print(V.round(2))
    print()

    W, H, losses = fit_nmf(V, k=2, max_iter=300, seed=0)
    print(f"  W (docs x topics):\n{W.round(2)}\n")
    print(f"  H (topics x words):\n{H.round(2)}")
    print(f"  Word interpretations:")
    for tid in range(2):
        top_words = np.argsort(-H[tid])[:3]
        print(f"    topic {tid}: {[words[i] for i in top_words]}   weights {H[tid, top_words].round(2).tolist()}")

    print(f"\n  reconstruction error (Frobenius): {losses[-1]:.4f}"
          f"   (start {losses[0]:.4f})\n")
    print("--- library cross-check (sklearn.decomposition.NMF; nimfa; R NMF) ---")
