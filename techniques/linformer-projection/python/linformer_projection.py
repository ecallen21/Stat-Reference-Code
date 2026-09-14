"""Linformer projection (Wang, Li, Khabsa, Fang & Ma 2020).

Project the SEQUENCE dimension of K and V from n to k using
low-rank learned projections E_k, E_v (n x k):

    K_proj = E_k.T K,   V_proj = E_v.T V
    attention(Q, K_proj, V_proj)   # O(n k) instead of O(n^2)

For fixed k << n, complexity is LINEAR in n.  Random Gaussian
projections work well enough for a demo; learned ones are
better in practice.
"""

import numpy as np    # arrays


def full_attention(Q, K, V):
    scores = Q @ K.T / np.sqrt(K.shape[1])
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    return p @ V


def linformer_attention(Q, K, V, k):
    """Chunk-average projection (n -> k pooling) as a simple stand-in
    for Linformer's learned E_k, E_v projections."""
    n, d = K.shape
    E = np.zeros((n, k))
    for i in range(n):
        E[i, min(k - 1, i * k // n)] = 1.0
    E = E / E.sum(axis=0, keepdims=True)    # column-normalise
    K_proj = E.T @ K
    V_proj = E.T @ V
    scores = Q @ K_proj.T / np.sqrt(d)
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    return p @ V_proj


def demo():
    print("=== Linformer projection (Wang et al 2020) ===")
    rng = np.random.default_rng(2026)
    n, d = 512, 32
    # Diffuse attention — softmax matrix has low effective rank so
    # random-projection Linformer approximates it well.
    Q = rng.standard_normal((n, d)) / np.sqrt(d)
    K = rng.standard_normal((n, d)) / np.sqrt(d)
    V = rng.standard_normal((n, d))

    out_full = full_attention(Q, K, V)
    print(f"  n = {n}, d = {d}")
    for k in [16, 64, 128, 256]:
        out_lin = linformer_attention(Q, K, V, k=k)
        rel = np.linalg.norm(out_full - out_lin) / np.linalg.norm(out_full)
        pairs = n * k
        print(f"  k = {k:3d}: relative err = {rel:.4f}, "
              f"score pairs = {pairs}/{n * n} "
              f"({100 * pairs / (n * n):.1f}%)")


if __name__ == "__main__":
    demo()
