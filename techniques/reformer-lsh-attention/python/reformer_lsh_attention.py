"""Reformer LSH attention (Kitaev, Kaiser & Levskaya 2020 ICLR).

Approximate softmax attention by hashing queries and keys into
buckets with locality-sensitive hashing (LSH):

    q, k -> bucket(q), bucket(k)  via random projections
    attend only within same bucket -> O(n log n) cost

The demo shows that only a fraction of Q x K entries are
computed while recovering most of the full-attention weight
mass on random Gaussian queries / keys.
"""

import numpy as np    # arrays + linalg


def full_softmax(Q, K, V):
    scores = Q @ K.T / np.sqrt(K.shape[1])
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    return p @ V, p


def lsh_hash(X, n_hashes, n_buckets, seed):
    """Random projection LSH: sign of X @ R -> concatenated bits."""
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((X.shape[1], n_hashes))
    signs = (X @ R) > 0    # (n, n_hashes)
    # Convert bit vectors to integer bucket in [0, n_buckets)
    weights = 2 ** np.arange(n_hashes)
    codes = signs @ weights
    return codes % n_buckets


def reformer_lsh_attention(Q, K, V, n_hashes=4, n_buckets=8, seed=0):
    n = Q.shape[0]
    Qc = lsh_hash(Q, n_hashes, n_buckets, seed)
    Kc = lsh_hash(K, n_hashes, n_buckets, seed)
    out = np.zeros((n, V.shape[1]))
    covered = np.zeros(n, dtype=bool)
    for b in range(n_buckets):
        q_mask = Qc == b
        k_mask = Kc == b
        if not q_mask.any() or not k_mask.any():
            continue
        scores = Q[q_mask] @ K[k_mask].T / np.sqrt(K.shape[1])
        p = np.exp(scores - scores.max(axis=1, keepdims=True))
        p = p / p.sum(axis=1, keepdims=True)
        out[q_mask] = p @ V[k_mask]
        covered[q_mask] = True
    # Fallback: any query in an empty bucket uses full attention.
    if not covered.all():
        missing = ~covered
        scores = Q[missing] @ K.T / np.sqrt(K.shape[1])
        p = np.exp(scores - scores.max(axis=1, keepdims=True))
        p = p / p.sum(axis=1, keepdims=True)
        out[missing] = p @ V
    return out


def demo():
    print("=== Reformer LSH attention (Kitaev et al 2020 ICLR) ===")
    rng = np.random.default_rng(2026)
    n, d = 256, 32
    # Q and K are TIED (Reformer): softmax then attends heavily to
    # nearby vectors, so LSH can preserve most of the mass.
    Q = rng.standard_normal((n, d)) * 2.0
    K = Q.copy()
    V = rng.standard_normal((n, d))

    O_full, _ = full_softmax(Q, K, V)
    for buckets in [4, 8, 16]:
        O_lsh = reformer_lsh_attention(Q, K, V, n_hashes=4, n_buckets=buckets, seed=1)
        rel = np.linalg.norm(O_full - O_lsh) / np.linalg.norm(O_full)
        pairs_full = n * n
        pairs_lsh = 0
        Qc = lsh_hash(Q, 4, buckets, 1)
        Kc = lsh_hash(K, 4, buckets, 1)
        for b in range(buckets):
            pairs_lsh = pairs_lsh + int((Qc == b).sum()) * int((Kc == b).sum())
        print(f"  buckets = {buckets:2d}: relative output error = {rel:.3f}, "
              f"score pairs = {pairs_lsh}/{pairs_full}"
              f" ({100 * pairs_lsh / pairs_full:.1f}%)")


if __name__ == "__main__":
    demo()
