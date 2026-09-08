"""MinHash + LSH (Reference Sec 47.76).

Broder 1997 'On the resemblance and containment of documents',
Compression and Complexity of Sequences. For two sets A, B:

    P(min h(A) = min h(B))  =  |A ∩ B| / |A ∪ B|  = J(A, B)

so K independent hashes give an UNBIASED Jaccard estimate with
variance ~1/K. Locality-Sensitive Hashing (Indyk-Motwani 1998):
split the K-length signature into b bands of r rows; declare a
candidate pair if any band matches -- gives O(n) approximate
near-neighbour retrieval.
"""
from __future__ import annotations    # stdlib

import hashlib    # deterministic hash
import numpy as np    # numerical arrays


def _hash(x, seed):
    h = hashlib.sha1(f"{seed}:{x}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def minhash_signature(shingles, K, seeds=None):
    """Return length-K MinHash signature for a set of shingles."""
    if seeds is None:
        seeds = list(range(K))
    return np.array([min(_hash(s, k) for s in shingles) for k in seeds], dtype=np.uint64)


def approx_jaccard(sig1, sig2):
    return float((sig1 == sig2).mean())


def lsh_index(sigs, b, r):
    """Bucket signatures by b bands of r rows into a dict of candidate lists."""
    K = sigs.shape[1]
    assert b * r == K, f"K = b*r required, got {K}, {b}, {r}"
    buckets = {}
    for i, sig in enumerate(sigs):
        for band in range(b):
            key = (band, tuple(sig[band * r:(band + 1) * r].tolist()))
            buckets.setdefault(key, []).append(i)
    candidates = set()
    for members in buckets.values():
        for a in members:
            for c in members:
                if a < c:
                    candidates.add((a, c))
    return candidates


def true_jaccard(A, B):
    A = set(A); B = set(B)
    return len(A & B) / max(len(A | B), 1)


if __name__ == "__main__":
    print("=== MinHash + LSH (Broder 1997; Indyk-Motwani 1998) ===\n")
    rng = np.random.default_rng(0)

    # 3 documents as sets of shingles
    docs = [
        set("the quick brown fox jumps over the lazy dog".split()),
        set("the quick brown fox jumps over the lazy cat".split()),
        set("machine learning is fun and productive".split()),
    ]

    for K in [16, 64, 256]:
        sigs = np.array([minhash_signature(d, K) for d in docs])
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            J_true = true_jaccard(docs[i], docs[j])
            J_est = approx_jaccard(sigs[i], sigs[j])
            print(f"  K={K:3d}  doc{i}-doc{j}  J_true={J_true:.3f}  J_est={J_est:.3f}   "
                  f"|err|={abs(J_true - J_est):.3f}")
        print()

    # LSH on many random shingled documents
    n_docs, K, b, r = 200, 128, 32, 4
    vocab = list(range(500))
    def rand_doc():
        return set(rng.choice(vocab, size=rng.integers(20, 50), replace=False))
    docs = [rand_doc() for _ in range(n_docs)]
    # Insert 5 duplicate pairs (perturb one word each)
    for k in range(5):
        docs[k + 100] = docs[k].copy()
        docs[k + 100].add(int(rng.integers(500, 1000)))
    sigs = np.array([minhash_signature(d, K) for d in docs])
    cand = lsh_index(sigs, b=b, r=r)
    hits = sum(1 for k in range(5) if (k, k + 100) in cand)
    print(f"  LSH (K=128, b=32, r=4): {len(cand)} candidate pairs among {n_docs} docs")
    print(f"  Duplicate pairs recovered: {hits} / 5")

    print("\n--- library cross-check (LSHR / textreuse R; datasketch Python) ---")
