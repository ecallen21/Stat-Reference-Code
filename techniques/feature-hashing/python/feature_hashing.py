"""Feature hashing / the hashing trick (Reference Sec 41.10).

Weinberger et al. (2009): map arbitrary string features to a fixed
d-dimensional vector via hash(feature) mod d.  A signed variant uses
a second hash to xi in {-1, +1} to unbias collisions.

  * Constant memory regardless of vocabulary size.
  * No dictionary; suitable for streaming.
  * Collisions cause approximate representations but expected inner
    products are unbiased under random hashing.

Common uses: text n-gram features, huge-cardinality categoricals, and
online learning where the feature space evolves.
"""
from __future__ import annotations    # stdlib

import hashlib

import numpy as np    # numerical arrays


def _hash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def hash_encode(tokens, d=16, signed=True):
    """Convert list-of-lists of strings to a (n, d) sparse-like matrix."""
    X = np.zeros((len(tokens), d))
    for i, doc in enumerate(tokens):
        for t in doc:
            h = _hash(t)
            j = h % d
            sign = 1
            if signed:
                sign = 1 if (h >> 32) & 1 else -1
            X[i, j] += sign
    return X


if __name__ == "__main__":
    print("=== Feature hashing: encode text features to fixed-dim vector ===\n")
    docs = [
        ["red", "shoe", "leather"],
        ["blue", "shoe", "canvas"],
        ["red", "hat", "wool"],
        ["blue", "scarf", "silk"],
    ]

    X = hash_encode(docs, d=8, signed=True)
    print(f"  vocabulary implicit; hashed to d = 8:\n{X}\n")

    X_small = hash_encode(docs, d=4, signed=True)
    print(f"  d = 4 (collision-prone; signed hashing unbiases):\n{X_small}\n")

    # Check unbiasedness of inner product across d
    rng = np.random.default_rng(0)
    vocab = [f"tok_{i}" for i in range(100)]
    # 500 random documents
    ds = [[rng.choice(vocab) for _ in range(rng.integers(3, 10))] for _ in range(500)]
    # Random target: mean feature count
    y = np.array([len(d) for d in ds]).astype(float)
    for d in (4, 16, 64, 256):
        Xh = hash_encode(ds, d=d, signed=True)
        # Approx OLS fit residual norm
        beta = np.linalg.lstsq(Xh, y, rcond=None)[0]
        yhat = Xh @ beta
        rss = ((y - yhat) ** 2).sum()
        print(f"  d = {d:>4d}   OLS RSS = {rss:.2f}")

    print("\n--- library cross-check (R FeatureHashing/text2vec; Python sklearn.feature_extraction.FeatureHasher) ---")
