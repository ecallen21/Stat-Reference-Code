"""Entity embeddings for categorical inputs (Guo-Berkhahn 2016; Reference §27.12).

Turn a categorical variable (K levels) into a learnable dense vector of
dimension d << K -- an "entity embedding".  Advantages over one-hot:
  * fewer parameters when combined with a downstream layer,
  * captures similarity between levels (levels that behave similarly get
    similar vectors),
  * transferable across tasks.

Implementation: a K x d lookup matrix; forward pass = row indexing; gradient
= scatter-add.  Train jointly with the downstream head.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=1, keepdims=True)


def train_entity_embedding(cat_ids, y, K: int, dim: int = 4,
                            n_classes: int = None, lr: float = 0.05,
                            epochs: int = 200, seed: int = 0) -> dict:
    """cat_ids: (N,)  y: (N,).  Single categorical + softmax classifier."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    if n_classes is None:
        n_classes = int(y.max()) + 1
    E = rng.normal(scale=0.1, size=(K, dim))              # embedding matrix
    W = rng.normal(scale=0.1, size=(dim, n_classes))
    b = np.zeros(n_classes)
    for ep in range(epochs):
        emb = E[cat_ids]                                  # (N, dim)
        logits = emb @ W + b
        probs = _softmax(logits)
        Y = np.zeros_like(probs); Y[np.arange(len(y)), y] = 1
        d_logits = (probs - Y) / len(y)
        d_W = emb.T @ d_logits; d_b = d_logits.sum(axis=0)
        d_emb = d_logits @ W.T
        # scatter-add gradient into E
        d_E = np.zeros_like(E)
        for i, cid in enumerate(cat_ids):
            d_E[cid] += d_emb[i]
        W -= lr * d_W; b -= lr * d_b; E -= lr * d_E
    return {"E": E, "W": W, "b": b, "n_classes": n_classes,
            "method": f"entity embedding (K={K}, dim={dim}) + softmax head"}


def predict_ee(cat_ids, m):
    emb = m["E"][cat_ids]
    return _softmax(emb @ m["W"] + m["b"]).argmax(axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 12 categories arranged in 3 clusters of behaviour; classify into 3 clusters.
    # Levels 0..3 -> class 0; levels 4..7 -> class 1; levels 8..11 -> class 2.
    K = 12; n_per = 60
    cat_ids = np.repeat(np.arange(K), n_per)
    labels = np.repeat([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], n_per)
    idx = rng.permutation(len(cat_ids))
    cat_ids = cat_ids[idx]; labels = labels[idx]
    n_tr = int(0.8 * len(cat_ids))
    cats_tr, cats_te = cat_ids[:n_tr], cat_ids[n_tr:]
    y_tr, y_te = labels[:n_tr], labels[n_tr:]

    m = train_entity_embedding(cats_tr, y_tr, K=K, dim=2, epochs=400, lr=0.1)
    acc_tr = float((predict_ee(cats_tr, m) == y_tr).mean())
    acc_te = float((predict_ee(cats_te, m) == y_te).mean())
    print(f"=== Entity embedding demo: K={K} categories, dim=2, 3 latent groups ===")
    print(f"  train accuracy = {acc_tr:.3f}")
    print(f"  test  accuracy = {acc_te:.3f}")
    print(f"\n  learned 2-D embedding for each category (grouped by true class):")
    for c in range(K):
        true_cls = c // 4
        print(f"    cat {c:>2} (true class {true_cls}): {np.round(m['E'][c], 3).tolist()}")

    # cosine similarity between embeddings within same class vs across
    def _cos(u, v): return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    within_sims = []
    between_sims = []
    for i in range(K):
        for j in range(i + 1, K):
            s = _cos(m["E"][i], m["E"][j])
            if i // 4 == j // 4:
                within_sims.append(s)
            else:
                between_sims.append(s)
    print(f"\n  mean cosine within-class = {np.mean(within_sims):+.3f}")
    print(f"  mean cosine across-class = {np.mean(between_sims):+.3f}   "
          f"(embeddings for co-behaving categories are more similar)")

    print("\n--- library cross-check (torch.nn.Embedding + Linear head) ---")
