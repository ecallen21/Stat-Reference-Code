"""t-SNE and UMAP: nonlinear dimensionality reduction for visualization
(Reference §26.5).

Both algorithms embed a high-dimensional dataset into 2 or 3 dimensions
preserving LOCAL neighborhood structure -- clusters remain clusters.
They are visualization tools, NOT for downstream distance-preserving
tasks or reversible dimensionality reduction.

t-SNE (van der Maaten-Hinton 2008)
    1. Compute pairwise probabilities p_ij in high-D via Gaussians tuned
       to a target PERPLEXITY (~5-50).
    2. Compute pairwise probabilities q_ij in low-D via a Student-t kernel
       (heavy tails to avoid crowding).
    3. Minimize KL(P || Q) via gradient descent.

UMAP (McInnes-Healy 2018)
    Based on Riemannian geometry / topological data analysis.  Faster than
    t-SNE, tends to preserve GLOBAL structure better, deterministic default.
    Typical parameter: n_neighbors (~15) controls local vs global emphasis.

This module ships a small demo showing the *interface* and calls sklearn's
TSNE / (optional) umap-learn.  Full from-scratch implementations of
either method are substantial multi-hundred-line optimizers; not
duplicated here.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def run_tsne(X, n_components: int = 2, perplexity: float = 30.0, seed: int = 0) -> dict:
    from sklearn.manifold import TSNE
    ts = TSNE(n_components=n_components, perplexity=perplexity,
              random_state=seed, init="pca")
    Z = ts.fit_transform(X)
    return {"embedding": Z, "kl_divergence": float(ts.kl_divergence_),
            "perplexity": float(perplexity),
            "method": "t-SNE (sklearn)"}


def run_umap(X, n_components: int = 2, n_neighbors: int = 15,
             min_dist: float = 0.1, seed: int = 0) -> dict:
    try:
        import umap
    except ImportError:
        return {"error": "umap-learn not installed"}
    u = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                  min_dist=min_dist, random_state=seed)
    Z = u.fit_transform(X)
    return {"embedding": Z,
            "n_neighbors": n_neighbors, "min_dist": min_dist,
            "method": "UMAP"}


def cluster_purity(Z, labels, n_clusters: int = None) -> float:
    """Simple diagnostic: k-means-cluster the embedding, purity vs true labels."""
    from sklearn.cluster import KMeans
    labels = np.asarray(labels)
    if n_clusters is None: n_clusters = len(np.unique(labels))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(Z)
    from collections import Counter
    total_hit = 0
    for c in range(n_clusters):
        cnt = Counter(labels[km.labels_ == c])
        if cnt: total_hit += cnt.most_common(1)[0][1]
    return total_hit / len(labels)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Three well-separated 20-D Gaussians
    n_per = 100
    X = np.vstack([rng.normal(0, 1, (n_per, 20)),
                    rng.normal(5, 1, (n_per, 20)),
                    rng.normal(-5, 1, (n_per, 20))])
    y = np.repeat([0, 1, 2], n_per)

    print("=== t-SNE (sklearn) ===")
    try:
        r = run_tsne(X, perplexity=30, seed=0)
        purity = cluster_purity(r["embedding"], y)
        print(f"  KL divergence = {r['kl_divergence']:.4f}")
        print(f"  cluster purity on embedding = {purity:.3f}")
    except Exception as ex:
        print(f"  sklearn TSNE unavailable: {ex}")

    print("\n=== UMAP ===")
    r = run_umap(X, n_neighbors=15, seed=0)
    if "error" not in r:
        purity = cluster_purity(r["embedding"], y)
        print(f"  cluster purity on embedding = {purity:.3f}")
    else:
        print(f"  {r['error']} (install with 'pip install umap-learn')")

    print("\n--- Notes ---")
    print("  Both methods are stochastic / dependent on random-state. Report multiple seeds.")
    print("  t-SNE and UMAP embeddings should NOT be used for downstream distances or clustering.")
