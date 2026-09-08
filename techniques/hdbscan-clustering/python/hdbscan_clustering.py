"""HDBSCAN clustering (Reference Sec 47.79).

Campello, Moulavi & Sander 2013 'Density-based clustering based on
hierarchical density estimates', PAKDD. Improves DBSCAN by:

    1. mutual reachability distance d_m(a, b) = max(core(a), core(b),
                                                   dist(a, b));
    2. minimum-spanning tree on d_m;
    3. hierarchical single-linkage dendrogram;
    4. extract flat clusters by MAXIMUM PERSISTENCE.

Removes DBSCAN's eps parameter; requires only `min_cluster_size`.
Handles clusters of VARYING DENSITY, marks low-density points as
noise (label = -1).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays

try:
    import hdbscan    # library implementation (fallback if unavailable)
    HDBSCAN_OK = True
except Exception:
    HDBSCAN_OK = False


def hdbscan_or_sklearn(X, min_cluster_size=10):
    if HDBSCAN_OK:
        m = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        return m.fit_predict(X)
    # sklearn 1.3+ ships HDBSCAN
    from sklearn.cluster import HDBSCAN
    return HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X)


if __name__ == "__main__":
    print("=== HDBSCAN (Campello-Moulavi-Sander 2013) ===\n")
    rng = np.random.default_rng(0)

    # Three clusters of very different density + uniform noise
    Xa = rng.normal(loc=[0, 0], scale=0.15, size=(150, 2))
    Xb = rng.normal(loc=[4, 0], scale=0.6, size=(150, 2))
    Xc = rng.normal(loc=[2, 3.5], scale=1.2, size=(150, 2))
    Xn = rng.uniform(low=-2, high=6, size=(60, 2))
    X = np.vstack([Xa, Xb, Xc, Xn])
    y_true = np.array([0]*150 + [1]*150 + [2]*150 + [-1]*60)

    for mcs in [5, 15, 30]:
        labels = hdbscan_or_sklearn(X, min_cluster_size=mcs)
        n_clusters = int(labels.max() + 1) if labels.max() >= 0 else 0
        n_noise = int((labels == -1).sum())
        from sklearn.metrics import adjusted_rand_score    # ARI vs truth
        ari = float(adjusted_rand_score(y_true, labels))
        print(f"  min_cluster_size = {mcs:3d}  ->  {n_clusters} clusters, "
              f"{n_noise} noise pts  ARI = {ari:.3f}")

    # Contrast: k-means always splits noise into clusters
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=3, n_init=20, random_state=0).fit_predict(X)
    from sklearn.metrics import adjusted_rand_score
    print(f"\n  K-means k=3 baseline           ARI = "
          f"{adjusted_rand_score(y_true, km):.3f}   "
          f"(no noise label; noise absorbed into clusters)")

    print("\n--- library cross-check (dbscan R; hdbscan / sklearn.cluster.HDBSCAN Python) ---")
