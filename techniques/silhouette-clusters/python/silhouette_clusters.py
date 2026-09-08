"""Silhouette clustering validation (Reference Sec 47.74).

Rousseeuw 1987 'Silhouettes: a graphical aid to the interpretation
and validation of cluster analysis', J Comp Appl Math 20. For each
point i in cluster A:

    a_i = mean distance to other points in A
    b_i = min over other clusters B of mean distance to B
    s_i = (b_i - a_i) / max(a_i, b_i)         in [-1, 1]

Average silhouette width  = 1/n sum s_i.
    s ~ 1  -- well-clustered (a << b).
    s ~ 0  -- boundary point.
    s < 0  -- probably in the wrong cluster.

Selecting k by the value maximising mean s (up to elbow) is a
standard heuristic.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from sklearn.cluster import KMeans    # clusterer for demo


def silhouette_scores(X, labels):
    """Per-point + mean silhouette (n^2 memory)."""
    n = len(X)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    s = np.zeros(n)
    for i in range(n):
        li = labels[i]
        same = (labels == li) & (np.arange(n) != i)
        a = float(D[i, same].mean()) if same.any() else 0.0
        b = np.inf
        for L in np.unique(labels):
            if L == li: continue
            mask = labels == L
            if not mask.any(): continue
            b = min(b, float(D[i, mask].mean()))
        if max(a, b) > 0:
            s[i] = (b - a) / max(a, b)
    return {"s": s, "mean": float(s.mean())}


if __name__ == "__main__":
    print("=== Silhouette (Rousseeuw 1987) ===\n")
    rng = np.random.default_rng(0)
    n_per = 100

    # Well-separated 3 clusters
    centers = np.array([[0, 0], [4, 0], [2, 3.5]])
    Xc = np.vstack([c + 0.4 * rng.normal(size=(n_per, 2)) for c in centers])

    print("  Well-separated 3-cluster data (n=300):")
    for k in [2, 3, 4, 5]:
        labs = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(Xc)
        r = silhouette_scores(Xc, labs)
        print(f"    k = {k}:  mean silhouette = {r['mean']:+.3f}")

    # Overlapping clusters
    Xo = np.vstack([c + 1.5 * rng.normal(size=(n_per, 2)) for c in centers])
    print("\n  Overlapping 3-cluster data (n=300):")
    for k in [2, 3, 4, 5]:
        labs = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(Xo)
        r = silhouette_scores(Xo, labs)
        print(f"    k = {k}:  mean silhouette = {r['mean']:+.3f}")

    print("\n  Rule of thumb (Rousseeuw): mean s > 0.7 strong; 0.5-0.7 reasonable;")
    print("  < 0.25 weak / no substantial structure.")
    print("\n--- library cross-check (cluster::silhouette R; sklearn.metrics.silhouette_score Python) ---")
