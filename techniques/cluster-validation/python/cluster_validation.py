"""Cluster-validation indices for choosing k (Reference §9.14).

Internal indices (data + labels; no external labels needed):

    Silhouette      : per-point:  (b - a) / max(a, b)
                      where a = mean distance to same-cluster points
                            b = min mean distance to any OTHER cluster
                      Overall silhouette = mean of per-point silhouettes.
                      Range [-1, 1]; higher = better.

    Calinski-Harabasz (CH) : between-SS / within-SS  scaled by df ratio.
                             Higher = better; often peaks at the true k.

    Davies-Bouldin (DB)    : mean over clusters of the "worst neighbor" ratio
                             (dispersion(i) + dispersion(j)) / d(centroid_i, centroid_j).
                             Lower = better.

Alongside these, the classic informal choices:
    Elbow (WCSS knee)  : plot inertia vs k and eyeball the elbow.
    Gap statistic (Tibshirani et al. 2001): compare WCSS to uniform-null WCSS.

All indices are useful together -- they disagree by construction on some data;
pick k where several indices agree.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _pdist_euclidean(X):
    diffs = X[:, None, :] - X[None, :, :]
    return np.sqrt((diffs ** 2).sum(axis=-1))


def silhouette(X, labels) -> dict:
    """Overall + per-cluster silhouette scores."""
    X = np.asarray(X, dtype=float); labels = np.asarray(labels)
    n = X.shape[0]
    D = _pdist_euclidean(X)
    unique = np.unique(labels)
    per_point = np.zeros(n)
    for i in range(n):
        own = labels[i]
        same_mask = (labels == own) & (np.arange(n) != i)
        if not same_mask.any():
            per_point[i] = 0.0; continue
        a = D[i, same_mask].mean()
        b = np.inf
        for c in unique:
            if c == own: continue
            m = labels == c
            if not m.any(): continue
            mean_dist = D[i, m].mean()
            if mean_dist < b: b = mean_dist
        per_point[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    per_cluster = {int(c): float(per_point[labels == c].mean()) for c in unique}
    return {"overall": float(per_point.mean()),
            "per_cluster": per_cluster,
            "n_clusters": int(len(unique))}


def calinski_harabasz(X, labels) -> float:
    """CH = (B / (k-1)) / (W / (n-k)) where B, W are between/within SS."""
    X = np.asarray(X, dtype=float); labels = np.asarray(labels)
    n, p = X.shape
    unique = np.unique(labels); k = len(unique)
    if k < 2 or k >= n: return float("nan")
    grand = X.mean(axis=0)
    B = W = 0.0
    for c in unique:
        m = labels == c; Xc = X[m]; nc = Xc.shape[0]
        cent = Xc.mean(axis=0)
        B += nc * float(((cent - grand) ** 2).sum())
        W += float(((Xc - cent) ** 2).sum())
    if W == 0: return float("inf")
    return (B / (k - 1)) / (W / (n - k))


def davies_bouldin(X, labels) -> float:
    """DB = (1/k) sum_i max_{j != i} (s_i + s_j) / d(c_i, c_j),  lower = better."""
    X = np.asarray(X, dtype=float); labels = np.asarray(labels)
    unique = np.unique(labels); k = len(unique)
    if k < 2: return float("nan")
    cents = np.array([X[labels == c].mean(axis=0) for c in unique])
    disp = np.array([float(np.sqrt(((X[labels == c] - cents[i]) ** 2).sum(axis=1)).mean())
                     for i, c in enumerate(unique)])
    d_cent = _pdist_euclidean(cents)
    np.fill_diagonal(d_cent, np.inf)
    ratios = (disp[:, None] + disp[None, :]) / d_cent
    return float(ratios.max(axis=1).mean())


def elbow_wcss(X, k_grid, kmeans_fit) -> list:
    """WCSS (inertia) for each k using a supplied k-means fitter.

    ``kmeans_fit(X, k)`` should return a dict with an ``inertia`` key.
    """
    return [{"k": k, "inertia": kmeans_fit(X, k)["inertia"]} for k in k_grid]


def gap_statistic(X, k_grid, kmeans_fit, n_ref: int = 20, seed: int = 0) -> dict:
    """Tibshirani et al. (2001) gap statistic.

    Gap(k) = E[log W_uniform(k)] - log W_data(k)

    We generate ``n_ref`` uniform reference datasets over the bounding box of X,
    run kmeans on each, and average log W_ref.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    rng = np.random.default_rng(seed)
    lo, hi = X.min(axis=0), X.max(axis=0)
    data_logW = []
    ref_logW_mean = []
    ref_logW_sd = []
    for k in k_grid:
        logW_data = math.log(max(kmeans_fit(X, k)["inertia"], 1e-12))
        data_logW.append(logW_data)
        logW_refs = []
        for _ in range(n_ref):
            Xr = rng.uniform(lo, hi, size=(n, p))
            logW_refs.append(math.log(max(kmeans_fit(Xr, k)["inertia"], 1e-12)))
        ref_logW_mean.append(float(np.mean(logW_refs)))
        ref_logW_sd.append(float(np.std(logW_refs, ddof=1)))
    gap = [r - d for r, d in zip(ref_logW_mean, data_logW)]
    s = [sd * math.sqrt(1 + 1 / n_ref) for sd in ref_logW_sd]
    # "1-SE" k choice: smallest k such that gap(k) >= gap(k+1) - s(k+1)
    best = k_grid[0]
    for i in range(len(k_grid) - 1):
        if gap[i] >= gap[i + 1] - s[i + 1]:
            best = k_grid[i]; break
        best = k_grid[i + 1]
    return {"k_grid": list(k_grid), "gap": gap, "s_k": s,
            "data_logW": data_logW, "ref_logW_mean": ref_logW_mean,
            "best_k_1SE": best}


def library_versions(X, labels):
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    return {"sklearn silhouette": float(silhouette_score(X, labels)),
            "sklearn calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "sklearn davies_bouldin": float(davies_bouldin_score(X, labels))}


if __name__ == "__main__":
    rng = np.random.default_rng(83)
    # 3 well-separated Gaussian blobs
    centers = np.array([[0, 0], [4, 4], [8, 0]])
    X = np.vstack([rng.normal(c, 0.7, size=(50, 2)) for c in centers])

    # Quick k-means (reuse from-scratch is heavier; use sklearn for the fitter)
    from sklearn.cluster import KMeans
    def kfit(X, k):
        m = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        return {"inertia": float(m.inertia_), "labels": m.labels_}

    print("=== Validation indices at k=3 (known truth) ===")
    labels = kfit(X, 3)["labels"]
    sil = silhouette(X, labels)
    print(f"  silhouette: {sil['overall']:.4f}  per-cluster: {sil['per_cluster']}")
    print(f"  Calinski-Harabasz: {calinski_harabasz(X, labels):.4f}")
    print(f"  Davies-Bouldin:    {davies_bouldin(X, labels):.4f}")

    print("\n=== Elbow (WCSS) for k = 1..7 ===")
    for r in elbow_wcss(X, range(1, 8), kfit):
        print(f"  k={r['k']}: inertia={r['inertia']:.4f}")

    print("\n=== Gap statistic (k = 1..7) ===")
    g = gap_statistic(X, range(1, 8), kfit, n_ref=10)
    for k, gap, s in zip(g["k_grid"], g["gap"], g["s_k"]):
        print(f"  k={k}: gap={gap:+.4f}, s={s:.4f}")
    print(f"  best k (1-SE rule): {g['best_k_1SE']}")

    print("\n--- library (sklearn) at k=3 ---")
    for k, v in library_versions(X, labels).items():
        print(f"  {k}: {v}")
