"""Gap statistic for choosing K (Reference Sec 47.81).

Tibshirani, Walther & Hastie 2001 'Estimating the number of
clusters in a data set via the gap statistic', JRSS-B 63. Compare
observed within-cluster dispersion W_k to a null reference
generated from a uniform distribution over the data's bounding box:

    Gap(k) = E*[log W_k*] - log W_k

with se_k the SD of log W_k* over B references. Choose the SMALLEST
k such that

    Gap(k) >= Gap(k+1) - se_{k+1}.

Robust to k=1 (no clusters) case.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from sklearn.cluster import KMeans    # base clusterer


def _within_dispersion(X, labels):
    W = 0.0
    for L in np.unique(labels):
        pts = X[labels == L]
        if len(pts) < 2:
            continue
        d = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        W += float(d.sum() / (2 * len(pts)))
    return W


def gap_statistic(X, k_range=range(1, 10), B=20, seed=0):
    rng = np.random.default_rng(seed)
    lo = X.min(axis=0); hi = X.max(axis=0)
    log_W = np.zeros(len(k_range))
    log_W_ref = np.zeros((len(k_range), B))
    for i, k in enumerate(k_range):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        log_W[i] = np.log(max(_within_dispersion(X, km.labels_), 1e-12))
        for b in range(B):
            Xr = rng.uniform(lo, hi, size=X.shape)
            kmr = KMeans(n_clusters=k, n_init=10, random_state=b).fit(Xr)
            log_W_ref[i, b] = np.log(max(_within_dispersion(Xr, kmr.labels_), 1e-12))
    gap = log_W_ref.mean(axis=1) - log_W
    sd = log_W_ref.std(axis=1)
    se = sd * np.sqrt(1 + 1.0 / B)
    # Tibshirani rule: smallest k with Gap(k) >= Gap(k+1) - se(k+1)
    k_hat = None
    for i in range(len(k_range) - 1):
        if gap[i] >= gap[i + 1] - se[i + 1]:
            k_hat = list(k_range)[i]; break
    if k_hat is None:
        k_hat = list(k_range)[np.argmax(gap)]
    return {"k_range": list(k_range), "gap": gap, "se": se, "k_hat": k_hat}


if __name__ == "__main__":
    print("=== Gap statistic (Tibshirani-Walther-Hastie 2001) ===\n")
    rng = np.random.default_rng(0)

    # 3 well-separated clusters in 2D
    centers = np.array([[0, 0], [5, 0], [2.5, 4]])
    X3 = np.vstack([c + 0.3 * rng.normal(size=(80, 2)) for c in centers])

    res = gap_statistic(X3, k_range=range(1, 8), B=20)
    print("  3-cluster data:")
    for k, g, s in zip(res["k_range"], res["gap"], res["se"]):
        marker = "  <-- chosen" if k == res["k_hat"] else ""
        print(f"    k={k}   Gap={g:+.3f}   se={s:.3f}{marker}")
    print(f"  Selected K = {res['k_hat']}   (truth 3)")

    # Uniform data -- expect k=1
    Xu = rng.uniform(size=(240, 2))
    res_u = gap_statistic(Xu, k_range=range(1, 8), B=20)
    print("\n  Uniform data (no cluster structure):")
    print(f"  Selected K = {res_u['k_hat']}   (truth 1)")

    print("\n--- library cross-check (cluster::clusGap R; gap-statistic Python) ---")
