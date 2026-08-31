"""Random projections (Reference Sec 25.12).

Johnson & Lindenstrauss (1984); Achlioptas (2001).

Project X (n x d) to Y (n x k) with a RANDOM matrix R (d x k):

  Y = X R,  R_{ij} iid ~ (1/sqrt(k)) * N(0, 1)  or  Achlioptas' {-1, 0, +1}.

JL Lemma: for any epsilon in (0, 1) and n points, k = O(log n / eps^2)
suffices so that all pairwise distances are preserved within factor
(1 +/- eps):

  (1 - eps) ||x_i - x_j||^2 <= ||y_i - y_j||^2 <= (1 + eps) ||x_i - x_j||^2

with probability >= 1 - 2 exp(-c k eps^2).

Advantages:
  * NO training required (data-independent).
  * Streaming / online friendly.
  * Distance-preserving guarantee.

Here we implement Gaussian + Achlioptas random projections, verify the
JL distance-preservation bound empirically, and demo a downstream
1-nearest-neighbour accuracy check.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def gaussian_projection(X, k, seed=0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    R = rng.normal(0, 1.0 / np.sqrt(k), (d, k))
    return X @ R, R


def achlioptas_projection(X, k, seed=0):
    """Sparse random projection with entries in {-1, 0, +1}."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    u = rng.random((d, k))
    R = np.where(u < 1/6, np.sqrt(3), np.where(u > 5/6, -np.sqrt(3), 0.0))
    R = R / np.sqrt(k)
    return X @ R, R


def distortion_stats(X, Y):
    """Report distortion of pairwise distances after projection."""
    n = X.shape[0]
    idx = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    D_orig = np.sqrt(((X[idx[:, 0]] - X[idx[:, 1]]) ** 2).sum(axis=1))
    D_proj = np.sqrt(((Y[idx[:, 0]] - Y[idx[:, 1]]) ** 2).sum(axis=1))
    ratio = D_proj / (D_orig + 1e-12)
    return {"mean_ratio": float(ratio.mean()),
             "min_ratio": float(ratio.min()),
             "max_ratio": float(ratio.max()),
             "distortion_pct": float(100 * np.mean(np.abs(1 - ratio)))}


if __name__ == "__main__":
    print("=== Random projections (Johnson-Lindenstrauss) ===\n")
    rng = np.random.default_rng(0)
    n, d = 100, 500
    X = rng.normal(0, 1, (n, d))

    print(f"  original dim = {d}   n = {n}")
    print(f"\n  {'k':>4}  {'method':>13}  {'mean ratio':>10}  {'min ratio':>10}"
          f"  {'max ratio':>10}  {'mean distortion':>16}")
    for k in (20, 50, 100, 200):
        for name, fn in (("Gaussian", gaussian_projection), ("Achlioptas", achlioptas_projection)):
            Y, _ = fn(X, k, seed=1)
            s = distortion_stats(X, Y)
            print(f"  {k:>4}  {name:>13}  {s['mean_ratio']:>10.3f}"
                  f"  {s['min_ratio']:>10.3f}  {s['max_ratio']:>10.3f}"
                  f"  {s['distortion_pct']:>15.2f}%")

    # 1-NN accuracy: use CLUSTERED data so NN is meaningful.
    K = 5
    centres = rng.normal(0, 6, (K, d))
    per_cluster = 20
    X_cl = np.vstack([centres[k] + rng.normal(0, 1, (per_cluster, d)) for k in range(K)])
    labels = np.repeat(np.arange(K), per_cluster)
    # Pick 20 queries from each cluster; look-up in the rest.
    query_idx = np.array([k * per_cluster for k in range(K)] + [k * per_cluster + 1 for k in range(K)]
                          + [k * per_cluster + 2 for k in range(K)] + [k * per_cluster + 3 for k in range(K)])
    ref_idx = np.array([i for i in range(len(X_cl)) if i not in set(query_idx.tolist())])
    def nn_labels(A, ref, ref_labels):
        D = ((A[:, None] - ref[None]) ** 2).sum(axis=2)
        return ref_labels[D.argmin(axis=1)]
    nn_orig = nn_labels(X_cl[query_idx], X_cl[ref_idx], labels[ref_idx])
    Yq, _ = gaussian_projection(X_cl, k=50, seed=1)
    nn_proj = nn_labels(Yq[query_idx], Yq[ref_idx], labels[ref_idx])
    acc_orig = float((nn_orig == labels[query_idx]).mean())
    acc_proj = float((nn_proj == labels[query_idx]).mean())
    print(f"\n  1-NN cluster-label accuracy on {K} clusters:")
    print(f"    original (d={d}): {acc_orig:.3f}")
    print(f"    projected (k=50): {acc_proj:.3f}\n")
    print("--- library cross-check (sklearn.random_projection.GaussianRandomProjection / SparseRandomProjection) ---")
