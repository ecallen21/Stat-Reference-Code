"""Hierarchical (agglomerative) clustering (Reference §9.8).

Starts with each observation as its own cluster and repeatedly merges the two
closest clusters until one remains. The full history of merges is the
'dendrogram' -- cut it at any height to get a specific number of clusters.

Linkage methods (how "distance between clusters" is defined given a pair-distance):
    single   : min distance between any two members
    complete : max distance
    average  : mean pairwise distance
    Ward     : merge that minimally increases total within-cluster variance
               (implicitly assumes Euclidean distance)

Metrics: Euclidean by default; any pdist metric works for single/complete/average.
Ward requires Euclidean.

Cophenetic correlation compares the dendrogram's implied pairwise distances to
the original pair distances -- a diagnostic of how faithfully the tree
represents the data.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _pdist_euclidean(X):
    X = np.asarray(X, dtype=float)
    diffs = X[:, None, :] - X[None, :, :]
    return np.sqrt((diffs ** 2).sum(axis=2))


def agglomerative(X, linkage: str = "average", metric: str = "euclidean") -> dict:
    """From-scratch agglomerative clustering.

    Returns
    -------
    dict with the merge history ``linkage_matrix`` (SciPy-compatible: each row
    is [cluster_a, cluster_b, distance, n_in_new_cluster]) and per-observation
    ``labels`` if cut at various k.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if metric != "euclidean":
        raise ValueError("only Euclidean supported in this from-scratch demo")
    if linkage == "ward":
        # Ward implicitly uses the point-to-point distances, but the *update*
        # formula (Lance-Williams) computes squared distances.
        pass

    D = _pdist_euclidean(X)
    if linkage == "ward":
        D2 = D * D
    else:
        D2 = D
    D2 = D2.astype(float)
    np.fill_diagonal(D2, np.inf)

    # cluster sizes and IDs; new cluster id when merged
    sizes = np.ones(n, dtype=int)
    ids = list(range(n))
    next_id = n
    Z = np.zeros((n - 1, 4))
    active = list(range(n))

    for step in range(n - 1):
        # find current smallest distance between any two active clusters
        # (rows/cols indexed by position in `active`)
        sub = D2[np.ix_(active, active)]
        idx = np.unravel_index(np.argmin(sub), sub.shape)
        i_pos, j_pos = min(idx), max(idx)
        if i_pos == j_pos:
            break
        ci = active[i_pos]; cj = active[j_pos]
        d_val = sub[i_pos, j_pos]
        s_i = sizes[ci]; s_j = sizes[cj]
        if linkage == "ward":
            Z[step] = [ci, cj, math.sqrt(d_val), s_i + s_j]
        else:
            Z[step] = [ci, cj, d_val, s_i + s_j]

        # Lance-Williams update to get distances from the merged cluster to all others
        for k_pos, ck in enumerate(active):
            if ck == ci or ck == cj: continue
            d_ik = D2[ci, ck]
            d_jk = D2[cj, ck]
            d_ij = d_val
            n_i = s_i; n_j = s_j; n_k = sizes[ck]
            if linkage == "single":
                new = min(d_ik, d_jk)
            elif linkage == "complete":
                new = max(d_ik, d_jk)
            elif linkage == "average":
                new = (n_i * d_ik + n_j * d_jk) / (n_i + n_j)
            elif linkage == "ward":
                new = ((n_i + n_k) * d_ik + (n_j + n_k) * d_jk - n_k * d_ij) / (n_i + n_j + n_k)
            else:
                raise ValueError("linkage must be single / complete / average / ward")
            D2[ci, ck] = new; D2[ck, ci] = new
            D2[cj, ck] = np.inf; D2[ck, cj] = np.inf
        # extend D2 to make room for the new cluster id
        # (we reuse cluster ci as the merged cluster's id)
        sizes = np.append(sizes, s_i + s_j)
        next_id += 1
        # simulate the merge by keeping ci and removing cj
        active.remove(cj)
    return {"linkage_matrix": Z.tolist(),
            "linkage": linkage, "metric": metric, "n": n,
            "method": f"agglomerative ({linkage})"}


def cut_tree(linkage_matrix, n, k: int):
    """Cut a dendrogram to yield k clusters. Returns length-n label array."""
    Z = np.asarray(linkage_matrix)
    # perform first (n - k) merges
    parent = list(range(n + len(Z)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for step in range(n - k):
        a = int(Z[step, 0]); b = int(Z[step, 1])
        parent[find(b)] = find(a)
    roots = {}
    labels = np.empty(n, dtype=int)
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels[i] = roots[r]
    return labels


def cophenetic_correlation(X, linkage_matrix):
    """Correlation between original pairwise distances and dendrogram-implied
    distances (height at which each pair first joins the same cluster)."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    orig = _pdist_euclidean(X)
    Z = np.asarray(linkage_matrix)
    # union-find with join heights
    parent = list(range(n + len(Z)))
    join_h = {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    cluster_members = {i: [i] for i in range(n)}
    for step, row in enumerate(Z):
        a = int(row[0]); b = int(row[1]); h = row[2]
        ra, rb = find(a), find(b)
        for i in cluster_members[ra]:
            for j in cluster_members[rb]:
                join_h[(min(i, j), max(i, j))] = h
        new_id = n + step
        parent.append(new_id)
        parent[ra] = new_id; parent[rb] = new_id
        cluster_members[new_id] = cluster_members[ra] + cluster_members[rb]
        del cluster_members[ra]; del cluster_members[rb]
    coph = np.zeros_like(orig)
    for (i, j), h in join_h.items():
        coph[i, j] = h; coph[j, i] = h
    # correlation on upper triangle
    iu = np.triu_indices(n, k=1)
    orig_flat = orig[iu]; coph_flat = coph[iu]
    return float(np.corrcoef(orig_flat, coph_flat)[0, 1])


def library_versions(X, k=3):
    from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster
    from scipy.spatial.distance import pdist
    out = {}
    for method in ("single", "complete", "average", "ward"):
        Z = scipy_linkage(pdist(X), method=method)
        labels = fcluster(Z, t=k, criterion="maxclust")
        # relabel 0-based for comparability
        remap = {v: i for i, v in enumerate(sorted(set(labels)))}
        labels = np.array([remap[l] for l in labels])
        out[f"scipy {method}"] = {"first_row_of_Z": Z[0].tolist(),
                                   "last_row_of_Z": Z[-1].tolist(),
                                   "labels_at_k=" + str(k) + " head": labels[:15].tolist()}
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(41)
    # three well-separated Gaussian blobs in 2D
    n_per = 30
    centers = np.array([[0, 0], [4, 4], [8, 0]])
    X = np.vstack([rng.normal(c, 0.7, size=(n_per, 2)) for c in centers])

    for method in ("single", "complete", "average", "ward"):
        print(f"\n=== Agglomerative ({method}) ===")
        out = agglomerative(X, linkage=method)
        Z = np.asarray(out["linkage_matrix"])
        labels = cut_tree(Z, X.shape[0], k=3)
        # count cluster sizes
        from collections import Counter
        print(f"  cluster sizes at k=3: {dict(Counter(labels))}")
        if method != "ward":
            print(f"  cophenetic corr = {cophenetic_correlation(X, Z):.4f}")

    print("\n--- library (scipy) ---")
    for k, v in library_versions(X, k=3).items():
        print(f"  {k}:")
        for kk, vv in v.items():
            print(f"    {kk}: {vv}")
