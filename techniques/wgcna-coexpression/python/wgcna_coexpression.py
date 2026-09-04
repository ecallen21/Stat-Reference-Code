"""Weighted gene co-expression network analysis (Reference Sec 40.6).

Langfelder-Horvath 2008.  From an expression matrix:

  1. Compute pairwise gene-gene correlations.
  2. Soft-threshold to a weighted adjacency
     a_ij = |cor(x_i, x_j)|^beta   (beta chosen for scale-free
     topology).
  3. Compute TOPOLOGICAL OVERLAP MATRIX (TOM) then 1 - TOM as a
     dissimilarity.
  4. Hierarchical clustering on 1 - TOM -> DYNAMIC TREE CUT gives
     modules of co-expressed genes.
  5. MODULE EIGENGENE = first PC of each module's expression;
     correlate with clinical traits to find biologically-relevant
     modules.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.cluster.hierarchy import linkage, fcluster    # hclust


def soft_threshold_adjacency(X, beta=6):
    """|corr|^beta soft-threshold."""
    R = np.corrcoef(X, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    A = np.abs(R) ** beta
    np.fill_diagonal(A, 0)
    return A


def topological_overlap(A):
    """TOM_ij = (l_ij + a_ij) / (min(k_i, k_j) + 1 - a_ij)."""
    k = A.sum(axis=1)
    L = A @ A
    denom = np.minimum.outer(k, k) + 1 - A
    T = (L + A) / np.where(denom > 0, denom, 1.0)
    np.fill_diagonal(T, 1.0)
    return T


def wgcna_modules(X, beta=6, min_module_size=5, n_clusters=None,
                  dist_threshold=None):
    """Cluster genes; either fix number of clusters or a distance cutoff."""
    A = soft_threshold_adjacency(X, beta)
    T = topological_overlap(A)
    D = 1 - T
    from scipy.spatial.distance import squareform
    Dc = squareform(D, checks=False)
    Z = linkage(Dc, method="average")
    if n_clusters is not None:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    else:
        labels = fcluster(Z, t=dist_threshold, criterion="distance")
    # Drop tiny modules
    from collections import Counter
    counts = Counter(labels)
    keep = {k: (v if v >= min_module_size else 0) for k, v in counts.items()}
    labels = np.array([lab if keep[lab] else 0 for lab in labels])
    return labels


def module_eigengene(X, labels, mod):
    """First PC of the module's expression matrix."""
    idx = np.where(labels == mod)[0]
    if len(idx) < 2:
        return None
    M = X[:, idx] - X[:, idx].mean(axis=0)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return M @ Vt[0]


if __name__ == "__main__":
    print("=== WGCNA: soft-thresholded co-expression network + module eigengenes ===\n")
    rng = np.random.default_rng(0)
    n_samp = 100; p = 60
    # 3 modules of 15 genes driven by common latent factors + noise features
    factors = rng.normal(0, 1, (n_samp, 3))
    loadings = np.zeros((3, p))
    loadings[0, :15] = 1.0
    loadings[1, 15:30] = 1.0
    loadings[2, 30:45] = 1.0
    X = factors @ loadings + rng.normal(0, 0.5, (n_samp, p))

    labels = wgcna_modules(X, beta=6, min_module_size=5, n_clusters=4)
    from collections import Counter
    cts = Counter(labels)
    print(f"  Module sizes (0 = unassigned): {dict(cts)}")

    trait = 1.2 * factors[:, 0] + rng.normal(0, 0.5, n_samp)
    for mod in sorted(set(labels) - {0}):
        eg = module_eigengene(X, labels, mod)
        r = np.corrcoef(eg, trait)[0, 1]
        print(f"    module {mod:>2d}  n_genes = {cts[mod]:>2d}   corr(eigengene, trait) = {r:+.3f}")

    print("\n--- library cross-check (R WGCNA::blockwiseModules; Python PyWGCNA/hdWGCNA) ---")
