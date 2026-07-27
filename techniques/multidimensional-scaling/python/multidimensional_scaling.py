"""Multidimensional Scaling: classical + non-metric (Reference §9.32).

Given a pairwise distance matrix D between n objects (obtained however you like
-- Euclidean, Jaccard, subjective similarity, ...), MDS finds coordinates in a
low-dimensional space (typically 2 or 3) such that the pairwise distances
approximate D as closely as possible.

Classical (metric) MDS -- Torgerson-Gower
-----------------------------------------
    1. Square the distances: D^2.
    2. Double-center:  B = -0.5 * H D^2 H,  where H = I - (1/n) 1 1'.
    3. Eigen-decompose B; take top k eigenvalues and vectors.
       Coordinates = eigenvectors * sqrt(eigenvalues).
When D IS the Euclidean distance matrix of some data, classical MDS recovers
that data (up to rotation and reflection); equivalent to PCA on the data.

Non-metric MDS (Kruskal 1964)
-----------------------------
Preserves only the ORDER of distances -- useful when distances come from
ordinal / subjective ratings. Iteratively:
    1. Given current coords, compute fitted distances d_hat.
    2. Isotonic regression of d_hat on D to get monotone-transformed disparities.
    3. Move coords to reduce STRESS = sqrt(sum (d_hat - disparity)^2 / sum d_hat^2).
    4. Iterate.

Kruskal's stress interpretation:
    < 0.05  excellent   0.05-0.1  good   0.1-0.2  fair   > 0.2  poor
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def classical_mds(D, n_components: int = 2) -> dict:
    """Classical (metric) MDS via double-centering + eigendecomposition."""
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square distance matrix")
    n = D.shape[0]
    D2 = D ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H
    B = (B + B.T) / 2                        # numerical symmetrization
    w, V = np.linalg.eigh(B)
    order = np.argsort(-w)
    w = w[order]; V = V[:, order]
    w_pos = np.clip(w[:n_components], 0, None)
    coords = V[:, :n_components] * np.sqrt(w_pos)[None, :]
    total_pos_var = float(w[w > 0].sum())
    explained = (w_pos / total_pos_var).tolist() if total_pos_var > 0 else [0.0] * n_components
    return {"coordinates": coords.tolist(),
            "eigenvalues": w[:n_components].tolist(),
            "explained_variance_ratio": explained,
            "n_components": n_components,
            "method": "Classical (metric) MDS (Torgerson-Gower)"}


def _isotonic(y, w=None):
    """Pool-adjacent-violators isotonic regression: non-decreasing fit to y."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if w is None: w = np.ones(n)
    else: w = np.asarray(w, dtype=float)
    y_out = y.copy(); w_out = w.copy()
    ranges = [(i, i) for i in range(n)]
    i = 0
    while i < len(y_out) - 1:
        if y_out[i] <= y_out[i + 1] + 1e-15:
            i += 1; continue
        # merge i and i+1
        wsum = w_out[i] + w_out[i + 1]
        merged = (w_out[i] * y_out[i] + w_out[i + 1] * y_out[i + 1]) / wsum
        y_out = np.delete(y_out, i + 1); y_out[i] = merged
        w_out[i] = wsum; w_out = np.delete(w_out, i + 1)
        r_i, r_j = ranges[i][0], ranges[i + 1][1]
        del ranges[i + 1]; ranges[i] = (r_i, r_j)
        if i > 0: i -= 1
    fitted = np.empty(n)
    for k, (lo, hi) in enumerate(ranges):
        fitted[lo:hi + 1] = y_out[k]
    return fitted


def non_metric_mds(D, n_components: int = 2, max_iter: int = 200,
                    tol: float = 1e-6, seed: int = 0) -> dict:
    """Kruskal-style non-metric MDS with stress minimization + isotonic step."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    orig = D[iu]
    order = np.argsort(orig)
    # init: classical MDS coords
    coords = np.array(classical_mds(D, n_components)["coordinates"])
    prev_stress = np.inf
    for it in range(max_iter):
        diffs = coords[:, None, :] - coords[None, :, :]
        d_hat = np.sqrt((diffs ** 2).sum(-1))
        d_flat = d_hat[iu]
        # isotonic regression: rearrange d_flat by original-distance order,
        # do PAV, then unshuffle
        sorted_d = d_flat[order]
        disp_sorted = _isotonic(sorted_d)
        disparities = np.empty_like(d_flat)
        disparities[order] = disp_sorted
        # Kruskal stress-1
        denom = (d_flat ** 2).sum()
        stress = math.sqrt(((d_flat - disparities) ** 2).sum() / denom) if denom > 0 else 0.0
        if abs(prev_stress - stress) < tol:
            break
        prev_stress = stress
        # Guttman transform (majorization update) to move coords toward disparities
        d_safe = np.where(d_hat > 1e-12, d_hat, 1e-12)
        B = np.zeros((n, n))
        disp_full = np.zeros((n, n))
        disp_full[iu] = disparities; disp_full[(iu[1], iu[0])] = disparities
        with np.errstate(divide="ignore", invalid="ignore"):
            B = np.where(d_hat > 1e-12, -disp_full / d_safe, 0.0)
        np.fill_diagonal(B, 0)
        B_diag = -B.sum(axis=1)
        np.fill_diagonal(B, B_diag)
        coords = (1.0 / n) * B @ coords
        # recenter
        coords = coords - coords.mean(axis=0)
    return {"coordinates": coords.tolist(),
            "stress_1": stress,
            "n_iter": it + 1,
            "n_components": n_components,
            "method": "Non-metric MDS (Kruskal stress-1 minimization)"}


def library_versions(D, n_components=2):
    import warnings
    from sklearn.manifold import MDS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metric = MDS(n_components=n_components, dissimilarity="precomputed",
                     n_init=1, random_state=0, metric=True).fit_transform(D)
        nonmetric = MDS(n_components=n_components, dissimilarity="precomputed",
                        n_init=1, random_state=0, metric=False).fit_transform(D)
    return {"sklearn metric coords[:3]": metric[:3].tolist(),
            "sklearn non-metric coords[:3]": nonmetric[:3].tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(103)
    # 5 well-separated 2D points; MDS on their Euclidean distances should recover them
    true_coords = np.array([[0, 0], [3, 0], [3, 4], [0, 4], [1.5, 2]])
    diffs = true_coords[:, None, :] - true_coords[None, :, :]
    D = np.sqrt((diffs ** 2).sum(-1))

    print("=== Classical MDS on Euclidean distances ===")
    out = classical_mds(D, n_components=2)
    print(f"  eigenvalues: {out['eigenvalues']}")
    print(f"  explained variance ratio: {out['explained_variance_ratio']}")
    print(f"  coords:")
    for c in out["coordinates"]:
        print(f"    {c}")

    print("\n=== Non-metric MDS ===")
    out = non_metric_mds(D, n_components=2)
    print(f"  stress_1: {out['stress_1']:.5f}  ({'excellent' if out['stress_1'] < 0.05 else 'good' if out['stress_1'] < 0.1 else 'fair'})")
    print(f"  n_iter: {out['n_iter']}")

    print("\n--- library (sklearn) ---")
    for k, v in library_versions(D).items():
        print(f"  {k}: {v}")
