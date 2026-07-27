"""Principal Component Analysis (Reference §9.3).

PCA finds an orthogonal rotation of the p variables into new axes (principal
components) ordered by the variance they explain.

Algorithm (SVD form)
--------------------
    Center (and optionally scale) X to Xc  (n x p)
    Xc / sqrt(n - 1)  =  U Sigma V'         (SVD)
    Loadings (rotations) : columns of V
    Scores               : U Sigma sqrt(n - 1) = Xc V
    Explained variances  : Sigma^2

Equivalent to eigen-decomposing the sample covariance matrix S = Xc' Xc / (n - 1).

Options
-------
- ``scale=True`` : divide each column by its SD first, so PCA is on the
  correlation matrix. Use when variables have very different units.
- Return loadings, scores, per-component variance and cumulative variance,
  and the biplot coordinates.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def pca(X, n_components: int | None = None, scale: bool = False,
        var_names=None) -> dict:
    """SVD-based PCA on an n x p data matrix.

    Returns
    -------
    dict with loadings (V, p x k), scores (n x k), singular_values, per-component
    variance, cumulative variance %, center and scale used, and biplot coords.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n, p = X.shape
    if n_components is None or n_components > min(n, p):
        n_components = min(n, p)
    if var_names is None:
        var_names = [f"V{i}" for i in range(p)]

    center = X.mean(axis=0)
    scale_arr = X.std(axis=0, ddof=1) if scale else np.ones(p)
    Xc = (X - center) / np.where(scale_arr > 0, scale_arr, 1.0)

    # SVD on Xc / sqrt(n - 1) -> singular values are SDs of the PCs
    U, sigma, Vt = np.linalg.svd(Xc / math.sqrt(n - 1), full_matrices=False)
    # keep top k
    V = Vt.T[:, :n_components]                  # p x k loadings
    sigma = sigma[:n_components]                # singular values (= PC SDs)
    scores = Xc @ V                              # n x k
    variances = sigma ** 2                       # eigenvalues of cov(Xc)
    total_var = float(np.sum(Xc.var(axis=0, ddof=1)))
    explained_ratio = variances / total_var if total_var > 0 else np.zeros(n_components)
    cum_ratio = np.cumsum(explained_ratio)
    # Biplot coordinates -- loadings scaled by singular values
    biplot_loadings = V * sigma[None, :]

    return {"n_components": n_components,
            "center": center.tolist(),
            "scale": scale_arr.tolist() if scale else None,
            "loadings": V.tolist(),               # each column = one PC's direction
            "loadings_variables": list(var_names),
            "scores": scores.tolist(),            # each row = one obs's coord in PC space
            "singular_values": sigma.tolist(),
            "explained_variance": variances.tolist(),
            "explained_variance_ratio": explained_ratio.tolist(),
            "cumulative_variance_ratio": cum_ratio.tolist(),
            "biplot_loadings": biplot_loadings.tolist(),
            "n": n, "p": p,
            "method": "PCA via SVD" + (" (on correlation matrix)" if scale else " (on covariance matrix)")}


def library_versions(X, n_components=None, scale=False):
    from sklearn.decomposition import PCA as SkPCA
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X) if scale else X - np.asarray(X).mean(axis=0)
    if n_components is None:
        n_components = min(np.asarray(X).shape)
    m = SkPCA(n_components=n_components).fit(Xs)
    return {"sklearn explained_variance": m.explained_variance_.tolist(),
            "sklearn explained_variance_ratio": m.explained_variance_ratio_.tolist(),
            "sklearn singular_values": m.singular_values_.tolist(),
            "sklearn components (rows = PCs)": m.components_.tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(23)
    # Create data with two dominant directions in 5D
    n = 200
    # true loadings (unit-norm columns)
    L = np.array([
        [ 0.6,  0.1],
        [ 0.5, -0.2],
        [ 0.4,  0.3],
        [ 0.3, -0.5],
        [ 0.2,  0.7],
    ])
    L /= np.linalg.norm(L, axis=0)
    scores_true = rng.normal(0, [3.0, 1.5], size=(n, 2))
    noise = rng.normal(0, 0.3, size=(n, 5))
    X = scores_true @ L.T + noise + np.array([1, 2, 3, 4, 5])
    print("=== PCA on 5D data (should recover 2 dominant components) ===")
    out = pca(X, n_components=5)
    print(f"  center = {out['center']}")
    print(f"  explained variance ratio: {[f'{v:.4f}' for v in out['explained_variance_ratio']]}")
    print(f"  cumulative: {[f'{v:.4f}' for v in out['cumulative_variance_ratio']]}")
    print(f"  singular values: {[f'{v:.4f}' for v in out['singular_values']]}")

    print("\n--- library (sklearn) ---")
    for k, v in library_versions(X, n_components=5).items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            print(f"  {k}: (rows shown)"); [print(f"    {r}") for r in v[:2]]
        else:
            print(f"  {k}: {v}")
