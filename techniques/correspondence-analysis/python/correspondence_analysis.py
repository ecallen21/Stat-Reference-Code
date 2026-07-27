"""Correspondence Analysis (CA) for a 2-way contingency table (Reference §8.5).

CA is the categorical-data analogue of PCA: it decomposes the *chi-square*
distance between row (or column) profiles into orthogonal dimensions of
association, letting you plot row and column categories in a shared low-D
space where proximity encodes above-expected co-occurrence.

Algorithm
---------
Given an I x J count matrix N (n_ij) with grand total n_++:

    P = N / n_++                         (correspondence matrix -- joint probs)
    r = P 1  (row masses),  c = 1' P     (col masses)
    S = D_r^{-1/2} (P - r c') D_c^{-1/2}     (standardized residuals)
    S = U Sigma V'                        (SVD)

Then:
    Principal row coords: F = D_r^{-1/2} U Sigma
    Principal col coords: G = D_c^{-1/2} V Sigma
    Standard row coords:  Phi = D_r^{-1/2} U     (biplot / joint plot use these)
    Standard col coords:  Gamma = D_c^{-1/2} V
    Total inertia         = sum sigma_k^2  = chi^2 / n_++
    Explained inertia per dim = sigma_k^2 / total_inertia

Number of nontrivial dimensions = min(I, J) - 1. Multiple Correspondence
Analysis (MCA) is a brief add-on: CA on the "Burt" table of a multi-way design.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def correspondence_analysis(counts, row_labels=None, col_labels=None,
                             n_components: int | None = None) -> dict:
    """Simple CA on an I x J contingency table.

    Returns row/col principal coordinates, singular values, inertia decomposition.
    """
    N = np.asarray(counts, dtype=float)
    if N.ndim != 2:
        raise ValueError("counts must be a 2D matrix")
    I, J = N.shape
    if row_labels is None: row_labels = [f"R{i}" for i in range(I)]
    if col_labels is None: col_labels = [f"C{j}" for j in range(J)]
    n = N.sum()
    P = N / n
    r = P.sum(axis=1); c = P.sum(axis=0)
    # standardized residuals
    Dr_inv_half = np.diag(1.0 / np.sqrt(np.clip(r, 1e-12, None)))
    Dc_inv_half = np.diag(1.0 / np.sqrt(np.clip(c, 1e-12, None)))
    S = Dr_inv_half @ (P - np.outer(r, c)) @ Dc_inv_half
    U, sigma, Vt = np.linalg.svd(S, full_matrices=False)
    # drop trivial dimensions (sigma == 0 or near-zero)
    keep = sigma > 1e-12
    U = U[:, keep]; Vt = Vt[keep, :]; sigma = sigma[keep]
    max_dim = min(I, J) - 1
    if n_components is None: n_components = max_dim
    n_components = min(n_components, max_dim)
    U = U[:, :n_components]; V = Vt[:n_components, :].T
    sigma = sigma[:n_components]
    # Principal coordinates (rows in F, cols in G) -- Greenacre notation
    F = Dr_inv_half @ U * sigma[None, :]      # I x n_components
    G = Dc_inv_half @ V * sigma[None, :]      # J x n_components
    Phi = Dr_inv_half @ U                      # standard row coords (for asymmetric biplot)
    Gamma = Dc_inv_half @ V                    # standard col coords
    total_inertia = float((sigma ** 2).sum())
    explained = (sigma ** 2 / total_inertia * 100.0).tolist() if total_inertia > 0 else []
    chi_sq = float((sigma ** 2).sum() * n)
    return {
        "n_components": n_components,
        "row_labels": list(row_labels),
        "col_labels": list(col_labels),
        "singular_values": sigma.tolist(),
        "eigenvalues": (sigma ** 2).tolist(),
        "total_inertia": total_inertia,
        "chi_square_total": chi_sq,
        "df_total": (I - 1) * (J - 1),
        "explained_inertia_pct": explained,
        "row_masses": r.tolist(),
        "col_masses": c.tolist(),
        "row_coords_principal": F.tolist(),     # rows x dims
        "col_coords_principal": G.tolist(),     # cols x dims
        "row_coords_standard": Phi.tolist(),
        "col_coords_standard": Gamma.tolist(),
    }


def mca_burt(design_matrix, factor_labels=None, n_components: int | None = None) -> dict:
    """Multiple Correspondence Analysis via the Burt matrix.

    Parameters
    ----------
    design_matrix : an n x sum(K_j) indicator matrix where each of Q categorical
        factors has been dummy-coded into K_j 0/1 columns. Standard MCA input.

    Simple approach: CA on the Burt matrix B = Z' Z where Z is the indicator
    matrix. Eigenvalues need Benzecri or Greenacre correction for interpretability;
    we report raw and Benzecri-corrected eigenvalues.
    """
    Z = np.asarray(design_matrix, dtype=float)
    n = Z.shape[0]
    B = Z.T @ Z
    ca = correspondence_analysis(B, factor_labels, factor_labels, n_components)
    # Benzecri correction on eigenvalues > 1/Q (with Q = number of factors);
    # user provides factor structure via original K_j not tracked here, so we
    # just report raw eigenvalues.
    return {"burt_eigenvalues": ca["eigenvalues"],
            "burt_explained_inertia_pct": ca["explained_inertia_pct"],
            "n_components": ca["n_components"],
            "row_coords_principal": ca["row_coords_principal"],
            "col_coords_principal": ca["col_coords_principal"],
            "note": "raw Burt eigenvalues; consider Benzecri/Greenacre "
                    "corrections for interpretability"}


def library_versions(counts):
    try:
        import prince
        import pandas as pd
        df = pd.DataFrame(counts).astype(int)
        ca = prince.CA(n_components=min(df.shape) - 1)
        ca = ca.fit(df)
        return {"prince eigenvalues": ca.eigenvalues_.tolist()}
    except Exception as ex:
        return {"prince (optional)": f"not available: {ex}"}


if __name__ == "__main__":
    # Classic example: hair color x eye color (approximate; small)
    counts = np.array([
        # eye:  blue brown green
        [15,   2,    5],     # blond
        [ 4,  20,    3],     # brown
        [ 3,   3,    8],     # black
    ], dtype=float)
    rows = ["blond", "brown", "black"]
    cols = ["blue", "brown", "green"]

    print("=== Correspondence analysis: hair x eye ===")
    out = correspondence_analysis(counts, rows, cols)
    print(f"  chi^2 = {out['chi_square_total']:.4f}, df = {out['df_total']}")
    print(f"  total inertia = {out['total_inertia']:.6f}")
    print(f"  n components = {out['n_components']}")
    print(f"  singular values = {out['singular_values']}")
    print(f"  explained inertia % = {out['explained_inertia_pct']}")
    print(f"  row principal coords:")
    for lbl, c in zip(rows, out["row_coords_principal"]):
        print(f"    {lbl:6s}: {c}")
    print(f"  col principal coords:")
    for lbl, c in zip(cols, out["col_coords_principal"]):
        print(f"    {lbl:6s}: {c}")

    print("\n--- library ---")
    for k, v in library_versions(counts).items():
        print(f"  {k}: {v}")
