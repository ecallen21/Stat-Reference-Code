"""Exploratory Factor Analysis (Reference §9.4).

EFA models each observed variable X_j as a linear combination of k unobserved
common factors F plus a variable-specific "unique" component:

    X_j  =  Lambda_j1 * F_1 + ... + Lambda_jk * F_k + U_j

so the correlation / covariance matrix decomposes as:

    Sigma  =  Lambda Lambda'  +  Psi
              (common part)    (diagonal uniquenesses)

Extraction methods:
    - Principal Axis Factoring (PAF): iterate on the reduced correlation matrix
      (diagonal replaced by communalities); a simple, robust default.
    - Maximum Likelihood (ML): joint MLE over Lambda and Psi (statsmodels/factor_analyzer).

Rotation (interpretation aid; does not change fit):
    - Varimax: orthogonal rotation maximizing the variance of squared loadings
      within columns (each factor loads highly on a few variables).
    - Promax: oblique rotation obtained by raising varimax loadings to a power
      (typically 4) then Procrustes-fitting the target.

Key outputs: loadings (Lambda), communalities h^2_j = sum_k Lambda_jk^2, and
uniquenesses u^2_j = 1 - h^2_j.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _paf(R, n_factors, max_iter=100, tol=1e-6):
    """Principal Axis Factoring on a correlation matrix R."""
    R = np.asarray(R, dtype=float)
    p = R.shape[0]
    # start with squared multiple correlations for communalities
    R_inv = np.linalg.pinv(R)
    h2 = 1 - 1.0 / np.clip(np.diag(R_inv), 1e-12, None)
    h2 = np.clip(h2, 0.05, 0.99)
    L = None
    for _ in range(max_iter):
        R_reduced = R.copy(); np.fill_diagonal(R_reduced, h2)
        w, v = np.linalg.eigh(R_reduced)
        # sort descending
        order = np.argsort(-w)
        w = w[order]; v = v[:, order]
        w_top = np.clip(w[:n_factors], 1e-12, None)
        v_top = v[:, :n_factors]
        L = v_top * np.sqrt(w_top)[None, :]
        h2_new = (L ** 2).sum(axis=1)
        h2_new = np.clip(h2_new, 0.05, 0.99)
        if np.max(np.abs(h2_new - h2)) < tol:
            h2 = h2_new; break
        h2 = h2_new
    return L, h2


def varimax(L, gamma: float = 1.0, max_iter: int = 100, tol: float = 1e-8):
    """Kaiser's varimax rotation. Returns rotated loadings and rotation matrix."""
    L = np.asarray(L, dtype=float)
    p, k = L.shape
    R = np.eye(k)
    d = 0
    for _ in range(max_iter):
        d_old = d
        Lam = L @ R
        # Kaiser: compute variance of columns of Lam^2 minus their column means
        u, s, vh = np.linalg.svd(L.T @ (Lam ** 3 - (gamma / p) * Lam @ np.diag((Lam ** 2).sum(axis=0))))
        R = u @ vh
        d = s.sum()
        if abs(d - d_old) < tol:
            break
    return L @ R, R


def promax(L, kappa: int = 4):
    """Oblique promax rotation via varimax then Procrustes to Lambda_varimax^kappa."""
    L_var, R_var = varimax(L)
    # target: sign of L_var * |L_var|^kappa
    target = np.sign(L_var) * np.abs(L_var) ** kappa
    # least-squares Procrustes: T = (L_var' L_var)^{-1} L_var' target
    T = np.linalg.solve(L_var.T @ L_var, L_var.T @ target)
    # Rescale T so diag(Phi) = diag((T'T)^{-1}) = 1 (Phi becomes a correlation matrix).
    # If T -> T D, then (T'T)^{-1} -> D^{-1} (T'T)^{-1} D^{-1}, whose diagonal is
    # d_j^{-2} * ((T'T)^{-1})_jj. Setting that to 1 gives d_j = sqrt(((T'T)^{-1})_jj).
    A_inv = np.linalg.inv(T.T @ T)
    D = np.diag(np.sqrt(np.diag(A_inv)))
    T = T @ D
    L_promax = L_var @ T
    Phi = np.linalg.inv(T.T @ T)         # factor correlation matrix (diag = 1)
    return L_promax, T, Phi


def fit_efa(X, n_factors: int, method: str = "paf", rotation: str = "varimax") -> dict:
    """Fit EFA and return loadings, communalities, uniquenesses, and rotation."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    # correlation matrix
    Xc = X - X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    Z = Xc / np.where(sd > 0, sd, 1.0)
    R = np.corrcoef(Z, rowvar=False)

    if method == "paf":
        L, h2 = _paf(R, n_factors)
    else:
        raise ValueError("only 'paf' is implemented from-scratch; use library MLE for 'ml'")

    if rotation is None or rotation == "none":
        L_rot = L; rot_matrix = np.eye(n_factors); Phi = None
    elif rotation == "varimax":
        L_rot, rot_matrix = varimax(L); Phi = None
    elif rotation == "promax":
        L_rot, rot_matrix, Phi = promax(L)
    else:
        raise ValueError("rotation must be 'none', 'varimax', or 'promax'")

    if Phi is None:
        communalities = (L_rot ** 2).sum(axis=1)
    else:
        # oblique: h^2_j = L_j Phi L_j' where L_j is the j-th ROW (length k)
        communalities = np.array([float(L_rot[j, :] @ Phi @ L_rot[j, :]) for j in range(L_rot.shape[0])])
    uniquenesses = 1 - communalities
    # variance explained per factor (post rotation) -- for oblique, we report the
    # raw sum-of-squared-loadings (ignoring Phi) as psych::fa does; a proper
    # variance decomposition under oblique factors is not unique.
    ss_loadings = (L_rot ** 2).sum(axis=0)
    return {"loadings": L_rot.tolist(),
            "communalities": communalities.tolist(),
            "uniquenesses": uniquenesses.tolist(),
            "ss_loadings_per_factor": ss_loadings.tolist(),
            "prop_variance_per_factor": (ss_loadings / p).tolist(),
            "rotation_matrix": rot_matrix.tolist(),
            "factor_correlation_matrix": Phi.tolist() if Phi is not None else None,
            "method": f"EFA ({method}) + {rotation} rotation",
            "n": n, "p": p, "k": n_factors}


def library_versions(X, n_factors):
    try:
        from factor_analyzer import FactorAnalyzer
        fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax", method="principal")
        fa.fit(X)
        return {"factor_analyzer varimax loadings":
                fa.loadings_.tolist(),
                "communalities": fa.get_communalities().tolist()}
    except Exception as ex:
        return {"factor_analyzer (optional)": f"not available: {ex}"}


if __name__ == "__main__":
    rng = np.random.default_rng(31)
    # 6 variables loading on 2 factors
    n = 300
    F = rng.normal(0, 1, size=(n, 2))
    L_true = np.array([
        [0.8, 0.1],
        [0.7, 0.2],
        [0.9, 0.0],
        [0.1, 0.7],
        [0.2, 0.8],
        [0.0, 0.9],
    ])
    U = rng.normal(0, 0.5, size=(n, 6))
    X = F @ L_true.T + U

    print("=== EFA (PAF + varimax), k=2 ===")
    out = fit_efa(X, n_factors=2, method="paf", rotation="varimax")
    print("  loadings (rotated):")
    for i, row in enumerate(out["loadings"]):
        print(f"    V{i}: [{row[0]:+.3f}, {row[1]:+.3f}]  h^2={out['communalities'][i]:.3f}")
    print(f"  SS loadings per factor: {out['ss_loadings_per_factor']}")
    print(f"  prop variance per factor: {out['prop_variance_per_factor']}")

    print("\n=== EFA (PAF + promax), k=2 ===")
    out = fit_efa(X, n_factors=2, method="paf", rotation="promax")
    print("  loadings (rotated):")
    for i, row in enumerate(out["loadings"]):
        print(f"    V{i}: [{row[0]:+.3f}, {row[1]:+.3f}]")
    print(f"  factor correlations Phi: {out['factor_correlation_matrix']}")

    print("\n--- library ---")
    for k, v in library_versions(X, 2).items():
        print(f"  {k}: {v}")
