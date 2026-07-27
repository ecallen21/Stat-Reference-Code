"""Canonical Correlation Analysis (Reference §9.29).

Given two sets of variables X (n x p) and Y (n x q) measured on the same
subjects, CCA finds the linear combinations U = X a and V = Y b that are
maximally correlated. Then the next pair (u_2, v_2) is found that is also
maximally correlated but uncorrelated with (u_1, v_1). And so on for
min(p, q) pairs.

Algorithm (via generalized eigenproblem or double SVD):
    Compute cross-covariance R_xy = X_c' Y_c / (n - 1)
    Compute R_xx = X_c' X_c / (n - 1)  and R_yy = Y_c' Y_c / (n - 1)
    Solve R_xx^{-1/2} R_xy R_yy^{-1} R_yx R_xx^{-1/2} = A Lambda A'
    Canonical correlations = sqrt(Lambda_ii)
    Canonical weights (a, b) come from A and the analogous factor for Y.

Bartlett's chi-square test (based on Wilks' Lambda):
    Test that AT LEAST k+1 canonical correlations are non-zero given the
    first k are, for k = 0..s-1.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _matrix_pow(M, p, tol=1e-12):
    """Symmetric matrix power via eigendecomposition."""
    w, V = np.linalg.eigh(M)
    w = np.where(w > tol, w, 0.0)
    return V @ np.diag(w ** p if not np.all(w == 0) else w) @ V.T


def canonical_correlation(X, Y) -> dict:
    """Standard CCA on centered X (n x p) and Y (n x q).

    Returns canonical correlations, canonical weights, canonical variates,
    and Bartlett sequential test.
    """
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    n, p = X.shape
    n2, q = Y.shape
    if n != n2:
        raise ValueError("X and Y must have the same number of rows")
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    Rxx = Xc.T @ Xc / (n - 1)
    Ryy = Yc.T @ Yc / (n - 1)
    Rxy = Xc.T @ Yc / (n - 1)
    # M = Rxx^{-1/2} Rxy Ryy^{-1} Rxy' Rxx^{-1/2}
    Rxx_ihalf = _matrix_pow(Rxx, -0.5)
    Ryy_inv = np.linalg.pinv(Ryy)
    M = Rxx_ihalf @ Rxy @ Ryy_inv @ Rxy.T @ Rxx_ihalf
    lam, A = np.linalg.eigh(M)
    # sort descending
    order = np.argsort(-lam)
    lam = np.clip(lam[order], 0.0, 1.0)
    A = A[:, order]
    r = np.sqrt(lam)
    # Canonical weights
    Wx = Rxx_ihalf @ A                                   # p x min(p,q)
    s = min(p, q)
    Wx = Wx[:, :s]; r = r[:s]
    # Corresponding Y weights via Wy = Ryy^{-1} Rxy' Wx / r (with safe divide)
    Wy = Ryy_inv @ Rxy.T @ Wx
    with np.errstate(divide="ignore", invalid="ignore"):
        Wy = Wy / np.where(r > 0, r, 1.0)
    # Bartlett sequential chi-square (Bartlett's Lambda test)
    # After removing the first k canonical correlations, test remainder = 0
    bartlett = []
    for k in range(s):
        rem = r[k:]
        wilks = float(np.prod(1 - rem ** 2))
        # Bartlett approximation
        stat = -(n - 1 - (p + q + 1) / 2) * math.log(max(wilks, 1e-300))
        df = (p - k) * (q - k)
        p_val = float(stats.chi2.sf(stat, df)) if df > 0 else float("nan")
        bartlett.append({"after_k_kept": k, "wilks_lambda": wilks,
                         "chi_square": stat, "df": df, "p_value": p_val})
    # Canonical variates
    U = Xc @ Wx; V = Yc @ Wy
    return {"canonical_correlations": r.tolist(),
            "X_weights": Wx.tolist(),
            "Y_weights": Wy.tolist(),
            "X_variates_head": U[:5].tolist(),
            "Y_variates_head": V[:5].tolist(),
            "bartlett_sequential": bartlett,
            "n": n, "p": p, "q": q, "n_pairs": s,
            "method": "CCA via generalized eigendecomposition"}


def library_versions(X, Y):
    from sklearn.cross_decomposition import CCA
    n_components = min(X.shape[1], Y.shape[1])
    cca = CCA(n_components=n_components).fit(X, Y)
    Xc_t, Yc_t = cca.transform(X, Y)
    corrs = [float(np.corrcoef(Xc_t[:, i], Yc_t[:, i])[0, 1]) for i in range(n_components)]
    return {"sklearn canonical correlations (sample)": corrs}


if __name__ == "__main__":
    rng = np.random.default_rng(97)
    n = 300
    # Two shared latent factors; X gets loadings, Y gets loadings
    F = rng.normal(0, 1, size=(n, 2))
    Lx = np.array([[0.8, 0.1],
                   [0.7, 0.2],
                   [0.6, 0.3]])
    Ly = np.array([[0.1, 0.8],
                   [0.2, 0.7]])
    X = F @ Lx.T + rng.normal(0, 0.4, size=(n, 3))
    Y = F @ Ly.T + rng.normal(0, 0.4, size=(n, 2))

    print("=== CCA on X (n x 3), Y (n x 2) ===")
    out = canonical_correlation(X, Y)
    print(f"  canonical correlations: {out['canonical_correlations']}")
    print(f"  X weights (columns = canonical variates):")
    for row in out["X_weights"]:
        print(f"    {row}")
    print(f"  Y weights:")
    for row in out["Y_weights"]:
        print(f"    {row}")
    print(f"\n  Bartlett sequential tests:")
    for b in out["bartlett_sequential"]:
        print(f"    after keeping {b['after_k_kept']}: Wilks={b['wilks_lambda']:.4f}, "
              f"chi2={b['chi_square']:.2f}, df={b['df']}, p={b['p_value']:.4g}")

    print("\n--- library (sklearn) ---")
    for k, v in library_versions(X, Y).items():
        print(f"  {k}: {v}")
