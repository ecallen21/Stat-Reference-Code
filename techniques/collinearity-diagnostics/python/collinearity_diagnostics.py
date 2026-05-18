"""Collinearity diagnostics for OLS regression (Reference §5.23).

Collinearity = strong linear relationships among the predictors (the design
matrix columns), making Var(beta_hat) explode and individual coefficients
unstable / hard to interpret.

Diagnostics implemented
-----------------------
- Pairwise correlations    : easy first pass; misses joint multicollinearity
- VIF (Variance Inflation Factor) for each predictor x_j:
        VIF_j = 1 / (1 - R_j^2)
  where R_j^2 is from regressing x_j on the OTHER predictors.
  Tolerance_j = 1 / VIF_j.
- Condition indices and variance-decomposition proportions (Belsley/Kuh/Welsch):
        kappa_k = sqrt(lambda_max / lambda_k)
  for the singular values of the SCALED design matrix.

Rules of thumb
  VIF > 5     : noticeable collinearity (some say > 4)
  VIF > 10    : serious collinearity, consider remedies
  kappa > 30  : moderate-to-severe; check the variance-decomposition proportions
                for pairs of coefficients with > 0.5 of the variance loaded on
                the same near-singular dimension.

Remedies (see §5.29 of the reference):
  - drop one of a near-redundant pair (if substantively justifiable)
  - centering helps for interaction / polynomial terms (techniques/interaction-terms)
  - use regularization (techniques/regularization -- ridge / elastic net)
  - PCR / PLS (future batch)
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def pairwise_correlations(X, names: Sequence[str] | None = None) -> dict:
    """Pearson correlation matrix among the non-intercept columns of X."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    has_int = bool(np.allclose(X[:, 0], 1.0))
    Xc = X[:, 1:] if has_int else X
    if names is None:
        names = [f"x{i+1}" for i in range(Xc.shape[1])]
    else:
        if has_int and len(names) == p:
            names = list(names)[1:]
    return {"names": list(names),
            "correlation": np.corrcoef(Xc, rowvar=False).tolist()}


def vif(X, names: Sequence[str] | None = None) -> dict:
    """Variance Inflation Factor for each non-intercept column of X.

    VIF_j = 1 / (1 - R_j^2) where R_j^2 is from regressing x_j on the others.
    """
    X = np.asarray(X, dtype=float); n, p = X.shape
    has_int = bool(np.allclose(X[:, 0], 1.0))
    Xc = X[:, 1:] if has_int else X
    k = Xc.shape[1]
    if names is None:
        names = [f"x{i+1}" for i in range(k)]
    out = {"names": list(names), "vif": [], "tolerance": []}
    for j in range(k):
        y = Xc[:, j]
        Xj = np.column_stack([np.ones(n), np.delete(Xc, j, axis=1)])
        beta, *_ = np.linalg.lstsq(Xj, y, rcond=None)
        e = y - Xj @ beta
        rss = float(e @ e); tss = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - rss / tss if tss > 0 else 0.0
        v = 1 / max(1e-15, 1 - r2)
        out["vif"].append(v); out["tolerance"].append(1 / v)
    return out


def condition_indices(X, names: Sequence[str] | None = None) -> dict:
    """Belsley-Kuh-Welsch condition indices on the SCALED design matrix.

    Each column of X is divided by its Euclidean norm. The condition index for
    singular value lambda_k is sqrt(lambda_max / lambda_k). The
    variance-decomposition proportions show how much of Var(beta_j) is loaded
    on each near-singular dimension.
    """
    X = np.asarray(X, dtype=float); n, p = X.shape
    if names is None:
        names = [("intercept" if np.allclose(X[:, i], 1.0) else f"x{i}")
                 for i in range(p)]
    # Scale columns to unit Euclidean norm
    norms = np.linalg.norm(X, axis=0)
    Xs = X / np.where(norms == 0, 1, norms)
    U, sing, Vt = np.linalg.svd(Xs, full_matrices=False)
    sing = np.where(sing == 0, 1e-15, sing)
    kappa = sing.max() / sing
    # Variance-decomposition proportions: pi_{kj} = (V_{jk}^2 / sing_k^2) / sum_k (V_{jk}^2 / sing_k^2)
    V = Vt.T                                    # p x p
    raw = (V ** 2) / (sing ** 2)
    col_sum = raw.sum(axis=1, keepdims=True)
    proportions = raw / np.where(col_sum == 0, 1, col_sum)
    return {"names": list(names),
            "singular_values": sing.tolist(),
            "condition_indices": kappa.tolist(),
            "max_condition_index": float(kappa.max()),
            "variance_proportions": proportions.tolist()}


def library_versions(X):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    return {f"statsmodels VIF[{j}]": float(variance_inflation_factor(np.asarray(X), j))
            for j in range(np.asarray(X).shape[1])}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 80
    # Build x1, x2 with x2 highly collinear with x1
    x1 = rng.normal(0, 1, n)
    x2 = 0.95 * x1 + rng.normal(0, 0.2, n)
    x3 = rng.normal(0, 1, n)
    X = np.column_stack([np.ones(n), x1, x2, x3])

    print("=== Pairwise correlations among predictors ===")
    pc = pairwise_correlations(X, names=["intercept", "x1", "x2", "x3"])
    print(f"  cols: {pc['names']}")
    for row in pc["correlation"]:
        print("   ", [round(v, 3) for v in row])

    print("\n=== VIF ===")
    res = vif(X, names=["x1", "x2", "x3"])
    for nm, v, tol in zip(res["names"], res["vif"], res["tolerance"]):
        print(f"  {nm:6s}: VIF = {v:7.2f}   tolerance = {tol:.4f}")

    print("\n=== Condition indices / variance proportions ===")
    ci = condition_indices(X, names=["intercept", "x1", "x2", "x3"])
    print(f"  max condition index: {ci['max_condition_index']:.2f}")
    print("  condition indices:", [round(k, 2) for k in ci["condition_indices"]])
    print("  variance proportions (rows = coefficients, cols = singular dims):")
    print(f"  {'name':10s} " + "  ".join(f"sd{i+1}" for i in range(len(ci['singular_values']))))
    for nm, row in zip(ci["names"], ci["variance_proportions"]):
        print(f"  {nm:10s} " + "  ".join(f"{v:.3f}" for v in row))

    print("\n--- library (statsmodels VIF) ---")
    for k, v in library_versions(X).items():
        print(f"  {k}: {v}")
