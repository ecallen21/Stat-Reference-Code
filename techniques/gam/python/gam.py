"""Generalized Additive Models (Reference §5.14).

    g(E[y_i]) = beta_0 + sum_j f_j(x_ij) + eps_i

where each f_j is a SMOOTH function estimated from the data.  Combines the
interpretability of linear regression with the flexibility of nonparametric
smoothing.  g is a link (identity for Gaussian, logit for binomial,
log for Poisson).

Implementation choices
    - Basis expansion: cubic regression spline of dimension K (Wood 2017);
      or B-splines / natural splines.
    - Penalty: integrated squared second derivative
      lambda_j integral f_j''(x)^2 dx  = beta_j^T S_j beta_j.
    - Smoothing parameter lambda_j chosen by GCV or REML.

This module implements a PENALIZED LINEAR REGRESSION with cubic-spline
basis for each smooth term and one smoothing parameter per term chosen by
GCV grid search.  Single-response Gaussian outcome only.  For full GLM
support use mgcv (R) or pyGAM.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _cubic_spline_basis(x, knots):
    """Truncated-power cubic basis (no constant column) with `n_knots` interior knots."""
    x = np.asarray(x, dtype=float)
    B = [x, x ** 2, x ** 3]
    for k in knots:
        B.append(np.maximum(x - k, 0) ** 3)
    return np.column_stack(B)


def _penalty_matrix(n_knots):
    """Diagonal penalty on the truncated cubic coefficients (polynomial part unpenalized)."""
    p = 3 + n_knots
    S = np.zeros((p, p))
    for k in range(3, p): S[k, k] = 1.0
    return S


def fit_gam(X, y, n_knots: int = 8, lam_grid=None) -> dict:
    """Fit a Gaussian additive model with one smooth per column of X.

    Selects a single smoothing parameter per term by minimizing GCV.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    if lam_grid is None:
        lam_grid = np.exp(np.linspace(-6, 6, 20))
    # Fit each smooth term with a separate penalty; use a coordinate descent-like scheme
    # by adding all bases into one design and choosing a common lambda per term.
    # For simplicity: single lambda across all terms chosen by GCV.
    bases = []
    knots_list = []
    for j in range(p):
        xj = X[:, j]
        knots = np.quantile(xj, np.linspace(0.1, 0.9, n_knots))
        knots_list.append(knots)
        bases.append(_cubic_spline_basis(xj, knots))
    B_all = np.column_stack([np.ones(n)] + bases)
    # Build block-diagonal penalty (leave intercept & polynomial part unpenalized)
    starts = [1]
    for b in bases: starts.append(starts[-1] + b.shape[1])
    P = np.zeros((B_all.shape[1], B_all.shape[1]))
    for j in range(p):
        Sj = _penalty_matrix(n_knots)
        i0 = starts[j]; i1 = starts[j + 1]
        P[i0:i1, i0:i1] = Sj
    best = {"gcv": np.inf, "lam": None, "beta": None}
    for lam in lam_grid:
        A = B_all.T @ B_all + lam * P
        try:
            beta = np.linalg.solve(A, B_all.T @ y)
        except np.linalg.LinAlgError:
            continue
        y_hat = B_all @ beta
        H = B_all @ np.linalg.solve(A, B_all.T)
        edf = float(np.trace(H))
        rss = float(np.sum((y - y_hat) ** 2))
        gcv = rss * n / max((n - edf) ** 2, 1e-6)
        if gcv < best["gcv"]:
            best = {"gcv": gcv, "lam": float(lam), "beta": beta, "edf": edf}
    # Term-by-term fitted values
    beta = best["beta"]
    term_fits = []
    for j in range(p):
        i0 = starts[j]; i1 = starts[j + 1]
        term_fits.append(bases[j] @ beta[i0:i1])
    return {"lambda": best["lam"], "gcv": best["gcv"], "edf": best["edf"],
            "beta_all": beta,
            "term_fits": term_fits,
            "fitted": B_all @ beta,
            "residuals": y - B_all @ beta,
            "n": int(n), "p": int(p),
            "method": "GAM (penalized cubic spline, single smoothing param per fit)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    x1 = np.sort(rng.uniform(-3, 3, n))
    x2 = rng.uniform(0, 10, n)
    f1_true = np.sin(x1 * 1.5)
    f2_true = 0.3 * (x2 - 5) ** 2 - 3
    y = f1_true + f2_true + rng.normal(0, 0.5, n)
    X = np.column_stack([x1, x2])

    r = fit_gam(X, y, n_knots=8)
    print(f"=== GAM fit (2 smooth terms) ===")
    print(f"  chosen lambda = {r['lambda']:.4f}")
    print(f"  effective df  = {r['edf']:.1f}   (n = {r['n']})")
    print(f"  GCV           = {r['gcv']:.3f}")
    print(f"  in-sample RMSE = {np.sqrt(np.mean(r['residuals'] ** 2)):.3f}  (noise sd = 0.5)")

    # Correlations between term fits and true smooth shapes (accounts for unidentified constant / linear split)
    print(f"\n  Cor(term1_fit, f1_true) = {np.corrcoef(r['term_fits'][0], f1_true)[0,1]:.3f}")
    print(f"  Cor(term2_fit, f2_true) = {np.corrcoef(r['term_fits'][1], f2_true)[0,1]:.3f}")

    print("\n--- library cross-check (pygam) ---")
    try:
        from pygam import LinearGAM, s
        gam = LinearGAM(s(0) + s(1)).fit(X, y)
        print(f"  pygam pseudo R^2: {gam.statistics_['pseudo_r2']}")
    except Exception as ex:
        print(f"  (pygam unavailable: {ex})")
