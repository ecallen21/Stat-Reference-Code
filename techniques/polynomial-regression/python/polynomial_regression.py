"""Polynomial regression (Reference §5.3).

Two ways to put polynomial terms into a linear model:

  Raw       y = b0 + b1 x + b2 x^2 + ... + bd x^d + eps
            Easy to interpret coefficients, but x, x^2, ... are highly
            correlated -> VIF explodes; coefficients aren't stable.

  Orthogonal y = c0 P0(x) + c1 P1(x) + ... + cd Pd(x) + eps
            P_k(x) are an orthonormal polynomial basis (Gram-Schmidt on
            1, x, x^2, ..., x^d). Coefficients ARE uncorrelated, so a
            test on c_k is a test on "the degree-k contribution above
            and beyond the lower-degree fit." The fitted values and R^2
            are identical to the raw fit.

This file:
  - builds raw and orthogonal design matrices (`poly_features`, `ortho_poly`)
  - fits both via OLS and prints coefficients side by side
  - shows that the FITS are identical despite different coefficients
  - includes a "what degree?" walk-through: fit increasing degrees and stop
    when the highest-order coefficient is no longer significant
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats


def poly_features(x, degree: int, raw: bool = True) -> np.ndarray:
    """Build n x (degree + 1) polynomial feature matrix.

    ``raw=True`` -> [1, x, x^2, ..., x^d]
    ``raw=False`` -> orthonormal polynomials via Gram-Schmidt (column 0 is all 1/sqrt(n))
    """
    x = np.asarray(x, dtype=float)
    P = np.column_stack([x ** k for k in range(degree + 1)])
    if raw:
        return P
    # Modified Gram-Schmidt
    Q = np.zeros_like(P)
    for k in range(P.shape[1]):
        v = P[:, k].copy()
        for j in range(k):
            v -= (Q[:, j] @ v) * Q[:, j]
        norm = np.linalg.norm(v)
        Q[:, k] = v / (norm if norm > 0 else 1)
    return Q


def fit_polynomial(x, y, degree: int, raw: bool = True) -> dict:
    """Fit y on a polynomial of x. Returns coefficients with SEs and overall fit."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    X = poly_features(x, degree, raw=raw)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta; e = y - yhat
    n, p = X.shape
    rss = float(e @ e); df_r = n - p
    sigma2 = rss / df_r
    var_beta = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(var_beta))
    t = beta / se; p_vals = 2 * stats.t.sf(np.abs(t), df_r)
    tss = float(((y - y.mean()) ** 2).sum())
    return {"degree": degree, "raw": raw,
            "beta": beta.tolist(), "se": se.tolist(),
            "t_stats": t.tolist(), "p_values": p_vals.tolist(),
            "r_squared": 1 - rss / tss,
            "sigma_hat": float(np.sqrt(sigma2)),
            "df_residual": df_r,
            "fitted": yhat.tolist(), "residuals": e.tolist()}


def choose_degree(x, y, max_degree: int = 6, alpha: float = 0.05) -> dict:
    """Fit degrees 1..max_degree on the *orthogonal* basis; choose the largest
    degree whose top-order coefficient is significant at level ``alpha``.

    Using the orthogonal basis is the key here -- the test on coefficient c_k
    is the test for "the k-th degree component" without being confounded by the
    lower-order terms (which is what makes raw-polynomial t-tests misleading
    for selection).
    """
    results = []
    chosen = 1
    for d in range(1, max_degree + 1):
        fit = fit_polynomial(x, y, d, raw=False)
        results.append({"degree": d,
                        "top_t": fit["t_stats"][-1],
                        "top_p": fit["p_values"][-1],
                        "r_squared": fit["r_squared"]})
        if fit["p_values"][-1] < alpha:
            chosen = d
    return {"results": results, "chosen_degree": chosen}


def library_versions(x, y, degree=3):
    import statsmodels.api as sm
    import numpy as np
    # statsmodels has no built-in orthogonal polynomials; use raw via column_stack
    X = sm.add_constant(np.column_stack([x ** k for k in range(1, degree + 1)]))
    res = sm.OLS(y, X).fit()
    return {"statsmodels raw poly coefs": res.params.tolist(),
            "statsmodels raw poly p-values": res.pvalues.tolist(),
            "statsmodels rsquared": float(res.rsquared)}


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    n = 100
    x = np.linspace(-3, 3, n) + rng.normal(0, 0.1, n)
    # True model: y = 1 + 0.5 x + 0.8 x^2 + noise
    y = 1 + 0.5 * x + 0.8 * x ** 2 + rng.normal(0, 0.7, n)

    print("=== Raw polynomial, degree 3 ===")
    raw = fit_polynomial(x, y, 3, raw=True)
    for k in range(4):
        print(f"  beta_{k}  : {raw['beta'][k]:+.4f}  SE {raw['se'][k]:.4f}  "
              f"t {raw['t_stats'][k]:.2f}  p {raw['p_values'][k]:.4g}")
    print(f"  R^2 = {raw['r_squared']:.4f}   sigma = {raw['sigma_hat']:.4f}")

    print("\n=== Orthogonal polynomial, degree 3 ===")
    orth = fit_polynomial(x, y, 3, raw=False)
    for k in range(4):
        print(f"  c_{k}     : {orth['beta'][k]:+.4f}  SE {orth['se'][k]:.4f}  "
              f"t {orth['t_stats'][k]:.2f}  p {orth['p_values'][k]:.4g}")
    print(f"  R^2 = {orth['r_squared']:.4f}   sigma = {orth['sigma_hat']:.4f}")
    print("  Note: raw and orthogonal R^2 / sigma are identical; the COEFS differ.")

    print("\n=== Choose degree via orthogonal basis (alpha = 0.05) ===")
    sel = choose_degree(x.tolist(), y.tolist(), max_degree=6)
    for r in sel["results"]:
        print(f"  d = {r['degree']}: top t = {r['top_t']:+.3f}  p = {r['top_p']:.4g}  R^2 = {r['r_squared']:.4f}")
    print(f"  Chosen degree: {sel['chosen_degree']}")

    print("\n--- library (statsmodels, raw polynomial) ---")
    for k, v in library_versions(x, y, degree=3).items():
        print(f"  {k}: {v}")
