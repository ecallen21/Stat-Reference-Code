"""Fractional polynomials (FP) (Reference Sec 5.14).

Royston & Altman 1994 'Regression using fractional polynomials of
continuous covariates', JRSS-C. A flexible parametric alternative to
splines for modelling nonlinear covariate effects.

Powers taken from the fixed set:

    P = { -2, -1, -0.5, 0, 0.5, 1, 2, 3 }   (0 = log(x))

Degree-1 FP:   f(x) = beta * x^p
Degree-2 FP:   f(x) = beta1 * x^p1 + beta2 * x^p2
   (with p1 == p2:  beta1 * x^p + beta2 * x^p * log(x))

Best-fitting FP is chosen by exhaustive search over the 8 (degree 1)
or 36 (degree 2) power combinations, comparing model deviance / log-lik.
Selection with FP1 vs FP2 vs linear typically uses a closed-test
procedure comparing deviances at prescribed significance levels.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays

POWERS = [-2, -1, -0.5, 0, 0.5, 1, 2, 3]


def transform(x, p):
    """Fractional-polynomial term x^p, with p=0 meaning log(x)."""
    if p == 0:
        return np.log(x)
    return x ** p


def fit_ols(X, y):
    """Closed-form OLS: beta = (X'X)^-1 X'y; also returns SSE."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, float(np.sum(resid ** 2))


def fp1(x, y):
    """Best degree-1 fractional polynomial."""
    n = len(x)
    best = None
    for p in POWERS:
        Xd = np.c_[np.ones(n), transform(x, p)]
        _, sse = fit_ols(Xd, y)
        if best is None or sse < best[1]:
            best = (p, sse)
    return {"p": best[0], "SSE": best[1]}


def fp2(x, y):
    """Best degree-2 fractional polynomial."""
    n = len(x)
    best = None
    for p1 in POWERS:
        for p2 in POWERS:
            if p2 < p1:
                continue
            t1 = transform(x, p1)
            if p1 == p2:
                t2 = t1 * np.log(x)      # repeated-power form
            else:
                t2 = transform(x, p2)
            Xd = np.c_[np.ones(n), t1, t2]
            _, sse = fit_ols(Xd, y)
            if best is None or sse < best[2]:
                best = (p1, p2, sse)
    return {"p1": best[0], "p2": best[1], "SSE": best[2]}


def closed_test_fp(x, y, alpha=0.05):
    """Royston-Sauerbrei function-selection procedure: FP2 vs FP1 vs linear vs null."""
    n = len(x)
    #  Null (intercept only)
    _, sse0 = fit_ols(np.ones((n, 1)), y)
    #  Linear
    _, sse_lin = fit_ols(np.c_[np.ones(n), x], y)
    fp1_ = fp1(x, y)
    fp2_ = fp2(x, y)

    def F_test(sse_reduced, sse_full, df_reduced, df_full):
        num = (sse_reduced - sse_full) / (df_full - df_reduced)
        den = sse_full / (n - df_full)
        F = num / den
        from scipy.stats import f
        p = 1 - f.cdf(F, df_full - df_reduced, n - df_full)
        return F, p

    #  Step 1: FP2 vs null (4 df: intercept + 2 terms + best-of-search penalty ~4)
    F1, p_2vs0 = F_test(sse0, fp2_["SSE"], 1, 3)
    if p_2vs0 > alpha:
        return {"chosen": "null", "SSE": sse0}
    #  Step 2: FP2 vs linear
    F2, p_2vs_lin = F_test(sse_lin, fp2_["SSE"], 2, 3)
    if p_2vs_lin > alpha:
        return {"chosen": "linear", "SSE": sse_lin}
    #  Step 3: FP2 vs FP1
    F3, p_2vs1 = F_test(fp1_["SSE"], fp2_["SSE"], 2, 3)
    if p_2vs1 > alpha:
        return {"chosen": "FP1", "p": fp1_["p"], "SSE": fp1_["SSE"]}
    return {"chosen": "FP2", "p1": fp2_["p1"], "p2": fp2_["p2"], "SSE": fp2_["SSE"]}


if __name__ == "__main__":
    print("=== Fractional polynomials -- Royston & Altman ===\n")
    rng = np.random.default_rng(0)
    n = 400
    #  True curve: J-shape, y = 3 - 4/sqrt(x) + 0.02*x  (needs power -0.5 and 1)
    x = rng.uniform(1, 40, size=n)
    y_true = 3 - 4 / np.sqrt(x) + 0.02 * x
    y = y_true + rng.normal(scale=0.4, size=n)

    print("  True function: y = 3 - 4/sqrt(x) + 0.02*x   (powers -0.5 and 1)")
    print()

    r1 = fp1(x, y)
    print(f"  Best FP1:  power = {r1['p']:>+5.2f}   SSE = {r1['SSE']:.2f}")

    r2 = fp2(x, y)
    print(f"  Best FP2:  powers = ({r2['p1']:>+5.2f}, {r2['p2']:>+5.2f})   SSE = {r2['SSE']:.2f}")

    r_full = closed_test_fp(x, y)
    print(f"\n  Closed-test selection: {r_full}")

    print("\n--- library cross-check (mfp R; from-scratch Python) ---")
