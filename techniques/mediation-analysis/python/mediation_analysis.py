"""Mediation analysis: Baron-Kenny + natural direct/indirect effects
(Reference §15.15; Imai-Keele-Tingley 2010).

Model:
    T -> M -> Y             indirect / mediated path
    T -> Y                  direct path

Baron-Kenny (1986) three-regression product method:
    M = alpha + a * T + eps_M
    Y = beta_0 + c' T + b * M + eps_Y
    indirect effect  =  a * b
    direct effect    =  c'
    total effect     =  a * b + c'

Modern causal formulation (Robins-Greenland 1992; Pearl 2001; Imai-Keele-
Tingley 2010) using POTENTIAL OUTCOMES:
    NDE = E[Y(t = 1, M = M(0))] - E[Y(t = 0, M = M(0))]
    NIE = E[Y(t = 1, M = M(1))] - E[Y(t = 1, M = M(0))]
    Total effect = NDE + NIE

Under NO INTERACTION and linear models, NDE = c', NIE = a * b (matches BK).
With interactions, NDE / NIE need explicit standardization (imai/mediation).

Bootstrap for SEs / CIs (nonparametric percentile CI).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def baron_kenny(T, M, Y) -> dict:
    """Baron-Kenny mediation."""
    T = np.asarray(T, dtype=float); M = np.asarray(M, dtype=float); Y = np.asarray(Y, dtype=float)
    n = len(T)
    # M ~ 1 + T
    X = np.column_stack([np.ones(n), T])
    beta_m, *_ = np.linalg.lstsq(X, M, rcond=None)
    a = float(beta_m[1])
    # Y ~ 1 + T + M
    X2 = np.column_stack([np.ones(n), T, M])
    beta_y, *_ = np.linalg.lstsq(X2, Y, rcond=None)
    c_prime = float(beta_y[1]); b = float(beta_y[2])
    # Total from Y ~ 1 + T
    beta_total, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), T]), Y, rcond=None)
    total = float(beta_total[1])
    return {"a_path (T->M)": a, "b_path (M->Y|T)": b,
            "c_prime (direct T->Y|M)": c_prime,
            "indirect ab": a * b,
            "total effect (T->Y)": total,
            "proportion_mediated": (a * b) / total if abs(total) > 1e-9 else float("nan"),
            "method": "Baron-Kenny product-of-coefficients"}


def bootstrap_indirect(T, M, Y, B: int = 1000, seed: int = 0) -> dict:
    """Bootstrap CI for the indirect (a*b) effect."""
    rng = np.random.default_rng(seed)
    n = len(T)
    ab_star = np.empty(B)
    for k in range(B):
        idx = rng.integers(0, n, n)
        r = baron_kenny(T[idx], M[idx], Y[idx])
        ab_star[k] = r["indirect ab"]
    return {"ab_mean": float(ab_star.mean()),
            "ab_se": float(ab_star.std(ddof=1)),
            "ab_ci_95": (float(np.quantile(ab_star, 0.025)),
                         float(np.quantile(ab_star, 0.975))),
            "B": int(B)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    T = rng.binomial(1, 0.5, n).astype(float)
    M = 1 + 0.8 * T + rng.normal(0, 1, n)       # a = 0.8
    Y = 2 + 0.3 * T + 0.5 * M + rng.normal(0, 1, n)  # c' = 0.3, b = 0.5
    # true indirect = a*b = 0.4; total = 0.3 + 0.4 = 0.7

    print("=== Baron-Kenny mediation (true a = 0.8, b = 0.5, c' = 0.3) ===")
    r = baron_kenny(T, M, Y)
    for k, v in r.items():
        if isinstance(v, float): print(f"  {k}: {v:.4f}")
        else: print(f"  {k}: {v}")

    print("\n=== Bootstrap 95% CI for indirect effect ===")
    b = bootstrap_indirect(T, M, Y, B=1000, seed=0)
    print(f"  indirect ab mean = {b['ab_mean']:.3f}, SE = {b['ab_se']:.3f}")
    print(f"  95% CI = ({b['ab_ci_95'][0]:.3f}, {b['ab_ci_95'][1]:.3f})   (true 0.4)")

    print("\n--- library cross-check (mediation R package) ---")
    print("  R: mediation::mediate(model.m, model.y, treat = 'T', mediator = 'M', sims = 1000)")
