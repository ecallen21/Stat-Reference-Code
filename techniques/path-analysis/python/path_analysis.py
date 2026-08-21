"""Path analysis: structural regression with observed variables (Reference §19.4).

Path analysis is SEM restricted to OBSERVED variables (no latent factors).
Represents a system of regressions linked by a directed acyclic graph (DAG).

Example DAG:
    W -> M -> Y
    W -----> Y
    (M and Y have direct predictors; W is exogenous.)

Two connected regressions:
    M = a_W * W + eps_M
    Y = b_M * M + c_W * W + eps_Y

Total effect of W on Y = c_W + a_W * b_M   (direct + indirect)

Estimation
    Recursive DAGs: fit each equation by OLS (efficient); assemble covariance
    matrix of residuals; standard errors via bootstrap or delta method.
    Non-recursive systems need 2SLS / SEM ML.

Contrast with mediation-analysis: mediation-analysis is a common special
case of path analysis with a single mediator.  Full path analysis handles
arbitrary DAG structures.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def path_analysis(data: dict, equations: list) -> dict:
    """Fit each equation via OLS; return coefficient dict per equation.

    data      : {name: 1-D array}
    equations : list of (outcome, [predictors]) tuples in topological order.
    """
    fits = {}
    residuals = {}
    for lhs, rhs in equations:
        y = np.asarray(data[lhs], dtype=float)
        X = np.column_stack([np.ones(len(y))] + [np.asarray(data[v], dtype=float) for v in rhs])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        fits[lhs] = {"intercept": float(beta[0]),
                     **{v: float(beta[i + 1]) for i, v in enumerate(rhs)},
                     "R2": float(1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))}
        residuals[lhs] = y - y_hat
    return {"equations": fits, "residuals": residuals}


def total_effect(fits: dict, source: str, target: str) -> float:
    """Compute total effect of `source` on `target` by summing all path products."""
    # Traverse DAG: at each equation with `target`, get its direct predictors;
    # each predictor contributes coef * total_effect(source, predictor).
    if source == target: return 1.0
    if target not in fits: return 0.0
    coefs = {k: v for k, v in fits[target].items() if k not in ("intercept", "R2")}
    total = 0.0
    for pred, coef in coefs.items():
        if pred == source:
            total += coef
        else:
            total += coef * total_effect(fits, source, pred)
    return total


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    W = rng.normal(size=n)
    M = 0.6 * W + rng.normal(0, 1, n)                # a_W = 0.6
    Y = 0.4 * M + 0.3 * W + rng.normal(0, 1, n)       # b_M = 0.4, c_W = 0.3

    r = path_analysis(
        data={"W": W, "M": M, "Y": Y},
        equations=[("M", ["W"]), ("Y", ["M", "W"])]
    )
    print("=== Path analysis: W -> M -> Y, W -> Y ===")
    for lhs, coefs in r["equations"].items():
        print(f"  {lhs} eq: {coefs}")

    # Total effect W -> Y = c_W + a_W * b_M
    te = total_effect(r["equations"], source="W", target="Y")
    print(f"\n  Total effect W -> Y = {te:.4f}   (true 0.3 + 0.6 * 0.4 = 0.54)")

    # Indirect part
    a = r["equations"]["M"]["W"]; b = r["equations"]["Y"]["M"]; c_dir = r["equations"]["Y"]["W"]
    print(f"  Direct W -> Y      = {c_dir:.4f}   (true 0.30)")
    print(f"  Indirect W -> M -> Y = {a * b:.4f}   (true 0.24)")

    print("\n--- library cross-check (lavaan / semopy) ---")
    print("  R: lavaan::sem('M ~ W; Y ~ M + W', data = df)")
