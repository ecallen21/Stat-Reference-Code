"""Egger Test for Publication Bias (Ref Sec 47.271).

Egger, Davey Smith, Schneider & Minder 1997 BMJ. Regression-
based funnel-plot asymmetry test:

    y_k / SE_k = a + b * (1 / SE_k)      (weighted least squares)

A non-zero intercept 'a' indicates SMALL-STUDY EFFECTS
(publication bias, methodological quality gradient, or true
heterogeneity by size). Also computes the more robust Peters
test for OR meta-analyses.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def egger_test(y, v):
    """Egger regression: t_stat, p-value for intercept."""
    from scipy.stats import t as tdist
    y = np.asarray(y); v = np.asarray(v); se = np.sqrt(v)
    # Precision = 1 / SE, standardized effect = y / SE
    x = 1.0 / se
    z = y / se
    K = len(y)
    # WLS reduces to plain OLS on (z ~ 1 + x) because weights are already absorbed
    X = np.column_stack([np.ones(K), x])
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    resid = z - X @ beta
    sigma2 = (resid ** 2).sum() / (K - 2)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se_int = np.sqrt(cov[0, 0])
    t_stat = beta[0] / se_int
    p = 2 * (1 - tdist.cdf(np.abs(t_stat), df=K - 2))
    return dict(intercept=float(beta[0]), slope=float(beta[1]), t=float(t_stat), p=float(p))


if __name__ == "__main__":
    print("=== Egger Test for Publication Bias (Egger et al 1997 BMJ) ===\n")
    rng = np.random.default_rng(0)

    # Scenario A: no bias, homogeneous studies
    K = 20
    v_a = 1.0 / rng.integers(30, 500, K); y_a = rng.normal(0.4, np.sqrt(v_a), K)
    res_a = egger_test(y_a, v_a)
    print(f"  A) No publication bias (K={K}):")
    print(f"     intercept a = {res_a['intercept']:.3f}   t = {res_a['t']:.2f}   p = {res_a['p']:.3f}")
    print(f"     -> {'reject H0 (asymmetry)' if res_a['p'] < 0.05 else 'no evidence of asymmetry'}\n")

    # Scenario B: small studies WITH LARGER effects (classic publication bias)
    # Small studies (large v) get inflated effects because null small studies are missing
    v_b = 1.0 / rng.integers(20, 500, K)
    inflation = 2.0 * np.sqrt(v_b)                                # small studies boosted
    y_b = rng.normal(0.4 + inflation, np.sqrt(v_b))
    res_b = egger_test(y_b, v_b)
    print(f"  B) Small-study effects (K={K}):")
    print(f"     intercept a = {res_b['intercept']:.3f}   t = {res_b['t']:.2f}   p = {res_b['p']:.4f}")
    print(f"     -> {'reject H0 (asymmetry)' if res_b['p'] < 0.05 else 'no evidence'}\n")

    # Scenario C: trim-and-fill would counter this in practice
    print(f"  Egger intercept ~ 0 under symmetric funnel; positive under small-study inflation.")
    print(f"  Slope estimates the true underlying effect (bias-corrected in a linear-bias model).\n")

    # Funnel visualization: report positions
    print(f"  Funnel-plot snapshot (scenario B): (y_k, SE_k) top / bottom 3 by SE")
    order = np.argsort(np.sqrt(v_b))
    for k in np.concatenate([order[:3], order[-3:]]):
        print(f"    study {k+1:>2}:  y = {y_b[k]:>+.3f}   SE = {np.sqrt(v_b[k]):.3f}")

    print("\n--- library cross-check (metafor::regtest R; meta::metabias R; PythonMeta) ---")
