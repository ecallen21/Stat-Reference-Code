"""Ratio metrics for A/B tests via the delta method (Reference Sec 44.10).

Many business metrics are ratios of two random variables (revenue
per user, CTR = clicks / impressions).  Naive t-test on the ratio
IS INVALID when the unit of randomisation differs from the ratio
denominator, or when numerator/denominator are correlated.

Deng-Knoblich-Lu 2018:
  Delta method: Var(ratio) ≈ (mu_y / mu_x)^2 * (Var(y)/mu_y^2
                            - 2 Cov(y, x)/(mu_y mu_x)
                            + Var(x)/mu_x^2)  divided by n.

Compute lift(t) - lift(c) with delta-method SE for a difference of
ratios across groups.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def ratio_stats(numer, denom):
    n = len(numer)
    mu_y = numer.mean(); mu_x = denom.mean()
    var_y = numer.var(ddof=1); var_x = denom.var(ddof=1)
    cov = np.cov(numer, denom, ddof=1)[0, 1]
    ratio = mu_y / mu_x
    var_ratio = (ratio ** 2 / n) * (var_y / mu_y ** 2
                                     - 2 * cov / (mu_y * mu_x)
                                     + var_x / mu_x ** 2)
    return {"ratio": float(ratio), "var_ratio": float(var_ratio)}


def diff_of_ratios(nC, dC, nT, dT):
    a = ratio_stats(nC, dC); b = ratio_stats(nT, dT)
    diff = b["ratio"] - a["ratio"]
    se = np.sqrt(a["var_ratio"] + b["var_ratio"])
    return {"ratio_C": a["ratio"], "ratio_T": b["ratio"],
            "abs_lift": float(diff), "SE": float(se),
            "CI95": (float(diff - 1.96 * se), float(diff + 1.96 * se))}


if __name__ == "__main__":
    print("=== Ratio metrics + delta method for A/B tests ===\n")
    rng = np.random.default_rng(0)
    # 5000 users; per-user (clicks, impressions).
    n = 5000
    imps_C = rng.poisson(20, n).astype(float)
    imps_T = rng.poisson(20, n).astype(float)
    p_click_C = 0.05
    p_click_T = 0.055
    clicks_C = rng.binomial(imps_C.astype(int), p_click_C).astype(float)
    clicks_T = rng.binomial(imps_T.astype(int), p_click_T).astype(float)

    r = diff_of_ratios(clicks_C, imps_C, clicks_T, imps_T)
    print(f"  Ratio_C (click/imp) = {r['ratio_C']:.4f}")
    print(f"  Ratio_T             = {r['ratio_T']:.4f}")
    print(f"  Abs lift            = {r['abs_lift']:+.4f}")
    print(f"  Delta-method SE     = {r['SE']:.4f}")
    print(f"  95% CI              = ({r['CI95'][0]:+.4f}, {r['CI95'][1]:+.4f})\n")

    # Naive per-user CTR (mean of per-user CTR) is different from pooled ratio
    per_user_C = clicks_C / np.maximum(imps_C, 1)
    per_user_T = clicks_T / np.maximum(imps_T, 1)
    print(f"  Naive per-user mean CTR C = {per_user_C.mean():.4f}   T = {per_user_T.mean():.4f}")
    print(f"    (per-user mean != pooled ratio -- different quantities)\n")
    print("--- library cross-check (R msm::deltamethod, sandwich, boot; Python custom + scipy) ---")
