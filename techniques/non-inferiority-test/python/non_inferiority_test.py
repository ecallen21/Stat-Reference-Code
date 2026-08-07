"""Non-inferiority tests (Reference §17.7).

Distinct from superiority (H_0: delta = 0 vs H_a: delta != 0) and
equivalence (H_0: |delta| >= margin vs H_a: |delta| < margin):

    Non-inferiority (one-sided):
        H_0: delta <= -margin      (new is inferior by more than margin)
        H_a: delta > -margin       (new is at most margin worse)

Test: one-sided at alpha; equivalent to checking whether the LOWER bound
of the 95% CI for delta > -margin.

Two common effect measures
    - Mean difference (Normal outcomes; use t distribution).
    - Proportion difference (binary; use asymptotic Normal / Farrington-Manning).

The margin is a SUBSTANTIVE choice (regulatory guidance, minimum
clinically important difference).  Report both the effect and CI, not
just a p-value.

Contrast with equivalence (TOST)
    Equivalence uses TWO one-sided tests to bound delta within +/- margin.
    Non-inferiority uses ONE side only.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def non_inferiority_means(y_new, y_std, margin: float, alpha: float = 0.05,
                           lower_is_better: bool = False) -> dict:
    """Non-inferiority t-test on the mean difference (new - std).

    If lower_is_better = True, "new inferior" means new > std by more than margin,
    so H_a becomes delta < margin.
    """
    y_new = np.asarray(y_new, dtype=float); y_std = np.asarray(y_std, dtype=float)
    n1 = len(y_new); n2 = len(y_std)
    d_hat = float(y_new.mean() - y_std.mean())
    s2_pooled = ((n1 - 1) * y_new.var(ddof=1) + (n2 - 1) * y_std.var(ddof=1)) / (n1 + n2 - 2)
    se = math.sqrt(s2_pooled * (1 / n1 + 1 / n2))
    df = n1 + n2 - 2
    if lower_is_better:
        # Prefer smaller y; non-inferior if d_hat < margin (i.e. new not much bigger than std)
        t_stat = (d_hat - margin) / se
        p_ni = float(stats.t.cdf(t_stat, df=df))
        ci_upper = d_hat + stats.t.ppf(1 - alpha, df) * se
        result_ci = ("-inf", float(ci_upper))
        conclude = "non-inferior" if ci_upper < margin else "not shown"
    else:
        # Prefer larger y; non-inferior if d_hat > -margin (new not much smaller than std)
        t_stat = (d_hat - (-margin)) / se
        p_ni = float(1 - stats.t.cdf(t_stat, df=df))
        ci_lower = d_hat - stats.t.ppf(1 - alpha, df) * se
        result_ci = (float(ci_lower), "inf")
        conclude = "non-inferior" if ci_lower > -margin else "not shown"
    return {"mean_new": float(y_new.mean()), "mean_std": float(y_std.mean()),
            "diff": d_hat, "se": float(se), "df": int(df),
            "margin": float(margin),
            "one_sided_p_NI": p_ni,
            "one_sided_CI_bound": result_ci,
            "conclusion": conclude,
            "method": "Non-inferiority t-test on mean difference"}


def non_inferiority_props(x1: int, n1: int, x2: int, n2: int,
                          margin: float, alpha: float = 0.05) -> dict:
    """Non-inferiority test for two proportions (new - std >= -margin).

    Uses Farrington-Manning restricted MLE for the SE under the null.
    """
    p1 = x1 / n1; p2 = x2 / n2
    d_hat = p1 - p2
    # Farrington-Manning restricted MLE at null delta = -margin
    delta0 = -margin
    # Solve for constrained MLE p_1_tilde satisfying p_1 - p_2 = delta0
    a = n1 + n2
    b = -(n1 + n2 + x1 + x2 - delta0 * (n2 + 2 * n1))
    c = x1 + x2 - (2 * x1 + n1 + n2) * delta0 - (n1 + n2) * delta0
    d = x1 * delta0 * (1 - delta0)
    # Numerical inner solve; use bisection over p1_tilde in feasible range
    def score(p1t):
        p2t = p1t - delta0
        if p2t <= 0 or p2t >= 1: return None
        val = -((x1 - n1 * p1t) / (p1t * (1 - p1t)) + (x2 - n2 * p2t) / (p2t * (1 - p2t)))
        return val
    # bisection
    lo = max(1e-6, delta0 + 1e-6); hi = 1 - 1e-6
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        s = score(mid)
        if s is None: break
        if s < 0: lo = mid
        else: hi = mid
    p1_tilde = 0.5 * (lo + hi); p2_tilde = p1_tilde - delta0
    se_null = math.sqrt(p1_tilde * (1 - p1_tilde) / n1 + p2_tilde * (1 - p2_tilde) / n2)
    z_stat = (d_hat - delta0) / se_null
    p_ni = float(1 - stats.norm.cdf(z_stat))
    # Wald 95% one-sided CI for reporting
    se_wald = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci_lower = d_hat - stats.norm.ppf(1 - alpha) * se_wald
    return {"p1_new": float(p1), "p2_std": float(p2),
            "diff": float(d_hat), "z_stat_FM": float(z_stat),
            "one_sided_p_NI": p_ni, "margin": float(margin),
            "one_sided_CI_lower": float(ci_lower),
            "conclusion": "non-inferior" if ci_lower > -margin else "not shown",
            "method": "Non-inferiority proportions test (Farrington-Manning)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_new = rng.normal(9.8, 2.0, 60)
    y_std = rng.normal(10.0, 2.0, 60)
    print("=== Non-inferiority mean (margin = 1.0, higher is better) ===")
    r = non_inferiority_means(y_new, y_std, margin=1.0)
    print(f"  mean new = {r['mean_new']:.3f}, mean std = {r['mean_std']:.3f}, diff = {r['diff']:.3f}")
    print(f"  one-sided CI lower bound = {r['one_sided_CI_bound'][0]:.3f}  (margin threshold -1.0)")
    print(f"  p_NI = {r['one_sided_p_NI']:.4f}   conclusion: {r['conclusion']}")

    print("\n=== Non-inferiority proportions (new 82/100 vs std 80/100, margin = 0.10) ===")
    r = non_inferiority_props(82, 100, 80, 100, margin=0.10)
    print(f"  p1 = {r['p1_new']}, p2 = {r['p2_std']}, diff = {r['diff']}")
    print(f"  z = {r['z_stat_FM']:.3f}, p_NI = {r['one_sided_p_NI']:.4f}")
    print(f"  95% one-sided CI lower = {r['one_sided_CI_lower']:.3f}")
    print(f"  conclusion: {r['conclusion']}")
