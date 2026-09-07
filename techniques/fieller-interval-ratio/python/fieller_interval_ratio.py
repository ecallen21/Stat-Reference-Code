"""Fieller confidence interval for a ratio (Reference Sec 3.27).

Fieller 1954 'Some problems in interval estimation', JRSS-B. Exact-
under-normality CI for the RATIO of two normal means / regression
slopes, robust to small denominators (unlike the delta-method
approximation).

For estimators theta1, theta2 with joint covariance V = ((v11, v12),
(v12, v22)) and correlation rho, the ratio r = theta1 / theta2 CI at
level (1 - alpha) is obtained by solving the quadratic:

    A r^2 - 2 B r + C = 0
    A = theta2^2 - z^2 * v22
    B = theta1 * theta2 - z^2 * v12
    C = theta1^2 - z^2 * v11

If A > 0, CI = (B - sqrt(B^2 - AC), B + sqrt(B^2 - AC)) / A.
If A <= 0, the CI is UNBOUNDED (denominator estimate too close to 0)
-- Fieller correctly reports this, delta-method would silently give
a narrow CI.

Applications: ratio of two means (e.g., cost-effectiveness ICER),
relative potency (bioassay), IV / 2SLS ratios, log-linear model
transformations.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def fieller_ci(t1, t2, v11, v22, v12, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    A = t2 ** 2 - z ** 2 * v22
    B = t1 * t2 - z ** 2 * v12
    C = t1 ** 2 - z ** 2 * v11
    if A > 0 and B ** 2 - A * C > 0:
        sq = np.sqrt(B ** 2 - A * C)
        return ((B - sq) / A, (B + sq) / A)
    else:
        return (-np.inf, np.inf)


def delta_method_ci(t1, t2, v11, v22, v12, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    r = t1 / t2
    var_r = (v11 - 2 * r * v12 + r ** 2 * v22) / t2 ** 2
    se = np.sqrt(var_r)
    return (r - z * se, r + z * se)


if __name__ == "__main__":
    print("=== Fieller confidence interval for a ratio ===\n")
    #  Two examples: strong denominator vs weak denominator (near 0)
    for label, t1, t2, v11, v22, v12 in [
        ("Strong denominator", 4.0, 2.0, 0.4, 0.1, 0.02),
        ("Weak   denominator", 4.0, 0.5, 0.4, 0.3, 0.02),
    ]:
        r_hat = t1 / t2
        fci = fieller_ci(t1, t2, v11, v22, v12)
        dci = delta_method_ci(t1, t2, v11, v22, v12)
        print(f"  {label}:  r_hat = {r_hat:.3f}")
        print(f"    Fieller CI       = ({fci[0]:.3f}, {fci[1]:.3f})")
        print(f"    Delta method CI  = ({dci[0]:.3f}, {dci[1]:.3f})")
        print()

    print("  Weak denominator: delta method looks fine but under-covers; Fieller")
    print("  becomes unbounded (correct diagnostic of non-identification).")

    print("\n--- library cross-check (ratio / mratios R; from-scratch scipy Python) ---")
