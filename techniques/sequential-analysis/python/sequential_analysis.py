"""Sequential probability ratio test (Reference Sec 37.6).

Wald (1945). At each observation, compute the LOG LIKELIHOOD RATIO

  Lambda_n = sum_{i=1}^n log( f_1(x_i) / f_0(x_i) )

and decide:
  Lambda_n >= B  -> ACCEPT H_1 (stop).
  Lambda_n <= A  -> ACCEPT H_0 (stop).
  A < Lambda_n < B  -> continue sampling.

Wald boundaries:
  A = log(beta / (1 - alpha)),  B = log((1 - beta) / alpha).

Expected sample size is dramatically lower than fixed-n tests for the
same (alpha, beta) errors.

Here we test H_0: p = 0.5  vs  H_1: p = 0.7 on Bernoulli data with
alpha = beta = 0.05 and compare average stopping times.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sprt_bernoulli(x, p0, p1, alpha=0.05, beta=0.05):
    """Sequential test H_0: p = p0 vs H_1: p = p1 on Bernoulli x_1..x_n."""
    A = np.log(beta / (1 - alpha))
    B = np.log((1 - beta) / alpha)
    Lambda = 0.0
    for i, xi in enumerate(x, start=1):
        if xi == 1:
            Lambda += np.log(p1 / p0)
        else:
            Lambda += np.log((1 - p1) / (1 - p0))
        if Lambda >= B:
            return {"stop": i, "decision": "reject H_0"}
        if Lambda <= A:
            return {"stop": i, "decision": "accept H_0"}
    return {"stop": len(x), "decision": "undecided"}


if __name__ == "__main__":
    print("=== Wald sequential probability ratio test (SPRT) ===\n")
    rng = np.random.default_rng(0)
    p0, p1 = 0.5, 0.7
    alpha = beta = 0.05

    print(f"  H_0: p = {p0} vs H_1: p = {p1}, alpha = beta = {alpha}\n")
    # Case A: true p = p_1 (should reject H_0)
    ns_A = []; decs_A = []
    for trial in range(500):
        x = (rng.random(1000) < p1).astype(int)
        r = sprt_bernoulli(x, p0, p1, alpha, beta)
        ns_A.append(r['stop']); decs_A.append(r['decision'])
    from collections import Counter
    print(f"  True p = {p1} (H_1 correct):")
    print(f"    mean stopping time = {np.mean(ns_A):.1f}   median = {np.median(ns_A):.0f}")
    print(f"    decisions: {dict(Counter(decs_A))}")

    # Case B: true p = p_0 (should accept H_0)
    ns_B = []; decs_B = []
    for trial in range(500):
        x = (rng.random(1000) < p0).astype(int)
        r = sprt_bernoulli(x, p0, p1, alpha, beta)
        ns_B.append(r['stop']); decs_B.append(r['decision'])
    print(f"  True p = {p0} (H_0 correct):")
    print(f"    mean stopping time = {np.mean(ns_B):.1f}   median = {np.median(ns_B):.0f}")
    print(f"    decisions: {dict(Counter(decs_B))}")

    # Fixed-n test to reach the same power would need
    from math import ceil
    from scipy.stats import norm
    z_a = norm.ppf(1 - alpha); z_b = norm.ppf(1 - beta)
    p_avg = (p0 + p1) / 2
    fixed_n = ceil((z_a * np.sqrt(p0 * (1 - p0)) + z_b * np.sqrt(p1 * (1 - p1))) ** 2 / (p1 - p0) ** 2)
    print(f"\n  Fixed-n test for same (alpha, beta) needs ~ {fixed_n} obs;\n"
          f"  SPRT averages {np.mean(ns_A + ns_B):.0f} obs.\n")
    print("--- library cross-check (R gsDesign; Python scipy.stats sequential helpers) ---")
