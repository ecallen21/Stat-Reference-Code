"""Acceptance sampling (Reference Sec 37.8 / 37.13).

SINGLE sampling plan (n, c): inspect n items; accept lot if defects
<= c, reject if > c.

DOUBLE sampling plan (n_1, c_1, n_2, c_2, r_2): may re-sample based
on the first result.

Key performance measures:
  * OC curve: P(accept | p) = sum_{d=0..c} Binom(d; n, p).
  * Producer's risk: P(reject | p = AQL).
  * Consumer's risk: P(accept | p = LTPD).
  * AOQ / AOQL (average outgoing quality) after 100% rectification of
    rejected lots.

Sequential SPRT sampling plan uses Wald (see sequential-analysis).

Here we compute an OC curve, ATI (average total inspection), and AOQL
for a single-sampling plan.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays

from scipy.stats import binom as _binom


def oc_curve(n, c, ps):
    """P(accept lot with fraction p defective) under single sampling plan."""
    return np.array([_binom.cdf(c, n, p) for p in ps])


def average_outgoing_quality(n, c, N, ps):
    """AOQ assuming rejected lots are 100% inspected and defects removed."""
    P_acc = oc_curve(n, c, ps)
    return P_acc * ps * (N - n) / N


def average_total_inspection(n, c, N, ps):
    """ATI: expected inspected items per lot after rectification."""
    P_acc = oc_curve(n, c, ps)
    return n + (1 - P_acc) * (N - n)


if __name__ == "__main__":
    print("=== Acceptance sampling ===\n")
    n, c, N = 50, 2, 1000
    ps = np.array([0.005, 0.01, 0.02, 0.04, 0.05, 0.08, 0.10])
    print(f"  Single-sampling plan (n = {n}, c = {c}, lot N = {N})\n")
    print(f"  {'p':>7}  {'P(accept)':>10}  {'AOQ':>9}  {'ATI':>7}")
    P_acc = oc_curve(n, c, ps)
    AOQ = average_outgoing_quality(n, c, N, ps)
    ATI = average_total_inspection(n, c, N, ps)
    for p, pa, aoq, ati in zip(ps, P_acc, AOQ, ATI):
        print(f"  {p:>7.3f}  {pa:>10.3f}  {aoq:>9.4f}  {ati:>7.1f}")

    print(f"\n  AOQL (max AOQ over p): {AOQ.max():.4f}   at p = {ps[AOQ.argmax()]:.3f}")
    print(f"  Producer's risk at AQL = 0.01: alpha = {1 - _binom.cdf(c, n, 0.01):.3f}")
    print(f"  Consumer's risk at LTPD = 0.08: beta = {_binom.cdf(c, n, 0.08):.3f}\n")
    print("--- library cross-check (R AcceptanceSampling; Python spc / custom) ---")
