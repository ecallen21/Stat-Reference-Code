"""Disparate impact (Reference Ch 31 Fairness).

The DISPARATE IMPACT RATIO is the numerical form of the US EEOC
four-fifths rule (29 CFR Sec. 1607.4D, 'Uniform Guidelines on Employee
Selection Procedures', 1978):

  DI  =  P(Y_hat = 1 | A = minority) / P(Y_hat = 1 | A = majority).

Guidance:
  - DI >= 0.8 -> presumptively COMPLIANT.
  - DI  < 0.8 -> presumptive evidence of adverse impact.

Distinct from DEMOGRAPHIC PARITY: DI compares each minority group to
the MAX-SELECTION-RATE reference group. Feldman et al. (2015) also
report a MITIGATION that geometrically 'repairs' features so DI passes.

Statistical test (Fisher-exact / two-proportion z) can accompany DI to
report a 95% CI (Morris-Lobsenz 2001).

Here we compute DI + a two-proportion z-test 95% CI, and demonstrate a
simple mitigation via reweighing that raises DI above 0.8.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays

from scipy.stats import norm as _norm    # normal quantile


def selection_rate(y_hat, mask):
    return float(y_hat[mask].mean()) if mask.any() else float("nan")


def disparate_impact_ratio(y_hat, groups, reference=None):
    """Report DI = sel_rate(minority) / sel_rate(reference).
    'reference' defaults to the group with the LARGEST selection rate."""
    a_all = np.unique(groups)
    rates = {int(a): selection_rate(y_hat, groups == a) for a in a_all}
    if reference is None:
        reference = int(max(rates, key=rates.get))
    ref_rate = rates[reference]
    return {int(a): (r / ref_rate if ref_rate > 0 else float("nan"))
             for a, r in rates.items()}, reference, rates


def di_two_proportion_ci(y_hat, groups, group_a, group_b, alpha=0.05):
    """95% CI for sel_rate(a)/sel_rate(b) via delta method on log ratio."""
    ma = groups == group_a; mb = groups == group_b
    n_a, n_b = int(ma.sum()), int(mb.sum())
    x_a, x_b = int(y_hat[ma].sum()), int(y_hat[mb].sum())
    p_a, p_b = x_a / n_a, x_b / n_b
    ratio = p_a / p_b
    se_log = np.sqrt((1 - p_a) / max(x_a, 1) + (1 - p_b) / max(x_b, 1))
    z = _norm.ppf(1 - alpha / 2)
    lo = float(np.exp(np.log(ratio) - z * se_log))
    hi = float(np.exp(np.log(ratio) + z * se_log))
    return float(ratio), lo, hi


if __name__ == "__main__":
    print("=== Disparate impact / four-fifths rule ===\n")
    rng = np.random.default_rng(0)
    n_per = 400
    scores0 = rng.normal(0.6, 0.3, n_per)
    scores1 = rng.normal(0.4, 0.3, n_per)
    scores = np.clip(np.concatenate([scores0, scores1]), 0, 1)
    groups = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(int)

    y_hat = (scores >= 0.5).astype(int)
    di, ref, rates = disparate_impact_ratio(y_hat, groups)
    print(f"  selection rates: {rates}   reference group (max rate): {ref}")
    print(f"  disparate-impact ratios: {di}")
    r, lo, hi = di_two_proportion_ci(y_hat, groups, 1, 0)
    print(f"  DI(minority=1 / majority=0) = {r:.3f}   95% CI: [{lo:.3f}, {hi:.3f}]")
    print(f"  {'PASSES' if r >= 0.8 else 'FAILS'} the 4/5 rule.\n")

    # ---- Mitigation: reweigh minority so that its sel rate matches ----
    # Simple 'threshold shift' mitigation: shift the minority-group threshold down.
    def find_threshold(scores_g, target_rate):
        return float(np.quantile(scores_g, 1 - target_rate))

    target = rates[ref]                                # match majority selection rate
    t1 = find_threshold(scores[groups == 1], target)
    y_hat_mit = y_hat.copy()
    y_hat_mit[groups == 1] = (scores[groups == 1] >= t1).astype(int)
    di_mit, ref_mit, rates_mit = disparate_impact_ratio(y_hat_mit, groups)
    r_mit, lo_mit, hi_mit = di_two_proportion_ci(y_hat_mit, groups, 1, 0)
    print(f"  after group-1 threshold shift to {t1:.3f}:")
    print(f"    rates: {rates_mit}   DI ratio(minority/majority) = {r_mit:.3f}   CI [{lo_mit:.3f}, {hi_mit:.3f}]")
    print(f"    {'PASSES' if r_mit >= 0.8 else 'FAILS'} the 4/5 rule.\n")

    print("--- library cross-check (aif360.metrics.BinaryLabelDatasetMetric.disparate_impact;"
          " fairlearn.metrics.demographic_parity_ratio) ---")
