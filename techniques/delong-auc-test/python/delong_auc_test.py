"""DeLong's test for comparing AUCs (Reference Sec 21.3).

DeLong-DeLong-Clarke-Pearson 1988.  Compare two AUCs computed on
the SAME cases using Mann-Whitney U-statistic structure theory
(Hanley-McNeil variance):

  Var(AUC_hat) = (V_10' V_10 - n1 * AUC^2) / (n1(n1-1))
                + (V_01' V_01 - n0 * AUC^2) / (n0(n0-1))

For paired AUCs, use the between-model covariance of the V's.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def _auc_and_placement(scores, y):
    """AUC via Mann-Whitney + per-observation placement values."""
    pos = scores[y == 1]; neg = scores[y == 0]
    n1 = len(pos); n0 = len(neg)
    # Placement values V10 (positive) and V01 (negative)
    V10 = np.array([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])
    V01 = np.array([np.mean(n < pos) + 0.5 * np.mean(n == pos) for n in neg])
    auc = float(V10.mean())
    return auc, V10, V01


def delong_test(scores_A, scores_B, y):
    aA, V10_A, V01_A = _auc_and_placement(scores_A, y)
    aB, V10_B, V01_B = _auc_and_placement(scores_B, y)
    n1 = len(V10_A); n0 = len(V01_A)
    S10 = np.cov(np.stack([V10_A, V10_B], axis=0))
    S01 = np.cov(np.stack([V01_A, V01_B], axis=0))
    var = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / n1 \
        + (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n0
    se = np.sqrt(max(var, 1e-12))
    z = (aA - aB) / se
    p = 2 * stats.norm.sf(abs(z))
    return {"AUC_A": aA, "AUC_B": aB, "diff": float(aA - aB),
            "SE": float(se), "z": float(z), "p_value": float(p)}


if __name__ == "__main__":
    print("=== DeLong's test: compare two paired AUCs ===\n")
    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 2, n)
    # Model A: strong signal; Model B: weaker signal
    scores_A = rng.normal(y * 1.0, 1, n)
    scores_B = rng.normal(y * 0.5, 1, n)
    r = delong_test(scores_A, scores_B, y)
    print(f"  AUC_A = {r['AUC_A']:.3f}   AUC_B = {r['AUC_B']:.3f}"
          f"   diff = {r['diff']:+.3f}")
    print(f"  DeLong SE = {r['SE']:.4f}   z = {r['z']:.2f}   p = {r['p_value']:.3e}\n")

    # Same-model sanity check: p should be ~1
    r2 = delong_test(scores_A, scores_A, y)
    print(f"  Same-model sanity check: p = {r2['p_value']:.3f}   (should be ~1)\n")
    print("--- library cross-check (R pROC::roc.test method='delong'; Python custom) ---")
