"""Benefit-risk assessment via MCDA + NNT/NNH (Reference Sec 43.10).

Multi-Criteria Decision Analysis (MCDA):
  * Score each drug on multiple benefits and harms.
  * Normalise scores to [0, 1].
  * Weight criteria by clinical importance (elicited from experts).
  * WEIGHTED SUM = overall benefit-risk score.

Complementary:
  Number Needed to Treat  (NNT) = 1 / absolute risk reduction (benefit).
  Number Needed to Harm   (NNH) = 1 / absolute risk increase (harm).
  Likelihood of being helped vs harmed = NNH / NNT.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def normalise(x, higher_is_better=True):
    x = np.asarray(x, dtype=float)
    if higher_is_better:
        return (x - x.min()) / max(x.max() - x.min(), 1e-12)
    return (x.max() - x) / max(x.max() - x.min(), 1e-12)


def mcda(scores_matrix, higher_is_better, weights):
    """scores_matrix : (n_drugs, n_criteria);   higher_is_better : list of bool."""
    normed = np.stack([normalise(scores_matrix[:, j], higher_is_better[j])
                       for j in range(scores_matrix.shape[1])], axis=1)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return normed @ weights, normed


def nnt(rate_treat, rate_control):
    arr = rate_control - rate_treat
    return float(1 / arr) if arr > 0 else float("inf")


def nnh(rate_harm_treat, rate_harm_control):
    ari = rate_harm_treat - rate_harm_control
    return float(1 / ari) if ari > 0 else float("inf")


if __name__ == "__main__":
    print("=== Benefit-risk assessment: MCDA + NNT / NNH ===\n")

    drugs = ["A", "B", "C"]
    # Columns: efficacy score (0-100), major adverse event %, minor AE %, cost
    scores = np.array([
        [80, 5, 20, 400],
        [75, 3, 15, 700],
        [60, 2, 10, 200],
    ])
    higher_is_better = [True, False, False, False]  # efficacy is up; AE / cost down
    weights = [0.5, 0.3, 0.1, 0.1]

    overall, normed = mcda(scores, higher_is_better, weights)
    print(f"  {'drug':<6s}   {'efficacy':>9s} {'majorAE':>8s} {'minorAE':>8s} {'cost':>6s}"
          f"   {'MCDA_score':>10s}")
    for i, d in enumerate(drugs):
        print(f"  {d:<6s}   {scores[i, 0]:>9.0f} {scores[i, 1]:>8.1f} {scores[i, 2]:>8.1f}"
              f" {scores[i, 3]:>6.0f}   {overall[i]:>10.3f}")

    print(f"\n  Winner (max MCDA): drug {drugs[int(np.argmax(overall))]}\n")

    # NNT / NNH for drug A
    p_treat_benefit = 0.20; p_ctrl_benefit = 0.05
    p_treat_harm = 0.06; p_ctrl_harm = 0.01
    print(f"  Drug A vs placebo:")
    print(f"    NNT (benefit) = {nnt(1 - p_treat_benefit, 1 - p_ctrl_benefit):.1f}")
    print(f"    NNH (harm)    = {nnh(p_treat_harm, p_ctrl_harm):.1f}")
    print(f"    LHH = NNH / NNT = {nnh(p_treat_harm, p_ctrl_harm) / nnt(1 - p_treat_benefit, 1 - p_ctrl_benefit):.2f}\n")

    print("--- library cross-check (R drugCombo, MCDA (custom); Python custom + scipy.optimize) ---")
