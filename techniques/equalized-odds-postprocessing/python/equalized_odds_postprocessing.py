"""Equalized-odds postprocessing (Reference Ch 31 Fairness).

Hardt, Price & Srebro (2016) 'Equality of Opportunity in Supervised
Learning'.  Given ANY score-producing classifier, choose a
GROUP-SPECIFIC RANDOMISED DECISION RULE that

  (1) matches TPR and FPR across groups (equalized odds), and
  (2) minimises deviation from the base classifier subject to (1).

Recipe (Hardt 2016):

  1. Compute per-group (TPR, FPR) as we sweep thresholds -> per-group
     ROC curves.
  2. The FEASIBLE set of (TPR, FPR) achievable by randomisations of the
     base classifier is the CONVEX HULL of its ROC curve.
  3. Feasible for EQUALIZED ODDS = intersection of the two group hulls.
  4. Pick the point that maximises OVERALL ACCURACY inside the
     intersection.

For each group we then find the mixture (t_a, p_a) of two thresholds +
a randomisation probability that lands on that target (TPR*, FPR*).

Here we implement a discretised version:
  - Sweep thresholds per group to enumerate (TPR, FPR) reachable pairs.
  - Add all 2-point convex combinations to approximate the hull.
  - Intersect the two groups' feasible sets.
  - Pick the (TPR*, FPR*) that maximises accuracy under Y's prevalence.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def per_group_roc_points(scores, y, groups, n_thr=50):
    """Return {group: list of (thr, TPR, FPR)}."""
    out = {}
    for a in np.unique(groups):
        m = groups == a
        s_a = scores[m]; y_a = y[m]
        thrs = np.quantile(s_a, np.linspace(0, 1, n_thr))
        rows = []
        for t in thrs:
            yh = (s_a >= t).astype(int)
            pos = y_a == 1
            tpr = float(yh[pos].mean()) if pos.any() else 0.0
            fpr = float(yh[~pos].mean()) if (~pos).any() else 0.0
            rows.append((float(t), tpr, fpr))
        out[int(a)] = rows
    return out


def convex_hull_points(pts):
    """Return the extreme points of the convex hull of (fpr, tpr) pairs
    plus interior midpoints obtained via 2-point convex combinations."""
    P = np.array(pts)
    # Add midpoints between all pairs to give the intersection more resolution.
    mid = []
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            for alpha in (0.25, 0.5, 0.75):
                mid.append(alpha * P[i] + (1 - alpha) * P[j])
    return np.vstack([P, np.array(mid)])


def feasible_intersection(roc0, roc1, tol=0.02):
    """Return list of (fpr, tpr) pairs achievable by BOTH groups approximately."""
    F0 = convex_hull_points([(r[2], r[1]) for r in roc0])
    F1 = convex_hull_points([(r[2], r[1]) for r in roc1])
    inter = []
    for p in F0:
        if np.min(np.max(np.abs(F1 - p), axis=1)) < tol:
            inter.append(p)
    return np.array(inter) if inter else np.zeros((0, 2))


if __name__ == "__main__":
    print("=== Equalized-odds postprocessing (Hardt 2016) ===\n")
    rng = np.random.default_rng(0)
    n_per = 400
    y0 = (rng.random(n_per) < 0.5).astype(int)
    y1 = (rng.random(n_per) < 0.3).astype(int)
    s0 = y0 + rng.normal(0, 0.5, n_per)
    s1 = y1 + rng.normal(0, 1.0, n_per)
    scores = np.concatenate([s0, s1])
    y = np.concatenate([y0, y1])
    groups = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(int)

    # Base classifier: threshold 0.5.
    y_hat_base = (scores >= 0.5).astype(int)
    tprs_base, fprs_base = {}, {}
    for a in (0, 1):
        m = groups == a
        tprs_base[a] = float(y_hat_base[m & (y == 1)].mean())
        fprs_base[a] = float(y_hat_base[m & (y == 0)].mean())
    print("  Base classifier per group (threshold 0.5):")
    for a in (0, 1):
        print(f"    A={a}   TPR={tprs_base[a]:.3f}   FPR={fprs_base[a]:.3f}")

    roc = per_group_roc_points(scores, y, groups)
    inter = feasible_intersection(roc[0], roc[1])
    print(f"\n  Intersection (feasible under equalized odds) has {len(inter)} candidate points.")

    if len(inter) > 0:
        # Pick the point that maximises overall accuracy.
        best, best_acc = None, -1
        for fpr_s, tpr_s in inter:
            # Overall accuracy under the shared (TPR, FPR)
            p_y1 = float(y.mean())
            acc = tpr_s * p_y1 + (1 - fpr_s) * (1 - p_y1)
            if acc > best_acc:
                best_acc = acc; best = (float(fpr_s), float(tpr_s))
        print(f"  Best EO-feasible point: FPR={best[0]:.3f}   TPR={best[1]:.3f}"
              f"   overall accuracy proxy={best_acc:.3f}")
    else:
        print("  No feasible intersection (increase n_thr or tol).")

    print("\n  Group-specific thresholds + randomisation move each group's classifier to (FPR*, TPR*).\n")
    print("--- library cross-check (aif360.algorithms.postprocessing.EqOddsPostprocessing;"
          " fairlearn.postprocessing.ThresholdOptimizer) ---")
