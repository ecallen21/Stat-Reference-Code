"""Nomograms (Reference Sec 39.3).

Iasonos-Schrag-Raj-Panageas (2008); Harrell (2015 ch 14).  A nomogram
turns a fitted regression model into a POINT-BASED graphic:

  * Each predictor -> a points scale.  Reference level = 0 points;
    each unit gets points proportional to its beta.
  * SUM total points.
  * Map total points -> linear predictor -> probability via the
    inverse link.

Rounded so a clinician can compute it at the bedside with paper.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def build_nomogram(beta, feature_ranges, feature_names, max_points=100):
    """Return per-feature point conversion + total-to-linear-predictor mapping.

    beta            : coefficient vector (excluding intercept)
    feature_ranges  : list of (lo, hi) covered by each predictor
    feature_names   : names for display
    """
    beta = np.asarray(beta, dtype=float)
    # Contribution range of each predictor over its (lo, hi)
    contribs = [abs(b) * (hi - lo) for b, (lo, hi) in zip(beta, feature_ranges)]
    max_contrib = max(contribs)
    # Points per unit of predictor
    points_per_unit = [b * (max_points / max_contrib) for b in beta]
    scales = []
    for name, (lo, hi), ppu in zip(feature_names, feature_ranges, points_per_unit):
        marks = np.linspace(lo, hi, 5)
        pts = [(m - lo) * ppu for m in marks]           # 0 at lo
        scales.append({"name": name, "marks": list(marks), "points": pts})
    # Points-to-linear-predictor conversion factor
    total_points_to_lp = max_contrib / max_points       # per point
    return {"scales": scales, "total_to_lp": float(total_points_to_lp)}


def score_patient(patient, feature_ranges, scales, total_to_lp, intercept):
    """Compute total points and predicted probability for one patient."""
    total = 0.0
    breakdown = []
    for x, (lo, hi), s in zip(patient, feature_ranges, scales):
        ppu = (s["points"][-1] - s["points"][0]) / (hi - lo) if hi != lo else 0.0
        pts = (x - lo) * ppu
        total += pts
        breakdown.append({"name": s["name"], "value": float(x), "points": float(pts)})
    lp = intercept + total * total_to_lp
    p = 1 / (1 + np.exp(-lp))
    return {"total_points": float(total), "linear_predictor": float(lp),
            "prob": float(p), "breakdown": breakdown}


if __name__ == "__main__":
    print("=== Nomogram: turn logistic coefficients into a points-based scoring sheet ===\n")
    rng = np.random.default_rng(0)
    # Toy: 30-day mortality after cardiac surgery
    names = ["age", "creatinine", "ef", "diabetes"]
    beta = np.array([0.06, 0.4, -0.05, 0.6])
    intercept = -6.0
    ranges = [(40, 90), (0.5, 3.0), (20, 70), (0, 1)]

    nomo = build_nomogram(beta, ranges, names, max_points=100)
    print("  Per-predictor point scales:")
    for s in nomo["scales"]:
        print(f"    {s['name']:<12s}   marks: " +
              "  ".join(f"{m:6.2f}={p:6.1f}pt" for m, p in zip(s["marks"], s["points"])))

    print(f"\n  Total-points -> linear-predictor factor: {nomo['total_to_lp']:.4f}\n")

    # Score two patients
    patients = [(70, 1.2, 55, 0), (85, 2.4, 30, 1)]
    for i, pt in enumerate(patients):
        r = score_patient(pt, ranges, nomo["scales"], nomo["total_to_lp"], intercept)
        print(f"  Patient {i + 1}: {dict(zip(names, pt))}")
        for b in r["breakdown"]:
            print(f"    {b['name']:<12s}  x = {b['value']:>5.2f}   points = {b['points']:>6.1f}")
        print(f"    Total points = {r['total_points']:.1f}"
              f"   LP = {r['linear_predictor']:.3f}"
              f"   Predicted P(30-day mortality) = {r['prob']:.3f}\n")

    print("--- library cross-check (R rms::nomogram; Python custom + matplotlib) ---")
