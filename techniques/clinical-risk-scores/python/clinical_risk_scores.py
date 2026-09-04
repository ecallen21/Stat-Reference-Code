"""Clinical risk scores (Reference Sec 39.12).

Sullivan-Massaro-D'Agostino (2004) method: turn a fitted logistic
regression model into a bedside-usable INTEGER POINT SYSTEM
(Framingham, CHA2DS2-VASc, Wells, APACHE, SOFA, QRISK all follow
this template):

  1. Choose REFERENCE VALUE W_ref_j for each predictor.
  2. Compute contribution beta_j * (W_j - W_ref_j) for each unit /
     category.
  3. Divide by a CONSTANT B (units-per-point) chosen so points are
     small integers.  Sullivan uses B = beta_1 * (W_1_high - W_1_low)
     / desired-max-points-for-var-1.
  4. Round to integers.
  5. Sum categories -> total score -> table of predicted risks
     (fit a smoother from linear predictor -> score).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def build_points_system(beta, categories, ref_index, points_max=20):
    """Sullivan et al. 2004 integer-points scoring.

    beta        : coefficients (single value per level for categorical, per unit
                  for continuous already binned)
    categories  : list of dicts {"name": ..., "levels": [(label, value), ...]}
                  for each predictor
    ref_index   : index of reference level within each predictor
    points_max  : approximate maximum points assigned to the widest predictor
    """
    # Contributions relative to reference
    contribs = []
    for b, cats, ri in zip(beta, categories, ref_index):
        ref_val = cats["levels"][ri][1]
        vals = np.array([v for _, v in cats["levels"]])
        contribs.append(b * (vals - ref_val))
    # Units-per-point B = widest predictor's max abs contribution / points_max
    max_abs = max(np.abs(c).max() for c in contribs)
    B = max_abs / points_max
    # Integer points per level
    tables = []
    for cats, c in zip(categories, contribs):
        rows = []
        for (label, _), contrib in zip(cats["levels"], c):
            rows.append({"level": label, "beta_contrib": float(contrib),
                         "points": int(round(contrib / B))})
        tables.append({"name": cats["name"], "rows": rows})
    return {"tables": tables, "B": float(B)}


def score_to_risk(intercept, B, total_points):
    """Convert total integer points back to predicted probability."""
    lp = intercept + total_points * B
    return float(1 / (1 + np.exp(-lp)))


if __name__ == "__main__":
    print("=== Clinical risk score (Sullivan et al. 2004 integer points) ===\n")

    # Toy CHA2DS2-VASc-style model: 5 categorical predictors -> stroke risk
    beta = [0.63, 0.55, 1.10, 0.42, 0.75]
    intercept = -3.5
    categories = [
        {"name": "age_group",  "levels": [("<65", 0), ("65-74", 1), (">=75", 2)]},
        {"name": "sex",        "levels": [("male", 0), ("female", 1)]},
        {"name": "prior_stroke", "levels": [("no", 0), ("yes", 1)]},
        {"name": "diabetes",   "levels": [("no", 0), ("yes", 1)]},
        {"name": "chf",        "levels": [("no", 0), ("yes", 1)]},
    ]
    ref = [0, 0, 0, 0, 0]

    sys = build_points_system(beta, categories, ref_index=ref, points_max=6)
    print(f"  Sullivan units-per-point B = {sys['B']:.3f}")
    print(f"  Points per predictor:")
    for t in sys["tables"]:
        rows = "  ".join(f"{r['level']}={r['points']:+d}" for r in t["rows"])
        print(f"    {t['name']:<14s}  {rows}")

    # Score two patients
    patients = [
        {"age_group": "65-74", "sex": "female", "prior_stroke": "no",  "diabetes": "yes", "chf": "no"},
        {"age_group": ">=75",  "sex": "female", "prior_stroke": "yes", "diabetes": "yes", "chf": "yes"},
    ]
    print("\n  Patient scoring:")
    for i, pt in enumerate(patients):
        total = 0
        for t, choice_name in zip(sys["tables"], pt.keys()):
            pts = next(r["points"] for r in t["rows"] if r["level"] == pt[choice_name])
            total += pts
        risk = score_to_risk(intercept, sys["B"], total)
        print(f"    patient {i + 1}: {pt}   -> total = {total}   P(stroke) = {risk:.3f}")

    print("\n--- library cross-check (R rms::nomogram; Python custom) ---")
