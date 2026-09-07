"""Youden's J / ROC-optimal cutpoint (Reference Sec 26.10).

Youden 1950 'Index for rating diagnostic tests', Cancer. Choose the
threshold t that maximises

    J(t) = Sensitivity(t) + Specificity(t) - 1
         = TPR(t) - FPR(t)

Geometrically, the ROC point maximally distant from the chance
diagonal. Alternative criteria include:

    ROC01     : minimise (1 - Sensitivity)^2 + (1 - Specificity)^2
    IU / MCC  : maximise informedness or MCC
    Cost-weighted: minimise c_FN * (1 - Sen) * prev + c_FP * (1 - Spe) * (1 - prev)
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def tpr_fpr(y, prob, t):
    yp = (prob >= t).astype(int)
    tp = int(((yp == 1) & (y == 1)).sum())
    fn = int(((yp == 0) & (y == 1)).sum())
    fp = int(((yp == 1) & (y == 0)).sum())
    tn = int(((yp == 0) & (y == 0)).sum())
    sen = tp / (tp + fn) if tp + fn > 0 else 0
    spe = tn / (tn + fp) if tn + fp > 0 else 0
    return sen, spe


def youden_optimum(y, prob, n_grid=201):
    grid = np.linspace(0.001, 0.999, n_grid)
    js = np.zeros(len(grid))
    sens = np.zeros(len(grid))
    spes = np.zeros(len(grid))
    for k, t in enumerate(grid):
        s, sp = tpr_fpr(y, prob, t)
        sens[k] = s; spes[k] = sp
        js[k] = s + sp - 1
    idx = int(np.argmax(js))
    return {"t_opt": float(grid[idx]), "J_opt": float(js[idx]),
            "Sens": float(sens[idx]), "Spec": float(spes[idx])}


def roc01_optimum(y, prob, n_grid=201):
    grid = np.linspace(0.001, 0.999, n_grid)
    d2 = np.zeros(len(grid))
    for k, t in enumerate(grid):
        s, sp = tpr_fpr(y, prob, t)
        d2[k] = (1 - s) ** 2 + (1 - sp) ** 2
    idx = int(np.argmin(d2))
    s, sp = tpr_fpr(y, prob, grid[idx])
    return {"t_opt": float(grid[idx]), "d2_opt": float(d2[idx]),
            "Sens": float(s), "Spec": float(sp)}


if __name__ == "__main__":
    print("=== Youden J and ROC01 optimal cutpoints ===\n")
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.binomial(1, 0.25, size=n)
    score = rng.normal(loc=1.5 * y, scale=1.0)
    prob = 1 / (1 + np.exp(-score))

    j = youden_optimum(y, prob)
    r = roc01_optimum(y, prob)
    print(f"  Youden J:  t* = {j['t_opt']:.3f}   J = {j['J_opt']:.3f}   "
          f"Sens = {j['Sens']:.3f}  Spec = {j['Spec']:.3f}")
    print(f"  ROC01   :  t* = {r['t_opt']:.3f}   d2 = {r['d2_opt']:.3f}   "
          f"Sens = {r['Sens']:.3f}  Spec = {r['Spec']:.3f}")

    #  Compare naive t = 0.5
    s5, sp5 = tpr_fpr(y, prob, 0.5)
    print(f"\n  Naive t = 0.5   Sens = {s5:.3f}  Spec = {sp5:.3f}  "
          f"J = {s5 + sp5 - 1:.3f}")

    print("\n--- library cross-check (OptimalCutpoints / cutpointr R; sklearn Python) ---")
