"""Discrimination vs calibration (Reference Sec 39.17).

Two DIFFERENT aspects of prediction performance:

  DISCRIMINATION -- ability to rank patients by risk.
    Metrics: C-statistic (AUC), c-index for time-to-event.
    A random re-scaling of probabilities preserves it.

  CALIBRATION   -- agreement between predicted probability and
                   observed event rate.
    Metrics: CITL, calibration slope, ICI, HL test, calibration plot.
    Different scalings of the same score give DIFFERENT calibration.

Van Calster et al. (2019): "Calibration is the Achilles heel of
predictive analytics."  Reporting AUC alone is inadequate; two models
with identical AUC can differ dramatically in clinical usefulness.
"""
from __future__ import annotations    # stdlib

import warnings                                          # suppress noise

import numpy as np    # numerical arrays
from sklearn.linear_model import LogisticRegression    # logistic fits

warnings.filterwarnings("ignore")


def auc(y, p):
    order = np.argsort(-p)
    y_o = y[order]
    pos = y_o == 1; neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return float(np.cumsum(pos)[neg].sum() / (pos.sum() * neg.sum()))


def brier(y, p):
    return float(((p - y) ** 2).mean())


def calibration_metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    m = LogisticRegression(C=1e12, solver="lbfgs", max_iter=500).fit(lp[:, None], y)
    # Integrated Calibration Index (ICI, Austin-Steyerberg 2019): mean |p_hat - loess(y|p_hat)|
    # Approximate loess by 10-bin means
    order = np.argsort(p)
    bins = np.array_split(order, 10)
    ici = 0.0; n = 0
    for b in bins:
        if len(b) == 0: continue
        ici += len(b) * abs(y[b].mean() - p[b].mean())
        n += len(b)
    ici /= max(n, 1)
    return {"CITL": float(y.mean() - p.mean()),
            "slope": float(m.coef_[0][0]),
            "intercept": float(m.intercept_[0]),
            "brier": brier(y, p),
            "ICI": float(ici)}


if __name__ == "__main__":
    print("=== Discrimination vs calibration: two models with same AUC ===\n")
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)

    # Well-calibrated model: correct logit
    p_good = 1 / (1 + np.exp(-x))

    # Same discrimination, WORSE calibration: shift + scale the logit
    p_bad = 1 / (1 + np.exp(-(0.7 * x - 0.6)))    # scale-shift, same ranking

    print(f"  Model A (well-calibrated):   AUC = {auc(y, p_good):.3f}")
    ca = calibration_metrics(y, p_good)
    print(f"    CITL = {ca['CITL']:+.3f}   slope = {ca['slope']:.3f}"
          f"   Brier = {ca['brier']:.3f}   ICI = {ca['ICI']:.3f}")

    print(f"\n  Model B (miscalibrated but same ranking): AUC = {auc(y, p_bad):.3f}")
    cb = calibration_metrics(y, p_bad)
    print(f"    CITL = {cb['CITL']:+.3f}   slope = {cb['slope']:.3f}"
          f"   Brier = {cb['brier']:.3f}   ICI = {cb['ICI']:.3f}")

    print("\n  Same AUC, very different clinical usefulness.  Report BOTH.\n")
    print("--- library cross-check (R rms::validate/val.prob, pROC; Python sklearn.metrics) ---")
