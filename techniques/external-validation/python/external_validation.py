"""External validation of clinical prediction models (Reference Sec 39.5).

Fitting a model on one cohort (development) and evaluating it in an
INDEPENDENT cohort (validation) tests TRANSPORTABILITY:

  Temporal   -- different time period from same setting.
  Geographic -- different sites / regions.
  Domain     -- different case-mix (e.g., primary vs tertiary care).

Report both DISCRIMINATION (AUC) and CALIBRATION:

  Calibration-in-the-large (CITL)  = mean(y) - mean(p_hat)     [ideal 0]
  Calibration slope                 = slope of logit(y) on logit(p_hat)
                                       [ideal 1; < 1 = over-fit; > 1 = under-fit]

Recalibrate if calibration is off but discrimination survives.
"""
from __future__ import annotations    # stdlib

import warnings                                          # suppress noise

import numpy as np    # numerical arrays
from sklearn.linear_model import LogisticRegression    # baseline logistic

warnings.filterwarnings("ignore")


def _auc(y_true, p):
    order = np.argsort(-p)
    y_o = y_true[order]
    pos = y_o == 1; neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return float(np.cumsum(pos)[neg].sum() / (pos.sum() * neg.sum()))


def calibration_metrics(y, p):
    """CITL, calibration slope, Brier score."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    # Slope: logistic regression of y on lp
    m = LogisticRegression(C=1e12, solver="lbfgs", max_iter=500).fit(lp[:, None], y)
    return {"CITL": float(y.mean() - p.mean()),
            "slope": float(m.coef_[0][0]),
            "intercept": float(m.intercept_[0]),
            "brier": float(((p - y) ** 2).mean())}


if __name__ == "__main__":
    print("=== External validation: apparent vs external discrimination + calibration ===\n")
    rng = np.random.default_rng(0)
    n_dev, n_ext = 800, 500
    p = 4
    beta = np.array([0.8, -0.5, 0.6, 0.3])

    # Development cohort
    X_dev = rng.normal(0, 1, (n_dev, p))
    y_dev = (rng.random(n_dev) < 1 / (1 + np.exp(-(X_dev @ beta - 0.4)))).astype(int)

    # External cohort with case-mix shift + calibration drift
    X_ext = rng.normal(0.4, 1.2, (n_ext, p))                # different mean/SD
    y_ext = (rng.random(n_ext) < 1 / (1 + np.exp(-(X_ext @ (0.85 * beta) - 0.8)))).astype(int)

    m = LogisticRegression(C=1e12, solver="lbfgs", max_iter=500).fit(X_dev, y_dev)

    # Discrimination
    p_dev = m.predict_proba(X_dev)[:, 1]
    p_ext = m.predict_proba(X_ext)[:, 1]
    print(f"  Development cohort:  AUC = {_auc(y_dev, p_dev):.3f}, n = {n_dev}, events = {y_dev.sum()}")
    print(f"  External cohort   :  AUC = {_auc(y_ext, p_ext):.3f}, n = {n_ext}, events = {y_ext.sum()}\n")

    # Calibration
    cal_dev = calibration_metrics(y_dev, p_dev)
    cal_ext = calibration_metrics(y_ext, p_ext)
    print(f"  Dev calibration : CITL = {cal_dev['CITL']:+.3f}, slope = {cal_dev['slope']:.3f}"
          f", Brier = {cal_dev['brier']:.3f}")
    print(f"  Ext calibration : CITL = {cal_ext['CITL']:+.3f}, slope = {cal_ext['slope']:.3f}"
          f", Brier = {cal_ext['brier']:.3f}\n")

    verdict = "over-fit" if cal_ext['slope'] < 0.9 else ("under-fit" if cal_ext['slope'] > 1.1 else "well-calibrated")
    print(f"  External slope < 0.9 -> {verdict}; consider logistic recalibration.\n")
    print("--- library cross-check (R rms::val.prob; Python sklearn + custom) ---")
