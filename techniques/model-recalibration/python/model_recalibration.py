"""Model recalibration (Reference Sec 39.6).

When calibration drifts on an external cohort but discrimination
survives, RECALIBRATE:

  A. RECALIBRATION IN THE LARGE  (intercept-only update)
     new_lp = old_lp + a          -- fit a on external data

  B. LOGISTIC RECALIBRATION      (intercept + slope update)
     new_lp = a + b * old_lp      -- fit (a, b) on external data

  C. MODEL REVISION              (refit individual coefficients)

A is the cheapest fix; B is the standard workhorse; C loses the
original identity and needs its own external validation.
"""
from __future__ import annotations    # stdlib

import warnings                                          # suppress noise

import numpy as np    # numerical arrays
from sklearn.linear_model import LogisticRegression    # baseline logistic

warnings.filterwarnings("ignore")


def _lr(**kw):
    return LogisticRegression(C=1e12, solver="lbfgs", max_iter=500, **kw)


def recalibrate_intercept(lp, y):
    """Fit an offset alpha so that logit^-1(lp + alpha) matches P(y=1)."""
    m = _lr().fit(np.zeros((len(y), 1)), y)         # dummy X for shape
    # Equivalent: alpha = logit(y.mean()) - logit(mean(logit^-1(lp)))
    p_bar = 1 / (1 + np.exp(-lp))
    def _logit(q): return np.log(q / (1 - q))
    alpha = _logit(y.mean()) - _logit(p_bar.mean())
    return float(alpha)


def logistic_recalibration(lp, y):
    """Fit (alpha, beta) in y ~ Bernoulli(logit^-1(alpha + beta * lp))."""
    m = _lr().fit(lp[:, None], y)
    return {"intercept": float(m.intercept_[0]), "slope": float(m.coef_[0][0])}


def _auc(y_true, p):
    order = np.argsort(-p)
    y_o = y_true[order]
    pos = y_o == 1; neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return float(np.cumsum(pos)[neg].sum() / (pos.sum() * neg.sum()))


def _brier(y, p):
    return float(((p - y) ** 2).mean())


if __name__ == "__main__":
    print("=== Model recalibration: intercept + logistic (Steyerberg 2019 Ch 20) ===\n")
    rng = np.random.default_rng(0)
    # Development cohort: fit; External: known drift
    n_dev, n_ext, p = 800, 500, 4
    beta_dev = np.array([0.8, -0.5, 0.6, 0.3])
    X_dev = rng.normal(0, 1, (n_dev, p))
    y_dev = (rng.random(n_dev) < 1 / (1 + np.exp(-(X_dev @ beta_dev - 0.4)))).astype(int)
    m = _lr().fit(X_dev, y_dev)

    # External: coefficients 15 % smaller + higher baseline risk (shift intercept -0.4 -> -1.0)
    X_ext = rng.normal(0, 1, (n_ext, p))
    y_ext = (rng.random(n_ext) < 1 / (1 + np.exp(-(X_ext @ (0.85 * beta_dev) - 1.0)))).astype(int)
    p_ext_raw = m.predict_proba(X_ext)[:, 1]
    lp_ext = np.log(p_ext_raw / (1 - p_ext_raw))

    print(f"  External raw     : AUC = {_auc(y_ext, p_ext_raw):.3f}"
          f"   CITL = {y_ext.mean() - p_ext_raw.mean():+.3f}"
          f"   Brier = {_brier(y_ext, p_ext_raw):.3f}")

    # A. Intercept-only recalibration
    alpha = recalibrate_intercept(lp_ext, y_ext)
    p_int = 1 / (1 + np.exp(-(lp_ext + alpha)))
    print(f"  A) intercept fix : alpha = {alpha:+.3f}"
          f"   -> CITL = {y_ext.mean() - p_int.mean():+.3f}"
          f"   Brier = {_brier(y_ext, p_int):.3f}")

    # B. Logistic recalibration
    lr = logistic_recalibration(lp_ext, y_ext)
    lp_new = lr["intercept"] + lr["slope"] * lp_ext
    p_log = 1 / (1 + np.exp(-lp_new))
    print(f"  B) log recalib.  : intercept = {lr['intercept']:+.3f}"
          f"   slope = {lr['slope']:.3f}"
          f"   -> CITL = {y_ext.mean() - p_log.mean():+.3f}"
          f"   Brier = {_brier(y_ext, p_log):.3f}\n")

    print("  Discrimination (AUC) is unchanged by A or B -- they are monotone.")
    print("  Prefer the least invasive fix that fixes the drift.\n")
    print("--- library cross-check (R rms::val.prob, predtools::recalibrate; Python custom) ---")
