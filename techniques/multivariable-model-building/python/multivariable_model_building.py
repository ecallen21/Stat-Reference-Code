"""Multivariable model building for prediction (Reference Sec 39.2).

Harrell (2015) argues that stepwise selection is HARMFUL for
prediction:
  * Inflated apparent performance (optimism ~10-20% AUC).
  * Unstable coefficients across bootstrap resamples.
  * P-values from the final model are wrong (data snooping).

Recommended (Harrell "full model" strategy):
  1. Choose predictors from CLINICAL knowledge + literature -- not
     data-driven univariable screening.
  2. Fit the FULL model with candidate predictors (respect the
     effective sample size constraint m / EPV >= ~15).
  3. Apply a GLOBAL SHRINKAGE FACTOR (or ridge / LASSO) to protect
     against optimism.
  4. Assess bootstrap-optimism-corrected performance.

Here we compare:
  A. Full model
  B. Backward-elimination stepwise (AIC)
  C. Full model with van Houwelingen-Le Cessie global shrinkage
"""
from __future__ import annotations    # stdlib

import warnings                                          # suppress sklearn deprecation
import numpy as np    # numerical arrays
from sklearn.linear_model import LogisticRegression    # baseline logistic
from sklearn.model_selection import KFold              # CV folds

warnings.filterwarnings("ignore")


def _lr(**kw):
    """Effectively-unpenalised logistic regression (big C)."""
    return LogisticRegression(C=1e12, solver="lbfgs", max_iter=500, **kw)


def full_model_fit(X, y):
    return _lr().fit(X, y)


def stepwise_backward_aic(X, y, feat_names):
    """Backward elimination by AIC on training data."""
    kept = list(range(X.shape[1]))
    def aic(idx):
        if not idx:
            m = _lr().fit(np.ones((len(y), 1)), y)
            return -2 * m.score(np.ones((len(y), 1)), y) * len(y) + 2
        m = _lr().fit(X[:, idx], y)
        p_hat = m.predict_proba(X[:, idx])[:, 1].clip(1e-6, 1 - 1e-6)
        loglik = (y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat)).sum()
        return -2 * loglik + 2 * (len(idx) + 1)
    cur = aic(kept)
    improved = True
    while improved and kept:
        improved = False
        best_drop = None; best_aic = cur
        for j in kept:
            trial = [i for i in kept if i != j]
            a = aic(trial)
            if a < best_aic:
                best_aic = a; best_drop = j
        if best_drop is not None:
            kept.remove(best_drop); cur = best_aic; improved = True
    m = _lr().fit(X[:, kept], y)
    return m, kept


def van_houwelingen_shrinkage(X, y):
    """Global shrinkage factor s = (model chi2 - df) / model chi2  (LR statistic)."""
    m_full = _lr().fit(X, y)
    p_full = m_full.predict_proba(X)[:, 1].clip(1e-6, 1 - 1e-6)
    ll_full = (y * np.log(p_full) + (1 - y) * np.log(1 - p_full)).sum()
    ybar = y.mean()
    ll_null = (y * np.log(ybar) + (1 - y) * np.log(1 - ybar)).sum()
    lr_chi2 = 2 * (ll_full - ll_null)
    df = X.shape[1]
    s = max(0.0, (lr_chi2 - df) / max(lr_chi2, 1e-9))
    return {"shrinkage": float(s), "beta_shrunk": s * m_full.coef_[0],
            "beta_full": m_full.coef_[0], "intercept": float(m_full.intercept_[0])}


def _auc(y_true, p):
    order = np.argsort(-p)
    y_o = y_true[order]
    pos = y_o == 1; neg = ~pos
    n_pos = pos.sum(); n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    cum_pos = np.cumsum(pos)
    return float((cum_pos[neg]).sum() / (n_pos * n_neg))


if __name__ == "__main__":
    print("=== Multivariable model building: full model vs stepwise vs shrinkage ===\n")
    rng = np.random.default_rng(0)
    n, p = 300, 8
    # Some real predictors (2, 3), noise (0, 1, 4, 5, 6, 7).  EPV ~= 12.5 for 8 preds.
    beta = np.array([0.0, 0.0, 1.2, -0.9, 0.0, 0.0, 0.0, 0.0])
    X = rng.normal(0, 1, (n, p))
    logit = X @ beta - 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    # 5-fold cross-validated AUC for each strategy
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    aucs = {"full": [], "stepwise": [], "shrinkage": []}
    for tr, te in kf.split(X):
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]
        # Full model
        m = full_model_fit(X_tr, y_tr)
        aucs["full"].append(_auc(y_te, m.predict_proba(X_te)[:, 1]))
        # Stepwise
        m, kept = stepwise_backward_aic(X_tr, y_tr, list(range(p)))
        aucs["stepwise"].append(_auc(y_te, m.predict_proba(X_te[:, kept])[:, 1]))
        # Shrunk full model
        sh = van_houwelingen_shrinkage(X_tr, y_tr)
        logit_te = X_te @ sh["beta_shrunk"] + sh["intercept"]
        aucs["shrinkage"].append(_auc(y_te, 1 / (1 + np.exp(-logit_te))))

    for k, v in aucs.items():
        print(f"  {k:>10s}   CV AUC = {np.mean(v):.3f}   (per fold: {[f'{x:.3f}' for x in v]})")

    sh = van_houwelingen_shrinkage(X, y)
    print(f"\n  van Houwelingen-Le Cessie global shrinkage factor: s = {sh['shrinkage']:.3f}")
    print(f"  betas full  = {np.round(sh['beta_full'], 3)}")
    print(f"  betas shrunk= {np.round(sh['beta_shrunk'], 3)}\n")

    print("--- library cross-check (R rms::pentrace/validate; Python sklearn + custom) ---")
