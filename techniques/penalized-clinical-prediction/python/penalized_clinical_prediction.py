"""Penalized regression for clinical prediction (Reference Sec 39.9).

Steyerberg (2019 Ch 12); van Houwelingen-Le Cessie (1990).  For a
low-EPV prediction problem, unpenalised MLE overfits.  Options:

  RIDGE       -- L2 penalty; shrinks all betas toward 0 uniformly.
                 Choice of lambda by CV or effective-df target.
  LASSO       -- L1 penalty; shrinks + selects (some betas exactly 0).
  ELASTIC NET -- alpha * L1 + (1 - alpha) * L2; useful when
                 correlated predictors.

  van Houwelingen-Le Cessie GLOBAL SHRINKAGE (see
  multivariable-model-building) -- one-shot analytic shrinkage
  factor s applied to all coefficients.
"""
from __future__ import annotations    # stdlib

import warnings

import numpy as np    # numerical arrays
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")


def _auc(y, p):
    order = np.argsort(-p)
    y_o = y[order]
    pos = y_o == 1; neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return float(np.cumsum(pos)[neg].sum() / (pos.sum() * neg.sum()))


def unpenalised_logistic(X, y):
    return LogisticRegression(C=1e12, solver="lbfgs", max_iter=500).fit(X, y)


def ridge_cv(X, y, Cs=np.logspace(-3, 3, 30)):
    return LogisticRegressionCV(Cs=Cs, penalty="l2", cv=10, solver="lbfgs", max_iter=500).fit(X, y)


def lasso_cv(X, y, Cs=np.logspace(-3, 3, 30)):
    return LogisticRegressionCV(Cs=Cs, penalty="l1", cv=10, solver="saga", max_iter=2000).fit(X, y)


if __name__ == "__main__":
    print("=== Penalised regression for clinical prediction (low EPV) ===\n")
    rng = np.random.default_rng(0)
    n, p = 200, 15
    # 4 real predictors + 11 noise, sparse-ish truth
    beta = np.concatenate([rng.normal(0, 0.8, 4), np.zeros(p - 4)])
    X = rng.normal(0, 1, (n, p))
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ beta - 0.3)))).astype(int)
    print(f"  n = {n}, p = {p}, EPV = {y.sum() / p:.1f}\n")

    # CV AUC for each method
    def cv_auc(model_fn):
        aucs = []
        from sklearn.model_selection import KFold
        for tr, te in KFold(n_splits=10, shuffle=True, random_state=0).split(X):
            m = model_fn(X[tr], y[tr])
            aucs.append(_auc(y[te], m.predict_proba(X[te])[:, 1]))
        return float(np.mean(aucs))

    for name, fn in [("unpenalised", unpenalised_logistic),
                     ("ridge-CV",    ridge_cv),
                     ("lasso-CV",    lasso_cv)]:
        auc = cv_auc(fn)
        m = fn(X, y)
        coefs = m.coef_[0]
        n_nonzero = int((np.abs(coefs) > 1e-6).sum()) if name == "lasso-CV" else "-"
        print(f"  {name:>12s}   CV AUC = {auc:.3f}   |beta|_max = {np.max(np.abs(coefs)):.3f}"
              f"   nonzero (LASSO) = {n_nonzero}")

    print("\n--- library cross-check (R glmnet::cv.glmnet, rms::pentrace; Python sklearn) ---")
