"""Inverse Probability of Treatment Weighting (Reference Sec 15.6).

Rosenbaum-Rubin 1983.  Given a propensity score e(X) = P(T=1|X),
reweight observations to mimic a randomised trial:

  ATE weight  : w = T/e(X) + (1-T)/(1-e(X))
  ATT weight  : w = T + (1-T) * e(X)/(1-e(X))
  ATC weight  : w = T * (1-e(X))/e(X) + (1-T)
  ATO weight  : w = T*(1-e(X)) + (1-T)*e(X)  (overlap weights, Li-Morgan-Zaslavsky)

Estimand:
  ATE_hat = mean(w * T * Y) / mean(w * T) - mean(w * (1-T) * Y) / mean(w * (1-T))

Careful: TRIMMING/STABILISATION for extreme weights (positivity).
"""
from __future__ import annotations    # stdlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np    # numerical arrays
from sklearn.linear_model import LogisticRegression


def iptw_weights(ps, T, estimand="ATE", trim=(0.01, 0.99)):
    ps = np.clip(ps, trim[0], trim[1])
    if estimand == "ATE":
        return np.where(T == 1, 1 / ps, 1 / (1 - ps))
    if estimand == "ATT":
        return np.where(T == 1, 1.0, ps / (1 - ps))
    if estimand == "ATC":
        return np.where(T == 1, (1 - ps) / ps, 1.0)
    if estimand == "ATO":
        return np.where(T == 1, 1 - ps, ps)
    raise ValueError(estimand)


def iptw_estimate(y, T, ps, estimand="ATE"):
    w = iptw_weights(ps, T, estimand)
    num_T = (w * T * y).sum(); den_T = (w * T).sum()
    num_C = (w * (1 - T) * y).sum(); den_C = (w * (1 - T)).sum()
    return float(num_T / den_T - num_C / den_C)


if __name__ == "__main__":
    print("=== IPTW: ATE / ATT / ATC / ATO estimators ===\n")
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(0, 1, (n, 3))
    logit_T = 0.5 * X[:, 0] + 0.4 * X[:, 1] - 0.3 * X[:, 2]
    T = (rng.random(n) < 1 / (1 + np.exp(-logit_T))).astype(int)
    # True ATE = 0.5
    y = 1.0 + 0.5 * T + 0.6 * X[:, 0] + 0.3 * X[:, 1] + rng.normal(0, 1, n)

    # Fit PS
    ps = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500).fit(X, T).predict_proba(X)[:, 1]

    naive = y[T == 1].mean() - y[T == 0].mean()
    print(f"  Naive difference   = {naive:+.3f}   (biased)")
    for est in ("ATE", "ATT", "ATC", "ATO"):
        print(f"  {est:>4s}  IPTW estimate = {iptw_estimate(y, T, ps, est):+.3f}")
    print(f"\n  True ATE = 0.5.  Overlap weighting (ATO) is most stable under near-violated positivity.\n")

    print("--- library cross-check (R WeightIt/ipw/cobalt; Python causalinference/zepid/econml) ---")
