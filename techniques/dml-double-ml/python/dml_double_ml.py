"""Double / Debiased Machine Learning (Reference Sec 15.25).

Chernozhukov et al. 2018.  Estimate a parameter of interest
theta while allowing arbitrary ML nuisance estimation with
Neyman-orthogonal moments + K-FOLD CROSS-FITTING to remove
regularisation bias.

Partially Linear Model (PLR):
    Y = D * theta + g(X) + eps
    D = m(X) + v

  1. Split into K folds.
  2. On each fold's OUT-OF-FOLD data, predict Y_hat = g(X) and
     D_hat = m(X) with any ML method.
  3. Residualise: y_tilde = Y - Y_hat,   d_tilde = D - D_hat.
  4. Estimate theta = mean(d_tilde * y_tilde) / mean(d_tilde^2).

Neyman-orthogonality means the moment condition's derivative wrt
the nuisance is zero at truth, so ML bias attenuates faster than
sqrt(n) and doesn't corrupt theta.
"""
from __future__ import annotations    # stdlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np    # numerical arrays
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold


def dml_plr(X, D, y, K=5, seed=0):
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    y_res = np.zeros_like(y, dtype=float)
    d_res = np.zeros_like(D, dtype=float)
    for tr, te in kf.split(X):
        g = GradientBoostingRegressor(random_state=0, n_estimators=100, max_depth=3).fit(X[tr], y[tr])
        m = GradientBoostingRegressor(random_state=0, n_estimators=100, max_depth=3).fit(X[tr], D[tr])
        y_res[te] = y[te] - g.predict(X[te])
        d_res[te] = D[te] - m.predict(X[te])
    theta = float((d_res * y_res).sum() / (d_res ** 2).sum())
    se = float(np.std((d_res * (y_res - theta * d_res)) / (d_res ** 2).mean()) / np.sqrt(len(y)))
    return {"theta": theta, "SE": se, "CI95": (theta - 1.96 * se, theta + 1.96 * se)}


if __name__ == "__main__":
    print("=== Double ML (partially linear model) ===\n")
    rng = np.random.default_rng(0)
    n, p = 1000, 5
    X = rng.normal(0, 1, (n, p))
    # Highly nonlinear nuisances
    g = 0.5 * np.sin(2 * X[:, 0]) + 0.3 * X[:, 1] ** 2 - 0.2 * X[:, 2]
    m = 0.4 * X[:, 0] * X[:, 1] + 0.3 * X[:, 2]
    D = m + rng.normal(0, 1, n)
    theta_true = 0.60
    y = theta_true * D + g + rng.normal(0, 1, n)

    r = dml_plr(X, D, y)
    print(f"  True theta = {theta_true}")
    print(f"  DML estimate = {r['theta']:+.3f}   SE = {r['SE']:.3f}"
          f"   95%CI = ({r['CI95'][0]:+.3f}, {r['CI95'][1]:+.3f})")

    # Naive OLS ignores confounding
    from numpy.linalg import lstsq
    beta, *_ = lstsq(np.column_stack([np.ones(n), D]), y, rcond=None)
    print(f"  Naive OLS of y on D: theta_hat = {beta[1]:+.3f}   (biased)\n")

    print("--- library cross-check (R DoubleML; Python DoubleML / econml.dml) ---")
