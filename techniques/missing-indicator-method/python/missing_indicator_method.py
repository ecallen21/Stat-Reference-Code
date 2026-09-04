"""Missing indicator method (Reference Sec 41.12).

For a variable X with some missing values:
  1. IMPUTE X (mean, zero, MICE, etc.) -> X_imp.
  2. Create a BINARY INDICATOR: M_x = 1 if X is missing, 0 otherwise.
  3. Include BOTH X_imp and M_x in the regression.

Groenwold-White-Donders-Carpenter-Altman-Moons 2012 warn this is
biased for CAUSAL inference unless data are MCAR.  However, it is
useful for PREDICTION when missingness itself carries information
(a lab not ordered may reflect clinical judgement).

Prediction vs. inference is again the axis: for prediction, missing
indicators can genuinely help.
"""
from __future__ import annotations    # stdlib

import warnings

import numpy as np    # numerical arrays
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


def _mean_impute(X):
    X = np.asarray(X, dtype=float).copy()
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = np.nanmean(X[:, j])
    return X


def add_missing_indicators(X):
    """Return (X_imp, X_with_indicators)."""
    X = np.asarray(X, dtype=float)
    M = np.isnan(X).astype(float)
    X_imp = _mean_impute(X)
    X_out = np.hstack([X_imp, M])
    return X_imp, X_out


if __name__ == "__main__":
    print("=== Missing indicator method: prediction with vs without indicators ===\n")
    rng = np.random.default_rng(0)
    n, p = 800, 3
    X = rng.normal(0, 1, (n, p))
    y = X @ [1.5, -1.0, 0.5] + rng.normal(0, 1, n)

    # Make X[:, 0] MISSING NOT AT RANDOM: prob of missing depends on y
    miss_prob = 1 / (1 + np.exp(-(y - y.mean())))
    mask = rng.random(n) < miss_prob
    X_nan = X.copy(); X_nan[mask, 0] = np.nan

    X_imp, X_ind = add_missing_indicators(X_nan)

    # 5-fold CV RMSE
    def _rmse(fit_X):
        rmses = []
        for tr, te in KFold(n_splits=5, shuffle=True, random_state=0).split(fit_X):
            m = LinearRegression().fit(fit_X[tr], y[tr])
            yhat = m.predict(fit_X[te])
            rmses.append(float(np.sqrt(((yhat - y[te]) ** 2).mean())))
        return float(np.mean(rmses))

    print(f"  Fraction missing in x0: {mask.mean():.2%} (MNAR: prob depends on y)")
    print(f"  CV RMSE (mean-imputed only): {_rmse(X_imp):.3f}")
    print(f"  CV RMSE (imputed + indicators): {_rmse(X_ind):.3f}")
    print("\n  Under MNAR the indicators improve prediction; for INFERENCE they can bias\n"
          "  the target coefficient (Groenwold et al. 2012).\n")

    print("--- library cross-check (R mice, recipes::step_indicate_na; Python sklearn.impute.MissingIndicator) ---")
