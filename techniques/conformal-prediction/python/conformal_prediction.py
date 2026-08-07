"""Conformal prediction intervals (Reference §10.19; Vovk-Gammerman-Shafer 2005).

Distribution-free prediction intervals with FINITE-SAMPLE coverage guarantee:
under just exchangeability of (X_i, y_i), the split-conformal interval
covers the true y_{new} with probability >= 1 - alpha.  Works with ANY
underlying model (LR, RF, xgboost, NN) as a black box.

Split-conformal (Papadopoulos et al. 2002)
    1. Split data into TRAIN and CALIBRATION sets.
    2. Fit a model on TRAIN -> hat_mu.
    3. Compute nonconformity scores s_i = |y_i - hat_mu(x_i)| on CALIBRATION.
    4. Let q = ceil((n + 1) (1 - alpha)) / n -quantile of {s_i}.
    5. Prediction interval:  hat_mu(x_new) +/- q.

Locally-adaptive variant (Lei et al. 2018): use s_i = |y_i - hat_mu(x_i)| / hat_sigma(x_i)
where hat_sigma is a variance model.  Intervals then have length that
varies with x.

CQR (Romano-Patterson-Candes 2019): use QUANTILE regressors and score
    s_i = max(hat_lo(x_i) - y_i,  y_i - hat_hi(x_i))
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def split_conformal_regression(X, y, X_test, alpha: float = 0.1,
                               fit_predict=None, calib_frac: float = 0.5,
                               seed: int = 0) -> dict:
    """Split-conformal prediction intervals for regression.

    fit_predict(X_train, y_train, X_pred) -> predictions   defaults to OLS.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    n = len(y); rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_cal = int(calib_frac * n)
    cal_idx, tr_idx = perm[:n_cal], perm[n_cal:]
    if fit_predict is None:
        def fit_predict(Xtr, ytr, Xp):
            beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
            return Xp @ beta
    y_hat_cal = fit_predict(X[tr_idx], y[tr_idx], X[cal_idx])
    scores = np.abs(y[cal_idx] - y_hat_cal)
    q_level = math.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q = float(np.quantile(scores, min(q_level, 1.0)))
    y_hat_test = fit_predict(X[tr_idx], y[tr_idx], X_test)
    return {"prediction": y_hat_test,
            "lower": y_hat_test - q, "upper": y_hat_test + q,
            "q": q, "alpha": float(alpha),
            "n_train": int(len(tr_idx)), "n_calibration": int(n_cal),
            "method": "Split-conformal prediction (Lei-Wasserman)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Split-conformal 90% intervals for linear regression ===")
    n = 500
    x = rng.normal(size=(n, 3))
    X = np.column_stack([np.ones(n), x])
    y = X @ np.array([1, 2, -1, 0.5]) + rng.normal(0, 1.0, n)

    X_test = X[:100]; y_test = y[:100]
    r = split_conformal_regression(X[100:], y[100:], X_test, alpha=0.1, seed=0)
    covered = ((y_test >= r["lower"]) & (y_test <= r["upper"])).mean()
    width = float(np.mean(r["upper"] - r["lower"]))
    print(f"  quantile q = {r['q']:.3f}")
    print(f"  empirical 90% coverage on held-out: {covered:.3f}  (target 0.90)")
    print(f"  average interval width: {width:.3f}")

    print("\n=== Split-conformal with heteroscedastic data ===")
    n = 500
    x = rng.uniform(-3, 3, n).reshape(-1, 1)
    X = np.column_stack([np.ones(n), x])
    y = 1 + 2 * x[:, 0] + (0.5 + np.abs(x[:, 0])) * rng.normal(size=n)
    X_test = X[:100]; y_test = y[:100]
    r = split_conformal_regression(X[100:], y[100:], X_test, alpha=0.1, seed=0)
    print(f"  empirical 90% coverage: {(np.abs(y_test - r['prediction']) <= r['q']).mean():.3f}")
    print(f"  interval width (constant): {r['upper'][0] - r['lower'][0]:.3f}")
    print("  (locally-adaptive conformal or CQR would vary width with |x|)")
