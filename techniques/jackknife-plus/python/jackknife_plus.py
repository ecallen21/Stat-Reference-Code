"""Jackknife+ (Reference Ch 29 Uncertainty Quantification).

Barber, Candes, Ramdas & Tibshirani (2021) "Predictive inference with the
jackknife+."

Unlike split-conformal (data-splitting cost) and full conformal (retrain per
test point), jackknife+ uses n LOO refits of the base model. For each
training point i, fit mu_hat_{-i} and score the residual:

  R_i = |y_i - mu_hat_{-i}(x_i)|

Prediction interval for x_new at level 1 - alpha:

  [ Q_alpha( { mu_hat_{-i}(x_new) - R_i } ),
    Q_{1-alpha}( { mu_hat_{-i}(x_new) + R_i } ) ]

Coverage guarantee: P(y_new in interval) >= 1 - 2 alpha (worst-case);
in practice close to 1 - alpha. Distribution-free, no split needed.

CV+ generalises to K-fold in place of LOO.

Here we implement jackknife+ for linear regression (fast LOO via the
Sherman-Morrison update) on a synthetic problem, and empirically verify
coverage across many trials.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def loo_linear_predict(X, y, x_new):
    """Return (mu_hat_{-i}(x_i), mu_hat_{-i}(x_new)) for each i."""
    n = X.shape[0]
    XtX = X.T @ X
    Xty = X.T @ y
    A_inv = np.linalg.inv(XtX)
    # ridge-less; assume XtX invertible.
    r_i_train = np.zeros(n)
    r_i_new = np.zeros((n, x_new.shape[0]))
    for i in range(n):
        # Sherman-Morrison: (A - x_i x_i^T)^{-1} = A^-1 + (A^-1 x_i x_i^T A^-1) / (1 - x_i^T A^-1 x_i)
        xi = X[i]
        Ainv_xi = A_inv @ xi
        denom = 1.0 - xi @ Ainv_xi
        A_inv_loo = A_inv + np.outer(Ainv_xi, Ainv_xi) / denom
        Xty_loo = Xty - xi * y[i]
        beta_loo = A_inv_loo @ Xty_loo
        r_i_train[i] = xi @ beta_loo
        r_i_new[i] = x_new @ beta_loo
    return r_i_train, r_i_new


def jackknife_plus_interval(X, y, x_new, alpha=0.1):
    n = X.shape[0]
    yhat_train, yhat_new = loo_linear_predict(X, y, x_new)
    R = np.abs(y - yhat_train)                                # (n,)
    # For each x_new col, form n candidate lower/upper endpoints:
    lows = yhat_new - R[:, None]
    highs = yhat_new + R[:, None]
    lo_q = np.floor(alpha * (n + 1)) / n
    hi_q = np.ceil((1 - alpha) * (n + 1)) / n
    lo_q = max(min(lo_q, 1.0), 0.0)
    hi_q = max(min(hi_q, 1.0), 0.0)
    lower = np.quantile(lows,  lo_q, axis=0, method="lower")
    upper = np.quantile(highs, hi_q, axis=0, method="higher")
    return lower, upper


if __name__ == "__main__":
    print("=== Jackknife+ prediction intervals (Barber 2021) ===\n")
    rng = np.random.default_rng(0)
    n, d = 40, 3

    n_trials = 200
    alpha = 0.1
    covered = 0
    widths = []
    for trial in range(n_trials):
        # y = X beta + eps
        X_all = rng.normal(0, 1, (n + 50, d))
        beta_true = rng.normal(0, 1, d)
        eps = rng.normal(0, 0.5, n + 50)
        y_all = X_all @ beta_true + eps
        X, y = X_all[:n], y_all[:n]
        X_te, y_te = X_all[n:], y_all[n:]
        lo, hi = jackknife_plus_interval(X, y, X_te, alpha=alpha)
        cov_trial = ((y_te >= lo) & (y_te <= hi)).mean()
        covered += cov_trial
        widths.append((hi - lo).mean())

    empirical_cov = covered / n_trials
    print(f"  n_train={n}  d={d}  alpha={alpha}  n_trials={n_trials}")
    print(f"  target coverage:      {1-alpha:.2f}")
    print(f"  empirical coverage:   {empirical_cov:.3f}"
          f"  (guarantee: >= {1 - 2*alpha:.2f})")
    print(f"  average interval width: {np.mean(widths):.3f}\n")

    print("  Single test-set example intervals:")
    for i in range(6):
        w = hi[i] - lo[i]
        hit = "in " if lo[i] <= y_te[i] <= hi[i] else "out"
        print(f"    y_true={y_te[i]:7.3f}  interval=[{lo[i]:7.3f}, {hi[i]:7.3f}]  width={w:.3f}  {hit}")

    print("\n--- library cross-check (mapie.regression.JackknifeAB; nonconformist) ---")
