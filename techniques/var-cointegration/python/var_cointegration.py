"""VAR + cointegration + error correction (Reference §13.12, §13.13, §13.44).

VAR(p): Vector Autoregression on K series y_t = (y_{1t}, ..., y_{Kt}):
    y_t  =  c  +  A_1 y_{t-1}  +  ...  +  A_p y_{t-p}  +  u_t

Each variable is regressed on lags of ALL variables. Fitted equation-by-equation
via OLS. Order p chosen by AIC or BIC.

Cointegration:
    Two (or more) non-stationary I(1) series are COINTEGRATED if some linear
    combination is stationary. Economic meaning: they share a common long-run
    relationship, even though each individually wanders.

Engle-Granger 2-step (§13.44):
    Step 1: OLS regression y_1_t = alpha + beta * y_2_t + u_t
    Step 2: Test u_t for a unit root (ADF on OLS residuals).
        Reject unit root => y_1 and y_2 are cointegrated with rank 1.

Error Correction Model (§13.13):
    Delta y_1_t  =  gamma * (y_1_{t-1} - beta * y_2_{t-1})   +  lagged diffs  +  eps
                    ^^^^^ error-correction term ^^^^^^^^
    gamma < 0 => y_1 adjusts back toward the long-run equilibrium.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions


def fit_var(Y, p: int) -> dict:
    """VAR(p) fit by equation-by-equation OLS.

    Parameters
    ----------
    Y : n x K matrix of the K series.
    p : lag order.
    """
    Y = np.asarray(Y, dtype=float)
    n, K = Y.shape
    # Build lagged design: rows = t = p..n-1
    Xrows = []; yrows = []
    for t in range(p, n):
        lag_stack = [1.0]                             # intercept
        for k in range(1, p + 1):
            lag_stack.extend(Y[t - k])
        Xrows.append(lag_stack); yrows.append(Y[t])
    X = np.array(Xrows); y_mat = np.array(yrows)
    beta_all, *_ = np.linalg.lstsq(X, y_mat, rcond=None)  # (1 + K*p) x K
    resid = y_mat - X @ beta_all
    Sigma = (resid.T @ resid) / (len(y_mat) - X.shape[1])
    n_params = beta_all.size + K * (K + 1) / 2
    log_lik = -0.5 * len(y_mat) * (K * math.log(2 * math.pi) + math.log(max(np.linalg.det(Sigma), 1e-300)))
    log_lik -= 0.5 * np.trace(resid @ np.linalg.inv(Sigma) @ resid.T)
    aic = 2 * n_params - 2 * log_lik
    return {"p": p, "K": K, "n_obs": int(len(y_mat)),
            "coefficients": beta_all.tolist(),
            "Sigma_residual": Sigma.tolist(),
            "AIC": float(aic), "log_lik": float(log_lik),
            "method": f"VAR({p}) via equation-by-equation OLS"}


def engle_granger_cointegration(y1, y2) -> dict:
    """Engle-Granger 2-step cointegration test on I(1) series y1, y2."""
    from statsmodels.tsa.stattools import adfuller
    y1 = np.asarray(y1, dtype=float); y2 = np.asarray(y2, dtype=float)
    # Step 1: OLS regression y1 = alpha + beta y2
    X = np.column_stack([np.ones(len(y1)), y2])
    beta, *_ = np.linalg.lstsq(X, y1, rcond=None)
    resid = y1 - X @ beta
    # Step 2: ADF on residuals (no constant, no trend -- Engle-Granger critical values differ but we use the p-value from adfuller)
    stat, p, *_ = adfuller(resid, regression="n")
    return {"alpha": float(beta[0]), "beta": float(beta[1]),
            "residual_first_5": resid[:5].tolist(),
            "ADF_on_resid_statistic": float(stat),
            "ADF_on_resid_p_value": float(p),
            "cointegrated": bool(p < 0.05),
            "interpretation": "small p => residuals are stationary => cointegrated",
            "method": "Engle-Granger 2-step cointegration"}


def error_correction_model(y1, y2, lags: int = 1) -> dict:
    """1-lag ECM: Delta y1 = alpha + gamma (y1_{t-1} - beta y2_{t-1}) + ..."""
    y1 = np.asarray(y1, dtype=float); y2 = np.asarray(y2, dtype=float)
    # Get long-run beta from cointegrating regression
    Xlong = np.column_stack([np.ones(len(y1)), y2])
    beta_long, *_ = np.linalg.lstsq(Xlong, y1, rcond=None)
    beta_lr = float(beta_long[1])
    # Cointegrating residual (error correction term)
    ect = y1 - beta_long[0] - beta_lr * y2

    dy1 = np.diff(y1); dy2 = np.diff(y2)
    n = len(dy1)
    # Delta y1_t = c + gamma * ect_{t-1} + phi1 * Delta y1_{t-1} + phi2 * Delta y2_{t-1} + eps
    ect_lag = ect[:-1]
    X_rows = []; y_rows = []
    for t in range(lags, n):
        row = [1.0, ect_lag[t]]
        for k in range(1, lags + 1):
            row.extend([dy1[t - k], dy2[t - k]])
        X_rows.append(row); y_rows.append(dy1[t])
    X = np.array(X_rows); y = np.array(y_rows)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"long_run_beta": beta_lr,
            "ecm_intercept": float(beta[0]),
            "gamma_speed_of_adjustment": float(beta[1]),
            "gamma_interpretation": ("negative gamma => y1 corrects back toward "
                                      "the long-run relation; positive => explosive"),
            "n_used": int(len(y)),
            "method": "Error correction model (1 lag)"}


def library_versions(Y, p=2):
    from statsmodels.tsa.api import VAR
    m = VAR(Y).fit(p)
    return {"statsmodels VAR AIC": float(m.aic),
            "statsmodels VAR loglik": float(m.llf)}


if __name__ == "__main__":
    rng = np.random.default_rng(29)
    n = 300
    # Simulate a cointegrated pair: y2 is a random walk; y1 = 2 * y2 + stationary noise
    y2 = np.cumsum(rng.normal(0, 1, n))
    y1 = 2.0 * y2 + rng.normal(0, 1, n)          # cointegrated (cointegrating beta = 2)
    Y = np.column_stack([y1, y2])

    print("=== VAR(2) fit on (y1, y2) ===")
    vf = fit_var(Y, p=2)
    print(f"  AIC = {vf['AIC']:.2f}, log-lik = {vf['log_lik']:.2f}")
    print(f"  Residual Sigma:")
    for row in vf["Sigma_residual"]:
        print(f"    {row}")

    print("\n=== Engle-Granger 2-step cointegration ===")
    eg = engle_granger_cointegration(y1, y2)
    print(f"  long-run: y1 ~ {eg['alpha']:+.3f} + {eg['beta']:+.3f} * y2   (true beta = 2.0)")
    print(f"  ADF on residuals: stat = {eg['ADF_on_resid_statistic']:.3f}, p = {eg['ADF_on_resid_p_value']:.4g}")
    print(f"  Cointegrated? {eg['cointegrated']}")

    print("\n=== Error Correction Model ===")
    ecm = error_correction_model(y1, y2)
    print(f"  long-run beta = {ecm['long_run_beta']:+.3f}")
    print(f"  gamma (speed of adjustment) = {ecm['gamma_speed_of_adjustment']:+.4f}")
    print(f"  {ecm['gamma_interpretation']}")

    print("\n--- library (statsmodels VAR) ---")
    for k, v in library_versions(Y, p=2).items():
        print(f"  {k}: {v}")
