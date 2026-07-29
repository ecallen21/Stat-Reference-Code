"""Granger causality (Reference §13.50).

Does past X help predict Y beyond past Y itself?

    Model_R (restricted): Y_t = c + sum phi_k Y_{t-k} + eps_t         (Y only)
    Model_U (unrestricted): Y_t = c + sum phi_k Y_{t-k}
                                    + sum psi_k X_{t-k} + eps_t       (Y + X lags)

    F-test: does adding X's lags reduce SSR significantly?
        F  =  ((SSR_R - SSR_U) / p_x)  /  (SSR_U / (n - p_y - p_x - 1))
        p-value under F(p_x, n - p_y - p_x - 1)

Small p => X GRANGER-causes Y (helps predict future Y).

Limitations to state clearly (§13.50):
    - "Granger causality" is a PREDICTION statement, not a causal one. If X and Y are
      both driven by a common Z that leads them both, X may appear to Granger-cause
      Y without causing it in any real sense.
    - Assumes correctly-specified lag structure and stationarity.
    - Multivariate extension: fit a full VAR and test the block of X-coefficients.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions


def _lagged(y, p):
    n = len(y); out = np.zeros((n - p, p))
    for k in range(1, p + 1):
        out[:, k - 1] = y[p - k: n - k]
    return out


def granger_causality(x, y, max_lag: int = 5) -> dict:
    """Test whether X Granger-causes Y at lag = max_lag."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    x = x[:n]; y = y[:n]
    p = max_lag
    y_lags = _lagged(y, p)
    x_lags = _lagged(x, p)
    y_target = y[p:]
    # Restricted: Y on Y's lags
    X_R = np.column_stack([np.ones(n - p), y_lags])
    beta_R, *_ = np.linalg.lstsq(X_R, y_target, rcond=None)
    resid_R = y_target - X_R @ beta_R
    SSR_R = float((resid_R ** 2).sum())
    # Unrestricted: Y on Y's + X's lags
    X_U = np.column_stack([np.ones(n - p), y_lags, x_lags])
    beta_U, *_ = np.linalg.lstsq(X_U, y_target, rcond=None)
    resid_U = y_target - X_U @ beta_U
    SSR_U = float((resid_U ** 2).sum())
    df_num = p
    df_den = n - p - 2 * p - 1
    F = ((SSR_R - SSR_U) / df_num) / (SSR_U / df_den) if df_den > 0 and SSR_U > 0 else float("inf")
    p_val = float(stats.f.sf(F, df_num, df_den)) if df_den > 0 else float("nan")
    return {"max_lag": p,
            "F_statistic": float(F),
            "df_num": int(df_num), "df_den": int(df_den),
            "p_value": p_val,
            "SSR_restricted": SSR_R, "SSR_unrestricted": SSR_U,
            "interpretation": ("small p => reject H0 that X does NOT Granger-cause Y "
                               "(X's past improves Y's prediction)"),
            "method": "Granger causality (F-test on nested regressions)"}


def granger_causality_bidirectional(x, y, max_lag: int = 5) -> dict:
    """Both directions: X -> Y and Y -> X."""
    return {"X_Granger_causes_Y": granger_causality(x, y, max_lag),
            "Y_Granger_causes_X": granger_causality(y, x, max_lag)}


def library_versions(y, x, max_lag=5):
    from statsmodels.tsa.stattools import grangercausalitytests
    import io, contextlib
    data = np.column_stack([y, x])           # statsmodels convention: [target, predictor]
    with contextlib.redirect_stdout(io.StringIO()):
        r = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    return {f"statsmodels F @ lag {max_lag}":
            float(r[max_lag][0]["ssr_ftest"][0]),
            f"statsmodels p @ lag {max_lag}":
            float(r[max_lag][0]["ssr_ftest"][1])}


if __name__ == "__main__":
    rng = np.random.default_rng(31)
    n = 500
    # Simulate: X causes Y (via lagged X), no reverse
    x = np.zeros(n); y = np.zeros(n)
    x[0] = rng.normal(); y[0] = rng.normal()
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.normal()
        y[t] = 0.3 * y[t - 1] + 0.6 * x[t - 1] + rng.normal()

    print("=== X -> Y (should REJECT null; X truly Granger-causes Y) ===")
    r = granger_causality(x, y, max_lag=5)
    print(f"  F({r['df_num']}, {r['df_den']}) = {r['F_statistic']:.4f}, p = {r['p_value']:.4g}")

    print("\n=== Y -> X (should NOT reject; no reverse causality) ===")
    r = granger_causality(y, x, max_lag=5)
    print(f"  F({r['df_num']}, {r['df_den']}) = {r['F_statistic']:.4f}, p = {r['p_value']:.4g}")

    print("\n--- library (statsmodels grangercausalitytests) ---")
    for k, v in library_versions(y, x, max_lag=5).items():
        print(f"  {k}: {v}")
