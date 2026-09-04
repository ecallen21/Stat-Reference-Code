"""Panel cointegration (Reference Sec 35.26).

Pedroni (1999, 2004); Westerlund (2007); Pesaran-Shin-Smith PMG (1999).

For a PANEL of I(1) series, test whether y_it and x_it are COINTEGRATED
(share a long-run equilibrium):

  y_it = alpha_i + beta_i x_it + e_it,

with residuals e_it stationary under H_1 (cointegration).

Pedroni approach: run regression per unit, then pool a unit-root test
statistic on the residuals; two variants:
  * Panel PP / ADF (group-mean or panel-level)
  * Panel variance ratio.

Here we implement a compact panel-ADF residual test and apply to
synthetic cointegrated vs non-cointegrated panels.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _adf_stat(x, max_lag=0):
    """ADF t-statistic on rho in Delta x_t = rho x_{t-1} + eps_t."""
    dx = np.diff(x)
    lx = x[:-1]
    if max_lag > 0:
        lags = [dx[max_lag - k - 1:-k - 1] if k > 0 else dx[max_lag:] for k in range(max_lag)]
        X = np.column_stack([lx[max_lag:]] + lags)
        y_reg = dx[max_lag:]
    else:
        X = lx.reshape(-1, 1)
        y_reg = dx
    beta, *_ = np.linalg.lstsq(X, y_reg, rcond=None)
    resid = y_reg - X @ beta
    n = len(y_reg); k = X.shape[1]
    sigma2 = float(resid @ resid / (n - k))
    Vb = sigma2 * np.linalg.inv(X.T @ X + 1e-8 * np.eye(k))
    return float(beta[0] / np.sqrt(Vb[0, 0]))


def pedroni_panel_test(y, x, unit):
    """Panel ADF on cross-unit residuals."""
    stats = []
    for u in np.unique(unit):
        m = unit == u
        y_u = y[m]; x_u = x[m]
        # Cointegrating regression
        A = np.stack([np.ones(len(x_u)), x_u], axis=1)
        beta, *_ = np.linalg.lstsq(A, y_u, rcond=None)
        e = y_u - A @ beta
        stats.append(_adf_stat(e))
    group_mean = float(np.mean(stats))
    # Reject H_0 (no cointegration) if group-mean ADF is sufficiently negative.
    # Critical value ~ -2 at 5% (rough); use permutation for a real test.
    return group_mean, stats


if __name__ == "__main__":
    print("=== Panel cointegration (Pedroni 1999) ===\n")
    rng = np.random.default_rng(0)
    N, T = 30, 60

    # Case A: cointegrated. x is a random walk; y = beta * x + stationary noise.
    y_list = []; x_list = []; unit = []
    for i in range(N):
        x = np.cumsum(rng.normal(0, 1, T))          # random walk (I(1))
        y = 0.7 * x + rng.normal(0, 0.5, T)         # stationary noise -> cointegrated
        y_list.append(y); x_list.append(x); unit.extend([i] * T)
    y = np.concatenate(y_list); x = np.concatenate(x_list); u = np.array(unit)
    gm_A, _ = pedroni_panel_test(y, x, u)

    # Case B: not cointegrated. y is an independent random walk from x.
    y_list = []; x_list = []
    for i in range(N):
        x = np.cumsum(rng.normal(0, 1, T))
        y = np.cumsum(rng.normal(0, 1, T))
        y_list.append(y); x_list.append(x)
    y = np.concatenate(y_list); x = np.concatenate(x_list)
    gm_B, _ = pedroni_panel_test(y, x, u)

    print(f"  Case A (cointegrated):     group-mean ADF t = {gm_A:.3f}   (very negative = reject no-cointegration)")
    print(f"  Case B (not cointegrated): group-mean ADF t = {gm_B:.3f}   (near 0 = don't reject)")
    print("\n  Critical values from Pedroni tables; here we compare relative magnitudes.\n")
    print("--- library cross-check (R plm::pcointtest; punitroots; Python arch.unitroot; egcm) ---")
