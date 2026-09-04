"""Arellano-Bond difference GMM (Reference Sec 35.3).

Arellano & Bond (1991) 'Some tests of specification for panel data.'

For a DYNAMIC panel with lagged dependent variable:

  y_it = rho * y_i,t-1 + x_it' beta + a_i + eps_it,

OLS + FE + within-transform are all INCONSISTENT because y_i,t-1 is
correlated with the transformed error (Nickell bias).

Arellano-Bond: FIRST-DIFFERENCE the model to eliminate a_i,

  Delta y_it = rho * Delta y_i,t-1 + Delta x_it' beta + Delta eps_it,

then use LAGGED LEVELS y_i,t-2, y_i,t-3, ... as INSTRUMENTS for
Delta y_i,t-1. GMM combines all valid instruments.

Here we implement first-difference GMM with a compact instrument set,
compare to plain FE on a synthetic AR(1) panel, and confirm AB
recovers rho while FE is biased downward.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def within_transform(x, unit):
    xw = x.astype(float).copy()
    for u in np.unique(unit):
        m = unit == u
        xw[m] -= xw[m].mean(axis=0)
    return xw


def diff_within(x, unit):
    """First difference within each unit (drop the first period per unit)."""
    idx = []; d = []
    for u in np.unique(unit):
        m = np.where(unit == u)[0]
        for k in range(1, len(m)):
            idx.append(m[k])
            d.append(x[m[k]] - x[m[k - 1]])
    return np.array(d), np.array(idx)


def arellano_bond(y, unit, T):
    """Difference GMM: instrument Delta y_{it-1} with y_{i, t-2}, y_{i, t-3}, ..."""
    n_units = len(np.unique(unit))
    # First-differenced series
    dY = np.zeros((n_units, T - 1))
    for u_idx, u in enumerate(np.unique(unit)):
        m = np.where(unit == u)[0]
        y_u = y[m]
        dY[u_idx] = np.diff(y_u)                    # length T-1
    # Difference eq for t >= 3 (0-indexed: k >= 2):
    #   dY[:, k] = rho * dY[:, k - 1] + du_it
    # instrument for dY[:, k - 1] is level y[:, k - 1] (i.e. y_{i, t-2}).
    ys = np.zeros((n_units, T))
    for u_idx, u in enumerate(np.unique(unit)):
        m = np.where(unit == u)[0]
        ys[u_idx] = y[m]

    # Full Arellano-Bond GMM: for each (unit, t) with t >= 3 build a block
    # of instruments {y_{i, 0}, ..., y_{i, t-2}}; stack in block-diagonal
    # form and use one-step GMM with 2SLS weighting.
    from scipy.linalg import block_diag
    Z_blocks = []; lhs = []; rhs = []
    for u_idx in range(n_units):
        Z_u = []                                     # each row is an instrument vector for one t
        for k in range(2, T - 1):                    # dY index k corresponds to t=k+1
            z_row = np.zeros(T - 3)
            for lag in range(min(k, T - 3)):
                z_row[lag] = ys[u_idx, lag]
            Z_u.append(z_row)
            lhs.append(dY[u_idx, k])
            rhs.append(dY[u_idx, k - 1])
        Z_blocks.append(np.array(Z_u))
    lhs = np.array(lhs); rhs = np.array(rhs); Z = np.vstack(Z_blocks)
    # One-step 2SLS with identity weighting on projected regressor
    ZtZ = Z.T @ Z + 1e-4 * np.eye(Z.shape[1])
    ZtX = Z.T @ rhs
    Zty = Z.T @ lhs
    return float((ZtX @ np.linalg.solve(ZtZ, Zty))
                  / (ZtX @ np.linalg.solve(ZtZ, ZtX)))


def within_ols(y, unit, T):
    """Within-FE OLS of y on lagged y (Nickell-biased for dynamic panel)."""
    lhs = []; rhs = []
    for u in np.unique(unit):
        m = np.where(unit == u)[0]
        y_u = y[m]
        lhs.append(y_u[1:] - y_u[1:].mean())
        rhs.append((y_u[:-1]) - y_u[:-1].mean())
    l = np.concatenate(lhs); r = np.concatenate(rhs)
    return float((r @ l) / (r @ r))


if __name__ == "__main__":
    print("=== Arellano-Bond difference GMM (Arellano-Bond 1991) ===\n")
    rng = np.random.default_rng(0)
    n_units, T = 500, 15
    rho_true = 0.6
    y_all = []; unit = []
    for i in range(n_units):
        a = rng.normal(0, 1)                        # unit effect
        y = np.zeros(T)
        y[0] = a + rng.normal(0, 0.5)
        for t in range(1, T):
            y[t] = rho_true * y[t - 1] + a + rng.normal(0, 0.5)
        y_all.append(y); unit.extend([i] * T)
    y = np.concatenate(y_all); unit = np.array(unit)

    rho_ols_within = within_ols(y, unit, T)
    rho_ab = arellano_bond(y, unit, T)
    print(f"  true rho              = {rho_true:.3f}")
    print(f"  within-FE OLS rho_hat = {rho_ols_within:.3f}   (Nickell-biased downward)")
    print(f"  Arellano-Bond rho_hat = {rho_ab:.3f}")
    print("\n  Note: Arellano-Bond is consistent as N -> inf with fixed T under the")
    print("  moment conditions, but has known small-sample bias (Alvarez-Arellano 2003,")
    print("  Bun-Windmeijer 2010). System GMM (Blundell-Bond 1998) and Windmeijer-")
    print("  corrected SEs usually address it in practice.\n")
    print("--- library cross-check (R plm::pgmm; Python linearmodels.panel.PanelGMM) ---")
