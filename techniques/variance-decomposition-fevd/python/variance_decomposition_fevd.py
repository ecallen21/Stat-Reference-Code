"""Forecast Error Variance Decomposition - FEVD (Ref Sec 47.308).

Sims 1980. Decompose the h-step forecast error variance of each
variable INTO CONTRIBUTIONS from each structural shock:

    Var(y_{k, t+h} - y_hat) = sum_j sum_{s=0}^{h-1} (Psi_s * B0^-1)_{k, j}^2
    FEVD(k, j, h) = share attributable to shock j at horizon h

Complements IRFs: IRFs show DIRECTION and MAGNITUDE, FEVD shows
RELATIVE IMPORTANCE of each shock.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def var_companion(A_lags):
    k, kp = A_lags.shape; p = kp // k
    C = np.zeros((k * p, k * p))
    C[:k] = A_lags
    if p > 1: C[k:, :-k] = np.eye(k * (p - 1))
    return C


def fevd(A_lags, B0_inv, H=20):
    """Return FEVD shares of shape (H+1, k, k) where axis 1 is response
    variable and axis 2 is structural shock."""
    k = A_lags.shape[0]
    C = var_companion(A_lags)
    C_pow = np.eye(C.shape[0])
    Psi_list = []
    for h in range(H + 1):
        Psi_list.append(C_pow[:k, :k])
        C_pow = C_pow @ C
    # For each variable k, sum of squared IRF contributions from each shock j
    shares = np.zeros((H + 1, k, k))
    for h in range(H + 1):
        # Cumulative sum of (Psi_s @ B0_inv)^2 up to s = h
        contributions = np.zeros((k, k))
        for s in range(h + 1):
            M = Psi_list[s] @ B0_inv
            contributions += M ** 2
        # Normalise so each row (variable) sums to 1
        row_sums = contributions.sum(axis=1, keepdims=True)
        shares[h] = contributions / row_sums
    return shares


if __name__ == "__main__":
    print("=== FEVD - Forecast Error Variance Decomposition (Sims 1980) ===\n")
    rng = np.random.default_rng(0)

    T = 500; e = rng.standard_normal((T, 2)); Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t, 0] = 0.5 * Y[t - 1, 0] + e[t, 0]
        Y[t, 1] = 0.3 * Y[t - 1, 0] + 0.7 * Y[t - 1, 1] + 0.5 * e[t, 0] + e[t, 1]

    p = 1; X = np.column_stack([np.ones(T - p)] + [Y[p - l - 1:T - l - 1] for l in range(p)])
    A_full, *_ = np.linalg.lstsq(X, Y[p:], rcond=None)
    A_lags = A_full[1:].T
    resid = Y[p:] - X @ A_full
    Sigma = resid.T @ resid / (T - p - X.shape[1])
    B0_inv = np.linalg.cholesky(Sigma)

    shares = fevd(A_lags, B0_inv, H=15)
    print(f"  FEVD shares (percentage) of forecast-error variance by shock:")
    print(f"    {'h':>3}  |  y1 <- e1  y1 <- e2   |   y2 <- e1  y2 <- e2")
    for h in [0, 1, 2, 5, 10, 15]:
        s = shares[h] * 100
        print(f"    {h:>3}  |  {s[0,0]:>7.1f}%  {s[0,1]:>7.1f}%   |   {s[1,0]:>7.1f}%  {s[1,1]:>7.1f}%")

    print(f"\n  y1 is driven entirely by e1 (by construction; no e2 contemporaneous term).")
    print(f"  y2 receives both shocks (via the 0.5 * e1 loading in the structural DGP).")
    print(f"  FEVD complements IRFs: relative importance vs absolute impact.")

    print("\n--- library cross-check (statsmodels VARResults.fevd; vars::fevd R) ---")
