"""Impulse Response Function (Reference Sec 47.307).

Sims 1980. For a VAR(p), simulate the dynamic response of each
variable to a UNIT-SIZE shock in one identified structural
innovation, over horizon h = 0, 1, ..., H:

    y_t = sum_{s>=0} Psi_s * e_{t-s}       (MA representation)
    IRF(h, j -> k) = element k of Psi_h * e_j
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def var_companion(A_lags):
    """Companion-matrix form of VAR(p) lag coefficients (k x k*p)."""
    k, kp = A_lags.shape                                          # A_lags has shape (k, k*p)
    p = kp // k
    C = np.zeros((k * p, k * p))
    C[:k] = A_lags
    if p > 1:
        C[k:, :-k] = np.eye(k * (p - 1))
    return C


def compute_irf(A_lags, B0_inv, H=20):
    """Structural IRF up to horizon H."""
    k = A_lags.shape[0]
    C = var_companion(A_lags)
    C_pow = np.eye(C.shape[0])
    irfs = []
    for h in range(H + 1):
        Psi_h = C_pow[:k, :k]                                    # top-left k x k
        irfs.append(Psi_h @ B0_inv)
        C_pow = C_pow @ C
    return np.array(irfs)                                         # (H+1, k, k)


if __name__ == "__main__":
    print("=== Impulse Response Function (Sims 1980) ===\n")
    rng = np.random.default_rng(0)

    # Simulate bivariate VAR(1); estimate; compute IRFs
    T = 500
    e = rng.standard_normal((T, 2))
    Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t, 0] = 0.5 * Y[t - 1, 0] + e[t, 0]
        Y[t, 1] = 0.3 * Y[t - 1, 0] + 0.7 * Y[t - 1, 1] + 0.5 * e[t, 0] + e[t, 1]

    from numpy.linalg import lstsq
    p = 1; X = np.column_stack([np.ones(T - p)] + [Y[p - l - 1:T - l - 1] for l in range(p)])
    A_full, *_ = lstsq(X, Y[p:], rcond=None)
    A_lags = A_full[1:].T                                          # (k x k*p)
    resid = Y[p:] - X @ A_full
    Sigma = resid.T @ resid / (T - p - X.shape[1])
    B0_inv = np.linalg.cholesky(Sigma)

    irfs = compute_irf(A_lags, B0_inv, H=15)
    print(f"  Horizon  |   e1 -> y1   e1 -> y2   |   e2 -> y1   e2 -> y2")
    for h in [0, 1, 2, 5, 10, 15]:
        r = irfs[h]
        print(f"    h = {h:>2}  |  {r[0,0]:>+7.3f}   {r[1,0]:>+7.3f}   |  {r[0,1]:>+7.3f}   {r[1,1]:>+7.3f}")

    print(f"\n  Structural shock e1 initially raises y1 by {irfs[0,0,0]:.2f}, y2 by {irfs[0,1,0]:.2f}.")
    print(f"  Responses DECAY over horizon (stable VAR); persistence controlled by A_1 eigenvalues.")
    print(f"  Cumulative IRF over 15 periods: y1 <- e1 = {irfs[:16,0,0].sum():.2f}")

    print("\n--- library cross-check (statsmodels.tsa.vector_ar.var_model.VARResults.irf; vars R irf) ---")
