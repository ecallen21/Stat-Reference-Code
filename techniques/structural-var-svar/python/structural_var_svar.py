"""Structural Vector Autoregression - SVAR (Reference Sec 47.306).

Sims 1980 Econometrica; Blanchard-Quah 1989 AER. A reduced-form
VAR(p):

    y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + u_t,   Cov(u_t) = Sigma

is transformed to a STRUCTURAL representation:

    B_0 y_t = B_1 y_{t-1} + ... + B_p y_{t-p} + e_t,  Cov(e_t) = D (diag)

by identifying B_0 (or its inverse). Cholesky (recursive)
identification is the simplest: order variables, then set B_0^{-1}
to a lower-triangular matrix such that B_0^{-1} B_0^{-T} = Sigma.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def fit_var(Y, p=1):
    """OLS estimate of VAR(p) coefficients and residual covariance."""
    T, k = Y.shape
    X = np.column_stack([np.ones(T - p)] +
                         [Y[p - l - 1:T - l - 1] for l in range(p)])
    Yt = Y[p:]
    A, *_ = np.linalg.lstsq(X, Yt, rcond=None)
    resid = Yt - X @ A
    Sigma = resid.T @ resid / (T - p - X.shape[1])
    return A, Sigma, resid


def cholesky_identification(Sigma):
    """Recursive (Cholesky) identification. Returns B0^{-1} lower-triangular."""
    L = np.linalg.cholesky(Sigma)
    return L


if __name__ == "__main__":
    print("=== Structural VAR (Sims 1980; Blanchard-Quah 1989) ===\n")
    rng = np.random.default_rng(0)

    # Simulate a bivariate structural DGP:
    #   e1_t and e2_t iid N(0, 1) STRUCTURAL shocks
    #   y1_t = 0.5 y1_{t-1} + e1_t
    #   y2_t = 0.3 y1_{t-1} + 0.7 y2_{t-1} + 0.5 e1_t + e2_t
    T = 500
    e = rng.standard_normal((T, 2))
    Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t, 0] = 0.5 * Y[t - 1, 0] + e[t, 0]
        Y[t, 1] = 0.3 * Y[t - 1, 0] + 0.7 * Y[t - 1, 1] + 0.5 * e[t, 0] + e[t, 1]

    A, Sigma, _ = fit_var(Y, p=1)
    print(f"  Estimated VAR(1) coefficient matrix (const + lag 1):")
    print(f"    intercept = {A[0].round(3).tolist()}")
    print(f"    A_1       = \n{np.array2string(A[1:].T, precision=3)}")
    print(f"\n  Reduced-form residual covariance Sigma:")
    print(f"{np.array2string(Sigma, precision=3)}\n")

    B0_inv = cholesky_identification(Sigma)
    print(f"  Cholesky B0^-1 (structural impact matrix):")
    print(f"{np.array2string(B0_inv, precision=3)}")
    print(f"\n  Diagonal = per-shock impact on OWN variable; off-diagonal = ordering-dependent.")
    print(f"  Reordering variables gives a DIFFERENT identification.")

    # Sanity: B0_inv @ B0_inv.T should reproduce Sigma
    print(f"\n  Sanity: |B0^-1 B0^-T - Sigma|_F = {np.linalg.norm(B0_inv @ B0_inv.T - Sigma):.4f}")
    print(f"  Blanchard-Quah / sign-restrictions / Rubio-Ramirez alternatives")
    print(f"  provide identification WITHOUT the recursive ordering assumption.")

    print("\n--- library cross-check (statsmodels.tsa.vector_ar.svar_model; vars R) ---")
