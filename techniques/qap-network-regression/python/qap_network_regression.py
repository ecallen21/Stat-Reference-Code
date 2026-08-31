"""QAP network regression (Reference Sec 30.6).

Krackhardt (1988) 'Predicting with networks: nonparametric multiple
regression analyses of dyadic data.'

Standard OLS on network dyads is invalid because dyads are NOT
independent: they share nodes.  QAP (Quadratic Assignment Procedure)
tests give VALID p-values via label permutations of the DEPENDENT
NETWORK.

Procedure (Krackhardt-Dekker 2007 double semi-partialing):
  1. Fit OLS on vec(Y) ~ vec(X_1), vec(X_2), ...  Record coefficients.
  2. For B permutations, permute node LABELS of Y (row + col
     simultaneously) and refit; get null distribution of coefficients.
  3. p-value = fraction of |b_perm| >= |b_obs|.

Here we implement QAP on synthetic dyadic data: Y is a communication
frequency matrix; X_1 = friendship matrix (relevant predictor); X_2 =
random noise matrix (irrelevant); check that only X_1 is significant.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _mask_offdiag(M):
    n = M.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return M[mask]


def dyadic_ols(Y, X_list):
    """Vectorised OLS on the off-diagonal entries."""
    y = _mask_offdiag(Y)
    X = np.column_stack([np.ones_like(y)] + [_mask_offdiag(x) for x in X_list])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def qap_permutation_test(Y, X_list, B=500, seed=0):
    rng = np.random.default_rng(seed)
    b_obs = dyadic_ols(Y, X_list)
    K = len(X_list)
    b_null = np.zeros((B, K))
    n = Y.shape[0]
    for b in range(B):
        perm = rng.permutation(n)
        Y_p = Y[np.ix_(perm, perm)]
        b_p = dyadic_ols(Y_p, X_list)
        b_null[b] = b_p[1:]                            # skip intercept
    p_vals = np.array([(np.abs(b_null[:, k]) >= abs(b_obs[k + 1])).mean() for k in range(K)])
    return b_obs, p_vals


if __name__ == "__main__":
    print("=== QAP network regression (Krackhardt 1988) ===\n")
    rng = np.random.default_rng(0)
    n = 25
    # X_1: friendship (relevant): symmetric 0/1
    X1 = (rng.random((n, n)) < 0.3).astype(float)
    X1 = np.triu(X1, 1); X1 = X1 + X1.T
    # X_2: irrelevant random matrix
    X2 = rng.normal(0, 1, (n, n))
    X2 = (X2 + X2.T) / 2
    np.fill_diagonal(X2, 0)
    # Y: communication frequency, related to X_1 only.
    Y = 2.0 * X1 + rng.normal(0, 0.5, (n, n))
    Y = (Y + Y.T) / 2
    np.fill_diagonal(Y, 0)

    b_obs, p_vals = qap_permutation_test(Y, [X1, X2], B=500, seed=1)
    print(f"  OLS beta hat  = {b_obs.round(3).tolist()}   (intercept, friendship, noise)")
    print(f"  QAP p-values  = {p_vals.round(3).tolist()}   (friendship, noise)")
    print(f"\n  Only the friendship coefficient should be significant.\n")
    print("--- library cross-check (R sna::netlm; statnet::qap.lm; Python netperm) ---")
