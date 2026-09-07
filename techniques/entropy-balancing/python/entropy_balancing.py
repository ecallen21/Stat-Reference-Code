"""Entropy balancing (Reference Sec 15.55).

Hainmueller 2012.  Directly reweight control units so weighted
covariate moments EXACTLY match treated moments, subject to keeping
weights as close to the base distribution as possible in KL sense.

Solve:  min  sum_i w_i log(w_i / q_i)
        s.t. sum_i w_i c_ij = m_j    for each moment j
             sum_i w_i = 1

Dual: closed-form weights via exponential tilting given Lagrange
multipliers lambda:

  w_i = exp(- sum_j lambda_j c_ij) / Z(lambda)

Solve for lambda by Newton on the moment-matching residuals.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def entropy_balance(C, targets, n_iter=100, tol=1e-8):
    """C : (n_ctrl, J) control covariates; targets : (J,) treated means."""
    n, J = C.shape
    lam = np.zeros(J)
    for _ in range(n_iter):
        eta = -C @ lam
        w = np.exp(eta - eta.max())
        w = w / w.sum()
        moment_diff = C.T @ w - targets
        if np.abs(moment_diff).max() < tol:
            break
        # Newton step
        Cw = C * w[:, None]
        cov = (C.T @ Cw) - np.outer(C.T @ w, C.T @ w)
        try:
            step = np.linalg.solve(cov + 1e-8 * np.eye(J), moment_diff)
        except np.linalg.LinAlgError:
            step = moment_diff * 0.5
        lam = lam + step
    return w


if __name__ == "__main__":
    print("=== Entropy balancing (Hainmueller 2012) ===\n")
    rng = np.random.default_rng(0)
    n_c = 2000; n_t = 500
    p = 3
    X_c = rng.normal(0, 1, (n_c, p))
    X_t = rng.normal(0.5, 1, (n_t, p))       # treated mean shifted by +0.5

    # Match first + second moments (mean and mean-square)
    C = np.column_stack([X_c, X_c ** 2])
    targets = np.concatenate([X_t.mean(axis=0), (X_t ** 2).mean(axis=0)])
    w = entropy_balance(C, targets)

    print(f"  Target (treated) means : {X_t.mean(axis=0).round(3)}")
    print(f"  Unweighted ctrl means  : {X_c.mean(axis=0).round(3)}")
    print(f"  Weighted   ctrl means  : {(w[:, None] * X_c).sum(axis=0).round(3)}")
    print(f"\n  Target treated E[X^2] : {(X_t ** 2).mean(axis=0).round(3)}")
    print(f"  Weighted ctrl E[X^2]  : {(w[:, None] * X_c ** 2).sum(axis=0).round(3)}")

    # Effective sample size = 1 / sum(w^2) scaled by n_ctrl
    ess = 1 / (w ** 2).sum() / n_c * n_c
    print(f"\n  Effective sample size (weighted controls): {ess:.0f} / {n_c}")

    print("\n--- library cross-check (R WeightIt::ebalance, ebal; Python custom + econml) ---")
