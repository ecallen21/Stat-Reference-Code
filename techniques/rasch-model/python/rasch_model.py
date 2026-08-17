"""Rasch model (1PL) for dichotomous items (Reference §22.5; Rasch 1960).

    Pr(y_ij = 1 | theta_i, b_j) = 1 / (1 + exp(-(theta_i - b_j)))

    theta_i : person ability
    b_j     : item difficulty
    common discrimination = 1 (that's the "one parameter" in 1PL)

Joint maximum likelihood (JML) estimation
    Maximize the joint log-likelihood over (theta, b) simultaneously.
    Simple but has a small-sample bias; corrected variants (WML) exist.
    Marginal MLE (MML) integrates theta out under a Normal(0, sigma^2) prior --
    the standard for research use.

Identifiability: fix sum(b) = 0 (or theta_1 = 0).

Special property: SUFFICIENT STATISTICS -- person's raw score is sufficient
for theta given b, and item's raw score is sufficient for b given theta.
That's what makes Rasch distinct from 2PL / 3PL (see 2PL/3PL module).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def _sigmoid(x): return 1 / (1 + np.exp(-x))


def rasch_jml(Y, max_iter: int = 200, tol: float = 1e-6) -> dict:
    """Joint MLE of Rasch model.  Y is n_persons x n_items binary."""
    Y = np.asarray(Y, dtype=float); n, J = Y.shape
    theta = np.zeros(n); b = np.zeros(J)
    for it in range(max_iter):
        # Fix b, update theta (each person's log-likelihood)
        for i in range(n):
            def neg_ll_i(t):
                p = _sigmoid(t - b)
                return -np.sum(Y[i] * np.log(p + 1e-12) + (1 - Y[i]) * np.log(1 - p + 1e-12))
            res = minimize(neg_ll_i, theta[i], method="BFGS")
            theta[i] = float(res.x[0])
        # Fix theta, update b (each item's log-likelihood)
        for j in range(J):
            def neg_ll_j(bj):
                p = _sigmoid(theta - bj)
                return -np.sum(Y[:, j] * np.log(p + 1e-12) + (1 - Y[:, j]) * np.log(1 - p + 1e-12))
            res = minimize(neg_ll_j, b[j], method="BFGS")
            b[j] = float(res.x[0])
        b = b - b.mean()                             # centered for identifiability
        if it > 0 and np.max(np.abs(b - b_prev)) < tol: break
        b_prev = b.copy()
    return {"theta": theta, "b_difficulty": b, "iterations": int(it + 1),
            "method": "Rasch 1PL Joint MLE"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, J = 300, 20
    theta_true = rng.normal(0, 1, n)
    b_true = rng.uniform(-2, 2, J); b_true -= b_true.mean()
    P = _sigmoid(theta_true[:, None] - b_true[None, :])
    Y = (rng.uniform(size=P.shape) < P).astype(int)

    fit = rasch_jml(Y, max_iter=20)
    print("=== Rasch JML ===")
    print(f"  Person theta: mean = {fit['theta'].mean():.3f}, sd = {fit['theta'].std():.3f}   (true mean 0, sd 1)")
    print(f"  correlation of theta_hat with theta_true = {np.corrcoef(fit['theta'], theta_true)[0, 1]:.4f}")
    print(f"  correlation of b_hat with b_true = {np.corrcoef(fit['b_difficulty'], b_true)[0, 1]:.4f}")

    print("\n--- library cross-check (R eRm::RM) ---")
    print("  R: eRm::RM(Y) for CML; ltm::rasch for MML")
