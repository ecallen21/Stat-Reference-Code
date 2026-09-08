"""Turnbull NPMLE for interval-censored survival (Reference Sec 11.30).

Turnbull 1976 'The empirical distribution function with arbitrarily
grouped, censored and truncated data', JRSS-B. Generalises the
Kaplan-Meier estimator to observations known only to lie in intervals
(L_i, R_i]:

    * Exact times: L_i = R_i.
    * Right-censored: R_i = infinity.
    * Interval-censored: L_i < R_i, both finite.

Turnbull's self-consistent EM iterates on the mass at each candidate
'jump point' (unique L, R values):

    p_j^{(t+1)} = p_j^{(t)} * (1 / n) * sum_i alpha_ij(p^{(t)})
    alpha_ij = 1[interval i contains jump j] / sum_k alpha_ik * p_k

Converges to the NPMLE of the survival function.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def turnbull_ec(L, R, tol=1e-6, max_iter=500):
    """Self-consistency algorithm. L, R : arrays of interval bounds.

    Right-censored: pass R_i = np.inf.
    """
    #  Candidate jump points = unique finite L and R
    pts = np.unique(np.r_[L, R[R < np.inf]])
    m = len(pts)
    #  Indicator matrix A[i, j] = 1 if (L_i, R_i] contains pts[j] intersection
    A = np.zeros((len(L), m))
    for i in range(len(L)):
        for j in range(m):
            if L[i] < pts[j] and pts[j] <= R[i]:
                A[i, j] = 1
    p = np.ones(m) / m
    for it in range(max_iter):
        weight = A * p[None, :]
        row_sum = weight.sum(axis=1, keepdims=True) + 1e-12
        posterior = weight / row_sum
        p_new = posterior.sum(axis=0) / len(L)
        if np.max(np.abs(p_new - p)) < tol:
            p = p_new; break
        p = p_new
    #  Survival: S(t) = P(T > t) = 1 - cumsum(p at pts <= t)
    F = np.cumsum(p)
    S = 1 - F
    return {"points": pts, "mass": p, "S": S}


if __name__ == "__main__":
    print("=== Turnbull NPMLE for interval-censored survival ===\n")
    rng = np.random.default_rng(0)
    n = 400
    #  True T ~ Exponential(0.1) (mean 10)
    T = rng.exponential(10.0, size=n)
    #  Coarsen: each subject observed at check-ups spaced by ~5 -> interval-censored
    L = np.zeros(n); R = np.zeros(n)
    for i in range(n):
        checkups = np.sort(rng.uniform(0, 25, size=6))
        prior = checkups[checkups < T[i]]
        later = checkups[checkups >= T[i]]
        L[i] = prior[-1] if len(prior) else 0.0
        R[i] = later[0] if len(later) else np.inf

    r = turnbull_ec(L, R, max_iter=300)
    print(f"  n = {n}, interval-censored via check-ups")
    print(f"  Turnbull estimate at times 5, 10, 15, 20:")
    for t in [5, 10, 15, 20]:
        idx = np.searchsorted(r["points"], t)
        S_hat = float(r["S"][idx - 1]) if idx > 0 else 1.0
        S_true = float(np.exp(-t * 0.1))
        print(f"    S({t}) = {S_hat:.3f}   (truth exp(-0.1 * {t}) = {S_true:.3f})")

    print("\n--- library cross-check (icenReg / interval / survival R; lifelines KaplanMeierFitter Python) ---")
