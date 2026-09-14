"""Mahalanobis Distance Matching (Reference Sec 47.326).

Rubin 1980 Biometrics; Rosenbaum & Rubin 1985 Amer Stat.
Match treated to control units by covariate Mahalanobis
distance:

    d(x_i, x_j) = sqrt((x_i - x_j)^T S^{-1} (x_i - x_j))

where S is the pooled covariance of covariates. Optionally
combined with a CALIPER (max acceptable distance) or with
propensity-score matching.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def mahalanobis_match(X, treat, caliper=None):
    """1:1 nearest-neighbour Mahalanobis matching with optional caliper."""
    S_inv = np.linalg.inv(np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1]))
    treated_idx = np.where(treat == 1)[0]
    control_idx = np.where(treat == 0)[0]
    matches = []; used = set()
    for i in treated_idx:
        best = None; best_d = np.inf
        for j in control_idx:
            if j in used: continue
            diff = X[i] - X[j]
            d = float(np.sqrt(diff @ S_inv @ diff))
            if d < best_d:
                best_d = d; best = j
        if best is None: continue
        if caliper is not None and best_d > caliper: continue
        matches.append((int(i), int(best), best_d))
        used.add(best)
    return matches


def standardized_diff(X_treated, X_control):
    """Standardised mean difference for balance check."""
    smd = []
    for k in range(X_treated.shape[1]):
        m_t = X_treated[:, k].mean(); s_t = X_treated[:, k].std()
        m_c = X_control[:, k].mean(); s_c = X_control[:, k].std()
        pooled = np.sqrt((s_t ** 2 + s_c ** 2) / 2)
        smd.append((m_t - m_c) / (pooled + 1e-9))
    return smd


if __name__ == "__main__":
    print("=== Mahalanobis Distance Matching (Rubin 1980) ===\n")
    rng = np.random.default_rng(0)

    n = 600
    X0 = rng.multivariate_normal([0, 0, 0], np.eye(3), n // 2)
    X1 = rng.multivariate_normal([0.5, 0.3, -0.2], np.eye(3), n // 2)
    X = np.vstack([X0, X1])
    treat = np.array([0] * (n // 2) + [1] * (n // 2))

    print(f"  n = {n}, {int(treat.sum())} treated, {int((1 - treat).sum())} control")
    print(f"  {'Covariate':>10}  {'SMD before':>10}   {'SMD after':>10}")
    smd_before = standardized_diff(X[treat == 1], X[treat == 0])

    matches = mahalanobis_match(X, treat, caliper=0.5)
    print(f"\n  Matched pairs: {len(matches)} of {int(treat.sum())} treated")
    print(f"  Mean Mahalanobis distance: {np.mean([m[2] for m in matches]):.3f}")

    matched_treat_idx = [m[0] for m in matches]
    matched_control_idx = [m[1] for m in matches]
    smd_after = standardized_diff(X[matched_treat_idx], X[matched_control_idx])

    for k in range(3):
        print(f"  feature {k+1}    {smd_before[k]:>+.3f}       {smd_after[k]:>+.3f}")

    print(f"\n  Cochrane guideline: |SMD| < 0.1 = good balance.")
    print(f"  Mahalanobis matching outperforms Euclidean when covariates are correlated.")

    print("\n--- library cross-check (MatchIt R; causalmatch Python; SciPy cdist with 'mahalanobis') ---")
